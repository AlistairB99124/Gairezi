"""Work Package 3 of ``Aster_Porting.md``: fail the build on a bad mesh/groups.

Checks performed, each independent of the exporter's own bookkeeping:

* the Gmsh file on disk actually contains the physical groups/counts the
  mesh manifest claims (no silent drift between exporter and export);
* every required group (``Elmer/regions/model_groups.REQUIRED_GROUPS``) is
  present and nonempty;
* no cell repeats a node id (degenerate/collapsed element);
* every volume cell decomposes into strictly positive-volume tetrahedra
  (reuses ``Elmer.regions.tetrahedral.split_cell``, which already raises on
  a degenerate tetrahedron);
* re-running ``audit_combined_structure`` on a freshly built structure still
  passes, which independently re-checks bonded-interface/duplicate-node
  topology.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Aster.build_med_mesh import DEFAULT_OUTPUT_DIR
from Elmer.regions.model_groups import REQUIRED_GROUPS
from Elmer.regions.tetrahedral import split_cell
from Elmer.structural_model import audit_combined_structure, build_combined_structure


class MeshValidationError(ValueError):
    """A mesh or named-group problem that must fail the Code_Aster build."""


def parse_gmsh_physical_groups(path: Path) -> dict[tuple[int, int], str]:
    text = path.read_text(encoding="ascii")
    match = re.search(r"\$PhysicalNames\n(\d+)\n(.*?)\$EndPhysicalNames", text, re.DOTALL)
    if not match:
        raise MeshValidationError(f"{path} has no $PhysicalNames section")
    count = int(match.group(1))
    lines = [line for line in match.group(2).splitlines() if line.strip()]
    if len(lines) != count:
        raise MeshValidationError(f"{path} declares {count} physical names but lists {len(lines)}")
    groups: dict[tuple[int, int], str] = {}
    for line in lines:
        dim_text, tag_text, name_text = line.split(None, 2)
        groups[(int(dim_text), int(tag_text))] = name_text.strip().strip('"')
    return groups


# Gmsh element-type codes used by this exporter, mapped to their topological
# dimension so that a volume physical tag is never confused with a face
# physical tag that happens to share the same numeric id (REGION_GROUP_IDS
# and FACE_GROUP_IDS are independent namespaces, each starting at 1).
_ELEMENT_TYPE_DIM = {2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3}


def parse_gmsh_counts(path: Path) -> tuple[int, dict[tuple[int, int], int]]:
    """(node count, {(dim, physical tag): element count}) from $Elements."""
    text = path.read_text(encoding="ascii")
    node_match = re.search(r"\$Nodes\n(\d+)\n", text)
    if not node_match:
        raise MeshValidationError(f"{path} has no $Nodes section")
    node_count = int(node_match.group(1))
    elements_match = re.search(r"\$Elements\n(\d+)\n(.*?)\$EndElements", text, re.DOTALL)
    if not elements_match:
        raise MeshValidationError(f"{path} has no $Elements section")
    declared_count = int(elements_match.group(1))
    counts: dict[tuple[int, int], int] = {}
    lines = [line for line in elements_match.group(2).splitlines() if line.strip()]
    if len(lines) != declared_count:
        raise MeshValidationError(f"{path} declares {declared_count} elements but lists {len(lines)}")
    for line in lines:
        fields = line.split()
        element_type = int(fields[1])
        tag = int(fields[3])
        dim = _ELEMENT_TYPE_DIM[element_type]
        counts[(dim, tag)] = counts.get((dim, tag), 0) + 1
    return node_count, counts


def check_degenerate_cells(structure) -> list[str]:
    problems: list[str] = []
    for cell_index, (element_type, cell) in enumerate(structure.cells):
        if len(set(cell)) != len(cell):
            problems.append(f"cell {cell_index} (type {element_type}) repeats a node id: {cell}")
            continue
        try:
            split_cell(structure.nodes, element_type, cell)
        except ValueError as error:
            problems.append(f"cell {cell_index} (type {element_type}) is degenerate: {error}")
    return problems


def validate(root: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    manifest_path = output_dir / "mesh_manifest.json"
    gmsh_path = output_dir / "mesh.msh"
    if not manifest_path.is_file() or not gmsh_path.is_file():
        raise MeshValidationError(
            f"Run Aster.build_med_mesh first; missing {manifest_path} or {gmsh_path}",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    problems: list[str] = []

    physical_names = parse_gmsh_physical_groups(gmsh_path)
    node_count, counts_by_tag = parse_gmsh_counts(gmsh_path)
    if node_count != manifest["node_count"]:
        problems.append(f"mesh.msh has {node_count} nodes; manifest says {manifest['node_count']}")

    for name, entity in {"WALL": "volume", **{name: "face" for name in (
        "FOUNDATION", "UPSTREAM", "DOWNSTREAM", "CREST", "LEFT_ABUTMENT", "RIGHT_ABUTMENT", "OTHER_EXTERIOR",
    )}}.items():
        group_manifest = manifest["groups"][name]
        expected_count = group_manifest["cell_count"] if entity == "volume" else group_manifest["face_count"]
        expected_dim = 3 if entity == "volume" else 2
        dim_tags = [(dim, tag) for (dim, tag), group_name in physical_names.items() if group_name == name]
        if expected_count and not dim_tags:
            problems.append(f"required/populated group {name} has no $PhysicalNames entry in mesh.msh")
        mismatched_dim = [(dim, tag) for dim, tag in dim_tags if dim != expected_dim]
        if mismatched_dim:
            problems.append(f"group {name} is declared at dimension(s) {mismatched_dim}; expected dimension {expected_dim}")
        actual_count = sum(counts_by_tag.get((dim, tag), 0) for dim, tag in dim_tags)
        if actual_count != expected_count:
            problems.append(f"group {name}: manifest says {expected_count} elements; mesh.msh has {actual_count}")

    missing_required = [name for name in REQUIRED_GROUPS if not manifest["groups"].get(name, {}).get(
        "cell_count" if name == "WALL" else "face_count",
    )]
    if missing_required:
        problems.append(f"required groups are empty: {', '.join(missing_required)}")

    structure = build_combined_structure(root)
    try:
        audit_combined_structure(structure)
    except ValueError as error:
        problems.append(f"audit_combined_structure failed on a freshly built structure: {error}")

    degenerate = check_degenerate_cells(structure)
    problems.extend(degenerate)

    report = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "gmsh_path": str(gmsh_path),
        "checked_node_count": node_count,
        "problems": problems,
        "passed": not problems,
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if problems:
        raise MeshValidationError(
            f"Mesh validation failed ({len(problems)} problem(s)); see {report_path}:\n" + "\n".join(problems),
        )
    return report


def main() -> None:
    report = validate(ROOT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
