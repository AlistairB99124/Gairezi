"""Export the active bonded wall mesh with named Code_Aster groups.

Work Package 2/3 of ``Aster_Porting.md``. The canonical region/face group
contract is ``Elmer/regions/model_groups.py``; this module only consumes it
(``build_named_groups``, ``REGION_GROUP_IDS``, ``FACE_GROUP_IDS``), it never
re-derives group membership from geometry.

Mesh format note
-----------------
MED is HDF5-based. Writing it here requires the optional ``h5py`` dependency,
which currently has no prebuilt wheel for the Python 3.15 interpreter used by
this repository's virtual environment (building it from source needs a local
HDF5 toolchain that is not installed). Rather than block the whole port on
that packaging gap, this exporter always writes a Gmsh 2.2 ASCII mesh with an
embedded ``$PhysicalNames`` section (``mesh.msh``), which Code_Aster reads
natively (``LIRE_MAILLAGE(FORMAT='GMSH', ...)`` or via ``PRE_GMSH``) and which
carries the same named groups a MED file would. When ``h5py`` is importable
(for example under Code_Aster's own bundled Python/Salome environment) it
additionally writes ``mesh.med`` via ``meshio``. The manifest records which
formats were actually produced so downstream tooling never has to guess.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Elmer.regions.combined_structure import CombinedStructure
from Elmer.regions.model_groups import (
    FACE_GROUP_IDS,
    GROUP_ENTITIES,
    REGION_GROUP_IDS,
    REQUIRED_GROUPS,
    is_wall_joint_boundary_id,
)
from Elmer.regions.tetrahedral import _signed_volume, split_cell, split_face
from Elmer.structure_assembly import apply_load_case
from Elmer.structural_model import (
    audit_combined_structure,
    build_combined_structure,
    build_named_groups,
)

WALL_JOINT_PHYSICAL_TAG = 100
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "00_elastic_baseline"


def _physical_tag(group_name: str, group_entities: dict[str, str]) -> tuple[int, int]:
    """(dimension, physical tag) Gmsh expects for a named group."""
    if group_entities[group_name] == "volume":
        return 3, REGION_GROUP_IDS[group_name]
    if group_name == "WALL_JOINT":
        return 2, WALL_JOINT_PHYSICAL_TAG
    return 2, FACE_GROUP_IDS[group_name]


def _face_area_m2(nodes: list[tuple[float, float, float]], face: tuple[int, ...]) -> float:
    area = 0.0
    for a, b, c in split_face(tuple(face)):
        p0, p1, p2 = nodes[a], nodes[b], nodes[c]
        u = [p1[i] - p0[i] for i in range(3)]
        v = [p2[i] - p0[i] for i in range(3)]
        cross = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        area += 0.5 * (cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) ** 0.5
    return area


def _cell_volume_m3(nodes: list[tuple[float, float, float]], element_type: int, cell: tuple[int, ...]) -> float:
    return sum(
        abs(_signed_volume(nodes, tetrahedron))
        for tetrahedron in split_cell(nodes, element_type, cell)
    )


def _bounds(nodes: list[tuple[float, float, float]], indices: list[int] | None = None) -> dict[str, list[float]]:
    points = nodes if indices is None else [nodes[index] for index in indices]
    if not points:
        return {"min_xyz_m": None, "max_xyz_m": None}
    return {
        "min_xyz_m": [min(point[axis] for point in points) for axis in range(3)],
        "max_xyz_m": [max(point[axis] for point in points) for axis in range(3)],
    }


def write_named_gmsh(
    structure: CombinedStructure,
    groups: dict[str, tuple[int, ...]],
    path: Path,
) -> None:
    """Gmsh 2.2 ASCII with $PhysicalNames; physical tags match the numeric
    region/boundary ids already carried by the structure, so this is a
    relabeling, not a re-derivation, of group membership."""
    group_of_cell: dict[int, str] = {}
    for name, entity in GROUP_ENTITIES.items():
        if entity != "volume":
            continue
        for cell_index in groups.get(name, ()):
            group_of_cell[cell_index] = name
    group_of_face: dict[int, str] = {}
    for name, entity in GROUP_ENTITIES.items():
        if entity != "face":
            continue
        for face_index in groups.get(name, ()):
            group_of_face[face_index] = name

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        physical_names = sorted(
            {_physical_tag(name, GROUP_ENTITIES) for name in groups if groups[name]},
        )
        stream.write("$PhysicalNames\n")
        stream.write(f"{len(physical_names)}\n")
        tag_to_name = {
            (_physical_tag(name, GROUP_ENTITIES)): name for name in groups if groups[name]
        }
        for dim, tag in physical_names:
            stream.write(f'{dim} {tag} "{tag_to_name[(dim, tag)]}"\n')
        stream.write("$EndPhysicalNames\n")
        stream.write("$Nodes\n")
        stream.write(f"{len(structure.nodes)}\n")
        for node_id, (x_m, y_m, z_m) in enumerate(structure.nodes, start=1):
            stream.write(f"{node_id} {x_m:.12g} {y_m:.12g} {z_m:.12g}\n")
        stream.write("$EndNodes\n$Elements\n")
        stream.write(f"{len(structure.cells) + len(structure.boundaries)}\n")
        element_id = 1
        for cell_index, (element_type, cell) in enumerate(structure.cells):
            region_id = structure.region_ids[cell_index]
            node_ids = " ".join(str(node_id + 1) for node_id in cell)
            stream.write(f"{element_id} {element_type} 2 {region_id} {region_id} {node_ids}\n")
            element_id += 1
        for face_index, (boundary_id, face) in enumerate(structure.boundaries):
            element_type = 2 if len(face) == 3 else 3
            tag = WALL_JOINT_PHYSICAL_TAG if is_wall_joint_boundary_id(boundary_id) else boundary_id
            node_ids = " ".join(str(node_id + 1) for node_id in face)
            stream.write(f"{element_id} {element_type} 2 {tag} {tag} {node_ids}\n")
            element_id += 1
        stream.write("$EndElements\n")


def try_write_med(structure: CombinedStructure, groups: dict[str, tuple[int, ...]], path: Path) -> bool:
    """Best-effort MED write; returns False (without raising) if h5py is unavailable."""
    try:
        import h5py  # noqa: F401
        import meshio
    except ImportError:
        return False
    vtk_cell_type = {4: "tetra", 5: "hexahedron", 6: "wedge", 7: "pyramid"}
    cells_by_type: dict[str, list[tuple[int, ...]]] = {}
    cell_group_sets: dict[str, dict[str, list[int]]] = {}
    for cell_index, (element_type, cell) in enumerate(structure.cells):
        kind = vtk_cell_type[element_type]
        cells_by_type.setdefault(kind, [])
        local_index = len(cells_by_type[kind])
        cells_by_type[kind].append(cell)
        name = next((name for name, entity in GROUP_ENTITIES.items() if entity == "volume" and cell_index in groups.get(name, ())), None)
        if name is not None:
            cell_group_sets.setdefault(kind, {}).setdefault(name, []).append(local_index)
    face_cell_type = {3: "triangle", 4: "quad"}
    for face_index, (_boundary_id, face) in enumerate(structure.boundaries):
        kind = face_cell_type[len(face)]
        cells_by_type.setdefault(kind, [])
        local_index = len(cells_by_type[kind])
        cells_by_type[kind].append(face)
        name = next((name for name, entity in GROUP_ENTITIES.items() if entity == "face" and face_index in groups.get(name, ())), None)
        if name is not None:
            cell_group_sets.setdefault(kind, {}).setdefault(name, []).append(local_index)
    mesh = meshio.Mesh(
        points=structure.nodes,
        cells=[(kind, cells) for kind, cells in cells_by_type.items()],
        cell_sets={
            name: [cell_group_sets.get(kind, {}).get(name, []) for kind in cells_by_type]
            for name in groups
            if groups[name]
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(str(path), mesh, file_format="med")
    return True


def build_manifest(structure: CombinedStructure, groups: dict[str, tuple[int, ...]]) -> dict[str, object]:
    missing_required = [name for name in REQUIRED_GROUPS if not groups.get(name)]
    if missing_required:
        raise ValueError(f"Required mesh groups are empty: {', '.join(missing_required)}")
    group_manifest: dict[str, object] = {}
    for name, entity in GROUP_ENTITIES.items():
        indices = groups.get(name, ())
        if entity == "volume":
            cells = [structure.cells[index] for index in indices]
            node_ids = sorted({node_id for _, cell in cells for node_id in cell})
            volume_m3 = sum(_cell_volume_m3(structure.nodes, element_type, cell) for element_type, cell in cells)
            group_manifest[name] = {
                "entity": "volume",
                "cell_count": len(indices),
                "node_count": len(node_ids),
                "volume_m3": volume_m3,
                **_bounds(structure.nodes, node_ids),
            }
        else:
            faces = [structure.boundaries[index][1] for index in indices]
            node_ids = sorted({node_id for face in faces for node_id in face})
            area_m2 = sum(_face_area_m2(structure.nodes, face) for face in faces)
            group_manifest[name] = {
                "entity": "face",
                "face_count": len(indices),
                "node_count": len(node_ids),
                "area_m2": area_m2,
                **_bounds(structure.nodes, node_ids),
            }
    return {
        "schema_version": 1,
        "source": "Elmer.structural_model.build_combined_structure (monolithic bonded wall, no split joints)",
        "node_count": len(structure.nodes),
        "cell_count": len(structure.cells),
        "boundary_face_count": len(structure.boundaries),
        "element_types": {
            "hexahedra": sum(1 for element_type, _ in structure.cells if element_type == 5),
            "triangular_prisms": sum(1 for element_type, _ in structure.cells if element_type == 6),
        },
        "coordinate_bounds": _bounds(structure.nodes),
        "coordinate_convention": "metres; vertical axis is Z; crest at Z=0",
        "groups": group_manifest,
        "required_groups": list(REQUIRED_GROUPS),
        "material": {
            "density_kg_m3": structure.material.density_kg_m3,
            "youngs_modulus_pa": structure.material.youngs_modulus_pa,
            "poissons_ratio": structure.material.poissons_ratio,
            "tensile_strength_pa": structure.material.tensile_strength_pa,
        },
        "foundation_support_n_per_m3": {
            "x": structure.foundation_support.spring_x_n_per_m3,
            "y": structure.foundation_support.spring_y_n_per_m3,
            "z": structure.foundation_support.spring_z_n_per_m3,
        },
    }


def write_mesh_arrays(structure: CombinedStructure, groups: dict[str, tuple[int, ...]], path: Path) -> None:
    """Dump node coordinates plus per-type cell/face connectivity (and the named
    group each one belongs to) as a plain-numpy ``.npz``. This has no ``h5py``
    dependency, unlike a true ``.med`` write, so it works in this repo's Python
    3.15 ``.venv``. A separate script running under Code_Aster's own bundled
    Python (which has ``medcoupling`` but not a standalone importable ``h5py``)
    reads this file and builds the real ``.med`` via the ``medcoupling`` API
    instead of re-parsing Gmsh text, since Code_Aster's native ``LIRE_MAILLAGE
    (FORMAT='GMSH')`` reader was observed to take >9 minutes on this mesh size
    without finishing (see Aster_Porting.md amendment)."""
    import numpy as np

    volume_group_of_cell: dict[int, str] = {}
    for name, entity in GROUP_ENTITIES.items():
        if entity != "volume":
            continue
        for cell_index in groups.get(name, ()):
            volume_group_of_cell[cell_index] = name
    face_group_of_face: dict[int, str] = {}
    for name, entity in GROUP_ENTITIES.items():
        if entity != "face":
            continue
        for face_index in groups.get(name, ()):
            face_group_of_face[face_index] = name

    hex8, hex8_group = [], []
    wedge6, wedge6_group = [], []
    for cell_index, (element_type, cell) in enumerate(structure.cells):
        group_name = volume_group_of_cell.get(cell_index, "")
        if element_type == 5:
            hex8.append(cell)
            hex8_group.append(group_name)
        elif element_type == 6:
            wedge6.append(cell)
            wedge6_group.append(group_name)
        else:
            raise ValueError(f"Unsupported volume element_type={element_type} at cell {cell_index}")

    tri3, tri3_group = [], []
    quad4, quad4_group = [], []
    for face_index, (_boundary_id, face) in enumerate(structure.boundaries):
        group_name = face_group_of_face.get(face_index, "")
        if len(face) == 3:
            tri3.append(face)
            tri3_group.append(group_name)
        elif len(face) == 4:
            quad4.append(face)
            quad4_group.append(group_name)
        else:
            raise ValueError(f"Unsupported boundary face with {len(face)} nodes at face {face_index}")

    # RIGI_PARASOL (Code_Aster's distributed-foundation-spring mechanism; see
    # generate_elastic_case.py) needs an explicit POI1 (single-node) element
    # at every FOUNDATION surface node, one group covering all of them, so it
    # can distribute the per-area spring stiffness onto them weighted by
    # tributary area. These aren't present in the Gmsh/MED exports otherwise.
    foundation_faces = [structure.boundaries[index][1] for index in groups.get("FOUNDATION", ())]
    poi1_nodes = np.asarray(sorted({node_id for face in foundation_faces for node_id in face}), dtype=np.int64)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        nodes=np.asarray(structure.nodes, dtype=np.float64),
        hex8=np.asarray(hex8, dtype=np.int64).reshape(-1, 8),
        hex8_group=np.asarray(hex8_group, dtype=object),
        wedge6=np.asarray(wedge6, dtype=np.int64).reshape(-1, 6),
        wedge6_group=np.asarray(wedge6_group, dtype=object),
        tri3=np.asarray(tri3, dtype=np.int64).reshape(-1, 3),
        tri3_group=np.asarray(tri3_group, dtype=object),
        quad4=np.asarray(quad4, dtype=np.int64).reshape(-1, 4),
        quad4_group=np.asarray(quad4_group, dtype=object),
        foundation_poi1_nodes=poi1_nodes,
    )


def export(root: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    structure = build_combined_structure(root)
    apply_load_case(structure, root / "config.json")
    audit_combined_structure(structure)
    groups = build_named_groups(structure)
    manifest = build_manifest(structure, groups)

    output_dir.mkdir(parents=True, exist_ok=True)
    gmsh_path = output_dir / "mesh.msh"
    write_named_gmsh(structure, groups, gmsh_path)
    med_path = output_dir / "mesh.med"
    med_written = try_write_med(structure, groups, med_path)
    arrays_path = output_dir / "mesh_arrays.npz"
    write_mesh_arrays(structure, groups, arrays_path)

    manifest["mesh_formats_written"] = ["gmsh_named"] + (["med"] if med_written else [])
    manifest["gmsh_path"] = str(gmsh_path)
    manifest["med_path"] = str(med_path) if med_written else None
    manifest["mesh_arrays_path"] = str(arrays_path)
    if not med_written:
        manifest["med_unavailable_reason"] = (
            "h5py has no prebuilt wheel for this interpreter; MED export needs a Code_Aster/"
            "Salome Python environment with h5py/medcoupling available, or a local HDF5 build. "
            "mesh_arrays.npz is written instead for Aster/convert_arrays_to_med.py to consume "
            "under Code_Aster's own bundled Python (medcoupling only, no h5py needed)."
        )
    manifest["gmsh_sha256"] = hashlib.sha256(gmsh_path.read_bytes()).hexdigest()
    manifest_path = output_dir / "mesh_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    manifest = export(ROOT)
    print(json.dumps({key: manifest[key] for key in (
        "node_count", "cell_count", "boundary_face_count", "mesh_formats_written", "gmsh_path", "manifest_path",
    )}, indent=2))


if __name__ == "__main__":
    main()
