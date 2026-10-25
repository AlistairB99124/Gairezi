"""Work Package 1 of ``Aster_Porting.md``: freeze the Elmer result being used
for corroboration so that a later Code_Aster comparison is against a fixed,
hashed, independently-recomputed snapshot rather than a result that could
silently change underneath it.

This does not re-run Elmer. It operates on the already-completed solve under
``Elmer/structure_output/combined_solver`` (mesh, SIF, PVTU + partitions,
``analysis/stress_summary.json``) and:

* hashes every input file it reads, so later tampering/staleness is detectable;
* re-parses the raw VTU fields with ``Elmer.analyze_stress.parse_vtu`` and
  independently recomputes the max tensile/compressive principal stress,
  rather than trusting the stored ``stress_summary.json`` verbatim, and
  fails loudly if the two disagree beyond floating-point tolerance;
* writes a compressed field archive (``elmer_baseline_fields.npz``) plus a
  JSON manifest (``elmer_baseline.json``) that the (not yet runnable)
  Code_Aster postprocessing step will diff against.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Aster.build_med_mesh import DEFAULT_OUTPUT_DIR
from Elmer.analyze_stress import parse_vtu, stress_tensor

DEFAULT_SOLVE_DIR = ROOT / "Elmer" / "structure_output" / "combined_solver"
RELATIVE_TOLERANCE = 1e-6


class BaselineFreezeError(ValueError):
    """The stored Elmer result is missing, inconsistent, or stale."""


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pvtu_partition_paths(pvtu_path: Path) -> list[Path]:
    manifest = pvtu_path.read_text()
    # Optional whitespace before "/>": the legacy combined_solver writer emits
    # '<Piece Source="..."/>' (no space) but Elmer.three_body_solver's writer
    # emits '<Piece Source="..." />' (one space) -- both are valid, equivalent
    # self-closing XML tags, just from two different ResultOutputSolver write
    # paths in this codebase.
    sources = re.findall(r'<Piece Source="([^"]+)"\s*/>', manifest)
    if not sources:
        raise BaselineFreezeError(f"{pvtu_path} references no partition pieces")
    paths = []
    for source in sources:
        partition = (pvtu_path.parent / source).resolve()
        if not partition.is_relative_to(pvtu_path.parent.resolve()):
            raise BaselineFreezeError(f"Partition path escapes result directory: {source!r}")
        if not partition.is_file():
            raise BaselineFreezeError(f"Missing Elmer partition file: {partition}")
        paths.append(partition)
    return paths


def recompute_peak_principal_stresses(points: np.ndarray, stress: np.ndarray, cells) -> tuple[dict, dict]:
    max_tension = {"element_id": -1, "value_pa": float("-inf"), "centroid_xyz_m": [0.0, 0.0, 0.0]}
    max_compression = {"element_id": -1, "value_pa": float("inf"), "centroid_xyz_m": [0.0, 0.0, 0.0]}
    for element_id, (_cell_type, node_ids) in enumerate(cells, start=1):
        coords = points[node_ids]
        mean_stress = stress[node_ids].mean(axis=0)
        principal = np.linalg.eigvalsh(stress_tensor(mean_stress))
        centroid = coords.mean(axis=0).tolist()
        if principal[2] > max_tension["value_pa"]:
            max_tension = {"element_id": element_id, "value_pa": float(principal[2]), "centroid_xyz_m": centroid}
        if principal[0] < max_compression["value_pa"]:
            max_compression = {"element_id": element_id, "value_pa": float(principal[0]), "centroid_xyz_m": centroid}
    return max_tension, max_compression


def freeze(solve_dir: Path = DEFAULT_SOLVE_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    mesh_path = solve_dir / "combined_structure.msh"
    sif_path = solve_dir / "dam_model.sif"
    pvtu_path = solve_dir / "results" / "combined_results_mpi_t0001.pvtu"
    stored_summary_path = solve_dir / "results" / "analysis" / "stress_summary.json"
    for path in (mesh_path, sif_path, pvtu_path, stored_summary_path):
        if not path.is_file():
            raise BaselineFreezeError(f"Expected Elmer baseline artifact is missing: {path}")

    partitions = pvtu_partition_paths(pvtu_path)
    hashed_paths = [mesh_path, sif_path, pvtu_path, stored_summary_path, *partitions]
    hashes = {str(path.resolve()): file_hash(path) for path in hashed_paths}

    points, displacement, stress, cells = parse_vtu(pvtu_path)
    if not cells:
        raise BaselineFreezeError(f"{pvtu_path} contains no volume elements")
    if not (
        points.ndim == 2 and points.shape[1] == 3
        and displacement.shape == points.shape
        and stress.shape == (len(points), 6)
        and np.isfinite(points).all()
        and np.isfinite(displacement).all()
        and np.isfinite(stress).all()
    ):
        raise BaselineFreezeError("Elmer baseline fields have invalid shapes or nonfinite values")

    # Elmer's VTU "points" DataArray is the DEFORMED mesh (Elmer's result-output
    # solver writes current, not reference, node positions when a displacement
    # field is active) -- NOT the undeformed reference geometry the manifest
    # below claims. Verified directly: subtracting displacement from the raw
    # VTU points makes every node coincide with Code_Aster's undeformed MED
    # mesh node to ~1e-10 m (floating-point noise), whereas leaving the raw
    # VTU points as-is produces per-node gaps up to the displacement magnitude
    # itself (~8mm on this mesh) against Code_Aster's undeformed mesh. Use the
    # undeformed coordinates everywhere below so this archive's
    # "coordinate_frame" claim is actually true and the Code_Aster comparison
    # can match nodes exactly.
    undeformed_points = points - displacement

    max_tension, max_compression = recompute_peak_principal_stresses(undeformed_points, stress, cells)

    stored_summary = json.loads(stored_summary_path.read_text(encoding="utf-8"))
    stored_tension = stored_summary["max_tensile_principal_stress_pa"]["value_pa"]
    stored_compression = stored_summary["max_compressive_principal_stress_pa"]["value_pa"]
    if not np.isclose(max_tension["value_pa"], stored_tension, rtol=RELATIVE_TOLERANCE):
        raise BaselineFreezeError(
            f"Recomputed max tensile stress {max_tension['value_pa']!r} Pa does not match "
            f"stored stress_summary.json value {stored_tension!r} Pa; the stored result is stale.",
        )
    if not np.isclose(max_compression["value_pa"], stored_compression, rtol=RELATIVE_TOLERANCE):
        raise BaselineFreezeError(
            f"Recomputed max compressive stress {max_compression['value_pa']!r} Pa does not match "
            f"stored stress_summary.json value {stored_compression!r} Pa; the stored result is stale.",
        )

    if any(file_hash(path) != hashes[str(path.resolve())] for path in hashed_paths):
        raise BaselineFreezeError("Elmer baseline files changed while being frozen")

    output_dir.mkdir(parents=True, exist_ok=True)
    field_archive_name = "elmer_baseline_fields.npz"
    np.savez_compressed(
        output_dir / field_archive_name,
        points_m=undeformed_points, displacement_m=displacement, stress_pa=stress,
    )

    manifest = {
        "schema_version": 1,
        "source_solve_dir": str(solve_dir.resolve()),
        "source_sha256": hashes,
        "node_count": int(len(points)),
        "element_count": len(cells),
        "coordinate_frame": "undeformed Elmer mesh; crest z=0",
        "units": {"coordinates": "m", "displacement": "m", "stress": "Pa"},
        "stress_components": ["xx", "yy", "zz", "xy", "yz", "xz"],
        "max_tensile_principal_stress_pa": max_tension,
        "max_compressive_principal_stress_pa": max_compression,
        "tensile_strength_pa": stored_summary.get("tensile_strength_pa"),
        "tension_utilization": stored_summary.get("tension_utilization"),
        "field_archive": field_archive_name,
        "recomputed_from_raw_vtu": True,
        "note": (
            "This is the Elmer linear-elastic baseline frozen for independent cross-check by "
            "Code_Aster (Aster_Porting.md)."
        ),
    }
    manifest_path = output_dir / "elmer_baseline.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    manifest = freeze()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
