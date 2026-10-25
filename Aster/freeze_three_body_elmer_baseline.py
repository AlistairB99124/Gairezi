"""Three-body counterpart of ``Aster/freeze_elmer_baseline.py``.

The legacy ``freeze_elmer_baseline.py``/``00_elastic_baseline`` pair only
ever compared Code_Aster against the single-WALL spring-surrogate Elmer
model (``Elmer/structure_output/combined_solver``), which is checked into
this repo. The real, current app/API/CLI model is
``Elmer.three_body_solver`` (WALL+FOUNDATION+BEDROCK); its solved output
lives in an ephemeral per-run job directory under the user's local app data
(``<...>/three-body-solver/<run-id>/``), not in this repo, so there is no
fixed path to hardcode the way the legacy script does.

Instead, this module locates the exact Elmer run that was used to build the
Aster three-body mesh by reading back ``Aster/build_three_body_mesh_arrays.py``'s
own ``mesh_manifest.json`` (written by ``Aster/convert_three_body_to_med.py``),
which already records the absolute ``source_mesh`` path
(``<run>/model/three_body.msh``). This guarantees the Elmer result frozen
here is the SAME solve the Aster mesh was derived from -- not just "a"
three-body run that happens to be lying around -- so the comparison is a
genuine like-for-like cross-check.

Unlike the legacy script, this one does NOT verify the recomputed peak
stress against a stored ``stress_summary.json``: the three-body pipeline's
own reporting step (``Elmer.three_body_solver``'s
``model/reporting/stress_summary.json``) reports per-body, sharp-edge-
excluded, near-rock-clipped peaks (a deliberately different, narrower
methodology -- see that file's own ``limitations`` field), not the simple
whole-domain extremum this module recomputes directly from the raw PVTU.
Comparing the two would be comparing different things, not confirming/
denying freshness, so the stored summary is recorded for reference only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Elmer.analyze_stress does a bare "import kaleido_safety" (a module that
# lives inside Elmer/, not a package-qualified import), so Elmer/ itself must
# also be on sys.path, not just its parent. Elmer/structure_assembly.py does
# this same insertion as an import-time side effect, which is how the legacy
# Aster/freeze_elmer_baseline.py gets away without doing it explicitly (it
# imports Aster.build_med_mesh first, which pulls in Elmer.structure_assembly
# transitively); this module has no such transitive import, so it must do it
# directly.
ELMER_DIR = ROOT / "Elmer"
if str(ELMER_DIR) not in sys.path:
    sys.path.insert(0, str(ELMER_DIR))

from Aster.build_three_body_mesh_arrays import DEFAULT_OUTPUT_DIR
from Aster.freeze_elmer_baseline import (
    BaselineFreezeError,
    file_hash,
    pvtu_partition_paths,
)
from Elmer.analyze_stress import parse_vtu
from Aster.postprocess_baseline import _peak_principal_stresses


def _locate_solve_dir(output_dir: Path) -> Path:
    mesh_manifest_path = output_dir / "mesh_manifest.json"
    if not mesh_manifest_path.is_file():
        raise BaselineFreezeError(
            f"Missing {mesh_manifest_path}; run Aster.convert_three_body_to_med "
            "(via Aster.run_three_body_case or Aster.run_case) first so the exact "
            "source Elmer run is recorded.",
        )
    mesh_manifest = json.loads(mesh_manifest_path.read_text(encoding="utf-8"))
    source_mesh = Path(mesh_manifest["source_mesh"])
    if not source_mesh.is_file():
        raise BaselineFreezeError(
            f"{mesh_manifest_path} records source_mesh={source_mesh}, but that file no "
            "longer exists -- the Elmer run directory it came from was an ephemeral "
            "per-job output and has likely been cleaned up. Re-run Elmer.three_body_solver "
            "and Aster.convert_three_body_to_med before freezing a baseline against it.",
        )
    # source_mesh is "<run>/model/three_body.msh"; the run's results/reporting
    # live alongside "model", not inside it.
    return source_mesh.parent.parent


def freeze(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    solve_dir = _locate_solve_dir(output_dir)
    model_dir = solve_dir / "model"
    pvtu_path = model_dir / "results" / "three_body_results_t0001.pvtu"
    if not pvtu_path.is_file():
        raise BaselineFreezeError(f"Expected Elmer three-body baseline artifact is missing: {pvtu_path}")

    partitions = pvtu_partition_paths(pvtu_path)
    hashed_paths = [pvtu_path, *partitions]
    hashes = {str(path.resolve()): file_hash(path) for path in hashed_paths}

    points, displacement, stress, cells = parse_vtu(pvtu_path, body_ids=(1, 3))
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
        raise BaselineFreezeError("Elmer three-body baseline fields have invalid shapes or nonfinite values")

    # Same ResultOutputSolver behaviour as the legacy single-WALL case (see
    # Aster/freeze_elmer_baseline.py's docstring): the PVTU "points" DataArray
    # is the DEFORMED mesh, so subtract displacement to get the undeformed
    # reference geometry that Code_Aster's MED mesh actually matches.
    # Aster/postprocess_baseline.py's coordinate-matching step independently
    # verifies this is correct -- it raises ComparisonError if no Aster node
    # is found within 1e-6 m of each Elmer node, so a wrong sign/omission here
    # would fail loudly at comparison time, not silently produce a bogus diff.
    undeformed_points = points - displacement
    from Aster.build_three_body_mesh_arrays import _read_gmsh_ascii
    from Aster.parity_checks import source_node_ids
    source_points, source_cells, source_tags, _, _ = _read_gmsh_ascii(model_dir / "three_body.msh")
    source_cells = source_cells[np.isin(source_tags, (1, 3))]
    ids = source_node_ids(undeformed_points, cells, source_points, source_cells)

    max_tension, max_compression = _peak_principal_stresses(undeformed_points, stress)

    if any(file_hash(path) != hashes[str(path.resolve())] for path in hashed_paths):
        raise BaselineFreezeError("Elmer three-body baseline files changed while being frozen")

    output_dir.mkdir(parents=True, exist_ok=True)
    field_archive_name = "elmer_baseline_fields.npz"
    np.savez_compressed(
        output_dir / field_archive_name,
        points_m=undeformed_points, displacement_m=displacement, stress_pa=stress,
        reporting_scope="concrete",
        source_node_ids=ids,
    )

    manifest = {
        "schema_version": 1,
        "reporting_scope": "concrete",
        "stress_method": "recovered_nodal",
        "source_solve_dir": str(solve_dir.resolve()),
        "source_sha256": hashes,
        "node_count": int(len(points)),
        "element_count": len(cells),
        "coordinate_frame": "undeformed Elmer mesh; crest z=0",
        "units": {"coordinates": "m", "displacement": "m", "stress": "Pa"},
        "stress_components": ["xx", "yy", "zz", "xy", "yz", "xz"],
        "max_tensile_principal_stress_pa": max_tension,
        "max_compressive_principal_stress_pa": max_compression,
        "field_archive": field_archive_name,
        "recomputed_from_raw_vtu": True,
        "note": (
            "Wall and optional concrete foundation only. Principal stress extrema use "
            "recovered nodal tensors, matching the Code_Aster reporting measure. "
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
