"""Work Package 7 of ``Aster_Porting.md``: compare a completed Code_Aster
elastic result against the frozen Elmer baseline.

Both solvers ultimately reference the SAME underlying undeformed node
array (the Code_Aster mesh was built directly from Elmer's raw node/cell
arrays by Aster/build_med_mesh.py + Aster/convert_arrays_to_med.py), but
their two result EXPORTS do not share row order or even row count:
Elmer's PVTU is split across MPI partitions and
Elmer.analyze_stress.parse_vtu concatenates each partition's own point
list without deduplicating, so shared partition-boundary nodes appear
twice (observed: 119150 Elmer rows vs 116708 Aster rows, a gap of exactly
2442 -- matching the number of duplicate-coordinate groups found in the
Elmer export). Elmer's VTU "points" DataArray is also the DEFORMED mesh,
not the undeformed reference geometry (Aster/freeze_elmer_baseline.py
subtracts displacement before freezing the archive so this module gets
undeformed coordinates from both solvers; verified by that script to
coincide to ~1e-10 m, i.e. floating-point noise). ``_match_by_coordinate``
below still builds an explicit coordinate-based correspondence (a
dependency-free spatial grid-bucket nearest-neighbour search, not naive
independent per-axis rounding, which was tried first and found to
misclassify points that straddle a rounding boundary on one array but not
the other) to handle the row-order/duplicate-row differences, and
verifies every match is within tolerance before any stress/displacement
diff is computed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Aster.build_med_mesh import DEFAULT_OUTPUT_DIR

# Acceptance threshold for global quantities (Aster_Porting.md,
# "Baseline Acceptance Criteria"): 2% initial target; larger gaps must be
# investigated, not averaged away.
TOLERANCE_FRACTION = 0.02
# Coordinates come from two independently-written mesh exports of the same
# underlying undeformed node array (Elmer's VTU, post displacement-removal,
# vs Code_Aster's MED). Both are full double-precision, so after matching
# the gap should be floating-point noise (~1e-10 m observed); the tolerance
# just needs to be far finer than the mesh's node spacing (order of 0.1-1m)
# so distinct nodes are never confused with each other.
GRID_CELL_SIZE_M = 0.01  # 1cm buckets: comfortably above floating-point noise
COORDINATE_MATCH_TOLERANCE_M = 1e-6


class ComparisonError(ValueError):
    """The Elmer/Code_Aster result fields cannot be meaningfully compared."""


def _match_by_coordinate(reference_points: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    """For each row of ``query_points``, find the nearest row of
    ``reference_points`` by true Euclidean distance using a spatial
    grid-bucket search (dependency-free; no scipy in this repo's .venv).
    Independent per-axis rounding was tried first and rejected: two points
    separated by ordinary floating-point/export noise can still round to
    DIFFERENT grid cells on each axis if they straddle a rounding boundary
    (observed on this mesh), silently producing a spurious "no match". This
    checks the real distance to every candidate in the 3x3x3 block of
    neighbouring cells instead, so a boundary-straddling pair is still
    found. Raises ComparisonError if any query point has no reference point
    within COORDINATE_MATCH_TOLERANCE_M."""
    cell_of = np.floor(reference_points / GRID_CELL_SIZE_M).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(map(tuple, cell_of.tolist())):
        buckets.setdefault(cell, []).append(index)

    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    query_cell_of = np.floor(query_points / GRID_CELL_SIZE_M).astype(np.int64)
    matches = np.empty(len(query_points), dtype=np.int64)
    for query_index, (cx, cy, cz) in enumerate(map(tuple, query_cell_of.tolist())):
        candidates: list[int] = []
        for dx, dy, dz in offsets:
            candidates.extend(buckets.get((cx + dx, cy + dy, cz + dz), ()))
        if not candidates:
            raise ComparisonError(
                f"No Elmer node within {GRID_CELL_SIZE_M} m of Aster node {query_index} "
                f"at {query_points[query_index].tolist()}; the shared-geometry assumption "
                "this comparison relies on does not hold for this pair of results.",
            )
        candidate_points = reference_points[candidates]
        distances = np.linalg.norm(candidate_points - query_points[query_index], axis=1)
        best = int(np.argmin(distances))
        if distances[best] > COORDINATE_MATCH_TOLERANCE_M:
            raise ComparisonError(
                f"Nearest Elmer node to Aster node {query_index} "
                f"at {query_points[query_index].tolist()} is {distances[best]} m away "
                f"(tolerance {COORDINATE_MATCH_TOLERANCE_M} m); the shared-geometry "
                "assumption this comparison relies on does not hold for this pair of results.",
            )
        matches[query_index] = candidates[best]
    return matches


def _peak_principal_stresses(points: np.ndarray, stress: np.ndarray) -> tuple[dict, dict]:
    """Per-node principal stress extrema (not per-element, unlike
    Elmer.freeze_elmer_baseline.recompute_peak_principal_stresses, since
    Code_Aster's SIGM_NOEU/the extracted field here is already nodal, not
    per-element-with-connectivity). Reports the node index, not an element
    id, with the same field names as elmer_baseline.json's "centroid_xyz_m"
    for a like-for-like diff (a single node here, not an element centroid).
    Vectorised (batched numpy.linalg.eigvalsh over all nodes at once)
    rather than a per-node Python loop, since there are ~1e5+ nodes.

    Nodes with a NaN stress tensor (Aster/extract_result_fields.py can leave
    a handful of far-field nodes uncovered when SIGM_NOEU's support is a
    strict subset of the full node set -- see that module's docstring) are
    excluded from the argmax/argmin via nan-aware reductions so a handful
    of NaNs can't silently win "peak" by NaN-propagation/comparison quirks.
    """
    sxx, syy, szz, sxy, syz, sxz = stress.T
    tensors = np.empty((len(stress), 3, 3), dtype=np.float64)
    tensors[:, 0, 0], tensors[:, 1, 1], tensors[:, 2, 2] = sxx, syy, szz
    tensors[:, 0, 1] = tensors[:, 1, 0] = sxy
    tensors[:, 0, 2] = tensors[:, 2, 0] = sxz
    tensors[:, 1, 2] = tensors[:, 2, 1] = syz
    valid = ~np.isnan(stress).any(axis=1)
    principal = np.full((len(stress), 3), np.nan, dtype=np.float64)
    principal[valid] = np.linalg.eigvalsh(tensors[valid])  # ascending order per row, shape (N, 3)

    tension_node = int(np.nanargmax(principal[:, 2]))
    compression_node = int(np.nanargmin(principal[:, 0]))
    max_tension = {
        "node_id": tension_node,
        "value_pa": float(principal[tension_node, 2]),
        "centroid_xyz_m": points[tension_node].tolist(),
    }
    max_compression = {
        "node_id": compression_node,
        "value_pa": float(principal[compression_node, 0]),
        "centroid_xyz_m": points[compression_node].tolist(),
    }
    return max_tension, max_compression


def _percent_diff(aster_value: float, elmer_value: float) -> float:
    if elmer_value == 0.0:
        return float("inf") if aster_value != 0.0 else 0.0
    return (aster_value - elmer_value) / abs(elmer_value)


def compare(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    elmer_path = output_dir / "elmer_baseline.json"
    aster_run_path = output_dir / "run_manifest.json"

    if not elmer_path.is_file():
        raise FileNotFoundError(f"Run Aster.freeze_elmer_baseline first; missing {elmer_path}")
    elmer = json.loads(elmer_path.read_text(encoding="utf-8"))

    if not aster_run_path.is_file():
        return _blocked_report(output_dir, "Aster.run_case has not been run yet (missing run_manifest.json).")

    run_manifest = json.loads(aster_run_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "completed":
        return _blocked_report(
            output_dir,
            f"Aster.run_case status is {run_manifest.get('status')!r}, not 'completed'; "
            "no Code_Aster result is available to compare against the frozen Elmer baseline. "
            f"{run_manifest.get('message', '')}",
        )

    aster_fields_path = run_manifest.get("result_fields_path")
    if not aster_fields_path or not Path(aster_fields_path).is_file():
        return _blocked_report(
            output_dir,
            "Aster.run_case completed but result_fields_path is missing; "
            "Aster/extract_result_fields.py did not run or failed.",
        )

    elmer_fields_path = output_dir / elmer["field_archive"]
    if not elmer_fields_path.is_file():
        raise FileNotFoundError(f"Missing Elmer baseline field archive: {elmer_fields_path}")

    elmer_fields = np.load(elmer_fields_path)
    aster_fields = np.load(aster_fields_path)
    scope = elmer.get("reporting_scope", "legacy_wall")
    aster_scope = str(aster_fields["reporting_scope"]) if "reporting_scope" in aster_fields else "legacy_wall"
    if scope != aster_scope:
        return _blocked_report(output_dir, "Reporting scopes differ; regenerate both concrete field archives.")

    elmer_points_raw = elmer_fields["points_m"]
    aster_points = aster_fields["points_m"]
    # Elmer's PVTU is split across MPI partitions and Elmer.analyze_stress.parse_vtu
    # concatenates each partition's own point list without deduplicating -- shared
    # partition-boundary nodes appear twice. Code_Aster's mesh was built directly
    # from the single canonical node array (Aster/build_med_mesh.py), so it has no
    # duplicates; match by true nearest coordinate (not raw array index) to build a
    # correspondence that is robust to both the duplicates and any row-order
    # differences between the two exports, rather than assuming the arrays are
    # already aligned index-for-index.
    if ("source_node_ids" in aster_fields) != ("source_node_ids" in elmer_fields):
        raise ComparisonError("Source identity is missing from one archive; regenerate both comparison archives")
    if "source_node_ids" in aster_fields and "source_node_ids" in elmer_fields:
        source_ids = elmer_fields["source_node_ids"]
        ids, first = np.unique(source_ids, return_index=True)
        positions = np.searchsorted(ids, aster_fields["source_node_ids"])
        if (positions >= len(ids)).any() or not np.array_equal(ids[positions], aster_fields["source_node_ids"]):
            raise ComparisonError("Concrete source node identities do not match")
        elmer_index_for_aster = first[positions]
    else:
        elmer_index_for_aster = _match_by_coordinate(elmer_points_raw, aster_points)
    elmer_points = elmer_points_raw[elmer_index_for_aster]
    max_coord_gap_m = float(np.max(np.abs(elmer_points - aster_points)))
    if max_coord_gap_m > COORDINATE_MATCH_TOLERANCE_M:
        raise ComparisonError("Matched source node identities have inconsistent coordinates")

    elmer_disp = elmer_fields["displacement_m"]
    aster_disp = aster_fields["displacement_m"]
    elmer_disp_norm = np.linalg.norm(elmer_disp, axis=1)
    aster_disp_norm = np.linalg.norm(aster_disp, axis=1)
    elmer_max_disp_node = int(elmer_disp_norm.argmax())
    aster_max_disp_node = int(aster_disp_norm.argmax())
    # Elmer's displacement at the SAME physical node as Aster's peak, via the
    # coordinate-matched index map (not elmer_max_disp_node, which may be a
    # different node/location entirely).
    elmer_disp_norm_at_aster_peak = float(elmer_disp_norm[elmer_index_for_aster[aster_max_disp_node]])

    aster_tension, aster_compression = _peak_principal_stresses(aster_points, aster_fields["stress_pa"])

    displacement_comparison = {
        "elmer_max_displacement_m": {
            "value_m": float(elmer_disp_norm[elmer_max_disp_node]),
            "node_id": elmer_max_disp_node,
            "xyz_m": elmer_points_raw[elmer_max_disp_node].tolist(),
        },
        "aster_max_displacement_m": {
            "value_m": float(aster_disp_norm[aster_max_disp_node]),
            "node_id": aster_max_disp_node,
            "xyz_m": aster_points[aster_max_disp_node].tolist(),
        },
        "percent_diff_at_aster_peak_node": _percent_diff(
            float(aster_disp_norm[aster_max_disp_node]), elmer_disp_norm_at_aster_peak,
        ),
    }

    tensile_comparison = {
        "elmer": elmer["max_tensile_principal_stress_pa"],
        "aster": aster_tension,
        "percent_diff": _percent_diff(
            aster_tension["value_pa"], elmer["max_tensile_principal_stress_pa"]["value_pa"],
        ),
    }
    compressive_comparison = {
        "elmer": elmer["max_compressive_principal_stress_pa"],
        "aster": aster_compression,
        "percent_diff": _percent_diff(
            aster_compression["value_pa"], elmer["max_compressive_principal_stress_pa"]["value_pa"],
        ),
    }

    flags = []
    if abs(displacement_comparison["percent_diff_at_aster_peak_node"]) > TOLERANCE_FRACTION:
        flags.append("peak_displacement_exceeds_tolerance")
    if abs(tensile_comparison["percent_diff"]) > TOLERANCE_FRACTION:
        flags.append("max_tensile_stress_exceeds_tolerance")
    if abs(compressive_comparison["percent_diff"]) > TOLERANCE_FRACTION:
        flags.append("max_compressive_stress_exceeds_tolerance")

    report = {
        "schema_version": 1,
        "reporting_scope": scope or "legacy_wall",
        "status": "ok",
        "tolerance_fraction": TOLERANCE_FRACTION,
        "max_node_coordinate_gap_m": max_coord_gap_m,
        "node_count": int(len(elmer_points)),
        "displacement": displacement_comparison,
        "max_tensile_principal_stress_pa": tensile_comparison,
        "max_compressive_principal_stress_pa": compressive_comparison,
        "flags_outside_tolerance": flags,
        "note": (
            ("Both solvers report principal stresses of recovered nodal tensors. "
               if scope == "concrete" else
               "Code_Aster's peak principal stresses are per-node (SIGM_NOEU/SIEQ_NOEU "
            "extrapolated to nodes); Elmer's are per-element (mean nodal stress at each "
            "element's own nodes) -- a real, expected methodological difference, not a bug, "
            "that can by itself produce gaps beyond TOLERANCE_FRACTION near stress "
            "concentrations/supports. ")
            + "Differences outside tolerance require investigation; agreement is not validation."
        ),
    }
    report_path = output_dir / "comparison_report.json"
    if "source_node_ids" in aster_fields and "body_1_mask" in aster_fields:
        from Aster.parity_checks import concrete_checks
        report.update(concrete_checks(elmer_fields, aster_fields, elmer_index_for_aster))
        report["stress_reporting"] = "Separate wall and optional concrete foundation; no combined stress peak."
        report["tensile_peak_comparison_by_region"] = {
            name: region["tensile_peak_comparison"] for name, region in report["regions"].items()
        }
        report.pop("max_tensile_principal_stress_pa")
        report.pop("max_compressive_principal_stress_pa")
        regional_flags = [
            flag for flag in flags if flag == "peak_displacement_exceeds_tolerance"
        ]
        for name, region in report["regions"].items():
            for measure in ("tension", "compression"):
                if abs(_percent_diff(region[f"aster_{measure}_pa"]["value_pa"],
                                     region[f"elmer_{measure}_pa"]["value_pa"])) > TOLERANCE_FRACTION:
                    regional_flags.append(f"{name}_max_{measure}_stress_exceeds_tolerance")
        report["flags_outside_tolerance"] = regional_flags
    report["baseline_verified"] = False
    report["acceptance_status"] = "requires_verification"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def _blocked_report(output_dir: Path, reason: str) -> dict:
    report = {
        "schema_version": 1,
        "status": "blocked",
        "reason": reason,
        "tolerance_fraction": TOLERANCE_FRACTION,
        "note": (
            "This is a comparison skeleton, not a result. No Code_Aster installation was "
            "available while building this tooling, so no real elastic cross-check has run. "
            "See Aster_Porting.md for the full acceptance-criteria checklist this script "
            "will evaluate once a real Code_Aster result exists."
        ),
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = compare()
    print(json.dumps(report, indent=2))
    if report.get("status") != "ok":
        sys.exit(2)


if __name__ == "__main__":
    main()
