"""Automatic whole-wall tensile screening of native intact FEniCSx cell stresses.

This is initial screening, not a predicted final crack set: it reports wall
cells whose unsmoothed maximum principal stress exceeds the configured intact
tensile strength in the verified intact equilibrium. It never alters
coefficients, launches cracking or substitutes recovered nodal stress.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

WALL_REGION = 1
SCREENING_VERSION = 1
# Detailed rows are listed for the strongest cells only; counts and source rows stay complete.
MAXIMUM_LISTED_CELLS = 500


def _cell_geometry(coordinates, cells):
    points = coordinates[cells]
    volumes = np.abs(np.linalg.det(points[:, 1:] - points[:, :1])) / 6
    return volumes, points.mean(axis=1)


FACET_CRITERION = "one-sided: max over the two owner cells of n.sigma.n on interior wall-wall facets"


def _wall_faces(tetrahedra, regions):
    """Interior wall-wall facets: sorted source triples and their two owner cells."""
    wall = np.flatnonzero(regions == WALL_REGION)
    local = np.array(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
    keys = np.sort(tetrahedra[wall][:, local].reshape(-1, 3), axis=1)
    owners = np.repeat(wall, 4)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    interior = counts == 2
    pairs = np.stack((owners[order][starts[interior]], owners[order][starts[interior] + 1]), axis=1)
    return unique[interior], pairs


def interior_wall_facet_side_stress(coordinates_m, tetrahedra, regions, cell_stress_pa):
    """n.sigma_cell.n from each owner cell of every interior wall-wall facet.

    ``cell_stress_pa`` holds one tensor per tetrahedron (rows of non-wall cells are unused).
    Returns (owner pairs (m, 2), side normal stress (m, 2)).
    """
    coordinates = np.asarray(coordinates_m, dtype=float)
    faces, pairs = _wall_faces(np.asarray(tetrahedra), np.asarray(regions))
    if not len(faces):
        return pairs, np.zeros((0, 2))
    points = coordinates[faces]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    stress = np.asarray(cell_stress_pa, dtype=float)
    sides = np.stack([np.einsum("fi,fij,fj->f", normals, stress[pairs[:, k]], normals) for k in (0, 1)], axis=1)
    return pairs, sides


def interior_wall_facet_normal_stress(coordinates_m, tetrahedra, regions, cell_stress_pa):
    """Network initiation measure (one-sided): max of the two owner cells' n.sigma.n per facet.

    Returns (owner pairs (m, 2), facet normal stress (m,)).
    """
    pairs, sides = interior_wall_facet_side_stress(coordinates_m, tetrahedra, regions, cell_stress_pa)
    return pairs, (sides.max(axis=1) if len(sides) else np.zeros(0))


def facet_check(coordinates_m, tetrahedra, regions, cell_stress_pa, tensile_strength_pa, relative_tolerance):
    """Intact facet tractions with the cracking runner's initiation rule (and the two-cell average for reference)."""
    pairs, sides = interior_wall_facet_side_stress(coordinates_m, tetrahedra, regions, cell_stress_pa)
    one_sided = sides.max(axis=1) if len(sides) else np.zeros(0)
    average = sides.mean(axis=1) if len(sides) else np.zeros(0)
    critical = one_sided >= float(tensile_strength_pa) * (1 - float(relative_tolerance))
    peak = float(one_sided.max(initial=-np.inf)) if len(one_sided) else 0.
    return pairs, one_sided, critical, {
        "criterion": FACET_CRITERION,
        "interior_wall_facets": int(len(one_sided)),
        "peak_facet_normal_stress_pa": peak,
        "peak_ratio_to_strength": peak / float(tensile_strength_pa),
        "critical_facets": int(critical.sum()),
        "peak_two_cell_average_facet_normal_stress_pa": float(average.max(initial=0.)) if len(average) else 0.,
        "initiates": bool(critical.any()),
    }


def screen_cells(cell_stress_pa, cell_region_tags, source_cell_rows, coordinates_m, geometry_cells,
                 tensile_strength_pa, *, relative_tolerance: float) -> dict:
    """Classify wall cells by max principal stress against f_t (tension positive)."""
    stress = np.asarray(cell_stress_pa, dtype=float)
    regions = np.asarray(cell_region_tags)
    rows = np.asarray(source_cell_rows)
    coordinates = np.asarray(coordinates_m, dtype=float)
    cells = np.asarray(geometry_cells)
    count = len(stress)
    if (stress.shape != (count, 3, 3) or regions.shape != (count,) or rows.shape != (count,)
            or cells.shape != (count, 4) or regions.dtype.kind not in "iu" or rows.dtype.kind not in "iu"
            or cells.dtype.kind not in "iu" or coordinates.ndim != 2 or coordinates.shape[1] != 3
            or not np.isfinite(stress).all() or not np.isfinite(coordinates).all()
            or np.any(cells < 0) or np.any(cells >= len(coordinates))):
        raise ValueError("Native cell stress, regions, source rows and geometry must match in local cell order")
    if np.any(rows < 0):
        raise ValueError("Native source rows must be nonnegative")
    if not np.allclose(stress, np.swapaxes(stress, 1, 2), rtol=1e-10,
                       atol=1e-9 * max(float(np.abs(stress).max(initial=0.)), 1.)):
        raise ValueError("Native cell stress tensors must be symmetric")
    if (isinstance(tensile_strength_pa, bool) or not np.isfinite(tensile_strength_pa)
            or tensile_strength_pa <= 0):
        raise ValueError("Tensile strength must be finite and positive")
    if (isinstance(relative_tolerance, bool) or not np.isfinite(relative_tolerance)
            or not 0 < relative_tolerance < 1):
        raise ValueError("relative_tolerance must lie in (0, 1)")
    wall = np.flatnonzero(regions == WALL_REGION)
    if not len(wall):
        raise ValueError("No wall-concrete cells are present in the native mesh")
    eigen, vectors = np.linalg.eigh(.5 * (stress[wall] + np.swapaxes(stress[wall], 1, 2)))
    principal = eigen[:, -1]
    volumes, centroids = _cell_geometry(coordinates, cells[wall])
    if np.any(volumes <= 0):
        raise ValueError("Degenerate native wall tetrahedra")
    limit = float(tensile_strength_pa) * (1 + relative_tolerance)
    over = principal > limit
    near = ~over & (principal >= float(tensile_strength_pa) * (1 - relative_tolerance))
    peak = int(np.argmax(principal))
    order = np.argsort(-principal[over], kind="stable")
    over_ids = np.flatnonzero(over)[order]
    _, _, _, facets = facet_check(coordinates, cells, regions, stress, tensile_strength_pa, relative_tolerance)
    return {
        "screening_version": SCREENING_VERSION,
        "status": "overstressed_cells_found" if over.any() else "no_overstressed_cells",
        "criterion": "native unsmoothed cell maximum principal stress > f_t * (1 + relative_tolerance)",
        "stress_association": "cellwise intact FEniCSx equilibrium tensors; not recovered nodal Elmer stress",
        "tensile_strength_pa": float(tensile_strength_pa),
        "relative_tolerance": float(relative_tolerance),
        "wall_cells": int(len(wall)), "wall_volume_m3": float(volumes.sum()),
        "overstressed_cells": int(over.sum()),
        "overstressed_source_rows": np.unique(rows[wall][over]).tolist(),
        "overstressed_volume_m3": float(volumes[over].sum()),
        "near_critical_cells": int(near.sum()),
        "peak_principal_stress_pa": float(principal[peak]),
        "peak_ratio_to_strength": float(principal[peak] / tensile_strength_pa),
        "peak_location_m": centroids[peak].tolist(),
        "peak_source_row": int(rows[wall][peak]),
        "peak_principal_direction": vectors[peak][:, -1].tolist(),
        "overstressed": [{
            "source_row": int(rows[wall][index]),
            "principal_stress_pa": float(principal[index]),
            "volume_m3": float(volumes[index]),
            "centroid_m": centroids[index].tolist(),
            "principal_direction": vectors[index][:, -1].tolist(),
        } for index in over_ids[:MAXIMUM_LISTED_CELLS]],
        "overstressed_listed": int(min(len(over_ids), MAXIMUM_LISTED_CELLS)),
        # The automatic cracking runner's initiation measure: cracking starts only if a facet reaches f_t.
        "facet_check": facets,
        "is_final_crack_prediction": False,
        "solver_ready": False, "engineering_validated": False,
        "limitations": [
            "Initial screening only; redistribution after cracking can extend or arrest damage.",
            "Principal cell stress is not the normal traction on any mesh facet; facet_check reports the facet "
            "tractions that automatic cracking uses to initiate.",
            "Foundation/rock and existing structural joints are excluded from concrete screening.",
        ],
    }


def screening_from_archives(intact_cell_stress: Path, native_inputs: Path, tensile_strength_pa: float,
                            *, relative_tolerance: float) -> dict:
    with np.load(intact_cell_stress, allow_pickle=False) as stress, np.load(native_inputs, allow_pickle=False) as native:
        result = screen_cells(
            stress["cell_stress_pa"], stress["cell_region_tags"], stress["source_cell_rows"],
            native["coordinates_m"], native["geometry_cells"], tensile_strength_pa,
            relative_tolerance=relative_tolerance)
    result["native_archive_sha256"] = {
        "intact_cell_stress.npz": hashlib.sha256(Path(intact_cell_stress).read_bytes()).hexdigest(),
        "native_inputs.npz": hashlib.sha256(Path(native_inputs).read_bytes()).hexdigest(),
    }
    return result


COMPLETED = "completed_automatic_screening"


def config_digest(config: dict) -> str:
    """Same canonical digest as the app job record and painted runner."""
    return hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def prepare_and_screen(configuration: Path, model_directory: Path, baseline_directory: Path,
                       output: Path, *, job_id: str, maximum_volume_change: float,
                       relative_tolerance: float) -> dict:
    """Solve the verified intact native equilibrium in a new directory, then screen it.

    Writes manifest.json (running, completed or failed) so callers never mistake a
    partial or foreign directory for a result. Nothing is cracked or submitted.
    """
    if __package__:
        from .run import submitted_structure, write_json
        from .tensile_loss_event import EventSolverOptions, solve_source_event
        from .three_body_model import require_verified_baseline
    else:
        from run import submitted_structure, write_json
        from tensile_loss_event import EventSolverOptions, solve_source_event
        from three_body_model import require_verified_baseline

    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("An explicit nonempty source job_id is required")
    output = Path(output)
    config_bytes = Path(configuration).read_bytes()
    config = json.loads(config_bytes)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "status": "running_automatic_screening", "source_job_id": job_id,
        "screening_version": SCREENING_VERSION,
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_digest": config_digest(config),
        "solver_ready": False, "engineering_validated": False,
    }

    def persist():
        manifest["elapsed_s"] = time.monotonic() - started
        write_json(output / "manifest.json", manifest)

    persist()
    try:
        inputs = output / "inputs"
        inputs.mkdir()
        snapshot = inputs / "config.json"
        snapshot.write_bytes(config_bytes)
        concrete = config["material"]["concrete"]
        structure = submitted_structure(snapshot, Path(model_directory))
        baseline = require_verified_baseline(structure, Path(baseline_directory))
        manifest["phase"] = "solving the verified intact native equilibrium"
        persist()
        event_directory = output / "intact"
        solve_source_event(structure, np.empty(0, dtype=int), maximum_volume_change=maximum_volume_change,
                           options=EventSolverOptions(), output=event_directory, intact_only=True)
        manifest["phase"] = "screening wall cells"
        persist()
        result = screening_from_archives(event_directory / "intact_cell_stress.npz",
                                         event_directory / "native_inputs.npz",
                                         float(concrete["tensile_strength_pa"]),
                                         relative_tolerance=relative_tolerance)
        energy = concrete.get("cohesive_fracture_energy_j_m2")
        result.update({
            "source_job_id": job_id,
            "configuration_sha256": manifest["configuration_sha256"],
            "config_digest": manifest["config_digest"],
            "verified_baseline_source_sha256": baseline["source_sha256"],
            "cohesive_fracture_energy_j_m2": energy,
            "cohesive_fracture_energy_provenance": (
                "user-supplied; not independently verified" if energy is not None
                else "absent from the configuration; automatic cracking cannot be submitted"),
            "maximum_volume_change": maximum_volume_change,
        })
        write_json(output / "screening.json", result)
        manifest.update({"status": COMPLETED, "phase": None, "accepted_results": {"screening": "screening.json"},
                         "screening_sha256": hashlib.sha256((output / "screening.json").read_bytes()).hexdigest()})
        persist()
        return result
    except BaseException as error:
        manifest.update({"status": "failed_automatic_screening", "error": f"{type(error).__name__}: {error}"})
        persist()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--maximum-volume-change", type=float, required=True)
    parser.add_argument("--relative-tolerance", type=float, required=True)
    arguments = parser.parse_args(argv)
    result = prepare_and_screen(arguments.config, arguments.model, arguments.baseline, arguments.output,
                                job_id=arguments.job_id,
                                maximum_volume_change=arguments.maximum_volume_change,
                                relative_tolerance=arguments.relative_tolerance)
    print(json.dumps({key: result[key] for key in (
        "status", "wall_cells", "overstressed_cells", "peak_principal_stress_pa")}))


if __name__ == "__main__":
    main()
