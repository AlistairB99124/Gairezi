"""Automatic whole-wall tensile screening of native intact FEniCSx cell stresses.

This is initial screening, not a predicted final crack set: it reports wall
cells whose unsmoothed maximum principal stress exceeds the configured intact
tensile strength in the verified intact equilibrium. It never alters
coefficients, launches cracking or substitutes recovered nodal stress.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

WALL_REGION = 1
SCREENING_VERSION = 3
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


def _validate_count(max_damage_count):
    if max_damage_count is not None and (isinstance(max_damage_count, bool) or int(max_damage_count) < 2
                                         or int(max_damage_count) != max_damage_count):
        raise ValueError("max_damage_count must be null or an integer of at least 2")


def _node_star_patch(tetrahedra, regions, pairs, facet_stress, critical_ids, max_damage_count):
    """Connected whole wall node stars, highest incident facet tension first, within the count.

    An interior facet only opens where the cut faces separate a node's cells, so the patch
    takes every wall cell around a node (nodes touching other regions are skipped) and
    grows to neighbouring nodes. Returns (selected cells, star centre nodes).
    """
    faces, _ = _wall_faces(tetrahedra, regions)
    node_score = np.full(int(tetrahedra.max(initial=-1)) + 1, -np.inf)
    for corner in range(3):
        np.maximum.at(node_score, faces[:, corner], facet_stress)
    wall = regions == WALL_REGION
    other = np.zeros(len(node_score), dtype=bool)
    other[np.unique(tetrahedra[~wall])] = True
    incidence = sparse.csr_matrix(
        (np.ones(tetrahedra.size), (tetrahedra.ravel(), np.repeat(np.arange(len(tetrahedra)), 4))),
        shape=(len(node_score), len(tetrahedra)))

    def star(node):
        return incidence.indices[incidence.indptr[node]:incidence.indptr[node + 1]]

    seeds = np.unique(faces[critical_ids])
    seeds = seeds[np.argsort(-node_score[seeds], kind="stable")]
    seed = next((int(node) for node in seeds
                 if not other[node] and len(star(node)) <= max_damage_count), None)
    chosen: set[int] = set()
    centres: list[int] = []
    if seed is None:
        return np.empty(0, dtype=int), centres
    heap, used = [(-node_score[seed], seed)], set()
    while heap:
        _, node = heapq.heappop(heap)
        if node in used:
            continue
        used.add(node)
        if other[node]:
            continue
        new = [int(cell) for cell in star(node) if int(cell) not in chosen]
        if not new or len(chosen) + len(new) > max_damage_count:
            continue
        chosen.update(new)
        centres.append(node)
        for cell in new:
            for neighbour in tetrahedra[cell]:
                if int(neighbour) not in used:
                    heapq.heappush(heap, (-node_score[neighbour], int(neighbour)))
    return np.array(sorted(chosen), dtype=int), centres


def openable_facets(coordinates_m, tetrahedra, regions, selected_cells):
    """Pre-check with the cracking runner's own rule: wall facets between selected cells are cut,
    and fully tied cuts (owners still sharing every node) are dropped. Returns (candidates, openable)."""
    if __package__:
        from .fracture_topology import split_interior_wall_faces
    else:
        from fracture_topology import split_interior_wall_faces

    faces, owners = _wall_faces(tetrahedra, regions)
    member = np.zeros(len(tetrahedra), dtype=bool)
    member[np.asarray(selected_cells, dtype=int)] = True
    candidates = faces[member[owners[:, 0]] & member[owners[:, 1]]]
    if not len(candidates):
        return 0, 0
    try:
        split = split_interior_wall_faces(coordinates_m, tetrahedra, regions, candidates.astype(np.int64),
                                          drop_fully_tied=True)
    except ValueError as error:
        if not str(error).startswith("Every cut is fully tied"):
            raise
        return int(len(candidates)), 0
    return int(len(candidates)), int(len(candidates) - len(split.tied_source_faces))


def select_damage_cells(pairs, facet_stress_pa, critical, source_rows, max_damage_count=None, *,
                        coordinates_m=None, tetrahedra=None, regions=None) -> dict:
    """Automatic element selection from the facet initiation check.

    Eligible cells own a critical facet. ``None`` selects every eligible cell (the cracking
    runner then adds neighbour rings). A count needs the mesh: a connected patch of whole
    wall node stars is grown from the highest-tension critical facet, taking the next most
    tensioned neighbouring node whose star still fits, and is checked for openability.
    """
    pairs = np.asarray(pairs).reshape(-1, 2)
    stress = np.asarray(facet_stress_pa, dtype=float)
    rows = np.asarray(source_rows)
    critical_ids = np.flatnonzero(np.asarray(critical, dtype=bool))
    eligible = np.unique(pairs[critical_ids]) if len(critical_ids) else np.empty(0, dtype=int)
    _validate_count(max_damage_count)
    patch = {}
    if max_damage_count is None:
        chosen = eligible
    else:
        if tetrahedra is None or regions is None or coordinates_m is None:
            raise ValueError("A capped selection needs the mesh coordinates, tetrahedra and regions")
        tetrahedra, regions = np.asarray(tetrahedra), np.asarray(regions)
        if len(critical_ids):
            chosen, centres = _node_star_patch(tetrahedra, regions, pairs, stress, critical_ids,
                                               int(max_damage_count))
        else:
            chosen, centres = np.empty(0, dtype=int), []
        candidates, opening = (openable_facets(coordinates_m, tetrahedra, regions, chosen) if len(chosen)
                               else (0, 0))
        patch = {"patch_nodes": len(centres), "candidate_facets": candidates, "openable_facets": opening,
                 "openable": bool(opening) if len(critical_ids) else None, "message": None}
        if len(critical_ids) and not opening:
            patch["message"] = (
                f"The {int(max_damage_count)} allowed elements cannot form an openable crack at the most "
                "tensioned facets: no whole group of wall elements around one mesh node fits the count, so every "
                "cut would stay tied. Increase Max Damage Count or leave it blank.")
    member = np.zeros(max(int(pairs.max(initial=-1)), int(np.max(chosen, initial=-1))) + 1, dtype=bool)
    member[chosen] = True
    facets_inside = (member[pairs[critical_ids]].all(axis=1) if len(critical_ids)
                     else np.zeros(0, dtype=bool))
    inside, outside = critical_ids[facets_inside], critical_ids[~facets_inside]
    return {
        "rule": ("cells owning an interior wall facet whose one-sided normal stress >= f_t (1 - tolerance)"
                 if max_damage_count is None else
                 "connected patch of whole wall node stars (every wall cell around a mesh node), grown from the "
                 "highest one-sided facet normal stress at or above f_t (1 - tolerance) to the next most tensioned "
                 "neighbouring node whose star still fits max_damage_count; nodes touching other regions are "
                 "skipped"),
        "max_damage_count": None if max_damage_count is None else int(max_damage_count),
        "capped": max_damage_count is not None and len(chosen) < len(eligible),
        "eligible_cells": int(len(eligible)),
        "eligible_facets": int(len(critical_ids)),
        "selected_cells": int(len(chosen)),
        "selected_facets": int(facets_inside.sum()),
        "selected_source_rows": np.unique(rows[chosen]).astype(int).tolist(),
        "lowest_selected_facet_stress_pa": float(stress[inside].min()) if len(inside) else None,
        "peak_unselected_facet_stress_pa": float(stress[outside].max()) if len(outside) else None,
        **patch,
    }


def screen_cells(cell_stress_pa, cell_region_tags, source_cell_rows, coordinates_m, geometry_cells,
                 tensile_strength_pa, *, relative_tolerance: float, max_damage_count=None) -> dict:
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
    pairs, one_sided, critical, facets = facet_check(coordinates, cells, regions, stress, tensile_strength_pa,
                                                     relative_tolerance)
    selection = select_damage_cells(pairs, one_sided, critical, rows, max_damage_count,
                                    coordinates_m=coordinates, tetrahedra=cells, regions=regions)
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
        # Automatic element selection: the cells automatic cracking seeds from (and, when capped, is confined to).
        "selection": selection,
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
                            *, relative_tolerance: float, max_damage_count=None) -> dict:
    with np.load(intact_cell_stress, allow_pickle=False) as stress, np.load(native_inputs, allow_pickle=False) as native:
        result = screen_cells(
            stress["cell_stress_pa"], stress["cell_region_tags"], stress["source_cell_rows"],
            native["coordinates_m"], native["geometry_cells"], tensile_strength_pa,
            relative_tolerance=relative_tolerance, max_damage_count=max_damage_count)
    result["native_archive_sha256"] = {
        "intact_cell_stress.npz": hashlib.sha256(Path(intact_cell_stress).read_bytes()).hexdigest(),
        "native_inputs.npz": hashlib.sha256(Path(native_inputs).read_bytes()).hexdigest(),
    }
    return result


COMPLETED = "completed_automatic_screening"


def max_damage_count(config: dict):
    """The configured ``damage_selection.max_damage_count`` (None = every eligible cell)."""
    selection = config.get("damage_selection") if isinstance(config, dict) else None
    return selection.get("max_damage_count") if isinstance(selection, dict) else None


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
                                         relative_tolerance=relative_tolerance,
                                         max_damage_count=max_damage_count(config))
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
