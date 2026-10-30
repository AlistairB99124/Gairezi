"""Automatic stress-driven cohesive cracking of the canonical three-body dam.

Runs the whole-wall extrinsic cohesive network (``cohesive_network``) on the
native FEniCSx mesh from a completed automatic screening. All configured loads
(gravity, exterior reservoir/tailwater/top water) are applied proportionally
from 0 to 1; reservoir water enters connected open cracks with its unit weight
scaled by the same factor. Under proportional loading the intact response is
linear, so a screening peak principal stress below f_t * (1 - tolerance)
proves no facet can initiate and no mesh work is needed. Otherwise the
verified screening archives give the intact one-sided facet-normal stress (the
network's own initiation measure: the larger n.sigma.n of the two owner cells) on
every interior wall facet: if all are
below f_t * (1 - tolerance) no facet initiates; else the cells adjacent to
critical facets seed the candidate region.

The candidate facets are the interior wall facets of a bounded region (seed
cells plus neighbour rings, shed down to zero rings to fit the face-count
preflight). The final state is accepted only if no interior wall facet outside
the candidate set reaches f_t; the region is enlarged otherwise, up to the
preflight limit. This is not engineering
validation: Gf provenance, mesh dependence and stability are not established.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

if __package__:
    from .automatic_screening import (COMPLETED as SCREENING_COMPLETED, WALL_REGION, _wall_faces, config_digest,
                                      facet_check, interior_wall_facet_normal_stress)
    from .cohesive_contact import CohesiveOptions
    from .cohesive_mode_i import LinearModeICohesiveLaw
    from .cohesive_network import LIMITATIONS, CohesiveNetwork, NetworkOptions, NetworkWater
else:
    from automatic_screening import (COMPLETED as SCREENING_COMPLETED, WALL_REGION, _wall_faces, config_digest,
                                     facet_check, interior_wall_facet_normal_stress)
    from cohesive_contact import CohesiveOptions
    from cohesive_mode_i import LinearModeICohesiveLaw
    from cohesive_network import LIMITATIONS, CohesiveNetwork, NetworkOptions, NetworkWater

RUNNER_VERSION = 2
COMPLETED = "completed_automatic_cracking"
FAILED = "failed_automatic_cracking"
RUNNING = "running_automatic_cracking"
ACCEPTED_OUTCOMES = ("no_damage_initiated", "accepted_equilibrium")
UPSTREAM_TAG = 2
# Cohesive force-balance tolerance relative to the norm of the configured nodal load vector.
FORCE_TOLERANCE_RELATIVE = 1e-9


def isotropic_elasticity(young_pa, poisson):
    """6x6 Voigt [xx, yy, zz, xy, yz, zx] with engineering shear strains."""
    mu = young_pa / (2 * (1 + poisson))
    lam = young_pa * poisson / ((1 + poisson) * (1 - 2 * poisson))
    matrix = np.zeros((6, 6))
    matrix[:3, :3] = lam
    matrix[np.arange(3), np.arange(3)] += 2 * mu
    matrix[np.arange(3, 6), np.arange(3, 6)] = mu
    return matrix


def cell_max_principal_stress(coordinates_m, cells, displacement_m, elasticity):
    """Exact constant P1 cell stress, maximum principal value (tension positive)."""
    return np.linalg.eigvalsh(cell_stress_tensors(coordinates_m, cells, displacement_m, elasticity))[:, -1]


def cell_stress_tensors(coordinates_m, cells, displacement_m, elasticity):
    """Exact constant P1 cell stress tensors (n, 3, 3), tension positive."""
    cells = np.asarray(cells)
    if not len(cells):
        return np.zeros((0, 3, 3))
    points = np.asarray(coordinates_m)[cells]
    homogeneous = np.concatenate((np.ones((len(cells), 4, 1)), points), axis=2)
    gradients = np.linalg.inv(homogeneous)[:, 1:, :]
    grad_u = np.einsum("nia,naj->nij", gradients, np.asarray(displacement_m).reshape(-1, 3)[cells])
    strain = .5 * (grad_u + np.swapaxes(grad_u, 1, 2))
    voigt = np.stack((strain[:, 0, 0], strain[:, 1, 1], strain[:, 2, 2],
                      2 * strain[:, 0, 1], 2 * strain[:, 1, 2], 2 * strain[:, 2, 0]), axis=1)
    s = voigt @ np.asarray(elasticity).T
    return np.stack((np.stack((s[:, 0], s[:, 3], s[:, 5]), 1), np.stack((s[:, 3], s[:, 1], s[:, 4]), 1),
                     np.stack((s[:, 5], s[:, 4], s[:, 2]), 1)), 1)


def screening_facet_bound(screening_output: Path, screening: dict, tensile_strength_pa, relative_tolerance):
    """Intact facet-traction check from the verified screening archives, or None if they are unavailable.

    Under proportional linear loading the intact facet stress scales with the load
    factor, so a maximum below f_t (1 - tolerance) proves no facet initiates
    using the network's own (one-sided) criterion. Otherwise the cells adjacent
    to critical facets are the cracking seeds.
    """
    hashes = screening.get("native_archive_sha256")
    if not isinstance(hashes, dict):
        return None
    intact = Path(screening_output) / "intact"
    paths = {name: intact / name for name in ("intact_cell_stress.npz", "native_inputs.npz")}
    if set(hashes) != set(paths) or not all(path.is_file() for path in paths.values()):
        return None
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f"Screening archive {name} does not match the verified screening")
    with np.load(paths["intact_cell_stress.npz"], allow_pickle=False) as stress, \
            np.load(paths["native_inputs.npz"], allow_pickle=False) as native:
        rows = np.asarray(stress["source_cell_rows"])
        pairs, _, critical, check = facet_check(
            native["coordinates_m"], native["geometry_cells"], stress["cell_region_tags"], stress["cell_stress_pa"],
            tensile_strength_pa, relative_tolerance)
    return {
        **check,
        "critical_source_rows": np.unique(rows[pairs[critical]]).astype(int).tolist(),
        "bounds_initiation": not check["initiates"],
    }


def candidate_region(tetrahedra, regions, seed_cells, rings):
    """Seed wall cells grown by ``rings`` vertex-neighbour layers (wall cells only)."""
    tetrahedra = np.asarray(tetrahedra)
    wall = np.asarray(regions) == WALL_REGION
    incidence = sparse.csr_matrix(
        (np.ones(tetrahedra.size), (np.repeat(np.arange(len(tetrahedra)), 4), tetrahedra.ravel())),
        shape=(len(tetrahedra), int(tetrahedra.max()) + 1))
    selected = np.zeros(len(tetrahedra), dtype=bool)
    selected[np.asarray(seed_cells, dtype=int)] = True
    selected &= wall
    for _ in range(rings):
        nodes = np.asarray(incidence[selected].sum(axis=0)).ravel() > 0
        selected |= (np.asarray(incidence @ nodes.astype(float)).ravel() > 0) & wall
    return selected


def region_candidate_faces(tetrahedra, regions, selected):
    faces, owners = _wall_faces(np.asarray(tetrahedra), np.asarray(regions))
    keep = selected[owners[:, 0]] & selected[owners[:, 1]]
    return faces[keep]


def upstream_inlet_edges(imported):
    """Source-node edges of the upstream reservoir boundary (tag 2), from native connectivity."""
    from dolfinx import mesh

    domain = imported.domain
    facets = imported.facet_tags.find(UPSTREAM_TAG)
    if not len(facets):
        return np.empty((0, 2), dtype=np.int64)
    triangles = domain.geometry.input_global_indices[mesh.entities_to_geometry(domain, 2, facets)]
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    return np.unique(edges.astype(np.int64), axis=0)


class NativeNetworkProblem:
    """Assembled native elasticity, supports, springs and loads in split node-major DOFs.

    The prepared (split) mesh is imported with its source-node identity equal to
    the split node index, so ``bulk_face_mapping`` gives the DOLFINx DOF block
    of each split node. Loads are the configured gravity plus exact exterior
    water nodal forces (as in the screening equilibrium), scaled by the factor.
    """

    def __init__(self, prepared, structure):
        from dolfinx import fem
        from dolfinx.fem.petsc import assemble_matrix, assemble_vector
        from petsc4py import PETSc
        import ufl

        if __package__:
            from .dam_elasticity import external_load, water_load_parameters
            from .fracture_faces import bulk_face_mapping
            from .hydrostatic_load import assemble_water_forces
            from .interface_springs import InterfaceSprings
            from .three_body_model import materials, require_support_tags, support_bcs, support_form
        else:
            from dam_elasticity import external_load, water_load_parameters
            from fracture_faces import bulk_face_mapping
            from hydrostatic_load import assemble_water_forces
            from interface_springs import InterfaceSprings
            from three_body_model import materials, require_support_tags, support_bcs, support_form

        imported = prepared.imported
        space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
        coefficients = materials(imported, structure)
        require_support_tags(imported, structure)
        bcs = support_bcs(imported, space, structure)
        ds = ufl.Measure("ds", domain=imported.domain, subdomain_data=imported.facet_tags)
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        strain = ufl.sym(ufl.grad(trial))
        stress = 2 * coefficients["mu"] * strain + coefficients["lam"] * ufl.tr(strain) * ufl.Identity(3)
        form = ufl.inner(stress, ufl.sym(ufl.grad(test))) * ufl.dx
        support = support_form(trial, test, ds, structure)
        if support is not None:
            form += support
        operator = assemble_matrix(fem.form(form))
        operator.assemble()
        springs = InterfaceSprings(imported, structure, space, multiplier_parent_node_ids=prepared.parent_node_ids)
        try:
            operator.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            springs.build_matrix(operator)
            springs.add_to(operator)
            offsets, columns, values = operator.getValuesCSR()
            stiffness = sparse.csr_matrix((values.copy(), columns.copy(), offsets.copy()), shape=operator.getSize())
        finally:
            operator.destroy()
            if springs.matrix is not None:
                springs.matrix.destroy()
        vector = assemble_vector(fem.form(external_load(
            imported, structure, coefficients["density"], test, ds, include_water=False)[0]))
        try:
            gravity = vector.array.copy()
        finally:
            vector.destroy()
        water = assemble_water_forces(imported, space, *water_load_parameters(imported, structure)).ravel()
        _, _, mapping = bulk_face_mapping(space, prepared.faces)
        permutation = (3 * mapping[:, None].astype(np.int64) + np.arange(3)).ravel()
        if len(permutation) != stiffness.shape[0] or len(np.unique(permutation)) != len(permutation):
            raise ValueError("Split nodes must map one-to-one onto native DOF blocks")
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(len(permutation))
        fixed = np.unique(np.concatenate([bc.dof_indices()[0] for bc in bcs])) if bcs else np.empty(0, int)
        self.stiffness = stiffness[permutation][:, permutation].tocsr()
        self.stiffness = (.5 * (self.stiffness + self.stiffness.T)).tocsr()
        self.load = (gravity + water)[permutation]
        self.fixed = np.sort(inverse[fixed]).astype(np.int64)
        self.split = prepared.split
        self.spring_report = springs.report

    def __call__(self, split, factor):
        if split is not self.split:
            raise ValueError("Native network problem belongs to a different split inventory")
        return self.stiffness, float(factor) * self.load, self.fixed, np.zeros(len(self.fixed))


def _seed_cells(original, screening, seed_rows=None):
    if seed_rows is None:
        rows = set(map(int, screening.get("overstressed_source_rows") or []))
        rows.add(int(screening["peak_source_row"]))
    else:
        rows = set(map(int, seed_rows))
    parents = np.asarray(original.parent_ids)
    return np.flatnonzero(np.isin(parents, sorted(rows)) & (np.asarray(original.regions) == WALL_REGION))


def run_network_on_structure(structure, screening, *, gf_j_m2, tensile_strength_pa, relative_tolerance,
                             maximum_volume_change, rings=2, maximum_candidate_faces=20000,
                             maximum_region_expansions=3, maximum_solves=5000, progress=print,
                             should_stop=None, seed_rows=None):
    """Bounded-region cohesive network solve; returns (NetworkResult, prepared, region report).

    ``seed_rows`` (source cell rows) overrides the screening's principal-stress seeds, for
    example with the cells adjacent to intact critical facets.
    """
    if __package__:
        from .dam_mesh import import_structure, prepare_wall_candidates
    else:
        from dam_mesh import import_structure, prepare_wall_candidates

    progress("importing the native mesh")
    original = import_structure(structure, maximum_volume_change)
    concrete = structure.three_body_config["material"]["concrete"]
    elasticity = isotropic_elasticity(float(concrete["youngs_modulus"]), float(concrete["poisson_ratio"]))
    law = LinearModeICohesiveLaw(float(tensile_strength_pa), float(gf_j_m2))
    tetrahedra, regions = np.asarray(original.tetrahedra), np.asarray(original.regions)
    seeds = _seed_cells(original, screening, seed_rows)
    if not len(seeds):
        raise ValueError("Screening source rows do not identify native wall cells")
    loads = structure.loads
    unit_weight = float(loads.water_density_kg_m3) * abs(float(loads.gravity_z_m_s2))
    inlets = upstream_inlet_edges(original)
    attempts = []
    for attempt in range(maximum_region_expansions + 1):
        selected = candidate_region(tetrahedra, regions, seeds, rings)
        candidates = region_candidate_faces(tetrahedra, regions, selected)
        # The first region may shed neighbour rings to fit the preflight; the seeds themselves never shrink.
        while attempt == 0 and len(candidates) > maximum_candidate_faces and rings > 0:
            attempts.append({"rings": rings, "region_cells": int(selected.sum()),
                             "candidate_faces": int(len(candidates)), "status": "reduced_rings"})
            rings -= 1
            selected = candidate_region(tetrahedra, regions, seeds, rings)
            candidates = region_candidate_faces(tetrahedra, regions, selected)
        record = {"rings": rings, "region_cells": int(selected.sum()), "candidate_faces": int(len(candidates))}
        attempts.append(record)
        if not len(candidates):
            raise ValueError("The candidate region has no interior wall facets")
        if len(candidates) > maximum_candidate_faces:
            record["status"] = "resource_limit"
            return None, None, {"attempts": attempts, "status": "resource_limit",
                                "message": f"{len(candidates)} candidate facets exceed the preflight limit "
                                           f"{maximum_candidate_faces}"}
        progress(f"preparing {len(candidates)} candidate facets in {int(selected.sum())} wall cells "
                 f"(attempt {attempt + 1})")
        prepared = prepare_wall_candidates(structure, None, maximum_volume_change,
                                           candidate_triangles=candidates, original=original)
        progress("assembling native stiffness, supports, springs and loads")
        problem = NativeNetworkProblem(prepared, structure)
        tolerance = float(relative_tolerance)
        scale = max(float(np.linalg.norm(problem.load)), 1.)
        options = NetworkOptions(
            CohesiveOptions(gap_tolerance_m=1e-9, force_tolerance_n=FORCE_TOLERANCE_RELATIVE * scale,
                            relative_residual_tolerance=1e-8, initiation_relative_tolerance=tolerance,
                            tangential_response="tied"),
            maximum_solves=maximum_solves)
        water = NetworkWater(float(loads.upstream_surface_z_m), unit_weight, inlets, 1e-9,
                             scale_with_load_factor=True) if len(inlets) else None
        network = CohesiveNetwork.from_split(prepared.split, elasticity=elasticity, law=law, options=options,
                                             problem_at=problem, water=water, should_stop=should_stop,
                                             progress=progress)
        progress("solving the cohesive network under proportional loading")
        result = network.run(0., 1.)
        record.update(status=result.status, solves=result.report["solves"])
        if result.status not in ACCEPTED_OUTCOMES or result.displacement_m is None:
            return result, prepared, {"attempts": attempts, "status": result.status}
        split = prepared.split
        # Non-candidate interior wall facets still share nodes in the split mesh; check them with the
        # same one-sided facet-normal criterion the network uses for initiation.
        split_cells, split_regions = np.asarray(split.cells), np.asarray(split.region_ids)
        stress = np.zeros((len(split_cells), 3, 3))
        wall_cells = split_regions == WALL_REGION
        stress[wall_cells] = cell_stress_tensors(split.coordinates_m, split_cells[wall_cells],
                                                 result.displacement_m, elasticity)
        pairs, normal = interior_wall_facet_normal_stress(split.coordinates_m, split_cells, split_regions, stress)
        peak = float(normal.max(initial=-np.inf))
        record["outside_region_peak_facet_normal_stress_pa"] = peak
        critical = normal >= float(tensile_strength_pa) * (1 - tolerance)
        if not critical.any():
            return result, prepared, {"attempts": attempts, "status": "region_sufficient",
                                      "outside_region_peak_facet_normal_stress_pa": peak}
        record["outside_region_critical_facets"] = int(critical.sum())
        seeds = np.union1d(np.flatnonzero(selected),
                           np.unique(np.asarray(split.parent_cell_ids)[pairs[critical].ravel()]))
        rings += 2
    return result, prepared, {"attempts": attempts, "status": "candidate_region_exceeded",
                              "message": "Wall facets outside the candidate region still reach f_t"}


def _verified_screening(screening_output: Path, job_id: str, digest: str) -> dict:
    manifest = json.loads((screening_output / "manifest.json").read_text())
    if manifest.get("status") != SCREENING_COMPLETED:
        raise ValueError(f"Screening did not complete ({manifest.get('status')})")
    if manifest.get("source_job_id") != job_id or manifest.get("config_digest") != digest:
        raise ValueError("Screening does not belong to this job's submitted configuration")
    data = (screening_output / "screening.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest.get("screening_sha256"):
        raise ValueError("Screening result does not match its manifest")
    screening = json.loads(data)
    if screening.get("source_job_id") != job_id:
        raise ValueError("Screening provenance is invalid")
    return screening


def screening_bound(screening, tensile_strength_pa, relative_tolerance):
    """True when the intact screening proves no facet can initiate under proportional loading."""
    return float(screening["peak_principal_stress_pa"]) < float(tensile_strength_pa) * (1 - relative_tolerance)


def _write_state(output, result, prepared, water_head, unit_weight):
    if __package__:
        from .fracture_faces import write_surface_state
    else:
        from fracture_faces import write_surface_state
    split, faces = prepared.split, prepared.split.faces
    np.savez_compressed(
        output / "cohesive_state.npz",
        source_faces=split.source_faces, coordinates_m=faces.coordinates, minus=faces.minus, plus=faces.plus,
        normals=faces.normals, areas_m2=faces.areas, initiated=result.initiated,
        face_status=np.asarray(result.face_status, dtype="U16"), face_opening_m=result.face_opening_m,
        face_normal_stress_pa=result.face_normal_stress_pa,
        maximum_opening_history_m=result.maximum_opening_history_m, wet_faces=result.wet_faces,
        displacement_m=result.displacement_m.reshape(-1, 3), parent_node_ids=split.parent_node_ids,
        load_factor=result.load_factor)
    if not result.initiated.any():
        return None
    z = faces.coordinates[faces.minus, 2]
    pressure = np.where(result.wet_faces[:, None],
                        np.maximum(unit_weight * result.load_factor * (water_head - z), 0.), 0.)
    path = output / "cohesive_surface.vtu"
    write_surface_state(path, faces, result.displacement_m.reshape(-1, 3), released_face_mask=result.initiated,
                        wet_face_mask=result.wet_faces & result.initiated, vertex_pressures_pa=pressure,
                        parent_node_ids=split.parent_node_ids)
    return path.name


def prepare_and_run(configuration: Path, model_directory: Path, screening_output: Path, output: Path, *,
                    job_id: str, maximum_volume_change: float, rings: int = 2,
                    maximum_candidate_faces: int = 20000, maximum_solves: int = 5000) -> dict:
    """Verify the confirmed screening, then solve or bound the automatic cracking outcome."""
    if __package__:
        from .run import submitted_structure, write_json
    else:
        from run import submitted_structure, write_json

    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("An explicit nonempty source job_id is required")
    output = Path(output)
    config_bytes = Path(configuration).read_bytes()
    config = json.loads(config_bytes)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"status": RUNNING, "runner_version": RUNNER_VERSION, "source_job_id": job_id,
                "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "config_digest": config_digest(config), "phase": "verifying screening",
                "solver_ready": False, "engineering_validated": False}

    def persist():
        manifest["elapsed_s"] = time.monotonic() - started
        write_json(output / "manifest.json", manifest)

    def progress(message):
        manifest["phase"] = message
        persist()
        print(f"Automatic cracking: {message}", flush=True)

    persist()
    try:
        screening = _verified_screening(Path(screening_output), job_id, manifest["config_digest"])
        concrete = config["material"]["concrete"]
        strength = float(concrete["tensile_strength_pa"])
        energy = concrete.get("cohesive_fracture_energy_j_m2")
        if energy is None:
            raise ValueError("The configuration has no cohesive fracture energy (Gf)")
        if float(screening["tensile_strength_pa"]) != strength:
            raise ValueError("Screening tensile strength differs from the configuration")
        tolerance = float(screening["relative_tolerance"])
        law = LinearModeICohesiveLaw(strength, float(energy))
        summary = {
            "runner_version": RUNNER_VERSION, "source_job_id": job_id,
            "config_digest": manifest["config_digest"],
            "screening_sha256": hashlib.sha256((Path(screening_output) / "screening.json").read_bytes()).hexdigest(),
            "tensile_strength_pa": strength, "fracture_energy_j_m2": float(energy),
            "fracture_energy_provenance": "user-supplied; not independently verified",
            "critical_opening_m": law.critical_opening_m,
            "initiation_relative_tolerance": tolerance,
            "loading": "all configured loads proportional 0 -> 1; reservoir crack water scaled by the same factor",
            "screening_peak_principal_stress_pa": float(screening["peak_principal_stress_pa"]),
            "screening_overstressed_cells": int(screening["overstressed_cells"]),
            "limitations": LIMITATIONS, "engineering_validated": False,
        }
        if screening_bound(screening, strength, tolerance):
            summary.update({
                "outcome": "no_damage_initiated", "load_factor": 1.0,
                "outcome_basis": "intact screening bound: linear proportional loading and peak principal stress "
                                 "< f_t (1 - tolerance), which bounds every facet-normal stress",
                "initiated_faces": 0, "opened_area_m2": 0., "cohesive_dissipated_energy_j": 0.,
                "steps": [],
            })
            facets = None
        else:
            progress("checking intact facet tractions")
            same_mesh = screening.get("maximum_volume_change") in (None, maximum_volume_change)
            facets = (screening_facet_bound(Path(screening_output), screening, strength, tolerance)
                      if same_mesh else None)
            if facets is not None:
                summary["intact_facet_screening"] = facets
        if facets is not None and facets["bounds_initiation"]:
            summary.update({
                "outcome": "no_damage_initiated", "load_factor": 1.0,
                "outcome_basis": "intact facet bound: linear proportional loading and every interior wall facet "
                                 "has normal stress < f_t (1 - tolerance) on both owner cells; cell principal "
                                 "peaks above f_t do not act normal to any shared facet",
                "initiated_faces": 0, "opened_area_m2": 0., "cohesive_dissipated_energy_j": 0.,
                "steps": [],
            })
        elif "outcome" not in summary:
            structure = submitted_structure(Path(configuration), Path(model_directory))
            result, prepared, region = run_network_on_structure(
                structure, screening, gf_j_m2=float(energy), tensile_strength_pa=strength,
                relative_tolerance=tolerance, maximum_volume_change=maximum_volume_change, rings=rings,
                maximum_candidate_faces=maximum_candidate_faces, maximum_solves=maximum_solves,
                progress=progress, seed_rows=facets["critical_source_rows"] if facets else None)
            summary["candidate_region"] = region
            if result is None:
                summary.update({"outcome": region["status"], "outcome_basis": region["message"]})
            else:
                outcome = (result.status if result.status not in ACCEPTED_OUTCOMES
                           or region["status"] == "region_sufficient" else region["status"])
                summary.update({"outcome": outcome, "load_factor": result.load_factor,
                                "outcome_basis": "cohesive network solve", "network": result.report,
                                "steps": result.steps, "source_preparation": prepared.report})
                if result.displacement_m is not None:
                    loads = structure.loads
                    surface = _write_state(output, result, prepared, float(loads.upstream_surface_z_m),
                                           float(loads.water_density_kg_m3) * abs(float(loads.gravity_z_m_s2)))
                    summary["state"] = "cohesive_state.npz"
                    summary["surface"] = surface
        summary["accepted"] = summary["outcome"] in ACCEPTED_OUTCOMES
        write_json(output / "summary.json", summary)
        manifest.update({
            "status": COMPLETED, "phase": None, "outcome": summary["outcome"], "accepted": summary["accepted"],
            "summary_sha256": hashlib.sha256((output / "summary.json").read_bytes()).hexdigest()})
        persist()
        return summary
    except BaseException as error:
        manifest.update({"status": FAILED, "error": f"{type(error).__name__}: {error}"})
        persist()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--screening", type=Path, required=True, help="completed screening output directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--maximum-volume-change", type=float, required=True)
    parser.add_argument("--neighbour-rings", type=int, default=2)
    parser.add_argument("--maximum-candidate-faces", type=int, default=20000)
    parser.add_argument("--maximum-solves", type=int, default=5000)
    arguments = parser.parse_args(argv)
    summary = prepare_and_run(arguments.config, arguments.model, arguments.screening, arguments.output,
                              job_id=arguments.job_id, maximum_volume_change=arguments.maximum_volume_change,
                              rings=arguments.neighbour_rings,
                              maximum_candidate_faces=arguments.maximum_candidate_faces,
                              maximum_solves=arguments.maximum_solves)
    print(json.dumps({key: summary.get(key) for key in ("outcome", "accepted", "load_factor")}))


if __name__ == "__main__":
    main()
