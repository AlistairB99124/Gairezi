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
critical facets seed the candidate region. Screenings that carry an automatic
element ``selection`` (``damage_selection.max_damage_count``) seed from the
selected cells instead; with a count set, cracking is confined to them.

The candidate facets are the interior wall facets of a bounded region (seed
cells plus neighbour rings, shed down to zero rings to fit the face-count
preflight). The final state is accepted only if no interior wall facet outside
the candidate set reaches f_t; the region is enlarged otherwise, up to the
preflight limit. A confined (capped) run uses the facets between selected cells
only and reports, without rejecting, outside facets that reach f_t. This is not engineering
validation: Gf provenance, mesh dependence and stability are not established.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

# MUMPS factorisations run in a threaded BLAS (OpenBLAS). Its default of one thread per
# core oversubscribes and was 3.4x slower than 4 threads on the full-dam operator. Must be
# set before NumPy/SciPy/PETSc load the library; an explicit environment value wins.
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(min(4, os.cpu_count() or 1)))

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

RUNNER_VERSION = 5
COMPLETED = "completed_automatic_cracking"
FAILED = "failed_automatic_cracking"
RUNNING = "running_automatic_cracking"
ACCEPTED_OUTCOMES = ("no_damage_initiated", "accepted_equilibrium")
UPSTREAM_TAG = 2
# Cohesive force-balance tolerance relative to the norm of the configured nodal load vector.
FORCE_TOLERANCE_RELATIVE = 1e-9
OVERLOAD_INCREMENT = 0.1
MAXIMUM_OVERLOAD_FACTOR = 10.0


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


def region_cell_stresses(coordinates_m, cells, region_ids, displacement_m, material):
    """Exact constant P1 cell stress per body: concrete wall (1) and foundation (3), bedrock (2)."""
    cells, region_ids = np.asarray(cells), np.asarray(region_ids)
    if not np.isin(region_ids, (1, 2, 3)).all():
        raise ValueError("Cell stress needs the canonical wall/bedrock/foundation body tags")
    stress = np.zeros((len(cells), 3, 3))
    for name, tags in (("concrete", (1, 3)), ("bedrock", (2,))):
        mask = np.isin(region_ids, tags)
        if mask.any():
            props = material[name]
            stress[mask] = cell_stress_tensors(
                coordinates_m, cells[mask], displacement_m,
                isotropic_elasticity(float(props["youngs_modulus"]), float(props["poisson_ratio"])))
    return stress


def stress_comparison(split, material, intact_displacement_m, cracked_displacement_m):
    """Same-mesh, same-measure intact vs cracked wall stresses; returns (report, intact, cracked) cell tensors.

    Both states are raw constant P1 cell stresses on the split mesh at the same load factor; the
    intact state keeps every candidate facet bonded. Compression is negative (signed principal).
    """
    coordinates, cells, regions = split.coordinates_m, np.asarray(split.cells), np.asarray(split.region_ids)
    intact_u = np.asarray(intact_displacement_m).reshape(-1, 3)
    cracked_u = np.asarray(cracked_displacement_m).reshape(-1, 3)
    intact = region_cell_stresses(coordinates, cells, regions, intact_u, material)
    cracked = region_cell_stresses(coordinates, cells, regions, cracked_u, material)
    wall = np.flatnonzero(regions == WALL_REGION)
    wall_nodes = np.unique(cells[wall])
    centroids = np.asarray(coordinates)[cells[wall]].mean(axis=1)
    states = {}
    for name, stress, displacement in (("intact", intact, intact_u), ("cracked", cracked, cracked_u)):
        principal = np.linalg.eigvalsh(stress[wall])
        tension, compression = int(np.argmax(principal[:, -1])), int(np.argmin(principal[:, 0]))
        states[name] = {
            "maximum_principal_tensile_stress_pa": float(principal[tension, -1]),
            "maximum_principal_compressive_stress_pa": float(principal[compression, 0]),
            "tensile_peak_cell": int(wall[tension]), "compressive_peak_cell": int(wall[compression]),
            "tensile_peak_centroid_xyz_m": centroids[tension].tolist(),
            "compressive_peak_centroid_xyz_m": centroids[compression].tolist(),
            "maximum_displacement_m": float(np.linalg.norm(displacement, axis=1).max()),
            "maximum_wall_displacement_m": float(np.linalg.norm(displacement[wall_nodes], axis=1).max()),
        }
    for key, cell in (("cracked_at_intact_tensile_peak", states["intact"]["tensile_peak_cell"]),
                      ("cracked_at_intact_compressive_peak", states["intact"]["compressive_peak_cell"])):
        states["cracked"][key] = {"cell": cell, "principal_stresses_pa": np.linalg.eigvalsh(cracked[cell]).tolist(),
                                  "coordinate_xyz_m": np.asarray(coordinates)[cells[cell]].mean(axis=0).tolist()}
    displacement_change = float(np.linalg.norm(cracked_u - intact_u, axis=1).max())
    scale = max(states["intact"]["maximum_displacement_m"], 1e-300)
    report = {
        "measure": "raw_cellwise",
        "method": "exact constant P1 cell stress from nodal displacement, per-body isotropic elasticity",
        "scope": "wall (body 1) cells of the split mesh; tension positive, compression negative",
        "units": "Pa", "intact": states["intact"], "cracked": states["cracked"],
        "intact_definition": "fully bonded candidate facets, same mesh, operator, loads and load factor",
        "maximum_wall_stress_change_pa": float(np.abs(cracked[wall] - intact[wall]).max()),
        "maximum_displacement_change_m": displacement_change,
        "relative_displacement_change": displacement_change / scale,
        "mesh_nodes": int(len(coordinates)), "mesh_cells": int(len(cells)),
    }
    return report, intact, cracked


def opening_assessment(result, law, comparison):
    """Plain statement of whether any initiated facet opened and whether the response changed."""
    initiated = np.asarray(result.initiated, dtype=bool)
    opened = np.isin(result.face_status, ("softening", "separated"))
    traction = np.asarray(result.face_normal_stress_pa)[initiated]
    unchanged = comparison is not None and comparison["relative_displacement_change"] <= 1e-6
    assessment = {
        "initiated_faces": int(initiated.sum()), "opened_faces": int(opened.sum()),
        "separated_faces": int((np.asarray(result.face_status) == "separated").sum()),
        "peak_opening_m": float(np.max(result.face_opening_m, initial=0.)),
        "opening_limit_m": float(law.critical_opening_m),
        "initiated_peak_interface_traction_pa": float(traction.max()) if len(traction) else None,
        "tensile_strength_pa": float(law.tensile_strength_pa),
        "cracked_equals_intact": bool(unchanged),
    }
    if not initiated.any():
        note = "No facet initiated; the cracked state is the intact elastic response."
    elif not opened.any():
        note = (f"{int(initiated.sum())} facets reached f_t and were released, but none opened (peak opening "
                f"{assessment['peak_opening_m']:.3g} m); the cracked response "
                + ("equals the intact bonded response." if unchanged else "differs only through tied facets."))
    else:
        note = f"{int(opened.sum())} of {int(initiated.sum())} initiated facets opened."
    assessment["note"] = note
    return assessment


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
                             should_stop=None, seed_rows=None, confined=False, overload_factor=1.0,
                             overload_increment=OVERLOAD_INCREMENT):
    """Bounded-region cohesive network solve; returns (NetworkResult, prepared, region report).

    The region report carries ``design_load`` (the state at load factor 1). When no
    facet opens at the design load and ``overload_factor`` > 1, loading continues
    proportionally in ``overload_increment`` steps up to that factor (``overload``) and
    the returned state is the last accepted overload equilibrium.

    ``seed_rows`` (source cell rows) overrides the screening's principal-stress seeds, for
    example with the cells adjacent to intact critical facets. ``confined`` limits the
    candidate facets to those between seed cells (no neighbour rings, no region growth);
    wall facets outside that still reach f_t are reported, not used to reject the result.
    """
    if confined:
        rings, maximum_region_expansions = 0, 0
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
        pairs, critical, peak = _outside_region_check(split, result.displacement_m, elasticity,
                                                      tensile_strength_pa, tolerance)
        record["outside_region_peak_facet_normal_stress_pa"] = peak
        if not critical.any():
            region = {"attempts": attempts, "status": "region_sufficient",
                      "outside_region_peak_facet_normal_stress_pa": peak}
            break
        record["outside_region_critical_facets"] = int(critical.sum())
        if confined:
            region = {
                "attempts": attempts, "status": "region_confined",
                "outside_region_peak_facet_normal_stress_pa": peak,
                "outside_region_critical_facets": int(critical.sum()),
                "message": f"{int(critical.sum())} wall facets outside the selected elements reach f_t; "
                           "cracking was confined to the selection (max_damage_count) and they were not opened"}
            break
        seeds = np.union1d(np.flatnonzero(selected),
                           np.unique(np.asarray(split.parent_cell_ids)[pairs[critical].ravel()]))
        rings += 2
    else:
        return result, prepared, {"attempts": attempts, "status": "candidate_region_exceeded",
                                  "message": "Wall facets outside the candidate region still reach f_t"}
    region["design_load"] = design_load_state(result, law)
    if float(overload_factor) > result.load_factor and not region["design_load"]["opened_faces"]:
        result, region["overload"] = continue_overload(network, result, float(overload_factor),
                                                       float(overload_increment), progress)
        if result.status not in ACCEPTED_OUTCOMES or result.displacement_m is None:
            return result, prepared, region
        _, outside, outside_peak = _outside_region_check(prepared.split, result.displacement_m, elasticity,
                                                         tensile_strength_pa, tolerance)
        region["overload"].update(outside_region_peak_facet_normal_stress_pa=outside_peak,
                                  outside_region_critical_facets=int(outside.sum()))
    progress("solving the fully bonded (intact) reference at the accepted load factor")
    result.intact_displacement_m = network.bonded_displacement(result.load_factor)
    return result, prepared, region


def _outside_region_check(split, displacement_m, elasticity, tensile_strength_pa, tolerance):
    """Non-candidate interior wall facets still share nodes in the split mesh; check them with the
    same one-sided facet-normal criterion the network uses for initiation."""
    split_cells, split_regions = np.asarray(split.cells), np.asarray(split.region_ids)
    stress = np.zeros((len(split_cells), 3, 3))
    wall_cells = split_regions == WALL_REGION
    stress[wall_cells] = cell_stress_tensors(split.coordinates_m, split_cells[wall_cells], displacement_m, elasticity)
    pairs, normal = interior_wall_facet_normal_stress(split.coordinates_m, split_cells, split_regions, stress)
    critical = normal >= float(tensile_strength_pa) * (1 - tolerance)
    return pairs, critical, float(normal.max(initial=-np.inf))


def _opened(status):
    return np.isin(np.asarray(status), ("softening", "separated"))


def design_load_state(result, law):
    """Crack state at the design load factor and the interface stress margin to f_t."""
    initiated = np.asarray(result.initiated, dtype=bool)
    opened = _opened(result.face_status)
    stress = np.asarray(result.face_normal_stress_pa, dtype=float)
    released = stress[initiated]
    peak = float(released.max()) if len(released) else (float(stress.max()) if len(stress) else None)
    strength = float(law.tensile_strength_pa)
    first = next((step["load_factor"] for step in result.steps if step["initiated_faces"]), None)
    state = {
        "load_factor": float(result.load_factor),
        "initiated_faces": int(initiated.sum()), "opened_faces": int(opened.sum()),
        "initiated_area_m2": float(result.report.get("initiated_area_m2", 0.)),
        "opened_area_m2": float(result.report.get("opened_area_m2", 0.)),
        "peak_opening_m": float(np.max(result.face_opening_m, initial=0.)),
        "first_initiation_load_factor": first,
        "peak_stress_measure": ("interface traction on released facets" if len(released)
                                else "bonded candidate facet normal stress"),
        "peak_stress_pa": peak,
        "peak_stress_to_strength_ratio": None if peak is None else peak / strength,
        "linear_first_opening_load_factor_estimate": None,
    }
    if not opened.any() and peak is not None and peak > 0:
        # Valid while the released facets stay closed: the response is linear in the load factor.
        state["linear_first_opening_load_factor_estimate"] = float(result.load_factor) * strength / peak
    return state


def continue_overload(network, result, maximum_factor, increment, progress):
    """Proportional loading beyond the design load in fixed increments from the committed state.

    Returns (state, report). A stop other than cancellation keeps the last accepted
    equilibrium and records why loading could not continue.
    """
    if increment <= 0:
        raise ValueError("The overload increment must be positive")
    design_factor = float(result.load_factor)
    count = int(np.ceil((maximum_factor - design_factor) / increment - 1e-9))
    targets = [round(min(design_factor + (index + 1) * increment, maximum_factor), 9) for index in range(count)]
    steps = [dict(step, phase="design") for step in result.steps]
    current, stopped = result, None
    for target in targets:
        progress(f"overload search: loading {current.load_factor:.3g} -> {target:.3g} x design load")
        trial = network.run(current.load_factor, target, initiated=current.initiated,
                            history=current.maximum_opening_history_m, progress_target=maximum_factor)
        steps += [dict(step, phase="overload") for step in trial.steps]
        if trial.status == "cancelled":
            current, stopped = trial, {"status": trial.status, "message": trial.report.get("message"),
                                       "attempted_load_factor": target}
            break
        if trial.status not in ACCEPTED_OUTCOMES:
            stopped = {"status": trial.status, "message": trial.report.get("message"),
                       "attempted_load_factor": target}
            if trial.displacement_m is not None and trial.load_factor > current.load_factor:
                status = "accepted_equilibrium" if trial.initiated.any() else "no_damage_initiated"
                current = replace(trial, status=status, report=dict(
                    trial.report, status=status,
                    message=f"Last accepted overload step; loading stopped: {trial.report.get('message')}"))
            break
        current = trial
    current.steps = steps
    opening = [index for index, step in enumerate(steps) if step["opened_faces"]]
    first = opening[0] if opening else None
    report = {
        "requested_load_factor": float(maximum_factor), "increment": float(increment),
        "design_load_factor": design_factor, "reached_load_factor": float(current.load_factor),
        "first_opening_load_factor": steps[first]["load_factor"] if first is not None else None,
        "first_opening_bracket": ([steps[first - 1]["load_factor"] if first else design_factor,
                                   steps[first]["load_factor"]] if first is not None else None),
        "initiated_faces": int(np.asarray(current.initiated).sum()),
        "opened_faces": int(_opened(current.face_status).sum()),
        "initiated_area_m2": float(current.report.get("initiated_area_m2", 0.)),
        "opened_area_m2": float(current.report.get("opened_area_m2", 0.)),
        "peak_opening_m": float(np.max(current.face_opening_m, initial=0.)),
        "stopped": stopped,
        "basis": "every configured load (self-weight, reservoir and crack water) scaled together beyond the "
                 "design load; a strength-margin search, not a physical load case",
    }
    return current, report


def crack_verdict(design, overload, tensile_strength_pa):
    """One-paragraph headline of what the automatic cracking run found."""
    mpa = 1e-6

    def facets(count):
        return f"{count} facet" + ("" if count == 1 else "s")

    if design is None:
        return None
    if design["opened_faces"]:
        return (f"Cracks open at the design load: {facets(design['opened_faces'])} "
                f"({design['opened_area_m2']:.3g} m2), peak opening {design['peak_opening_m'] * 1e3:.3g} mm.")
    if design["initiated_faces"]:
        text = (f"No crack opens at the design load: {facets(design['initiated_faces'])} reached f_t on the "
                f"one-sided cell stress and were released, but their interface traction peaks at "
                f"{design['peak_stress_pa'] * mpa:.3g} MPa ({design['peak_stress_to_strength_ratio']:.0%} of "
                f"f_t {tensile_strength_pa * mpa:.3g} MPa), so they stay closed.")
    else:
        text = "No candidate facet reaches f_t at the design load."
    if overload is None:
        estimate = design["linear_first_opening_load_factor_estimate"]
        if estimate is not None:
            text += (f" Linear estimate: cracks would first open near {estimate:.3g}x the design load "
                     "(set damage_selection.maximum_overload_factor above 1 to solve beyond the design load).")
        return text
    if overload["first_opening_load_factor"] is not None:
        low, high = overload["first_opening_bracket"]
        text += (f" Under proportional overload cracks first open between {low:.3g}x and {high:.3g}x the design "
                 f"load; at {overload['reached_load_factor']:.3g}x, {facets(overload['opened_faces'])} "
                 f"({overload['opened_area_m2']:.3g} m2) open, peak opening "
                 f"{overload['peak_opening_m'] * 1e3:.3g} mm.")
    else:
        text += f" No crack opened up to {overload['reached_load_factor']:.3g}x the design load."
    if overload["stopped"]:
        text += (f" Overload stopped before {overload['requested_load_factor']:.3g}x: "
                 f"{overload['stopped']['message']}.")
    return text


def _bound_verdict(measure, peak_pa, tensile_strength_pa):
    text = (f"No crack at the design load: the intact {measure} {peak_pa * 1e-6:.3g} MPa is below f_t "
            f"{tensile_strength_pa * 1e-6:.3g} MPa.")
    if peak_pa > 0:
        text += f" Under linear proportional loading no facet initiates below {tensile_strength_pa / peak_pa:.3g}x the design load."
    return text


def maximum_overload_factor(config: dict) -> float:
    """The configured ``damage_selection.maximum_overload_factor`` (absent/null = design load only)."""
    selection = config.get("damage_selection")
    value = selection.get("maximum_overload_factor") if isinstance(selection, dict) else None
    if value is None:
        return 1.0
    if isinstance(value, bool) or not 1 <= float(value) <= MAXIMUM_OVERLOAD_FACTOR:
        raise ValueError(f"maximum_overload_factor must be between 1 and {MAXIMUM_OVERLOAD_FACTOR}")
    return float(value)


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


def _write_state(output, result, prepared, water_head, unit_weight, intact_stress=None, cracked_stress=None):
    if __package__:
        from .fracture_faces import write_surface_state
    else:
        from fracture_faces import write_surface_state
    split, faces = prepared.split, prepared.split.faces
    volume = {}
    if intact_stress is not None:
        # Volume fields for inspection: split cells, their source rows and both same-mesh states.
        adjacent = np.asarray(split.adjacent_cells).ravel()
        candidate_cells = np.zeros(len(split.cells), dtype=bool)
        candidate_cells[adjacent[adjacent >= 0]] = True
        volume = {"cells": np.asarray(split.cells), "region_ids": np.asarray(split.region_ids),
                  "source_cell_rows": np.asarray(prepared.original.parent_ids)[np.asarray(split.parent_cell_ids)],
                  "candidate_cells": candidate_cells, "cell_stress_pa": cracked_stress,
                  "intact_cell_stress_pa": intact_stress,
                  "intact_displacement_m": np.asarray(result.intact_displacement_m).reshape(-1, 3)}
    np.savez_compressed(
        output / "cohesive_state.npz",
        source_faces=split.source_faces, coordinates_m=faces.coordinates, minus=faces.minus, plus=faces.plus,
        normals=faces.normals, areas_m2=faces.areas, initiated=result.initiated,
        face_status=np.asarray(result.face_status, dtype="U16"), face_opening_m=result.face_opening_m,
        face_normal_stress_pa=result.face_normal_stress_pa,
        maximum_opening_history_m=result.maximum_opening_history_m, wet_faces=result.wet_faces,
        displacement_m=result.displacement_m.reshape(-1, 3), parent_node_ids=split.parent_node_ids,
        load_factor=result.load_factor, **volume)
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
        overload_factor = maximum_overload_factor(config)
        summary = {
            "runner_version": RUNNER_VERSION, "source_job_id": job_id,
            "config_digest": manifest["config_digest"],
            "screening_sha256": hashlib.sha256((Path(screening_output) / "screening.json").read_bytes()).hexdigest(),
            "tensile_strength_pa": strength, "fracture_energy_j_m2": float(energy),
            "fracture_energy_provenance": "user-supplied; not independently verified",
            "critical_opening_m": law.critical_opening_m,
            "initiation_relative_tolerance": tolerance,
            "loading": "all configured loads proportional 0 -> 1; reservoir crack water scaled by the same factor"
                       + (f"; if no crack opens at 1, continued in {OVERLOAD_INCREMENT:g} steps up to "
                          f"{overload_factor:g} (overload search)" if overload_factor > 1 else ""),
            "maximum_overload_factor": overload_factor,
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
                "verdict": _bound_verdict("peak principal stress", float(screening["peak_principal_stress_pa"]),
                                          strength),
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
                "verdict": _bound_verdict("peak facet normal stress", float(facets["peak_facet_normal_stress_pa"]),
                                          strength),
            })
        elif "outcome" not in summary:
            selection = screening.get("selection") if isinstance(screening.get("selection"), dict) else None
            confined = selection is not None and selection.get("max_damage_count") is not None
            if selection is not None:
                seed_rows = selection["selected_source_rows"]
                summary["damage_selection"] = {key: value for key, value in selection.items()
                                               if key != "selected_source_rows"}
                if not seed_rows:
                    raise ValueError("The automatic element selection is empty although a facet reaches f_t")
            else:
                seed_rows = facets["critical_source_rows"] if facets else None
            structure = submitted_structure(Path(configuration), Path(model_directory))
            result, prepared, region = run_network_on_structure(
                structure, screening, gf_j_m2=float(energy), tensile_strength_pa=strength,
                relative_tolerance=tolerance, maximum_volume_change=maximum_volume_change, rings=rings,
                maximum_candidate_faces=maximum_candidate_faces, maximum_solves=maximum_solves,
                progress=progress, seed_rows=seed_rows, confined=confined, overload_factor=overload_factor)
            design, overload = region.pop("design_load", None), region.pop("overload", None)
            summary["candidate_region"] = region
            if design is not None:
                summary["design_load"] = design
                summary["overload"] = overload
                summary["verdict"] = crack_verdict(design, overload, strength)
            if result is None:
                summary.update({"outcome": region["status"], "outcome_basis": region["message"]})
            else:
                outcome = (result.status if result.status not in ACCEPTED_OUTCOMES
                           or region["status"] in ("region_sufficient", "region_confined") else region["status"])
                basis = "cohesive network solve"
                if confined:
                    basis += (f" confined to the {selection['selected_cells']} selected elements "
                              f"(max_damage_count {selection['max_damage_count']}); facets outside the "
                              "selection were not allowed to open")
                summary.update({"outcome": outcome, "load_factor": result.load_factor,
                                "outcome_basis": basis, "network": result.report,
                                "steps": result.steps, "source_preparation": prepared.report})
                if result.displacement_m is not None:
                    loads = structure.loads
                    comparison = intact = cracked = None
                    if result.intact_displacement_m is not None:
                        progress("comparing cracked and intact stresses")
                        comparison, intact, cracked = stress_comparison(
                            prepared.split, structure.three_body_config["material"],
                            result.intact_displacement_m, result.displacement_m)
                        summary["stress_comparison"] = comparison
                    summary["opening_assessment"] = opening_assessment(result, law, comparison)
                    surface = _write_state(output, result, prepared, float(loads.upstream_surface_z_m),
                                           float(loads.water_density_kg_m3) * abs(float(loads.gravity_z_m_s2)),
                                           intact, cracked)
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
