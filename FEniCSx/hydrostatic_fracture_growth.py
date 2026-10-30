"""Experimental reservoir-connected, drained mesh-facet Griffith growth.

Reservoir work belongs in the potential: Pi=U-W_exterior-integral(p*w)dA.
Reference geometry and fixed head make p independent of displacement. This
does not define a growth criterion for dissipative different-head through-flow.
"""
from dataclasses import dataclass
import hashlib
import json
import numbers
from pathlib import Path

import numpy as np

if __package__:
    from .dry_fracture_growth import _solve_contact_growth
    from .fracture_faces import PairedCrackFaces, bonded_node_representatives
    from .fracture_hydrostatic import HydrostaticBoundary
    from .fracture_steady_flow import HydrostaticEquilibriumHydraulics
    from .reservoir_crack_load import ReservoirCrackHydraulics
else:
    from dry_fracture_growth import _solve_contact_growth
    from fracture_faces import PairedCrackFaces, bonded_node_representatives
    from fracture_hydrostatic import HydrostaticBoundary
    from fracture_steady_flow import HydrostaticEquilibriumHydraulics
    from reservoir_crack_load import ReservoirCrackHydraulics


@dataclass(frozen=True)
class HydrostaticGrowthEquilibrium:
    total_potential_j: float
    problem_sha256: str
    evidence: dict

    def __post_init__(self):
        if (isinstance(self.total_potential_j, bool)
                or not isinstance(self.total_potential_j, numbers.Real)
                or not np.isfinite(self.total_potential_j)):
            raise ValueError("Common-head total potential must be finite")
        if (not isinstance(self.problem_sha256, str) or len(self.problem_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.problem_sha256)):
            raise ValueError("Common-head growth requires SHA256-bound mechanical and reservoir identities")
        if (not isinstance(self.evidence, dict) or self.evidence.get("equilibrium_verified") is not True
                or self.evidence.get("common_head_reservoir_work_included") is not True):
            raise ValueError("Common-head growth requires equilibrium evidence including reservoir work")

    @property
    def potential_j(self):
        return self.total_potential_j


def solve_hydrostatic_contact_growth(
    space, faces, parent_node_ids, marker_edges, *, boundaries,
    water_density_kg_m3, gravity_m_s2, closure_tolerance_m, flow_relative_tolerance,
    young_pa, poisson, selected_cells, boundary_conditions, external_load,
    event_options, contact_options, growth_options, output,
    interface_springs=None, linear_support=None, additional_nodal_forces=None,
    opening_limit_m=.002, previously_broken=(), selected_tensile_strength_pa=0.,
    selected_young_pa=None,
):
    """Compare actual equilibria with pressure on reservoir-connected cuts.

    Unreleased inventory edges do not connect reservoirs. Disconnected cuts
    are explicitly unpressurised, not a fallback for failed wet equilibrium.
    Connected submerged faces must have open hydraulic access at acceptance.
    Prescribed-head pressure can open a closed trial without an artificial gap.
    No trapped-fluid state or wetting history is supplied.
    Inadmissible/failed trials fail the search instead of claiming arrest.
    """
    boundaries = tuple(boundaries)
    if not isinstance(faces, PairedCrackFaces):
        raise ValueError("Common-head growth requires explicit prepared paired candidate faces")
    if (not boundaries or any(not isinstance(b, HydrostaticBoundary) for b in boundaries)
            or len({b.name for b in boundaries}) != len(boundaries)):
        raise ValueError("Hydrostatic growth requires uniquely named reservoir boundaries")
    parents = np.asarray(parent_node_ids)
    if (parents.dtype.kind not in "iu" or parents.shape != (len(faces.coordinates),)
            or np.any(parents < 0) or not np.array_equal(parents[faces.minus], parents[faces.plus])):
        raise ValueError("Corresponding candidate lips require matching nonnegative parent source IDs")
    # Validate numerical hydraulic inputs even when the initial cut is empty.
    HydrostaticEquilibriumHydraulics(
        parents, boundaries[:1], water_density_kg_m3, gravity_m_s2,
        np.ones(1, dtype=bool), closure_tolerance_m, flow_relative_tolerance,
        allow_tied_fronts=True, inventory_face_ids=[0])
    source_edges = set()
    for triangle in parents[faces.minus]:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            source_edges.add(tuple(sorted(map(int, triangle[[i, j]]))))
    if any(tuple(sorted(map(int, edge))) not in source_edges
           for boundary in boundaries for edge in boundary.inlet_source_edges):
        raise ValueError("Reservoir edges must belong to the candidate source inventory")
    reservoir_identity = {
        "head_m": (float(boundaries[0].elevation_m)
                   if len({b.elevation_m for b in boundaries}) == 1 else None),
        "water_density_kg_m3": float(water_density_kg_m3), "gravity_m_s2": float(gravity_m_s2),
        "boundaries": [
            {"name": b.name, "head_m": float(b.elevation_m),
             "edges": sorted(sorted(map(int, edge)) for edge in b.inlet_source_edges)}
            for b in sorted(boundaries, key=lambda b: b.name)],
        "saturation": "drained reservoir-connected released surfaces; no trapped-fluid or wetting history",
    }

    def hydraulics(released):
        representatives = bonded_node_representatives(faces, ~released)
        edge_owners = {}
        source_edges = {}
        for face in np.flatnonzero(released):
            minus, plus = representatives[faces.minus[face]], representatives[faces.plus[face]]
            for i, j in ((0, 1), (1, 2), (2, 0)):
                key = tuple(sorted(tuple(sorted((int(minus[k]), int(plus[k]))))
                                   for k in (i, j)))
                edge_owners.setdefault(key, []).append(int(face))
                source = tuple(sorted(map(int, parents[minus[[i, j]]])))
                source_edges.setdefault(source, set()).add(key)
        if any(len(owners) > 2 for owners in edge_owners.values()):
            raise ValueError("Nonmanifold released hydraulic edges require a branch-junction formulation")
        adjacency = {int(face): set() for face in np.flatnonzero(released)}
        for owners in edge_owners.values():
            for face in owners:
                adjacency[face].update(set(owners) - {face})
        active = []
        seeds = set()
        for boundary in boundaries:
            edges = []
            for edge in boundary.inlet_source_edges:
                matches = source_edges.get(tuple(sorted(map(int, edge))), set())
                exterior = [key for key in matches if len(edge_owners[key]) == 1]
                if len(exterior) > 1:
                    raise ValueError("Reservoir inlet matches multiple unrelated split crack edges")
                if exterior:
                    edges.append(edge)
                    seeds.update(edge_owners[exterior[0]])
            if edges:
                active.append(HydrostaticBoundary(boundary.name, boundary.elevation_m, edges))
        if not active:
            return None
        if len({b.elevation_m for b in active}) != 1:
            raise ValueError(
                "Released crack connects different reservoir heads; a flow-coupled growth law is required")
        connected = set(seeds)
        pending = list(seeds)
        while pending:
            face = pending.pop()
            for other in adjacency[face] - connected:
                connected.add(other)
                pending.append(other)
        hydraulic_ids = np.array(sorted(connected), dtype=np.int64)
        return ReservoirCrackHydraulics(
            parents, tuple(active), water_density_kg_m3, gravity_m_s2,
            np.ones(len(hydraulic_ids), dtype=bool), closure_tolerance_m, flow_relative_tolerance,
            allow_tied_fronts=True, inventory_face_ids=hydraulic_ids)

    def problem_identity(report):
        return hashlib.sha256(json.dumps(
            {"mechanical": report["dry_problem_sha256"], "reservoir": reservoir_identity},
            sort_keys=True, allow_nan=False).encode()).hexdigest()

    def trial_bound(error):
        if __package__:
            from .dry_fracture_growth import UnacceptedTrialEnergyBound
        else:
            from dry_fracture_growth import UnacceptedTrialEnergyBound
        report = error.report
        return UnacceptedTrialEnergyBound(
            report["optimistic_pressure_potential_lower_bound_j"], problem_identity(report),
            {"mechanical_contact_equilibrium_verified": True,
             "hydraulic_equilibrium_accepted": False,
             "reason": report["hydraulic_access_error"],
             "opening": report["opening"],
             "final_material_contact_audit": report["iterations"][-1],
             "scope": "Nonnegative fully connected pressure maximises reservoir work for nonpenetrating gaps. The convex material/contact minimum is an optimistic energy bound only; it cannot accept an extension."})

    def state(event):
        report = event.report
        iterations = report.get("iterations")
        if (report.get("status") != "passed_reduced_tensile_contact_event" or not iterations
                or "scaled_force_imbalance" not in iterations[-1]):
            raise ValueError("Common-head growth requires accepted independently audited event equilibrium")
        potential = report.get("conservative_hydrostatic_potential_j")
        reservoir_work = report.get("hydrostatic_reservoir_work_j")
        if report.get("hydraulics") is None:
            potential = report["dry_mechanical_potential_j"]
            reservoir_work = 0.
        if potential is None or reservoir_work is None:
            raise ValueError("Wet growth cannot omit conservative common-head reservoir work")
        return HydrostaticGrowthEquilibrium(
            potential, problem_identity(report), {"equilibrium_verified": True,
                                  "common_head_reservoir_work_included": True,
                                  "reservoir_work_j": reservoir_work,
                                  "internal_water_state": (
                                      "reservoir_connected_released_crack" if report.get("hydraulics") is not None
                                      else "bonded_intact_cut" if not report["released_face_ids"]
                                      else "reservoir_disconnected_unpressurised_cut"),
                                  "opening_assessment": report["opening_assessment"],
                                  "final_opening_location_m": report["opening"]["peak_location_m"],
                                  "final_material_contact_audit": iterations[-1]})

    result = _solve_contact_growth(
        space, faces, parents, marker_edges, young_pa=young_pa, poisson=poisson,
        selected_tensile_strength_pa=selected_tensile_strength_pa, selected_young_pa=selected_young_pa,
        selected_cells=selected_cells, boundary_conditions=boundary_conditions, external_load=external_load,
        event_options=event_options, contact_options=contact_options, growth_options=growth_options,
        output=output, interface_springs=interface_springs, linear_support=linear_support,
        additional_nodal_forces=additional_nodal_forces, opening_limit_m=opening_limit_m,
        previously_broken=previously_broken, hydraulic_factory=hydraulics, state_factory=state,
        equilibrium_type=HydrostaticGrowthEquilibrium, mode="common_head_hydrostatic",
        problem_provenance={"reservoir_problem": reservoir_identity}, trial_bound_factory=trial_bound)
    result["reservoir_problem"] = reservoir_identity
    (Path(output) / "growth_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def source_reservoir_boundaries(prepared, structure):
    """Inventory exterior reservoir edges; access is tested on each released cut."""
    return imported_reservoir_boundaries(
        prepared.imported, structure, prepared.parent_node_ids,
        prepared.parent_node_ids[prepared.faces.minus])


def imported_reservoir_boundaries(imported, structure, parent_node_ids, candidate_triangles):
    """Same inlet proof before or after splitting, using inherited native tags."""
    from dolfinx import mesh
    if __package__:
        from .dam_elasticity import water_load_parameters
    else:
        from dam_elasticity import water_load_parameters
    domain = imported.domain
    gradient, elevations, constant = water_load_parameters(imported, structure)
    if gradient <= 0:
        raise ValueError("Source hydrostatic growth requires positive canonical water density/gravity")
    parents = np.asarray(parent_node_ids)
    candidate_edges = np.unique(np.sort(np.vstack((
        candidate_triangles[:, [0, 1]], candidate_triangles[:, [1, 2]],
        candidate_triangles[:, [2, 0]])), axis=1), axis=0)
    edge_dtype = np.dtype([("first", candidate_edges.dtype), ("second", candidate_edges.dtype)])
    candidate_keys = np.ascontiguousarray(candidate_edges).view(edge_dtype).ravel()
    result = []
    for tag in sorted(set(elevations) | set(constant)):
        facets = imported.facet_tags.find(tag)
        geometry = mesh.entities_to_geometry(domain, 2, facets)
        source = domain.geometry.input_global_indices[geometry]
        nodes = np.vstack((source[:, [0, 1]], source[:, [1, 2]], source[:, [2, 0]]))
        edges = np.ascontiguousarray(np.sort(parents[nodes], axis=1), dtype=candidate_edges.dtype)
        matches = np.isin(edges.view(edge_dtype).ravel(), candidate_keys)
        matched = {tuple(map(int, edge)): imported.points[row, 2]
                   for edge, row in zip(edges[matches], nodes[matches])}
        if tag in constant:
            if constant[tag] > 0 and matched:
                z = np.concatenate(list(matched.values()))
                if not np.allclose(z, z[0], rtol=0., atol=1e-10):
                    raise ValueError("Constant-pressure inlet must be horizontal to define a reservoir head")
                result.append(HydrostaticBoundary(
                    f"exterior_tag_{tag}", float(z[0] + constant[tag] / gradient), sorted(matched)))
            continue
        head = elevations[tag]
        inlets = []
        for edge, z in sorted(matched.items()):
            if z.min() < head:
                inlets.append(edge)
        if inlets:
            result.append(HydrostaticBoundary(f"exterior_tag_{tag}", head, inlets))
    if not result:
        raise ValueError("No submerged candidate edge belongs to a canonical exterior reservoir boundary")
    return tuple(result)


def solve_source_hydrostatic_growth(
    structure, affected_source_rows, marker_source_edges, *, maximum_volume_change,
    event_options, contact_options, growth_options, closure_tolerance_m,
    flow_relative_tolerance, output, opening_limit_m=.002, selection_provenance=None,
    selected_tensile_strength_pa=0., selected_young_pa=None,
):
    """Experimental canonical source adapter with proven exterior hydraulic inlets."""
    if __package__:
        from .dry_fracture_growth import _solve_source_growth
    else:
        from dry_fracture_growth import _solve_source_growth

    def driver(prepared, space, **kwargs):
        boundaries = source_reservoir_boundaries(prepared, structure)
        return solve_hydrostatic_contact_growth(
            space, prepared.faces, prepared.parent_node_ids, marker_source_edges,
            boundaries=boundaries, water_density_kg_m3=structure.loads.water_density_kg_m3,
            gravity_m_s2=abs(structure.loads.gravity_z_m_s2),
            closure_tolerance_m=closure_tolerance_m,
            flow_relative_tolerance=flow_relative_tolerance, **kwargs)

    return _solve_source_growth(
        structure, affected_source_rows, marker_source_edges,
        maximum_volume_change=maximum_volume_change, event_options=event_options,
        contact_options=contact_options, growth_options=growth_options, output=output,
        opening_limit_m=opening_limit_m, selection_provenance=selection_provenance,
        selected_tensile_strength_pa=selected_tensile_strength_pa, selected_young_pa=selected_young_pa,
        prepared_driver=driver, mode="common_head_hydrostatic",
        water_model="canonical exterior-tag reservoirs; pressure only on connected released surfaces; exact waterline clipping; no filling history")


def solve_bound_hydrostatic_growth(
    configuration, model_directory, baseline_directory, fields, request, *,
    maximum_volume_change, event_options, contact_options, growth_options,
    closure_tolerance_m, flow_relative_tolerance, output,
):
    """Verified painting/baseline binding; still experimental mesh-facet growth."""
    if __package__:
        from .tensile_loss_event import verified_source_selection
        from .dry_fracture_growth import _record_bound_growth
    else:
        from tensile_loss_event import verified_source_selection
        from dry_fracture_growth import _record_bound_growth
    structure, binding = verified_source_selection(
        configuration, model_directory, baseline_directory, fields, request)
    result = solve_source_hydrostatic_growth(
        structure, binding["affected_element_ids"], binding["marker_source_edges"],
        maximum_volume_change=maximum_volume_change, event_options=event_options,
        contact_options=contact_options, growth_options=growth_options,
        closure_tolerance_m=closure_tolerance_m, flow_relative_tolerance=flow_relative_tolerance,
        output=output, opening_limit_m=binding["opening_limit_m"], selection_provenance=binding)
    return _record_bound_growth(result, output)
