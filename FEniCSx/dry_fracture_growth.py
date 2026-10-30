"""Experimental irreversible finite-candidate Griffith growth at dry equilibrium.

This is a discrete energy search, not an unrestricted crack tracking method.
The caller supplies justified surface patches and an audited equilibrium solve.
No hydraulic energy, wet propagation, nucleation strength or stability verdict
is inferred from the mechanical potential.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import numbers
from pathlib import Path
from typing import Callable

import numpy as np


def report_progress(message: str) -> None:
    """One flushed ``Progress:`` line; the app's activity monitor shows the latest one."""
    print(f"Progress: {message}", flush=True)


FRACTURE_PHASES = ("preparing", "initial_equilibrium", "growth")


def report_phase(phase: str) -> None:
    """One flushed ``Phase:`` line with the solver's UTC clock; the app maps it to pipeline stages."""
    if phase not in FRACTURE_PHASES:
        raise ValueError(f"Unknown fracture phase {phase!r}")
    print("Phase: " + json.dumps({"phase": phase, "time": datetime.now(timezone.utc).isoformat()}), flush=True)


@dataclass(frozen=True)
class DryGrowthEquilibrium:
    mechanical_potential_j: float
    problem_sha256: str
    evidence: dict

    def __post_init__(self):
        if (isinstance(self.mechanical_potential_j, bool)
                or not isinstance(self.mechanical_potential_j, numbers.Real)
                or not np.isfinite(self.mechanical_potential_j)):
            raise ValueError("Mechanical potential must be finite")
        if (not isinstance(self.problem_sha256, str) or len(self.problem_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.problem_sha256)):
            raise ValueError("Equilibria require the same explicit SHA256-bound mechanical problem")
        if (not isinstance(self.evidence, dict) or self.evidence.get("equilibrium_verified") is not True
                or self.evidence.get("dry_conservative_problem") is not True):
            raise ValueError("Dry growth requires independently verified conservative dry equilibrium evidence")

    @property
    def potential_j(self):
        return self.mechanical_potential_j


def contact_event_growth_state(event) -> DryGrowthEquilibrium:
    """Bind the existing integrated material/contact event to dry energy search."""
    report = event.report
    iterations = report.get("iterations")
    if (report.get("status") != "passed_reduced_tensile_contact_event"
            or report.get("dry_mechanical_potential_j") is None
            or not iterations or "scaled_force_imbalance" not in iterations[-1]
            or report.get("hydraulics") is not None):
        raise ValueError("Growth requires an accepted dry material/contact event with force audit and potential")
    return DryGrowthEquilibrium(
        report["dry_mechanical_potential_j"], report["dry_problem_sha256"],
        {"equilibrium_verified": True, "dry_conservative_problem": True,
         "event_status": report["status"], "cut_released": report["cut_released"],
         "opening_assessment": report["opening_assessment"],
         "final_opening_location_m": report["opening"]["peak_location_m"],
         "final_material_contact_audit": iterations[-1]},
    )


@dataclass(frozen=True)
class UnacceptedTrialEnergyBound:
    potential_lower_bound_j: float
    problem_sha256: str
    evidence: dict

    def __post_init__(self):
        if (isinstance(self.potential_lower_bound_j, bool)
                or not isinstance(self.potential_lower_bound_j, numbers.Real)
                or not np.isfinite(self.potential_lower_bound_j)
                or not isinstance(self.problem_sha256, str) or len(self.problem_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.problem_sha256)
                or not isinstance(self.evidence, dict)
                or self.evidence.get("mechanical_contact_equilibrium_verified") is not True):
            raise ValueError("An audited, problem-bound finite trial energy bound is required")


@dataclass(frozen=True)
class GrowthOptions:
    fracture_energy_j_m2: float
    energy_tolerance_j: float
    maximum_extensions: int
    maximum_patch_faces: int = 1
    resolve_locked_patches: bool = False

    def __post_init__(self):
        for name in ("fracture_energy_j_m2", "energy_tolerance_j"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if type(self.maximum_extensions) is not int or self.maximum_extensions < 1:
            raise ValueError("maximum_extensions must be a positive integer")
        if type(self.maximum_patch_faces) is not int or self.maximum_patch_faces < 1:
            raise ValueError("maximum_patch_faces must be a positive integer")
        if type(self.resolve_locked_patches) is not bool:
            raise ValueError("resolve_locked_patches must be an explicit boolean")


def grow_dry_fracture(
    coordinates_m, surface_triangles, marker_edges, *,
    equilibrate: Callable[[tuple[int, ...], Path], DryGrowthEquilibrium],
    options: GrowthOptions, output: Path, previously_broken=(),
) -> dict:
    """Dry specialization of the shared edge-connected conservative search."""
    return _grow_conservative_fracture(
        coordinates_m, surface_triangles, marker_edges, equilibrate=equilibrate,
        options=options, output=output, previously_broken=previously_broken,
        equilibrium_type=DryGrowthEquilibrium, mode="dry")


def _grow_conservative_fracture(
    coordinates_m, surface_triangles, marker_edges, *, equilibrate,
    options, output, previously_broken, equilibrium_type, mode, problem_provenance=None,
    opening_limit_m=None, trial_admissible=None, resolve_patch=None,
):
    """Greedy energy descent over edge-connected explicit candidate patches.

    Each trial re-equilibrates the ENTIRE accumulated cut at unchanged loads,
    supports and selected material loss. Failed/unresolved trial solves abort
    the search; they are never treated as evidence of arrested growth.
    Previously broken faces cannot heal. Area counts one geometric surface,
    not both lips. A tie within energy tolerance remains unbroken.
    Arrest means no tested patch up to the explicit size bound lowered energy;
    it does not exclude larger jumps or unrepresented orientations.
    """
    if not isinstance(options, GrowthOptions):
        raise ValueError("Explicit growth options are required")
    if options.resolve_locked_patches and (resolve_patch is None or trial_admissible is None):
        raise ValueError("Resolving locked patches requires native node-star topology and an admissibility audit")
    xyz = np.asarray(coordinates_m)
    triangles = np.asarray(surface_triangles)
    edges = np.asarray(marker_edges)
    if (xyz.dtype.kind not in "fiu" or xyz.ndim != 2 or xyz.shape[1] != 3
            or not np.isfinite(xyz).all() or triangles.dtype.kind not in "iu"
            or triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or np.any(triangles < 0) or np.any(triangles >= len(xyz))
            or any(len(set(row)) != 3 for row in triangles)
            or len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles)
            or edges.dtype.kind not in "iu" or edges.ndim != 2 or edges.shape[1] != 2
            or not len(edges) or np.any(edges < 0) or np.any(edges >= len(xyz))
            or np.any(edges[:, 0] == edges[:, 1])
            or len(np.unique(np.sort(edges, axis=1), axis=0)) != len(edges)):
        raise ValueError("Explicit unique surface triangles/marker edges and finite coordinates are required")
    points = xyz[triangles]
    area = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1) / 2
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Candidate surface triangles must have positive finite area")
    face_edges = [{tuple(sorted(map(int, row[[a, b]]))) for a, b in ((0, 1), (1, 2), (2, 0))}
                  for row in triangles]
    edge_faces = {}
    if options.maximum_patch_faces > 1:
        for face, face_edge_set in enumerate(face_edges):
            for edge in face_edge_set:
                edge_faces.setdefault(edge, set()).add(face)
    markers = {tuple(sorted(map(int, row))) for row in edges}
    if not markers.issubset(set().union(*face_edges)):
        raise ValueError("Every marker must be an actual candidate surface edge")
    prior = tuple(previously_broken)
    if (any(type(i) is not int or not 0 <= i < len(triangles) for i in prior)
            or len(set(prior)) != len(prior)):
        raise ValueError("Previously broken surface IDs must be unique candidate indices")
    broken = set(prior)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "growth_inputs.npz", coordinates_m=xyz,
                        surface_triangles=triangles, marker_edges=edges,
                        previously_broken=np.array(prior, dtype=np.int64))
    report = {
        "status": f"running_experimental_{mode}_growth", "solver_ready": False,
        "potential_model": mode,
        "problem_provenance": problem_provenance,
        "engineering_validated": False, "fracture_energy_j_m2": options.fracture_energy_j_m2,
        "energy_tolerance_j": options.energy_tolerance_j, "maximum_extensions": options.maximum_extensions,
        "maximum_patch_faces": options.maximum_patch_faces,
        "resolve_locked_patches": options.resolve_locked_patches,
        "generated_node_star_patches": [],
        "accepted_surface_ids": sorted(broken), "trials": [], "accepted_extensions": [],
        "excluded_kinematically_locked_patches": [],
        "initial_equilibrium_started": False,
        "accepted_equilibria": [],
        "limitations": "Greedy conservative finite-candidate energy search only. The explicit patch bound may be augmented by audited node-star completions when enabled. Other larger jumps, mesh-independent paths, tensile-strength nucleation, general wet growth and engineering stability are not solved.",
    }
    arrested_status = ("arrested_in_tested_node_star_patch_candidate_set" if options.resolve_locked_patches
                       else "arrested_in_tested_single_face_candidate_set" if options.maximum_patch_faces == 1
                       else "arrested_in_tested_connected_patch_candidate_set")

    observations = []

    def record_state(state):
        if opening_limit_m is not None:
            if __package__:
                from .horizontal_fracture_contract import EquilibriumOpeningObservation
            else:
                from horizontal_fracture_contract import EquilibriumOpeningObservation
            assessment = state.evidence.get("opening_assessment")
            if not isinstance(assessment, dict) or assessment.get("completed") is not True:
                raise ValueError("Native growth requires measured opening evidence for every accepted equilibrium")
            prefix = f"accepted_growth_{len(report['accepted_equilibria']):04d}"
            observations.extend((
                EquilibriumOpeningObservation(
                    prefix + "_peak", assessment["peak_opening_m"],
                    tuple(assessment["peak_opening_location_m"]), assessment["fracture_initiated"]),
                EquilibriumOpeningObservation(
                    prefix + "_final", assessment["final_opening_m"],
                    tuple(state.evidence["final_opening_location_m"]), assessment["fracture_initiated"])))
        report["accepted_equilibria"].append({
            "surface_ids": sorted(broken), "potential_j": float(state.potential_j),
            "evidence": state.evidence})

    def persist():
        if observations:
            if __package__:
                from .horizontal_fracture_contract import assess_equilibrium_opening
            else:
                from horizontal_fracture_contract import assess_equilibrium_opening
            report["opening_assessment"] = assess_equilibrium_opening(
                opening_limit_m, tuple(observations),
                completed=report["status"] == arrested_status)
            report["opening_history"] = [
                {**asdict(item), "location_m": list(item.location_m)} for item in observations]
        (output / "growth_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def candidate_patches():
        front = markers | set().union(*(face_edges[i] for i in broken))
        candidates = [i for i in range(len(triangles)) if i not in broken and face_edges[i] & front]
        patches = {(i,) for i in candidates}
        layer = patches.copy()
        for _ in range(1, min(options.maximum_patch_faces, len(triangles) - len(broken))):
            expanded = set()
            for patch in layer:
                neighbours = set().union(*(
                    edge_faces[edge] for face in patch for edge in face_edges[face]))
                expanded.update(tuple(sorted((*patch, face)))
                                for face in neighbours - broken - set(patch))
            patches.update(expanded)
            layer = expanded
            if not layer:
                break
        return sorted(patches, key=lambda ids: (len(ids), ids))

    def screen_patches(patches, step):
        admissible = set()
        current_ids = tuple(sorted(broken))
        for patch in patches:
            proposed = tuple(sorted(broken | set(patch)))
            if trial_admissible is None or trial_admissible(current_ids, proposed):
                admissible.add(patch)
            else:
                report["excluded_kinematically_locked_patches"].append({
                    "step": step, "candidate_surface_ids": list(patch),
                    "reason": "Remaining all-component bonds still tie every released lip pair; no new opening degree of freedom",
                })
                if options.resolve_locked_patches:
                    for resolved in resolve_patch(current_ids, patch):
                        proposed = tuple(sorted(broken | set(resolved)))
                        if trial_admissible(current_ids, proposed) and resolved not in admissible:
                            admissible.add(resolved)
                            report["generated_node_star_patches"].append({
                                "step": step, "seed_surface_ids": list(patch),
                                "candidate_surface_ids": list(resolved),
                                "method": "minimum-cardinality node-star bond cut; full geometric area charged",
                            })
        if patches and not admissible:
            report["failure_kind"] = "no_kinematically_admissible_extension"
            raise ValueError(
                "Every tested patch remains tied shut by neighbouring bonded faces. "
                "No opening-capable crack extension was tested; this is NOT evidence of crack arrest. "
                "A larger connected surface patch or a different fracture discretisation is required.")
        return list(enumerate(sorted(admissible, key=lambda ids: (len(ids), ids))))

    persist()
    try:
        report_progress("checking candidate opening degrees of freedom before initial equilibrium")
        patches = candidate_patches()
        admissible = screen_patches(patches, 0)
        report["initial_equilibrium_started"] = True
        persist()
        report_phase("initial_equilibrium")
        report_progress("equilibrium of the painted source (initial solve)")
        current = equilibrate(tuple(sorted(broken)), output / "initial")
        if not isinstance(current, equilibrium_type):
            raise ValueError(f"Equilibrium callback must return {equilibrium_type.__name__}")
        report["problem_sha256"] = current.problem_sha256
        report["initial_potential_j"] = float(current.potential_j)
        record_state(current)
        persist()
        extensions = 0
        while True:
            if extensions:
                patches = candidate_patches()
                admissible = screen_patches(patches, extensions)
            best = None
            if extensions == 0:
                report_phase("growth")
            report_progress(f"growth step {extensions + 1}: testing {len(admissible)} candidate extensions")
            for patch_index, patch in admissible:
                proposed = tuple(sorted(broken | set(patch)))
                report_progress(f"growth step {extensions + 1}: candidate {patch_index + 1}/{len(admissible)}")
                suffix = f"{patch[0]:06d}" if len(patch) == 1 else f"patch_{patch_index:06d}"
                state = equilibrate(proposed,
                                    output / f"step_{extensions:04d}_candidate_{suffix}")
                if isinstance(state, UnacceptedTrialEnergyBound):
                    if state.problem_sha256 != current.problem_sha256:
                        raise ValueError("Trial energy bound changed the bound loads/supports/material problem")
                    new_area = float(area[list(patch)].sum())
                    cost = options.fracture_energy_j_m2 * new_area
                    release_upper = float(current.potential_j - state.potential_lower_bound_j)
                    change_lower = cost - release_upper
                    if not np.isfinite([cost, release_upper, change_lower]).all():
                        raise RuntimeError("Fracture energy bound arithmetic overflowed")
                    report["trials"].append({
                        "step": extensions, "candidate": patch[0] if len(patch) == 1 else None,
                        "candidate_surface_ids": list(patch), "new_surface_area_m2": new_area,
                        "surface_energy_cost_j": cost,
                        "conservative_energy_release_upper_bound_j": release_upper,
                        "total_energy_change_lower_bound_j": change_lower,
                        "assessment": ("unresolved_hydraulic_trial_energy_bound"
                                       if change_lower < -options.energy_tolerance_j
                                       else "rejected_by_optimistic_pressure_energy_bound"),
                        "evidence": state.evidence,
                    })
                    persist()
                    if change_lower < -options.energy_tolerance_j:
                        report["failure_kind"] = "unresolved_energetically_favourable_hydraulic_trial"
                        raise RuntimeError(
                            "An optimistic pressure bound is energetically favourable but has no open "
                            "reservoir path; a coupled wet/contact state is unresolved, not arrested")
                    continue
                if not isinstance(state, equilibrium_type) or state.problem_sha256 != current.problem_sha256:
                    raise ValueError("Trial equilibrium changed the bound loads/supports/material problem")
                new_area = float(area[list(patch)].sum())
                cost = options.fracture_energy_j_m2 * new_area
                release = float(current.potential_j - state.potential_j)
                change = cost - release
                if not np.isfinite([cost, release, change]).all():
                    raise RuntimeError("Fracture energy arithmetic overflowed")
                row = {"step": extensions, "candidate": patch[0] if len(patch) == 1 else None,
                       "candidate_surface_ids": list(patch), "new_surface_area_m2": new_area,
                       "conservative_energy_release_j": release, "surface_energy_cost_j": cost,
                       "total_energy_change_j": change}
                if mode == "dry":
                    row["mechanical_energy_release_j"] = release
                report["trials"].append(row)
                persist()
                if change < -options.energy_tolerance_j and (best is None or change < best[0]):
                    best = (change, patch, state, row)
            if best is None:
                report_progress(f"growth arrested after {extensions} accepted extension(s)")
                report["status"] = arrested_status
                report["final_potential_j"] = float(current.potential_j)
                report["total_fracture_surface_energy_j"] = float(
                    options.fracture_energy_j_m2 * area[list(broken)].sum())
                report["new_fracture_surface_energy_j"] = float(
                    options.fracture_energy_j_m2 * area[list(broken - set(prior))].sum())
                persist()
                return report
            if extensions >= options.maximum_extensions:
                raise RuntimeError("Growth extension limit reached with an energetically favourable trial; not arrested")
            _, patch, current, row = best
            report_progress(f"growth step {extensions + 1}: accepted {len(patch)} new face(s)")
            broken.update(patch)
            report["accepted_surface_ids"] = sorted(broken)
            report["accepted_extensions"].append(row)
            record_state(current)
            extensions += 1
            persist()
    except (ValueError, RuntimeError) as error:
        report["status"] = f"failed_experimental_{mode}_growth"
        report["error"] = str(error)
        persist()
        raise RuntimeError(f"{mode} growth failed; no arrested-growth result accepted: {error}") from error


def solve_dry_contact_growth(
    space, faces, parent_node_ids, marker_edges, *, young_pa, poisson,
    selected_cells, boundary_conditions, external_load, event_options,
    contact_options, growth_options: GrowthOptions, output: Path,
    interface_springs=None, linear_support=None, additional_nodal_forces=None,
    opening_limit_m=0.002, previously_broken=(), selected_tensile_strength_pa=0.,
):
    """Dry native selected-material/contact candidate growth; no crack pressure."""
    return _solve_contact_growth(
        space, faces, parent_node_ids, marker_edges, young_pa=young_pa, poisson=poisson,
        selected_cells=selected_cells, boundary_conditions=boundary_conditions,
        external_load=external_load, event_options=event_options, contact_options=contact_options,
        growth_options=growth_options, output=output, interface_springs=interface_springs,
        linear_support=linear_support, additional_nodal_forces=additional_nodal_forces,
        opening_limit_m=opening_limit_m, previously_broken=previously_broken,
        selected_tensile_strength_pa=selected_tensile_strength_pa,
        hydraulic_factory=None, state_factory=contact_event_growth_state,
        equilibrium_type=DryGrowthEquilibrium, mode="dry")


def _solve_contact_growth(
    space, faces, parent_node_ids, marker_edges, *, young_pa, poisson,
    selected_cells, boundary_conditions, external_load, event_options,
    contact_options, growth_options, output, interface_springs, linear_support,
    additional_nodal_forces, opening_limit_m, previously_broken,
    hydraulic_factory, state_factory, equilibrium_type, mode, problem_provenance=None,
    selected_tensile_strength_pa=0., trial_bound_factory=None,
):
    """Native candidate search retaining unbroken faces as full vector bonds.

    Requires a prepared paired candidate mesh with canonical parent identities.
    No user-painted surface is manufactured or seeded with an artificial gap.
    Dry and common-head specializations supply distinct audited energy states.
    This mesh-constrained experimental path is not a production dam route.
    """
    if __package__:
        from .fracture_faces import bulk_face_mapping, bonded_node_representatives
        from .tensile_loss_event import solve_contact_event, UnacceptedHydrostaticBound
    else:
        from fracture_faces import bulk_face_mapping, bonded_node_representatives
        from tensile_loss_event import solve_contact_event, UnacceptedHydrostaticBound
    coordinates, _, _ = bulk_face_mapping(space, faces)
    parents = np.asarray(parent_node_ids)
    if (parents.dtype.kind not in "iu" or parents.shape != (len(coordinates),)
            or np.any(parents < 0) or not np.array_equal(parents[faces.minus], parents[faces.plus])):
        raise ValueError("Corresponding candidate lips require matching nonnegative parent source IDs")
    triangles = parents[faces.minus]
    if len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles):
        raise ValueError("Candidate parent triangles must be unique; unrelated coincident sheets cannot be welded")
    # Compact parent identities without allocating by an arbitrary source ID.
    ids, first, inverse = np.unique(parents, return_index=True, return_inverse=True)
    source_points = coordinates[first].copy()
    if not np.allclose(coordinates, source_points[inverse], rtol=0., atol=1e-12):
        raise ValueError("Parent identity maps to inconsistent candidate coordinates")
    edges = np.asarray(marker_edges)
    if edges.dtype.kind not in "iu" or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Marker edges must be explicit parent source-node pairs")
    mapped_edges = np.searchsorted(ids, edges)
    if np.any(mapped_edges >= len(ids)) or not np.array_equal(ids[mapped_edges], edges):
        raise ValueError("Marker source identity is absent from the prepared mesh")
    bcs = tuple(boundary_conditions)
    accepted_paths = {}
    initial_event = None
    bond_class_counts = {}
    prepared_inputs_written = False

    def persist_prepared_inputs():
        nonlocal prepared_inputs_written
        if not prepared_inputs_written:
            np.savez_compressed(
                output / "prepared_candidate_inputs.npz", coordinates_m=coordinates,
                parent_node_ids=parents, compact_source_ids=ids,
                candidate_parent_triangles=triangles, marker_source_edges=edges,
                minus=faces.minus, plus=faces.plus, normals=faces.normals,
            )
            prepared_inputs_written = True

    def trial_admissible(current, proposed):
        persist_prepared_inputs()
        for surface_ids in (current, proposed):
            if surface_ids not in bond_class_counts:
                bonded = np.ones(len(faces.minus), dtype=bool)
                bonded[list(surface_ids)] = False
                representatives = bonded_node_representatives(faces, bonded)
                bond_class_counts[surface_ids] = len(np.unique(representatives))
        return bond_class_counts[proposed] > bond_class_counts[current]

    def equilibrate(surface_ids, directory):
        nonlocal initial_event
        persist_prepared_inputs()
        released = np.zeros(len(faces.minus), dtype=bool)
        released[list(surface_ids)] = True
        try:
            event = solve_contact_event(
                space, faces, young_pa=young_pa, poisson=poisson, selected_cells=selected_cells,
                boundary_conditions=bcs, external_load=external_load, options=event_options,
                contact_options=contact_options, output=directory, interface_springs=interface_springs,
                linear_support=linear_support, additional_nodal_forces=additional_nodal_forces,
                opening_limit_m=opening_limit_m, released_face_mask=released,
                selected_tensile_strength_pa=selected_tensile_strength_pa,
                equilibrium_seed=initial_event,
                hydraulics=hydraulic_factory(released) if hydraulic_factory is not None and released.any() else None,
            )
        except UnacceptedHydrostaticBound as error:
            if trial_bound_factory is None:
                raise
            return trial_bound_factory(error)
        if initial_event is None:
            initial_event = event
        accepted_paths[tuple(surface_ids)] = directory
        return state_factory(event)

    output = Path(output)
    if growth_options.resolve_locked_patches:
        if __package__:
            from .fracture_topology import NodeStarPatchResolver
        else:
            from fracture_topology import NodeStarPatchResolver
        resolver = NodeStarPatchResolver(faces, parents)
    else:
        resolver = None
    result = _grow_conservative_fracture(
        source_points, inverse[faces.minus], mapped_edges, equilibrate=equilibrate,
        options=growth_options, output=output, previously_broken=previously_broken,
        equilibrium_type=equilibrium_type, mode=mode, problem_provenance=problem_provenance,
        opening_limit_m=opening_limit_m, trial_admissible=trial_admissible, resolve_patch=resolver)
    final = accepted_paths[tuple(result["accepted_surface_ids"])]
    result["accepted_event_directory"] = str(final.relative_to(output))
    result["prepared_candidate_scope"] = f"{mode} mesh-constrained selected-material/contact search; not unrestricted wet growth"
    (output / "growth_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def solve_bound_dry_growth(
    configuration: Path, model_directory: Path, baseline_directory: Path, fields: Path,
    request: dict, *, maximum_volume_change, event_options, contact_options,
    growth_options: GrowthOptions, output: Path,
):
    """Verified selection adapter for experimental dry, mesh-constrained growth."""
    if __package__:
        from .tensile_loss_event import verified_source_selection
    else:
        from tensile_loss_event import verified_source_selection
    structure, binding = verified_source_selection(
        configuration, model_directory, baseline_directory, fields, request)
    result = solve_source_dry_growth(
        structure, binding["affected_element_ids"], binding["marker_source_edges"],
        maximum_volume_change=maximum_volume_change, event_options=event_options,
        contact_options=contact_options, growth_options=growth_options, output=output,
        opening_limit_m=binding["opening_limit_m"], selection_provenance=binding)
    return _record_bound_growth(result, output)


def _record_bound_growth(result, output):
    """Mark binding only after the verified source specialization succeeds."""
    result["source_binding_verified"] = True
    output = Path(output)
    (output / "growth_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    source = json.loads((output / "source_report.json").read_text())
    source["source_binding_verified"] = True
    (output / "source_report.json").write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
    return result


def solve_source_dry_growth(
    structure, affected_source_rows, marker_source_edges, *, maximum_volume_change,
    event_options, contact_options, growth_options: GrowthOptions, output: Path,
    opening_limit_m=0.002, selection_provenance=None, selected_tensile_strength_pa=0.,
):
    """Experimental source-to-equilibrium dry mesh-facet energy search.

    Original exterior water/gravity loads are retained. "Dry" refers only to
    the newly created crack surfaces; no internal crack pressure is introduced.
    The supplied structure must already be bound to the user's source snapshot.
    This does not bypass baseline verification or create an API submission route.
    """
    def driver(prepared, space, **kwargs):
        return solve_dry_contact_growth(
            space, prepared.faces, prepared.parent_node_ids, marker_source_edges, **kwargs)

    return _solve_source_growth(
        structure, affected_source_rows, marker_source_edges,
        maximum_volume_change=maximum_volume_change, event_options=event_options,
        contact_options=contact_options, growth_options=growth_options, output=output,
        opening_limit_m=opening_limit_m, selection_provenance=selection_provenance,
        selected_tensile_strength_pa=selected_tensile_strength_pa,
        prepared_driver=driver, mode="dry",
        water_model="explicitly dry new surfaces; unchanged exterior reservoir/tailwater loads")


def _solve_source_growth(
    structure, affected_source_rows, marker_source_edges, *, maximum_volume_change,
    event_options, contact_options, growth_options, output, opening_limit_m,
    selection_provenance, prepared_driver, mode, water_model, selected_tensile_strength_pa=0.,
):
    """Canonical source/load/support preparation shared by growth specializations."""
    from dolfinx import fem
    import ufl

    if __package__:
        from .dam_mesh import prepare_wall_candidates, source_cell_tags, selected_wall_cells
        from .dam_elasticity import external_load, water_load_parameters
        from .hydrostatic_load import assemble_water_forces
        from .interface_springs import InterfaceSprings
        from .three_body_model import is_three_body, materials, require_support_tags, support_bcs, support_form
    else:
        from dam_mesh import prepare_wall_candidates, source_cell_tags, selected_wall_cells
        from dam_elasticity import external_load, water_load_parameters
        from hydrostatic_load import assemble_water_forces
        from interface_springs import InterfaceSprings
        from three_body_model import is_three_body, materials, require_support_tags, support_bcs, support_form
    if not is_three_body(structure):
        raise ValueError("Source growth requires the canonical multi-body structure")
    report_progress("preparing crack candidates on the native mesh (can take tens of minutes)")
    prepared = prepare_wall_candidates(structure, marker_source_edges, maximum_volume_change)
    report_progress("assembling loads, supports and interface springs")
    imported = prepared.imported
    source_tags = source_cell_tags(imported)
    selected = selected_wall_cells(source_tags, imported.cell_tags, affected_source_rows)
    space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
    coefficients = materials(imported, structure)
    require_support_tags(imported, structure)
    bcs = support_bcs(imported, space, structure)
    ds = ufl.Measure("ds", domain=imported.domain, subdomain_data=imported.facet_tags)
    forces = assemble_water_forces(imported, space, *water_load_parameters(imported, structure))
    springs = InterfaceSprings(imported, structure, space,
                               multiplier_parent_node_ids=prepared.parent_node_ids)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_report = {
        "status": f"running_experimental_source_{mode}_growth",
        "solver_ready": False, "engineering_validated": False,
        "source_preparation": prepared.report, "source_binding_verified": False,
        "selection_provenance": selection_provenance,
        "internal_crack_water_model": water_model,
        "selected_tensile_strength_pa": selected_tensile_strength_pa,
    }

    def persist_source():
        (output / "source_report.json").write_text(json.dumps(source_report, indent=2, allow_nan=False) + "\n")

    persist_source()
    try:
        np.savez_compressed(
            output / "source_mapping.npz", local_source_cell_rows=source_tags.values,
            selected_local_cells=selected, candidate_source_triangles=prepared.source_triangles,
            parent_node_ids=prepared.parent_node_ids, exterior_water_forces_n=forces,
        )
        result = prepared_driver(
            prepared, space,
            young_pa=coefficients["youngs_modulus"], poisson=coefficients["poisson_ratio"],
            selected_cells=selected, boundary_conditions=bcs,
            external_load=lambda v: external_load(imported, structure, coefficients["density"],
                                                  v, ds, include_water=False)[0],
            linear_support=lambda u, v: support_form(u, v, ds, structure),
            interface_springs=springs, additional_nodal_forces=forces,
            event_options=event_options, contact_options=contact_options,
            growth_options=growth_options, opening_limit_m=opening_limit_m, output=output / "trials",
            selected_tensile_strength_pa=selected_tensile_strength_pa,
        )
        result.update({key: value for key, value in source_report.items() if key != "status"})
        result["accepted_event_directory"] = str(Path("trials") / result["accepted_event_directory"])
        (output / "growth_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        source_report["status"] = f"completed_experimental_source_{mode}_growth"
        source_report["accepted_event_directory"] = result["accepted_event_directory"]
        persist_source()
        return result
    except (ValueError, RuntimeError) as error:
        source_report["status"] = f"failed_experimental_source_{mode}_growth"
        source_report["error"] = str(error)
        persist_source()
        raise RuntimeError(f"Source {mode} growth failed; no arrested-growth result accepted: {error}") from error
    finally:
        if springs.matrix is not None:
            springs.matrix.destroy()
