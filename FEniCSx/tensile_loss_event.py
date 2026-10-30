"""Reduced intact-load -> selected tensile-loss equilibrium driver.

The same external load functional and boundary conditions are used in both
states. No density/load ramp, seed gap or residual tensile stiffness is added.
Source/bound adapters retain canonical import/provenance. The explicit-cut
contact entry point also supports steady saturated hydraulics; none of these
entry points implements marker-driven surfaces or irreversible fracture growth.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from dolfinx import fem, io
from dolfinx.fem.petsc import NonlinearProblem, assemble_vector
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

if __package__:
    from .no_tension_fem import NoTensionMaterial
    from .interface_springs import InterfaceSprings, SpringLinearProblem
else:
    from no_tension_fem import NoTensionMaterial
    from interface_springs import InterfaceSprings, SpringLinearProblem


@dataclass(frozen=True)
class EventSolverOptions:
    relative_tolerance: float = 1e-8
    absolute_tolerance_n: float = 1e-7
    maximum_iterations: int = 50

    def __post_init__(self):
        for name in ("relative_tolerance", "absolute_tolerance_n"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.relative_tolerance >= 1:
            raise ValueError("relative_tolerance must be less than one")
        if type(self.maximum_iterations) is not int or self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be a positive integer")


@dataclass
class TensileLossEvent:
    intact_displacement: fem.Function
    post_event_displacement: fem.Function
    material: NoTensionMaterial
    report: dict


def solve_bound_event(configuration: Path, model_directory: Path, baseline_directory: Path,
                      fields: Path, request: dict, *, maximum_volume_change,
                      options: EventSolverOptions, output: Path) -> TensileLossEvent:
    """Experimental verified-baseline mechanics only, never a full app scenario."""
    structure, binding = verified_source_selection(
        configuration, model_directory, baseline_directory, fields, request)
    return solve_source_event(
        structure, binding["affected_element_ids"], maximum_volume_change=maximum_volume_change,
        options=options, output=output, selection_provenance={
            **binding,
            "component_scope": "intact/post-event equilibrium only; not crack growth or hydraulic evolution",
        },
    )


def verified_source_selection(configuration: Path, model_directory: Path, baseline_directory: Path,
                              fields: Path, request: dict):
    """Rebuild the source and bind the selection to existing baseline evidence."""
    if __package__:
        from .run import submitted_structure
        from .three_body_model import require_verified_baseline
        from .damage_selection import bind_source_selection
    else:
        from run import submitted_structure
        from three_body_model import require_verified_baseline
        from damage_selection import bind_source_selection
    configuration, model_directory = Path(configuration), Path(model_directory)
    baseline_directory, fields = Path(baseline_directory), Path(fields)
    structure = submitted_structure(configuration, model_directory)
    baseline = require_verified_baseline(structure, baseline_directory)
    binding = bind_source_selection(
        model_directory / "three_body.msh", fields, configuration, structure, request,
    )
    manifest = json.loads((baseline_directory / "manifest.json").read_text())
    field_record = manifest.get("crack_seed_source")
    if (not isinstance(field_record, dict)
            or field_record.get("sha256") != binding["baseline_fields_sha256"]
            or binding["source_mesh_sha256"] != baseline["source_sha256"]["three_body.msh"]
            or binding["configuration_sha256"] != structure.submitted_configuration_sha256):
        raise ValueError("Selection source/fields do not match the verified baseline evidence")
    for name, digest in baseline["source_sha256"].items():
        if hashlib.sha256((model_directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Verified source mesh/configuration/diagnostic changed before event preparation")
    return structure, {**binding, "verified_baseline": baseline}


def solve_source_event(structure, affected_source_rows, *, maximum_volume_change,
                       options: EventSolverOptions, output: Path,
                       selection_provenance=None) -> TensileLossEvent:
    """Experimental source-import adapter, not an authenticated app scenario.

    Source row IDs refer to the supplied structure's volume cells. This entry
    point imports that structure itself, never reuses unrelated native tags.
    No selected-edge growth, crack water flow or saved-catalog binding is implied.
    """
    if __package__:
        from .dam_mesh import import_structure, source_cell_tags, selected_wall_cells
        from .dam_elasticity import external_load, water_load_parameters
        from .hydrostatic_load import assemble_water_forces
        from .three_body_model import is_three_body, materials, require_support_tags, support_bcs, support_form
    else:
        from dam_mesh import import_structure, source_cell_tags, selected_wall_cells
        from dam_elasticity import external_load, water_load_parameters
        from hydrostatic_load import assemble_water_forces
        from three_body_model import is_three_body, materials, require_support_tags, support_bcs, support_form
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("Source tensile-loss events currently require one MPI process")
    if not is_three_body(structure):
        raise ValueError("Source tensile-loss events require the canonical multi-body structure")
    if not isinstance(options, EventSolverOptions):
        raise ValueError("Explicit EventSolverOptions are required")
    if output is None:
        raise ValueError("Source events require a new persistent output directory")
    if (isinstance(maximum_volume_change, bool)
            or not isinstance(maximum_volume_change, (int, float))
            or not np.isfinite(maximum_volume_change) or not 0 <= maximum_volume_change < 1):
        raise ValueError("maximum_volume_change must be finite and in [0, 1)")
    imported = import_structure(structure, maximum_volume_change)
    source_tags = source_cell_tags(imported)
    selected = selected_wall_cells(source_tags, imported.cell_tags, affected_source_rows)
    space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
    coefficients = materials(imported, structure)
    require_support_tags(imported, structure)
    bcs = support_bcs(imported, space, structure)
    ds = ufl.Measure("ds", domain=imported.domain, subdomain_data=imported.facet_tags)
    water_forces = assemble_water_forces(imported, space, *water_load_parameters(imported, structure))
    springs = InterfaceSprings(imported, structure, space)
    try:
        return solve_event(
            space, young_pa=coefficients["youngs_modulus"], poisson=coefficients["poisson_ratio"],
            selected_cells=selected, boundary_conditions=bcs,
            external_load=lambda test: external_load(
                imported, structure, coefficients["density"], test, ds, include_water=False,
            )[0],
            linear_support=lambda trial, test: support_form(trial, test, ds, structure),
            interface_springs=springs, options=options, output=output,
            source_cell_rows=source_tags.values,
            additional_nodal_forces=water_forces,
            selection_provenance=selection_provenance,
        )
    finally:
        if springs.matrix is not None:
            springs.matrix.destroy()


def solve_event(space, *, young_pa, poisson, selected_cells, boundary_conditions,
                external_load, options: EventSolverOptions, output: Path | None = None,
                interface_springs: InterfaceSprings | None = None, linear_support=None,
                source_cell_rows=None, additional_nodal_forces=None,
                selection_provenance=None) -> TensileLossEvent:
    """Solve both states using an external_load(test_function) UFL functional.

    Serial affine P1 tetrahedra, cellwise isotropic material, fixed load/BC
    snapshot, direct MUMPS. A failed solve is not an engineering instability
    verdict; no residual-stiffness retry or equilibrium-shaped fallback exists.
    If requested, output must be a new directory and retains failure diagnostics.
    linear_support(trial, test) supplies the unchanged UFL support bilinear form;
    its boundary measure must share the load's facet tags. Existing mortar
    springs must use this space. Serial spring/Dirichlet lifting and elimination
    preserve imposed displacements and raw spring forces for reaction checks.
    additional_nodal_forces is a fixed serial load in native scalar DOF order;
    constrained entries contribute to reactions, not nonlinear free residuals.
    """
    if not isinstance(options, EventSolverOptions):
        raise ValueError("Explicit EventSolverOptions are required")
    domain = space.mesh
    if domain.comm.size != 1:
        raise ValueError("Reduced tensile-loss event currently requires one MPI process")
    displacement = fem.Function(space, name="post_event_displacement_m")
    material = NoTensionMaterial(displacement, young_pa, poisson, selected_cells)
    if not np.any(material.selected):
        raise ValueError("At least one affected local cell is required")
    selection = material.selected.copy()
    provenance = {}
    if source_cell_rows is not None:
        rows = np.asarray(source_cell_rows)
        if rows.shape != selection.shape or rows.dtype.kind not in "iu" or np.any(rows < 0):
            raise ValueError("Source cell rows must be nonnegative integer IDs in local cell order")
        provenance["source_cell_rows"] = rows.copy()
    bcs = list(boundary_conditions)
    fixed_dofs = np.unique(np.concatenate([bc.dof_indices()[0] for bc in bcs])) if bcs else np.empty(0, dtype=int)
    extra_forces = np.zeros_like(displacement.x.array)
    if additional_nodal_forces is not None:
        forces = np.asarray(additional_nodal_forces)
        if (forces.shape != extra_forces.shape or forces.dtype.kind not in "fiu"
                or not np.isfinite(forces).all()):
            raise ValueError("Additional nodal forces must be finite real values matching serial displacement DOFs")
        extra_forces[:] = forces
    free_extra_forces = extra_forces.copy()
    free_extra_forces[fixed_dofs] = 0.
    test, trial = ufl.TestFunction(space), ufl.TrialFunction(space)
    load = external_load(test)
    if interface_springs is not None:
        if not isinstance(interface_springs, InterfaceSprings) or interface_springs.space is not space:
            raise ValueError("Interface springs must be the existing mortar operator on the event displacement space")
    strain = ufl.sym(ufl.grad(trial))
    elastic_stress = 2 * material.mu * strain + material.lam * ufl.tr(strain) * ufl.Identity(3)
    bilinear = ufl.inner(elastic_stress, ufl.sym(ufl.grad(test))) * ufl.dx
    support = linear_support(trial, test) if linear_support is not None else None
    support_residual = ufl.action(support, displacement) if support is not None else None
    if support is not None:
        bilinear += support
    intact = fem.Function(space, name="intact_displacement_m")
    material.selected[:] = False
    report = {
        "status": "running_reduced_tensile_loss_event", "solver_ready": False,
        "engineering_validated": False, "selected_cells": int(selection.sum()),
        "initial_gap_m": 0., "unchanged_external_loads": True,
        "additional_nodal_forces_included": additional_nodal_forces is not None,
        "material_groups": [
            {"young_pa": young, "poisson": ratio, "local_cells": int(members.sum())}
            for members, young, ratio in material.material_groups
        ],
        "state_output_axis": "0=intact_loaded, 1=post_event; not hydraulic physical time",
        "selection_provenance": selection_provenance,
        "relative_tolerance": options.relative_tolerance,
        "absolute_tolerance_n": options.absolute_tolerance_n,
        "maximum_iterations": options.maximum_iterations,
        "interface_springs": interface_springs.report if interface_springs is not None else None,
        "interface_dirichlet_elimination": "serial rows/columns with raw-force lifting" if interface_springs is not None else None,
        "linear_support_included": support is not None,
        "limitations": "Cellwise material equilibrium with optional existing mortar/linear supports; no surface growth, friction, hydraulic flow or full dam adapter.",
        "stages": [],
    }
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)

    def save_report():
        if output is not None:
            (output / "event_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save_report()
    load_vector = assemble_vector(fem.form(load))
    try:
        load_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        load_vector.array[:] += extra_forces
        force_scale = max(float(load_vector.norm()), options.absolute_tolerance_n)
        applied_force = load_vector.array.reshape(-1, 3).sum(axis=0).copy()
        if not np.isfinite(force_scale) or not np.isfinite(applied_force).all():
            raise ValueError("Applied loads must assemble to finite nodal forces")
        if output is not None:
            np.savez_compressed(
                output / "native_inputs.npz", coordinates_m=domain.geometry.x,
                geometry_cells=domain.geometry.dofmaps[0],
                geometry_input_global_indices=domain.geometry.input_global_indices,
                original_cell_index=domain.topology.original_cell_index,
                selected_local_cells=selection, assembled_external_force_n=load_vector.array,
                additional_nodal_force_n=extra_forces,
                young_pa=material.young, poisson=material.poisson, **provenance,
            )
    finally:
        load_vector.destroy()

    def check_state(stage, reason, iterations):
        material.update()
        internal = ufl.inner(material.stress, ufl.sym(ufl.grad(test))) * ufl.dx - load
        if support_residual is not None:
            internal += support_residual
        residual = assemble_vector(fem.form(internal))
        try:
            residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            residual.array[:] -= extra_forces
            if interface_springs is not None:
                work = interface_springs.matrix.createVecLeft()
                try:
                    interface_springs.matrix.mult(displacement.x.petsc_vec, work)
                    residual.axpy(1., work)
                finally:
                    work.destroy()
            fixed = np.unique(np.concatenate([bc.dof_indices()[0] for bc in bcs])) if bcs else np.empty(0, dtype=int)
            free = np.setdiff1d(np.arange(len(residual.array)), fixed)
            error = float(np.linalg.norm(residual.array[free]))
            reactions = residual.array.copy()
            reactions[free] = 0.
            balance = reactions.reshape(-1, 3).sum(axis=0)
            grounded_reaction = np.zeros(3)
            support_energy = 0.
            if support_residual is not None:
                support_vector = assemble_vector(fem.form(support_residual))
                try:
                    grounded_reaction = -support_vector.array.reshape(-1, 3).sum(axis=0)
                    support_energy = float(.5 * support_vector.dot(displacement.x.petsc_vec))
                finally:
                    support_vector.destroy()
                balance += grounded_reaction
            force_imbalance = float(np.linalg.norm(balance + applied_force))
            stress = material.stress.x.array.reshape(-1, 3, 3)[material.tensor_dofs]
            row = {
                "stage": stage, "solver_reason": int(reason), "solver_iterations": int(iterations),
                "free_residual_norm_n": error, "scaled_free_residual": error / force_scale,
                "support_reaction_n": balance.tolist(),
                "linear_support_reaction_n": grounded_reaction.tolist(),
                "linear_support_energy_j": support_energy,
                "applied_force_n": applied_force.tolist(),
                "scaled_force_imbalance": force_imbalance / force_scale,
                "maximum_displacement_m": float(np.max(np.linalg.norm(displacement.x.array.reshape(-1, 3), axis=1))),
                "maximum_selected_principal_stress_pa": float(np.linalg.eigvalsh(stress[selection])[:, -1].max()),
                "elastic_energy_j": float(fem.assemble_scalar(fem.form(material.energy * ufl.dx))),
                "interface_spring_energy_j": interface_springs.energy(displacement) if interface_springs is not None else 0.,
            }
        finally:
            residual.destroy()
        report["stages"].append(row)
        save_report()
        threshold = options.absolute_tolerance_n + options.relative_tolerance * force_scale
        if (reason <= 0 or not np.isfinite(error) or error > threshold
                or not np.isfinite(force_imbalance) or force_imbalance > threshold * max(1., np.sqrt(len(free)))
                or not np.isfinite(displacement.x.array).all()):
            raise RuntimeError(f"{stage} equilibrium failed independent residual check: {row}")
        return row

    try:
        baseline = SpringLinearProblem(
            bilinear, load, u=displacement, bcs=bcs, interface_springs=interface_springs,
            additional_nodal_forces=extra_forces,
            eliminate_interface_bcs=True,
            petsc_options_prefix="tensile_loss_intact_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                           "ksp_error_if_not_converged": True},
        )
        if interface_springs is not None:
            offsets, columns, values = interface_springs.matrix.getValuesCSR()
            if output is not None:
                np.savez_compressed(
                    output / "interface_springs.npz", indptr=offsets, indices=columns, data=values,
                    shape=interface_springs.matrix.getSize(),
                    displacement_cell_dofs=space.dofmap.list,
                )
        baseline.solve()
        check_state("intact_loaded", baseline.solver.getConvergedReason(), baseline.solver.getIterationNumber())
        intact.x.array[:] = displacement.x.array
        intact.x.scatter_forward()
        if output is not None:
            np.savez_compressed(output / "intact_state.npz", displacement_m=intact.x.array,
                                stress_pa=material.stress.x.array,
                                energy_j_m3=material.energy.x.array)
        material.selected[:] = selection
        material.update()
        post_residual = ufl.inner(material.stress, ufl.sym(ufl.grad(test))) * ufl.dx - load
        post_tangent = ufl.inner(material.tangent_stress(trial), ufl.sym(ufl.grad(test))) * ufl.dx
        if support is not None:
            post_residual += support_residual
            post_tangent += support
        problem = NonlinearProblem(
            post_residual, displacement, J=post_tangent,
            bcs=bcs, petsc_options_prefix="tensile_loss_event_",
            petsc_options={"snes_type": "newtonls", "snes_rtol": options.relative_tolerance * .1,
                           "snes_atol": options.absolute_tolerance_n * .1, "snes_stol": 0.,
                           "snes_max_it": options.maximum_iterations, "snes_error_if_not_converged": False,
                           "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                           "ksp_error_if_not_converged": False},
        )
        if interface_springs is not None:
            problem.A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        def update(x):
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            fem.petsc.assign(x, displacement)
            material.update()

        def residual(snes, x, vector):
            update(x)
            fem.petsc.assemble_residual(snes, x, vector, displacement, problem.F, problem.J, bcs)
            vector.array[:] -= free_extra_forces
            if interface_springs is not None:
                work = interface_springs.matrix.createVecLeft()
                try:
                    interface_springs.matrix.mult(x, work)
                    work.array[fixed_dofs] = 0.
                    vector.axpy(1., work)
                finally:
                    work.destroy()

        def jacobian(snes, x, matrix, preconditioner):
            update(x)
            fem.petsc.assemble_jacobian(snes, x, matrix, preconditioner, displacement,
                                       problem.J, problem.preconditioner, bcs)
            if interface_springs is not None:
                interface_springs.add_to(matrix, operator=baseline.spring_jacobian)
                if preconditioner is not None and preconditioner.handle != matrix.handle:
                    interface_springs.add_to(preconditioner, operator=baseline.spring_jacobian)

        problem.solver.setFunction(residual, problem.b)
        problem.solver.setJacobian(jacobian, problem.A, problem.P_mat)
        trace = []

        def monitor(snes, iteration, norm):
            if not np.isfinite(norm):
                raise RuntimeError(f"Post-event nonlinear residual is non-finite at iteration {iteration}")
            trace.append({"iteration": int(iteration), "residual_norm_n": float(norm)})

        problem.solver.setMonitor(monitor)
        report["post_event_iterations"] = trace
        problem.solve()
        check_state("post_event", problem.solver.getConvergedReason(), problem.solver.getIterationNumber())
        report["status"] = "passed_reduced_tensile_loss_event"
        if output is not None:
            np.savez_compressed(output / "post_event_state.npz", displacement_m=displacement.x.array,
                                stress_pa=material.stress.x.array,
                                relaxed_strain_not_opening=material.relaxed_strain.x.array,
                                energy_j_m3=material.energy.x.array)
            with io.VTKFile(domain.comm, output / "displacement.pvd", "w") as writer:
                writer.write_function(intact, 0.)
                writer.write_function(displacement, 1.)
            with io.VTKFile(domain.comm, output / "post_event_material.pvd", "w") as writer:
                writer.write_function([material.stress, material.relaxed_strain, material.energy], 1.)
        save_report()
        return TensileLossEvent(intact, displacement, material, report)
    except (RuntimeError, ValueError, PETSc.Error) as error:
        report["status"] = "failed_reduced_tensile_loss_event"
        report["error"] = str(error)
        save_report()
        raise RuntimeError(f"Tensile-loss event failed; no equilibrium or opening verdict accepted: {error}") from error


class UnacceptedHydrostaticBound(RuntimeError):
    """Optimistic pressure energy bound, never an accepted wet equilibrium."""

    def __init__(self, report):
        super().__init__(report["hydraulic_access_error"])
        self.report = report


def solve_contact_event(
    space, faces, *, young_pa, poisson, selected_cells, boundary_conditions,
    external_load, options: EventSolverOptions, contact_options, output: Path,
    interface_springs=None, linear_support=None, additional_nodal_forces=None,
    face_pressures_pa=None, hydraulics=None, opening_limit_m=0.002, release_cut=True,
    released_face_mask=None, selected_tensile_strength_pa=0., equilibrium_seed=None,
) -> TensileLossEvent:
    """Loaded bonded state -> selected tensile loss plus explicit cut release.

    The caller must supply a justified cut: this function does not infer a
    fracture surface from painted cells/edges or choose an irreversible growth
    law. All lips are bonded in all components for the intact solve. Only the
    selected cells lose tensile capacity after release; other cells retain the
    original elastic law. Contact is frictionless. New-face water is explicitly
    prescribed post-event traction unless explicit steady hydraulics are
    supplied. Steady through-flow requires a dry-equilibrium-open surface
    (optional exact tied front). The explicit common-head saturated equilibrium
    specialization permits pressure opening, then audits final aperture/access.
    Steady pressure is recomputed from each candidate's own aperture.
    Closure/isolated water cannot acquire an invented residual aperture.
    Explicit hydraulic face IDs can select released faces from a partially
    bonded candidate inventory. Omitted faces are declared dry, not drained by
    a wetting or trapped-fluid algorithm. Hydraulic front equivalence follows
    the same all-component bond constraints as the mechanics.
    """
    from dolfinx.fem.petsc import assemble_matrix, set_bc
    from scipy import sparse

    if __package__:
        from .fracture_faces import bulk_face_mapping, source_vector_values
        from .paired_contact import ContactOptions, solve_contact
        from .fracture_steady_flow import SteadyCrackHydraulics
        from .horizontal_fracture_contract import EquilibriumOpeningObservation, assess_equilibrium_opening
        from .reservoir_crack_load import ReservoirAccessError
    else:
        from fracture_faces import bulk_face_mapping, source_vector_values
        from paired_contact import ContactOptions, solve_contact
        from fracture_steady_flow import SteadyCrackHydraulics
        from horizontal_fracture_contract import EquilibriumOpeningObservation, assess_equilibrium_opening
        from reservoir_crack_load import ReservoirAccessError
    if not isinstance(options, EventSolverOptions) or not isinstance(contact_options, ContactOptions):
        raise ValueError("Explicit event and contact solver options are required")
    if type(release_cut) is not bool:
        raise ValueError("release_cut must be an explicit boolean")
    if not release_cut and (hydraulics is not None or face_pressures_pa is not None):
        raise ValueError("Bonded post-loss surfaces cannot receive crack-water loads")
    if hydraulics is not None:
        if not isinstance(hydraulics, SteadyCrackHydraulics):
            raise ValueError("Explicit SteadyCrackHydraulics are required for coupled flow")
        if face_pressures_pa is not None:
            raise ValueError("Choose transported steady pressure or prescribed pressure, not both")
    coordinates, cells, mapping = bulk_face_mapping(space, faces)
    if released_face_mask is None:
        released = np.full(len(faces.minus), release_cut, dtype=bool)
    else:
        mask = np.asarray(released_face_mask)
        if mask.dtype != np.dtype(bool) or mask.shape != (len(faces.minus),) or not release_cut:
            raise ValueError("Released face mask must match inventory with release_cut enabled")
        if face_pressures_pa is not None:
            raise ValueError("Partial candidate release requires explicit hydraulic face identities, not prescribed pressure")
        released = mask.copy()
    hydraulic_ids = hydraulics.face_ids(faces) if hydraulics is not None else None
    if hydraulic_ids is not None and not np.all(released[hydraulic_ids]):
        raise ValueError("Hydraulic inventory may load only released faces; unbroken candidates must remain bonded and dry")
    has_release = bool(released.any())
    opening_history = [EquilibriumOpeningObservation(
        "intact_loaded_bonded", 0., tuple(map(float, coordinates[faces.minus[0, 0]])), False)]
    assess_equilibrium_opening(opening_limit_m, tuple(opening_history), completed=False)
    if interface_springs is not None and interface_springs.space is not space:
        raise ValueError("Interface springs must belong to the supplied event displacement space")
    displacement = fem.Function(space, name="post_event_displacement_m")
    material = NoTensionMaterial(
        displacement, young_pa, poisson, selected_cells, selected_tensile_strength_pa,
    )
    selection = material.selected.copy()
    if not np.any(selection):
        raise ValueError("At least one affected element is required")
    material.selected[:] = False
    bcs = list(boundary_conditions)
    fixed = np.unique(np.concatenate([bc.dof_indices()[0] for bc in bcs])) if bcs else np.empty(0, dtype=int)
    free = np.setdiff1d(np.arange(len(displacement.x.array)), fixed)
    extra = np.zeros_like(displacement.x.array)
    if additional_nodal_forces is not None:
        raw = np.asarray(additional_nodal_forces)
        if raw.shape != extra.shape or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
            raise ValueError("Additional nodal loads must match serial displacement DOFs")
        extra[:] = raw
    pressure = np.zeros(faces.minus.shape) if face_pressures_pa is None else np.asarray(face_pressures_pa)
    water_source = faces.pressure_forces(pressure)
    water = np.empty_like(water_source)
    water[mapping] = water_source
    test, trial = ufl.TestFunction(space), ufl.TrialFunction(space)
    load_form = external_load(test)
    support = linear_support(trial, test) if linear_support is not None else None
    internal_form = ufl.inner(material.stress, ufl.sym(ufl.grad(test))) * ufl.dx
    tangent_form = ufl.inner(material.tangent_stress(trial), ufl.sym(ufl.grad(test))) * ufl.dx
    if support is not None:
        internal_form += ufl.action(support, displacement)
        tangent_form += support
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running_reduced_tensile_contact_event", "solver_ready": False,
        "engineering_validated": False, "initial_gap_m": 0.,
        "cut_is_prescribed": True, "fracture_growth_solved": False,
        "cut_released": has_release,
        "released_face_ids": np.flatnonzero(released).tolist(),
        "hydraulic_inventory_face_ids": hydraulic_ids.tolist() if hydraulic_ids is not None else [],
        "explicitly_dry_released_face_ids": (
            np.flatnonzero(released & ~np.isin(np.arange(len(released)), hydraulic_ids)).tolist()
            if hydraulic_ids is not None else np.flatnonzero(released).tolist()),
        "fluid_history_solved": False,
        "hydraulic_requires_dry_opening": hydraulics.requires_dry_opening if hydraulics is not None else None,
        "physical_time_solved": False,
        "contact_options": asdict(contact_options),
        "selected_cells": int(selection.sum()), "iterations": [],
        "selected_tensile_strength_pa": material.tensile_strength_pa,
        "selected_material_law": "reversible tensile cap; not irreversible strength-calibrated nucleation",
        "relative_tolerance": options.relative_tolerance,
        "absolute_tolerance_n": options.absolute_tolerance_n,
        "maximum_iterations": options.maximum_iterations,
        "opening_limit_m": opening_limit_m,
        "state_output_axis": "0=intact_loaded_bonded, 1=post_event; not physical seconds",
        "hydraulic_mode": (
            "drained reservoir equilibrium with final opening/access audit and exact waterline clipping" if not hydraulics.requires_dry_opening else
            "explicit saturated released-face subset steady flow" if len(hydraulic_ids) != len(faces.minus) else
            "fully saturated steady flow with exact tied front" if hydraulics.allow_tied_fronts
            else "fully saturated open-cut steady flow"
        ) if hydraulics is not None else "prescribed pressure",
        "limitations": "Explicit trial cut, not an unrestricted growth law. Frictionless contact; fixed-head pressure opening audits final submerged access. Exact reference-waterline clipping is not transient filling, trapped fluid, moving free surfaces or a different-head growth law.",
    }

    def persist():
        (output / "event_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def stiffness():
        operator = assemble_matrix(fem.form(tangent_form))
        operator.assemble()
        try:
            if interface_springs is not None:
                operator.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
                if interface_springs.matrix is None:
                    interface_springs.build_matrix(operator)
                interface_springs.add_to(operator)
            offsets, columns, coefficients = operator.getValuesCSR()
            return sparse.csr_matrix((coefficients.copy(), columns.copy(), offsets.copy()), shape=operator.getSize())
        finally:
            operator.destroy()

    def internal():
        vector = assemble_vector(fem.form(internal_form))
        try:
            values = vector.array.copy()
        finally:
            vector.destroy()
        if interface_springs is not None:
            work = interface_springs.matrix.createVecLeft()
            try:
                interface_springs.matrix.mult(displacement.x.petsc_vec, work)
                values += work.array
            finally:
                work.destroy()
        return values

    persist()
    try:
        load = assemble_vector(fem.form(load_form))
        try:
            exterior = load.array.copy() + extra
            prescribed = load.duplicate()
            try:
                prescribed.set(0.)
                set_bc(prescribed, bcs)
                imposed = prescribed.array[fixed].copy()
            finally:
                prescribed.destroy()
        finally:
            load.destroy()
        if not np.isfinite(exterior).all():
            raise ValueError("External loads must be finite")
        np.savez_compressed(
            output / "native_inputs.npz", coordinates_m=coordinates, cells=cells,
            minus=faces.minus, plus=faces.plus, normals=faces.normals,
            source_to_dof=mapping, selected_local_cells=selection,
            released_face_mask=released,
            young_pa=material.young, poisson=material.poisson,
            selected_tensile_strength_pa=material.tensile_strength_pa,
            fixed_dofs=fixed, prescribed_m=imposed, assembled_exterior_force_n=exterior,
            pressure_pa=pressure, new_face_water_forces_n=water_source,
        )
        if hydraulics is not None:
            np.savez_compressed(
                output / "hydraulic_inputs.npz", parent_node_ids=hydraulics.parent_node_ids,
                saturated_face_mask=hydraulics.saturated_face_mask,
                mechanical_inventory_face_ids=hydraulic_ids,
            )
            report["hydraulic_boundaries"] = [
                {"name": boundary.name, "elevation_m": float(boundary.elevation_m),
                 "inlet_source_edges": boundary.inlet_source_edges.tolist()}
                for boundary in hydraulics.boundaries
            ]
            report["hydraulic_parameters"] = {
                name: float(getattr(hydraulics, name)) for name in (
                    "water_density_kg_m3", "gravity_m_s2", "closure_tolerance_m", "relative_tolerance")
            }
            report["hydraulic_parameters"]["allow_tied_fronts"] = hydraulics.allow_tied_fronts
        material.update()
        intact_operator = stiffness()
        identity = hashlib.sha256()
        for array in (coordinates, cells, mapping, faces.minus, faces.plus, faces.normals,
                      material.young, material.poisson, selection,
                      np.asarray(material.tensile_strength_pa),
                      fixed, imposed, exterior, intact_operator.indptr, intact_operator.indices,
                      intact_operator.data):
            contiguous = np.ascontiguousarray(array)
            identity.update(json.dumps((contiguous.dtype.str, contiguous.shape)).encode("ascii"))
            identity.update(contiguous.tobytes())
        report["dry_problem_sha256"] = identity.hexdigest()
        if equilibrium_seed is None:
            baseline = solve_contact(intact_operator, exterior, fixed, imposed, faces, mapping,
                                     options=contact_options, bonded=True)
            intact_values = baseline.displacement_m
            intact_report = baseline.report
        else:
            if (not isinstance(equilibrium_seed, TensileLossEvent)
                    or equilibrium_seed.report.get("status") != "passed_reduced_tensile_contact_event"
                    or equilibrium_seed.report.get("dry_problem_sha256") != report["dry_problem_sha256"]
                    or equilibrium_seed.intact_displacement.function_space is not space
                    or equilibrium_seed.post_event_displacement.function_space is not space
                    or not np.isfinite(equilibrium_seed.intact_displacement.x.array).all()
                    or not np.isfinite(equilibrium_seed.post_event_displacement.x.array).all()
                    or not set(equilibrium_seed.report["released_face_ids"]).issubset(set(np.flatnonzero(released)))):
                raise ValueError("Equilibrium seed must be accepted for this exact mechanical problem and a subset of this cut")
            intact_values = equilibrium_seed.intact_displacement.x.array
            intact_report = equilibrium_seed.report["intact_bonded"]
        report["reused_bound_intact_equilibrium"] = equilibrium_seed is not None
        report["workflow"] = [
            "intact bonded equilibrium under unchanged exterior gravity and water loads",
            "selected-cell tensile strength reduction under the same exterior loads",
            "trial cut equilibrium including reservoir-connected crack-face pressure and contact",
        ]
        if interface_springs is not None:
            offsets, columns, coefficients = interface_springs.matrix.getValuesCSR()
            np.savez_compressed(
                output / "interface_springs.npz", indptr=offsets, indices=columns, data=coefficients,
                displacement_cell_dofs=space.dofmap.list,
                **({"multiplier_row_map": interface_springs.multiplier_map}
                   if interface_springs.multiplier_map is not None else {}),
            )
            report["interface_springs"] = interface_springs.report
        displacement.x.array[:] = intact_values
        displacement.x.scatter_forward()
        material.update()
        intact = fem.Function(space, name="intact_displacement_m")
        intact.x.array[:] = displacement.x.array
        _, intact_source = source_vector_values(intact)
        report["intact_bonded"] = intact_report
        np.savez_compressed(output / "intact_state.npz", displacement_m=intact_source,
                            stress_pa=material.stress.x.array, energy_j_m3=material.energy.x.array)
        intact_force_norm = float(np.linalg.norm(internal()))
        material.selected[:] = selection
        if equilibrium_seed is not None:
            displacement.x.array[:] = equilibrium_seed.post_event_displacement.x.array
            displacement.x.scatter_forward()
            report["post_event_initial_guess"] = "bound accepted equilibrium; independently re-equilibrated and audited"
        total = exterior + water.ravel()
        force_scale = max(float(np.linalg.norm(total)), intact_force_norm, options.absolute_tolerance_n)
        awaiting_dry = hydraulics is not None and hydraulics.requires_dry_opening
        hydraulic_state = None
        if hydraulics is not None and not awaiting_dry:
            hydraulic_state = hydraulics.trial_load(
                faces, source_vector_values(displacement)[1], bonded_face_mask=~released)
            water[mapping] = hydraulic_state.nodal_forces_n
            total = exterior + water.ravel()
            force_scale = max(float(np.linalg.norm(total)), intact_force_norm, options.absolute_tolerance_n)
        for iteration in range(1, options.maximum_iterations + 1):
            material.update()
            tangent = stiffness()
            old = displacement.x.array.copy()
            # Absolute-displacement constrained Newton step. Contact is solved
            # on the candidate, not an unconstrained increment that may overlap.
            effective = total - internal() + tangent @ old
            candidate = solve_contact(tangent, effective, fixed, imposed, faces, mapping,
                                      options=contact_options, bonded_face_mask=~released)
            displacement.x.array[:] = candidate.displacement_m
            displacement.x.scatter_forward()
            material.update()
            if hydraulics is not None and not awaiting_dry:
                hydraulic_state = hydraulics.trial_load(
                    faces, source_vector_values(displacement)[1], bonded_face_mask=~released)
                water[mapping] = hydraulic_state.nodal_forces_n
                total = exterior + water.ravel()
                force_scale = max(float(np.linalg.norm(total)), intact_force_norm, options.absolute_tolerance_n)
            residual = internal() - total - candidate.contact_nodal_forces_n
            error = float(np.linalg.norm(residual[free]))
            principal = np.linalg.eigvalsh(
                material.stress.x.array.reshape(-1, 3, 3)[material.tensor_dofs][selection])[:, -1]
            row = {
                "iteration": iteration, "free_residual_norm_n": error,
                "stage": "dry_post_event" if awaiting_dry else "post_event",
                "scaled_free_residual": error / force_scale,
                "maximum_selected_principal_stress_pa": float(principal.max()),
                "contact": candidate.report,
            }
            report["iterations"].append(row)
            persist()
            # One flushed line per equilibrium iteration for the app's live subtask view.
            print("Iteration: " + json.dumps({
                "iteration": iteration, "stage": row["stage"], "scaled_residual": row["scaled_free_residual"],
                "target": options.relative_tolerance,
                "time": datetime.now(timezone.utc).isoformat()}), flush=True)
            if not np.isfinite(error):
                raise RuntimeError("Non-finite post-event residual")
            if error <= options.absolute_tolerance_n + options.relative_tolerance * force_scale:
                if awaiting_dry:
                    dry_opening = faces.measure(source_vector_values(displacement)[1])
                    opening_history.append(EquilibriumOpeningObservation(
                        "dry_post_event", dry_opening["maximum_opening_m"],
                        tuple(dry_opening["peak_location_m"]), True))
                    np.savez_compressed(output / "dry_post_event_state.npz",
                                        displacement_m=source_vector_values(displacement)[1],
                                        stress_pa=material.stress.x.array,
                                        contact_multipliers_n=candidate.multipliers_n)
                    hydraulic_state = hydraulics.load(
                        faces, source_vector_values(displacement)[1], bonded_face_mask=~released)
                    water[mapping] = hydraulic_state.nodal_forces_n
                    total = exterior + water.ravel()
                    awaiting_dry = False
                    continue
                reactions = np.zeros_like(residual)
                reactions[fixed] = residual[fixed]
                grounded = np.zeros(3)
                if support is not None:
                    support_vector = assemble_vector(fem.form(ufl.action(support, displacement)))
                    try:
                        grounded = -support_vector.array.reshape(-1, 3).sum(axis=0)
                    finally:
                        support_vector.destroy()
                balance = (reactions.reshape(-1, 3).sum(axis=0) + grounded
                           + total.reshape(-1, 3).sum(axis=0))
                imbalance = float(np.linalg.norm(balance))
                row["scaled_force_imbalance"] = imbalance / force_scale
                if (not np.isfinite(imbalance) or imbalance > (
                        options.absolute_tolerance_n + options.relative_tolerance * force_scale
                ) * max(1., np.sqrt(len(free)))):
                    raise RuntimeError("Post-event material/contact global force audit failed")
                report["opening"] = faces.measure(source_vector_values(displacement)[1])
                opening_history.append(EquilibriumOpeningObservation(
                    "post_event", report["opening"]["maximum_opening_m"] if has_release else 0.,
                    tuple(report["opening"]["peak_location_m"]), has_release))
                report["opening_assessment"] = assess_equilibrium_opening(
                    opening_limit_m, tuple(opening_history), completed=True)
                report["support_reaction_n"] = reactions.reshape(-1, 3).sum(axis=0).tolist()
                report["linear_support_reaction_n"] = grounded.tolist()
                stored_energy = float(fem.assemble_scalar(fem.form(material.energy * ufl.dx)))
                if support is not None:
                    stored_energy += .5 * float(fem.assemble_scalar(fem.form(
                        ufl.action(ufl.action(support, displacement), displacement))))
                if interface_springs is not None:
                    report["interface_spring_energy_j"] = interface_springs.energy(displacement)
                    stored_energy += report["interface_spring_energy_j"]
                report["stored_mechanical_energy_j"] = stored_energy
                report["exterior_dead_load_work_j"] = float(exterior @ displacement.x.array)
                report["dry_mechanical_potential_j"] = (
                    stored_energy - report["exterior_dead_load_work_j"]
                    if hydraulics is None and np.all(pressure == 0.) else None
                )
                report["hydrostatic_reservoir_work_j"] = None
                report["conservative_hydrostatic_potential_j"] = None
                if hydraulics is not None and len({b.elevation_m for b in hydraulics.boundaries}) == 1:
                    head = hydraulics.boundaries[0].elevation_m
                    expected_pressure = hydraulics.water_density_kg_m3 * hydraulics.gravity_m_s2 * (
                        np.maximum(head - faces.coordinates[faces.minus[hydraulic_ids], 2], 0.))
                    if not np.allclose(hydraulic_state.vertex_pressures_pa, expected_pressure,
                                       rtol=hydraulics.relative_tolerance, atol=0.):
                        raise RuntimeError("Common-head hydraulic state failed independent hydrostatic pressure audit")
                    audited_water = hydraulics.trial_load(
                        faces, source_vector_values(displacement)[1], bonded_face_mask=~released)
                    if not np.allclose(hydraulic_state.nodal_forces_n, audited_water.nodal_forces_n,
                                       rtol=hydraulics.relative_tolerance, atol=options.absolute_tolerance_n):
                        raise RuntimeError("Reservoir traction failed independent clipped-load audit")
                    reservoir_work = float(water.ravel() @ displacement.x.array)
                    report["hydrostatic_reservoir_work_j"] = reservoir_work
                    report["conservative_hydrostatic_potential_j"] = (
                        stored_energy - report["exterior_dead_load_work_j"] - reservoir_work)
                    report["hydrostatic_potential_scope"] = (
                        "Fixed common reservoir head, reference geometry, small displacement, "
                        "explicit boundary-connected saturation; includes reservoir work -integral(p*w)dA. "
                        "Not a different-head through-flow or trapped-fluid growth potential.")
                if hydraulics is not None:
                    try:
                        accepted_hydraulics = hydraulics.load(
                            faces, source_vector_values(displacement)[1], bonded_face_mask=~released)
                    except ReservoirAccessError as error:
                        if report["conservative_hydrostatic_potential_j"] is None:
                            raise
                        report["optimistic_pressure_potential_lower_bound_j"] = report.pop(
                            "conservative_hydrostatic_potential_j")
                        report["hydraulic_access_error"] = str(error)
                        report["failure_kind"] = "unaccepted_hydraulic_access_energy_bound"
                        np.savez_compressed(
                            output / "unaccepted_pressure_bound_state.npz",
                            displacement_m=source_vector_values(displacement)[1],
                            signed_gaps_m=candidate.signed_gaps_m,
                            contact_multipliers_n=candidate.multipliers_n)
                        raise UnacceptedHydrostaticBound(report) from error
                    if not np.allclose(accepted_hydraulics.nodal_forces_n, hydraulic_state.nodal_forces_n,
                                       rtol=hydraulics.relative_tolerance, atol=options.absolute_tolerance_n):
                        raise RuntimeError("Accepted hydraulic pressure differs from equilibrium trial pressure")
                    hydraulic_state = accepted_hydraulics
                    report["hydraulics"] = hydraulic_state.report
                    np.savez_compressed(
                        output / "hydraulic_state.npz", head_m=hydraulic_state.vertex_heads_m,
                        pressure_pa=hydraulic_state.vertex_pressures_pa,
                        nodal_water_forces_n=hydraulic_state.nodal_forces_n,
                        mechanical_inventory_face_ids=hydraulic_ids,
                    )
                report["status"] = "passed_reduced_tensile_contact_event"
                np.savez_compressed(
                    output / "post_event_state.npz", displacement_m=source_vector_values(displacement)[1],
                    stress_pa=material.stress.x.array, relaxed_strain_not_opening=material.relaxed_strain.x.array,
                    energy_j_m3=material.energy.x.array,
                    nodal_contact_forces_n=candidate.contact_nodal_forces_n.reshape(-1, 3)[mapping],
                    support_reaction_n=reactions.reshape(-1, 3)[mapping],
                    signed_gaps_m=candidate.signed_gaps_m, contact_multipliers_n=candidate.multipliers_n,
                )
                with io.VTKFile(space.mesh.comm, output / "displacement.pvd", "w") as writer:
                    writer.write_function(intact, 0.)
                    writer.write_function(displacement, 1.)
                with io.VTKFile(space.mesh.comm, output / "post_event_material.pvd", "w") as writer:
                    writer.write_function([material.stress, material.relaxed_strain, material.energy], 1.)
                persist()
                return TensileLossEvent(intact, displacement, material, report)
        raise RuntimeError("Post-event material/contact iteration limit reached")
    except (ValueError, RuntimeError, PETSc.Error) as error:
        report["status"] = "failed_reduced_tensile_contact_event"
        report["error"] = str(error)
        report["opening_assessment"] = assess_equilibrium_opening(
            opening_limit_m, tuple(opening_history), completed=False)
        persist()
        if isinstance(error, UnacceptedHydrostaticBound):
            raise
        raise RuntimeError(f"Tensile-contact event failed; no post-event equilibrium accepted: {error}") from error
