"""Reduced serial 3-D elasticity/steady-flow coupling on an explicit open cut.

Dry equilibrium must already open every hydraulic vertex. This does not
initiate cracks, impose an opening gap, enforce contact or propagate damage.
"""
from dataclasses import dataclass
import json
from pathlib import Path

from dolfinx import fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import numpy as np
from petsc4py import PETSc
import ufl

if __package__:
    from .fracture_faces import PairedCrackFaces, source_vector_values, bulk_face_mapping
    from .fracture_steady_flow import SteadyCouplingOptions, SteadyCrackLoad, steady_crack_load
    from .interface_springs import SpringLinearProblem
else:
    from fracture_faces import PairedCrackFaces, source_vector_values, bulk_face_mapping
    from fracture_steady_flow import SteadyCouplingOptions, SteadyCrackLoad, steady_crack_load
    from interface_springs import SpringLinearProblem


@dataclass
class OpenCrackEquilibrium:
    displacement: fem.Function
    dry_displacement: fem.Function
    hydraulic_load: SteadyCrackLoad
    report: dict


def solve_open_crack_equilibrium(
    space, faces: PairedCrackFaces, parent_node_ids, boundaries, *,
    elastic_form, external_load, boundary_conditions,
    water_density_kg_m3, gravity_m_s2, saturated_face_mask,
    closure_tolerance_m, options: SteadyCouplingOptions, output: Path,
    interface_springs=None, additional_nodal_forces=None, allow_tied_fronts=False,
) -> OpenCrackEquilibrium:
    """Iterate actual bulk P1 elasticity and audited steady surface flow.

    elastic_form(trial,test) and external_load(test) are unchanged UFL forms.
    No artificial aperture, residual stiffness or inferred water source exists.
    Fixed crack geometry is reference small-displacement geometry.
    """
    if not isinstance(options, SteadyCouplingOptions):
        raise ValueError("Explicit SteadyCouplingOptions are required")
    if not isinstance(faces, PairedCrackFaces):
        raise ValueError("Explicit PairedCrackFaces are required")
    if type(allow_tied_fronts) is not bool:
        raise ValueError("allow_tied_fronts must be an explicit boolean")
    field = fem.Function(space, name="equilibrium_displacement_m")
    if interface_springs is not None and interface_springs.space is not space:
        raise ValueError("Interface springs must belong to the supplied displacement space")
    coordinates, source, source_to_dof = bulk_face_mapping(space, faces)
    fixed_loads = np.zeros_like(field.x.array)
    if additional_nodal_forces is not None:
        forces = np.asarray(additional_nodal_forces)
        if (forces.shape != field.x.array.shape or forces.dtype.kind not in "fiu"
                or not np.isfinite(forces).all()):
            raise ValueError("Additional nodal forces must be a finite vector in displacement DOF order")
        fixed_loads[:] = forces
    bcs = list(boundary_conditions)
    fixed = np.unique(np.concatenate([bc.dof_indices()[0] for bc in bcs])) if bcs else np.empty(0, dtype=int)
    free = np.setdiff1d(np.arange(len(field.x.array)), fixed)
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    bilinear, load_form = elastic_form(trial, test), external_load(test)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running_reduced_open_crack_equilibrium",
        "solver_ready": False, "engineering_validated": False,
        "physical_time_solved": False, "iterations": [],
        "relative_tolerance": options.relative_tolerance,
        "absolute_aperture_tolerance_m": options.absolute_aperture_tolerance_m,
        "maximum_iterations": options.maximum_iterations, "relaxation": options.relaxation,
        "water_density_kg_m3": water_density_kg_m3, "gravity_m_s2": gravity_m_s2,
        "closure_tolerance_m": closure_tolerance_m,
        "allow_tied_fronts": allow_tied_fronts,
        "limitations": "Prescribed open cut with elastic bulk only; no tensile-loss event, contact/friction, irreversible growth, trapped fluid or dam submission.",
    }

    def persist():
        (output / "equilibrium_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    boundaries = tuple(boundaries)
    audit_matrix = None
    base_vector = None
    persist()
    try:
        problem = SpringLinearProblem(
            bilinear, load_form, bcs=bcs, u=field,
            interface_springs=interface_springs, eliminate_interface_bcs=True,
            additional_nodal_forces=fixed_loads,
            petsc_options_prefix="open_crack_equilibrium_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps", "ksp_error_if_not_converged": True},
        )
        audit_matrix = assemble_matrix(fem.form(bilinear))
        audit_matrix.assemble()
        if interface_springs is not None:
            audit_matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            interface_springs.add_to(audit_matrix)
            report["interface_springs"] = interface_springs.report
            offsets, columns, coefficients = interface_springs.matrix.getValuesCSR()
            np.savez_compressed(output / "interface_springs.npz",
                                indptr=offsets, indices=columns, data=coefficients,
                                displacement_cell_dofs=space.dofmap.list)
        base_vector = assemble_vector(fem.form(load_form))
        base_vector.array[:] += fixed_loads
        if not np.isfinite(base_vector.array).all():
            raise ValueError("External bulk loads must be finite")
        problem.solve()
        if problem.solver.getConvergedReason() <= 0:
            raise RuntimeError("Dry bulk equilibrium did not converge")
        dry_residual = base_vector.duplicate()
        try:
            audit_matrix.mult(field.x.petsc_vec, dry_residual)
            dry_residual.axpy(-1., base_vector)
            dry_scale = max(float(base_vector.norm()), np.finfo(float).tiny)
            dry_error = float(np.linalg.norm(dry_residual.array[free]) / dry_scale)
            if not np.isfinite(dry_error) or dry_error > options.relative_tolerance:
                raise RuntimeError("Dry bulk equilibrium failed independent residual check")
            report["dry_mechanical_relative_residual"] = dry_error
            if interface_springs is not None:
                report["dry_interface_spring_energy_j"] = interface_springs.energy(field)
        finally:
            dry_residual.destroy()
        dry = fem.Function(space, name="dry_displacement_m")
        dry.x.array[:] = field.x.array
        _, dry_values = source_vector_values(dry)
        np.savez_compressed(output / "dry_state.npz", displacement_m=dry_values)
        np.savez_compressed(output / "native_inputs.npz", coordinates_m=coordinates,
                            cells=source, minus=faces.minus, plus=faces.plus,
                            normals=faces.normals, parent_node_ids=parent_node_ids,
                            saturated_face_mask=saturated_face_mask,
                            assembled_external_force_n=base_vector.array,
                            additional_nodal_forces_n=fixed_loads)
        residual = base_vector.duplicate()
        try:
            for iteration in range(1, options.maximum_iterations + 1):
                _, values = source_vector_values(field)
                hydraulic = steady_crack_load(
                    faces, values, parent_node_ids, boundaries,
                    water_density_kg_m3=water_density_kg_m3, gravity_m_s2=gravity_m_s2,
                    saturated_face_mask=saturated_face_mask, closure_tolerance_m=closure_tolerance_m,
                    relative_tolerance=options.flow_relative_tolerance,
                    allow_tied_fronts=allow_tied_fronts,
                )
                nodal = np.empty_like(field.x.array.reshape(-1, 3))
                nodal[source_to_dof] = hydraulic.nodal_forces_n
                problem.additional_nodal_forces[:] = fixed_loads + nodal.ravel()
                old = field.x.array.copy()
                problem.solve()
                if problem.solver.getConvergedReason() <= 0:
                    raise RuntimeError("Wet bulk equilibrium did not converge")
                correction = float(np.max(np.abs(field.x.array - old)))
                field.x.array[:] = old + options.relaxation * (field.x.array - old)
                field.x.scatter_forward()
                # Recompute pressure from the accepted candidate's aperture,
                # not the previous iterate used to form the mechanical RHS.
                _, values = source_vector_values(field)
                accepted = steady_crack_load(
                    faces, values, parent_node_ids, boundaries,
                    water_density_kg_m3=water_density_kg_m3, gravity_m_s2=gravity_m_s2,
                    saturated_face_mask=saturated_face_mask, closure_tolerance_m=closure_tolerance_m,
                    relative_tolerance=options.flow_relative_tolerance,
                    allow_tied_fronts=allow_tied_fronts,
                )
                nodal[source_to_dof] = accepted.nodal_forces_n
                total = base_vector.array + nodal.ravel()
                audit_matrix.mult(field.x.petsc_vec, residual)
                residual.array[:] -= total
                force_scale = max(float(np.linalg.norm(total)), np.finfo(float).tiny)
                error = float(np.linalg.norm(residual.array[free]) / force_scale)
                threshold = (options.absolute_aperture_tolerance_m
                             + options.relative_tolerance * float(np.max(np.abs(field.x.array))))
                reaction = residual.array.copy()
                reaction[free] = 0.
                balance = reaction.reshape(-1, 3).sum(axis=0) + total.reshape(-1, 3).sum(axis=0)
                force_error = float(np.linalg.norm(balance) / force_scale)
                if not np.isfinite([error, force_error, correction]).all():
                    raise RuntimeError("Non-finite coupled equilibrium residual")
                row = {
                    "iteration": iteration, "displacement_correction_m": correction,
                    "displacement_limit_m": threshold, "mechanical_relative_residual": error,
                    "relative_global_force_imbalance": force_error,
                    "flow_relative_residual": accepted.report["relative_free_residual"],
                    "maximum_opening_m": faces.measure(values)["maximum_opening_m"],
                }
                report["iterations"].append(row)
                persist()
                if (correction <= threshold and error <= options.relative_tolerance
                        and force_error <= options.relative_tolerance):
                    report["status"] = "passed_reduced_open_crack_equilibrium"
                    report["hydraulics"] = accepted.report
                    if interface_springs is not None:
                        report["interface_spring_energy_j"] = interface_springs.energy(field)
                    np.savez_compressed(output / "equilibrium_state.npz",
                                        displacement_m=values, pressure_pa=accepted.vertex_pressures_pa,
                                        head_m=accepted.vertex_heads_m, nodal_water_forces_n=accepted.nodal_forces_n)
                    with io.VTKFile(space.mesh.comm, output / "displacement.pvd", "w") as writer:
                        writer.write_function(dry, 0.)
                        writer.write_function(field, 1.)
                    report["state_output_axis"] = "0=dry, 1=coupled equilibrium; not physical seconds"
                    persist()
                    return OpenCrackEquilibrium(field, dry, accepted, report)
        finally:
            residual.destroy()
        raise RuntimeError("Coupled bulk/flow iteration limit reached; no equilibrium accepted")
    except (ValueError, RuntimeError, PETSc.Error) as error:
        report["status"] = "failed_reduced_open_crack_equilibrium"
        report["error"] = str(error)
        persist()
        raise RuntimeError(f"Open-crack equilibrium failed; no coupled state accepted: {error}") from error
    finally:
        if audit_matrix is not None:
            audit_matrix.destroy()
        if base_vector is not None:
            base_vector.destroy()
