"""Staggered hybrid AT2 phase-field fracture solver for DOLFINx 0.11.

The mechanics equation uses isotropic elasticity degraded by
``g(alpha) = (1 - alpha)^2 + kappa``. Damage is driven only by the tensile
volumetric/deviatoric energy history, which excludes volumetric compression
but retains deviatoric driving energy. This hybrid formulation does not enforce crack-face contact, so
interpenetration under compression remains a known limitation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Callable

from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

from crack_field import CrackSpecification, PhaseFieldParameters, PhaseFieldSolverOptions, crack_phase_field
from elasticity_solver import rigid_body_near_nullspace

GROSS_SMALL_STRAIN_BREAKDOWN = 1.0


@dataclass
class PhaseFieldResult:
    displacement: fem.Function
    damage: fem.Function
    history: fem.Function
    report: dict


def strain(displacement):
    return ufl.sym(ufl.grad(displacement))


def elastic_parameters(young: float, poisson: float) -> tuple[float, float, float]:
    if young <= 0 or not -1.0 < poisson < 0.5:
        raise ValueError("Elastic material requires E > 0 and -1 < nu < 0.5")
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    bulk = lam + 2.0 * mu / 3.0
    return lam, mu, bulk


def undamaged_stress(displacement, lam: float, mu: float):
    eps = strain(displacement)
    return 2.0 * mu * eps + lam * ufl.tr(eps) * ufl.Identity(3)


def tensile_energy(displacement, bulk: float, mu: float):
    """Amor volumetric/deviatoric tensile driving energy."""
    eps = strain(displacement)
    trace = ufl.tr(eps)
    positive_trace = ufl.max_value(trace, 0.0)
    deviator = ufl.dev(eps)
    return 0.5 * bulk * positive_trace**2 + mu * ufl.inner(deviator, deviator)


def degradation(damage, residual_stiffness: float):
    return (1.0 - damage) ** 2 + residual_stiffness


def _global_scalar(domain, expression) -> float:
    local = fem.assemble_scalar(fem.form(expression))
    return float(domain.comm.allreduce(local, op=MPI.SUM))


def _relative_change(current: np.ndarray, previous: np.ndarray, comm=MPI.COMM_SELF) -> float:
    numerator = float(np.linalg.norm(current - previous, ord=np.inf))
    denominator = max(float(np.linalg.norm(current, ord=np.inf)), 1.0)
    numerator = comm.allreduce(numerator, op=MPI.MAX)
    denominator = comm.allreduce(denominator, op=MPI.MAX)
    return numerator / denominator


def _aitken_factor(residual, previous_residual, previous_factor, owned_dofs, comm) -> float:
    if previous_residual is None:
        return 1.0
    difference = residual[:owned_dofs] - previous_residual[:owned_dofs]
    numerator = comm.allreduce(float(np.dot(previous_residual[:owned_dofs], difference)), op=MPI.SUM)
    denominator = comm.allreduce(float(np.dot(difference, difference)), op=MPI.SUM)
    if denominator <= np.finfo(float).tiny:
        return 1.0
    # Positive feedback gives a negative extrapolation; retain the original
    # minimisation update rather than stalling it with artificial damping.
    if numerator >= 0:
        return 1.0
    return float(np.clip(-previous_factor * numerator / denominator, 0.5, 2.0))


def load_schedule(load_steps: int, final_load_substeps: int = 1) -> list[float]:
    """Preserve the configured ramp, optionally subdividing only its last interval."""
    for name, value in (("load_steps", load_steps), ("final_load_substeps", final_load_substeps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return [step / load_steps for step in range(1, load_steps)] + [
        (load_steps - 1 + substep / final_load_substeps) / load_steps
        for substep in range(1, final_load_substeps + 1)
    ]


def _update_history(history, candidate, accepted_history=None):
    lower = history.x.array if accepted_history is None else accepted_history
    history.x.array[:] = np.maximum(lower, candidate.x.array)
    history.x.scatter_forward()


def _damage_change_details(damage, previous, lower_bound) -> dict:
    """Locate the largest unrelaxed change using owned scalar DOFs only."""
    space = damage.function_space
    owned = space.dofmap.index_map.size_local
    local = None
    if owned:
        change = damage.x.array[:owned] - previous[:owned]
        index = int(np.argmax(np.abs(change)))
        local = {
            "maximum_absolute_change": float(abs(change[index])),
            "signed_change": float(change[index]),
            "coordinate_xyz_m": space.tabulate_dof_coordinates()[index].tolist(),
            "previous_iterate_damage": float(previous[index]),
            "candidate_damage": float(damage.x.array[index]),
            "previous_load_damage": float(lower_bound.x.array[index]),
        }
    candidates = [record for record in space.mesh.comm.allgather(local) if record is not None]
    if not candidates:
        raise RuntimeError("Cannot locate damage change: no owned damage DOFs")
    return max(candidates, key=lambda record: record["maximum_absolute_change"])


def _strain_breakdown_details(displacement, damage, strain_norm) -> dict:
    domain = displacement.function_space.mesh
    owned_cells = domain.topology.index_map(domain.topology.dim).size_local
    norm_dofs = strain_norm.function_space.dofmap.list[:owned_cells, 0]
    norms = strain_norm.x.array[norm_dofs]
    invalid = np.flatnonzero(~np.isfinite(norms))
    cell = int(invalid[0] if invalid.size else np.argmax(norms))
    centroid = domain.geometry.x[domain.geometry.dofmaps[0][cell]].mean(axis=0)
    eigenvalues = None
    if np.isfinite(norms[cell]):
        expression = fem.Expression(
            strain(displacement),
            strain_norm.function_space.element.interpolation_points,
        )
        tensor = expression.eval(domain, np.array([cell], dtype=np.int32)).reshape(3, 3)
        eigenvalues = np.linalg.eigvalsh(tensor).tolist()
    local = {
        "maximum_strain_norm": float(norms[cell]) if np.isfinite(norms[cell]) else None,
        "centroid_xyz_m": centroid.tolist(),
        "principal_strains": eigenvalues,
        "damage_at_vertices": damage.x.array[damage.function_space.dofmap.cell_dofs(cell)].tolist(),
    }
    return max(
        domain.comm.allgather(local),
        key=lambda record: (
            record["maximum_strain_norm"] is None,
            record["maximum_strain_norm"] or 0.0,
        ),
    )


def _solve_damage_obstacle(
    damage: fem.Function,
    lower_bound: fem.Function,
    history: fem.Function,
    gc: float,
    length_scale: float,
    options: PhaseFieldSolverOptions,
    fixed_damage_dofs=None,
) -> dict:
    """Minimise the quadratic AT2 functional subject to old <= alpha <= 1."""
    started = time.perf_counter()
    space = damage.function_space
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    dx = ufl.Measure("dx", domain=space.mesh)
    bilinear = (
        (gc / length_scale + 2.0 * history) * trial * test
        + gc * length_scale * ufl.inner(ufl.grad(trial), ufl.grad(test))
    ) * dx
    linear = 2.0 * history * test * dx
    matrix = fem.petsc.assemble_matrix(fem.form(bilinear))
    matrix.assemble()
    rhs = fem.petsc.assemble_vector(fem.form(linear))
    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    lower = damage.x.petsc_vec.duplicate()
    upper = damage.x.petsc_vec.duplicate()
    lower_bound.x.petsc_vec.copy(lower)
    upper.set(1.0)
    if fixed_damage_dofs is not None:
        with upper.localForm() as local:
            local.array[fixed_damage_dofs] = 0.0
    work = damage.x.petsc_vec.duplicate()

    def objective_gradient(_tao, x, gradient):
        matrix.mult(x, gradient)
        gradient.axpy(-1.0, rhs)
        matrix.mult(x, work)
        return 0.5 * x.dot(work) - rhs.dot(x)

    def hessian(_tao, _x, _hessian, _preconditioner):
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    tao = PETSc.TAO().create(space.mesh.comm)
    tao.setType(PETSc.TAO.Type.TRON)
    tao.setObjectiveGradient(objective_gradient)
    tao.setHessian(hessian, matrix, matrix)
    tao.setVariableBounds(lower, upper)
    def projected_gradient_norm():
        matrix.mult(damage.x.petsc_vec, work)
        work.axpy(-1.0, rhs)
        gradient = work.array
        values = damage.x.petsc_vec.array
        active = (
            ((values <= lower.array) & (gradient > 0))
            | ((values >= upper.array) & (gradient < 0))
        )
        gradient[active] = 0
        return float(work.norm())

    initial_gradient_norm = projected_gradient_norm()
    tao.setTolerances(
        gatol=options.damage_gradient_tolerance,
        # The quadratic omits a history-dependent constant, so |f| is not a
        # meaningful scale for deciding whether the projected gradient is small.
        grtol=0.0,
        gttol=options.damage_gradient_tolerance,
    )
    tao.setMaximumIterations(options.damage_maximum_iterations)
    tao.setFromOptions()
    setup_seconds = time.perf_counter() - started
    solve_started = time.perf_counter()
    tao.solve(damage.x.petsc_vec)
    solve_seconds = time.perf_counter() - solve_started
    damage.x.scatter_forward()
    reason = int(tao.getConvergedReason())
    report = {
        "convergence_reason": reason,
        "iterations": int(tao.getIterationNumber()),
        "objective": float(tao.getObjectiveValue()),
        "gradient_norm": projected_gradient_norm(),
        "tao_gradient_norm": float(tao.getSolutionStatus()[2]),
        "initial_gradient_norm": initial_gradient_norm,
        "gradient_tolerance": options.damage_gradient_tolerance,
        "setup_seconds": setup_seconds,
        "solve_seconds": solve_seconds,
    }
    tao.destroy()
    matrix.destroy()
    rhs.destroy()
    lower.destroy()
    upper.destroy()
    work.destroy()
    if reason <= 0:
        raise RuntimeError(f"Bound-constrained damage solve failed: {report}")
    if not np.isfinite(initial_gradient_norm):
        raise RuntimeError(f"Damage solve did not record a finite initial gradient: {report}")
    gradient_limit = options.damage_gradient_tolerance * max(1.0, initial_gradient_norm)
    if not np.isfinite(report["gradient_norm"]) or report["gradient_norm"] > gradient_limit:
        raise RuntimeError(f"Damage projected gradient exceeds tolerance {gradient_limit}: {report}")
    tolerance = 64 * np.finfo(damage.x.array.dtype).eps
    violated = (
        np.any(damage.x.array < lower_bound.x.array - tolerance)
        or np.any(damage.x.array > 1.0 + tolerance)
    )
    if space.mesh.comm.allreduce(bool(violated), op=MPI.LOR):
        raise RuntimeError("Damage solve violated irreversibility or the alpha <= 1 bound")
    return report


def solve_staggered(
    domain,
    young: float,
    poisson: float,
    specifications: list[CrackSpecification],
    options: PhaseFieldSolverOptions,
    external_work: Callable,
    *,
    support_form: Callable | None = None,
    boundary_conditions: Callable | None = None,
    output: Path | None = None,
    damage_cell_tags=None,
    damage_region_id: int = 2,
    parameters: PhaseFieldParameters | None = None,
    mechanics_formulation: str = "hybrid",
    damage_acceleration: bool = False,
    final_load_substeps: int = 1,
    elastic_coefficients=None,
    interface_factory=None,
) -> PhaseFieldResult:
    """Solve mechanics and irreversible AT2 damage by alternate minimisation.

    ``external_work(test, load_factor, damage)`` returns the linear load form.
    ``support_form(trial, test)`` may return a spring bilinear form.
    ``boundary_conditions(space, load_factor)`` may return displacement BCs.
    """
    if not specifications and parameters is None:
        raise ValueError("At least one crack specification is required")
    if mechanics_formulation not in ("hybrid", "experimental_amor", "experimental_spectral"):
        raise ValueError(f"Unknown mechanics formulation: {mechanics_formulation!r}")
    if damage_acceleration and mechanics_formulation == "hybrid":
        raise ValueError("Damage acceleration is supported only with experimental mechanics")
    first = parameters if parameters is not None else specifications[0]
    for specification in specifications:
        if (
            specification.model != first.model
            or specification.energy_split != first.energy_split
            or specification.length_scale_m != first.length_scale_m
            or specification.critical_energy_release_rate_j_m2
            != first.critical_energy_release_rate_j_m2
            or specification.residual_stiffness != first.residual_stiffness
            or specification.load_steps != first.load_steps
        ):
            raise ValueError("All cracks in one coupled solve must share phase-field and loading parameters")
    if first.model != "AT2":
        raise ValueError(f"Only AT2 is implemented, got {first.model!r}")
    if first.energy_split != "volumetric_deviatoric":
        raise ValueError(f"Only volumetric_deviatoric energy split is implemented, got {first.energy_split!r}")
    if first.load_steps < 1:
        raise ValueError("load_steps must be positive")
    load_factors = load_schedule(first.load_steps, final_load_substeps)
    total_load_steps = len(load_factors)
    if options.field_output_interval < 1:
        raise ValueError("field_output_interval must be positive")

    lam, mu, bulk = elastic_parameters(young, poisson)
    damage_mask = 1.
    if elastic_coefficients is not None:
        if mechanics_formulation == "experimental_spectral":
            raise ValueError("Three-body spectral mechanics is not verified; use hybrid or experimental_amor")
        lam, mu, bulk = (elastic_coefficients[key] for key in ("lam", "mu", "bulk"))
        damage_mask = elastic_coefficients["damage_mask"]
    displacement_space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    interfaces = interface_factory(displacement_space) if interface_factory else None
    near_nullspace = rigid_body_near_nullspace(displacement_space)
    damage_space = fem.functionspace(domain, ("Lagrange", 1))
    history_space = fem.functionspace(domain, ("DG", 0))
    displacement = fem.Function(displacement_space, name="displacement_m")
    damage = (
        crack_phase_field(domain, specifications, name="damage", check_resolution=False)
        if specifications else fem.Function(damage_space, name="damage")
    )
    fixed_damage_dofs = None
    if damage_cell_tags is not None:
        non_wall_cells = damage_cell_tags.indices[damage_cell_tags.values != damage_region_id]
        if not domain.comm.allreduce(len(damage_cell_tags.find(damage_region_id)), op=MPI.SUM):
            raise ValueError("Damage region has no tagged cells.")
        domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
        fixed_damage_dofs = fem.locate_dofs_topological(
            damage_space, domain.topology.dim, non_wall_cells,
        )
        damage.x.array[fixed_damage_dofs] = 0.0
        damage.x.scatter_forward()
    old_damage = fem.Function(damage_space, name="damage_lower_bound")
    old_damage.x.array[:] = damage.x.array
    old_damage.x.scatter_forward()
    history = fem.Function(history_space, name="tensile_energy_history_pa")
    history_candidate = fem.Function(history_space)
    strain_norm = fem.Function(history_space)
    strain_norm_expression = fem.Expression(
        ufl.sqrt(ufl.inner(strain(displacement), strain(displacement))),
        history_space.element.interpolation_points,
    )

    trial, test = ufl.TrialFunction(displacement_space), ufl.TestFunction(displacement_space)
    dx = ufl.Measure("dx", domain=domain)
    factor = damage_mask * degradation(damage, first.residual_stiffness) + (1 - damage_mask)
    stiffness = factor * ufl.inner(
        undamaged_stress(trial, lam, mu), strain(test)
    ) * dx
    if support_form is not None:
        stiffness += support_form(trial, test)

    step_reports = []
    vtk_displacement = vtk_damage = vtk_history = None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        vtk_displacement = io.VTKFile(domain.comm, output / "displacement.pvd", "w")
        vtk_damage = io.VTKFile(domain.comm, output / "damage.pvd", "w")
        vtk_history = io.VTKFile(domain.comm, output / "tensile_history.pvd", "w")
        vtk_damage.write_function(damage, 0.0)
        suffix = "" if domain.comm.size == 1 else f"_rank{domain.comm.rank}"
        np.savez(
            output / f"checkpoint_layout{suffix}.npz",
            displacement_cell_dofs=displacement_space.dofmap.list,
            damage_cell_dofs=damage_space.dofmap.list,
            history_cell_dofs=history_space.dofmap.list,
            cell_input_indices=domain.topology.original_cell_index,
            geometry_cell_dofs=domain.geometry.dofmaps[0],
            geometry_input_global_indices=domain.geometry.input_global_indices,
            coordinates=domain.geometry.x,
        )

    previous_energy = None
    mechanics_problem = None
    nonlinear_mechanics_problem = None
    initial_mechanics_iterations = None
    load_coefficient = fem.Constant(domain, PETSc.ScalarType(0))
    for step, load_factor in enumerate(load_factors, start=1):
        load_coefficient.value = load_factor
        accepted_history = (
            history.x.array.copy() if mechanics_formulation != "hybrid" else None
        )
        if boundary_conditions is not None:
            mechanics_problem = None
            nonlinear_mechanics_problem = None
            initial_mechanics_iterations = None
        converged = False
        iterations = []
        previous_damage_residual = None
        relaxation_factor = 1.0
        for staggered_iteration in range(1, options.maximum_staggered_iterations + 1):
            iteration_started = time.perf_counter()
            previous_displacement = displacement.x.array.copy()
            previous_damage = damage.x.array.copy()
            if output is not None:
                suffix = "" if domain.comm.size == 1 else f"_rank{domain.comm.rank}"
                checkpoint = output / f"iteration_checkpoint{suffix}.npz"
                temporary = output / f"iteration_checkpoint{suffix}.tmp.npz"
                np.savez(
                    temporary, displacement=displacement.x.array,
                    damage=damage.x.array, history=history.x.array,
                    lower_bound=old_damage.x.array, step=step,
                    iteration=staggered_iteration, load_factor=load_factor,
                    coordinates=domain.geometry.x,
                    input_global_indices=domain.geometry.input_global_indices,
                    mpi_rank=domain.comm.rank, mpi_processes=domain.comm.size,
                )
                temporary.replace(checkpoint)
            bcs = boundary_conditions(displacement_space, load_factor) if boundary_conditions else []
            setup_started = time.perf_counter()
            if mechanics_problem is None and mechanics_formulation == "hybrid":
                from interface_springs import SpringLinearProblem
                problem_type = SpringLinearProblem if interfaces is not None else LinearProblem
                spring_arguments = {"interface_springs": interfaces} if interfaces is not None else {}
                mechanics_problem = problem_type(
                    stiffness,
                    external_work(test, load_coefficient, damage),
                    bcs=bcs,
                    **spring_arguments,
                    u=displacement,
                    petsc_options_prefix=f"phase_mechanics_{step}_",
                    petsc_options={
                        "ksp_type": "gmres",
                        "ksp_gmres_restart": 100,
                        "ksp_norm_type": "unpreconditioned",
                        "ksp_pc_side": "right",
                        "pc_type": "gamg",
                        "ksp_rtol": 0.1 * options.mechanics_relative_tolerance,
                        "ksp_max_it": options.mechanics_maximum_iterations,
                        "ksp_error_if_not_converged": True,
                        "ksp_initial_guess_nonzero": True,
                    },
                )
            problem = mechanics_problem
            if problem is not None:
                problem.A.setNearNullSpace(near_nullspace)
                fem.petsc.assign(displacement, problem.x)
            setup_seconds = time.perf_counter() - setup_started
            context = {
                "step": step, "load_factor": load_factor, "iteration": staggered_iteration,
                "solver": "gmres", "preconditioner": "gamg",
                "mechanics_formulation": mechanics_formulation,
                "mechanics_residual_reference": (
                    "assembled load vector" if mechanics_formulation == "hybrid"
                    else "applied load (initial residual if zero load); absolute tolerance also reported"
                ),
            }

            def mechanics_progress(event, **details):
                record = {**context, "event": event, "timestamp": time.time(), **details}
                if domain.comm.rank == 0:
                    print("Phase mechanics: " + json.dumps(record, allow_nan=False), flush=True)
                    if output is not None:
                        with (output / "mechanics_diagnostics.jsonl").open("a") as stream:
                            stream.write(json.dumps(record, allow_nan=False) + "\n")

            mechanics_progress("started")
            solve_started = time.perf_counter()
            nonlinear_report = None
            if mechanics_formulation != "hybrid":
                from compression_preserving import MechanicsProblem

                try:
                    if nonlinear_mechanics_problem is None:
                        nonlinear_mechanics_problem = MechanicsProblem(
                            displacement, damage, bulk, mu, first.residual_stiffness,
                            lambda test, alpha: external_work(test, load_coefficient, alpha),
                            bcs=bcs, support_form=support_form,
                            interface_springs=interfaces, damage_mask=damage_mask,
                            relative_tolerance=options.mechanics_relative_tolerance,
                            linear_maximum_iterations=options.mechanics_maximum_iterations,
                            spectral=mechanics_formulation == "experimental_spectral",
                        )
                    nonlinear_report = nonlinear_mechanics_problem.solve()
                except RuntimeError as error:
                    mechanics_progress("failed", error=str(error))
                    raise RuntimeError(
                        f"Experimental mechanics failed at load step {step}/{total_load_steps}, "
                        f"staggered iteration {staggered_iteration}: {error}"
                    ) from error
                mechanics_reason = nonlinear_report["nonlinear_reason"]
                mechanics_iterations = nonlinear_report["linear_iterations"]
                relative_residual = nonlinear_report["true_residual_norm"] / max(
                    nonlinear_report["residual_reference_norm"], np.finfo(float).tiny,
                )
            else:
                mechanics_error = None
                try:
                    displacement = problem.solve()
                except PETSc.Error as error:
                    mechanics_error = error
                reason = int(problem.solver.getConvergedReason())
                if mechanics_error is not None or reason <= 0:
                    error_message = str(mechanics_error) if mechanics_error is not None else f"PETSc reason {reason}"
                    mechanics_progress(
                        "failed", reason=reason,
                        iterations=int(problem.solver.getIterationNumber()), error=error_message,
                    )
                    raise RuntimeError(
                        f"Mechanics solve failed at load step {step}/{total_load_steps}, "
                        f"staggered iteration {staggered_iteration}: {error_message}"
                    ) from mechanics_error
                displacement.x.scatter_forward()
                mechanics_reason = int(problem.solver.getConvergedReason())
                mechanics_iterations = int(problem.solver.getIterationNumber())
                residual_vector = problem.b.duplicate()
                try:
                    problem.A.mult(displacement.x.petsc_vec, residual_vector)
                    residual_vector.axpy(-1.0, problem.b)
                    rhs_norm = float(problem.b.norm())
                    true_residual = float(residual_vector.norm())
                finally:
                    residual_vector.destroy()
                relative_residual = true_residual / rhs_norm if rhs_norm else true_residual
                if not np.isfinite(relative_residual) or (
                    rhs_norm and relative_residual > options.mechanics_relative_tolerance
                ):
                    mechanics_progress(
                        "failed", reason=mechanics_reason, iterations=mechanics_iterations,
                        error=f"True residual {relative_residual} exceeds tolerance",
                    )
                    raise RuntimeError(
                        f"Mechanics true residual {relative_residual} exceeds tolerance "
                        f"{options.mechanics_relative_tolerance} at step {step}, iteration {staggered_iteration}"
                    )
            solve_seconds = time.perf_counter() - solve_started
            mechanics_progress(
                "completed", reason=mechanics_reason,
                iterations=mechanics_iterations,
                relative_true_residual=relative_residual,
                setup_seconds=setup_seconds, solve_seconds=solve_seconds,
                nonlinear_solver=nonlinear_report,
            )
            if problem is not None:
                if initial_mechanics_iterations is None:
                    initial_mechanics_iterations = max(1, mechanics_iterations)
                reuse_preconditioner = mechanics_iterations <= 2 * initial_mechanics_iterations
                problem.solver.getPC().setReusePreconditioner(reuse_preconditioner)
                if not reuse_preconditioner:
                    mechanics_progress("preconditioner_refresh_requested", iterations=mechanics_iterations)

            strain_norm.interpolate(strain_norm_expression)
            finite_strain = not domain.comm.allreduce(
                bool(np.any(~np.isfinite(strain_norm.x.array))), op=MPI.LOR,
            )
            maximum_strain_norm = float(domain.comm.allreduce(
                np.max(strain_norm.x.array, initial=0), op=MPI.MAX,
            ))
            strain_detail = maximum_strain_norm if finite_strain else None
            mechanics_progress("kinematics_checked", maximum_strain_norm=strain_detail)
            if not finite_strain or maximum_strain_norm >= GROSS_SMALL_STRAIN_BREAKDOWN:
                details = _strain_breakdown_details(displacement, damage, strain_norm)
                mechanics_progress("model_invalid", **details)
                raise RuntimeError(
                    f"Small-strain fracture model is invalid at load step {step}/{total_load_steps}, "
                    f"staggered iteration {staggered_iteration}: maximum strain norm "
                    f"{strain_detail} is outside small-strain kinematics near "
                    f"{details['centroid_xyz_m']} m, principal strains "
                    f"{details['principal_strains']}. "
                    "Review fracture/contact and kinematics assumptions before retrying; "
                    "this is not a dam-safety verdict."
                )

            tensile_density = damage_mask * tensile_energy(displacement, bulk, mu)
            if mechanics_formulation == "experimental_spectral":
                tensile_density = nonlinear_mechanics_problem.spectral_split.tensile_energy
            history_candidate.interpolate(
                fem.Expression(
                    tensile_density,
                    history_space.element.interpolation_points,
                )
            )
            _update_history(history, history_candidate, accepted_history)
            mechanics_progress("damage_started")
            damage_started = time.perf_counter()
            damage_report = _solve_damage_obstacle(
                damage,
                old_damage,
                history,
                first.critical_energy_release_rate_j_m2,
                first.length_scale_m,
                options,
                fixed_damage_dofs,
            )
            damage_seconds = time.perf_counter() - damage_started
            mechanics_progress(
                "damage_completed", elapsed_seconds=damage_seconds, **damage_report,
            )
            displacement_change = _relative_change(
                displacement.x.array, previous_displacement, domain.comm,
            )
            damage_change = _relative_change(damage.x.array, previous_damage, domain.comm)
            damage_change_details = _damage_change_details(damage, previous_damage, old_damage)
            residual = max(displacement_change, damage_change)
            if damage_acceleration and residual > options.staggered_tolerance:
                damage_residual = damage.x.array - previous_damage
                relaxation_factor = _aitken_factor(
                    damage_residual, previous_damage_residual, relaxation_factor,
                    damage_space.dofmap.index_map.size_local, domain.comm,
                )
                previous_damage_residual = damage_residual.copy()
                damage.x.array[:] = np.clip(
                    previous_damage + relaxation_factor * damage_residual,
                    old_damage.x.array, 1.0,
                )
                if fixed_damage_dofs is not None:
                    damage.x.array[fixed_damage_dofs] = 0
                damage.x.scatter_forward()
            else:
                relaxation_factor = 1.0
            old_degradation = (1.0 - previous_damage) ** 2 + first.residual_stiffness
            new_degradation = (1.0 - damage.x.array) ** 2 + first.residual_stiffness
            stiffness_change_factor = float(domain.comm.allreduce(max(
                np.max(old_degradation / new_degradation),
                np.max(new_degradation / old_degradation),
            ), op=MPI.MAX))
            if stiffness_change_factor > 2.0 and problem is not None:
                problem.solver.getPC().setReusePreconditioner(False)
                mechanics_progress(
                    "preconditioner_refresh_requested",
                    stiffness_change_factor=stiffness_change_factor,
                )
            mechanics_progress(
                "staggered_completed", relative_displacement_change=displacement_change,
                maximum_damage_change=damage_change, staggered_residual=residual,
                damage_change_location=damage_change_details,
                damage_relaxation_factor=relaxation_factor,
                stiffness_change_factor=stiffness_change_factor,
                maximum_displacement_m=float(domain.comm.allreduce(np.linalg.norm(
                    displacement.x.array.reshape(-1, 3), axis=1,
                ).max(), op=MPI.MAX)),
                minimum_damage=float(domain.comm.allreduce(damage.x.array.min(), op=MPI.MIN)),
                maximum_damage=float(domain.comm.allreduce(damage.x.array.max(), op=MPI.MAX)),
                fully_damaged_dofs=int(domain.comm.allreduce(np.count_nonzero(
                    damage.x.array[:damage_space.dofmap.index_map.size_local] >= 0.99,
                ), op=MPI.SUM)),
            )
            iterations.append(
                {
                    "iteration": staggered_iteration,
                    "relative_displacement_change": displacement_change,
                    "maximum_damage_change": damage_change,
                    "damage_change_location": damage_change_details,
                    "damage_relaxation_factor": relaxation_factor,
                    "staggered_residual": residual,
                    "mechanics_iterations": mechanics_iterations,
                    "nonlinear_mechanics": nonlinear_report,
                    "mechanics_reason": mechanics_reason,
                    "mechanics_relative_true_residual": relative_residual,
                    "maximum_strain_norm": maximum_strain_norm,
                    "damage_solver": damage_report,
                    "mechanics_setup_seconds": setup_seconds,
                    "mechanics_solve_seconds": solve_seconds,
                    "damage_seconds": damage_seconds,
                    "elapsed_seconds": time.perf_counter() - iteration_started,
                }
            )
            if residual <= options.staggered_tolerance:
                converged = True
                break
        if not converged:
            raise RuntimeError(
                f"Staggered phase-field solve failed to converge at load step {step}/"
                f"{total_load_steps}; residual={iterations[-1]['staggered_residual']}"
            )

        old_damage.x.array[:] = np.maximum(old_damage.x.array, damage.x.array)
        old_damage.x.scatter_forward()
        eps = strain(displacement)
        elastic_density = factor * (
            0.5 * lam * ufl.tr(eps) ** 2 + mu * ufl.inner(eps, eps)
        )
        if mechanics_formulation == "experimental_amor":
            from compression_preserving import elastic_energy

            elastic_density = elastic_energy(displacement, damage, bulk, mu, first.residual_stiffness,
                                             damage_mask=damage_mask)
        elif mechanics_formulation == "experimental_spectral":
            elastic_density = nonlinear_mechanics_problem.spectral_split.energy
        fracture_density = first.critical_energy_release_rate_j_m2 * (
            damage**2 / (2.0 * first.length_scale_m)
            + first.length_scale_m * ufl.inner(ufl.grad(damage), ufl.grad(damage)) / 2.0
        )
        elastic_energy = _global_scalar(domain, elastic_density * dx)
        fracture_energy = _global_scalar(domain, fracture_density * dx)
        spring_energy = interfaces.energy(displacement) if interfaces else 0.
        total_energy = elastic_energy + fracture_energy + spring_energy
        if previous_energy is not None and not np.isfinite(total_energy):
            raise RuntimeError("Phase-field energy became non-finite")
        previous_energy = total_energy
        step_report = {
            "step": step,
            "load_factor": load_factor,
            "staggered_iterations": len(iterations),
            "iterations": iterations,
            "maximum_displacement_m": float(
                domain.comm.allreduce(np.linalg.norm(displacement.x.array.reshape(-1, 3), axis=1).max(), op=MPI.MAX)
            ),
            "minimum_damage": float(domain.comm.allreduce(damage.x.array.min(), op=MPI.MIN)),
            "maximum_damage": float(domain.comm.allreduce(damage.x.array.max(), op=MPI.MAX)),
            "elastic_energy_j": elastic_energy,
            "interface_spring_energy_j": spring_energy,
            "fracture_energy_j": fracture_energy,
            "total_internal_energy_j": total_energy,
        }
        step_reports.append(step_report)
        if output is not None and domain.comm.rank == 0:
            progress = output / "load_step_progress.json"
            temporary = output / "load_step_progress.tmp.json"
            temporary.write_text(json.dumps(
                {"status": "partial_progress", "engineering_validated": False, "steps": step_reports},
                indent=2, allow_nan=False,
            ) + "\n")
            temporary.replace(progress)
        if output is not None and (
            step % options.field_output_interval == 0 or step == total_load_steps
        ):
            vtk_displacement.write_function(displacement, load_factor)
            vtk_damage.write_function(damage, load_factor)
            vtk_history.write_function(history, load_factor)

    if output is not None:
        vtk_displacement.close()
        vtk_damage.close()
        vtk_history.close()
    report = {
        "model": (
            "hybrid AT2 phase-field fracture" if mechanics_formulation == "hybrid"
            else (
                "experimental compression-preserving Amor AT2 phase-field fracture"
                if mechanics_formulation == "experimental_amor"
                else "experimental principal-strain spectral AT2 phase-field fracture"
            )
        ),
        "mechanics_formulation": mechanics_formulation,
        "damage_acceleration": "bounded Aitken [0.5, 2]" if damage_acceleration else "none",
        "configured_load_steps": first.load_steps,
        "final_load_substeps": final_load_substeps,
        "load_factors": load_factors,
        "damage_convention": "0 intact, 1 fully damaged",
        "energy_split": (
            "spectral_principal_strain" if mechanics_formulation == "experimental_spectral"
            else first.energy_split
        ),
        "residual_stiffness": first.residual_stiffness,
        "irreversibility": "PETSc TAO bound-constrained minimisation: alpha_n <= alpha <= 1",
        "history_update": (
            "maximum of previous accepted load-step history and current tensile energy"
            if mechanics_formulation != "hybrid"
            else "running maximum over load steps and staggered iterates"
        ),
        "compression_contact": False,
        "engineering_validated": False,
        "kinematics": "small strain; gross breakdown rejected at strain Frobenius norm >= 1",
        "initial_damage": "configured cracks" if specifications else "zero everywhere; no seeded crack",
        "mechanics_solver": {
            "ksp_type": "gmres", "pc_type": "gamg", "near_nullspace": "six rigid-body modes",
            "relative_tolerance": options.mechanics_relative_tolerance,
            "maximum_iterations": options.mechanics_maximum_iterations,
            "residual_check": (
                "explicit unpreconditioned norm(Au-b)/norm(b)"
                if mechanics_formulation == "hybrid"
                else "reassembled nonlinear residual; load-relative and absolute stopping"
            ),
            "initial_guess": "previous displacement",
            "preconditioner_reuse": (
                "within each mechanics problem; true residual required every iteration"
                if mechanics_formulation == "hybrid"
                else "conditional GAMG reuse; refresh for further Newton corrections, large damage changes or iteration growth"
            ),
            "mpi_processes": domain.comm.size,
        },
        "steps": step_reports,
    }
    if output is not None and domain.comm.rank == 0:
        (output / "phase_field_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return PhaseFieldResult(displacement, damage, history, report)
