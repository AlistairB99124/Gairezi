"""Staggered hybrid AT2 phase-field fracture solver for DOLFINx 0.11.

The mechanics equation uses isotropic elasticity degraded by
``g(alpha) = (1 - alpha)^2 + kappa``. Damage is driven only by the tensile
volumetric/deviatoric energy history, which prevents compression from creating
new damage. This hybrid formulation does not enforce crack-face contact, so
interpenetration under compression remains a known limitation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

from crack_field import CrackSpecification, PhaseFieldSolverOptions, crack_phase_field


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


def _relative_change(current: np.ndarray, previous: np.ndarray) -> float:
    numerator = float(np.linalg.norm(current - previous, ord=np.inf))
    denominator = max(float(np.linalg.norm(current, ord=np.inf)), 1.0)
    return numerator / denominator


def _solve_damage_obstacle(
    damage: fem.Function,
    lower_bound: fem.Function,
    history: fem.Function,
    gc: float,
    length_scale: float,
    options: PhaseFieldSolverOptions,
) -> dict:
    """Minimise the quadratic AT2 functional subject to old <= alpha <= 1."""
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
    tao.setTolerances(
        gatol=options.damage_gradient_tolerance,
        grtol=options.damage_gradient_tolerance,
        gttol=options.damage_gradient_tolerance,
    )
    tao.setMaximumIterations(options.damage_maximum_iterations)
    tao.setFromOptions()
    tao.solve(damage.x.petsc_vec)
    damage.x.scatter_forward()
    reason = int(tao.getConvergedReason())
    report = {
        "convergence_reason": reason,
        "iterations": int(tao.getIterationNumber()),
        "objective": float(tao.getObjectiveValue()),
        "gradient_norm": float(tao.getSolutionStatus()[4]),
    }
    tao.destroy()
    matrix.destroy()
    rhs.destroy()
    lower.destroy()
    upper.destroy()
    work.destroy()
    if reason <= 0:
        raise RuntimeError(f"Bound-constrained damage solve failed: {report}")
    tolerance = 64 * np.finfo(damage.x.array.dtype).eps
    if np.any(damage.x.array < lower_bound.x.array - tolerance) or np.any(damage.x.array > 1.0 + tolerance):
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
) -> PhaseFieldResult:
    """Solve mechanics and irreversible AT2 damage by alternate minimisation.

    ``external_work(test, load_factor, damage)`` returns the linear load form.
    ``support_form(trial, test)`` may return a spring bilinear form.
    ``boundary_conditions(space, load_factor)`` may return displacement BCs.
    """
    if not specifications:
        raise ValueError("At least one crack specification is required")
    first = specifications[0]
    for specification in specifications[1:]:
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
    if options.field_output_interval < 1:
        raise ValueError("field_output_interval must be positive")

    lam, mu, bulk = elastic_parameters(young, poisson)
    displacement_space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    damage_space = fem.functionspace(domain, ("Lagrange", 1))
    history_space = fem.functionspace(domain, ("DG", 0))
    displacement = fem.Function(displacement_space, name="displacement_m")
    damage = crack_phase_field(
        domain,
        specifications,
        name="damage",
        check_resolution=False,
    )
    old_damage = fem.Function(damage_space, name="damage_lower_bound")
    old_damage.x.array[:] = damage.x.array
    old_damage.x.scatter_forward()
    history = fem.Function(history_space, name="tensile_energy_history_pa")
    history_candidate = fem.Function(history_space)

    trial, test = ufl.TrialFunction(displacement_space), ufl.TestFunction(displacement_space)
    dx = ufl.Measure("dx", domain=domain)
    stiffness = degradation(damage, first.residual_stiffness) * ufl.inner(
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

    previous_energy = None
    for step in range(1, first.load_steps + 1):
        load_factor = step / first.load_steps
        converged = False
        iterations = []
        for staggered_iteration in range(1, options.maximum_staggered_iterations + 1):
            previous_displacement = displacement.x.array.copy()
            previous_damage = damage.x.array.copy()
            bcs = boundary_conditions(displacement_space, load_factor) if boundary_conditions else []
            problem = LinearProblem(
                stiffness,
                external_work(test, load_factor, damage),
                bcs=bcs,
                u=displacement,
                petsc_options_prefix=f"phase_mechanics_{step}_{staggered_iteration}_",
                petsc_options={
                    "ksp_type": "cg",
                    "pc_type": "gamg",
                    "ksp_rtol": options.mechanics_relative_tolerance,
                    "ksp_max_it": options.mechanics_maximum_iterations,
                    "ksp_error_if_not_converged": True,
                },
            )
            displacement = problem.solve()
            displacement.x.scatter_forward()
            mechanics_reason = int(problem.solver.getConvergedReason())
            if mechanics_reason <= 0:
                raise RuntimeError(f"Mechanics solve failed: PETSc reason {mechanics_reason}")

            history_candidate.interpolate(
                fem.Expression(
                    tensile_energy(displacement, bulk, mu),
                    history_space.element.interpolation_points,
                )
            )
            history.x.array[:] = np.maximum(history.x.array, history_candidate.x.array)
            history.x.scatter_forward()
            damage_report = _solve_damage_obstacle(
                damage,
                old_damage,
                history,
                first.critical_energy_release_rate_j_m2,
                first.length_scale_m,
                options,
            )
            displacement_change = _relative_change(displacement.x.array, previous_displacement)
            damage_change = _relative_change(damage.x.array, previous_damage)
            residual = max(displacement_change, damage_change)
            iterations.append(
                {
                    "iteration": staggered_iteration,
                    "relative_displacement_change": displacement_change,
                    "maximum_damage_change": damage_change,
                    "staggered_residual": residual,
                    "mechanics_iterations": int(problem.solver.getIterationNumber()),
                    "mechanics_reason": mechanics_reason,
                    "damage_solver": damage_report,
                }
            )
            if residual <= options.staggered_tolerance:
                converged = True
                break
        if not converged:
            raise RuntimeError(
                f"Staggered phase-field solve failed to converge at load step {step}/"
                f"{first.load_steps}; residual={iterations[-1]['staggered_residual']}"
            )

        old_damage.x.array[:] = np.maximum(old_damage.x.array, damage.x.array)
        old_damage.x.scatter_forward()
        eps = strain(displacement)
        elastic_density = degradation(damage, first.residual_stiffness) * (
            0.5 * lam * ufl.tr(eps) ** 2 + mu * ufl.inner(eps, eps)
        )
        fracture_density = first.critical_energy_release_rate_j_m2 * (
            damage**2 / (2.0 * first.length_scale_m)
            + first.length_scale_m * ufl.inner(ufl.grad(damage), ufl.grad(damage)) / 2.0
        )
        elastic_energy = _global_scalar(domain, elastic_density * dx)
        fracture_energy = _global_scalar(domain, fracture_density * dx)
        total_energy = elastic_energy + fracture_energy
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
            "fracture_energy_j": fracture_energy,
            "total_internal_energy_j": total_energy,
        }
        step_reports.append(step_report)
        if output is not None and (
            step % options.field_output_interval == 0 or step == first.load_steps
        ):
            vtk_displacement.write_function(displacement, load_factor)
            vtk_damage.write_function(damage, load_factor)
            vtk_history.write_function(history, load_factor)

    if output is not None:
        vtk_displacement.close()
        vtk_damage.close()
        vtk_history.close()
    report = {
        "model": "hybrid AT2 phase-field fracture",
        "damage_convention": "0 intact, 1 fully damaged",
        "energy_split": first.energy_split,
        "residual_stiffness": first.residual_stiffness,
        "irreversibility": "PETSc TAO bound-constrained minimisation: alpha_n <= alpha <= 1",
        "compression_contact": False,
        "engineering_validated": False,
        "steps": step_reports,
    }
    if output is not None and domain.comm.rank == 0:
        (output / "phase_field_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return PhaseFieldResult(displacement, damage, history, report)
