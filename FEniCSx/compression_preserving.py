"""Experimental Amor split mechanics, not unilateral crack-face contact.

Only the negative volumetric energy is undegraded. Deviatoric energy remains
degraded, including in compression. This is not a spectral split or friction.
The dam workflow does not select this experimental formulation automatically.
"""
from __future__ import annotations

from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

from elasticity_solver import rigid_body_near_nullspace
from phase_field_solver import degradation, strain, tensile_energy


def elastic_energy(displacement, damage, bulk: float, mu: float, kappa: float):
    trace = ufl.tr(strain(displacement))
    return (
        degradation(damage, kappa) * tensile_energy(displacement, bulk, mu)
        + 0.5 * bulk * ufl.min_value(trace, 0.0)**2
    )


def stress(displacement, damage, bulk: float, mu: float, kappa: float):
    eps = strain(displacement)
    trace = ufl.tr(eps)
    factor = degradation(damage, kappa)
    return (
        2 * mu * factor * ufl.dev(eps)
        + bulk * (factor * ufl.max_value(trace, 0.0) + ufl.min_value(trace, 0.0))
        * ufl.Identity(3)
    )


class MechanicsProblem:
    """Reusable fixed-damage equilibrium with checked, conditional AMG reuse."""

    def __init__(
        self, displacement, damage, bulk: float, mu: float, kappa: float,
        external_work, *, bcs=(), support_form=None,
        relative_tolerance: float = 1e-8, absolute_tolerance: float = 1e-10,
        maximum_iterations: int = 30, linear_maximum_iterations: int = 2000,
        direct_solver: bool = False,
        spectral: bool = False,
    ):
        for name, value in (
            ("bulk modulus", bulk), ("shear modulus", mu), ("residual stiffness", kappa),
            ("relative tolerance", relative_tolerance), ("absolute tolerance", absolute_tolerance),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Experimental {name} must be finite and positive")
        for value in (maximum_iterations, linear_maximum_iterations):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Experimental iteration limits must be positive integers")
        self.displacement, self.damage = displacement, damage
        self.kappa, self.bcs = kappa, bcs
        self.rtol, self.atol = relative_tolerance, absolute_tolerance
        self.direct_solver = direct_solver
        self.previous_degradation = None
        self.initial_linear_iterations = None
        self.rebuild_requested = True
        self.norms = []
        self.spectral_split = None
        space = displacement.function_space
        test, trial = ufl.TestFunction(space), ufl.TrialFunction(space)
        load = external_work(test, damage)
        self.load_form = fem.form(load)
        residual = ufl.inner(stress(displacement, damage, bulk, mu, kappa), strain(test)) * ufl.dx - load
        factor = degradation(damage, kappa)
        # Choose the tensile-side generalized derivative at zero trace.
        tangent_stress = (
            2 * mu * factor * ufl.dev(strain(trial))
            + bulk * ufl.conditional(ufl.ge(ufl.tr(strain(displacement)), 0), factor, 1.0)
            * ufl.tr(strain(trial)) * ufl.Identity(3)
        )
        tangent = ufl.inner(tangent_stress, strain(test)) * ufl.dx
        if spectral:
            from spectral_split import SpectralSplit

            self.spectral_split = SpectralSplit(displacement, damage, bulk, mu, kappa)
            residual = ufl.inner(self.spectral_split.stress, strain(test)) * ufl.dx - load
            tangent = ufl.inner(self.spectral_split.tangent_stress(trial), strain(test)) * ufl.dx
        if support_form is not None:
            residual += support_form(displacement, test)
            tangent += support_form(trial, test)
        linear_options = (
            {"ksp_type": "preonly", "pc_type": "lu"}
            if direct_solver else {
                "ksp_type": "gmres", "ksp_gmres_restart": 100,
                "ksp_norm_type": "unpreconditioned", "ksp_pc_side": "right",
                "ksp_rtol": relative_tolerance, "ksp_max_it": linear_maximum_iterations,
                "ksp_atol": 0.1 * absolute_tolerance, "pc_type": "gamg",
            }
        )
        self.problem = NonlinearProblem(
            residual, displacement, J=tangent, bcs=bcs, petsc_options_prefix="experimental_amor_",
            petsc_options={
                "snes_type": "newtonls", "snes_linesearch_type": "bt",
                "snes_rtol": relative_tolerance, "snes_atol": absolute_tolerance,
                "snes_stol": 0.0, "snes_max_it": maximum_iterations,
                "snes_lag_preconditioner": -2,
                "snes_lag_preconditioner_persists": True,
                "snes_error_if_not_converged": True,
                "ksp_error_if_not_converged": True, **linear_options,
            },
        )
        self.problem.A.setNearNullSpace(rigid_body_near_nullspace(space))
        self.problem.solver.setMonitor(self._monitor)
        if self.spectral_split is not None:
            self.problem.solver.setFunction(self._assemble_spectral_residual, self.problem.b)
            self.problem.solver.setJacobian(
                self._assemble_spectral_jacobian, self.problem.A, self.problem.P_mat,
            )

    def _update_spectral(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        fem.petsc.assign(x, self.displacement)
        self.spectral_split.update()

    def _assemble_spectral_residual(self, snes, x, b):
        self._update_spectral(x)
        fem.petsc.assemble_residual(
            snes, x, b, self.displacement, self.problem.F, self.problem.J, self.bcs,
        )

    def _assemble_spectral_jacobian(self, snes, x, matrix, preconditioner):
        self._update_spectral(x)
        fem.petsc.assemble_jacobian(
            snes, x, matrix, preconditioner, self.displacement,
            self.problem.J, self.problem.preconditioner, self.bcs,
        )

    def _monitor(self, snes, iteration, norm):
        self.norms.append(float(norm))
        # A further Newton correction may switch compression branches.
        rtol, atol, _, _ = snes.getTolerances()
        if iteration > 0 and norm > max(atol, rtol * self.norms[0]):
            self._set_preconditioner_lag(rebuild=True)
        if not self.direct_solver:
            snes.getKSP().setTolerances(atol=self.linear_absolute_tolerance)

    def _set_preconditioner_lag(self, *, rebuild):
        # SNES otherwise overrides PC-level reuse on its first Jacobian.
        if not rebuild:
            return
        options = PETSc.Options(self.problem.solver.getOptionsPrefix())
        options["snes_lag_preconditioner"] = -2
        options["snes_lag_preconditioner_persists"] = True
        tolerances = self.problem.solver.getTolerances()
        try:
            self.problem.solver.setFromOptions()
        finally:
            self.problem.solver.setTolerances(*tolerances)
            del options["snes_lag_preconditioner"]
            del options["snes_lag_preconditioner_persists"]

    def solve(self) -> dict:
        domain = self.displacement.function_space.mesh
        invalid_damage = bool(np.any(
            ~np.isfinite(self.damage.x.array)
            | (self.damage.x.array < 0) | (self.damage.x.array > 1 + 1e-12),
        ))
        if domain.comm.allreduce(invalid_damage, op=MPI.LOR):
            raise ValueError("Experimental mechanics requires finite damage in [0, 1]")
        load_vector = fem.petsc.assemble_vector(self.load_form)
        try:
            load_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(load_vector, self.bcs, alpha=0.0)
            load_norm = float(load_vector.norm())
        finally:
            load_vector.destroy()
        if not np.isfinite(load_norm):
            raise ValueError("Experimental mechanics requires a finite applied load vector")
        problem = self.problem
        problem.solver.setTolerances(
            rtol=0.0 if load_norm else self.rtol, atol=max(self.atol, self.rtol * load_norm),
        )
        current = (1 - self.damage.x.array)**2 + self.kappa
        refresh = self.rebuild_requested or self.direct_solver or self.previous_degradation is None
        if self.previous_degradation is not None:
            change = domain.comm.allreduce(float(max(
                np.max(current / self.previous_degradation),
                np.max(self.previous_degradation / current),
            )), op=MPI.MAX)
            refresh = refresh or change > 2.0
        self._set_preconditioner_lag(rebuild=refresh)
        self.linear_absolute_tolerance = 0.1 * max(self.atol, self.rtol * load_norm)
        self.norms.clear()
        try:
            problem.solve()
        except PETSc.Error as error:
            raise RuntimeError(
                "Experimental compression-preserving mechanics failed: "
                f"SNES reason={int(problem.solver.getConvergedReason())}, "
                f"KSP reason={int(problem.solver.getKSP().getConvergedReason())}, "
                f"nonlinear residuals={self.norms}"
            ) from error
        self.displacement.x.scatter_forward()
        vector = problem.b.duplicate()
        try:
            problem.solver.computeFunction(problem.x, vector)
            final_norm = float(vector.norm())
        finally:
            vector.destroy()
        reason = int(problem.solver.getConvergedReason())
        initial_norm = self.norms[0]
        reference_norm = load_norm if load_norm else initial_norm
        limit = max(self.atol, self.rtol * reference_norm)
        nonlinear_iterations = int(problem.solver.getIterationNumber())
        linear_iterations = int(problem.solver.getLinearSolveIterations())
        report = {
            "formulation": (
                "experimental principal-strain spectral split" if self.spectral_split is not None
                else "experimental compression-preserving Amor volumetric/deviatoric"
            ),
            "compression_contact": False, "engineering_validated": False,
            "nonlinear_reason": reason, "nonlinear_iterations": nonlinear_iterations,
            "linear_iterations": linear_iterations,
            "linear_solver": "direct LU" if self.direct_solver else "GMRES/GAMG",
            "initial_residual_norm": initial_norm, "load_vector_norm": load_norm,
            "residual_reference_norm": reference_norm,
            "residual_reference": "applied load" if load_norm else "initial residual (zero applied load)",
            "true_residual_norm": final_norm, "residual_limit": limit,
            "linear_absolute_tolerance": self.linear_absolute_tolerance,
        }
        if reason <= 0 or not np.isfinite(initial_norm) or not np.isfinite(final_norm) or final_norm > limit:
            raise RuntimeError(f"Experimental compression-preserving mechanics failed: {report}")
        if self.initial_linear_iterations is None and nonlinear_iterations:
            self.initial_linear_iterations = max(1, linear_iterations / nonlinear_iterations)
        reuse = (
            not self.direct_solver
            and self.initial_linear_iterations is not None
            and problem.solver.getKSP().getIterationNumber() <= 2 * self.initial_linear_iterations
        )
        self.rebuild_requested = not reuse
        self.previous_degradation = current.copy()
        report["preconditioner_rebuild_requested"] = not reuse
        return report


def solve_mechanics(displacement, damage, bulk, mu, kappa, external_work, **kwargs) -> dict:
    """One-off benchmark/probe; coupled callers retain a MechanicsProblem."""
    return MechanicsProblem(displacement, damage, bulk, mu, kappa, external_work, **kwargs).solve()
