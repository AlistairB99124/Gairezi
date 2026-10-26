"""Experimental Amor split mechanics, not unilateral crack-face contact.

Only the negative volumetric energy is undegraded. Deviatoric energy remains
degraded, including in compression. This is not a spectral split or friction.
The desktop app submits it by default; the CLI and Python defaults remain hybrid.
"""
from __future__ import annotations

from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

from elasticity_solver import rigid_body_near_nullspace
from phase_field_solver import degradation, strain, tensile_energy, use_mumps_out_of_core_directory

# PETSc SNESConvergedReason for "maximum Newton iterations reached".
SNES_DIVERGED_MAX_IT = -5
# Refactorise the MUMPS preconditioner once a lagged factor needs this many GMRES iterations.
DIRECT_REFACTOR_LINEAR_ITERATIONS = 30


class NewtonIterationLimit(RuntimeError):
    """Newton stopped at its iteration limit; the linear solves themselves converged.

    A different linear solver cannot help, so callers must not treat this as a
    linear-solver failure.
    """


def elastic_energy(displacement, damage, bulk: float, mu: float, kappa: float, *, damage_mask=1.):
    trace = ufl.tr(strain(displacement))
    return (
        (damage_mask * degradation(damage, kappa) + 1 - damage_mask) * tensile_energy(displacement, bulk, mu)
        + 0.5 * bulk * ufl.min_value(trace, 0.0)**2
    )


def stress(displacement, damage, bulk: float, mu: float, kappa: float, *, damage_mask=1.):
    eps = strain(displacement)
    trace = ufl.tr(eps)
    factor = damage_mask * degradation(damage, kappa) + 1 - damage_mask
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
        interface_springs=None,
        damage_mask=1.,
        maximum_continuations: int = 3,
        newton_progress=None,
    ):
        for name, value in (
            ("bulk modulus", bulk), ("shear modulus", mu), ("residual stiffness", kappa),
            ("relative tolerance", relative_tolerance), ("absolute tolerance", absolute_tolerance),
        ):
            if isinstance(value, ufl.core.expr.Expr):
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Experimental {name} must be finite and positive")
        for value in (maximum_iterations, linear_maximum_iterations):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Experimental iteration limits must be positive integers")
        if isinstance(maximum_continuations, bool) or not isinstance(maximum_continuations, int) \
                or maximum_continuations < 0:
            raise ValueError("Experimental Newton continuations must be a non-negative integer")
        self.displacement, self.damage = displacement, damage
        self.kappa, self.bcs = kappa, bcs
        self.rtol, self.atol = relative_tolerance, absolute_tolerance
        self.direct_solver = direct_solver
        self.maximum_iterations = maximum_iterations
        self.maximum_continuations = maximum_continuations
        # Called as newton_progress(iteration, residual_norm, residual_limit) after each Newton update.
        self.newton_progress = newton_progress
        self.residual_limit = None
        self.round_initial_norm = 0.0
        self.completed_newton_iterations = 0
        self.previous_degradation = None
        self.initial_linear_iterations = None
        self.rebuild_requested = True
        self.norms = []
        self.spectral_split = None
        self.interface_springs = interface_springs
        space = displacement.function_space
        test, trial = ufl.TestFunction(space), ufl.TrialFunction(space)
        load = external_work(test, damage)
        self.load_form = fem.form(load)
        residual = ufl.inner(stress(displacement, damage, bulk, mu, kappa, damage_mask=damage_mask), strain(test)) * ufl.dx - load
        factor = damage_mask * degradation(damage, kappa) + 1 - damage_mask
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
        linear_options = {
            "ksp_type": "gmres", "ksp_gmres_restart": 100,
            "ksp_norm_type": "unpreconditioned", "ksp_pc_side": "right",
            "ksp_rtol": relative_tolerance, "ksp_max_it": linear_maximum_iterations,
            "ksp_atol": 0.1 * absolute_tolerance,
        }
        if direct_solver:
            # GMRES keeps each Newton step exact while a MUMPS factor from an earlier
            # tangent is reused, so a full-dam factorisation is not repeated every iteration.
            linear_options.update({
                "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                # Extra MUMPS workspace (percent) for fill-in from near-singular cracked zones.
                "mat_mumps_icntl_14": 80,
                # Out-of-core factors on disk; a full-dam factor can exceed RAM.
                "mat_mumps_icntl_22": 1,
            })
            use_mumps_out_of_core_directory()
        else:
            linear_options["pc_type"] = "gamg"
        self.problem = NonlinearProblem(
            residual, displacement, J=tangent, bcs=bcs, petsc_options_prefix="experimental_amor_",
            petsc_options={
                "snes_type": "newtonls", "snes_linesearch_type": "bt",
                "snes_rtol": relative_tolerance, "snes_atol": absolute_tolerance,
                "snes_stol": 0.0, "snes_max_it": maximum_iterations,
                "snes_lag_preconditioner": -2,
                "snes_lag_preconditioner_persists": True,
                # Checked in solve(): a raised SNES error stops later monitor callbacks,
                # which would silence progress and preconditioner refreshes on continuation.
                "snes_error_if_not_converged": False,
                "ksp_error_if_not_converged": True, **linear_options,
            },
        )
        self.problem.A.setNearNullSpace(rigid_body_near_nullspace(space))
        self.problem.solver.setMonitor(self._monitor)
        if interface_springs is not None:
            self.problem.A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            if interface_springs.matrix is None:
                interface_springs.build_matrix(self.problem.A)
        if self.spectral_split is not None or interface_springs is not None:
            self.problem.solver.setFunction(self._assemble_spectral_residual, self.problem.b)
            self.problem.solver.setJacobian(
                self._assemble_spectral_jacobian, self.problem.A, self.problem.P_mat,
            )

    def _update_spectral(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        fem.petsc.assign(x, self.displacement)
        if self.spectral_split is not None:
            self.spectral_split.update()

    def _assemble_spectral_residual(self, snes, x, b):
        self._update_spectral(x)
        fem.petsc.assemble_residual(
            snes, x, b, self.displacement, self.problem.F, self.problem.J, self.bcs,
        )
        if self.interface_springs is not None:
            work = self.interface_springs.matrix.createVecLeft()
            try:
                self.interface_springs.matrix.mult(x, work)
                b.axpy(1., work)
            finally:
                work.destroy()

    def _assemble_spectral_jacobian(self, snes, x, matrix, preconditioner):
        self._update_spectral(x)
        fem.petsc.assemble_jacobian(
            snes, x, matrix, preconditioner, self.displacement,
            self.problem.J, self.problem.preconditioner, self.bcs,
        )
        if self.interface_springs is not None:
            self.interface_springs.add_to(matrix)
            if preconditioner is not None and preconditioner.handle != matrix.handle:
                self.interface_springs.add_to(preconditioner)

    def _monitor(self, snes, iteration, norm):
        self.norms.append(float(norm))
        if iteration == 0:
            self.round_initial_norm = float(norm)
            if self.residual_limit is None:
                # Displacement-driven (zero-load) solves converge relative to this residual.
                self.residual_limit = max(self.atol, self.rtol * float(norm))
        if self.newton_progress is not None and iteration > 0:
            self.newton_progress(
                self.completed_newton_iterations + iteration, float(norm), self.residual_limit,
            )
        # A further Newton correction may switch compression branches.
        rtol, atol, _, _ = snes.getTolerances()
        unconverged = iteration > 0 and norm > max(atol, rtol * self.round_initial_norm)
        if self.direct_solver:
            # A lagged MUMPS factor stays a near-exact preconditioner unless GMRES struggles.
            unconverged = unconverged and (
                snes.getKSP().getIterationNumber() > DIRECT_REFACTOR_LINEAR_ITERATIONS
            )
        if unconverged:
            self._set_preconditioner_lag(rebuild=True)
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

    def _restore(self, values):
        # A failed solve leaves the caller's initial guess unchanged, as before.
        self.displacement.x.array[:] = values
        self.displacement.x.scatter_forward()
        if self.spectral_split is not None:
            self.spectral_split.update()

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
        self.residual_limit = max(self.atol, self.rtol * load_norm) if load_norm else None
        nonlinear_iterations = linear_iterations = continuations = 0
        initial_guess = self.displacement.x.array.copy()
        while True:
            round_start = len(self.norms)
            self.completed_newton_iterations = nonlinear_iterations
            linear_error = None
            try:
                problem.solve()
            except PETSc.Error as error:
                linear_error = error
            snes_reason = int(problem.solver.getConvergedReason())
            ksp_reason = int(problem.solver.getKSP().getConvergedReason())
            nonlinear_iterations += int(problem.solver.getIterationNumber())
            linear_iterations += int(problem.solver.getLinearSolveIterations())
            if linear_error is None and snes_reason > 0:
                break
            message = (
                "Experimental compression-preserving mechanics failed: "
                f"SNES reason={snes_reason}, KSP reason={ksp_reason}, "
                f"Newton continuations={continuations}, nonlinear residuals={self.norms}"
            )
            round_norms = self.norms[round_start:]
            # Early damped Newton steps can reduce the residual very little (as in the
            # full dam); the continuation cap, not a reduction ratio, bounds wasted work.
            progressing = (
                len(round_norms) > 1 and all(np.isfinite(round_norms))
                and round_norms[-1] < round_norms[0]
            )
            if linear_error is not None or snes_reason != SNES_DIVERGED_MAX_IT or ksp_reason <= 0:
                self._restore(initial_guess)
                raise RuntimeError(message) from linear_error
            if not progressing or continuations >= self.maximum_continuations:
                self._restore(initial_guess)
                raise NewtonIterationLimit(message)
            # Newton was still converging: DOLFINx has copied its last accepted iterate
            # into the displacement, so resume from it. The convergence test is unchanged.
            continuations += 1
            if not load_norm:
                # Keep the zero-load test relative to the first round's initial residual.
                problem.solver.setTolerances(rtol=0.0, atol=max(self.atol, self.rtol * self.norms[0]))
            self._set_preconditioner_lag(rebuild=True)
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
        report = {
            "formulation": (
                "experimental principal-strain spectral split" if self.spectral_split is not None
                else "experimental compression-preserving Amor volumetric/deviatoric"
            ),
            "compression_contact": False, "engineering_validated": False,
            "nonlinear_reason": reason, "nonlinear_iterations": nonlinear_iterations,
            "newton_continuations": continuations,
            "linear_iterations": linear_iterations,
            "linear_solver": "GMRES/MUMPS LU" if self.direct_solver else "GMRES/GAMG",
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
