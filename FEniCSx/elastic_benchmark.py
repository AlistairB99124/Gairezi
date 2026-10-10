"""Exact uniaxial elastic patch test on a one-metre cube; not a dam solve."""
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc


def solve(scenario: dict, material: dict, output: Path) -> dict:
    young = float(material["youngs_modulus"])
    poisson = float(material["poisson_ratio"])
    if not np.isfinite(young) or young <= 0 or not np.isfinite(poisson) or not -1 < poisson < 0.5:
        raise ValueError("Elastic constants require E > 0 and -1 < Poisson ratio < 0.5")
    divisions = scenario["mesh_divisions"]
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, divisions, divisions, divisions)
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    exact = fem.Function(space)
    strain = scenario["axial_strain"]
    exact.interpolate(lambda x: np.vstack((strain * x[0], -poisson * strain * x[1], -poisson * strain * x[2])))
    facets = mesh.locate_entities_boundary(
        domain, 2, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    dofs = fem.locate_dofs_topological(space, 2, facets)
    boundary = fem.dirichletbc(exact, dofs)
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))

    def stress(displacement):
        epsilon = ufl.sym(ufl.grad(displacement))
        return 2 * mu * epsilon + lam * ufl.tr(epsilon) * ufl.Identity(3)

    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    body = fem.Constant(domain, np.zeros(3, dtype=PETSc.ScalarType))
    problem = LinearProblem(
        ufl.inner(stress(trial), ufl.sym(ufl.grad(test))) * ufl.dx,
        ufl.inner(body, test) * ufl.dx,
        bcs=[boundary],
        petsc_options_prefix=f"{scenario['id']}_",
        petsc_options={
            "ksp_type": "cg", "pc_type": "jacobi",
            "ksp_rtol": scenario["solver"]["relative_tolerance"],
            "ksp_max_it": scenario["solver"]["maximum_iterations"],
            "ksp_error_if_not_converged": True,
        },
    )
    displacement = problem.solve()
    displacement.x.scatter_forward()
    reason = problem.solver.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"Elastic benchmark failed to converge: PETSc reason {reason}")
    error = displacement - exact
    squared_error = fem.assemble_scalar(fem.form(ufl.inner(error, error) * ufl.dx))
    squared_exact = fem.assemble_scalar(fem.form(ufl.inner(exact, exact) * ufl.dx))
    relative_error = float(np.sqrt(squared_error / squared_exact))
    energy = float(fem.assemble_scalar(fem.form(
        0.5 * ufl.inner(stress(displacement), ufl.sym(ufl.grad(displacement))) * ufl.dx
    )))
    expected_energy = 0.5 * young * strain**2
    relative_energy_error = abs(energy - expected_energy) / expected_energy
    threshold = scenario["maximum_relative_l2_error"]
    if not np.isfinite(relative_error) or relative_error > threshold or relative_energy_error > threshold:
        raise RuntimeError(
            f"Patch test failed: displacement error={relative_error}, "
            f"energy error={relative_energy_error}, limit={threshold}"
        )
    displacement.name = "displacement_m"
    tensor_space = fem.functionspace(domain, ("DG", 0, (3, 3)))
    sigma = fem.Function(tensor_space)
    sigma.name = "stress_pa"
    sigma.interpolate(fem.Expression(stress(displacement), tensor_space.element.interpolation_points))
    with io.VTKFile(domain.comm, output / "displacement.pvd", "w") as writer:
        writer.write_function(displacement)
    with io.VTKFile(domain.comm, output / "stress.pvd", "w") as writer:
        writer.write_function(sigma)
    return {
        "model": "one-metre cube uniaxial elastic patch test, not the dam",
        "relative_l2_displacement_error": relative_error,
        "relative_energy_error": relative_energy_error,
        "strain_energy_j": energy,
        "expected_strain_energy_j": expected_energy,
        "expected_axial_stress_pa": young * strain,
        "petsc_convergence_reason": reason,
        "iterations": problem.solver.getIterationNumber(),
        "residual_norm": problem.solver.getResidualNorm(),
    }
