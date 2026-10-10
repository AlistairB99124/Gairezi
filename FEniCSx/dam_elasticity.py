"""Port the current Elmer SIF: isotropic elasticity with grounded face springs."""
from pathlib import Path

from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import ufl


def solve(imported, structure, options: dict, output: Path):
    domain = imported.domain
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    young = structure.material.youngs_modulus_pa
    poisson = structure.material.poissons_ratio
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))

    def stress(u):
        strain = ufl.sym(ufl.grad(u))
        return 2 * mu * strain + lam * ufl.tr(strain) * ufl.Identity(3)

    ds = ufl.Measure("ds", domain=domain, subdomain_data=imported.facet_tags)
    dx = ufl.Measure("dx", domain=domain)
    a = ufl.inner(stress(trial), ufl.sym(ufl.grad(test))) * dx
    springs = {}
    for tag, support in ((1, structure.foundation_support), (8, structure.plinth_support)):
        coefficients = (support.spring_x_n_per_m3, support.spring_y_n_per_m3, support.spring_z_n_per_m3)
        springs[tag] = coefficients
        for component, stiffness in enumerate(coefficients):
            a += stiffness * trial[component] * test[component] * ds(tag)
    loads = structure.loads
    gravity = ufl.as_vector((0.0, 0.0, structure.material.density_kg_m3 * loads.gravity_z_m_s2))
    z = ufl.SpatialCoordinate(domain)[2]
    normal = ufl.FacetNormal(domain)
    pressure_gradient = loads.water_density_kg_m3 * abs(loads.gravity_z_m_s2)
    water_gradient = loads.water_density_kg_m3 * abs(loads.gravity_z_m_s2)
    tailwater_elevation = float(imported.points[:, 2].min()) + loads.tailwater_head_m
    pressures = {
        2: pressure_gradient * ufl.max_value(loads.overflow_head_m - z, 0),
        3: water_gradient * ufl.max_value(tailwater_elevation - z, 0),
        4: water_gradient * loads.overflow_head_m,
    }
    L = ufl.inner(gravity, test) * dx
    for tag, pressure in pressures.items():
        L += ufl.inner(-pressure * normal, test) * ds(tag)
    problem = LinearProblem(
        a, L, petsc_options_prefix="dam_elastic_",
        petsc_options={
            "ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": options["relative_tolerance"],
            "ksp_max_it": options["maximum_iterations"], "ksp_error_if_not_converged": True,
        },
    )
    # AMG needs the six rigid-body modes of a 3-D elastic body.
    from dolfinx import la
    from petsc4py import PETSc
    coordinates = space.tabulate_dof_coordinates()
    modes = [la.vector(space.dofmap.index_map, bs=3) for _ in range(6)]
    for mode in modes:
        mode.array[:] = 0
    for component in range(3):
        modes[component].array[component::3] = 1
    x, y, z_coord = coordinates.T
    modes[3].array[0::3], modes[3].array[1::3] = -y, x
    modes[4].array[0::3], modes[4].array[2::3] = z_coord, -x
    modes[5].array[1::3], modes[5].array[2::3] = -z_coord, y
    la.orthonormalize(modes)
    vectors = [PETSc.Vec().createWithArray(mode.array, comm=domain.comm) for mode in modes]
    nullspace = PETSc.NullSpace().create(vectors=vectors, comm=domain.comm)
    problem.A.setNearNullSpace(nullspace)
    displacement = problem.solve()
    displacement.x.scatter_forward()
    reason = problem.solver.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"Elastostatic solve failed: PETSc reason {reason}")

    def integral(expression):
        return float(fem.assemble_scalar(fem.form(expression)))

    applied = np.asarray([
        integral(gravity[i] * dx) + sum(integral(-p * normal[i] * ds(tag)) for tag, p in pressures.items())
        for i in range(3)
    ])
    reactions = {
        str(tag): [integral(-coefficients[i] * displacement[i] * ds(tag)) for i in range(3)]
        for tag, coefficients in springs.items()
    }
    imbalance = applied + np.asarray(list(reactions.values())).sum(axis=0)
    relative_balance = float(np.linalg.norm(imbalance) / np.linalg.norm(applied))
    displacement.name = "displacement_m"
    with io.VTKFile(domain.comm, output / "displacement.pvd", "w") as writer:
        writer.write_function(displacement)
    tensor_space = fem.functionspace(domain, ("DG", 0, (3, 3)))
    sigma = fem.Function(tensor_space)
    sigma.name = "stress_pa"
    sigma.interpolate(fem.Expression(stress(displacement), tensor_space.element.interpolation_points))
    with io.VTKFile(domain.comm, output / "stress.pvd", "w") as writer:
        writer.write_function(sigma)
    local_source = imported.domain.geometry.input_global_indices
    source_displacement = np.empty((len(imported.points), 3))
    geometry_nodes = domain.geometry.dofmaps[0]
    field_nodes = space.dofmap.list.reshape(-1, 4)
    source_displacement[local_source[geometry_nodes]] = displacement.x.array.reshape(-1, 3)[field_nodes]
    np.savez(output / "nodal_displacement.npz", points=imported.points, displacement=source_displacement)
    report = {
        "petsc_convergence_reason": reason, "iterations": problem.solver.getIterationNumber(),
        "residual_norm": problem.solver.getResidualNorm(),
        "applied_force_n": applied.tolist(), "ground_spring_reactions_n": reactions,
        "force_imbalance_n": imbalance.tolist(), "relative_force_balance_error": relative_balance,
        "maximum_displacement_m": float(np.linalg.norm(source_displacement, axis=1).max()),
        "wall_plinth_model": "wall-side grounded spring, exactly as current Elmer SIF; not interbody coupling",
    }
    return source_displacement, sigma, report
