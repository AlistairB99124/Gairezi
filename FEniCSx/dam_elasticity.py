"""Elmer-compatible elasticity for canonical three-body and legacy source meshes."""
from pathlib import Path
import json

from dolfinx import fem, io
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc
if __package__:
    from .three_body_model import is_three_body
else:
    from three_body_model import is_three_body


def ground_supports(imported, structure):
    """Return legacy foundation springs; the canonical mesh uses explicit support regions."""
    support = structure.foundation_support
    return {1: (support.spring_x_n_per_m3, support.spring_y_n_per_m3, support.spring_z_n_per_m3)}, (
        "single_monolithic_body"
    )


def water_load_parameters(imported, structure):
    """Existing hydrostatic heads/constant top pressure, without reinterpretation."""
    domain = imported.domain
    loads = structure.loads
    water_gradient = loads.water_density_kg_m3 * abs(loads.gravity_z_m_s2)
    tailwater_elevation = (
        structure.concrete_minimum_z if is_three_body(structure)
        else float(domain.comm.allreduce(domain.geometry.x[:, 2].min(), op=MPI.MIN))
    ) + loads.tailwater_head_m
    return water_gradient, {2: loads.upstream_surface_z_m, 3: tailwater_elevation}, {
        4: water_gradient * max(loads.upstream_surface_z_m, 0.0),
    }


def external_load(imported, structure, density, test, ds, *, include_water=True):
    """Canonical unchanged gravity and exterior water loads, with audit terms."""
    domain = imported.domain
    gravity = ufl.as_vector((0.0, 0.0, density * structure.loads.gravity_z_m_s2))
    z = ufl.SpatialCoordinate(domain)[2]
    normal = ufl.FacetNormal(domain)
    water_gradient, elevations, constants = water_load_parameters(imported, structure)
    pressures = {
        tag: water_gradient * ufl.max_value(elevation - z, 0)
        for tag, elevation in elevations.items()
    }
    pressures.update(constants)
    load = ufl.inner(gravity, test) * ufl.dx(domain=domain)
    if include_water:
        for tag, pressure in pressures.items():
            load += ufl.inner(-pressure * normal, test) * ds(tag)
    return load, gravity, pressures


def solve(imported, structure, options: dict, output: Path, *, save_source_displacement: bool = True,
          reference_stress=None):
    domain = imported.domain
    three_body = is_three_body(structure)
    if three_body and domain.comm.rank == 0:
        print("Three-body elastic assembling system", flush=True)
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    young = structure.material.youngs_modulus_pa
    poisson = structure.material.poissons_ratio
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    coefficients = None
    interfaces = None
    bcs = []
    if three_body:
        from three_body_model import materials, require_support_tags, support_bcs
        from interface_springs import InterfaceSprings

        coefficients = materials(imported, structure)
        lam, mu = coefficients["lam"], coefficients["mu"]
        interfaces = InterfaceSprings(imported, structure, space)
        require_support_tags(imported, structure)
        bcs = support_bcs(imported, space, structure)

    def stress(u):
        strain = ufl.sym(ufl.grad(u))
        return 2 * mu * strain + lam * ufl.tr(strain) * ufl.Identity(3)

    ds = ufl.Measure("ds", domain=domain, subdomain_data=imported.facet_tags)
    dx = ufl.Measure("dx", domain=domain)
    a = ufl.inner(stress(trial), ufl.sym(ufl.grad(test))) * dx
    springs, connection = (
        ({}, getattr(structure, "wall_connection", "single_monolithic_body"))
        if three_body else ground_supports(imported, structure)
    )
    for tag, coefficients in springs.items():
        for component, stiffness in enumerate(coefficients):
            a += stiffness * trial[component] * test[component] * ds(tag)
    if three_body:
        from three_body_model import support_form

        rock_rollers = support_form(trial, test, ds, structure)
        if rock_rollers is not None:
            a += rock_rollers
    density = coefficients["density"] if three_body else structure.material.density_kg_m3
    normal = ufl.FacetNormal(domain)
    L, gravity, pressures = external_load(imported, structure, density, test, ds)
    from interface_springs import SpringLinearProblem
    solver_options = {
        "ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 0.1 * options["relative_tolerance"],
        "ksp_norm_type": "unpreconditioned",
        "ksp_pc_side": "left",
        "ksp_max_it": options["maximum_iterations"], "ksp_error_if_not_converged": True,
    }
    if three_body:
        solver_options.update(ksp_type="gmres", ksp_gmres_restart=100, ksp_pc_side="right")
    problem = SpringLinearProblem(
        a, L, bcs=bcs, interface_springs=interfaces, petsc_options_prefix="dam_elastic_",
        petsc_options=solver_options, eliminate_interface_bcs=domain.comm.size == 1,
    )
    from elasticity_solver import rigid_body_near_nullspace

    nullspace = rigid_body_near_nullspace(space)
    problem.A.setNearNullSpace(nullspace)
    if three_body:
        def monitor(ksp, iteration, norm):
            if domain.comm.rank == 0 and iteration % 50 == 0:
                print(f"Three-body elastic GMRES iteration {iteration}: residual={norm:.6g}", flush=True)
        problem.solver.setMonitor(monitor)
        if domain.comm.rank == 0:
            # Lets progress monitors normalise the residuals above by this solve's own tolerance.
            print(f"Three-body elastic GMRES tolerance: rtol={solver_options['ksp_rtol']:.6g}", flush=True)
            print("Three-body elastic solving displacements", flush=True)
    try:
        displacement = problem.solve()
    except PETSc.Error as error:
        failure = {
            "method": problem.solver.getType(), "preconditioner": problem.solver.getPC().getType(),
            "convergence_reason": int(problem.solver.getConvergedReason()),
            "iterations": int(problem.solver.getIterationNumber()),
            "residual_norm": float(problem.solver.getResidualNorm()),
            "error": str(error),
        }
        if domain.comm.rank == 0:
            (output / "linear_solver_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise RuntimeError(f"Elastostatic linear solve failed: {failure}") from error
    displacement.x.scatter_forward()
    reason = problem.solver.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"Elastostatic solve failed: PETSc reason {reason}")
    residual = problem.b.duplicate()
    try:
        problem.A.mult(problem.x, residual)
        residual.axpy(-1, problem.b)
        rhs_norm = float(problem.b.norm())
        true_residual = float(residual.norm())
        relative_residual = true_residual / rhs_norm if rhs_norm else true_residual
    finally:
        residual.destroy()
    if not np.isfinite(relative_residual) or relative_residual > options["relative_tolerance"]:
        raise RuntimeError(f"Elastostatic true residual {relative_residual} exceeds tolerance")
    solver_report = {
        "petsc_convergence_reason": int(reason),
        "iterations": int(problem.solver.getIterationNumber()),
        "linear_solver": {"method": problem.solver.getType(), "preconditioner": problem.solver.getPC().getType()},
        "residual_norm": float(problem.solver.getResidualNorm()),
    }
    if three_body and domain.comm.rank == 0:
        (output / "linear_solver_report.json").write_text(
            json.dumps({**solver_report, "relative_true_residual": relative_residual},
                       indent=2, allow_nan=False) + "\n",
        )
        print("Three-body elastic mechanics converged; assembling reactions", flush=True)
    # Release the stiffness matrix and AMG hierarchy before reaction/stress assembly.
    del problem
    nullspace.destroy()

    def integral(expression):
        return float(domain.comm.allreduce(fem.assemble_scalar(fem.form(expression)), op=MPI.SUM))

    applied = np.asarray([
        integral(gravity[i] * dx) + sum(integral(-p * normal[i] * ds(tag)) for tag, p in pressures.items())
        for i in range(3)
    ])
    reactions = {
        str(tag): [integral(-coefficients[i] * displacement[i] * ds(tag)) for i in range(3)]
        for tag, coefficients in springs.items()
    }
    if three_body:
        from three_body_model import support_reactions
        reactions = support_reactions(
            imported, space, structure, displacement, ufl.action(a, displacement) - L, ds,
            interface_springs=interfaces)
    imbalance = applied + np.asarray(list(reactions.values())).sum(axis=0)
    relative_balance = float(np.linalg.norm(imbalance) / np.linalg.norm(applied))
    if three_body and domain.comm.rank == 0:
        print(f"Three-body elastic relative force imbalance={relative_balance:.6g}; "
              "exporting displacement and raw stress", flush=True)
    displacement.name = "displacement_m"
    with io.VTKFile(domain.comm, output / "displacement.pvd", "w") as writer:
        writer.write_function(displacement)
    tensor_space = fem.functionspace(domain, ("DG", 0, (3, 3)))
    sigma = fem.Function(tensor_space)
    sigma.name = "stress_pa"
    sigma.interpolate(fem.Expression(stress(displacement), tensor_space.element.interpolation_points))
    with io.VTKFile(domain.comm, output / "stress.pvd", "w") as writer:
        writer.write_function([sigma, coefficients["body_id"]] if three_body else sigma)
    source_displacement = displacement.x.array.reshape(-1, 3)
    if save_source_displacement:
        if domain.comm.size != 1:
            raise ValueError("Source-order displacement archive requires a serial mesh")
        local_source = imported.domain.geometry.input_global_indices
        source_displacement = np.empty((len(imported.points), 3))
        geometry_nodes = domain.geometry.dofmaps[0]
        field_nodes = space.dofmap.list.reshape(-1, 4)
        source_displacement[local_source[geometry_nodes]] = displacement.x.array.reshape(-1, 3)[field_nodes]
        np.savez(output / "nodal_displacement.npz", points=imported.points, displacement=source_displacement)
    report = {
        **solver_report,
        "relative_true_residual": relative_residual,
        "applied_force_n": applied.tolist(), "ground_spring_reactions_n": reactions,
        "force_imbalance_n": imbalance.tolist(), "relative_force_balance_error": relative_balance,
        "maximum_displacement_m": float(domain.comm.allreduce(
            np.linalg.norm(source_displacement, axis=1).max(), op=MPI.MAX,
        )),
        "mesh_nodes": int(space.dofmap.index_map.size_global),
        "mesh_cells": int(domain.topology.index_map(3).size_global),
        "wall_connection": connection,
    }
    if three_body:
        from three_body_model import rock_support, connection_report

        support = rock_support(structure)
        report.update(
            model_kind="three_body", interface_springs=interfaces.report,
            wall_connection=connection,
            wall_joints=[{"name": joint["name"], "z_m": joint["z_m"]} for joint in getattr(structure, "wall_joints", [])],
            **connection_report(structure),
            rock_outer_support=support,
            source_mesh=structure.three_body_source,
            stress_scope="wall; rock and optional foundation reported separately",
        )
    from stress_recovery import recover_stress

    owned_cells = domain.topology.index_map(3).size_local
    raw_principal = np.linalg.eigvalsh(
        sigma.x.array.reshape(-1, 3, 3)[tensor_space.dofmap.list[:owned_cells, 0]],
    )
    if three_body:
        concrete = imported.cell_tags.find(1)
        raw_principal = raw_principal[concrete[concrete < owned_cells]]
    report["raw_element_stress"] = {
        "maximum_principal_tensile_stress_pa": float(domain.comm.allreduce(
            raw_principal[:, -1].max() if len(raw_principal) else -np.inf, op=MPI.MAX)),
        "maximum_principal_compressive_stress_pa": float(domain.comm.allreduce(
            raw_principal[:, 0].min() if len(raw_principal) else np.inf, op=MPI.MIN)),
    }
    if three_body and domain.comm.rank == 0:
        print("Three-body elastic recovering nodal stress", flush=True)
    report.update(recover_stress(domain, stress(displacement), output,
                                 cell_tags=imported.cell_tags if three_body else None,
                                 reference_stress=reference_stress))
    return source_displacement, sigma, report
