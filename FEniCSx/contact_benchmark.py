"""Two elastic bars with exact unilateral end contact; a reduced 1-D benchmark."""
from pathlib import Path
import json
import warnings

import basix.ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import ufl


def constrained_step(stiffness, fixed, prescribed, gap_operator, initial_gap, active, options):
    free = np.setdiff1d(np.arange(stiffness.shape[0]), fixed)
    scale = float(np.max(stiffness.diagonal()))
    reduced = stiffness[free][:, free] / scale
    rhs = -stiffness[free][:, fixed] @ prescribed / scale
    constraint = gap_operator[free]
    constant_gap = initial_gap + gap_operator[fixed] @ prescribed
    for iteration in range(1, options["maximum_active_set_iterations"] + 1):
        if active:
            column = sparse.csr_matrix(constraint[:, None])
            matrix = sparse.bmat([[reduced, -column], [column.T, None]], format="csr")
            load = np.concatenate((rhs, [-constant_gap]))
        else:
            matrix, load = reduced, rhs
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            solution = spsolve(matrix, load)
        if not np.all(np.isfinite(solution)):
            raise RuntimeError("Contact linear solve returned non-finite values")
        displacement = np.zeros(stiffness.shape[0])
        displacement[fixed] = prescribed
        displacement[free] = solution[:len(free)]
        contact_force = float(solution[-1] * scale) if active else 0.0
        gap = float(initial_gap + gap_operator @ displacement)
        if not active and gap < -options["gap_tolerance_m"]:
            active = True
            continue
        if active and contact_force < 0:
            active = False
            continue
        residual = stiffness @ displacement - gap_operator * contact_force
        return displacement, contact_force, gap, active, iteration, residual, free
    raise RuntimeError("Contact active set did not converge within maximum_active_set_iterations")


def solve(scenario: dict, material: dict, output: Path) -> dict:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("The two-bar contact benchmark supports one MPI process")
    young = float(material["youngs_modulus"])
    if not np.isfinite(young) or young <= 0:
        raise ValueError("Contact benchmark requires a finite positive Young's modulus")
    left_length, right_length = scenario["bar_lengths_m"]
    area = scenario["cross_section_m2"]
    initial_gap = scenario["initial_gap_m"]
    options = scenario["verification"]
    all_steps = []
    for divisions in scenario["mesh_divisions"]:
        left = np.linspace(0, left_length, divisions + 1)
        right = np.linspace(left_length + initial_gap, left_length + initial_gap + right_length, divisions + 1)
        coordinates = np.concatenate((left, right))[:, None]
        ids = np.arange(divisions, dtype=np.int64)
        cells = np.vstack((np.column_stack((ids, ids + 1)),
                           np.column_stack((ids + divisions + 1, ids + divisions + 2))))
        domain = mesh.create_mesh(
            MPI.COMM_WORLD, cells, ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(1,))),
            coordinates
        )
        space = fem.functionspace(domain, ("Lagrange", 1))
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        assembled = assemble_matrix(fem.form(young * area * ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx))
        assembled.assemble()
        try:
            offsets, columns, values = assembled.getValuesCSR()
            stiffness = sparse.csr_matrix((values.copy(), columns.copy(), offsets.copy()), shape=assembled.getSize())
        finally:
            assembled.destroy()
        dof_x = space.tabulate_dof_coordinates()[:, 0]
        # Resolve endpoints by source node identity, including when the initial gap is zero.
        field_nodes = space.dofmap.list.reshape(-1, 2)
        source_nodes = domain.geometry.input_global_indices[domain.geometry.dofmaps[0]]
        source_to_dof = np.empty(len(coordinates), dtype=np.int32)
        source_to_dof[source_nodes] = field_nodes
        fixed = source_to_dof[[0, len(coordinates) - 1]]
        gap_operator = np.zeros(len(coordinates))
        gap_operator[source_to_dof[divisions]] = -1
        gap_operator[source_to_dof[divisions + 1]] = 1
        displacement_field = fem.Function(space)
        displacement_field.name = "axial_displacement_m"
        stress_space = fem.functionspace(domain, ("DG", 0))
        stress_field = fem.Function(stress_space)
        stress_field.name = "axial_stress_pa"
        cell_x = stress_space.tabulate_dof_coordinates()[:, 0]
        active = False
        folder = output / f"mesh_{divisions}"
        folder.mkdir()
        with io.VTKFile(domain.comm, folder / "displacement.pvd", "w") as displacement_writer, io.VTKFile(
            domain.comm, folder / "stress.pvd", "w"
        ) as stress_writer:
            for step, closure in enumerate(scenario["outer_closures_m"]):
                displacement, force, gap, active, iterations, residual, free = constrained_step(
                    stiffness, fixed, np.asarray([0.0, -closure]), gap_operator, initial_gap, active, options
                )
                pressure = force / area
                expected_pressure = max(closure - initial_gap, 0) * young / (left_length + right_length)
                expected_gap = max(initial_gap - closure, 0)
                expected_displacement = np.empty_like(displacement)
                # Coincident crack lips must be distinguished by connectivity, not coordinates.
                left_dofs = source_to_dof[:divisions + 1]
                right_dofs = source_to_dof[divisions + 1:]
                expected_displacement[left_dofs] = -expected_pressure * dof_x[left_dofs] / young
                expected_displacement[right_dofs] = (
                    -closure + expected_pressure * (coordinates[-1, 0] - dof_x[right_dofs]) / young
                )
                displacement_field.x.array[:] = displacement
                displacement_field.x.scatter_forward()
                stress_field.interpolate(fem.Expression(
                    young * displacement_field.dx(0), stress_space.element.interpolation_points
                ))
                expected_stress = np.full(len(cell_x), -expected_pressure)
                force_scale = max(area * young * max(initial_gap, abs(closure)) / (left_length + right_length), 1.0)
                pressure_scale = force_scale / area
                pressure_error = abs(pressure - expected_pressure) / pressure_scale
                stress_error = float(np.max(np.abs(stress_field.x.array - expected_stress))) / pressure_scale
                displacement_error = float(np.max(np.abs(displacement - expected_displacement)))
                gap_error = abs(gap - expected_gap)
                equilibrium_error = float(np.max(np.abs(residual[free]))) / force_scale
                reaction_error = float(np.max(np.abs(residual[fixed] - [force, -force]))) / force_scale
                complementarity = abs(force * gap) / force_scale
                row = {
                    "mesh_divisions_per_bar": divisions, "step": step, "outer_closure_m": closure,
                    "active_contact": active, "active_set_iterations": iterations,
                    "gap_m": gap, "expected_gap_m": expected_gap,
                    "contact_pressure_pa": pressure, "expected_contact_pressure_pa": expected_pressure,
                    "contact_force_n": force, "outer_reactions_n": residual[fixed].tolist(),
                    "maximum_displacement_error_m": displacement_error, "gap_error_m": gap_error,
                    "scaled_pressure_error": pressure_error, "scaled_stress_error": stress_error,
                    "scaled_equilibrium_error": equilibrium_error, "scaled_reaction_error": reaction_error,
                    "scaled_complementarity_error_m": complementarity,
                }
                all_steps.append(row)
                failures = (
                    gap < -options["gap_tolerance_m"] or force < 0
                    or gap_error > options["gap_tolerance_m"]
                    or displacement_error > options["displacement_tolerance_m"]
                    or complementarity > options["gap_tolerance_m"]
                    or max(pressure_error, stress_error, equilibrium_error, reaction_error) > options["relative_tolerance"]
                )
                if failures:
                    (output / "contact_steps.json").write_text(json.dumps(all_steps, indent=2, allow_nan=False) + "\n")
                    raise RuntimeError(f"Two-bar contact verification failed for mesh {divisions}, step {step}: {row}")
                displacement_writer.write_function(displacement_field, float(step))
                stress_writer.write_function(stress_field, float(step))
    return {
        "model": "two-bar 1-D frictionless unilateral contact benchmark, not a dam crack solve",
        "formulation": "assembled P1 axial elasticity with a single exact contact-force Lagrange multiplier",
        "gap_convention": "initial_gap + right_lip_displacement - left_lip_displacement; positive is open",
        "pressure_convention": "positive is compression; bulk axial stress is negative in contact",
        "analytical_pressure_formula": "max(outer_closure - initial_gap, 0) * E / (L_left + L_right)",
        "youngs_modulus_pa": young, "cross_section_m2": area,
        "initial_gap_m": initial_gap, "mesh_divisions": scenario["mesh_divisions"],
        "passed_steps": len(all_steps), "steps": all_steps,
        "limitations": ["One normal contact constraint only; no 3-D surface search, friction, water, or crack growth.",
                       "This benchmark does not validate a future dam contact adapter."],
    }
