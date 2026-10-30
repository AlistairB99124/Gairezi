"""Reduced elastic bars with unilateral contact and optional pressure/Coulomb shear."""
from pathlib import Path
from contextlib import ExitStack
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


def constrained_step(stiffness, fixed, prescribed, gap_operator, initial_gap, active, options,
                     *, applied_force=None):
    load_vector = np.zeros(stiffness.shape[0]) if applied_force is None else np.asarray(applied_force)
    if load_vector.shape != (stiffness.shape[0],) or not np.all(np.isfinite(load_vector)):
        raise ValueError("Applied contact-benchmark force must be a finite vector matching the displacement space")
    free = np.setdiff1d(np.arange(stiffness.shape[0]), fixed)
    scale = float(np.max(stiffness.diagonal()))
    reduced = stiffness[free][:, free] / scale
    rhs = (load_vector[free] - stiffness[free][:, fixed] @ prescribed) / scale
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
        residual = stiffness @ displacement - load_vector - gap_operator * contact_force
        return displacement, contact_force, gap, active, iteration, residual, free
    raise RuntimeError("Contact active set did not converge within maximum_active_set_iterations")


def tangential_step(stiffness, fixed, prescribed, gap_operator, previous_slip,
                    contact_force, friction_coefficient):
    """Scalar Coulomb return for two elastic shear bars and one split interface."""
    free = np.setdiff1d(np.arange(stiffness.shape[0]), fixed)
    reduced = stiffness[free][:, free]
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        responses = spsolve(reduced, np.column_stack((
            -stiffness[free][:, fixed] @ prescribed, gap_operator[free],
        )))
    if not np.all(np.isfinite(responses)):
        raise RuntimeError("Tangential contact equilibrium returned non-finite values")
    free_gap = float(gap_operator[free] @ responses[:, 0] + gap_operator[fixed] @ prescribed)
    compliance = float(gap_operator[free] @ responses[:, 1])
    if compliance <= 0 or not np.isfinite(compliance):
        raise RuntimeError("Tangential interface compliance must be finite and positive")
    trial_force = (previous_slip - free_gap) / compliance
    limit = friction_coefficient * contact_force
    sliding = abs(trial_force) > limit
    force = float(np.clip(trial_force, -limit, limit))
    displacement = np.zeros(stiffness.shape[0])
    displacement[fixed] = prescribed
    displacement[free] = responses[:, 0] + responses[:, 1] * force
    jump = float(gap_operator @ displacement)
    slip = jump if sliding else previous_slip
    dissipation = -force * (slip - previous_slip)
    residual = stiffness @ displacement - gap_operator * force
    return displacement, force, jump, slip, sliding, dissipation, residual, free


def solve(scenario: dict, material: dict, output: Path, *, face_pressures_pa=None,
          tangential_shifts_m=None, friction_coefficient=None, shear_modulus_pa=None) -> dict:
    """Verify axial contact with optional prescribed pressure on both crack lips.

    Pressure is an externally specified traction, not a pressure transported
    through closed contact. It exerts equal and opposite opening forces.
    This test does not model access/wetting or compute pressure from flow.
    Optional scalar tangential loading requires explicit friction and shear
    modulus. Coulomb strength uses solid contact force, excluding water load.
    """
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("The two-bar contact benchmark supports one MPI process")
    young = float(material["youngs_modulus"])
    if not np.isfinite(young) or young <= 0:
        raise ValueError("Contact benchmark requires a finite positive Young's modulus")
    left_length, right_length = scenario["bar_lengths_m"]
    area = scenario["cross_section_m2"]
    initial_gap = scenario["initial_gap_m"]
    options = scenario["verification"]
    pressures = np.zeros(len(scenario["outer_closures_m"]))
    if face_pressures_pa is not None:
        raw = np.asarray(face_pressures_pa)
        if (raw.dtype.kind not in "fiu" or raw.shape != pressures.shape
                or not np.all(np.isfinite(raw)) or np.any(raw < 0)):
            raise ValueError("Face pressures must be a finite nonnegative history matching outer closures")
        pressures = raw.astype(float)
    friction = tangential_shifts_m is not None
    if friction:
        shifts = np.asarray(tangential_shifts_m)
        if (shifts.dtype.kind not in "fiu" or shifts.shape != pressures.shape
                or not np.all(np.isfinite(shifts))):
            raise ValueError("Tangential shifts must be a finite history matching outer closures")
        for name, value, positive in (("friction_coefficient", friction_coefficient, False),
                                      ("shear_modulus_pa", shear_modulus_pa, True)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value < 0 or (positive and value == 0)):
                raise ValueError(f"{name} must be explicitly supplied, finite and {'positive' if positive else 'nonnegative'}")
    elif friction_coefficient is not None or shear_modulus_pa is not None:
        raise ValueError("Tangential shifts are required when supplying friction properties")
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
        if friction:
            shear_stiffness = stiffness * (shear_modulus_pa / young)
            tangential_field = fem.Function(space, name="tangential_displacement_m")
            shear_field = fem.Function(stress_space, name="shear_stress_pa")
            plastic_slip = 0.
        cell_x = stress_space.tabulate_dof_coordinates()[:, 0]
        active = False
        folder = output / f"mesh_{divisions}"
        folder.mkdir()
        with ExitStack() as writers:
            displacement_writer = writers.enter_context(io.VTKFile(domain.comm, folder / "displacement.pvd", "w"))
            stress_writer = writers.enter_context(io.VTKFile(domain.comm, folder / "stress.pvd", "w"))
            if friction:
                tangential_writer = writers.enter_context(
                    io.VTKFile(domain.comm, folder / "tangential_displacement.pvd", "w"))
                shear_writer = writers.enter_context(io.VTKFile(domain.comm, folder / "shear_stress.pvd", "w"))
            for step, closure in enumerate(scenario["outer_closures_m"]):
                water_pressure = float(pressures[step])
                water_load = gap_operator * (water_pressure * area)
                displacement, force, gap, active, iterations, residual, free = constrained_step(
                    stiffness, fixed, np.asarray([0.0, -closure]), gap_operator, initial_gap, active, options,
                    applied_force=water_load,
                )
                pressure = force / area
                expected_pressure = max(
                    (closure - initial_gap) * young / (left_length + right_length) - water_pressure, 0,
                )
                expected_gap = max(
                    initial_gap - closure + water_pressure * (left_length + right_length) / young, 0,
                )
                total_pressure = expected_pressure + water_pressure
                expected_displacement = np.empty_like(displacement)
                # Coincident crack lips must be distinguished by connectivity, not coordinates.
                left_dofs = source_to_dof[:divisions + 1]
                right_dofs = source_to_dof[divisions + 1:]
                expected_displacement[left_dofs] = -total_pressure * dof_x[left_dofs] / young
                expected_displacement[right_dofs] = (
                    -closure + total_pressure * (coordinates[-1, 0] - dof_x[right_dofs]) / young
                )
                displacement_field.x.array[:] = displacement
                displacement_field.x.scatter_forward()
                stress_field.interpolate(fem.Expression(
                    young * displacement_field.dx(0), stress_space.element.interpolation_points
                ))
                expected_stress = np.full(len(cell_x), -total_pressure)
                force_scale = max(area * young * max(initial_gap, abs(closure)) / (left_length + right_length),
                                  area * water_pressure, 1.0)
                pressure_scale = force_scale / area
                pressure_error = abs(pressure - expected_pressure) / pressure_scale
                stress_error = float(np.max(np.abs(stress_field.x.array - expected_stress))) / pressure_scale
                displacement_error = float(np.max(np.abs(displacement - expected_displacement)))
                gap_error = abs(gap - expected_gap)
                equilibrium_error = float(np.max(np.abs(residual[free]))) / force_scale
                total_force = force + water_pressure * area
                reaction_error = float(np.max(np.abs(residual[fixed] - [total_force, -total_force]))) / force_scale
                complementarity = abs(force * gap) / force_scale
                row = {
                    "mesh_divisions_per_bar": divisions, "step": step, "outer_closure_m": closure,
                    "active_contact": active, "active_set_iterations": iterations,
                    "gap_m": gap, "expected_gap_m": expected_gap,
                    "contact_pressure_pa": pressure, "expected_contact_pressure_pa": expected_pressure,
                    "prescribed_face_pressure_pa": water_pressure,
                    "water_opening_forces_n": [-water_pressure * area, water_pressure * area],
                    "contact_force_n": force, "outer_reactions_n": residual[fixed].tolist(),
                    "maximum_displacement_error_m": displacement_error, "gap_error_m": gap_error,
                    "scaled_pressure_error": pressure_error, "scaled_stress_error": stress_error,
                    "scaled_equilibrium_error": equilibrium_error, "scaled_reaction_error": reaction_error,
                    "scaled_complementarity_error_m": complementarity,
                }
                if friction:
                    old_slip = plastic_slip
                    (tangential, shear_force, jump, plastic_slip, sliding, dissipation,
                     shear_residual, shear_free) = tangential_step(
                        shear_stiffness, fixed, np.asarray([0., shifts[step]]),
                        gap_operator, old_slip, force, friction_coefficient,
                    )
                    expected_trial = (old_slip - shifts[step]) * area * shear_modulus_pa / (left_length + right_length)
                    expected_force = float(np.clip(expected_trial, -friction_coefficient * force,
                                                   friction_coefficient * force))
                    expected_jump = shifts[step] + expected_force * (left_length + right_length) / (area * shear_modulus_pa)
                    tangential_field.x.array[:] = tangential
                    tangential_field.x.scatter_forward()
                    shear_field.interpolate(fem.Expression(
                        shear_modulus_pa * tangential_field.dx(0), stress_space.element.interpolation_points,
                    ))
                    shear_scale = max(force_scale, abs(expected_force), 1.)
                    shear_errors = [
                        float(np.max(np.abs(shear_residual[shear_free]))) / shear_scale,
                        float(np.max(np.abs(shear_residual[fixed] - [shear_force, -shear_force]))) / shear_scale,
                        abs(shear_force - expected_force) / shear_scale,
                        float(np.max(np.abs(shear_field.x.array * area + shear_force))) / shear_scale,
                    ]
                    row.update({
                        "tangential_outer_shift_m": float(shifts[step]), "sliding": sliding,
                        "tangential_force_n": shear_force, "friction_limit_n": friction_coefficient * force,
                        "tangential_jump_m": jump, "plastic_slip_m": plastic_slip,
                        "friction_dissipation_j": dissipation,
                        "scaled_shear_equilibrium_error": max(shear_errors),
                    })
                    if (not np.all(np.isfinite([jump, plastic_slip, dissipation, *shear_errors]))
                            or max(shear_errors) > options["relative_tolerance"]
                            or abs(jump - expected_jump) > options["displacement_tolerance_m"]
                            or (not sliding and abs(jump - old_slip) > options["displacement_tolerance_m"])
                            or dissipation < -shear_scale * options["displacement_tolerance_m"]):
                        raise RuntimeError(f"Tangential Coulomb verification failed for mesh {divisions}, step {step}: {row}")
                    tangential_writer.write_function(tangential_field, float(step))
                    shear_writer.write_function(shear_field, float(step))
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
        "model": "two-bar 1-D unilateral contact benchmark, not a dam crack solve",
        "formulation": "assembled P1 axial elasticity with exact normal-contact multiplier and optional scalar Coulomb shear",
        "gap_convention": "initial_gap + right_lip_displacement - left_lip_displacement; positive is open",
        "pressure_convention": "positive is compression; bulk axial stress is negative in contact",
        "analytical_pressure_formula": "max((outer_closure - initial_gap) * E / (L_left + L_right) - water_pressure, 0)",
        "youngs_modulus_pa": young, "cross_section_m2": area,
        "initial_gap_m": initial_gap, "mesh_divisions": scenario["mesh_divisions"],
        "passed_steps": len(all_steps), "steps": all_steps,
        "solver_ready": False, "engineering_validated": False,
        "friction_model": "scalar Coulomb contact" if friction else "frictionless",
        "friction_coefficient": friction_coefficient,
        "shear_modulus_pa": shear_modulus_pa,
        "hydraulic_model": "prescribed face traction only; no flow, access/wetting or mechanical feedback to pressure",
        "limitations": ["One normal constraint and optional scalar shear slip only; no 3-D surface search, mixed-mode growth, water transport or wetting.",
                       "This benchmark does not validate a future dam contact adapter."],
    }
