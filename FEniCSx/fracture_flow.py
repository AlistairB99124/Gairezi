"""Reduced saturated parallel-plate flow on an open 1-D fracture.

Per unit fracture width: beta*w*dp/dt + dw/dt + dq/ds = 0,
q = -w**3/(12*mu) * (dp/ds + rho*g*dz/ds).
Linearised fluid storage, backward Euler, continuous P1 pressure. The balance
checks verify this equation, not a nonlinear density/large-aperture mass model.
The prescribed variant supplies aperture; the compliant variant calculates it
from an explicit local linear normal compliance, not from a 3-D dam solve.
This verification component has no wetting front, cavitation, contact,
branching or propagation. Closed paths are rejected,
never assigned an artificial minimum aperture or pressurised through contact.
"""
from dataclasses import dataclass
import numbers

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem, assemble_vector
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl


@dataclass(frozen=True)
class FractureFluid:
    viscosity_pa_s: float
    compressibility_pa_inv: float
    density_kg_m3: float
    gravity_m_s2: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive; no physical default is assumed")


def solve_prescribed_fracture(
    fluid: FractureFluid, *, length_m: float, elevation_change_m: float,
    divisions: int, times_s, apertures_m, inlet_pressures_pa, outlet_pressures_pa,
    initial_pressure, relative_tolerance: float = 1e-10,
) -> dict:
    """Return checked nodal histories, boundary inflows and per-step mass balance.

    Pressure is gauge pressure; this reduced saturated verification requires
    nonnegative pressure. Values needing a suction/cavitation model fail.
    """
    return _solve_fracture(
        fluid, length_m=length_m, elevation_change_m=elevation_change_m,
        divisions=divisions, times_s=times_s, apertures_m=apertures_m,
        inlet_pressures_pa=inlet_pressures_pa, outlet_pressures_pa=outlet_pressures_pa,
        initial_pressure=initial_pressure, relative_tolerance=relative_tolerance,
        normal_compliance_m_pa=0., reference_pressure_pa=0., maximum_iterations=50,
    )


def solve_compliant_fracture(
    fluid: FractureFluid, *, normal_compliance_m_pa: float, reference_pressure_pa: float,
    reference_apertures_m, length_m: float, elevation_change_m: float,
    divisions: int, times_s, inlet_pressures_pa, outlet_pressures_pa,
    initial_pressure, relative_tolerance: float = 1e-10,
    maximum_iterations: int = 50,
) -> dict:
    """Reduced two-way pressure/opening coupling on an already-wet open path.

    Eliminate local linear normal mechanics: w = w_ref(t) + C*(p - p_ref).
    Pressure changes actual opening, fluid storage and cubic conductivity;
    opening changes feed back on pressure. C is explicit normal compliance in
    m/Pa, not an artificial aperture. This is not a contact/wetting or dam model.
    """
    if (isinstance(normal_compliance_m_pa, bool)
            or not isinstance(normal_compliance_m_pa, numbers.Real)
            or not np.isfinite(normal_compliance_m_pa) or normal_compliance_m_pa <= 0):
        raise ValueError("normal_compliance_m_pa must be finite and positive")
    if (isinstance(reference_pressure_pa, bool)
            or not isinstance(reference_pressure_pa, numbers.Real)
            or not np.isfinite(reference_pressure_pa) or reference_pressure_pa < 0):
        raise ValueError("reference_pressure_pa must be finite and nonnegative")
    if isinstance(maximum_iterations, bool) or not isinstance(maximum_iterations, int) or maximum_iterations < 1:
        raise ValueError("maximum_iterations must be a positive integer")
    return _solve_fracture(
        fluid, length_m=length_m, elevation_change_m=elevation_change_m,
        divisions=divisions, times_s=times_s, apertures_m=reference_apertures_m,
        inlet_pressures_pa=inlet_pressures_pa, outlet_pressures_pa=outlet_pressures_pa,
        initial_pressure=initial_pressure, relative_tolerance=relative_tolerance,
        normal_compliance_m_pa=normal_compliance_m_pa, reference_pressure_pa=reference_pressure_pa,
        maximum_iterations=maximum_iterations,
    )


def _solve_fracture(
    fluid, *, length_m, elevation_change_m, divisions, times_s, apertures_m,
    inlet_pressures_pa, outlet_pressures_pa, initial_pressure, relative_tolerance,
    normal_compliance_m_pa, reference_pressure_pa, maximum_iterations,
) -> dict:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Prescribed-fracture verification currently requires one MPI process")
    if not isinstance(fluid, FractureFluid):
        raise ValueError("Explicit FractureFluid properties are required")
    for name, value in (("length_m", length_m), ("relative_tolerance", relative_tolerance)):
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if relative_tolerance >= 1:
        raise ValueError("relative_tolerance must be less than one")
    if isinstance(elevation_change_m, bool) or not isinstance(elevation_change_m, numbers.Real) \
            or not np.isfinite(elevation_change_m) or abs(elevation_change_m) > length_m:
        raise ValueError("elevation_change_m must be finite and no larger than fracture length")
    if isinstance(divisions, bool) or not isinstance(divisions, int) or divisions < 2:
        raise ValueError("divisions must be an integer >= 2")
    arrays = {}
    for name, values in (("times_s", times_s), ("apertures_m", apertures_m),
                         ("inlet_pressures_pa", inlet_pressures_pa),
                         ("outlet_pressures_pa", outlet_pressures_pa)):
        raw = np.asarray(values)
        if raw.dtype.kind not in "fiu" or raw.ndim != 1 or len(raw) < 2 or not np.all(np.isfinite(raw)):
            raise ValueError(f"{name} must be a finite real sequence with at least two states")
        arrays[name] = raw.astype(float)
    times, aperture = arrays["times_s"], arrays["apertures_m"]
    inlet, outlet = arrays["inlet_pressures_pa"], arrays["outlet_pressures_pa"]
    if any(len(values) != len(times) for values in arrays.values()):
        raise ValueError("All histories must have matching lengths")
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Physical times must start at zero and increase strictly")
    if np.any(aperture <= 0):
        raise ValueError("Closed fractures require contact/wetting treatment; aperture must be positive")
    if np.any(inlet < 0) or np.any(outlet < 0):
        raise ValueError("Negative gauge pressure requires a suction/cavitation model")
    for boundary_pressure in (inlet, outlet):
        boundary_width = aperture + normal_compliance_m_pa * (boundary_pressure - reference_pressure_pa)
        if np.any(~np.isfinite(boundary_width)) or np.any(boundary_width <= 0):
            raise ValueError("Boundary aperture reaches closure; contact/wetting model required")

    domain = mesh.create_interval(MPI.COMM_WORLD, divisions, [0., float(length_m)])
    space = fem.functionspace(domain, ("Lagrange", 1))
    pressure, previous = fem.Function(space), fem.Function(space)
    previous.interpolate(initial_pressure)
    previous.x.scatter_forward()
    if np.any(~np.isfinite(previous.x.array)) or np.any(previous.x.array < 0):
        raise ValueError("Initial pressure must be finite and nonnegative")
    pressure.x.array[:] = previous.x.array
    initial_aperture = aperture[0] + normal_compliance_m_pa * (previous.x.array - reference_pressure_pa)
    if np.any(~np.isfinite(initial_aperture)) or np.any(initial_aperture <= 0):
        raise ValueError("Initial compliant aperture must be positive; closed contact/wetting is not implemented")
    positions = space.tabulate_dof_coordinates()[:, 0]
    endpoints = [fem.locate_dofs_geometrical(space, lambda x, end=end: np.isclose(x[0], end))
                 for end in (0., length_m)]
    if any(len(ids) != 1 for ids in endpoints):
        raise RuntimeError("Fracture endpoints did not map uniquely")
    if not np.allclose([previous.x.array[ids[0]] for ids in endpoints], [inlet[0], outlet[0]],
                       rtol=relative_tolerance, atol=1e-9):
        raise ValueError("Initial pressure must match the initial boundary pressures")
    test, trial = ufl.TestFunction(space), ufl.TrialFunction(space)
    order = np.argsort(positions)
    history = [{"physical_time_s": 0., "pressure_pa": previous.x.array[order].tolist(),
                "aperture_m": float(aperture[0]),
                "calculated_apertures_m": initial_aperture[order].tolist()}]
    storage_coefficient = fem.Constant(domain, PETSc.ScalarType(1.))
    conductivity_coefficient = fem.Constant(domain, PETSc.ScalarType(1.))
    aperture_rate = fem.Constant(domain, PETSc.ScalarType(0.))
    boundary_values = [fem.Constant(domain, PETSc.ScalarType(0.)) for _ in endpoints]
    bcs = [fem.dirichletbc(value, ids, space) for value, ids in zip(boundary_values, endpoints)]
    gravity_gradient = fluid.density_kg_m3 * fluid.gravity_m_s2 * elevation_change_m / length_m
    if not np.isfinite(gravity_gradient):
        raise ValueError("Hydraulic gravity gradient overflowed; check units and physical inputs")
    bilinear = (storage_coefficient * trial * test
                + conductivity_coefficient * trial.dx(0) * test.dx(0)) * ufl.dx
    load = (storage_coefficient * previous * test - aperture_rate * test
            - conductivity_coefficient * gravity_gradient * test.dx(0)) * ufl.dx
    if normal_compliance_m_pa:
        width_reference = fem.Constant(domain, PETSc.ScalarType(aperture[0]))
        previous_reference = fem.Constant(domain, PETSc.ScalarType(aperture[0]))
        inverse_dt = fem.Constant(domain, PETSc.ScalarType(1.))
        width_expression = width_reference + normal_compliance_m_pa * (pressure - reference_pressure_pa)
        previous_width = previous_reference + normal_compliance_m_pa * (previous - reference_pressure_pa)
        volume_expression = (fluid.compressibility_pa_inv * width_expression * (pressure - previous)
                             + width_expression - previous_width) * inverse_dt
        flux_coefficient = width_expression**3 / (12 * fluid.viscosity_pa_s)
        residual_expression = (volume_expression * test + flux_coefficient
                               * (pressure.dx(0) + gravity_gradient) * test.dx(0)) * ufl.dx
        residual_form = fem.form(residual_expression)
        rate_form = fem.form(volume_expression * ufl.dx)
        problem = NonlinearProblem(
            residual_expression, pressure, bcs=bcs,
            petsc_options_prefix="compliant_fracture_flow_",
            petsc_options={"snes_type": "newtonls", "snes_rtol": relative_tolerance * .01,
                           "snes_atol": 1e-30, "snes_max_it": maximum_iterations,
                           "snes_error_if_not_converged": False, "ksp_type": "preonly",
                           "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        )
    else:
        residual_form = fem.form(ufl.action(bilinear, pressure) - load)
        rate_form = fem.form((storage_coefficient * (pressure - previous) + aperture_rate) * ufl.dx)
        problem = LinearProblem(
            bilinear, load, u=pressure, bcs=bcs, petsc_options_prefix="prescribed_fracture_flow_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps", "ksp_error_if_not_converged": True},
        )
    for step in range(1, len(times)):
        dt = times[step] - times[step - 1]
        width = aperture[step]
        storage = fluid.compressibility_pa_inv * width / dt
        conductivity = width**3 / (12 * fluid.viscosity_pa_s)
        volume_rate = (width - aperture[step - 1]) / dt
        if not (np.isfinite(storage) and storage > 0 and np.isfinite(conductivity)
                and conductivity > 0 and np.isfinite(volume_rate)):
            raise ValueError("Hydraulic coefficients overflowed or underflowed; check units and physical inputs")
        storage_coefficient.value = storage
        conductivity_coefficient.value = conductivity
        aperture_rate.value = volume_rate
        for value, pressure_value in zip(boundary_values, (inlet[step], outlet[step])):
            value.value = pressure_value
        if normal_compliance_m_pa:
            width_reference.value = width
            previous_reference.value = aperture[step - 1]
            inverse_dt.value = 1 / dt
        problem.solve()
        reason = int(problem.solver.getConvergedReason())
        iterations = int(problem.solver.getIterationNumber())
        if reason <= 0 or np.any(~np.isfinite(pressure.x.array)):
            raise RuntimeError(
                f"Fracture pressure solve failed at physical time {times[step]:.9g} s: "
                f"solver reason {reason}, iterations {iterations}"
            )
        pressure_scale = max(float(np.max(np.abs(pressure.x.array))), 1.)
        if np.min(pressure.x.array) < -relative_tolerance * pressure_scale:
            raise RuntimeError("Saturated fracture predicted negative gauge pressure; cavitation/wetting model required")
        calculated_width = width + normal_compliance_m_pa * (pressure.x.array - reference_pressure_pa)
        if np.any(~np.isfinite(calculated_width)) or np.any(calculated_width <= 0):
            raise RuntimeError("Compliant fracture reached closure; contact/wetting model required")
        residual = assemble_vector(residual_form)
        try:
            residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            reactions = [float(residual.array[ids[0]]) for ids in endpoints]
            free = np.setdiff1d(np.arange(len(residual.array)), np.concatenate(endpoints))
            free_error = float(np.max(np.abs(residual.array[free])))
        finally:
            residual.destroy()
        storage_rate = float(fem.assemble_scalar(rate_form))
        net_inflow = sum(reactions)
        balance_error = abs(storage_rate - net_inflow)
        current_conductivity = float(np.max(calculated_width))**3 / (12 * fluid.viscosity_pa_s)
        if not np.all(np.isfinite([storage_rate, net_inflow, free_error, balance_error,
                                   current_conductivity, *reactions])):
            raise RuntimeError("Fracture flow independent equilibrium/mass-balance check is non-finite")
        flow_scale = max(abs(storage_rate), sum(abs(value) for value in reactions),
                         current_conductivity * pressure_scale / length_m, np.finfo(float).tiny)
        if max(free_error, balance_error) > relative_tolerance * flow_scale:
            raise RuntimeError("Fracture flow independent equilibrium/mass-balance check failed")
        history.append({
            "physical_time_s": float(times[step]), "aperture_m": float(width),
            "calculated_apertures_m": calculated_width[order].tolist(),
            "pressure_pa": pressure.x.array[order].tolist(),
            "boundary_inflows_m2_s": reactions, "net_inflow_m2_s": net_inflow,
            "storage_and_aperture_rate_m2_s": storage_rate,
            "scaled_mass_balance_error": balance_error / flow_scale,
            "scaled_free_residual": free_error / flow_scale,
            "solver_converged_reason": reason, "solver_iterations": iterations,
        })
        previous.x.array[:] = pressure.x.array
        previous.x.scatter_forward()
    return {
        "status": "passed_compliant_fracture_flow" if normal_compliance_m_pa else "passed_prescribed_fracture_flow",
        "engineering_validated": False, "solver_ready": False,
        "coupled_to_mechanics": bool(normal_compliance_m_pa), "wetting_front": False,
        "mechanics_model": "local linear normal compliance" if normal_compliance_m_pa else "prescribed aperture",
        "normal_compliance_m_pa": normal_compliance_m_pa,
        "reference_pressure_pa": reference_pressure_pa,
        "positions_m": positions[order].tolist(), "steps": history,
        "limitations": (
            "Already-wet open path with local linear normal compliance only; no contact, propagation or 3D dam mechanics."
            if normal_compliance_m_pa else
            "Prescribed uniformly open saturated path only; no contact, propagation or hydraulic feedback."
        ),
    }
