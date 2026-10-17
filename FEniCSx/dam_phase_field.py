"""Couple the verified phase-field core to the existing dam loads and supports."""
from __future__ import annotations

import json
import math
from pathlib import Path

from dolfinx import fem, io
from mpi4py import MPI
import numpy as np
import ufl

from crack_field import (
    CrackSpecification,
    PhaseFieldParameters,
    PhaseFieldSolverOptions,
    validate_crack_placement,
)
from phase_field_solver import (
    PhaseFieldResult,
    degradation,
    elastic_parameters,
    strain,
    solve_staggered,
    undamaged_stress,
)


def _global_scalar(domain, expression) -> float:
    local = fem.assemble_scalar(fem.form(expression))
    return float(domain.comm.allreduce(local, op=MPI.SUM))


def implied_at2_tensile_strength(young: float, gc: float, length_scale: float) -> float:
    """One-dimensional homogeneous AT2 strength used only as a calibration diagnostic."""
    return math.sqrt(27.0 * young * gc / (256.0 * length_scale))


def solve(
    imported,
    structure,
    specifications: list[CrackSpecification],
    options: PhaseFieldSolverOptions,
    output: Path,
    *,
    parameters: PhaseFieldParameters | None = None,
) -> PhaseFieldResult:
    """Run the configured dam phase-field simulation on an imported/refined mesh."""
    if not specifications and parameters is None:
        raise ValueError("A crack specification or baseline phase-field parameters are required")
    first = parameters if parameters is not None else specifications[0]
    placement = validate_crack_placement(imported, specifications) if specifications else []
    domain = imported.domain
    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=imported.facet_tags)
    loads = structure.loads
    material = structure.material
    gravity = ufl.as_vector((0.0, 0.0, material.density_kg_m3 * loads.gravity_z_m_s2))
    z = ufl.SpatialCoordinate(domain)[2]
    normal = ufl.FacetNormal(domain)
    pressure_gradient = loads.water_density_kg_m3 * abs(loads.gravity_z_m_s2)
    minimum_elevation = float(domain.geometry.x[:, 2].min())
    tailwater_elevation = minimum_elevation + loads.tailwater_head_m
    pressures = {
        2: pressure_gradient * ufl.max_value(loads.overflow_head_m - z, 0),
        3: pressure_gradient * ufl.max_value(tailwater_elevation - z, 0),
        4: pressure_gradient * loads.overflow_head_m,
    }
    springs = {
        1: (
            structure.foundation_support.spring_x_n_per_m3,
            structure.foundation_support.spring_y_n_per_m3,
            structure.foundation_support.spring_z_n_per_m3,
        ),
        8: (
            structure.plinth_support.spring_x_n_per_m3,
            structure.plinth_support.spring_y_n_per_m3,
            structure.plinth_support.spring_z_n_per_m3,
        ),
    }

    def support_form(trial, test):
        result = 0
        for tag, coefficients in springs.items():
            for component, stiffness in enumerate(coefficients):
                result += stiffness * trial[component] * test[component] * ds(tag)
        return result

    pressure_value = first.internal_pressure_pa
    if any(specification.internal_pressure_pa != pressure_value for specification in specifications):
        raise ValueError("All cracks in one diffuse pressure solve must use the same internal pressure")

    def external_work(test, load_factor, damage):
        work = load_factor * ufl.inner(gravity, test) * dx
        for tag, pressure in pressures.items():
            work += load_factor * ufl.inner(-pressure * normal, test) * ds(tag)
        if pressure_value:
            # -grad(alpha) points away from the diffuse crack on both faces.
            work += load_factor * ufl.inner(-pressure_value * ufl.grad(damage), test) * dx
        return work

    result = solve_staggered(
        domain,
        material.youngs_modulus_pa,
        material.poissons_ratio,
        specifications,
        options,
        external_work,
        support_form=support_form,
        output=output,
        damage_cell_tags=imported.cell_tags,
        damage_region_id=2,
        parameters=parameters,
    )

    lam, mu, _ = elastic_parameters(material.youngs_modulus_pa, material.poissons_ratio)
    tensor_space = fem.functionspace(domain, ("DG", 0, (3, 3)))
    stress = fem.Function(tensor_space, name="degraded_stress_pa")
    stress.interpolate(
        fem.Expression(
            degradation(result.damage, first.residual_stiffness)
            * undamaged_stress(result.displacement, lam, mu),
            tensor_space.element.interpolation_points,
        )
    )
    strain_field = fem.Function(tensor_space, name="strain")
    strain_field.interpolate(
        fem.Expression(strain(result.displacement), tensor_space.element.interpolation_points)
    )
    scalar_space = fem.functionspace(domain, ("DG", 0))
    crack_density = fem.Function(scalar_space, name="crack_surface_density_per_m")
    length_scale = first.length_scale_m
    crack_density.interpolate(
        fem.Expression(
            result.damage**2 / (2.0 * length_scale)
            + length_scale * ufl.inner(ufl.grad(result.damage), ufl.grad(result.damage)) / 2.0,
            scalar_space.element.interpolation_points,
        )
    )
    with io.VTKFile(domain.comm, output / "degraded_stress.pvd", "w") as writer:
        writer.write_function(stress)
    with io.VTKFile(domain.comm, output / "strain.pvd", "w") as writer:
        writer.write_function(strain_field)
    with io.VTKFile(domain.comm, output / "crack_surface_density.pvd", "w") as writer:
        writer.write_function(crack_density)

    def integral(expression):
        return _global_scalar(domain, expression)

    applied = np.asarray(
        [
            integral(gravity[i] * dx)
            + sum(integral(-pressure * normal[i] * ds(tag)) for tag, pressure in pressures.items())
            + (
                integral(-pressure_value * ufl.grad(result.damage)[i] * dx)
                if pressure_value
                else 0.0
            )
            for i in range(3)
        ]
    )
    reactions = {
        str(tag): [
            integral(-coefficients[i] * result.displacement[i] * ds(tag))
            for i in range(3)
        ]
        for tag, coefficients in springs.items()
    }
    imbalance = applied + np.asarray(list(reactions.values())).sum(axis=0)
    applied_norm = float(np.linalg.norm(applied))
    relative_balance = float(np.linalg.norm(imbalance) / applied_norm) if applied_norm else 0.0
    configured_strength = material.tensile_strength_pa
    implied_strength = implied_at2_tensile_strength(
        material.youngs_modulus_pa,
        first.critical_energy_release_rate_j_m2,
        length_scale,
    )
    stress_values = stress.x.array.reshape(-1, 3, 3)
    eigenvalues = np.linalg.eigvalsh(stress_values)
    maximum_tensile_stress = float(
        domain.comm.allreduce(eigenvalues[:, -1].max(), op=MPI.MAX)
    )
    maximum_compressive_stress = float(
        domain.comm.allreduce(eigenvalues[:, 0].min(), op=MPI.MIN)
    )
    result.report.update(
        {
            "crack_placement": placement,
            "damage_region_id": 2,
            "non_wall_damage": "fixed at zero, including shared interface DOFs",
            "applied_force_n": applied.tolist(),
            "ground_spring_reactions_n": reactions,
            "force_imbalance_n": imbalance.tolist(),
            "relative_force_balance_error": relative_balance,
            "maximum_degraded_stress_pa": float(
                domain.comm.allreduce(np.abs(stress.x.array).max(), op=MPI.MAX)
            ),
            "maximum_principal_tensile_stress_pa": maximum_tensile_stress,
            "maximum_principal_compressive_stress_pa": maximum_compressive_stress,
            "crack_surface_measure_m2": integral(crack_density * dx),
            "configured_tensile_strength_pa": configured_strength,
            "at2_implied_homogeneous_tensile_strength_pa": implied_strength,
            "implied_to_configured_strength_ratio": implied_strength / configured_strength,
            "internal_pressure_model": (
                "diffuse pressure force -p*grad(alpha)" if pressure_value else "disabled (0 Pa)"
            ),
            "limitations": [
                "Hybrid AT2 degrades full stiffness but only tensile energy drives damage.",
                "No unilateral crack-face contact or friction is enforced.",
                "Diffuse pressure is an energy-consistent regularised traction, not an explicit face load.",
                "Convergence and successful execution do not constitute engineering validation.",
                "Damage is restricted to wall region 2; non-wall and shared interface damage DOFs are fixed at zero.",
            ],
        }
    )
    from stress_recovery import recover_stress

    result.report["raw_element_stress"] = {
        "maximum_principal_tensile_stress_pa": maximum_tensile_stress,
        "maximum_principal_compressive_stress_pa": maximum_compressive_stress,
    }
    result.report.update(recover_stress(
        domain,
        degradation(result.damage, first.residual_stiffness)
        * undamaged_stress(result.displacement, lam, mu),
        output,
    ))
    if domain.comm.rank == 0:
        (output / "phase_field_report.json").write_text(
            json.dumps(result.report, indent=2, allow_nan=False) + "\n"
        )
    return result
