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
    mechanics_formulation: str = "hybrid",
    damage_acceleration: bool = False,
    final_load_substeps: int = 1,
    refined_baseline: bool = False,
) -> PhaseFieldResult:
    """Run the configured dam phase-field simulation on an imported/refined mesh."""
    if not specifications and parameters is None:
        raise ValueError("A crack specification or baseline phase-field parameters are required")
    first = parameters if parameters is not None else specifications[0]
    domain = imported.domain
    from three_body_model import (
        DAMAGE_REGION_IDS, is_three_body, materials, require_support_tags, rock_support, support_bcs,
        support_reactions, support_form as rock_support_form,
    )
    three_body = is_three_body(structure)
    damage_regions = DAMAGE_REGION_IDS if three_body else (2,)
    placement = (
        validate_crack_placement(imported, specifications, centre_regions=damage_regions)
        if specifications else []
    )
    coefficients = materials(imported, structure) if three_body else None
    baseline_report = None
    def record_stage(stage):
        if domain.comm.rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            temporary = output / "analysis_stage.tmp.json"
            temporary.write_text(json.dumps({"stage": stage}) + "\n")
            temporary.replace(output / "analysis_stage.json")

    if refined_baseline:
        from dam_elasticity import solve as solve_elastic

        baseline_output = output / "refined_baseline"
        baseline_output.mkdir(parents=True, exist_ok=True)
        record_stage("fenicsx_refined_baseline")
        if domain.comm.rank == 0:
            print("=== FEniCSx refined-mesh uncracked baseline (before cracking) ===", flush=True)
        _, _, baseline_report = solve_elastic(
            imported, structure,
            {"relative_tolerance": options.mechanics_relative_tolerance,
             "maximum_iterations": options.mechanics_maximum_iterations},
            baseline_output, save_source_displacement=False,
        )
        baseline_report.update(
            scenario_name="Refined_NoPhase",
            model="linear isotropic elasticity; zero damage; same mesh as crack scenario",
            configured_tensile_strength_pa=structure.material.tensile_strength_pa,
            crack_surface_measure_m2=0.0,
            engineering_validated=False,
            comparison_basis="identical mesh, material, external loads and spring supports; no crack pressure",
        )
        if three_body and (not np.isfinite(baseline_report["relative_force_balance_error"])
                           or baseline_report["relative_force_balance_error"] > 1e-6):
            raise RuntimeError("Refined three-body zero-damage equilibrium gate failed")
        if domain.comm.rank == 0:
            (baseline_output / "baseline_report.json").write_text(
                json.dumps(baseline_report, indent=2, allow_nan=False) + "\n",
            )
            print("=== Refined-mesh baseline completed; starting FEniCSx cracking ===", flush=True)
        record_stage("cracking")
    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=imported.facet_tags)
    loads = structure.loads
    material = structure.material
    density = coefficients["density"] if three_body else material.density_kg_m3
    gravity = ufl.as_vector((0.0, 0.0, density * loads.gravity_z_m_s2))
    z = ufl.SpatialCoordinate(domain)[2]
    normal = ufl.FacetNormal(domain)
    pressure_gradient = loads.water_density_kg_m3 * abs(loads.gravity_z_m_s2)
    minimum_elevation = (structure.concrete_minimum_z if three_body else
                         float(domain.comm.allreduce(domain.geometry.x[:, 2].min(), op=MPI.MIN)))
    tailwater_elevation = minimum_elevation + loads.tailwater_head_m
    pressures = {
        2: pressure_gradient * ufl.max_value(loads.overflow_head_m - z, 0),
        3: pressure_gradient * ufl.max_value(tailwater_elevation - z, 0),
        4: pressure_gradient * loads.overflow_head_m,
    }
    from dam_elasticity import ground_supports

    springs, _ = ({}, "three_body_springs") if three_body else ground_supports(imported, structure)

    def support_form(trial, test):
        result = 0
        for tag, coefficients in springs.items():
            for component, stiffness in enumerate(coefficients):
                result += stiffness * trial[component] * test[component] * ds(tag)
        rock_rollers = rock_support_form(trial, test, ds, structure) if three_body else None
        if rock_rollers is not None:
            result += rock_rollers
        return result

    if three_body:
        require_support_tags(imported, structure)
    # Rollers are springs in support_form; only the legacy fixed bottom needs Dirichlet BCs.
    fixed_bottom = three_body and rock_support(structure)["model"] == "fixed_bottom"

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

    from interface_springs import InterfaceSprings
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
        damage_region_ids=damage_regions,
        parameters=parameters,
        mechanics_formulation=mechanics_formulation,
        damage_acceleration=damage_acceleration,
        final_load_substeps=final_load_substeps,
        elastic_coefficients=coefficients,
        interface_factory=(lambda space: InterfaceSprings(imported, structure, space)) if three_body else None,
        boundary_conditions=(
            (lambda space, load: support_bcs(imported, space, structure)) if fixed_bottom else None
        ),
    )

    lam, mu, bulk = elastic_parameters(material.youngs_modulus_pa, material.poissons_ratio)
    mask = 1.
    if three_body:
        lam, mu, bulk = (coefficients[key] for key in ("lam", "mu", "bulk"))
        mask = coefficients["damage_mask"]
    stress_expression = (
        (mask * degradation(result.damage, first.residual_stiffness) + 1 - mask)
        * undamaged_stress(result.displacement, lam, mu)
    )
    if mechanics_formulation == "experimental_amor":
        from compression_preserving import stress as compression_preserving_stress

        stress_expression = compression_preserving_stress(
            result.displacement, result.damage, bulk, mu, first.residual_stiffness,
            damage_mask=mask,
        )
    elif mechanics_formulation == "experimental_spectral":
        from spectral_split import SpectralSplit

        spectral = SpectralSplit(
            result.displacement, result.damage, bulk, mu, first.residual_stiffness,
        )
        stress_expression = spectral.stress
    tensor_space = fem.functionspace(domain, ("DG", 0, (3, 3)))
    stress = fem.Function(tensor_space, name="degraded_stress_pa")
    stress.interpolate(
        fem.Expression(
            stress_expression,
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
        writer.write_function([stress, coefficients["body_id"]] if three_body else stress)
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
    if three_body:
        test = ufl.TestFunction(result.displacement.function_space)
        reactions = support_reactions(
            imported, result.displacement.function_space, structure, result.displacement,
            ufl.inner(stress_expression, strain(test)) * dx - external_work(test, 1., result.damage), ds,
        )
    imbalance = applied + np.asarray(list(reactions.values())).sum(axis=0)
    applied_norm = float(np.linalg.norm(applied))
    relative_balance = float(np.linalg.norm(imbalance) / applied_norm) if applied_norm else 0.0
    configured_strength = material.tensile_strength_pa
    implied_strength = implied_at2_tensile_strength(
        material.youngs_modulus_pa,
        first.critical_energy_release_rate_j_m2,
        length_scale,
    )
    owned_cells = domain.topology.index_map(domain.topology.dim).size_local
    owned_stress_dofs = tensor_space.dofmap.list[:owned_cells, 0]
    stress_values = stress.x.array.reshape(-1, 3, 3)[owned_stress_dofs]
    if three_body:
        concrete = np.concatenate((imported.cell_tags.find(1), imported.cell_tags.find(2)))
        stress_values = stress_values[concrete[concrete < owned_cells]]
        if not np.isfinite(relative_balance) or relative_balance > 1e-6:
            raise RuntimeError(f"Three-body fracture equilibrium gate failed: {relative_balance}")
    eigenvalues = np.linalg.eigvalsh(stress_values)
    maximum_tensile_stress = float(
        domain.comm.allreduce(eigenvalues[:, -1].max() if len(eigenvalues) else -np.inf, op=MPI.MAX)
    )
    maximum_compressive_stress = float(
        domain.comm.allreduce(eigenvalues[:, 0].min() if len(eigenvalues) else np.inf, op=MPI.MIN)
    )
    result.report.update(
        {
            "crack_placement": placement,
            "damage_region_ids": list(damage_regions),
            "non_wall_damage": (
                "bedrock fixed at zero" if three_body else "fixed at zero, including shared interface DOFs"
            ),
            "applied_force_n": applied.tolist(),
            "ground_spring_reactions_n": reactions,
            "force_imbalance_n": imbalance.tolist(),
            "relative_force_balance_error": relative_balance,
            "maximum_degraded_stress_pa": float(
                domain.comm.allreduce(np.abs(stress_values).max() if len(stress_values) else 0., op=MPI.MAX)
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
                (
                    "Hybrid AT2 degrades full stiffness but only tensile energy drives damage."
                    if mechanics_formulation == "hybrid"
                    else "Experimental Amor mechanics preserves negative volumetric energy only; shear remains degraded."
                ),
                "No unilateral crack-face contact or friction is enforced.",
                "Diffuse pressure is an energy-consistent regularised traction, not an explicit face load.",
                "Convergence and successful execution do not constitute engineering validation.",
                (
                    "Damage is allowed in plinth (1) and wall (2); bedrock damage DOFs are fixed at zero. "
                    "Wall and plinth damage do not couple across the spring joint, and the joint springs do not soften."
                    if three_body else
                    "Damage is restricted to wall region 2; non-wall and shared interface damage DOFs are fixed at zero."
                ),
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
        stress_expression,
        output,
        sample_points={
            f"refined_baseline_{kind}_peak": {
                "coordinate_xyz_m": baseline_report["stress_recovery"][f"{kind}_peak_centroid_xyz_m"],
                "region_id": baseline_report["stress_recovery"].get(f"{kind}_peak_region_id"),
            }
            for kind in ("tensile", "compressive")
        } if baseline_report is not None else None,
        cell_tags=imported.cell_tags if three_body else None,
    ))
    if three_body:
        result.report.update(
            model_kind="three_body", source_mesh=structure.three_body_source,
            wall_plinth_connection="three_body_springs",
            plinth_bedrock_connection=(
                "bonded" if structure.three_body_config["geometry"]["bedrock"]["support"].get("model") == "bonded"
                else "spring"
            ),
            stress_scope="wall and plinth; full-domain rock reported separately",
            fracture_verification="implementation verification only; full-dam validation required",
        )
    result.report["mesh_nodes"] = int(result.damage.function_space.dofmap.index_map.size_global)
    result.report["mesh_cells"] = int(domain.topology.index_map(3).size_global)
    if baseline_report is not None:
        result.report["refined_mesh_baseline"] = baseline_report
    if domain.comm.rank == 0:
        (output / "phase_field_report.json").write_text(
            json.dumps(result.report, indent=2, allow_nan=False) + "\n"
        )
    return result
