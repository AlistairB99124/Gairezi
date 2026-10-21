"""Render a civil-engineering-oriented summary of a phase-field dam run."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _number(value: float, decimals: int = 3) -> str:
    return f"{value:,.{decimals}f}"


def _vector_meganewtons(values: list[float]) -> str:
    return "(" + ", ".join(f"{value / 1e6:,.3f}" for value in values) + ") MN"


def _vector_newtons(values: list[float]) -> str:
    return "(" + ", ".join(f"{value:,.4f}" for value in values) + ") N"


def _percentage_change(initial: float, final: float) -> float | None:
    if initial == 0.0:
        return None
    return 100.0 * (final - initial) / abs(initial)


def build_engineering_results(
    configuration: dict[str, Any],
    result: dict[str, Any],
    refinement: dict[str, Any],
    placement: list[dict[str, Any]],
    import_report: dict[str, Any],
    *,
    configuration_sha256: str,
    environment: dict[str, Any],
    started_at_utc: str,
    completed_at_utc: str,
) -> str:
    """Build a readable report from the persisted run data."""
    steps = result["steps"]
    if not steps:
        raise ValueError("Cannot create results.md without load-step results")
    final = steps[-1]
    initial = steps[0]
    concrete = configuration["material"]["concrete"]
    loads = configuration["loads"]
    cracks = configuration["crack_simulations"]
    first_crack = cracks[0]
    parameters = first_crack["phase_field_parameters"]
    crack_loading = first_crack["loading"]
    final_refinement = refinement["levels"][-1]["cracks"]
    least_resolved = min(
        crack["elements_per_length_scale"] for crack in final_refinement
    )
    maximum_local_diameter = max(
        crack["maximum_cell_diameter_near_crack_m"] for crack in final_refinement
    )
    recommended_resolution = refinement[
        "recommended_minimum_elements_per_length_scale"
    ]
    fracture_change = _percentage_change(
        initial["fracture_energy_j"], final["fracture_energy_j"]
    )
    strength = result["configured_tensile_strength_pa"]
    tensile_stress = result["maximum_principal_tensile_stress_pa"]
    strength_utilisation = tensile_stress / strength
    placement_valid = all(item["valid"] for item in placement)
    pressure = first_crack["loading"]["internal_pressure_pa"]

    lines = [
        "# Dam phase-field fracture result",
        "",
        "## Executive summary",
        "",
        "This directory contains one completed numerical run of the configured",
        "concrete-dam phase-field model. It is a **preliminary research result, not an",
        "engineering assessment or a dam-safety determination**.",
        "",
        "At the applied full load (`load factor = "
        f"{final['load_factor']:.1f}`):",
        "",
        f"- Maximum displacement magnitude: **{final['maximum_displacement_m'] * 1000:.2f} mm**.",
        f"- Maximum principal tensile stress: **{tensile_stress / 1e6:.3f} MPa**.",
        f"- Configured concrete tensile strength: **{strength / 1e6:.3f} MPa**.",
        f"- Tensile stress / configured tensile strength: **{strength_utilisation:.1%}**.",
        f"- Final diffuse crack surface measure: **{result['crack_surface_measure_m2']:.2f} m2**.",
        f"- Relative global force-balance error: **{result['relative_force_balance_error']:.3e}**.",
        "",
        f"The refinement status is **`{refinement['status']}`**. The least-resolved",
        f"crack has **{least_resolved:.3f} elements per phase-field length scale**,",
        f"below the workflow recommendation of **{recommended_resolution:.1f}**.",
        "The crack initiation, propagation, and damage distribution should therefore",
        "not be treated as quantitatively reliable or mesh independent.",
        "",
        (
            "Internal crack pressure was **not applied (0 Pa)**."
            if pressure == 0.0
            else f"Configured internal crack pressure: **{pressure / 1e6:.3f} MPa**."
        ),
        "",
        "## What was analysed",
        "",
        f"- Model: `{result['model']}`.",
        f"- Energy split: `{result['energy_split']}`.",
        "- Three-dimensional tetrahedral dam and support mesh.",
        f"- {len(steps)} load increments ending at load factor {final['load_factor']:.1f}.",
        "- Staggered mechanics/damage solution with irreversible damage.",
        "- Distributed elastic spring supports for the bedrock boundary.",
        "- Gravity and configured external water loads.",
        f"- Internal crack pressure model: `{result['internal_pressure_model']}`.",
        "- No unilateral crack-face contact or friction.",
        "",
        "The phase-field damage variable is dimensionless:",
        "",
        "| Damage value | Engineering interpretation |",
        "|---:|---|",
        "| `0` | Intact material in this model |",
        "| Between `0` and `1` | Diffuse transition/damage zone |",
        "| `1` | Fully damaged phase-field value |",
        "",
        "Damage is not crack opening, crack width, percentage strength loss, or a",
        "factor of safety. A contour such as `damage = 0.5` is a convenient view of",
        "the regularised crack zone, not an explicit physical crack face.",
        "",
        "## Principal input values",
        "",
        "The authoritative dam input snapshot is [config.json](./config.json).",
        "For user-located crack runs, the selected location, derived crack geometry and the",
        "reused solver defaults are recorded separately in [crack_location.json](./crack_location.json).",
        "Its annular-sector size is radial depth, arc length, and vertical opening,",
        "not the dimensions of a straight rectangular box.",
        "",
        "| Quantity | Value | Units |",
        "|---|---:|---|",
        f"| Concrete density | {_number(concrete['density'], 0)} | kg/m3 |",
        f"| Concrete Young's modulus | {_number(concrete['youngs_modulus'] / 1e6, 0)} | MPa |",
        f"| Concrete Poisson ratio | {concrete['poisson_ratio']:.3f} | dimensionless |",
        f"| Configured concrete tensile strength | {concrete['tensile_strength_pa'] / 1e6:.3f} | MPa |",
        f"| Fracture energy, `Gc` | {_number(parameters['critical_energy_release_rate_J_m2'], 1)} | J/m2 |",
        f"| Phase-field length scale, `l` | {parameters['length_scale_m']:.4f} | m |",
        f"| Residual stiffness | {parameters['residual_stiffness']:.3e} | dimensionless |",
        f"| Gravity acceleration | {loads['gravity']:.3f} | m/s2 |",
        f"| Water height | {loads['water_height']:.3f} | m |",
        f"| Tailwater head | {loads['tailwater_head']:.3f} | m |",
        f"| Internal crack pressure | {pressure:.3f} | Pa |",
        "",
        "## Configured cracks",
        "",
        "| # | Name | Centre (x, y, z), m | Nominal size (x, y, z), m | Normal | Placement valid |",
        "|---:|---|---|---|---|---|",
    ]
    for index, crack in enumerate(cracks, start=1):
        name = crack.get("name", "")
        centre = crack["central_crack_pos_m"]
        size = crack["size_m"]
        normal = crack["orientation"]
        is_valid = placement[index - 1]["valid"] if index <= len(placement) else False
        lines.append(
            f"| {index} | {name} | ({centre['x']:.3f}, {centre['y']:.3f}, {centre['z']:.3f}) "
            f"| ({size['x']:.3f}, {size['y']:.3f}, {size['z']:.3f}) "
            f"| ({normal['normal_x']:.3f}, {normal['normal_y']:.3f}, "
            f"{normal['normal_z']:.3f}) | {'yes' if is_valid else 'no'} |"
        )

    lines.extend(
        [
            "",
            f"All placement checks valid: **{'yes' if placement_valid else 'no'}**.",
            "",
            "## Load-step results",
            "",
            "Values below come from [phase_field_report.json](./phase_field_report.json).",
            "",
            "| Load factor | Maximum displacement | Elastic energy | Fracture energy | Staggered iterations |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for step in steps:
        lines.append(
            f"| {step['load_factor']:.1f} "
            f"| {step['maximum_displacement_m'] * 1000:.3f} mm "
            f"| {step['elastic_energy_j'] / 1000:.2f} kJ "
            f"| {step['fracture_energy_j'] / 1000:.2f} kJ "
            f"| {step['staggered_iterations']} |"
        )

    fracture_change_text = (
        "not defined because the first reported fracture energy is zero"
        if fracture_change is None
        else f"{fracture_change:.2f}%"
    )
    lines.extend(
        [
            "",
            "### Interpretation of the load history",
            "",
            "The relation between load factor and maximum displacement should be reviewed",
            "for departures from proportional response. The fracture energy includes the",
            "energy of the prescribed initial diffuse crack, so its starting value is not",
            "zero. Across the reported load history, fracture energy changed from",
            f"**{initial['fracture_energy_j'] / 1000:.2f} kJ** to",
            f"**{final['fracture_energy_j'] / 1000:.2f} kJ** ({fracture_change_text}).",
            "A small change is consistent with limited additional damage in this model,",
            "but crack growth must also be checked by comparing the initial and final",
            "damage fields. It is not evidence that the real dam cannot crack.",
            "",
            "## Stress and fracture calibration",
            "",
            "| Indicator | Result |",
            "|---|---:|",
            f"| Maximum degraded stress component magnitude | {result['maximum_degraded_stress_pa'] / 1e6:.3f} MPa |",
            f"| Maximum principal tensile stress | {tensile_stress / 1e6:.3f} MPa |",
            f"| Maximum principal compressive stress | {result.get('maximum_principal_compressive_stress_pa', 0.0) / 1e6:.3f} MPa |",
            f"| Configured tensile strength | {strength / 1e6:.3f} MPa |",
            f"| Tensile stress / configured tensile strength | {strength_utilisation:.1%} |",
            f"| AT2 implied homogeneous tensile strength | {result['at2_implied_homogeneous_tensile_strength_pa'] / 1e6:.3f} MPa |",
            f"| AT2 implied / configured strength | {result['implied_to_configured_strength_ratio']:.3f} |",
            "",
            (
                "Principal-stress peaks use consistent L2 recovery to nodal tensors, "
                "then element nodal-tensor averaging before eigenvalues, matching the "
                "Elmer comparison convention. Raw element peaks remain in "
                "`raw_element_stress`; `degraded_stress.pvd` is the raw DG0 field and "
                "`recovered_stress.pvd` is the recovered nodal field. Recovery can "
                "smooth localised crack-tip peaks; inspect both fields."
                if "stress_recovery" in result
                else "Historical report: stress-recovery metadata was not recorded."
            ),
            "",
            "The AT2 implied strength is a calibration diagnostic. If it differs materially",
            "from the intended concrete tensile strength, `Gc` and `l` are not a mutually",
            "calibrated parameter pair for that target. They should be selected from",
            "appropriate concrete test data or a documented calibration procedure, not",
            "adjusted merely to force a preferred crack contour.",
            "",
            "## Equilibrium and reactions",
            "",
            f"Applied force vector: `{_vector_meganewtons(result['applied_force_n'])}`.",
            "",
            "| Support tag | Reaction vector (x, y, z) |",
            "|---:|---|",
        ]
    )
    for tag, reaction in result["ground_spring_reactions_n"].items():
        lines.append(f"| {tag} | `{_vector_meganewtons(reaction)}` |")
    lines.extend(
        [
            "",
            f"Force imbalance: `{_vector_newtons(result['force_imbalance_n'])}`.",
            f"Relative force-balance error: **{result['relative_force_balance_error']:.3e}**.",
            "",
            "This is a numerical equilibrium check. It does not validate the support",
            "stiffnesses, boundary surfaces, reservoir loads, or their representation of",
            "the actual structure.",
            "",
            "## Mesh and phase-field resolution",
            "",
            f"- Imported nodes: **{import_report['imported_nodes']:,}**.",
            f"- Imported tetrahedra before crack-band refinement: **{import_report['tetrahedra']:,}**.",
            f"- Imported volume: **{import_report['tetrahedral_volume_m3']:,.2f} m3**.",
            f"- Finest reported maximum cell diameter near a crack: **{maximum_local_diameter:.4f} m**.",
            f"- Least-resolved crack: **{least_resolved:.3f} elements per length scale**.",
            f"- Recommended minimum: **{recommended_resolution:.1f} elements per length scale**.",
            f"- Resolution status: **`{refinement['status']}`**.",
            "",
            "The current mesh cannot establish mesh-independent fracture behaviour. A",
            "convergence study should compare crack initiation load, damage contours,",
            "crack surface measure, agreed displacement/stress probes, reactions, and",
            "energy quantities on progressively finer crack-band meshes.",
            "",
            "## Output files and ParaView",
            "",
            "| File | Contents | Units / interpretation |",
            "|---|---|---|",
            "| [damage.pvd](./damage.pvd) | Damage history | Dimensionless; 0 intact, 1 fully damaged |",
            "| [displacement.pvd](./displacement.pvd) | Displacement vector | m |",
            "| [degraded_stress.pvd](./degraded_stress.pvd) | Degraded Cauchy stress tensor | Pa |",
            "| [tensile_history.pvd](./tensile_history.pvd) | Tensile energy history | J/m3; damage-driving model variable |",
            "| [strain.pvd](./strain.pvd) | Small-strain tensor | dimensionless |",
            "| [crack_surface_density.pvd](./crack_surface_density.pvd) | Diffuse crack density | 1/m; not crack width |",
            "",
            "Open the `.pvd` collection files rather than individual `.vtu` pieces. For",
            "damage, use a fixed colour range from 0 to 1. A Threshold filter from about",
            "0.5 to 1.0 can isolate the diffuse crack core. A value such as `7.4e-03`",
            "means `damage = 0.0074`, which is nearly intact material; it is not 0.74%",
            "crack opening.",
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in result["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "- The fracture prediction is not mesh converged.",
            "- The model has not been validated against concrete fracture tests or field measurements.",
            "- No independent dam-safety review is represented by this output.",
            "",
            "## Recommended next steps",
            "",
            "1. Confirm crack location, orientation, dimensions, and the governing load case.",
            "2. Confirm whether reservoir water pressure should act within the crack.",
            "3. Calibrate `Gc` and `l` against documented concrete fracture data.",
            "4. Verify the implementation on a smaller fracture benchmark.",
            "5. Perform a mesh/length-scale convergence study, using MPI where required.",
            "6. Compare agreed probes, stresses, and reactions with an independent reference model.",
            "7. Obtain review by a qualified civil/structural engineer before decision-making.",
            "",
            "## Reproducibility",
            "",
            f"- Started (UTC): `{started_at_utc}`.",
            f"- Completed (UTC): `{completed_at_utc}`.",
            f"- Configuration SHA-256: `{configuration_sha256}`.",
            f"- Python: `{environment.get('python', 'not recorded')}`.",
            f"- DOLFINx: `{environment.get('dolfinx', 'not recorded')}`.",
            f"- PETSc: `{environment.get('petsc', 'not recorded')}`.",
            "",
            "See [manifest.json](./manifest.json) for the complete environment and run",
            "status. Successful execution and convergence do not constitute engineering",
            "validation or establish that the dam is safe or unsafe.",
            "",
        ]
    )
    refined = result.get("refined_mesh_baseline")
    if refined is not None:
        lines.extend([
            "## Mesh-controlled crack comparison",
            "",
            "The uncracked refined baseline was solved before introducing damage,",
            "on the identical mesh with the same material, external loads and supports.",
            "See [refined baseline report](./refined_baseline/baseline_report.json)",
            "and [refined baseline stress field](./refined_baseline/recovered_stress.pvd).",
            "",
            "| Recovered global peak | Refined uncracked, Pa | Cracked, Pa | Cracked minus refined, Pa |",
            "|---|---:|---:|---:|",
        ])
        for label, key in (
            ("Tension", "maximum_principal_tensile_stress_pa"),
            ("Compression (signed)", "maximum_principal_compressive_stress_pa"),
        ):
            lines.append(
                f"| {label} | {refined[key]:.6g} | {result[key]:.6g} | {result[key] - refined[key]:.6g} |"
            )
        lines.extend([
            "",
            "Global extrema can move: their difference is not local stress relief.",
            "Negative compression means larger compressive magnitude. Raw element",
            "stresses and recovered stresses are distinct measures; compare like with like.",
            "The configured tensile strength is a reporting reference, not an imposed stress cap.",
            "",
            "| Fixed refined-baseline location | Coordinates, m | Baseline stress, Pa | Cracked stress, Pa |",
            "|---|---|---:|---:|",
        ])
        for label, name, index, key in (
            ("Tensile peak", "refined_baseline_tensile_peak", 2, "maximum_principal_tensile_stress_pa"),
            ("Compressive peak", "refined_baseline_compressive_peak", 0, "maximum_principal_compressive_stress_pa"),
        ):
            sample = result["matched_point_stresses"][name]
            point = ", ".join(f"{value:.3f}" for value in sample["coordinate_xyz_m"])
            lines.append(
                f"| {label} | ({point}) | {refined[key]:.6g} | {sample['principal_stresses_pa'][index]:.6g} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_engineering_results(
    directory: Path,
    configuration: dict[str, Any],
    result: dict[str, Any],
    refinement: dict[str, Any],
    placement: list[dict[str, Any]],
    import_report: dict[str, Any],
    *,
    configuration_sha256: str,
    environment: dict[str, Any],
    started_at_utc: str,
    completed_at_utc: str | None = None,
) -> Path:
    """Atomically write results.md for a completed phase-field run."""
    completed = completed_at_utc or datetime.now(timezone.utc).isoformat()
    contents = build_engineering_results(
        configuration,
        result,
        refinement,
        placement,
        import_report,
        configuration_sha256=configuration_sha256,
        environment=environment,
        started_at_utc=started_at_utc,
        completed_at_utc=completed,
    )
    path = directory / "results.md"
    temporary = directory / "results.md.tmp"
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)
    return path


def write_engineering_results_from_directory(directory: Path) -> Path:
    """Regenerate results.md from the JSON artifacts in a completed job."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not str(manifest.get("status", "")).startswith("completed"):
        raise ValueError("results.md can only be generated for a completed job")
    return write_engineering_results(
        directory,
        json.loads((directory / "config.json").read_text(encoding="utf-8")),
        json.loads((directory / "phase_field_report.json").read_text(encoding="utf-8")),
        json.loads((directory / "refinement_report.json").read_text(encoding="utf-8")),
        json.loads((directory / "crack_placement.json").read_text(encoding="utf-8")),
        json.loads((directory / "import_report.json").read_text(encoding="utf-8")),
        configuration_sha256=manifest["configuration_sha256"],
        environment=manifest["environment"],
        started_at_utc=manifest["started_at_utc"],
        completed_at_utc=manifest["completed_at_utc"],
    )
