"""Report native planar seed refinement separately from nonlinear convergence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Aster.source_contract import sha256


def zero_load_planar_reference(z, initial_damage: float, seed_thickness_m: float, band_width_m: float):
    """R5.03.28: c*a''=k in the active zone, a'=0 at its free boundary."""
    parameters = np.asarray([initial_damage, seed_thickness_m, band_width_m], dtype=float)
    z = np.asarray(z, dtype=float)
    if (not np.isfinite(parameters).all() or not 0 < initial_damage < 1
            or seed_thickness_m <= 0 or band_width_m <= 0 or not np.isfinite(z).all()):
        raise ValueError("Invalid planar zero-load reference parameters")
    half_band = band_width_m / 2
    distance = np.maximum(np.abs(z) - seed_thickness_m / 2, 0)
    profile = np.maximum(np.sqrt(initial_damage) - distance / half_band, 0) ** 2
    return profile, {
        "scalar_damage_integral_m3_for_one_m2_area": (
            initial_damage * seed_thickness_m + 2 * half_band * initial_damage ** 1.5 / 3),
        "support_half_width_m": seed_thickness_m / 2 + half_band * np.sqrt(initial_damage),
        "equation": "c*a''=k; k=3*GF/(4*D), c=3*GF*D/8, D=LARG_BANDE/2",
        "scope": "zero displacement, infinite planar seed, continuum obstacle equilibrium",
        "not_an_initial_damage_assignment": True,
        "documentation": "https://codeaster.gitlab.io/doc/docaster/manuals/man_r/r5/r5.03.28/Mod_le_continu.html",
    }


def refinement_checks(rows: list[dict], relative_tolerance: float) -> dict:
    """Require two consecutive refinements, for both damage integral and profile."""
    if (isinstance(relative_tolerance, bool) or not np.isfinite(relative_tolerance)
            or not 0 < relative_tolerance < 1):
        raise ValueError("Relative refinement tolerance must lie strictly between zero and one")
    if len(rows) < 2:
        raise ValueError("At least two explicit successful resolutions are required")
    comparisons = []
    for first, second in zip(rows, rows[1:]):
        if second["normal_divisions"] != 2 * first["normal_divisions"]:
            raise ValueError("Consecutive specimens must double the normal resolution")
        integrals = np.array([first["equilibrium_integral_m3"], second["equilibrium_integral_m3"]])
        if not np.isfinite(integrals).all() or np.any(integrals <= 0):
            raise ValueError("Equilibrium damage integrals must be finite and positive")
        profiles = [np.asarray(row["profile"], dtype=float) for row in (first, second)]
        for profile in profiles:
            if (profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 2
                    or not np.isfinite(profile).all() or np.any(np.diff(profile[:, 0]) <= 0)
                    or np.any((profile[:, 1] < -1e-7) | (profile[:, 1] > 1 + 1e-7))):
                raise ValueError("Expected ordered, finite, bounded planar profiles")
        # Use the same physical grid for both piecewise-linear profile interpolants.
        z = np.unique(np.concatenate([profile[:, 0] for profile in profiles]))
        low = max(profile[0, 0] for profile in profiles)
        high = min(profile[-1, 0] for profile in profiles)
        z = z[(z >= low) & (z <= high)]
        if len(z) < 2:
            raise ValueError("Specimen profile supports do not overlap")
        values = [np.interp(z, profile[:, 0], profile[:, 1]) for profile in profiles]
        denominator = float(np.trapezoid(values[0] ** 2, z))
        if denominator <= 0:
            raise ValueError("Zero reference profile norm")
        profile_error = float(np.sqrt(np.trapezoid((values[1] - values[0]) ** 2, z) / denominator))
        integral_error = float(abs(integrals[1] - integrals[0]) / integrals[0])
        comparisons.append({
            "coarse_normal_divisions": first["normal_divisions"],
            "fine_normal_divisions": second["normal_divisions"],
            "relative_integral_change": integral_error,
            "relative_profile_l2_change": profile_error,
            "within_requested_tolerance": bool(max(integral_error, profile_error) <= relative_tolerance),
            "profile_measure": "piecewise-linear interpolation over common Gauss-profile support",
        })
    met = len(comparisons) >= 2 and all(row["within_requested_tolerance"] for row in comparisons[-2:])
    return {
        "status": "successive_refinement_tolerance_met" if met else "normal_resolution_sensitivity_unresolved",
        "relative_tolerance": relative_tolerance, "required_consecutive_refinements": 2,
        "comparisons": comparisons, "engineering_validated": False,
        "limitations": [
            "A successive-refinement tolerance is not proof of an asymptotic mesh limit.",
            "Planar seed equilibration does not validate finite crack edges, propagation or dam safety.",
            "Scalar damage integral is not physical crack volume or opening.",
        ],
    }


def report(directories: list[Path], output: Path, relative_tolerance: float = .01) -> dict:
    rows = []
    contract = None
    for directory in directories:
        path = directory / "benchmark_report.json"
        evidence = json.loads(path.read_text(encoding="utf-8"))
        inputs_path = directory / "inputs.json"
        if (evidence["status"] != "seed_placement_and_irreversibility_passed"
                or not evidence.get("specimen_divisions")
                or sha256(inputs_path) != evidence["inputs_sha256"]):
            raise ValueError(f"Missing, changed or unsuccessful planar specimen evidence: {directory}")
        inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
        if inputs.get("finite_sector", False):
            raise ValueError("Finite-sector specimens do not satisfy the infinite-planar reference")
        native_path = directory / "native_seed.json"
        native = json.loads(native_path.read_text(encoding="utf-8"))
        shared = {key: inputs[key] for key in ("shared_seed", "material", "centre_m", "times")}
        if contract is None:
            contract = shared
        elif contract != shared:
            raise ValueError("Specimen physical seed, material, domain centre or time history differs")
        metrics = evidence["specimen_metrics"]
        from Aster.seed_benchmark import specimen_metrics
        recalculated = specimen_metrics(native, evidence["specimen_divisions"], inputs["centre_m"][2],
                                        evidence.get("lateral_divisions", 8),
                                        evidence.get("normal_grading_power", 1))
        if metrics != recalculated:
            raise ValueError("Persisted specimen metrics differ from the native history")
        if (metrics["maximum_zero_load_equilibrium_change"] > 1e-7
                or metrics["profile_reflection_error"] > 1e-6
                or metrics["maximum_damage_beyond_0_3_m"] > 1e-7):
            raise ValueError("Specimen has not passed equilibrium, symmetry and remote-damage gates")
        z = np.round(np.asarray(native["audit"]["coordinates_m"])[:, 2] - inputs["centre_m"][2], 10)
        levels, inverse = np.unique(z, return_inverse=True)
        damage = np.asarray(native["history"][2]["damage_values"])
        mean = np.bincount(inverse, weights=damage) / np.bincount(inverse)
        geometry = inputs["shared_seed"]["payload"]["geometry"]
        exact_profile, exact = zero_load_planar_reference(
            levels, native["audit"]["initial_damage"], geometry["seed_thickness_m"],
            inputs["material"]["parameters"]["LARG_BANDE"],
        )
        if exact["support_half_width_m"] >= .5:
            raise ValueError("Analytic planar support reaches the specimen boundary")
        profile_norm = float(np.trapezoid(exact_profile ** 2, levels))
        reference_error = float(np.sqrt(np.trapezoid((mean - exact_profile) ** 2, levels) / profile_norm))
        rows.append({
            "directory": str(directory.resolve()), "report_sha256": sha256(path),
            "native_history_sha256": sha256(native_path),
            "normal_divisions": evidence["specimen_divisions"],
            "lateral_divisions": evidence.get("lateral_divisions", 8),
            "normal_grading_power": evidence.get("normal_grading_power", 1),
            "equilibrium_integral_m3": metrics["integrated_damage_volume_m3_by_state"][2],
            "profile": np.column_stack([levels, mean]).tolist(),
            "continuum_reference": {
                **exact,
                "relative_integral_error": abs(metrics["integrated_damage_volume_m3_by_state"][2] /
                                               exact["scalar_damage_integral_m3_for_one_m2_area"] - 1),
                "relative_profile_l2_error": reference_error,
            },
        })
    if len({row["lateral_divisions"] for row in rows}) != 1:
        raise ValueError("Normal-resolution sequence must hold lateral discretisation fixed")
    if len({row["normal_grading_power"] for row in rows}) != 1:
        raise ValueError("Normal-resolution sequence must hold its grading policy fixed")
    result = refinement_checks(rows, relative_tolerance)
    finest = rows[-1]["continuum_reference"]
    result["finest_continuum_reference_within_tolerance"] = bool(
        max(finest["relative_integral_error"], finest["relative_profile_l2_error"]) <= relative_tolerance)
    result["continuum_reference_scope"] = (
        "Independent zero-load planar obstacle solution; separate from the two-successive-refinements test")
    result["specimens"] = [{key: value for key, value in row.items() if key != "profile"} for row in rows]
    result["shared_seed_sha256"] = contract["shared_seed"]["sha256"]
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=.01)
    args = parser.parse_args()
    print(json.dumps(report(args.directories, args.output, args.relative_tolerance), indent=2))
