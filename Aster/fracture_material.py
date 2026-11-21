"""Auditable engineering inputs for native Aster ENDO_FISS_EXP, in SI units."""
from __future__ import annotations

import json
import math
from pathlib import Path

from Elmer.model_config import load_config

DEFAULTS = Path(__file__).with_name("fracture_defaults.json")
DOCUMENTATION = (
    "https://codeaster.gitlab.io/doc/docaster/manuals/man_u/u4/u4.42.07/"
    "Mot_clef_facteur_ENDO_FISS_EXP.html"
)


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _positive_root(function, name: str) -> float:
    low, high = 1e-12, 1e-6
    value_low = function(low)
    while value_low * function(high) > 0.0 and high < 1e3:
        high *= 2.0
    if value_low * function(high) > 0.0:
        raise ValueError(f"Could not bracket {name} for the Code_Aster damage model")
    for _ in range(120):
        middle = (low + high) / 2.0
        if value_low * function(middle) > 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def internal_parameters(parameters: dict) -> dict:
    """Convert engineering inputs without the installed macro's high-ratio overflow."""
    e, nu = parameters["E"], parameters["NU"]
    ft, fc = parameters["FT"], parameters["FC"]
    beta, p = 0.1, parameters["P"]
    if fc / ft < 5.83:
        raise ValueError("ENDO_FISS_EXP requires compressive/tensile strength ratio >= 5.83")
    a = math.sqrt(2.0 / 3.0 + 3.0 * beta ** 2)
    ratio = fc / ft
    length = a * (ratio - 1.0)

    def damage_root(x: float) -> float:
        negative = 0.0 if -2.0 * ratio * x < -745.0 else math.exp(-2.0 * ratio * x)
        positive = math.inf if 2.0 * x > 709.0 else math.exp(2.0 * x)
        return length * x + math.sqrt(2.0 + negative) - math.sqrt(2.0 + positive)

    x_damage = _positive_root(damage_root, "tensile-damage parameter")
    tau = a * x_damage + math.sqrt(2.0 + math.exp(2.0 * x_damage))
    sig0 = ft / x_damage
    directional = (1.0 - nu, nu, nu)
    confined_length = math.sqrt(
        2.0 / 3.0 * (1.0 - 2.0 * nu) ** 2
        + 3.0 * beta ** 2 * (1.0 + nu) ** 2
    )

    def confined_root(x: float) -> float:
        exponentials = [
            math.inf if x * component > 709.0 else math.exp(x * component)
            for component in directional
        ]
        return confined_length * x + math.sqrt(sum(value * value for value in exponentials)) - tau

    x_confined = _positive_root(confined_root, "confined-tension parameter")
    sigc = x_confined * sig0 * (1.0 - nu)
    half_width = parameters["LARG_BANDE"] / 2.0
    rigidity = e * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    m = 1.5 * rigidity * parameters["GF"] / (half_width * sigc ** 2)
    if m < p + 2.0:
        raise ValueError(f"ENDO_FISS_EXP M={m:g} must be at least P+2={p + 2:g}")
    restoration = -1.0 / (fc / e * math.log(parameters["REST_RIGI_FC"]))
    return {
        "M": m, "P": p, "Q": parameters["Q"], "K": 0.75 * parameters["GF"] / half_width,
        "TAU": tau, "SIG0": sig0, "BETA": beta,
        "REST_RIGIDITE": restoration, "COEF_RIGI_MINI": parameters["COEF_RIGI_MINI"],
        "C_GRAD_VARI": 0.375 * parameters["GF"] * half_width,
        "PENA_LAGR": 1000.0 * 0.75 * parameters["GF"] / half_width,
    }


def material_contract(
    config: dict,
    *,
    band_width_m: float | None = None,
    softening_p: float | None = None,
    defaults_path: Path = DEFAULTS,
) -> dict:
    """Read shared strengths/energy without altering the source configuration.

    Overrides are for sensitivity studies, not an implicit fit to elastic peaks.
    The native macro additionally checks its coupled admissibility conditions.
    """
    concrete = load_config(config)["material"]["concrete"]
    debug = config.get("fracture_debug") or {}
    aster_debug = debug.get("aster") or {}
    if "compressive_strength_pa" not in concrete:
        raise ValueError("material.concrete.compressive_strength_pa is required for Aster damage")
    if "cohesive_fracture_energy_j_m2" not in concrete:
        raise ValueError("material.concrete.cohesive_fracture_energy_j_m2 is required for Aster damage")
    defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
    # Precedence: explicit argument (sensitivity studies), then fracture_debug, then defaults.
    if band_width_m is None:
        band_width_m = aster_debug.get("band_width_m", defaults["band_width_m"])
    if softening_p is None:
        softening_p = aster_debug.get("softening_p", defaults["softening_p"])
    fracture_energy = concrete["cohesive_fracture_energy_j_m2"]
    width = _positive(band_width_m, "band_width_m")
    p = _positive(softening_p, "softening_p")
    if p < 1:
        raise ValueError("softening_p must be at least 1 for ENDO_FISS_EXP")
    parameters = {
        "E": _positive(concrete["youngs_modulus"], "youngs_modulus"),
        "NU": concrete["poisson_ratio"],
        "FT": _positive(concrete["tensile_strength_pa"], "tensile_strength_pa"),
        "FC": _positive(concrete["compressive_strength_pa"], "compressive_strength_pa"),
        "GF": _positive(fracture_energy, "fracture_energy"),
        "LARG_BANDE": width,
        "P": p,
        "Q": 0.0,
        "REST_RIGI_FC": 0.9,
        "COEF_RIGI_MINI": 1e-5,
    }
    if not 0 <= parameters["NU"] < .5:
        raise ValueError("ENDO_FISS_EXP requires 0 <= poisson_ratio < 0.5")
    if parameters["FC"] / parameters["FT"] < 5.83:
        raise ValueError("ENDO_FISS_EXP requires compressive/tensile strength ratio >= 5.83")
    # U4.42.07 / the inverse P(G_INIT) in Aster 17.4's DEFI_MATER_GC.
    initial_energy = 3 * math.pi * parameters["GF"] / (4 * (p + 2) ** 1.5)
    internal = internal_parameters(parameters)
    return {
        "schema_version": 1,
        "model": "ENDO_FISS_EXP",
        "modelisation": "3D_GRAD_VARI",
        "calibration_status": defaults["calibration_status"],
        "parameters": parameters,
        "internal_parameters": internal,
        "derived": {
            "half_band_width_m": width / 2,
            "equivalent_initial_fracture_energy_j_m2": initial_energy,
            "initial_cohesive_slope_pa_per_m": -parameters["FT"] ** 2 / (2 * initial_energy),
            "hillerborg_length_m": parameters["E"] * parameters["GF"] / parameters["FT"] ** 2,
        },
        "provenance": {
            "strengths_and_elasticity": "submitted config: material.concrete",
            "fracture_energy": "submitted config: material.concrete.cohesive_fracture_energy_j_m2",
            "regularisation_and_softening": ("submitted config: fracture_debug.aster" if aster_debug
                                             else str(defaults_path.resolve())),
            "documentation": DOCUMENTATION,
            "fracture_energy_basis": (
                "Configured concrete cohesive fracture energy; not inferred from the retired "
                "FEniCSx fracture_debug phase-field energy."
            ),
            "softening_basis": "P=5 is a provisional starting value used in installed ssnv234b.comm",
            "width_basis": (
                "0.5 m is a provisional numerical full band, not a measured crack width "
                "or an equivalence to the FEniCSx phase-field length scale"
            ),
        },
        "limitations": [
            "Not calibrated to concrete fracture tests; requires width, mesh and softening sensitivity.",
            "Configured cohesive fracture energy is not a test-calibrated dam-concrete value.",
            "Damage V1 is not a literal crack opening or directly comparable to FEniCSx phase-field damage.",
            "Material admissibility does not establish seeded dam convergence or engineering validity.",
        ],
    }
