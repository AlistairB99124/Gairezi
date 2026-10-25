"""Auditable engineering inputs for native Aster ENDO_FISS_EXP, in SI units."""
from __future__ import annotations

import json
import math
from pathlib import Path

from Elmer.model_config import load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = Path(__file__).with_name("fracture_defaults.json")
FENICSX_DEFAULTS = ROOT / "FEniCSx" / "fracture_defaults.json"
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


def material_contract(
    config: dict,
    *,
    band_width_m: float | None = None,
    softening_p: float | None = None,
    defaults_path: Path = DEFAULTS,
    fenicsx_defaults_path: Path = FENICSX_DEFAULTS,
) -> dict:
    """Read shared strengths/energy without altering the source configuration.

    Overrides are for sensitivity studies, not an implicit fit to elastic peaks.
    The native macro additionally checks its coupled admissibility conditions.
    """
    concrete = load_config(config)["material"]["concrete"]
    if "compressive_strength_pa" not in concrete:
        raise ValueError("material.concrete.compressive_strength_pa is required for Aster damage")
    defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
    shared = json.loads(fenicsx_defaults_path.read_text(encoding="utf-8"))
    width = _positive(defaults["band_width_m"] if band_width_m is None else band_width_m,
                      "band_width_m")
    p = _positive(defaults["softening_p"] if softening_p is None else softening_p, "softening_p")
    if p < 1:
        raise ValueError("softening_p must be at least 1 for ENDO_FISS_EXP")
    parameters = {
        "E": _positive(concrete["youngs_modulus"], "youngs_modulus"),
        "NU": concrete["poisson_ratio"],
        "FT": _positive(concrete["tensile_strength_pa"], "tensile_strength_pa"),
        "FC": _positive(concrete["compressive_strength_pa"], "compressive_strength_pa"),
        "GF": _positive(shared["critical_energy_release_rate_j_m2"], "fracture_energy"),
        "LARG_BANDE": width,
        "P": p,
        "Q": 0.0,
        "REST_RIGI_FC": 0.9,
        "COEF_RIGI_MINI": 0.0,
    }
    if not 0 <= parameters["NU"] < .5:
        raise ValueError("ENDO_FISS_EXP requires 0 <= poisson_ratio < 0.5")
    if parameters["FC"] / parameters["FT"] < 5.83:
        raise ValueError("ENDO_FISS_EXP requires compressive/tensile strength ratio >= 5.83")
    # U4.42.07 / the inverse P(G_INIT) in Aster 17.4's DEFI_MATER_GC.
    initial_energy = 3 * math.pi * parameters["GF"] / (4 * (p + 2) ** 1.5)
    return {
        "schema_version": 1,
        "model": "ENDO_FISS_EXP",
        "modelisation": "3D_GRAD_VARI",
        "calibration_status": defaults["calibration_status"],
        "parameters": parameters,
        "derived": {
            "half_band_width_m": width / 2,
            "equivalent_initial_fracture_energy_j_m2": initial_energy,
            "initial_cohesive_slope_pa_per_m": -parameters["FT"] ** 2 / (2 * initial_energy),
            "hillerborg_length_m": parameters["E"] * parameters["GF"] / parameters["FT"] ** 2,
        },
        "provenance": {
            "strengths_and_elasticity": "submitted config: material.concrete",
            "fracture_energy": str(fenicsx_defaults_path.resolve()),
            "regularisation_and_softening": str(defaults_path.resolve()),
            "documentation": DOCUMENTATION,
            "fracture_energy_basis": (
                "fib MC2010 GF = 73*fcm^0.18 with fcm = fc + 8 MPa = 28 MPa gives about 133 J/m2; "
                "raised to a provisional 200 J/m2 for large dam aggregate (published dam concrete "
                "is typically 150-400 J/m2)"
            ),
            "softening_basis": "P=5 is a provisional starting value used in installed ssnv234b.comm",
            "width_basis": (
                "0.5 m is a provisional numerical full band, not a measured crack width "
                "or an equivalence to the FEniCSx phase-field length scale"
            ),
        },
        "limitations": [
            "Not calibrated to concrete fracture tests; requires width, mesh and softening sensitivity.",
            "Shared fracture energy is a provisional fib Model Code 2010 estimate with a "
            "large-aggregate allowance, not a measured dam-concrete value.",
            "Damage V1 is not a literal crack opening or directly comparable to FEniCSx phase-field damage.",
            "Material admissibility does not establish seeded dam convergence or engineering validity.",
        ],
    }
