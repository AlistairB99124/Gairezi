"""Runtime-neutral crack and solver inputs shared with native solver adapters."""
from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import hashlib
import json
import math

import numpy as np

PHASE_FIELD_MODELS = ("AT1", "AT2")


def implied_tensile_strength(model: str, young_pa: float, gc_j_m2: float, length_scale_m: float) -> float:
    """Peak stress of the homogeneous 1-D phase-field solution (Pa).

    AT1 (w = alpha, c_w = 8/3) is linear elastic up to sqrt(3 E Gc / (8 l)).
    AT2 (w = alpha^2, c_w = 2) has no elastic phase; its stress peaks at
    sqrt(27 E Gc / (256 l)) after diffuse damage has already started.
    """
    _check_calibration_inputs(model, young_pa, gc_j_m2, length_scale_m)
    factor = 3.0 / 8.0 if model == "AT1" else 27.0 / 256.0
    return math.sqrt(factor * young_pa * gc_j_m2 / length_scale_m)


def strength_calibrated_length_scale(model: str, young_pa: float, gc_j_m2: float, tensile_strength_pa: float) -> float:
    """Length scale (m) whose homogeneous peak stress equals the tensile strength.

    Inverts :func:`implied_tensile_strength`; for AT1 this is 3/8 of the Irwin
    length E Gc / ft^2 (Tanne et al. 2018, JMPS 110:80-99).
    """
    _check_calibration_inputs(model, young_pa, gc_j_m2, tensile_strength_pa)
    factor = 3.0 / 8.0 if model == "AT1" else 27.0 / 256.0
    return factor * young_pa * gc_j_m2 / tensile_strength_pa**2


def _check_calibration_inputs(model: str, *values: float) -> None:
    if model not in PHASE_FIELD_MODELS:
        raise ValueError(f"Unknown phase-field model {model!r}; expected one of {PHASE_FIELD_MODELS}")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in values):
        raise ValueError("Phase-field calibration inputs must be finite and positive")


STRENGTH_CALIBRATION_REFERENCE = "Tanne et al. (2018), J. Mech. Phys. Solids 110:80-99"


def calibrate_phase_field_defaults(defaults: dict, concrete: dict) -> tuple[dict, dict]:
    """Replace ``length_scale_basis`` with the strength-calibrated ``length_scale_m``.

    ``concrete`` is ``config["material"]["concrete"]``. Returns the parameter
    dictionary (without the basis key) and a JSON-ready calibration record.
    """
    parameters = dict(defaults)
    basis = parameters.pop("length_scale_basis", None)
    model = parameters["model"]
    young = float(concrete["youngs_modulus"])
    tensile_strength = float(concrete["tensile_strength_pa"])
    gc = float(parameters["critical_energy_release_rate_j_m2"])
    if basis is None and "length_scale_m" in parameters:
        # A fixed fracture_debug length scale: record the strength it implies instead.
        basis = "fixed (fracture_debug.length_scale_m)"
        length_scale = float(parameters["length_scale_m"])
    elif basis != "concrete_tensile_strength":
        raise ValueError(f"Unsupported length_scale_basis {basis!r}; expected 'concrete_tensile_strength'.")
    else:
        length_scale = strength_calibrated_length_scale(model, young, gc, tensile_strength)
    parameters["length_scale_m"] = length_scale
    return parameters, {
        "basis": basis,
        "model": model,
        "reference": STRENGTH_CALIBRATION_REFERENCE,
        "youngs_modulus_pa": young,
        "critical_energy_release_rate_j_m2": gc,
        "configured_tensile_strength_pa": tensile_strength,
        "length_scale_m": length_scale,
        "implied_homogeneous_tensile_strength_pa": implied_tensile_strength(model, young, gc, length_scale),
    }


@dataclass(frozen=True)
class CrackSpecification:
    """One physical seed with explicit FEniCSx constitutive/loading controls."""

    type: str
    shape: str
    centre_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    length_scale_m: float
    critical_energy_release_rate_j_m2: float
    internal_pressure_pa: float
    name: str = ""
    model: str = "AT2"
    energy_split: str = "volumetric_deviatoric"
    residual_stiffness: float = 1e-8
    load_steps: int = 1
    annular_sector: tuple[float, float, float, float] | None = None

    @property
    def half_extents_m(self) -> np.ndarray:
        return np.asarray(self.size_m, dtype=float) / 2.0

    @property
    def unit_normal(self) -> np.ndarray:
        normal = np.asarray(self.normal, dtype=float)
        return normal / np.linalg.norm(normal)


@dataclass(frozen=True)
class PhaseFieldParameters:
    """Constitutive and loading controls for an initially intact baseline."""

    length_scale_m: float
    critical_energy_release_rate_j_m2: float
    internal_pressure_pa: float
    load_steps: int
    model: str = "AT2"
    energy_split: str = "volumetric_deviatoric"
    residual_stiffness: float = 1e-8


@dataclass(frozen=True)
class PhaseFieldSolverOptions:
    maximum_staggered_iterations: int = 50
    staggered_tolerance: float = 1e-5
    mechanics_relative_tolerance: float = 1e-8
    mechanics_maximum_iterations: int = 2000
    damage_gradient_tolerance: float = 1e-8
    damage_maximum_iterations: int = 500
    field_output_interval: int = 1
    minimum_elements_per_length_scale: float = 2.0
    maximum_refinement_levels: int = 5
    maximum_refined_cells: int = 4_000_000


# Debug overrides of fracture_defaults.json carried by the dam config's optional
# ``fracture_debug`` block, so each job freezes its own values (process of elimination).
FRACTURE_DEBUG_KEYS = (
    "model", "critical_energy_release_rate_j_m2", "length_scale_m", "residual_stiffness",
    "crack_water_pressure_factor", "load_steps", "solver", "aster",
)
FRACTURE_DEBUG_ASTER_KEYS = ("band_width_m", "softening_p")
DEFAULT_RESIDUAL_STIFFNESS = 1e-8
DEFAULT_CRACK_WATER_PRESSURE_FACTOR = 1.0


def default_fracture_debug(defaults: dict, aster_defaults: dict) -> dict:
    """Fully populated ``fracture_debug`` block holding the current shipped defaults.

    ``length_scale_m`` null keeps the strength calibration (Tanne et al. 2018);
    a crack water factor of 1 applies the full hydrostatic head in the crack.
    """
    return {
        "model": defaults["model"],
        "critical_energy_release_rate_j_m2": defaults["critical_energy_release_rate_j_m2"],
        "length_scale_m": defaults.get("length_scale_m"),
        "residual_stiffness": defaults.get("residual_stiffness", DEFAULT_RESIDUAL_STIFFNESS),
        "crack_water_pressure_factor": DEFAULT_CRACK_WATER_PRESSURE_FACTOR,
        "load_steps": defaults["load_steps"],
        "solver": {field.name: defaults["solver"].get(field.name, field.default)
                   for field in dataclasses.fields(PhaseFieldSolverOptions)},
        "aster": {key: aster_defaults[key] for key in FRACTURE_DEBUG_ASTER_KEYS},
    }


def apply_fracture_debug(defaults: dict, config: dict) -> tuple[dict, float]:
    """fracture_defaults.json with ``config["fracture_debug"]`` overrides applied.

    Returns the defaults dictionary (same shape as the JSON file) and the crack water
    pressure factor (0 = dry crack, 1 = full hydrostatic head). Missing keys keep the
    defaults. Elmer/model_config.py validates ranges before a job is accepted.
    """
    parameters = json.loads(json.dumps(defaults))
    debug = config.get("fracture_debug")
    if debug is None:
        return parameters, DEFAULT_CRACK_WATER_PRESSURE_FACTOR
    if not isinstance(debug, dict):
        raise ValueError("fracture_debug must be an object")
    unknown = set(debug) - set(FRACTURE_DEBUG_KEYS)
    if unknown:
        raise ValueError(f"Unsupported fracture_debug fields: {sorted(unknown)}")
    if "model" in debug:
        if debug["model"] not in PHASE_FIELD_MODELS:
            raise ValueError(f"fracture_debug.model must be one of {PHASE_FIELD_MODELS}")
        parameters["model"] = debug["model"]
    if "critical_energy_release_rate_j_m2" in debug:
        parameters["critical_energy_release_rate_j_m2"] = _debug_number(
            debug, "critical_energy_release_rate_j_m2")
    if debug.get("length_scale_m") is not None:
        parameters.pop("length_scale_basis", None)
        parameters["length_scale_m"] = _debug_number(debug, "length_scale_m")
    if "residual_stiffness" in debug:
        parameters["residual_stiffness"] = _debug_number(debug, "residual_stiffness")
    if "load_steps" in debug:
        parameters["load_steps"] = _debug_integer(debug, "load_steps", 1)
    solver = debug.get("solver") or {}
    if not isinstance(solver, dict):
        raise ValueError("fracture_debug.solver must be an object")
    known = {field.name: field.type for field in dataclasses.fields(PhaseFieldSolverOptions)}
    for key, value in solver.items():
        if key not in known:
            raise ValueError(f"Unsupported fracture_debug.solver field {key!r}")
        parameters["solver"][key] = (
            _debug_integer(solver, key, 0 if key == "maximum_refinement_levels" else 1)
            if known[key] in (int, "int") else _debug_number(solver, key)
        )
    factor = DEFAULT_CRACK_WATER_PRESSURE_FACTOR
    if "crack_water_pressure_factor" in debug:
        factor = _debug_number(debug, "crack_water_pressure_factor", allow_zero=True)
        if factor > 1:
            raise ValueError("fracture_debug.crack_water_pressure_factor must lie between 0 and 1")
    return parameters, factor


def _debug_number(values: dict, key: str, allow_zero: bool = False) -> float:
    value = values[key]
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or (value == 0 and not allow_zero)):
        raise ValueError(f"fracture_debug {key} must be a finite {'nonnegative' if allow_zero else 'positive'} number")
    return float(value)


def _debug_integer(values: dict, key: str, minimum: int) -> int:
    value = values[key]
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"fracture_debug {key} must be an integer of at least {minimum}")
    return value

def load_schedule(load_steps: int, final_load_substeps: int = 1) -> list[float]:
    """Preserve the configured ramp, optionally subdividing only its last interval."""
    for name, value in (("load_steps", load_steps), ("final_load_substeps", final_load_substeps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return [step / load_steps for step in range(1, load_steps)] + [
        (load_steps - 1 + substep / final_load_substeps) / load_steps
        for substep in range(1, final_load_substeps + 1)
    ]


def freeze_selected_seed(
    specification: CrackSpecification, selection: dict, final_load_substeps: int = 1,
) -> dict:
    """Freeze physical placement/loading, not either backend's damage profile."""
    if specification.annular_sector is None or specification.shape != "annular_sector":
        raise ValueError("A shared UI seed requires an annular-sector specification")
    payload = {
        "schema_version": 1,
        "selected_location_m": dict(selection["selected_location_m"]),
        "seed_face": selection["seed_face"],
        "geometry": {
            "shape": "horizontal_annular_sector",
            "annular_sector": list(specification.annular_sector),
            "elevation_m": specification.centre_m[2],
            "seed_thickness_m": specification.size_m[2],
            "normal": list(specification.normal),
            "region_ids": list(selection["required_region_ids"]),
        },
        "loading": {
            "load_factors": [0., *load_schedule(specification.load_steps, final_load_substeps)],
            "internal_pressure_pa": specification.internal_pressure_pa,
            "fracture_energy_j_m2": specification.critical_energy_release_rate_j_m2,
        },
    }
    contract = {"payload": payload, "sha256": _seed_hash(payload)}
    validate_selected_seed(contract)
    return contract


def _seed_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def validate_selected_seed(contract: dict) -> dict:
    """Reject edited or unsupported shared placement before a backend uses it."""
    payload = contract["payload"]
    if contract["sha256"] != _seed_hash(payload):
        raise ValueError("Shared UI crack seed hash mismatch")
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported shared UI crack seed schema")
    geometry, loading = payload["geometry"], payload["loading"]
    if geometry["shape"] != "horizontal_annular_sector" or geometry["region_ids"] != [1]:
        raise ValueError("Shared UI crack seed must be a wall-only horizontal annular sector")
    sector = np.asarray(geometry["annular_sector"], dtype=float)
    if sector.shape != (4,) or not np.isfinite(sector).all():
        raise ValueError("Shared crack sector must contain four finite values")
    inner, outer, start, end = sector
    if not (0 < inner < outer and 0 <= start < end <= math.pi):
        raise ValueError("Invalid shared UI crack sector extents")
    if geometry["normal"] != [0., 0., 1.] or payload["seed_face"] not in ("upstream", "downstream"):
        raise ValueError("Unsupported shared UI seed face or normal")
    location = payload["selected_location_m"]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) for value in
               (location["y"], location["z"], geometry["elevation_m"], geometry["seed_thickness_m"],
                loading["internal_pressure_pa"], loading["fracture_energy_j_m2"])):
        raise ValueError("Shared seed coordinates and loading must be finite numbers")
    if (geometry["elevation_m"] != location["z"] or geometry["seed_thickness_m"] <= 0
            or loading["fracture_energy_j_m2"] <= 0 or loading["internal_pressure_pa"] < 0):
        raise ValueError("Invalid shared seed elevation, thickness or loading")
    factors = np.asarray(loading["load_factors"], dtype=float)
    if (factors.ndim != 1 or len(factors) < 2 or not np.isfinite(factors).all()
            or factors[0] != 0 or factors[-1] != 1 or np.any(np.diff(factors) <= 0)):
        raise ValueError("Shared seed load factors must increase from zero to one")
    return payload
