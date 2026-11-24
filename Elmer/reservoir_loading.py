"""Physical reservoir depths (metres above the surveyed wall-seat datum)."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import math


class ReservoirLoadingError(ValueError):
    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(message)


def _object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReservoirLoadingError(path, "must be an object")
    return value


def _number(values: Mapping[str, object], key: str, path: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReservoirLoadingError(f"{path}.{key}", "must be a finite number")
    return float(value)


@dataclass(frozen=True)
class ReservoirLoading:
    depth_reference_z_m: float
    final_water_depth_m: float
    water_depth_start_m: float
    water_depth_increment_m: float
    gradient_pa_per_m: float
    tailwater_surface_z_m: float
    crest_z_m: float = 0.0
    loading_mode: str = "reservoir_filling"

    def metadata(self) -> dict:
        return dict(asdict(self), gravity_loading="constant_full_self_weight",
                    tailwater_loading="constant_configured_head")

    def targets(self) -> list[float]:
        values = [self.water_depth_start_m]
        while values[-1] < self.final_water_depth_m:
            following = min(self.final_water_depth_m,
                            round(values[-1] + self.water_depth_increment_m, 10))
            if following <= values[-1]:
                raise ReservoirLoadingError("fracture_debug.aster.water_depth_increment_m",
                                            "is too small to advance the depth schedule")
            values.append(following)
        return values

    def surface_z(self, depth: float) -> float:
        if not math.isfinite(depth) or not 0 <= depth <= self.final_water_depth_m:
            raise ValueError("Trial reservoir depth must lie between zero and final depth")
        return self.depth_reference_z_m + depth

    def crest_pressure(self, depth: float) -> float:
        return self.gradient_pa_per_m * max(self.surface_z(depth) - self.crest_z_m, 0.0)


def reservoir_loading(config: Mapping[str, object],
                      source_inputs: Mapping[str, object] | None = None) -> ReservoirLoading | None:
    """Validate an explicit filling contract; absent mode keeps legacy proportional loads."""
    debug_value = config.get("fracture_debug")
    debug = {} if debug_value is None else _object(debug_value, "fracture_debug")
    controls_value = debug.get("aster", {})
    controls = _object(controls_value, "fracture_debug.aster")
    path = "fracture_debug.aster"
    mode = controls.get("loading_mode", "proportional")
    if mode not in ("reservoir_filling", "proportional"):
        raise ReservoirLoadingError(f"{path}.loading_mode", "must be reservoir_filling or proportional")
    if mode == "proportional":
        if "loading_mode" not in controls and any(
                key in controls for key in ("water_depth_start_m", "water_depth_increment_m")):
            raise ReservoirLoadingError(f"{path}.loading_mode",
                                        "must explicitly select reservoir_filling for metre-based controls")
        return None
    start = _number(controls, "water_depth_start_m", path)
    increment = _number(controls, "water_depth_increment_m", path)
    loads = _object(config.get("loads"), "loads")
    final = _number(loads, "water_height", "loads") + _number(loads, "overflow_head", "loads")
    geometry = _object(config.get("geometry"), "geometry")
    wall = _object(geometry.get("wall"), "geometry.wall")
    capacity = _number(wall, "height_m", "geometry.wall") + _number(loads, "overflow_head", "loads")
    if final <= 0 or not math.isclose(final, capacity, rel_tol=0., abs_tol=1e-8):
        raise ReservoirLoadingError("loads.water_height",
                                    "reservoir filling requires final depth to equal wall height plus surcharge")
    if not math.isclose(_number(loads, "overflow_water_level", "loads"), final, rel_tol=0., abs_tol=1e-8):
        raise ReservoirLoadingError("loads.overflow_water_level", "must equal water height plus overflow head")
    if not 0 <= start <= final:
        raise ReservoirLoadingError(f"{path}.water_depth_start_m", "must lie between zero and final water depth")
    if increment <= 0:
        raise ReservoirLoadingError(f"{path}.water_depth_increment_m", "must be positive")
    bedrock = _object(geometry.get("bedrock"), "geometry.bedrock")
    contours = bedrock.get("contours")
    if not isinstance(contours, list) or not contours:
        raise ReservoirLoadingError("geometry.bedrock.contours", "must contain the surveyed wall-seat profile")
    # The minimum surveyed elevation lies inside the wall extent used by load_contours.
    reference = min(_number(_object(row, "geometry.bedrock.contours"), "height",
                            "geometry.bedrock.contours") for row in contours)
    gradient = _number(loads, "water_density", "loads") * abs(_number(loads, "gravity", "loads"))
    result = ReservoirLoading(reference, final, start, increment, gradient,
                              reference + _number(loads, "tailwater_head", "loads"))
    if start < final and round(start + increment, 10) <= start:
        raise ReservoirLoadingError(f"{path}.water_depth_increment_m",
                                    "is too small to advance the depth schedule")
    if source_inputs is not None:
        expected = {
            "upstream_water_surface_z_m": result.surface_z(final),
            "water_pressure_gradient_pa_per_m": gradient,
            "overflow_surcharge_pa": result.crest_pressure(final),
        }
        for key, value in expected.items():
            actual = _number(source_inputs, key, "source_loads")
            if not math.isclose(actual, value, rel_tol=1e-10, abs_tol=1e-6):
                raise ReservoirLoadingError(f"source_loads.{key}",
                                            "does not match the authoritative reservoir depth/datum")
        # Tailwater keeps the source workflow's foundation-based datum, not the upstream wall-seat datum.
        result = replace(result, tailwater_surface_z_m=_number(
            source_inputs, "tailwater_elevation_m", "source_loads"))
    return result
