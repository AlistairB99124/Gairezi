"""Read the root dam configuration shared by mesh and solver builders."""
from __future__ import annotations

import json
import math
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    payload = json.loads(path.read_text())
    geometry = payload["geometry"]
    for region in ("wall", "plinth"):
        values = geometry[region]
        upstream = float(values["upstream_radius_m"])
        downstream = float(values["downstream_radius_m"])
        width = float(values["radial_width_m"])
        centerline = float(values["centerline_radius_m"])
        if not all(math.isfinite(value) for value in (upstream, downstream, width, centerline)):
            raise ValueError(f"geometry.{region} radii and width must be finite")
        if downstream <= 0 or width <= 0 or not math.isclose(upstream - downstream, width, abs_tol=1e-8):
            raise ValueError(f"geometry.{region}.radial_width_m must equal upstream_radius_m - downstream_radius_m")
        if not math.isclose(centerline, 0.5 * (upstream + downstream), abs_tol=1e-8):
            raise ValueError(f"geometry.{region}.centerline_radius_m must be midway between its faces")
    for name in ("global_element_size_m", "local_element_size_m"):
        value = float(geometry[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"geometry.{name} must be finite and positive")
    if geometry["local_element_size_m"] > geometry["global_element_size_m"]:
        raise ValueError("Local element size must not exceed global element size")
    if (
        geometry["plinth"]["downstream_radius_m"] > geometry["wall"]["downstream_radius_m"]
        or geometry["plinth"]["upstream_radius_m"] < geometry["wall"]["upstream_radius_m"]
    ):
        raise ValueError("The plinth must cover the wall radial width")
    for name, values in payload["material"].items():
        for key in ("density", "youngs_modulus", "tensile_strength_pa"):
            value = float(values[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"material.{name}.{key} must be finite and positive")
        poisson = float(values["poisson_ratio"])
        if not math.isfinite(poisson) or not -1 < poisson < 0.5:
            raise ValueError(f"material.{name}.poisson_ratio must lie between -1 and 0.5")
    loads = payload["loads"]
    for key in ("gravity", "water_height", "tailwater_head", "overflow_head", "overflow_water_level", "water_density"):
        if not math.isfinite(float(loads[key])):
            raise ValueError(f"loads.{key} must be finite")
    if loads["water_density"] <= 0 or any(loads[key] < 0 for key in ("water_height", "tailwater_head", "overflow_head")):
        raise ValueError("Water density must be positive and water heads nonnegative")
    if not math.isclose(loads["overflow_water_level"], loads["water_height"] + loads["overflow_head"], abs_tol=1e-8):
        raise ValueError("loads.overflow_water_level must equal water_height + overflow_head")
    for region in ("bedrock", "plinth"):
        for direction, value in geometry[region]["support"]["spring_n_per_m3"].items():
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"geometry.{region}.support spring {direction} must be finite and nonnegative")
    return payload
