"""Generate the horizontal upstream crack from the consumer's Elmer baseline."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DEFAULTS = Path(__file__).with_name("fracture_defaults.json")


def plinth_intersections(contours: list[dict], z: float) -> tuple[float, float]:
    """Interpolate the two chainages where the plinth reaches signed crest-relative Z."""
    if not math.isfinite(z) or len(contours) < 2:
        raise ValueError("A finite crack Z and at least two plinth contours are required.")
    points = [(float(row["chainage"]), float(row["height"])) for row in contours]
    if not all(math.isfinite(s) and math.isfinite(h) for s, h in points):
        raise ValueError("Plinth contours must be finite.")
    if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
        raise ValueError("Plinth chainages must be strictly increasing.")
    crossings = []
    for (s0, h0), (s1, h1) in zip(points, points[1:]):
        if h0 == h1:
            if z == h0:
                raise ValueError("Crack Z coincides with a horizontal plinth segment.")
            continue
        if min(h0, h1) <= z <= max(h0, h1):
            s = s0 + (z - h0) * (s1 - s0) / (h1 - h0)
            if not any(math.isclose(s, other, abs_tol=1e-9) for other in crossings):
                crossings.append(s)
    if len(crossings) != 2:
        raise ValueError(f"Crack Z={z} requires exactly two plinth intersections; found {len(crossings)}.")
    left, right = sorted(crossings)
    interior = [h for s, h in points if left < s < right]
    if not interior or any(h >= z for h in interior):
        raise ValueError("Plinth intersections do not bound one continuous wall section.")
    return left, right


def generate_crack(config: dict, baseline: dict):
    """Use the global element-mean principal-stress peak, as in the baseline report."""
    from crack_field import CrackSpecification, PhaseFieldSolverOptions

    peak = baseline["max_principal_stress"]
    if not math.isfinite(peak["value_pa"]) or peak["value_pa"] <= 0:
        raise ValueError("Elmer baseline has no positive tensile-stress peak; no automatic crack can be placed.")
    z = float(peak["centroid_xyz_m"][2])
    geometry = config["geometry"]
    wall = geometry["wall"]
    upstream = float(wall["upstream_radius_m"])
    thickness = float(wall["radial_width_m"])
    chainage_radius = float(wall["centerline_radius_m"])
    if not all(math.isfinite(v) and v > 0 for v in (upstream, thickness, chainage_radius)):
        raise ValueError("Wall radii and thickness must be finite and positive.")
    inner = upstream - 0.75 * thickness
    if inner <= 0:
        raise ValueError("Automatic crack inner radius must be positive.")
    left, right = plinth_intersections(geometry["plinth"]["contours"], z)
    theta0, theta1 = left / chainage_radius, right / chainage_radius
    if not 0 < theta1 - theta0 < math.pi:
        raise ValueError("Automatic crack requires an angular span between zero and pi.")
    middle = (theta0 + theta1) / 2
    radius = (inner + upstream) / 2
    defaults = json.loads(DEFAULTS.read_text(encoding="utf-8"))
    options = PhaseFieldSolverOptions(**defaults.pop("solver"))
    specification = CrackSpecification(
        name="automatic_peak_tension", type="phase_field", shape="annular_sector",
        centre_m=(radius * math.sin(middle), radius * math.cos(middle), z),
        size_m=(upstream - inner, radius * (theta1 - theta0), 0.002),
        normal=(0.0, 0.0, 1.0),
        annular_sector=(inner, upstream, theta0, theta1),
        **defaults,
    )
    report = {
        "method": "horizontal annular-sector crack at global peak principal-stress element centroid Z",
        "source_peak": peak,
        "chainage_intersections_m": [left, right],
        "upstream_intersections_xyz_m": [
            [upstream * math.sin(theta), upstream * math.cos(theta), z]
            for theta in (theta0, theta1)
        ],
        "opening_m": 0.002,
        "depth_fraction": 0.75,
        "depth_m": upstream - inner,
        "annular_sector": list(specification.annular_sector),
        "defaults": json.loads(DEFAULTS.read_text(encoding="utf-8")),
        "report_crack": {
            "name": specification.name,
            "central_crack_pos_m": dict(zip(("x", "y", "z"), specification.centre_m)),
            "size_m": dict(zip(("x", "y", "z"), specification.size_m)),
            "orientation": {"normal_x": 0.0, "normal_y": 0.0, "normal_z": 1.0},
            "phase_field_parameters": {
                "length_scale_m": specification.length_scale_m,
                "critical_energy_release_rate_J_m2": specification.critical_energy_release_rate_j_m2,
                "residual_stiffness": specification.residual_stiffness,
            },
            "loading": {
                "internal_pressure_pa": specification.internal_pressure_pa,
                "load_steps": specification.load_steps,
            },
        },
        "engineering_validated": False,
    }
    return specification, options, report


def sector_signed_distance(points: np.ndarray, sector: tuple, z: float, opening: float) -> np.ndarray:
    """Exact distance to a horizontal annular-sector prism (x=r sin(theta))."""
    inner, outer, start, end = sector
    xy = points[:, :2]
    radius = np.linalg.norm(xy, axis=1)
    angle = np.arctan2(xy[:, 0], xy[:, 1])
    middle = (start + end) / 2
    angle = middle + (angle - middle + np.pi) % (2 * np.pi) - np.pi
    clamped_angle = np.clip(angle, start, end)
    direction = np.column_stack((np.sin(clamped_angle), np.cos(clamped_angle)))
    distances = [np.linalg.norm(xy - r * direction, axis=1) for r in (inner, outer)]
    for theta in (start, end):
        axis = np.array([math.sin(theta), math.cos(theta)])
        closest = np.clip(xy @ axis, inner, outer)[:, None] * axis
        distances.append(np.linalg.norm(xy - closest, axis=1))
    planar = np.min(distances, axis=0)
    inside = (radius >= inner) & (radius <= outer) & (angle >= start) & (angle <= end)
    planar = np.where(inside, -planar, planar)
    vertical = np.abs(points[:, 2] - z) - opening / 2
    components = np.column_stack((planar, vertical))
    return np.linalg.norm(np.maximum(components, 0), axis=1) + np.minimum(components.max(axis=1), 0)
