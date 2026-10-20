"""Generate a local horizontal crack through wall and plinth at the Elmer concrete peak tension."""
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
    """Seed a local horizontal crack across the wall/plinth junction at the concrete peak.

    The peak is the baseline's concrete (wall and plinth) maximum principal stress. The crack
    lies at the peak elevation, runs inward from the concrete face nearer the peak, and its
    arc is centred on the peak but shifted so it also spans the wall/plinth junction.
    """
    from crack_field import CrackSpecification, PhaseFieldSolverOptions

    peak = baseline["max_principal_stress"]
    if not math.isfinite(peak["value_pa"]) or peak["value_pa"] <= 0:
        raise ValueError("Elmer baseline has no positive tensile-stress peak; no automatic crack can be placed.")
    # Three-body baselines report a nodal position; older single-body ones an element centroid.
    peak_position = np.asarray(peak.get("position_xyz_m", peak.get("centroid_xyz_m")), dtype=float)
    if peak_position.shape != (3,) or not np.isfinite(peak_position).all():
        raise ValueError("Elmer peak position must contain three finite coordinates.")
    z = float(peak_position[2])
    geometry = config["geometry"]
    wall, plinth = geometry["wall"], geometry["plinth"]
    upstream = float(wall["upstream_radius_m"])
    thickness = float(wall["radial_width_m"])
    chainage_radius = float(wall["centerline_radius_m"])
    plinth_upstream = float(plinth["upstream_radius_m"])
    plinth_downstream = float(plinth["downstream_radius_m"])
    if not all(math.isfinite(v) and v > 0 for v in (
        upstream, thickness, chainage_radius, plinth_upstream, plinth_downstream,
    )):
        raise ValueError("Wall and plinth radii and wall thickness must be finite and positive.")
    downstream = upstream - thickness
    if downstream <= 0:
        raise ValueError("Wall downstream radius must be positive.")
    defaults = json.loads(DEFAULTS.read_text(encoding="utf-8"))
    options = PhaseFieldSolverOptions(**defaults.pop("solver"))
    placement = defaults.pop("automatic_crack")
    depth_fraction = float(placement["depth_fraction"])
    arc_length = float(placement["arc_length_m"])
    if not 0 < depth_fraction < 1:
        raise ValueError("automatic_crack.depth_fraction must lie strictly between 0 and 1.")
    if not math.isfinite(arc_length) or arc_length <= 0:
        raise ValueError("automatic_crack.arc_length_m must be finite and positive.")
    depth = depth_fraction * thickness
    peak_x, peak_y, _ = peak_position
    peak_radius = math.hypot(peak_x, peak_y)
    peak_chainage = chainage_radius * math.atan2(peak_x, peak_y)
    contours = plinth["contours"]
    if not float(contours[0]["chainage"]) <= peak_chainage <= float(contours[-1]["chainage"]):
        raise ValueError("Elmer peak chainage lies outside the configured plinth contours.")
    # Concrete exists above the rock seat; the wall exists above the plinth top contour.
    concrete_left, concrete_right = plinth_intersections(geometry["bedrock"]["contours"], z)
    if not concrete_left <= peak_chainage <= concrete_right:
        raise ValueError("Elmer peak chainage lies outside the concrete section at the peak elevation.")
    try:
        wall_left, wall_right = plinth_intersections(contours, z)
    except ValueError as error:
        raise ValueError(
            f"The wall does not reach the peak elevation Z={z}; a crack through wall and plinth "
            f"cannot be placed ({error})"
        ) from error
    junction = min((wall_left, wall_right), key=lambda chainage: abs(chainage - peak_chainage))
    # Seed from the concrete face carrying the peak; plinth shoulders extend beyond the wall faces.
    seed_face = "upstream" if peak_radius >= chainage_radius else "downstream"
    if seed_face == "upstream":
        inner, outer = upstream - depth, max(upstream, plinth_upstream)
    else:
        inner, outer = min(downstream, plinth_downstream), downstream + depth
    if not inner <= peak_radius <= outer:
        raise ValueError("Automatic crack depth does not reach the Elmer peak radius.")
    span = min(arc_length, concrete_right - concrete_left)
    length_scale = float(defaults["length_scale_m"])
    margin = min(2 * length_scale, span / 4)
    # Admissible arc starts keep the arc in concrete, containing the peak and the junction.
    lowest = max(concrete_left, peak_chainage - span, junction + margin - span)
    highest = min(concrete_right - span, peak_chainage, junction - margin)
    if lowest > highest:
        raise ValueError(
            f"A {span:.6g} m crack arc cannot contain both the peak (chainage {peak_chainage:.6g} m) "
            f"and the wall/plinth junction (chainage {junction:.6g} m) at Z={z}."
        )
    start = min(max(peak_chainage - span / 2, lowest), highest)
    end = start + span
    theta0, theta1 = start / chainage_radius, end / chainage_radius
    if not 0 < theta1 - theta0 < math.pi:
        raise ValueError("Automatic crack requires an angular span between zero and pi.")
    middle = (theta0 + theta1) / 2
    radius = (inner + outer) / 2
    specification = CrackSpecification(
        name="automatic_peak_tension", type="phase_field", shape="annular_sector",
        centre_m=(radius * math.sin(middle), radius * math.cos(middle), z),
        size_m=(outer - inner, radius * (theta1 - theta0), 0.002),
        normal=(0.0, 0.0, 1.0),
        annular_sector=(inner, outer, theta0, theta1),
        **defaults,
    )
    plinth_top_at_peak = float(np.interp(
        peak_chainage,
        [float(row["chainage"]) for row in contours],
        [float(row["height"]) for row in contours],
    ))
    report = {
        "method": (
            "local horizontal annular-sector crack at the wall/plinth nodal peak principal stress, "
            "seeded from the nearer concrete face and spanning the wall/plinth junction"
        ),
        "source_peak": peak,
        "seed_face": seed_face,
        "concrete_chainage_extent_m": [concrete_left, concrete_right],
        "wall_chainage_intersections_m": [wall_left, wall_right],
        "wall_plinth_junction_chainage_m": junction,
        "crack_chainage_m": [start, end],
        "arc_length_m": end - start,
        "opening_m": 0.002,
        "depth_fraction": depth_fraction,
        "depth_m": depth,
        "radial_extent_m": outer - inner,
        "annular_sector": list(specification.annular_sector),
        "damaged_bodies": ["plinth", "wall"],
        "placement_provenance": {
            "elevation_source": "Elmer concrete nodal peak stress; not a fixed physical prescription",
            "peak_body": {1: "plinth", 2: "wall"}.get(peak.get("region_id"), "unknown"),
            "peak_radius_m": peak_radius,
            "peak_chainage_m": peak_chainage,
            "plinth_top_contour_at_peak_m": plinth_top_at_peak,
            "junction_margin_m": margin,
            "regenerated_for_each_baseline": True,
        },
        "warnings": [],
        "limitations": [
            "The wall and plinth are separate meshes joined by springs: damage is computed in each "
            "body, but the phase-field gradient does not couple across the joint and the joint "
            "springs do not soften.",
            "The peak is a nodal value at the cut-back wall/plinth corner and is mesh sensitive.",
        ],
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
