"""Local horizontal crack through concrete at the user-selected Z-Y location (no automatic placement)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

DEFAULTS = Path(__file__).with_name("fracture_defaults.json")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def contour_intersections(contours: list[dict], z: float) -> tuple[float, float]:
    """Return chainages where bedrock meets a wall elevation below the crest."""
    if not math.isfinite(z) or len(contours) < 2:
        raise ValueError("A finite crack Z and at least two bedrock contours are required.")
    points = [(float(row["chainage"]), float(row["height"])) for row in contours]
    if not all(math.isfinite(s) and math.isfinite(h) for s, h in points):
        raise ValueError("Bedrock contours must be finite.")
    if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
        raise ValueError("Bedrock contour chainages must be strictly increasing.")
    if z > 0 or z <= min(height for _, height in points):
        raise ValueError("Crack Z must lie above the bedrock minimum and at or below the crest.")
    crossings = []
    for (s0, h0), (s1, h1) in zip(points, points[1:]):
        if h0 == h1:
            if z == h0:
                raise ValueError("Crack Z coincides with a horizontal bedrock segment.")
            continue
        if min(h0, h1) <= z <= max(h0, h1):
            s = s0 + (z - h0) * (s1 - s0) / (h1 - h0)
            if not any(math.isclose(s, other, abs_tol=1e-9) for other in crossings):
                crossings.append(s)
    if len(crossings) != 2:
        raise ValueError(f"Crack Z={z} requires exactly two bedrock contour intersections; found {len(crossings)}.")
    left, right = sorted(crossings)
    interior = [height for chainage, height in points if left < chainage < right]
    if not interior or any(height >= z for height in interior):
        raise ValueError("Bedrock intersections do not bound one continuous wall section.")
    return left, right


def locate_crack(config: dict, location, fields):
    """Seed a local horizontal crack at the user-selected Z-Y location.

    ``location`` is the (y, z) point picked on the baseline Z-Y plot (see crack_selection):
    z is the crack elevation and y selects the chainage R acos(y/R) at the wall centreline
    radius R. ``fields`` is (coordinates_m, stress_pa, node_regions) from the Elmer baseline;
    they choose the seeded face (the face of the highest local principal stress) and confirm
    which concrete bodies the crack crosses. No location is chosen automatically.
    """
    from FEniCSx.crack_contract import (
        CrackSpecification, PhaseFieldSolverOptions, calibrate_phase_field_defaults,
    )
    if __package__:
        from .crack_selection import plan_angle, principal_stresses
    else:
        from crack_selection import plan_angle, principal_stresses

    try:
        y, z = (float(value) for value in location)
    except (TypeError, ValueError) as error:
        raise ValueError("Crack location must be two numbers: Y and Z in metres.") from error
    if not (math.isfinite(y) and math.isfinite(z)):
        raise ValueError("Crack location Y and Z must be finite.")
    geometry = config["geometry"]
    wall = geometry["wall"]
    upstream = float(wall["upstream_radius_m"])
    thickness = float(wall["radial_width_m"])
    chainage_radius = float(wall["centerline_radius_m"])
    downstream = float(wall["downstream_radius_m"])
    # Optional thickness steps widen the wall downstream below each top_z_m.
    for step in wall.get("thickness_steps") or ():
        if z < float(step["top_z_m"]):
            thickness = float(step["radial_width_m"])
            downstream = upstream - thickness
    if not all(math.isfinite(v) and v > 0 for v in (
        upstream, downstream, thickness, chainage_radius,
    )):
        raise ValueError("Wall radii and thickness must be finite and positive.")
    if downstream <= 0:
        raise ValueError("Wall downstream radius must be positive.")
    if not -chainage_radius <= y <= chainage_radius:
        raise ValueError(f"Selected Y={y} m lies outside the +/-{chainage_radius} m centreline projection.")
    chainage = chainage_radius * math.acos(y / chainage_radius)
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
    # Strength-based calibration (Tanne et al. 2018): the homogeneous peak stress equals ft.
    defaults, calibration = calibrate_phase_field_defaults(defaults, config["material"]["concrete"])
    length_scale = calibration["length_scale_m"]
    # The wall exists above the surveyed bedrock and up to the crest.
    try:
        concrete_left, concrete_right = contour_intersections(geometry["bedrock"]["contours"], z)
    except ValueError as error:
        raise ValueError(f"No concrete section exists at the selected Z={z} m ({error})") from error
    if not concrete_left <= chainage <= concrete_right:
        raise ValueError(
            f"Selected location (Y={y} m, chainage {chainage:.6g} m) lies outside the concrete "
            f"section at Z={z} m (chainage {concrete_left:.6g} to {concrete_right:.6g} m)."
        )
    span = min(arc_length, concrete_right - concrete_left)
    start = min(max(chainage - span / 2, concrete_left), concrete_right - span)
    end = start + span
    warnings = []
    if span < arc_length:
        warnings.append(f"Crack arc shortened to the {span:.6g} m concrete width at Z={z} m.")
    if not math.isclose(start, chainage - span / 2, abs_tol=1e-9):
        warnings.append("Crack arc shifted along the dam to stay inside the concrete section.")

    points, stress, node_regions = (np.asarray(array) for array in fields)
    if (points.ndim != 2 or points.shape[1] != 3 or stress.shape != (len(points), 6)
            or node_regions.shape != (len(points),)):
        raise ValueError("Baseline fields must be nodal coordinates, six stress components and regions.")
    node_chainage = chainage_radius * plan_angle(points)
    node_radius = np.hypot(points[:, 0], points[:, 1])
    concrete = (node_regions & 1) > 0
    along = (node_chainage >= start) & (node_chainage <= end)
    # Search one coarse wall element layer (about 0.5 m) either side of the click so a
    # selection between node layers still finds wall concrete; the crack stays at z.
    search_half_height = 0.5
    nearby = np.flatnonzero(concrete & along & (np.abs(points[:, 2] - z) <= search_half_height))
    if not len(nearby):
        raise ValueError(f"No wall baseline nodes lie near the selected location (Y={y} m, Z={z} m).")
    principal = principal_stresses(stress[nearby])[:, -1]
    face_node = int(nearby[int(np.argmax(principal))])
    seed_face = "upstream" if node_radius[face_node] >= upstream - thickness / 2 else "downstream"
    if seed_face == "upstream":
        inner, outer = upstream - depth, upstream
    else:
        inner, outer = downstream, downstream + depth
    crossed = concrete & along & (np.abs(points[:, 2] - z) <= max(length_scale / 2, search_half_height)) & (
        (node_radius >= inner) & (node_radius <= outer)
    )
    required_regions = [1] if np.any(node_regions[crossed] & 1) else []
    if not required_regions:
        raise ValueError(
            f"The {seed_face} crack band at the selected location (Y={y} m, Z={z} m) contains no wall nodes."
        )
    theta0, theta1 = start / chainage_radius, end / chainage_radius
    if not 0 < theta1 - theta0 < math.pi:
        raise ValueError("Crack requires an angular span between zero and pi.")
    middle = (theta0 + theta1) / 2
    radius = (inner + outer) / 2
    specification = CrackSpecification(
        name="user_located_crack", type="phase_field", shape="annular_sector",
        centre_m=(radius * math.sin(middle), radius * math.cos(middle), z),
        size_m=(outer - inner, radius * (theta1 - theta0), 0.002),
        normal=(0.0, 0.0, 1.0),
        annular_sector=(inner, outer, theta0, theta1),
        **defaults,
    )
    names = {1: "wall"}
    report = {
        "method": (
            "local horizontal annular-sector crack at the user-selected Z-Y location, seeded from "
            "the concrete face carrying the highest nearby baseline principal stress"
        ),
        "selected_location_m": {"y": y, "z": z},
        "selection_projection": "chainage = R acos(y / R) at the wall centreline radius R",
        "selected_chainage_m": chainage,
        "seed_face": seed_face,
        "seed_face_source": {
            "node_id": face_node + 1,
            "position_xyz_m": points[face_node].tolist(),
            "max_principal_stress_pa": float(principal.max()),
        },
        "concrete_chainage_extent_m": [concrete_left, concrete_right],
        "crack_chainage_m": [start, end],
        "arc_length_m": end - start,
        "opening_m": 0.002,
        "depth_fraction": depth_fraction,
        "depth_m": depth,
        "radial_extent_m": outer - inner,
        "annular_sector": list(specification.annular_sector),
        "required_region_ids": required_regions,
        "damaged_bodies": [names[region] for region in required_regions],
        "placement_provenance": {
            "elevation_source": "user selection on the baseline Z-Y stress plot; not chosen automatically",
            "chainage_radius_m": chainage_radius,
        },
        "warnings": warnings,
        "limitations": [
            "Damage is restricted to the wall; bedrock remains intact and does not share the wall damage field. "
            "Seat springs do not soften as damage evolves.",
            "The Z-Y view cannot distinguish faces; the seeded face follows the baseline stress near "
            "the selected point.",
        ],
        "defaults": json.loads(DEFAULTS.read_text(encoding="utf-8")),
        "length_scale_calibration": calibration,
        "report_crack": {
            "name": specification.name,
            "central_crack_pos_m": dict(zip(("x", "y", "z"), specification.centre_m)),
            "size_m": dict(zip(("x", "y", "z"), specification.size_m)),
            "orientation": {"normal_x": 0.0, "normal_y": 0.0, "normal_z": 1.0},
            "phase_field_parameters": {
                "model": specification.model,
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
