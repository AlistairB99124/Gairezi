"""Z-Y crack-location selection map built from the verified Elmer baseline stress field.

The app shows this map after the elastic baseline so the user can choose the crack
location; no crack elevation is chosen automatically. Coordinates are undeformed metres
with crest z=0; stresses are reported in MPa with tension positive.

Projection: each concrete node is plotted at y = R cos(theta), where theta = atan2(x, y)
and R is the wall centreline (chainage) radius. Using one radius removes the radial
smearing of the 75-81 m concrete thickness, and a selected Y maps uniquely back to
chainage = R acos(y / R) for the dam's 0 <= theta <= pi plan angles.
"""
from __future__ import annotations

import math

import numpy as np

Y_BIN_M = 1.0
Z_BIN_M = 0.5
BODY_BITS = {"wall": 1}


def principal_stresses(stress: np.ndarray) -> np.ndarray:
    """Ascending principal stresses (Pa) for nodal tensors ordered xx, yy, zz, xy, yz, xz."""
    xx, yy, zz, xy, yz, xz = np.asarray(stress, dtype=float).T
    tensors = np.stack((
        np.stack((xx, xy, xz), axis=-1), np.stack((xy, yy, yz), axis=-1), np.stack((xz, yz, zz), axis=-1),
    ), axis=-2)
    return np.linalg.eigvalsh(tensors)


def plan_angle(points: np.ndarray) -> np.ndarray:
    """Plan angle theta (radians) with x = r sin(theta), y = r cos(theta)."""
    points = np.asarray(points, dtype=float)
    return np.arctan2(points[:, 0], points[:, 1])


def chainage_radius(config: dict) -> float:
    radius = float(config["geometry"]["wall"]["centerline_radius_m"])
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Wall centreline radius must be finite and positive.")
    return radius


def _outline(name: str, contours: list[dict], radius: float) -> dict:
    """Contour polyline in the projected Z-Y view, densified because y is nonlinear in chainage."""
    rows = [(float(row["chainage"]), float(row["height"])) for row in contours]
    points = []
    for (s0, h0), (s1, h1) in zip(rows, rows[1:]):
        steps = max(1, math.ceil(abs(s1 - s0)))
        for step in range(steps):
            fraction = step / steps
            s = s0 + fraction * (s1 - s0)
            points.append([radius * math.cos(s / radius), h0 + fraction * (h1 - h0)])
    s, h = rows[-1]
    points.append([radius * math.cos(s / radius), h])
    return {"name": name, "points_yz_m": points}


def selection_map(points: np.ndarray, stress: np.ndarray, node_regions: np.ndarray, config: dict) -> dict:
    """Binned concrete maximum principal stress and per-body peaks for crack-location selection."""
    points = np.asarray(points, dtype=float)
    stress = np.asarray(stress, dtype=float)
    node_regions = np.asarray(node_regions)
    if (points.ndim != 2 or points.shape[1] != 3 or stress.shape != (len(points), 6)
            or node_regions.shape != (len(points),)
            or not np.isfinite(points).all() or not np.isfinite(stress).all()):
        raise ValueError("Selection map requires finite nodal coordinates, six stress components and regions.")
    radius = chainage_radius(config)
    concrete = np.flatnonzero((node_regions & 1) > 0)
    if not len(concrete):
        raise ValueError("Baseline has no wall nodes for the crack-selection map.")
    principal = principal_stresses(stress[concrete])
    theta = plan_angle(points[concrete])
    if theta.min() < 0 or theta.max() > math.pi:
        raise ValueError("Concrete plan angles must lie between 0 and pi for a unique Z-Y projection.")
    y = radius * np.cos(theta)
    z = points[concrete, 2]
    y_min = math.floor(y.min() / Y_BIN_M) * Y_BIN_M
    z_min = math.floor(z.min() / Z_BIN_M) * Z_BIN_M
    y_count = max(1, math.ceil((y.max() - y_min) / Y_BIN_M))
    z_count = max(1, math.ceil((z.max() - z_min) / Z_BIN_M))
    iy = np.clip(np.floor((y - y_min) / Y_BIN_M).astype(int), 0, y_count - 1)
    iz = np.clip(np.floor((z - z_min) / Z_BIN_M).astype(int), 0, z_count - 1)
    grid = np.full((z_count, y_count), -np.inf)
    np.maximum.at(grid, (iz, iy), principal[:, -1])
    values = [
        [round(float(value) / 1e6, 4) if np.isfinite(value) else None for value in row]
        for row in grid
    ]
    peaks = []
    regions = node_regions[concrete]
    for body, bit in BODY_BITS.items():
        members = np.flatnonzero((regions & bit) > 0)
        if not len(members):
            continue
        for kind, index in (
            ("tension", members[int(np.argmax(principal[members, -1]))]),
            ("compression", members[int(np.argmin(principal[members, 0]))]),
        ):
            value = principal[index, -1 if kind == "tension" else 0]
            peaks.append({
                "body": body, "kind": kind,
                "value_mpa": float(value) / 1e6,
                "x_m": float(points[concrete[index], 0]),
                "y_m": float(y[index]),
                "z_m": float(z[index]),
                "chainage_m": float(radius * theta[index]),
                "radius_m": float(np.hypot(*points[concrete[index], :2])),
            })
    geometry = config["geometry"]
    return {
        "schema_version": 1,
        "purpose": "user selection of the phase-field crack location; no location is chosen automatically",
        "units": {"coordinates": "m", "stress": "MPa"},
        "coordinate_frame": "undeformed baseline mesh; crest z=0",
        "projection": (
            "y = R cos(theta) at the wall centreline radius R; selected y maps to chainage = R acos(y/R)"
        ),
        "chainage_radius_m": radius,
        "field": "Elmer nodal maximum principal stress over wall nodes (tension positive)",
        "bin_statistic": "maximum over the radial thickness and nodes in each bin",
        "y_bin_m": Y_BIN_M, "z_bin_m": Z_BIN_M,
        "y_min_m": y_min, "z_min_m": z_min,
        "y_count": y_count, "z_count": z_count,
        "max_principal_stress_mpa": values,
        "peaks": peaks,
        "outlines": [
            _outline("wall_base", geometry["bedrock"]["contours"], radius),
        ],
        "engineering_validated": False,
    }
