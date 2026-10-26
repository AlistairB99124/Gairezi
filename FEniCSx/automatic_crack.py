"""Horizontal crack at the user-selected elevation, clear of the wall/foundation seat."""
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


# Tolerance (m) when testing the foundation clearance; snapped points lie exactly on it.
CLEARANCE_TOLERANCE_M = 1e-9
# A zero buffer still keeps snapped seeds strictly inside the wall concrete (m).
MINIMUM_SNAP_CLEARANCE_M = 0.01
SEED_FACES = ("upstream", "downstream")


def cracking_settings(config: dict, defaults: dict | None = None) -> dict:
    """Seed settings from the config ``cracking`` array, falling back to fracture_defaults.json.

    ``cracking`` is a list of crack definitions; one seed is currently supported, so it
    must hold exactly one entry here (a config without ``cracking`` uses the defaults;
    an empty list means no crack, and the app skips the cracking stages).
    face: wall face the seed opens from; depth_m: radial depth from that face (m);
    opening_m: seed band thickness in Z (m); full_width: run to both buffer limits;
    buffer_m: clearance from the wall/foundation contour (m); length_m: seed arc length
    at the centreline radius (m), used only when full_width is false.
    """
    if defaults is None:
        defaults = json.loads(DEFAULTS.read_text(encoding="utf-8"))
    placement = defaults["automatic_crack"]
    cracks = config.get("cracking", [{}])
    if not isinstance(cracks, list) or not all(isinstance(crack, dict) for crack in cracks):
        raise ValueError("cracking must be an array of crack objects.")
    if not cracks:
        raise ValueError("cracking defines no crack, so there is no seed to place.")
    if len(cracks) != 1:
        raise ValueError(f"Exactly one cracking entry is currently supported; found {len(cracks)}.")
    cracking = cracks[0]
    settings = {
        "face": cracking.get("face", "upstream"),
        "depth_m": cracking.get("depth_m", placement["depth_m"]),
        "opening_m": cracking.get("opening_m", placement.get("opening_m", 0.002)),
        "full_width": cracking.get("full_width", True),
        "buffer_m": cracking.get("buffer_m", placement["foundation_clearance_m"]),
        "length_m": cracking.get("length_m"),
    }
    if settings["face"] not in SEED_FACES:
        raise ValueError(f"cracking[0].face must be one of {SEED_FACES}.")
    if not isinstance(settings["full_width"], bool):
        raise ValueError("cracking[0].full_width must be true or false.")
    for key in ("depth_m", "opening_m", "buffer_m"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"cracking[0].{key} must be a finite number.")
        settings[key] = float(value)
    if settings["depth_m"] <= 0 or settings["opening_m"] <= 0 or settings["buffer_m"] < 0:
        raise ValueError("cracking depth and opening must be positive and the buffer nonnegative.")
    length = settings["length_m"]
    if length is not None:
        if (isinstance(length, bool) or not isinstance(length, (int, float))
                or not math.isfinite(length) or length <= 0):
            raise ValueError("cracking[0].length_m must be null or a positive number.")
        settings["length_m"] = float(length)
    if not settings["full_width"] and settings["length_m"] is None:
        raise ValueError("cracking[0].length_m is required when full_width is false.")
    return settings


def wall_radii_at(config: dict, z: float) -> tuple[float, float]:
    """Upstream radius and local downstream radius (m) of the wall at elevation z."""
    wall = config["geometry"]["wall"]
    upstream = float(wall["upstream_radius_m"])
    downstream = float(wall["downstream_radius_m"])
    # Optional thickness steps widen the wall downstream below each top_z_m.
    for step in wall.get("thickness_steps") or ():
        if z < float(step["top_z_m"]):
            downstream = upstream - float(step["radial_width_m"])
    return upstream, downstream


class SeedGeometry:
    """Seed depth and foundation clearance in the wall's chainage-Z elevation.

    The bedrock contour chainages are measured at the centreline radius R. Clearances
    are measured at the seed's smallest possible radius, where the same angle spans the
    shortest arc (conservative): upstream seeds reach ``upstream - depth_m``; downstream
    seeds start at the downstream face, taken at its smallest radius over all thickness
    steps. Points are ``(x, z)`` with ``x = chainage * inner / R`` in metres of arc.
    """

    def __init__(self, config: dict, defaults: dict | None = None):
        settings = cracking_settings(config, defaults)
        self.face = settings["face"]
        self.depth_m = settings["depth_m"]
        self.buffer_m = settings["buffer_m"]
        self.clearance_m = max(self.buffer_m, MINIMUM_SNAP_CLEARANCE_M)
        wall = config["geometry"]["wall"]
        self.upstream_m = float(wall["upstream_radius_m"])
        self.chainage_radius_m = float(wall["centerline_radius_m"])
        if not all(math.isfinite(v) and v > 0 for v in (
            self.depth_m, self.clearance_m, self.upstream_m, self.chainage_radius_m,
        )):
            raise ValueError("Seed depth, foundation clearance and wall radii must be finite and positive.")
        if self.face == "upstream":
            self.inner_m = self.upstream_m - self.depth_m
        else:
            widths = [float(wall["radial_width_m"])] + [
                float(step["radial_width_m"]) for step in wall.get("thickness_steps") or ()
            ]
            self.inner_m = min(float(wall["downstream_radius_m"]), self.upstream_m - max(widths))
        if self.inner_m <= 0:
            raise ValueError("Seed depth must be less than the upstream wall radius.")
        self.scale = self.inner_m / self.chainage_radius_m
        contours = config["geometry"]["bedrock"]["contours"]
        self.vertices = np.array(
            [(self.scale * float(row["chainage"]), float(row["height"])) for row in contours], dtype=float,
        )
        if len(self.vertices) < 2 or not np.isfinite(self.vertices).all() or np.any(np.diff(self.vertices[:, 0]) <= 0):
            raise ValueError("Bedrock contours must be finite with strictly increasing chainages.")

    def clearance(self, points) -> np.ndarray:
        """Distance (m) from each (x, z) point to the wall/foundation contour."""
        p = np.atleast_2d(np.asarray(points, dtype=float))
        a, b = self.vertices[:-1], self.vertices[1:]
        d = b - a
        t = np.clip(np.einsum("pij,ij->pi", p[:, None, :] - a, d) / np.einsum("ij,ij->i", d, d), 0, 1)
        closest = a + t[..., None] * d
        return np.linalg.norm(p[:, None, :] - closest, axis=2).min(axis=1)

    def valid(self, points) -> np.ndarray:
        """Concrete points at or below the crest and at least the clearance from the foundation."""
        p = np.atleast_2d(np.asarray(points, dtype=float))
        above = p[:, 1] > np.interp(p[:, 0], self.vertices[:, 0], self.vertices[:, 1])
        inside = (p[:, 0] >= self.vertices[0, 0]) & (p[:, 0] <= self.vertices[-1, 0])
        return (inside & above & (p[:, 1] <= 0.0)
                & (self.clearance(p) >= self.clearance_m - CLEARANCE_TOLERANCE_M))

    def _boundary_candidates(self, point) -> list:
        """Projections of ``point`` onto every piece of the valid region's boundary."""
        c, candidates = self.clearance_m, [np.array((point[0], 0.0))]
        for a, b in zip(self.vertices[:-1], self.vertices[1:]):
            d = b - a
            normal = np.array((-d[1], d[0])) / np.linalg.norm(d)
            start = a + c * normal
            t = float(np.clip(np.dot(point - start, d) / np.dot(d, d), 0, 1))
            candidates.append(start + t * d)
            if d[1] != 0:
                t = -start[1] / d[1]
                if 0 <= t <= 1:
                    candidates.append(start + t * d)
        for vertex in self.vertices:
            offset = point - vertex
            length = float(np.linalg.norm(offset))
            candidates.append(vertex + c * (offset / length if length > 1e-12 else np.array((0.0, 1.0))))
            if abs(vertex[1]) <= c:
                half = math.sqrt(c * c - vertex[1] ** 2)
                candidates.extend((np.array((vertex[0] - half, 0.0)), np.array((vertex[0] + half, 0.0))))
        return candidates

    def _sampled_boundary(self) -> np.ndarray:
        c, samples = self.clearance_m, []
        for a, b in zip(self.vertices[:-1], self.vertices[1:]):
            d = b - a
            normal = np.array((-d[1], d[0])) / np.linalg.norm(d)
            t = np.linspace(0, 1, max(2, math.ceil(np.linalg.norm(d) / 0.01)))
            samples.append(a + c * normal + t[:, None] * d)
        angles = np.linspace(0, 2 * math.pi, 721)
        for vertex in self.vertices:
            samples.append(vertex + c * np.column_stack((np.cos(angles), np.sin(angles))))
        return np.vstack(samples)

    def snap(self, chainage: float, z: float) -> tuple[float, float]:
        """Nearest (chainage, z) with the required foundation clearance; unchanged if already valid."""
        point = np.array((self.scale * chainage, z))
        if self.valid(point)[0]:
            return chainage, z
        candidates = np.array(self._boundary_candidates(point))
        candidates = candidates[self.valid(candidates)]
        if not len(candidates):
            candidates = self._sampled_boundary()
            candidates = candidates[self.valid(candidates)]
        if not len(candidates):
            raise ValueError(
                f"No wall location lies at least {self.clearance_m:g} m from the foundation."
            )
        best = candidates[int(np.argmin(np.linalg.norm(candidates - point, axis=1)))]
        snapped_chainage = chainage if abs(best[0] - point[0]) <= 1e-9 else float(best[0] / self.scale)
        return snapped_chainage, float(best[1])

    def extent(self, chainage: float, z: float) -> tuple[float, float]:
        """Chainages where the seed at z, containing ``chainage``, comes within the clearance."""
        x0 = self.scale * chainage
        if not self.valid((x0, z))[0]:
            raise ValueError(
                f"Chainage {chainage:.6g} m at Z={z} m is within {self.clearance_m:g} m of the foundation."
            )
        ends = []
        for limit in (self.vertices[0, 0], self.vertices[-1, 0]):
            xs = np.append(np.arange(x0, limit, 0.01 if limit > x0 else -0.01), limit)
            ok = self.valid(np.column_stack((xs, np.full(len(xs), z))))
            first_bad = int(np.argmin(ok)) if not ok.all() else len(xs) - 1
            good, bad = xs[max(first_bad - 1, 0)], xs[first_bad]
            for _ in range(60):
                middle = (good + bad) / 2
                if self.valid((middle, z))[0]:
                    good = middle
                else:
                    bad = middle
            ends.append(good / self.scale)
        return ends[0], ends[1]


def snap_crack_location(config: dict, location) -> tuple[float, float, bool]:
    """Move a selected (Y, Z) to the nearest wall point with the configured foundation clearance."""
    try:
        y, z = (float(value) for value in location)
    except (TypeError, ValueError) as error:
        raise ValueError("Crack location must be two numbers: Y and Z in metres.") from error
    if not (math.isfinite(y) and math.isfinite(z)):
        raise ValueError("Crack location Y and Z must be finite.")
    geometry = SeedGeometry(config)
    radius = geometry.chainage_radius_m
    if not -radius <= y <= radius:
        raise ValueError(f"Selected Y={y} m lies outside the +/-{radius} m centreline projection.")
    chainage = radius * math.acos(y / radius)
    snapped_chainage, snapped_z = geometry.snap(chainage, z)
    if snapped_chainage == chainage and snapped_z == z:
        return y, z, False
    return radius * math.cos(snapped_chainage / radius), snapped_z, True

def hydrostatic_crack_pressure(config: dict, seed_face: str, z: float) -> dict:
    """Water pressure (Pa) in a crack open to ``seed_face`` at elevation z.

    The crack fills from the face it opens on, so it carries that face's hydrostatic
    pressure at the crack elevation, uniform along the horizontal crack (full head in
    the cracked length, as in USACE EM 1110-2-2100 for cracked bases). Datums match
    the Elmer water loads: water_height and tailwater_head are measured from the
    lowest bedrock contour, and the reservoir surface includes the overflow head.
    """
    loads = config["loads"]
    bed = min(float(row["height"]) for row in config["geometry"]["bedrock"]["contours"])
    gradient = float(loads["water_density"]) * abs(float(loads["gravity"]))
    if seed_face == "upstream":
        surface = bed + float(loads["water_height"]) + float(loads["overflow_head"])
    elif seed_face == "downstream":
        surface = bed + float(loads["tailwater_head"])
    else:
        raise ValueError(f"Unknown crack seed face {seed_face!r}")
    if not all(math.isfinite(value) for value in (gradient, surface, z)) or gradient <= 0:
        raise ValueError("Crack water pressure needs finite water density, gravity and levels")
    return {
        "pressure_pa": gradient * max(surface - z, 0.0),
        "water_surface_z_m": surface,
        "pressure_gradient_pa_per_m": gradient,
        "seed_face": seed_face,
        "model": (f"hydrostatic {seed_face} water at the crack elevation, uniform over the "
                  "seeded crack and ramped with the external water loads"),
    }


def locate_crack(config: dict, location, fields):
    """Seed a horizontal crack at the selected elevation, clear of the wall/foundation seat.

    ``location`` is the (y, z) point picked on the baseline Z-Y plot (see crack_selection):
    z is the crack elevation and y selects the chainage R acos(y/R) at the wall centreline
    radius R. A point closer than the ``cracking[0].buffer_m`` to the bedrock/foundation
    contour (or above the crest) snaps to the nearest valid point. The crack runs along the
    wall at z in both directions until it comes within that buffer of the contour (or, when
    ``full_width`` is false, for ``length_m`` centred on the selection), and is opened from
    ``cracking[0].face`` to a fixed ``depth_m`` with Z thickness ``opening_m``; missing settings
    fall back to fracture_defaults.json. ``fields`` is (coordinates_m,
    stress_pa, node_regions) from the Elmer baseline; they confirm the crack crosses wall
    concrete and record the peak principal stress near the selection.
    """
    from FEniCSx.crack_contract import (
        CrackSpecification, PhaseFieldSolverOptions, apply_fracture_debug, calibrate_phase_field_defaults,
    )
    if __package__:
        from .crack_selection import plan_angle, principal_stresses
    else:
        from crack_selection import plan_angle, principal_stresses

    y, z, snapped = snap_crack_location(config, location)
    requested_y, requested_z = (float(value) for value in location)
    settings = cracking_settings(config)
    seed_geometry = SeedGeometry(config)
    geometry = config["geometry"]
    upstream, downstream = wall_radii_at(config, z)
    thickness = upstream - downstream
    chainage_radius = seed_geometry.chainage_radius_m
    if not all(math.isfinite(v) and v > 0 for v in (downstream, thickness)):
        raise ValueError("Wall radii and thickness must be finite and positive.")
    depth = seed_geometry.depth_m
    if depth >= thickness:
        raise ValueError(
            f"Seed depth {depth:g} m must be less than the {thickness:g} m wall thickness at Z={z} m."
        )
    chainage = chainage_radius * math.acos(y / chainage_radius)
    defaults, water_factor = apply_fracture_debug(
        json.loads(DEFAULTS.read_text(encoding="utf-8")), config)
    effective_defaults = json.loads(json.dumps(defaults))
    options = PhaseFieldSolverOptions(**defaults.pop("solver"))
    placement = defaults.pop("automatic_crack")
    face_half_length = float(placement["face_search_half_length_m"])
    if not math.isfinite(face_half_length) or face_half_length <= 0:
        raise ValueError("automatic_crack.face_search_half_length_m must be finite and positive.")
    # Strength-based calibration (Tanne et al. 2018): the homogeneous peak stress equals ft.
    defaults, calibration = calibrate_phase_field_defaults(defaults, config["material"]["concrete"])
    length_scale = calibration["length_scale_m"]
    concrete_left, concrete_right = contour_intersections(geometry["bedrock"]["contours"], z)
    # The seed stops where it would come within the clearance of the wall/foundation seat.
    start, end = seed_geometry.extent(chainage, z)
    warnings = []
    if not settings["full_width"]:
        half = settings["length_m"] / 2
        if chainage - half < start or chainage + half > end:
            warnings.append(
                f"Seed length {settings['length_m']:g} m was clipped to the {seed_geometry.buffer_m:g} m "
                f"foundation buffer ({end - start:.3f} m available at Z={z} m)."
            )
        start, end = max(start, chainage - half), min(end, chainage + half)
    if seed_geometry.clearance_m > seed_geometry.buffer_m:
        warnings.append(
            f"A {seed_geometry.buffer_m:g} m buffer keeps a {seed_geometry.clearance_m:g} m minimum "
            "clearance so the seed stays inside wall concrete."
        )

    points, stress, node_regions = (np.asarray(array) for array in fields)
    if (points.ndim != 2 or points.shape[1] != 3 or stress.shape != (len(points), 6)
            or node_regions.shape != (len(points),)):
        raise ValueError("Baseline fields must be nodal coordinates, six stress components and regions.")
    node_chainage = chainage_radius * plan_angle(points)
    node_radius = np.hypot(points[:, 0], points[:, 1])
    concrete = (node_regions & 1) > 0
    along = (node_chainage >= start) & (node_chainage <= end)
    near_selection = np.abs(node_chainage - chainage) <= face_half_length
    # Search one coarse wall element layer (about 0.5 m) either side of the click so a
    # selection between node layers still finds wall concrete; the crack stays at z.
    search_half_height = 0.5
    nearby = np.flatnonzero(concrete & along & near_selection & (np.abs(points[:, 2] - z) <= search_half_height))
    if not len(nearby):
        raise ValueError(f"No wall baseline nodes lie near the selected location (Y={y} m, Z={z} m).")
    principal = principal_stresses(stress[nearby])[:, -1]
    peak_node = int(nearby[int(np.argmax(principal))])
    # The seed opens from the configured face at a fixed radial depth from that face.
    seed_face = settings["face"]
    water = hydrostatic_crack_pressure(config, seed_face, z)
    if water_factor != 1.0:
        water["pressure_pa"] *= water_factor
        water["model"] += f", scaled by fracture_debug.crack_water_pressure_factor = {water_factor:g}"
    water["factor"] = water_factor
    defaults["internal_pressure_pa"] = water["pressure_pa"]
    if seed_face == "upstream":
        inner, outer = upstream - depth, upstream
    else:
        inner, outer = downstream, downstream + depth
    opening = settings["opening_m"]
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
        size_m=(outer - inner, radius * (theta1 - theta0), opening),
        normal=(0.0, 0.0, 1.0),
        annular_sector=(inner, outer, theta0, theta1),
        **defaults,
    )
    names = {1: "wall"}
    report = {
        "method": (
            f"horizontal annular-sector crack at the user-selected elevation, opened a fixed depth "
            f"from the {seed_face} face and stopping the foundation clearance short of the "
            "wall/foundation seat at both ends"
            + ("" if settings["full_width"] else ", limited to the configured length about the selection")
        ),
        "cracking_settings": settings,
        "requested_location_m": {"y": requested_y, "z": requested_z},
        "selected_location_m": {"y": y, "z": z},
        "snapped_to_foundation_clearance": snapped,
        "selection_projection": "chainage = R acos(y / R) at the wall centreline radius R",
        "selected_chainage_m": chainage,
        "seed_face": seed_face,
        "seed_face_rule": f"{seed_face} (configured)",
        "seed_face_source": {
            "node_id": peak_node + 1,
            "position_xyz_m": points[peak_node].tolist(),
            "max_principal_stress_pa": float(principal.max()),
        },
        "concrete_chainage_extent_m": [concrete_left, concrete_right],
        "crack_chainage_m": [start, end],
        "arc_length_m": end - start,
        "opening_m": opening,
        "depth_m": depth,
        "foundation_clearance_m": seed_geometry.clearance_m,
        "buffer_m": seed_geometry.buffer_m,
        "clearance_measurement": (
            "minimum distance in the chainage-Z elevation, using arc length at the seed's inner "
            "radius (conservative)"
        ),
        "radial_extent_m": outer - inner,
        "crack_water_pressure": water,
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
            "The seed runs along the wall at the selected elevation until it comes within the "
            "foundation clearance of the wall/foundation seat (or the configured length), at constant "
            "depth from the configured face; it is a horizontal annular sector, not a surface conforming to the sloped "
            "bedrock. The diffuse AT1 band (about 2l) can extend past the seed ends, but damage "
            "remains restricted to wall cells.",
            "The seed face comes from the configuration; the peak stress near the selection "
            "is recorded for information only and does not choose the face.",
            "Crack water pressure is the hydrostatic pressure of the seeded face at the crack "
            "elevation (full head along the crack, no drainage or head loss); it is not a coupled "
            "seepage solution, and base uplift at the wall-foundation interface is not modelled.",
        ],
        "defaults": effective_defaults,
        "fracture_debug": config.get("fracture_debug"),
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
