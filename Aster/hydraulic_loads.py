"""Small-strain undrained uplift and reservoir-connected crack-water loads (SI)."""
from __future__ import annotations

import numpy as np

SETTINGS = {
    "base_uplift": "undrained_heel_to_toe",
    "crack_water": "reservoir_connected_open_faces",
    "opening_tolerance_m": 1e-9,
    "maximum_iterations": 20,
}
CRACK_WATER_SETTINGS = dict(SETTINGS, base_uplift="none")
MAX_LOAD_INTERVAL_BISECTIONS = 4
TET_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


def validate_settings(settings: dict) -> dict:
    if set(settings) != set(SETTINGS):
        raise ValueError("Hydraulic settings require the documented uplift/crack-water modes and iteration controls")
    if (settings["base_uplift"] not in (SETTINGS["base_uplift"], "none")
            or settings["crack_water"] != SETTINGS["crack_water"]):
        raise ValueError("Unsupported hydraulic loading assumption")
    tolerance = settings["opening_tolerance_m"]
    iterations = settings["maximum_iterations"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Hydraulic opening tolerance must be finite and positive")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 2:
        raise ValueError("Hydraulic iterations must be an integer >= 2")
    return dict(settings)


def bisect_load_interval(start: float, target: float, level: int,
                         maximum_levels: int) -> tuple[float, int] | None:
    if (not np.isfinite(start) or not np.isfinite(target) or target <= start
            or isinstance(level, bool) or not isinstance(level, int) or level < 0
            or isinstance(maximum_levels, bool) or not isinstance(maximum_levels, int)
            or maximum_levels < 0):
        raise ValueError("Invalid hydraulic load-interval subdivision")
    midpoint = (start + target) / 2.0
    minimum_interval = 1e-12 * max(1.0, abs(start), abs(target))
    if level >= maximum_levels or midpoint <= start + minimum_interval:
        return None
    return midpoint, level + 1


def boundary_faces(cells: np.ndarray, nodes: np.ndarray,
                   source_node_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    faces = cells[:, TET_FACES].reshape(-1, 3)
    identities = faces if source_node_ids is None else source_node_ids[faces]
    _, first, counts = np.unique(np.sort(identities, axis=1), axis=0, return_index=True, return_counts=True)
    rows = first[counts == 1]
    faces = faces[rows].copy()
    cross = np.cross(nodes[faces[:, 1]] - nodes[faces[:, 0]], nodes[faces[:, 2]] - nodes[faces[:, 0]])
    outward = nodes[faces].mean(axis=1) - nodes[cells[rows // 4]].mean(axis=1)
    flip = np.einsum("ij,ij->i", cross, outward) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    cross[flip] *= -1
    lengths = np.linalg.norm(cross, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Degenerate hydraulic boundary face")
    return faces, cross / lengths[:, None]


def triangle_pressure_forces(vertices: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Integrate N_i * max(linear head, 0) exactly, including clipped waterlines."""
    polygon = [np.eye(3)[i] for i in range(3)]
    clipped = []
    for left, right in zip(polygon, polygon[1:] + polygon[:1]):
        a, b = float(left @ head), float(right @ head)
        if a >= 0:
            clipped.append(left)
        if (a < 0) != (b < 0):
            clipped.append(left + a / (a - b) * (right - left))
    result = np.zeros(3)
    quadrature = (np.ones((3, 3)) + 3 * np.eye(3)) / 6
    for index in range(1, len(clipped) - 1):
        bary = np.asarray([clipped[0], clipped[index], clipped[index + 1]])
        xyz = bary @ vertices
        area = np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])) / 2
        shape = quadrature @ bary
        result += area / 3 * (shape.T @ (shape @ head))
    return result


def radial_fraction(nodes: np.ndarray, ids: np.ndarray, upstream_ids: np.ndarray, downstream_ids: np.ndarray,
                    *, label: str = "Base", strict: bool = True) -> np.ndarray:
    """Position of ``ids`` across the wall: 0 on the upstream face, 1 on the downstream face.

    The source arch is centred on (0, 0). The upstream and downstream radii are measured
    from the given face nodes and interpolated along the arch angle, so the wall width is
    never inferred from the crest thickness. ``strict`` rejects nodes outside the faces'
    angular coverage or footprint (the base uplift contract); otherwise the end radii are
    held and the fraction clipped (a lift plane's end nodes at the abutments).
    """
    ids = np.asarray(ids, dtype=np.int64)
    anchor = float(np.arctan2(nodes[ids[0], 1], nodes[ids[0], 0]))

    def angles(rows):
        return anchor + np.angle(np.exp(1j * (np.arctan2(nodes[rows, 1], nodes[rows, 0]) - anchor)))

    theta = angles(ids)

    def radii(rows):
        angle = angles(rows)
        order = np.argsort(angle)
        if strict and (theta.min() < angle.min() - 1e-6 or theta.max() > angle.max() + 1e-6):
            raise ValueError(f"{label} lies outside measured heel/toe angular coverage")
        return np.interp(theta, angle[order], np.linalg.norm(nodes[rows, :2], axis=1)[order])

    upstream_r, downstream_r = radii(np.asarray(upstream_ids)), radii(np.asarray(downstream_ids))
    width = upstream_r - downstream_r
    if np.any(width <= 0):
        raise ValueError(f"Measured upstream {label.lower()} radius must exceed downstream {label.lower()} radius")
    fraction = (upstream_r - np.linalg.norm(nodes[ids, :2], axis=1)) / width
    if strict and np.any((fraction < -1e-3) | (fraction > 1 + 1e-3)):
        raise ValueError(f"{label} nodes lie outside the measured heel/toe footprint")
    return np.clip(fraction, 0, 1)


def uplift_forces(arrays: dict, upstream_z: float, downstream_z: float, gradient: float) -> tuple[np.ndarray, dict]:
    nodes = arrays["nodes"]
    wall = arrays["tetra4_concrete"][arrays["concrete_body_ids"] == 1]
    source = arrays["source_node_ids"]
    faces, normals = boundary_faces(wall, nodes, source)
    pairs = arrays["seg2_wall_foundation_springs"]
    names = arrays["interface_names"]
    interface = (names == "wall_foundation") | np.char.startswith(names.astype(str), "wall_foundation/")
    seat_nodes = np.unique(source[pairs[interface, 0]])
    selected = np.isin(source[faces], seat_nodes).all(axis=1)
    faces, normals = faces[selected], normals[selected]
    if not len(faces):
        raise ValueError("Wall-side wall/foundation base faces are missing")
    base_nodes = np.unique(faces)
    heel = base_nodes[np.isin(source[base_nodes], source[arrays["tri3_upstream"]])]
    toe = base_nodes[np.isin(source[base_nodes], source[arrays["tri3_downstream"]])]
    if len(heel) < 2 or len(toe) < 2:
        raise ValueError("Base heel/toe intersections with reservoir/tailwater surfaces are missing")
    fraction = radial_fraction(nodes, base_nodes, heel, toe)
    pressure = np.zeros(len(nodes))
    pressure[base_nodes] = gradient * ((1 - fraction) * np.maximum(upstream_z - nodes[base_nodes, 2], 0)
                                       + fraction * np.maximum(downstream_z - nodes[base_nodes, 2], 0))
    forces = np.zeros_like(nodes)
    for face, normal in zip(faces, normals):
        weights = triangle_pressure_forces(nodes[face], pressure[face])
        np.add.at(forces, face, -weights[:, None] * normal)
    return forces, {"mode": SETTINGS["base_uplift"], "faces": len(faces),
                    "area_m2": float(np.linalg.norm(np.cross(nodes[faces[:, 1]] - nodes[faces[:, 0]],
                                                           nodes[faces[:, 2]] - nodes[faces[:, 0]]), axis=1).sum() / 2),
                    "resultant_n": forces.sum(axis=0).tolist(),
                    "moment_n_m": np.cross(nodes, forces).sum(axis=0).tolist(),
                    "minimum_pressure_pa": float(pressure[base_nodes].min()),
                    "maximum_pressure_pa": float(pressure[base_nodes].max())}


class CrackWater:
    """Reservoir water on opened, reservoir-connected joint faces; optional uplift in a lift plane.

    Wet faces carry the full reservoir head. With ``pore_faces`` (one boolean per candidate
    facet, the lift-plane facets) the faces of that plane that are not wet carry a pore
    pressure varying linearly across the wall from the reservoir head at the upstream face
    to the tailwater head at the downstream face (FERC Engineering Guidelines ch. 3: uplift
    on a plane within the dam; full head over the cracked part). The pressure acts on both
    lips as equal and opposite forces, so the joint carries the effective stress.
    """

    def __init__(self, arrays: dict, upstream_z: float, gradient: float, tolerance: float,
                 pore_faces: np.ndarray | None = None, downstream_z: float | None = None):
        self.nodes = arrays["nodes"]
        self.faces = arrays["penta6_joints"]
        self.head = upstream_z - self.nodes[self.faces[:, :3], 2]
        self.gradient = gradient
        self.tolerance = tolerance
        vertices = self.nodes[self.faces[:, :3]]
        cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        area = np.linalg.norm(cross, axis=1)
        if np.any(area <= 0):
            raise ValueError("Degenerate crack-water triangle")
        self.normals = cross / area[:, None]
        self.weights = np.array([triangle_pressure_forces(v, h) * gradient
                                 for v, h in zip(vertices, self.head)]).reshape(-1, 3)
        self.pore = None
        self.pore_report = {"mode": "none"}
        if pore_faces is not None:
            pore = np.asarray(pore_faces)
            if pore.dtype.kind != "b" or pore.shape != (len(self.faces),):
                raise ValueError("Lift-plane uplift needs one boolean per candidate facet")
            if downstream_z is None or not np.isfinite(downstream_z):
                raise ValueError("Lift-plane uplift needs the tailwater elevation")
            if not pore.any():
                raise ValueError("Lift-plane uplift has no lift facets")
            rows = self.faces[pore, :3]
            ids = np.unique(rows)
            edges = np.concatenate([np.linalg.norm(self.nodes[rows[:, i]] - self.nodes[rows[:, (i + 1) % 3]], axis=1)
                                    for i in range(3)])
            band = 0.5 * float(edges.mean())
            upstream_nodes, downstream_nodes = np.unique(arrays["tri3_upstream"]), np.unique(arrays["tri3_downstream"])
            fraction = np.zeros(len(self.nodes))
            order = ids[np.argsort(self.nodes[ids, 2], kind="stable")]
            # One plane per elevation cluster; the wall faces are sampled at that elevation.
            for members in np.split(order, np.flatnonzero(np.diff(self.nodes[order, 2]) > band) + 1):
                low, high = self.nodes[members, 2].min() - band, self.nodes[members, 2].max() + band
                heel = upstream_nodes[(self.nodes[upstream_nodes, 2] >= low) & (self.nodes[upstream_nodes, 2] <= high)]
                toe = downstream_nodes[(self.nodes[downstream_nodes, 2] >= low)
                                       & (self.nodes[downstream_nodes, 2] <= high)]
                if len(heel) < 2 or len(toe) < 2:
                    raise ValueError("Lift-plane uplift needs the wall's upstream and downstream faces at the "
                                     f"plane elevation z = {float(self.nodes[members, 2].mean()):g} m")
                fraction[members] = radial_fraction(self.nodes, members, heel, toe, label="Lift plane", strict=False)
            z = self.nodes[:, 2]
            linear = ((1 - fraction) * np.maximum(upstream_z - z, 0) + fraction * np.maximum(downstream_z - z, 0))
            self.pore = pore
            self.pore_weights = np.zeros_like(self.weights)
            self.pore_weights[pore] = np.array([triangle_pressure_forces(self.nodes[face], linear[face]) * gradient
                                                for face in rows]).reshape(-1, 3)
            self.pore_report = {
                "mode": "linear_upstream_to_downstream",
                "faces": int(pore.sum()), "area_m2": float(area[pore].sum() / 2),
                "upstream_z_m": float(upstream_z), "downstream_z_m": float(downstream_z),
                "maximum_pressure_pa": float(gradient * linear[ids].max()),
                "minimum_pressure_pa": float(gradient * linear[ids].min()),
                "resultant_on_one_lip_n": float(np.linalg.norm(
                    (self.pore_weights[pore].sum(axis=1)[:, None] * self.normals[pore]).sum(axis=0))),
                "assumption": ("closed lift-plane faces: pore pressure linear from the reservoir head at the "
                               "upstream face to the tailwater head at the downstream face; wet (opened, "
                               "reservoir-connected) faces: full reservoir head"),
            }
        source = arrays["source_node_ids"]
        upstream = source[arrays["tri3_upstream"]]
        reservoir_edges = {tuple(sorted(edge)) for face in upstream
                           for edge in (face[[0, 1]], face[[1, 2]], face[[2, 0]])}
        self.edges = {}
        self.seeds = []
        for row, face in enumerate(source[self.faces[:, :3]]):
            for i, j in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(sorted((face[i], face[j])))
                self.edges.setdefault(edge, []).append((row, i, j))
                if edge in reservoir_edges:
                    self.seeds.append((row, i, j))

    def evaluate(self, displacement: np.ndarray, eligible_faces: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        minus, plus = self.faces[:, :3], self.faces[:, 3:]
        opening = np.einsum("fki,fi->fk", displacement[minus] - displacement[plus], self.normals)
        available = (opening.mean(axis=1) > self.tolerance) & (self.head.max(axis=1) > 0)
        if eligible_faces is not None:
            eligible = np.asarray(eligible_faces)
            if eligible.shape != available.shape or eligible.dtype.kind != "b":
                raise ValueError("Crack-water eligibility must be one boolean per original facet")
            available &= eligible

        def edge_open(row, i, j):
            return available[row] and (opening[row, i] + opening[row, j]) / 2 > self.tolerance \
                and max(self.head[row, i], self.head[row, j]) > 0

        wet = np.zeros(len(self.faces), dtype=bool)
        stack = []
        for row, i, j in self.seeds:
            if edge_open(row, i, j) and not wet[row]:
                wet[row] = True
                stack.append(row)
        neighbours = [[] for _ in self.faces]
        for members in self.edges.values():
            accessible = [row for row, i, j in members if edge_open(row, i, j)]
            for row in accessible:
                neighbours[row].extend(other for other in accessible if other != row)
        while stack:
            for other in neighbours[stack.pop()]:
                if not wet[other]:
                    wet[other] = True
                    stack.append(other)
        forces = np.zeros_like(self.nodes)
        values = self.weights[wet, :, None] * self.normals[wet, None, :]
        np.add.at(forces, minus[wet], values)
        np.add.at(forces, plus[wet], -values)
        if self.pore is not None:
            closed = self.pore & ~wet
            values = self.pore_weights[closed, :, None] * self.normals[closed, None, :]
            np.add.at(forces, minus[closed], values)
            np.add.at(forces, plus[closed], -values)
        return wet, forces


def force_entries(forces: np.ndarray) -> list[dict]:
    if not np.isfinite(forces).all():
        raise ValueError("Hydraulic nodal forces must be finite")
    return [{"NOEUD": f"N{row + 1}", "FX": float(value[0]), "FY": float(value[1]), "FZ": float(value[2])}
            for row, value in enumerate(forces) if np.any(value != 0)]


def render_uplift(arrays: dict, upstream_z: float, downstream_z: float, gradient: float) -> tuple[str, dict]:
    forces, report = uplift_forces(arrays, upstream_z, downstream_z, gradient)
    entries = force_entries(forces)
    if not entries:
        raise ValueError("Enabled base uplift has no wet resultant")
    return ("UPLIFT = AFFE_CHAR_MECA(MODELE=MODELE, FORCE_NODALE=tuple(\n"
            f"    _F(**entry) for entry in {entries!r}))\n\n"), report


COUPLED_SOLVE = '''    try:
        previous = RESU
        displacement = (nodal_displacement(previous.getField('DEPL', previous.getLastIndex()))
                        if previous is not None else np.zeros_like(hydraulic_arrays['nodes']))
        wet, water_forces = crack_water.evaluate(displacement, hydraulic_open_faces(previous))
        for hydraulic_iteration in range(1, hydraulic_settings['maximum_iterations'] + 1):
            entries = force_entries(water_forces)
            water_load = (AFFE_CHAR_MECA(MODELE=MODELE, FORCE_NODALE=tuple(_F(**entry) for entry in entries))
                          if entries else None)
            trial_options = dict(options)
            trial_options['EXCIT'] = EXCIT + ((_F(CHARGE=water_load, FONC_MULT=RAMP),) if entries else ())
            if previous is not None:
                trial_options['ETAT_INIT'] = _F(EVOL_NOLI=previous, INST=reached)
            mechanical_started = time.perf_counter()
            trial = STAT_NON_LINE(**trial_options)
            mechanical_s += time.perf_counter() - mechanical_started
            trial_displacement = nodal_displacement(trial.getField('DEPL', trial.getLastIndex()))
            updated_wet, updated_forces = crack_water.evaluate(trial_displacement, hydraulic_open_faces(trial))
            print(f"ASTER_HYDRAULIC_UPDATE load_factor={target:.6g} iteration={hydraulic_iteration} "
                  f"wet_faces={int(updated_wet.sum())}", flush=True)
            if np.array_equal(wet, updated_wet):
                RESU = trial
                hydraulic_history.append({
                    'load_factor': target, 'iterations': hydraulic_iteration,
                    'active_set_stable': True,
                    'wet_faces': int(wet.sum()),
                    'paired_resultant_n': (target * water_forces.sum(axis=0)).tolist(),
                    'paired_moment_n_m': (target * np.cross(hydraulic_arrays['nodes'], water_forces).sum(axis=0)).tolist(),
                    'sum_nodal_force_magnitudes_n': float(target * np.linalg.norm(water_forces, axis=1).sum()),
                    'uplift_resultant_n': (target * hydraulic_uplift_resultant).tolist(),
                    'pore_pressure_faces': (int((crack_water.pore & ~wet).sum())
                                            if crack_water.pore is not None else 0),
                })
                np.savez(STEPS_PATH.replace('.npz', '_hydraulic.npz'),
                         load_factor=np.array([row['load_factor'] for row in hydraulic_history]),
                         wet_faces=np.asarray(hydraulic_masks + [wet]))
                hydraulic_masks.append(wet.copy())
                break
            # Every trial starts from the last accepted mechanical state. Rejecting
            # a hydraulic iterate must not commit its irreversible damage.
            DETRUIRE(CONCEPT=_F(NOM=trial))
            if water_load is not None:
                DETRUIRE(CONCEPT=_F(NOM=water_load))
            wet, water_forces = updated_wet, updated_forces
        else:
            raise RuntimeError("Crack-water active set did not converge; no hydraulic step accepted")
    except (CA.AsterError, RuntimeError) as error:
        detail = str(error).strip() or type(error).__name__
        retry = bisect_load_interval(reached, target, target_level, hydraulic_subdivision_levels)
        if retry is not None:
            midpoint, next_level = retry
            TARGETS[target_index:target_index + 1] = [midpoint, target]
            TARGET_LEVELS[target_index:target_index + 1] = [next_level, next_level]
            status["targets"] = TARGETS
            status.setdefault("hydraulic_subdivisions", []).append({
                "from_load_factor": reached, "failed_target": target,
                "retry_target": midpoint, "level": next_level,
                "reason": detail,
            })
            save()
            print(f"ASTER_HYDRAULIC_SUBDIVIDE from={reached:.6g} target={target:.6g} "
                  f"retry={midpoint:.6g} level={next_level}: {detail}", flush=True)
            continue
        save({"status": "stopped", "stopped_at_load_factor": target,
              "message": (f"Coupled Code_Aster step failed from {reached:.6g} to {target:.6g} after "
                          f"{target_level} hydraulic load-step subdivisions: {detail}")})
        print(f"ASTER_COHESIVE_STEP failed load_factor={target:.6g}", flush=True)
        break
'''


def validate_crack_water_evidence(report: dict) -> None:
    """Reject missing or unstable pressure history for a requested wet solve."""
    metadata = report.get("hydraulic_loads")
    if not isinstance(metadata, dict):
        raise ValueError("Requested Aster crack-water pressure has no hydraulic metadata")
    if not isinstance(metadata.get("settings"), dict):
        raise ValueError("Requested Aster crack-water pressure has invalid hydraulic settings")
    settings = validate_settings(metadata["settings"])
    if settings != CRACK_WATER_SETTINGS:
        raise ValueError("Requested crack-water-only solve has different hydraulic settings")
    steps = report.get("steps", [])
    hydraulic = report.get("hydraulic_steps", [])
    if not isinstance(steps, list) or not isinstance(hydraulic, list) or not steps or len(steps) != len(hydraulic):
        raise ValueError("Crack-water pressure history does not cover every accepted mechanical step")
    for mechanical, water in zip(steps, hydraulic):
        if not isinstance(mechanical, dict) or not isinstance(water, dict):
            raise ValueError("Crack-water history contains an invalid accepted step")
        iterations = water.get("iterations")
        if (water.get("active_set_stable") is not True
                or water.get("load_factor") != mechanical["load_factor"]
                or isinstance(water.get("wet_faces"), bool)
                or not isinstance(water.get("wet_faces"), int) or water["wet_faces"] < 0
                or isinstance(iterations, bool) or not isinstance(iterations, int)
                or not 1 <= iterations <= settings["maximum_iterations"]):
            raise ValueError("Crack-water history contains an unstable or inconsistent accepted step")
        for key in ("paired_resultant_n", "paired_moment_n_m", "uplift_resultant_n"):
            values = np.asarray(water.get(key), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid hydraulic diagnostic: {key}")
        if np.any(np.asarray(water["uplift_resultant_n"]) != 0):
            raise ValueError("Crack-water-only solve unexpectedly applied base uplift")
        force = water.get("sum_nodal_force_magnitudes_n")
        if (isinstance(force, bool) or not isinstance(force, (int, float))
                or not np.isfinite(force) or force < 0):
            raise ValueError("Invalid crack-water force magnitude")
