"""Small-strain undrained uplift and reservoir-connected crack-water loads (SI)."""
from __future__ import annotations

import numpy as np

SETTINGS = {
    "base_uplift": "undrained_heel_to_toe",
    "crack_water": "reservoir_connected_open_faces",
    "opening_tolerance_m": 1e-9,
    "maximum_iterations": 20,
}
TET_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


def validate_settings(settings: dict) -> dict:
    if set(settings) != set(SETTINGS):
        raise ValueError("Hydraulic settings require the documented uplift/crack-water modes and iteration controls")
    if (settings["base_uplift"] != SETTINGS["base_uplift"]
            or settings["crack_water"] != SETTINGS["crack_water"]):
        raise ValueError("Unsupported hydraulic loading assumption")
    tolerance = settings["opening_tolerance_m"]
    iterations = settings["maximum_iterations"]
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Hydraulic opening tolerance must be finite and positive")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 2:
        raise ValueError("Hydraulic iterations must be an integer >= 2")
    return dict(settings)


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
    # The source arch is centred on (0,0). Interpolate measured base boundary
    # radii along its angle; do not infer base width from the crest thickness.
    anchor = float(np.arctan2(nodes[base_nodes[0], 1], nodes[base_nodes[0], 0]))

    def angles(ids):
        return anchor + np.angle(np.exp(1j * (np.arctan2(nodes[ids, 1], nodes[ids, 0]) - anchor)))

    theta = angles(base_nodes)

    def radii(ids):
        angle = angles(ids)
        order = np.argsort(angle)
        if theta.min() < angle.min() - 1e-6 or theta.max() > angle.max() + 1e-6:
            raise ValueError("Base lies outside measured heel/toe angular coverage")
        return np.interp(theta, angle[order], np.linalg.norm(nodes[ids, :2], axis=1)[order])

    upstream_r, downstream_r = radii(heel), radii(toe)
    width = upstream_r - downstream_r
    if np.any(width <= 0):
        raise ValueError("Measured upstream base radius must exceed downstream base radius")
    fraction = (upstream_r - np.linalg.norm(nodes[base_nodes, :2], axis=1)) / width
    if np.any((fraction < -1e-3) | (fraction > 1 + 1e-3)):
        raise ValueError("Base nodes lie outside the measured heel/toe footprint")
    fraction = np.clip(fraction, 0, 1)
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
    def __init__(self, arrays: dict, upstream_z: float, gradient: float, tolerance: float):
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

    def evaluate(self, displacement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        minus, plus = self.faces[:, :3], self.faces[:, 3:]
        opening = np.einsum("fki,fi->fk", displacement[minus] - displacement[plus], self.normals)
        available = (opening.mean(axis=1) > self.tolerance) & (self.head.max(axis=1) > 0)

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
        wet, water_forces = crack_water.evaluate(displacement)
        for hydraulic_iteration in range(1, hydraulic_settings['maximum_iterations'] + 1):
            entries = force_entries(water_forces)
            water_load = (AFFE_CHAR_MECA(MODELE=MODELE, FORCE_NODALE=tuple(_F(**entry) for entry in entries))
                          if entries else None)
            trial_options = dict(options)
            trial_options['EXCIT'] = EXCIT + ((_F(CHARGE=water_load, FONC_MULT=RAMP),) if entries else ())
            if previous is not None:
                trial_options['ETAT_INIT'] = _F(EVOL_NOLI=previous, INST=reached)
            trial = STAT_NON_LINE(**trial_options)
            trial_displacement = nodal_displacement(trial.getField('DEPL', trial.getLastIndex()))
            updated_wet, updated_forces = crack_water.evaluate(trial_displacement)
            print(f"ASTER_HYDRAULIC_UPDATE load_factor={target:.6g} iteration={hydraulic_iteration} "
                  f"wet_faces={int(updated_wet.sum())}", flush=True)
            if np.array_equal(wet, updated_wet):
                RESU = trial
                hydraulic_history.append({
                    'load_factor': target, 'iterations': hydraulic_iteration,
                    'wet_faces': int(wet.sum()),
                    'paired_resultant_n': (target * water_forces.sum(axis=0)).tolist(),
                    'paired_moment_n_m': (target * np.cross(hydraulic_arrays['nodes'], water_forces).sum(axis=0)).tolist(),
                    'sum_nodal_force_magnitudes_n': float(target * np.linalg.norm(water_forces, axis=1).sum()),
                    'uplift_resultant_n': (target * hydraulic_uplift_resultant).tolist(),
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
        save({"status": "stopped", "stopped_at_load_factor": target,
              "message": f"Coupled Aster/hydraulic update failed from {reached:.6g} to {target:.6g}: {detail}"})
        print(f"ASTER_COHESIVE_STEP failed load_factor={target:.6g}", flush=True)
        break
'''
