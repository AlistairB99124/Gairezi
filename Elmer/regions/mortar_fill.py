"""Mortar fill on the wall's thickness steps.

A stepped wall leaves a horizontal ledge on its downstream face at each step. The fill
is a sloped body of mortar that sits on a ledge against the wall face above it, so the
downstream face runs smoothly from the ledge's outer edge up to the wall. It is a body of
its own: it has its own nodes, coincident with the wall's where the two touch, so a
solver can join them with interface springs (material.mortar.bond_spring_n_per_m3).

The fill is built from the wall's own boundary triangles after the wall has been split
into tetrahedra. The wall face above a ledge is a regular grid of triangles; each triangle
is swept away from the wall by the fill's thickness at that height, which falls linearly
from the step's width at the ledge to nothing at the top. A ledge that spans several radial
cells of the wall gives the fill as many layers, so the fill's underside repeats the wall's
ledge triangles exactly.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model_groups import FACE_GROUP_IDS

TOLERANCE_M = 1.0e-6
ANGLE_DECIMALS = 9
_TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


class MortarFillError(ValueError):
    """The wall mesh does not have the shape the fill is built on."""


@dataclass
class MortarFill:
    """The fill's own mesh. Face and cell rows index ``points``; ``wall_faces`` index the wall's boundary list."""
    points: np.ndarray
    tetrahedra: np.ndarray
    tie_faces: np.ndarray       # coincident with the wall faces the fill covers
    slope_faces: np.ndarray     # the new downstream face
    end_faces: np.ndarray       # where a fill ends against the abutment
    wall_faces: list[int]       # the wall's boundary faces that the fill covers
    report: dict = field(default_factory=dict)


def _fit_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    matrix = np.column_stack([2 * points, np.ones(len(points))])
    solution, *_ = np.linalg.lstsq(matrix, (points ** 2).sum(axis=1), rcond=None)
    centre = solution[:2]
    return centre, float(np.sqrt(solution[2] + centre @ centre))


def _volumes(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    corners = points[tetrahedra]
    return np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0


def _chain_ranks(prefer_first: list[bool]) -> list[tuple[int, int]]:
    """Sort keys for the nodes along a ledge so that node j sorts before node j + 1 exactly where asked."""
    ranks, rank = [(0, 0)], 0
    for index, first in enumerate(prefer_first, start=1):
        rank += 1 if first else -1
        ranks.append((rank, index))
    return ranks


def _build_step(step: dict, nodes: np.ndarray, faces: np.ndarray, face_rows: np.ndarray,
                radius: np.ndarray, angle: np.ndarray, centre: np.ndarray) -> dict | None:
    """The fill on one step, or None when the wall has no ledge there (the rock is above the step everywhere)."""
    z0, wall_radius, edge_radius = step["top_z_m"], step["wall_face_radius_m"], step["ledge_edge_radius_m"]
    label = f"step at z = {z0:g} m"
    z = nodes[:, 2]
    on_ledge = ((np.abs(z[faces] - z0) < TOLERANCE_M).all(axis=1)
                & (radius[faces] > edge_radius - TOLERANCE_M).all(axis=1)
                & (radius[faces] < wall_radius + TOLERANCE_M).all(axis=1))
    if not on_ledge.any():
        return None
    ledge, ledge_nodes = faces[on_ledge], np.unique(faces[on_ledge])

    # Radial layers of the ledge, from the wall face out to the ledge's edge.
    layer_radii = np.unique(np.round(radius[ledge_nodes], 6))[::-1]
    if abs(layer_radii[0] - wall_radius) > TOLERANCE_M or abs(layer_radii[-1] - edge_radius) > TOLERANCE_M:
        raise MortarFillError(f"The ledge of the {label} does not span from the wall face to the step's edge")
    rows = []
    for layer_radius in layer_radii:
        row = ledge_nodes[np.abs(radius[ledge_nodes] - layer_radius) < TOLERANCE_M]
        rows.append(row[np.argsort(angle[row])])
    chain = rows[0]
    if len(chain) < 2 or any(len(row) != len(chain) or not np.allclose(angle[row], angle[chain], atol=1e-8)
                             for row in rows):
        raise MortarFillError(f"The ledge of the {label} is not a regular strip of the wall mesh")
    fractions = (wall_radius - layer_radii) / (wall_radius - edge_radius)

    # Courses of the wall face above the ledge; the fill's top snaps to the one nearest the configured rise.
    on_wall_face = np.unique(faces[(np.abs(radius[faces] - wall_radius) < TOLERANCE_M).all(axis=1)])
    column = on_wall_face[np.abs(angle[on_wall_face] - angle[chain[0]]) < 1e-8]
    courses = np.unique(np.round(z[column], 6))
    courses = courses[(courses > z0 - TOLERANCE_M) & (courses < step["limit_z_m"] + TOLERANCE_M)]
    if len(courses) < 2:
        raise MortarFillError(f"The wall has no course above the {label} for the fill to stand against")
    top = 1 + int(np.argmin(np.abs(courses[1:] - (z0 + step["rise_m"]))))
    courses = courses[:top + 1]
    rise = float(courses[-1] - z0)

    located = {(round(float(angle[node]), ANGLE_DECIMALS), round(float(z[node]), 6)): int(node) for node in on_wall_face}
    grid = np.empty((len(chain), len(courses)), dtype=np.int64)
    for j, start in enumerate(chain):
        for k, level in enumerate(courses):
            node = located.get((round(float(angle[start]), ANGLE_DECIMALS), round(float(level), 6)))
            if node is None:
                raise MortarFillError(f"The wall face above the {label} is not a regular grid of courses")
            grid[j, k] = node
    position = {int(node): (j, k) for (j, k), node in np.ndenumerate(grid)}
    on_grid = np.isin(faces, grid).all(axis=1) & ~on_ledge
    wall_face = faces[on_grid]
    if len(wall_face) != 2 * (len(chain) - 1) * (len(courses) - 1):
        raise MortarFillError(f"The wall face above the {label} is not fully exposed where the fill would stand")

    # The fill's own nodes: (layer, station, course). Layer 0 lies on the wall face; the top course is one line.
    points: list[np.ndarray] = []
    index: dict[tuple[int, int, int], int] = {}

    def node_id(layer: int, j: int, k: int) -> int:
        key = (0 if k == len(courses) - 1 else layer, j, k)
        if key not in index:
            wall_node = grid[j, k]
            if key[0] == 0:
                point = nodes[wall_node]
            elif k == 0:
                point = nodes[rows[key[0]][j]]  # exactly the wall's ledge node
            else:
                thickness = step["run_m"] * (1.0 - (courses[k] - z0) / rise)
                outward = (nodes[wall_node, :2] - centre) / radius[wall_node]
                point = np.array([*(centre + (wall_radius - thickness * fractions[key[0]]) * outward), courses[k]])
            index[key] = len(points)
            points.append(np.asarray(point, dtype=float))
        return index[key]

    ledge_triangles = {tuple(sorted(triangle)) for triangle in ledge.tolist()}
    tetrahedra: list[tuple[int, int, int, int]] = []
    for layer in range(len(layer_radii) - 1):
        inner, outer = rows[layer], rows[layer + 1]
        prefer_first = []
        for j in range(len(chain) - 1):
            p, q, p_out, q_out = int(inner[j]), int(inner[j + 1]), int(outer[j]), int(outer[j + 1])
            if (tuple(sorted((p, q, q_out))) in ledge_triangles
                    and tuple(sorted((p, q_out, p_out))) in ledge_triangles):
                prefer_first.append(True)   # the wall cut this ledge quad from p to q_out
            elif (tuple(sorted((q, p, p_out))) in ledge_triangles
                  and tuple(sorted((q, p_out, q_out))) in ledge_triangles):
                prefer_first.append(False)
            else:
                raise MortarFillError(f"The ledge of the {label} is not two triangles per mesh cell")
        ranks = _chain_ranks(prefer_first)
        after_chain = len(chain) + 1

        def key(wall_node: int) -> tuple[int, int]:
            j, k = position[int(wall_node)]
            return ranks[j] if k == 0 else (after_chain, int(wall_node))

        for triangle in wall_face:
            ordered = sorted((int(node) for node in triangle), key=key)
            a = [node_id(layer, *position[node]) for node in ordered]
            b = [node_id(layer + 1, *position[node]) for node in ordered]
            for cell in ((a[0], a[1], a[2], b[2]), (a[0], a[1], b[1], b[2]), (a[0], b[0], b[1], b[2])):
                if len(set(cell)) == 4:  # a cell that touches the top line loses the corners that coincide there
                    tetrahedra.append(cell)
    own = np.asarray(points)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    volumes = _volumes(own, cells)
    if np.any(np.abs(volumes) < 1e-12):
        raise MortarFillError(f"The fill on the {label} has a cell with no volume")
    cells[volumes < 0] = cells[volumes < 0][:, [0, 2, 1, 3]]

    # Exterior faces by where they lie.
    logical = {own_id: key for key, own_id in index.items()}
    counts: Counter[tuple[int, ...]] = Counter()
    oriented = {}
    for cell in cells:
        for pattern in _TET_FACES:
            face_key = tuple(sorted(cell[list(pattern)].tolist()))
            counts[face_key] += 1
            oriented[face_key] = (cell[list(pattern)], cell)
    tie, slope, ends = [], [], []
    last_station = len(chain) - 1
    for face_key, count in counts.items():
        if count != 1:
            continue
        face, cell = oriented[face_key]
        # Outward orientation: away from the cell's fourth node.
        opposite = own[next(node for node in cell if node not in face_key)]
        normal = np.cross(own[face[1]] - own[face[0]], own[face[2]] - own[face[0]])
        if normal @ (opposite - own[face[0]]) > 0:
            face = face[[0, 2, 1]]
        keys = [logical[int(node)] for node in face]
        if all(layer == 0 for layer, _, _ in keys) or all(k == 0 for _, _, k in keys):
            tie.append(face)
        elif all(j == 0 for _, j, _ in keys) or all(j == last_station for _, j, _ in keys):
            ends.append(face)
        else:
            slope.append(face)
    covered = np.concatenate([face_rows[on_ledge], face_rows[on_grid]])
    span = float(angle[chain[-1]] - angle[chain[0]])
    return {
        "points": own, "tetrahedra": cells, "tie_faces": np.asarray(tie, dtype=np.int64).reshape(-1, 3),
        "slope_faces": np.asarray(slope, dtype=np.int64).reshape(-1, 3),
        "end_faces": np.asarray(ends, dtype=np.int64).reshape(-1, 3), "wall_faces": covered,
        "report": {
            "top_z_m": z0, "run_m": step["run_m"], "requested_rise_m": step["rise_m"], "built_rise_m": rise,
            "built_slope_run_per_rise": step["run_m"] / rise, "top_of_fill_z_m": float(courses[-1]),
            "courses": int(len(courses) - 1), "radial_layers": int(len(layer_radii) - 1),
            "stations": int(len(chain)), "arc_length_at_wall_face_m": wall_radius * span,
            "volume_m3": float(np.abs(volumes).sum()), "tetrahedra": int(len(cells)), "nodes": int(len(own)),
            "tie_triangles": int(len(tie)), "slope_triangles": int(len(slope)), "end_triangles": int(len(ends)),
        },
    }


def wall_with_fill(root: Path):
    """The wall of ``root / config.json`` as the solvers see it, and its fill (None without one).

    The wall is split into tetrahedra and moved to the survey frame exactly as
    Elmer.three_body_diagnostic.prepare does before it adds the foundation.
    """
    from ..model_config import load_config, mortar_fill
    from ..three_body_diagnostic import survey_coordinates
    from .combined_structure import build_combined_structure
    from .tetrahedral import tetrahedralize_structure

    config = load_config(root / "config.json")
    wall = build_combined_structure(root)
    wall.nodes = [tuple(point) for point in survey_coordinates(
        np.asarray(wall.nodes), config["geometry"]["wall"]["centerline_radius_m"], 80.)]
    wall = tetrahedralize_structure(wall)
    definition = mortar_fill(config)
    return wall, None if definition is None else build_mortar_fill(definition, wall.nodes, wall.boundaries)


def build_mortar_fill(definition: dict, nodes, boundaries) -> MortarFill | None:
    """The fill for ``definition`` (Elmer.model_config.mortar_fill) on a wall already split into tetrahedra.

    ``nodes`` and ``boundaries`` (tag, triangle) are the wall's, in the frame the solvers use: the
    arch axis is vertical and is found from the upstream face. Returns None when no step of this
    wall has a ledge.
    """
    points = np.asarray(nodes, dtype=float)
    tags = np.asarray([tag for tag, _ in boundaries])
    triangles = [face for _, face in boundaries]
    if any(len(face) != 3 for face in triangles):
        raise MortarFillError("The fill is built on a wall that has been split into tetrahedra")
    triangles = np.asarray(triangles, dtype=np.int64)
    upstream = np.unique(triangles[tags == FACE_GROUP_IDS["UPSTREAM"]])
    if not len(upstream):
        raise MortarFillError("The wall has no upstream face to find the arch axis from")
    centre, _ = _fit_circle(points[upstream, :2])
    radius = np.linalg.norm(points[:, :2] - centre, axis=1)
    angle = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    downstream_rows = np.flatnonzero(tags == FACE_GROUP_IDS["DOWNSTREAM"])
    downstream = triangles[downstream_rows]

    parts, skipped = [], []
    for step in definition["steps"]:
        part = _build_step(step, points, downstream, downstream_rows, radius, angle, centre)
        if part is None:
            skipped.append(step["top_z_m"])
        else:
            parts.append(part)
    if not parts:
        return None
    offsets = np.cumsum([0] + [len(part["points"]) for part in parts])
    wall_faces = np.concatenate([part["wall_faces"] for part in parts])
    if len(np.unique(wall_faces)) != len(wall_faces):
        raise MortarFillError("Two fills cover the same wall face")
    fill = MortarFill(
        points=np.vstack([part["points"] for part in parts]),
        tetrahedra=np.vstack([part["tetrahedra"] + offset for part, offset in zip(parts, offsets)]),
        tie_faces=np.vstack([part["tie_faces"] + offset for part, offset in zip(parts, offsets)]),
        slope_faces=np.vstack([part["slope_faces"] + offset for part, offset in zip(parts, offsets)]),
        end_faces=np.vstack([part["end_faces"] + offset for part, offset in zip(parts, offsets)]),
        wall_faces=[int(row) for row in wall_faces],
    )
    # Every covered wall triangle has exactly one coincident fill triangle.
    def located(coordinates: np.ndarray) -> list[tuple]:
        return sorted(tuple(sorted(tuple(point) for point in np.round(face, 8))) for face in coordinates)
    if located(fill.points[fill.tie_faces]) != located(points[triangles[fill.wall_faces]]):
        raise MortarFillError("The fill's underside does not repeat the wall faces it covers")
    volumes = _volumes(fill.points, fill.tetrahedra)
    edges = fill.points[fill.tetrahedra][:, [0, 0, 0, 1, 1, 2]] - fill.points[fill.tetrahedra][:, [1, 2, 3, 2, 3, 3]]
    longest = np.linalg.norm(edges, axis=2).max(axis=1)
    fill.report = {
        "steps": [part["report"] for part in parts], "steps_without_a_ledge_z_m": skipped,
        "volume_m3": float(volumes.sum()), "tetrahedra": int(len(fill.tetrahedra)), "nodes": int(len(fill.points)),
        "requested_slope_run_per_rise": definition["slope_run_per_rise"],
        # Longest edge cubed over volume: 8.5 for a regular tetrahedron, larger for slivers.
        "largest_edge_cubed_per_volume": float((longest ** 3 / volumes).max()),
        "smallest_tetrahedron_volume_m3": float(volumes.min()),
    }
    return fill
