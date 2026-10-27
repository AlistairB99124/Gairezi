"""Whole-model crack mesh: an exact seed slab and a quadratic damage band in the wall.

The three-body baseline mesh is reused unchanged except for the wall. Wall cells
crossing the seed faces z = Z -/+ t/2 are cut so the shared seed slab is a conforming
layer of cells whose Gauss points lie inside the seed. Wall cells near Z form the
quadratic damage band; every other cell keeps its linear baseline element. Band
mid-edge nodes shared with linear cells are returned so the case can tie them to the
linear trace. The wall shares no nodes with the foundation or bedrock (it is coupled
by springs), so cutting the wall never changes those bodies.
"""
from __future__ import annotations

import numpy as np

TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
TET_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
TRI_EDGES = ((0, 1), (1, 2), (2, 0))
FACE_ARRAYS = ("tri3_upstream", "tri3_downstream", "tri3_crest")
# Nodes closer than this to a cutting plane are treated as lying on it.
PLANE_TOLERANCE_M = 1e-9


def signed_volumes(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    points = np.asarray(nodes)[np.asarray(tets)]
    return np.einsum("ij,ij->i", np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
                     points[:, 3] - points[:, 0]) / 6


def _clip(polygon, side, sign, cut):
    """Clip a cyclic vertex list to ``sign * side >= 0``, preserving cyclic order."""
    kept = []
    for index, current in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        if side[current] * sign >= 0:
            kept.append(current)
        if side[current] * side[following] < 0:
            kept.append(cut(current, following))
    unique = []
    for vertex in kept:
        if not unique or unique[-1] != vertex:
            unique.append(vertex)
    if len(unique) > 1 and unique[0] == unique[-1]:
        unique.pop()
    return unique


def _triangulate(polygon):
    """Fan from the lowest node id, so neighbours split a shared quad identically."""
    if len(polygon) < 3:
        return []
    start = polygon.index(min(polygon))
    ordered = polygon[start:] + polygon[:start]
    return [(ordered[0], ordered[i], ordered[i + 1]) for i in range(1, len(ordered) - 1)]


def slice_plane(nodes: np.ndarray, tets: np.ndarray, selected: np.ndarray, faces: dict,
                elevation: float, face_origins: dict | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Cut selected tetrahedra (and their boundary triangles) by the plane z = elevation.

    Returns new nodes (old ids unchanged, cut nodes appended), new cells, the parent
    cell index of every new cell and the new face arrays. Each piece of a cut cell is
    convex, so a fan from its lowest node id over its other faces gives positive tets.
    If supplied, face_origins is updated in place alongside the new face arrays.
    """
    nodes = np.asarray(nodes, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    distance = nodes[:, 2] - elevation
    side = np.where(np.abs(distance) <= PLANE_TOLERANCE_M, 0, np.sign(distance)).astype(int)
    cell_sides = side[tets]
    crossing = np.asarray(selected, dtype=bool) & (cell_sides.max(1) > 0) & (cell_sides.min(1) < 0)
    added: list[np.ndarray] = []
    cut_ids: dict[tuple[int, int], int] = {}

    def cut(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in cut_ids:
            fraction = distance[key[0]] / (distance[key[0]] - distance[key[1]])
            point = nodes[key[0]] + fraction * (nodes[key[1]] - nodes[key[0]])
            point[2] = elevation
            cut_ids[key] = len(nodes) + len(added)
            added.append(point)
        return cut_ids[key]

    side_of = _SideLookup(side)
    new_cells, parents = [], []
    for index in np.flatnonzero(crossing):
        cell = [int(value) for value in tets[index]]
        for sign in (1, -1):
            polygons = [_clip([cell[i] for i in face], side_of, sign, cut) for face in TET_FACES]
            polygons = [polygon for polygon in polygons if len(polygon) >= 3]
            cap = sorted({vertex for polygon in polygons for vertex in polygon
                          if side_of[vertex] == 0})
            if len(cap) >= 3:
                points = _coordinates(nodes, added, cap)
                centre = points.mean(0)
                order = np.argsort(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))
                polygons.append([cap[i] for i in order])
            apex = min(vertex for polygon in polygons for vertex in polygon)
            for polygon in polygons:
                if apex in polygon:
                    continue
                for triangle in _triangulate(polygon):
                    new_cells.append((apex, *triangle))
                    parents.append(index)
    all_nodes = np.vstack([nodes, np.asarray(added).reshape(-1, 3)])
    pieces = np.asarray(new_cells, dtype=np.int64).reshape(-1, 4)
    volumes = signed_volumes(all_nodes, pieces)
    flip = volumes < 0
    pieces[flip] = pieces[flip][:, [0, 2, 1, 3]]
    if np.any(np.abs(volumes) <= 1e-18):
        raise ValueError("Cutting the wall at the seed plane produced a degenerate cell")
    parent_volumes = np.abs(signed_volumes(nodes, tets[crossing]))
    child_volumes = np.bincount(np.searchsorted(np.flatnonzero(crossing), parents),
                                weights=np.abs(volumes), minlength=len(parent_volumes))
    if not np.allclose(child_volumes, parent_volumes, rtol=1e-9, atol=1e-15):
        raise ValueError("Cut cells do not conserve the parent cell volume")
    keep = np.flatnonzero(~crossing)
    cells = np.vstack([tets[keep], pieces])
    parent = np.concatenate([keep, np.asarray(parents, dtype=np.int64)])
    new_faces = {}
    for name, triangles in faces.items():
        triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        triangle_sides = side[triangles]
        cut_faces = (triangle_sides.max(1) > 0) & (triangle_sides.min(1) < 0)
        rows = [triangles[~cut_faces]]
        origins = [face_origins[name][~cut_faces]] if face_origins is not None else []
        for index in np.flatnonzero(cut_faces):
            triangle = triangles[index]
            for sign in (1, -1):
                polygon = _clip([int(value) for value in triangle], side_of, sign, _strict(cut_ids))
                if len(polygon) >= 3:
                    pieces = np.asarray(_triangulate(polygon), dtype=np.int64)
                    rows.append(pieces)
                    if face_origins is not None:
                        origins.append(np.full(len(pieces), face_origins[name][index], dtype=np.int64))
        new_faces[name] = np.vstack(rows)
        if face_origins is not None:
            face_origins[name] = np.concatenate(origins)
    return all_nodes, cells, parent, new_faces


class _SideLookup:
    """Side of an original node; cut nodes always lie on the plane."""

    def __init__(self, side: np.ndarray):
        self.side = side

    def __getitem__(self, vertex: int) -> int:
        return int(self.side[vertex]) if vertex < len(self.side) else 0


def _strict(cut_ids):
    def lookup(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in cut_ids:
            raise ValueError("A cut boundary face edge is not an edge of a cut wall cell")
        return cut_ids[key]
    return lookup


def _coordinates(nodes, added, ids):
    return np.asarray([nodes[i] if i < len(nodes) else added[i - len(nodes)] for i in ids])


def elevate_band(nodes: np.ndarray, cells: np.ndarray, band: np.ndarray, faces: dict) -> dict:
    """TETRA10 band cells (MED edge order) with -1 padding for retained TETRA4 cells.

    Band mid-edge nodes lying on an edge of a linear cell are hanging; the case ties
    them to the linear trace so displacement stays continuous.
    """
    cells = np.asarray(cells, dtype=np.int64)
    band = np.asarray(band, dtype=bool)
    if not band.any():
        raise ValueError("The damage band contains no wall cells")
    edges = np.sort(np.stack([cells[:, list(pair)] for pair in TET_EDGES], axis=1), axis=2)
    stride = len(nodes)
    keys = edges[..., 0] * stride + edges[..., 1]
    band_keys = np.unique(keys[band])
    midpoint_ids = len(nodes) + np.arange(len(band_keys))
    quadratic = np.full((len(cells), 10), -1, dtype=np.int64)
    quadratic[:, :4] = cells
    quadratic[band, 4:] = midpoint_ids[np.searchsorted(band_keys, keys[band])]
    ends = np.column_stack([band_keys // stride, band_keys % stride])
    midpoints = (nodes[ends[:, 0]] + nodes[ends[:, 1]]) / 2
    linear_keys = np.unique(keys[~band])
    hanging = np.isin(band_keys, linear_keys)
    neighbours = ~band & np.isin(keys, band_keys[hanging]).any(1)
    band_faces = {tuple(face) for face in np.sort(
        np.concatenate([cells[band][:, list(face)] for face in TET_FACES]), axis=1).tolist()}
    elevated_faces = {}
    for name, triangles in faces.items():
        triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        face_keys = np.sort(np.stack([triangles[:, list(pair)] for pair in TRI_EDGES], axis=1), axis=2)
        face_keys = face_keys[..., 0] * stride + face_keys[..., 1]
        position = np.minimum(np.searchsorted(band_keys, face_keys), len(band_keys) - 1)
        quadratic_face = np.array([tuple(face) in band_faces
                                   for face in np.sort(triangles, axis=1).tolist()], dtype=bool)
        if not (band_keys[position[quadratic_face]] == face_keys[quadratic_face]).all():
            raise ValueError("A band boundary face edge has no band mid-edge node")
        rows = np.full((len(triangles), 6), -1, dtype=np.int64)
        rows[:, :3] = triangles
        rows[quadratic_face, 3:] = midpoint_ids[position[quadratic_face]]
        elevated_faces[name] = rows
    return {
        "nodes": np.vstack([nodes, midpoints]), "cells": quadratic,
        "faces": elevated_faces, "hanging_nodes": midpoint_ids[hanging],
        "hanging_master_cells": np.flatnonzero(neighbours),
    }


def crack_faces(nodes: np.ndarray, cells: np.ndarray, seed_layer: np.ndarray, geometry: dict) -> dict:
    """TRIA6 faces bounding the seed slab inside the seeded sector, ordered for PRES_REP.

    Each face lies on a cut plane z = Z -/+ t/2 and belongs to a seed-layer cell. Its
    node order makes the right-hand normal point into the slab (outward from the solid
    it loads), so a positive PRES_REP pushes the solids above and below apart, as water
    in the crack does. Faces are kept where their centroid satisfies the same sector test
    as the native seed Gauss points.
    """
    nodes = np.asarray(nodes, dtype=float)
    cells = np.asarray(cells, dtype=np.int64)
    inner, outer, start, end = (float(value) for value in geometry["annular_sector"])
    elevation = float(geometry["elevation_m"])
    half = float(geometry["seed_thickness_m"]) / 2
    if cells.shape[1] != 10 or np.any(cells[np.asarray(seed_layer, dtype=bool), 4:] < 0):
        raise ValueError("Crack faces need quadratic seed-layer cells")
    rows, areas = [], {"top": 0.0, "bottom": 0.0}
    edge_column = {frozenset(pair): 4 + index for index, pair in enumerate(TET_EDGES)}
    for cell in cells[np.flatnonzero(seed_layer)]:
        for local in TET_FACES:
            corner = cell[list(local)]
            heights = nodes[corner, 2]
            for name, plane, inward in (("top", elevation + half, -1.0), ("bottom", elevation - half, 1.0)):
                if np.abs(heights - plane).max() > PLANE_TOLERANCE_M:
                    continue
                centre = nodes[corner].mean(0)
                radius, angle = float(np.hypot(centre[0], centre[1])), float(np.arctan2(centre[0], centre[1]))
                if not (inner <= radius <= outer and start <= angle <= end):
                    continue
                order = list(local)
                normal = np.cross(*(nodes[cell[order[1:]]] - nodes[cell[order[0]]]))
                if normal[2] * inward < 0:
                    order = [order[0], order[2], order[1]]
                    normal = -normal
                corners = [int(cell[i]) for i in order]
                middles = [int(cell[edge_column[frozenset((order[a], order[b]))]]) for a, b in TRI_EDGES]
                rows.append(corners + middles)
                areas[name] += float(np.linalg.norm(normal)) / 2
    if not rows or not areas["top"] or not areas["bottom"]:
        raise ValueError("The seed slab has no crack faces on both cut planes inside the sector")
    nominal = (outer ** 2 - inner ** 2) * (end - start) / 2
    return {"faces": np.asarray(rows, dtype=np.int64),
            "report": {"crack_faces": len(rows), "top_area_m2": areas["top"],
                       "bottom_area_m2": areas["bottom"], "nominal_seed_area_m2": nominal}}


def build(arrays: dict, geometry: dict, band_half_height_m: float) -> dict:
    """Cut the wall at both seed faces, then elevate the wall band around the seed."""
    elevation = float(geometry["elevation_m"])
    half = float(geometry["seed_thickness_m"]) / 2
    if not (np.isfinite(band_half_height_m) and band_half_height_m > 2 * half):
        raise ValueError("The damage band must be thicker than the seed")
    nodes = np.asarray(arrays["nodes"], dtype=float)
    concrete = np.asarray(arrays["tetra4_concrete"], dtype=np.int64)
    bedrock = np.asarray(arrays["tetra4_bedrock"], dtype=np.int64)
    bodies = np.asarray(arrays["concrete_body_ids"], dtype=np.int64)
    wall = bodies == 1
    wall_nodes = np.unique(concrete[wall])
    if np.intersect1d(wall_nodes, concrete[~wall]).size or np.intersect1d(wall_nodes, bedrock).size:
        raise ValueError("The wall shares nodes with the foundation or bedrock; it cannot be cut alone")
    faces = {name: np.asarray(arrays[name], dtype=np.int64) for name in FACE_ARRAYS}
    for side in ("slave", "master"):
        faces[f"joint_{side}"] = np.asarray(arrays[f"joint_{side}_tri3"], dtype=np.int64)
    if len(faces["joint_slave"]) != len(faces["joint_master"]):
        raise ValueError("Incomplete source interface face correspondence")
    wall_faces = {name: triangles[np.isin(triangles, wall_nodes).all(1)] for name, triangles in faces.items()}
    other_faces = {name: triangles[~np.isin(triangles, wall_nodes).all(1)] for name, triangles in faces.items()}
    wall_origins = {name: np.flatnonzero(np.isin(triangles, wall_nodes).all(1))
                    for name, triangles in faces.items()}
    other_origins = {name: np.flatnonzero(~np.isin(triangles, wall_nodes).all(1))
                     for name, triangles in faces.items()}
    cells, body = concrete, bodies
    report = {"original_concrete_cells": int(len(concrete)), "cut_planes_m": []}
    for plane in (elevation - half, elevation + half):
        before = len(cells)
        nodes, cells, parent, wall_faces = slice_plane(nodes, cells, body == 1, wall_faces, plane,
                                                      wall_origins)
        body = body[parent]
        report["cut_planes_m"].append({"z_m": plane, "cells_added": int(len(cells) - before)})
    centroid_z = nodes[cells, 2].mean(1)
    band = (body == 1) & (np.abs(centroid_z - elevation) <= band_half_height_m)
    seed_layer = (body == 1) & (np.abs(nodes[cells, 2] - elevation) <= half + PLANE_TOLERANCE_M).all(1)
    if not seed_layer.any():
        raise ValueError("No wall cells lie inside the seed slab after cutting")
    if not band[seed_layer].all():
        raise ValueError("The damage band does not contain the whole seed slab")
    if np.any(signed_volumes(nodes, cells) <= 0):
        raise ValueError("The cut wall mesh contains inverted cells")
    all_faces = {name: np.vstack([other_faces[name], wall_faces[name]]) for name in faces}
    elevated = elevate_band(nodes, cells, band, all_faces)
    from Aster.refined_interface import interface_blocks
    source_xyz = np.asarray(arrays["nodes"])[arrays["joint_slave_tri3"]]
    source_areas = np.linalg.norm(np.cross(source_xyz[:, 1] - source_xyz[:, 0],
                                          source_xyz[:, 2] - source_xyz[:, 0]), axis=1) / 2
    joint_pairs, joint_blocks, joint_audit = interface_blocks(
        elevated["nodes"], elevated["faces"]["joint_slave"], elevated["faces"]["joint_master"],
        np.concatenate([other_origins["joint_slave"], wall_origins["joint_slave"]]),
        np.concatenate([other_origins["joint_master"], wall_origins["joint_master"]]),
        arrays["joint_coefficients_n_per_m3"], source_areas_m2=source_areas)
    crack = crack_faces(elevated["nodes"], elevated["cells"], seed_layer, geometry)
    report.update({
        "concrete_cells": int(len(cells)), "damage_band_cells": int(band.sum()),
        "seed_layer_cells": int(seed_layer.sum()), "hanging_nodes": int(len(elevated["hanging_nodes"])),
        "band_half_height_m": band_half_height_m, "nodes": int(len(elevated["nodes"])),
        **crack["report"],
        "interface_integration": joint_audit,
    })
    return {**elevated, "body_ids": body, "band": band, "seed_layer": seed_layer,
            "joint_block_pairs": joint_pairs, "joint_block_matrices_n_per_m": joint_blocks,
            "crack_faces": crack["faces"], "report": report}


def archive_arrays(arrays: dict, built: dict) -> dict:
    """Arrays for convert_whole_model_to_med.py: concrete rows first, then bedrock."""
    concrete = built["cells"]
    bedrock = np.asarray(arrays["tetra4_bedrock"], dtype=np.int64)
    padded_bedrock = np.column_stack([bedrock, np.full((len(bedrock), 6), -1, dtype=np.int64)])
    cells = np.vstack([concrete, padded_bedrock])
    count, rock = len(concrete), np.zeros(len(bedrock), dtype=bool)
    body = built["body_ids"]
    master = np.zeros(count, dtype=bool)
    master[built["hanging_master_cells"]] = True

    def concrete_mask(mask):
        return np.concatenate([mask, rock])

    def pad(triangles):
        triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        return np.column_stack([triangles, np.full((len(triangles), 3), -1, dtype=np.int64)])

    return {
        "nodes": built["nodes"], "cells": cells,
        "group_concrete": concrete_mask(np.ones(count, dtype=bool)),
        "group_bedrock": np.concatenate([np.zeros(count, dtype=bool), ~rock]),
        "group_wall": concrete_mask(body == 1), "group_foundation": concrete_mask(body == 3),
        "group_damage_band": concrete_mask(built["band"]),
        "group_seed_layer": concrete_mask(built["seed_layer"]),
        "group_linear_volumes": np.concatenate([~built["band"], ~rock]),
        "group_hanging_master": concrete_mask(master),
        "faces_upstream": built["faces"]["tri3_upstream"],
        "faces_downstream": built["faces"]["tri3_downstream"],
        "faces_crest": built["faces"]["tri3_crest"],
        "faces_crack_faces": built["crack_faces"],
        "faces_rock_bottom": pad(arrays["tri3_rock_bottom"]),
        "faces_rock_sides": pad(arrays["tri3_rock_sides"]),
        "surface_block_pairs": np.vstack([
            built["joint_block_pairs"],
            np.repeat(np.asarray(arrays["support_nodes"], dtype=np.int64)[:, None], 2, axis=1)]),
        "surface_block_matrices_n_per_m": np.concatenate([
            built["joint_block_matrices_n_per_m"], arrays["support_matrices_n_per_m"]]),
        "hanging_nodes": np.asarray(built["hanging_nodes"], dtype=np.int64),
        **crack_node_groups(built),
    }


def crack_node_groups(built: dict) -> dict:
    """Crack-face nodes on the upper and lower seed planes (separation diagnostics)."""
    faces = np.asarray(built["crack_faces"], dtype=np.int64)
    ids = np.unique(faces)
    upper = built["nodes"][ids, 2] > float(np.mean(built["nodes"][ids, 2]))
    return {"crack_top_nodes": ids[upper], "crack_bottom_nodes": ids[~upper]}
