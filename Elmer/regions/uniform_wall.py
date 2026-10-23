"""Curved wall seated directly on the (locally excavated) surveyed bedrock.

The wall runs from the rock surface up to the crest (z = 0) between the
chainages where the rock profile reaches the crest. Every wall node lies on a
horizontal course: at each station the seat is the highest course at or below
the surveyed rock, i.e. the rock is excavated by less than one element. The
rock mesh conforms to this seat.
The upstream face is a single cylinder; optional geometry.wall.thickness_steps
thicken the wall downstream below given elevations, leaving horizontal treads
on the downstream face.
With geometry.foundation present, station tops are the lower adjacent column
treads and the seat interpolates between them. Former risers become approximately
45-degree chamfers; triangular foundation wedges become wall prisms, each
wholly inside one thickness/joint band. Optional
geometry.wall.step_end_taper_ratio narrows each step's extra width where the
seat is shallow below the step (whole radial-strip columns per station
interval), so the step ends taper into the abutment.
With split_joints=True each thickness band is its own concrete block: nodes on
a step level are duplicated and the two sides of every joint are tagged
(joint_boundary_ids) so a solver can join the blocks with springs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path

from .bedrock_contours import CREST_Z_M, bedrock_elevation, load_global_element_size, load_model_contours
from .model_groups import FACE_GROUP_IDS, joint_boundary_ids
from ..model_config import load_config, tapered_wall_downstream_radius_at, wall_thickness_steps


DOWNSTREAM_WALL_RADIUS_M = float(load_config()["geometry"]["wall"]["downstream_radius_m"])
UPSTREAM_RADIUS_M = float(load_config()["geometry"]["wall"]["upstream_radius_m"])
BASE_BOUNDARY_ID = FACE_GROUP_IDS["FOUNDATION"]
UPSTREAM_BOUNDARY_ID = FACE_GROUP_IDS["UPSTREAM"]
DOWNSTREAM_BOUNDARY_ID = FACE_GROUP_IDS["DOWNSTREAM"]
CREST_BOUNDARY_ID = FACE_GROUP_IDS["CREST"]
LEFT_END_BOUNDARY_ID = FACE_GROUP_IDS["LEFT_ABUTMENT"]
RIGHT_END_BOUNDARY_ID = FACE_GROUP_IDS["RIGHT_ABUTMENT"]
OTHER_BOUNDARY_ID = FACE_GROUP_IDS["OTHER_EXTERIOR"]


@dataclass
class UniformWallMesh:
    nodes: list[tuple[float, float, float]]
    cells: list[tuple[int, tuple[int, ...]]]
    boundaries: list[tuple[int, tuple[int, ...]]]
    chainages_m: list[float]
    base_levels_m: list[float]
    element_size_m: float
    # Thickness band (0 = crest block) of every node; all zero for a monolithic wall.
    node_blocks: list[int] | None = None
    # Step elevations (top to bottom) that are split into spring joints.
    joint_levels_m: tuple[float, ...] = ()


def target_chainages(start_m: float, end_m: float, anchors_m: list[float], target_size_m: float) -> list[float]:
    anchors = [start_m] + [value for value in anchors_m if start_m < value < end_m] + [end_m]
    values = [start_m]
    for interval_start_m, interval_end_m in zip(anchors, anchors[1:]):
        segment_count = max(1, round((interval_end_m - interval_start_m) / target_size_m))
        values.extend(
            interval_start_m + (interval_end_m - interval_start_m) * index / segment_count
            for index in range(1, segment_count + 1)
        )
    return values


def excavated_seat_level(
    bedrock_z_m: float,
    element_size_m: float,
    step_levels_m: tuple[float, ...] = (),
    tolerance_m: float = 1.0e-6,
) -> float:
    """Highest horizontal wall course at or below the surveyed rock (or the crest).

    The wall keeps perfectly horizontal courses; the rock under the wall is
    excavated by less than one element so the seat lies on a course.
    """
    if bedrock_z_m >= CREST_Z_M - tolerance_m:
        return CREST_Z_M
    grid_z_m = math.floor((bedrock_z_m + tolerance_m) / element_size_m) * element_size_m
    candidates = [grid_z_m, *(z_m for z_m in step_levels_m if z_m <= bedrock_z_m + tolerance_m)]
    return round(max(candidates), 9)


def vertical_levels(
    bottom_z_m: float,
    top_z_m: float,
    element_size_m: float,
    step_levels_m: tuple[float, ...] = (),
) -> list[float]:
    """Seat, global grid courses at least half an element clear of both ends, and crest.

    Thickness-step elevations become courses too (when at least half an element
    above the seat), so each step tread is a flat course of the mesh.
    """
    if top_z_m - bottom_z_m <= 1.0e-9:
        return [round(top_z_m, 9)]
    clearance_m = 0.5 * element_size_m
    levels = [bottom_z_m]
    first = math.ceil((bottom_z_m + clearance_m) / element_size_m)
    last = math.floor((top_z_m - clearance_m) / element_size_m)
    levels.extend(index * element_size_m for index in range(first, last + 1))
    levels.extend(z_m for z_m in step_levels_m if bottom_z_m + clearance_m <= z_m < top_z_m)
    levels.append(top_z_m)
    return sorted(set(round(level, 9) for level in levels))


def radial_node_radii(
    downstream_radii_m: list[float],
    upstream_radius_m: float,
    element_size_m: float,
    extra_radial_fractions: tuple[float, ...] = (),
) -> list[float]:
    """Radii from the deepest downstream face to the upstream face.

    Every downstream step radius is a node radius; each interval between them is
    split into near-element-size divisions. Extra fractions are measured over
    the full span (innermost downstream radius to upstream radius).
    """
    anchors = sorted({*downstream_radii_m, upstream_radius_m})
    radii = {anchors[0]}
    for inner_m, outer_m in zip(anchors, anchors[1:]):
        divisions = max(1, math.ceil((outer_m - inner_m) / element_size_m - 1.0e-9))
        radii.update(inner_m + (outer_m - inner_m) * index / divisions for index in range(1, divisions + 1))
    span_m = upstream_radius_m - anchors[0]
    for fraction in extra_radial_fractions:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("Extra radial fractions must lie within the wall thickness")
        radii.add(anchors[0] + fraction * span_m)
    return sorted({round(radius, 9) for radius in radii})


def _triangulate_fan(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    return [
        (vertices[0], vertices[index], vertices[index + 1])
        for index in range(1, len(vertices) - 1)
        if len({side for side, _ in (vertices[0], vertices[index], vertices[index + 1])}) > 1
    ]


def top_aligned_section_faces(
    left_levels: list[float],
    right_levels: list[float],
) -> list[tuple[tuple[int, int], ...]]:
    """Quads between matching crest-down courses; triangles take up extra seat levels."""
    left_extra = max(len(left_levels) - len(right_levels), 0)
    right_extra = max(len(right_levels) - len(left_levels), 0)
    faces = _triangulate_fan(
        [(0, index) for index in range(left_extra + 1)]
        + [(1, index) for index in range(right_extra, -1, -1)]
    )
    faces.extend(
        (
            (0, left_extra + index),
            (1, right_extra + index),
            (1, right_extra + index + 1),
            (0, left_extra + index + 1),
        )
        for index in range(min(len(left_levels), len(right_levels)) - 1)
    )
    return faces


def _determinant(a, b, c) -> float:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _edge(nodes, cell, start: int, end: int) -> tuple[float, float, float]:
    first, second = nodes[cell[start] - 1], nodes[cell[end] - 1]
    return tuple(second[index] - first[index] for index in range(3))


def _prism_orientation(nodes: list[tuple[float, float, float]], cell: tuple[int, ...]) -> float:
    return _determinant(_edge(nodes, cell, 0, 1), _edge(nodes, cell, 0, 2), _edge(nodes, cell, 0, 3))


def _hex_orientation(nodes: list[tuple[float, float, float]], cell: tuple[int, ...]) -> float:
    return _determinant(_edge(nodes, cell, 0, 1), _edge(nodes, cell, 0, 3), _edge(nodes, cell, 0, 4))


def build_uniform_wall(
    root: Path,
    start_chainage_m: float,
    end_chainage_m: float,
    anchor_chainages_m: list[float],
    extra_radial_fractions: tuple[float, ...] = (),
    split_joints: bool = False,
) -> UniformWallMesh:
    element_size_m = load_global_element_size(root / "config.json")
    geometry = load_config(root / "config.json")["geometry"]
    bedrock_rows = geometry["bedrock"]["contours"]
    wall = geometry["wall"]
    centerline_radius_m = float(wall["centerline_radius_m"])
    upstream_radius_m = float(wall["upstream_radius_m"])
    bands = wall_thickness_steps(wall)
    step_levels_m = tuple(top_z_m for top_z_m, _ in bands[1:])
    downstream_radii_m = [upstream_radius_m - width_m for _, width_m in bands]
    chainages_m = target_chainages(start_chainage_m, end_chainage_m, anchor_chainages_m, element_size_m)
    surveyed_levels_m = [min(bedrock_elevation(bedrock_rows, value), CREST_Z_M) for value in chainages_m]
    base_levels_m = [
        excavated_seat_level(z_m, element_size_m, step_levels_m) for z_m in surveyed_levels_m
    ]
    radii_m = radial_node_radii(downstream_radii_m, upstream_radius_m, element_size_m, extra_radial_fractions)
    section_levels_m = [
        vertical_levels(base_z_m, CREST_Z_M, element_size_m, step_levels_m) for base_z_m in base_levels_m
    ]
    staircase = "foundation" in geometry
    tread_levels_m = [min(a, b) for a, b in zip(base_levels_m, base_levels_m[1:])]
    if staircase:
        if any(abs(a-b)>1e-8 and abs(abs(a-b)-element_size_m)>1e-8
               for a,b in zip(tread_levels_m,tread_levels_m[1:])):
            raise ValueError("Chamfered foundation requires every nonzero riser to be exactly one course")
        base_levels_m = [min(tread_levels_m[max(0,i-1):min(len(tread_levels_m),i+1)])
                         for i in range(len(chainages_m))]
        section_levels_m = [vertical_levels(z, CREST_Z_M, element_size_m, step_levels_m)
                            for z in base_levels_m]
    tolerance = 1.0e-6

    def seat_level_at(station_m: float) -> float:
        # The seat is linear in chainage between stations, like the wall underside.
        for index in range(len(chainages_m) - 1):
            left_m, right_m = chainages_m[index], chainages_m[index + 1]
            if left_m - tolerance <= station_m <= right_m + tolerance:
                weight = min(max((station_m - left_m) / (right_m - left_m), 0.0), 1.0)
                return base_levels_m[index] + weight * (base_levels_m[index + 1] - base_levels_m[index])
        return math.nan

    # Lattice coordinates; only nodes used by a wall cell are kept below.
    lattice: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    for station_index, (chainage_m, levels_m) in enumerate(zip(chainages_m, section_levels_m)):
        angle = chainage_m / centerline_radius_m
        for vertical_index, z_m in enumerate(levels_m):
            for radial_index, radius_m in enumerate(radii_m):
                lattice[(station_index, vertical_index, radial_index)] = (
                    radius_m * math.sin(angle), radius_m * math.cos(angle), z_m,
                )
    split_joints = split_joints and bool(step_levels_m)
    # Cells reference (lattice key, block) pairs; ids are assigned once all cells exist.
    first_touch: dict[tuple[int, int, int], int] = {}
    keyed_cells: list[tuple[int, tuple[tuple[tuple[int, int, int], int], ...]]] = []
    for station_index in range(len(chainages_m) - 1):
        faces = top_aligned_section_faces(section_levels_m[station_index], section_levels_m[station_index + 1])
        face_z_m = [
            sum(section_levels_m[station_index + side][level] for side, level in face) / len(face)
            for face in faces
        ]
        # Shallower end of this station interval; tapered step ends use whole columns.
        column_seat_m = max(base_levels_m[station_index], base_levels_m[station_index + 1])
        for radial_index in range(len(radii_m) - 1):
            for face, z_m in zip(faces, face_z_m):
                # Keep the radial strip only where the wall at this course is thick enough.
                if radii_m[radial_index] < tapered_wall_downstream_radius_at(wall, z_m, column_seat_m) - tolerance:
                    continue
                block = sum(1 for level_m in step_levels_m if z_m < level_m) if split_joints else 0
                inner = tuple((station_index + side, level, radial_index) for side, level in face)
                outer = tuple((station_index + side, level, radial_index + 1) for side, level in face)
                lattice_cell = inner + outer
                if split_joints:
                    # A block's cells must lie wholly within its band, or the joint would cut a cell.
                    top_m = step_levels_m[block - 1] if block else math.inf
                    bottom_m = step_levels_m[block] if block < len(step_levels_m) else -math.inf
                    if any(not bottom_m - tolerance <= lattice[key][2] <= top_m + tolerance for key in lattice_cell):
                        raise ValueError("A wall cell crosses a thickness step; cannot split the wall into blocks")
                for key in lattice_cell:
                    first_touch.setdefault(key, len(first_touch))
                points = [lattice[key] for key in lattice_cell]
                if len(face) == 3:
                    order = (0, 1, 2, 3, 4, 5) if _prism_orientation(points, tuple(range(1, 7))) >= 0.0 else (0, 2, 1, 3, 5, 4)
                    element_type = 6
                else:
                    order = (
                        (0, 1, 2, 3, 4, 5, 6, 7) if _hex_orientation(points, tuple(range(1, 9))) >= 0.0
                        else (0, 3, 2, 1, 4, 7, 6, 5)
                    )
                    element_type = 5
                keyed_cells.append((element_type, tuple((lattice_cell[i], block) for i in order)))

    # Number nodes by first use of their lattice point, then block. Without joints this is the
    # original first-touch numbering; with joints the two copies of a node are adjacent, so
    # tetrahedral.split_face (lowest id) cuts both sides of a joint quad along the same diagonal.
    node_keys = sorted({key for _, cell in keyed_cells for key in cell}, key=lambda item: (first_touch[item[0]], item[1]))
    node_ids = {key: index for index, key in enumerate(node_keys, start=1)}
    nodes = [lattice[lattice_key] for lattice_key, _ in node_keys]
    node_blocks = [block for _, block in node_keys]
    cells: list[tuple[int, tuple[int, ...]]] = [
        (element_type, tuple(node_ids[key] for key in cell)) for element_type, cell in keyed_cells
    ]
    # Lattice faces shared by cells of two different blocks are the joints between bands.
    lattice_face_blocks: dict[tuple[tuple[int, int, int], ...], set[int]] = {}

    face_patterns = {
        5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
        6: ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
    }
    face_counts: Counter[tuple[int, ...]] = Counter()
    oriented_faces: dict[tuple[int, ...], tuple[int, ...]] = {}
    for (element_type, cell), (_, keyed_cell) in zip(cells, keyed_cells):
        for pattern in face_patterns[element_type]:
            face = tuple(cell[index] for index in pattern)
            key = tuple(sorted(face))
            face_counts[key] += 1
            oriented_faces[key] = face
            lattice_face = tuple(sorted(keyed_cell[index][0] for index in pattern))
            lattice_face_blocks.setdefault(lattice_face, set()).add(keyed_cell[pattern[0]][1])
    joint_lattice_faces = {face for face, blocks in lattice_face_blocks.items() if len(blocks) > 1}
    lattice_of_node = [lattice_key for lattice_key, _ in node_keys]

    boundaries = []
    thickening_inner_m, crest_downstream_m = min(downstream_radii_m), downstream_radii_m[0]
    for key, count in face_counts.items():
        if count != 1:
            continue
        face = oriented_faces[key]
        if tuple(sorted(lattice_of_node[value - 1] for value in face)) in joint_lattice_faces:
            # Upper side of joint k belongs to block k, the lower side to block k + 1.
            block = node_blocks[face[0] - 1]
            z_m = nodes[face[0] - 1][2]
            joint = min(range(len(step_levels_m)), key=lambda index: abs(step_levels_m[index] - z_m))
            upper_id, lower_id = joint_boundary_ids(joint)
            boundaries.append((upper_id if block == joint else lower_id, face))
            continue
        coordinates = [nodes[value - 1] for value in face]
        radii = [math.hypot(x_m, y_m) for x_m, y_m, _ in coordinates]
        stations_m = [centerline_radius_m * math.atan2(x_m, y_m) for x_m, y_m, _ in coordinates]
        z_values_m = [z_m for _, _, z_m in coordinates]
        if max(abs(radius_m - upstream_radius_m) for radius_m in radii) < tolerance:
            boundary_id = UPSTREAM_BOUNDARY_ID
        elif any(
            max(abs(radius_m - downstream_m) for radius_m in radii) < tolerance
            for downstream_m in downstream_radii_m
        ):
            boundary_id = DOWNSTREAM_BOUNDARY_ID
        elif max(abs(z_m - CREST_Z_M) for z_m in z_values_m) < tolerance:
            boundary_id = CREST_BOUNDARY_ID
        elif all(
            abs(z_m - seat_level_at(station_m)) < tolerance
            for station_m, z_m in zip(stations_m, z_values_m)
        ):
            boundary_id = BASE_BOUNDARY_ID
        elif all(
            thickening_inner_m - tolerance <= radius_m <= crest_downstream_m + tolerance
            for radius_m in radii
        ):
            # Step treads and the exposed ends of thickened bands face downstream.
            boundary_id = DOWNSTREAM_BOUNDARY_ID
        elif max(abs(station_m - start_chainage_m) for station_m in stations_m) < tolerance:
            boundary_id = LEFT_END_BOUNDARY_ID
        elif max(abs(station_m - end_chainage_m) for station_m in stations_m) < tolerance:
            boundary_id = RIGHT_END_BOUNDARY_ID
        else:
            boundary_id = OTHER_BOUNDARY_ID
        boundaries.append((boundary_id, face))

    return UniformWallMesh(
        nodes, cells, boundaries, chainages_m, base_levels_m, element_size_m,
        node_blocks, step_levels_m if split_joints else (),
    )


def build_bedrock_seated_wall(
    root: Path, extra_radial_fractions: tuple[float, ...] = (), split_joints: bool = False,
) -> UniformWallMesh:
    contours = load_model_contours(root)
    return build_uniform_wall(
        root,
        contours[0].chainage_m,
        contours[-1].chainage_m,
        [point.chainage_m for point in contours],
        extra_radial_fractions,
        split_joints,
    )
