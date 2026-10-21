"""Export three attributed bodies for geometry review, without running a solve."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import meshio
import numpy as np
from matplotlib.path import Path as PlotPath
from matplotlib.tri import LinearTriInterpolator, Triangulation

from .model_config import load_config
from .regions.combined_structure import build_combined_structure
from .regions.tetrahedral import split_face


BODY_IDS = {"wall": 1, "bedrock": 2}
# Agreed rock domain extension (m): upstream/downstream of the concrete and
# beyond each bank end of the surveyed cross-section.
STREAM_MARGIN_M = 20.
BANK_MARGIN_M = 5.
NEAR_FIELD_DISTANCE_M = 5.
MESH_GROWTH_FACTOR = 1.2
FAR_FIELD_ELEMENT_SIZE_M = 5.
CELL_TYPES = {4: "tetra", 5: "hexahedron", 6: "wedge", 7: "pyramid"}
FIELD_TYPES = {
    "RegionId": "Int32",
    "MaterialId": "Int32",
    "Density": "Float64",
    "YoungsModulus": "Float64",
    "PoissonsRatio": "Float64",
    "TensileStrength": "Float64",
}


def bedrock_geometry(
    contours: list[dict[str, float]],
    concrete_points: np.ndarray,
    survey_radius_m: float = 80.0,
    bottom_z_m: float = -31.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Extrude the approved chord section through the concrete's depth bounds."""
    stations = np.asarray([row["chainage"] for row in contours], dtype=float)
    elevations = np.asarray([row["height"] for row in contours], dtype=float)
    if (
        len(stations) < 2
        or not np.isfinite(stations).all()
        or not np.isfinite(elevations).all()
        or not np.all(np.diff(stations) > 0)
        or not math.isfinite(survey_radius_m)
        or survey_radius_m <= 0
        or not math.isfinite(bottom_z_m)
        or not np.all(elevations > bottom_z_m)
    ):
        raise ValueError("Bedrock requires increasing finite stations and tops above its bottom")
    if (
        concrete_points.ndim != 2
        or concrete_points.shape[1] != 3
        or not len(concrete_points)
        or not np.isfinite(concrete_points).all()
    ):
        raise ValueError("Concrete coordinates must be a nonempty finite N by 3 array")
    angles = stations / survey_radius_m
    if angles[-1] - angles[0] >= math.pi:
        raise ValueError("The survey arc must span less than pi radians for monotonic projection")
    surveyed_xy = survey_radius_m * np.column_stack((np.sin(angles), np.cos(angles)))
    origin = surveyed_xy[0]
    chord = surveyed_xy[-1] - origin
    span = float(np.linalg.norm(chord))
    across = chord / span
    depth = np.array([-across[1], across[0]])
    projected = (surveyed_xy - origin) @ across
    concrete_across = (concrete_points[:, :2] - origin) @ across
    concrete_depth = (concrete_points[:, :2] - origin) @ depth
    depth_min = float(concrete_depth.min())
    depth_max = float(concrete_depth.max())
    if depth_max - depth_min <= 1e-9:
        raise ValueError("Concrete has no upstream/downstream extent")
    points = []
    for x, top in zip(projected, elevations):
        for d, z in (
            (depth_min, bottom_z_m), (depth_min, top),
            (depth_max, bottom_z_m), (depth_max, top),
        ):
            xy = origin + x * across + d * depth
            points.append((float(xy[0]), float(xy[1]), float(z)))
    # One coarse volume per survey interval: visualization, not an analysis mesh.
    cells = np.asarray([
        (4*i, 4*i+4, 4*i+6, 4*i+2, 4*i+1, 4*i+5, 4*i+7, 4*i+3)
        for i in range(len(stations) - 1)
    ], dtype=np.int64)
    concrete_section_top = np.interp(concrete_across, projected, elevations)
    inside = (concrete_across >= 0) & (concrete_across <= span)
    report = {
        "survey_radius_m": survey_radius_m,
        "bottom_z_m": bottom_z_m,
        "projected_span_m": span,
        "projected_stations_m": projected.tolist(),
        "extrusion_depth_limits_m": [depth_min, depth_max],
        "extrusion_depth_m": depth_max - depth_min,
        "section_origin_xy_m": origin.tolist(),
        "across_unit_vector_xy": across.tolist(),
        "depth_unit_vector_xy": depth.tolist(),
        "concrete_projected_limits_m": [
            float(concrete_across.min()), float(concrete_across.max()),
        ],
        "unseated_extrusion_vertices_outside_section_span": int(np.count_nonzero(~inside)),
        "unseated_extrusion_vertices_below_rock_top": int(np.count_nonzero(
            inside & (concrete_points[:, 2] < concrete_section_top - 1e-8)
        )),
    }
    return np.asarray(points), cells, report


def survey_coordinates(points: np.ndarray, old_radius: float, survey_radius: float) -> np.ndarray:
    """Preserve physical radii and elevations while correcting survey station angles."""
    result = points.copy()
    radii = np.linalg.norm(points[:, :2], axis=1)
    angles = np.arctan2(points[:, 0], points[:, 1]) * old_radius / survey_radius
    result[:, 0] = radii * np.sin(angles)
    result[:, 1] = radii * np.cos(angles)
    return result


def constrain_edges(
    points: np.ndarray, triangles: np.ndarray, required: set[tuple[int, int]],
) -> np.ndarray:
    """Insert footprint constraints by flipping crossed diagonals of convex pairs."""
    triangles = triangles.copy()

    def cross(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def crosses(a: int, b: int, c: int, d: int) -> bool:
        if len({a, b, c, d}) < 4:
            return False
        pa, pb, pc, pd = points[[a, b, c, d]]
        return (
            cross(pb-pa, pc-pa) * cross(pb-pa, pd-pa) < -1e-18
            and cross(pd-pc, pa-pc) * cross(pd-pc, pb-pc) < -1e-18
        )

    initial_edges = {
        tuple(sorted((int(tri[i]), int(tri[(i+1) % 3]))))
        for tri in triangles for i in range(3)
    }
    for target in sorted(required - initial_edges):
        for _ in range(len(triangles)):
            adjacency: dict[tuple[int, int], list[int]] = {}
            for index, tri in enumerate(triangles):
                for i in range(3):
                    edge = tuple(sorted((int(tri[i]), int(tri[(i+1) % 3]))))
                    adjacency.setdefault(edge, []).append(index)
            if target in adjacency:
                break
            changed = False
            for (a, b), owners in adjacency.items():
                if len(owners) != 2 or (a, b) in required or not crosses(*target, a, b):
                    continue
                first, second = owners
                c = next(int(n) for n in triangles[first] if n not in (a, b))
                d = next(int(n) for n in triangles[second] if n not in (a, b))
                if not crosses(a, b, c, d):
                    continue
                triangles[first] = (c, d, a)
                triangles[second] = (d, c, b)
                changed = True
                break
            if not changed:
                raise ValueError(f"Cannot insert rock-seat perimeter constraint {target}")
        else:
            raise ValueError(f"Rock-seat edge insertion did not terminate: {target}")
    return triangles


def seated_bedrock_geometry(
    contours: list[dict[str, float]],
    concrete: np.ndarray,
    foundation_faces: list[tuple[int, ...]],
    stream_margin_m: float = 0.,
    bank_margin_m: float = 0.,
    extension_spacing_m: float = 5.,
    *,
    element_size_m: float = 0.5,
    bottom_z_m: float = -31.,
    grade_mesh: bool = False,
    resample_surface: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Use the actual wall underside as the rock seat, extending the canyon outside it.

    stream_margin_m extends the rock upstream and downstream of the concrete
    footprint; the canyon cross-section is unchanged along that direction.
    bank_margin_m extends it beyond each bank end of the survey/concrete span at
    the outermost surveyed bank elevation, because no survey exists beyond it.
    Sample parallel exterior rows without changing the existing wall seat or
    the piecewise-linear topography. Returned columns retain bottom/top ordering.
    """
    if (
        not all(math.isfinite(value) for value in (
            stream_margin_m, bank_margin_m, extension_spacing_m, element_size_m,
        ))
        or stream_margin_m < 0 or bank_margin_m < 0
        or extension_spacing_m <= 0 or element_size_m <= 0
    ):
        raise ValueError("Rock extension margins must be non-negative with positive spacing")
    _, _, report = bedrock_geometry(contours, concrete, bottom_z_m=bottom_z_m)
    origin = np.array(report["section_origin_xy_m"])
    across = np.array(report["across_unit_vector_xy"])
    depth = np.array(report["depth_unit_vector_xy"])
    local = np.column_stack(((concrete[:, :2] - origin) @ across,
                             (concrete[:, :2] - origin) @ depth))
    face_triangles = [tri for face in foundation_faces for tri in split_face(face)]
    if not face_triangles:
        raise ValueError("No wall foundation faces found")
    edge_counts = Counter(
        tuple(sorted((tri[i], tri[(i + 1) % 3])))
        for tri in face_triangles for i in range(3)
    )
    if max(edge_counts.values()) > 2:
        raise ValueError("Non-manifold wall underside")
    perimeter_edges = [edge for edge, count in edge_counts.items() if count == 1]
    neighbours: dict[int, list[int]] = {}
    for a, b in perimeter_edges:
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    if any(len(values) != 2 for values in neighbours.values()):
        raise ValueError("Wall footprint must have a single closed perimeter")
    start = min(neighbours)
    loop = [start]
    previous = -1
    current = start
    while True:
        following = next(node for node in neighbours[current] if node != previous)
        if following == start:
            break
        if following in loop:
            raise ValueError("Invalid wall perimeter cycle")
        loop.append(following)
        previous, current = current, following
    if len(loop) != len(neighbours):
        raise ValueError("Multiple wall perimeter loops are not supported")
    outline = PlotPath(local[loop])
    x_min = min(0., float(local[:, 0].min()))
    x_max = max(float(report["projected_span_m"]), float(local[:, 0].max()))
    d_min, d_max = float(local[:, 1].min()), float(local[:, 1].max())
    seat_d_min, seat_d_max = d_min, d_max
    x_min -= bank_margin_m
    x_max += bank_margin_m
    d_min -= stream_margin_m
    d_max += stream_margin_m
    projected = np.array(report["projected_stations_m"])
    heights = np.array([row["height"] for row in contours])
    xy: list[tuple[float, float]] = []
    tops: list[float] = []
    index_by_xy: dict[tuple[float, float], int] = {}

    def vertex(x: float, d: float, z: float) -> int:
        key = (round(float(x), 8), round(float(d), 8))
        if key in index_by_xy:
            index = index_by_xy[key]
            if abs(tops[index] - z) > 1e-7:
                raise ValueError("Rock top is not single-valued at a shared footprint vertex")
            return index
        index = len(xy)
        index_by_xy[key] = index
        xy.append((float(x), float(d)))
        tops.append(float(z))
        return index

    foundation_map = {}
    for node in sorted({n for tri in face_triangles for n in tri}):
        foundation_map[node] = vertex(*local[node], concrete[node, 2])
    # Outer rectangle retains the approved projected profile, with flat bank
    # extensions beyond the survey endpoints. Intermediate rows keep the
    # upstream/downstream extension triangles reasonably shaped.
    rows = {d_min, d_max}
    for start, end in ((d_min, seat_d_min), (seat_d_max, d_max)):
        if end > start:
            divisions = max(1, math.ceil((end - start) / extension_spacing_m))
            rows.update(np.linspace(start, end, divisions + 1).tolist())
    for x in sorted({x_min, x_max, *projected.tolist()}):
        if x_min <= x <= x_max:
            for d in sorted(rows):
                key = (round(x, 8), round(d, 8))
                if key not in index_by_xy:
                    vertex(x, d, float(np.interp(x, projected, heights)))
    points_2d = np.array(xy)
    triangulation = Triangulation(points_2d[:, 0], points_2d[:, 1])
    required_edges = {
        tuple(sorted((foundation_map[a], foundation_map[b]))) for a, b in perimeter_edges
    }
    all_triangles = constrain_edges(points_2d, triangulation.triangles, required_edges)
    all_edges = {
        tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
        for tri in all_triangles for i in range(3)
    }
    missing = required_edges - all_edges
    if missing:
        raise ValueError(
            f"Triangulation does not preserve {len(missing)} seat perimeter edges: "
            f"{[(edge, points_2d[list(edge)].tolist()) for edge in sorted(missing)[:3]]}"
        )
    centroids = points_2d[all_triangles].mean(axis=1)
    exterior = all_triangles[~outline.contains_points(centroids)]
    seat = np.array([[foundation_map[n] for n in tri] for tri in face_triangles])
    surface = np.column_stack((points_2d, tops))
    if resample_surface:
        surface, exterior = parallel_rock_surface(
            surface, exterior, required_edges,
            outline, element_size_m, grade_mesh=grade_mesh,
        )
    points_2d = surface[:, :2]
    triangles = np.vstack((seat, exterior))
    first_edges = points_2d[triangles[:, 1]] - points_2d[triangles[:, 0]]
    second_edges = points_2d[triangles[:, 2]] - points_2d[triangles[:, 0]]
    signed_areas = (
        first_edges[:, 0] * second_edges[:, 1] - first_edges[:, 1] * second_edges[:, 0]
    ) / 2
    if np.any(np.abs(signed_areas) < 1e-10):
        raise ValueError("Degenerate rock footprint triangle")
    area = float(np.abs(signed_areas).sum())
    rectangle_area = (x_max - x_min) * (d_max - d_min)
    if not math.isclose(area, rectangle_area, rel_tol=1e-8):
        raise ValueError(f"Rock footprint has holes or overlap: {area} versus {rectangle_area}")
    negative = signed_areas < 0
    triangles[negative] = triangles[negative][:, [0, 2, 1]]
    top_array = surface[:, 2]
    if np.any(top_array <= bottom_z_m):
        raise ValueError("Wall underside reaches or crosses the rock bottom")
    plan = origin + points_2d[:, :1] * across + points_2d[:, 1:] * depth
    count = len(plan)
    nodes = np.vstack((
        np.column_stack((plan, np.full(count, bottom_z_m))),
        np.column_stack((plan, top_array)),
    ))
    wedges = np.column_stack((triangles, triangles + count))
    volume = float(np.sum(
        np.abs(signed_areas) * (top_array[triangles].mean(axis=1) - bottom_z_m)
    ))
    report.update({
        "rock_footprint_area_m2": area,
        "rock_volume_m3": volume,
        "seat_triangles": len(seat),
        "seat_perimeter_edges_verified": len(required_edges),
        "seat_max_coordinate_difference_m": float(max(
            np.linalg.norm(nodes[foundation_map[n] + count] - concrete[n])
            for n in foundation_map
        )),
        "rock_across_limits_m": [x_min, x_max],
        "rock_depth_limits_m": [d_min, d_max],
        "stream_margin_m": stream_margin_m,
        "bank_margin_m": bank_margin_m,
        "element_size_m": element_size_m,
        "mesh_grading": (
            {
                "near_field_distance_m": NEAR_FIELD_DISTANCE_M,
                "growth_factor": MESH_GROWTH_FACTOR,
                "far_field_element_size_m": max(element_size_m, FAR_FIELD_ELEMENT_SIZE_M),
                "fine_layer_bottom_z_m": float(top_array.min()) - NEAR_FIELD_DISTANCE_M,
            } if grade_mesh else None
        ),
        "surface_triangles": len(triangles),
        "maximum_exterior_surface_edge_m": float(max(
            np.linalg.norm(surface[exterior[:, (i + 1) % 3]] - surface[exterior[:, i]], axis=1).max()
            for i in range(3)
        )) if len(exterior) else 0.,
        "surface_edge_size_policy": (
            "Nominally spaced parallel local-X rows outside the wall seat; retain exact "
            "seat nodes and use unstructured triangles only for the interface transition."
        ),
        "surface_row_axis": "local cross-valley X; across_unit_vector_xy in global coordinates",
        "surface_definition": (
            "Exact existing wall underside at the seat; approved projected canyon "
            "profile at the outer rectangle; triangulated transition between them. "
            "Flat bank extensions at the end survey heights beyond the survey chord "
            "endpoints; canyon section held constant upstream and downstream."
        ),
    })
    return nodes, wedges, report


def parallel_rock_surface(
    points: np.ndarray, triangles: np.ndarray,
    seat_edges: set[tuple[int, int]], outline: PlotPath, element_size_m: float,
    *, grade_mesh: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample the existing terrain on parallel rows, keeping the seat unchanged."""
    if not math.isfinite(element_size_m) or element_size_m <= 0:
        raise ValueError("Bedrock element size must be finite and positive")
    limits = np.array([points[:, :2].min(axis=0), points[:, :2].max(axis=0)])
    seat_nodes = {node for edge in seat_edges for node in edge}
    outer_nodes = sorted(set(triangles.ravel()) - seat_nodes)
    seat_bounds = np.array([
        points[sorted(seat_nodes), :2].min(axis=0),
        points[sorted(seat_nodes), :2].max(axis=0),
    ])
    axes = [
        np.unique(np.concatenate((
            graded_axis(
                low, high,
                seat_bounds[0, axis] - NEAR_FIELD_DISTANCE_M,
                seat_bounds[1, axis] + NEAR_FIELD_DISTANCE_M,
                element_size_m,
            ) if grade_mesh else np.linspace(
                low, high, max(1, math.ceil((high - low) / element_size_m)) + 1,
            ),
            points[outer_nodes, axis],
        )))
        for axis, (low, high) in enumerate(limits.T)
    ]
    x, y = np.meshgrid(*axes)
    grid = np.column_stack((x.ravel(), y.ravel()))
    keep = ~outline.contains_points(grid)
    # Leave room for an exact, conforming transition to the curved seat.
    for a, b in seat_edges:
        start, end = points[[a, b], :2]
        direction = end - start
        fractions = np.clip((grid - start) @ direction / (direction @ direction), 0., 1.)
        distances = np.linalg.norm(grid - start - fractions[:, None] * direction, axis=1)
        keep &= distances > element_size_m / 4
    grid = grid[keep]
    source = Triangulation(points[:, 0], points[:, 1], triangles)
    elevations = LinearTriInterpolator(source, points[:, 2])(grid[:, 0], grid[:, 1])
    if np.ma.getmaskarray(elevations).any():
        raise ValueError("Parallel bedrock rows extend outside the existing terrain surface")
    existing = {tuple(np.round(point, 8)) for point in points[:, :2]}
    unique = np.array([tuple(np.round(point, 8)) not in existing for point in grid])
    result = np.vstack((points, np.column_stack((grid[unique], elevations.data[unique]))))
    triangulation = Triangulation(result[:, 0], result[:, 1])
    cells = constrain_edges(result[:, :2], triangulation.triangles, seat_edges)
    centroids = result[cells, :2].mean(axis=1)
    return result, cells[~outline.contains_points(centroids)]


def graded_axis(
    low: float, high: float, fine_low: float, fine_high: float, spacing: float,
) -> np.ndarray:
    """Retain fine spacing near the dam, growing by 20% to at most 5 m outside."""
    if not all(math.isfinite(value) for value in (low, high, fine_low, fine_high, spacing)):
        raise ValueError("Graded mesh coordinates and spacing must be finite")
    if low >= high or fine_low > fine_high or spacing <= 0:
        raise ValueError("Graded mesh needs ordered limits and positive spacing")
    start, end = np.clip([fine_low, fine_high], low, high)
    values = np.linspace(start, end, max(1, math.ceil((end - start) / spacing)) + 1).tolist()
    maximum = max(spacing, FAR_FIELD_ELEMENT_SIZE_M)
    for origin, limit, direction in ((start, low, -1), (end, high, 1)):
        position = origin
        step = spacing
        while direction * (limit - position) > 1e-10:
            step = min(step * MESH_GROWTH_FACTOR, maximum)
            position += direction * min(step, abs(limit - position))
            values.append(position)
    return np.unique(values)


def refine_rock_surface(
    points: np.ndarray, triangles: np.ndarray,
    seat_edges: set[tuple[int, int]], element_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Conformingly bisect long surface edges, retaining the concrete seat."""
    if not math.isfinite(element_size_m) or element_size_m <= 0:
        raise ValueError("Bedrock element size must be finite and positive")
    limit = math.sqrt(2) * element_size_m
    while len(triangles):
        edges, inverse = np.unique(
            np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1),
            axis=0, return_inverse=True,
        )
        lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
        protected = np.fromiter(
            (tuple(edge) in seat_edges for edge in edges), dtype=bool, count=len(edges),
        )
        limits = np.full(len(edges), limit)
        # A matching seat can have longer diagonals on steep terrain. Allow
        # its immediate transition triangles the same size, without hanging nodes.
        seat_lengths = np.where(protected, lengths, 0.)[inverse].reshape(-1, 3).max(axis=1)
        np.maximum.at(limits, inverse, np.repeat(seat_lengths, 3))
        split = (lengths > limits * (1 + 1e-10)) & ~protected
        if not np.any(split):
            break
        midpoint_ids = np.full(len(edges), -1, dtype=np.int64)
        midpoint_ids[split] = np.arange(len(points), len(points) + np.count_nonzero(split))
        points = np.vstack((points, points[edges[split]].mean(axis=1)))
        midpoints = midpoint_ids[inverse].reshape(-1, 3)
        marked = midpoints >= 0
        counts = marked.sum(axis=1)
        blocks = [triangles[counts == 0]]
        for rotation in range(3):
            a, b, c = np.roll(triangles, -rotation, axis=1).T
            ab, bc, ca = np.roll(midpoints, -rotation, axis=1).T
            one = (counts == 1) & (ab >= 0)
            blocks.extend((
                np.column_stack((a[one], ab[one], c[one])),
                np.column_stack((ab[one], b[one], c[one])),
            ))
            two = (counts == 2) & (ab >= 0) & (bc >= 0)
            blocks.extend((
                np.column_stack((a[two], ab[two], c[two])),
                np.column_stack((ab[two], bc[two], c[two])),
                np.column_stack((ab[two], b[two], bc[two])),
            ))
        all_three = counts == 3
        a, b, c = triangles[all_three].T
        ab, bc, ca = midpoints[all_three].T
        blocks.extend((
            np.column_stack((a, ab, ca)), np.column_stack((ab, b, bc)),
            np.column_stack((ca, bc, c)), np.column_stack((ab, bc, ca)),
        ))
        triangles = np.vstack(blocks)
    return points, triangles


def layered_bedrock_geometry(
    columns: np.ndarray, wedges: np.ndarray, element_size_m: float,
    *, fine_layer_bottom_z_m: float | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], int]:
    """Extrude matching surface triangles, optionally grading below the near field."""
    if not math.isfinite(element_size_m) or element_size_m <= 0:
        raise ValueError("Bedrock element size must be finite and positive")
    count = len(columns) // 2
    bottom, top = columns[:count], columns[count:]
    bottom_z, top_z = float(bottom[:, 2].min()), float(top[:, 2].max())
    if fine_layer_bottom_z_m is None:
        layers = max(1, math.ceil((top_z - bottom_z) / element_size_m))
        levels = np.minimum(bottom_z + np.arange(1, layers + 1) * element_size_m, top_z)
    else:
        levels = graded_axis(
            bottom_z, top_z, fine_layer_bottom_z_m, top_z, element_size_m,
        )[1:]
        layers = len(levels)
    triangles = wedges[:, :3]
    # Share horizontal levels, clipping each column at its own terrain height.
    # Collapsed wedges close with pyramids/tetrahedra, as in the wall builder.
    chunks = [top, bottom]
    node_count = 2 * count
    lower = np.arange(count, 2 * count, dtype=np.int64)
    blocks: dict[int, list[np.ndarray]] = {4: [], 6: [], 7: []}
    for level in levels:
        active = top[:, 2] > level + 1e-10
        upper = np.arange(count, dtype=np.int64)
        upper[active] = np.arange(node_count, node_count + np.count_nonzero(active))
        coordinates = top[active].copy()
        coordinates[:, 2] = level
        chunks.append(coordinates)
        node_count += len(coordinates)
        low, high = lower[triangles], upper[triangles]
        changed = low != high
        counts = changed.sum(axis=1)
        blocks[6].append(np.column_stack((low[counts == 3], high[counts == 3])))
        for rotation in range(3):
            a, b, c = np.roll(low, -rotation, axis=1).T
            above_a, above_b, above_c = np.roll(high, -rotation, axis=1).T
            one = (counts == 1) & (a != above_a)
            blocks[4].append(np.column_stack((a[one], b[one], c[one], above_a[one])))
            two = (counts == 2) & (c == above_c)
            blocks[7].append(np.column_stack((
                a[two], above_a[two], above_b[two], b[two], c[two],
            )))
        lower = upper
    return np.vstack(chunks), {
        kind: np.vstack(parts) for kind, parts in blocks.items()
        if any(len(part) for part in parts)
    }, layers


def configured_bedrock_geometry(
    config: dict, concrete: np.ndarray, foundation_faces: list[tuple[int, ...]],
    stream_margin_m: float | None = None, bank_margin_m: float | None = None,
    bottom_z_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Apply one authoritative domain definition to preview and diagnostic."""
    domain = config["geometry"]["bedrock"].get("domain", {})
    return seated_bedrock_geometry(
        config["geometry"]["bedrock"]["contours"], concrete, foundation_faces,
        stream_margin_m if stream_margin_m is not None else float(domain.get("stream_margin_m", STREAM_MARGIN_M)),
        bank_margin_m if bank_margin_m is not None else float(domain.get("bank_margin_m", BANK_MARGIN_M)),
        element_size_m=float(config["geometry"]["global_element_size_m"]),
        bottom_z_m=bottom_z_m if bottom_z_m is not None else float(domain.get("bottom_z_m", -31.)),
        grade_mesh=bool(domain),
    )


def bedrock_layer_options(report: dict) -> dict[str, float]:
    grading = report.get("mesh_grading")
    return {"fine_layer_bottom_z_m": grading["fine_layer_bottom_z_m"]} if grading else {}


def configured_rock_mesh(config, concrete, foundation_faces, *overrides):
    """Shared preview/solver volume mesh; retain old configs' layered defaults."""
    from .graded_bedrock import RockMesh, build_graded_rock
    if "mesh" in config["geometry"]["bedrock"]:
        return build_graded_rock(config, concrete, foundation_faces, *overrides)
    points, wedges, report = configured_bedrock_geometry(
        config, concrete, foundation_faces, *overrides,
    )
    count = len(points) // 2
    points, blocks, layers = layered_bedrock_geometry(
        points, wedges, float(config["geometry"]["global_element_size_m"]),
        **bedrock_layer_options(report),
    )
    report["vertical_layers"] = layers
    return RockMesh(points, blocks, wedges[:, :3], wedges[:, :3] + count, report)


def attributed_mesh(
    points: np.ndarray,
    cells: list[tuple[str, np.ndarray]],
    body_id: int,
    material_id: int,
    material: dict[str, float],
) -> meshio.Mesh:
    # The procedural Gmsh prisms use the opposite triangular winding from
    # meshio's VTK wedge convention.
    cells = [
        (kind, block[:, [0, 2, 1, 3, 5, 4]] if kind == "wedge" else block)
        for kind, block in cells
    ]
    values = {
        "RegionId": body_id,
        "MaterialId": material_id,
        "Density": material["density"],
        "YoungsModulus": material["youngs_modulus"],
        "PoissonsRatio": material["poisson_ratio"],
        "TensileStrength": material["tensile_strength_pa"],
    }
    return meshio.Mesh(points, cells, cell_data={
        name: [
            np.full(len(block), value, dtype=np.int32 if name.endswith("Id") else np.float64)
            for _, block in cells
        ]
        for name, value in values.items()
    })


def export_preview(
    root: Path, output: Path,
    stream_margin_m: float | None = None, bank_margin_m: float | None = None,
    bottom_z_m: float | None = None,
) -> Path:
    config = load_config(root / "config.json")
    structure = build_combined_structure(root)
    concrete_points = survey_coordinates(
        np.asarray(structure.nodes),
        float(config["geometry"]["wall"]["centerline_radius_m"]), 80.,
    )
    rock = configured_rock_mesh(
        config, concrete_points,
        [face for tag, face in structure.boundaries if tag == 1],
        stream_margin_m, bank_margin_m, bottom_z_m,
    )
    rock_points, rock_blocks, report = rock.points, rock.blocks, rock.report
    output.mkdir(parents=True, exist_ok=True)
    manifest = ET.Element("VTKFile", type="PUnstructuredGrid", version="0.1",
                          byte_order="LittleEndian")
    grid = ET.SubElement(manifest, "PUnstructuredGrid", GhostLevel="0")
    ET.SubElement(grid, "PPointData")
    data = ET.SubElement(grid, "PCellData", Scalars="RegionId")
    for name, dtype in FIELD_TYPES.items():
        ET.SubElement(data, "PDataArray", type=dtype, Name=name, NumberOfComponents="1")
    coordinates = ET.SubElement(grid, "PPoints")
    ET.SubElement(coordinates, "PDataArray", type="Float64", NumberOfComponents="3")
    counts = {}
    for name, body_id in BODY_IDS.items():
        if name == "bedrock":
            mesh = attributed_mesh(
                rock_points, [(CELL_TYPES[kind], cells) for kind, cells in rock_blocks.items()],
                body_id, 2, config["material"]["bedrock"],
            )
        else:
            selected = [
                (kind, cell) for (kind, cell), region in zip(
                    structure.cells, structure.region_ids, strict=True,
                ) if region == body_id
            ]
            used = sorted({node for _, cell in selected for node in cell})
            mapping = {node: index for index, node in enumerate(used)}
            blocks: dict[str, list[tuple[int, ...]]] = {}
            for kind, cell in selected:
                blocks.setdefault(CELL_TYPES[kind], []).append(tuple(mapping[n] for n in cell))
            mesh = attributed_mesh(
                concrete_points[used],
                [(kind, np.asarray(block)) for kind, block in blocks.items()],
                body_id, 1, config["material"]["concrete"],
            )
        piece = output / f"{name}.vtu"
        meshio.write(piece, mesh)
        checked = meshio.read(piece)
        expected_count = sum(len(block.data) for block in mesh.cells)
        if sum(len(block.data) for block in checked.cells) != expected_count:
            raise RuntimeError(f"VTU readback cell-count failure: {piece}")
        for field in FIELD_TYPES:
            if not np.array_equal(
                np.concatenate(checked.cell_data[field]),
                np.concatenate(mesh.cell_data[field]),
            ):
                raise RuntimeError(f"VTU readback field mismatch: {piece}, {field}")
        if not np.array_equal(checked.points, mesh.points):
            raise RuntimeError(f"VTU readback coordinate mismatch: {piece}")
        counts[name] = expected_count
        ET.SubElement(grid, "Piece", Source=piece.name)
    path = output / "three_body_geometry.pvtu"
    ET.indent(manifest)
    ET.ElementTree(manifest).write(path, encoding="utf-8", xml_declaration=True)
    report.update({
        "purpose": "Wall and bedrock geometry preview only; NOT an Elmer solution",
        "body_ids": BODY_IDS,
        "cell_counts": counts,
        "concrete_chainage_reference_radius_m": 80.,
        "legacy_concrete_chainage_reference_radius_m":
            config["geometry"]["wall"]["centerline_radius_m"],
        "bedrock_mesh": report.get("mesh_method") or (
            "Surface-refined, vertically layered wedges with pyramid/tetrahedron terrain "
            "closures; configured near-field spacing with far-field grading when "
            "mesh_grading is set; unsolved"
        ),
        "matching_wall_rock_surface": True,
        "solver_interface_law_assigned": False,
        "materials_source": str((root / "config.json").resolve()),
    })
    (output / "geometry_review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream-margin-m", type=float, help="Override the configured stream margin (m)")
    parser.add_argument("--bank-margin-m", type=float, help="Override the configured bank margin (m)")
    parser.add_argument("--bottom-z-m", type=float, help="Override the configured rock bottom elevation (m)")
    args = parser.parse_args()
    path = export_preview(
        args.root, args.output, args.stream_margin_m, args.bank_margin_m, args.bottom_z_m,
    )
    print(path)
    print("GEOMETRY ONLY: inspect geometry_review.json before any solver integration.")


if __name__ == "__main__":
    main()
