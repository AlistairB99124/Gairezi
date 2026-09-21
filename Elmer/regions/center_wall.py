"""Generate the center wall with the continuous downstream shelf wedge."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct


START_CHAINAGE_M = 101.0
END_CHAINAGE_M = 116.0
WEDGE_TOE_RADIUS_M = 74.0
DOWNSTREAM_WALL_RADIUS_M = 76.0
UPSTREAM_RADIUS_M = 80.0
BASE_Z_M = -28.5
WEDGE_HEIGHT_M = 1.5
WEDGE_WIDTH_M = 2.0
WEDGE_SHELF_WIDTH_M = 0.5
WEDGE_TOP_Z_M = BASE_Z_M + WEDGE_HEIGHT_M
CREST_Z_M = 0.0
BODY_ID = 1
BASE_BOUNDARY_ID = 1
UPSTREAM_BOUNDARY_ID = 2
DOWNSTREAM_BOUNDARY_ID = 3
CREST_BOUNDARY_ID = 4
LEFT_END_BOUNDARY_ID = 5
RIGHT_END_BOUNDARY_ID = 6


@dataclass
class CenterWallMesh:
    nodes: list[tuple[float, float, float]]
    cells: list[tuple[int, tuple[int, ...]]]
    boundaries: list[tuple[int, tuple[int, ...]]]
    element_size_m: float


def _global_element_size(root: Path) -> float:
    controls = json.loads((root / "Data" / "Computational_Grid_Controls.json").read_text())
    matches = [float(row["Client Value"]) for row in controls if row["Mesh Control"] == "Global Element Size"]
    if len(matches) != 1 or matches[0] <= 0.0:
        raise ValueError("Expected one positive Global Element Size")
    return matches[0]


def _levels(start: float, end: float, step: float) -> list[float]:
    count = round((end - start) / step)
    if abs(start + count * step - end) > 1.0e-9:
        raise ValueError(f"Range {start:g}..{end:g} is not divisible by element size {step:g}")
    return [start + index * step for index in range(count + 1)]


def _wedge_section_faces(element_size_m: float) -> list[tuple[tuple[int, int], ...]]:
    radial_divisions = round(WEDGE_WIDTH_M / element_size_m)
    vertical_divisions = round(WEDGE_HEIGHT_M / element_size_m)
    slope_width_m = WEDGE_WIDTH_M - WEDGE_SHELF_WIDTH_M
    if (
        not math.isclose(radial_divisions * element_size_m, WEDGE_WIDTH_M, abs_tol=1.0e-9)
        or not math.isclose(vertical_divisions * element_size_m, WEDGE_HEIGHT_M, abs_tol=1.0e-9)
        or not math.isclose(vertical_divisions * element_size_m, slope_width_m, abs_tol=1.0e-9)
    ):
        raise ValueError("Center wedge dimensions must align with the global element grid")

    faces: list[tuple[tuple[int, int], ...]] = []
    for vertical_index in range(vertical_divisions):
        upper_start = vertical_index + 1
        faces.append((
            (vertical_index, vertical_index),
            (upper_start, vertical_index),
            (upper_start, vertical_index + 1),
        ))
        for radial_index in range(upper_start, radial_divisions):
            faces.append((
                (radial_index, vertical_index),
                (radial_index + 1, vertical_index),
                (radial_index + 1, vertical_index + 1),
                (radial_index, vertical_index + 1),
            ))
    return faces


def build_center_wall(root: Path) -> CenterWallMesh:
    element_size_m = _global_element_size(root)
    config = json.loads((root / "config.json").read_text())
    centerline_radius_m = float(config["wall_centerline_radius_m"])
    chainages_m = _levels(START_CHAINAGE_M, END_CHAINAGE_M, element_size_m)
    radii_m = _levels(WEDGE_TOE_RADIUS_M, UPSTREAM_RADIUS_M, element_size_m)
    z_levels_m = _levels(BASE_Z_M, CREST_Z_M, element_size_m)
    wedge_faces = _wedge_section_faces(element_size_m)

    nodes = []
    node_ids = {}

    def node_id(chainage_index: int, z_index: int, radial_index: int) -> int:
        key = (chainage_index, z_index, radial_index)
        if key in node_ids:
            return node_ids[key]
        chainage_m = chainages_m[chainage_index]
        radius_m = radii_m[radial_index]
        z_m = z_levels_m[z_index]
        angle = chainage_m / centerline_radius_m
        node_ids[key] = len(nodes) + 1
        nodes.append((radius_m * math.sin(angle), radius_m * math.cos(angle), z_m))
        return node_ids[key]

    cells: list[tuple[int, tuple[int, ...]]] = []
    wall_start_index = round((DOWNSTREAM_WALL_RADIUS_M - WEDGE_TOE_RADIUS_M) / element_size_m)
    for chainage_index in range(len(chainages_m) - 1):
        for z_index in range(len(z_levels_m) - 1):
            for radial_index in range(wall_start_index, len(radii_m) - 1):
                cell = (
                    node_id(chainage_index, z_index, radial_index),
                    node_id(chainage_index + 1, z_index, radial_index),
                    node_id(chainage_index + 1, z_index, radial_index + 1),
                    node_id(chainage_index, z_index, radial_index + 1),
                    node_id(chainage_index, z_index + 1, radial_index),
                    node_id(chainage_index + 1, z_index + 1, radial_index),
                    node_id(chainage_index + 1, z_index + 1, radial_index + 1),
                    node_id(chainage_index, z_index + 1, radial_index + 1),
                )
                cells.append((5, cell))
        for face in wedge_faces:
            start_face = tuple(node_id(chainage_index, vertical, radial) for radial, vertical in face)
            end_face = tuple(node_id(chainage_index + 1, vertical, radial) for radial, vertical in face)
            if len(face) == 3:
                cells.append((6, start_face + end_face))
            else:
                cells.append((5, (
                    start_face[0], end_face[0], end_face[1], start_face[1],
                    start_face[3], end_face[3], end_face[2], start_face[2],
                )))

    face_patterns = {
        5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
        6: ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
    }
    face_counts: Counter[tuple[int, ...]] = Counter()
    oriented_faces = {}
    for element_type, cell in cells:
        for pattern in face_patterns[element_type]:
            face = tuple(cell[index] for index in pattern)
            key = tuple(sorted(face))
            face_counts[key] += 1
            oriented_faces[key] = face

    boundaries = []
    tolerance = 1.0e-7
    for key, count in face_counts.items():
        if count != 1:
            continue
        face = oriented_faces[key]
        coordinates = [nodes[node_id_value - 1] for node_id_value in face]
        radii = [math.hypot(x_m, y_m) for x_m, y_m, _ in coordinates]
        z_values = [z_m for _, _, z_m in coordinates]
        stations = [centerline_radius_m * math.atan2(x_m, y_m) for x_m, y_m, _ in coordinates]
        if max(abs(z_m - BASE_Z_M) for z_m in z_values) < tolerance:
            boundary_id = BASE_BOUNDARY_ID
        elif max(abs(radius_m - UPSTREAM_RADIUS_M) for radius_m in radii) < tolerance:
            boundary_id = UPSTREAM_BOUNDARY_ID
        elif max(abs(z_m - CREST_Z_M) for z_m in z_values) < tolerance:
            boundary_id = CREST_BOUNDARY_ID
        elif max(abs(station_m - START_CHAINAGE_M) for station_m in stations) < tolerance:
            boundary_id = LEFT_END_BOUNDARY_ID
        elif max(abs(station_m - END_CHAINAGE_M) for station_m in stations) < tolerance:
            boundary_id = RIGHT_END_BOUNDARY_ID
        else:
            boundary_id = DOWNSTREAM_BOUNDARY_ID
        boundaries.append((boundary_id, face))

    return CenterWallMesh(nodes, cells, boundaries, element_size_m)


def write_gmsh(mesh: CenterWallMesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")
        stream.write(f"{len(mesh.nodes)}\n")
        for node_id, (x_m, y_m, z_m) in enumerate(mesh.nodes, start=1):
            stream.write(f"{node_id} {x_m:.12g} {y_m:.12g} {z_m:.12g}\n")
        stream.write("$EndNodes\n$Elements\n")
        stream.write(f"{len(mesh.cells) + len(mesh.boundaries)}\n")
        element_id = 1
        for element_type, cell in mesh.cells:
            stream.write(f"{element_id} {element_type} 2 {BODY_ID} {BODY_ID} {' '.join(map(str, cell))}\n")
            element_id += 1
        for boundary_id, face in mesh.boundaries:
            element_type = 2 if len(face) == 3 else 3
            stream.write(f"{element_id} {element_type} 2 {boundary_id} {boundary_id} {' '.join(map(str, face))}\n")
            element_id += 1
        stream.write("$EndElements\n")


def write_vtu(mesh: CenterWallMesh, path: Path) -> None:
    connectivity = [node_id - 1 for _, cell in mesh.cells for node_id in cell]
    offsets = []
    offset = 0
    for _, cell in mesh.cells:
        offset += len(cell)
        offsets.append(offset)
    vtk_cell_types = {4: 10, 5: 12, 6: 13}
    cell_types = [vtk_cell_types[element_type] for element_type, _ in mesh.cells]
    points = [coordinate for point in mesh.nodes for coordinate in point]
    chunks = [
        struct.pack(f"<I{len(points)}d", 8 * len(points), *points),
        struct.pack(f"<I{len(connectivity)}i", 4 * len(connectivity), *connectivity),
        struct.pack(f"<I{len(offsets)}i", 4 * len(offsets), *offsets),
        struct.pack(f"<I{len(cell_types)}B", len(cell_types), *cell_types),
    ]
    offsets_bytes = []
    byte_offset = 0
    for chunk in chunks:
        offsets_bytes.append(byte_offset)
        byte_offset += len(chunk)
    header = "\n".join((
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32">',
        "  <UnstructuredGrid>",
        f'    <Piece NumberOfPoints="{len(mesh.nodes)}" NumberOfCells="{len(mesh.cells)}">',
        "      <PointData/>",
        "      <CellData/>",
        "      <Points>",
        f'        <DataArray type="Float64" NumberOfComponents="3" format="appended" offset="{offsets_bytes[0]}"/>',
        "      </Points>",
        "      <Cells>",
        f'        <DataArray type="Int32" Name="connectivity" format="appended" offset="{offsets_bytes[1]}"/>',
        f'        <DataArray type="Int32" Name="offsets" format="appended" offset="{offsets_bytes[2]}"/>',
        f'        <DataArray type="UInt8" Name="types" format="appended" offset="{offsets_bytes[3]}"/>',
        "      </Cells>",
        "    </Piece>",
        "  </UnstructuredGrid>",
        '  <AppendedData encoding="raw">',
    )).encode("ascii")
    path.write_bytes(header + b"\n_" + b"".join(chunks) + b"\n  </AppendedData>\n</VTKFile>\n")


def audit_center_wall(mesh: CenterWallMesh) -> dict[str, object]:
    chainage_cells = round((END_CHAINAGE_M - START_CHAINAGE_M) / mesh.element_size_m)
    radial_cells = round((UPSTREAM_RADIUS_M - DOWNSTREAM_WALL_RADIUS_M) / mesh.element_size_m)
    vertical_cells = round((CREST_Z_M - BASE_Z_M) / mesh.element_size_m)
    wedge_cells_per_section = len(_wedge_section_faces(mesh.element_size_m))
    expected_cells = chainage_cells * (radial_cells * vertical_cells + wedge_cells_per_section)
    boundary_counts = Counter(boundary_id for boundary_id, _ in mesh.boundaries)
    if len(mesh.cells) != expected_cells:
        raise ValueError(f"Expected {expected_cells} center-wall cells, found {len(mesh.cells)}")
    if set(boundary_counts) != set(range(1, 7)):
        raise ValueError(f"Missing center-wall boundaries: found {sorted(boundary_counts)}")
    if any(abs(z_m / mesh.element_size_m - round(z_m / mesh.element_size_m)) > 1.0e-9 for _, _, z_m in mesh.nodes):
        raise ValueError("Center-wall Z levels are not on the global element grid")
    element_counts = Counter(element_type for element_type, _ in mesh.cells)
    return {
        "nodes": len(mesh.nodes),
        "cells": len(mesh.cells),
        "hexahedra": element_counts[5],
        "triangular_prisms": element_counts[6],
        "boundary_faces": dict(sorted(boundary_counts.items())),
        "chainage_m": [START_CHAINAGE_M, END_CHAINAGE_M],
        "wall_radius_m": [DOWNSTREAM_WALL_RADIUS_M, UPSTREAM_RADIUS_M],
        "wedge_radius_m": [WEDGE_TOE_RADIUS_M, DOWNSTREAM_WALL_RADIUS_M],
        "wedge_z_m": [BASE_Z_M, WEDGE_TOP_Z_M],
        "wedge_shelf_width_m": WEDGE_SHELF_WIDTH_M,
        "z_m": [BASE_Z_M, CREST_Z_M],
        "element_size_m": mesh.element_size_m,
    }
