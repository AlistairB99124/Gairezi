"""Merge all approved structural regions into one attributed VTU dataset."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct

from .plinth import build_conforming_plinth, build_plinth, load_contours
from .uniform_wall import (
    DOWNSTREAM_BATTER_BASE_Z_M,
    DOWNSTREAM_WALL_RADIUS_M,
    UPSTREAM_RADIUS_M,
    build_uniform_wall,
)
from .model_groups import FACE_GROUP_IDS, REGION_GROUP_IDS, build_named_groups


def build_wall(root: Path, extra_radial_fractions: tuple[float, ...] = ()):
    contours = load_contours(root / "Data" / "plinth.json")
    return build_uniform_wall(
        root,
        contours[0].chainage_m,
        contours[-1].chainage_m,
        [point.chainage_m for point in contours],
        extra_radial_fractions,
    )


REGION_BUILDERS = (
    (REGION_GROUP_IDS["PLINTH"], "plinth", build_plinth),
    (REGION_GROUP_IDS["WALL"], "wall", build_wall),
)

STRUCTURAL_REGION_BUILDERS = REGION_BUILDERS[1:]


@dataclass(frozen=True)
class MaterialProperties:
    density_kg_m3: float
    youngs_modulus_pa: float
    poissons_ratio: float
    tensile_strength_pa: float


@dataclass(frozen=True)
class EnvironmentalLoads:
    gravity_z_m_s2: float
    maximum_water_height_m: float
    peak_water_pressure_pa: float
    water_density_kg_m3: float
    tailwater_head_m: float
    overflow_head_m: float


@dataclass(frozen=True)
class FoundationSupport:
    spring_x_n_per_m3: float
    spring_y_n_per_m3: float
    spring_z_n_per_m3: float


@dataclass
class CombinedStructure:
    nodes: list[tuple[float, float, float]]
    cells: list[tuple[int, tuple[int, ...]]]
    boundaries: list[tuple[int, tuple[int, ...]]]
    region_ids: list[int]
    material: MaterialProperties
    loads: EnvironmentalLoads
    foundation_support: FoundationSupport
    plinth_support: FoundationSupport
    region_cell_counts: dict[str, int]


def _named_values(path: Path, name_key: str, value_key: str) -> dict[str, float]:
    return {str(row[name_key]): float(row[value_key]) for row in json.loads(path.read_text())}


def load_material(path: Path) -> MaterialProperties:
    values = _named_values(path, "Parameter Name", "Client Input")
    return MaterialProperties(
        density_kg_m3=values["Concrete Density"],
        youngs_modulus_pa=values["Young's Modulus (E)"],
        poissons_ratio=values["Poisson's Ratio (v)"],
        tensile_strength_pa=values["Concrete Tensile Strength"],
    )


def load_environment(path: Path) -> EnvironmentalLoads:
    values = _named_values(path, "Boundary", "Value")
    return EnvironmentalLoads(
        gravity_z_m_s2=values["Gravity"],
        maximum_water_height_m=values["MaximumWaterHeight"],
        peak_water_pressure_pa=values["Peak Water Pressure"],
        water_density_kg_m3=values["WaterDensity"],
        tailwater_head_m=values["TailwaterHead"],
        overflow_head_m=values["OverflowHead"],
    )


def load_foundation_support(path: Path) -> FoundationSupport:
    values = _named_values(path, "Boundary", "Value")
    return FoundationSupport(
        spring_x_n_per_m3=values["FoundationSpringX"],
        spring_y_n_per_m3=values["FoundationSpringY"],
        spring_z_n_per_m3=values["FoundationSpringZ"],
    )


def build_combined_structure(root: Path, extra_radial_fractions: tuple[float, ...] = ()) -> CombinedStructure:
    material = load_material(root / "Data" / "Concrete_Material_Properties.json")
    load_path = root / "Data" / "Env_Boundaries_And_Loads.json"
    loads = load_environment(load_path)
    foundation_support = load_foundation_support(load_path)
    plinth_support = foundation_support
    nodes: list[tuple[float, float, float]] = []
    coordinate_nodes: dict[tuple[float, float, float], int] = {}
    cells: list[tuple[int, tuple[int, ...]]] = []
    region_ids: list[int] = []
    boundary_faces: dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]] = {}
    region_cell_counts: dict[str, int] = {}

    wall = build_wall(root, extra_radial_fractions)
    plinth = build_conforming_plinth(root, [wall])
    wall_base_nodes = {
        node_id
        for boundary_id, face in wall.boundaries
        if boundary_id == FACE_GROUP_IDS["FOUNDATION"]
        for node_id in face
    }
    wall_base_coordinate_faces = {
        tuple(sorted(tuple(round(value, 9) for value in wall.nodes[node_id - 1]) for node_id in face))
        for boundary_id, face in wall.boundaries
        if boundary_id == FACE_GROUP_IDS["FOUNDATION"]
    }
    regions = [
        (REGION_GROUP_IDS["PLINTH"], "plinth", plinth),
        (REGION_GROUP_IDS["WALL"], "wall", wall),
    ]

    for region_id, region_name, mesh in regions:
        local_to_global: dict[int, int] = {}
        for local_id, coordinate in enumerate(mesh.nodes, start=1):
            key = tuple(round(value, 9) for value in coordinate)
            separate_spring_interface_node = region_name == "wall" and local_id in wall_base_nodes
            if key not in coordinate_nodes or separate_spring_interface_node:
                node_id = len(nodes)
                nodes.append(coordinate)
                if not separate_spring_interface_node:
                    coordinate_nodes[key] = node_id
            else:
                node_id = coordinate_nodes[key]
            local_to_global[local_id] = node_id

        for element_type, cell in mesh.cells:
            cells.append((element_type, tuple(local_to_global[node_id] for node_id in cell)))
            region_ids.append(region_id)
        region_cell_counts[region_name] = region_cell_counts.get(region_name, 0) + len(mesh.cells)

        for boundary_id, face in mesh.boundaries:
            oriented_face = tuple(local_to_global[node_id] for node_id in face)
            global_face = set(oriented_face)
            face_key = tuple(sorted(global_face))
            boundary_faces.setdefault(face_key, []).append((region_id, boundary_id, oriented_face))

    boundaries = []
    for entries in boundary_faces.values():
        if len(entries) != 1:
            continue
        region_id, boundary_id, oriented_face = entries[0]
        coordinate_face = tuple(sorted(tuple(round(value, 9) for value in nodes[node_id]) for node_id in oriented_face))
        if coordinate_face in wall_base_coordinate_faces:
            boundary_id = (
                FACE_GROUP_IDS["WALL_PLINTH_WALL"]
                if region_id == REGION_GROUP_IDS["WALL"]
                else FACE_GROUP_IDS["WALL_PLINTH_PLINTH"]
            )
        boundaries.append((boundary_id, oriented_face))
    if any(len(entries) > 2 for entries in boundary_faces.values()):
        raise ValueError("More than two regions share a boundary face")
    return CombinedStructure(
        nodes=nodes,
        cells=cells,
        boundaries=boundaries,
        region_ids=region_ids,
        material=material,
        loads=loads,
        foundation_support=foundation_support,
        plinth_support=plinth_support,
        region_cell_counts=region_cell_counts,
    )


def write_combined_gmsh(mesh: CombinedStructure, path: Path) -> None:
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
            node_ids = " ".join(str(node_id + 1) for node_id in cell)
            region_id = mesh.region_ids[element_id - 1]
            stream.write(f"{element_id} {element_type} 2 {region_id} {region_id} {node_ids}\n")
            element_id += 1
        for boundary_id, face in mesh.boundaries:
            element_type = 2 if len(face) == 3 else 3
            node_ids = " ".join(str(node_id + 1) for node_id in face)
            stream.write(f"{element_id} {element_type} 2 {boundary_id} {boundary_id} {node_ids}\n")
            element_id += 1
        stream.write("$EndElements\n")


def _packed_chunk(format_code: str, values: list[float] | list[int]) -> bytes:
    item_size = struct.calcsize(format_code)
    return struct.pack(f"<I{len(values)}{format_code}", item_size * len(values), *values)


def write_combined_vtu(mesh: CombinedStructure, path: Path) -> None:
    connectivity = [node_id for _, cell in mesh.cells for node_id in cell]
    offsets = []
    offset = 0
    for _, cell in mesh.cells:
        offset += len(cell)
        offsets.append(offset)
    vtk_cell_types = {4: 10, 5: 12, 6: 13, 7: 14}
    cell_types = [vtk_cell_types[element_type] for element_type, _ in mesh.cells]
    points = [coordinate for point in mesh.nodes for coordinate in point]
    upstream_nodes = {
        node_id
        for boundary_id, face in mesh.boundaries
        if boundary_id == FACE_GROUP_IDS["UPSTREAM"]
        for node_id in face
    }
    hydrostatic_pressure = [0.0] * len(mesh.nodes)
    pressure_traction_vectors = [(0.0, 0.0, 0.0)] * len(mesh.nodes)
    for node_id in upstream_nodes:
        x_m, y_m, z_m = mesh.nodes[node_id]
        depth_fraction = min(max(-z_m / mesh.loads.maximum_water_height_m, 0.0), 1.0)
        pressure_pa = mesh.loads.peak_water_pressure_pa * depth_fraction
        radius_m = math.hypot(x_m, y_m)
        hydrostatic_pressure[node_id] = pressure_pa
        pressure_traction_vectors[node_id] = (
            -pressure_pa * x_m / radius_m,
            -pressure_pa * y_m / radius_m,
            0.0,
        )
    pressure_traction = [component for vector in pressure_traction_vectors for component in vector]
    gravity = [component for _ in mesh.cells for component in (0.0, 0.0, mesh.loads.gravity_z_m_s2)]
    foundation_nodes = [0] * len(mesh.nodes)
    for boundary_id, face in mesh.boundaries:
        if boundary_id == FACE_GROUP_IDS["FOUNDATION"]:
            for node_id in face:
                foundation_nodes[node_id] = 1
    cell_count = len(mesh.cells)
    material_ids = [1] * cell_count
    densities = [mesh.material.density_kg_m3] * cell_count
    youngs_moduli = [mesh.material.youngs_modulus_pa] * cell_count
    poissons_ratios = [mesh.material.poissons_ratio] * cell_count

    arrays = [
        ("points", "Float64", 3, _packed_chunk("d", points)),
        ("connectivity", "Int32", 1, _packed_chunk("i", connectivity)),
        ("offsets", "Int32", 1, _packed_chunk("i", offsets)),
        ("types", "UInt8", 1, _packed_chunk("B", cell_types)),
        ("FoundationNode", "UInt8", 1, _packed_chunk("B", foundation_nodes)),
        ("HydrostaticPressure", "Float64", 1, _packed_chunk("d", hydrostatic_pressure)),
        ("WaterPressureTraction", "Float64", 3, _packed_chunk("d", pressure_traction)),
        ("RegionId", "Int32", 1, _packed_chunk("i", mesh.region_ids)),
        ("MaterialId", "Int32", 1, _packed_chunk("i", material_ids)),
        ("ConcreteDensity", "Float64", 1, _packed_chunk("d", densities)),
        ("YoungsModulus", "Float64", 1, _packed_chunk("d", youngs_moduli)),
        ("PoissonsRatio", "Float64", 1, _packed_chunk("d", poissons_ratios)),
        ("ConcreteTensileStrength", "Float64", 1, _packed_chunk("d", [mesh.material.tensile_strength_pa] * cell_count)),
        ("GravityAcceleration", "Float64", 3, _packed_chunk("d", gravity)),
    ]
    byte_offsets: dict[str, int] = {}
    byte_offset = 0
    for name, _, _, chunk in arrays:
        byte_offsets[name] = byte_offset
        byte_offset += len(chunk)

    def data_array(name: str, indent: str) -> str:
        _, value_type, components, _ = next(array for array in arrays if array[0] == name)
        component_attribute = f' NumberOfComponents="{components}"' if components > 1 else ""
        name_attribute = "" if name == "points" else f' Name="{name}"'
        return f'{indent}<DataArray type="{value_type}"{name_attribute}{component_attribute} format="appended" offset="{byte_offsets[name]}"/>'

    header = "\n".join((
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32">',
        "  <UnstructuredGrid>",
        f'    <FieldData>',
        f'      <DataArray type="Float64" Name="MaximumWaterHeight" NumberOfTuples="1" format="ascii">{mesh.loads.maximum_water_height_m:.12g}</DataArray>',
        f'      <DataArray type="Float64" Name="PeakWaterPressure" NumberOfTuples="1" format="ascii">{mesh.loads.peak_water_pressure_pa:.12g}</DataArray>',
        "    </FieldData>",
        f'    <Piece NumberOfPoints="{len(mesh.nodes)}" NumberOfCells="{cell_count}">',
        '      <PointData Scalars="HydrostaticPressure" Vectors="WaterPressureTraction">',
        data_array("FoundationNode", "        "),
        data_array("HydrostaticPressure", "        "),
        data_array("WaterPressureTraction", "        "),
        "      </PointData>",
        '      <CellData Scalars="RegionId" Vectors="GravityAcceleration">',
        data_array("RegionId", "        "),
        data_array("MaterialId", "        "),
        data_array("ConcreteDensity", "        "),
        data_array("YoungsModulus", "        "),
        data_array("PoissonsRatio", "        "),
        data_array("ConcreteTensileStrength", "        "),
        data_array("GravityAcceleration", "        "),
        "      </CellData>",
        "      <Points>",
        data_array("points", "        "),
        "      </Points>",
        "      <Cells>",
        data_array("connectivity", "        "),
        data_array("offsets", "        "),
        data_array("types", "        "),
        "      </Cells>",
        "    </Piece>",
        "  </UnstructuredGrid>",
        '  <AppendedData encoding="raw">',
    )).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + b"\n_" + b"".join(chunk for _, _, _, chunk in arrays) + b"\n  </AppendedData>\n</VTKFile>\n")


def audit_combined_structure(mesh: CombinedStructure) -> dict[str, object]:
    element_counts = Counter(element_type for element_type, _ in mesh.cells)
    if any(
        element_type != 5
        for (element_type, _), region_id in zip(mesh.cells, mesh.region_ids)
        if region_id == REGION_GROUP_IDS["WALL"]
    ):
        raise ValueError("Wall contains non-hexahedral cells")
    expected_cells = sum(mesh.region_cell_counts.values())
    if len(mesh.cells) != expected_cells or len(mesh.region_ids) != expected_cells:
        raise ValueError("Combined cell arrays do not preserve all regional cells")
    coincident_interface_nodes = {
        node_id
        for boundary_id, face in mesh.boundaries
        if boundary_id in (
            FACE_GROUP_IDS["WALL_PLINTH_WALL"],
            FACE_GROUP_IDS["WALL_PLINTH_PLINTH"],
        )
        for node_id in face
    }
    coordinate_node_ids: dict[tuple[float, float, float], list[int]] = {}
    for node_id, point in enumerate(mesh.nodes):
        coordinate_node_ids.setdefault(tuple(round(value, 9) for value in point), []).append(node_id)
    duplicate_nodes = {
        node_id
        for node_ids in coordinate_node_ids.values()
        if len(node_ids) > 1
        for node_id in node_ids
    }
    if duplicate_nodes - coincident_interface_nodes:
        raise ValueError("Combined structure contains duplicate coordinates outside the wall/plinth interface")
    face_patterns = {
        4: ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
        5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
        6: ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
        7: ((0, 1, 2, 3), (0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0)),
    }
    parents = list(range(len(mesh.cells)))

    def find(cell_index: int) -> int:
        while parents[cell_index] != cell_index:
            parents[cell_index] = parents[parents[cell_index]]
            cell_index = parents[cell_index]
        return cell_index

    face_owners: dict[tuple[int, ...], int] = {}
    region_interfaces: Counter[tuple[int, int]] = Counter()
    plinth_boundary_nodes: set[int] = set()
    for boundary_id, face in mesh.boundaries:
        if boundary_id == FACE_GROUP_IDS["FOUNDATION"]:
            plinth_boundary_nodes.update(face)
    for cell_index, (element_type, cell) in enumerate(mesh.cells):
        if (
            mesh.region_ids[cell_index] == REGION_GROUP_IDS["PLINTH"]
            and element_type != 5
            and not set(cell) & plinth_boundary_nodes
        ):
            raise ValueError("Plinth has a non-hexahedral cell outside the bedrock closure")
        for pattern in face_patterns[element_type]:
            face = tuple(sorted(cell[index] for index in pattern))
            if face in face_owners:
                owner_index = face_owners[face]
                first_region = mesh.region_ids[owner_index]
                second_region = mesh.region_ids[cell_index]
                if first_region != second_region:
                    region_interfaces[tuple(sorted((first_region, second_region)))] += 1
                first_root = find(cell_index)
                second_root = find(owner_index)
                if first_root != second_root:
                    parents[second_root] = first_root
            else:
                face_owners[face] = cell_index
    component_count = len({find(cell_index) for cell_index in range(len(mesh.cells))})
    if component_count not in (1, 2):
        raise ValueError(f"Combined structure has {component_count} face-connected components")
    foundation_node_count = len({
        node_id
        for boundary_id, face in mesh.boundaries
        if boundary_id == FACE_GROUP_IDS["FOUNDATION"]
        for node_id in face
    })
    upstream_node_ids = {
        node_id
        for boundary_id, face in mesh.boundaries
        if boundary_id == FACE_GROUP_IDS["UPSTREAM"]
        for node_id in face
    }
    wall_interface_face_count = sum(
        1 for boundary_id, _ in mesh.boundaries
        if boundary_id == FACE_GROUP_IDS["WALL_PLINTH_WALL"]
    )
    plinth_interface_face_count = sum(
        1 for boundary_id, _ in mesh.boundaries
        if boundary_id == FACE_GROUP_IDS["WALL_PLINTH_PLINTH"]
    )
    if wall_interface_face_count == 0 or wall_interface_face_count != plinth_interface_face_count:
        raise ValueError("Wall/plinth interface faces are missing or unpaired")
    return {
        "nodes": len(mesh.nodes),
        "cells": len(mesh.cells),
        "tetrahedra": element_counts[4],
        "hexahedra": element_counts[5],
        "triangular_prisms": element_counts[6],
        "pyramids": element_counts[7],
        "face_connected_components": component_count,
        "wall_plinth_interface_faces": wall_interface_face_count,
        "region_interface_faces": {
            f"{first_region}-{second_region}": count
            for (first_region, second_region), count in sorted(region_interfaces.items())
        },
        "boundary_faces": dict(sorted(Counter(boundary_id for boundary_id, _ in mesh.boundaries).items())),
        "regions": mesh.region_cell_counts,
        "wall_geometry": {
            "downstream_radius_m": DOWNSTREAM_WALL_RADIUS_M,
            "upstream_radius_m": UPSTREAM_RADIUS_M,
            "uniform_thickness_m": UPSTREAM_RADIUS_M - DOWNSTREAM_WALL_RADIUS_M,
            "wall_base_elevation_m": DOWNSTREAM_BATTER_BASE_Z_M,
        },
        "foundation_nodes": foundation_node_count,
        "pressurized_upstream_nodes": sum(
            -mesh.nodes[node_id][2] > 0.0 for node_id in upstream_node_ids
        ),
        "maximum_applied_pressure_pa": mesh.loads.peak_water_pressure_pa,
        "material": {
            "density_kg_m3": mesh.material.density_kg_m3,
            "youngs_modulus_pa": mesh.material.youngs_modulus_pa,
            "poissons_ratio": mesh.material.poissons_ratio,
            "tensile_strength_pa": mesh.material.tensile_strength_pa,
        },
        "loads": {
            "gravity_z_m_s2": mesh.loads.gravity_z_m_s2,
            "maximum_water_height_m": mesh.loads.maximum_water_height_m,
            "peak_water_pressure_pa": mesh.loads.peak_water_pressure_pa,
        },
    }


__all__ = [
    "audit_combined_structure",
    "build_named_groups",
    "build_combined_structure",
    "write_combined_gmsh",
    "write_combined_vtu",
]
