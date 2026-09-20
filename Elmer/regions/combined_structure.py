"""Merge all approved structural regions into one attributed VTU dataset."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct

from regions.center_wall import build_center_wall
from regions.left_wall import build_left_wall
from regions.left_wedged_wall import build_left_wedged_wall
from regions.left_wedged_wall_transition import build_left_wedged_wall_transition
from regions.plinth import build_plinth
from regions.right_wall import build_right_wall
from regions.right_wedged_wall import build_right_wedged_wall
from regions.right_wedged_wall_transition import build_right_wedged_wall_transition


REGION_BUILDERS = (
    (1, "plinth", build_plinth),
    (2, "left_wall", build_left_wall),
    (3, "left_wedged_wall_transition", build_left_wedged_wall_transition),
    (4, "left_wedged_wall", build_left_wedged_wall),
    (5, "center_wall", build_center_wall),
    (6, "right_wedged_wall", build_right_wedged_wall),
    (7, "right_wedged_wall_transition", build_right_wedged_wall_transition),
    (8, "right_wall", build_right_wall),
)


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


@dataclass
class CombinedStructure:
    nodes: list[tuple[float, float, float]]
    cells: list[tuple[int, tuple[int, ...]]]
    boundaries: list[tuple[int, tuple[int, ...]]]
    region_ids: list[int]
    fixed_base: list[int]
    hydrostatic_pressure_pa: list[float]
    water_pressure_traction_pa: list[tuple[float, float, float]]
    material: MaterialProperties
    loads: EnvironmentalLoads
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
    )


def build_combined_structure(root: Path) -> CombinedStructure:
    material = load_material(root / "Data" / "Concrete_Material_Properties.json")
    loads = load_environment(root / "Data" / "Env_Boundaries_And_Loads.json")
    nodes: list[tuple[float, float, float]] = []
    coordinate_nodes: dict[tuple[float, float, float], int] = {}
    cells: list[tuple[int, tuple[int, ...]]] = []
    region_ids: list[int] = []
    fixed_node_ids: set[int] = set()
    upstream_node_ids: set[int] = set()
    boundary_faces: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = {}
    region_cell_counts: dict[str, int] = {}

    for region_id, region_name, builder in REGION_BUILDERS:
        mesh = builder(root)
        local_to_global: dict[int, int] = {}
        for local_id, coordinate in enumerate(mesh.nodes, start=1):
            key = tuple(round(value, 9) for value in coordinate)
            if key not in coordinate_nodes:
                coordinate_nodes[key] = len(nodes)
                nodes.append(coordinate)
            local_to_global[local_id] = coordinate_nodes[key]

        for element_type, cell in mesh.cells:
            cells.append((element_type, tuple(local_to_global[node_id] for node_id in cell)))
            region_ids.append(region_id)
        region_cell_counts[region_name] = len(mesh.cells)

        for boundary_id, face in mesh.boundaries:
            oriented_face = tuple(local_to_global[node_id] for node_id in face)
            global_face = set(oriented_face)
            boundary_faces.setdefault(tuple(sorted(global_face)), []).append((boundary_id, oriented_face))
            if region_name == "plinth" and boundary_id == 1:
                fixed_node_ids.update(global_face)
            if boundary_id == 2:
                upstream_node_ids.update(global_face)

    boundaries = [entries[0] for entries in boundary_faces.values() if len(entries) == 1]
    if any(len(entries) > 2 for entries in boundary_faces.values()):
        raise ValueError("More than two regions share a boundary face")

    fixed_base = [int(node_id in fixed_node_ids) for node_id in range(len(nodes))]
    hydrostatic_pressure_pa = [0.0] * len(nodes)
    water_pressure_traction_pa = [(0.0, 0.0, 0.0)] * len(nodes)
    for node_id in upstream_node_ids:
        x_m, y_m, z_m = nodes[node_id]
        depth_fraction = min(max(-z_m / loads.maximum_water_height_m, 0.0), 1.0)
        pressure_pa = loads.peak_water_pressure_pa * depth_fraction
        radius_m = math.hypot(x_m, y_m)
        hydrostatic_pressure_pa[node_id] = pressure_pa
        water_pressure_traction_pa[node_id] = (
            -pressure_pa * x_m / radius_m,
            -pressure_pa * y_m / radius_m,
            0.0,
        )

    return CombinedStructure(
        nodes=nodes,
        cells=cells,
        boundaries=boundaries,
        region_ids=region_ids,
        fixed_base=fixed_base,
        hydrostatic_pressure_pa=hydrostatic_pressure_pa,
        water_pressure_traction_pa=water_pressure_traction_pa,
        material=material,
        loads=loads,
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
            stream.write(f"{element_id} {element_type} 2 1 1 {node_ids}\n")
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
    vtk_cell_types = {4: 10, 5: 12, 6: 13}
    cell_types = [vtk_cell_types[element_type] for element_type, _ in mesh.cells]
    points = [coordinate for point in mesh.nodes for coordinate in point]
    pressure_traction = [component for vector in mesh.water_pressure_traction_pa for component in vector]
    gravity = [component for _ in mesh.cells for component in (0.0, 0.0, mesh.loads.gravity_z_m_s2)]
    cell_count = len(mesh.cells)

    arrays = [
        ("points", "Float64", 3, _packed_chunk("d", points)),
        ("connectivity", "Int32", 1, _packed_chunk("i", connectivity)),
        ("offsets", "Int32", 1, _packed_chunk("i", offsets)),
        ("types", "UInt8", 1, _packed_chunk("B", cell_types)),
        ("FixedBase", "UInt8", 1, _packed_chunk("B", mesh.fixed_base)),
        ("HydrostaticPressure", "Float64", 1, _packed_chunk("d", mesh.hydrostatic_pressure_pa)),
        ("WaterPressureTraction", "Float64", 3, _packed_chunk("d", pressure_traction)),
        ("RegionId", "Int32", 1, _packed_chunk("i", mesh.region_ids)),
        ("MaterialId", "Int32", 1, _packed_chunk("i", [1] * cell_count)),
        ("ConcreteDensity", "Float64", 1, _packed_chunk("d", [mesh.material.density_kg_m3] * cell_count)),
        ("YoungsModulus", "Float64", 1, _packed_chunk("d", [mesh.material.youngs_modulus_pa] * cell_count)),
        ("PoissonsRatio", "Float64", 1, _packed_chunk("d", [mesh.material.poissons_ratio] * cell_count)),
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
        data_array("FixedBase", "        "),
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
    expected_cells = sum(mesh.region_cell_counts.values())
    if len(mesh.cells) != expected_cells or len(mesh.region_ids) != expected_cells:
        raise ValueError("Combined cell arrays do not preserve all regional cells")
    if len({tuple(round(value, 9) for value in point) for point in mesh.nodes}) != len(mesh.nodes):
        raise ValueError("Combined structure contains duplicate coordinates")
    return {
        "nodes": len(mesh.nodes),
        "cells": len(mesh.cells),
        "hexahedra": element_counts[5],
        "triangular_prisms": element_counts[6],
        "boundary_faces": dict(sorted(Counter(boundary_id for boundary_id, _ in mesh.boundaries).items())),
        "regions": mesh.region_cell_counts,
        "fixed_base_nodes": sum(mesh.fixed_base),
        "pressurized_upstream_nodes": sum(value > 0.0 for value in mesh.hydrostatic_pressure_pa),
        "maximum_applied_pressure_pa": max(mesh.hydrostatic_pressure_pa),
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
    "build_combined_structure",
    "write_combined_gmsh",
    "write_combined_vtu",
]
