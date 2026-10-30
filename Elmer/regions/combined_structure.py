"""Merge all approved structural regions into one attributed VTU dataset."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct

from .bedrock_contours import load_contours
from .uniform_wall import build_bedrock_seated_wall
from .model_groups import FACE_GROUP_IDS, REGION_GROUP_IDS, build_named_groups, is_wall_joint_boundary_id
from ..mesh_sizing import (
    course_compatible_size,
    size_for_element_target,
    tetrahedral_element_count,
    wall_tetrahedral_target,
)
from ..model_config import load_config, wall_joint_support, wall_thickness_steps


def build_wall(
    root: Path,
    extra_radial_fractions: tuple[float, ...] = (),
    split_joints: bool = False,
    element_size_m: float | None = None,
):
    return build_bedrock_seated_wall(root, extra_radial_fractions, split_joints, element_size_m)


REGION_BUILDERS = (
    (REGION_GROUP_IDS["WALL"], "wall", build_wall),
)

STRUCTURAL_REGION_BUILDERS = REGION_BUILDERS


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
    # Reservoir surface in model coordinates (crest at z=0). None keeps the legacy crest-plus-overflow datum.
    upstream_water_surface_z_m: float | None = None

    @property
    def water_pressure_gradient_pa_per_m(self) -> float:
        return self.water_density_kg_m3 * abs(self.gravity_z_m_s2)

    @property
    def upstream_surface_z_m(self) -> float:
        if self.upstream_water_surface_z_m is None:
            return self.overflow_head_m
        return self.upstream_water_surface_z_m

    @property
    def crest_surcharge_pa(self) -> float:
        return self.water_pressure_gradient_pa_per_m * max(self.upstream_surface_z_m, 0.0)

    def upstream_pressure_pa(self, elevation_m: float) -> float:
        return self.water_pressure_gradient_pa_per_m * max(self.upstream_surface_z_m - elevation_m, 0.0)


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
    region_cell_counts: dict[str, int]
    # Step elevations split into spring joints between wall blocks; empty for a monolithic wall.
    wall_joint_levels_m: tuple[float, ...] = ()
    suggested_global_element_size_m: float | None = None
    element_size_target_m: float | None = None
    target_wall_tetrahedral_elements: int | None = None
    wall_tetrahedral_element_count: int | None = None


def load_material(path: Path) -> MaterialProperties:
    values = load_config(path)["material"]["concrete"]
    return MaterialProperties(
        density_kg_m3=float(values["density"]),
        youngs_modulus_pa=float(values["youngs_modulus"]),
        poissons_ratio=float(values["poisson_ratio"]),
        tensile_strength_pa=float(values["tensile_strength_pa"]),
    )


def load_environment(path: Path) -> EnvironmentalLoads:
    values = load_config(path)["loads"]
    # water_height is measured from the deepest wall seat (lowest bedrock contour, crest at z = 0).
    bed_elevation_m = min(point.bedrock_z_m for point in load_contours(path))
    loads = EnvironmentalLoads(
        gravity_z_m_s2=float(values["gravity"]),
        maximum_water_height_m=float(values["water_height"]),
        peak_water_pressure_pa=float(values["water_density"]) * abs(float(values["gravity"])) * (
            float(values["water_height"]) + float(values["overflow_head"])
        ),
        water_density_kg_m3=float(values["water_density"]),
        tailwater_head_m=float(values["tailwater_head"]),
        overflow_head_m=float(values["overflow_head"]),
        upstream_water_surface_z_m=bed_elevation_m + float(values["water_height"]) + float(values["overflow_head"]),
    )
    expected_peak = loads.upstream_pressure_pa(bed_elevation_m)
    if not math.isclose(loads.peak_water_pressure_pa, expected_peak, rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError(
            f"Peak Water Pressure {loads.peak_water_pressure_pa} Pa must equal "
            f"rho*abs(g)*(MaximumWaterHeight+OverflowHead) = {expected_peak} Pa"
        )
    return loads


def load_foundation_support(path: Path, region: str = "bedrock") -> FoundationSupport:
    values = load_config(path)["geometry"][region]["support"]["spring_n_per_m3"]
    return FoundationSupport(
        spring_x_n_per_m3=float(values["x"]),
        spring_y_n_per_m3=float(values["y"]),
        spring_z_n_per_m3=float(values["z"]),
    )


def build_combined_structure(
    root: Path, extra_radial_fractions: tuple[float, ...] = (), *, split_wall_joints: bool | None = None,
) -> CombinedStructure:
    """Attributed wall mesh.

    split_wall_joints=True separates the thickness bands into blocks (duplicated joint
    nodes, joint tags) when geometry.wall.joint_support is "spring"; the default keeps
    one monolithic wall for consumers that do not model the joint springs.
    """
    load_path = root / "config.json"
    config = load_config(load_path)
    geometry = config["geometry"]
    wall = geometry["wall"]
    split_joints = bool(split_wall_joints) and wall_joint_support(wall) is not None
    material = load_material(load_path)
    loads = load_environment(load_path)
    foundation_support = load_foundation_support(load_path)
    nodes: list[tuple[float, float, float]] = []
    coordinate_nodes: dict[tuple[float, float, float], int] = {}
    cells: list[tuple[int, tuple[int, ...]]] = []
    region_ids: list[int] = []
    boundary_faces: dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]] = {}
    region_cell_counts: dict[str, int] = {}

    suggested_size_m = float(geometry["global_element_size_m"])
    target = wall_tetrahedral_target(geometry)
    step_levels_m = tuple(top_z_m for top_z_m, _ in wall_thickness_steps(wall)[1:])
    constrained_steps = step_levels_m if "foundation" in geometry else ()
    element_size_m = course_compatible_size(
        suggested_size_m, constrained_steps, not_coarser_than=True,
    )
    wall_mesh = build_wall(root, extra_radial_fractions, split_joints, element_size_m)
    tetra_count = tetrahedral_element_count(wall_mesh.cells)
    for _ in range(12):
        if tetra_count <= target:
            break
        requested_size_m = size_for_element_target(
            wall_mesh.element_size_m, tetra_count, target,
        )
        next_size_m = course_compatible_size(
            requested_size_m, constrained_steps, not_coarser_than=False,
        )
        if next_size_m <= wall_mesh.element_size_m * (1.0 + 1e-12):
            raise ValueError(
                f"Cannot reduce the wall mesh to {target:,} "
                "tetrahedra without changing the configured wall geometry"
            )
        wall_mesh = build_wall(root, extra_radial_fractions, split_joints, next_size_m)
        tetra_count = tetrahedral_element_count(wall_mesh.cells)
    else:
        raise RuntimeError(
            f"Wall mesh sizing did not reach the {target:,}-tetrahedron target "
            f"after 12 builds; the last mesh has {tetra_count:,} tetrahedra"
        )
    regions = [(REGION_GROUP_IDS["WALL"], "wall", wall_mesh)]

    for region_id, region_name, mesh in regions:
        local_to_global: dict[int, int] = {}
        blocks = mesh.node_blocks or [0] * len(mesh.nodes)
        for local_id, coordinate in enumerate(mesh.nodes, start=1):
            # Joint nodes have one copy per block; never merge copies of different blocks.
            key = (*(round(value, 9) for value in coordinate), blocks[local_id - 1])
            if key not in coordinate_nodes:
                node_id = len(nodes)
                nodes.append(coordinate)
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
        boundaries.append((boundary_id, oriented_face))
    if any(len(entries) > 2 for entries in boundary_faces.values()):
        raise ValueError("More than two regions share a boundary face")
    combined = CombinedStructure(
        nodes=nodes,
        cells=cells,
        boundaries=boundaries,
        region_ids=region_ids,
        material=material,
        loads=loads,
        foundation_support=foundation_support,
        region_cell_counts=region_cell_counts,
        wall_joint_levels_m=tuple(wall_mesh.joint_levels_m),
        suggested_global_element_size_m=suggested_size_m,
        element_size_target_m=wall_mesh.element_size_m,
        target_wall_tetrahedral_elements=target,
        wall_tetrahedral_element_count=tetra_count,
    )
    return combined


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
        pressure_pa = mesh.loads.upstream_pressure_pa(z_m)
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
        element_type not in (5, 6)
        for (element_type, _), region_id in zip(mesh.cells, mesh.region_ids)
        if region_id == REGION_GROUP_IDS["WALL"]
    ):
        raise ValueError("Wall contains unsupported cells")
    expected_cells = sum(mesh.region_cell_counts.values())
    if len(mesh.cells) != expected_cells or len(mesh.region_ids) != expected_cells:
        raise ValueError("Combined cell arrays do not preserve all regional cells")
    coordinate_node_ids: dict[tuple[float, float, float], list[int]] = {}
    for node_id, point in enumerate(mesh.nodes):
        coordinate_node_ids.setdefault(tuple(round(value, 9) for value in point), []).append(node_id)
    duplicate_nodes = {
        node_id
        for node_ids in coordinate_node_ids.values()
        if len(node_ids) > 1
        for node_id in node_ids
    }
    joint_nodes = {
        node_id
        for boundary_id, face in mesh.boundaries
        if is_wall_joint_boundary_id(boundary_id)
        for node_id in face
    }
    if duplicate_nodes - joint_nodes:
        raise ValueError("Bonded structure contains duplicate coordinates; regenerate a conforming mesh")
    joint_count = len(mesh.wall_joint_levels_m)
    if joint_nodes and not joint_count:
        raise ValueError("Wall joint faces present without configured joint levels")
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
    for cell_index, (element_type, cell) in enumerate(mesh.cells):
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
    if component_count != joint_count + 1:
        raise ValueError(
            f"Combined structure has {component_count} face-connected components; expected {joint_count + 1}"
        )
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
    if not foundation_node_count:
        raise ValueError("Wall has no foundation seat faces on the bedrock")
    wall_node_ids = {
        node_id
        for (_, cell), region_id in zip(mesh.cells, mesh.region_ids)
        if region_id == REGION_GROUP_IDS["WALL"]
        for node_id in cell
    }
    wall_radii = [math.hypot(*mesh.nodes[node_id][:2]) for node_id in wall_node_ids]
    wall_min_radius, wall_max_radius = min(wall_radii), max(wall_radii)
    return {
        "nodes": len(mesh.nodes),
        "cells": len(mesh.cells),
        "tetrahedra": element_counts[4],
        "hexahedra": element_counts[5],
        "triangular_prisms": element_counts[6],
        "pyramids": element_counts[7],
        "face_connected_components": component_count,
        "wall_connection": "spring_jointed_blocks" if joint_count else "single_monolithic_body",
        "connection_assumption": (
            f"Wall is {joint_count + 1} concrete blocks split at thickness steps z="
            + ", ".join(f"{level:g}" for level in mesh.wall_joint_levels_m)
            + " m, joined by linear springs from geometry.wall.joint_support; seated directly on the bedrock"
            if joint_count else
            "Wall is one concrete body seated directly on the bedrock; no internal joints"
        ),
        "wall_joint_levels_m": list(mesh.wall_joint_levels_m),
        "active_support": "geometry.bedrock.support on foundation tag 1 only",
        "inactive_inputs": ["geometry.wall.height_m: nominal metadata; the seat follows geometry.bedrock.contours"],
        "fracture_hold_reason": "Cracking is on hold pending review of the corrected bonded intact reference.",
        "region_interface_faces": {
            f"{first_region}-{second_region}": count
            for (first_region, second_region), count in sorted(region_interfaces.items())
        },
        "boundary_faces": dict(sorted(Counter(boundary_id for boundary_id, _ in mesh.boundaries).items())),
        "regions": mesh.region_cell_counts,
        "wall_geometry": {
            "downstream_radius_m": wall_min_radius,
            "upstream_radius_m": wall_max_radius,
            "max_thickness_m": wall_max_radius - wall_min_radius,
            "wall_base_elevation_m": min(mesh.nodes[node_id][2] for node_id in wall_node_ids),
        },
        "foundation_nodes": foundation_node_count,
        "pressurized_upstream_nodes": sum(
            mesh.loads.upstream_pressure_pa(mesh.nodes[node_id][2]) > 0.0 for node_id in upstream_node_ids
        ),
        "maximum_applied_pressure_pa": max(
            mesh.loads.upstream_pressure_pa(mesh.nodes[node_id][2]) for node_id in upstream_node_ids
        ),
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
