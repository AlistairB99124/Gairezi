"""Export a whole-model nodal displacement field without using deformed coordinates."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def export_mesh(arrays: dict, displacement: np.ndarray, output: Path,
                damage: np.ndarray, opening: np.ndarray,
                concrete_cell_stress: dict | None = None) -> dict:
    import meshio

    points = np.asarray(arrays["nodes"], dtype=float)
    native_wall = "hexa8_wall" in arrays
    concrete = np.asarray(
        arrays["hexa8_wall"] if native_wall else arrays["tetra4_concrete"], dtype=np.int64)
    hex_foundation = np.asarray(
        arrays.get("hexa8_foundation", np.empty((0, 8), dtype=np.int64)),
        dtype=np.int64,
    ) if native_wall else np.empty((0, 8), dtype=np.int64)
    tet_foundation = np.asarray(
        arrays["tetra4_foundation"],
        dtype=np.int64,
    ) if native_wall else np.empty((0, 4), dtype=np.int64)
    tet_rock = np.asarray(arrays["tetra4_bedrock"], dtype=np.int64)
    hex_rock = np.asarray(
        arrays.get("hexa8_bedrock", np.empty((0, 8), dtype=np.int64)),
        dtype=np.int64,
    )
    rock_count = len(tet_rock) + len(hex_rock)
    tetra = (np.vstack((tet_foundation, tet_rock)) if native_wall and (len(tet_foundation) or len(tet_rock))
             else np.vstack((concrete, tet_rock)) if not native_wall and (len(concrete) or len(tet_rock))
             else np.empty((0, 4), dtype=int))
    if native_wall:
        regions = np.concatenate((
            np.ones(len(concrete), int),
            np.full(len(tet_foundation) + len(hex_foundation), 3, int),
            np.full(rock_count, 2, int),
        ))
    else:
        regions = np.concatenate((arrays["concrete_body_ids"], np.full(rock_count, 2)))
    if displacement.shape != points.shape or not np.isfinite(displacement).all():
        raise ValueError("Aster ParaView export requires a finite full nodal displacement field")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "aster_final.vtu"
    hexa = np.vstack((concrete, hex_foundation, hex_rock)) if native_wall and (
        len(concrete) or len(hex_foundation) or len(hex_rock)
    ) else concrete if native_wall else np.empty((0, 8), dtype=np.int64)
    blocks = ([("hexahedron", hexa)] if len(hexa) else [])
    block_lengths = [len(hexa)] if blocks else []
    if len(tetra):
        blocks.append(("tetra", tetra))
        block_lengths.append(len(tetra))
    def by_block(values):
        if len(block_lengths) <= 1:
            return [values]
        result = []
        offset = 0
        for length in block_lengths:
            result.append(values[offset:offset + length])
            offset += length
        return result
    cell_data = {"GeometryIds": by_block(regions)}
    if "damaged_cells" in arrays:
        cell_data["selected_material"] = by_block(arrays["damaged_cells"].astype(np.int32))
    if concrete_cell_stress:
        # Final-state constant P1 stress per concrete (wall + foundation) cell; rock
        # cells carry NaN because no rock constitutive stress is recovered here.
        for name, values in concrete_cell_stress.items():
            values = np.asarray(values, dtype=float)
            if values.shape != (len(concrete) + len(foundation),):
                raise ValueError(f"Concrete cell stress array {name} does not match the concrete cells")
            cell_data[name] = by_block(np.concatenate((
                values,
                np.full(rock_count, np.nan),
            )))
    meshio.write(path, meshio.Mesh(points, blocks,
                                  point_data={"displacement": displacement},
                                  cell_data=cell_data))
    joints = np.asarray(
        arrays["quad8_joints"] if "quad8_joints" in arrays else arrays["penta6_joints"],
        dtype=np.int64)
    if damage.shape != (len(joints),) or opening.shape != damage.shape or not (
            np.isfinite(damage).all() and np.isfinite(opening).all()):
        raise ValueError("Aster joint export requires finite per-facet damage and opening")
    joint_path = output / "aster_joints.vtu"
    if len(joints):
        width = joints.shape[1] // 2
        joint_type = "quad" if width == 4 else "triangle"
        meshio.write(joint_path, meshio.Mesh(points, [(joint_type, joints[:, :width])],
                                           cell_data={"joint_damage_state": [damage], "normal_opening_m": [opening]}))
    root = ET.Element("VTKFile", type="vtkMultiBlockDataSet", version="1.0", byte_order="LittleEndian")
    blocks = ET.SubElement(root, "vtkMultiBlockDataSet")
    ET.SubElement(blocks, "DataSet", index="0", name="Whole model displacement", file=path.name)
    if len(joints):
        ET.SubElement(blocks, "DataSet", index="1", name="Wall cohesive joints", file=joint_path.name)
    collection = output / "aster_results.vtm"
    ET.ElementTree(root).write(collection, encoding="utf-8", xml_declaration=True)
    return {"vtu": str(path.resolve()), "joint_vtu": str(joint_path.resolve()) if len(joints) else None, "vtm": str(collection.resolve()),
            "coordinate_convention": "undeformed",
            "displacement_units": "m"}


def export_joint_series(arrays: dict, load_factors: np.ndarray, damage: np.ndarray,
                        opening: np.ndarray, output: Path) -> dict | None:
    """Write one VTP/VTU time state per converged cohesive load factor."""
    import meshio

    points = np.asarray(arrays["nodes"], dtype=float)
    joints = np.asarray(
        arrays["quad8_joints"] if "quad8_joints" in arrays else arrays["penta6_joints"],
        dtype=np.int64)
    damage = np.asarray(damage, dtype=float)
    opening = np.asarray(opening, dtype=float)
    factors = np.asarray(load_factors, dtype=float)
    if not len(joints):
        return None
    if damage.shape != opening.shape or damage.shape != (len(factors), len(joints)):
        raise ValueError("Cohesive time-series fields do not match joint and load-step counts")
    output.mkdir(parents=True, exist_ok=True)
    root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")
    snapshots = []
    for index, factor in enumerate(factors):
        path = output / f"joint_state_{index:04d}.vtu"
        width = joints.shape[1] // 2
        joint_type = "quad" if width == 4 else "triangle"
        meshio.write(path, meshio.Mesh(
            points, [(joint_type, joints[:, :width])],
            cell_data={
                "joint_damage_state": [damage[index]],
                "normal_opening_m": [opening[index]],
                "normal_opening_mm": [1000.0 * opening[index]],
            }))
        ET.SubElement(collection, "DataSet", timestep=repr(float(factor)),
                      part="0", file=path.name)
        snapshots.append({
            "load_factor": float(factor),
            "vtu": path.name,
            "peak_opening_mm": float(1000.0 * max(0.0, opening[index].max())),
        })
    pvd = output / "joint_history.pvd"
    ET.ElementTree(root).write(pvd, encoding="utf-8", xml_declaration=True)
    return {
        "pvd": str(pvd.resolve()),
        "snapshots": snapshots,
        "time_parameter": "load_factor",
        "opening_units": "normal_opening_m and normal_opening_mm",
    }
