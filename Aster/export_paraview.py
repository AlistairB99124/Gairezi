"""Export a whole-model nodal displacement field without using deformed coordinates."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def export_mesh(arrays: dict, displacement: np.ndarray, output: Path,
                damage: np.ndarray, opening: np.ndarray,
                concrete_cell_stress: dict | None = None) -> dict:
    import meshio

    points = np.asarray(arrays["nodes"], dtype=float)
    concrete = np.asarray(arrays["tetra4_concrete"], dtype=np.int64)
    rock = np.asarray(arrays["tetra4_bedrock"], dtype=np.int64)
    cells = np.vstack((concrete, rock))
    regions = np.concatenate((arrays["concrete_body_ids"], np.full(len(rock), 2)))
    if displacement.shape != points.shape or not np.isfinite(displacement).all():
        raise ValueError("Aster ParaView export requires a finite full nodal displacement field")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "aster_final.vtu"
    cell_data = {"GeometryIds": [regions]}
    if "damaged_cells" in arrays:
        cell_data["selected_material"] = [arrays["damaged_cells"].astype(np.int32)]
    if concrete_cell_stress:
        # Final-state constant P1 stress per concrete (wall + foundation) cell; rock
        # cells carry NaN because no rock constitutive stress is recovered here.
        for name, values in concrete_cell_stress.items():
            values = np.asarray(values, dtype=float)
            if values.shape != (len(concrete),):
                raise ValueError(f"Concrete cell stress array {name} does not match the concrete cells")
            cell_data[name] = [np.concatenate((values, np.full(len(rock), np.nan)))]
    meshio.write(path, meshio.Mesh(points, [("tetra", cells)],
                                  point_data={"displacement": displacement},
                                  cell_data=cell_data))
    joints = np.asarray(arrays["penta6_joints"], dtype=np.int64)
    if damage.shape != (len(joints),) or opening.shape != damage.shape or not (
            np.isfinite(damage).all() and np.isfinite(opening).all()):
        raise ValueError("Aster joint export requires finite per-facet damage and opening")
    joint_path = output / "aster_joints.vtu"
    if len(joints):
        meshio.write(joint_path, meshio.Mesh(points, [("triangle", joints[:, :3])],
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
