"""Bundled MED-runtime converter for the curved dam-wall crack submodel block."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

FACES = ("CUT_START", "CUT_END", "BOTTOM", "TOP", "UPSTREAM", "DOWNSTREAM")


def face_names(centres: np.ndarray, theta: np.ndarray, radius: np.ndarray, z: np.ndarray,
               top_is_crest: bool) -> np.ndarray:
    """Name boundary faces from their (theta, r, z) parameter-space centres."""
    names = np.full(len(centres), "", dtype="U10")
    for axis, value, name in ((0, theta[0], "CUT_START"), (0, theta[-1], "CUT_END"),
                              (2, z[0], "BOTTOM"), (2, z[-1], "CREST" if top_is_crest else "TOP"),
                              (1, radius[-1], "UPSTREAM"), (1, radius[0], "DOWNSTREAM")):
        mask = np.isclose(centres[:, axis], value, rtol=0., atol=1e-9)
        if not mask.any():
            raise ValueError(f"Missing submodel face {name}")
        names[mask] = name
    if np.any(names == ""):
        raise ValueError("Unnamed submodel boundary faces")
    return names


def mapped(parameters: np.ndarray) -> np.ndarray:
    theta, radius, z = parameters.T
    return np.column_stack([radius * np.sin(theta), radius * np.cos(theta), z])


def convert(inputs_path: Path, output: Path) -> None:
    from convert_three_body_to_med import mc

    layout = json.loads(inputs_path.read_text(encoding="utf-8"))["layout"]
    theta, radius, z = (np.asarray(layout[key], dtype=float)
                        for key in ("theta_edges_rad", "radius_edges_m", "z_edges_m"))
    for name, edges in (("theta", theta), ("radius", radius), ("z", z)):
        if len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError(f"Submodel {name} edges must increase")
    cartesian = mc.MEDCouplingCMesh.New("CRACK_SUBMODEL")
    # (theta, r, z) keeps a positive Jacobian after mapping to x = r sin(theta), y = r cos(theta).
    cartesian.setCoords(mc.DataArrayDouble(theta), mc.DataArrayDouble(radius), mc.DataArrayDouble(z))
    volume = cartesian.buildUnstructured()
    surface = volume.buildBoundaryMesh(True)
    names = face_names(surface.computeCellCenterOfMass().toNumPyArray(), theta, radius, z,
                       layout["top_is_crest"])
    coordinates = mc.DataArrayDouble(mapped(volume.getCoords().toNumPyArray()))
    coordinates.setInfoOnComponents(["X [m]", "Y [m]", "Z [m]"])
    volume.setCoords(coordinates)
    surface.setCoords(coordinates)
    volume.setName("CRACK_SUBMODEL")
    surface.setName("CRACK_SUBMODEL")
    if np.any(volume.getMeasureField(False).getArray().toNumPyArray() <= 0):
        raise ValueError("Submodel mapping produced inverted cells")
    med = mc.MEDFileUMesh.New()
    med.setMeshAtLevel(0, volume)
    med.setMeshAtLevel(-1, surface)
    med.setFamilyFieldArr(0, mc.DataArrayInt([-1] * volume.getNumberOfCells()))
    med.setFamilyId("VOL", -1)
    med.setFamiliesOnGroup("VOLUME", ["VOL"])
    med.setFamiliesOnGroup("WALL", ["VOL"])
    families = np.zeros(len(names), dtype=np.int32)
    present = [name for name in (*FACES, "CREST") if np.any(names == name)]
    for identifier, name in enumerate(present, start=2):
        families[names == name] = -identifier
        med.setFamilyId(name, -identifier)
        med.setFamiliesOnGroup(name, [name])
    cut = [name for name in ("CUT_START", "CUT_END", "BOTTOM", "TOP") if name in present]
    med.setFamiliesOnGroup("CUT", cut)
    med.setFamilyFieldArr(-1, mc.DataArrayInt(families.tolist()))
    med.write(str(output), 2)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Submodel MED export produced no mesh")
    print(json.dumps({"cells": volume.getNumberOfCells(), "nodes": volume.getNumberOfNodes(),
                      "faces": {name: int(np.count_nonzero(names == name)) for name in present}}))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
