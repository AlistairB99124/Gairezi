"""Bundled MED-runtime converter for a band-resolved synthetic seed specimen."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


def normal_edges(divisions: int, grading_power: int = 1) -> np.ndarray:
    if (isinstance(divisions, bool) or not isinstance(divisions, int)
            or divisions not in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)):
        raise ValueError("Specimen divisions must be a power of two from 8 to 4096")
    if isinstance(grading_power, bool) or not isinstance(grading_power, int) or grading_power not in (1, 2):
        raise ValueError("Normal grading power must be 1 or 2")
    if grading_power == 1:
        return np.concatenate([np.linspace(-.5, -.001, divisions // 2 + 1),
                               np.linspace(.001, .5, divisions // 2 + 1)])
    positive = .001 + .499 * np.linspace(0., 1., divisions // 2 + 1) ** grading_power
    return np.concatenate([-positive[::-1], positive])


def lateral_edges(divisions: int, width_m: float = 1.) -> np.ndarray:
    if (isinstance(divisions, bool) or not isinstance(divisions, int)
            or divisions not in (1, 2, 4, 8, 16, 32)):
        raise ValueError("Lateral specimen divisions must be 1, 2, 4, 8, 16 or 32")
    if (isinstance(width_m, bool) or not isinstance(width_m, (int, float))
            or not np.isfinite(width_m) or width_m < 1.):
        raise ValueError("Lateral specimen width must be finite and at least 1 m")
    return np.linspace(-width_m / 2, width_m / 2, divisions + 1)


def convert(inputs_path: Path, output: Path) -> None:
    from convert_three_body_to_med import mc

    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    n = inputs["specimen_divisions"]
    z = normal_edges(n, inputs.get("normal_grading_power", 1))
    lateral = inputs.get("lateral_divisions", 8)
    width = inputs.get("lateral_width_m", 1.)
    xy = lateral_edges(lateral, width)
    centre = np.asarray(inputs["centre_m"])
    cartesian = mc.MEDCouplingCMesh.New("SEED_SPECIMEN")
    cartesian.setCoords(
        mc.DataArrayDouble(centre[0] + xy),
        mc.DataArrayDouble(centre[1] + xy),
        mc.DataArrayDouble(centre[2] + z),
    )
    volume = cartesian.buildUnstructured()
    tetrahedral = inputs.get("tetrahedral", False)
    if not isinstance(tetrahedral, bool):
        raise ValueError("Tetrahedral specimen selection must be boolean")
    if tetrahedral:
        volume.simplexize(mc.PLANAR_FACE_6)
    surface = volume.buildBoundaryMesh(True)
    coordinates = volume.getCoords()
    coordinates.setInfoOnComponents(["X [m]", "Y [m]", "Z [m]"])
    volume.setName("SEED_SPECIMEN")
    surface.setName("SEED_SPECIMEN")
    centre_points = surface.computeCellCenterOfMass().toNumPyArray()
    names = np.full(surface.getNumberOfCells(), "", dtype="U8")
    for axis, side, name in ((0, -width / 2, "BACK"), (1, -width / 2, "LEFT"),
                              (2, -.5, "BOTTOM"), (2, .5, "TOP")):
        mask = np.isclose(centre_points[:, axis], centre[axis] + side, rtol=0., atol=1e-10)
        if not mask.any():
            raise ValueError(f"Missing synthetic specimen face {name}")
        names[mask] = name
    med = mc.MEDFileUMesh.New()
    med.setMeshAtLevel(0, volume)
    med.setMeshAtLevel(-1, surface)
    med.setFamilyFieldArr(0, mc.DataArrayInt([-1] * volume.getNumberOfCells()))
    med.setFamilyId("VOL", -1)
    med.setFamiliesOnGroup("VOLUME", ["VOL"])
    med.setFamiliesOnGroup("WALL", ["VOL"])
    families = np.zeros(len(names), dtype=np.int32)
    for identifier, name in enumerate(("BACK", "LEFT", "BOTTOM", "TOP"), start=2):
        families[names == name] = -identifier
        med.setFamilyId(name, -identifier)
        med.setFamiliesOnGroup(name, [name])
    med.setFamilyFieldArr(-1, mc.DataArrayInt(families.tolist()))
    med.write(str(output), 2)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Synthetic specimen MED export produced no mesh")


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
