"""Run with pvpython after attribute_diagnostic_result.py: joint gaps per body pair.

The gap at a joint location is the upper body's displacement minus the lower
body's (wall - bedrock), in metres. Bonded joints give zero.
Prints one JSON line.
"""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

BODY_NAMES = {1: "wall", 2: "bedrock"}
JOINTS = {"wall_bedrock": (1, 2)}


def measure(result: Path) -> dict[str, object]:
    grid = ET.parse(result).getroot().find("PUnstructuredGrid")
    if grid is None:
        raise ValueError("Invalid PVTU")
    by_location: dict[tuple[float, ...], dict[int, np.ndarray]] = {}
    peak = {name: 0.0 for name in BODY_NAMES.values()}
    finite = True
    finite_stress = True
    for piece in grid.findall("Piece"):
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(result.parent / piece.attrib["Source"]))
        reader.Update()
        mesh = reader.GetOutput()
        bodies = mesh.GetCellData().GetArray("GeometryIds")
        if bodies is None:
            raise ValueError("Run attribute_diagnostic_result.py first")
        bodies = vtk_to_numpy(bodies)
        displacement = vtk_to_numpy(mesh.GetPointData().GetArray("displacement")).astype(float)
        finite = finite and bool(np.isfinite(displacement).all())
        for name in ("stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_yz", "stress_xz"):
            values = mesh.GetPointData().GetArray(name)
            if values is None:
                raise ValueError(f"Missing solved stress field: {name}")
            finite_stress = finite_stress and bool(np.isfinite(vtk_to_numpy(values)).all())
        # Elmer writes deformed coordinates; recover the reference position.
        reference = np.round(vtk_to_numpy(mesh.GetPoints().GetData()) - displacement, 6)
        cells = mesh.GetCells()
        offsets = vtk_to_numpy(cells.GetOffsetsArray())
        connectivity = vtk_to_numpy(cells.GetConnectivityArray())
        for body in np.unique(bodies):
            members = np.flatnonzero(bodies == body)
            used = np.zeros(len(reference), dtype=bool)
            for start in range(0, len(members), 20000):
                chunk = members[start:start + 20000]
                used[np.concatenate(
                    [connectivity[offsets[c]:offsets[c + 1]] for c in chunk],
                )] = True
            points = np.flatnonzero(used)
            name = BODY_NAMES[int(body)]
            peak[name] = max(peak[name], float(np.linalg.norm(displacement[points], axis=1).max()))
            for point in points:
                by_location.setdefault(tuple(reference[point]), {})[int(body)] = displacement[point]
    joints = {}
    for name, (upper, lower) in JOINTS.items():
        gaps = np.array([
            sides[upper] - sides[lower]
            for sides in by_location.values() if upper in sides and lower in sides
        ]).reshape(-1, 3)
        joints[name] = {
            "locations": int(len(gaps)),
            "max_abs_gap_m": float(np.linalg.norm(gaps, axis=1).max()) if len(gaps) else 0.0,
            "max_abs_gap_by_axis_m": np.abs(gaps).max(axis=0).tolist() if len(gaps) else None,
            "mean_gap_m": gaps.mean(axis=0).tolist() if len(gaps) else None,
            "min_gap_z_m": float(gaps[:, 2].min()) if len(gaps) else None,
            "max_gap_z_m": float(gaps[:, 2].max()) if len(gaps) else None,
        }
    return {"finite_displacement": finite, "finite_stress": finite_stress,
            "max_displacement_m": peak, "joints": joints}


if __name__ == "__main__":
    print(json.dumps(measure(Path(sys.argv[1]))))
