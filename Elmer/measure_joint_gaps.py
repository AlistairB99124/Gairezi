"""Run with pvpython after attribute_diagnostic_result.py: joint gaps per body pair.

The gap at a joint location is the upper body's displacement minus the lower
body's (wall - bedrock), in metres. Bonded joints give zero.
Optional extra arguments are the elevations (m) of spring joints between wall
thickness blocks; wall_joint_k gives upper block minus lower block there.
Prints one JSON line.
"""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

BODY_NAMES = {1: "wall", 2: "bedrock", 3: "foundation"}
JOINTS = {"wall_bedrock": (1, 2)}


def summarise(gaps: np.ndarray) -> dict[str, object]:
    return {
        "locations": int(len(gaps)),
        "max_abs_gap_m": float(np.linalg.norm(gaps, axis=1).max()) if len(gaps) else 0.0,
        "max_abs_gap_by_axis_m": np.abs(gaps).max(axis=0).tolist() if len(gaps) else None,
        "mean_gap_m": gaps.mean(axis=0).tolist() if len(gaps) else None,
        "min_gap_z_m": float(gaps[:, 2].min()) if len(gaps) else None,
        "max_gap_z_m": float(gaps[:, 2].max()) if len(gaps) else None,
    }


def measure(result: Path, joint_levels_m: tuple[float, ...] = ()) -> dict[str, object]:
    grid = ET.parse(result).getroot().find("PUnstructuredGrid")
    if grid is None:
        raise ValueError("Invalid PVTU")
    by_location: dict[tuple[float, ...], dict[int, dict[tuple[float, ...], np.ndarray]]] = {}
    latest_by_location: dict[tuple[float, ...], dict[int, np.ndarray]] = {}
    peak = {name: 0.0 for name in BODY_NAMES.values()}
    finite = True
    finite_stress = True
    # Per wall joint: reference location -> {"upper"/"lower": displacement}.
    wall_sides: list[dict[tuple[float, ...], dict[str, np.ndarray]]] = [{} for _ in joint_levels_m]
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
        if joint_levels_m:
            sizes = np.diff(offsets)
            cell_ids = np.repeat(np.arange(len(sizes)), sizes)
            centre_z = np.add.reduceat(reference[connectivity, 2], offsets[:-1]) / sizes
            wall = bodies[cell_ids] == 1
            for index, level_m in enumerate(joint_levels_m):
                on_level = wall & (np.abs(reference[connectivity, 2] - level_m) < 1e-5)
                above = np.zeros(len(reference), dtype=bool)
                below = np.zeros(len(reference), dtype=bool)
                above[connectivity[on_level & (centre_z[cell_ids] > level_m)]] = True
                below[connectivity[on_level & (centre_z[cell_ids] < level_m)]] = True
                # A joint node copy touches cells of one block only; points touching both are not on a joint.
                for side, mask in (("upper", above & ~below), ("lower", below & ~above)):
                    for point in np.flatnonzero(mask):
                        wall_sides[index].setdefault(tuple(reference[point]), {})[side] = displacement[point]
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
                copies = by_location.setdefault(tuple(reference[point]), {}).setdefault(int(body), {})
                copies[tuple(np.round(displacement[point], 14))] = displacement[point]
                latest_by_location.setdefault(tuple(reference[point]), {})[int(body)] = displacement[point]
    joints = {}
    pairs = ({"wall_foundation": (1, 3), "foundation_bedrock": (3, 2)}
             if any(3 in sides for sides in by_location.values()) else JOINTS)
    if pairs is JOINTS:
        peak.pop("foundation")
    for name, (upper, lower) in pairs.items():
        gaps = np.array([
            upper_value - lower_value
            for sides in by_location.values() if upper in sides and lower in sides
            for upper_value in sides[upper].values() for lower_value in sides[lower].values()
        ]).reshape(-1, 3)
        if pairs is JOINTS:
            # Preserve the legacy per-location sample convention without a foundation.
            gaps = np.array([
                sides[upper] - sides[lower]
                for sides in latest_by_location.values() if upper in sides and lower in sides
            ]).reshape(-1, 3)
        joints[name] = summarise(gaps)
    diagnostic_path = result.parent.parent / "diagnostic.json"
    if diagnostic_path.is_file():
        diagnostic = json.loads(diagnostic_path.read_text())
        logical = diagnostic.get("spring_pairs", {})
        tag_groups = {}
        for name, entry in logical.items():
            if isinstance(entry, dict):
                for part, tags in entry["faces"].items():
                    tag_groups[diagnostic["boundary_id_map"][str(tags[0])]] = (name, part)
        if tag_groups:
            source_points, locations = {}, {}
            with (result.parent.parent / "three_body.msh").open() as stream:
                for line in stream:
                    if line.strip() == "$Nodes":
                        for _ in range(int(next(stream))):
                            fields = next(stream).split()
                            source_points[int(fields[0])] = tuple(round(float(x), 6) for x in fields[1:4])
                    elif line.strip() == "$Elements":
                        for _ in range(int(next(stream))):
                            fields = [int(v) for v in next(stream).split()]
                            if fields[1] == 2 and fields[3] in tag_groups:
                                name, part = tag_groups[fields[3]]
                                locations.setdefault((name, part), set()).update(
                                    source_points[n] for n in fields[3 + fields[2]:])
            for (name, part), keys in locations.items():
                upper, lower = pairs[name]
                gaps = np.asarray([
                    u - l for key in keys for sides in [by_location.get(key, {})]
                    if upper in sides and lower in sides
                    for u in sides[upper].values() for l in sides[lower].values()
                ]).reshape(-1, 3)
                joints[name].setdefault("faces", {})[part] = summarise(gaps)
    for index, level_m in enumerate(joint_levels_m):
        gaps = np.array([
            sides["upper"] - sides["lower"]
            for sides in wall_sides[index].values() if "upper" in sides and "lower" in sides
        ]).reshape(-1, 3)
        joints[f"wall_joint_{index + 1}"] = {"z_m": level_m, **summarise(gaps)}
    return {"finite_displacement": finite, "finite_stress": finite_stress,
            "max_displacement_m": peak, "joints": joints}


if __name__ == "__main__":
    # Levels arrive as one "levels=-10,-20" token: pvpython would consume bare negative numbers.
    tokens = [arg.split("=", 1)[1] for arg in sys.argv[2:] if arg.startswith("levels=")]
    levels = tuple(float(value) for token in tokens for value in token.split(",") if value)
    print(json.dumps(measure(Path(sys.argv[1]), levels)))
