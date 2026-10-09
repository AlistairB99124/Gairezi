"""Measure the opening and slip between the two lips of an XFEM crack (run inside the container).

Usage: python3 crack_opening.py result.med nx ny nz px py pz
The lips are found as pairs of nodes with identical coordinates; (nx, ny, nz) is the crack
plane normal and (px, py, pz) a point on the plane. Positive opening means the lips separate.
"""
import json
import sys

import numpy as np
import medcoupling as mc


def main(path, normal, origin):
    name = mc.GetMeshNames(path)[0]
    mesh = mc.ReadUMeshFromFile(path, name, 0)
    points = mesh.getCoords().toNumPyArray()
    field_name = next(n for n in mc.GetAllFieldNamesOnMesh(path, name) if n.endswith("DEPL"))
    # The field lives on a node profile, so it must be read with its profile and scattered back.
    med_mesh = mc.MEDFileMesh.New(path, name)
    time_step = mc.MEDFileFields(path)[field_name][0]
    values, profile = time_step.getFieldWithProfile(mc.ON_NODES, 0, med_mesh)
    displacement = np.zeros((len(points), 3))
    displacement[profile.toNumPyArray()] = values.toNumPyArray()

    centres = mesh.computeCellCenterOfMass().toNumPyArray()
    conn = mesh.getNodalConnectivity().toNumPyArray()
    index = mesh.getNodalConnectivityIndex().toNumPyArray()
    side = np.zeros(len(points))
    signed_cell = np.sign((centres - origin) @ normal)
    for cell in range(mesh.getNumberOfCells()):
        side[conn[index[cell] + 1 : index[cell + 1]]] += signed_cell[cell]

    keys = {}
    for node, point in enumerate(np.round(points, 5)):
        keys.setdefault(tuple(point), []).append(node)
    pairs = [v for v in keys.values() if len(v) == 2]
    result = {"lip_node_pairs": len(pairs)}
    if pairs:
        upper = np.array([a if side[a] >= side[b] else b for a, b in pairs])
        lower = np.array([b if side[a] >= side[b] else a for a, b in pairs])
        jump = displacement[upper] - displacement[lower]
        opening = jump @ normal
        slip = np.linalg.norm(jump - np.outer(opening, normal), axis=1)
        location = points[upper]
        result.update(
            opening_mm_min=1e3 * opening.min(),
            opening_mm_max=1e3 * opening.max(),
            opening_mm_mean=1e3 * opening.mean(),
            slip_mm_max=1e3 * slip.max(),
            fraction_open=float((opening > 1e-9).mean()),
            location_min=location.min(axis=0).tolist(),
            location_max=location.max(axis=0).tolist(),
        )
    print("OPENING " + json.dumps(result))


if __name__ == "__main__":
    values = [float(v) for v in sys.argv[2:8]]
    main(sys.argv[1], np.array(values[:3]), np.array(values[3:]))
