"""Export the nodal displacement of an Aster result MED to npz (run inside the container).

Usage: python3 export_depl.py result.med out.npz
"""
import sys

import numpy as np
import medcoupling as mc


def main(path, out):
    name = mc.GetMeshNames(path)[0]
    mesh = mc.ReadUMeshFromFile(path, name, 0)
    count = len(mesh.getCoords().toNumPyArray())
    field_name = next(n for n in mc.GetAllFieldNamesOnMesh(path, name) if n.endswith("DEPL"))
    med_mesh = mc.MEDFileMesh.New(path, name)
    time_step = mc.MEDFileFields(path)[field_name][0]
    values, profile = time_step.getFieldWithProfile(mc.ON_NODES, 0, med_mesh)
    displacement = np.zeros((count, 3))
    displacement[profile.toNumPyArray()] = values.toNumPyArray()[:, :3]
    np.savez(out, points=mesh.getCoords().toNumPyArray(), displacement=displacement)
    print("exported", displacement.shape)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
