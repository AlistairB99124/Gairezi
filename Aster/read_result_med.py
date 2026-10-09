"""Convert an Aster result MED to npz (run inside the container).

Usage: python3 read_result_med.py result.med out.npz
"""
import sys

import numpy as np
import medcoupling as mc


def read_node_field(path, mesh_name, suffix):
    name = next(n for n in mc.GetAllFieldNamesOnMesh(path, mesh_name) if n.endswith("__" + suffix))
    order, iteration = mc.GetAllFieldIterations(path, name)[-1][:2]
    field = mc.ReadFieldNode(path, mesh_name, 0, name, order, iteration)
    components = list(field.getArray().getInfoOnComponents())
    return field.getArray().toNumPyArray(), components


def main(med_path, npz_path):
    mesh_name = mc.GetMeshNames(med_path)[0]
    mesh = mc.ReadUMeshFromFile(med_path, mesh_name, 0)
    arrays = {"points": mesh.getCoords().toNumPyArray()}
    for key, suffix in (("displacement", "DEPL"), ("stress", "SIGM_NOEU"), ("reaction", "REAC_NODA")):
        values, components = read_node_field(med_path, mesh_name, suffix)
        arrays[key] = values
        print(key, values.shape, components)
    np.savez(npz_path, **arrays)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
