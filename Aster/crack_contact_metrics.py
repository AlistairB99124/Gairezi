"""Aperture and slip between the lips of a split-node crack (run inside the container).

Usage: python3 crack_contact_metrics.py result.med crack_pairs.npy crack_meta.json
Aperture is the initial gap plus the relative displacement along the crack normal; zero means the lips touch.
"""
import json
import sys

import numpy as np
import medcoupling as mc


def main(path, pairs_path, meta_path):
    name = mc.GetMeshNames(path)[0]
    mesh = mc.ReadUMeshFromFile(path, name, 0)
    n_nodes = len(mesh.getCoords().toNumPyArray())
    field_name = next(n for n in mc.GetAllFieldNamesOnMesh(path, name) if n.endswith("DEPL"))
    med_mesh = mc.MEDFileMesh.New(path, name)
    time_step = mc.MEDFileFields(path)[field_name][0]
    values, profile = time_step.getFieldWithProfile(mc.ON_NODES, 0, med_mesh)
    displacement = np.zeros((n_nodes, 3))
    displacement[profile.toNumPyArray()] = values.toNumPyArray()[:, :3]

    pairs = np.load(pairs_path)
    meta = json.load(open(meta_path))
    normal = np.array(meta["normal"])
    jump = displacement[pairs[:, 1]] - displacement[pairs[:, 0]]
    normal_jump = jump @ normal
    aperture = meta["gap_m"] + normal_jump
    slip = np.linalg.norm(jump - np.outer(normal_jump, normal), axis=1)
    result = {
        "node_pairs": int(len(pairs)),
        "initial_gap_mm": 1e3 * meta["gap_m"],
        "aperture_mm_min": 1e3 * float(aperture.min()),
        "aperture_mm_max": 1e3 * float(aperture.max()),
        "aperture_mm_mean": 1e3 * float(aperture.mean()),
        "fraction_in_contact": float((aperture < 1e-6).mean()),
        "opening_mm_max": 1e3 * float(normal_jump.max()),
        "fraction_open_over_1um": float((normal_jump > 1e-6).mean()),
        "fraction_open_over_100um": float((normal_jump > 1e-4).mean()),
        "slip_mm_max": 1e3 * float(slip.max()),
        "fraction_slipping_over_1um": float((slip > 1e-6).mean()),
    }
    print("METRICS " + json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
