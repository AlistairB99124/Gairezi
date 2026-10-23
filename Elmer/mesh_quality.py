"""Report per-body tetrahedral dihedrals and worst cells near the wall.

Requires meshio, NumPy and SciPy. Distances are centroid-to-wall-node distances,
not exact surface distances; this is a mesh diagnostic, not a safety assessment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import cKDTree


def quality(mesh_path: Path) -> dict:
    mesh = meshio.read(mesh_path)
    points = mesh.points
    tetrahedra = mesh.cells_dict["tetra"]
    labels = mesh.cell_data_dict["gmsh:physical"]["tetra"]
    p = points[tetrahedra]
    normals = []
    for a, b, c, opposite in ((1, 2, 3, 0), (0, 3, 2, 1), (0, 1, 3, 2), (0, 2, 1, 3)):
        normal = np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a])
        inward = np.einsum("ij,ij->i", normal, p[:, opposite] - p[:, a]) > 0
        normal[inward] *= -1
        normals.append(normal / np.linalg.norm(normal, axis=1)[:, None])
    minimum = np.full(len(p), 180.)
    for a in range(4):
        for b in range(a):
            cosine = -np.einsum("ij,ij->i", normals[a], normals[b])
            minimum = np.minimum(minimum, np.degrees(np.arccos(np.clip(cosine, -1., 1.))))
    centres = p.mean(axis=1)
    distance = cKDTree(points[np.unique(tetrahedra[labels == 1])]).query(centres)[0]
    edges = np.stack([np.linalg.norm(p[:, a] - p[:, b], axis=1) for a in range(4) for b in range(a)], axis=1)
    report = {"mesh": str(mesh_path), "distance_definition": "tet centroid to nearest wall vertex", "bodies": {}}
    for body, name in ((1, "wall"), (2, "bedrock"), (3, "foundation")):
        selected = labels == body
        if not selected.any():
            continue
        entry = {}
        for region, mask in (("all", selected), ("near_wall_lt_2m", selected & (distance < 2.)), ("far_wall_ge_2m", selected & (distance >= 2.))):
            ids = np.flatnonzero(mask)
            if not len(ids):
                continue
            worst = ids[np.argsort(minimum[ids])[:5]]
            entry[region] = {"cells": len(ids), "minimum_dihedral_deg": float(minimum[ids].min()),
                             "count_below_5deg": int((minimum[ids] < 5.).sum()),
                             "percent_below_5deg": float(100 * np.mean(minimum[ids] < 5.)),
                             "minimum_edge_m": float(edges[ids].min()),
                             "worst_cells": [{"dihedral_deg": float(minimum[i]), "reference_xyz_m": centres[i].tolist(),
                                              "distance_to_wall_m": float(distance[i]), "minimum_edge_m": float(edges[i].min())}
                                             for i in worst]}
        report["bodies"][name] = entry
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = quality(arguments.mesh)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
