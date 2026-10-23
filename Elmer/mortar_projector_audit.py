"""Audit Elmer-exported Galerkin projectors against coincident triangle mass.

Run with meshio/NumPy/SciPy, e.g. in WSL. Local projector indices are resolved
through partition node-file row order, not solver-global numbering.
"""
import argparse
import json
from pathlib import Path
import re

import meshio
import numpy as np
from scipy import sparse

if __package__ in (None, ""):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def audit(model, projectors):
    diagnostic = json.loads((model / "diagnostic.json").read_text())
    mesh = meshio.read(model / "three_body.msh")
    points = mesh.points
    triangles = mesh.cells_dict["triangle"]
    labels = mesh.cell_data_dict["gmsh:physical"]["triangle"]
    tags = {int(k): v for k, v in diagnostic["boundary_id_map"].items()}
    source_pairs = diagnostic["spring_pairs"]
    pairs = {}
    for name in ("wall_foundation", "foundation_bedrock"):
        entry = source_pairs[name]
        if isinstance(entry, dict):
            pairs.update({f"{name}/{part}": pair for part, pair in entry["faces"].items()})
        else:
            pairs[name] = entry
    text = (model / diagnostic["solver_input"]).read_text()
    bc_ids = {}
    for number, block in re.findall(r"Boundary Condition (\d+)\n(.*?)\nEnd", text, re.S):
        name = re.search(r'Name = "([^"]+)_spring"', block)
        if name:
            bc_ids[name[1]] = int(number)
    parts = sorted((model / "mesh").glob("partitioning.*"))
    if not parts:
        raise ValueError("Partition node maps are required")
    mappings = {int(path.name.split(".")[1])-1: np.loadtxt(path, usecols=0, dtype=int)-1
                for path in parts[0].glob("part.*.nodes")}
    report = {}
    exports = {p.name: p for p in projectors.glob("p*_part*.dat")}
    def location(face):
        return tuple(sorted(tuple(p) for p in np.round(points[list(face)], 8)))
    for name, (slave, master) in pairs.items():
        first = triangles[labels == tags[slave]]
        second = {location(face): face for face in triangles[labels == tags[master]]}
        rows, cols, values = [], [], []
        for face in first:
            p = points[face]
            other = second[location(face)]
            other = other[np.argmin(np.linalg.norm(p[:, None]-points[other][None], axis=2), axis=1)]
            area = np.linalg.norm(np.cross(p[1]-p[0], p[2]-p[0]))/2
            mass = area/12 * (np.ones((3, 3)) + np.eye(3))
            rows.extend(np.repeat(face, 6))
            cols.extend(np.tile(np.r_[face, other], 3))
            values.extend(np.c_[mass, -mass].ravel())
        data_parts, row_parts, col_parts = [], [], []
        for rank, mapping in mappings.items():
            path = exports.get(f"p{bc_ids[name]}_part{rank}.dat")
            if path is None or path.stat().st_size == 0:
                continue
            data = np.loadtxt(path, ndmin=2)
            if not data.size:
                continue
            data_parts.extend(data[:, 2])
            row_parts.extend(mapping[data[:, 0].astype(int)-1])
            col_parts.extend(mapping[data[:, 1].astype(int)-1])
        if not data_parts:
            raise ValueError(f"Missing exported projector: {name}")
        # Audit in the small pair-local index space, retaining any unexpected
        # exported nodes. Global-size CSR arrays make thousands of planes O(NP).
        source_ids = np.unique(np.r_[rows,cols,row_parts,col_parts]).astype(int)
        shape = (len(source_ids),len(source_ids))
        expected = sparse.coo_matrix((values,(np.searchsorted(source_ids,rows),
                    np.searchsorted(source_ids,cols))),shape=shape).tocsr()
        actual = sparse.coo_matrix((data_parts,(np.searchsorted(source_ids,row_parts),
                    np.searchsorted(source_ids,col_parts))),shape=shape).tocsr()
        difference = actual - expected
        norm = sparse.linalg.norm(expected)
        entry = {"relative_frobenius_error": float(sparse.linalg.norm(difference)/norm),
                 "maximum_entry_error_m2": float(abs(difference).max()),
                 "exported_nonzeros": actual.nnz, "triangles_per_side": len(first)}
        if "/" in name:
            from Elmer.mortar_interfaces import fitted_plane, orientation_class
            logical,part = name.split("/",1)
            contract = source_pairs[logical]
            tolerance = contract.get("tolerance_m",1e-7)
            q = points[np.unique(first)]
            normal,offset,fit_error = fitted_plane(q)
            coo = actual.tocoo()
            used = source_ids[np.unique(np.r_[coo.row[np.abs(coo.data)>1e-15], coo.col[np.abs(coo.data)>1e-15]])]
            distance = np.abs(points[used] @ normal-offset)
            xyz = points[first]
            normals = np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
            normals /= np.linalg.norm(normals,axis=1)[:,None]
            spread = float(np.degrees(np.arccos(np.clip(np.abs(normals @ normals.T).min(),-1.,1.))))
            classes = {orientation_class(points[f]) for f in first}
            declared = contract["planes"][part].get("orientation_class")
            allowed = set(np.r_[first.ravel(),np.asarray(list(second.values())).ravel()])
            unexpected = set(int(n) for n in used)-allowed
            entry.update(maximum_exported_node_plane_distance_m=float(distance.max()),
                         maximum_fitted_plane_distance_m=fit_error,
                         tolerance_m=tolerance,
                         exported_nodes_off_plane_above_tolerance=int((distance>tolerance+1e-8).sum()),
                         exported_nodes_off_plane_above_1e_7m=int((distance>1e-7).sum()),
                         orientation_classes=sorted(classes),
                         actual_normal_spread_deg=spread,
                         normal_spread_within_limit=spread<=contract.get("max_angle_deg",.001)+1e-6,
                         orientation_class_consistent=len(classes)==1 and (declared is None or declared in classes),
                         unexpected_exported_nodes=len(unexpected))
        report[name] = entry
    planar = [v["maximum_exported_node_plane_distance_m"] for v in report.values()
              if "maximum_exported_node_plane_distance_m" in v]
    maximum_error = max(v["relative_frobenius_error"] for v in report.values())
    geometry_passed = bool(planar) and all(
        v.get("exported_nodes_off_plane_above_tolerance",0)==0
        and v.get("normal_spread_within_limit",False)
        and v.get("orientation_class_consistent",False)
        and v.get("unexpected_exported_nodes",1)==0 for v in report.values())
    exact_mass_passed = maximum_error < 1e-8
    passed = geometry_passed and exact_mass_passed
    return {"passed": passed,"geometry_and_orientation_passed":geometry_passed,
            "exact_triangle_mass_passed":exact_mass_passed,"exact_mass_relative_tolerance":1e-8,
            "projector_numbering": "local, partition node-file row order",
            "comparison": "exact coincident linear triangle consistent mass [M,-M]",
            "pairs": report,
            "maximum_relative_frobenius_error": maximum_error,
            "maximum_exported_node_plane_distance_m": max(planar) if planar else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projectors", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.model, args.projectors or args.model)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "pairs"}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
