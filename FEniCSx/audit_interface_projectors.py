"""Compare production FEniCSx group mortar projectors/operators to Elmer exports.

This does not solve equilibrium, create a verified baseline, or establish parity.
Local export indices are mapped through each partition node-file row order.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import meshio
import numpy as np
from scipy import sparse

from interface_springs import interface_blocks, mortar_projector, projected_spring_matrix
from three_body_model import source_foundation, source_wall_joints, validate_foundation_faces


def audit(model, projectors):
    diagnostic = json.loads((model / "diagnostic.json").read_text())
    config = json.loads((model / "config.json").read_text())
    _, joints = source_wall_joints(config, diagnostic)
    _, pairs = source_foundation(config, diagnostic, joints)
    source = meshio.read(model / "three_body.msh")
    points = source.points
    triangles = source.cells_dict["triangle"]
    labels = source.cell_data_dict["gmsh:physical"]["triangle"]
    inverse = {value: int(key) for key, value in diagnostic["boundary_id_map"].items()}
    boundaries = [(inverse[int(tag)], tuple(face)) for tag, face in zip(labels, triangles, strict=True)]
    cells = source.cells_dict["tetra"]
    regions = source.cell_data_dict["gmsh:physical"]["tetra"]
    body_nodes = {tag: set(np.unique(cells[regions == tag])) for tag in (1, 2, 3)}
    validate_foundation_faces(points, boundaries, body_nodes, pairs)
    by_tag = {}
    for tag, face in boundaries:
        by_tag.setdefault(tag, []).append(face)
    text = (model / diagnostic["solver_input"]).read_text()
    bc_ids = {}
    for number, block in re.findall(r"Boundary Condition (\d+)\n(.*?)\nEnd", text, re.S):
        name = re.search(r'Name = "([^"]+)_spring"', block)
        if name:
            bc_ids[name[1]] = int(number)
    partitions = sorted((model / "mesh").glob("partitioning.*"))
    if not partitions:
        raise ValueError("Partition node maps are required for local projector exports")
    mappings = {int(path.name.split(".")[1])-1: np.loadtxt(path, usecols=0, dtype=int)-1
                for path in partitions[0].glob("part.*.nodes")}
    results = {}
    for pair in pairs:
        name = pair["name"]
        if "/" not in name:
            continue
        first, second = (np.asarray(by_tag[tag]) for tag in pair["tags"])
        export_rows, export_cols, export_values = [], [], []
        for rank, mapping in mappings.items():
            path = projectors / f"p{bc_ids[name]}_part{rank}.dat"
            if not path.is_file() or path.stat().st_size == 0:
                continue
            data = np.loadtxt(path, ndmin=2)
            if not data.size:
                continue
            export_rows.extend(mapping[data[:, 0].astype(int)-1])
            export_cols.extend(mapping[data[:, 1].astype(int)-1])
            export_values.extend(data[:, 2])
        if not export_values:
            raise ValueError(f"No exported projector for {name}")
        ids = np.unique(np.r_[first.ravel(), second.ravel(), export_rows, export_cols]).astype(int)
        count = len(ids)
        records = [[(np.searchsorted(ids, face), points[face]) for face in side] for side in (first, second)]
        blocks, area = interface_blocks(*records, name)
        expected, _ = mortar_projector(blocks, count)
        actual = sparse.coo_matrix((export_values, (np.searchsorted(ids, export_rows),
                                                  np.searchsorted(ids, export_cols))),
                                  shape=(count, count)).tocsr()
        projected = projected_spring_matrix(blocks, count)
        positive = actual.copy()
        positive.data = np.maximum(positive.data, 0.)
        weights = np.asarray(positive.sum(axis=1)).ravel()
        inverse_weights = np.zeros(count)
        inverse_weights[weights > 0] = 1 / weights[weights > 0]
        reference = actual.T @ sparse.diags(inverse_weights) @ actual
        # The same configured diagonal x/y/z stiffness multiplies both scalar operators.
        # Its Frobenius factor cancels, so this is also the vector-operator relative error.
        results[name] = {
            "projector_relative_frobenius_error": float(sparse.linalg.norm(expected-actual)/sparse.linalg.norm(actual)),
            "operator_relative_frobenius_error": float(sparse.linalg.norm(projected-reference)/sparse.linalg.norm(reference)),
            "spring_n_per_m3": pair["spring_n_per_m3"], "area_m2": area,
            "triangles_per_side": [len(first), len(second)],
        }
    maximum_projector = max(item["projector_relative_frobenius_error"] for item in results.values())
    maximum_operator = max(item["operator_relative_frobenius_error"] for item in results.values())
    return {
        "passed": maximum_projector <= 1e-8 and maximum_operator <= 1e-8,
        "geometry_validation_passed": True, "groups": len(results),
        "maximum_projector_relative_frobenius_error": maximum_projector,
        "maximum_operator_relative_frobenius_error": maximum_operator,
        "comparison_relative_tolerance": 1e-8,
        "integration": "actual coincident slave triangle surfaces; independent group multiplier elimination",
        "equilibrium_solved": False, "full_elastic_parity_available": False,
        "engineering_validated": False,
        "source_sha256": {name: hashlib.sha256((model / name).read_bytes()).hexdigest()
                          for name in ("three_body.msh", "config.json", "diagnostic.json")},
        "pairs": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.model, args.projectors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
