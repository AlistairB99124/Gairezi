"""Export concrete-only quadratic results with topology-proven source identities."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from extract_result_fields import export_stress_vtu, ml, np


def extract(result_path: Path, candidate: Path, output: Path) -> dict:
    with np.load(candidate / "quadratic_mesh_arrays.npz") as archive:
        nodes, cells, bodies = archive["nodes"], archive["tetra10"], archive["body_ids"]
        originals = archive["original_source_node_ids"]
    mesh_names = ml.GetMeshNames(str(result_path))
    if len(mesh_names) != 1:
        raise ValueError("Expected exactly one native result mesh")
    names = ml.GetAllFieldNames(str(result_path))
    loaded = {}
    for suffix in ("DEPL", "SIGM_NOEU"):
        matches = [name for name in names if name.endswith("_" + suffix)]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous native field {suffix}")
        iterations = ml.GetAllFieldIterations(str(result_path), matches[0])
        if len(iterations) != 1:
            raise ValueError("Elastic verification requires exactly one retained state")
        iteration, order, _ = iterations[0]
        loaded[suffix] = ml.ReadFieldNode(str(result_path), mesh_names[0], 0, matches[0], iteration, order)
    native = ml.MEDFileUMesh(str(result_path))
    # IMPR_RESU retains full mesh groups, but fields omit unrequested volumes.
    region_ids = {group: np.asarray(native.getGroupArr(0, group).toNumPyArray())
                  for group in ("WALL", "FOUNDATION")}
    retained = np.unique(np.concatenate(list(region_ids.values())))
    full_mesh = native.getMeshAtLevel(0)
    field_mesh = loaded["SIGM_NOEU"].getMesh()
    if field_mesh.getNumberOfCells() != len(retained):
        raise ValueError("Native field support differs from requested concrete volumes")
    full_conn = np.asarray(full_mesh.getNodalConnectivity().toNumPyArray())
    full_offsets = np.asarray(full_mesh.getNodalConnectivityIndex().toNumPyArray())
    field_conn = np.asarray(field_mesh.getNodalConnectivity().toNumPyArray())
    field_offsets = np.asarray(field_mesh.getNodalConnectivityIndex().toNumPyArray())
    full_positions = np.concatenate([np.arange(full_offsets[cell] + 1, full_offsets[cell + 1])
                                     for cell in retained])
    field_positions = np.concatenate([np.arange(field_offsets[cell] + 1, field_offsets[cell + 1])
                                      for cell in range(len(retained))])
    if len(full_positions) != len(field_positions):
        raise ValueError("Native concrete connectivity inventory differs")
    full_node_ids = full_conn[full_positions]
    field_node_ids = field_conn[field_positions]
    identity = np.full(field_mesh.getNumberOfNodes(), -1, dtype=np.int64)
    identity[field_node_ids] = full_node_ids
    if np.any(identity < 0) or not np.array_equal(identity[field_node_ids], full_node_ids):
        raise ValueError("Native field topology does not identify full mesh nodes")
    np.testing.assert_allclose(field_mesh.getCoords().toNumPyArray(),
                               full_mesh.getCoords().toNumPyArray()[identity], rtol=0., atol=1e-9)
    report = {"status": "finite_concrete_fields_exported_not_baseline_acceptance", "regions": {}}
    for body, label, group in ((1, "wall", "WALL"), (3, "foundation", "FOUNDATION")):
        ids = np.searchsorted(retained, region_ids[group]).tolist()
        stress, displacement = loaded["SIGM_NOEU"][ids], loaded["DEPL"][ids]
        support = stress.getMesh()
        coords = np.asarray(support.getCoords().toNumPyArray())
        if (not np.array_equal(coords, displacement.getMesh().getCoords().toNumPyArray())
                or not np.array_equal(support.getNodalConnectivity().toNumPyArray(),
                                      displacement.getMesh().getNodalConnectivity().toNumPyArray())):
            raise ValueError("Native stress and displacement supports differ")
        expected = cells[bodies == body]
        width = 10 if np.all(expected[:, 4:] >= 0) else 4
        expected = expected[:, :width]
        flat = np.asarray(support.getNodalConnectivity().toNumPyArray())
        if flat.size != len(expected) * (width + 1):
            raise ValueError(f"Unexpected {label} result topology")
        connectivity = flat.reshape(-1, width + 1)[:, 1:]
        mapping = np.full(len(coords), -1, dtype=np.int64)
        mapping[connectivity.ravel()] = expected.ravel()
        if np.any(mapping < 0) or not np.array_equal(mapping[connectivity], expected):
            raise ValueError(f"Inconsistent {label} node identities")
        np.testing.assert_allclose(coords, nodes[mapping], rtol=0., atol=1e-9)
        source_ids = np.full(len(coords), -1, dtype=np.int64)
        original_mask = mapping < len(originals)
        source_ids[original_mask] = originals[mapping[original_mask]]
        raw = np.asarray(stress.getArray().toNumPyArray())
        u = np.asarray(displacement.getArray().toNumPyArray())
        if raw.shape != (len(coords), 6) or u.shape != coords.shape or not np.isfinite(u).all():
            raise ValueError(f"Incomplete/nonfinite {label} result")
        vtu = export_stress_vtu(stress, output / f"{label}_stress.vtu")
        archive_path = output / f"{label}_fields.npz"
        np.savez_compressed(archive_path, points_m=coords, displacement_m=u,
                            stress_pa=raw[:, [0, 1, 2, 3, 5, 4]],
                            candidate_node_ids=mapping, source_node_ids=source_ids)
        report["regions"][label] = {
            "nodes": len(coords), "cells": len(expected), "vtu": vtu.name,
            "fields": archive_path.name, "original_nodes": int(original_mask.sum()),
        }
    (output / "field_export.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(extract(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])), indent=2))
