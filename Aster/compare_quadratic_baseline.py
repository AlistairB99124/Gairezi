"""Compare concrete regions at original source identities, not nearest coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Aster.postprocess_baseline import _peak_principal_stresses
from Aster.source_contract import sha256


def unique_reference(reference):
    """Elmer partition outputs may repeat identities, but never average discrepancies."""
    ids, first, inverse = np.unique(reference["source_node_ids"], return_index=True, return_inverse=True)
    result = {"source_node_ids": ids}
    for key in ("points_m", "displacement_m", "stress_pa"):
        values = reference[key]
        expected = values[first][inverse]
        tolerance = 8 * np.finfo(np.float64).eps * np.maximum(np.abs(expected), np.abs(values))
        if (not np.isfinite(values).all()
                or np.any(np.abs(expected - values) > tolerance)):
            raise ValueError(f"Conflicting repeated reference source identities in {key}")
        result[key] = values[first]
    return result


def tetrahedron_kinematics(points, cells, displacement, barycentric):
    """Exact strain at one barycentric location in straight-sided tetrahedra."""
    from Aster.elastic_diagnostics import cell_kinematics
    barycentric = np.asarray(barycentric, dtype=float)
    if (barycentric.shape != (4,) or not np.isfinite(barycentric).all()
            or np.any(barycentric < 0) or not np.isclose(barycentric.sum(), 1., atol=1e-14, rtol=0)):
        raise ValueError("Expected four nonnegative barycentric coordinates summing to one")
    if cells.shape[1] == 4:
        return cell_kinematics(points, cells, displacement)
    if cells.shape[1] != 10:
        raise ValueError("Expected TETRA4 or TETRA10 connectivity")
    xyz = points[cells[:, :4]]
    edges = xyz[:, 1:] - xyz[:, :1]
    volume = np.abs(np.linalg.det(edges)) / 6
    if not np.isfinite(volume).all() or np.any(volume <= 0):
        raise ValueError("Invalid tetrahedron volume")
    expected = np.stack([(xyz[:, i] + xyz[:, j]) / 2
                         for i, j in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))], axis=1)
    np.testing.assert_allclose(points[cells[:, 4:]], expected, rtol=0., atol=1e-9)
    gradients = np.zeros((len(cells), 4, 3))
    gradients[:, 1:] = np.linalg.inv(edges).transpose(0, 2, 1)
    gradients[:, 0] = -gradients[:, 1:].sum(axis=1)
    vertex_gradients = gradients * (4 * barycentric - 1)[None, :, None]
    edge_gradients = np.stack([4 * (barycentric[i] * gradients[:, j] +
                                     barycentric[j] * gradients[:, i])
                               for i, j in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))], axis=1)
    shape_gradients = np.concatenate([vertex_gradients, edge_gradients], axis=1)
    gradient = np.einsum("nki,nkj->nij", displacement[cells], shape_gradients)
    return (gradient + gradient.transpose(0, 2, 1)) / 2, volume


def centroid_kinematics(points, cells, displacement):
    return tetrahedron_kinematics(points, cells, displacement, np.full(4, .25))


def recovered_peak_audit(nodes, cells, native, young, poisson, faces=None, face_tags=None,
                         boundary_blocks=None):
    """Evaluate direct element strains at the recovered peak's own identity."""
    from Aster.elastic_diagnostics import elastic_stress
    stress = native["stress_pa"]
    tensor = np.zeros((len(stress), 3, 3))
    tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2] = stress[:, :3].T
    for component, i, j in ((3, 0, 1), (4, 1, 2), (5, 0, 2)):
        tensor[:, i, j] = tensor[:, j, i] = stress[:, component]
    row = int(np.linalg.eigvalsh(tensor)[:, -1].argmax())
    node = int(native["candidate_node_ids"][row])
    incident = np.flatnonzero(np.any(cells == node, axis=1))
    if not len(incident):
        raise ValueError("Recovered peak has no incident volume cells")
    displacement = np.zeros_like(nodes)
    displacement[native["candidate_node_ids"]] = native["displacement_m"]
    values, weights, centroids = [], [], []
    locations = list(np.eye(4))
    locations += [(np.eye(4)[i] + np.eye(4)[j]) / 2
                  for i, j in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))]
    for cell in cells[incident]:
        local = int(np.flatnonzero(cell == node)[0])
        strain, volume = tetrahedron_kinematics(nodes, cell[None], displacement, locations[local])
        values.append(elastic_stress(strain, young, poisson)[0])
        weights.append(float(volume[0]))
        strain, _ = centroid_kinematics(nodes, cell[None], displacement)
        centroids.append(float(np.linalg.eigvalsh(elastic_stress(strain, young, poisson)[0])[-1]))
    values = np.asarray(values)
    arithmetic = values.mean(axis=0)
    weighted = np.average(values, axis=0, weights=weights)
    report = {
        "candidate_node_id": node, "xyz_m": nodes[node].tolist(),
        "incident_cell_count": len(incident), "incident_volume_cell_rows": incident.tolist(),
        "recovered_tensile_mpa": float(np.linalg.eigvalsh(tensor[row])[-1] / 1e6),
        "direct_incident_vertex_tensile_mpa": (np.linalg.eigvalsh(values)[:, -1] / 1e6).tolist(),
        "direct_incident_centroid_tensile_mpa": (np.asarray(centroids) / 1e6).tolist(),
        "arithmetic_tensor_average_tensile_mpa": float(np.linalg.eigvalsh(arithmetic)[-1] / 1e6),
        "volume_weighted_tensor_average_tensile_mpa": float(np.linalg.eigvalsh(weighted)[-1] / 1e6),
        "relative_recovered_vs_arithmetic_tensor_error": float(
            np.linalg.norm(tensor[row] - arithmetic) / np.linalg.norm(arithmetic)),
        "interpretation": "direct element stress evaluated at the node, not a converged stress estimate",
    }
    if faces is not None and face_tags is not None and boundary_blocks is not None:
        import re
        from Aster.source_contract import target
        tags = np.unique(face_tags[np.any(faces == node, axis=1)])
        boundaries = []
        for number, block in boundary_blocks.items():
            if target(block) not in tags:
                continue
            name = re.search(r'Name\s*=\s*"([^"]+)"', block)
            boundaries.append({"source_boundary": number, "face_tag": target(block),
                               "name": name[1] if name else None})
        report["incident_source_boundaries"] = boundaries
    return report


def centroid_comparison(nodes, cells, original_ids, reference, native, young, poisson):
    from Aster.elastic_diagnostics import elastic_stress
    reference_order = np.argsort(reference["source_node_ids"])
    requested = original_ids[cells[:, :4]]
    indices = np.searchsorted(reference["source_node_ids"][reference_order], requested)
    if (np.any(indices >= len(reference_order)) or not np.array_equal(
            reference["source_node_ids"][reference_order[indices]], requested)):
        raise ValueError("Missing original element displacement identities")
    ref_u = np.zeros_like(nodes)
    ref_u[cells[:, :4].ravel()] = reference["displacement_m"][reference_order[indices]].reshape(-1, 3)
    port_u = np.zeros_like(nodes)
    port_u[native["candidate_node_ids"]] = native["displacement_m"]
    error = norm = 0.
    peaks = [-np.inf, -np.inf]
    peak_xyz = [None, None]
    for start in range(0, len(cells), 20000):
        batch = cells[start:start + 20000]
        first, volume = centroid_kinematics(nodes, batch[:, :4], ref_u)
        second, _ = centroid_kinematics(nodes, batch, port_u)
        stresses = [elastic_stress(strain, young, poisson) for strain in (first, second)]
        error += float(volume @ ((stresses[1] - stresses[0]) ** 2).sum(axis=(1, 2)))
        norm += float(volume @ (stresses[0] ** 2).sum(axis=(1, 2)))
        for i, stress in enumerate(stresses):
            tension = np.linalg.eigvalsh(stress)[:, -1]
            row = int(tension.argmax())
            if tension[row] > peaks[i]:
                peaks[i] = float(tension[row])
                peak_xyz[i] = nodes[batch[row, :4]].mean(axis=0).tolist()
    if norm <= 0 or not np.isfinite([error, norm, *peaks]).all():
        raise ValueError("Invalid centroid stress comparison")
    return {
        "sampling": "same source tetrahedron centroids; not recovered nodes or full integration-point extrema",
        "relative_volume_weighted_l2_tensor_difference": float(np.sqrt(error / norm)),
        "elmer_tensile_peak_mpa": peaks[0] / 1e6, "aster_tensile_peak_mpa": peaks[1] / 1e6,
        "elmer_peak_xyz_m": peak_xyz[0], "aster_peak_xyz_m": peak_xyz[1],
    }


def compare_region(reference, native) -> dict:
    reference = unique_reference(reference)
    ids = native["source_node_ids"]
    selected = np.flatnonzero(ids >= 0)
    ref_ids = reference["source_node_ids"]
    order = np.argsort(ref_ids)
    position = np.searchsorted(ref_ids[order], ids[selected])
    if (not len(selected) or len(np.unique(ids[selected])) != len(selected)
            or np.any(position >= len(ref_ids))
            or not np.array_equal(ref_ids[order[position]], ids[selected])):
        raise ValueError("Incomplete or ambiguous original source identity correspondence")
    rows = order[position]
    points = native["points_m"][selected]
    np.testing.assert_allclose(points, reference["points_m"][rows], rtol=0., atol=1e-9)
    u, ref_u = native["displacement_m"][selected], reference["displacement_m"][rows]
    stress, ref_stress = native["stress_pa"][selected], reference["stress_pa"][rows]
    if (not all(np.isfinite(value).all() for value in (u, ref_u, stress, ref_stress))
            or np.linalg.norm(ref_u) == 0):
        raise ValueError("Invalid finite/nonzero baseline fields")
    e_peak, _ = _peak_principal_stresses(points, ref_stress)
    a_peak, _ = _peak_principal_stresses(points, stress)
    full_peak, _ = _peak_principal_stresses(native["points_m"], native["stress_pa"])
    if e_peak["value_pa"] == 0:
        raise ValueError("Zero reference tensile peak cannot define a relative difference")
    return {
        "matched_original_nodes": len(selected),
        "relative_l2_displacement_difference": float(np.linalg.norm(u - ref_u) / np.linalg.norm(ref_u)),
        "recovered_tension_at_common_original_nodes": {
            "elmer_mpa": e_peak["value_pa"] / 1e6, "aster_mpa": a_peak["value_pa"] / 1e6,
            "difference_percent": 100 * (a_peak["value_pa"] / e_peak["value_pa"] - 1),
            "elmer_xyz_m": e_peak["centroid_xyz_m"], "aster_xyz_m": a_peak["centroid_xyz_m"],
        },
        "aster_all_nodes_recovered_tension": {
            "mpa": full_peak["value_pa"] / 1e6, "xyz_m": full_peak["centroid_xyz_m"],
            "includes_quadratic_midsides": bool(np.any(ids < 0)),
            "not_same_sampling_as_elmer": bool(np.any(ids < 0)),
        },
    }


def compare(directory: Path, reference_path: Path) -> dict:
    from Aster.source_contract import sif_boundaries
    candidate = directory.parent
    manifest = json.loads((candidate / "quadratic_mesh_manifest.json").read_text(encoding="utf-8"))
    source_path = Path(manifest["source_mesh_manifest"])
    if (sha256(candidate / "quadratic_mesh_arrays.npz") != manifest["quadratic_array_sha256"]
            or sha256(source_path) != manifest["source_mesh_manifest_sha256"]):
        raise ValueError("Changed candidate/source evidence cannot be compared")
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source_directory = Path(source_manifest["source_mesh"]).parent
    for name, digest in manifest["source_hashes"].items():
        if sha256(source_directory / name) != digest:
            raise ValueError(f"Changed source evidence: {name}")
    verification = json.loads((source_path.parent / "verification_report.json").read_text(encoding="utf-8"))
    if sha256(reference_path) != verification["verified_artifact_hashes"]["elmer_baseline_fields.npz"]:
        raise ValueError("Changed reference displacement/stress archive")
    material = json.loads(Path(source_manifest["source_config"]).read_text(encoding="utf-8"))["material"]["concrete"]
    boundary_blocks = sif_boundaries((source_directory / "dam_model.sif").read_text(encoding="utf-8"))
    report = {
        "status": "comparison_produced_pending_baseline_acceptance",
        "engineering_validated": False, "baseline_verified": False,
        "reference_sha256": sha256(reference_path), "regions": {},
        "candidate_manifest_sha256": sha256(candidate / "quadratic_mesh_manifest.json"),
        "export_report_sha256": sha256(directory / "field_export.json"),
        "partition_duplicate_relative_tolerance": 8 * np.finfo(np.float64).eps,
        "stress_method": "principal maximum of recovered nodal tensors, not integration-point stress",
    }
    with np.load(reference_path) as reference, np.load(candidate / "quadratic_mesh_arrays.npz") as mesh:
        reference = unique_reference(reference)
        for label in ("wall", "foundation"):
            path = directory / f"{label}_fields.npz"
            with np.load(path) as native:
                report["regions"][label] = compare_region(reference, native)
                selected = mesh["tetra10"][mesh["body_ids"] == (1 if label == "wall" else 3)]
                if np.all(selected[:, 4:] == -1):
                    selected = selected[:, :4]
                report["regions"][label]["constitutive_centroid_stress"] = centroid_comparison(
                    mesh["nodes"], selected, mesh["original_source_node_ids"], reference, native,
                    material["youngs_modulus"], material["poisson_ratio"],
                )
                report["regions"][label]["recovered_peak_audit"] = recovered_peak_audit(
                    mesh["nodes"], selected, native, material["youngs_modulus"], material["poisson_ratio"],
                    mesh["tri6"], mesh["face_tags"], boundary_blocks,
                )
            report["regions"][label]["native_fields_sha256"] = sha256(path)
    (directory / "elastic_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.directory, args.reference), indent=2))
