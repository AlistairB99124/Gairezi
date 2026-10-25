"""Topology-aware correspondence and concrete-only elastic acceptance checks."""
from __future__ import annotations

import numpy as np


def boundary_triangles(cells):
    faces = np.concatenate([np.delete(cells, i, axis=1) for i in range(4)])
    faces, count = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    if (count > 2).any():
        raise ValueError("Nonmanifold region in stress-sampling mesh")
    return faces[count == 1]


def surface_clearance(point, triangles):
    """Exact distance to triangles, including their edges, rather than vertices."""
    a, b, c = triangles.transpose(1, 0, 2)
    ab, ac, ap = b - a, c - a, point - a
    normals = np.cross(ab, ac)
    squared_normal = np.einsum("ij,ij->i", normals, normals)
    if not len(triangles) or (squared_normal <= 0.).any():
        raise ValueError("Empty or degenerate stress-sampling surface")
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)
    aa, bb, cc = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    denominator = aa * cc - bb ** 2
    v = (cc * dot(ap, ab) - bb * dot(ap, ac)) / denominator
    w = (aa * dot(ap, ac) - bb * dot(ap, ab)) / denominator
    inside = (v >= 0.) & (w >= 0.) & (v + w <= 1.)
    distance_sq = np.where(inside, dot(ap, normals) ** 2 / squared_normal, np.inf)
    for first, second in ((a, b), (b, c), (c, a)):
        edge, delta = second - first, point - first
        fraction = np.clip(dot(delta, edge) / dot(edge, edge), 0., 1.)
        residual = delta - fraction[:, None] * edge
        distance_sq = np.minimum(distance_sq, dot(residual, residual))
    return float(np.sqrt(distance_sq.min()))


def select_stress_samples(points, cells, bodies, minimum_clearance_m=.25):
    """Freeze reproducible interior coordinates for later meshes and both solvers."""
    if not np.isfinite(minimum_clearance_m) or minimum_clearance_m <= 0.:
        raise ValueError("Stress samples require a positive finite clearance")
    samples = []
    for body, name in ((1, "wall"), (3, "foundation")):
        region = cells[bodies == body]
        if not len(region):
            continue
        boundary = points[boundary_triangles(region)]
        centres = points[region].mean(axis=1)
        low, high = points[np.unique(region)].min(axis=0), points[np.unique(region)].max(axis=0)
        fractions = (.2, .5, .8) if body == 1 else (.5,)
        for fraction in fractions:
            target = (low + high) / 2.
            target[2] = low[2] + fraction * (high[2] - low[2])
            # A curved region's bounding-box centre can lie outside the solid.
            # Spread the bounded search across nearby cells, not one surface patch.
            band = max(2. * minimum_clearance_m, .05 * (high[2] - low[2]))
            eligible = np.flatnonzero(abs(centres[:, 2] - target[2]) <= band)
            if not len(eligible):
                raise ValueError(f"No {name} cells near the requested sample elevation")
            order = eligible[np.argsort(np.linalg.norm(centres[eligible] - target, axis=1))]
            candidates = order[np.unique(np.linspace(0, min(len(order), 8192) - 1, 512).astype(int))]
            for candidate in candidates:
                location = centres[candidate]
                if any(np.linalg.norm(location - sample["xyz_m"]) < 1e-8 for sample in samples):
                    continue
                clearance = surface_clearance(location, boundary)
                if clearance >= minimum_clearance_m:
                    samples.append({
                        "name": f"{name}_interior_{round(100 * fraction)}",
                        "body_id": body, "xyz_m": location.tolist(),
                        "minimum_clearance_m": minimum_clearance_m,
                        "selection_clearance_m": clearance,
                    })
                    break
            else:
                raise ValueError(f"No {name} sample in bounded interior search meets {minimum_clearance_m} m clearance")
    if not samples:
        raise ValueError("No concrete regions available for stress samples")
    return {
        "schema_version": 1, "coordinate_frame": "undeformed", "units": "m",
        "selection": "First suitable distinct cell centroid in a bounded distance-ordered search "
                     "towards region midplan and 20/50/80% height; foundation at 50%. "
                     "Clearance from every region boundary, including joints.",
        "engineering_sample_locations_approved": False, "samples": samples,
    }


def fixed_sample_checks(points, cells, bodies, reference_u, port_u, young, poisson, contract):
    from Aster.elastic_diagnostics import cell_kinematics, elastic_stress

    if contract.get("schema_version") != 1 or contract.get("coordinate_frame") != "undeformed" \
            or contract.get("units") != "m" or not contract.get("samples"):
        raise ValueError("Invalid fixed stress-sample contract")
    result = {}
    for sample in contract["samples"]:
        name, body = sample["name"], sample["body_id"]
        point = np.asarray(sample["xyz_m"], dtype=float)
        offset = sample["minimum_clearance_m"]
        if name in result or body not in (1, 3) or point.shape != (3,) \
                or not np.isfinite(point).all() or not np.isfinite(offset) or offset <= 0.:
            raise ValueError("Invalid or duplicate concrete stress sample")
        region = cells[bodies == body]
        if not len(region):
            raise ValueError(f"Missing concrete region for sample {name}")
        clearance = surface_clearance(point, points[boundary_triangles(region)])
        if clearance < offset - 1e-8:
            raise ValueError(f"Sample {name} violates its fixed boundary clearance")
        found = []
        for start in range(0, len(region), 20000):
            batch = region[start:start + 20000]
            xyz = points[batch]
            bary3 = np.linalg.solve((xyz[:, 1:] - xyz[:, :1]).transpose(0, 2, 1),
                                    (point - xyz[:, 0])[..., None])[..., 0]
            weights = np.column_stack([1. - bary3.sum(axis=1), bary3])
            inside = (weights >= -1e-9).all(axis=1) & (weights <= 1. + 1e-9).all(axis=1)
            found.extend((cell, weight) for cell, weight in zip(batch[inside], weights[inside]))
        if not found:
            raise ValueError(f"Fixed sample {name} is outside its concrete region")
        # A point on an element face can have two stresses; retain both instead of
        # silently choosing whichever tetrahedron happens to be stored first.
        tensors = [[], []]
        displacements = [[], []]
        for cell, weights in found:
            for i, displacement in enumerate((reference_u, port_u)):
                strain, _ = cell_kinematics(points, cell[None], displacement)
                tensors[i].append(elastic_stress(strain, young, poisson)[0])
                displacements[i].append(weights @ displacement[cell])
        reference, port = np.asarray(tensors)
        norm = np.linalg.norm(reference)
        if norm == 0.:
            raise ValueError(f"Zero reference tensor norm at stress sample {name}")
        result[name] = {
            "body_id": body, "xyz_m": point.tolist(), "clearance_m": clearance,
            "incident_cell_count": len(found),
            "elmer_element_stress_tensors_pa": reference.tolist(),
            "aster_element_stress_tensors_pa": port.tolist(),
            "elmer_displacement_m": np.mean(displacements[0], axis=0).tolist(),
            "aster_displacement_m": np.mean(displacements[1], axis=0).tolist(),
            "relative_tensor_difference": float(np.linalg.norm(port - reference) / norm),
        }
    return result


def source_node_ids(points, cells, source_points, source_cells):
    """Identify coincident interface sides by their incident tetrahedron, not position alone."""
    if any(kind != 10 for kind, _ in cells):
        raise ValueError("Source identity matching requires tetrahedral results")
    result_cells = np.asarray([ids for _, ids in cells])
    rounded = np.round(np.vstack([source_points, points]), 7)
    _, geometry_ids = np.unique(rounded, axis=0, return_inverse=True)
    src_geo = geometry_ids[:len(source_points)][source_cells]
    res_geo = geometry_ids[len(source_points):][result_cells]
    src_order = np.argsort(src_geo, axis=1)
    res_order = np.argsort(res_geo, axis=1)
    src_keys = np.take_along_axis(src_geo, src_order, axis=1)
    res_keys = np.take_along_axis(res_geo, res_order, axis=1)
    dtype = np.dtype([(f"n{i}", np.int64) for i in range(4)])
    src_struct = np.ascontiguousarray(src_keys).view(dtype).ravel()
    res_struct = np.ascontiguousarray(res_keys).view(dtype).ravel()
    order = np.argsort(src_struct)
    sorted_keys = src_struct[order]
    if np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError("Source has geometrically duplicate tetrahedra")
    index = np.searchsorted(sorted_keys, res_struct)
    if (index >= len(order)).any() or not np.array_equal(sorted_keys[index], res_struct):
        raise ValueError("Result tetrahedra do not match the saved source mesh")
    matched = source_cells[order[index]]
    matched = np.take_along_axis(matched, src_order[order[index]], axis=1)
    result_nodes = np.take_along_axis(result_cells, res_order, axis=1)
    ids = np.full(len(points), -1, dtype=np.int64)
    ids[result_nodes.ravel()] = matched.ravel()
    if (ids < 0).any() or not np.array_equal(ids[result_nodes], matched):
        raise ValueError("Inconsistent source node identity across incident cells")
    if np.max(np.linalg.norm(points - source_points[ids], axis=1)) > 1e-6:
        raise ValueError("Source node correspondence exceeds coordinate tolerance")
    return ids


def concrete_checks(elmer, aster, index):
    from Aster.postprocess_baseline import _peak_principal_stresses, _percent_diff

    points = aster["points_m"]
    e_disp = elmer["displacement_m"][index]
    a_disp = aster["displacement_m"]
    regions = {}
    for body, name in ((1, "wall"), (3, "foundation")):
        key = f"body_{body}_mask"
        if key not in aster or not aster[key].any():
            continue
        mask = aster[key]
        e_t, e_c = _peak_principal_stresses(points[mask], elmer["stress_pa"][index][mask])
        a_t, a_c = _peak_principal_stresses(points[mask], aster["stress_pa"][mask])
        rows = np.flatnonzero(mask)
        for peak in (e_t, e_c, a_t, a_c):
            peak["node_id"] = int(rows[peak["node_id"]])
            peak["source_node_id"] = int(aster["source_node_ids"][peak["node_id"]])
        regions[name] = {
            "elmer_tension_pa": e_t, "aster_tension_pa": a_t,
            "elmer_compression_pa": e_c, "aster_compression_pa": a_c,
            "tensile_peak_comparison": {
                "stress_method": "principal maximum of recovered nodal stress tensor",
                "elmer_mpa": e_t["value_pa"] / 1e6,
                "aster_mpa": a_t["value_pa"] / 1e6,
                "difference_mpa": (a_t["value_pa"] - e_t["value_pa"]) / 1e6,
                "relative_difference_fraction": _percent_diff(a_t["value_pa"], e_t["value_pa"]),
                "difference_percent": 100. * _percent_diff(a_t["value_pa"], e_t["value_pa"]),
                "elmer_xyz_m": e_t["centroid_xyz_m"],
                "aster_xyz_m": a_t["centroid_xyz_m"],
            },
            "relative_l2_displacement_difference": float(
                np.linalg.norm(a_disp[mask] - e_disp[mask]) / np.linalg.norm(e_disp[mask])
            ),
            "unweighted_mean_recovered_nodal_stress_components_pa": {
                "elmer": elmer["stress_pa"][index][mask].mean(axis=0).tolist(),
                "aster": aster["stress_pa"][mask].mean(axis=0).tolist(),
            },
        }
    wall = aster["body_1_mask"]
    wall_ids = np.flatnonzero(wall)
    xyz = points[wall_ids]
    crest = wall_ids[np.isclose(xyz[:, 2], xyz[:, 2].max(), atol=1e-6)]
    crown = crest[np.argmin(abs(points[crest, 1] - np.median(points[crest, 1])))]
    targets = {"crest_crown": crown, "wall_toe": wall_ids[np.argmin(xyz[:, 2])]}
    if "body_3_mask" in aster and aster["body_3_mask"].any():
        ids = np.flatnonzero(aster["body_3_mask"])
        targets["foundation_centre"] = ids[np.argmin(np.linalg.norm(points[ids] - points[ids].mean(axis=0), axis=1))]
    probes = {
        name: {"source_node_id": int(aster["source_node_ids"][row]),
               "xyz_m": points[row].tolist(), "elmer_displacement_m": e_disp[row].tolist(),
               "aster_displacement_m": a_disp[row].tolist(),
               "elmer_stress_pa": elmer["stress_pa"][index[row]].tolist(),
               "aster_stress_pa": aster["stress_pa"][row].tolist()}
        for name, row in targets.items()
    }
    return {"regions": regions, "probes": probes,
            "stress_component_order": ["xx", "yy", "zz", "xy", "yz", "xz"]}


def common_element_stress_checks(points, cells, bodies, reference_u, port_u, young, poisson):
    """Compare constitutive stresses before either solver's nodal recovery."""
    from Aster.elastic_diagnostics import cell_kinematics, elastic_stress

    result = {}
    for body, name in ((1, "wall"), (3, "foundation")):
        selected = cells[bodies == body]
        if not len(selected):
            continue
        total_volume = error = norm = 0.
        means = [np.zeros((3, 3)), np.zeros((3, 3))]
        peaks = [-np.inf, -np.inf]
        for start in range(0, len(selected), 20000):
            ids = selected[start:start + 20000]
            e_strain, volume = cell_kinematics(points, ids, reference_u)
            a_strain, _ = cell_kinematics(points, ids, port_u)
            e_stress = elastic_stress(e_strain, young, poisson)
            a_stress = elastic_stress(a_strain, young, poisson)
            error += float(volume @ ((a_stress - e_stress) ** 2).sum(axis=(1, 2)))
            norm += float(volume @ (e_stress ** 2).sum(axis=(1, 2)))
            total_volume += float(volume.sum())
            for i, stress in enumerate((e_stress, a_stress)):
                means[i] += np.einsum("n,nij->ij", volume, stress)
                peaks[i] = max(peaks[i], float(np.linalg.eigvalsh(stress)[:, -1].max()))
        result[name] = {
            "relative_volume_weighted_l2_tensor_difference": float(np.sqrt(error / norm)),
            "volume_weighted_mean_stress_tensor_pa": {
                "elmer": (means[0] / total_volume).tolist(),
                "aster": (means[1] / total_volume).tolist(),
            },
            "maximum_element_tensile_principal_stress_pa": dict(zip(("elmer", "aster"), peaks)),
        }
    return result
