"""Consistent two-sided surface laws on independently refined flat joint traces."""
from __future__ import annotations

import numpy as np

from Aster.quadratic_surface import compact_blocks, triangle_mass


def _barycentric(points, corners):
    basis = (corners[1:] - corners[0]).T
    local = np.linalg.lstsq(basis, (points - corners[0]).T, rcond=None)[0].T
    if np.linalg.matrix_rank(basis) != 2:
        raise ValueError("Degenerate interface triangle")
    if np.max(np.linalg.norm(corners[0] + local @ basis.T - points, axis=1)) > 1e-7:
        raise ValueError("Interface traces are not coplanar")
    return np.column_stack([1 - local.sum(axis=1), local])


def _intersection(master_in_slave):
    """Clip the master triangle by the slave's three barycentric half-planes."""
    polygon = list(master_in_slave)
    for axis in range(3):
        clipped = []
        for i, current in enumerate(polygon):
            following = polygon[(i + 1) % len(polygon)]
            a, b = current[axis], following[axis]
            if a >= 0:
                clipped.append(current)
            if (a < 0 < b) or (b < 0 < a):
                clipped.append(current + a / (a - b) * (following - current))
        polygon = clipped
        if len(polygon) < 3:
            return []
    return polygon


def _shapes(barycentric, width):
    if width == 3:
        return barycentric
    return np.column_stack([barycentric * (2 * barycentric - 1),
                            4 * barycentric[:, 0] * barycentric[:, 1],
                            4 * barycentric[:, 1] * barycentric[:, 2],
                            4 * barycentric[:, 2] * barycentric[:, 0]])


def _faces(nodes, faces):
    faces = np.asarray(faces)
    if (faces.ndim != 2 or faces.shape[1] not in (3, 6)
            or not np.issubdtype(faces.dtype, np.integer)):
        raise ValueError("Interface faces must be TRI3 or padded TRI6")
    rows = []
    for face in faces:
        row = face[face >= 0]
        if len(row) not in (3, 6) or np.any(row >= len(nodes)) or len(np.unique(row)) != len(row):
            raise ValueError("Invalid interface node identities")
        if not np.array_equal(face[:len(row)], row) or np.any(face[len(row):] != -1):
            raise ValueError("Invalid padded interface connectivity")
        if len(row) == 6 and not np.allclose(
                nodes[row[3:]], (nodes[row[:3]] + nodes[row[[1, 2, 0]]]) / 2,
                rtol=0, atol=1e-8):
            raise ValueError("Only straight-sided interface TRI6 faces are supported")
        rows.append(row)
    return rows


def interface_blocks(nodes, slave_faces, master_faces, slave_origins, master_origins, coefficients,
                     *, source_areas_m2=None):
    """Integrate (u_slave-u_master)^T K (u_slave-u_master) on the common refinement.

    Origins identify corresponding *source* triangles, not coincident node IDs.
    Either trace can be sliced and enriched independently. Each overlap triangle
    uses exact TRI6 mass integration after polynomial trace interpolation.
    """
    nodes, coefficients = np.asarray(nodes, dtype=float), np.asarray(coefficients, dtype=float)
    if (nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all()
            or coefficients.ndim != 2 or coefficients.shape[1] != 3
            or len(coefficients) == 0 or not np.isfinite(coefficients).all()
            or np.any(coefficients <= 0)):
        raise ValueError("Expected finite interface geometry and positive source coefficients")
    slaves, masters = _faces(nodes, slave_faces), _faces(nodes, master_faces)
    origins = []
    for supplied, rows in ((slave_origins, slaves), (master_origins, masters)):
        supplied = np.asarray(supplied)
        if (supplied.shape != (len(rows),) or not np.issubdtype(supplied.dtype, np.integer)
                or np.any(supplied < 0) or np.any(supplied >= len(coefficients))
                or len(np.unique(supplied)) != len(coefficients)):
            raise ValueError("Incomplete source interface face coverage")
        order = np.argsort(supplied, kind="stable")
        origins.append(np.split(order, np.cumsum(np.bincount(
            supplied, minlength=len(coefficients)))[:-1]))
    mass = triangle_mass(1.)
    pairs, blocks = [], []
    covered_slave, covered_master = np.zeros(len(slaves)), np.zeros(len(masters))
    slave_area = np.array([np.linalg.norm(np.cross(
        nodes[row[1]] - nodes[row[0]], nodes[row[2]] - nodes[row[0]])) / 2 for row in slaves])
    master_area = np.array([np.linalg.norm(np.cross(
        nodes[row[1]] - nodes[row[0]], nodes[row[2]] - nodes[row[0]])) / 2 for row in masters])
    if np.any(slave_area <= 0) or np.any(master_area <= 0):
        raise ValueError("Degenerate interface triangle")
    if source_areas_m2 is not None:
        expected = np.asarray(source_areas_m2, dtype=float)
        if (expected.shape != (len(coefficients),) or not np.isfinite(expected).all()
                or np.any(expected <= 0)):
            raise ValueError("Invalid source interface areas")
        for supplied, areas in ((slave_origins, slave_area), (master_origins, master_area)):
            if not np.allclose(np.bincount(supplied, weights=areas, minlength=len(expected)),
                               expected, rtol=2e-8, atol=1e-15):
                raise ValueError("Refined interface does not conserve each source triangle area")
    overlaps = 0
    for parent, (slave_ids, master_ids) in enumerate(zip(*origins)):
        if np.intersect1d(np.concatenate([slaves[i] for i in slave_ids]),
                          np.concatenate([masters[i] for i in master_ids])).size:
            raise ValueError("Two-sided interfaces require split node identities")
        for i in slave_ids:
            slave = slaves[i]
            corners = nodes[slave[:3]]
            for j in master_ids:
                master = masters[j]
                polygon = _intersection(_barycentric(nodes[master[:3]], corners))
                for k in range(1, len(polygon) - 1):
                    piece_bary = np.array([polygon[0], polygon[k], polygon[k + 1]])
                    area_fraction = abs(np.linalg.det(piece_bary))
                    if area_fraction <= 1e-13:
                        continue
                    area = slave_area[i] * area_fraction
                    piece = piece_bary @ corners
                    sample = np.vstack([piece, (piece + piece[[1, 2, 0]]) / 2])
                    interpolation = np.column_stack([
                        _shapes(_barycentric(sample, corners), len(slave)),
                        -_shapes(_barycentric(sample, nodes[master[:3]]), len(master))])
                    local_mass = area * (interpolation.T @ mass @ interpolation)
                    local_mass = (local_mass + local_mass.T) / 2
                    identities = np.concatenate([slave, master])
                    pairs.append(np.stack(np.meshgrid(identities, identities, indexing="ij"),
                                          axis=-1).reshape(-1, 2))
                    blocks.append((local_mass[..., None, None] * np.diag(coefficients[parent]))
                                  .reshape(-1, 3, 3))
                    covered_slave[i] += area
                    covered_master[j] += area
                    overlaps += 1
    for covered, expected in ((covered_slave, slave_area), (covered_master, master_area)):
        if not np.allclose(covered, expected, rtol=2e-8, atol=1e-15):
            raise ValueError("Interface common refinement has gaps or overlapping coverage")
    pairs, blocks = compact_blocks(np.concatenate(pairs), np.concatenate(blocks))
    return pairs, blocks, {
        "law": "consistent_two_sided_TRI3_TRI6",
        "source_triangles": len(coefficients), "slave_faces": len(slaves),
        "master_faces": len(masters), "overlap_triangles": overlaps,
        "area_m2": float(slave_area.sum()),
        "integrated_directional_stiffness_n_per_m": (slave_area @ coefficients[
            np.asarray(slave_origins)]).tolist(),
        "maximum_relative_coverage_error": float(max(
            np.max(np.abs(covered_slave - slave_area) / slave_area),
            np.max(np.abs(covered_master - master_area) / master_area))),
    }
