"""Consistent flat TRI3/TRI6 surface laws for native displacement meshes."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

# MED TRI6 order: vertices (0,1,2), then edge midpoints (01,12,20).
_SHAPES = (
    {(2, 0, 0): 2., (1, 0, 0): -1.},
    {(0, 2, 0): 2., (0, 1, 0): -1.},
    {(0, 0, 2): 2., (0, 0, 1): -1.},
    {(1, 1, 0): 4.},
    {(0, 1, 1): 4.},
    {(1, 0, 1): 4.},
)
_LINEAR_SHAPES = (
    {(1, 0, 0): 1.},
    {(0, 1, 0): 1.},
    {(0, 0, 1): 1.},
)


def triangle_mass(area_m2: float, rows: int = 6, columns: int = 6) -> np.ndarray:
    """Exact integral of N_i*N_j; barycentric monomials avoid tuned quadrature."""
    if not math.isfinite(area_m2) or area_m2 <= 0:
        raise ValueError("Quadratic surface area must be finite and positive")
    if rows not in (3, 6) or columns not in (3, 6):
        raise ValueError("Surface traces must be TRI3 or TRI6")
    mass = np.zeros((rows, columns))
    for i, first in enumerate(_LINEAR_SHAPES if rows == 3 else _SHAPES):
        for j, second in enumerate(_LINEAR_SHAPES if columns == 3 else _SHAPES):
            value = Fraction(0)
            for power_a, coefficient_a in first.items():
                for power_b, coefficient_b in second.items():
                    powers = tuple(a + b for a, b in zip(power_a, power_b))
                    integral = Fraction(
                        2 * math.prod(math.factorial(value) for value in powers),
                        math.factorial(sum(powers) + 2),
                    )
                    value += int(coefficient_a * coefficient_b) * integral
            mass[i, j] = area_m2 * float(value)
    return mass


def group_stiffness_blocks(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Share numerically equal blocks with a checked 1e-10 relative error bound.

    Each block has its own decimal scale, so large supports cannot erase
    smaller joints. This is coefficient encoding precision, not calibration.
    """
    blocks = np.asarray(blocks, dtype=float)
    if blocks.ndim != 3 or blocks.shape[1:] != (3, 3) or not np.isfinite(blocks).all():
        raise ValueError("Expected finite 3x3 stiffness blocks")
    magnitude = np.max(np.abs(blocks), axis=(1, 2))
    if np.any(magnitude <= 0):
        raise ValueError("Cannot group zero stiffness blocks")
    exponent = np.floor(np.log10(magnitude))
    scale = 10. ** exponent
    encoded = np.round(blocks / scale[:, None, None], decimals=10)
    keys, inverse = np.unique(np.column_stack([exponent, encoded.reshape(-1, 9)]),
                              axis=0, return_inverse=True)
    grouped = keys[:, 1:].reshape(-1, 3, 3) * (10. ** keys[:, 0])[:, None, None]
    error = np.max(np.abs(grouped[inverse] - blocks), axis=(1, 2)) / magnitude
    maximum = float(np.max(error))
    if not np.isfinite(maximum) or maximum > 1e-10:
        raise ValueError(f"Grouped stiffness exceeds per-block relative encoding bound: {maximum}")
    return grouped, inverse, {"blocks": len(blocks), "groups": len(grouped),
                              "maximum_block_relative_error": maximum,
                              "allowed_block_relative_error": 1e-10}


def surface_blocks(
    nodes: np.ndarray, faces: np.ndarray, stiffness_n_per_m3: np.ndarray,
    master_faces: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return COO node-pair 3x3 blocks of the integrated surface operator.

    Two-sided laws integrate (u_slave-u_master), including mixed TRI3/TRI6
    traces. Coincident node coordinates never imply shared identities.
    Blocks include both triangles of the matrix, not only an upper half.
    """
    nodes = np.asarray(nodes, dtype=float)
    faces = np.asarray(faces)
    stiffness = np.asarray(stiffness_n_per_m3, dtype=float)
    if (nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all()
            or faces.ndim != 2 or faces.shape[1] not in (3, 6) or len(faces) == 0
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= len(nodes))):
        raise ValueError("Expected finite nodes and nonempty valid TRI3/TRI6 connectivity")
    if stiffness.shape != (3, 3) or not np.isfinite(stiffness).all():
        raise ValueError("Surface stiffness must be a finite 3x3 matrix")
    if not np.allclose(stiffness, stiffness.T, rtol=1e-12, atol=1e-10):
        raise ValueError("Surface stiffness must be symmetric")
    if np.linalg.eigvalsh(stiffness).min() < -1e-12 * max(1., np.linalg.norm(stiffness)):
        raise ValueError("Surface stiffness must be positive semidefinite")
    xyz = nodes[faces]
    midpoint = (xyz[:, [0, 1, 2]] + xyz[:, [1, 2, 0]]) / 2
    if faces.shape[1] == 6 and not np.allclose(xyz[:, 3:], midpoint, rtol=0., atol=1e-8):
        raise ValueError("Only straight-sided TRI6 faces are supported")
    area = .5 * np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    if np.any(area <= 0):
        raise ValueError("Degenerate quadratic surface triangle")
    traces = [(faces, faces, 1.)]
    if master_faces is not None:
        master = np.asarray(master_faces)
        if (master.ndim != 2 or len(master) != len(faces) or master.shape[1] not in (3, 6)
                or not np.issubdtype(master.dtype, np.integer)
                or np.any(master < 0) or np.any(master >= len(nodes))):
            raise ValueError("Invalid quadratic master connectivity")
        if (np.intersect1d(faces, master).size
                or not np.allclose(nodes[master[:, :3]], xyz[:, :3], rtol=0., atol=1e-7)):
            raise ValueError("Quadratic master/slave faces must match geometrically with split node identities")
        if master.shape[1] == 6 and not np.allclose(nodes[master[:, 3:]], midpoint, rtol=0., atol=1e-8):
            raise ValueError("Only straight-sided TRI6 master faces are supported")
        traces += [(master, master, 1.), (faces, master, -1.), (master, faces, -1.)]
    pairs, blocks = [], []
    for rows, columns, sign in traces:
        mass = area[:, None, None] * triangle_mass(1., rows.shape[1], columns.shape[1])
        pairs.append(np.stack((
            np.broadcast_to(rows[:, :, None], mass.shape),
            np.broadcast_to(columns[:, None, :], mass.shape),
        ), axis=-1).reshape(-1, 2))
        blocks.append((sign * mass[..., None, None] * stiffness).reshape(-1, 3, 3))
    return np.concatenate(pairs), np.concatenate(blocks)


def compact_blocks(pairs: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum contributions by node identity without welding coincident surfaces."""
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    summed = np.zeros((len(unique), 3, 3))
    np.add.at(summed, inverse, blocks)
    keep = np.any(summed != 0., axis=(1, 2))
    return unique[keep], summed[keep]


def split_blocks(pairs: np.ndarray, blocks: np.ndarray, node_count: int) -> tuple:
    """Compact a symmetric operator into native point diagonals and upper links."""
    pairs, blocks = compact_blocks(pairs, blocks)
    if np.any(pairs < 0) or np.any(pairs >= node_count):
        raise ValueError("Surface operator references an invalid node identity")
    points, links = pairs[:, 0] == pairs[:, 1], pairs[:, 0] < pairs[:, 1]
    keys = pairs[:, 0] * node_count + pairs[:, 1]
    transpose = np.searchsorted(keys, pairs[:, 1] * node_count + pairs[:, 0])
    if (np.any(transpose >= len(keys))
            or not np.array_equal(pairs[transpose], pairs[:, ::-1])
            or not np.isfinite(blocks).all()
            or not np.allclose(blocks[transpose], blocks.transpose(0, 2, 1), rtol=1e-12, atol=1e-8)):
        raise ValueError("Surface operator has missing/nonsymmetric transpose blocks")
    return pairs[points, 0], pairs[links], blocks[points], blocks[links]


def discrete_entries(
    pairs: np.ndarray, blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    """Encode the global surface matrix as POI1 diagonals and SEG2 off-diagonals.

    K_T_L has a general symmetric 6x6 matrix here, NOT a conventional
    two-node axial spring. The assembled operator, not each off-diagonal
    contribution in isolation, is positive semidefinite.
    """
    pairs, blocks = compact_blocks(pairs, blocks)
    lookup = {tuple(pair): block for pair, block in zip(pairs.tolist(), blocks)}
    point_nodes, segment_nodes, point_entries, segment_entries = [], [], [], []
    for (first, second), block in lookup.items():
        if first == second:
            if not np.allclose(block, block.T, rtol=1e-12, atol=1e-8):
                raise ValueError("Nonsymmetric quadratic diagonal stiffness")
            point_nodes.append(first)
            point_entries.append({
                "group": f"QS_P_{len(point_entries):06d}",
                "matrix": block.tolist(),
            })
        elif first < second:
            other = lookup.get((second, first))
            if other is None or not np.allclose(block, other.T, rtol=1e-12, atol=1e-8):
                raise ValueError("Missing transpose quadratic surface block")
            matrix = np.zeros((6, 6))
            matrix[:3, 3:] = block
            matrix[3:, :3] = block.T
            segment_nodes.append((first, second))
            segment_entries.append({
                "group": f"QS_L_{len(segment_entries):06d}",
                "matrix": matrix.tolist(),
            })
    return (np.asarray(point_nodes, dtype=np.int64),
            np.asarray(segment_nodes, dtype=np.int64).reshape(-1, 2),
            point_entries, segment_entries)
