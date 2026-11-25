"""Aster-owned topology helpers for the native mixed concrete mesh.

This module deliberately has no FEniCSx dependency. Production wall,
foundation, and bedrock volumes are native HEXA8; wall crack interfaces are
QUAD4 faces. Legacy TETRA4 fixtures remain readable.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

WALL = 1
BEDROCK = 2
FOUNDATION = 3

HEX_FACES = (
    (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
    (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
)
TET_FACES = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))


@dataclass(frozen=True)
class NativeSplit:
    coordinates_m: np.ndarray
    parent_node_ids: np.ndarray
    source_faces: np.ndarray
    adjacent_cells: np.ndarray
    tied_source_faces: np.ndarray

    @property
    def faces(self):
        return self


def has_native_wall(arrays: dict) -> bool:
    return "hexa8_wall" in arrays and len(arrays["hexa8_wall"]) > 0


def wall_faces(cells: np.ndarray) -> np.ndarray:
    rows = np.asarray(cells, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 8:
        raise ValueError("Native wall topology requires HEXA8 cells")
    faces = rows[:, HEX_FACES].reshape(-1, 4)
    return np.sort(faces, axis=1)


def interior_wall_faces(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique interior QUAD4 faces and their two wall-cell owners."""
    faces = wall_faces(cells)
    owners = np.repeat(np.arange(len(cells), dtype=np.int64), len(HEX_FACES))
    unique, inverse, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    interior = counts == 2
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    pairs = np.stack((owners[order][starts[interior]], owners[order][starts[interior] + 1]), axis=1)
    return unique[interior], pairs


def interior_facet_normals(coordinates: np.ndarray, faces: np.ndarray) -> np.ndarray:
    points = np.asarray(coordinates, dtype=float)[np.asarray(faces, dtype=np.int64)]
    if points.size == 0:
        return np.empty((0, 3))
    centred = points - points.mean(axis=1)[:, None, :]
    normals = np.linalg.svd(centred, full_matrices=False)[2][:, -1]
    magnitude = np.linalg.norm(normals, axis=1)
    if np.any(magnitude <= 0):
        raise ValueError("Interior wall QUAD4 facet is degenerate")
    return normals / magnitude[:, None]


def facet_normal_stress(coordinates: np.ndarray, faces: np.ndarray, owners: np.ndarray,
                        cell_stress: np.ndarray) -> np.ndarray:
    normal = interior_facet_normals(coordinates, faces)
    stress = np.asarray(cell_stress, dtype=float)
    side = np.stack([
        np.einsum("fi,fij,fj->f", normal, stress[owners[:, side]], normal)
        for side in (0, 1)
    ], axis=1)
    return side.max(axis=1)


def candidate_region(cells: np.ndarray, selected: np.ndarray, rings: int) -> np.ndarray:
    wall = np.asarray(cells, dtype=np.int64)
    member = np.zeros(len(wall), dtype=bool)
    member[np.asarray(selected, dtype=np.int64)] = True
    for _ in range(rings):
        nodes = np.unique(wall[member])
        member |= np.isin(wall, nodes).any(axis=1)
    return member


def candidate_faces(cells: np.ndarray, member: np.ndarray) -> np.ndarray:
    faces, owners = interior_wall_faces(cells)
    return faces[np.asarray(member, dtype=bool)[owners].all(axis=1)]


def split_wall_faces(
    coordinates: np.ndarray,
    wall: np.ndarray,
    faces: np.ndarray,
) -> NativeSplit:
    """Split copies consistently across a node star; retain crack tips.

    The returned provenance is sufficient to build the paired QUAD8 joints
    directly in ``automatic_cohesive.split_for_joints``.  Faces are not
    triangulated.
    """
    points = np.asarray(coordinates, dtype=float)
    cells = np.asarray(wall, dtype=np.int64)
    cuts = np.sort(np.asarray(faces, dtype=np.int64).reshape(-1, 4), axis=1)
    inventory, owners = interior_wall_faces(cells)
    lookup = {tuple(face): pair for face, pair in zip(inventory.tolist(), owners.tolist())}
    adjacent = []
    for face in cuts.tolist():
        if tuple(face) not in lookup:
            raise ValueError("Cut facet is not a native HEXA8 interior wall face")
        adjacent.append(lookup[tuple(face)])
    adjacent = np.asarray(adjacent, dtype=np.int64).reshape(-1, 2)
    # This preparatory split records the topology and exact owners.  The Aster
    # native split below performs copy assignment while preserving node stars.
    return NativeSplit(
        points,
        np.arange(len(points), dtype=np.int64),
        cuts,
        adjacent,
        np.empty((0, 4), dtype=np.int64),
    )


def native_screening(
    arrays: dict,
    stress: np.ndarray,
    source_rows: np.ndarray,
    tensile_strength_pa: float,
    relative_tolerance: float = 0.0,
    max_damage_count: int | None = None,
) -> dict:
    """Initial wall screening from native Aster cell tensors."""
    wall = np.asarray(arrays["hexa8_wall"], dtype=np.int64)
    stress = np.asarray(stress, dtype=float)
    if stress.shape != (len(wall), 3, 3):
        raise ValueError("Native HEXA8 screening requires one stress tensor per wall cell")
    principal = np.linalg.eigvalsh(0.5 * (stress + stress.transpose(0, 2, 1)))[:, -1]
    faces, owners = interior_wall_faces(wall)
    facet = facet_normal_stress(arrays["nodes"], faces, owners, stress)
    threshold = float(tensile_strength_pa) * (1.0 - float(relative_tolerance))
    critical = facet >= threshold
    selected = np.unique(owners[critical]) if critical.any() else np.empty(0, dtype=np.int64)
    if max_damage_count is not None and len(selected) > max_damage_count:
        score = principal[selected]
        selected = selected[np.argsort(-score, kind="stable")[:int(max_damage_count)]]
    peak = int(np.argmax(principal))
    return {
        "maximum_principal_tensile_stress_pa": float(principal[peak]),
        "peak_location_m": np.asarray(arrays["nodes"])[wall[peak]].mean(0).tolist(),
        "peak_source_row": int(source_rows[peak]),
        "overstressed_source_rows": np.asarray(source_rows)[principal > threshold].astype(int).tolist(),
        "selection": {
            "enabled": True,
            "rule": "native Aster HEXA8 cell principal stress and one-sided QUAD4 facet normal stress",
            "selected_source_rows": np.unique(np.asarray(source_rows)[selected]).astype(int).tolist(),
            "selected_cells": int(len(selected)),
            "max_damage_count": max_damage_count,
        },
        "facet_check": {
            "criterion": "one-sided: max over the two native wall cells of n.sigma.n",
            "interior_wall_facets": int(len(faces)),
            "critical_facets": int(critical.sum()),
            "peak_facet_normal_stress_pa": float(facet.max(initial=0.0)),
            "peak_ratio_to_strength": float(facet.max(initial=0.0) / tensile_strength_pa),
            "initiates": bool(critical.any()),
        },
        "_cell_principal_stress_pa": principal,
        "_selected_indices": selected,
        "_faces": faces,
        "_owners": owners,
    }
