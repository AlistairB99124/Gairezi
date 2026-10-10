"""Deterministic conforming subdivision, retaining source node identities."""
from dataclasses import dataclass
from pathlib import Path

import basix.ufl
from dolfinx import mesh
from mpi4py import MPI
import numpy as np
import ufl

FACES = {
    4: ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
    5: ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
    6: ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
}


def triangles(face):
    face = list(face)
    start = face.index(min(face))
    face = face[start:] + face[:start]
    return [(face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1)]


def subdivide(points, cells, regions, boundaries):
    coordinates = points.tolist()
    face_centres = {}
    interpolation_parents = []
    for kind, cell in cells:
        if kind not in FACES:
            raise ValueError(f"Unsupported Gmsh volume type {kind}; no implicit conversion")
        for pattern in FACES[kind]:
            face = tuple(cell[i] for i in pattern)
            if len(face) != 4:
                continue
            xyz = points[list(face)]
            scale = np.linalg.norm(xyz - xyz[0], axis=1).max()
            warped = abs(np.linalg.det(xyz[1:] - xyz[:1])) > 1e-10 * scale**3
            key = tuple(sorted(face))
            if warped and key not in face_centres:
                face_centres[key] = len(coordinates)
                coordinates.append(xyz.mean(axis=0).tolist())
                interpolation_parents.append(key)

    def face_triangles(face):
        centre = face_centres.get(tuple(sorted(face)))
        if centre is None:
            return triangles(face)
        return [(face[i], face[(i + 1) % len(face)], centre) for i in range(len(face))]

    tetrahedra, parents = [], []
    for parent, (kind, cell) in enumerate(cells):
        centred = any(
            tuple(sorted(cell[i] for i in pattern)) in face_centres for pattern in FACES[kind]
        )
        if centred:
            anchor = len(coordinates)
            coordinates.append(points[list(cell)].mean(axis=0).tolist())
            interpolation_parents.append(tuple(cell))
            for pattern in FACES[kind]:
                face = [cell[i] for i in pattern]
                for triangle in face_triangles(face):
                    tetrahedra.append((anchor, *triangle))
                    parents.append(parent)
            continue
        anchor = min(cell)
        for pattern in FACES[kind]:
            face = [cell[i] for i in pattern]
            if anchor not in face:
                for triangle in triangles(face):
                    tetrahedra.append((anchor, *triangle))
                    parents.append(parent)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    coordinates = np.asarray(coordinates)
    xyz = coordinates[tetrahedra]
    determinants = np.linalg.det(xyz[:, 1:] - xyz[:, :1])
    if not np.all(np.isfinite(determinants)) or np.any(np.abs(determinants) <= 1e-14):
        raise ValueError("Subdivision contains non-finite or degenerate tetrahedra")
    negative = determinants < 0
    tetrahedra[negative, 1], tetrahedra[negative, 2] = (
        tetrahedra[negative, 2].copy(), tetrahedra[negative, 1].copy()
    )
    parent_ids = np.asarray(parents, dtype=np.int64)
    boundary_triangles, boundary_values = [], []
    for tag, face in boundaries:
        for triangle in face_triangles(face):
            boundary_triangles.append(sorted(triangle))
            boundary_values.append(tag)
    return (
        tetrahedra, np.asarray(regions, dtype=np.int32)[parent_ids], parent_ids,
        np.asarray(boundary_triangles, dtype=np.int64),
        np.asarray(boundary_values, dtype=np.int32), np.abs(determinants) / 6,
        coordinates, interpolation_parents,
    )


def source_cell_volumes(points, cells):
    # Exact integration of linear prism/trilinear hex geometry, before subdivision.
    result = np.zeros(len(cells))
    gauss = 1 / np.sqrt(3)
    signs = np.asarray([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)])
    for kind in FACES:
        indices = np.asarray([i for i, (cell_kind, _) in enumerate(cells) if cell_kind == kind], dtype=np.int64)
        block = np.asarray([cells[i][1] for i in indices], dtype=np.int64)
        if not len(block):
            continue
        xyz = points[block]
        if kind == 4:
            result[indices] = np.abs(np.linalg.det(xyz[:, 1:] - xyz[:, :1])) / 6
        elif kind == 5:
            for r in (-gauss, gauss):
                for s in (-gauss, gauss):
                    for t in (-gauss, gauss):
                        derivative = 0.125 * np.stack([
                            signs[:, 0] * (1 + signs[:, 1] * s) * (1 + signs[:, 2] * t),
                            signs[:, 1] * (1 + signs[:, 0] * r) * (1 + signs[:, 2] * t),
                            signs[:, 2] * (1 + signs[:, 0] * r) * (1 + signs[:, 1] * s),
                        ])
                        result[indices] += np.abs(np.linalg.det(np.einsum("ij,njk->nik", derivative, xyz)))
        else:
            for r, s in ((1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)):
                for t in (-gauss, gauss):
                    lam = np.asarray([1 - r - s, r, s])
                    derivative = np.zeros((3, 6))
                    derivative[:2, :3] = np.asarray([[-1, 1, 0], [-1, 0, 1]]) * (1 - t) / 2
                    derivative[:2, 3:] = np.asarray([[-1, 1, 0], [-1, 0, 1]]) * (1 + t) / 2
                    derivative[2, :3], derivative[2, 3:] = -lam / 2, lam / 2
                    result[indices] += np.abs(np.linalg.det(np.einsum("ij,njk->nik", derivative, xyz))) / 6
    return result


@dataclass
class ImportedDam:
    domain: mesh.Mesh
    cell_tags: mesh.MeshTags
    facet_tags: mesh.MeshTags
    points: np.ndarray
    tetrahedra: np.ndarray
    regions: np.ndarray
    parent_ids: np.ndarray
    volumes: np.ndarray
    report: dict
    interpolation_parents: list[tuple[int, ...]]


def import_structure(structure, maximum_volume_change: float) -> ImportedDam:
    points = np.asarray(structure.nodes, dtype=float)
    tets, regions, parents, boundary, values, volumes, coordinates, interpolation = subdivide(
        points, structure.cells, structure.region_ids, structure.boundaries
    )
    original_cells = source_cell_volumes(points, structure.cells)
    original_volume = float(original_cells.sum())
    if not np.all(np.isfinite(original_cells)) or np.any(original_cells <= 0):
        raise ValueError("Source contains non-positive or non-finite cell volumes")
    converted_cells = np.bincount(parents, weights=volumes, minlength=len(original_cells))
    maximum_cell_change = float(np.max(np.abs(converted_cells - original_cells) / original_cells))
    volume_change = abs(float(volumes.sum()) - original_volume) / original_volume
    if max(volume_change, maximum_cell_change) > maximum_volume_change:
        raise ValueError(
            f"Tetrahedral volume change (total={volume_change}, worst cell={maximum_cell_change}) "
            f"exceeds {maximum_volume_change}"
        )
    element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    domain = mesh.create_mesh(MPI.COMM_WORLD, tets, ufl.Mesh(element), coordinates)
    domain.topology.create_connectivity(2, 0)
    domain.topology.create_connectivity(2, 3)
    domain.topology.create_connectivity(0, 3)
    original_vertices = domain.geometry.input_global_indices[
        mesh.entities_to_geometry(domain, 0, np.arange(domain.topology.index_map(0).size_local, dtype=np.int32)).ravel()
    ]
    connectivity = domain.topology.connectivity(2, 0)
    facets = mesh.exterior_facet_indices(domain.topology)
    facet_vertices = np.sort(original_vertices[connectivity.array.reshape(-1, 3)[facets]], axis=1)
    lookup = {tuple(face): int(tag) for face, tag in zip(boundary, values)}
    if len(lookup) != len(boundary):
        raise ValueError("Source boundary contains duplicate tagged triangles")
    imported_values = []
    for face in facet_vertices:
        if tuple(face) not in lookup:
            raise ValueError(f"Untagged exterior triangle after subdivision: {face}")
        imported_values.append(lookup.pop(tuple(face)))
    if lookup:
        raise ValueError(f"{len(lookup)} source boundary triangles did not map to exterior facets")
    cell_ids = domain.topology.original_cell_index
    cell_tags = mesh.meshtags(domain, 3, np.arange(len(cell_ids), dtype=np.int32), regions[cell_ids])
    facet_tags = mesh.meshtags(domain, 2, facets, np.asarray(imported_values, dtype=np.int32))
    report = {
        "source_nodes": len(points), "imported_nodes": domain.geometry.x.shape[0],
        "added_nodes": len(interpolation),
        "source_cells": len(structure.cells), "tetrahedra": len(tets),
        "source_volume_m3": original_volume, "tetrahedral_volume_m3": float(volumes.sum()),
        "relative_volume_change": volume_change,
        "maximum_parent_cell_relative_volume_change": maximum_cell_change,
        "minimum_tetrahedron_volume_m3": float(volumes.min()),
        "bounds_m": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "facet_triangle_counts": {str(tag): int(np.count_nonzero(values == tag)) for tag in np.unique(values)},
        "region_tetrahedron_counts": {str(tag): int(np.count_nonzero(regions == tag)) for tag in np.unique(regions)},
        "conversion": "pulling triangulation with shared centres on warped faces and affected cell centres",
    }
    if len(coordinates) != report["imported_nodes"]:
        raise ValueError("Mesh import changed the converted node count")
    if not np.array_equal(coordinates[:len(points)], points):
        raise ValueError("Subdivision changed original source nodes")
    return ImportedDam(domain, cell_tags, facet_tags, coordinates, tets, regions, parents, volumes, report, interpolation)
