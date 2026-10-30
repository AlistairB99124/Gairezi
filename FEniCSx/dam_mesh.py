"""Deterministic conforming subdivision, retaining source node identities."""
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import basix.ufl
from dolfinx import mesh
from mpi4py import MPI
import numpy as np
import ufl

if TYPE_CHECKING:
    if __package__:
        from .fracture_faces import PairedCrackFaces
    else:
        from fracture_faces import PairedCrackFaces

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


def source_cell_tags(imported: ImportedDam) -> mesh.MeshTags:
    """Label native cells by source volume row, never by coordinates or local ID.

    The caller must bind the imported source to the validated catalog hash.
    Preserve these labels with transfer_meshtag at every refinement; do not
    reinterpret a refined mesh's original_cell_index as a source row.
    """
    domain = imported.domain
    if domain.comm.size != 1:
        raise ValueError("Painted source-cell transfer currently requires one MPI process")
    parents = np.asarray(imported.parent_ids)
    original = np.asarray(domain.topology.original_cell_index)
    count = imported.report["source_cells"]
    cell_map = domain.topology.index_map(domain.topology.dim)
    size = cell_map.size_local + cell_map.num_ghosts
    if (type(count) is not int or count <= 0 or count > np.iinfo(np.int32).max
            or parents.dtype.kind not in "iu" or parents.shape != (len(imported.tetrahedra),)
            or np.any(parents < 0) or np.any(parents >= count)
            or len(np.unique(parents)) != count
            or original.dtype.kind not in "iu" or original.shape != (size,)
            or np.any(original < 0) or np.any(original >= len(parents))):
        raise ValueError("Incomplete or invalid source-cell provenance after import")
    tags = mesh.meshtags(domain, domain.topology.dim, np.arange(size, dtype=np.int32),
                         parents[original].astype(np.int32))
    tags.name = "source_volume_row"
    return tags


def selected_wall_cells(source_tags: mesh.MeshTags, region_tags: mesh.MeshTags,
                        affected_source_rows) -> np.ndarray:
    """Make a local mask from complete source/region tags on the same serial mesh."""
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("Painted source-cell selection currently requires one MPI process")
    if (source_tags.dim != 3 or region_tags.dim != 3
            or source_tags.topology is not region_tags.topology):
        raise ValueError("Source and region tags must describe the same tetrahedral mesh")
    cell_map = source_tags.topology.index_map(3)
    size = cell_map.size_local + cell_map.num_ghosts
    indices = np.arange(size, dtype=np.int32)
    if (not np.array_equal(source_tags.indices, indices)
            or not np.array_equal(region_tags.indices, indices)
            or source_tags.values.dtype.kind not in "iu" or np.any(source_tags.values < 0)):
        raise ValueError("Complete local source/region cell tags are required")
    selected = np.asarray(affected_source_rows)
    if (selected.ndim != 1 or selected.dtype.kind not in "iu" or not len(selected)
            or np.any(selected < 0) or len(np.unique(selected)) != len(selected)
            or not np.all(np.isin(selected, source_tags.values))):
        raise ValueError("Affected source rows must be nonempty unique known integer IDs")
    mask = np.isin(source_tags.values, selected)
    if np.any(region_tags.values[mask] != 1):
        raise ValueError("Tensile-strength loss may only select wall concrete (region 1)")
    return mask


def compact_unused_nodes(structure):
    """Remove volume-unused nodes without merging coincident interface identities."""
    from dataclasses import replace
    used = np.unique(np.asarray([cell for _, cell in structure.cells]).ravel())
    if len(used) == len(structure.nodes):
        return structure
    mapping = np.full(len(structure.nodes), -1, dtype=np.int64)
    mapping[used] = np.arange(len(used))
    boundaries = [(tag, tuple(int(mapping[n]) for n in face)) for tag, face in structure.boundaries]
    if any(min(face) < 0 for _, face in boundaries):
        raise ValueError("A boundary references a node outside the volume mesh")
    return replace(structure, nodes=[tuple(structure.nodes[n]) for n in used],
                   cells=[(kind, tuple(int(mapping[n]) for n in cell)) for kind, cell in structure.cells],
                   boundaries=boundaries)


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
    cell_ids = domain.topology.original_cell_index
    facet_cells = domain.topology.connectivity(2, 3)
    for facet, face in zip(facets, facet_vertices):
        if tuple(face) not in lookup:
            owner = facet_cells.links(int(facet))
            if (getattr(structure, "three_body_config", None) is not None and len(owner) == 1
                    and regions[cell_ids[owner[0]]] in (2, 3)):
                imported_values.append(7)
                continue
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


@dataclass
class PreparedWallCandidates:
    imported: ImportedDam
    faces: "PairedCrackFaces"
    parent_node_ids: np.ndarray
    source_triangles: np.ndarray
    report: dict


def wall_candidate_triangles(imported, marker_source_edges):
    """Read actual serial interior wall facets without Python cell-owner maps."""
    domain = imported.domain
    if domain.comm.size != 1:
        raise ValueError("Wall-candidate inventory requires one MPI process")
    domain.topology.create_connectivity(2, 3)
    adjacency = domain.topology.connectivity(2, 3)
    counts = np.diff(adjacency.offsets)
    if np.any(counts > 2):
        raise ValueError("Nonmanifold tetrahedral face has more than two owners")
    internal = np.flatnonzero(counts == 2).astype(np.int32)
    starts = adjacency.offsets[internal]
    owners = adjacency.array[starts[:, None] + np.arange(2)]
    regions = np.zeros(domain.topology.index_map(3).size_local, dtype=np.int32)
    regions[imported.cell_tags.indices] = imported.cell_tags.values
    wall = np.all(regions[owners] == 1, axis=1)
    internal, owners = internal[wall], owners[wall]
    if not len(internal):
        raise ValueError("No interior wall-concrete growth facets exist")
    geometry = mesh.entities_to_geometry(domain, 2, internal)
    triangles = np.sort(domain.geometry.input_global_indices[geometry], axis=1)
    points = imported.points[triangles]
    centres = domain.geometry.x[domain.geometry.dofmaps[0][owners]].mean(axis=2)
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    sides = np.einsum("fci,fi->fc", centres - points[:, None, 0], normals)
    if np.any(sides[:, 0] * sides[:, 1] >= 0):
        raise ValueError("Interior wall candidate owners must lie on opposite sides")
    triangles = triangles[np.lexsort(triangles[:, ::-1].T)]
    edges = np.asarray(marker_source_edges)
    if (edges.dtype.kind not in "iu" or edges.ndim != 2 or edges.shape[1] != 2 or not len(edges)
            or np.any(edges < 0) or np.any(edges >= len(imported.points))
            or np.any(edges[:, 0] == edges[:, 1])
            or len(np.unique(np.sort(edges, axis=1), axis=0)) != len(edges)):
        raise ValueError("Markers must be unique source-node edge pairs")
    for edge in edges:
        if not np.any(np.sum(np.isin(triangles, edge), axis=1) == 2):
            raise ValueError("Every marker must touch an actual interior wall-concrete candidate facet")
    return triangles


def prepare_wall_candidates(structure, marker_source_edges, maximum_volume_change) -> PreparedWallCandidates:
    """Split actual interior wall facets, preserving source rows/exterior tags.

    All candidates start geometrically closed. Their bond/contact state is
    subsequently decided by the energy trial, not by this preparation. This
    experimental all-facet preparation is serial and not resource-validated
    for the full dam; it does not choose unrestricted physical crack paths.
    """
    from copy import copy
    from dolfinx import fem

    if __package__:
        from .fracture_faces import bulk_face_mapping
        from .fracture_topology import split_interior_wall_faces
    else:
        from fracture_faces import bulk_face_mapping
        from fracture_topology import split_interior_wall_faces
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("Wall-candidate preparation requires one MPI process")
    if (isinstance(maximum_volume_change, bool)
            or not isinstance(maximum_volume_change, (int, float))
            or not np.isfinite(maximum_volume_change) or not 0 <= maximum_volume_change < 1):
        raise ValueError("maximum_volume_change must be finite and in [0, 1)")
    original = import_structure(structure, maximum_volume_change)
    candidates = wall_candidate_triangles(original, marker_source_edges)
    split = split_interior_wall_faces(original.points, original.tetrahedra, original.regions, candidates)
    # Export inherited exterior identities from native connectivity, including
    # canonical default rock boundaries; never infer labels from coordinates.
    domain = original.domain
    facet_ids = original.facet_tags.indices
    geometry = mesh.entities_to_geometry(domain, 2, facet_ids)
    source_boundary = domain.geometry.input_global_indices[geometry]
    boundary, values = split.transfer_boundary_faces(source_boundary, original.facet_tags.values)
    prepared_structure = copy(structure)
    prepared_structure.nodes = split.coordinates_m
    prepared_structure.cells = [(4, tuple(map(int, cell))) for cell in split.cells]
    prepared_structure.region_ids = split.region_ids.tolist()
    prepared_structure.boundaries = (
        [(int(tag), tuple(map(int, triangle))) for triangle, tag in zip(boundary, values)]
        + [(7, tuple(map(int, triangle))) for triangle in np.vstack((split.faces.minus, split.faces.plus))]
    )
    prepared = import_structure(prepared_structure, maximum_volume_change)
    if not np.array_equal(prepared.points, split.coordinates_m):
        raise ValueError("Candidate import changed split source-node coordinates or identity")
    if not np.array_equal(np.sort(prepared.tetrahedra, axis=1), np.sort(split.cells, axis=1)):
        raise ValueError("Candidate import changed source tetrahedral row identity")
    # Reimport parents refer to source tetrahedra; recover original volume rows
    # so painting mixed-element source cells still selects every child.
    prepared.parent_ids = original.parent_ids[prepared.parent_ids]
    prepared.report["source_cells"] = original.report["source_cells"]
    source_cell_tags(prepared)
    space = fem.functionspace(prepared.domain, ("Lagrange", 1, (3,)))
    bulk_face_mapping(space, split.faces)
    report = {
        "status": "prepared_experimental_wall_facet_candidates",
        "solver_ready": False, "engineering_validated": False,
        "candidate_faces": len(candidates), "original_nodes": len(original.points),
        "prepared_nodes": len(prepared.points), "source_cells": original.report["source_cells"],
        "tetrahedra": len(prepared.tetrahedra),
        "original_import": original.report,
        "limitations": "All interior wall mesh facets only; zero initial gap, no growth decision, no full-dam resource validation or API submission.",
    }
    return PreparedWallCandidates(prepared, split.faces, split.parent_node_ids, candidates, report)
