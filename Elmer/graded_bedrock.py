"""Conforming 3-D rock tetrahedra with a locked plinth seat and distance grading."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import warnings

import gmsh
import numpy as np

from .bedrock_preview import (
    BANK_MARGIN_M, STREAM_MARGIN_M, seated_bedrock_geometry,
)


@dataclass
class RockMesh:
    points: np.ndarray
    blocks: dict[int, np.ndarray]
    top_faces: np.ndarray
    bottom_faces: np.ndarray
    report: dict


def build_graded_rock(
    config: dict, concrete: np.ndarray, foundation_faces: list[tuple[int, ...]],
    stream_margin_m: float | None = None, bank_margin_m: float | None = None,
    bottom_z_m: float | None = None,
) -> RockMesh:
    rock = config["geometry"]["bedrock"]
    domain = rock.get("domain", {})
    settings = rock["mesh"]
    columns, wedges, report = seated_bedrock_geometry(
        rock["contours"], concrete, foundation_faces,
        stream_margin_m if stream_margin_m is not None else domain.get("stream_margin_m", STREAM_MARGIN_M),
        bank_margin_m if bank_margin_m is not None else domain.get("bank_margin_m", BANK_MARGIN_M),
        bottom_z_m=bottom_z_m if bottom_z_m is not None else domain.get("bottom_z_m", -31.),
        element_size_m=settings["near_size_m"], resample_surface=False,
    )
    count = len(columns) // 2
    triangles = wedges[:, :3]
    seat_count = report["seat_triangles"]
    edge_counts = Counter(
        tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
        for tri in triangles for i in range(3)
    )
    perimeter = [edge for edge, total in edge_counts.items() if total == 1]
    if any(total not in (1, 2) for total in edge_counts.values()):
        raise ValueError("Non-manifold rock top")
    # Retain the faceted terrain exactly; only the seat triangles are mesh-locked.
    facets = [tuple(int(n) + count for n in tri) for tri in triangles]
    top_count = len(facets)
    # A coarse bottom fan is planar; its interior edges do not force fine columns.
    boundary_nodes = sorted({n for edge in perimeter for n in edge})
    centre = columns[boundary_nodes].mean(axis=0)
    shell_points = np.vstack((columns, centre))
    bottom_start = len(facets)
    facets.extend((a, b, 2 * count) for a, b in perimeter)
    bottom_end = len(facets)
    for a, b in perimeter:
        facets.extend(((a, b, b + count), (a, b + count, a + count)))

    if gmsh.isInitialized():
        raise RuntimeError("Rock mesher requires its own Gmsh session")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add("graded_bedrock")
        used = sorted({n for face in facets for n in face})
        vertices = {n: gmsh.model.geo.addPoint(*shell_points[n]) for n in used}
        curves = {}
        seat_edges = {
            tuple(sorted((face[i], face[(i + 1) % 3])))
            for face in facets[:seat_count] for i in range(3)
        }
        surfaces = []
        for face in facets:
            loop = []
            for a, b in zip(face, (*face[1:], face[0])):
                key = tuple(sorted((a, b)))
                if key not in curves:
                    curves[key] = gmsh.model.geo.addLine(vertices[key[0]], vertices[key[1]])
                loop.append(curves[key] if a < b else -curves[key])
            wire = gmsh.model.geo.addCurveLoop(loop)
            surfaces.append(gmsh.model.geo.addPlaneSurface([wire]))
        gmsh.model.geo.synchronize()
        for edge in seat_edges:
            gmsh.model.mesh.setTransfiniteCurve(curves[edge], 2)
        for face, surface in zip(facets[:seat_count], surfaces[:seat_count]):
            gmsh.model.mesh.setTransfiniteSurface(surface, cornerTags=[vertices[n] for n in face])
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", surfaces[:seat_count])
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 2)
        threshold = gmsh.model.mesh.field.add("Threshold")
        for key, value in {
            "InField": distance, "SizeMin": settings["near_size_m"],
            "SizeMax": settings["far_size_m"], "DistMin": settings["near_distance_m"],
            "DistMax": settings["far_distance_m"],
        }.items():
            gmsh.model.mesh.field.setNumber(threshold, key, value)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        for key, value in {
            "Mesh.MeshSizeFromPoints": 0, "Mesh.MeshSizeFromCurvature": 0,
            "Mesh.MeshSizeExtendFromBoundary": 0, "Mesh.MeshSizeMin": settings["near_size_m"],
            "Mesh.MeshSizeMax": settings["far_size_m"], "Mesh.Algorithm3D": 10,
            "Mesh.ElementOrder": 1,
            "Mesh.OptimizeNetgen": 1,
            "Mesh.RandomSeed": 1,
            "Mesh.MeshOnlyEmpty": 1,
        }.items():
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.generate(2)
        # The volume sees a single closed discrete shell, not artificial CAD
        # seams between coplanar terrain/seat facets. Keep every shell triangle.
        closed_shell = gmsh.model.addDiscreteEntity(2)
        shell_nodes = np.concatenate([
            gmsh.model.mesh.getElements(2, tag)[2][0] for tag in surfaces
        ])
        gmsh.model.mesh.addElementsByType(closed_shell, 2, [], shell_nodes)
        shell = gmsh.model.geo.addSurfaceLoop([closed_shell])
        volume = gmsh.model.geo.addVolume([shell])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(node_tags)
        sorted_tags = node_tags[order]
        points = np.asarray(coordinates).reshape(-1, 3)[order]

        def connectivity(dim, tags, kind, width):
            parts = []
            for tag in tags:
                types, _, nodes = gmsh.model.mesh.getElements(dim, tag)
                for element_type, members in zip(types, nodes):
                    if element_type != kind:
                        raise ValueError(f"Unexpected Gmsh element type {element_type}")
                    parts.append(np.searchsorted(sorted_tags, members).reshape(-1, width))
            return np.vstack(parts)

        tetrahedra = connectivity(3, [volume], 4, 4)
        surface_faces = connectivity(2, surfaces[:top_count], 2, 3)
        bottom_faces = connectivity(2, surfaces[bottom_start:bottom_end], 2, 3)
        shell_faces = connectivity(2, surfaces, 2, 3)
        actual_seat = connectivity(2, surfaces[:seat_count], 2, 3)
        expected = {
            tuple(sorted(tuple(np.round(shell_points[n], 9)) for n in face))
            for face in facets[:seat_count]
        }
        actual = {
            tuple(sorted(tuple(np.round(points[n], 9)) for n in face))
            for face in actual_seat
        }
        if actual != expected or len(actual_seat) != seat_count:
            raise ValueError(
                f"Gmsh changed the locked plinth/rock interface: expected {seat_count}, "
                f"got {len(actual_seat)}; missing {list(expected - actual)[:1]}"
            )
        expected_points = {
            tuple(np.round(shell_points[n], 9)): shell_points[n]
            for face in facets[:seat_count] for n in face
        }
        coordinate_error = max(
            np.linalg.norm(points[n] - expected_points[tuple(np.round(points[n], 9))])
            for n in np.unique(actual_seat)
        )
        p = points[tetrahedra]
        signed = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                           p[:, 3] - p[:, 0]) / 6.
        if np.any(signed <= 0):
            raise ValueError("Graded rock contains non-positive tetrahedra")
        if not np.isclose(signed.sum(), report["rock_volume_m3"], rtol=1e-8):
            raise ValueError("Graded mesh changed the rock volume")
        audit_tetrahedral_boundary(tetrahedra, shell_faces)
        qualities = gmsh.model.mesh.getElementQualities(
            gmsh.model.mesh.getElements(3, volume)[1][0], "minSICN",
        )
        if np.min(qualities) <= 1e-8:
            raise ValueError("Graded rock contains numerically degenerate tetrahedra")
        report.update(
            mesh_method="conforming_distance_graded_tetrahedra",
            mesh_grading=dict(settings), bedrock_cells=len(tetrahedra),
            minimum_cell_volume_m3=float(signed.min()),
            minimum_scaled_jacobian=float(np.min(qualities)),
            seat_faces_verified=len(actual_seat), surface_triangles=len(surface_faces),
            seat_max_coordinate_difference_m=max(
                report.get("seat_max_coordinate_difference_m", 0.), float(coordinate_error),
            ),
            mesher_version=gmsh.__version__, vertical_layers=None,
            surface_edge_size_policy="Unstructured conforming triangles; locked plinth seat",
            surface_row_axis=None,
            maximum_exterior_surface_edge_m=float(max(
                np.linalg.norm(points[surface_faces[:, i]] - points[surface_faces[:, (i + 1) % 3]],
                               axis=1).max() for i in range(3)
            )),
            conformity_audit="Every interior face paired; exterior equals the closed CAD shell",
            spacing_samples=spacing_summary(points, tetrahedra, actual_seat, settings),
        )
        if report["minimum_scaled_jacobian"] < .01:
            warnings.warn(
                f"Rock minimum scaled Jacobian is {report['minimum_scaled_jacobian']:.6g}; "
                "inspect constrained-boundary element quality before engineering use",
                stacklevel=2,
            )
        return RockMesh(points, {4: tetrahedra}, surface_faces, bottom_faces, report)
    finally:
        gmsh.finalize()


def audit_tetrahedral_boundary(tetrahedra: np.ndarray, shell: np.ndarray) -> None:
    faces = np.sort(np.concatenate([
        tetrahedra[:, indices] for indices in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    ]), axis=1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    expected = np.sort(shell, axis=1)
    expected = expected[np.lexsort(expected.T[::-1])]
    if np.any(counts > 2) or not np.array_equal(unique[counts == 1], expected):
        raise ValueError("Rock has non-manifold, unpaired or hanging volume faces")
    edges, counts = np.unique(np.sort(np.concatenate([
        shell[:, indices] for indices in ((0, 1), (1, 2), (2, 0))
    ]), axis=1), axis=0, return_counts=True)
    if not len(edges) or np.any(counts != 2):
        raise ValueError("Rock exterior is not a closed two-manifold")


def spacing_summary(points, tetrahedra, seat, settings) -> dict:
    """Bounded deterministic sample; distance to seat vertices approximates its surface."""
    selected = tetrahedra[::max(1, len(tetrahedra) // 3000)]
    centres = points[selected].mean(axis=1)
    reference = points[np.unique(seat)]
    distances = np.concatenate([
        np.sqrt(np.min(np.sum((chunk[:, None] - reference[None]) ** 2, axis=2), axis=1))
        for chunk in np.array_split(centres, max(1, len(centres) // 100))
    ])
    lengths = np.array([
        np.linalg.norm(points[selected[:, a]] - points[selected[:, b]], axis=1)
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    ]).T
    result = {
        "sampled_cells": len(selected),
        "distance_method": "Cell centroid to nearest locked seat vertex (not exact surface distance)",
    }
    for name, mask in (
        ("near", distances <= settings["near_distance_m"]),
        ("far", distances >= settings["far_distance_m"]),
    ):
        values = lengths[mask]
        result[name] = {
            "sampled_cells": int(mask.sum()),
            "median_edge_m": float(np.median(values)) if values.size else None,
            "edge_10th_percentile_m": float(np.quantile(values, .1)) if values.size else None,
            "edge_90th_percentile_m": float(np.quantile(values, .9)) if values.size else None,
        }
    return result
