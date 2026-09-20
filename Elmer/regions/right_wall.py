"""Generate the standalone uniform right wall from the transition to the abutment."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from regions.center_wall import write_gmsh, write_vtu
from regions.plinth import _monotone_values, load_contours
from regions.right_wedged_wall_transition import WEDGE_END_CHAINAGE_M, WEDGE_HEIGHT_M, transition_end_chainage
from regions.uniform_wall import (
    DOWNSTREAM_WALL_RADIUS_M,
    UPSTREAM_RADIUS_M,
    UniformWallMesh,
    build_uniform_wall,
)


def build_right_wall(root: Path) -> UniformWallMesh:
    contours = load_contours(root / "Data" / "plinth.json")
    reference_base_z_m = _monotone_values(contours, "plinth_z_m", WEDGE_END_CHAINAGE_M)
    start_chainage_m = transition_end_chainage(contours, reference_base_z_m + WEDGE_HEIGHT_M)
    end_chainage_m = contours[-1].chainage_m
    anchors_m = [point.chainage_m for point in contours]
    return build_uniform_wall(root, start_chainage_m, end_chainage_m, reference_base_z_m, anchors_m)


def audit_right_wall(mesh: UniformWallMesh) -> dict[str, object]:
    element_counts = Counter(element_type for element_type, _ in mesh.cells)
    boundary_counts = Counter(boundary_id for boundary_id, _ in mesh.boundaries)
    expected_boundaries = {1, 2, 3, 4, 5}
    if set(boundary_counts) != expected_boundaries:
        raise ValueError(f"Missing right-wall boundaries: found {sorted(boundary_counts)}")
    return {
        "nodes": len(mesh.nodes),
        "cells": len(mesh.cells),
        "hexahedra": element_counts[5],
        "triangular_prisms": element_counts[6],
        "boundary_faces": dict(sorted(boundary_counts.items())),
        "chainage_m": [mesh.chainages_m[0], mesh.chainages_m[-1]],
        "plinth_z_m": [mesh.base_levels_m[0], mesh.base_levels_m[-1]],
        "wall_radius_m": [DOWNSTREAM_WALL_RADIUS_M, UPSTREAM_RADIUS_M],
        "wall_thickness_m": UPSTREAM_RADIUS_M - DOWNSTREAM_WALL_RADIUS_M,
        "element_size_m": mesh.element_size_m,
    }


__all__ = ["audit_right_wall", "build_right_wall", "write_gmsh", "write_vtu"]
