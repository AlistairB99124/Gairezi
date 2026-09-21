"""Generate the wedge-free transition beyond the right wedged wall."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from regions.center_wall import write_gmsh, write_vtu
from regions.left_wedged_wall_transition import (
    FULL_WEDGE_HEIGHT_M,
    LeftWedgedWallTransitionMesh,
    build_wedged_wall_transition,
)
from regions.left_wedged_wall import WEDGE_SHELF_WIDTH_M
from regions.plinth import _monotone_values, load_contours
from regions.uniform_wall import (
    DOWNSTREAM_WALL_RADIUS_M,
    UPSTREAM_RADIUS_M,
)


WEDGE_END_CHAINAGE_M = 140.37597894015022
WEDGE_HEIGHT_M = FULL_WEDGE_HEIGHT_M


def transition_end_chainage(contours, target_z_m: float) -> float:
    lower_chainage_m = WEDGE_END_CHAINAGE_M
    upper_chainage_m = next(point.chainage_m for point in contours if point.chainage_m > lower_chainage_m and point.plinth_z_m >= target_z_m)
    for _ in range(80):
        midpoint_m = 0.5 * (lower_chainage_m + upper_chainage_m)
        if _monotone_values(contours, "plinth_z_m", midpoint_m) < target_z_m:
            lower_chainage_m = midpoint_m
        else:
            upper_chainage_m = midpoint_m
    return 0.5 * (lower_chainage_m + upper_chainage_m)


def build_right_wedged_wall_transition(root: Path) -> LeftWedgedWallTransitionMesh:
    contours = load_contours(root / "Data" / "plinth.json")
    reference_base_z_m = _monotone_values(contours, "plinth_z_m", WEDGE_END_CHAINAGE_M)
    end_chainage_m = transition_end_chainage(contours, reference_base_z_m + WEDGE_HEIGHT_M)
    return build_wedged_wall_transition(root, WEDGE_END_CHAINAGE_M, end_chainage_m)


def audit_right_wedged_wall_transition(mesh: LeftWedgedWallTransitionMesh) -> dict[str, object]:
    element_counts = Counter(element_type for element_type, _ in mesh.cells)
    boundary_counts = Counter(boundary_id for boundary_id, _ in mesh.boundaries)
    if set(boundary_counts) != set(range(1, 7)):
        raise ValueError(f"Missing right transition boundaries: found {sorted(boundary_counts)}")
    return {
        "nodes": len(mesh.nodes),
        "cells": len(mesh.cells),
        "tetrahedra": element_counts[4],
        "hexahedra": element_counts[5],
        "triangular_prisms": element_counts[6],
        "boundary_faces": dict(sorted(boundary_counts.items())),
        "chainage_m": [mesh.chainages_m[0], mesh.chainages_m[-1]],
        "wall_radius_m": [DOWNSTREAM_WALL_RADIUS_M, UPSTREAM_RADIUS_M],
        "wall_thickness_m": UPSTREAM_RADIUS_M - DOWNSTREAM_WALL_RADIUS_M,
        "fixed_reference_z_m": mesh.wedge_top_z_m,
        "wedge_removed": True,
        "wedge_shelf_width_m": WEDGE_SHELF_WIDTH_M,
        "element_size_m": mesh.element_size_m,
    }


__all__ = [
    "audit_right_wedged_wall_transition",
    "build_right_wedged_wall_transition",
    "transition_end_chainage",
    "write_gmsh",
    "write_vtu",
]
