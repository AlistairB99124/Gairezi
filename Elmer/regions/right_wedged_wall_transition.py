"""Generate the wedge-free transition beyond the right wedged wall."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from regions.center_wall import write_gmsh, write_vtu
from regions.plinth import _monotone_values, load_contours
from regions.uniform_wall import (
    DOWNSTREAM_WALL_RADIUS_M,
    UPSTREAM_RADIUS_M,
    UniformWallMesh,
    build_uniform_wall,
)


WEDGE_END_CHAINAGE_M = 140.37597894015022
WEDGE_HEIGHT_M = 2.0


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


def build_right_wedged_wall_transition(root: Path) -> UniformWallMesh:
    contours = load_contours(root / "Data" / "plinth.json")
    reference_base_z_m = _monotone_values(contours, "plinth_z_m", WEDGE_END_CHAINAGE_M)
    end_chainage_m = transition_end_chainage(contours, reference_base_z_m + WEDGE_HEIGHT_M)
    return build_uniform_wall(root, WEDGE_END_CHAINAGE_M, end_chainage_m, reference_base_z_m, [])


def audit_right_wedged_wall_transition(mesh: UniformWallMesh) -> dict[str, object]:
    element_counts = Counter(element_type for element_type, _ in mesh.cells)
    boundary_counts = Counter(boundary_id for boundary_id, _ in mesh.boundaries)
    if set(boundary_counts) != set(range(1, 7)):
        raise ValueError(f"Missing right transition boundaries: found {sorted(boundary_counts)}")
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
        "fixed_reference_z_m": mesh.base_levels_m[0] + WEDGE_HEIGHT_M,
        "wedge_removed": True,
        "element_size_m": mesh.element_size_m,
    }


__all__ = [
    "audit_right_wedged_wall_transition",
    "build_right_wedged_wall_transition",
    "transition_end_chainage",
    "write_gmsh",
    "write_vtu",
]
