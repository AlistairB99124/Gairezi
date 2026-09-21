"""Generate the standalone right wall with its integral smooth downstream toe."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from regions.left_wedged_wall import (
    DOWNSTREAM_WALL_RADIUS_M,
    UPSTREAM_RADIUS_M,
    WEDGE_HEIGHT_M,
    WEDGE_SHELF_WIDTH_M,
    WEDGE_WIDTH_M,
    WEDGE_TOE_RADIUS_M,
    LeftWedgedWallMesh,
    build_wedged_wall,
    write_gmsh,
    write_vtu,
)


START_CHAINAGE_M = 116.0
END_CHAINAGE_M = 140.37597894015022


def build_right_wedged_wall(root: Path) -> LeftWedgedWallMesh:
    return build_wedged_wall(root, START_CHAINAGE_M, END_CHAINAGE_M)


def audit_right_wedged_wall(mesh: LeftWedgedWallMesh) -> dict[str, object]:
    element_counts = Counter(element_type for element_type, _ in mesh.cells)
    boundary_counts = Counter(boundary_id for boundary_id, _ in mesh.boundaries)
    if set(boundary_counts) != set(range(1, 7)):
        raise ValueError(f"Missing right-wedged-wall boundaries: found {sorted(boundary_counts)}")
    return {
        "nodes": len(mesh.nodes),
        "cells": len(mesh.cells),
        "hexahedra": element_counts[5],
        "triangular_prisms": element_counts[6],
        "boundary_faces": dict(sorted(boundary_counts.items())),
        "chainage_m": [START_CHAINAGE_M, END_CHAINAGE_M],
        "plinth_z_m": [-28.5, -17.03],
        "wall_radius_m": [DOWNSTREAM_WALL_RADIUS_M, UPSTREAM_RADIUS_M],
        "wedge_radius_m": [WEDGE_TOE_RADIUS_M, DOWNSTREAM_WALL_RADIUS_M],
        "wedge_height_m": WEDGE_HEIGHT_M,
        "wedge_width_m": WEDGE_WIDTH_M,
        "wedge_shelf_width_m": WEDGE_SHELF_WIDTH_M,
        "element_size_m": mesh.element_size_m,
    }


__all__ = ["audit_right_wedged_wall", "build_right_wedged_wall", "write_gmsh", "write_vtu"]
