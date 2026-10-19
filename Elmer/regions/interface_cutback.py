"""Replace staircase plinth corners by wall wedges without moving any nodes."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
from typing import TYPE_CHECKING
from .model_groups import FACE_GROUP_IDS

from .tetrahedral import _CELL_FACES, _signed_volume, split_cell

if TYPE_CHECKING:
    from .combined_structure import CombinedStructure

CELL_FACES = {
    **_CELL_FACES,
    7: ((0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
}


def cutback_wall_plinth(mesh: CombinedStructure, reference_radius_m: float) -> CombinedStructure:
    """Chamfer on a 1:1 chainage/elevation slope, entirely inside existing concrete.

    Chainage is the longitudinal design coordinate. On a curved wall, physical
    slopes vary slightly across the radial width, as do its other mesh lengths.
    """
    stations = [
        round(reference_radius_m * math.atan2(x, y), 8) for x, y, _ in mesh.nodes
    ]
    radii = [round(math.hypot(x, y), 8) for x, y, _ in mesh.nodes]
    heights = [round(p[2], 8) for p in mesh.nodes]
    owners: dict[tuple[int, ...], tuple[int, int]] = {}
    steps: dict[tuple[float, float, float, float], tuple[float, set[int]]] = {}
    for index, ((kind, cell), region) in enumerate(zip(mesh.cells, mesh.region_ids, strict=True)):
        for pattern in CELL_FACES[kind]:
            face = tuple(sorted(cell[i] for i in pattern))
            if face not in owners:
                owners[face] = (index, region)
                continue
            other, other_region = owners[face]
            if {region, other_region} != {1, 2}:
                continue
            if len({stations[n] for n in face}) != 1:
                continue
            low, high = min(heights[n] for n in face), max(heights[n] for n in face)
            if high - low < 1e-8:
                continue
            plinth_index = index if region == 1 else other
            plinth_kind, plinth_cell = mesh.cells[plinth_index]
            if plinth_kind != 5:
                raise ValueError("45-degree cutback requires hexahedral plinth step corners")
            step_station = stations[face[0]]
            ends = sorted({stations[n] for n in plinth_cell})
            if len(ends) != 2 or step_station not in ends:
                raise ValueError("Invalid longitudinal plinth step")
            away = ends[1] if ends[0] == step_station else ends[0]
            if not math.isclose(abs(away - step_station), high - low, abs_tol=1e-7):
                raise ValueError(
                    "45-degree cutback requires step rise equal to adjacent chainage run: "
                    f"chainage {step_station:g}, rise {high-low:g}, run {abs(away-step_station):g}"
                )
            key = (ends[0], ends[1], low, high)
            previous = steps.setdefault(key, (step_station, set()))
            if previous[0] != step_station:
                raise ValueError("Opposing cutbacks overlap within a plinth course")
            previous[1].add(plinth_index)

    if not steps:
        raise ValueError("No wall/plinth staircase corners found for 45-degree cutback")
    boundary_tags = {tuple(sorted(face)): tag for tag, face in mesh.boundaries}
    removed_boundary_faces = set()
    new_boundaries = []
    cells = []
    regions = []
    transferred = 0.
    split_count = 0
    transfers = 0
    footprint_transfers = 0
    removed = 0.
    removed_wedges = 0
    wall_radii = [radii[n] for (_, cell), region in zip(mesh.cells, mesh.region_ids)
                  if region == 2 for n in cell]
    wall_mid_radius = (min(wall_radii) + max(wall_radii)) / 2
    for index, ((kind, cell), region) in enumerate(zip(mesh.cells, mesh.region_ids, strict=True)):
        key = (
            min(stations[n] for n in cell), max(stations[n] for n in cell),
            min(heights[n] for n in cell), max(heights[n] for n in cell),
        )
        if region != 1 or key not in steps:
            cells.append((kind, cell))
            regions.append(region)
            continue
        if kind != 5 or len({radii[n] for n in cell}) != 2:
            raise ValueError("Unsupported plinth closure intersecting a 45-degree cutback")
        step, selected = steps[key]
        away = key[1] if step == key[0] else key[0]
        low, high = key[2:]
        radial_ends = sorted({radii[n] for n in cell})
        lookup = {(stations[n], radii[n], heights[n]): n for n in cell}
        upper = []
        lower = []
        for radius in radial_ends:
            a = lookup[(away, radius, high)]
            b = lookup[(step, radius, high)]
            c = lookup[(step, radius, low)]
            d = lookup[(away, radius, low)]
            upper.extend((a, b, c))
            lower.extend((a, c, d))
        wedges = [tuple(upper), tuple(lower)]
        original_volume = sum(_signed_volume(mesh.nodes, t) for t in split_cell(mesh.nodes, kind, cell))
        volumes = [
            sum(_signed_volume(mesh.nodes, t) for t in split_cell(mesh.nodes, 6, wedge))
            for wedge in wedges
        ]
        if min(volumes) <= 0 or not math.isclose(sum(volumes), original_volume, rel_tol=1e-8):
            raise ValueError("Cutback does not preserve positive concrete volume")
        retained = index in selected
        new_regions = [2, 1] if retained else [1]
        retained_wedges = wedges if retained else wedges[1:]
        cells.extend((6, wedge) for wedge in retained_wedges)
        regions.extend(new_regions)
        split_count += 1
        if retained:
            transferred += volumes[0]
            transfers += 1
        else:
            removed += volumes[0]
            removed_wedges += 1
        footprint_transfers += index in selected
        exterior = []
        for pattern in _CELL_FACES[5]:
            old_face = tuple(sorted(cell[i] for i in pattern))
            if old_face in boundary_tags:
                removed_boundary_faces.add(old_face)
                exterior.append((set(old_face), boundary_tags[old_face]))
        for wedge in retained_wedges:
            for pattern in _CELL_FACES[6]:
                face = tuple(wedge[i] for i in pattern)
                for old_nodes, tag in exterior:
                    if set(face) <= old_nodes:
                        new_boundaries.append((tag, face))
                        break
    if footprint_transfers != sum(len(selected) for _, selected in steps.values()):
        raise ValueError("Not every plinth corner was transferred to the wall")
    boundaries = [
        (tag, face) for tag, face in mesh.boundaries
        if tuple(sorted(face)) not in removed_boundary_faces
    ] + new_boundaries
    known_tags = {tuple(sorted(face)): tag for tag, face in boundaries}
    exterior_faces: dict[tuple[int, ...], tuple[tuple[int, ...], tuple[int, ...]]] = {}
    counts_by_face: Counter[tuple[int, ...]] = Counter()
    for kind, cell in cells:
        for pattern in CELL_FACES[kind]:
            face = tuple(cell[i] for i in pattern)
            key = tuple(sorted(face))
            counts_by_face[key] += 1
            exterior_faces[key] = (face, cell)
    if max(counts_by_face.values()) > 2:
        raise ValueError("Nonmanifold concrete after shoulder cutback")
    boundaries = []
    for key, (face, cell) in exterior_faces.items():
        if counts_by_face[key] != 1:
            continue
        if key in known_tags:
            tag = known_tags[key]
        else:
            # Newly exposed shoulder slopes and wall sides carry the same
            # upstream/downstream hydrostatic law as their respective bank.
            mean_radius = sum(radii[n] for n in face) / len(face)
            tag = FACE_GROUP_IDS["UPSTREAM" if mean_radius > wall_mid_radius else "DOWNSTREAM"]
            p, q, r = (mesh.nodes[n] for n in face[:3])
            u = [q[i] - p[i] for i in range(3)]
            v = [r[i] - p[i] for i in range(3)]
            normal = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
            outward = [
                sum(mesh.nodes[n][i] for n in face)/len(face)
                - sum(mesh.nodes[n][i] for n in cell)/len(cell)
                for i in range(3)
            ]
            if sum(normal[i]*outward[i] for i in range(3)) < 0:
                face = tuple(reversed(face))
        boundaries.append((tag, face))
    counts = Counter(regions)
    return replace(
        mesh, cells=cells, region_ids=regions, boundaries=boundaries,
        region_cell_counts={"plinth": counts[1], "wall": counts[2]},
        wall_plinth_cutback={
            "method": "plinth_cutback_45deg", "slope_coordinate": "design chainage / elevation",
            "radial_extent": "full_plinth_width_including_shoulders",
            "step_count": len(steps), "split_plinth_cells": split_count,
            "wall_wedges": transfers, "transferred_volume_m3": transferred,
            "shoulder_fill": "empty_outside_original_wall_footprint",
            "removed_shoulder_wedges": removed_wedges,
            "removed_shoulder_volume_m3": removed,
            "shoulder_wall_wedges": 0,
            "volume_coordinate_reference_radius_m": reference_radius_m,
            "nodes_moved_or_added": 0,
        },
    )
