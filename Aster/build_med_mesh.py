"""Export the combined dam mesh to a MED file with named volume and face groups."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Aster"))

from run_case import run_in_container  # noqa: E402
from structural_model import (  # noqa: E402
    FACE_GROUP_IDS,
    REGION_GROUP_IDS,
    REQUIRED_GROUPS,
    audit_combined_structure,
    build_combined_structure,
    build_named_groups,
)
from Elmer.regions.uniform_wall import UPSTREAM_RADIUS_M, downstream_radius  # noqa: E402

# Gmsh element type -> meshio/MED cell type.
VOLUME_TYPES = {4: "tetra", 5: "hexahedron", 6: "wedge", 7: "pyramid"}
FACE_TYPES = {3: "triangle", 4: "quad"}

# MED families: cells use negative numbers by convention.
GROUP_FAMILY_IDS = {
    name: -index
    for index, name in enumerate([*REGION_GROUP_IDS, *FACE_GROUP_IDS], start=1)
}

_GAUSS = 1.0 / np.sqrt(3.0)


def _hex_volume(xyz: np.ndarray) -> float:
    total = 0.0
    for r in (-_GAUSS, _GAUSS):
        for s in (-_GAUSS, _GAUSS):
            for t in (-_GAUSS, _GAUSS):
                signs = np.array([
                    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
                ], float)
                dn = 0.125 * np.stack([
                    signs[:, 0] * (1 + signs[:, 1] * s) * (1 + signs[:, 2] * t),
                    signs[:, 1] * (1 + signs[:, 0] * r) * (1 + signs[:, 2] * t),
                    signs[:, 2] * (1 + signs[:, 0] * r) * (1 + signs[:, 1] * s),
                ])
                total += np.linalg.det(dn @ xyz)
    return total


def _wedge_volume(xyz: np.ndarray) -> float:
    total = 0.0
    tri_points = ((1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3))
    for r, s in tri_points:
        for t in (-_GAUSS, _GAUSS):
            lam = np.array([1 - r - s, r, s])
            dn_tri = np.array([[-1, 1, 0], [-1, 0, 1]], float)
            dn = np.zeros((3, 6))
            for half, sign in ((slice(0, 3), (1 - t) / 2), (slice(3, 6), (1 + t) / 2)):
                dn[0:2, half] = dn_tri * sign
            dn[2, 0:3] = -lam / 2
            dn[2, 3:6] = lam / 2
            total += np.linalg.det(dn @ xyz) / 6.0
    return total


def _tet_volume(xyz: np.ndarray) -> float:
    return np.linalg.det(xyz[1:] - xyz[0]) / 6.0


def cell_volume(element_type: int, xyz: np.ndarray) -> float:
    if element_type == 4:
        return _tet_volume(xyz)
    if element_type == 5:
        return _hex_volume(xyz)
    if element_type == 6:
        return _wedge_volume(xyz)
    raise NotImplementedError(f"Volume for Gmsh element type {element_type}")


def face_area(xyz: np.ndarray) -> float:
    area = 0.5 * np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0]))
    if len(xyz) == 4:
        area += 0.5 * np.linalg.norm(np.cross(xyz[2] - xyz[0], xyz[3] - xyz[0]))
    return float(area)


def nodal_tributary_areas(structure) -> dict[int, float]:
    """Row sums of the consistent surface mass matrix, i.e. the integral of each shape function."""
    points = np.asarray(structure.nodes, dtype=float)
    areas: dict[int, float] = Counter()
    gauss = ((-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS), (_GAUSS, _GAUSS), (-_GAUSS, _GAUSS))
    corners = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    for boundary_id, face in structure.boundaries:
        if boundary_id != FACE_GROUP_IDS["FOUNDATION"]:
            continue
        xyz = points[list(face)]
        if len(face) == 3:
            for node in face:
                areas[node] += face_area(xyz) / 3.0
            continue
        for r, s in gauss:
            shape = np.array([0.25 * (1 + cr * r) * (1 + cs * s) for cr, cs in corners])
            d_r = np.array([0.25 * cr * (1 + cs * s) for cr, cs in corners]) @ xyz
            d_s = np.array([0.25 * cs * (1 + cr * r) for cr, cs in corners]) @ xyz
            jacobian = np.linalg.norm(np.cross(d_r, d_s))
            for node, weight in zip(face, shape):
                areas[node] += weight * jacobian
    return dict(areas)


def case_inputs(structure, spring_areas: dict[int, float]) -> dict:
    nodes = sorted(spring_areas)
    return {
        "material": {
            "density_kg_m3": structure.material.density_kg_m3,
            "youngs_modulus_pa": structure.material.youngs_modulus_pa,
            "poissons_ratio": structure.material.poissons_ratio,
        },
        "loads": {
            "gravity_z_m_s2": structure.loads.gravity_z_m_s2,
            "maximum_water_height_m": structure.loads.maximum_water_height_m,
            "peak_water_pressure_pa": structure.loads.peak_water_pressure_pa,
            "water_density_kg_m3": structure.loads.water_density_kg_m3,
            "tailwater_head_m": structure.loads.tailwater_head_m,
            "overflow_head_m": structure.loads.overflow_head_m,
            "foundation_elevation_m": min(p[2] for p in structure.nodes),
        },
        "foundation_spring_n_per_m3": [
            structure.foundation_support.spring_x_n_per_m3,
            structure.foundation_support.spring_y_n_per_m3,
            structure.foundation_support.spring_z_n_per_m3,
        ],
        "spring_nodes": nodes,
        "spring_areas_m2": [spring_areas[n] for n in nodes],
    }


def build_arrays(root: Path, extra_radial_fractions: tuple[float, ...] = ()): 
    structure = build_combined_structure(root, extra_radial_fractions)
    audit_combined_structure(structure)
    groups = build_named_groups(structure)
    points = np.asarray(structure.nodes, dtype=float)
    region_names = {rid: name for name, rid in REGION_GROUP_IDS.items()}
    face_names = {gid: name for name, gid in FACE_GROUP_IDS.items()}

    blocks: dict[str, list[tuple[tuple[int, ...], int]]] = {}
    for (element_type, cell), region in zip(structure.cells, structure.region_ids):
        if element_type not in VOLUME_TYPES:
            raise ValueError(f"Unsupported volume element type {element_type}")
        blocks.setdefault(VOLUME_TYPES[element_type], []).append(
            (cell, GROUP_FAMILY_IDS[region_names[region]])
        )
    for boundary_id, face in structure.boundaries:
        blocks.setdefault(FACE_TYPES[len(face)], []).append(
            (face, GROUP_FAMILY_IDS[face_names[boundary_id]])
        )

    arrays: dict[str, np.ndarray] = {"points": points}
    for cell_type, entries in blocks.items():
        arrays[f"conn_{cell_type}"] = np.array([c for c, _ in entries], dtype=np.int64)
        arrays[f"fam_{cell_type}"] = np.array([f for _, f in entries], dtype=np.int64)
    spring_areas = nodal_tributary_areas(structure)
    arrays["spring_nodes"] = np.array(sorted(spring_areas), dtype=np.int64)
    # Cells in generator order (-1 padded); the client stress pipeline depends on this order.
    arrays["orig_types"] = np.array([t for t, _ in structure.cells], dtype=np.int64)
    arrays["orig_conn"] = np.full((len(structure.cells), 8), -1, dtype=np.int64)
    for index, (_, cell) in enumerate(structure.cells):
        arrays["orig_conn"][index, : len(cell)] = cell
    return structure, groups, arrays, spring_areas


def build_manifest(structure, groups, mesh_path: Path) -> dict:
    points = np.asarray(structure.nodes, dtype=float)
    region_names = {rid: name for name, rid in REGION_GROUP_IDS.items()}
    volumes: dict[str, float] = Counter()
    for (element_type, cell), region in zip(structure.cells, structure.region_ids):
        volumes[region_names[region]] += cell_volume(element_type, points[list(cell)])
    areas: dict[str, float] = Counter()
    face_names = {gid: name for name, gid in FACE_GROUP_IDS.items()}
    for boundary_id, face in structure.boundaries:
        areas[face_names[boundary_id]] += face_area(points[list(face)])
    return {
        "mesh_file": str(mesh_path),
        "unit": "m",
        "node_count": len(points),
        "volume_cell_counts": dict(Counter(VOLUME_TYPES[t] for t, _ in structure.cells)),
        "group_counts": {name: len(indices) for name, indices in groups.items()},
        "required_groups": list(REQUIRED_GROUPS),
        "region_volumes_m3": dict(volumes),
        "exterior_areas_m2": dict(areas),
        "bounds_min": points.min(axis=0).tolist(),
        "bounds_max": points.max(axis=0).tolist(),
        "family_ids": GROUP_FAMILY_IDS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Aster" / "results" / "00_elastic_baseline")
    parser.add_argument("--align-crack-front-depth-m", type=float)
    parser.add_argument("--crack-z-m", type=float, default=-25.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    extra_radial_fractions: tuple[float, ...] = ()
    if args.align_crack_front_depth_m is not None:
        downstream_radius_m = downstream_radius(args.crack_z_m)
        fraction = (UPSTREAM_RADIUS_M - args.align_crack_front_depth_m - downstream_radius_m) / (
            UPSTREAM_RADIUS_M - downstream_radius_m
        )
        if not 0.0 < fraction < 1.0:
            parser.error("The aligned crack front must lie within the wall thickness at --crack-z-m")
        extra_radial_fractions = (fraction,)

    structure, groups, arrays, spring_areas = build_arrays(ROOT, extra_radial_fractions)
    (args.output_dir / "case_inputs.json").write_text(json.dumps(case_inputs(structure, spring_areas)))
    npz_path = args.output_dir / "mesh_intermediate.npz"
    groups_path = args.output_dir / "group_family_ids.json"
    mesh_path = args.output_dir / "mesh.med"
    np.savez(npz_path, **arrays)
    groups_path.write_text(json.dumps(GROUP_FAMILY_IDS))
    code = run_in_container(
        f"python3 {ROOT / 'Aster' / 'write_med.py'} {npz_path} {mesh_path} {groups_path}",
        args.output_dir,
    )
    if code != 0:
        sys.exit(f"MED writer failed with exit code {code}")
    manifest = build_manifest(structure, groups, mesh_path)
    (args.output_dir / "mesh_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
