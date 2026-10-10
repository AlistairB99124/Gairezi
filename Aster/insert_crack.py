"""Insert a planar crack into the MED input by splitting nodes on a mesh plane.

orientation 'horizontal': plane z = z_m, footprint width_m along the wall and depth_m from the upstream face.
orientation 'vertical': plane at chainage_m, footprint z_lo_m..z_hi_m and depth_m from the upstream face.
Nodes inside the footprint are duplicated for the cells on the + side of the plane, except the crack-front nodes
(shared with the plane faces outside the footprint). Lip faces become the groups CRACK_LOW and CRACK_UP.
An optional initial gap (gap_mm) moves the + side nodes along the plane normal.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aster"))
from run_case import run_in_container  # noqa: E402

BASE = ROOT / "Aster" / "results" / "00_elastic_baseline"
CENTERLINE_RADIUS_M = 78.0
TOLERANCE_M = 1e-6
CH_TOL = 0.05
HEX_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def insert_crack(spec: dict, out_dir: Path, base_dir: Path = BASE) -> dict:
    orientation = spec.get("orientation", "horizontal")
    mesh = np.load(base_dir / "mesh_intermediate.npz")
    points = mesh["points"].copy()
    hexes = mesh["conn_hexahedron"].copy()
    quads = mesh["conn_quad"].copy()
    quad_families = mesh["fam_quad"].copy()

    radius = np.hypot(points[:, 0], points[:, 1])
    chainage = CENTERLINE_RADIUS_M * np.arctan2(points[:, 0], points[:, 1])
    theta = spec["chainage_m"] / CENTERLINE_RADIUS_M
    full_width = bool(spec.get("full_width", False))
    if full_width and orientation != "horizontal":
        raise ValueError("full_width cracks are supported only on horizontal planes")

    if orientation == "horizontal":
        plane = np.abs(points[:, 2] - spec["z_m"]) < TOLERANCE_M
        normal = np.array([0.0, 0.0, 1.0])
        if full_width:
            window = np.ones(len(points), dtype=bool)
        else:
            half_width = 0.5 * spec["width_m"]
            window = (chainage > spec["chainage_m"] - half_width - CH_TOL) & (chainage < spec["chainage_m"] + half_width + CH_TOL)
        levels = np.unique(np.round(radius[plane & window & (radius > spec["upstream_radius_m"] - 6.0)], 3))
        r_cut = levels[np.argmin(np.abs(levels - (spec["upstream_radius_m"] - spec["depth_m"])))] - TOLERANCE_M
    else:
        plane = np.abs(chainage - spec["chainage_m"]) < CH_TOL
        normal = np.array([np.cos(theta), -np.sin(theta), 0.0])
        window = (points[:, 2] > spec["z_lo_m"] - TOLERANCE_M) & (points[:, 2] < spec["z_hi_m"] + TOLERANCE_M)
        r_cut = spec["upstream_radius_m"] - spec["depth_m"] - TOLERANCE_M
    if spec.get("through"):
        r_cut = 0.0
    footprint = plane & window & (radius > r_cut)

    def centroid_side(coordinates: np.ndarray) -> np.ndarray:
        """True for cells on the + side of the crack plane."""
        centre = points[coordinates].mean(axis=1)
        if orientation == "horizontal":
            return centre[:, 2] > spec["z_m"]
        return CENTERLINE_RADIUS_M * np.arctan2(centre[:, 0], centre[:, 1]) > spec["chainage_m"]

    above = centroid_side(hexes)

    # Plane faces seen from the - side cells; those fully inside the footprint are the lips.
    seen: set[tuple] = set()
    footprint_faces: list[list[int]] = []
    outside_nodes: set[int] = set()
    for cell, family in zip(hexes[~above], mesh["fam_hexahedron"][~above]):
        for pattern in HEX_FACES:
            nodes = [int(cell[i]) for i in pattern]
            if not all(plane[n] for n in nodes):
                continue
            key = tuple(sorted(nodes))
            if key in seen:
                continue
            seen.add(key)
            if all(footprint[n] for n in nodes) and (not full_width or family == -2):
                footprint_faces.append(nodes)
            else:
                outside_nodes.update(nodes)
    front = {n for face in footprint_faces for n in face} & outside_nodes
    split = np.array(sorted({n for face in footprint_faces for n in face} - front), dtype=np.int64)
    if len(split) == 0:
        raise SystemExit("Crack footprint has no interior nodes; make it larger")

    gap_m = spec.get("gap_mm", 0.0) / 1000.0
    duplicate = {int(n): len(points) + i for i, n in enumerate(split)}
    points = np.vstack([points, points[split] + gap_m * normal])
    lookup = np.arange(len(points))
    for old, new in duplicate.items():
        lookup[old] = new

    hexes[above] = lookup[hexes[above]]
    quads_above = centroid_side(quads)
    quads[quads_above] = lookup[quads[quads_above]]

    lower = np.array(footprint_faces, dtype=np.int64)
    upper = lookup[lower]
    low_family, up_family = -30, -31
    quads = np.vstack([quads, lower, upper])
    quad_families = np.concatenate([
        quad_families,
        np.full(len(lower), low_family, dtype=np.int64),
        np.full(len(upper), up_family, dtype=np.int64),
    ])

    arrays = {key: mesh[key] for key in mesh.files}
    arrays.update(points=points, conn_hexahedron=hexes, conn_quad=quads, fam_quad=quad_families)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "mesh_intermediate.npz", **arrays)
    groups = json.loads((base_dir / "group_family_ids.json").read_text())
    groups.update(CRACK_LOW=low_family, CRACK_UP=up_family)
    (out_dir / "group_family_ids.json").write_text(json.dumps(groups))
    shutil.copyfile(base_dir / "case_inputs.json", out_dir / "case_inputs.json")
    np.save(out_dir / "crack_pairs.npy", np.array([[n, d] for n, d in duplicate.items()], dtype=np.int64))
    (out_dir / "crack_meta.json").write_text(json.dumps({"normal": normal.tolist(), "gap_m": gap_m}))

    code = run_in_container(
        f"python3 {ROOT / 'Aster' / 'write_med.py'} {out_dir / 'mesh_intermediate.npz'} {out_dir / 'mesh.med'} {out_dir / 'group_family_ids.json'}",
        out_dir,
    )
    if code != 0:
        raise SystemExit("MED writer failed")
    footprint_nodes = np.unique(lower)
    info = {
        "orientation": orientation,
        "full_width": full_width,
        "split_nodes": len(split),
        "lip_faces": len(lower),
        "front_nodes": len(front),
        "actual_depth_m": float(spec["upstream_radius_m"] - radius[footprint_nodes].min()),
        "initial_gap_mm": gap_m * 1e3,
        "footprint_chainage_range_m": [float(chainage[footprint_nodes].min()), float(chainage[footprint_nodes].max())],
        "footprint_z_range_m": [float(points[footprint_nodes, 2].min()), float(points[footprint_nodes, 2].max())],
    }
    (out_dir / "crack_info.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    print(json.dumps(insert_crack(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])), indent=2))
