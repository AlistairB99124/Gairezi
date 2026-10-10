"""Opening and sliding check on the horizontal plane z = z0 using the uncracked Aster baseline.

Along the radial line at the crack chainage it reports the vertical stress across the wall, the
water-pressure opening criterion (crack stays open where the compression is below the water pressure)
and the friction demand tau / (mu * sigma_n).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Aster" / "results" / "00_elastic_baseline"
CENTERLINE_RADIUS_M = 78.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chainage", type=float, default=108.5)
    parser.add_argument("--z", type=float, default=-25.0)
    parser.add_argument("--head-above-crest", type=float, default=2.0)
    parser.add_argument("--friction", type=float, default=0.7)
    args = parser.parse_args()

    aster = np.load(RESULTS / "aster_result.npz")
    points, stress = aster["points"], aster["stress"]
    theta = args.chainage / CENTERLINE_RADIUS_M
    radial = np.array([np.sin(theta), np.cos(theta), 0.0])
    water_gradient = 1000.0 * 9.81
    water_pressure = water_gradient * (-args.z + args.head_above_crest)

    rows = []
    tree = cKDTree(points)
    for r in np.arange(75.75, 80.0001, 0.354167):
        target = np.array([r * radial[0], r * radial[1], args.z])
        distance, index = tree.query(target)
        if distance > 0.2:
            continue
        sxx, syy, szz, sxy, sxz, syz = stress[index]
        tangent = np.array([np.cos(theta), -np.sin(theta), 0.0])
        traction = np.array([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])[:, 2]  # traction on a horizontal plane
        shear = np.linalg.norm(traction[:2])
        sigma_n = -szz
        rows.append({
            "radius_m": round(float(r), 3),
            "depth_below_upstream_face_m": round(float(80.0 - r), 3),
            "sigma_zz_MPa": round(float(szz) / 1e6, 3),
            "shear_MPa": round(float(shear) / 1e6, 3),
            "net_normal_with_water_MPa": round((float(szz) + water_pressure) / 1e6, 3),
            "friction_demand_ratio": round(float(shear / (args.friction * sigma_n)), 3) if sigma_n > 0 else None,
        })
    rows.sort(key=lambda row: row["depth_below_upstream_face_m"])
    summary = {
        "chainage_m": args.chainage,
        "z_m": args.z,
        "water_pressure_kPa": water_pressure / 1e3,
        "friction_coefficient_assumed": args.friction,
        "profile_from_upstream_face": rows,
    }
    out = RESULTS / "baseline_plane_check.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"water pressure on a crack at z={args.z}: {water_pressure/1e3:.1f} kPa")
    print("depth  sigma_zz  shear  sigma_zz+p_water  tau/(mu*sn)")
    for row in rows:
        print(row["depth_below_upstream_face_m"], row["sigma_zz_MPa"], row["shear_MPa"], row["net_normal_with_water_MPa"], row["friction_demand_ratio"])


if __name__ == "__main__":
    main()
