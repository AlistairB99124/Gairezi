"""Solve the full-width z=-25 m, 3 m deep crack for initial gaps from 2 to 10 mm."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import meshio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aster"))

from insert_crack import insert_crack  # noqa: E402


RESULTS = ROOT / "Aster" / "results"
SWEEP = RESULTS / "crack_full_width_z25_d3_gap_sweep"
BASE = SWEEP / "uncracked_aligned"
GAPS_MM = range(2, 11)
BASE_SPEC = {
    "orientation": "horizontal",
    "chainage_m": 108.5,
    "upstream_radius_m": 80.0,
    "z_m": -25.0,
    "depth_m": 3.0,
    "full_width": True,
    "friction_coefficient": 0.7,
    "load_steps": 4,
}


def build_aligned_baseline() -> None:
    if (BASE / "mesh.med").exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "Aster" / "build_med_mesh.py"),
            "--output-dir",
            str(BASE),
            "--align-crack-front-depth-m",
            str(BASE_SPEC["depth_m"]),
            "--crack-z-m",
            str(BASE_SPEC["z_m"]),
        ],
        check=True,
    )


def response_metrics(case_dir: Path) -> dict[str, float]:
    mesh = meshio.read(case_dir / "contact_result.vtu")
    displacement = mesh.point_data["result__DEPL"][:, :3]
    pairs = np.load(case_dir / "crack_pairs.npy")
    meta = json.loads((case_dir / "crack_meta.json").read_text())
    normal = np.asarray(meta["normal"])
    normal_jump = (displacement[pairs[:, 1]] - displacement[pairs[:, 0]]) @ normal
    aperture = meta["gap_m"] + normal_jump
    slip = np.linalg.norm(
        displacement[pairs[:, 1]] - displacement[pairs[:, 0]] - np.outer(normal_jump, normal),
        axis=1,
    )
    return {
        "aperture_mm_min": float(aperture.min() * 1e3),
        "aperture_mm_max": float(aperture.max() * 1e3),
        "aperture_mm_mean": float(aperture.mean() * 1e3),
        "fraction_in_contact": float((aperture < 1e-6).mean()),
        "maximum_slip_mm": float(slip.max() * 1e3),
        "maximum_principal_tension_mpa": float(mesh.point_data["principal_stress_max"].max() / 1e6),
        "minimum_principal_compression_mpa": float(mesh.point_data["principal_stress_min"].min() / 1e6),
    }


def write_report(rows: list[dict[str, float]]) -> None:
    lines = [
        "# Full-Width Crack Gap Sweep",
        "",
        "All cases use the same full-wall horizontal crack at $z=-25$ m, exact 3.000 m radial depth, 621 lip faces, gravity, exterior hydrostatic loading, crest overflow, and hydrostatic pressure on both crack faces.",
        "",
        "| Initial gap (mm) | Aperture min-max (mm) | Mean aperture (mm) | Max slip (mm) | Max principal tension (MPa) | Min principal compression (MPa) | Contact fraction |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['gap_mm']:.0f} | {row['aperture_mm_min']:.3f} to {row['aperture_mm_max']:.3f} "
            f"| {row['aperture_mm_mean']:.3f} | {row['maximum_slip_mm']:.3f} "
            f"| {row['maximum_principal_tension_mpa']:.3f} | {row['minimum_principal_compression_mpa']:.3f} "
            f"| {row['fraction_in_contact']:.0%} |"
        )
    lines.extend(
        [
            "",
            "`principal_stress_max`, `principal_stress_mid`, `principal_stress_min`, and all six stress components are available in every case's `contact_result.vtu` for ParaView.",
        ]
    )
    (RESULTS / "cracking_results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    SWEEP.mkdir(parents=True, exist_ok=True)
    build_aligned_baseline()
    rows = []
    for gap_mm in GAPS_MM:
        case_dir = SWEEP / f"gap_{gap_mm:02d}mm"
        spec = {**BASE_SPEC, "gap_mm": float(gap_mm)}
        result_vtu = case_dir / "contact_result.vtu"
        if not result_vtu.exists():
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "spec.json").write_text(json.dumps(spec, indent=2))
            info = insert_crack(spec, case_dir, base_dir=BASE)
            if abs(info["actual_depth_m"] - 3.0) > 1e-9:
                raise RuntimeError(f"{case_dir}: crack depth is not 3 m")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "Aster" / "run_case.py"),
                    str(ROOT / "Aster" / "cases" / "04_crack_contact.comm"),
                    "--results",
                    str(case_dir),
                    "--spec",
                    str(case_dir / "spec.json"),
                    "--name",
                    "contact",
                    "--memory-mb",
                    "7000",
                    "--ncpus",
                    "4",
                ],
                check=True,
            )
        row = {"gap_mm": float(gap_mm), **response_metrics(case_dir)}
        (case_dir / "summary.json").write_text(json.dumps({"spec": spec, "response": row}, indent=2))
        rows.append(row)
        print(json.dumps(row), flush=True)
    write_report(rows)


if __name__ == "__main__":
    main()