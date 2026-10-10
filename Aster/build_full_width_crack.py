"""Build the full-wall, 3 m deep, 2 mm horizontal crack case at z = -25 m."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aster"))

from insert_crack import insert_crack  # noqa: E402


RESULTS = ROOT / "Aster" / "results" / "crack_full_width_z25_d3_gap2_aligned"
BASE = RESULTS / "uncracked_aligned"
SPEC = {
    "orientation": "horizontal",
    "chainage_m": 108.5,
    "upstream_radius_m": 80.0,
    "z_m": -25.0,
    "depth_m": 3.0,
    "full_width": True,
    "gap_mm": 2.0,
    "friction_coefficient": 0.7,
    "load_steps": 4,
}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "Aster" / "build_med_mesh.py"),
            "--output-dir",
            str(BASE),
            "--align-crack-front-depth-m",
            str(SPEC["depth_m"]),
            "--crack-z-m",
            str(SPEC["z_m"]),
        ],
        check=True,
    )
    (RESULTS / "spec.json").write_text(json.dumps(SPEC, indent=2))
    info = insert_crack(SPEC, RESULTS, base_dir=BASE)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()