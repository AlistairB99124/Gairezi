"""Run one split-node crack scenario with contact and friction.

Usage: python Aster/run_crack_contact.py <name> '<json overrides>'
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aster"))
from insert_crack import insert_crack  # noqa: E402
from run_case import SALOME_ROOT  # noqa: E402

DEFAULTS = {
    "orientation": "horizontal",
    "chainage_m": 108.5,
    "upstream_radius_m": 80.0,
    "z_m": -25.0,
    "width_m": 3.0,
    "depth_m": 2.0,
    "friction_coefficient": 0.7,
    "load_steps": 4,
}


def main(name: str, overrides: dict) -> None:
    spec = {**DEFAULTS, **overrides}
    out_dir = ROOT / "Aster" / "results" / f"crack_{name}"
    spec_path = out_dir / "spec.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2))
    info = insert_crack(spec, out_dir)
    print("crack:", json.dumps(info))
    subprocess.run(
        [sys.executable, str(ROOT / "Aster" / "run_case.py"), str(ROOT / "Aster" / "cases" / "04_crack_contact.comm"),
         "--results", str(out_dir), "--spec", str(spec_path), "--name", "contact", "--memory-mb", "12000",
         "--ncpus", str(spec.get("ncpus", 1))],
        check=False,
    )
    result_med = out_dir / "contact_result.med"
    log = out_dir / "metrics.log"
    script = f"source {SALOME_ROOT}/salome_prerequisites.sh; python3 {ROOT / 'Aster' / 'crack_contact_metrics.py'} {result_med} {out_dir / 'crack_pairs.npy'}"
    with log.open("w") as handle:
        subprocess.run(["/usr/bin/python3", "/usr/local/bin/run_aster", "--command", "bash", "--", "-c", script],
                       stdout=handle, stderr=subprocess.STDOUT, cwd=out_dir)
    metrics = next((json.loads(line.split(" ", 1)[1]) for line in log.read_text().splitlines() if line.startswith("METRICS ")), None)
    summary = {"spec": spec, "crack": info, "metrics": metrics}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
