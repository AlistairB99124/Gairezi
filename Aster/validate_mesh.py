"""Compare the Code_aster view of mesh.med with the Python mesh manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VOLUME_TOLERANCE = 1e-6  # relative


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "Aster" / "results" / "00_elastic_baseline")
    args = parser.parse_args()

    run = subprocess.run([sys.executable, str(ROOT / "Aster" / "run_case.py"),
                          str(ROOT / "Aster" / "cases" / "validate_mesh.comm"),
                          "--results", str(args.results)])
    if run.returncode != 0:
        print("FAIL: Code_aster could not read the mesh")
        return 1

    manifest = json.loads((args.results / "mesh_manifest.json").read_text())
    aster = json.loads((args.results / "validate_mesh_table.txt").read_text())
    failures = []

    for name, expected in manifest["region_volumes_m3"].items():
        got = aster["volume_m3"][name]
        if abs(got - expected) > VOLUME_TOLERANCE * expected:
            failures.append(f"{name} volume: Aster {got:.6f} vs source {expected:.6f}")
    for name, expected in manifest["group_counts"].items():
        got = aster["cell_counts"][name]
        if got != expected:
            failures.append(f"{name} cells: Aster {got} vs source {expected}")
    for name in manifest["required_groups"]:
        if aster["cell_counts"][name] == 0:
            failures.append(f"{name} is empty")

    for line in failures:
        print("FAIL:", line)
    print("mesh validation:", "FAILED" if failures else "passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
