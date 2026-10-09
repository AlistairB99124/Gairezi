"""Run crack scenarios and summarise crack opening and stress intensity factors.

Usage: python Aster/run_crack_scenarios.py vertical horizontal
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "Aster" / "results" / "00_elastic_baseline"
CASES = ROOT / "Aster" / "cases"


def plane(spec: dict) -> tuple[list[float], list[float]]:
    theta = spec["chainage_m"] / spec["centerline_radius_m"]
    radial = (math.sin(theta), math.cos(theta))
    tangent = (math.cos(theta), -math.sin(theta))
    half_depth = 0.5 * spec["depth_m"] + 0.5 * spec["overshoot_m"]
    radius = spec["upstream_radius_m"] + spec["overshoot_m"] - half_depth
    if spec["orientation"] == "horizontal":
        return [0.0, 0.0, 1.0], [radius * radial[0], radius * radial[1], spec["z_m"] + spec["offset_m"]]
    origin = [radius * radial[0] + spec["offset_m"] * tangent[0], radius * radial[1] + spec["offset_m"] * tangent[1], spec["z_m"]]
    return [tangent[0], tangent[1], 0.0], origin


def main(names: list[str]) -> None:
    summary = {}
    for name in names:
        spec_path = CASES / f"03_{name}_crack.json"
        spec = json.loads(spec_path.read_text())
        stem = f"03_{name}_crack"
        subprocess.run(
            [sys.executable, str(ROOT / "Aster" / "run_case.py"), str(CASES / "03_crack_scenarios.comm"),
             "--spec", str(spec_path), "--name", stem, "--memory-mb", "12000"],
            check=False,
        )
        normal, origin = plane(spec)
        args = " ".join(f"{v:.9g}" for v in normal + origin)
        result_med = RESULTS / f"{stem}_result.med"
        log = RESULTS / f"{stem}_opening.log"
        with log.open("w") as handle:
            run_in_container_args = f"python3 {ROOT / 'Aster' / 'crack_opening.py'} {result_med} {args}"
            subprocess.run(
                ["/usr/bin/python3", "/usr/local/bin/run_aster", "--command", "bash", "--", "-c",
                  f"source /opt/salome-meca/2025/V2025.1.0_scibian_univ/salome_prerequisites.sh; {run_in_container_args}"],
                stdout=handle, stderr=subprocess.STDOUT, cwd=RESULTS,
            )
        opening = next((json.loads(line.split(" ", 1)[1]) for line in log.read_text().splitlines() if line.startswith("OPENING ")), None)
        table_path = RESULTS / f"{stem}_table.txt"
        k = {}
        if table_path.exists():
            table = json.loads(table_path.read_text())["table"]
            k = {key: max(abs(v) for v in table[key]) / 1e6 for key in ("K1", "K2", "K3")}
            k["crack_pressure_kpa"] = json.loads(table_path.read_text())["crack_pressure_pa"] / 1e3
        summary[name] = {"opening": opening, "max_abs_K_MPa_sqrt_m": k}
        print(name, json.dumps(summary[name], indent=2))
    (RESULTS / "crack_scenarios_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or ["vertical", "horizontal"])
