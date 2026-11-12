"""Run the dam-joint test cases shipped with the installed Code_Aster, unchanged.

These are EDF's own validation cases for the dam joint laws of R7.01.25
(JOINT_MECA_RUPT / JOINT_MECA_FROT / JOINT_MECA_ENDO). Passing them shows that
this installation reproduces the reference results before the same recipe is
transferred to the dam model. Nothing in the installation is modified: each
test is run from its own export file and only its log is kept.

Run from the Gairezi directory (PowerShell):

    .venv\\Scripts\\python.exe -m Aster.bundled_dam_tests
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_DIR = ROOT / "Aster" / "results" / "bundled_dam_tests"
SUMMARY_NAME = "summary.json"
# What each test covers, as read in its command file (share/aster/tests/<name>.comm).
TESTS = {
    "ssnp142a": "Dam on its ground, reservoir filling, 2D, rupture law, with and without imposed joint pressure",
    "ssnp142b": "Same in 3D (3D_JOINT), rupture law with zero tensile strength",
    "ssnp142c": "2D, friction law",
    "ssnp142d": "3D, friction law",
    "ssnp142e": "2D, rupture law, water flow solved in the joint (PLAN_JOINT_HYME)",
    "ssnp142g": "2D, friction law, water flow solved in the joint",
    "ssnp142h": "3D, friction law, water flow solved in the joint (3D_JOINT_HYME, quadratic mesh)",
    "ssnp142k": "2D, rupture law, driven by an imposed displacement (PILOTAGE DDL_IMPO)",
    "ssnp143b": "Grouting of the joint between two dam blocks, 3D (PRES_CLAVAGE)",
    "ssnp143d": "Sawing of the joint between two dam blocks, 3D, both laws (SCIAGE)",
    "ssnp162b": "Joint law check, 3D, rupture law",
    "ssnp162e": "Joint law check, 3D, friction law",
    "ssnp162h": "Joint law check, 3D, rupture law with water flow in the joint",
}
PER_TEST_LIMIT_S = 420
_CHECK = re.compile(r"^\s*(OK|NOOK)\s", re.MULTILINE)


def parse_log(text: str, exit_code: int) -> dict:
    """Outcome of one test from run_aster's output: its own diagnostic and the count of checks."""
    checks = _CHECK.findall(text)
    diagnostics = re.findall(r"DIAGNOSTIC JOB : (\S+)", text)
    failed = checks.count("NOOK")
    passed = checks.count("OK")
    diagnostic = diagnostics[-1] if diagnostics else None
    # "<A>_ALARM" is Code_Aster's diagnostic for a run whose checks pass but which raised
    # an alarm; the alarm identifiers are kept so the reader can judge them.
    alarms = sorted(set(re.findall(r"<A> <(\w+)>", text)))
    return {
        "exit_code": exit_code,
        "diagnostic": diagnostic,
        "checks_ok": passed,
        "checks_failed": failed,
        "alarms": alarms,
        "passed": exit_code == 0 and diagnostic in ("OK", "<A>_ALARM") and failed == 0 and passed > 0,
    }


def run_tests(names: list[str], output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    from Aster.whole_model_crack import (DEFAULT_WSL_DISTRIBUTION, DISTRIBUTION_ENVIRONMENT, find_linux_installation,
                                         linux_command)
    import os

    distribution = os.environ.get(DISTRIBUTION_ENVIRONMENT, DEFAULT_WSL_DISTRIBUTION)
    root = find_linux_installation(distribution)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in names:
        export = f"{root}/share/aster/tests/{name}.export"
        # threads=1: the tests' own export files ask for one core.
        command = linux_command(distribution, root, output_dir, [
            "timeout", "--kill-after=10", str(PER_TEST_LIMIT_S), f"{root}/bin/run_aster", "--test", export], threads=1)
        started = time.monotonic()
        try:
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                       timeout=PER_TEST_LIMIT_S + 30, check=False)
            text, code = completed.stdout + completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as error:
            partial = [part if isinstance(part, str) else (part or b"").decode("utf-8", "replace")
                       for part in (error.stdout, error.stderr)]
            text, code = "".join(partial), 124
        (output_dir / f"{name}.log").write_text(text, encoding="utf-8")
        outcome = parse_log(text, code)
        outcome.update(covers=TESTS.get(name, ""), elapsed_s=round(time.monotonic() - started, 1), export=export)
        results[name] = outcome
        print(f"{name}: {'PASS' if outcome['passed'] else 'FAIL'}  checks ok {outcome['checks_ok']}, "
              f"failed {outcome['checks_failed']}, {outcome['elapsed_s']} s"
              + (f", alarms {', '.join(outcome['alarms'])}" if outcome["alarms"] else ""), flush=True)
    # Tests run earlier into the same folder stay in the summary; a rerun replaces its own entry.
    summary_path = output_dir / SUMMARY_NAME
    if summary_path.is_file():
        earlier = json.loads(summary_path.read_text(encoding="utf-8"))
        if earlier.get("installation") == root:
            results = {**earlier.get("tests", {}), **results}
    summary = {"schema_version": 1, "installation": root, "distribution": distribution,
               "all_passed": bool(results) and all(result["passed"] for result in results.values()), "tests": results}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tests", nargs="*", default=list(TESTS), help="test names (default: the dam-joint set)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    summary = run_tests(arguments.tests, arguments.output_dir)
    sys.exit(0 if summary["all_passed"] else 2)


if __name__ == "__main__":
    main()
