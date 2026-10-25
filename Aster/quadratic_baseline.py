"""Bounded developer verification of an isolated full-dam quadratic candidate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from Aster.run_case import bundled_python_for, detect_installation, run_solver
from Aster.source_contract import audit_source_materials, sha256

CASE = Path(__file__).with_name("cases") / "quadratic_baseline.comm"
CONVERTER = Path(__file__).with_name("convert_quadratic_to_med.py")


class NativeBaselineError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def verify_native_result(returncode: int, logs: str, native_path: Path, result_path: Path) -> dict:
    if "<JEVEUX_62>" in logs:
        raise NativeBaselineError(
            "native_memory_allocation",
            "Native quadratic solve exceeded its memory limit; see native.fort6.log",
        )
    if returncode or any(marker in logs for marker in (
        "<EXCEPTION>", "Traceback (most recent call last)", "Fatal Python error",
    )):
        raise NativeBaselineError("native_execution_failed", "Native quadratic solve failed; see native.fort6.log")
    if (not native_path.is_file() or not result_path.is_file() or result_path.stat().st_size == 0):
        raise NativeBaselineError("incomplete_native_outputs", "Native quadratic solve produced incomplete outputs")
    summary = json.loads(native_path.read_text(encoding="utf-8"))
    peak = summary.get("maximum_absolute_displacement_component_m")
    count = summary.get("displacement_components")
    if (summary.get("status") != "finite_native_elastic_solve_not_baseline_acceptance"
            or isinstance(peak, bool) or not isinstance(peak, (int, float))
            or not math.isfinite(peak) or peak < 0 or isinstance(count, bool)
            or not isinstance(count, int) or count <= 0):
        raise NativeBaselineError("invalid_native_summary", "Native quadratic displacement summary is invalid")
    return summary


def run(candidate_directory: Path, budget_s: int = 540, memory_mib: int = 8192,
        threads: int = 1, allow_extended_budget: bool = False) -> dict:
    """This timeout is for developer verification, not user-submitted jobs."""
    if (isinstance(budget_s, bool) or not isinstance(budget_s, int) or budget_s < 1
            or (budget_s > 570 and not allow_extended_budget) or memory_mib <= 0):
        raise ValueError("Positive developer budget required; more than 570 seconds needs explicit extended approval")
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 4:
        raise ValueError("Developer verification threads must be an integer from 1 to 4")
    installation = detect_installation()
    if not installation["available"]:
        raise RuntimeError("Native quadratic baseline requires Code_Aster")
    python = bundled_python_for(installation["launcher_path"])
    if python is None:
        raise RuntimeError("Native quadratic baseline requires the bundled MED interpreter")
    candidate_path = candidate_directory / "quadratic_mesh_manifest.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    archive = candidate_directory / "quadratic_mesh_arrays.npz"
    source_path = Path(candidate["source_mesh_manifest"])
    if (sha256(archive) != candidate["quadratic_array_sha256"] or
            sha256(source_path) != candidate["source_mesh_manifest_sha256"]):
        raise ValueError("Quadratic candidate or source manifest changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    mesh_source = Path(source["source_mesh"])
    for name, digest in candidate["source_hashes"].items():
        if sha256(mesh_source.parent / name) != digest:
            raise ValueError(f"Quadratic baseline source changed: {name}")
    audit_source_materials(mesh_source)
    config = json.loads(Path(source["source_config"]).read_text(encoding="utf-8"))
    directory = Path(tempfile.mkdtemp(prefix="baseline-", dir=candidate_directory.resolve()))
    inputs_path = directory / "inputs.json"
    def material(name: str) -> dict:
        value = config["material"][name]
        return {"E": value["youngs_modulus"], "NU": value["poisson_ratio"], "RHO": value["density"]}
    inputs_path.write_text(json.dumps({
        "repository_root": str(Path(__file__).resolve().parent.parent),
        "concrete": material("concrete"), "reference": material("bedrock"),
        "gravity_m_per_s2": abs(config["loads"]["gravity"]),
        "pressures": source["source_pressures"], "crest_pressure_pa": source["source_crest_pressure_pa"],
    }, indent=2), encoding="utf-8")
    report = {
        "status": "running", "directory": str(directory), "engineering_validated": False,
        "installation": installation,
        "baseline_verified": False, "candidate_manifest_sha256": sha256(candidate_path),
        "case_sha256": sha256(CASE), "converter_sha256": sha256(CONVERTER),
        "native_discrete_sha256": sha256(Path(__file__).with_name("native_discrete.py")),
        "surface_code_sha256": sha256(Path(__file__).with_name("quadratic_surface.py")),
        "inputs_sha256": sha256(inputs_path), "budget_s": budget_s, "memory_limit_mib": memory_mib,
        "threads": threads,
        "extended_budget_authorized": allow_extended_budget,
        "element_order_policy": candidate.get("element_order_policy", "all_quadratic"),
        "scope": "developer_full_dam_quadratic_zero_damage_elastic_check",
    }
    started = time.monotonic()
    native_work = directory / "native_work"
    try:
        mesh_path = directory / "mesh.med"
        # Keep stage diagnostics even when the bounded converter is terminated.
        with (directory / "conversion.stdout.log").open("w", encoding="utf-8") as stdout, \
                (directory / "conversion.stderr.log").open("w", encoding="utf-8") as stderr:
            conversion = subprocess.run(
                [str(python), str(CONVERTER), str(archive.resolve()), str(mesh_path)],
                stdout=stdout, stderr=stderr, text=True, timeout=min(120, budget_s), check=False,
            )
        if conversion.returncode or not mesh_path.is_file():
            raise RuntimeError(f"Quadratic MED conversion failed; see {directory}")
        discrete = directory / "quadratic_discrete.npz"
        report["discrete_sha256"] = sha256(discrete)
        report["stiffness_encoding"] = json.loads(discrete.with_suffix(".json").read_text(encoding="utf-8"))
        remaining = int(budget_s - (time.monotonic() - started))
        if remaining < 1:
            raise RuntimeError("Quadratic conversion exhausted the developer verification budget")
        native = directory / "native_summary.json"
        result = directory / "result.med"
        export = directory / "baseline.export"
        export.write_text(
            f"P time_limit {remaining}\nP memory_limit {memory_mib}\nP mpi_nbcpu 1\nP ncpus {threads}\n"
            f"F comm {CASE.resolve()} D 1\nF libr {inputs_path} D 91\n"
            f"F libr {discrete} D 92\nF mmed {mesh_path} D 20\n"
            f"F libr {native} R 90\nF rmed {result} R 80\n", encoding="utf-8",
        )
        completed = run_solver(
            [installation["launcher_path"], "--wrkdir", str(native_work), str(export)], directory, remaining,
        )
        (directory / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (directory / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        native_log = native_work / "fort.6"
        logs = completed.stdout + completed.stderr
        if native_log.is_file():
            logs += native_log.read_text(encoding="utf-8")
        report["native_summary"] = verify_native_result(completed.returncode, logs, native, result)
        report["status"] = "finite_elastic_solve_pending_comparison"
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        if isinstance(error, NativeBaselineError):
            report["failure_code"] = error.code
        elif isinstance(error, subprocess.TimeoutExpired):
            report["failure_code"] = "developer_verification_timeout"
        raise
    finally:
        if (native_work / "fort.6").is_file():
            shutil.copyfile(native_work / "fort.6", directory / "native.fort6.log")
        report["elapsed_s"] = time.monotonic() - started
        (directory / "baseline_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_directory", type=Path)
    parser.add_argument("--threads", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--budget-s", type=int, default=540)
    parser.add_argument("--allow-extended-budget", action="store_true",
                        help="Use only with explicit approval for a development run longer than ten minutes")
    args = parser.parse_args()
    print(json.dumps(run(args.candidate_directory, budget_s=args.budget_s, threads=args.threads,
                         allow_extended_budget=args.allow_extended_budget), indent=2))


if __name__ == "__main__":
    main()
