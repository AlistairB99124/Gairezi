"""Run the wall-only discrete cohesive workflow after an accepted Aster baseline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import math

from Aster import automatic_cohesive, automatic_preparation

OUTPUT_ROOT = "aster_crack_submodel"
REPORT_NAME = "crack_report.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(*, config: Path, model: Path, baseline: Path, output: Path, job_id: str,
        time_limit_s: int, relative_tolerance: float = 1e-3) -> dict:
    submitted = json.loads(config.read_text(encoding="utf-8"))
    started = time.monotonic()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output.resolve() / OUTPUT_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "solver": "code_aster",
        "analysis_scope": "wall_cohesive",
        "engineering_validated": False,
        "source_job_id": job_id,
        "configuration_sha256": file_hash(config),
        "source_mesh_sha256": file_hash(model / "three_body.msh"),
        "started_at_utc": timestamp,
    }
    write_json(run_dir / "manifest.json", manifest)
    try:
        screening_dir = run_dir / "screening"
        print("[1/3] Screening the accepted Code_Aster baseline for tensile wall facets ...", flush=True)
        screening = automatic_preparation.run(
            config, model, baseline, screening_dir, job_id=job_id,
            relative_tolerance=relative_tolerance, candidate_scope="all_overstressed")
        print("[2/3] Splitting selected wall facets into cohesive interfaces ...", flush=True)
        aster_controls = (submitted.get("fracture_debug") or {}).get("aster") or {}
        zero_tension_tolerance_pa = float(
            aster_controls.get("whole_wall_tension_tolerance_pa", 1.0))
        growth_warning = None
        print("[2/3] Preparing cohesive round 1 ...", flush=True)
        candidates_dir = run_dir / "prepared_candidates_1"
        preparation = automatic_preparation.run(
            config, model, baseline, candidates_dir, job_id=job_id,
            relative_tolerance=relative_tolerance, screening=screening_dir,
            candidate_scope="all_overstressed")
        final_round_dir = run_dir / "solve_round_1"
        manifest["candidate_faces"] = int(preparation["candidate_faces"])
        manifest["adaptive_round"] = 1
        write_json(run_dir / "manifest.json", manifest)
        print("[3/3] Solving cohesive round 1 ...", flush=True)
        round_started = time.monotonic()
        report = automatic_cohesive.run(
            baseline.resolve(), candidates_dir.resolve(), final_round_dir.resolve(), time_limit_s)
        growth_rows = report.get("growth_source_rows", [])
        full_load = math.isclose(report.get("reached_load_factor", 0.0), 1.0, rel_tol=0.0, abs_tol=1e-12)
        maximum_tension = report.get("maximum_principal_tensile_stress_pa")
        target_reached = (isinstance(maximum_tension, (int, float))
                          and math.isfinite(maximum_tension)
                          and maximum_tension <= zero_tension_tolerance_pa)
        iterations = [{
            "iteration": 1,
            "candidate_faces": int(preparation["candidate_faces"]),
            "solver_status": report.get("status"),
            "accepted": report.get("accepted") is True,
            "reached_load_factor": report.get("reached_load_factor"),
            "maximum_principal_tensile_stress_pa": report.get("maximum_principal_tensile_stress_pa"),
            "maximum_displacement_m": report.get("maximum_wall_displacement_m"),
            "residual_tension_assessment": report.get("residual_tension_assessment"),
            "opening_assessment": report.get("opening_assessment"),
            "duration_s": time.monotonic() - round_started,
            "result_directory": str(final_round_dir),
        }]
        if report.get("accepted") is not True or report.get("status") != "completed" or not full_load:
            stop_reason = "solver_failure_or_incomplete_load"
        elif growth_rows and not target_reached:
            stop_reason = "fracture_history_transfer_unavailable"
            growth_warning = (
                "Further overstressed wall cells require new cohesive facets, but this workflow "
                "cannot transfer the accepted fracture history to a rebuilt mesh. It stopped "
                "instead of restarting from an undamaged state. The accepted round is preserved.")
            print(growth_warning, flush=True)
        else:
            stop_reason = "assessment_complete"
        report["adaptive_rounds_completed"] = manifest["adaptive_round"]
        report["iterations"] = iterations
        report["baseline_assessment"] = screening.get("residual_tension_assessment")
        report["stop_reason"] = stop_reason
        if growth_warning is not None:
            report["adaptive_growth_warning"] = growth_warning
        report["engineering_goal"] = {
            "description": (
                "Assess the maximum wall-cell principal tensile stress against the configured "
                "residual stress target without changing the material tensile strength."),
            "whole_wall_tension_tolerance_pa": zero_tension_tolerance_pa,
            "full_load": bool(math.isclose(
                report.get("reached_load_factor", 0.0), 1.0, rel_tol=0.0, abs_tol=1e-12)),
            "no_unjointed_tensile_growth": not bool(report.get("growth_source_rows")),
            "maximum_principal_tensile_stress_pa":
                report.get("maximum_principal_tensile_stress_pa"),
        }
        report["near_base_crack_assessment"] = {
            "target_pa": 500000.0,
            "status": ("scenario_not_specified" if report["engineering_goal"]["full_load"]
                       else "incomplete"),
            "full_load": report["engineering_goal"]["full_load"],
            "message": ("A near-base crack's location, extent and face identities have not been "
                        "specified; all open candidate facets are reported separately and cannot "
                        "be treated as the requested near-base scenario."),
        }
        report["engineering_goal"]["zero_tension_reached"] = target_reached
        report["engineering_goal"]["residual_target_reached"] = report["engineering_goal"]["zero_tension_reached"]
        report["engineering_goal"]["passed"] = all((
            report.get("accepted") is True,
            report.get("status") == "completed",
            report["engineering_goal"]["full_load"],
            report["engineering_goal"]["zero_tension_reached"],
        ))
        goal_passed = report["engineering_goal"]["passed"]
        if not goal_passed and stop_reason == "assessment_complete":
            report["stop_reason"] = "residual_tension_target_not_met"
        write_json(run_dir / REPORT_NAME, report)
        if (report.get("accepted") is not True
                or not str(report.get("solver", "")).startswith("Code_Aster")
                or not report.get("paraview")
                or report.get("opening_assessment", {}).get("peak_opening_m") is None):
            raise RuntimeError("Discrete cohesive solve did not produce an accepted crack-opening result")
        manifest.update(
            status=("completed_not_engineering_validated" if goal_passed
                    else "partial_not_engineering_validated"),
            reached_load_factor=float(report["reached_load_factor"]),
            peak_opening_m=float(report["opening_assessment"]["peak_opening_m"]),
            candidate_faces=int(preparation["candidate_faces"]),
            remaining_growth_source_rows=len(report.get("growth_source_rows", [])),
            engineering_goal_passed=goal_passed,
            stop_reason=report["stop_reason"],
            warning=(None if goal_passed else " ".join(filter(None, (
                report.get("capacity_assessment", {}).get("message"),
                growth_warning,
            ))) or "The crack solve has not achieved the configured residual tensile-stress target."),
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, MemoryError) as error:
        detail = str(error) or type(error).__name__
        manifest.update(status="failed", error=detail,
                        stop_reason=("resource_limit" if isinstance(error, MemoryError)
                                     or "resource limit" in detail.lower() else "execution_failure"))
        print(f"Discrete cohesive wall cracking failed: {error}", flush=True)
    manifest["runtime_s"] = time.monotonic() - started
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--time-limit-s", type=int, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    result = run(config=args.config, model=args.model, baseline=args.baseline,
                 output=args.output, job_id=args.job_id, time_limit_s=args.time_limit_s,
                 relative_tolerance=args.relative_tolerance)
    print(json.dumps(result, indent=2))
    if result["status"] != "completed_not_engineering_validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
