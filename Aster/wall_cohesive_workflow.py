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

import numpy as np

from Aster import automatic_cohesive, automatic_preparation

OUTPUT_ROOT = "aster_crack_submodel"
REPORT_NAME = "crack_report.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def candidate_face_ids(directory: Path) -> set[tuple[int, int, int]]:
    with np.load(directory / "cohesive_state.npz", allow_pickle=False) as state:
        faces = automatic_cohesive.candidate_faces(state)
    return {tuple(int(node) for node in face) for face in faces}


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
        automatic_preparation.run(config, model, baseline, screening_dir,
                                  job_id=job_id, relative_tolerance=relative_tolerance,
                                  candidate_scope="all_overstressed")
        print("[2/3] Splitting selected wall facets into cohesive interfaces ...", flush=True)
        aster_controls = submitted.get("fracture_debug", {}).get("aster", {})
        maximum_rounds = int(aster_controls.get("adaptive_rounds", 5))
        zero_tension_tolerance_pa = float(
            aster_controls.get("whole_wall_tension_tolerance_pa", 1.0))
        preparation = report = final_round_dir = None
        previous_faces = None
        growth_warning = None
        for round_index in range(maximum_rounds):
            print(f"[2/3] Preparing adaptive cohesive round {round_index + 1} ...", flush=True)
            candidates_dir = run_dir / f"prepared_candidates_{round_index + 1}"
            next_preparation = automatic_preparation.run(
                config, model, baseline, candidates_dir, job_id=job_id,
                relative_tolerance=relative_tolerance, screening=screening_dir,
                candidate_scope="all_overstressed")
            next_faces = candidate_face_ids(candidates_dir)
            if previous_faces is not None and not (next_faces - previous_faces):
                growth_warning = (
                    "Adaptive cracking stopped: overstressed cells produced no new usable cohesive "
                    "facets after candidate selection and crack-front constraints. The last solved "
                    "round remains partial; the unjointed overstressed cells cannot crack in that model.")
                print(growth_warning, flush=True)
                break
            preparation = next_preparation
            previous_faces = next_faces
            final_round_dir = run_dir / f"solve_round_{round_index + 1}"
            manifest["candidate_faces"] = int(preparation["candidate_faces"])
            manifest["adaptive_round"] = round_index + 1
            write_json(run_dir / "manifest.json", manifest)
            print(f"[3/3] Solving adaptive cohesive round {round_index + 1} ...", flush=True)
            report = automatic_cohesive.run(
                baseline.resolve(), candidates_dir.resolve(), final_round_dir.resolve(), time_limit_s)
            growth_rows = report.get("growth_source_rows", [])
            if not growth_rows or round_index + 1 >= maximum_rounds:
                break
            screening_json = screening_dir / "screening.json"
            screening_data = json.loads(screening_json.read_text(encoding="utf-8"))
            existing = set(screening_data.get("overstressed_source_rows", []))
            existing.update(growth_rows)
            screening_data["overstressed_source_rows"] = sorted(existing)
            write_json(screening_json, screening_data)
            screening_manifest_path = screening_dir / "manifest.json"
            screening_manifest = json.loads(screening_manifest_path.read_text(encoding="utf-8"))
            screening_manifest["screening_sha256"] = file_hash(screening_json)
            write_json(screening_manifest_path, screening_manifest)
        assert report is not None and final_round_dir is not None and preparation is not None
        report["adaptive_rounds_completed"] = manifest["adaptive_round"]
        if growth_warning is not None:
            report["adaptive_growth_warning"] = growth_warning
        report["engineering_goal"] = {
            "description": (
                "Crack the wall spontaneously until the wall is in numerical compression, "
                "while reporting maximum displacement and maximum cohesive opening."),
            "whole_wall_tension_tolerance_pa": zero_tension_tolerance_pa,
            "full_load": bool(math.isclose(
                report.get("reached_load_factor", 0.0), 1.0, abs_tol=1e-12)),
            "no_unjointed_tensile_growth": not bool(report.get("growth_source_rows")),
            "maximum_principal_tensile_stress_pa":
                report.get("maximum_principal_tensile_stress_pa"),
        }
        maximum_tension = report.get("maximum_principal_tensile_stress_pa")
        report["engineering_goal"]["zero_tension_reached"] = (
            isinstance(maximum_tension, (int, float))
            and math.isfinite(maximum_tension)
            and maximum_tension <= zero_tension_tolerance_pa)
        report["engineering_goal"]["passed"] = all((
            report["engineering_goal"]["full_load"],
            report["engineering_goal"]["no_unjointed_tensile_growth"],
            report["engineering_goal"]["zero_tension_reached"],
        ))
        if (report.get("accepted") is not True
                or not str(report.get("solver", "")).startswith("Code_Aster")
                or not report.get("paraview")
                or report.get("opening_assessment", {}).get("peak_opening_m") is None):
            raise RuntimeError("Discrete cohesive solve did not produce an accepted crack-opening result")
        write_json(run_dir / REPORT_NAME, report)
        goal_passed = report["engineering_goal"]["passed"]
        manifest.update(
            status=("completed_not_engineering_validated" if goal_passed
                    else "partial_not_engineering_validated"),
            reached_load_factor=float(report["reached_load_factor"]),
            peak_opening_m=float(report["opening_assessment"]["peak_opening_m"]),
            candidate_faces=int(preparation["candidate_faces"]),
            remaining_growth_source_rows=len(report.get("growth_source_rows", [])),
            engineering_goal_passed=goal_passed,
            warning=(None if goal_passed else " ".join(filter(None, (
                report.get("capacity_assessment", {}).get("message"),
                growth_warning,
            ))) or "The crack solve has not yet achieved the whole-wall zero-tension goal."),
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        manifest.update(status="failed", error=str(error))
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
    if result["status"] not in ("completed_not_engineering_validated",
                                "partial_not_engineering_validated"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
