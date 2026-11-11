"""Isolated, sequential selected-material sweep targeting 2 mm normal opening."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
# Legacy degraded-element sweeps vary the selected material; lift-joint sweeps vary the
# weak construction-joint law. alpha (JOINT_MECA_RUPT, in [0, 2]) and friction_coefficient
# (JOINT_MECA_FROT, in [0.001, 10]) are absolute values, not factors.
VARIABLES = {"youngs_modulus", "tensile_strength_pa", "cohesive_fracture_energy_j_m2"}
LIFT_JOINT_VARIABLES = {"tensile_strength_pa", "cohesive_fracture_energy_j_m2"}
LIFT_JOINT_ABSOLUTE = {"alpha": (0.0, 2.0), "friction_coefficient": (1e-3, 10.0)}
NATIVE_TIME_LIMIT_S = 30 * 24 * 3600


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def positive(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _valid_friction(friction) -> bool:
    if friction is None:
        return True
    return (isinstance(friction, dict)
            and isinstance(friction.get("friction_coefficient"), (int, float))
            and not isinstance(friction.get("friction_coefficient"), bool)
            and friction.get("friction_coefficient") > 0
            and isinstance(friction.get("adhesion_pa", 0.), (int, float))
            and friction.get("adhesion_pa", 0.) >= 0)


def define_cases(config: dict, specification: dict) -> list[dict]:
    from Elmer.model_config import validate_config

    validate_config(config)
    if (specification.get("schema_version") != 1
            or specification.get("required_load_factor") != 1.0
            or specification.get("target_peak_normal_opening_m") != 0.002):
        raise ValueError("This sweep requires schema 1 and a 2 mm target at design load 1")
    selection = config.get("damage_selection", {})
    if (selection.get("enabled") is not True
            or selection.get("maximum_overload_factor", 1.0) != 1.0
            or not _valid_friction(selection.get("damaged_joint_friction"))
            or config.get("cracking")):
        raise ValueError("Sweep requires enabled cohesive selection, no manual cracks, and no overload")
    lift = selection.get("lift_joints")
    properties = selection.get("damaged_element_properties")
    # Which blocks may vary, with their factor variables and absolute-valued keys.
    blocks = {}
    if isinstance(lift, dict):
        blocks["lift_joints"] = (LIFT_JOINT_VARIABLES, LIFT_JOINT_ABSOLUTE)
    if isinstance(properties, dict):
        blocks["damaged_element_properties"] = (VARIABLES, {})
    if not blocks:
        raise ValueError("Baseline must define lift_joints and/or selected damaged-material properties")
    definitions = specification.get("cases")
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("At least the unmodified baseline control is required")

    def apply(name, derivative, block, factors):
        variables, absolute = blocks[block]
        if not isinstance(factors, dict) or set(factors) - variables - set(absolute):
            raise ValueError(f"{name}: only {sorted(variables | set(absolute))} of {block} may vary")
        assigned = derivative["damage_selection"][block]
        baseline_values = selection[block]
        for key, factor in factors.items():
            if key in absolute:
                low, high = absolute[key]
                if (isinstance(factor, bool) or not isinstance(factor, (int, float))
                        or not math.isfinite(factor) or not low <= factor <= high):
                    raise ValueError(f"{name}: {key} must lie in [{low:g}, {high:g}]")
                if key == "friction_coefficient" and baseline_values.get("law") != "JOINT_MECA_FROT":
                    raise ValueError(f"{name}: friction_coefficient needs a baseline with the JOINT_MECA_FROT lift law")
                assigned[key] = factor
                continue
            if not positive(factor) or factor > 1:
                raise ValueError(f"{name}: factors must be finite in (0, 1]")
            assigned[key] = baseline_values[key] * factor

    cases = []
    seen = set()
    for definition in definitions:
        name, factors = definition["id"], definition["factors"]
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,79}", name)
                or name in seen):
            raise ValueError(f"Unsafe or duplicate case ID: {name!r}")
        if not isinstance(factors, dict):
            raise ValueError(f"{name}: factors must be an object")
        derivative = deepcopy(config)
        nested = set(factors) & {"lift_joints", "damaged_element_properties"}
        if nested:
            # Both blocks present: factors are keyed by block.
            if set(factors) - nested:
                raise ValueError(f"{name}: mix of block-keyed and flat factors")
            for block in nested:
                if block not in blocks:
                    raise ValueError(f"{name}: the baseline has no {block} to vary")
                apply(name, derivative, block, factors[block])
            block = "+".join(sorted(nested))
        elif factors:
            if len(blocks) != 1:
                raise ValueError(f"{name}: with lift joints and degraded elements both configured, "
                                 "key the factors by block (lift_joints / damaged_element_properties)")
            block = next(iter(blocks))
            apply(name, derivative, block, factors)
        else:
            block = "+".join(sorted(blocks))
        validate_config(derivative)
        cases.append({"id": name, "factors": factors, "config": derivative, "block": block})
        seen.add(name)
    if cases[0]["factors"]:
        raise ValueError("First case must be the unchanged baseline control")
    return cases


def classify_report(report: dict, target_m: float = 0.002) -> dict:
    if not isinstance(report, dict):
        raise ValueError("Native report must be a JSON object")
    steps = report.get("steps") or []
    if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
        raise ValueError("Native report steps must be an array of objects")
    final = steps[-1] if steps else {}
    opening = final.get("peak_opening_m")
    reached = final.get("load_factor")
    valid_opening = (isinstance(opening, (int, float)) and not isinstance(opening, bool)
                     and math.isfinite(opening) and opening >= 0)
    full_load = (positive(reached) and math.isclose(reached, 1., rel_tol=0., abs_tol=1e-9)
                 and positive(report.get("reached_load_factor"))
                 and math.isclose(report["reached_load_factor"], 1., rel_tol=0., abs_tol=1e-9))
    accepted = (report.get("accepted") is True and report.get("status") == "completed"
                and report.get("solve_status") == "completed" and report.get("returncode") == 0
                and report.get("assessment_authority") == "code_aster"
                and report.get("analysis_scope") == "whole_model" and full_load and valid_opening)
    return {
        "status": ("target_reached" if opening >= target_m else "converged_below_target")
        if accepted else "incomplete_or_failed",
        "target_reached": bool(accepted and opening >= target_m),
        "accepted_full_design_load": accepted,
        "reached_load_factor": reached,
        "peak_normal_opening_mm": opening * 1000 if valid_opening else None,
        "initiated_faces": final.get("initiated_faces"),
        "opened_faces": final.get("opened_faces"),
        "separated_faces": final.get("separated_faces"),
        "opened_facet_area_m2": final.get("opened_area_m2"),
        "maximum_wall_displacement_mm": (
            report["maximum_wall_displacement_m"] * 1000
            if positive(report.get("maximum_wall_displacement_m")) else None),
        "error": None if accepted else (
            report.get("solve_message") or "No accepted, finite whole-model equilibrium at design load 1"),
    }


def materialize_case(case: dict, directory: Path, prepared: Path, baseline_config: Path) -> Path:
    directory.mkdir()
    candidate = directory / "candidate"
    shutil.copytree(prepared, candidate)
    config_path = directory / "config.json"
    write_json(config_path, case["config"])
    summary = read_json(candidate / "summary.json")
    assignment = summary.get("material_assignment")
    if assignment is None or not assignment.get("selected_source_rows"):
        raise ValueError("Sweep requires a nonempty fixed selected region")
    selection = case["config"]["damage_selection"]
    # Every block the derivative carries is re-applied; the fixed selection never changes.
    if "lift_joints" in selection:
        if "joint_properties" not in assignment:
            raise ValueError("Lift-joint sweep requires candidates prepared with lift joints")
        joints = selection["lift_joints"]
        assignment["joint_properties"].update(
            {key: joints[key] for key in ("tensile_strength_pa", "cohesive_fracture_energy_j_m2", "alpha",
                                          "pena_contact", "law", "friction_coefficient", "uplift") if key in joints},
            critical_opening_m=2.0 * joints["cohesive_fracture_energy_j_m2"] / joints["tensile_strength_pa"])
        if joints.get("law") == "JOINT_MECA_FROT":
            # The Mohr-Coulomb cone's apex stays at the (varied) tensile strength.
            assignment["joint_properties"]["adhesion_pa"] = (
                joints["friction_coefficient"] * joints["tensile_strength_pa"])
    elif "joint_properties" in assignment:
        raise ValueError("Candidates were prepared with lift joints the derivative no longer configures")
    if "damaged_element_properties" in selection:
        if "properties" not in assignment:
            raise ValueError("Degraded-element sweep requires candidates prepared with damaged_element_properties")
        assignment["properties"] = deepcopy(selection["damaged_element_properties"])
        # The degraded facets' CZM follows the degraded material; the lift law travels separately.
        summary.update(tensile_strength_pa=assignment["properties"]["tensile_strength_pa"],
                       fracture_energy_j_m2=assignment["properties"]["cohesive_fracture_energy_j_m2"])
    elif "properties" in assignment:
        raise ValueError("Candidates were prepared with degraded elements the derivative no longer configures")
    provenance = {
        "baseline_config_sha256": file_hash(baseline_config),
        "derivative_config_sha256": file_hash(config_path),
        "fixed_candidate_state_sha256": file_hash(prepared / "cohesive_state.npz"),
        "fixed_screening_sha256": file_hash(prepared / "screening.json")
        if (prepared / "screening.json").exists() else summary["screening_sha256"],
        "case_id": case["id"], "factors": case["factors"],
        "baseline_reuse": "Unmodified intact baseline and selection; selected properties changed after screening",
        "config_digest_scope": "Original solved baseline; derivative config is separately hashed, not a new elastic solve",
        "engineering_validated": False,
    }
    summary["sweep_provenance"] = provenance
    write_json(candidate / "summary.json", summary)
    manifest = read_json(candidate / "manifest.json")
    manifest.update(summary_sha256=file_hash(candidate / "summary.json"), sweep_provenance=provenance)
    write_json(candidate / "manifest.json", manifest)
    write_json(directory / "provenance.json", provenance)
    return candidate


def run_module(module: str, arguments: list[str], log: Path, allowed_codes: tuple[int, ...] = (0,)) -> int:
    command = [sys.executable, "-u", "-B", "-m", module, *arguments]
    print(f"Running {module}; log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write(subprocess.list2cmdline(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        try:
            code = process.wait()
        except KeyboardInterrupt:
            if os.name == "nt":
                stopped = subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                                         stdout=stream, stderr=subprocess.STDOUT, check=False)
                if stopped.returncode and process.poll() is None:
                    raise RuntimeError(f"Unable to stop sweep worker {process.pid}; see {log}") from None
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if code not in allowed_codes:
        raise RuntimeError(f"{module} exited {code}; see {log}")
    return code


def prepare_reference(run: Path) -> tuple[Path, Path]:
    from Aster.automatic_preparation import baseline_inventory

    run_module("Elmer.three_body_solver", [
        "--config-root", str(run), "--output-dir", str(run / "elmer"),
        "--report-path", str(run / "elmer_report.json"),
    ], run / "elmer.log")
    baseline = run / "baseline"
    pipeline_code = run_module("Aster.run_three_body_pipeline", [
        "--source-mesh", str(run / "elmer" / "model" / "three_body.msh"),
        "--output-dir", str(baseline), "--runtime", "wsl" if os.name == "nt" else "native",
        "--time-limit-s", str(NATIVE_TIME_LIMIT_S),
    ], run / "baseline.log", allowed_codes=(0, 2))
    baseline_inventory(baseline, run / "config.json", run / "elmer" / "model")
    write_json(run / "baseline_status.json", {
        "status": "completed", "pipeline_exit_code": pipeline_code,
        "comparison_requires_verification": pipeline_code == 2, "engineering_validated": False,
    })
    if pipeline_code == 2:
        print("Baseline completed; cross-solver comparison requires verification (see baseline.log).", flush=True)
    shared = ["--config", str(run / "config.json"), "--model", str(run / "elmer" / "model"),
              "--baseline", str(baseline), "--job-id", run.name, "--relative-tolerance", "0.001"]
    run_module("Aster.automatic_preparation", [*shared, "--output", str(run / "screening")],
               run / "screening.log")
    prepared = run / "prepared"
    run_module("Aster.automatic_preparation", [
        *shared, "--screening", str(run / "screening"), "--output", str(prepared),
    ], run / "preparation.log")
    if read_json(prepared / "summary.json").get("status") != "prepared_aster_candidates":
        raise ValueError("Native candidate preparation did not complete")
    return baseline, prepared


def save_results(run: Path, state: dict) -> None:
    write_json(run / "sweep.json", state)
    fields = ("case_id", "block", "status", "target_reached", "accepted_full_design_load",
              "youngs_modulus_pa", "tensile_strength_pa", "fracture_energy_j_m2",
              "lift_tensile_strength_pa", "lift_fracture_energy_j_m2", "alpha",
              "peak_normal_opening_mm", "reached_load_factor", "initiated_faces", "opened_faces",
              "separated_faces", "opened_facet_area_m2", "maximum_wall_displacement_mm", "error")
    temporary = run / "results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(state["results"])
    temporary.replace(run / "results.csv")


def execute(run: Path, cases: list[dict], candidate_root: Path | None = None) -> dict:
    candidate_root = candidate_root or ROOT / "candidate"
    state = {"status": "running", "target_peak_normal_opening_m": .002,
             "required_load_factor": 1., "engineering_validated": False,
             "results": [], "active_case": None}
    save_results(run, state)
    try:
        baseline, prepared = prepare_reference(run)
        immutable = {path: file_hash(path) for directory in (baseline, prepared, run / "screening")
                     for path in directory.iterdir() if path.is_file()}
        immutable[run / "config.json"] = file_hash(run / "config.json")
        for case in cases:
            for path, digest in immutable.items():
                if file_hash(path) != digest:
                    raise ValueError(f"Shared reference changed during sweep: {path}")
            state["active_case"] = case["id"]
            save_results(run, state)
            print(f"Starting {case['id']}", flush=True)
            directory = run / "cases" / case["id"]
            candidate = materialize_case(case, directory, prepared, run / "config.json")
            selection = case["config"]["damage_selection"]
            lift, elements = selection.get("lift_joints") or {}, selection.get("damaged_element_properties") or {}
            result = {"case_id": case["id"], "block": case["block"],
                      "youngs_modulus_pa": elements.get("youngs_modulus"),
                      "tensile_strength_pa": elements.get("tensile_strength_pa"),
                      "fracture_energy_j_m2": elements.get("cohesive_fracture_energy_j_m2"),
                      "lift_tensile_strength_pa": lift.get("tensile_strength_pa"),
                      "lift_fracture_energy_j_m2": lift.get("cohesive_fracture_energy_j_m2"),
                      "alpha": lift.get("alpha")}
            try:
                code = run_module("Aster.automatic_cohesive", [
                    "--baseline-dir", str(baseline), "--fenicsx-output", str(candidate),
                    "--output-dir", str(directory / "aster"),
                ], directory / "case.log", allowed_codes=(0, 2))
                report = read_json(directory / "aster" / "aster_cohesive_report.json")
                if (case["config"]["damage_selection"].get("crack_water_pressure") is True
                        and report.get("status") == "completed"):
                    from Aster.hydraulic_loads import validate_crack_water_evidence
                    validate_crack_water_evidence(report)
                result.update(classify_report(report))
                if code != 0 and result["accepted_full_design_load"]:
                    raise ValueError("Native exit code contradicts the accepted report")
            except (OSError, ValueError, RuntimeError) as error:
                result.update(status="execution_failed", target_reached=False,
                              accepted_full_design_load=False, error=f"{type(error).__name__}: {error}")
                print(f"{case['id']}: {result['error']}", file=sys.stderr, flush=True)
            write_json(directory / "result.json", result)
            state["results"].append(result)
            state["active_case"] = None
            save_results(run, state)
            print(f"{case['id']}: {result['status']}; opening_mm={result.get('peak_normal_opening_mm')}", flush=True)
            if result["target_reached"]:
                candidate_root.mkdir(parents=True, exist_ok=True)
                shutil.copy2(directory / "config.json", candidate_root / "config.json")
                print(f"Copied successful config from {case['id']} to {candidate_root / 'config.json'}", flush=True)
                print("Accepted 2 mm target reached at full design load; stopping the sweep.", flush=True)
                break
        hits = [row["case_id"] for row in state["results"] if row["target_reached"]]
        failures = [row["case_id"] for row in state["results"] if not row["accepted_full_design_load"]]
        state.update(status="completed_with_failures" if failures else "completed",
                     target_cases=hits, failed_cases=failures,
                     stop_reason="target_reached" if hits else "cases_exhausted",
                     skipped_cases=[case["id"] for case in cases[len(state["results"]):]])
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        raise
    except (OSError, ValueError, RuntimeError):
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        save_results(run, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--scenarios", type=Path, default=Path(__file__).with_name("scenarios.json"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).with_name("runs"))
    parser.add_argument("--execute", action="store_true", help="Launch solvers; otherwise materialize a plan only")
    args = parser.parse_args()
    config, specification = read_json(args.config), read_json(args.scenarios)
    cases = define_cases(config, specification)
    args.output_root.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run = (args.output_root / name).resolve()
    run.mkdir()
    (run / "cases").mkdir()
    write_json(run / "config.json", config)
    write_json(run / "scenarios.json", specification)
    write_json(run / "plan.json", {
        "status": "defined", "source_config": str(args.config.resolve()),
        "source_config_sha256": file_hash(args.config), "snapshot_sha256": file_hash(run / "config.json"),
        "target_peak_normal_opening_m": .002, "required_load_factor": 1.,
        "selection_policy": "Screen once from the unchanged baseline; reuse exact cells/facets for every case",
        "fixed_support_degradation": config["damage_selection"].get("support_degradation"),
        "fixed_crack_base": config["damage_selection"].get("crack_base", False),
        "crack_water_pressure": config["damage_selection"].get("crack_water_pressure", False),
        "engineering_validated": False, "cases": cases,
    })
    print(f"Sweep directory: {run}\nDefined {len(cases)} cases; target >= 2 mm at load factor 1.", flush=True)
    if not args.execute:
        print("Plan only: no solvers launched. Add --execute to run a fresh campaign.", flush=True)
        return 0
    state = execute(run, cases)
    print(f"Finished: {state['status']}; target cases: {state['target_cases']}; results: {run / 'results.csv'}")
    return 2 if state["failed_cases"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Sweep interrupted; completed evidence retained.", file=sys.stderr)
        raise SystemExit(130)
