"""Solve the configured wall-on-bedrock model in a fresh, isolated run directory."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .model_config import load_config
from .structure_assembly import mpi_launcher
from .three_body_diagnostic import interface_stiffness, prepare, pvpython_executable, solve


def default_output_root() -> Path:
    base = Path(os.environ["LOCALAPPDATA"]) if os.name == "nt" else Path.home() / ".local" / "share"
    return base / "Lamb" / "three-body-solver"


def export_reporting_region(result: Path, *, timeout: int | None = None) -> Path:
    output = result.parent.parent / "reporting"
    subprocess.run(
        [pvpython_executable(), str(Path(__file__).with_name("stress_reporting_region.py")),
         "--result", str(result), "--output", str(output)],
        check=True, timeout=timeout,
    )
    path = output / "structure_and_near_rock.vtu"
    if not path.is_file() or not (output / "stress_summary.json").is_file():
        raise RuntimeError("Reporting-region export did not produce its result and summary")
    return path


def run_job(root: Path, output: Path, report_path: Path, *, partitions: int = 4,
            timeout: int | None = None) -> dict:
    """Canonical app worker: the API owns the isolated job directory and status."""
    from .regions.combined_structure import audit_combined_structure, build_combined_structure
    config = _validate_run(root, partitions, timeout, prepare_only=False)
    if report_path.exists() or (output.exists() and any(output.iterdir())):
        raise ValueError("API job requires fresh output and report paths")
    print("Stage: meshing", flush=True)
    geometry = audit_combined_structure(build_combined_structure(root, split_wall_joints=True))
    prepare(root, output / "model", joints="spring", integrated=True)
    print("Stage: solving", flush=True)
    result = solve(output / "model", timeout=timeout, partitions=partitions)
    print("Stage: postprocessing", flush=True)
    reporting_result = export_reporting_region(result, timeout=timeout)
    summary_path = reporting_result.parent / "stress_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    concrete = [summary["bodies"]["wall"]]
    tension = max(entry["peak_tension_mpa"] for entry in concrete) * 1e6
    compression = -max(entry["peak_compression_mpa"] for entry in concrete) * 1e6
    strength = config["material"]["concrete"]["tensile_strength_pa"]
    diagnostic = json.loads((output / "model" / "diagnostic.json").read_text())
    if diagnostic.get("foundation", {}).get("present"):
        wall_foundation = diagnostic.get("wall_foundation_connection")
        foundation_bedrock = diagnostic.get("foundation_bedrock_connection")
        connection_assumption = (
            ("Bonded" if wall_foundation == "bonded" else "Spring-connected")
            + " wall/foundation interface; "
            + ("bonded" if foundation_bedrock == "bonded" else "spring-connected")
            + " foundation/bedrock interface"
        )
        active_support = connection_assumption + "; "
    elif diagnostic.get("wall_bedrock_connection") == "bonded":
        connection_assumption = "Wall perfectly bonded to the bedrock by shared seat nodes"
        active_support = "Bonded wall/bedrock seat; "
    else:
        connection_assumption = "Separate coincident seat nodes; linear springs carry tension, not frictional contact"
        active_support = "Configured wall/bedrock directional seat springs; "
    if diagnostic.get("wall_joints"):
        connection_assumption += (
            "; wall thickness blocks joined by linear springs "
            "(geometry.wall.joint_support) that carry tension"
        )
    active_support += (
        "rock bottom and sides on normal roller springs"
        if diagnostic.get("rock_outer_support", {}).get("model") == "rollers" else "fixed rock bottom"
    )
    inactive_inputs = [
        "geometry.wall.height_m: nominal metadata; the seat follows geometry.bedrock.contours",
        "outputs: selections do not change solver fields",
    ]
    if diagnostic.get("foundation_bedrock_connection") == "bonded":
        inactive_inputs.append(
            "geometry.bedrock.support.spring_n_per_m3: inactive for bonded foundation/bedrock interface"
        )
    elif diagnostic.get("wall_bedrock_connection") == "bonded":
        inactive_inputs.append("geometry.bedrock.support.spring_n_per_m3: inactive for bonded seat")
    report = {
        "material": geometry["material"],
        "loads": geometry["loads"],
        "wall_geometry": geometry["wall_geometry"],
        "maximum_applied_pressure_pa": geometry["maximum_applied_pressure_pa"],
        "model_kind": "three_body",
        "stress_result": str(reporting_result.resolve()),
        "full_domain_result": str(result.resolve()),
        "solver_input": str((output / "model" / "dam_model.sif").resolve()),
        "reporting_summary": str(summary_path.resolve()),
        "stress_bodies": summary["bodies"],
        "reporting_distance_m": summary["rock_reporting_distance_m"],
        "reporting_limitations": summary["limitations"],
        "foundation": diagnostic.get("foundation", {"present": False}),
        "wall_foundation_connection": diagnostic.get("wall_foundation_connection"),
        "foundation_bedrock_connection": diagnostic.get("foundation_bedrock_connection"),
        "spring_pairs": diagnostic.get("spring_pairs", {}),
        "solver_body_ids": diagnostic.get("solver_body_ids", {"wall": 1, "bedrock": 2}),
        "boundary_id_map": diagnostic.get("boundary_id_map", {}),
        "joint_gaps": diagnostic.get("joint_gaps", {}),
        "wall_connection": diagnostic.get("wall_connection", "single_monolithic_body"),
        "wall_joints": diagnostic.get("wall_joints", []),
        "wall_bedrock_connection": diagnostic.get("wall_bedrock_connection", "spring"),
        "connection_assumption": connection_assumption,
        "active_support": active_support,
        "rock_outer_support": diagnostic.get("rock_outer_support", {"model": "fixed_bottom"}),
        "inactive_inputs": inactive_inputs,
        "regions": diagnostic["cells_by_body"],
        "interface_stiffness_n_per_m3": diagnostic["interface_stiffness_n_per_m3"],
        "assumptions": diagnostic["assumptions"],
        "nodes": diagnostic["nodes"],
        "cells": sum(diagnostic["cells_by_body"].values()),
        "stress_analysis": {
            "scope": "wall only; foundation separately when present; bedrock separately within 5 m",
            "max_tensile_principal_stress_pa": {"value_pa": tension},
            "max_compressive_principal_stress_pa": {"value_pa": compression},
            "tensile_strength_pa": strength,
            "tension_utilization": tension / strength,
            "element_count": sum(entry["reporting_cells"] for entry in concrete),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Reporting result: {reporting_result}", flush=True)
    return report


def _validate_run(root: Path, partitions: int, timeout: int | None, *, prepare_only: bool) -> dict:
    if partitions < 1 or (timeout is not None and timeout <= 0):
        raise ValueError("Partitions and optional solver timeout must be positive")
    config = load_config(root / "config.json")
    interface_stiffness(config)
    if not prepare_only:
        for executable in ("ElmerGrid", "ElmerSolver_mpi"):
            if shutil.which(executable) is None:
                raise RuntimeError(f"{executable} unavailable")
        mpi_launcher()
        pvpython_executable()
    return config


def run(
    root: Path,
    output_root: Path,
    *,
    partitions: int = 4,
    timeout: int | None = None,
    prepare_only: bool = False,
) -> dict:
    _validate_run(root, partitions, timeout, prepare_only=prepare_only)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(
        prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=output_root,
    ))
    report = {"status": "preparing", "output_directory": str(output), "result": None}
    report_path = output / "run.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Three-body run: {output}", flush=True)
    try:
        prepare(root, output / "model", joints="spring", integrated=True)
        report["model_directory"] = str(output / "model")
        report["status"] = "prepared" if prepare_only else "solving"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not prepare_only:
            result = solve(output / "model", timeout=timeout, partitions=partitions)
            reporting_result = export_reporting_region(result, timeout=timeout)
            report.update(
                status="succeeded", result=str(result), reporting_result=str(reporting_result),
                stress_summary=str(reporting_result.parent / "stress_summary.json"),
            )
    except KeyboardInterrupt:
        report.update(status="cancelled", error="Run interrupted by user")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    except Exception as error:
        report.update(status="failed", error=str(error))
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] == "succeeded":
        latest = output / "latest-run.json"
        latest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        latest.replace(output_root / "latest-run.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, help="Optional explicit solver timeout; no default cap")
    parser.add_argument("--prepare-only", action="store_true", help="Build mesh/SIF without running Elmer")
    parser.add_argument("--config-root", type=Path, help="Isolated API job configuration directory")
    parser.add_argument("--output-dir", type=Path, help="Isolated API job output directory")
    parser.add_argument("--report-path", type=Path, help="Required API result report")
    args = parser.parse_args()
    if any(value is not None for value in (args.config_root, args.output_dir, args.report_path)):
        if not all(value is not None for value in (args.config_root, args.output_dir, args.report_path)):
            parser.error("--config-root, --output-dir and --report-path must be supplied together")
        if args.prepare_only:
            parser.error("--prepare-only is not an API job mode")
        run_job(args.config_root, args.output_dir, args.report_path,
                partitions=args.partitions, timeout=args.timeout_s)
        return
    print(json.dumps(run(
        args.root, args.output_root, partitions=args.partitions,
        timeout=args.timeout_s, prepare_only=args.prepare_only,
    ), indent=2))


if __name__ == "__main__":
    main()
