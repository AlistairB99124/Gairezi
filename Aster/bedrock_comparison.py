"""Isolated, baseline-only rigid-foundation sensitivity comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from Aster.generate_three_body_case import generate
from Aster.run_case import run_case
from bedrock_comparison import without_bedrock, write_fields

ROOT = Path(__file__).resolve().parents[1]


def _linux_path(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.drive:
        raise ValueError("WSL diagnostic requires a local Windows drive path")
    return "/mnt/" + resolved.drive[0].lower() + resolved.as_posix()[2:]


def run(config: Path, model: Path, baseline: Path, output: Path, time_limit_s: int) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    manifest = {"schema_version": 1, "status": "running", "configuration_sha256": digest, "engineering_validated": False,
                "support": "fixed foundation/rock interface; rigid-support sensitivity, not equivalent bedrock"}
    manifest_path = output / "manifest.json"

    def persist():
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")

    persist()
    try:
        with np.load(baseline / "three_body_mesh_arrays.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        reduced = without_bedrock(arrays)
        source = json.loads((baseline / "mesh_manifest.json").read_text(encoding="utf-8"))
        for name in ("config.json", "three_body.msh"):
            if source["source_hashes"][name] != hashlib.sha256((model / name).read_bytes()).hexdigest():
                raise ValueError(f"Diagnostic Aster source differs from saved model: {name}")
        if Path(source["source_mesh"]).resolve() != (model / "three_body.msh").resolve():
            raise ValueError("Diagnostic baseline belongs to another source mesh")
        full_run = json.loads((baseline / "run_manifest.json").read_text(encoding="utf-8"))
        if full_run.get("status") != "completed" or full_run.get("returncode") != 0:
            raise ValueError("Diagnostic requires a completed Aster full-bedrock solve")
        manifest["source_hashes"] = {name: source["source_hashes"][name] for name in ("config.json", "three_body.msh")}
        manifest["baseline_arrays_sha256"] = hashlib.sha256((baseline / "three_body_mesh_arrays.npz").read_bytes()).hexdigest()
        persist()
        distribution = os.environ.get("LAMB_FENICSX_WSL_DISTRIBUTION", "Ubuntu")
        command = ["wsl.exe", "-d", distribution, "--cd", str(ROOT), "--exec", "python3", "-u",
                   "-m", "FEniCSx.bedrock_comparison",
                   "--config", _linux_path(config), "--model", _linux_path(model),
                   "--arrays", _linux_path(baseline / "three_body_mesh_arrays.npz"),
                   "--output", _linux_path(output)]
        subprocess.run(command, check=True)
        reports = json.loads((output / "fenicsx.json").read_text(encoding="utf-8"))
        rigid_dir = output / "aster_rigid"
        rigid_dir.mkdir()
        np.savez_compressed(rigid_dir / "three_body_mesh_arrays.npz", **reduced)
        source["rigid_seat_sensitivity"] = True
        source["support_springs"] = []
        (rigid_dir / "mesh_manifest.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
        generate(case_path=rigid_dir / "case.comm", inputs_path=rigid_dir / "case_inputs.json",
                 mesh_manifest_path=rigid_dir / "mesh_manifest.json")
        print("BEDROCK_COMPARISON aster_rigid: solving elastic baseline", flush=True)
        result = run_case(case_path=rigid_dir / "case.comm", output_dir=rigid_dir, mesh_dir=rigid_dir,
                          arrays_filename="three_body_mesh_arrays.npz",
                          converter_filename="convert_three_body_to_med.py",
                          export_name="rigid_seat_baseline", time_limit_s=time_limit_s)
        if result.get("status") != "completed" or result.get("returncode") != 0:
            raise RuntimeError(f"Aster rigid baseline failed: {result}")
        material = json.loads(config.read_text(encoding="utf-8"))["material"]["concrete"]
        for name, directory, data in (("aster_full", baseline, arrays), ("aster_rigid", rigid_dir, reduced)):
            export = output / name
            export.mkdir(exist_ok=True)
            with np.load(directory / "run" / "_diagnostic_displacement.npz", allow_pickle=False) as archive:
                reports[name] = write_fields(export, data, archive["displacement_m"], material)
        for solver in ("fenicsx", "aster"):
            full, rigid = reports[f"{solver}_full"], reports[f"{solver}_rigid"]
            rigid["difference_from_full"] = {
                body: {key: rigid["bodies"][body][key] - full["bodies"][body][key]
                       for key in full["bodies"][body]}
                for body in full["bodies"]}
        manifest.update(status="completed", variants=reports,
                        files=[f"{name}/{name}.vtu" for name in reports])
        persist()
        return manifest
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        manifest.update(status="failed", error=str(error))
        persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "model", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--time-limit-s", type=int, default=30 * 24 * 3600)
    args = parser.parse_args()
    run(args.config, args.model, args.baseline, args.output, args.time_limit_s)


if __name__ == "__main__":
    main()
