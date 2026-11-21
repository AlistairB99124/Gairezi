"""Native Code_Aster no-tension damage analysis of the single concrete wall.

The complete concrete wall is linear TETRA4 ENDO_ISOT_BETON and starts intact.
The solve follows the nonlinear damage path with Newton equilibrium iterations.
Foundation and bedrock volumes are omitted; their configured series
compliances are represented by one exact distributed spring operator at the
wall base. There is no selected seed, crack plane, joint damage, or prescribed
crack-face pressure.
It runs in WSL with Linux Code_Aster because the Windows MUMPS build returns
NaN on this mesh and its MULT_FRONT factorisation is too slow for Newton steps.
The result is native ENDO_ISOT_BETON damage, not AT2 damage or a crack opening.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASE = Path(__file__).with_name("cases") / "whole_model_crack.comm"
CONVERTER = Path(__file__).with_name("convert_whole_model_to_med.py")
EXTRACTOR = Path(__file__).with_name("extract_whole_model_crack.py")
MESHER = Path(__file__).with_name("whole_model_crack_mesh.py")
# Kept from the local submodel so the API, progress monitor and desktop app find the run.
OUTPUT_ROOT_NAME = "aster_crack_submodel"
SCENARIO_NAME = "Aster_Full_Wall_Damage"
DEFAULT_DAMAGE_THRESHOLD = 0.95
NO_TENSION_STRENGTH_FRACTION = 1.0
RESIDUAL_TENSION_FRACTION = 0.00001
SOLVER_LOAD_INCREMENT = 0.01
SOFTENING_SLOPE_FRACTION = 0.10
LOAD_FACTORS = tuple(round(0.1 * step, 10) for step in range(1, 11))
DEFAULT_THREADS = 8
# The single Code_Aster damage solve may use most of a 24 GB WSL allocation.
DEFAULT_MEMORY_MIB = 20480
# run_aster's configured tmpdir is /tmp, which WSL backs with RAM (tmpfs); MUMPS out-of-core
# files written there consume the memory they are meant to relieve, so work on the ext4 disk.
LINUX_WORK_ROOT = "/var/tmp/lamb_aster_crack"
UNBOUNDED_S = 30 * 24 * 3600
LIVE_LOG_NAME = "solver.live.log"
DEFAULT_WSL_DISTRIBUTION = "Ubuntu"
ROOT_ENVIRONMENT = "LAMB_LINUX_CODE_ASTER_ROOT"
DISTRIBUTION_ENVIRONMENT = "LAMB_ASTER_WSL_DISTRIBUTION"
CONVERSION_TIMEOUT_S = 1800
INTERFACE_OPERATOR = "consistent_two_sided_TRI3_TRI6"
WALL_BASE_INTERFACE_OPERATOR = "consistent_one_sided_TRI3_TRI6"


def wsl_path(path: Path) -> str:
    """Windows path to its /mnt/<drive>/... form; POSIX paths pass through."""
    text = str(Path(path).resolve()) if os.name == "nt" else str(path)
    windows = PureWindowsPath(text)
    drive = windows.drive
    if not drive:
        return text.replace("\\", "/")
    if len(drive) != 2 or drive[1] != ":":
        raise ValueError("WSL paths require a local drive, not a UNC share")
    rest = windows.as_posix()[2:]
    return f"/mnt/{drive.rstrip(':').lower()}{rest.replace(chr(92), '/')}"


def linux_environment(root: str, threads: int) -> list[str]:
    """``env`` arguments for the relocated Code_Aster runtime (not the Windows PATH)."""
    # run_aster scratches its JEVEUX bases under mkdtemp(); WSL's /tmp is a small
    # RAM-backed tmpfs that large joint models overflow, so point temporary files
    # at the persistent disk. run_aster still removes the directory afterwards.
    return [f"PYTHONHOME={root}", f"LD_LIBRARY_PATH={root}/lib:{root}/lib64",
            f"PATH={root}/bin:/usr/bin:/bin", f"OMP_NUM_THREADS={threads}",
            "PYTHONUNBUFFERED=1", "GFORTRAN_UNBUFFERED_PRECONNECTED=y",
            "TMPDIR=/var/tmp"]


def linux_command(distribution: str, root: str, directory: Path, program: list[str],
                  threads: int = DEFAULT_THREADS) -> list[str]:
    return ["wsl.exe", "-d", distribution, "--cd", wsl_path(directory), "--exec", "env",
            *linux_environment(root, threads), *program]


def find_linux_installation(distribution: str) -> str:
    """Install root from ``LAMB_LINUX_CODE_ASTER_ROOT`` or the newest ``~/opt/code_aster_*``."""
    configured = os.environ.get(ROOT_ENVIRONMENT)
    probe = ('for d in "$HOME"/opt/code_aster_*; do [ -x "$d/bin/run_aster" ] && echo "$d"; done'
             if not configured else f'[ -x "{configured}/bin/run_aster" ] && echo "{configured}"')
    try:
        found = subprocess.run(["wsl.exe", "-d", distribution, "--exec", "sh", "-c", probe],
                               capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"WSL distribution {distribution} is not available: {error}") from error
    roots = sorted(line.strip() for line in found.stdout.splitlines() if line.strip().startswith("/"))
    if not roots:
        raise RuntimeError(
            f"Linux Code_Aster was not found in WSL {distribution}; install it under ~/opt/code_aster_<version> "
            f"or set {ROOT_ENVIRONMENT} to its install root")
    return roots[-1]


def spring_inputs(mesh_manifest: dict) -> tuple[list[dict], list[dict]]:
    """Baseline DIS_T stiffnesses; K_T_N is encoded column-major as in the baseline case."""
    wall = [{"group": spring["group"], "stiffness_n_per_m": [float(v) for v in spring["stiffness_n_per_m"]]}
            for spring in mesh_manifest["wall_foundation_springs"]]
    support = [{"group": spring["group"],
                "matrix_column_major_n_per_m": [float(spring["matrix_n_per_m"][i][j])
                                                for j in range(3) for i in range(3)]}
               for spring in mesh_manifest["support_springs"]]
    for spring in wall + support:
        values = spring.get("stiffness_n_per_m") or spring["matrix_column_major_n_per_m"]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite spring stiffness in {spring['group']}")
    return wall, support


def full_wall_damage_metrics(damage_archive: Path, radius_m: float, threshold: float = 0.95) -> dict:
    """Report native whole-wall damage point/extent metrics without a seed."""
    with np.load(damage_archive, allow_pickle=False) as data:
        xyz, damage, times = (data[key] for key in ("coordinates_m", "damage", "times"))
        point_ids = data["damage_point_ids"] if "damage_point_ids" in data.files else None
    if not (0.0 < threshold < 1.0):
        raise ValueError("Damage threshold must lie strictly between zero and one")
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    chainage = radius_m * np.arctan2(xyz[:, 0], xyz[:, 1])
    history = []
    for instant, values in zip(times, damage):
        cracked = values >= threshold
        damaged_cells = (len(np.unique(point_ids[cracked, 0])) if point_ids is not None and cracked.any() else 0)
        entry = {
            "time": float(instant), "load_factor": float(instant),
            "maximum_damage": float(values.max()),
            "points_above_threshold": int(cracked.sum()),
            "cells_above_threshold": int(damaged_cells),
        }
        if cracked.any():
            entry["cracked_radial_extent_m"] = [
                float(radius[cracked].min()), float(radius[cracked].max())]
            entry["cracked_chainage_m"] = [
                float(chainage[cracked].min()), float(chainage[cracked].max())]
            entry["cracked_elevation_extent_m"] = [
                float(xyz[cracked, 2].min()), float(xyz[cracked, 2].max())]
        history.append(entry)
    return {
        "basis": "native ENDO_ISOT_BETON V1 at WALL Gauss points; not crack area or opening",
        "threshold": threshold, "wall_gauss_points": int(len(xyz)),
        "history": history, "final": history[-1],
    }


def equivalent_wall_base_support(config: dict) -> dict:
    """Equivalent per-area spring for the removed wall/foundation/bedrock stack."""
    geometry = config["geometry"]
    concrete = config["material"]["concrete"]
    layers = []
    wall_only_body = bool((config.get("bedrock_analysis") or {}).get("wall_only_body"))
    foundation = geometry.get("foundation")
    if foundation is not None and not wall_only_body:
        support = foundation.get("support") or {}
        if support.get("model") == "spring":
            values = support["spring_n_per_m3"]
            interface = np.diag([float(values[axis]) for axis in ("x", "y", "z")])
            layers.append(("wall_foundation_interface", interface))
        elif support.get("model") != "bonded":
            raise ValueError("Foundation support must be bonded or spring for a wall-only model")
        depth = float(foundation["depth_m"])
        youngs, nu = float(concrete["youngs_modulus"]), float(concrete["poisson_ratio"])
        shear = youngs / (2.0 * (1.0 + nu))
        constrained = youngs * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        layers.append(("concrete_foundation_slab",
                       np.diag([shear / depth, shear / depth, constrained / depth])))
    bedrock_analysis = config.get("bedrock_analysis") or {}
    if bedrock_analysis.get("mode") == "elastic_layer":
        values = bedrock_analysis["spring_n_per_m3"]
        rock = np.diag([float(values[axis]) for axis in ("x", "y", "z")])
    else:
        support = geometry["bedrock"].get("support") or {}
        if support.get("model") != "distributed_elastic_granite" \
                and support.get("model") != "spring":
            raise ValueError(
                "Wall-only model needs configured bedrock spring compliance; "
                "a bonded or fixed seat cannot be represented by springs")
        values = support["spring_n_per_m3"]
        rock = np.diag([float(values[axis]) for axis in ("x", "y", "z")])
    layers.append(("bedrock_support", rock))
    if any(not np.isfinite(tensor).all() or np.linalg.eigvalsh(tensor).min() <= 0.0
           for _, tensor in layers):
        raise ValueError("Wall-only support layers must be positive finite tensors")
    inverse = sum((np.linalg.inv(tensor) for _, tensor in layers), np.zeros((3, 3)))
    equivalent = np.linalg.inv(inverse)
    return {
        "matrix_n_per_m3": equivalent,
        "components": [{"name": name, "matrix_n_per_m3": matrix.tolist()}
                      for name, matrix in layers],
        "equivalent_n_per_m3": {axis: float(equivalent[index, index])
                                for index, axis in enumerate(("x", "y", "z"))},
        "basis": (
            "configured direct elastic-layer wall-base springs; no foundation or bedrock volume retained"
            if wall_only_body else
            "series per-area translational compliances; no foundation or bedrock volume retained"
        ),
    }


def prepare(directory: Path, dam_config: Path, source_mesh: Path) -> dict:
    """Export the baseline arrays, elevate the complete wall and write inputs."""
    from Aster.build_three_body_mesh_arrays import export
    from Aster.fracture_material import material_contract
    from Aster.generate_three_body_case import render_comm
    from Aster.whole_model_crack_mesh import archive_wall_only, build_wall_only
    from Elmer.model_config import load_config

    config_bytes = dam_config.read_bytes()
    source = directory / "baseline_mesh"
    manifest = export(output_dir=source, source_mesh=source_mesh)
    _, case_inputs = render_comm(ROOT, source / "mesh_manifest.json")
    config = load_config(dam_config)
    aster_config = config.get("fracture_debug", {}).get("aster", {})
    no_tension_strength_fraction = float(
        aster_config.get("no_tension_strength_fraction", NO_TENSION_STRENGTH_FRACTION))
    residual_tension_fraction = float(
        aster_config.get("residual_tension_fraction", RESIDUAL_TENSION_FRACTION))
    with np.load(source / "three_body_mesh_arrays.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files if name != "interface_names"}
    if len(arrays.get("tetra4_mortar", ())) != 0:
        raise ValueError("Full-wall damage does not yet support a separate mortar-fill material")
    support = equivalent_wall_base_support(config)
    built = build_wall_only(arrays, support["matrix_n_per_m3"])
    np.savez(directory / "whole_model_mesh.npz", **archive_wall_only(built))
    (source / "three_body_mesh_arrays.npz").unlink()
    material = material_contract(config)
    material.update({
        "model": "ENDO_ISOT_BETON",
        "modelisation": "3D",
        "solution_method": "NEWTON",
        "post_peak_slope_pa":
            -SOFTENING_SLOPE_FRACTION * float(material["parameters"]["E"]),
    })
    material["parameters"]["NO_TENSION_SYT"] = (
        no_tension_strength_fraction * float(material["parameters"]["FT"]))
    material["parameters"]["D_SIGM_EPSI"] = material["post_peak_slope_pa"]
    concrete = case_inputs["material"]["concrete"]
    if (not math.isclose(material["parameters"]["E"], concrete["youngs_modulus_pa"], rel_tol=1e-12)
            or not math.isclose(material["parameters"]["NU"], concrete["poissons_ratio"], rel_tol=1e-12)):
        raise ValueError("Damage material elasticity differs from the baseline concrete")
    areas = np.asarray([spring["tributary_area_m2"] for spring in manifest["wall_foundation_springs"]])
    if not len(areas):
        areas = np.asarray([built["report"]["base_face_area_median_m2"] / 3.0])
    times = [0.0, *LOAD_FACTORS]
    linear_groups: list[str] = []
    linear_model_groups = ["UPSTREAM", "DOWNSTREAM", "CREST", "WALL_BASE"]
    inputs = {
        "repository_root": wsl_path(ROOT),
        "material": material, "elastic": case_inputs["material"],
        "loads": {**{key: case_inputs[key] for key in (
            "gravity_m_s2", "upstream_water_surface_z_m", "water_pressure_gradient_pa_per_m",
            "tailwater_elevation_m", "overflow_surcharge_pa")}},
        "times": times,
        "solver_times": [round(index * SOLVER_LOAD_INCREMENT, 12)
                         for index in range(round(1 / SOLVER_LOAD_INCREMENT) + 1)],
        "hanging_nodes": int(len(built["hanging_nodes"])),
        "has_bedrock_volume": False,
        "has_foundation": False,
        "elastic_groups": linear_groups,
        "linear_model_groups": linear_model_groups,
        "damage_threshold": DEFAULT_DAMAGE_THRESHOLD,
        "wall_tension_tolerance_pa":
            residual_tension_fraction * float(material["parameters"]["FT"]),
        "interface_operator": WALL_BASE_INTERFACE_OPERATOR,
        # Spring residual reference: tensile strength over a median interface tributary area.
        "spring_force_reference_n": float(material["parameters"]["FT"] * np.median(areas)),
    }
    (directory / "inputs.json").write_text(json.dumps(inputs, allow_nan=False), encoding="utf-8")
    support_report = {key: value for key, value in support.items() if key != "matrix_n_per_m3"}
    return {"inputs": inputs, "mesh": {**built["report"], "base_support": support_report},
            "material": material,
            "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "radius_m": float(config["geometry"]["wall"]["centerline_radius_m"])}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def export_file(directory: Path, threads: int, memory_mib: int, time_limit_s: int) -> Path:
    def linux(name):
        return wsl_path(directory / name)
    cpu_time_limit_s = max(int(time_limit_s) * max(1, int(threads)), 3600) if time_limit_s is not None else 7200
            
    export = directory / "crack.export"
    export.write_bytes((
        f"P time_limit {cpu_time_limit_s}\nP memory_limit {memory_mib}\nP mpi_nbcpu 1\nP ncpus {threads}\n"
        f"F comm {wsl_path(CASE)} D 1\nF libr {linux('inputs.json')} D 91\nF mmed {linux('mesh.med')} D 20\n"
        f"F libr {linux('quadratic_discrete.npz')} D 92\n"
        f"F libr {linux('native_crack.json')} R 90\nF libr {linux('damage_history.npz')} R 89\n"
        f"F rmed {linux('result.med')} R 80\n").encode("utf-8"))
    return export


def run(dam_config: Path, source_mesh: Path, output_root: Path, *,
        threads: int = DEFAULT_THREADS,
        memory_mib: int = DEFAULT_MEMORY_MIB, time_limit_s: int = UNBOUNDED_S) -> dict:
    from Aster.run_case import run_solver

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_root.resolve() / OUTPUT_ROOT_NAME / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    config_bytes = dam_config.read_bytes()
    (directory / "config.json").write_bytes(config_bytes)
    manifest = {
        "status": "running", "scenario_name": SCENARIO_NAME, "crack_scenario_names": [SCENARIO_NAME],
        "analysis_scope": "wall_only", "engineering_validated": False,
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "started_at_utc": timestamp, "case_sha256": file_hash(CASE),
        "converter_sha256": file_hash(CONVERTER), "mesher_sha256": file_hash(MESHER),
        "source_mesh_sha256": file_hash(source_mesh),
        "interface_operator": INTERFACE_OPERATOR,
        "interface_code_sha256": {name: file_hash(CASE.parent.parent / name) for name in (
            "source_contract.py", "build_three_body_mesh_arrays.py", "refined_interface.py",
            "quadratic_surface.py", "native_discrete.py", "convert_quadratic_to_med.py")},
    }
    _write(directory / "manifest.json", manifest)
    started = time.monotonic()
    distribution = os.environ.get(DISTRIBUTION_ENVIRONMENT, DEFAULT_WSL_DISTRIBUTION)
    try:
        if not (1 <= threads <= 64 and 2048 <= memory_mib <= 262144):
            raise ValueError("Threads must lie in 1-64 and memory in 2048-262144 MiB")
        print("[1/5] Building the wall-only damage mesh ...", flush=True)
        prepared = prepare(directory, dam_config, source_mesh)
        manifest.update({"source_config_sha256": prepared["source_config_sha256"],
                         "interface_operator": prepared["inputs"].get(
                             "interface_operator", WALL_BASE_INTERFACE_OPERATOR),
                         "mesh": prepared["mesh"]})
        _write(directory / "manifest.json", manifest)
        print(f"Mesh: {json.dumps(prepared['mesh'])}", flush=True)
        root = find_linux_installation(distribution)
        manifest["linux_code_aster_root"] = root
        print("[2/5] Writing the mixed-order MED mesh in WSL ...", flush=True)
        conversion = subprocess.run(
            linux_command(distribution, root, directory, [f"{root}/bin/python3", wsl_path(CONVERTER),
                                                          "whole_model_mesh.npz", "mesh.med"], 1),
            capture_output=True, text=True, timeout=CONVERSION_TIMEOUT_S, check=False)
        (directory / "conversion.log").write_text(conversion.stdout + conversion.stderr, encoding="utf-8")
        mesh = directory / "mesh.med"
        if conversion.returncode or not mesh.is_file() or mesh.stat().st_size == 0:
            raise RuntimeError("Whole-model crack MED conversion failed; see conversion.log")
        discrete = directory / "quadratic_discrete.npz"
        if not discrete.is_file() or discrete.stat().st_size == 0:
            raise RuntimeError("Whole-model crack conversion has no quadratic_discrete.npz")
        manifest["quadratic_discrete_sha256"] = file_hash(discrete)
        (directory / "whole_model_mesh.npz").unlink()
        export = export_file(directory, threads, memory_mib, time_limit_s)
        work = f"{LINUX_WORK_ROOT}/{timestamp}"
        manifest["linux_work_directory"] = work
        print(f"[3/5] Solving with Linux Code_Aster ({threads} threads) in {directory} ...", flush=True)
        try:
            completed = run_solver(
                linux_command(distribution, root, directory,
                              [f"{root}/bin/run_aster", "--wrkdir", work, export.name], threads),
                directory, time_limit_s, live_log=directory / LIVE_LOG_NAME)
        finally:
            # With --wrkdir run_aster keeps its work directory (tens of GB of out-of-core files).
            subprocess.run(["wsl.exe", "-d", distribution, "--exec", "rm", "-rf", work],
                           capture_output=True, timeout=600, check=False)
        (directory / "solver.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        native, result, history = (directory / name for name in
                                   ("native_crack.json", "result.med", "damage_history.npz"))
        markers = ("<EXCEPTION>", "Traceback (most recent call last)", "Fatal Python error", "<F>_ERROR")
        if (completed.returncode or not native.is_file() or not result.is_file() or not history.is_file()
                or result.stat().st_size == 0
                or any(marker in completed.stdout + completed.stderr for marker in markers)):
            raise RuntimeError("Native Code_Aster wall-only cracking failed; see solver.log")
        data = json.loads(native.read_text(encoding="utf-8"))
        if [state["time"] for state in data["history"]] != prepared["inputs"]["times"]:
            raise ValueError("Native cracking did not converge every requested load factor")
        print("[4/5] Exporting the ParaView series ...", flush=True)
        extraction = subprocess.run(
            linux_command(distribution, root, directory, [f"{root}/bin/python3", wsl_path(EXTRACTOR),
                                                          "result.med", "."], 1),
            capture_output=True, text=True, timeout=CONVERSION_TIMEOUT_S, check=False)
        if extraction.returncode:
            raise RuntimeError(f"Aster ParaView export failed: {extraction.stdout}\n{extraction.stderr}")
        visualisation = json.loads((directory / "damage_series.json").read_text(encoding="utf-8"))["CRACK"]
        if (not (directory / visualisation["pvd"]).is_file()
                or [row["time"] for row in visualisation["snapshots"]] != prepared["inputs"]["times"]):
            raise ValueError("Aster cracking has incomplete ParaView output")
        print("[5/5] Writing the crack report ...", flush=True)
        metrics = full_wall_damage_metrics(history, prepared["radius_m"])
        final_state = data["history"][-1]
        maximum_displacement_m = max(
            float(state["maximum_displacement_magnitude_m"]) for state in data["history"])
        residual_tension_pa = max(0.0, float(final_state["maximum_principal_stress_pa"]))
        cracked_residual_tension_pa = final_state["maximum_cracked_principal_stress_pa"]
        tolerance_pa = prepared["inputs"]["wall_tension_tolerance_pa"]
        zero_tension = residual_tension_pa <= tolerance_pa
        all_load_states_converged = [state["time"] for state in data["history"]] == prepared["inputs"]["times"]
        report = {
            "scenario_name": SCENARIO_NAME, "solver": "code_aster", "analysis_scope": "wall_only",
            "constitutive_law": "ENDO_ISOT_BETON", "modelisation": "3D linear TETRA4 WALL",
            "solution_method": "NEWTON",
            "engineering_validated": False, "material": prepared["material"],
            "loads": prepared["inputs"]["loads"], "mesh": prepared["mesh"],
            "initial_damage": data["audit"], "metrics": metrics,
            "region_stress_history": [{"time": state["time"], **state["region_stress"],
                                       "maximum_displacement_component_m":
                                           state["maximum_displacement_component_m"],
                                       "maximum_displacement_magnitude_m":
                                           state["maximum_displacement_magnitude_m"]}
                                      for state in data["history"]],
            "maximum_damage": metrics["final"]["maximum_damage"],
            "maximum_principal_tensile_stress_pa": final_state["maximum_principal_stress_pa"],
            "maximum_cracked_principal_stress_pa": cracked_residual_tension_pa,
            "maximum_displacement_m": maximum_displacement_m,
            "maximum_displacement_mm": 1000.0 * maximum_displacement_m,
            "maximum_crack_opening_mm": None,
            "crack_opening_basis":
                "Not available: local ENDO_ISOT_BETON damage is not a discrete crack jump.",
            "residual_tension_tolerance_pa": tolerance_pa,
            "wall_tension_tolerance_pa": tolerance_pa,
            "zero_tension_acceptance": {
                "passed": zero_tension,
                "damage_threshold": prepared["inputs"]["damage_threshold"],
                "measured_maximum_principal_tension_pa": residual_tension_pa,
                "tolerance_pa": tolerance_pa,
                "basis": "maximum principal tensile stress at every native WALL Gauss point",
            },
            "equilibrium": {
                "passed": all_load_states_converged,
                "maximum_relative_residual": None,
                "tolerance": 1e-4,
                "basis": "Code_Aster STAT_NON_LINE native residual convergence at every archived load state",
            },
            "stress_reporting_scope": "wall only; removed foundation/bedrock represented by base springs",
            "paraview": {"pvd": visualisation["pvd"], "med": result.name},
            "runtime_s": time.monotonic() - started,
            "limitations": [
                "The complete wall is the only body. Foundation/bedrock compliance is represented "
                "by the configured equivalent base-spring tensor.",
                "Foundation and bedrock stress results are intentionally unavailable in this reduced model.",
                "The run uses Newton with an explicit equilibrium/stress-acceptance audit.",
                "This continuum result is not a discrete crack analysis and cannot be used to report "
                "crack opening or a cracked load path.",
                "ENDO_ISOT_BETON damage is a local cell damage measure, not a sharp crack opening.",
                "Maximum crack opening is intentionally null because the continuous finite-element mesh "
                "has no separated crack faces from which a normal opening can be measured.",
                "Water pressure is applied only to the exterior reservoir faces, not to newly damaged surfaces.",
                "Mesh, band-width, softening and concrete-fracture calibration are unverified.",
            ],
        }
        _write(directory / "crack_report.json", report)
        manifest["status"] = "completed_not_engineering_validated"
    except (RuntimeError, ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        print(f"Whole-model Aster cracking failed: {error}", flush=True)
    manifest["runtime_s"] = time.monotonic() - started
    _write(directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dam-config", type=Path, required=True)
    parser.add_argument("--source-mesh", type=Path, required=True,
                        help="The job's Elmer three_body.msh used by the elastic baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--memory-mib", type=int, default=DEFAULT_MEMORY_MIB)
    parser.add_argument("--time-limit-s", type=int, default=UNBOUNDED_S)
    args = parser.parse_args()
    manifest = run(args.dam_config, args.source_mesh, args.output, threads=args.threads,
                   memory_mib=args.memory_mib, time_limit_s=args.time_limit_s)
    print(json.dumps({key: manifest.get(key) for key in ("status", "error", "runtime_s")}, indent=2))
    if manifest["status"] != "completed_not_engineering_validated":
        sys.exit(1)


if __name__ == "__main__":
    main()
