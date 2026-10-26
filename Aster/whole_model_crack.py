"""Native Code_Aster cracking of the whole three-body dam with a full-length wall crack.

Only the wall cracks: a quadratic ENDO_FISS_EXP (3D_GRAD_VARI) band around the
selected elevation carries the frozen shared seed, which runs bedrock to bedrock.
All other wall, foundation and bedrock cells keep their linear elastic baseline
elements, springs and loads, so the rest of the model responds to the softening
wall. It runs in WSL with Linux Code_Aster because the Windows MUMPS build returns
NaN on this mesh and its MULT_FRONT factorisation is too slow for Newton steps.
The result is native ENDO_FISS_EXP damage, not AT2 damage or a crack opening.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
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
SCENARIO_NAME = "Aster_Crack_Submodel"
SEED_DAMAGE = 0.999
LOAD_FACTORS = tuple(round(0.1 * step, 10) for step in range(1, 11))
# Wall cells whose centroid lies within this distance of the crack plane are quadratic and can damage.
DEFAULT_BAND_HALF_HEIGHT_M = 0.75
DEFAULT_THREADS = 8
# Aster cracking runs alone (after FEniCSx), so it may use most of a 24 GB WSL allocation.
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


def wsl_path(path: Path) -> str:
    """Windows path to its /mnt/<drive>/... form; POSIX paths pass through."""
    text = str(Path(path).resolve()) if os.name == "nt" else str(path)
    drive, rest = os.path.splitdrive(text)
    if not drive:
        return text.replace("\\", "/")
    return f"/mnt/{drive.rstrip(':').lower()}{rest.replace(chr(92), '/')}"


def linux_environment(root: str, threads: int) -> list[str]:
    """``env`` arguments for the relocated Code_Aster runtime (not the Windows PATH)."""
    return [f"PYTHONHOME={root}", f"LD_LIBRARY_PATH={root}/lib:{root}/lib64",
            f"PATH={root}/bin:/usr/bin:/bin", f"OMP_NUM_THREADS={threads}",
            "PYTHONUNBUFFERED=1", "GFORTRAN_UNBUFFERED_PRECONNECTED=y"]


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


def crack_metrics(damage_archive: Path, geometry: dict, generation: dict, radius_m: float,
                  wall_thickness_m: float) -> dict:
    """Native damage growth along the wall; counts Gauss points, not crack area or opening."""
    with np.load(damage_archive, allow_pickle=False) as data:
        xyz, initial, damage, times = (data[key] for key in ("coordinates_m", "initial_damage", "damage", "times"))
    seeded = initial > 0
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    chainage = radius_m * np.arctan2(xyz[:, 0], xyz[:, 1])
    extent = generation.get("concrete_chainage_extent_m") or [float(chainage.min()), float(chainage.max())]
    history = []
    for instant, values in zip(times, damage):
        cracked = values > 0.95
        entry = {
            "time": float(instant), "load_factor": float(instant),
            "maximum_damage": float(values.max()),
            "maximum_new_damage": float(values[~seeded].max()) if np.any(~seeded) else 0.0,
            "new_points_above_0_95": int(np.count_nonzero(cracked & ~seeded)),
            "new_points_above_0_5": int(np.count_nonzero((values > 0.5) & ~seeded)),
        }
        if np.any(cracked):
            entry["cracked_radial_extent_m"] = float(radius[cracked].max() - radius[cracked].min())
            entry["cracked_through_thickness_fraction"] = min(1.0, entry["cracked_radial_extent_m"]
                                                              / wall_thickness_m)
            entry["cracked_chainage_m"] = [float(chainage[cracked].min()), float(chainage[cracked].max())]
            entry["cracked_elevation_extent_m"] = [float(xyz[cracked, 2].min()), float(xyz[cracked, 2].max())]
        history.append(entry)
    inner, outer, theta0, theta1 = geometry["annular_sector"]
    return {
        "basis": "native ENDO_FISS_EXP V1 at band Gauss points; counts are not areas or crack openings",
        "seed_radial_extent_m": outer - inner,
        "seed_chainage_m": [radius_m * theta0, radius_m * theta1],
        "wall_chainage_extent_at_crack_m": list(extent),
        "seeded_gauss_points": int(np.count_nonzero(seeded)),
        "history": history, "final": history[-1],
    }


def wall_thickness(wall: dict, elevation: float) -> float:
    thickness = float(wall["radial_width_m"])
    for step in wall.get("thickness_steps") or ():
        if elevation < float(step["top_z_m"]):
            thickness = float(step["radial_width_m"])
    return thickness


def crack_water_pressure(shared: dict, loads: dict, factor: float = 1.0) -> dict:
    """Check the seed's frozen crack pressure against this case's own water datums.

    ``factor`` is the config's fracture_debug.crack_water_pressure_factor (1 = full head).
    """
    seed = shared["payload"]
    face, elevation = seed["seed_face"], float(seed["geometry"]["elevation_m"])
    surface = float(loads["upstream_water_surface_z_m"] if face == "upstream"
                    else loads["tailwater_elevation_m"])
    gradient = float(loads["water_pressure_gradient_pa_per_m"])
    expected = factor * gradient * max(surface - elevation, 0.0)
    frozen = float(seed["loading"]["internal_pressure_pa"])
    if not math.isclose(frozen, expected, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(
            f"The prepared crack seed carries {frozen:.6g} Pa of crack water pressure, but the "
            f"{face} water at z = {elevation:.6g} m gives {expected:.6g} Pa. Seeds prepared before "
            "crack water pressure was supported carry 0 Pa; select the crack location again.")
    return {"pressure_pa": expected, "seed_face": face, "water_surface_z_m": surface,
            "pressure_gradient_pa_per_m": gradient, "factor": factor,
            "applied_to": "CRACK_FACES (seeded crack faces)"}


def prepare(directory: Path, dam_config: Path, prepared_seed: Path, source_mesh: Path,
            band_half_height_m: float = DEFAULT_BAND_HALF_HEIGHT_M) -> dict:
    """Export the baseline arrays, build the band mesh archive and solver inputs (Windows side)."""
    from Aster.build_three_body_mesh_arrays import export
    from Aster.fracture_material import material_contract
    from Aster.generate_three_body_case import render_comm
    from Aster.whole_model_crack_mesh import archive_arrays, build
    from Elmer.model_config import load_config
    from FEniCSx.crack_contract import apply_fracture_debug, validate_selected_seed
    from FEniCSx.selected_seed import validate_prepared_seed

    config_bytes = dam_config.read_bytes()
    prepared = json.loads(prepared_seed.read_text(encoding="utf-8"))
    payload = validate_prepared_seed(prepared, config_bytes)
    generation = payload["generation"]
    shared = generation["shared_seed"]
    geometry = validate_selected_seed(shared)["geometry"]
    source = directory / "baseline_mesh"
    manifest = export(output_dir=source, source_mesh=source_mesh)
    _, case_inputs = render_comm(ROOT, source / "mesh_manifest.json")
    config = load_config(dam_config)
    _, water_factor = apply_fracture_debug({"solver": {}}, config)
    water = crack_water_pressure(shared, case_inputs, water_factor)
    with np.load(source / "three_body_mesh_arrays.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files if name != "interface_names"}
    built = build(arrays, geometry, band_half_height_m)
    np.savez(directory / "whole_model_mesh.npz", **archive_arrays(arrays, built))
    (source / "three_body_mesh_arrays.npz").unlink()
    material = material_contract(config)
    concrete = case_inputs["material"]["concrete"]
    if (not math.isclose(material["parameters"]["E"], concrete["youngs_modulus_pa"], rel_tol=1e-12)
            or not math.isclose(material["parameters"]["NU"], concrete["poissons_ratio"], rel_tol=1e-12)):
        raise ValueError("Damage material elasticity differs from the baseline concrete")
    wall_springs, support_springs = spring_inputs(manifest)
    areas = np.asarray([spring["tributary_area_m2"] for spring in manifest["wall_foundation_springs"]])
    times = [0.0, *LOAD_FACTORS]
    inputs = {
        "repository_root": wsl_path(ROOT), "shared_seed": shared, "seed_damage": SEED_DAMAGE,
        "material": material, "elastic": case_inputs["material"],
        "loads": {**{key: case_inputs[key] for key in (
            "gravity_m_s2", "upstream_water_surface_z_m", "water_pressure_gradient_pa_per_m",
            "tailwater_elevation_m", "overflow_surcharge_pa")}, "crack_pressure_pa": water["pressure_pa"]},
        "times": times, "hanging_nodes": int(len(built["hanging_nodes"])),
        "wall_foundation_springs": wall_springs, "support_springs": support_springs,
        # Spring residual reference: tensile strength over a median interface tributary area.
        "spring_force_reference_n": float(material["parameters"]["FT"] * np.median(areas)),
    }
    (directory / "inputs.json").write_text(json.dumps(inputs, allow_nan=False), encoding="utf-8")
    wall = config["geometry"]["wall"]
    return {"payload": payload, "prepared_sha256": prepared["sha256"], "geometry": geometry,
            "generation": generation, "inputs": inputs, "mesh": built["report"],
            "crack_water_pressure": {**water, "loaded_area_m2": {
                "top": built["report"]["top_area_m2"], "bottom": built["report"]["bottom_area_m2"],
                "nominal_seed": built["report"]["nominal_seed_area_m2"]}},
            "material": material, "radius_m": float(wall["centerline_radius_m"]),
            "wall_thickness_m": wall_thickness(wall, float(geometry["elevation_m"]))}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def export_file(directory: Path, threads: int, memory_mib: int, time_limit_s: int) -> Path:
    def linux(name):
        return wsl_path(directory / name)
    export = directory / "crack.export"
    export.write_bytes((
        f"P time_limit {time_limit_s}\nP memory_limit {memory_mib}\nP mpi_nbcpu 1\nP ncpus {threads}\n"
        f"F comm {wsl_path(CASE)} D 1\nF libr {linux('inputs.json')} D 91\nF mmed {linux('mesh.med')} D 20\n"
        f"F libr {linux('native_crack.json')} R 90\nF libr {linux('damage_history.npz')} R 89\n"
        f"F rmed {linux('result.med')} R 80\n").encode("utf-8"))
    return export


def run(dam_config: Path, prepared_seed: Path, source_mesh: Path, output_root: Path, *,
        band_half_height_m: float = DEFAULT_BAND_HALF_HEIGHT_M, threads: int = DEFAULT_THREADS,
        memory_mib: int = DEFAULT_MEMORY_MIB, time_limit_s: int = UNBOUNDED_S) -> dict:
    from Aster.run_case import run_solver

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_root.resolve() / OUTPUT_ROOT_NAME / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    config_bytes = dam_config.read_bytes()
    (directory / "config.json").write_bytes(config_bytes)
    manifest = {
        "status": "running", "scenario_name": SCENARIO_NAME, "crack_scenario_names": [SCENARIO_NAME],
        "analysis_scope": "whole_model", "engineering_validated": False,
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "started_at_utc": timestamp, "case_sha256": file_hash(CASE),
        "converter_sha256": file_hash(CONVERTER), "mesher_sha256": file_hash(MESHER),
        "source_mesh_sha256": file_hash(source_mesh),
    }
    _write(directory / "manifest.json", manifest)
    started = time.monotonic()
    distribution = os.environ.get(DISTRIBUTION_ENVIRONMENT, DEFAULT_WSL_DISTRIBUTION)
    try:
        if not (1 <= threads <= 64 and 2048 <= memory_mib <= 262144):
            raise ValueError("Threads must lie in 1-64 and memory in 2048-262144 MiB")
        print("[1/5] Building the whole-model crack mesh ...", flush=True)
        prepared = prepare(directory, dam_config, prepared_seed, source_mesh, band_half_height_m)
        manifest.update({"prepared_seed_sha256": prepared["prepared_sha256"],
                         "crack_location": prepared["generation"], "mesh": prepared["mesh"]})
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
            raise RuntimeError("Native Code_Aster whole-model cracking failed; see solver.log")
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
        metrics = crack_metrics(history, prepared["geometry"], prepared["generation"], prepared["radius_m"],
                                prepared["wall_thickness_m"])
        final_state = data["history"][-1]
        report = {
            "scenario_name": SCENARIO_NAME, "solver": "code_aster", "analysis_scope": "whole_model",
            "constitutive_law": "ENDO_FISS_EXP", "modelisation": "3D_GRAD_VARI band, 3D elsewhere",
            "engineering_validated": False, "material": prepared["material"],
            "loads": prepared["inputs"]["loads"], "mesh": prepared["mesh"],
            "crack_water_pressure": prepared["crack_water_pressure"],
            "native_seed": data["audit"], "metrics": metrics,
            "region_stress_history": [{"time": state["time"], **state["region_stress"],
                                       "maximum_displacement_component_m":
                                           state["maximum_displacement_component_m"],
                                       "mean_crack_face_separation_m":
                                           state["mean_crack_face_separation_m"]}
                                      for state in data["history"]],
            "maximum_damage": metrics["final"]["maximum_damage"],
            "maximum_principal_tensile_stress_pa": final_state["maximum_principal_stress_pa"],
            "stress_reporting_scope": "whole model: wall, foundation and bedrock (Gauss-point peaks per region)",
            "paraview": {"pvd": visualisation["pvd"], "med": result.name},
            "runtime_s": time.monotonic() - started,
            "limitations": [
                "Only wall cells within the damage band can crack; the rest of the model is linear elastic.",
                "The seed is a horizontal annular sector cut into the mesh as a 2 mm layer; it does not "
                "follow the sloped bedrock at its ends.",
                "Band mid-edge nodes are tied to neighbouring linear cells, which stiffens the band edge slightly.",
                "Native ENDO_FISS_EXP damage is not FEniCSx phase-field (AT1/AT2) damage or crack opening.",
                "Crack water pressure acts on the seeded crack faces only (full hydrostatic head of the "
                "seeded face, no drainage); it does not follow damage that grows beyond the seed, and "
                "base uplift at the wall-foundation interface is not modelled.",
                "Mesh and band-width convergence are unverified.",
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
    parser.add_argument("--prepared-seed", type=Path, required=True)
    parser.add_argument("--source-mesh", type=Path, required=True,
                        help="The job's Elmer three_body.msh used by the elastic baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--band-half-height-m", type=float, default=DEFAULT_BAND_HALF_HEIGHT_M)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--memory-mib", type=int, default=DEFAULT_MEMORY_MIB)
    parser.add_argument("--time-limit-s", type=int, default=UNBOUNDED_S)
    args = parser.parse_args()
    manifest = run(args.dam_config, args.prepared_seed, args.source_mesh, args.output,
                   band_half_height_m=args.band_half_height_m, threads=args.threads,
                   memory_mib=args.memory_mib, time_limit_s=args.time_limit_s)
    print(json.dumps({key: manifest.get(key) for key in ("status", "error", "runtime_s")}, indent=2))
    if manifest["status"] != "completed_not_engineering_validated":
        sys.exit(1)


if __name__ == "__main__":
    main()
