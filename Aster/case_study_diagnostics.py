"""Replay a saved Aster cracking mesh without altering the source job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import uuid

from Aster import whole_model_crack as wm
from Aster.run_case import run_solver

STAGES = ("inventory", "elastic", "unseeded", "seeded", "compare")
POSTPROCESS = Path(__file__).with_name("cases") / "diagnostic_summary.comm"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_run(study: Path) -> Path:
    runs = list((study / "cracking" / wm.OUTPUT_ROOT_NAME).glob("*/manifest.json"))
    if len(runs) != 1:
        raise ValueError("Select a study with exactly one current Aster cracking run")
    run = runs[0].parent
    manifest = read(runs[0])
    if manifest["status"] != "completed_not_engineering_validated":
        raise ValueError("The archived Aster cracking run must be complete")
    if digest(run / "config.json") != manifest["configuration_sha256"]:
        raise ValueError("Archived cracking configuration hash does not match its manifest")
    for name in ("inputs.json", "mesh.med", "crack_report.json"):
        if not (run / name).is_file():
            raise ValueError(f"Missing archived input: {run / name}")
    return run


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Production command template changed; review diagnostic transformation: {old}")
    return text.replace(old, new, 1)


def render_case(stage: str, *, legacy_lumped: bool = False) -> str:
    if stage not in ("elastic", "unseeded", "seeded"):
        raise ValueError(f"Not a solver stage: {stage}")
    prefix, separator, _ = wm.CASE.read_text(encoding="utf-8").partition("\nband_cells = ")
    if not separator:
        raise ValueError("Production command template has no expected postprocessing boundary")
    if legacy_lumped:
        prefix = replace_once(prefix, 'MESH = read_mesh("fort.20", "fort.92")',
                              'MESH = LIRE_MAILLAGE(FORMAT="MED", UNITE=20, NOM_MED="THREE_BODY", '
                              'VERI_MAIL=_F(VERIF="NON"))')
        prefix = replace_once(prefix, 'GROUP_MA=("QS_POINTS", "QS_LINKS")',
                              'GROUP_MA=("OUTER_SPRINGS", "WALL_FOUNDATION_SPRINGS")')
        prefix = replace_once(prefix, 'springs = characteristics(MESH, MODEL, "fort.92")', '''
springs = AFFE_CARA_ELEM(
    MODELE=MODEL,
    DISCRET=[_F(GROUP_MA=s["group"], REPERE="GLOBAL", CARA="K_T_D_L",
                VALE=tuple(s["stiffness_n_per_m"])) for s in inputs["wall_foundation_springs"]]
    + [_F(GROUP_MA=s["group"], REPERE="GLOBAL", CARA="K_T_N", SYME="NON",
          VALE=tuple(s["matrix_column_major_n_per_m"])) for s in inputs["support_springs"]],
)''')
    initial_start = prefix.index("INITIAL, audit = initial_field(")
    initial_end = prefix.index("# Band mid-edge nodes", initial_start)
    if stage != "seeded":
        prefix = prefix[:initial_start] + prefix[initial_end:]
        prefix = replace_once(prefix, ", ETAT_INIT=_F(VARI=INITIAL)", "")
    if stage == "elastic":
        prefix = replace_once(prefix, 'MODELISATION="3D_GRAD_VARI"', 'MODELISATION="3D"')
        prefix = replace_once(prefix, ',\n          _F(GROUP_MA="DAMAGE_BAND", MATER=DAMAGE)', "")
        prefix = replace_once(
            prefix,
            'COMPORTEMENT=(_F(TOUT="OUI", RELATION="ELAS"),\n'
            '                  _F(GROUP_MA="DAMAGE_BAND", RELATION="ENDO_FISS_EXP", DEFORMATION="PETIT"))',
            'COMPORTEMENT=_F(TOUT="OUI", RELATION="ELAS")',
        )
    return prefix + "\n" + POSTPROCESS.read_text(encoding="utf-8")


def inventory(study: Path, output: Path) -> dict:
    run = source_run(study)
    report = read(run / "crack_report.json")
    baseline = study / "cracking" / "aster_elastic_baseline" / "aster_baseline_report.json"
    result = {
        "engineering_validated": False,
        "source_study": str(study), "source_run": str(run),
        "source_hashes": {name: digest(run / name) for name in
                          ("manifest.json", "inputs.json", "config.json", "mesh.med", "crack_report.json")},
        "production_case_sha256": digest(wm.CASE),
        "archived_case_sha256": read(run / "manifest.json")["case_sha256"],
        "archived_elastic_baseline": read(baseline),
        "archived_cracked_final": report["region_stress_history"][-1],
        "loads": read(run / "inputs.json")["loads"],
        "limitations": [
            "The original elastic baseline uses a different mesh and nodal stress recovery.",
            "Elastic diagnostic combines slicing, mixed element order and hanging-node constraints.",
            "Unseeded permits spontaneous band damage; it is not a zero-damage elastic reference.",
            "Diagnostics default to corrected interface integration, rebuilding legacy meshes from "
            "hash-checked source artifacts; --legacy-lumped explicitly replays the old operator.",
        ],
    }
    if result["production_case_sha256"] != result["archived_case_sha256"]:
        print("WARNING: current production command differs from the archived command hash.", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write(output / "inventory.json", result)
    return result


def corrected_mesh(run: Path, directory: Path) -> dict:
    """Rebuild only a legacy archive's mesh/operator, preserving its saved physics."""
    import numpy as np
    from Aster.build_three_body_mesh_arrays import export
    from Aster.whole_model_crack_mesh import archive_arrays, build
    from FEniCSx.crack_contract import validate_selected_seed

    baseline = read(run / "baseline_mesh" / "mesh_manifest.json")
    source = directory / "baseline_mesh"
    manifest = export(source, Path(baseline["source_mesh"]))
    if manifest["source_hashes"] != baseline["source_hashes"]:
        raise ValueError("Source artifacts differ from archived baseline; cannot rebuild corrected interfaces")
    inputs = read(run / "inputs.json")
    geometry = validate_selected_seed(inputs["shared_seed"])["geometry"]
    band = read(run / "manifest.json")["mesh"]["band_half_height_m"]
    with np.load(source / "three_body_mesh_arrays.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files if name != "interface_names"}
    built = build(arrays, geometry, band)
    np.savez(directory / "whole_model_mesh.npz", **archive_arrays(arrays, built))
    distribution = os.environ.get(wm.DISTRIBUTION_ENVIRONMENT, wm.DEFAULT_WSL_DISTRIBUTION)
    root = wm.find_linux_installation(distribution)
    converted = subprocess.run(
        wm.linux_command(distribution, root, directory, [
            "timeout", "--signal=TERM", "--kill-after=5s", f"{wm.CONVERSION_TIMEOUT_S}s",
            f"{root}/bin/python3", wm.wsl_path(wm.CONVERTER), "whole_model_mesh.npz", "mesh.med"], 1),
        capture_output=True, text=True, timeout=wm.CONVERSION_TIMEOUT_S + 15, check=False)
    (directory / "conversion.log").write_text(converted.stdout + converted.stderr, encoding="utf-8")
    if converted.returncode or any(not (directory / name).is_file()
                                  or (directory / name).stat().st_size == 0
                                  for name in ("mesh.med", "quadratic_discrete.npz")):
        raise RuntimeError("Corrected diagnostic mesh conversion failed; see conversion.log")
    return built["report"]


def prepare(study: Path, output: Path, stage: str, budget: int, threads: int,
            memory: int, *, legacy_lumped: bool = False) -> Path:
    run = source_run(study)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output / stage / stamp
    inputs = read(run / "inputs.json")
    inputs["repository_root"] = wm.wsl_path(wm.ROOT)
    inputs["diagnostic_stage"] = stage
    # Pressure on seeded faces without a seed is not an uncracked reference.
    if stage != "seeded" and inputs["loads"]["crack_pressure_pa"] != 0:
        raise ValueError("Uncracked diagnostics require zero archived crack-face pressure")
    directory.mkdir(parents=True, exist_ok=False)
    corrected = inputs.get("interface_operator") == wm.INTERFACE_OPERATOR
    if legacy_lumped and corrected:
        raise ValueError("A corrected archive cannot be replayed with legacy lumped springs")
    mesh_report = None
    mesh = run / "mesh.med"
    discrete = run / "quadratic_discrete.npz"
    if not legacy_lumped:
        if not corrected:
            mesh_report = corrected_mesh(run, directory)
            mesh, discrete = directory / "mesh.med", directory / "quadratic_discrete.npz"
        elif (not discrete.is_file()
              or digest(discrete) != read(run / "manifest.json")["quadratic_discrete_sha256"]):
            raise ValueError("Archived quadratic interface operator is missing or changed")
        inputs["interface_operator"] = wm.INTERFACE_OPERATOR
        inputs.pop("wall_foundation_springs", None)
        inputs.pop("support_springs", None)
    write(directory / "inputs.json", inputs)
    (directory / "case.comm").write_text(render_case(stage, legacy_lumped=legacy_lumped), encoding="utf-8")
    export = (
        f"P time_limit {budget}\nP memory_limit {memory}\nP mpi_nbcpu 1\nP ncpus {threads}\n"
        f"F comm {wm.wsl_path(directory / 'case.comm')} D 1\n"
        f"F libr {wm.wsl_path(directory / 'inputs.json')} D 91\n"
        f"F mmed {wm.wsl_path(mesh)} D 20\n"
        + (f"F libr {wm.wsl_path(discrete)} D 92\n" if not legacy_lumped else "") +
        f"F libr {wm.wsl_path(directory / 'summary.json')} R 90\n"
        f"F rmed {wm.wsl_path(directory / 'result.med')} R 80\n"
    )
    (directory / "diagnostic.export").write_text(export, encoding="utf-8")
    write(directory / "manifest.json", {
        "status": "prepared", "stage": stage, "engineering_validated": False,
        "source_run": str(run), "budget_s": budget, "threads": threads, "memory_mib": memory,
        "source_hashes": {name: digest(run / name) for name in ("inputs.json", "mesh.med", "config.json")},
        "case_sha256": digest(directory / "case.comm"),
        "native_seed_sha256": digest(Path(wm.__file__).with_name("native_seed.py")),
        "archived_case_sha256": read(run / "manifest.json")["case_sha256"],
        "interface_operator": "legacy_lumped" if legacy_lumped else wm.INTERFACE_OPERATOR,
        "mesh": mesh_report,
        "interface_mesh_sha256": digest(mesh),
        "quadratic_discrete_sha256": None if legacy_lumped else digest(discrete),
        "interface_code_sha256": {name: digest(Path(wm.__file__).with_name(name)) for name in (
            "refined_interface.py", "quadratic_surface.py", "native_discrete.py",
            "whole_model_crack_mesh.py", "convert_whole_model_to_med.py")},
    })
    return directory


def validate_summary(directory: Path, times: list[float], stage: str) -> dict:
    if not (directory / "summary.json").is_file():
        raise ValueError(f"Diagnostic has no summary.json; see {directory / 'solver.live.log'} "
                         "and the retained Linux work directory in manifest.json")
    summary = read(directory / "summary.json")
    if summary["stage"] != stage or [row["time"] for row in summary["history"]] != times:
        raise ValueError("Diagnostic did not converge every archived load factor")
    for row in summary["history"]:
        for peaks in row["regions"].values():
            for name in ("maximum_principal_stress_pa", "minimum_principal_stress_pa"):
                value = peaks[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("Diagnostic has invalid stress extrema")
    result = directory / "result.med"
    if not result.is_file() or result.stat().st_size == 0:
        raise ValueError("Diagnostic has no MED result")
    return summary


def solve(directory: Path) -> None:
    diagnostic_roots = [parent for parent in directory.parents if parent.name == "diagnostics"]
    if not diagnostic_roots:
        raise ValueError("Solver directory must be inside a study's diagnostics directory")
    lock = diagnostic_roots[-1] / "solver.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "directory": str(directory)}))
    except FileExistsError as error:
        raise RuntimeError(f"Another diagnostic owns {lock}; do not run stages concurrently") from error
    try:
        _solve(directory)
    finally:
        lock.unlink()


def _solve(directory: Path) -> None:
    manifest = read(directory / "manifest.json")
    distribution = os.environ.get(wm.DISTRIBUTION_ENVIRONMENT, wm.DEFAULT_WSL_DISTRIBUTION)
    work = f"/var/tmp/lamb_aster_diagnostics/{uuid.uuid4().hex}"
    manifest.update(status="running", linux_work_directory=work)
    write(directory / "manifest.json", manifest)
    try:
        root = wm.find_linux_installation(distribution)
        manifest["linux_code_aster_root"] = root
        budget = manifest["budget_s"]
        # Always fork an isolated group and wait for its exit before cleanup and validation.
        launch = (
            'echo $$ > diagnostic.pid; exec timeout --signal=TERM --kill-after=20s '
            f'{budget}s "$1/bin/run_aster" --wrkdir "$2" diagnostic.export'
        )
        command = wm.linux_command(distribution, root, directory, [
            "/usr/bin/setsid", "--fork", "--wait", "/bin/sh", "-c", launch, "diagnostic", root, work,
        ], manifest["threads"])
        print(f"Running {manifest['stage']}: {directory}", flush=True)
        try:
            completed = run_solver(command, directory, budget + 30,
                                   live_log=directory / "solver.live.log")
        finally:
            cleanup = (
                "import os,signal,sys\n"
                "from pathlib import Path\n"
                "p=Path('diagnostic.pid')\n"
                "if p.exists():\n"
                " pid=int(p.read_text())\n"
                " try:\n"
                "  if os.getpgid(pid)!=pid or Path('/proc/%s/cwd'%pid).resolve()!=Path.cwd():\n"
                "   raise RuntimeError('Refusing to stop an unrelated process group')\n"
                "  os.killpg(pid,signal.SIGKILL)\n"
                " except ProcessLookupError: pass\n"
            )
            subprocess.run(wm.linux_command(distribution, root, directory,
                           [f"{root}/bin/python3", "-c", cleanup], 1),
                           check=True, capture_output=True, text=True, timeout=30)
        (directory / "solver.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode or any(marker in completed.stdout + completed.stderr for marker in
                                      ("<EXCEPTION>", "Traceback (most recent call last)", "<F>_ERROR")):
            raise RuntimeError(f"Aster diagnostic failed ({completed.returncode}); see solver.live.log")
        validate_summary(directory, read(directory / "inputs.json")["times"], manifest["stage"])
        subprocess.run(["wsl.exe", "-d", distribution, "--exec", "rm", "-rf", "--", work],
                       check=True, capture_output=True, text=True, timeout=60)
        manifest["status"] = "completed"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        manifest.update(status="failed", error=str(error) or type(error).__name__)
        raise
    finally:
        write(directory / "manifest.json", manifest)
        # Preserve native working files on failure for diagnosis; never delete source job data.
    print(f"Completed: {directory / 'summary.json'}", flush=True)


def compare(study: Path, output: Path) -> dict:
    archive = inventory(study, output)
    rows = []
    for stage in ("elastic", "unseeded", "seeded"):
        manifests = sorted((output / stage).glob("*/manifest.json"))
        if not manifests:
            rows.append({"stage": stage, "status": "not_run"})
            continue
        path = manifests[-1]
        manifest = read(path)
        row = {"stage": stage, "status": manifest["status"], "directory": str(path.parent),
               "interface_operator": manifest.get("interface_operator", "legacy_lumped")}
        if manifest["status"] == "completed":
            if manifest["source_hashes"] != {k: archive["source_hashes"][k]
                                             for k in ("inputs.json", "mesh.med", "config.json")}:
                raise ValueError(f"{stage} used different source inputs")
            row["final"] = validate_summary(path.parent, read(path.parent / "inputs.json")["times"],
                                            stage)["history"][-1]
        rows.append(row)
    if len({row["interface_operator"] for row in rows if row["status"] == "completed"}) > 1:
        raise ValueError("Diagnostic stages mix legacy lumped and corrected interface operators; "
                         "rerun all stages with the same operator or use separate output directories")
    result = {"engineering_validated": False, "archive": archive, "diagnostics": rows}
    write(output / "comparison.json", result)
    print("Stage          Status           Wall tension / compression (MPa)", flush=True)
    baseline = archive["archived_elastic_baseline"]
    wall = archive["archived_cracked_final"]["wall"]
    print(f"saved baseline (nodal)          {baseline['maximum_principal_tensile_stress_pa']/1e6:.6g}"
          f" / {baseline['maximum_principal_compressive_stress_pa']/1e6:.6g}", flush=True)
    print(f"saved cracked  (Gauss)          {wall['maximum_principal_stress_pa']/1e6:.6g}"
          f" / {wall['minimum_principal_stress_pa']/1e6:.6g}", flush=True)
    for row in rows:
        wall = row.get("final", {}).get("regions", {}).get("wall")
        peaks = (f"{wall['maximum_principal_stress_pa']/1e6:.6g} / "
                 f"{wall['minimum_principal_stress_pa']/1e6:.6g}") if wall else "not available"
        print(f"{row['stage']:14} {row['status']:16} {peaks}", flush=True)
    print(f"Comparison: {output / 'comparison.json'}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--budget-s", type=int, default=570)
    parser.add_argument("--allow-extended-budget", action="store_true")
    parser.add_argument("--threads", type=int, default=wm.DEFAULT_THREADS)
    parser.add_argument("--memory-mib", type=int, default=wm.DEFAULT_MEMORY_MIB)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--legacy-lumped", action="store_true",
                        help="Explicit historical replay of the uncorrected vertex-only springs")
    args = parser.parse_args()
    if not 1 <= args.budget_s <= (14400 if args.allow_extended_budget else 570):
        parser.error("Budget must be 1-570 seconds, or up to 14400 with --allow-extended-budget")
    if not 1 <= args.threads <= 64 or not 2048 <= args.memory_mib <= 262144:
        parser.error("Threads must be 1-64 and memory 2048-262144 MiB")
    study = args.study_dir.resolve()
    output = (args.output_dir or study / "diagnostics").resolve()
    if output == study or study not in output.parents or output.parts[len(study.parts)] != "diagnostics":
        parser.error("Diagnostic outputs must be inside <study-dir>\\diagnostics")
    if args.stage == "inventory":
        inventory(study, output)
        print(f"Inventory: {output / 'inventory.json'}", flush=True)
    elif args.stage == "compare":
        compare(study, output)
    else:
        directory = prepare(study, output, args.stage, args.budget_s, args.threads, args.memory_mib,
                            legacy_lumped=args.legacy_lumped)
        print(f"Prepared: {directory}", flush=True)
        if not args.prepare_only:
            solve(directory)


if __name__ == "__main__":
    main()
