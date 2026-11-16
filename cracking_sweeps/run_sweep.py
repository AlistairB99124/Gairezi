"""Wall-geometry sweep: the baselines and the contact-joint cracking stage for every scenario.

Each entry of ``scenarios.json`` is an overwrite of ``config.json`` (both beside this file).
For every scenario, in order and one at a time, the sweep runs what the app runs for a job:

1. the Elmer baseline,
2. the Code_Aster elastic baseline,
3. the Code_Aster contact joint at the wall/foundation contact: dry, then with water in the joint.

When the scenarios are done it writes one row per scenario into ``FEA_Matrix.xlsx``.
Nothing here is an engineering assessment; see ``../aster_dam_baseline.md``.
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Code_Aster's export needs a finite limit; the app gives its elastic baseline the same one.
ASTER_BASELINE_TIME_LIMIT_S = 30 * 24 * 3600
# Limit of each of the two contact-joint solves, as in the app.
JOINT_SOLVE_LIMIT_S = 600
JOINT_CASES = (("dry", "dry"), ("water", "water_in_joint"))
FIGURES = ("tension_mpa", "compression_mpa", "displacement_mm")

# ---- layout of the "Sweep" sheet of FEA_Matrix.xlsx
SHEET_NAME = "Sweep"
FIRST_ROW = 3
LAST_COLUMN = 29  # AC
SHEET_STEPS = 2  # the sheet has columns for two thickness steps
EXPECTED_HEADERS = {"A2": "#", "B2": "Width", "E2": "Width", "F2": "Height", "I2": "Upstream", "J2": "Elmer",
                    "K2": "Aster", "L2": "Ave", "S2": "Volume (m^3)", "U1": "Cracking Opening (mm)", "U2": "Dry",
                    "AB2": "Water"}
GEOMETRY_COLUMNS = {"width_m": "B", "foundation_depth_m": "C", "foundation_side_margin_m": "D",
                    "upstream_radius_m": "I"}
STEP_COLUMNS = (("E", "F"), ("G", "H"))
# Result figure -> (Elmer column, Aster column, column of their average)
BASELINE_COLUMNS = {"tension_mpa": ("J", "K", "L"), "compression_mpa": ("M", "N", "O"),
                    "displacement_mm": ("P", "Q", "R")}
VOLUME_COLUMN, VOLUME_DIFFERENCE_COLUMN = "S", "T"
# Cracking figure -> (dry column, water column)
CRACKING_COLUMNS = {"opening_mm": ("U", "V"), "tension_mpa": ("W", "X"), "compression_mpa": ("Y", "Z"),
                    "displacement_mm": ("AA", "AB")}
NOTE_COLUMN = "U"
# Volume of the mortar fill on the wall's steps (material.mortar); "-" for a wall without one. The
# column is added to a workbook that does not have it yet.
MORTAR_VOLUME_COLUMN = "AC"
MORTAR_HEADERS = {"AC1": "Mortar fill", "AC2": "Volume (m^3)"}


class SweepError(ValueError):
    """The sweep cannot be defined or its workbook cannot be written as asked."""


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# ------------------------------------------------------------------ scenarios

def merge(base, overwrite):
    """``base`` with ``overwrite`` applied: objects are merged key by key, anything else is replaced."""
    if isinstance(base, dict) and isinstance(overwrite, dict):
        merged = dict(base)
        for key, value in overwrite.items():
            merged[key] = merge(base[key], value) if key in base else deepcopy(value)
        return merged
    return deepcopy(overwrite)


def sheet_geometry(config: dict) -> dict:
    """What the sheet records of a scenario's geometry."""
    wall, foundation = config["geometry"]["wall"], config["geometry"]["foundation"]
    steps = wall.get("thickness_steps") or []
    if len(steps) > SHEET_STEPS:
        raise SweepError(f"the sheet has columns for {SHEET_STEPS} thickness steps; this wall has {len(steps)}")
    from Elmer.model_config import mortar_fill

    fill = mortar_fill(config)
    return {"width_m": wall["radial_width_m"], "foundation_depth_m": foundation["depth_m"],
            "foundation_side_margin_m": foundation["side_margin_m"], "upstream_radius_m": wall["upstream_radius_m"],
            "steps": [{"width_m": step["radial_width_m"], "top_z_m": step["top_z_m"]} for step in steps],
            # None unless material.mortar is given and the wall has a step for it to sit on.
            "mortar_fill": None if fill is None else {"slope_run_per_rise": fill["slope_run_per_rise"]}}


def define_scenarios(config: dict, overwrites: object) -> list[dict]:
    """One validated configuration per entry of ``overwrites``, numbered from 1 in file order."""
    from Elmer.model_config import ConfigurationValidationError, validate_config

    try:
        validate_config(config)
    except ConfigurationValidationError as error:
        raise SweepError(f"config.json: {error}") from None
    if not isinstance(overwrites, list) or not overwrites:
        raise SweepError("scenarios.json must be a non-empty array of overwrites of config.json")
    scenarios = []
    for number, overwrite in enumerate(overwrites, start=1):
        label = f"scenario {number}"
        if not isinstance(overwrite, dict):
            raise SweepError(f"{label}: must be an object")
        derived = merge(config, overwrite)
        try:
            validate_config(derived)
        except ConfigurationValidationError as error:
            raise SweepError(f"{label}: {error}") from None
        # What the contact-joint stage needs; the app blocks the stage for anything else.
        if derived["geometry"].get("foundation") is None:
            raise SweepError(f"{label}: the contact joint is placed between the wall and a concrete foundation; "
                             "this configuration has no foundation")
        if (derived.get("bedrock_analysis") or {}).get("mode") != "elastic_layer":
            raise SweepError(f"{label}: the contact-joint stage is implemented for the elastic-layer bedrock mode")
        try:
            geometry = sheet_geometry(derived)
        except SweepError as error:
            raise SweepError(f"{label}: {error}") from None
        scenarios.append({"number": number, "id": f"scenario_{number:02d}", "overwrite": overwrite,
                          "geometry": geometry, "config": derived})
    return scenarios


def describe(geometry: dict) -> str:
    steps = "".join(f", {step['width_m']:g} m below z = {step['top_z_m']:g} m" for step in geometry["steps"])
    fill = geometry.get("mortar_fill")
    slope = "" if not fill else f", mortar fill on the steps at 1 in {1 / fill['slope_run_per_rise']:g}"
    return f"wall {geometry['width_m']:g} m{steps}{slope}"


# ------------------------------------------------------------------ results of one scenario

def elmer_figures(report_path: Path) -> dict:
    report = read_json(report_path)
    stress = report["stress_analysis"]
    return _figures(stress["max_tensile_principal_stress_pa"]["value_pa"],
                    stress["max_compressive_principal_stress_pa"]["value_pa"],
                    report["stress_bodies"]["wall"]["maximum_displacement_m"], "Elmer report")


def aster_figures(baseline_dir: Path) -> dict:
    """Wall peaks of a completed Code_Aster elastic baseline, read from the pipeline's own files as the app does.

    Also writes ``aster_baseline_report.json``, which the app writes for its jobs and from which
    the contact-joint stage takes the elastic figures it reports beside its own.
    """
    manifest = read_json(baseline_dir / "run_manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError(f"The Code_Aster elastic solve did not complete ({manifest.get('status')})")
    comparison = read_json(baseline_dir / "comparison_report.json")
    wall = comparison["regions"]["wall"]
    tension, compression = wall["aster_tension_pa"]["value_pa"], wall["aster_compression_pa"]["value_pa"]
    displacement = wall.get("aster_maximum_displacement_m")
    figures = _figures(tension, compression, displacement, "Code_Aster baseline comparison")
    write_json(baseline_dir / "aster_baseline_report.json", {
        "schema_version": 1, "solver": "code_aster", "written_by": "cracking_sweeps.run_sweep",
        "maximum_principal_tensile_stress_pa": tension, "maximum_principal_compressive_stress_pa": compression,
        "maximum_wall_displacement_m": displacement, "peak_scope": "wall",
        "baseline_verified": comparison.get("baseline_verified") is True,
        "comparison_status": comparison.get("status")})
    return figures


def _figures(tension_pa, compression_pa, displacement_m, source: str) -> dict:
    if not all(finite(value) for value in (tension_pa, compression_pa, displacement_m)):
        raise ValueError(f"The {source} has no finite stress and displacement figures")
    return {"tension_mpa": tension_pa / 1e6, "compression_mpa": compression_pa / 1e6,
            "displacement_mm": displacement_m * 1e3}


def concrete_volume_m3(baseline_dir: Path) -> float:
    """Volume of the wall and foundation tetrahedra of the Code_Aster baseline mesh."""
    import numpy as np

    with np.load(baseline_dir / "three_body_mesh_arrays.npz", allow_pickle=False) as arrays:
        corners = arrays["nodes"][arrays["tetra4_concrete"]]
    volume = float(np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])).sum() / 6.0)
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("The baseline mesh has no concrete volume")
    return volume


def mortar_volume_m3(baseline_dir: Path) -> float | None:
    """Volume of the mortar fill tetrahedra of the Code_Aster baseline mesh; None for a model without a fill.

    The fill is its own body and is not part of concrete_volume_m3.
    """
    import numpy as np

    with np.load(baseline_dir / "three_body_mesh_arrays.npz", allow_pickle=False) as arrays:
        if "tetra4_mortar" not in arrays.files or not len(arrays["tetra4_mortar"]):
            return None
        corners = arrays["nodes"][arrays["tetra4_mortar"]]
    volume = float(np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])).sum() / 6.0)
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("The baseline mesh has a mortar fill without volume")
    return volume


def joint_figures(joint_dir: Path) -> tuple[dict, str | None]:
    """Figures at full reservoir of each contact-joint case that reached it, and why the stage failed (if it did)."""
    report = read_json(joint_dir / "joint_opening_report.json")
    figures = {}
    for label, key in JOINT_CASES:
        case = report["cases"].get(key) or {}
        levels = case.get("levels") or []
        stress = case.get("final_wall_stress")
        if case.get("status") != "completed" or not levels or levels[-1].get("fill_fraction") != 1.0 or not stress:
            continue
        level = levels[-1]
        figures[label] = {"opening_mm": level["largest_opening_m"] * 1e3,
                          **_figures(stress["maximum_principal_tensile_stress_pa"],
                                     stress["maximum_principal_compressive_stress_pa"],
                                     level["maximum_wall_displacement_m"], f"contact-joint report ({key})"),
                          "opened_share": level["opened_share"]}
    failure = None
    if report.get("status") != "completed" or len(figures) != len(JOINT_CASES):
        failure = str(report.get("message") or "The contact-joint stage did not complete both cases")
    return figures, failure


# ------------------------------------------------------------------ running

def run_module(module: str, arguments: list[str], log: Path, allowed_codes: tuple[int, ...] = (0,)) -> int:
    """Run one stage as its own process, with its output in ``log``; Ctrl+C stops its whole process tree."""
    command = [sys.executable, "-u", "-B", "-m", module, *arguments]
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
        raise RuntimeError(f"{module} exited with code {code}; see {log}")
    return code


def say(run: Path, message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {message}"
    print(line, flush=True)
    with (run / "progress.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def run_scenario(run: Path, scenario: dict, runner=None) -> dict:
    """Run the three stages of one scenario in its own directory; a failed stage ends the scenario.

    Figures of the stages that did complete are kept, so a row can still show its baselines.
    ``runner`` stands in for run_module in tests.
    """
    runner = runner or run_module
    directory = run / scenario["id"]
    directory.mkdir()
    write_json(directory / "config.json", scenario["config"])
    baseline, joint = directory / "aster_baseline", directory / "joint_opening"
    result = {"number": scenario["number"], "id": scenario["id"], "geometry": scenario["geometry"],
              "status": "running", "failed_stage": None, "error": None, "seconds": {},
              "elmer": None, "aster": None, "concrete_volume_m3": None, "mortar_volume_m3": None, "cracking": {}}

    def elmer() -> None:
        runner("Elmer.three_body_solver", [
            "--config-root", str(directory), "--output-dir", str(directory / "elmer"),
            "--report-path", str(directory / "elmer_report.json")], directory / "elmer.log")
        result["elmer"] = elmer_figures(directory / "elmer_report.json")

    def aster_baseline() -> None:
        # As the app does, the baseline is judged by its report: exit code 2 only says that the
        # comparison against Elmer still requires verification.
        runner("Aster.run_three_body_pipeline", [
            "--source-mesh", str(directory / "elmer" / "model" / "three_body.msh"), "--output-dir", str(baseline),
            "--time-limit-s", str(ASTER_BASELINE_TIME_LIMIT_S), "--runtime", "wsl" if os.name == "nt" else "native",
        ], directory / "aster_baseline.log", allowed_codes=(0, 2))
        result["aster"] = aster_figures(baseline)
        result["concrete_volume_m3"] = concrete_volume_m3(baseline)
        result["mortar_volume_m3"] = mortar_volume_m3(baseline)
        if (result["mortar_volume_m3"] is None) != (scenario["geometry"].get("mortar_fill") is None):
            raise ValueError("The baseline mesh and the scenario disagree on whether the wall has a mortar fill")

    def joint_opening() -> None:
        # Exit code 2 means a case did not complete; the report says which, and why.
        runner("Aster.dam_joint_baseline", [
            "app", "--baseline-dir", str(baseline), "--output-dir", str(joint),
            "--time-limit-s", str(JOINT_SOLVE_LIMIT_S), "--progress-log", str(run / "progress.log"),
        ], directory / "joint_opening.log", allowed_codes=(0, 2))
        result["cracking"], failure = joint_figures(joint)
        if failure:
            raise RuntimeError(failure)

    for stage, label, action in (("elmer", "Elmer baseline", elmer),
                                 ("aster_baseline", "Code_Aster elastic baseline", aster_baseline),
                                 ("joint_opening", "Code_Aster contact joint (dry, then water in the joint)",
                                  joint_opening)):
        say(run, f"Scenario {scenario['number']}: {label} started")
        started = time.monotonic()
        try:
            action()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            result["seconds"][stage] = round(time.monotonic() - started, 1)
            reason = str(error).strip().splitlines()
            result.update(status="failed", failed_stage=stage,
                          error=f"{type(error).__name__}: {reason[0] if reason else ''}")
            say(run, f"Scenario {scenario['number']}: {label} FAILED after {result['seconds'][stage]:.0f} s: "
                     f"{result['error']} (log: {directory / (stage + '.log')})")
            break
        result["seconds"][stage] = round(time.monotonic() - started, 1)
        say(run, f"Scenario {scenario['number']}: {label} finished in {result['seconds'][stage]:.0f} s")
    else:
        result["status"] = "completed"
    write_json(directory / "result.json", result)
    return result


def save_results(run: Path, state: dict) -> None:
    write_json(run / "results.json", state)
    fields = ["number", "wall", "status", "failed_stage", "error"]
    fields += [f"{solver}_{figure}" for solver in ("elmer", "aster") for figure in FIGURES]
    fields += ["concrete_volume_m3", "mortar_volume_m3"]
    fields += [f"cracking_{label}_{figure}" for figure in CRACKING_COLUMNS for label, _ in JOINT_CASES]
    temporary = run / "results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in state["results"]:
            row = {"number": result["number"], "wall": describe(result["geometry"]), "status": result["status"],
                   "failed_stage": result["failed_stage"], "error": result["error"],
                   "concrete_volume_m3": result["concrete_volume_m3"],
                   "mortar_volume_m3": result.get("mortar_volume_m3")}
            for solver in ("elmer", "aster"):
                for figure in FIGURES:
                    row[f"{solver}_{figure}"] = (result[solver] or {}).get(figure)
            for figure in CRACKING_COLUMNS:
                for label, _ in JOINT_CASES:
                    row[f"cracking_{label}_{figure}"] = (result["cracking"].get(label) or {}).get(figure)
            writer.writerow(row)
    temporary.replace(run / "results.csv")


def execute(run: Path, scenarios: list[dict], sheet: Path | None, runner=None) -> dict:
    """Run the scenarios one after the other, then write the sheet. Results are saved after every scenario."""
    state = {"status": "running", "run": run.name, "engineering_validated": False, "sheet": None, "results": []}
    save_results(run, state)
    try:
        for scenario in scenarios:
            say(run, f"=== Scenario {scenario['number']} of the {len(scenarios)} to run: {describe(scenario['geometry'])}")
            state["results"].append(run_scenario(run, scenario, runner))
            save_results(run, state)
        failed = [result["number"] for result in state["results"] if result["status"] != "completed"]
        state.update(status="completed_with_failures" if failed else "completed", failed_scenarios=failed)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        save_results(run, state)
        raise
    save_results(run, state)
    if sheet is not None:
        try:
            state["sheet"] = update_sheet(sheet, state["results"], run)
            say(run, f"Wrote runs {', '.join(str(number) for number in state['sheet']['rows_written'])} to {sheet}")
        except (SweepError, OSError) as error:
            state["sheet"] = {"error": str(error)}
            say(run, f"The sheet was NOT updated: {error}. Results are in {run / 'results.json'}; "
                     f"write them later with --update-sheet {run}")
        save_results(run, state)
    return state


# ------------------------------------------------------------------ the workbook

def _load_sheet(path: Path):
    import openpyxl

    if not path.is_file():
        raise SweepError(f"{path} does not exist")
    try:
        workbook = openpyxl.load_workbook(path)
    except PermissionError:
        raise SweepError(f"{path} cannot be read; close it in Excel") from None
    if SHEET_NAME not in workbook.sheetnames:
        raise SweepError(f"{path} has no sheet named '{SHEET_NAME}'")
    sheet = workbook[SHEET_NAME]
    wrong = [f"{cell} is {sheet[cell].value!r}, expected {text!r}" for cell, text in EXPECTED_HEADERS.items()
             if str(sheet[cell].value or "").strip() != text]
    if wrong:
        raise SweepError(f"{path}: the '{SHEET_NAME}' sheet does not have the expected columns ({'; '.join(wrong)})")
    return workbook, sheet


def check_sheet(path: Path) -> None:
    """Fail before any solver starts if the workbook cannot be written when the sweep ends."""
    _load_sheet(path)
    try:
        with path.open("r+b"):
            pass
    except PermissionError:
        raise SweepError(f"{path} is open in another program; close it before starting the sweep") from None


def update_sheet(path: Path, results: list[dict], run: Path) -> dict:
    """Write one row per result into the sheet, replacing what the row held; other rows are left as they are.

    A figure a scenario did not produce is left empty, so nothing from an earlier sweep stays
    in a row that this sweep wrote. The workbook as it was is copied into ``run`` first.
    """
    workbook, sheet = _load_sheet(path)
    if not results:
        raise SweepError("There are no results to write")
    template = {column: sheet.cell(FIRST_ROW, column) for column in range(1, LAST_COLUMN + 1)}
    # The mortar-fill column takes its look from the concrete volume column in a workbook that does not have it yet.
    for cell, text in MORTAR_HEADERS.items():
        if sheet[cell].value is None:
            sheet[cell].value = text
            like = sheet[f"{VOLUME_COLUMN}{sheet[cell].row}"]
            if not sheet[cell].has_style and like.has_style:
                sheet[cell]._style = copy(like._style)
    if not template[LAST_COLUMN].has_style:
        template[LAST_COLUMN] = sheet[f"{VOLUME_COLUMN}{FIRST_ROW}"]

    def put(row: int, column: str, value) -> None:
        cell = sheet[f"{column}{row}"]
        if not cell.has_style and template[cell.column].has_style:
            cell._style = copy(template[cell.column]._style)
        cell.value = None if value is None else value if isinstance(value, (str, int)) else round(float(value), 3)

    for result in results:
        row = FIRST_ROW + result["number"] - 1
        for column in range(1, LAST_COLUMN + 1):
            sheet.cell(row, column).value = None
        geometry = result["geometry"]
        put(row, "A", result["number"])
        for key, column in GEOMETRY_COLUMNS.items():
            put(row, column, geometry[key])
        for index, (width_column, height_column) in enumerate(STEP_COLUMNS):
            step = geometry["steps"][index] if index < len(geometry["steps"]) else None
            put(row, width_column, step["width_m"] if step else "-")
            put(row, height_column, step["top_z_m"] if step else "-")
        for figure, (elmer_column, aster_column, average_column) in BASELINE_COLUMNS.items():
            elmer, aster = (result["elmer"] or {}).get(figure), (result["aster"] or {}).get(figure)
            put(row, elmer_column, elmer)
            put(row, aster_column, aster)
            put(row, average_column, None if elmer is None or aster is None
                else f"=AVERAGE({elmer_column}{row}:{aster_column}{row})")
        volume = result["concrete_volume_m3"]
        put(row, VOLUME_COLUMN, None if volume is None else int(round(volume)))
        put(row, VOLUME_DIFFERENCE_COLUMN, None if volume is None
            else f"={VOLUME_COLUMN}{row}-${VOLUME_COLUMN}${FIRST_ROW}")
        for figure, columns in CRACKING_COLUMNS.items():
            for (label, _), column in zip(JOINT_CASES, columns):
                put(row, column, (result["cracking"].get(label) or {}).get(figure))
        # "-" for a wall without a fill; empty when a wall with a fill did not get as far as its volume.
        fill_volume = result.get("mortar_volume_m3")
        put(row, MORTAR_VOLUME_COLUMN, round(fill_volume, 1) if fill_volume is not None
            else None if geometry.get("mortar_fill") else "-")

    written = sorted(result["number"] for result in results)
    last_row = FIRST_ROW + max(written) - 1
    # The note sits below the table; an earlier one is replaced.
    for row in range(FIRST_ROW, sheet.max_row + 1):
        value = sheet[f"{NOTE_COLUMN}{row}"].value
        if isinstance(value, str) and value.startswith("Cracking columns:"):
            sheet[f"{NOTE_COLUMN}{row}"].value = None
    data_rows = [row for row in range(FIRST_ROW, sheet.max_row + 1) if sheet.cell(row, 1).value is not None]
    note_row = max(data_rows + [last_row]) + 2
    failed = [result["number"] for result in results if result["status"] != "completed"]
    sheet[f"{NOTE_COLUMN}{note_row}"].value = (
        "Cracking columns: Code_Aster contact joint at the wall/foundation contact, full reservoir. "
        "Dry = no water in the joint; Water = water pressure in the joint. Opening = largest opening of the "
        "contact. Generic joint values; not an engineering assessment. "
        + ("Mortar fill = soft fill sloping the wall's steps, a body of its own with placeholder material values; "
           "it is not concrete and is not in the Volume column. "
           if any(result["geometry"].get("mortar_fill") for result in results) else "") +
        f"Runs {', '.join(str(number) for number in written)} written by cracking_sweeps.run_sweep, sweep {run.name}"
        + (f"; runs {', '.join(str(number) for number in failed)} did not complete, their empty cells have no result"
           if failed else "") + ".")
    backup = run / f"{path.stem}.before{path.suffix}"
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_name(path.name + ".tmp")
    try:
        workbook.save(temporary)
        temporary.replace(path)
    except PermissionError:
        temporary.unlink(missing_ok=True)
        raise SweepError(f"{path} is open in another program; close it") from None
    return {"path": str(path), "rows_written": written, "backup": str(backup), "note_row": note_row}


# ------------------------------------------------------------------ command line

def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.json", help="the configuration the scenarios overwrite")
    parser.add_argument("--scenarios", type=Path, default=HERE / "scenarios.json", help="array of overwrites, one per run")
    parser.add_argument("--sheet", type=Path, default=HERE / "FEA_Matrix.xlsx", help="workbook to update when the sweep ends")
    parser.add_argument("--output-root", type=Path, default=HERE / "runs")
    parser.add_argument("--only", type=int, nargs="+", metavar="NUMBER",
                        help="run only these scenarios (numbered from 1); only their rows are written")
    parser.add_argument("--execute", action="store_true", help="start the solvers; without it only the plan is written")
    parser.add_argument("--update-sheet", type=Path, metavar="SWEEP_DIRECTORY",
                        help="no solving: write the results of a finished sweep directory into the workbook")
    args = parser.parse_args(arguments)
    try:
        if args.update_sheet is not None:
            results = read_json(args.update_sheet / "results.json")["results"]
            outcome = update_sheet(args.sheet, results, args.update_sheet.resolve())
            print(f"Wrote runs {', '.join(str(number) for number in outcome['rows_written'])} to {args.sheet}")
            return 0
        scenarios = define_scenarios(read_json(args.config), read_json(args.scenarios))
        if args.only:
            unknown = sorted(set(args.only) - {scenario["number"] for scenario in scenarios})
            if unknown:
                raise SweepError(f"--only: there is no scenario {', '.join(str(number) for number in unknown)}; "
                                 f"scenarios.json has {len(scenarios)}")
            scenarios = [scenario for scenario in scenarios if scenario["number"] in set(args.only)]
        if args.execute:
            check_sheet(args.sheet)
    except (SweepError, OSError, json.JSONDecodeError, KeyError) as error:
        print(f"Sweep not started: {error}", file=sys.stderr)
        return 1
    args.output_root.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run = (args.output_root / name).resolve()
    run.mkdir()
    shutil.copy2(args.config, run / "config.json")
    shutil.copy2(args.scenarios, run / "scenarios.json")
    write_json(run / "plan.json", {"status": "defined", "source_config": str(args.config.resolve()),
                                   "source_scenarios": str(args.scenarios.resolve()),
                                   "sheet": str(args.sheet.resolve()), "engineering_validated": False,
                                   "scenarios": scenarios})
    print(f"Sweep directory: {run}")
    for scenario in scenarios:
        print(f"  {scenario['number']}: {describe(scenario['geometry'])}")
    if not args.execute:
        print("Plan only: no solver was started. Add --execute to run the sweep.")
        return 0
    state = execute(run, scenarios, args.sheet)
    for result in state["results"]:
        cracking = result["cracking"]
        opening = " / ".join(f"{cracking[label]['opening_mm']:.3f}" if label in cracking else "-" for label, _ in JOINT_CASES)
        print(f"  {result['number']}: {describe(result['geometry'])}: {result['status']}"
              + (f" at {result['failed_stage']}" if result["failed_stage"] else "")
              + f"; opening dry / water {opening} mm")
    print(f"Finished: {state['status']}; results: {run / 'results.json'}")
    sheet_failed = isinstance(state["sheet"], dict) and "error" in state["sheet"]
    return 2 if state["status"] != "completed" or sheet_failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Sweep interrupted; finished scenarios are kept in the sweep directory.", file=sys.stderr)
        raise SystemExit(130)
