"""Isolated fixed-load selected-wall sweeps; never submit or modify application jobs."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import numpy as np

from Aster import automatic_cohesive as native
from Aster import automatic_preparation as preparation

TARGET_M = .1
CAPS = (1500, 3000, 6000, 9000)
MAX_LAUNCHES = 64
RESERVED = 8
REFERENCE_JOB = "d7052bc6d4a34315a6daacdb77a72b42"


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Case:
    name: str
    cap: int = 3000
    e_gpa: float = 35.
    nu: float = .2
    ft_mpa: float = .8
    gf: float = 100.
    bonded: bool = False
    fine: bool = False

    def properties(self, concrete: dict) -> dict:
        if (self.cap not in CAPS or not .1 <= self.e_gpa <= 35 or
                not .02 <= self.ft_mpa <= .8 or not 1 <= self.gf <= 150 or
                self.nu not in (.15, .2)):
            raise ValueError(f"Case outside authorised bounds: {self}")
        return dict(concrete, youngs_modulus=self.e_gpa * 1e9, poisson_ratio=self.nu,
                    tensile_strength_pa=self.ft_mpa * 1e6,
                    cohesive_fracture_energy_j_m2=self.gf)


def initial_cases() -> list[Case]:
    return [Case("A1-reproduction", e_gpa=10, nu=.15, gf=150),
            Case("A2-intact-topology"), Case("A3-stiffness-only", e_gpa=10),
            *[Case(f"B-ft-{ft:g}", ft_mpa=ft) for ft in (.8, .4, .2, .1, .05, .02)],
            *[Case(f"C-ft-{ft:g}-gf-{gf:g}", ft_mpa=ft, gf=gf)
              for ft in (.2, .05, .02) for gf in (25., 5., 1.)]]


def confirmations(case: Case) -> list[Case]:
    cases = [replace(case, name="F-finer-repeat", fine=True),
             replace(case, name="F-bonded-control", bonded=True, fine=True)]
    for field, value in (("ft_mpa", .8), ("gf", 100.), ("e_gpa", 35.)):
        if getattr(case, field) != value:
            cases.append(replace(case, name=f"F-restore-{field}", fine=True, **{field: value}))
    return cases


def validate_reference(job: Path) -> dict:
    config = json.loads((job / "config.json").read_text(encoding="utf-8"))
    baseline = job / "cracking" / "aster_elastic_baseline"
    manifest = json.loads((baseline / "mesh_manifest.json").read_text(encoding="utf-8"))
    model = Path(manifest["source_mesh"]).parent
    preparation.baseline_inventory(baseline, job / "config.json", model)
    concrete = config["material"]["concrete"]
    if (concrete["youngs_modulus"] != 35e9 or concrete["tensile_strength_pa"] != .8e6 or
            concrete["density"] != 2400 or concrete["poisson_ratio"] != .2):
        raise ValueError("Reference concrete differs from the approved campaign")
    return {"configuration": config, "baseline": str(baseline), "model": str(model),
            "hashes": {str(path): native.file_hash(path) for path in
                       [job / "config.json", model / "config.json", model / "three_body.msh",
                        *[baseline / name for name in preparation.BASELINE_FILES]]}}


def verify_hashes(hashes: dict) -> None:
    for path, expected in hashes.items():
        if native.file_hash(Path(path)) != expected:
            raise ValueError(f"Campaign reference changed: {path}")


def prepare_cap(root: Path, cap: int, reference: dict) -> dict:
    directory = root / "caps" / str(cap)
    result_path = directory / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result["status"] == "prepared":
            verify_hashes(result["hashes"])
        return result
    directory.mkdir(parents=True, exist_ok=False)
    config = json.loads(json.dumps(reference["configuration"]))
    config["damage_selection"]["max_damage_count"] = cap
    config["damage_selection"]["maximum_overload_factor"] = 1.
    # Only the screening cap differs. Use the authenticated original inventory;
    # never pass a derivative configuration off as the baseline's solved config.
    baseline, model = Path(reference["baseline"]), Path(reference["model"])
    original_config = Path(next(iter(reference["hashes"])))
    try:
        screening = directory / "screening"
        with_config = preparation.config_digest(config)
        with np.load(baseline / "three_body_mesh_arrays.npz") as data:
            arrays = {key: data[key].copy() for key in data.files}
        original, coords, cells, tags, displacement, physical, rows, hashes = (
            preparation.baseline_inventory(baseline, original_config, model))
        stress = preparation.region_cell_stresses(coords, cells, tags, displacement, original["material"])
        selected = preparation.screen_cells(stress, tags, rows, coords, cells,
                                             original["material"]["concrete"]["tensile_strength_pa"],
                                             relative_tolerance=1e-4, max_damage_count=cap)
        chosen = np.asarray(selected["selection"]["selected_source_rows"], dtype=np.int64)
        if not len(chosen):
            raise ValueError("Automatic selection is empty")
        indices = np.argsort(rows)[chosen]
        membership = preparation.candidate_region(cells, tags, indices, 0)
        faces = preparation.region_candidate_faces(cells, tags, membership)
        if len(faces) > 20000:
            raise ValueError("Existing 20000 candidate facet limit exceeded")
        split = preparation.split_interior_wall_faces(coords, cells, tags, faces, drop_fully_tied=True)
        if not len(split.source_faces):
            raise ValueError("Selected patch is not openable")
        screening.mkdir()
        np.savez_compressed(screening / "cohesive_state.npz", coordinates_m=split.coordinates_m,
                            parent_node_ids=physical[split.parent_node_ids], minus=split.faces.minus,
                            plus=split.faces.plus, areas_m2=split.faces.areas)
        summary = {"status": "prepared_aster_candidates", "solver": "code_aster",
                   "source_job_id": root.name, "config_digest": with_config, "baseline_sha256": hashes,
                   "load_factor": 1., "maximum_overload_factor": 1.,
                   "tensile_strength_pa": .8e6, "fracture_energy_j_m2": 100.,
                   "material_assignment": {"properties": original["material"]["concrete"],
                                           "selected_source_rows": chosen.tolist()}}
        write_json(screening / "summary.json", summary)
        write_json(screening / "manifest.json", {"status": "prepared_aster_candidates",
                   "summary_sha256": native.file_hash(screening / "summary.json"),
                   "candidate_state_sha256": native.file_hash(screening / "cohesive_state.npz")})
        xyz = coords[cells]
        volumes = np.abs(np.linalg.det(xyz[:, 1:] - xyz[:, :1])) / 6
        result = {"status": "prepared", "cap": cap, "actual_count": len(chosen),
                  "volume_fraction": float(volumes[indices].sum() / volumes[tags == 1].sum()),
                  "candidate_faces": len(split.source_faces), "selection": selected["selection"],
                  "selected_rows_sha256": hashlib.sha256(chosen.tobytes()).hexdigest(),
                  "hashes": {str(path): native.file_hash(path) for path in screening.iterdir()}}
        write_json(directory / "derivative_config.json", config)
        del arrays
    except (ValueError, KeyError, native.CorroborationError) as error:
        result = {"status": "preparation_limited", "cap": cap, "error": str(error)}
    write_json(result_path, result)
    return result


STEP_CAPTURE = '''
from pathlib import Path
CAMPAIGN_DIR = Path(STEPS_PATH).parent
CAMPAIGN_WALL_NODES = np.asarray(__WALL_NODES__, dtype=int)
CAMPAIGN_SOURCE_IDS = np.asarray(__SOURCE_IDS__, dtype=int)
CAMPAIGN_HISTORY = []
CAMPAIGN_SEEN = set()


def campaign_capture(result):
    access = result.getAccessParameters()
    for order, instant in zip(result.getIndexes(), access['INST']):
        if order in CAMPAIGN_SEEN:
            continue
        u = nodal_displacement(result.getField('DEPL', order))
        physical = u[CAMPAIGN_WALL_NODES]
        if not np.isfinite(physical).all():
            raise ValueError('Non-finite physical wall displacement')
        norm = np.linalg.norm(physical, axis=1)
        peak = int(CAMPAIGN_WALL_NODES[np.argmax(norm)])
        row = dict(load_factor=float(instant), maximum_wall_displacement_m=float(norm.max()),
                   component_maximum_m=np.abs(physical).max(axis=0).tolist(),
                   peak_source_node=int(CAMPAIGN_SOURCE_IDS[peak]),
                   peak_xyz_m=np.asarray(MAIL.getCoordinates().getValues()).reshape(-1, 3)[peak].tolist(),
                   native_equilibrium='STAT_NON_LINE accepted instant; raw residuals in solver.live.log')
        filename = f'campaign_displacement_{len(CAMPAIGN_HISTORY):04d}.npz'
        with (CAMPAIGN_DIR / (filename + '.tmp')).open('wb') as stream:
            np.savez_compressed(stream, displacement_m=u)
        (CAMPAIGN_DIR / (filename + '.tmp')).replace(CAMPAIGN_DIR / filename)
        row['displacement_file'] = filename
        CAMPAIGN_HISTORY.append(row)
        CAMPAIGN_SEEN.add(order)
        temporary = CAMPAIGN_DIR / 'campaign_steps.json.tmp'
        temporary.write_text(json.dumps(CAMPAIGN_HISTORY))
        temporary.replace(CAMPAIGN_DIR / 'campaign_steps.json')
'''


def configure_case(directory: Path, case: Case) -> None:
    path = directory / native.CASE_NAME
    text = path.read_text(encoding="utf-8")
    inputs = json.loads((directory / "case_inputs.json").read_text())
    targets = [float(v) for v in np.round(np.arange(.01, 1., .02 if case.fine else .05), 8)] + [1.]
    old_targets = "TARGETS = [" + ", ".join(f"{v:.10g}" for v in inputs["load_factors"]) + "]"
    if text.count(old_targets) != 1 or text.count("    reached = target\n") != 1:
        raise ValueError("Native renderer anchors changed; campaign patch refused")
    text = text.replace(old_targets, f"TARGETS = {targets!r}")
    with np.load(directory / native.SPLIT_ARRAYS_NAME) as data:
        wall_nodes = np.unique(data["tetra4_concrete"][data["concrete_body_ids"] == 1])
        if np.any(data["source_node_ids"][wall_nodes] < 0):
            raise ValueError("Auxiliary nodes found in physical wall")
        capture = STEP_CAPTURE.replace("__WALL_NODES__", repr(wall_nodes.tolist()))
        capture = capture.replace("__SOURCE_IDS__", repr(data["source_node_ids"].tolist()))
        if case.bonded:
            parent = np.arange(len(data["nodes"]))

            def find(node):
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = parent[node]
                return node

            for a, b in data["tie_node_pairs"]:
                parent[find(a)] = find(b)
            pairs = []
            for face in data["penta6_joints"]:
                for a, b in zip(face[:3], face[3:]):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
                        pairs.append((int(a), int(b)))
            capture += (
                "\nCAMPAIGN_BOND = AFFE_CHAR_MECA(MODELE=MODELE, LIAISON_DDL=[\n"
                "    _F(NOEUD=(f'N{a+1}', f'N{b+1}'), DDL=(dof,dof),\n"
                "       COEF_MULT=(1.,-1.), COEF_IMPO=0.)\n"
                f"    for a,b in {pairs!r} for dof in ('DX','DY','DZ')])\n")
            text = text.replace("EXCIT = (\n", "EXCIT = (\n    _F(CHARGE=CAMPAIGN_BOND),\n")
    text = text.replace("save()\nEXCIT = (", capture + "\nsave()\nEXCIT = (")
    text = text.replace("    reached = target\n", "    campaign_capture(RESU)\n    reached = target\n")
    text = text.replace("    except CA.AsterError as error:\n", (
        "    except CA.AsterError as error:\n"
        "        if RESU is not None:\n"
        "            campaign_capture(RESU)\n"))
    # A campaign continues to design load, never stops at first tiny softening opening.
    text = text.replace("    if True and target >= 1.0 and np.any(", "    if False and target >= 1.0 and np.any(")
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8", newline="\n")
    inputs.update(load_factors=targets, case_sha256=native.file_hash(path),
                  campaign={"case": asdict(case), "fixed_load": True,
                            "native_residual_evidence": "run/solver.live.log"})
    write_json(directory / "case_inputs.json", inputs)


def summarize_case(directory: Path, case: Case, status: dict) -> dict:
    steps_path = directory / "run" / "campaign_steps.json"
    steps = json.loads(steps_path.read_text()) if steps_path.is_file() else []
    reached = max((row["load_factor"] for row in steps), default=0.)
    peak = max((row["maximum_wall_displacement_m"] for row in steps), default=0.)
    crossing = next((row for row in steps if row["maximum_wall_displacement_m"] >= TARGET_M), None)
    completed = status["status"] == "completed" and status["returncode"] == 0 and reached >= 1.
    outcome = ("converged_at_or_above_100mm" if completed and crossing else
               "converged_below_100mm" if completed else "nonconverged_unresolved")
    lower = next((r["load_factor"] for r in reversed(steps)
                  if crossing and r["load_factor"] < crossing["load_factor"]), 0.)
    return {"case": asdict(case), "status": outcome, "reached_load_factor": reached,
            "maximum_wall_displacement_m": peak, "crossing": crossing, "accepted_steps": len(steps),
            "crossing_bracket": [lower, crossing["load_factor"]] if crossing else None,
            "native_status": status, "engineering_validated": False,
            "native_energy_balance": "unavailable; no inferred exact dissipation"}


def resource_check(root: Path, minimum_free_gb: float) -> None:
    free = shutil.disk_usage(root).free / 1024**3
    if free < minimum_free_gb:
        raise OSError(f"Insufficient campaign disk space: {free:.2f} GiB free")


def render_report(root: Path, state: dict) -> None:
    results = state["results"]
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["name", "cap", "e_gpa", "nu", "ft_mpa", "gf", "bonded", "status",
                  "reached_load_factor", "maximum_wall_displacement_m"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: {**result["case"], **result}.get(key) for key in fields})
    hits = [r for r in results if r["status"] == "converged_at_or_above_100mm"]
    write_json(root / "findings.json", {
        "threshold": "converged physical wall displacement >=0.100 m at LF<=1",
        "threshold_cases": [r["case"]["name"] for r in hits],
        "conclusion": "Threshold candidates require causal controls and equilibrium review" if hits else
                      "100 mm not reached in completed cases; unresolved cases do not establish failure",
        "finished": state["status"], "launches": state["launches"],
        "limitations": ["Synthetic materials", "No crushing or crack-water pressure",
                        "No mesh-convergence proof", "Raw native residual logs require review",
                        "No native nonlinear energy balance"]})
    # Standalone SVG, no plotting dependency or physical deformation magnification.
    valid = [r for r in results if "maximum_wall_displacement_m" in r]
    bars = []
    scale = max([TARGET_M, *[r["maximum_wall_displacement_m"] for r in valid]])
    for i, result in enumerate(valid):
        width = 500 * result["maximum_wall_displacement_m"] / scale
        bars.append(f'<rect x="260" y="{25+i*24}" width="{width:.3f}" height="16" fill="#267a99"/>'
                    f'<text x="5" y="{38+i*24}">{result["case"]["name"]}</text>')
    (root / "displacement.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="{max(80,50+len(valid)*24)}">'
        '<text x="5" y="16">Peak accepted wall displacement (metres); unresolved cases are not failure results</text>'
        + "".join(bars) + "</svg>", encoding="utf-8")


def run_campaign(job: Path, root: Path, *, preflight_only: bool, execute: bool,
                 case_time_limit: int | None, minimum_free_gb: float) -> dict:
    if execute == preflight_only:
        raise ValueError("Choose exactly one of --preflight-only and --execute")
    root, job = root.resolve(), job.resolve()
    if root.is_relative_to(job.parent) or job.is_relative_to(root):
        raise ValueError("Campaign output must be outside application job storage")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "campaign.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    finished = threading.Event()
    state_path = root / "campaign.json"
    state = {"status": "preparing", "launches": 0, "results": []}
    try:
        reference = validate_reference(job)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["reference"]["hashes"] != reference["hashes"]:
                raise ValueError("Existing campaign belongs to another reference")
            if state.get("active_case"):
                raise ValueError("Interrupted active case: preserve evidence and create a new campaign")
        state["reference"] = reference
        state["runtime"] = {"wall_clock_limit_s": case_time_limit,
                            "native_cpu_guard": "Installed run_aster default when wall-clock is uncapped",
                            "minimum_free_disk_gb": minimum_free_gb, "maximum_launches": MAX_LAUNCHES}
        write_json(state_path, state)

        def heartbeat():
            while not finished.wait(30):
                logs = list(root.glob("cases/*/native/run/solver.live.log"))
                latest = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
                write_json(root / "heartbeat.json", {"timestamp": time.time(), "pid": os.getpid(),
                           "latest_log": str(latest) if latest else None,
                           "last_log_time": latest.stat().st_mtime if latest else None})

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        caps = {}
        for cap in CAPS:
            resource_check(root, minimum_free_gb)
            verify_hashes(reference["hashes"])
            caps[cap] = prepare_cap(root, cap, reference)
        state["caps"] = caps
        if preflight_only:
            state["status"] = "preflight_completed"
            return state
        if caps[3000]["status"] != "prepared":
            raise ValueError("Reference cap cannot prepare; campaign blocked")
        queue = initial_cases()
        extent_added = False
        confirming = False
        done = {r["case"]["name"] for r in state["results"]}
        while queue:
            case = queue.pop(0)
            if case.name in done:
                continue
            if (root / "CANCEL").exists():
                raise KeyboardInterrupt
            limit = MAX_LAUNCHES if confirming else MAX_LAUNCHES - RESERVED
            if state["launches"] >= limit:
                state["status"] = "launch_budget_exhausted"
                break
            resource_check(root, minimum_free_gb)
            verify_hashes(reference["hashes"])
            directory = root / "cases" / case.name
            directory.mkdir(parents=True, exist_ok=False)
            candidate = directory / "candidate"
            shutil.copytree(root / "caps" / str(case.cap) / "screening", candidate)
            summary = json.loads((candidate / "summary.json").read_text())
            summary["material_assignment"]["properties"] = case.properties(reference["configuration"]["material"]["concrete"])
            summary.update(tensile_strength_pa=case.ft_mpa * 1e6, fracture_energy_j_m2=case.gf)
            write_json(candidate / "summary.json", summary)
            manifest = json.loads((candidate / "manifest.json").read_text())
            manifest["summary_sha256"] = native.file_hash(candidate / "summary.json")
            write_json(candidate / "manifest.json", manifest)
            derivative = json.loads(json.dumps(reference["configuration"]))
            derivative["damage_selection"].update(max_damage_count=case.cap,
                damaged_element_properties=summary["material_assignment"]["properties"])
            write_json(directory / "config.json", derivative)
            write_json(directory / "provenance.json", {"original_hashes": reference["hashes"],
                       "baseline_reuse": "Unmodified accepted intact baseline; selected changes applied only after screening",
                       "derivative_config_sha256": native.file_hash(directory / "config.json"),
                       "selection": caps[case.cap]["selected_rows_sha256"], "case": asdict(case)})
            try:
                native.prepare(Path(reference["baseline"]), candidate, directory / "native")
                configure_case(directory / "native", case)
            except (ValueError, native.CorroborationError) as error:
                result = {"case": asdict(case), "status": "preparation_limited", "error": str(error)}
            else:
                state.update(status="running", active_case=case.name, launches=state["launches"] + 1)
                write_json(state_path, state)
                started = time.monotonic()
                print(f"Starting {case.name}: {asdict(case)}", flush=True)
                status = native.solve(directory / "native", case_time_limit)
                result = summarize_case(directory / "native", case, status)
                result["elapsed_s"] = time.monotonic() - started
                if not case.bonded:
                    report = native.postprocess(directory / "native", candidate, status)
                    write_json(directory / "native" / "aster_cohesive_report.json", report)
                    result["joint_steps"] = report.get("steps", [])
                print(f"Finished {case.name}: {result['status']}", flush=True)
            write_json(directory / "result.json", result)
            state["results"].append(result)
            done.add(case.name)
            state.pop("active_case", None)
            write_json(state_path, state)
            render_report(root, state)
            if result["status"] == "converged_at_or_above_100mm" and not confirming:
                queue = confirmations(case)
                confirming = True
            if not queue and not extent_added and not confirming:
                candidates = [r for r in state["results"] if r["status"] == "converged_below_100mm"]
                if not candidates:
                    break
                best = sorted(candidates, key=lambda r: r["maximum_wall_displacement_m"], reverse=True)[:2]
                for cap, preparation_result in caps.items():
                    if preparation_result["status"] == "prepared" and cap != 3000:
                        queue.extend(replace(Case(**r["case"]), name=f"D-{cap}-{i}", cap=cap)
                                     for i, r in enumerate(best))
                largest = max(cap for cap, r in caps.items() if r["status"] == "prepared")
                queue.extend(Case(f"E-{e:g}", cap=largest, e_gpa=e) for e in (10., 5., 1., .1))
                chosen = Case(**best[0]["case"])
                queue.extend(replace(chosen, name=f"E-combined-{e:g}", cap=largest, e_gpa=e)
                             for e in (1., .1))
                extent_added = True
        else:
            state["status"] = "completed"
        if state["status"] == "running":
            state["status"] = "completed_with_unresolved_cases"
        return state
    except KeyboardInterrupt:
        state["status"] = "cancelled"
        raise
    except (OSError, ValueError, native.CorroborationError) as error:
        state.update(status="blocked", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        finished.set()
        write_json(state_path, state)
        render_report(root, state)
        lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true", help="Explicit full native campaign authorisation")
    parser.add_argument("--case-time-limit", type=int, default=None,
                        help="Optional outer seconds; omitted means no outer wall-clock cap (native CPU guard remains)")
    parser.add_argument("--minimum-free-gb", type=float, default=10.)
    args = parser.parse_args()
    if args.case_time_limit is not None and args.case_time_limit <= 0:
        parser.error("--case-time-limit must be positive")
    if args.minimum_free_gb < 1:
        parser.error("--minimum-free-gb must be at least 1")
    run_campaign(args.job, args.output, preflight_only=args.preflight_only, execute=args.execute,
                 case_time_limit=args.case_time_limit, minimum_free_gb=args.minimum_free_gb)


if __name__ == "__main__":
    main()
