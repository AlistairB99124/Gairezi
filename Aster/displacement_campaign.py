"""Isolated fixed-load selected-wall sweeps; never submit or modify application jobs."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

import numpy as np

from Aster import automatic_cohesive as native

TARGET_M = .1
CAPS = (1000, 2000, 4000, 8000)
MAX_LAUNCHES = 64
RESERVED = 8
REFERENCE_JOB = "8bef80bd1c324e808a7358271a5b578f"
# Sweep selection ranks interior wall facets purely by tensile stress and keeps the top
# max_damage_count connected cells; a 1 Pa floor makes every tensioned facet eligible so
# the configured concrete strength no longer gates which elements can be degraded.
SELECTION_RANKING_FLOOR_PA = 1.
SWEEP_LOG = Path(__file__).resolve().parents[1] / "sweep.log"
_PROGRESS_LOCK = threading.Lock()


def log_progress(event: str, *, path: Path = SWEEP_LOG, **details) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"{timestamp} {event} {json.dumps(details, allow_nan=False, ensure_ascii=True)}\n"
    with _PROGRESS_LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(line)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Case:
    name: str
    cap: int = 2000
    e_gpa: float = 35.
    nu: float = .2
    ft_mpa: float = 2.
    gf: float = 100.
    bonded: bool = False
    fine: bool = False

    def properties(self, concrete: dict) -> dict:
        if (self.cap not in CAPS or not .1 <= self.e_gpa <= 35 or
                not .02 <= self.ft_mpa <= 2. or not 1 <= self.gf <= 150 or
                self.nu not in (.15, .2)):
            raise ValueError(f"Case outside authorised bounds: {self}")
        return dict(concrete, youngs_modulus=self.e_gpa * 1e9, poisson_ratio=self.nu,
                    tensile_strength_pa=self.ft_mpa * 1e6,
                    cohesive_fracture_energy_j_m2=self.gf)


def initial_cases(*, hydraulic: bool = False) -> list[Case]:
    return [Case("A1-reproduction") if hydraulic else Case("A1-reproduction", e_gpa=10, nu=.15, ft_mpa=.8, gf=150),
            Case("A2-intact-topology"), Case("A3-stiffness-only", e_gpa=10),
            *([Case("A4-historical-material-full-load", e_gpa=10, nu=.15, ft_mpa=.8, gf=150)] if hydraulic else []),
            *[Case(f"B-ft-{ft:g}", ft_mpa=ft) for ft in (.8, .4, .2, .1, .05, .02)],
            *[Case(f"C-ft-{ft:g}-gf-{gf:g}", ft_mpa=ft, gf=gf)
              for ft in (.2, .05, .02) for gf in (25., 5., 1.)]]


def confirmations(case: Case, previous: list[dict] | None = None) -> list[Case]:
    cases = [replace(case, name="F-finer-repeat", fine=True),
             replace(case, name="F-bonded-control", bonded=True, fine=True)]
    for field, value in (("ft_mpa", 2.), ("gf", 100.), ("e_gpa", 35.)):
        if getattr(case, field) != value:
            cases.append(replace(case, name=f"F-restore-{field}", fine=True, **{field: value}))
    for result in reversed(previous or []):
        other = Case(**result["case"])
        axes = [key for key in ("e_gpa", "ft_mpa", "gf") if getattr(case, key) != getattr(other, key)]
        if (result["status"] == "converged_below_100mm" and other.cap == case.cap
                and other.nu == case.nu and not other.bonded and len(axes) == 1):
            axis = axes[0]
            for index, fraction in enumerate((.25, .5, .75)):
                value = getattr(case, axis)**fraction * getattr(other, axis)**(1 - fraction)
                cases.append(replace(case, name=f"F-refine-{index}", fine=True, **{axis: value}))
            break
    return cases


def validate_reference(job: Path) -> dict:
    from Aster import automatic_preparation as preparation
    from Aster.whole_model_crack import wsl_path
    config = json.loads((job / "config.json").read_text(encoding="utf-8"))
    baseline = job / "cracking" / "aster_elastic_baseline"
    manifest = json.loads((baseline / "mesh_manifest.json").read_text(encoding="utf-8"))
    model = Path(wsl_path(manifest["source_mesh"]) if os.name != "nt" else manifest["source_mesh"]).parent
    preparation.baseline_inventory(baseline, job / "config.json", model)
    concrete = config["material"]["concrete"]
    if (concrete["youngs_modulus"] != 35e9 or concrete["tensile_strength_pa"] != 2e6 or
            concrete["density"] != 2400 or concrete["poisson_ratio"] != .2):
        raise ValueError("Reference concrete differs from the approved campaign")
    saved_reports = list((job / "automatic_cracking").glob("*/aster/aster_cohesive_report.json"))
    if len(saved_reports) != 1:
        raise ValueError("Expected exactly one authoritative saved cracking reference")
    report = json.loads(saved_reports[0].read_text())
    if report["status"] != "completed" or report["reached_load_factor"] < 1.:
        raise ValueError("Saved cracking reference did not complete at or beyond the design load")
    return {"configuration": config, "baseline": str(baseline), "model": str(model),
            "saved_cracking_report": report,
            "hashes": {str(path): native.file_hash(path) for path in
                       [job / "config.json", model / "config.json", model / "three_body.msh", saved_reports[0],
                        *[baseline / name for name in preparation.BASELINE_FILES]]}}

def materialize_reference(root: Path, reference: dict) -> None:
    from Aster import automatic_preparation as preparation
    from Aster.whole_model_crack import wsl_path
    source = Path(reference["baseline"])
    destination = root / "intact_baseline_copy"
    if not destination.exists():
        destination.mkdir()
        for name in preparation.BASELINE_FILES:
            shutil.copy2(source / name, destination / name)
        manifest = json.loads((destination / "mesh_manifest.json").read_text())
        if os.name != "nt":
            for key in ("source_mesh", "source_config"):
                manifest[key] = wsl_path(manifest[key])
        write_json(destination / "mesh_manifest.json", manifest)
        write_json(destination / "reuse_provenance.json", {
            "original_hashes": reference["hashes"], "operation": "Path rebasing only; accepted fields unchanged"})
    provenance = json.loads((destination / "reuse_provenance.json").read_text())
    if provenance["original_hashes"] != reference["hashes"]:
        raise ValueError("Isolated baseline copy belongs to another source")
    for name in ("three_body_mesh_arrays.npz", "aster_full_displacement.npz", "run_manifest.json"):
        if native.file_hash(source / name) != native.file_hash(destination / name):
            raise ValueError(f"Reused intact baseline changed: {name}")
    reference["original_baseline"] = str(source)
    reference["baseline"] = str(destination)


def verify_hashes(hashes: dict) -> None:
    for path, expected in hashes.items():
        if native.file_hash(Path(path)) != expected:
            raise ValueError(f"Campaign reference changed: {path}")


def hydraulic_reference(root: Path, reference: dict, time_limit: int | None) -> None:
    from Aster import hydraulic_loads, generate_three_body_case, elastic_diagnostics
    from Aster.run_case import run_case
    source = Path(reference["baseline"])
    destination = root / "hydraulic_intact_baseline"
    evidence_path = destination / "hydraulic_reference.json"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text())
        if evidence["original_hashes"] != reference["hashes"]:
            raise ValueError("Hydraulic reference belongs to another historical job")
        verify_hashes(evidence["hashes"])
    else:
        destination.mkdir(exist_ok=False)
        shutil.copy2(source / native.SPLIT_ARRAYS_NAME, destination / native.SPLIT_ARRAYS_NAME)
        manifest = json.loads((source / native.MESH_MANIFEST_NAME).read_text())
        manifest["hydraulic_loads"] = dict(hydraulic_loads.SETTINGS)
        write_json(destination / native.MESH_MANIFEST_NAME, manifest)
        generate_three_body_case.generate(
            case_path=destination / native.CASE_NAME, inputs_path=destination / "case_inputs.json",
            mesh_manifest_path=destination / native.MESH_MANIFEST_NAME, linear_solver="MUMPS")
        log_progress("HYDRAULIC_BASELINE_START", output=str(destination))
        result = run_case(case_path=destination / native.CASE_NAME, output_dir=destination,
                          mesh_dir=destination, arrays_filename=native.SPLIT_ARRAYS_NAME,
                          converter_filename="convert_three_body_to_med.py",
                          export_name="hydraulic_intact_baseline", time_limit_s=time_limit, runtime="wsl")
        if result["status"] != "completed":
            raise ValueError(f"Hydraulic baseline solve failed: {result}")
        with np.load(destination / "aster_full_displacement.npz") as fields:
            diagnostic = elastic_diagnostics.diagnostics(destination, fields["displacement_m"])
        if diagnostic["relative_force_balance_error"] > .01 or diagnostic["relative_moment_balance_error"] > .01:
            raise ValueError(f"Hydraulic baseline equilibrium gate failed: {diagnostic}")
        evidence = {"original_hashes": reference["hashes"], "settings": dict(hydraulic_loads.SETTINGS),
                    "diagnostic": diagnostic,
                    "hashes": {str(path): native.file_hash(path) for path in [
                        destination / native.CASE_NAME, destination / "case_inputs.json",
                        destination / native.MESH_MANIFEST_NAME, destination / native.SPLIT_ARRAYS_NAME,
                        destination / "run_manifest.json", destination / "aster_full_displacement.npz"]}}
        write_json(evidence_path, evidence)
        log_progress("HYDRAULIC_BASELINE_FINISH", diagnostic=diagnostic)
    reference["baseline"] = str(destination)
    reference["hydraulic_reference"] = evidence
    with np.load(destination / native.SPLIT_ARRAYS_NAME) as arrays, \
            np.load(destination / "aster_full_displacement.npz") as fields:
        wall = np.unique(arrays["tetra4_concrete"][arrays["concrete_body_ids"] == 1])
        reference["hydraulic_intact_maximum_m"] = float(np.linalg.norm(fields["displacement_m"][wall], axis=1).max())


def validate_constitutive_gate(path: Path) -> dict:
    from Aster.whole_model_crack import wsl_path
    gate = json.loads(path.read_text())
    if (gate.get("status") != "passed" or not gate.get("unloading_retains_separation")
            or gate.get("native_status", {}).get("returncode") != 0):
        raise ValueError("Native constitutive separation/unloading gate has not passed")
    evidence = {str(Path(wsl_path(p) if os.name != "nt" else p)): digest
                for p, digest in gate["evidence"].items()}
    verify_hashes(evidence)
    step_files = [Path(p) for p in evidence if Path(p).name == native.STEPS_NAME]
    if len(step_files) != 1:
        raise ValueError("Constitutive gate lacks unique native step evidence")
    with np.load(step_files[0]) as fields:
        damage = fields["damage_state"]
        jump = fields["normal_jump_m"]
        if (len(damage) < 3 or not np.isfinite(damage).all() or not np.isfinite(jump).all()
                or not np.all(damage[0] < 1.) or not np.any((damage >= 1.) & (damage < 2.))
                or not np.any((damage[-2] >= 2. - 1e-9) & (damage[-1] >= 2. - 1e-9))):
            raise ValueError("Native fields do not demonstrate initiation, separation and retained damage")
    point_files = [Path(p) for p in evidence if Path(p).name == native.STEPS_NAME.replace(".npz", "_pointwise.npz")]
    if len(point_files) != 1:
        raise ValueError("Constitutive gate lacks unique pointwise traction/opening evidence")
    with np.load(point_files[0]) as fields:
        state, jump, traction = fields["state"], fields["normal_jump_m"], fields["normal_traction_pa"]
        if (state.shape != jump.shape or state.shape != traction.shape or state.ndim != 2
                or not all(np.isfinite(values).all() for values in (state, jump, traction))):
            raise ValueError("Constitutive integration-point fields are invalid")
        separated = state >= 2. - 1e-9
        tensile_separated = separated & (jump > 0.)
        if (not np.all(state[0] < 1.) or not np.any(tensile_separated)
                or np.any(separated[1:] < separated[:-1])
                or np.max(np.abs(traction[tensile_separated])) > 1.):
            raise ValueError("Native separated tensile points do not demonstrate zero traction and irreversibility")
    return {"path": str(path), "sha256": native.file_hash(path), "verified_evidence": evidence}


def materialize_constitutive_gate(source: Path, destination: Path) -> Path:
    verified = validate_constitutive_gate(source)
    destination.mkdir(parents=True, exist_ok=False)
    gate = json.loads(source.read_text())
    copied = {}
    for name, digest in verified["verified_evidence"].items():
        target = destination / Path(name).name
        if target.exists():
            raise ValueError("Constitutive evidence has duplicate file names")
        shutil.copy2(name, target)
        copied[str(target.resolve())] = digest
    gate["evidence"] = copied
    path = destination / "gate.json"
    write_json(path, gate)
    validate_constitutive_gate(path)
    return path


def prepare_cap(root: Path, cap: int, reference: dict) -> dict:
    from Aster import automatic_preparation as preparation
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
        original, coords, cells, tags, displacement, physical, rows, hashes = (
            preparation.baseline_inventory(baseline, original_config, model))
        stress = preparation.region_cell_stresses(coords, cells, tags, displacement, original["material"])
        # Rank-based sweep selection: every tensioned facet is eligible (1 Pa floor) and the
        # cap keeps the top-stress connected patch, decoupling selection from concrete strength.
        selected = preparation.screen_cells(stress, tags, rows, coords, cells,
                                             SELECTION_RANKING_FLOOR_PA,
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
                   "tensile_strength_pa": float(original["material"]["concrete"]["tensile_strength_pa"]),
                   "fracture_energy_j_m2": float(original["material"]["concrete"]["cohesive_fracture_energy_j_m2"]),
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
with np.load(CAMPAIGN_DIR.parent / 'campaign_reference.npz') as ref:
    CAMPAIGN_REF_U = ref['intact']
    CAMPAIGN_CREST = ref['crest_nodes']
    CAMPAIGN_AREAS = ref['areas']
    CAMPAIGN_REF_PEAK = int(ref['peak'])


def campaign_capture(result):
    access = result.getAccessParameters()
    for order, instant in zip(result.getIndexes(), access['INST']):
        if float(instant) <= 0 or float(instant) in CAMPAIGN_SEEN:
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
        row['maximum_local_vector_change_m'] = float(np.linalg.norm(
            physical - CAMPAIGN_REF_U[CAMPAIGN_WALL_NODES] * instant, axis=1).max())
        row['at_original_peak_displacement_m'] = u[CAMPAIGN_REF_PEAK].tolist()
        row['crest_mean_displacement_m'] = u[CAMPAIGN_CREST].mean(axis=0).tolist()
        if JOINT_CELLS:
            vari = result.getField('VARI_ELGA', order)
            state = joint_component(vari, 'V3').reshape(-1, 3).max(axis=1)
            normal = joint_component(vari, 'V7').reshape(-1, 3).max(axis=1)
            row['initiated_faces'] = int(np.sum(state >= 1. - 1e-9))
            row['separated_faces'] = int(np.sum(state >= 2. - 1e-9))
            row['separated_area_m2'] = float(CAMPAIGN_AREAS[state >= 2. - 1e-9].sum())
            row['peak_normal_jump_m'] = float(normal.max())
            row['peak_tangential_jump_m'] = float(max(
                np.abs(joint_component(vari, 'V8')).max(),
                np.abs(joint_component(vari, 'V9')).max()))
        filename = f'campaign_displacement_{len(CAMPAIGN_HISTORY):04d}.npz'
        with (CAMPAIGN_DIR / (filename + '.tmp')).open('wb') as stream:
            np.savez_compressed(stream, displacement_m=u)
        (CAMPAIGN_DIR / (filename + '.tmp')).replace(CAMPAIGN_DIR / filename)
        row['displacement_file'] = filename
        CAMPAIGN_HISTORY.append(row)
        CAMPAIGN_SEEN.add(float(instant))
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
        reference_u = np.zeros_like(data["nodes"])
        baseline_path = inputs.get("baseline_displacement_path")
        if baseline_path is not None:
            with np.load(baseline_path) as baseline:
                valid = baseline["source_node_ids"] >= 0
                ids, first = np.unique(baseline["source_node_ids"][valid], return_index=True)
                physical = data["source_node_ids"] >= 0
                positions = np.searchsorted(ids, data["source_node_ids"][physical])
                if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], data["source_node_ids"][physical]):
                    raise ValueError("Campaign reference physical node identity mismatch")
                reference_u[physical] = baseline["displacement_m"][valid][first][positions]
        crest = np.intersect1d(np.unique(data["tri3_crest"]) if "tri3_crest" in data else wall_nodes, wall_nodes)
        if not len(crest):
            raise ValueError("Campaign has no physical wall crest gauges")
        joint_faces = data["penta6_joints"][:, :3] if "penta6_joints" in data else np.empty((0, 3), int)
        xyz = data["nodes"][joint_faces]
        areas = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1) / 2
        peak = wall_nodes[np.argmax(np.linalg.norm(reference_u[wall_nodes], axis=1))]
        np.savez(directory / "campaign_reference.npz", intact=reference_u, crest_nodes=crest,
                 areas=areas, peak=peak)
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


def resource_check(root: Path, minimum_free_gb: float, *, solver: bool = False) -> int:
    free = shutil.disk_usage(root).free / 1024**3
    if free < minimum_free_gb:
        raise OSError(f"Insufficient campaign disk space: {free:.2f} GiB free")
    memory = available_memory_mb()
    if memory is not None and memory < (4096 if solver else 2048):
        raise OSError(f"Insufficient available memory for isolated native solve: {memory:.0f} MiB")
    return min(native.MEMORY_LIMIT_MB, int(memory / 2)) if memory is not None else native.MEMORY_LIMIT_MB


def available_memory_mb() -> float | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    raise OSError("Linux memory availability not reported")


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
                 case_time_limit: int | None, minimum_free_gb: float,
                 constitutive_gate: Path | None = None) -> dict:
    if execute == preflight_only:
        raise ValueError("Choose exactly one of --preflight-only and --execute")
    if execute and constitutive_gate is None:
        raise ValueError("Full campaign requires verified native constitutive separation/unloading evidence")
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
        log_progress("CAMPAIGN_START", output=str(root), mode="execute" if execute else "preflight")
        reference = validate_reference(job)
        materialize_reference(root, reference)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            scope = "full_hydraulic" if execute else "historical_preflight"
            if state.get("load_scope") != scope:
                raise ValueError("Existing campaign has another/legacy load scope; use a new output directory")
            if state["reference"]["hashes"] != reference["hashes"]:
                raise ValueError("Existing campaign belongs to another reference")
            if state.get("active_case"):
                raise ValueError("Interrupted active case: preserve evidence and create a new campaign")
        state["reference"] = reference
        state["load_scope"] = "full_hydraulic" if execute else "historical_preflight"
        if constitutive_gate is not None:
            state["constitutive_gate"] = validate_constitutive_gate(constitutive_gate)
        state["runtime"] = {"wall_clock_limit_s": case_time_limit,
                            "native_cpu_guard": "Installed run_aster default when wall-clock is uncapped",
                            "minimum_free_disk_gb": minimum_free_gb, "maximum_launches": MAX_LAUNCHES}
        write_json(state_path, state)

        def heartbeat():
            while not finished.wait(30):
                logs = [*root.glob("cases/*/native/run/solver.live.log"),
                        *root.glob("hydraulic_intact_baseline/run/solver.live.log")]
                latest = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
                write_json(root / "heartbeat.json", {"timestamp": time.time(), "pid": os.getpid(),
                           "active_case": state.get("active_case"), "launches": state["launches"],
                           "available_memory_mb": available_memory_mb(),
                           "latest_log": str(latest) if latest else None,
                           "last_log_time": latest.stat().st_mtime if latest else None})
                log_progress("HEARTBEAT", output=str(root), active_case=state.get("active_case"),
                             launches=state["launches"], available_memory_mb=available_memory_mb(),
                             solver_log=str(latest) if latest else None,
                             solver_log_last_update=latest.stat().st_mtime if latest else None)
                if (root / "CANCEL").exists():
                    os.kill(os.getpid(), signal.SIGINT)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        if execute:
            resource_check(root, minimum_free_gb, solver=True)
            if not (root / "hydraulic_intact_baseline").exists():
                state["launches"] += 1
                state["active_case"] = "hydraulic-intact-baseline"
                write_json(state_path, state)
            hydraulic_reference(root, reference, case_time_limit)
            state.pop("active_case", None)
            state["reference"] = reference
            write_json(state_path, state)
        else:
            log_progress("PREFLIGHT_SCOPE", note="Historical no-uplift selection preview only; execution "
                         "will solve a new uplift baseline and re-screen into a separate output directory")
        caps = {}
        for cap in CAPS:
            log_progress("CAP_PREFLIGHT_START", cap=cap)
            resource_check(root, minimum_free_gb)
            verify_hashes(reference["hashes"])
            caps[cap] = prepare_cap(root, cap, reference)
            log_progress("CAP_PREFLIGHT_FINISH", cap=cap,
                         result={key: value for key, value in caps[cap].items()
                                 if key in ("status", "actual_count", "volume_fraction", "candidate_faces", "error")})
        state["caps"] = caps
        if preflight_only:
            state["status"] = "preflight_completed"
            return state
        if caps[2000]["status"] != "prepared":
            raise ValueError("Reference cap cannot prepare; campaign blocked")
        queue = ([Case(**row) for row in state.get("pending_cases", [])] if "pending_cases" in state
                 else initial_cases(hydraulic=True))
        extent_added = state.get("extent_added", False)
        confirming = state.get("confirming", False)
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
            memory_limit = resource_check(root, minimum_free_gb, solver=True)
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
                log_progress("CASE_START", case=asdict(case), launch=state["launches"],
                             memory_limit_mb=memory_limit)
                try:
                    status = native.solve(directory / "native", case_time_limit, memory_limit_mb=memory_limit)
                except subprocess.TimeoutExpired as error:
                    status = {"status": "stopped", "returncode": 124,
                              "message": f"Explicit case timeout: {error}"}
                result = summarize_case(directory / "native", case, status)
                result["elapsed_s"] = time.monotonic() - started
                result["memory_limit_mb"] = memory_limit
                inputs_path = directory / "native" / "case_inputs.json"
                actual_inputs = json.loads(inputs_path.read_text())
                actual_inputs["executed_case_sha256"] = native.file_hash(directory / "native" / native.CASE_NAME)
                write_json(inputs_path, actual_inputs)
                if not case.bonded and (directory / "native" / "run" / native.STEPS_NAME).is_file():
                    report = native.postprocess(directory / "native", candidate, status)
                    write_json(directory / "native" / "aster_cohesive_report.json", report)
                    result["joint_steps"] = report.get("steps", [])
                if case.name in ("A1-reproduction", "A2-intact-topology") and result["status"].startswith("converged_"):
                    expected = reference["hydraulic_intact_maximum_m"]
                    discrepancy = abs(result["maximum_wall_displacement_m"] / expected - 1.)
                    result["reproduction_relative_difference"] = discrepancy
                    if discrepancy > .01:
                        result["status"] = "invalid_evidence"
                        result["error"] = "Reference control differs by more than 1%; investigate before sweeping"
                print(f"Finished {case.name}: {result['status']}", flush=True)
            log_progress("CASE_FINISH", case=case.name, status=result["status"],
                         elapsed_s=result.get("elapsed_s"),
                         maximum_wall_displacement_m=result.get("maximum_wall_displacement_m"),
                         crossing_bracket=result.get("crossing_bracket"), error=result.get("error"))
            write_json(directory / "result.json", result)
            state["results"].append(result)
            done.add(case.name)
            state.pop("active_case", None)
            write_json(state_path, state)
            render_report(root, state)
            if result["status"] == "converged_at_or_above_100mm" and not confirming:
                queue = confirmations(case, state["results"])
                confirming = True
            if case.name.startswith("A") and not result["status"].startswith("converged_"):
                state["status"] = "control_gate_failed"
                break
            if not queue and not extent_added and not confirming:
                candidates = [r for r in state["results"] if r["status"] == "converged_below_100mm"
                              and r["case"]["name"].startswith(("B", "C"))]
                if not candidates:
                    break
                best = sorted(candidates, key=lambda r: (
                    max((s.get("separated_faces", 0) for s in r.get("joint_steps", [])), default=0),
                    r["maximum_wall_displacement_m"]), reverse=True)[:1]
                aggressive = next((r for r in candidates if r["case"]["ft_mpa"] == .02 and r["case"]["gf"] == 1.), None)
                if aggressive is not None and aggressive not in best:
                    best.append(aggressive)
                for cap, preparation_result in caps.items():
                    if preparation_result["status"] == "prepared" and cap != 2000:
                        queue.extend(replace(Case(**r["case"]), name=f"D-{cap}-{i}", cap=cap)
                                     for i, r in enumerate(best))
                largest = max(cap for cap, r in caps.items() if r["status"] == "prepared")
                queue.extend(Case(f"E-{e:g}", cap=largest, e_gpa=e) for e in (10., 5., 1., .1))
                chosen = Case(**best[0]["case"])
                queue.extend(replace(chosen, name=f"E-combined-{e:g}", cap=largest, e_gpa=e)
                             for e in (1., .1))
                extent_added = True
            state.update(pending_cases=[asdict(c) for c in queue], extent_added=extent_added, confirming=confirming)
            write_json(state_path, state)
        else:
            state["status"] = "completed"
        if state["status"] == "running":
            state["status"] = "completed_with_unresolved_cases"
        return state
    except KeyboardInterrupt:
        state["status"] = "cancelled"
        log_progress("CANCELLED", output=str(root), active_case=state.get("active_case"))
        raise
    except (OSError, ValueError, native.CorroborationError) as error:
        state.update(status="blocked", error=f"{type(error).__name__}: {error}")
        log_progress("BLOCKED", output=str(root), error=state["error"])
        raise
    finally:
        finished.set()
        write_json(state_path, state)
        render_report(root, state)
        lock.unlink(missing_ok=True)
        log_progress("CAMPAIGN_FINISH", output=str(root), status=state["status"], launches=state["launches"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-output", action="store_true",
                        help="Create a timestamped campaign below --output; never reuse old load-scope results")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true", help="Explicit full native campaign authorisation")
    parser.add_argument("--case-time-limit", type=int, default=None,
                        help="Optional outer seconds; omitted means no outer wall-clock cap (native CPU guard remains)")
    parser.add_argument("--minimum-free-gb", type=float, default=10.)
    parser.add_argument("--accept-native-cpu-guard", action="store_true",
                        help="Acknowledge installed native CPU safeguard despite no outer wall-clock timeout")
    parser.add_argument("--constitutive-gate", type=Path,
                        help="Hash-authenticated native separation/unloading gate.json required for execution")
    args = parser.parse_args()
    if args.execute and os.name == "nt":
        parser.error("Run the full campaign in Ubuntu/WSL with SciPy and the activated Aster environment")
    if args.execute and args.case_time_limit is None and not args.accept_native_cpu_guard:
        parser.error("Uncapped wall-clock execution requires --accept-native-cpu-guard; the native CPU guard is not removed")
    if args.case_time_limit is not None and args.case_time_limit <= 0:
        parser.error("--case-time-limit must be positive")
    if args.minimum_free_gb < 1:
        parser.error("--minimum-free-gb must be at least 1")
    if os.name != "nt":
        from Aster.whole_model_crack import wsl_path
        args.job = Path(wsl_path(args.job))
        args.output = Path(wsl_path(args.output))
        if args.constitutive_gate is not None:
            args.constitutive_gate = Path(wsl_path(args.constitutive_gate))
    if args.new_output:
        args.output = args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_campaign(args.job, args.output, preflight_only=args.preflight_only, execute=args.execute,
                 case_time_limit=args.case_time_limit, minimum_free_gb=args.minimum_free_gb,
                 constitutive_gate=args.constitutive_gate)


if __name__ == "__main__":
    main()
