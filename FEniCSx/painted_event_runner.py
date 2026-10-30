"""Isolated experimental saved-selection runs, not a production app backend."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .damage_selection import read_saved_selection
from .dry_fracture_growth import (
    GrowthOptions, _record_bound_growth, report_phase, report_progress, solve_source_dry_growth,
)
from .hydrostatic_fracture_growth import imported_reservoir_boundaries, solve_source_hydrostatic_growth
from .fracture_faces import PairedCrackFaces, write_surface_state
from .dam_mesh import import_structure, wall_candidate_triangles
from .paired_contact import ContactOptions
from .run import environment, write_json
from .tensile_loss_event import EventSolverOptions, verified_source_selection


def solver_options(value: dict, water_mode: str) -> dict:
    """Require explicit numerical/resource settings, never physical fallbacks."""
    if water_mode not in ("dry", "common_head_hydrostatic"):
        raise ValueError("water_mode must be dry or common_head_hydrostatic")
    keys = {"maximum_source_cells", "maximum_volume_change", "event", "contact", "growth"}
    if water_mode == "common_head_hydrostatic":
        keys.add("hydraulics")
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"Solver options must contain exactly {sorted(keys)}")
    ceiling = value["maximum_source_cells"]
    if type(ceiling) is not int or ceiling < 1:
        raise ValueError("maximum_source_cells must be a positive integer")
    volume = value["maximum_volume_change"]
    if (isinstance(volume, bool) or not isinstance(volume, (int, float))
            or not 0 <= volume < 1):
        raise ValueError("maximum_volume_change must be finite and in [0, 1)")
    result = {"maximum_volume_change": volume}
    for key, cls in (("event", EventSolverOptions), ("contact", ContactOptions), ("growth", GrowthOptions)):
        if not isinstance(value[key], dict):
            raise ValueError(f"{key} options must be an object")
        try:
            result[f"{key}_options"] = cls(**value[key])
        except TypeError as error:
            raise ValueError(f"Invalid {key} option fields: {error}") from error
    if water_mode == "common_head_hydrostatic":
        hydraulic = value["hydraulics"]
        if (not isinstance(hydraulic, dict)
                or set(hydraulic) != {"closure_tolerance_m", "flow_relative_tolerance"}):
            raise ValueError("hydraulics requires closure_tolerance_m and flow_relative_tolerance")
        for key, number in hydraulic.items():
            if (isinstance(number, bool) or not isinstance(number, (int, float))
                    or not 0 < number < float("inf")
                    or key == "flow_relative_tolerance" and number >= 1):
                raise ValueError(f"{key} must be finite and positive; relative tolerance must be below one")
        result.update(hydraulic)
    return result


def source_preflight(structure, marker_source_edges, maximum_volume_change, water_mode, *, output=None):
    """Audit the original mesh/candidate inlets, without split DOFs or solves."""
    started = time.monotonic()
    imported = import_structure(structure, maximum_volume_change)
    candidates = wall_candidate_triangles(imported, marker_source_edges)
    report = {
        "status": "passed_unsplit_candidate_preflight_not_solved",
        "original_nodes": len(imported.points), "tetrahedra": len(imported.tetrahedra),
        "candidate_faces": len(candidates), "prepared_displacement_dofs_upper_bound": (
            3 * (len(imported.points) + 4 * int(np.count_nonzero(imported.regions == 1)))),
        "intact_condensed_displacement_dofs": 3 * len(imported.points),
        "candidate_mesh_split": False, "numerical_growth_solved": False,
        "resource_feasibility_verified": False, "engineering_validated": False,
    }
    def persist():
        report["elapsed_s"] = time.monotonic() - started
        if output is not None:
            write_json(Path(output), report)

    persist()
    try:
        if water_mode == "common_head_hydrostatic":
            boundaries = imported_reservoir_boundaries(
                imported, structure, np.arange(len(imported.points)), candidates)
            report["reservoir_boundaries"] = [
                {"name": boundary.name, "head_m": boundary.elevation_m,
                 "inlet_edges": len(boundary.inlet_source_edges)} for boundary in boundaries]
    except ValueError as error:
        report["status"] = "failed_unsplit_candidate_preflight_not_solved"
        report["error"] = str(error)
        persist()
        raise
    persist()
    return report


def run_saved_event(
    configuration: Path, model_directory: Path, baseline_directory: Path,
    fields: Path, saved_selection: Path, job_id: str, *, water_mode: str,
    options: dict, output: Path, prepare_only: bool = False,
) -> dict:
    """Revalidate a saved app selection and run a bounded isolated experiment.

    The resource ceiling is an explicit caller choice, not a time limit on
    user jobs. All physical restrictions of the source growth drivers apply.
    """
    prepared_options = solver_options(options, water_mode)
    started = time.monotonic()
    if type(prepare_only) is not bool:
        raise ValueError("prepare_only must be an explicit boolean")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("An explicit nonempty source job_id is required")
    configuration, saved_selection = Path(configuration), Path(saved_selection)
    configuration_bytes, selection_bytes = configuration.read_bytes(), saved_selection.read_bytes()
    config = json.loads(configuration_bytes)
    config_digest = hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running_experimental_painted_event", "solver_ready": False,
        "engineering_validated": False, "water_mode": water_mode,
        "prepare_only": prepare_only,
        "source_job_id": job_id, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_sha256": hashlib.sha256(configuration_bytes).hexdigest(),
        "saved_selection_file_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "source_paths": {key: str(Path(path).resolve()) for key, path in (
            ("model", model_directory), ("baseline", baseline_directory), ("fields", fields))},
        "limitations": (
            "Experimental mesh-constrained frictionless conservative patch search; "
            "no unrestricted/strength-consistent nucleation, different-head growth, "
            "trapped fluid, filling history, verified displacement uniqueness/stability or engineering safety verdict."),
    }

    def persist():
        manifest["elapsed_s"] = time.monotonic() - started
        write_json(output / "manifest.json", manifest)

    def phase(text):
        manifest["phase"] = text
        persist()
        report_progress(text)

    persist()
    try:
        report_phase("preparing")
        phase("snapshotting and checking the saved selection")
        inputs = output / "inputs"
        inputs.mkdir()
        snapshot_config = inputs / "config.json"
        snapshot_config.write_bytes(configuration_bytes)
        snapshot_selection = inputs / saved_selection.name
        if snapshot_selection == snapshot_config or snapshot_selection.name == "solver_options.json":
            raise ValueError("Saved selection filename conflicts with reserved input snapshots")
        snapshot_selection.write_bytes(selection_bytes)
        write_json(inputs / "solver_options.json", options)
        saved = read_saved_selection(snapshot_selection, job_id, config_digest)
        manifest["selection_id"] = saved["selection_id"]
        manifest["environment"] = environment()
        persist()
        phase("binding the selection to the native mesh and baseline fields")
        structure, binding = verified_source_selection(
            snapshot_config, Path(model_directory), Path(baseline_directory), Path(fields), saved["request"])
        manifest["verified_selection"] = binding
        manifest["source_cells"] = len(structure.cells)
        manifest["maximum_source_cells"] = options["maximum_source_cells"]
        persist()
        if len(structure.cells) > options["maximum_source_cells"]:
            raise ValueError(
                f"Source has {len(structure.cells)} cells, exceeding explicit experimental "
                f"maximum_source_cells={options['maximum_source_cells']}; no candidate splitting or solve started")
        manifest["growth_options"] = asdict(prepared_options["growth_options"])
        manifest["configured_fracture_energy_j_m2"] = config.get("fracture_debug", {}).get(
            "critical_energy_release_rate_j_m2")
        persist()
        manifest["preflight_report"] = "preflight.json"
        persist()
        phase("resource preflight")
        manifest["preflight"] = source_preflight(
            structure, binding["marker_source_edges"], prepared_options["maximum_volume_change"],
            water_mode, output=output / "preflight.json")
        persist()
        if prepare_only:
            manifest["status"] = "prepared_experimental_not_solved"
            manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            persist()
            return manifest
        driver = solve_source_dry_growth if water_mode == "dry" else solve_source_hydrostatic_growth
        concrete = config["material"]["concrete"]
        selected_strength = concrete.get("selected_tensile_strength_pa", concrete["tensile_strength_pa"])
        if (isinstance(selected_strength, bool) or not isinstance(selected_strength, (int, float))
                or not np.isfinite(selected_strength) or selected_strength < 0):
            raise ValueError("Selected tensile strength must be finite and nonnegative")
        manifest["selected_tensile_strength_pa"] = selected_strength
        manifest["selected_material_law"] = (
            "reversible tensile cap; elastic below strength, no irreversible strength-calibrated nucleation")
        phase(f"{water_mode} fracture growth solve")
        result = driver(
            structure, binding["affected_element_ids"], binding["marker_source_edges"],
            **prepared_options, output=output / "growth",
            opening_limit_m=binding["opening_limit_m"], selection_provenance={
                **binding, "saved_selection_id": saved["selection_id"],
                "saved_selection_sha256": saved["selection_sha256"], "source_job_id": job_id},
            selected_tensile_strength_pa=selected_strength)
        _record_bound_growth(result, output / "growth")
        phase("writing the accepted crack surface")
        accepted = output / "growth" / result["accepted_event_directory"]
        artifacts = {
            "growth_report": output / "growth" / "growth_report.json",
            "event_report": accepted / "event_report.json",
            "displacement": accepted / "displacement.pvd",
            "material": accepted / "post_event_material.pvd",
            "post_event_state": accepted / "post_event_state.npz",
        }
        for name, path in artifacts.items():
            if not path.resolve().is_relative_to(output.resolve()) or not path.is_file():
                raise RuntimeError(f"Accepted result is missing or outside isolated output: {name}")
        with (np.load(accepted / "native_inputs.npz", allow_pickle=False) as native,
              np.load(accepted / "post_event_state.npz", allow_pickle=False) as state,
              np.load(output / "growth" / "source_mapping.npz", allow_pickle=False) as source):
            faces = PairedCrackFaces(
                native["coordinates_m"], native["minus"], native["plus"], native["normals"])
            pressure = native["pressure_pa"].copy()
            wet = np.zeros(len(faces.minus), dtype=bool)
            if (accepted / "hydraulic_state.npz").is_file():
                with (np.load(accepted / "hydraulic_state.npz", allow_pickle=False) as water,
                      np.load(accepted / "hydraulic_inputs.npz", allow_pickle=False) as hydraulic):
                    ids = water["mechanical_inventory_face_ids"]
                    if not np.array_equal(ids, hydraulic["mechanical_inventory_face_ids"]):
                        raise RuntimeError("Accepted hydraulic input/state face identities differ")
                    pressure[ids] = water["pressure_pa"]
                    wet[ids] = np.any(water["pressure_pa"] > 0., axis=1)
            surface = output / "crack_surface.vtu"
            write_surface_state(
                surface, faces, state["displacement_m"],
                released_face_mask=native["released_face_mask"],
                wet_face_mask=wet, vertex_pressures_pa=pressure,
                parent_node_ids=source["parent_node_ids"])
        artifacts["crack_surface"] = surface
        manifest["accepted_results"] = {name: str(path.relative_to(output)) for name, path in artifacts.items()}
        manifest["opening_assessment"] = result["opening_assessment"]
        manifest["accepted_surface_ids"] = result["accepted_surface_ids"]
        manifest["status"] = "completed_experimental_not_engineering_validated"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        persist()
        return manifest
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        manifest.pop("accepted_results", None)
        manifest["status"] = "failed_experimental_painted_event"
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = str(error)
        persist()
        raise RuntimeError(f"Experimental painted event failed; inspect {output / 'manifest.json'}: {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimental", action="store_true", required=True,
                        help="Acknowledge this is not production or engineering-validated")
    for name in ("configuration", "model-directory", "baseline-directory", "fields",
                 "saved-selection", "options", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--water-mode", choices=("dry", "common_head_hydrostatic"), required=True)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Verify source and unsplit candidates/inlets; do not split or solve")
    args = vars(parser.parse_args(argv))
    args.pop("experimental")
    args["options"] = json.loads(args["options"].read_text(encoding="utf-8"))
    result = run_saved_event(**args)
    print(json.dumps({"status": result["status"], "output": str(args["output"]),
                      "accepted_results": result.get("accepted_results")}, indent=2))


if __name__ == "__main__":
    main()
