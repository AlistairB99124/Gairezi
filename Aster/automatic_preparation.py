"""Select and prepare wall facets from accepted Code_Aster elastic displacement.

No supporting solver is invoked and no nonlinear result is produced here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from Aster.automatic_cohesive import file_hash
from Elmer.model_config import residual_target_tensile_stress_pa
from FEniCSx.automatic_cracking import candidate_region, region_cell_stresses, region_candidate_faces
from FEniCSx.automatic_cracking import maximum_overload_factor
from FEniCSx.automatic_screening import (COMPLETED, config_digest, crack_base, crack_thickness,
                                         lift_joints, max_damage_count, screen_cells, selection_enabled)
from FEniCSx.fracture_topology import split_interior_wall_faces
from FEniCSx.lift_joints import lift_joint_settings

PREPARED = "prepared_aster_candidates"
BASELINE_FILES = ("mesh_manifest.json", "run_manifest.json", "three_body_mesh_arrays.npz",
                  "aster_full_displacement.npz")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def baseline_inventory(baseline: Path, configuration: Path, model: Path):
    config = json.loads(configuration.read_text(encoding="utf-8"))
    manifest = json.loads((baseline / "mesh_manifest.json").read_text(encoding="utf-8"))
    run = json.loads((baseline / "run_manifest.json").read_text(encoding="utf-8"))
    if run.get("status") != "completed":
        raise ValueError("Authoritative Aster baseline is not completed")
    source_mesh = Path(manifest["source_mesh"])
    if os.name != "nt":
        from Aster.whole_model_crack import wsl_path
        source_mesh = Path(wsl_path(source_mesh))
    if source_mesh.resolve() != (model / "three_body.msh").resolve():
        raise ValueError("Aster baseline belongs to another source mesh")
    for name, digest in manifest["source_hashes"].items():
        if file_hash(model / name) != digest:
            raise ValueError(f"Aster baseline source changed: {name}")
    if config_digest(config) != config_digest(json.loads((model / "config.json").read_text(encoding="utf-8"))):
        raise ValueError("Aster baseline configuration differs from the submitted configuration")
    with np.load(baseline / "three_body_mesh_arrays.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    with np.load(baseline / "aster_full_displacement.npz", allow_pickle=False) as fields:
        if (not np.array_equal(fields["source_node_ids"], arrays["source_node_ids"])
                or not np.allclose(fields["coordinates_m"], arrays["nodes"], rtol=0., atol=1e-9)):
            raise ValueError("Full Aster displacement node identities differ from the baseline mesh")
        displacement = fields["displacement_m"].copy()
    if displacement.shape != arrays["nodes"].shape or not np.isfinite(displacement).all():
        raise ValueError("Full Aster baseline displacement must be finite at every mesh node")
    cells = np.vstack((arrays["tetra4_concrete"], arrays["tetra4_bedrock"]))
    regions = np.concatenate((arrays["concrete_body_ids"], np.full(len(arrays["tetra4_bedrock"]), 2)))
    rows = arrays["source_cell_rows"]
    if rows.shape != (len(cells),) or not np.array_equal(np.sort(rows), np.arange(len(cells))):
        raise ValueError("Aster baseline source-cell identities are incomplete or duplicated")
    physical = np.unique(cells)
    inverse = np.full(len(arrays["nodes"]), -1, dtype=np.int64)
    inverse[physical] = np.arange(len(physical))
    hashes = {name: file_hash(baseline / name) for name in BASELINE_FILES}
    return config, arrays["nodes"][physical], inverse[cells], regions, displacement[physical], physical, rows, hashes


def run(configuration: Path, model: Path, baseline: Path, output: Path, *,
        job_id: str, relative_tolerance: float, screening: Path | None = None,
        candidate_scope: str = "configured", continuation: bool = False) -> dict:
    if not job_id.strip():
        raise ValueError("An explicit source job ID is required")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running_aster_preparation", "source_job_id": job_id,
                "config_digest": config_digest(json.loads(configuration.read_text(encoding="utf-8"))),
                "solver": "code_aster", "engineering_validated": False, "phase": "preparation"}
    write_json(output / "manifest.json", manifest)
    try:
        config, coordinates, cells, regions, displacement, physical, rows, hashes = baseline_inventory(
            baseline, configuration, model)
        concrete = config["material"]["concrete"]
        target = residual_target_tensile_stress_pa(config)
        if screening is None:
            stress = region_cell_stresses(coordinates, cells, regions, displacement, config["material"])
            result = screen_cells(stress, regions, rows, coordinates, cells,
                                  float(concrete["tensile_strength_pa"]),
                                  relative_tolerance=relative_tolerance,
                                  max_damage_count=(max_damage_count(config)
                                                    if candidate_scope == "configured" else None),
                                  crack_base=(crack_base(config) if candidate_scope == "configured" else False),
                                  selection_enabled=selection_enabled(config),
                                  crack_thickness=(crack_thickness(config)
                                                   if candidate_scope == "configured" else None),
                                  lift_joints=(lift_joints(config)
                                               if candidate_scope == "configured" else None))
            result.update(source_job_id=job_id, config_digest=manifest["config_digest"],
                          solver="code_aster", stress_association="raw P1 cell stress from accepted Aster displacement",
                          verified_baseline_source_sha256=hashes,
                          cohesive_fracture_energy_j_m2=concrete.get("cohesive_fracture_energy_j_m2"),
                          cohesive_fracture_energy_provenance="user-supplied; not independently verified",
                          configuration_sha256=file_hash(configuration))
            wall = regions == 1
            wall_principal = np.linalg.eigvalsh(stress[wall])[:, -1]
            residual_rows = rows[wall][wall_principal > target]
            wall_nodes = np.unique(cells[wall])
            result["residual_tension_assessment"] = {
                "threshold_pa": target,
                "maximum_principal_tensile_stress_pa": float(wall_principal.max()),
                "peak_centroid_xyz_m": result["peak_location_m"],
                "wall_cells_above_threshold": int(np.count_nonzero(wall_principal > target)),
                "source_rows_above_threshold": residual_rows.tolist(),
                "maximum_wall_displacement_m": float(np.linalg.norm(displacement[wall_nodes], axis=1).max()),
                "scope": "all wall P1 cells; accepted elastic baseline at design load",
            }
            if candidate_scope == "all_overstressed" and "residual_target_tensile_stress_pa" in config:
                selected_rows = residual_rows if selection_enabled(config) else np.empty(0, dtype=np.int64)
                result["selection"] = {
                    "rule": "wall-cell maximum principal tensile stress strictly exceeds the residual target",
                    "enabled": selection_enabled(config),
                    "selected_source_rows": selected_rows.tolist(),
                    "selected_cells": len(selected_rows),
                    "threshold_pa": target,
                }
            if lift_joints(config) is not None and result["selection"].get("lift_joints") is not None:
                # Fail here, before any solve is spent, if the lift-joint law cannot be given a
                # joint at least as stiff as the elements beside it (same measure as native.prepare).
                from Aster.automatic_cohesive import JOINT_PENALTY_FACTOR, CorroborationError, lift_joint_parameters
                recorded = result["selection"]["lift_joints"]
                neighbour = float(concrete["youngs_modulus"]) / float(recorded["mean_facet_edge_m"])
                try:
                    lift_joint_parameters(recorded, JOINT_PENALTY_FACTOR * neighbour, neighbour)
                except CorroborationError as error:
                    raise ValueError(str(error)) from error
            properties = ((config.get("damage_selection") or {}).get("damaged_element_properties")
                          if candidate_scope == "configured" else None)
            joints = lift_joints(config) if candidate_scope == "configured" else None
            if properties is not None or joints is not None:
                assignment = {"selected_source_rows": result["selection"]["selected_source_rows"]}
                scope = []
                if properties is not None:
                    assignment["properties"] = properties
                    scope.append("degraded material on the exact selected wall/foundation volume cells "
                                 "(baseline and unselected cells unchanged)")
                if joints is not None:
                    assignment["joint_properties"] = result["selection"]["lift_joints"]
                    scope.append("weak construction-joint law on the selected band's near-horizontal facets")
                if properties is None:
                    scope.append("every element keeps the intact concrete")
                assignment["scope"] = "; ".join(scope)
                result["material_assignment"] = assignment
            write_json(output / "screening.json", result)
            manifest.update(status=COMPLETED, screening_sha256=file_hash(output / "screening.json"))
        else:
            accepted = json.loads((screening / "manifest.json").read_text(encoding="utf-8"))
            result = json.loads((screening / "screening.json").read_text(encoding="utf-8"))
            if (accepted.get("status") != COMPLETED or accepted.get("source_job_id") != job_id
                    or accepted.get("screening_sha256") != file_hash(screening / "screening.json")
                    or result.get("config_digest") != manifest["config_digest"]
                    or result.get("solver") != "code_aster"
                    or result.get("verified_baseline_source_sha256") != hashes):
                raise ValueError("Candidate preparation requires this exact current Aster screening")
            if candidate_scope not in ("all_overstressed", "configured"):
                raise ValueError(f"Unsupported Aster candidate scope: {candidate_scope}")
            disabled = result["selection"].get("enabled") is False
            if disabled:
                selected_rows = np.empty(0, dtype=np.int64)
            elif candidate_scope == "all_overstressed":
                selected_rows = np.asarray(
                    result["residual_tension_assessment"]["source_rows_above_threshold"]
                    if "residual_target_tensile_stress_pa" in config else result["overstressed_source_rows"],
                    dtype=np.int64)
            elif candidate_scope == "configured":
                selected_rows = np.asarray(result["selection"]["selected_source_rows"], dtype=np.int64)
            overload_factor = maximum_overload_factor(config)
            # A disabled selection is an intentional control run: no candidates regardless
            # of facet stress, so neither overload screening nor empty-selection errors apply.
            if (candidate_scope == "configured" and not len(selected_rows) and not disabled
                    and result["facet_check"]["critical_facets"] == 0 and overload_factor > 1):
                stress = region_cell_stresses(coordinates, cells, regions, displacement, config["material"])
                overload = screen_cells(stress * overload_factor, regions, rows, coordinates, cells,
                                        float(concrete["tensile_strength_pa"]), relative_tolerance=relative_tolerance,
                                        max_damage_count=max_damage_count(config), crack_base=crack_base(config),
                                        crack_thickness=crack_thickness(config))
                selected_rows = np.asarray(overload["selection"]["selected_source_rows"], dtype=np.int64)
                if not len(selected_rows) and overload["facet_check"]["critical_facets"] > 0:
                    raise ValueError(overload["selection"].get("message") or "Overload candidate selection is empty")
            if (candidate_scope == "configured" and not len(selected_rows) and not disabled
                    and result["facet_check"]["critical_facets"] > 0):
                raise ValueError(result["selection"].get("message") or "Aster candidate selection is empty")
            if np.any(selected_rows < 0) or np.any(selected_rows >= len(cells)):
                raise ValueError("Aster screening selected no valid wall cells")
            inverse_rows = np.argsort(rows)
            selected = inverse_rows[selected_rows]
            eligible_regions = (1,) if candidate_scope == "all_overstressed" else (1, 3)
            if not np.all(np.isin(regions[selected], eligible_regions)):
                raise ValueError("Aster screening selected cells outside the wall and foundation")
            # The uncapped selection is a complete bank-to-bank band by construction;
            # capped patches are openable by construction. No neighbour rings either way.
            rings = 0
            membership = candidate_region(cells, regions, selected, rings)
            faces = region_candidate_faces(cells, regions, membership)
            initial_faces = faces
            if continuation:
                if candidate_scope != "all_overstressed":
                    raise ValueError("Fixed-mesh continuation requires wall-only production selection")
                faces = region_candidate_faces(cells, regions, regions == 1)
            while len(faces) > 20000 and rings > 0:
                rings -= 1
                membership = candidate_region(cells, regions, selected, rings)
                faces = region_candidate_faces(cells, regions, membership)
            maximum_faces = int(((config.get("fracture_debug") or {}).get("aster") or {}).get(
                "maximum_candidate_facets", 20000))
            if len(faces) > maximum_faces:
                raise ValueError(
                    f"Cohesive candidate resource limit: {len(faces)} wall facets would generate "
                    f"{3 * len(faces)} joint patches, exceeding the configured limit of "
                    f"{maximum_faces} facets. No candidates were truncated; review the material "
                    "strength and mesh or increase the limit only with sufficient resources."
                )
            if not len(faces) and len(selected):
                raise ValueError("Selected wall cells have no interior candidate facets")
            split = (split_interior_wall_faces(coordinates, cells, regions, faces, drop_fully_tied=True)
                     if len(faces) else None)
            continuation_arrays = {}
            if continuation and split is not None:
                initial_keys = {tuple(sorted(face)) for face in initial_faces.tolist()}
                continuation_arrays = {
                    "continuation_active_faces": np.asarray(
                        [tuple(sorted(face)) in initial_keys for face in split.source_faces.tolist()], dtype=bool),
                    "continuation_face_owners": split.adjacent_cells,
                }
            energy = concrete.get("cohesive_fracture_energy_j_m2")
            properties = ((config.get("damage_selection") or {}).get("damaged_element_properties")
                          if candidate_scope == "configured" else None)
            joints = lift_joints(config) if candidate_scope == "configured" else None
            if properties is not None:
                energy = properties["cohesive_fracture_energy_j_m2"]
            if (isinstance(energy, bool) or not isinstance(energy, (int, float))
                    or not np.isfinite(energy) or energy <= 0):
                raise ValueError("A positive supplied cohesive fracture energy is required")
            lift_faces = None
            if joints is not None:
                # The screening recorded the lift planes as sorted source-node triples; the
                # candidate faces are the same sorted triples, so membership is exact.
                recorded = result["selection"].get("lift_joints")
                if not isinstance(recorded, dict) or not recorded.get("source_faces"):
                    raise ValueError("The Aster screening has no lift-joint facets although lift_joints are configured")
                lift_settings = lift_joint_settings(joints)
                planes = {tuple(face) for face in recorded["source_faces"]}
                source_faces = (np.sort(np.asarray(split.source_faces, dtype=np.int64), axis=1)
                                if split is not None else np.empty((0, 3), dtype=np.int64))
                lift_faces = np.array([tuple(face) in planes for face in source_faces.tolist()], dtype=bool)
                if int(lift_faces.sum()) != len(planes):
                    raise ValueError(
                        f"{int(lift_faces.sum())} candidate facets match the screening's {len(planes)} lift planes; "
                        "the prepared candidates must contain every lift-joint facet")
            np.savez_compressed(output / "cohesive_state.npz",
                                coordinates_m=split.coordinates_m if split is not None else coordinates,
                                parent_node_ids=physical[split.parent_node_ids] if split is not None else physical,
                                minus=split.faces.minus if split is not None else np.empty((0, 3), dtype=np.int64),
                                plus=split.faces.plus if split is not None else np.empty((0, 3), dtype=np.int64),
                                areas_m2=split.faces.areas if split is not None else np.empty(0),
                                **continuation_arrays,
                                **({"lift_joint_faces": lift_faces} if lift_faces is not None else {}))
            result = {"status": PREPARED, "outcome": "candidates_prepared", "accepted": False,
                      "solver": "code_aster", "source_job_id": job_id,
                      "config_digest": manifest["config_digest"], "engineering_validated": False,
                      "screening_sha256": file_hash(screening / "screening.json"),
                      "peak_location_m": result.get("peak_location_m"),
                      "baseline_sha256": hashes, "load_factor": 1.,
                      "maximum_overload_factor": overload_factor,
                      "tensile_strength_pa": float(concrete["tensile_strength_pa"]),
                      "fracture_energy_j_m2": float(energy), "candidate_faces": len(split.source_faces) if split is not None else 0,
                      "candidate_basis": (("all wall cells whose maximum principal tensile stress exceeds "
                                           "residual_target_tensile_stress_pa; material strength is unchanged"
                                           if "residual_target_tensile_stress_pa" in config else
                                           "all wall cells exceeding the configured concrete tensile strength")
                                          if candidate_scope == "all_overstressed" else
                                          ("design-load screening, with neighbour rings where uncapped"
                                          if len(result["selection"]["selected_source_rows"]) else
                                          ("damage selection disabled: control preparation with no candidates"
                                           if result["selection"].get("enabled") is False else
                                           "proportional maximum-overload screening; no eligible facets means native elastic steps"))),
                      "outcome_basis": "Preparation only; native Aster nonlinear equilibrium is still required"}
            if "residual_target_tensile_stress_pa" in config:
                result["residual_target_tensile_stress_pa"] = target
            if continuation and split is not None:
                result["continuation"] = {
                    "maximum_rounds": int(((config.get("fracture_debug") or {}).get("aster") or {}).get(
                        "adaptive_rounds", 1)),
                    "target_pa": residual_target_tensile_stress_pa(config),
                    "initial_active_faces": int(continuation_arrays["continuation_active_faces"].sum()),
                }
            result["solver_controls"] = {
                key: value for key, value in ((config.get("fracture_debug") or {}).get("aster") or {}).items()
                if key in {
                    "loading_mode", "water_depth_start_m", "water_depth_increment_m",
                    "load_factor_start", "load_increment", "maximum_iterations", "relative_residual_tolerance",
                    "line_search_iterations", "subdivision_levels", "adaptive_rounds",
                    "whole_wall_tension_tolerance_pa"}
            }
            from Elmer.reservoir_loading import reservoir_loading
            loading = reservoir_loading(config)
            if loading is not None:
                result["reservoir_loading"] = dict(
                    loading.metadata(), water_depth_targets_m=loading.targets(),
                    selection_enabled=selection_enabled(config))
            if properties is not None or joints is not None:
                assignment = {"selected_source_rows": selected_rows.tolist()}
                scope = []
                if properties is not None:
                    assignment["properties"] = properties
                    scope.append("degraded material on the exact selected wall/foundation volume cells")
                    result["tensile_strength_pa"] = properties["tensile_strength_pa"]
                if joints is not None:
                    assignment["joint_properties"] = {
                        **recorded, "source_faces": None,
                        **{key: lift_settings[key] for key in ("alpha", "pena_contact", "law")}}
                    scope.append("weak construction-joint law on the selected band's near-horizontal facets")
                    result["lift_joint_faces"] = int(lift_faces.sum())
                if properties is None:
                    scope.append("every element keeps the intact concrete")
                assignment["scope"] = "; ".join(scope)
                result["material_assignment"] = assignment
            friction = (config.get("damage_selection") or {}).get("damaged_joint_friction")
            result["crack_water_pressure"] = (config.get("damage_selection") or {}).get("crack_water_pressure", False)
            if friction is not None:
                if properties is None:
                    raise ValueError("damaged_joint_friction requires damaged_element_properties")
                result["damaged_joint_friction"] = friction
            degradation = (config.get("damage_selection") or {}).get("support_degradation")
            if degradation is not None and selection_enabled(config):
                # Applied only in the cracked solve (automatic_cohesive.prepare); the
                # control run (enabled false) keeps the intact supports.
                result["support_degradation"] = degradation
            write_json(output / "summary.json", result)
            manifest.update(status=PREPARED, summary_sha256=file_hash(output / "summary.json"),
                            candidate_state_sha256=file_hash(output / "cohesive_state.npz"))
        manifest["phase"] = None
        write_json(output / "manifest.json", manifest)
        return result
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
        manifest.update(status="failed_aster_preparation", error=f"{type(error).__name__}: {error}")
        write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "model", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--screening", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--relative-tolerance", type=float, required=True)
    parser.add_argument("--candidate-scope", choices=("configured", "all_overstressed"),
                        default="configured")
    args = parser.parse_args()
    run(args.config, args.model, args.baseline, args.output, job_id=args.job_id,
        relative_tolerance=args.relative_tolerance, screening=args.screening,
        candidate_scope=args.candidate_scope)


if __name__ == "__main__":
    main()
