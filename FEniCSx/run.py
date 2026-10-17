"""Run validated FEniCSx scenarios from Ubuntu/WSL."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

from configuration import load_scenarios
from engineering_report import write_engineering_results

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOAD_CASE = ROOT / "config.json"


def environment() -> dict:
    import dolfinx
    import gmsh
    import meshio
    import numpy
    import scipy
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This first-stage runner supports one MPI process; do not launch it with mpirun.")
    if dolfinx.__version__.split(".")[:2] != ["0", "11"]:
        raise RuntimeError(f"This adapter is verified for DOLFINx 0.11; found {dolfinx.__version__}")
    return {
        "python": sys.version, "executable": sys.executable, "dolfinx": dolfinx.__version__,
        "petsc": PETSc.Sys.getVersion(), "mpi": MPI.Get_library_version().strip("\0\n"),
        "numpy": numpy.__version__, "ufl": ufl.__version__,
        "meshio": meshio.__version__, "gmsh": gmsh.__version__,
        "scipy": scipy.__version__,
    }


def inventory() -> dict:
    sys.path.insert(0, str(ROOT))
    from structural_model import audit_combined_structure, build_combined_structure, build_named_groups
    from Elmer.structure_assembly import apply_load_case

    structure = build_combined_structure(ROOT)
    apply_load_case(structure, LOAD_CASE)
    audit = audit_combined_structure(structure)
    groups = build_named_groups(structure)
    return {
        "status": "inventory_only_not_imported_or_solved",
        "source": "structural_model.build_combined_structure + config.json",
        "units": {"coordinates": "m", "stress": "Pa", "springs": "N/m^3"},
        "nodes": len(structure.nodes),
        "volume_cells": len(structure.cells),
        "gmsh_volume_element_counts": dict(Counter(kind for kind, _ in structure.cells)),
        "group_counts": {name: len(indices) for name, indices in groups.items()},
        "material": asdict(structure.material),
        "loads": asdict(structure.loads),
        "foundation_support": asdict(structure.foundation_support),
        "plinth_support": asdict(structure.plinth_support),
        "audit": audit,
        "porting_requirements": [
            "Preserve mixed-cell topology or verify a controlled conversion.",
            "Preserve bonded internal wall/plinth faces and shared displacement nodes.",
            "Foundation springs only; plinth support coefficients are inactive in the bonded reference.",
            "Verify displacement probes, reactions and global equilibrium against Elmer.",
            "Verify crack contact and initial-gap geometry on a small benchmark before dam scenarios.",
        ],
    }


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def submitted_structure(config: Path):
    """Rebuild the exact Elmer solver topology without modifying checkout inputs."""
    sys.path.insert(0, str(ROOT))
    from structural_model import audit_combined_structure, build_combined_structure
    from Elmer.structure_assembly import apply_load_case
    from Elmer.regions.tetrahedral import tetrahedralize_structure

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "config.json").write_bytes(config.read_bytes())
        structure = build_combined_structure(root)
    apply_load_case(structure, config)
    audit_combined_structure(structure)
    return tetrahedralize_structure(structure)


def assert_fracture_ready(structure):
    if not any(tag == 8 for tag, _ in structure.boundaries):
        raise RuntimeError(
            "Cracking is on hold pending review of the corrected bonded intact wall/plinth reference. "
            "Run the elastic baseline first; joint opening/sliding and fracture require separate verification."
        )


def run_baseline(dam_config: Path, output_root: Path, versions: dict, *, phase: bool = False) -> None:
    """Solve the submitted uncracked dam, with or without evolving AT2 damage."""
    from automatic_crack import DEFAULTS
    from crack_field import PhaseFieldParameters, PhaseFieldSolverOptions
    from dam_mesh import import_structure

    scenario = "Baseline_Phase" if phase else "Baseline_NoPhase"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_root / ("dam_phase_field" if phase else "dam_elastic_baseline") / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "config.json"
    snapshot.write_bytes(dam_config.read_bytes())
    manifest = {
        "status": "running", "scenario_name": scenario,
        "crack_scenario_names": [scenario], "environment": versions,
        "configuration_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "started_at_utc": timestamp, "engineering_validated": False,
    }
    write_json(directory / "manifest.json", manifest)
    try:
        structure = submitted_structure(snapshot)
        if phase:
            assert_fracture_ready(structure)
        imported = import_structure(structure, maximum_volume_change=0.001)
        write_json(directory / "import_report.json", imported.report)
        defaults = json.loads(DEFAULTS.read_text(encoding="utf-8"))
        options = PhaseFieldSolverOptions(**defaults.pop("solver"))
        if phase:
            from dam_phase_field import solve
            from dolfinx.cpp.mesh import h

            parameters = PhaseFieldParameters(**defaults)
            cells = imported.cell_tags.find(2)
            maximum_diameter = float(h(imported.domain._cpp_object, 3, cells).max())
            resolution = parameters.length_scale_m / maximum_diameter
            if resolution < options.minimum_elements_per_length_scale:
                print(
                    f"WARNING: {scenario} has only {resolution:.3g} elements per length scale "
                    f"(recommended {options.minimum_elements_per_length_scale:g}); "
                    "damage results are not mesh-resolved.",
                    flush=True,
                )
            result = solve(imported, structure, [], options, directory, parameters=parameters)
            report = result.report
            report["phase_field_parameters"] = asdict(parameters)
            report["mesh_resolution"] = {
                "minimum_elements_per_length_scale": resolution,
                "recommended_minimum": options.minimum_elements_per_length_scale,
                "meets_recommendation": resolution >= options.minimum_elements_per_length_scale,
                "refinement": "none; baseline uses the same tetrahedral mesh as Elmer",
            }
            report_name = "phase_field_report.json"
        else:
            from dam_elasticity import solve

            _, _, report = solve(imported, structure, {
                "relative_tolerance": options.mechanics_relative_tolerance,
                "maximum_iterations": options.mechanics_maximum_iterations,
            }, directory)
            report.update({
                "model": "linear isotropic elasticity; phase field disabled",
                "configured_tensile_strength_pa": structure.material.tensile_strength_pa,
                "crack_surface_measure_m2": 0.0,
            })
            report_name = "baseline_report.json"
        report.update({
            "scenario_name": scenario, "engineering_validated": False,
            "configuration_sha256": manifest["configuration_sha256"],
            "initial_cracks": [], "internal_crack_pressure_pa": 0.0,
        })
        if report.get("wall_plinth_connection") == "bonded_intact_provisional":
            report["fracture_hold_reason"] = (
                "Cracking is on hold pending review of the corrected bonded intact wall/plinth reference."
            )
        write_json(directory / report_name, report)
        (directory / "results.md").write_text(
            f"# {scenario}\n\n"
            "Uncracked dam with submitted material, gravity, water loads and spring supports.\n\n"
            + ("Damage starts at zero and may evolve under load; no crack is seeded.\n\n"
               "The baseline mesh is not adaptively refined; inspect mesh_resolution in the report.\n\n"
               if phase else "Phase-field damage is disabled.\n\n")
            + "This is a numerical baseline, not engineering validation or a dam-safety assessment.\n\n"
            + (
                "Wall/plinth connection: provisional fully bonded shared nodes. "
                "Foundation springs only; geometry.plinth.support is inactive. "
                "No joint opening/sliding. Cracking is on hold pending model review.\n\n"
                if report.get("wall_plinth_connection") == "bonded_intact_provisional" else ""
            )
            + "Principal-stress peaks use consistent L2 nodal recovery followed by "
            "element tensor averaging before eigenvalues, matching the Elmer comparison. "
            "Raw element peaks remain in raw_element_stress; recovered_stress.pvd "
            "contains the recovered nodal tensor field.\n\n"
            f"Diagnostics and SI-unit results: [{report_name}]({report_name}).\n",
            encoding="utf-8",
        )
        manifest.update(status="completed_not_engineering_validated",
                        completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory / "manifest.json", manifest)
        print(f"{scenario}: completed_not_engineering_validated; results at {directory}", flush=True)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc())
        write_json(directory / "manifest.json", manifest)
        raise


def run_fracture(
    dam_config: Path, output_root: Path, versions: dict, elmer_result: Path | None = None,
    *, processes: int = 1, mechanics_formulation: str = "hybrid", damage_acceleration: bool = False,
    final_load_substeps: int = 1,
) -> None:
    """Run one isolated phase-field job from the authoritative dam configuration."""
    if mechanics_formulation not in ("hybrid", "experimental_amor", "experimental_spectral"):
        raise ValueError(f"Unknown fracture mechanics formulation: {mechanics_formulation!r}")
    if damage_acceleration and mechanics_formulation == "hybrid":
        raise ValueError("Damage acceleration requires experimental mechanics")
    from phase_field_solver import load_schedule

    load_schedule(1, final_load_substeps)
    sys.path.insert(0, str(ROOT))
    from crack_field import (
        crack_placement_report,
        load_crack_specifications,
        load_phase_field_solver_options,
        refine_for_phase_field,
        validate_crack_placement,
    )
    from dam_mesh import import_structure
    from dam_phase_field import solve
    from dolfinx import io

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_root / "dam_phase_field" / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "config.json"
    snapshot.write_bytes(dam_config.read_bytes())
    report_config = json.loads(snapshot.read_text(encoding="utf-8"))
    specifications = load_crack_specifications(snapshot)
    manifest = {
        "status": "running",
        "analysis": "dam_phase_field",
        "refined_baseline_required": True,
        "mechanics_formulation": mechanics_formulation,
        "damage_acceleration": damage_acceleration,
        "final_load_substeps": final_load_substeps,
        "environment": versions,
        "configuration_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "started_at_utc": timestamp,
        "engineering_validated": False,
        "crack_scenario_names": [specification.name for specification in specifications],
    }
    write_json(directory / "manifest.json", manifest)
    try:
        structure = submitted_structure(snapshot)
        assert_fracture_ready(structure)
        if elmer_result is not None:
            from elmer_baseline import import_baseline

            baseline = import_baseline(structure, elmer_result, snapshot, directory)
            manifest["elmer_baseline"] = baseline
            write_json(directory / "manifest.json", manifest)
        if not specifications:
            if elmer_result is None:
                raise ValueError("Automatic crack generation requires --elmer-result.")
            from automatic_crack import generate_crack

            specification, options, generation = generate_crack(
                json.loads(snapshot.read_text(encoding="utf-8")), baseline,
            )
            specifications = [specification]
            generation["specification"] = asdict(specification)
            report_config["crack_simulations"] = [generation["report_crack"]]
            write_json(directory / "automatic_crack.json", generation)
            manifest["automatic_crack"] = generation
            manifest["crack_scenario_names"] = [specification.name]
            write_json(directory / "manifest.json", manifest)
            print(f"Generated {specification.name} at Z={specification.centre_m[2]:.6f} m", flush=True)
            for warning in generation["warnings"]:
                print(f"Automatic crack warning: {warning}", flush=True)
        else:
            options = load_phase_field_solver_options(snapshot)
        imported = import_structure(structure, maximum_volume_change=0.001)
        write_json(directory / "import_report.json", imported.report)
        placement = [
            crack_placement_report(imported, specification)
            for specification in specifications
        ]
        write_json(directory / "crack_placement.json", placement)
        validate_crack_placement(imported, specifications)
        refined = refine_for_phase_field(imported, specifications, options)
        write_json(directory / "refinement_report.json", refined.report)
        with io.XDMFFile(refined.domain.comm, directory / "mesh.xdmf", "w") as writer:
            writer.write_mesh(refined.domain)
            refined.cell_tags.name = "regions"
            refined.facet_tags.name = "boundaries"
            writer.write_meshtags(refined.cell_tags, refined.domain.geometry)
            writer.write_meshtags(refined.facet_tags, refined.domain.geometry)
        process_count = min(8, os.cpu_count() or 1) if processes == 0 else processes
        manifest["fracture_mpi_processes"] = process_count
        write_json(directory / "manifest.json", manifest)
        if process_count == 1:
            report = solve(
                refined, structure, specifications, options, directory,
                mechanics_formulation=mechanics_formulation,
                damage_acceleration=damage_acceleration,
                final_load_substeps=final_load_substeps,
                refined_baseline=True,
            ).report
        else:
            from parallel_fracture import solve_parallel

            report = solve_parallel(
                directory, specifications, options, process_count,
                mechanics_formulation=mechanics_formulation,
                damage_acceleration=damage_acceleration,
                final_load_substeps=final_load_substeps,
            )
        completed_at = datetime.now(timezone.utc).isoformat()
        write_engineering_results(
            directory,
            report_config,
            report,
            refined.report,
            placement,
            imported.report,
            configuration_sha256=manifest["configuration_sha256"],
            environment=versions,
            started_at_utc=timestamp,
            completed_at_utc=completed_at,
        )
        manifest.update(
            {
                "status": "completed_not_engineering_validated",
                "completed_at_utc": completed_at,
                "result": report,
            }
        )
        write_json(directory / "manifest.json", manifest)
        print(f"Wrote phase-field results to {directory}")
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        write_json(directory / "manifest.json", manifest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "inventory", "import-dam", "run", "run-fracture", "run-baseline"))
    parser.add_argument("--config", type=Path, default=HERE / "cracking.json")
    parser.add_argument("--dam-config", type=Path, default=LOAD_CASE)
    parser.add_argument("--elmer-result", type=Path, help="Elmer baseline PVTU for run-fracture")
    parser.add_argument("--baseline-phase", action="store_true",
                        help="Run an initially intact phase-field baseline without generating a crack")
    parser.add_argument("--fracture-processes", type=int, choices=range(9), default=1,
                        help="MPI workers for the prepared fracture mesh; 0 selects up to 8 local CPUs (default: 1)")
    parser.add_argument("--fracture-mechanics", choices=("hybrid", "experimental_amor", "experimental_spectral"), default="hybrid",
                        help="Fracture mechanics law; experimental energy splits do not enforce crack-face contact")
    parser.add_argument("--accelerate-damage", action="store_true",
                        help="Bounded Aitken acceleration for experimental laws; final acceptance is unchanged")
    parser.add_argument("--fracture-final-substeps", type=int, default=1,
                        help="Subdivide the final configured load interval (default: 1, unchanged ramp)")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--scenario", help="Run one enabled scenario by id")
    args = parser.parse_args()
    if args.elmer_result is not None and args.action != "run-fracture":
        parser.error("--elmer-result is supported only with run-fracture")
    if args.baseline_phase and (args.action != "run-fracture" or args.elmer_result is not None):
        parser.error("--baseline-phase requires run-fracture without --elmer-result")
    if args.fracture_processes != 1 and (args.action != "run-fracture" or args.baseline_phase):
        parser.error("--fracture-processes applies only to a cracking run-fracture")
    if args.fracture_mechanics != "hybrid" and (args.action != "run-fracture" or args.baseline_phase):
        parser.error("--fracture-mechanics applies only to a cracking run-fracture")
    if args.accelerate_damage and (
        args.action != "run-fracture" or args.baseline_phase or args.fracture_mechanics == "hybrid"
    ):
        parser.error("--accelerate-damage requires cracking run-fracture with experimental mechanics")
    if args.fracture_final_substeps < 1:
        parser.error("--fracture-final-substeps must be positive")
    if args.fracture_final_substeps != 1 and (args.action != "run-fracture" or args.baseline_phase):
        parser.error("--fracture-final-substeps applies only to a cracking run-fracture")
    versions = environment()
    if args.action == "run-baseline" or args.baseline_phase:
        run_baseline(args.dam_config.resolve(), args.output.resolve(), versions, phase=args.baseline_phase)
        return 0
    if args.action == "run-fracture":
        run_fracture(
            args.dam_config.resolve(), args.output.resolve(), versions,
            args.elmer_result.resolve() if args.elmer_result is not None else None,
            processes=args.fracture_processes,
            mechanics_formulation=args.fracture_mechanics,
            damage_acceleration=args.accelerate_damage,
            final_load_substeps=args.fracture_final_substeps,
        )
        return 0
    scenarios = load_scenarios(args.config)
    if args.action == "check":
        print(json.dumps({"environment": versions, "scenarios": scenarios}, indent=2))
        return 0
    output = args.output
    if args.action == "inventory":
        output.mkdir(exist_ok=True)
        write_json(output / "migration_inventory.json", inventory())
        print(f"Wrote {output / 'migration_inventory.json'}; no dam solve performed")
        return 0
    if args.action == "import-dam":
        from dam_mesh import import_structure
        from elastic_parity import reference_structure
        from dolfinx import io
        dam = next((s for s in scenarios if s["kind"] == "dam_elastic"), None)
        if dam is None:
            raise ValueError("import-dam requires a dam_elastic scenario")
        structure, hashes = reference_structure(ROOT, LOAD_CASE, ROOT / dam["reference"])
        imported = import_structure(structure, dam["maximum_relative_volume_change"])
        directory = output / "dam_import"
        directory.mkdir(parents=True, exist_ok=True)
        with io.XDMFFile(imported.domain.comm, directory / "mesh.xdmf", "w") as writer:
            writer.write_mesh(imported.domain)
            imported.cell_tags.name = "regions"
            imported.facet_tags.name = "boundaries"
            writer.write_meshtags(imported.cell_tags, imported.domain.geometry)
            writer.write_meshtags(imported.facet_tags, imported.domain.geometry)
        write_json(directory / "import_report.json", {**imported.report, "reference_sha256": hashes})
        print(json.dumps(imported.report, indent=2))
        return 0
    selected = [s for s in scenarios if s["enabled"] and (args.scenario is None or s["id"] == args.scenario)]
    if not selected:
        raise ValueError("No enabled scenarios match the requested run")
    if any(s["kind"] == "predefined_crack" for s in selected):
        raise RuntimeError(
            "Dam crack contact is not implemented or verified yet. A reduced contact benchmark "
            "does not validate 3-D crack surfaces; complete that adapter before enabling crack scenarios."
        )
    material = json.loads(LOAD_CASE.read_text())["material"]["concrete"]
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    load_hash = hashlib.sha256(LOAD_CASE.read_bytes()).hexdigest()
    for scenario in selected:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = output / scenario["id"] / timestamp
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {
            "status": "running", "scenario": scenario, "environment": versions,
            "configuration_sha256": config_hash, "load_case_sha256": load_hash,
            "started_at_utc": timestamp, "engineering_validated": False,
        }
        write_json(directory / "manifest.json", manifest)
        try:
            if scenario["kind"] == "elastic_benchmark":
                from elastic_benchmark import solve
                summary = solve(scenario, material, directory)
                status = "passed_benchmark"
            elif scenario["kind"] == "contact_benchmark":
                from contact_benchmark import solve
                summary = solve(scenario, material, directory)
                status = "passed_reduced_contact_benchmark"
            else:
                from dam_mesh import import_structure
                from dam_elasticity import solve
                from elastic_parity import compare, reference_structure
                reference = ROOT / scenario["reference"]
                structure, hashes = reference_structure(ROOT, LOAD_CASE, reference)
                manifest["reference_sha256"] = hashes
                write_json(directory / "manifest.json", manifest)
                imported = import_structure(structure, scenario["maximum_relative_volume_change"])
                write_json(directory / "import_report.json", imported.report)
                print(f"{scenario['id']}: imported {len(imported.tetrahedra)} tetrahedra", flush=True)
                displacement, stress, diagnostics = solve(imported, structure, scenario["solver"], directory)
                parity = compare(imported, structure, displacement, stress, reference, scenario["parity"])
                import numpy as np
                reaction_differences = {}
                for tag, expected in parity["elmer_ground_spring_reactions_n"].items():
                    expected = np.asarray(expected)
                    actual = np.asarray(diagnostics["ground_spring_reactions_n"][tag])
                    reaction_differences[tag] = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
                parity["relative_reaction_differences"] = reaction_differences
                summary = {"mesh_import": imported.report, "solver": diagnostics, "parity": parity}
                write_json(directory / "summary.json", summary)
                if not parity["displacement_checks_passed"] or (
                    diagnostics["relative_force_balance_error"] > scenario["parity"]["maximum_force_balance_error"]
                ) or any(
                    difference > scenario["parity"]["maximum_relative_l2_difference"]
                    for difference in reaction_differences.values()
                ):
                    raise RuntimeError("Dam elastic parity failed configured limits; inspect summary.json. Crack runs remain blocked.")
                status = "passed_provisional_elastic_parity"
        except Exception as error:
            manifest.update(status="failed", error=str(error), traceback=traceback.format_exc())
            write_json(directory / "manifest.json", manifest)
            raise
        write_json(directory / "summary.json", summary)
        manifest["status"] = status
        write_json(directory / "manifest.json", manifest)
        write_json(directory.parent / "latest.json", {"run_directory": timestamp, "status": manifest["status"]})
        print(f"{scenario['id']}: {status}; results at {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
