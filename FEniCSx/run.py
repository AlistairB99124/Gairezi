"""Run validated FEniCSx scenarios from Ubuntu/WSL."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

from configuration import load_scenarios

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOAD_CASE = ROOT / "Elmer" / "load_cases.json"


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
        "source": "structural_model.build_combined_structure + Elmer/load_cases.json",
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
            "Preserve separate coincident wall/plinth nodes and both interface tags.",
            "Reproduce the current Elmer directional spring laws without silently bonding the interface.",
            "Verify displacement probes, reactions and global equilibrium against Elmer.",
            "Verify crack contact and initial-gap geometry on a small benchmark before dam scenarios.",
        ],
    }


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "inventory", "import-dam", "run"))
    parser.add_argument("--config", type=Path, default=HERE / "cracking.json")
    parser.add_argument("--scenario", help="Run one enabled scenario by id")
    args = parser.parse_args()
    scenarios = load_scenarios(args.config)
    versions = environment()
    if args.action == "check":
        print(json.dumps({"environment": versions, "scenarios": scenarios}, indent=2))
        return 0
    output = HERE / "results"
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
    material = json.loads(LOAD_CASE.read_text())["material"]
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
