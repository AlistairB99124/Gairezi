"""Audit explicit finite-sector attempts without claiming mesh convergence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Aster.run_case import file_hash
from Aster.seed_benchmark import finite_sector_metrics, history_checks, specimen_metrics


def attempt(directory: Path) -> tuple[dict, dict]:
    inputs_path = directory / "inputs.json"
    report_path = directory / "benchmark_report.json"
    native_path = directory / "native_seed.json"
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report["status"] != "seed_placement_and_irreversibility_passed"
            or not inputs.get("finite_sector") or not inputs.get("specimen_divisions")
            or file_hash(inputs_path) != report["inputs_sha256"]):
        raise ValueError("Comparison requires unchanged successful finite-sector inputs")
    native = json.loads(native_path.read_text(encoding="utf-8"))
    checks = history_checks(native, inputs["shared_seed"], inputs["times"], True)
    metrics = specimen_metrics(
        native, inputs["specimen_divisions"], inputs["centre_m"][2],
        inputs["lateral_divisions"], inputs.get("normal_grading_power", 1),
        inputs.get("lateral_width_m", 1.), inputs.get("tetrahedral", False),
    )
    if checks != report["checks"] or metrics != report["specimen_metrics"]:
        raise ValueError("Persisted metrics differ from the native finite-sector history")
    sampling = finite_sector_metrics(native, inputs, metrics)
    if "finite_sector_metrics" in report and sampling != report["finite_sector_metrics"]:
        raise ValueError("Persisted sampling metrics differ from the native history")
    row = {
        "directory": str(directory.resolve()),
        "inputs_sha256": file_hash(inputs_path),
        "report_sha256": file_hash(report_path),
        "native_history_sha256": file_hash(native_path),
        "normal_divisions": inputs["specimen_divisions"],
        "normal_grading_power": inputs.get("normal_grading_power", 1),
        "lateral_divisions": inputs["lateral_divisions"],
        "lateral_width_m": inputs.get("lateral_width_m", 1.),
        "tetrahedral": inputs.get("tetrahedral", False),
        "equilibrium_integral_m3": metrics["integrated_damage_volume_m3_by_state"][2],
        "sampling": sampling,
    }
    return inputs, row


def compare(first: Path, second: Path) -> dict:
    inputs, rows = zip(attempt(first), attempt(second))
    for key in ("shared_seed", "centre_m", "times", "line_search"):
        if inputs[0][key] != inputs[1][key]:
            raise ValueError(f"Finite-sector comparison changed physical/history contract: {key}")
    materials = []
    provenance_paths = []
    for values in inputs:
        material = dict(values["material"])
        provenance = dict(material.get("provenance", {}))
        provenance_paths.append({key: provenance.pop(key) for key in (
            "fracture_energy", "regularisation_and_softening",
        ) if key in provenance})
        if "provenance" in material:
            material["provenance"] = provenance
        materials.append(material)
    if materials[0] != materials[1]:
        raise ValueError("Finite-sector comparison changed physical/history contract: material")
    settings = ("normal_divisions", "normal_grading_power", "lateral_divisions",
                "lateral_width_m", "tetrahedral")
    changed = [key for key in settings if rows[0][key] != rows[1][key]]
    if not changed:
        kind = "repeat"
    elif changed == ["normal_divisions"]:
        kind = "normal_resolution"
    elif changed == ["lateral_divisions"]:
        kind = "lateral_resolution"
    elif set(changed) == {"lateral_width_m", "lateral_divisions"} and (
        rows[0]["lateral_width_m"] * rows[1]["lateral_divisions"]
        == rows[1]["lateral_width_m"] * rows[0]["lateral_divisions"]
    ):
        kind = "extent_at_fixed_lateral_spacing"
    elif changed == ["tetrahedral"]:
        kind = "element_family"
    else:
        kind = "multiple_discretisation_changes"
    return {
        "status": "finite_seed_sensitivity_only",
        "comparison_kind": kind,
        "changed_discretisation_settings": changed,
        "relative_equilibrium_integral_change": (
            rows[1]["equilibrium_integral_m3"] / rows[0]["equilibrium_integral_m3"] - 1),
        "attempts": list(rows),
        "material_provenance_paths_by_attempt": provenance_paths,
        "engineering_validated": False,
        "mesh_convergence_established": False,
        "limitations": [
            "One pair is not two successive mesh refinements or an asymptotic limit.",
            "Runtime/build differences are not controlled by numerical-setting classification.",
            "An averaged integral does not establish pointwise field agreement.",
            "A finite seed does not satisfy the infinite-planar analytic reference.",
            "Scalar damage integral is not crack opening or physical crack volume.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.first, args.second)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "attempts"}, indent=2))
