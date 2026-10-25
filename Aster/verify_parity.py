"""Verify an existing parity solve without launching either solver again."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ELMER_DIR = Path(__file__).resolve().parent.parent / "Elmer"
if str(ELMER_DIR) not in sys.path:
    sys.path.insert(0, str(ELMER_DIR))

from Aster.build_three_body_mesh_arrays import _read_gmsh_ascii
from Aster.elastic_diagnostics import diagnostics
from Aster.parity_checks import (
    common_element_stress_checks, fixed_sample_checks, select_stress_samples, source_node_ids,
)
from Aster.source_contract import audit_source_materials, native_projector_audit, sha256
from Elmer.analyze_stress import parse_vtu


def verify(output_dir: Path) -> dict:
    manifest = json.loads((output_dir / "mesh_manifest.json").read_text())
    source = Path(manifest["source_mesh"])
    material_audit = audit_source_materials(source)
    for name, digest in manifest["source_hashes"].items():
        if sha256(source.parent / name) != digest:
            raise ValueError(f"Frozen source input changed: {name}")
    reference = json.loads((output_dir / "elmer_baseline.json").read_text())
    for name, digest in reference["source_sha256"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Frozen source result changed: {name}")
    run = json.loads((output_dir / "run_manifest.json").read_text())
    if run["status"] != "completed" or sha256(Path(run["case_path"])) != run["case_sha256"]:
        raise ValueError("No current completed solve for verification")
    if sha256(output_dir / "mesh_manifest.json") != run["mesh_manifest_sha256"]:
        raise ValueError("Mesh manifest changed since the solve")
    inputs = json.loads((output_dir / "case_inputs.json").read_text())
    if inputs["source_mesh_manifest_sha256"] != run["mesh_manifest_sha256"]:
        raise ValueError("Generated case was not pinned to the solved mesh manifest")
    if inputs["case_sha256"] != run["case_sha256"]:
        raise ValueError("Generated case inputs do not describe the completed solve")
    with np.load(output_dir / "three_body_mesh_arrays.npz") as arrays:
        aster = np.load(output_dir / "run" / "_diagnostic_displacement.npz")
        aster_diagnostics = diagnostics(output_dir, aster["displacement_m"])
        source_points, source_cells, source_tags, source_faces, face_tags = _read_gmsh_ascii(source)
        if list(source.parent.glob("p*_part*.dat")):
            projector_audit = native_projector_audit(source, source_points, source_faces, face_tags)
            projector_audit["status"] = "audited"
        else:
            projector_audit = {
                "status": "unavailable",
                "reason": "Saved source run contains no native mortar projector exports.",
            }
        projector_path = output_dir / "native_projector_audit.json"
        projector_path.write_text(json.dumps(projector_audit, indent=2, allow_nan=False) + "\n")
        if not np.array_equal(arrays["nodes"], source_points[arrays["source_node_ids"]]):
            raise ValueError("Exported coordinates differ from source node identities")
        for key, tags in (("tetra4_concrete", (1, 3)), ("tetra4_bedrock", (2,))):
            if not np.array_equal(arrays["source_node_ids"][arrays[key]], source_cells[np.isin(source_tags, tags)]):
                raise ValueError("Exported volume topology differs from source")
        points, displacement, _, cells = parse_vtu(source.parent / "results" / "three_body_results_t0001.pvtu")
        ids = source_node_ids(points - displacement, cells, source_points, source_cells)
        source_displacement = np.full(source_points.shape, np.nan)
        source_displacement[ids] = displacement
        if not np.allclose(displacement, source_displacement[ids], rtol=1e-8, atol=1e-12):
            raise ValueError("Partition duplicates have inconsistent source displacements")
        if not np.isfinite(source_displacement[arrays["source_node_ids"]]).all():
            raise ValueError("Source displacement does not cover the full model")
        # Evaluated with the port's lumped springs: not Elmer's native assembled reactions/energy.
        reference_diagnostics = diagnostics(output_dir, source_displacement[arrays["source_node_ids"]])
        concrete = json.loads(Path(manifest["source_config"]).read_text())["material"]["concrete"]
        element_checks = common_element_stress_checks(
            arrays["nodes"], arrays["tetra4_concrete"], arrays["concrete_body_ids"],
            source_displacement[arrays["source_node_ids"]], aster["displacement_m"],
            concrete["youngs_modulus"], concrete["poisson_ratio"],
        )
        sample_path = output_dir / "stress_samples.json"
        if sample_path.is_file():
            sample_contract = json.loads(sample_path.read_text())
        else:
            sample_contract = select_stress_samples(
                arrays["nodes"], arrays["tetra4_concrete"], arrays["concrete_body_ids"],
            )
            sample_path.write_text(json.dumps(sample_contract, indent=2, allow_nan=False) + "\n")
        sample_checks = fixed_sample_checks(
            arrays["nodes"], arrays["tetra4_concrete"], arrays["concrete_body_ids"],
            source_displacement[arrays["source_node_ids"]], aster["displacement_m"],
            concrete["youngs_modulus"], concrete["poisson_ratio"], sample_contract,
        )
    energy_difference = abs(
        aster_diagnostics["linear_strain_energy_j"] - reference_diagnostics["linear_strain_energy_j"]
    ) / abs(reference_diagnostics["linear_strain_energy_j"])
    comparison_path = output_dir / "comparison_report.json"
    comparison = json.loads(comparison_path.read_text())
    checks = {
        "source_volume_cell_count": len(source_cells) == manifest["concrete_cells"] + manifest["bedrock_cells"],
        "force_balance": aster_diagnostics["relative_force_balance_error"] <= .02,
        "moment_balance": aster_diagnostics["relative_moment_balance_error"] <= .02,
        "energy_comparison": energy_difference <= .02,
        "concrete_common_tensor_comparison": all(
            region["relative_volume_weighted_l2_tensor_difference"] <= .02
            for region in element_checks.values()
        ),
        "fixed_interior_tensor_comparison": all(
            sample["relative_tensor_difference"] <= .02 for sample in sample_checks.values()
        ),
        "native_interface_integration": (
            projector_audit["status"] == "audited"
            and projector_audit["maximum_relative_lumped_mass_difference"] <= .02
            and not projector_audit["interfaces_without_native_projector"]
        ),
    }
    result = {
        "schema_version": 1, "reporting_scope": "concrete",
        "source_hashes": manifest["source_hashes"],
        "verified_artifact_hashes": {
            name: sha256(output_dir / name) for name in (
                "case.comm", "case_inputs.json", "mesh_manifest.json", "three_body_mesh_arrays.npz",
                "mesh.med", "run/result.med", "run/aster_result_fields.npz", "elmer_baseline_fields.npz",
                "stress_samples.json",
                "native_projector_audit.json",
            )
        },
        "source_material_audit": material_audit,
        "tensile_peak_comparison_by_region": comparison["tensile_peak_comparison_by_region"],
        "common_linear_tetrahedron_stress_comparison": element_checks,
        "fixed_interior_samples": sample_checks,
        "native_projector_audit": {
            key: value for key, value in projector_audit.items()
            if key not in ("interfaces", "source_artifact_hashes")
        },
        "checks": checks, "aster_global_equilibrium": aster_diagnostics,
        "reference_energy_evaluated_with_port_springs_j": reference_diagnostics["linear_strain_energy_j"],
        "relative_energy_difference": energy_difference,
        "recovered_nodal_peak_diagnostic_within_tolerance": not comparison["flags_outside_tolerance"],
        "baseline_verified": False,
        "remaining_gates": [
            "Native source reactions/energy must be exported for an independent assembly comparison.",
            "Surface-spring lumping differs from the source consistent Galerkin integration.",
            "Provisional fixed-offset sample locations require engineering approval.",
            "Mesh-convergence study requires verification before baseline acceptance.",
        ],
    }
    path = output_dir / "verification_report.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    comparison["verification"] = {"checks": checks, "report_path": str(path.resolve())}
    comparison["status"] = "requires_verification"
    comparison_path.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = verify(args.output_dir.resolve())
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["baseline_verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
