"""Tests for the Code_Aster porting tooling under Aster/ (Aster_Porting.md).

These exercise the exporter/validator/generator against the real production
mesh builder. A local Code_Aster install exists on this machine, but
``RunCaseAvailabilityTests`` deliberately mocks launcher detection off so the
tests stay deterministic/fast regardless of host machine configuration;
``Aster.run_case`` / ``Aster.postprocess_baseline`` are tested only for their
"no installation found" / "no run yet" reporting paths, not a real solve.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
import subprocess
import unittest
from unittest.mock import patch
import numpy as np

from Aster import build_med_mesh, generate_elastic_case, postprocess_baseline, run_case, validate_mesh
from Elmer.regions.combined_structure import load_environment, load_foundation_support, load_material

ROOT = Path(__file__).resolve().parent.parent


class BuildMedMeshExportTests(unittest.TestCase):
    def test_export_then_validate_round_trip(self):
        with tempfile.TemporaryDirectory(suffix=".aster") as directory:
            output_dir = Path(directory)
            manifest = build_med_mesh.export(ROOT, output_dir)

            self.assertEqual(manifest["groups"]["WALL"]["cell_count"], manifest["cell_count"])
            self.assertGreater(manifest["groups"]["FOUNDATION"]["face_count"], 0)
            self.assertGreater(manifest["groups"]["UPSTREAM"]["face_count"], 0)
            self.assertGreater(manifest["groups"]["DOWNSTREAM"]["face_count"], 0)
            self.assertGreater(manifest["groups"]["CREST"]["face_count"], 0)
            self.assertIn("gmsh_named", manifest["mesh_formats_written"])
            self.assertTrue((output_dir / "mesh.msh").is_file())
            self.assertTrue((output_dir / "mesh_manifest.json").is_file())

            report = validate_mesh.validate(ROOT, output_dir)
            self.assertTrue(report["passed"])
            self.assertEqual(report["problems"], [])

    def test_validate_detects_manifest_drift_from_mesh_file(self):
        with tempfile.TemporaryDirectory(suffix=".aster") as directory:
            output_dir = Path(directory)
            build_med_mesh.export(ROOT, output_dir)

            manifest_path = output_dir / "mesh_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["groups"]["WALL"]["cell_count"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(validate_mesh.MeshValidationError, "WALL"):
                validate_mesh.validate(ROOT, output_dir)


class GmshParsingTests(unittest.TestCase):
    """Volume and face groups reuse the same integer id namespace (both start
    at 1), so the parser must key counts by (dimension, tag), not tag alone.
    """

    SAMPLE_MSH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
3 1 "WALL"
2 1 "FOUNDATION"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
$EndNodes
$Elements
2
1 4 2 1 0 1 2 3 4
2 2 2 1 0 1 2 3
$EndElements
"""

    def test_dimension_aware_counts_do_not_collide_on_shared_tag(self):
        with tempfile.TemporaryDirectory(suffix=".aster") as directory:
            path = Path(directory) / "sample.msh"
            path.write_text(self.SAMPLE_MSH, encoding="ascii")

            node_count, counts = validate_mesh.parse_gmsh_counts(path)
            self.assertEqual(node_count, 4)
            self.assertEqual(counts[(3, 1)], 1)  # one tetrahedron tagged volume-1
            self.assertEqual(counts[(2, 1)], 1)  # one triangle tagged face-1 (same numeric tag)

            physical = validate_mesh.parse_gmsh_physical_groups(path)
            self.assertEqual(physical[(3, 1)], "WALL")
            self.assertEqual(physical[(2, 1)], "FOUNDATION")


class GenerateElasticCaseTests(unittest.TestCase):
    def test_rendered_values_match_config_driven_formulas(self):
        # Uses the real exported mesh manifest under Aster/results/00_elastic_baseline
        # (already produced by Aster.build_med_mesh against the live model), since
        # render_comm reads that fixed path to get the foundation elevation.
        config_path = ROOT / "config.json"
        material = load_material(config_path)
        loads = load_environment(config_path)
        support = load_foundation_support(config_path)

        comm, inputs = generate_elastic_case.render_comm(ROOT)

        self.assertAlmostEqual(inputs["gravity_m_s2"], abs(loads.gravity_z_m_s2))
        self.assertAlmostEqual(inputs["material"]["youngs_modulus_pa"], material.youngs_modulus_pa)
        self.assertAlmostEqual(inputs["material"]["poissons_ratio"], material.poissons_ratio)
        self.assertAlmostEqual(inputs["foundation_spring_n_per_m3"]["z"], support.spring_z_n_per_m3)
        self.assertIn(f"E={material.youngs_modulus_pa:.12g}", comm)
        self.assertIn("RIGI_PARASOL", comm)
        self.assertIn("VERIFIED", comm)
        self.assertIn("FORMULE", comm)


class RunCaseAvailabilityTests(unittest.TestCase):
    def test_reports_unavailable_when_no_launcher_on_path(self):
        with tempfile.TemporaryDirectory(suffix=".aster") as directory:
            output_dir = Path(directory)
            build_med_mesh.export(ROOT, output_dir)
            generate_elastic_case.generate(
                ROOT, output_dir / "case.comm", output_dir / "case_inputs.json",
            )
            with patch.object(run_case.shutil, "which", return_value=None), \
                 patch.object(run_case, "KNOWN_INSTALL_PATHS", ()):
                with patch.dict(os.environ):
                    os.environ.pop("LAMB_CODE_ASTER_LAUNCHER", None)
                    manifest = run_case.run_case(output_dir / "case.comm", output_dir, output_dir)
            self.assertEqual(manifest["status"], "unavailable")
            self.assertFalse(manifest["installation"]["available"])

    def test_reports_missing_case_file(self):
        with tempfile.TemporaryDirectory(suffix=".aster") as directory:
            output_dir = Path(directory)
            manifest = run_case.run_case(output_dir / "does_not_exist.comm", output_dir, output_dir)
            self.assertEqual(manifest["status"], "case_file_missing")


class RunCaseResultTests(unittest.TestCase):
    def _run(self, output_dir, produce_med, extraction_error=None):
        case_path = output_dir / "case.comm"
        case_path.write_text("# test case", encoding="utf-8")
        mesh_path = output_dir / "mesh.med"
        mesh_path.touch()
        run_dir = output_dir / "run"
        run_dir.mkdir()
        for name in ("result.med", "aster_result_fields.npz", "result_stress.vtu"):
            (run_dir / name).write_text("stale", encoding="utf-8")

        def solve(*args, **kwargs):
            self.assertFalse((run_dir / "result.med").exists())
            self.assertFalse((run_dir / "aster_result_fields.npz").exists())
            self.assertFalse((run_dir / "result_stress.vtu").exists())
            if produce_med:
                (run_dir / "result.med").write_text("new result", encoding="utf-8")
            return subprocess.CompletedProcess(args[0], 0, "normal exit", "")

        with patch.object(run_case, "detect_installation", return_value={
            "available": True, "launcher_path": "test-launcher",
        }), patch.object(run_case, "build_med_mesh", return_value=mesh_path), \
             patch.object(run_case, "run_solver", side_effect=solve), \
             patch.object(run_case, "extract_result_fields", side_effect=extraction_error,
                          return_value=run_dir / "aster_result_fields.npz"):
            return run_case.run_case(case_path, output_dir, output_dir)

    def test_missing_med_cannot_report_success_or_reuse_stale_results(self):
        with tempfile.TemporaryDirectory(suffix=".aster") as directory:
            manifest = self._run(Path(directory), produce_med=False)
            self.assertEqual(manifest["status"], "failed")
            self.assertIsNone(manifest["result_med_path"])
            self.assertIsNone(manifest["result_vtu_path"])

    def test_export_failure_is_persisted_in_manifest(self):
        with tempfile.TemporaryDirectory(suffix=".aster") as directory:
            output_dir = Path(directory)
            manifest = self._run(output_dir, produce_med=True,
                                 extraction_error=RuntimeError("VTU export failed"))
            self.assertEqual(manifest["status"], "postprocessing_failed")
            self.assertEqual(manifest["message"], "VTU export failed")
            self.assertIsNone(manifest["result_vtu_path"])
            self.assertEqual(json.loads((output_dir / "run_manifest.json").read_text())["status"],
                             "postprocessing_failed")


class PostprocessBaselineTests(unittest.TestCase):
    def test_wall_and_foundation_peaks_are_reported_separately(self):
        with tempfile.TemporaryDirectory(suffix=".aster") as directory:
            output_dir = Path(directory)
            points = np.array([[0., 0., 0.], [0., 0., -1.], [0., 0., -2.], [1., 0., -2.]])
            elmer_stress = np.zeros((4, 6))
            elmer_stress[:, 0] = [1e6, .5e6, 2e6, 1.5e6]
            aster_stress = elmer_stress.copy()
            aster_stress[:2] *= .9
            aster_stress[2:] *= .8
            for name, stress in (("elmer.npz", elmer_stress), ("aster.npz", aster_stress)):
                np.savez(output_dir / name, points_m=points, displacement_m=np.ones((4, 3)) * .001,
                         stress_pa=stress, reporting_scope="concrete", source_node_ids=np.arange(10, 14),
                         body_1_mask=np.array([True, True, False, False]),
                         body_3_mask=np.array([False, False, True, True]))
            tension, compression = postprocess_baseline._peak_principal_stresses(points, elmer_stress)
            (output_dir / "elmer_baseline.json").write_text(json.dumps({
                "field_archive": "elmer.npz", "reporting_scope": "concrete",
                "max_tensile_principal_stress_pa": tension,
                "max_compressive_principal_stress_pa": compression,
            }))
            (output_dir / "run_manifest.json").write_text(json.dumps({
                "status": "completed", "result_fields_path": str(output_dir / "aster.npz"),
            }))
            report = postprocess_baseline.compare(output_dir)
            peaks = report["tensile_peak_comparison_by_region"]
            self.assertEqual(set(peaks), {"wall", "foundation"})
            self.assertEqual(peaks["wall"]["elmer_mpa"], 1.)
            self.assertAlmostEqual(peaks["wall"]["aster_mpa"], .9)
            self.assertAlmostEqual(peaks["wall"]["difference_percent"], -10.)
            self.assertEqual(peaks["foundation"]["elmer_mpa"], 2.)
            self.assertAlmostEqual(peaks["foundation"]["aster_mpa"], 1.6)
            self.assertAlmostEqual(peaks["foundation"]["difference_percent"], -20.)
            self.assertEqual(report["regions"]["foundation"]["aster_tension_pa"]["node_id"], 2)
            self.assertEqual(report["regions"]["foundation"]["aster_tension_pa"]["source_node_id"], 12)
            self.assertNotIn("max_tensile_principal_stress_pa", report)
            self.assertNotIn("max_compressive_principal_stress_pa", report)
            self.assertIn("wall_max_tension_stress_exceeds_tolerance", report["flags_outside_tolerance"])
            self.assertIn("foundation_max_tension_stress_exceeds_tolerance", report["flags_outside_tolerance"])
            self.assertNotIn("bedrock", json.dumps(report).lower())
            self.assertEqual(json.loads((output_dir / "comparison_report.json").read_text()), report)

    def test_matching_concrete_nodal_measure_and_mismatched_scope_guard(self):
        with tempfile.TemporaryDirectory(suffix=".aster") as directory:
            output_dir = Path(directory)
            points = np.array([[0., 0., 0.], [1., 0., 0.]])
            elmer_stress = np.array([[1.5e6, 0., 0., 0., 0., 0.], [0., -3e6, 0., 0., 0., 0.]])
            aster_stress = elmer_stress * .8
            for name, stress in (("elmer.npz", elmer_stress), ("aster.npz", aster_stress)):
                np.savez(output_dir / name, points_m=points, displacement_m=np.ones((2, 3)) * .001,
                         stress_pa=stress, reporting_scope="concrete")
            tension, compression = postprocess_baseline._peak_principal_stresses(points, elmer_stress)
            baseline = {"field_archive": "elmer.npz", "reporting_scope": "concrete",
                        "max_tensile_principal_stress_pa": tension,
                        "max_compressive_principal_stress_pa": compression}
            (output_dir / "elmer_baseline.json").write_text(json.dumps(baseline))
            (output_dir / "run_manifest.json").write_text(json.dumps({
                "status": "completed", "result_fields_path": str(output_dir / "aster.npz"),
            }))
            report = postprocess_baseline.compare(output_dir)
            self.assertEqual(report["reporting_scope"], "concrete")
            self.assertAlmostEqual(report["max_tensile_principal_stress_pa"]["percent_diff"], -.2)
            self.assertNotIn("bedrock", json.dumps(report).lower())
            baseline.pop("reporting_scope")
            (output_dir / "elmer_baseline.json").write_text(json.dumps(baseline))
            self.assertEqual(postprocess_baseline.compare(output_dir)["status"], "blocked")

    def test_blocked_without_elmer_baseline(self):
        with tempfile.TemporaryDirectory(suffix=".aster") as directory:
            output_dir = Path(directory)
            with self.assertRaises(FileNotFoundError):
                postprocess_baseline.compare(output_dir)

    def test_blocked_without_completed_run(self):
        with tempfile.TemporaryDirectory(suffix=".aster") as directory:
            output_dir = Path(directory)
            (output_dir / "elmer_baseline.json").write_text("{}", encoding="utf-8")
            report = postprocess_baseline.compare(output_dir)
            self.assertEqual(report["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
