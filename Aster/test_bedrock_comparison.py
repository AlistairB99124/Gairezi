"""Rigid-seat topology, common stress measure and installed-Aster checks."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from Aster.test_automatic_cohesive import ROOT, column_fixture, fixture_manifest
from Aster.generate_three_body_case import generate
from Aster.run_case import detect_installation, run_case
from bedrock_comparison import rigid_seat, without_bedrock, write_fields
from Elmer.model_config import load_config, validate_config, ConfigurationValidationError


class BedrockComparisonTests(unittest.TestCase):
    def test_only_rock_removed_and_seat_fixed_without_welding(self):
        source = column_fixture()
        retained, mapping, seat = rigid_seat(source)
        reduced = without_bedrock(source)
        self.assertEqual(len(seat), 4)
        self.assertEqual(len(reduced["fixed_support_nodes"]), 6)
        self.assertEqual(len(reduced["tetra4_bedrock"]), 0)
        np.testing.assert_array_equal(reduced["nodes"][reduced["tetra4_concrete"]],
                                      source["nodes"][source["tetra4_concrete"]])
        np.testing.assert_array_equal(reduced["interface_stiffness_n_per_m"], source["interface_stiffness_n_per_m"])
        self.assertFalse(np.any(reduced["seg2_wall_foundation_springs"][:, 0] ==
                                reduced["seg2_wall_foundation_springs"][:, 1]))
        self.assertEqual(len(retained), len(reduced["nodes"]))
        self.assertTrue(np.all(mapping[seat] >= 0))

    def test_no_foundation_rejected(self):
        arrays = column_fixture()
        arrays["concrete_body_ids"][:] = 1
        with self.assertRaisesRegex(ValueError, "foundation"):
            without_bedrock(arrays)

    def test_affine_stress_uses_correct_gradient_on_skew_tets(self):
        arrays = without_bedrock(column_fixture())
        arrays["nodes"] = arrays["nodes"] @ np.array(((1., .3, .1), (.2, 1., .4), (.1, .2, 1.)))
        gradient = np.array(((1e-6, 2e-6, 0), (0, -1e-6, 0), (0, 0, 3e-6)))
        u = arrays["nodes"] @ gradient
        material = {"youngs_modulus": 35e9, "poisson_ratio": .2}
        with tempfile.TemporaryDirectory() as directory:
            report = write_fields(Path(directory), arrays, u, material)
            header, payload = Path(report["vtu"]).read_bytes().split(b'<AppendedData encoding="raw">', 1)
            mesh = ET.fromstring(header + b"</VTKFile>")
            field = mesh.find(".//CellData/DataArray[@Name='stress_pa']")
            offset = int(field.attrib["offset"])
            payload = payload[1:]
            length = struct.unpack_from("<I", payload, offset)[0]
            exported_stress = np.frombuffer(payload, dtype="<f8", count=length // 8,
                                           offset=offset + 4).reshape(-1, 9)
            mu = 35e9 / 2.4
            lam = 35e9 * .2 / (1.2 * .6)
            expected = mu * (gradient + gradient.T) + lam * np.trace(gradient) * np.eye(3)
            np.testing.assert_allclose(exported_stress,
                                       np.repeat(expected.reshape(1, 9), len(arrays["tetra4_concrete"]), axis=0),
                                       atol=1e-8)
            self.assertEqual(set(report["bodies"]), {"wall", "foundation"})
            self.assertIsNotNone(mesh.find(".//PointData/DataArray[@Name='displacement_m']"))
            with self.assertRaisesRegex(ValueError, "finite"):
                write_fields(Path(directory), arrays, u * np.nan, material)

    def test_configuration_bool_and_foundation_gate(self):
        config = load_config(ROOT / "config.json")
        config["fracture_debug"] = {"compare_with_no_bedrock": True}
        validate_config(config)
        config["fracture_debug"]["compare_with_no_bedrock"] = "true"
        with self.assertRaises(ConfigurationValidationError):
            validate_config(config)
        config["fracture_debug"]["compare_with_no_bedrock"] = True
        config["geometry"].pop("foundation")
        with self.assertRaises(ConfigurationValidationError):
            validate_config(config)

    def test_generated_rigid_case_runs_with_installed_aster(self):
        if not detect_installation().get("launcher_path"):
            self.skipTest("Code_Aster is not installed")
        arrays = without_bedrock(column_fixture())
        with tempfile.TemporaryDirectory(prefix="lamb-rigid-seat-") as temporary:
            directory = Path(temporary)
            config = load_config(ROOT / "config.json")
            config_path = directory / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            source = fixture_manifest(arrays)
            source.update(
                rigid_seat_sensitivity=True, source_config=str(config_path), source_mesh=str(directory / "source.msh"),
                source_hashes={"config.json": hashlib.sha256(config_path.read_bytes()).hexdigest()},
                source_pressures={name: {"surface_z_m": -100., "gradient_pa_per_m": 9810.}
                                  for name in ("UpstreamHydrostaticPressure", "DownstreamTailwater")},
                source_crest_pressure_pa=0., groups={"CONCRETE": {"min_xyz_m": [0, 0, 0]},
                                                   "ROCK_BOTTOM": {"area_m2": 1.}},
                wall_foundation_spring_group_all="WALL_FOUNDATION_SPRINGS",
                wall_foundation_spring_total_area_m2=1., simplifications_vs_elmer=[])
            manifest = directory / "mesh_manifest.json"
            manifest.write_text(json.dumps(source), encoding="utf-8")
            np.savez_compressed(directory / "three_body_mesh_arrays.npz", **arrays)
            with patch("Aster.source_contract.audit_source_materials"):
                generate(case_path=directory / "case.comm", inputs_path=directory / "case_inputs.json",
                         mesh_manifest_path=manifest)
            text = (directory / "case.comm").read_text()
            self.assertIn("GROUP_NO='FIXED_SEAT'", text)
            self.assertNotIn("_F(GROUP_MA='BEDROCK', MATER=", text)
            result = run_case(case_path=directory / "case.comm", output_dir=directory, mesh_dir=directory,
                              arrays_filename="three_body_mesh_arrays.npz",
                              converter_filename="convert_three_body_to_med.py", time_limit_s=120)
            self.assertEqual(result["status"], "completed", result)
            with np.load(directory / "run" / "_diagnostic_displacement.npz") as archive:
                u = archive["displacement_m"]
            self.assertTrue(np.isfinite(u).all())
            np.testing.assert_allclose(u[arrays["fixed_support_nodes"]], 0., atol=1e-14)
            self.assertGreater(float(np.linalg.norm(u)), 0.)
