"""Preparation checks for the bonded and stiff-spring three-body diagnostics."""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from Elmer.three_body_diagnostic import (
    SPRING_PAIRS, check_spring_gaps, check_spring_partitions, interface_stiffness, prepare,
)
from Elmer.model_config import load_config


class ThreeBodyDiagnosticTests(unittest.TestCase):
    def setUp(self):
        # Keep full-dam SIF/topology checks bounded; surface refinement has its
        # own geometry tests and full export verification.
        refinement = patch(
            "Elmer.bedrock_preview.parallel_rock_surface",
            side_effect=lambda points, triangles, seat_edges, outline, size, **kwargs: (points, triangles),
        )
        refinement.start()
        self.addCleanup(refinement.stop)
        # Exercise legacy-config compatibility without regenerating a large
        # Gmsh volume in each SIF test. Graded-mesh tests cover the new mesher.
        config = load_config(Path(__file__).resolve().parents[1] / "config.json")
        del config["geometry"]["bedrock"]["mesh"]
        loader = patch("Elmer.three_body_diagnostic.load_config", return_value=config)
        loader.start()
        self.addCleanup(loader.stop)

    def test_prepared_model_has_connected_rock_and_explicit_assumptions(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = prepare(root, output)
            text = (output / "diagnostic.sif").read_text()
            self.assertEqual(report["solver_body_ids"], {"plinth": 1, "wall": 2, "bedrock": 3})
            self.assertGreater(report["plinth_rock_seat_triangles"], 0)
            self.assertEqual(report["joints"], "bonded")
            self.assertEqual(report["geometry"]["element_size_m"], .5)
            self.assertGreater(report["geometry"]["vertical_layers"], 72)
            self.assertEqual(report["geometry"]["bottom_z_m"], -87.)
            self.assertEqual(report["geometry"]["stream_margin_m"], 58.)
            self.assertEqual(report["geometry"]["bank_margin_m"], 58.)
            self.assertIn("fixed rock bottom z=-87 m", report["assumptions"])
            self.assertNotIn("Mortar", text)
            self.assertIn("Body 3\n", text)
            self.assertIn('Name = "Granite"', text)
            self.assertIn("Youngs Modulus = Real 50000000000", text)
            self.assertIn("Stress Bodyforce 3 = Real -26487", text)
            self.assertIn("Displacement 3 = Real 0", text)
            self.assertNotIn("Spring 1 =", text)
            self.assertIn("zero uplift", text)
            self.assertIn("Target Boundaries(1) = 7", text)
            self.assertEqual(
                (output / "config.json").read_bytes(), (root / "config.json").read_bytes(),
            )
            with self.assertRaisesRegex(ValueError, "new empty"):
                prepare(root, output)

    def test_spring_joints_split_nodes_and_pair_both_joints(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            bonded = prepare(root, Path(directory) / "bonded")
            output = Path(directory) / "spring"
            report = prepare(root, output, joints="spring")
        self.assertEqual(report["joints"], "spring")
        self.assertEqual(report["interface_stiffness_n_per_m3"],
                         {"wall_plinth": {"x": 1.25e9, "y": 1.25e9, "z": 3e9},
                          "plinth_bedrock": {"x": 1e9, "y": 1e9, "z": 5e9}})
        self.assertEqual(report["interface_stiffness_sources"],
                         {"wall_plinth": "geometry.plinth.support.spring_n_per_m3",
                          "plinth_bedrock": "geometry.bedrock.support.spring_n_per_m3"})
        self.assertEqual(report["plinth_rock_seat_triangles"], bonded["plinth_rock_seat_triangles"])
        self.assertGreater(report["wall_plinth_joint_faces"], 0)
        # Each joint node now exists once per side.
        self.assertGreater(report["nodes"], bonded["nodes"] + report["wall_plinth_joint_faces"])
        self.assertEqual(report["cells_by_body"], bonded["cells_by_body"])

    def test_spring_sif_declares_mortar_pairs_with_configured_stiffness(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = prepare(root, output, joints="spring", integrated=True)
            text = (output / "dam_model.sif").read_text()
        self.assertEqual(text.count("Apply Mortar BCs = Logical True"), 1)
        self.assertEqual(text.count("Mortar BC Coefficient Displacement"), 6)
        self.assertNotIn("700000000000", text)
        boundary_map = {int(k): v for k, v in report["boundary_id_map"].items()}
        sections = dict(re.findall(r"Boundary Condition (\d+)\n(.*?)End\n", text, re.S))
        for name, (slave, master) in SPRING_PAIRS.items():
            slave_bc = next(n for n, body in sections.items() if f'"{name}_spring"' in body)
            master_bc = re.search(r"Mortar BC = Integer (\d+)", sections[slave_bc]).group(1)
            self.assertIn(f"Target Boundaries(1) = {boundary_map[slave]}\n", sections[slave_bc])
            self.assertIn(f"Target Boundaries(1) = {boundary_map[master]}\n", sections[master_bc])
            for i, axis in enumerate(("x", "y", "z"), start=1):
                value = report["interface_stiffness_n_per_m3"][name][axis]
                self.assertIn(
                    f"Mortar BC Coefficient Displacement {i} = Real {value:.12g}\n",
                    sections[slave_bc],
                )
        self.assertIn("directional springs (global x/y/z)", text)
        self.assertNotIn("Spring 1 =", text)
        self.assertNotIn("Spring 2 =", text)
        self.assertNotIn("Spring 3 =", text)
        self.assertIn('Output File Name = "three_body_results"', text)
        self.assertNotIn("Foundation support only", text)
        self.assertNotIn("Provisional bonded wall/plinth", text)
        self.assertNotIn("BedrockBaseSpring", text)
        self.assertIn("THREE-BODY ANALYSIS", text)
        self.assertEqual(report["solver_input"], "dam_model.sif")
        self.assertEqual(report["concrete_element_types"], [4])
        self.assertIn("Linear System Preconditioning = ILU1\n", text)
        self.assertIn("Linear System ILU Factor = Real 0.01", text)
        self.assertIn("Linear System Residual Output = 20", text)
        self.assertNotIn("ILUT Tolerance", text)
        self.assertEqual(report["linear_solver"]["preconditioner"], "ILU1")

    def test_partition_check_rejects_split_joint_sides(self):
        boundary_map = {8: 9, 9: 10, 11: 11, 12: 12}
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory)
            folder = mesh / "partitioning.2"
            folder.mkdir()
            (folder / "part.1.boundary").write_text("1 9 5 0 404 1 2 3 4\n2 11 6 0 303 1 2 3\n")
            (folder / "part.2.boundary").write_text("3 10 7 0 404 1 2 3 4\n4 12 8 0 303 1 2 3\n")
            with self.assertRaisesRegex(RuntimeError, "wall_plinth"):
                check_spring_partitions(mesh, 2, boundary_map)
            (folder / "part.1.boundary").write_text(
                "1 9 5 0 404 1 2 3 4\n3 10 7 0 404 1 2 3 4\n"
                "2 11 6 0 303 1 2 3\n4 12 8 0 303 1 2 3\n",
            )
            (folder / "part.2.boundary").write_text("5 1 9 0 303 1 2 3\n")
            check_spring_partitions(mesh, 2, boundary_map)

    def test_gap_check_rejects_missing_joints_and_runaway_openings(self):
        stiffness = interface_stiffness(load_config(Path(__file__).resolve().parents[1] / "config.json"))
        sane = {"locations": 10, "max_abs_gap_by_axis_m": [2e-6, 2e-6, 2e-6]}
        check_spring_gaps({"joints": {"wall_plinth": sane, "plinth_bedrock": sane}}, stiffness)
        with self.assertRaisesRegex(RuntimeError, "No plinth_bedrock"):
            check_spring_gaps({"joints": {"wall_plinth": sane,
                                          "plinth_bedrock": {"locations": 0}}},
                              stiffness)
        with self.assertRaisesRegex(RuntimeError, "100 MPa"):
            check_spring_gaps({"joints": {"wall_plinth": {"locations": 10,
                                                        "max_abs_gap_by_axis_m": [0., 0., 1e10]},
                                          "plinth_bedrock": sane}}, stiffness)

    def test_directional_springs_use_snapshot_values_in_axis_order(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "config.json")
        del config["geometry"]["bedrock"]["mesh"]
        config["geometry"]["plinth"]["support"]["spring_n_per_m3"] = {
            "z": 3e8, "x": 1e8, "y": 2e8,
        }
        config["geometry"]["bedrock"]["support"]["spring_n_per_m3"] = {
            "y": 5e8, "z": 6e8, "x": 0.,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input"
            source.mkdir()
            (source / "config.json").write_text(json.dumps(config))
            with patch("Elmer.three_body_diagnostic.load_config", side_effect=load_config):
                report = prepare(source, Path(directory) / "model", joints="spring", integrated=True)
            text = (Path(directory) / "model" / "dam_model.sif").read_text()
            self.assertEqual(report["interface_stiffness_n_per_m3"]["wall_plinth"],
                             {"x": 1e8, "y": 2e8, "z": 3e8})
            sections = dict(re.findall(r'Name = "(\w+)_spring"\n(.*?)End\n', text, re.S))
            for name, values in (("wall_plinth", (1e8, 2e8, 3e8)),
                                 ("plinth_bedrock", (0., 5e8, 6e8))):
                for i, value in enumerate(values, start=1):
                    self.assertIn(
                        f"Mortar BC Coefficient Displacement {i} = Real {value:.12g}\n",
                        sections[name],
                    )

    def test_gap_checks_use_each_axis_stiffness_and_reject_invalid_measurements(self):
        stiffness = {name: {"x": 1e9, "y": 2e9, "z": 5e9} for name in SPRING_PAIRS}
        sane = {"locations": 1, "max_abs_gap_by_axis_m": [.09, 0., 0.]}
        check_spring_gaps({"joints": {name: sane for name in SPRING_PAIRS}}, stiffness)
        for axis in range(3):
            values = [0., 0., 0.]
            values[axis] = 100e6 / stiffness["wall_plinth"][("x", "y", "z")[axis]] * 1.01
            with self.subTest(axis=axis), self.assertRaisesRegex(RuntimeError, "100 MPa"):
                check_spring_gaps({"joints": {
                    "wall_plinth": {"locations": 1, "max_abs_gap_by_axis_m": values},
                    "plinth_bedrock": sane,
                }}, stiffness)
        for values in ([0., 0.], [0., float("nan"), 0.], [-1., 0., 0.], None):
            with self.subTest(values=values), self.assertRaisesRegex(RuntimeError, "Invalid"):
                check_spring_gaps({"joints": {
                    "wall_plinth": {"locations": 1, "max_abs_gap_by_axis_m": values},
                    "plinth_bedrock": sane,
                }}, stiffness)


if __name__ == "__main__":
    unittest.main()
