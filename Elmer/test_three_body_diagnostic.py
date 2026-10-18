"""Preparation checks for the bonded and stiff-spring three-body diagnostics."""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from Elmer.three_body_diagnostic import (
    SPRING_PAIRS, check_spring_gaps, check_spring_partitions, prepare,
)


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
                         {"wall_plinth": 7e11, "plinth_bedrock": 7e11})
        self.assertEqual(report["plinth_rock_seat_triangles"], bonded["plinth_rock_seat_triangles"])
        self.assertGreater(report["wall_plinth_joint_faces"], 0)
        # Each joint node now exists once per side.
        self.assertGreater(report["nodes"], bonded["nodes"] + report["wall_plinth_joint_faces"])
        self.assertEqual(report["cells_by_body"], bonded["cells_by_body"])

    def test_spring_sif_declares_mortar_pairs_with_configured_stiffness(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = prepare(root, output, joints="spring")
            text = (output / "diagnostic.sif").read_text()
        self.assertEqual(text.count("Apply Mortar BCs = Logical True"), 1)
        self.assertEqual(text.count("Mortar BC Coefficient Displacement"), 6)
        self.assertEqual(text.count("Real 700000000000\n"), 6)
        boundary_map = {int(k): v for k, v in report["boundary_id_map"].items()}
        sections = dict(re.findall(r"Boundary Condition (\d+)\n(.*?)End\n", text, re.S))
        for name, (slave, master) in SPRING_PAIRS.items():
            slave_bc = next(n for n, body in sections.items() if f'"{name}_spring"' in body)
            master_bc = re.search(r"Mortar BC = Integer (\d+)", sections[slave_bc]).group(1)
            self.assertIn(f"Target Boundaries(1) = {boundary_map[slave]}\n", sections[slave_bc])
            self.assertIn(f"Target Boundaries(1) = {boundary_map[master]}\n", sections[master_bc])
        self.assertIn("isotropic springs", text)

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
        stiffness = {"wall_plinth": 7e11, "plinth_bedrock": 7e11}
        sane = {"locations": 10, "max_abs_gap_m": 2e-6}
        check_spring_gaps({"joints": {"wall_plinth": sane, "plinth_bedrock": sane}}, stiffness)
        with self.assertRaisesRegex(RuntimeError, "No plinth_bedrock"):
            check_spring_gaps({"joints": {"wall_plinth": sane,
                                          "plinth_bedrock": {"locations": 0, "max_abs_gap_m": 0}}},
                              stiffness)
        with self.assertRaisesRegex(RuntimeError, "100 MPa"):
            check_spring_gaps({"joints": {"wall_plinth": {"locations": 10, "max_abs_gap_m": 1e10},
                                          "plinth_bedrock": sane}}, stiffness)


if __name__ == "__main__":
    unittest.main()
