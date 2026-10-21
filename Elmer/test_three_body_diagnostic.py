"""Preparation checks for the bonded and spring-seated wall-on-bedrock diagnostics."""
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
        config["geometry"]["bedrock"]["support"]["model"] = "distributed_elastic_granite"
        del config["geometry"]["bedrock"]["mesh"]
        loader = patch("Elmer.three_body_diagnostic.load_config", return_value=config)
        self.config = config
        loader.start()
        self.addCleanup(loader.stop)

    def test_prepared_model_has_connected_rock_and_explicit_assumptions(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = prepare(root, output)
            text = (output / "diagnostic.sif").read_text()
            self.assertEqual(report["solver_body_ids"], {"wall": 1, "bedrock": 2})
            self.assertGreater(report["wall_rock_seat_triangles"], 0)
            self.assertEqual(report["wall_underside"]["profile"], "piecewise_linear_bedrock_contours")
            self.assertEqual(report["joints"], "bonded")
            self.assertEqual(report["geometry"]["element_size_m"], .5)
            self.assertGreater(report["geometry"]["vertical_layers"], 72)
            self.assertEqual(report["geometry"]["bottom_z_m"], -87.)
            self.assertEqual(report["geometry"]["stream_margin_m"], 58.)
            self.assertEqual(report["geometry"]["bank_margin_m"], 58.)
            self.assertNotIn("fixed rock bottom z=-87 m", report["assumptions"])
            self.assertTrue(any("roller springs" in item for item in report["assumptions"]))
            self.assertNotIn("Mortar", text)
            self.assertIn("Body 2\n", text)
            self.assertNotIn("Body 3\n", text)
            self.assertNotIn("plinth", text.lower())
            self.assertIn('Name = "Granite"', text)
            self.assertIn("Youngs Modulus = Real 50000000000", text)
            rock_weight = self.config["material"]["bedrock"]["density"] * self.config["loads"]["gravity"] + 0.
            self.assertIn(f"Stress Bodyforce 3 = Real {rock_weight:.12g}\n", text)
            self.assertNotIn("Real -0\n", text)
            self.assertNotIn("Displacement 3 = Real 0", text)
            self.assertNotIn("DiagnosticFixedRockBottom", text)
            self.assertNotIn("Spring 1 =", text)
            self.assertIn("zero uplift", text)
            self.assertIn("Target Boundaries(1) = 7", text)
            self.assertEqual(
                (output / "config.json").read_bytes(), (root / "config.json").read_bytes(),
            )
            with self.assertRaisesRegex(ValueError, "new empty"):
                prepare(root, output)

    def test_rollers_put_normal_springs_on_bottom_and_all_four_sides(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = prepare(root, output)
            text = (output / "diagnostic.sif").read_text()
            mesh = (output / "three_body.msh").read_text()
        support = report["rock_outer_support"]
        self.assertEqual(support["model"], "rollers")
        self.assertEqual(set(support["faces"]), {"10", "13", "14", "15", "16"})
        boundary_map = {int(k): v for k, v in report["boundary_id_map"].items()}
        sections = dict(re.findall(r"Boundary Condition (\d+)\n(.*?)End\n", text, re.S))
        geometry = report["geometry"]
        across = geometry["rock_across_limits_m"][1] - geometry["rock_across_limits_m"][0]
        depth = geometry["rock_depth_limits_m"][1] - geometry["rock_depth_limits_m"][0]
        self.assertAlmostEqual(support["faces"]["10"]["area_m2"], across * depth, delta=1e-6 * across * depth)
        for tag, face in support["faces"].items():
            body = next(b for b in sections.values() if f'"RockRoller_{face["name"]}"' in b)
            self.assertIn(f"Target Boundaries(1) = {boundary_map[int(tag)]}\n", body)
            self.assertNotIn("Displacement", body)
            self.assertGreater(face["area_m2"], 0.)
            k = face["normal_stiffness_n_per_m3"]
            n = face["outward_normal"]
            self.assertAlmostEqual(sum(v * v for v in n), 1., places=12)
            values = {(int(i), int(j)): float(v) for i, j, v in
                      re.findall(r"Spring (\d)(\d) = Real (\S+)\n", body)}
            for i in range(3):
                for j in range(3):
                    expected = k * n[i] * n[j]
                    self.assertAlmostEqual(values.get((i + 1, j + 1), 0.), expected,
                                           delta=1e-9 * k, msg=(tag, i, j))
            # Rock boundary faces must be present in the mesh for every support tag.
            self.assertRegex(mesh, rf"\n\d+ [23] 2 {boundary_map[int(tag)]} {boundary_map[int(tag)]} ")
        k = self.config["material"]["bedrock"]["support"]["spring_n_per_m3"]
        self.assertEqual(support["faces"]["10"]["normal_stiffness_n_per_m3"], k["z"])
        self.assertEqual(support["faces"]["13"]["normal_stiffness_n_per_m3"], k["x"])
        self.assertEqual(support["faces"]["15"]["normal_stiffness_n_per_m3"], k["y"])
        self.assertEqual(text.count("RockRoller_"), 5)

    def test_missing_support_block_keeps_fixed_bottom(self):
        del self.config["material"]["bedrock"]["support"]
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = prepare(root, output)
            text = (output / "diagnostic.sif").read_text()
        self.assertEqual(report["rock_outer_support"]["model"], "fixed_bottom")
        self.assertIn("fixed rock bottom z=-87 m", report["assumptions"])
        self.assertIn("free rock sides", report["assumptions"])
        self.assertIn("DiagnosticFixedRockBottom", text)
        self.assertIn("Displacement 3 = Real 0", text)
        self.assertNotIn("RockRoller_", text)
        self.assertNotIn("Spring 11", text)

    def test_spring_seat_splits_nodes_between_wall_and_rock(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            bonded = prepare(root, Path(directory) / "bonded")
            output = Path(directory) / "spring"
            report = prepare(root, output, joints="spring")
        self.assertEqual(report["joints"], "spring")
        geometry = json.loads((root / "config.json").read_text(encoding="utf-8"))["geometry"]
        self.assertEqual(report["interface_stiffness_n_per_m3"],
                         {"wall_bedrock": geometry["bedrock"]["support"]["spring_n_per_m3"]})
        self.assertEqual(report["interface_stiffness_sources"],
                         {"wall_bedrock": "geometry.bedrock.support.spring_n_per_m3"})
        self.assertEqual(report["wall_bedrock_connection"], "spring")
        self.assertEqual(bonded["wall_bedrock_connection"], "bonded")
        self.assertEqual(report["wall_rock_seat_triangles"], bonded["wall_rock_seat_triangles"])
        # Each seat node now exists once on the wall and once on the rock.
        self.assertGreater(report["nodes"], bonded["nodes"])
        self.assertEqual(report["cells_by_body"], bonded["cells_by_body"])

    def test_spring_sif_declares_mortar_pairs_with_configured_stiffness(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = prepare(root, output, joints="spring", integrated=True)
            text = (output / "dam_model.sif").read_text()
        self.assertEqual(text.count("Apply Mortar BCs = Logical True"), 1)
        self.assertEqual(text.count("Mortar BC Coefficient Displacement"), 3)
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
        self.assertNotIn("Single wall body seated", text)
        self.assertNotIn("BedrockBaseSpring", text)
        self.assertIn("WALL-BEDROCK ANALYSIS", text)
        self.assertEqual(report["solver_input"], "dam_model.sif")
        self.assertEqual(report["concrete_element_types"], [4])
        self.assertIn("Linear System Preconditioning = ILU1\n", text)
        self.assertIn("Linear System ILU Factor = Real 0.01", text)
        self.assertIn("Linear System Residual Output = 20", text)
        self.assertNotIn("ILUT Tolerance", text)
        self.assertEqual(report["linear_solver"]["preconditioner"], "ILU1")

    def test_bonded_seat_shares_nodes_without_springs(self):
        import meshio
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "config.json")
        config["geometry"]["bedrock"]["support"]["model"] = "bonded"
        del config["geometry"]["bedrock"]["mesh"]
        with tempfile.TemporaryDirectory() as directory, patch(
            "Elmer.three_body_diagnostic.load_config", return_value=config,
        ):
            output = Path(directory)
            report = prepare(root, output, joints="spring", integrated=True)
            source = meshio.read(output / "three_body.msh")
            text = (output / "dam_model.sif").read_text()
        nodes = {region: set() for region in (1, 2)}
        for block, tags in zip(source.cells, source.cell_data["gmsh:physical"], strict=True):
            if block.type == "tetra":
                for cell, tag in zip(block.data, tags, strict=True):
                    nodes[int(tag)].update(int(n) for n in cell)
        self.assertTrue(nodes[1] & nodes[2])
        self.assertEqual(report["wall_bedrock_connection"], "bonded")
        self.assertEqual(report["interface_stiffness_n_per_m3"], {})
        self.assertEqual(text.count("Mortar BC Coefficient Displacement"), 0)
        self.assertNotIn("Apply Mortar BCs", text)
        self.assertNotIn("wall_bedrock_spring", text)
        self.assertEqual(text.count("Displacement 1 = Real 0"), 0)
        self.assertEqual(text.count("RockRoller_"), 5)
        self.assertNotIn(11, report["boundary_id_map"])
        self.assertNotIn(12, report["boundary_id_map"])

    def test_partition_check_rejects_split_joint_sides(self):
        boundary_map = {11: 11, 12: 12}
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory)
            folder = mesh / "partitioning.2"
            folder.mkdir()
            (folder / "part.1.boundary").write_text("2 11 6 0 303 1 2 3\n")
            (folder / "part.2.boundary").write_text("4 12 8 0 303 1 2 3\n")
            with self.assertRaisesRegex(RuntimeError, "wall_bedrock"):
                check_spring_partitions(mesh, 2, boundary_map)
            (folder / "part.1.boundary").write_text("2 11 6 0 303 1 2 3\n4 12 8 0 303 1 2 3\n")
            (folder / "part.2.boundary").write_text("5 1 9 0 303 1 2 3\n")
            check_spring_partitions(mesh, 2, boundary_map)

    def test_gap_check_rejects_missing_joints_and_runaway_openings(self):
        stiffness = interface_stiffness(load_config(Path(__file__).resolve().parents[1] / "config.json"))
        sane = {"locations": 10, "max_abs_gap_by_axis_m": [2e-6, 2e-6, 2e-6]}
        check_spring_gaps({"joints": {"wall_bedrock": sane}}, stiffness)
        with self.assertRaisesRegex(RuntimeError, "No wall_bedrock"):
            check_spring_gaps({"joints": {"wall_bedrock": {"locations": 0}}}, stiffness)
        with self.assertRaisesRegex(RuntimeError, "100 MPa"):
            check_spring_gaps({"joints": {"wall_bedrock": {"locations": 10,
                                                         "max_abs_gap_by_axis_m": [0., 0., 1e10]}}},
                              stiffness)

    def test_directional_springs_use_snapshot_values_in_axis_order(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "config.json")
        del config["geometry"]["bedrock"]["mesh"]
        config["geometry"]["bedrock"]["support"]["model"] = "distributed_elastic_granite"
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
            self.assertEqual(report["interface_stiffness_n_per_m3"]["wall_bedrock"],
                             {"x": 0., "y": 5e8, "z": 6e8})
            sections = dict(re.findall(r'Name = "(\w+)_spring"\n(.*?)End\n', text, re.S))
            for name, values in (("wall_bedrock", (0., 5e8, 6e8)),):
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
            values[axis] = 100e6 / stiffness["wall_bedrock"][("x", "y", "z")[axis]] * 1.01
            with self.subTest(axis=axis), self.assertRaisesRegex(RuntimeError, "100 MPa"):
                check_spring_gaps({"joints": {
                    "wall_bedrock": {"locations": 1, "max_abs_gap_by_axis_m": values},
                }}, stiffness)
        for values in ([0., 0.], [0., float("nan"), 0.], [-1., 0., 0.], None):
            with self.subTest(values=values), self.assertRaisesRegex(RuntimeError, "Invalid"):
                check_spring_gaps({"joints": {
                    "wall_bedrock": {"locations": 1, "max_abs_gap_by_axis_m": values},
                }}, stiffness)


if __name__ == "__main__":
    unittest.main()
