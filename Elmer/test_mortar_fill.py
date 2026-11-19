"""Tests for the mortar fill on the wall's thickness steps (no solver; builds a coarse wall mesh)."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Elmer.model_config import CONFIG_PATH, mortar_fill
from Elmer.regions.model_groups import FACE_GROUP_IDS
from Elmer.regions.mortar_fill import MortarFillError, build_mortar_fill, wall_with_fill

MORTAR = {"density": 2200.0, "youngs_modulus": 3.5e8, "poisson_ratio": 0.2,
          "bond_spring_n_per_m3": {"x": 1.25e9, "y": 1.25e9, "z": 1.25e9}}
TWO_STEPS = [{"top_z_m": -20.0, "radial_width_m": 4.5}, {"top_z_m": -25.0, "radial_width_m": 5.0}]


def stepped_config(steps, mortar=MORTAR, **mortar_changes) -> dict:
    """The checked-in configuration with a 4 m wall, these steps and a coarse mesh."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    wall = config["geometry"]["wall"]
    wall.update(radial_width_m=4.0, centerline_radius_m=78.0, downstream_radius_m=76.0,
                thickness_steps=copy.deepcopy(steps), joint_support={"model": "bonded"})
    wall.pop("step_end_taper_ratio", None)
    config["geometry"]["target_wall_tetrahedral_elements"] = 40000
    config["bedrock_analysis"] = {**config.get("bedrock_analysis", {}), "mode": "elastic_layer"}
    if mortar is not None:
        config["material"]["mortar"] = {**copy.deepcopy(mortar), **mortar_changes}
    return config


def build(config: dict):
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "config.json").write_text(json.dumps(config), encoding="utf-8")
        wall, fill = wall_with_fill(Path(directory))
    nodes = np.asarray(wall.nodes)
    return wall, fill, nodes, np.hypot(nodes[:, 0], nodes[:, 1])


def volumes(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    corners = points[tetrahedra]
    return np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0


class TwoStepFillTests(unittest.TestCase):
    """Option 5 of the sweep: 0.5 m steps at 20 m and 25 m depth, fill at 1 in 10."""

    @classmethod
    def setUpClass(cls):
        cls.config = stepped_config(TWO_STEPS)
        cls.wall, cls.fill, cls.nodes, cls.radius = build(cls.config)
        cls.steps = {step["top_z_m"]: step for step in cls.fill.report["steps"]}

    def test_each_step_gets_a_fill_of_the_configured_slope(self):
        self.assertEqual(sorted(self.steps), [-25.0, -20.0])
        self.assertEqual(self.fill.report["steps_without_a_ledge_z_m"], [])
        for top_z, step in self.steps.items():
            self.assertEqual((step["run_m"], step["requested_rise_m"], step["radial_layers"]), (0.5, 5.0, 1))
            self.assertAlmostEqual(step["built_rise_m"], 5.0)
            self.assertAlmostEqual(step["built_slope_run_per_rise"], 0.1)
            self.assertAlmostEqual(step["top_of_fill_z_m"], top_z + 5.0)
        # The lower fill ends exactly where the upper step's ledge begins.
        self.assertAlmostEqual(self.steps[-25.0]["top_of_fill_z_m"], -20.0)

    def test_volume_is_that_of_the_triangular_section_swept_along_the_wall(self):
        for top_z, face_radius in ((-20.0, 76.0), (-25.0, 75.5)):
            step = self.steps[top_z]
            swept_angle = step["arc_length_at_wall_face_m"] / face_radius
            # Area of the section times the path of its centroid, one third of the run out from the wall face.
            expected = 0.5 * step["run_m"] * step["built_rise_m"] * (face_radius - step["run_m"] / 3) * swept_angle
            self.assertAlmostEqual(step["volume_m3"] / expected, 1.0, places=3)
        self.assertAlmostEqual(self.fill.report["volume_m3"], sum(step["volume_m3"] for step in self.steps.values()))
        self.assertGreater(self.fill.report["volume_m3"], 100.0)

    def test_every_cell_has_volume_and_is_right_handed(self):
        cell_volumes = volumes(self.fill.points, self.fill.tetrahedra)
        self.assertGreater(cell_volumes.min(), 0.0)
        self.assertAlmostEqual(cell_volumes.sum(), self.fill.report["volume_m3"])
        self.assertEqual(self.fill.report["tetrahedra"], len(self.fill.tetrahedra))
        self.assertEqual(self.fill.tetrahedra.max(), len(self.fill.points) - 1)

    def test_the_fill_has_its_own_nodes_and_repeats_the_wall_faces_it_covers(self):
        wall_faces = np.asarray([self.wall.boundaries[row][1] for row in self.fill.wall_faces])
        self.assertEqual(len(wall_faces), len(self.fill.tie_faces))
        self.assertEqual(len(set(self.fill.wall_faces)), len(self.fill.wall_faces))
        self.assertTrue(all(self.wall.boundaries[row][0] == FACE_GROUP_IDS["DOWNSTREAM"] for row in self.fill.wall_faces))
        located = lambda points: sorted(tuple(sorted(tuple(p) for p in np.round(face, 8))) for face in points)  # noqa: E731
        self.assertEqual(located(self.fill.points[self.fill.tie_faces]), located(self.nodes[wall_faces]))
        # Covered: the ledges (horizontal) and the wall face above each of them (vertical), nothing else.
        heights = np.ptp(self.nodes[wall_faces][:, :, 2], axis=1)
        on_ledge = heights < 1e-9
        self.assertEqual(sorted(np.unique(np.round(self.nodes[wall_faces][on_ledge][:, :, 2], 6)).tolist()), [-25.0, -20.0])
        self.assertTrue((np.ptp(self.radius[wall_faces][~on_ledge], axis=1) < 1e-6).all())

    def test_the_surface_is_closed_and_every_outside_face_is_accounted_for(self):
        counts = Counter(tuple(sorted(cell[list(pattern)].tolist())) for cell in self.fill.tetrahedra
                         for pattern in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
        outside = {face for face, count in counts.items() if count == 1}
        self.assertEqual(max(counts.values()), 2)
        listed = [tuple(sorted(face)) for group in (self.fill.tie_faces, self.fill.slope_faces, self.fill.end_faces)
                  for face in group.tolist()]
        self.assertEqual(len(listed), len(set(listed)))
        self.assertEqual(set(listed), outside)
        edges = Counter(tuple(sorted((face[a], face[b]))) for face in outside for a, b in ((0, 1), (1, 2), (0, 2)))
        self.assertEqual(set(edges.values()), {2})

    def test_the_new_downstream_face_is_a_straight_slope_from_the_ledge_edge_to_the_wall(self):
        points = self.fill.points
        radius = np.hypot(points[:, 0], points[:, 1])
        for top_z, face_radius in ((-20.0, 76.0), (-25.0, 75.5)):
            nodes = np.unique(self.fill.slope_faces)
            nodes = nodes[(points[nodes, 2] > top_z - 1e-6) & (points[nodes, 2] < top_z + 5.0 + 1e-6)
                          & (radius[nodes] < face_radius + 1e-6) & (radius[nodes] > face_radius - 0.5 - 1e-6)]
            thickness = 0.5 * (1.0 - (points[nodes, 2] - top_z) / 5.0)
            np.testing.assert_allclose(radius[nodes], face_radius - thickness, atol=1e-6)
            self.assertAlmostEqual(radius[nodes].min(), face_radius - 0.5, places=6)
        # Nothing of the fill is inside the wall.
        cell_centres = points[self.fill.tetrahedra].mean(axis=1)
        centre_radius = np.hypot(cell_centres[:, 0], cell_centres[:, 1])
        upper = cell_centres[:, 2] > -20.0
        self.assertTrue((centre_radius[upper] < 76.0).all() and (centre_radius[~upper] < 75.5).all())
        # The slope faces look away from the wall (towards the arch centre) and never upstream.
        faces = points[self.fill.slope_faces]
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        outward = faces.mean(axis=1)[:, :2]
        self.assertTrue((np.einsum("ij,ij->i", normals[:, :2], outward) < 0).all())

    def test_a_fill_ends_with_a_flat_face_at_each_abutment(self):
        points = self.fill.points
        angle = np.arctan2(points[:, 1], points[:, 0])
        self.assertEqual(len(self.fill.end_faces), 20)  # two fills, two ends each, five triangles per end
        for face in self.fill.end_faces:
            self.assertLess(np.ptp(angle[face]), 1e-9)

    def test_the_fill_needs_the_wall_split_into_tetrahedra(self):
        quads = [(tag, (*face, face[0])) for tag, face in self.wall.boundaries]
        with self.assertRaisesRegex(MortarFillError, "split into tetrahedra"):
            build_mortar_fill(mortar_fill(self.config), self.wall.nodes, quads)


class FillShapeTests(unittest.TestCase):
    def test_the_top_of_the_fill_snaps_to_a_wall_course(self):
        # 1 in 5 asks for 2.5 m, which is not a whole number of this coarse wall's courses.
        wall, fill, nodes, radius = build(stepped_config([TWO_STEPS[0]], slope_run_per_rise=0.2))
        (step,) = fill.report["steps"]
        self.assertEqual(step["requested_rise_m"], 2.5)
        on_face = np.unique([face for tag, face in wall.boundaries if tag == FACE_GROUP_IDS["DOWNSTREAM"]])
        levels = np.unique(np.round(nodes[on_face][np.abs(radius[on_face] - 76.0) < 1e-6][:, 2], 6))
        above = levels[levels > -20.0 + 1e-6] + 20.0
        nearest = above[np.argmin(np.abs(above - 2.5))]
        self.assertNotAlmostEqual(nearest, 2.5, places=3)
        self.assertAlmostEqual(step["built_rise_m"], nearest, places=6)
        self.assertAlmostEqual(step["top_of_fill_z_m"], -20.0 + nearest, places=6)
        self.assertAlmostEqual(step["built_slope_run_per_rise"], 0.5 / nearest, places=6)
        self.assertEqual(step["courses"], int(np.sum(above <= nearest + 1e-6)))
        self.assertEqual(fill.report["requested_slope_run_per_rise"], 0.2)
        self.assertAlmostEqual(fill.points[:, 2].max(), -20.0 + nearest, places=6)

    def test_a_wide_step_gets_one_layer_per_radial_cell_of_the_ledge(self):
        # A 2 m step is wider than this wall's cells, so the wall's ledge is more than one cell wide.
        wall, fill, nodes, radius = build(stepped_config([{"top_z_m": -25.0, "radial_width_m": 6.0}],
                                                         slope_run_per_rise=0.2))
        (step,) = fill.report["steps"]
        on_face = np.unique([face for tag, face in wall.boundaries if tag == FACE_GROUP_IDS["DOWNSTREAM"]])
        ledge = on_face[np.abs(nodes[on_face, 2] + 25.0) < 1e-6]
        cells_across = len(np.unique(np.round(radius[ledge], 6))) - 1
        self.assertGreaterEqual(cells_across, 2)
        self.assertEqual((step["run_m"], step["radial_layers"]), (2.0, cells_across))
        self.assertLess(abs(step["built_rise_m"] - 10.0), wall.element_size_target_m / 2 + 1e-6)
        self.assertGreater(volumes(fill.points, fill.tetrahedra).min(), 0.0)
        swept_angle = step["arc_length_at_wall_face_m"] / 76.0
        expected = 0.5 * 2.0 * step["built_rise_m"] * (76.0 - 2.0 / 3) * swept_angle
        self.assertAlmostEqual(step["volume_m3"] / expected, 1.0, places=3)
        wall_faces = np.asarray([wall.boundaries[row][1] for row in fill.wall_faces])
        self.assertEqual(len(wall_faces), len(fill.tie_faces))
        # Its underside is as many triangles as the wall's ledge: two per cell, per layer.
        on_ledge = np.ptp(nodes[wall_faces][:, :, 2], axis=1) < 1e-9
        self.assertEqual(int(on_ledge.sum()), 2 * cells_across * (step["stations"] - 1))

    def test_a_step_below_the_rock_everywhere_has_no_ledge_and_no_fill(self):
        _, fill, _, _ = build(stepped_config([TWO_STEPS[0], {"top_z_m": -40.0, "radial_width_m": 5.0}]))
        self.assertEqual(fill.report["steps_without_a_ledge_z_m"], [-40.0])
        self.assertEqual([step["top_z_m"] for step in fill.report["steps"]], [-20.0])
        _, none, _, _ = build(stepped_config([{"top_z_m": -40.0, "radial_width_m": 5.0}]))
        self.assertIsNone(none)

    def test_without_the_mortar_block_there_is_no_fill(self):
        wall, fill, _, _ = build(stepped_config(TWO_STEPS, mortar=None))
        self.assertIsNone(fill)
        self.assertGreater(len(wall.cells), 0)


class ElmerModelTests(unittest.TestCase):
    """The fill in the prepared Elmer model (no solve): its own body, material and springs to the wall."""

    @classmethod
    def setUpClass(cls):
        from Elmer.three_body_diagnostic import prepare

        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        cls.reports, cls.models = {}, {}
        for name, mortar in (("plain", None), ("filled", MORTAR)):
            (root / name).mkdir()
            (root / name / "config.json").write_text(json.dumps(stepped_config(TWO_STEPS, mortar=mortar)), encoding="utf-8")
            cls.reports[name] = prepare(root / name, root / name / "model", joints="spring", integrated=True)
            cls.models[name] = root / name / "model"

    def mesh(self, name: str):
        """(points, tetrahedra by body, triangles by boundary tag) of the saved Gmsh file."""
        lines = (self.models[name] / "three_body.msh").read_text().splitlines()
        start = lines.index("$Nodes")
        count = int(lines[start + 1])
        points = np.array([[float(v) for v in line.split()[1:]] for line in lines[start + 2:start + 2 + count]])
        start = lines.index("$Elements")
        cells, faces = {}, {}
        for line in lines[start + 2:start + 2 + int(lines[start + 1])]:
            row = [int(v) for v in line.split()]
            (cells if row[1] == 4 else faces).setdefault(row[3], []).append([n - 1 for n in row[3 + row[2]:]])
        return points, {k: np.array(v) for k, v in cells.items()}, {k: np.array(v) for k, v in faces.items()}

    def test_the_fill_is_a_fourth_body_and_leaves_wall_and_foundation_as_they_were(self):
        plain, filled = self.reports["plain"], self.reports["filled"]
        self.assertEqual(plain["cells_by_body"].keys(), {1, 3})
        self.assertEqual(set(filled["cells_by_body"]), {1, 3, 4})
        self.assertEqual({body: filled["cells_by_body"][body] for body in (1, 3)}, plain["cells_by_body"])
        self.assertEqual(filled["cells_by_body"][4], filled["mortar_fill"]["tetrahedra"])
        self.assertEqual(filled["solver_body_ids"], {"wall": 1, "foundation": 2, "mortar_fill": 3})
        self.assertEqual(filled["source_body_ids"], {"wall": 1, "foundation": 3, "mortar_fill": 4})
        self.assertEqual(plain["solver_body_ids"], {"wall": 1, "foundation": 2})
        self.assertNotIn("mortar_fill", plain)
        # Wall and foundation cells and nodes are the plain model's, in the same order.
        plain_points, plain_cells, _ = self.mesh("plain")
        points, cells, _ = self.mesh("filled")
        np.testing.assert_array_equal(points[:len(plain_points)], plain_points)
        for body in (1, 3):
            np.testing.assert_array_equal(cells[body], plain_cells[body])
        self.assertEqual(len(points) - len(plain_points), filled["mortar_fill"]["nodes"])
        self.assertGreaterEqual(cells[4].min(), len(plain_points))  # the fill shares no node with the wall
        self.assertAlmostEqual(volumes(points, cells[4]).sum(), filled["mortar_fill"]["volume_m3"], places=6)
        self.assertGreater(volumes(points, cells[4]).min(), 0.0)

    def test_the_fill_is_joined_to_the_wall_by_its_bond_springs_on_matching_faces(self):
        filled = self.reports["filled"]
        self.assertEqual(filled["interface_stiffness_n_per_m3"]["wall_mortar"], MORTAR["bond_spring_n_per_m3"])
        self.assertEqual(filled["interface_stiffness_sources"]["wall_mortar"], "material.mortar.bond_spring_n_per_m3")
        self.assertNotIn("wall_mortar", self.reports["plain"]["interface_stiffness_n_per_m3"])
        pair = filled["spring_pairs"]["wall_mortar"]
        self.assertEqual(pair["legacy_tags"], [19, 20])
        points, _, faces = self.mesh("filled")
        boundary = filled["boundary_id_map"]
        located = lambda rows: sorted(tuple(sorted(tuple(p) for p in np.round(points[row], 6))) for row in rows)  # noqa: E731
        total = 0
        for part, (wall_tag, fill_tag) in pair["faces"].items():
            wall_side, fill_side = faces[boundary[wall_tag]], faces[boundary[fill_tag]]
            self.assertEqual(located(wall_side), located(fill_side), part)
            self.assertFalse(set(wall_side.ravel()) & set(fill_side.ravel()), part)
            total += len(wall_side)
        self.assertEqual(total, filled["mortar_fill"]["steps"][0]["tie_triangles"] + filled["mortar_fill"]["steps"][1]["tie_triangles"])
        # Every group is flat enough for Elmer's projector, and the two ledges are one group each.
        classes = [plane["orientation_class"] for plane in pair["planes"].values()]
        self.assertEqual(classes.count("tread"), 2)
        self.assertLessEqual(max(plane["maximum_off_plane_distance_m"] for plane in pair["planes"].values()), 0.005)
        # The plain model's wall/foundation groups are untouched.
        self.assertEqual(filled["spring_pairs"]["wall_foundation"], self.reports["plain"]["spring_pairs"]["wall_foundation"])

    def test_the_solver_input_gives_the_fill_its_own_material_and_weight(self):
        text = (self.models["filled"] / "dam_model.sif").read_text(encoding="utf-8")
        plain = (self.models["plain"] / "dam_model.sif").read_text(encoding="utf-8")
        self.assertIn('Body 3\n    Target Bodies(1) = 3\n    Name = "MortarFillBody"\n    Equation = 1\n'
                      '    Material = 2\n    Body Force = 2\nEnd\n', text)
        self.assertIn('Material 2\n    Name = "MortarFill"\n    Youngs Modulus = Real 350000000\n'
                      '    Poisson Ratio = Real 0.2\n    Density = Real 2200\nEnd\n', text)
        self.assertIn(f"Body Force 2\n    Stress Bodyforce 3 = Real {2200 * -9.81:.12g}\nEnd\n", text)
        self.assertNotIn("MortarFill", plain)
        self.assertNotIn("Material 2", plain)
        self.assertEqual(text.count('_spring"\n') - plain.count('_spring"\n'),
                         len(self.reports["filled"]["spring_pairs"]["wall_mortar"]["faces"]))
        self.assertIn("Mortar BC Coefficient Displacement 3 = Real 1250000000", text)
        # The concrete's own material and weight are unchanged.
        block = lambda source, name: source[source.index(name):source.index("End", source.index(name))]  # noqa: E731
        self.assertEqual(block(text, "Material 1\n"), block(plain, "Material 1\n"))
        self.assertEqual(block(text, "Body Force 1\n"), block(plain, "Body Force 1\n"))

    def test_the_sloped_face_takes_over_as_downstream_surface(self):
        points, _, faces = self.mesh("filled")
        plain_points, _, plain_faces = self.mesh("plain")
        boundary, plain_boundary = self.reports["filled"]["boundary_id_map"], self.reports["plain"]["boundary_id_map"]
        downstream, plain_downstream = faces[boundary[3]], plain_faces[plain_boundary[3]]
        fill_nodes = downstream >= len(plain_points)
        on_fill = fill_nodes.all(axis=1)
        self.assertFalse((fill_nodes.any(axis=1) & ~on_fill).any())
        fill = self.reports["filled"]["mortar_fill"]
        self.assertEqual(int(on_fill.sum()), sum(step["slope_triangles"] for step in fill["steps"]))
        covered = sum(step["tie_triangles"] for step in fill["steps"])
        self.assertEqual(len(downstream) - int(on_fill.sum()), len(plain_downstream) - covered)
        # No wall face under the fill is still loaded as a downstream face.
        radius = np.hypot(points[:, 0], points[:, 1])
        wall_faces = downstream[~on_fill]
        on_upper_ledge = (np.abs(points[wall_faces][:, :, 2] + 20.0) < 1e-6).all(axis=1) & (radius[wall_faces] < 76.0 + 1e-6).all(axis=1)
        self.assertFalse(on_upper_ledge.any())

    def test_the_report_says_what_the_fill_is_and_is_not(self):
        fill = self.reports["filled"]["mortar_fill"]
        self.assertEqual(fill["material"], {"density": 2200.0, "youngs_modulus": 3.5e8, "poisson_ratio": 0.2})
        self.assertEqual(fill["bond_spring_n_per_m3"], MORTAR["bond_spring_n_per_m3"])
        self.assertFalse(fill["engineering_validated"])
        self.assertIn("excluded from wall stress", fill["reporting"])
        self.assertAlmostEqual(fill["steps"][0]["built_slope_run_per_rise"], 0.1)
        self.assertTrue(any("mortar fill" in entry for entry in self.reports["filled"]["assumptions"]))
        self.assertFalse(any("mortar fill" in entry for entry in self.reports["plain"]["assumptions"]))
        self.assertEqual(self.reports["filled"]["stress_reporting_scope"], "wall")
        saved = json.loads((self.models["filled"] / "diagnostic.json").read_text())
        self.assertEqual(saved["mortar_fill"]["volume_m3"], fill["volume_m3"])


if __name__ == "__main__":
    unittest.main()
