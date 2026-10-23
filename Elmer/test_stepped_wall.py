"""Downstream thickness steps (geometry.wall.thickness_steps) on the seated wall."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from Elmer import structure_assembly as assembly
from Elmer.model_config import (
    ConfigurationValidationError,
    tapered_wall_downstream_radius_at,
    validate_config,
    wall_downstream_radius_at,
    wall_joint_support,
    wall_width_at,
)
from Elmer.regions.model_groups import is_wall_joint_boundary_id, joint_boundary_ids
from Elmer.regions.uniform_wall import (
    DOWNSTREAM_BOUNDARY_ID,
    OTHER_BOUNDARY_ID,
    UPSTREAM_BOUNDARY_ID,
    build_bedrock_seated_wall,
    excavated_seat_level,
)

STEPS = [{"top_z_m": -10.0, "radial_width_m": 4.5}, {"top_z_m": -20.0, "radial_width_m": 5.0}]


def base_payload() -> dict:
    payload = json.loads(assembly.CONFIG_PATH.read_text(encoding="utf-8-sig"))
    payload["geometry"].pop("foundation", None)
    payload["geometry"]["wall"]["thickness_steps"] = copy.deepcopy(STEPS)
    return payload


def build_wall(payload: dict, split_joints: bool = False):
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "config.json").write_text(json.dumps(payload))
        return build_bedrock_seated_wall(Path(directory), split_joints=split_joints)


def build_structure(payload: dict, split_wall_joints: bool):
    from Elmer.regions.combined_structure import build_combined_structure
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "config.json").write_text(json.dumps(payload))
        return build_combined_structure(Path(directory), split_wall_joints=split_wall_joints)


class ThicknessStepHelperTests(unittest.TestCase):
    def test_width_and_downstream_radius_by_band(self):
        wall = base_payload()["geometry"]["wall"]
        for z, width in ((0.0, 4.0), (-10.0, 4.0), (-10.01, 4.5), (-20.0, 4.5), (-25.0, 5.0)):
            self.assertEqual(wall_width_at(wall, z), width)
            self.assertEqual(wall_downstream_radius_at(wall, z), 80.0 - width)

    def test_wall_without_steps_has_crest_width_everywhere(self):
        wall = base_payload()["geometry"]["wall"]
        del wall["thickness_steps"]
        self.assertEqual(wall_downstream_radius_at(wall, -28.0), 76.0)


class ThicknessStepValidationTests(unittest.TestCase):
    def issues(self, steps) -> set[str]:
        payload = base_payload()
        payload["geometry"]["wall"]["thickness_steps"] = steps
        with self.assertRaises(ConfigurationValidationError) as raised:
            validate_config(payload)
        return {issue.path for issue in raised.exception.issues}

    def test_valid_steps_pass(self):
        validate_config(base_payload())

    def test_rejects_steps_that_rise_or_thin_the_wall(self):
        self.assertIn("geometry.wall.thickness_steps[1].top_z_m", self.issues(
            [{"top_z_m": -10.0, "radial_width_m": 4.5}, {"top_z_m": -5.0, "radial_width_m": 5.0}]
        ))
        self.assertIn("geometry.wall.thickness_steps[0].radial_width_m", self.issues(
            [{"top_z_m": -10.0, "radial_width_m": 3.5}]
        ))
        self.assertIn("geometry.wall.thickness_steps[0].top_z_m", self.issues(
            [{"top_z_m": 0.0, "radial_width_m": 4.5}]
        ))

    def test_rejects_malformed_steps(self):
        self.assertIn("geometry.wall.thickness_steps", self.issues({"top_z_m": -10.0}))
        self.assertIn("geometry.wall.thickness_steps[0].radial_width_m", self.issues([{"top_z_m": -10.0}]))


class StepEndTaperTests(unittest.TestCase):
    def test_without_ratio_matches_full_step_widths(self):
        wall = base_payload()["geometry"]["wall"]
        for z, seat in ((-5.0, -6.0), (-15.0, -10.5), (-25.0, -28.0)):
            self.assertEqual(tapered_wall_downstream_radius_at(wall, z, seat), wall_downstream_radius_at(wall, z))

    def test_extra_width_is_limited_by_seat_depth_below_each_step(self):
        wall = base_payload()["geometry"]["wall"]
        wall["step_end_taper_ratio"] = 1.0
        self.assertAlmostEqual(tapered_wall_downstream_radius_at(wall, -5.0, -6.0), 76.0)
        self.assertAlmostEqual(tapered_wall_downstream_radius_at(wall, -10.2, -10.25), 75.75)
        self.assertAlmostEqual(tapered_wall_downstream_radius_at(wall, -12.0, -12.0), 75.5)
        self.assertAlmostEqual(tapered_wall_downstream_radius_at(wall, -20.2, -20.25), 75.25)
        self.assertAlmostEqual(tapered_wall_downstream_radius_at(wall, -25.0, -30.0), 75.0)

    def issues(self, payload) -> set[str]:
        with self.assertRaises(ConfigurationValidationError) as raised:
            validate_config(payload)
        return {issue.path for issue in raised.exception.issues}

    def test_validation_requires_positive_ratio_steps_and_foundation(self):
        source = json.loads(assembly.CONFIG_PATH.read_text(encoding="utf-8-sig"))
        if "foundation" not in source["geometry"]:
            self.skipTest("config.json has no geometry.foundation")
        payload = base_payload()
        payload["geometry"]["foundation"] = copy.deepcopy(source["geometry"]["foundation"])
        payload["geometry"]["wall"]["step_end_taper_ratio"] = 1.0
        validate_config(payload)
        payload["geometry"]["wall"]["step_end_taper_ratio"] = 0.0
        self.assertIn("geometry.wall.step_end_taper_ratio", self.issues(payload))
        without_foundation = base_payload()
        without_foundation["geometry"]["wall"]["step_end_taper_ratio"] = 1.0
        self.assertIn("geometry.wall.step_end_taper_ratio", self.issues(without_foundation))

    def test_tapered_wall_has_no_full_width_shelf_on_shallow_seats(self):
        source = json.loads(assembly.CONFIG_PATH.read_text(encoding="utf-8-sig"))
        if "foundation" not in source["geometry"]:
            self.skipTest("config.json has no geometry.foundation")
        payload = base_payload()
        payload["geometry"]["foundation"] = copy.deepcopy(source["geometry"]["foundation"])
        full = build_wall(payload)
        payload["geometry"]["wall"]["step_end_taper_ratio"] = 1.0
        tapered = build_wall(payload)
        wall = payload["geometry"]["wall"]
        self.assertLess(len(tapered.cells), len(full.cells))
        radius = float(wall["centerline_radius_m"])
        for _, cell in tapered.cells:
            points = [tapered.nodes[node - 1] for node in cell]
            stations = {
                min(range(len(tapered.chainages_m)), key=lambda i: abs(tapered.chainages_m[i] - chainage))
                for chainage in (radius * math.atan2(x, y) for x, y, _ in points)
            }
            seat = max(tapered.base_levels_m[index] for index in stations)
            # Every thickened strip sits at least ratio x its extra width above the column seat.
            inner = min(math.hypot(x, y) for x, y, _ in points)
            mid_z = sum(z for _, _, z in points) / len(points)
            self.assertGreaterEqual(inner, tapered_wall_downstream_radius_at(wall, mid_z, seat) - 1e-6)
        self.assertAlmostEqual(min(math.hypot(x, y) for x, y, _ in tapered.nodes), 75.0)


class JointSupportValidationTests(unittest.TestCase):
    def issues(self, joint_support) -> set[str]:
        payload = base_payload()
        payload["geometry"]["wall"]["joint_support"] = joint_support
        with self.assertRaises(ConfigurationValidationError) as raised:
            validate_config(payload)
        return {issue.path for issue in raised.exception.issues}

    def test_spring_and_bonded_joints_validate(self):
        payload = base_payload()
        payload["geometry"]["wall"]["joint_support"] = {"model": "bonded"}
        validate_config(payload)
        self.assertIsNone(wall_joint_support(payload["geometry"]["wall"]))
        del payload["geometry"]["wall"]["joint_support"]
        validate_config(payload)
        self.assertIsNone(wall_joint_support(payload["geometry"]["wall"]))
        payload["geometry"]["wall"]["joint_support"] = {
            "model": "spring", "spring_n_per_m3": {"x": 1e9, "y": 2e9, "z": 3e9},
        }
        validate_config(payload)
        self.assertEqual(wall_joint_support(payload["geometry"]["wall"]), {
            "levels_m": [-10.0, -20.0], "spring_n_per_m3": {"x": 1e9, "y": 2e9, "z": 3e9},
        })

    def test_rejects_unknown_model_and_non_positive_springs(self):
        self.assertIn("geometry.wall.joint_support.model", self.issues({"model": "glued"}))
        self.assertIn("geometry.wall.joint_support.spring_n_per_m3", self.issues({"model": "spring"}))
        self.assertIn("geometry.wall.joint_support.spring_n_per_m3.z", self.issues(
            {"model": "spring", "spring_n_per_m3": {"x": 1e9, "y": 1e9, "z": 0.0}}
        ))


class JointedWallMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = base_payload()
        cls.payload["geometry"]["wall"]["joint_support"] = {
            "model": "spring", "spring_n_per_m3": {"x": 1.25e9, "y": 1.25e9, "z": 3.0e9},
        }
        cls.monolithic = build_wall(cls.payload)
        cls.blocks = build_wall(cls.payload, split_joints=True)

    def test_monolithic_default_has_no_duplicates_or_joint_faces(self):
        coordinates = {tuple(round(v, 9) for v in point) for point in self.monolithic.nodes}
        self.assertEqual(len(coordinates), len(self.monolithic.nodes))
        self.assertEqual(set(self.monolithic.node_blocks), {0})
        self.assertFalse(any(is_wall_joint_boundary_id(tag) for tag, _ in self.monolithic.boundaries))
        self.assertEqual(self.monolithic.joint_levels_m, ())

    def test_blocks_keep_the_same_cells_and_split_only_at_step_levels(self):
        self.assertEqual(self.blocks.joint_levels_m, (-10.0, -20.0))
        self.assertEqual(len(self.blocks.cells), len(self.monolithic.cells))
        for (kind_a, cell_a), (kind_b, cell_b) in zip(self.monolithic.cells, self.blocks.cells, strict=True):
            self.assertEqual(kind_a, kind_b)
            self.assertEqual([self.monolithic.nodes[n - 1] for n in cell_a], [self.blocks.nodes[n - 1] for n in cell_b])
            blocks = {self.blocks.node_blocks[n - 1] for n in cell_b}
            self.assertEqual(len(blocks), 1)
            block = blocks.pop()
            zs = [self.blocks.nodes[n - 1][2] for n in cell_b]
            top, bottom = (0.0, -10.0, -20.0)[block], (-10.0, -20.0, -math.inf)[block]
            self.assertTrue(all(bottom - 1e-9 <= z <= top + 1e-9 for z in zs))
        copies: dict[tuple[float, ...], int] = {}
        for point in self.blocks.nodes:
            key = tuple(round(v, 9) for v in point)
            copies[key] = copies.get(key, 0) + 1
        duplicated = [key for key, count in copies.items() if count > 1]
        self.assertTrue(duplicated)
        self.assertEqual({key[2] for key in duplicated}, {-10.0, -20.0})
        self.assertEqual(max(copies.values()), 2)
        self.assertEqual(len(self.blocks.nodes) - len(self.monolithic.nodes), len(duplicated))

    def test_joint_sides_coincide_and_triangulate_identically(self):
        from Elmer.regions.tetrahedral import split_face
        nodes = self.blocks.nodes

        def located(face):
            return tuple(sorted(tuple(round(v, 7) for v in nodes[n - 1]) for n in face))

        for index, level in enumerate((-10.0, -20.0)):
            upper_tag, lower_tag = joint_boundary_ids(index)
            upper = [face for tag, face in self.blocks.boundaries if tag == upper_tag]
            lower = [face for tag, face in self.blocks.boundaries if tag == lower_tag]
            self.assertTrue(upper)
            self.assertEqual(sorted(map(located, upper)), sorted(map(located, lower)))
            self.assertEqual(
                sorted(located(tri) for face in upper for tri in split_face(face)),
                sorted(located(tri) for face in lower for tri in split_face(face)),
            )
            self.assertEqual({nodes[n - 1][2] for face in upper + lower for n in face}, {level})
            self.assertEqual({self.blocks.node_blocks[n - 1] for face in upper for n in face}, {index})
            self.assertEqual({self.blocks.node_blocks[n - 1] for face in lower for n in face}, {index + 1})

    def test_structure_audit_reports_one_component_per_block(self):
        from Elmer.regions.combined_structure import audit_combined_structure
        jointed = audit_combined_structure(build_structure(self.payload, True))
        self.assertEqual(jointed["wall_connection"], "spring_jointed_blocks")
        self.assertEqual(jointed["face_connected_components"], 3)
        self.assertEqual(jointed["wall_joint_levels_m"], [-10.0, -20.0])
        bonded_payload = copy.deepcopy(self.payload)
        bonded_payload["geometry"]["wall"]["joint_support"] = {"model": "bonded"}
        bonded = audit_combined_structure(build_structure(bonded_payload, True))
        self.assertEqual(bonded["wall_connection"], "single_monolithic_body")
        self.assertEqual(bonded["face_connected_components"], 1)


class SteppedWallMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = build_wall(base_payload())

    def test_downstream_face_steps_out_while_upstream_face_stays_fixed(self):
        inner = {"top": math.inf, "mid": math.inf, "low": math.inf}
        for x, y, z in self.mesh.nodes:
            band = "top" if z > -10 + 1e-6 else "mid" if z > -20 + 1e-6 else "low"
            inner[band] = min(inner[band], math.hypot(x, y))
        self.assertAlmostEqual(inner["top"], 76.0)
        self.assertAlmostEqual(inner["mid"], 75.5)
        self.assertAlmostEqual(inner["low"], 75.0)
        self.assertAlmostEqual(max(math.hypot(x, y) for x, y, _ in self.mesh.nodes), 80.0)
        for tag, face in self.mesh.boundaries:
            if tag == UPSTREAM_BOUNDARY_ID:
                for node in face:
                    self.assertAlmostEqual(math.hypot(*self.mesh.nodes[node - 1][:2]), 80.0)

    def test_every_node_is_used_and_treads_face_downstream(self):
        used = {node for _, cell in self.mesh.cells for node in cell}
        self.assertEqual(used, set(range(1, len(self.mesh.nodes) + 1)))
        self.assertFalse(any(tag == OTHER_BOUNDARY_ID for tag, _ in self.mesh.boundaries))
        treads = {-10.0: 0, -20.0: 0}
        for tag, face in self.mesh.boundaries:
            points = [self.mesh.nodes[node - 1] for node in face]
            levels = {round(z, 6) for _, _, z in points}
            if len(levels) == 1 and levels.pop() in treads:
                radii = [math.hypot(x, y) for x, y, _ in points]
                level = round(points[0][2], 6)
                if max(radii) <= 80.0 - wall_width_at(base_payload()["geometry"]["wall"], level) + 1e-6:
                    self.assertEqual(tag, DOWNSTREAM_BOUNDARY_ID)
                    treads[level] += 1
        self.assertGreater(treads[-10.0], 0)
        self.assertGreater(treads[-20.0], 0)

    def test_every_wall_node_lies_on_a_horizontal_course(self):
        # Step ledges must stay flat right up to the rock instead of sloping into it.
        for _, _, z in self.mesh.nodes:
            self.assertAlmostEqual(z / 0.5, round(z / 0.5), places=6)
        # Cells in a thickened strip never rise above that strip's step level.
        for _, cell in self.mesh.cells:
            points = [self.mesh.nodes[node - 1] for node in cell]
            inner = min(math.hypot(x, y) for x, y, _ in points)
            top = max(z for _, _, z in points)
            if inner < 75.5 - 1e-6:
                self.assertLessEqual(top, -20.0 + 1e-6)
            elif inner < 76.0 - 1e-6:
                self.assertLessEqual(top, -10.0 + 1e-6)


class ExcavatedSeatTests(unittest.TestCase):
    def test_seat_is_the_highest_course_at_or_below_the_rock(self):
        self.assertEqual(excavated_seat_level(-20.1, 0.5), -20.5)
        self.assertEqual(excavated_seat_level(-20.0, 0.5), -20.0)
        self.assertEqual(excavated_seat_level(-19.99, 0.5), -20.0)
        self.assertEqual(excavated_seat_level(-10.3, 0.5, (-10.2,)), -10.5)
        self.assertEqual(excavated_seat_level(-10.1, 0.5, (-10.2,)), -10.2)
        self.assertEqual(excavated_seat_level(3.0, 0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
