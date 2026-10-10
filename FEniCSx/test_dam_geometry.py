"""Check the requested radial dimensions on the actual shared dam mesh."""
import ast
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Elmer.regions.combined_structure import build_combined_structure, audit_combined_structure
from Elmer.regions.model_groups import REGION_GROUP_IDS
from Elmer.regions.plinth import DOWNSTREAM_RADIUS_M, UPSTREAM_RADIUS_M
from Elmer.regions.plinth import load_contours, load_model_contours
from Elmer.regions.uniform_wall import DOWNSTREAM_WALL_RADIUS_M, UPSTREAM_RADIUS_M as WALL_UPSTREAM_RADIUS_M


class DamGeometryTests(unittest.TestCase):
    def test_raised_crest_rebases_contours_without_changing_survey(self):
        path = ROOT / "Data" / "plinth.json"
        original_bytes = path.read_bytes()
        original = load_contours(path)
        rebased = load_model_contours(ROOT)
        config = json.loads((ROOT / "config.json").read_text())
        self.assertEqual(config["wall_height_above_plinth_m"], 31.0)
        self.assertEqual(config["crest_raise_m"], 2.0)
        for before, after in zip(original, rebased):
            self.assertEqual(after.chainage_m, before.chainage_m)
            self.assertAlmostEqual(after.bedrock_z_m, before.bedrock_z_m - 2.0)
            self.assertAlmostEqual(after.plinth_z_m, before.plinth_z_m - 2.0)
            self.assertAlmostEqual(
                after.plinth_z_m - after.bedrock_z_m,
                before.plinth_z_m - before.bedrock_z_m,
            )
        self.assertEqual(path.read_bytes(), original_bytes)

    def test_nonfinite_crest_raise_is_rejected(self):
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                load_contours(ROOT / "Data" / "plinth.json", value)

    def test_standalone_generator_uses_same_signed_rebased_profile(self):
        path = ROOT / "Elmer" / "build_curved_dam_geometry.py"
        tree = ast.parse(path.read_text())
        profile_block = next(
            node for node in tree.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "geometry_mode == 'v2'"
        )
        namespace = {
            "geometry_mode": "v2",
            "config": json.loads((ROOT / "config.json").read_text()),
            "plinth_path": ROOT / "Data" / "plinth.json",
            "load_contours": load_contours,
            "rows": [], "profile_points": [],
            "radius": 77.5, "math": math, "dam_height": 30.0,
        }
        exec(compile(ast.Module(body=[profile_block], type_ignores=[]), str(path), "exec"), namespace)
        for profile, contour in zip(namespace["profile_points"], load_model_contours(ROOT)):
            self.assertEqual(profile["station"], contour.chainage_m)
            self.assertEqual(profile["base_z"], contour.plinth_z_m)
            self.assertEqual(profile["ground_z"], contour.bedrock_z_m)
            self.assertEqual(profile["crest_z"], 0.0)
            self.assertGreater(profile["crest_z"] - profile["base_z"], 0.0)

    def test_configuration_and_crack_locations_agree(self):
        config = json.loads((ROOT / "config.json").read_text())
        geometry = json.loads((ROOT / "Elmer" / "load_cases.json").read_text())["geometry"]
        self.assertEqual(config["wall_thickness_m"], 5.0)
        self.assertEqual(geometry["thickness"], 5.0)
        self.assertEqual(config["wall_downstream_radius_m"], DOWNSTREAM_WALL_RADIUS_M)
        self.assertEqual(config["wall_upstream_radius_m"], WALL_UPSTREAM_RADIUS_M)
        self.assertEqual(config["wall_centerline_radius_m"], 77.5)
        self.assertEqual(config["wedge_anchor_radius_m"], DOWNSTREAM_WALL_RADIUS_M)
        for name in (
            "01_xfem_feasibility", "02_crack_stress_intensity",
            "03_horizontal_crack", "03_vertical_crack", "03_control_crack",
        ):
            with self.subTest(case=name):
                crack = json.loads((ROOT / "Aster" / "cases" / f"{name}.json").read_text())
                self.assertEqual(crack["centerline_radius_m"], config["wall_centerline_radius_m"])
                self.assertEqual(crack["upstream_radius_m"], WALL_UPSTREAM_RADIUS_M)

    def test_generated_wall_and_plinth_have_one_metre_overhangs(self):
        structure = build_combined_structure(ROOT)
        audit_combined_structure(structure)
        radial_bounds = {}
        vertical_bounds = {}
        for name in ("WALL", "PLINTH"):
            node_ids = {
                node_id
                for (_, cell), region in zip(structure.cells, structure.region_ids)
                if region == REGION_GROUP_IDS[name]
                for node_id in cell
            }
            radii = [math.hypot(*structure.nodes[node_id][:2]) for node_id in node_ids]
            radial_bounds[name] = min(radii), max(radii)
            elevations = [structure.nodes[node_id][2] for node_id in node_ids]
            vertical_bounds[name] = min(elevations), max(elevations)
        self.assertEqual(vertical_bounds["WALL"], (-30.0, 0.0))
        self.assertEqual(min(z for _, _, z in structure.nodes), -31.0)
        self.assertAlmostEqual(structure.loads.upstream_pressure_pa(-31.0), 333540)
        self.assertAlmostEqual(structure.loads.upstream_pressure_pa(-30.0), 323730)
        wall_min, wall_max = radial_bounds["WALL"]
        plinth_min, plinth_max = radial_bounds["PLINTH"]
        self.assertAlmostEqual(wall_min, DOWNSTREAM_WALL_RADIUS_M)
        self.assertAlmostEqual(wall_max, WALL_UPSTREAM_RADIUS_M)
        self.assertAlmostEqual(plinth_min, DOWNSTREAM_RADIUS_M)
        self.assertAlmostEqual(plinth_max, UPSTREAM_RADIUS_M)
        self.assertAlmostEqual(wall_max - wall_min, 5.0)
        self.assertAlmostEqual(plinth_max - plinth_min, 7.0)
        self.assertAlmostEqual(wall_min - plinth_min, 1.0)
        self.assertAlmostEqual(plinth_max - wall_max, 1.0)
        self.assertEqual(
            sum(tag == 8 for tag, _ in structure.boundaries),
            sum(tag == 9 for tag, _ in structure.boundaries),
        )


if __name__ == "__main__":
    unittest.main()
