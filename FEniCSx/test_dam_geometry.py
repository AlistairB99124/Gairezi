"""Check the requested radial dimensions on the actual shared dam mesh."""
import ast
import json
import math
from pathlib import Path
import sys
import tempfile
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
        path = ROOT / "config.json"
        original_bytes = path.read_bytes()
        original = load_contours(path)
        rebased = load_model_contours(ROOT)
        config = json.loads((ROOT / "config.json").read_text())
        self.assertEqual(config["geometry"]["wall"]["height_above_plinth_m"], 29.0)
        for before, after in zip(original, rebased):
            self.assertEqual(after.chainage_m, before.chainage_m)
            self.assertAlmostEqual(after.bedrock_z_m, before.bedrock_z_m)
            self.assertAlmostEqual(after.plinth_z_m, before.plinth_z_m)
            self.assertAlmostEqual(
                after.plinth_z_m - after.bedrock_z_m,
                before.plinth_z_m - before.bedrock_z_m,
            )
        self.assertEqual(path.read_bytes(), original_bytes)

    def test_nonfinite_crest_raise_is_rejected(self):
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                load_contours(ROOT / "config.json", value)

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
            "plinth_path": ROOT / "config.json",
            "load_contours": load_contours,
            "rows": [], "profile_points": [],
            "radius": 78.0, "math": math, "dam_height": 29.0,
        }
        exec(compile(ast.Module(body=[profile_block], type_ignores=[]), str(path), "exec"), namespace)
        for profile, contour in zip(namespace["profile_points"], load_model_contours(ROOT)):
            self.assertEqual(profile["station"], contour.chainage_m)
            self.assertEqual(profile["base_z"], contour.plinth_z_m)
            self.assertEqual(profile["ground_z"], contour.bedrock_z_m)
            self.assertEqual(profile["crest_z"], 0.0)
            self.assertGreaterEqual(profile["crest_z"] - profile["base_z"], 0.0)

    def test_configuration_and_mesh_radii_agree(self):
        config = json.loads((ROOT / "config.json").read_text())
        wall = config["geometry"]["wall"]
        self.assertEqual(wall["radial_width_m"], 4.0)
        self.assertEqual(wall["downstream_radius_m"], DOWNSTREAM_WALL_RADIUS_M)
        self.assertEqual(wall["upstream_radius_m"], WALL_UPSTREAM_RADIUS_M)
        self.assertEqual(wall["centerline_radius_m"], 78.0)

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
        self.assertEqual(vertical_bounds["WALL"], (-28.0, 0.0))
        self.assertEqual(min(z for _, _, z in structure.nodes), -29.0)
        self.assertAlmostEqual(structure.loads.upstream_pressure_pa(-29.0), 313920)
        self.assertAlmostEqual(structure.loads.upstream_pressure_pa(-28.0), 304110)
        wall_min, wall_max = radial_bounds["WALL"]
        plinth_min, plinth_max = radial_bounds["PLINTH"]
        self.assertAlmostEqual(wall_min, DOWNSTREAM_WALL_RADIUS_M)
        self.assertAlmostEqual(wall_max, WALL_UPSTREAM_RADIUS_M)
        self.assertAlmostEqual(plinth_min, DOWNSTREAM_RADIUS_M)
        self.assertAlmostEqual(plinth_max, UPSTREAM_RADIUS_M)
        self.assertAlmostEqual(wall_max - wall_min, 4.0)
        self.assertAlmostEqual(plinth_max - plinth_min, 6.0)
        self.assertAlmostEqual(wall_min - plinth_min, 1.0)
        self.assertAlmostEqual(plinth_max - wall_max, 1.0)
        self.assertEqual(
            sum(tag == 8 for tag, _ in structure.boundaries),
            sum(tag == 9 for tag, _ in structure.boundaries),
        )

    def test_changed_config_controls_mesh_without_deleted_inputs(self):
        config = json.loads((ROOT / "config.json").read_text())
        config["geometry"]["global_element_size_m"] = 1.0
        config["geometry"]["wall"].update(
            radial_width_m=3.0, downstream_radius_m=77.0, centerline_radius_m=78.5
        )
        config["geometry"]["plinth"].update(
            radial_width_m=5.0, downstream_radius_m=76.0, centerline_radius_m=78.5
        )
        config["material"]["concrete"]["youngs_modulus"] = 30e9
        config["geometry"]["bedrock"]["support"]["spring_n_per_m3"]["x"] = 2e9
        config["geometry"]["plinth"]["support"]["spring_n_per_m3"]["x"] = 3e9
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config))
            structure = build_combined_structure(root)
            report = audit_combined_structure(structure)
        self.assertAlmostEqual(report["wall_geometry"]["uniform_thickness_m"], 3.0)
        self.assertAlmostEqual(report["wall_geometry"]["downstream_radius_m"], 77.0)
        self.assertEqual(structure.material.youngs_modulus_pa, 30e9)
        self.assertEqual(structure.foundation_support.spring_x_n_per_m3, 2e9)
        self.assertEqual(structure.plinth_support.spring_x_n_per_m3, 3e9)


if __name__ == "__main__":
    unittest.main()
