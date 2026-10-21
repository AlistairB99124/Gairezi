"""Check the canonical wall-on-bedrock geometry contract."""
import json
import math
from pathlib import Path
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Elmer.regions.bedrock_contours import bedrock_elevation, load_contours, wall_extent
from Elmer.regions.uniform_wall import DOWNSTREAM_WALL_RADIUS_M, UPSTREAM_RADIUS_M


class DamGeometryTests(unittest.TestCase):
    def test_configured_wall_radii_and_height(self):
        config = json.loads((ROOT / "config.json").read_text())
        wall = config["geometry"]["wall"]
        self.assertEqual(wall["height_m"], 29.0)
        self.assertEqual(wall["downstream_radius_m"], DOWNSTREAM_WALL_RADIUS_M)
        self.assertEqual(wall["upstream_radius_m"], UPSTREAM_RADIUS_M)
        self.assertEqual(wall["centerline_radius_m"], 78.0)
        self.assertEqual((DOWNSTREAM_WALL_RADIUS_M, UPSTREAM_RADIUS_M), (76.0, 80.0))

    def test_wall_extent_is_where_bedrock_reaches_the_crest(self):
        config = json.loads((ROOT / "config.json").read_text())
        rows = config["geometry"]["bedrock"]["contours"]
        self.assertEqual(wall_extent(rows), (6.0, 174.0))
        contours = load_contours(ROOT / "config.json")
        self.assertEqual(contours[0].chainage_m, 6.0)
        self.assertEqual(contours[-1].chainage_m, 174.0)
        self.assertEqual(contours[0].bedrock_z_m, 0.0)
        self.assertEqual(contours[-1].bedrock_z_m, 0.0)
        self.assertEqual(min(point.bedrock_z_m for point in contours), -29.0)

    def test_bedrock_elevation_is_piecewise_linear(self):
        rows = json.loads((ROOT / "config.json").read_text())["geometry"]["bedrock"]["contours"]
        self.assertAlmostEqual(bedrock_elevation(rows, 47.5), -14.62)
        with self.assertRaisesRegex(ValueError, "outside"):
            bedrock_elevation(rows, 181.0)

    def test_nonfinite_crest_raise_is_rejected(self):
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                load_contours(ROOT / "config.json", value)


if __name__ == "__main__":
    unittest.main()
