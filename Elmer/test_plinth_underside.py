"""Exact piecewise-linear plinth seat geometry and unchanged upper cutbacks."""
from collections import Counter
from pathlib import Path
import unittest

import numpy as np

from Elmer.model_config import load_config
from Elmer.regions.combined_structure import build_combined_structure, build_wall
from Elmer.regions.plinth import bedrock_elevation, build_conforming_plinth, build_plinth
from Elmer.regions.tetrahedral import _CELL_FACES, tetrahedralize_structure


ROOT = Path(__file__).resolve().parents[1]


class PlinthUndersideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "config.json")
        cls.mesh = build_combined_structure(ROOT)
        cls.radius = cls.config["geometry"]["wall"]["centerline_radius_m"]
        cls.rows = cls.config["geometry"]["bedrock"]["contours"]

    def assert_linear_seat(self, nodes, faces):
        points = np.asarray(nodes)
        used = np.unique([n for face in faces for n in face])
        stations = self.radius * np.arctan2(points[used, 0], points[used, 1])
        expected = np.interp(stations, [r["chainage"] for r in self.rows],
                             [r["height"] for r in self.rows])
        np.testing.assert_allclose(points[used, 2], expected, atol=1e-7, rtol=0)
        self.assertGreater(np.count_nonzero(
            np.abs(points[used, 2] / .5 - np.round(points[used, 2] / .5)) > 1e-6,
        ), 100)
        self.assertAlmostEqual(np.linalg.norm(points[used, :2], axis=1).min(), 75., places=7)
        self.assertAlmostEqual(np.linalg.norm(points[used, :2], axis=1).max(), 81., places=7)
        inclined = 0
        for face in faces:
            p = points[list(face)]
            s = self.radius * np.arctan2(p[:, 0], p[:, 1])
            self.assertGreater(float(np.ptp(s)), 1e-8, "Vertical foundation riser")
            if np.ptp(p[:, 2]) > 1e-8:
                inclined += 1
        self.assertGreater(inclined, 100)

    def test_combined_seat_is_linear_full_width_without_vertical_risers(self):
        faces = [face for tag, face in self.mesh.boundaries if tag == 1]
        self.assert_linear_seat(self.mesh.nodes, faces)
        points = np.asarray(self.mesh.nodes)
        used = np.unique([n for (kind, cell), region in zip(self.mesh.cells, self.mesh.region_ids)
                          if region == 1 for n in cell])
        station = np.round(self.radius * np.arctan2(points[used, 0], points[used, 1]), 7)
        for s in np.unique(station):
            self.assertAlmostEqual(float(points[used[station == s], 2].min()),
                                   bedrock_elevation(self.rows, s), places=6)

    def test_conforming_and_standalone_previews_use_the_same_linear_profile(self):
        for mesh in (build_conforming_plinth(ROOT, [build_wall(ROOT)]), build_plinth(ROOT)):
            with self.subTest(cells=len(mesh.cells)):
                faces = [tuple(n - 1 for n in face) for tag, face in mesh.boundaries if tag == 1]
                self.assert_linear_seat(mesh.nodes, faces)

    def test_interpolation_is_exact_at_survey_knots_and_between_them(self):
        for left, right in zip(self.rows, self.rows[1:]):
            s = (left["chainage"] + right["chainage"]) / 2
            self.assertAlmostEqual(bedrock_elevation(self.rows, s),
                                   (left["height"] + right["height"]) / 2)
        for row in self.rows:
            self.assertAlmostEqual(bedrock_elevation(self.rows, row["chainage"]), row["height"])
        with self.assertRaisesRegex(ValueError, "outside"):
            bedrock_elevation(self.rows, -1)

    def test_positive_conforming_tetrahedra_and_unchanged_wall_cutbacks(self):
        mesh = tetrahedralize_structure(self.mesh)
        self.assertEqual(sum(region == 2 for region in mesh.region_ids), 501024)
        report = self.mesh.wall_plinth_cutback
        self.assertEqual(report["step_count"], 110)
        self.assertEqual(report["wall_wedges"], 880)
        self.assertAlmostEqual(report["transferred_volume_m3"], 54.99962332972414, places=7)
        self.assertAlmostEqual(report["removed_shoulder_volume_m3"], 27.499811664862094, places=7)
        faces = Counter(tuple(sorted(cell[i] for i in pattern))
                        for kind, cell in mesh.cells for pattern in _CELL_FACES[kind])
        self.assertEqual(max(faces.values()), 2)
        self.assertEqual({tuple(sorted(face)) for _, face in mesh.boundaries},
                         {face for face, count in faces.items() if count == 1})
        self.assert_linear_seat(mesh.nodes, [face for tag, face in mesh.boundaries if tag == 1])


if __name__ == "__main__":
    unittest.main()
