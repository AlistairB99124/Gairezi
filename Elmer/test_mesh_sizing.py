import unittest

from Elmer.mesh_sizing import (
    TARGET_WALL_TETRAHEDRAL_ELEMENTS,
    config_for_wall_element_size,
    course_compatible_size,
    size_for_element_target,
    tetrahedral_element_count,
)


class MeshSizingTests(unittest.TestCase):
    def test_element_size_estimate_scales_by_cube_root_of_count_ratio(self):
        size = size_for_element_target(0.5, 606414, TARGET_WALL_TETRAHEDRAL_ELEMENTS)
        self.assertAlmostEqual(size, 0.7965, places=3)

    def test_step_grid_snapping_chooses_coarser_compatible_size(self):
        size = course_compatible_size(
            0.796, (-20.0, -25.0), not_coarser_than=False,
        )
        self.assertAlmostEqual(size, 5 / 6)
        self.assertAlmostEqual(-20.0 / size, round(-20.0 / size))
        self.assertAlmostEqual(-25.0 / size, round(-25.0 / size))

    def test_tetrahedral_count_matches_mixed_cell_subdivision(self):
        cells = [(5, tuple(range(8))), (6, tuple(range(6))), (4, tuple(range(4)))]
        self.assertEqual(tetrahedral_element_count(cells), 10)

    def test_attached_body_settings_follow_effective_wall_size_without_mutating_config(self):
        source = {
            "geometry": {
                "global_element_size_m": 0.5,
                "bedrock": {"mesh": {"near_size_m": 0.5, "far_size_m": 15.0}},
            },
        }
        adjusted = config_for_wall_element_size(source, 0.8333333333333334)
        self.assertEqual(source["geometry"]["global_element_size_m"], 0.5)
        self.assertEqual(source["geometry"]["bedrock"]["mesh"]["near_size_m"], 0.5)
        self.assertAlmostEqual(adjusted["geometry"]["global_element_size_m"], 5 / 6)
        self.assertAlmostEqual(adjusted["geometry"]["bedrock"]["mesh"]["near_size_m"], 5 / 6)
        self.assertEqual(adjusted["geometry"]["bedrock"]["mesh"]["far_size_m"], 15.0)
        adjusted = config_for_wall_element_size({
            "geometry": {
                "global_element_size_m": 0.5,
                "bedrock": {"mesh": {"near_size_m": 0.5, "far_size_m": 0.6}},
            },
        }, 0.8333333333333334)
        self.assertAlmostEqual(adjusted["geometry"]["bedrock"]["mesh"]["far_size_m"], 5 / 6)

    def test_invalid_element_sizes_and_counts_are_rejected(self):
        for args in ((0.0, 10, 5), (0.5, 0, 5), (0.5, 10, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                size_for_element_target(*args)
        with self.assertRaisesRegex(ValueError, "Unsupported source"):
            tetrahedral_element_count([(7, tuple(range(5)))])


if __name__ == "__main__":
    unittest.main()
