"""Small real-Gmsh checks of locked boundaries, grading and volume conformity."""
from pathlib import Path
import unittest
from unittest.mock import patch

import gmsh
import numpy as np

from Elmer.graded_bedrock import audit_tetrahedral_boundary, build_graded_rock
from Elmer.model_config import ConfigurationValidationError, load_config, validate_config


class GradedBedrockTests(unittest.TestCase):
    def test_planar_bottom_has_no_artificial_fan_seams(self):
        config = load_config(Path(__file__).resolve().parents[1] / "config.json")
        top = np.array([[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.]])
        bottom = top.copy()
        bottom[:, 2] = -10.
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        with (
            patch("Elmer.graded_bedrock.seated_bedrock_geometry", return_value=(
                np.vstack((bottom, top)), np.column_stack((triangles, triangles + 4)),
                {"seat_triangles": 2, "rock_volume_m3": 40.},
            )),
            patch("gmsh.model.geo.addCurveLoop", wraps=gmsh.model.geo.addCurveLoop) as loops,
            patch("gmsh.model.geo.addPoint", wraps=gmsh.model.geo.addPoint) as points,
            patch("gmsh.model.mesh.generate", side_effect=RuntimeError("stop before meshing")),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop before meshing"):
                build_graded_rock(config, top, [])
        # Two top triangles, one bottom polygon, eight side triangles; no centre point.
        self.assertEqual([len(call.args[0]) for call in loops.call_args_list],
                         [3, 3, 4] + [3] * 8)
        self.assertEqual(points.call_count, 8)
        self.assertFalse(gmsh.isInitialized())

    def test_actual_mesher_preserves_seat_and_grades_in_three_dimensions(self):
        config = load_config(Path(__file__).resolve().parents[1] / "config.json")
        config["geometry"]["bedrock"]["mesh"] = dict(
            near_size_m=.5, far_size_m=3., near_distance_m=1., far_distance_m=5.,
        )
        # 20 x 20 x 10 m box with an embedded, mesh-locked 2 x 2 m seat.
        top = np.array([
            [9., 9., 0.], [11., 9., 0.], [11., 11., 0.], [9., 11., 0.],
            [0., 0., 0.], [20., 0., 0.], [20., 20., 0.], [0., 20., 0.],
        ])
        triangles = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 1], [4, 1, 0], [5, 6, 2], [5, 2, 1],
            [6, 7, 3], [6, 3, 2], [7, 4, 0], [7, 0, 3],
        ])
        bottom = top.copy()
        bottom[:, 2] = -10.
        columns = np.vstack((bottom, top))
        with patch("Elmer.graded_bedrock.seated_bedrock_geometry", return_value=(
            columns, np.column_stack((triangles, triangles + 8)),
            {"seat_triangles": 2, "rock_volume_m3": 4000.},
        )):
            result = build_graded_rock(config, top, [])
            with patch("gmsh.model.mesh.generate", side_effect=RuntimeError("meshing failed")):
                with self.assertRaisesRegex(RuntimeError, "meshing failed"):
                    build_graded_rock(config, top, [])
            with patch("gmsh.model.mesh.getElementQualities", return_value=np.array([1e-16])):
                with self.assertRaisesRegex(ValueError, "numerically degenerate"):
                    build_graded_rock(config, top, [])
        self.assertEqual(result.report["seat_faces_verified"], 2)
        self.assertFalse(gmsh.isInitialized())
        self.assertGreater(result.report["minimum_cell_volume_m3"], 0.)
        samples = result.report["spacing_samples"]
        self.assertGreater(samples["near"]["sampled_cells"], 0)
        self.assertGreater(samples["far"]["sampled_cells"], 0)
        self.assertGreater(samples["far"]["median_edge_m"], 2.)
        self.assertLess(samples["near"]["median_edge_m"], 1.5)
        self.assertGreater(samples["far"]["median_edge_m"],
                           2 * samples["near"]["median_edge_m"])

    def test_audit_rejects_missing_and_duplicated_faces(self):
        tetra = np.array([[0, 1, 2, 3]])
        shell = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        audit_tetrahedral_boundary(tetra, shell)
        with self.assertRaisesRegex(ValueError, "unpaired"):
            audit_tetrahedral_boundary(tetra, shell[:-1])
        with self.assertRaisesRegex(ValueError, "non-manifold"):
            audit_tetrahedral_boundary(np.vstack((tetra, tetra, tetra)), shell)

    def test_mesh_policy_rejects_invalid_values(self):
        for key, value in (
            ("near_size_m", 0), ("far_size_m", .1),
            ("near_distance_m", -1), ("far_distance_m", 2),
            ("near_size_m", float("nan")), ("far_size_m", True),
        ):
            with self.subTest(key=key, value=value):
                config = load_config(Path(__file__).resolve().parents[1] / "config.json")
                config["geometry"]["bedrock"]["mesh"][key] = value
                with self.assertRaisesRegex(ConfigurationValidationError, f"mesh.{key}"):
                    validate_config(config)

    def test_mesh_policy_is_optional_for_existing_configurations(self):
        config = load_config(Path(__file__).resolve().parents[1] / "config.json")
        config["geometry"].pop("foundation", None)
        del config["geometry"]["bedrock"]["mesh"]
        validate_config(config)


if __name__ == "__main__":
    unittest.main()
