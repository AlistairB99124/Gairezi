"""Native HEXA8 wall and LEGO foundation regression tests."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Elmer.foundation_trench import build_foundation
from Elmer.graded_bedrock import (
    positive_tetrahedron_orientation,
    projected_boundary_loop,
    separate_coincident_shell_edges,
)
from Elmer.model_config import load_config
from Elmer.regions.bedrock_contours import bedrock_elevation
from Elmer.regions.combined_structure import build_combined_structure
from Elmer.regions.model_groups import FACE_GROUP_IDS
from Elmer.three_body_diagnostic import config_for_wall_element_size
from Elmer.three_body_diagnostic import prepare


class NativeHexaWallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        config = load_config(Path(__file__).resolve().parents[1] / "config.json")
        config["geometry"]["target_wall_elements"] = 1000
        (cls.root / "config.json").write_text(json.dumps(config))
        cls.config = config
        cls.mesh = build_combined_structure(cls.root, split_wall_joints=False)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_wall_is_all_hexa8_and_under_every_surveyed_contour_interval(self):
        self.assertTrue(all(kind == 5 for kind, _ in self.mesh.cells))
        points = np.asarray(self.mesh.nodes)
        contours = self.config["geometry"]["bedrock"]["contours"]
        for boundary, face in self.mesh.boundaries:
            if boundary != FACE_GROUP_IDS["FOUNDATION"] or len(face) != 4:
                continue
            xyz = points[list(face)]
            centre_chainage = self.config["geometry"]["wall"]["centerline_radius_m"] * np.mean(
                np.arctan2(xyz[:, 0], xyz[:, 1]))
            survey = bedrock_elevation(contours, float(centre_chainage))
            self.assertLessEqual(float(xyz[:, 2].max()), survey + 1e-7)

    def test_foundation_is_built_from_the_same_wall_seat_blocks(self):
        foundation = build_foundation(
            config_for_wall_element_size(self.config, self.mesh.element_size_target_m),
            self.mesh,
        )
        self.assertGreater(len(foundation.hexahedra), 0)
        self.assertGreater(len(foundation.wall_faces), 0)
        seat = {
            tuple(np.round(point, 7))
            for face in foundation.wall_faces
            for point in foundation.points[list(face)]
        }
        wall = {
            tuple(np.round(self.mesh.nodes[node], 7))
            for node in range(len(self.mesh.nodes))
        }
        self.assertTrue(seat <= wall)

    def test_lego_foundation_never_rises_above_the_surveyed_contour(self):
        foundation = build_foundation(
            config_for_wall_element_size(self.config, self.mesh.element_size_target_m),
            self.mesh,
        )
        points = np.asarray(foundation.points)
        radius = float(self.config["geometry"]["wall"]["centerline_radius_m"])
        contours = self.config["geometry"]["bedrock"]["contours"]
        for face in foundation.terrain_faces:
            for node_id in np.unique(face):
                x, y, z = points[node_id]
                chainage = radius * np.arctan2(x, y)
                self.assertLessEqual(
                    z,
                    bedrock_elevation(contours, float(chainage)) + 1e-7,
                )

    def test_lego_roof_has_one_closed_projected_footprint(self):
        foundation = build_foundation(
            config_for_wall_element_size(self.config, self.mesh.element_size_target_m),
            self.mesh,
        )
        loop = projected_boundary_loop(
            foundation.points,
            foundation.terrain_faces,
            "LEGO test footprint",
        )
        self.assertGreater(len(loop), 4)
        self.assertEqual(len(loop), len(set(loop)))

    def test_projected_footprint_ignores_zero_area_riser_faces(self):
        points = np.asarray([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        faces = np.asarray([
            [0, 1, 2],
            [0, 2, 3],
            [4, 1, 0],
        ])
        self.assertEqual(
            len(projected_boundary_loop(points, faces, "synthetic footprint")),
            4,
        )

    def test_full_assembly_has_conforming_hexa8_interfaces_without_hanging_faces(self):
        output = self.root / "full-conforming"
        report = prepare(
            self.root, output, joints="spring", integrated=True,
        )
        audit = report["conforming_interface_audit"]
        self.assertEqual(audit["hanging_faces"], 0)
        self.assertGreater(audit["wall_foundation_faces"], 0)
        self.assertGreater(audit["foundation_bedrock_faces"], 0)
        self.assertEqual(audit["solver_cell_types"], {
            "wall": "HEXA8", "foundation": "TETRA4", "bedrock": "TETRA4",
        })
        self.assertEqual(report["foundation"]["solver_cell_type"], "tetrahedron")
        self.assertEqual(report["geometry"]["solver_cell_type"], "tetrahedron")
        self.assertGreater(report["wall_rock_seat_triangles"], 0)
        self.assertTrue((output / "three_body.msh").is_file())

    def test_coincident_inner_and_outer_shells_are_separated(self):
        points = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        faces = [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
        ]
        separated, updated, count = separate_coincident_shell_edges(
            points,
            faces,
            4,
        )
        edge_counts = {}
        for face in updated:
            for a, b in zip(face, face[1:] + face[:1]):
                edge = tuple(sorted((a, b)))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        self.assertEqual(count, 4)
        self.assertTrue(all(value == 2 for value in edge_counts.values()))
        self.assertTrue(np.allclose(
            separated[4:, 2] - points[:, 2],
            1e-2,
        ))

    def test_gmsh_tetrahedron_winding_is_normalized(self):
        points = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        oriented, count, valid = positive_tetrahedron_orientation(
            points,
            np.asarray([[0, 2, 1, 3]]),
        )
        p = points[oriented[0]]
        volume = np.dot(
            np.cross(p[1] - p[0], p[2] - p[0]),
            p[3] - p[0],
        ) / 6.
        self.assertEqual(count, 1)
        self.assertTrue(np.all(valid))
        self.assertGreater(volume, 0)

    def test_exact_zero_volume_tetrahedra_are_removed(self):
        points = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        oriented, _, valid = positive_tetrahedron_orientation(
            points,
            np.asarray([
                [0, 1, 2, 3],
                [0, 1, 2, 0],
            ]),
        )
        self.assertEqual(valid.tolist(), [False, False])
        self.assertEqual(len(oriented), 0)


if __name__ == "__main__":
    unittest.main()
