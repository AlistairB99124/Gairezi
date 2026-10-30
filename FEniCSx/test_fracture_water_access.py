"""Water access does not pressurise dry/isolated cracks or cross closed edges."""
import unittest

import numpy as np

if __package__:
    from .fracture_faces import PairedCrackFaces
    from .fracture_topology import split_interior_wall_faces
    from .fracture_water_access import water_access
else:
    from fracture_faces import PairedCrackFaces
    from fracture_topology import split_interior_wall_faces
    from fracture_water_access import water_access


class FractureWaterAccessTests(unittest.TestCase):
    def fixture(self):
        points = np.array([(0., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                           (0., 1., 1.), (0., 0., 2.), (0., 1., 2.)])
        triangles = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]])
        faces = PairedCrackFaces(np.vstack((points, points)), triangles, triangles + 6, [[1., 0., 0.]] * 4)
        state = np.zeros((12, 3))
        state[6:, 0] = .001
        return faces, state, np.tile(np.arange(6), 2)

    def access(self, faces, state, parents, inlet=((0, 1),), tolerance=1e-10):
        return water_access(faces, state, inlet, parents, closure_tolerance_m=tolerance)

    def test_explicit_inlet_reaches_connected_open_faces_without_assigning_water(self):
        faces, state, parents = self.fixture()
        report = self.access(faces, state, parents)
        self.assertEqual(report["reachable_open_face_ids"], [0, 1, 2, 3])
        self.assertEqual(report["isolated_open_face_ids"], [])
        self.assertFalse(report["wetting_solved"])
        self.assertFalse(report["pressure_assigned"])
        self.assertFalse(report["solver_ready"])

    def test_closed_shared_edge_blocks_flow_even_when_both_faces_have_open_area(self):
        faces, state, parents = self.fixture()
        state[[8, 9], 0] = 0.  # shared source edge (2,3) is closed
        report = self.access(faces, state, parents)
        self.assertEqual(report["reachable_open_face_ids"], [0, 1])
        self.assertEqual(report["isolated_open_face_ids"], [2, 3])
        self.assertEqual(report["closed_face_ids"], [])
        state[8, 0] = .001  # an actual open segment, not a minimum-aperture rule
        self.assertEqual(self.access(faces, state, parents)["reachable_open_face_ids"], [0, 1, 2, 3])

    def test_closed_inlet_and_reclosed_surface_do_not_inherit_pressure(self):
        faces, state, parents = self.fixture()
        state[[6, 7], 0] = 0.
        report = self.access(faces, state, parents)
        self.assertEqual(report["reachable_open_face_ids"], [])
        self.assertEqual(report["blocked_inlet_source_edges"], [[0, 1]])
        state[:] = 0.
        report = self.access(faces, state, parents)
        self.assertEqual(report["closed_face_ids"], [0, 1, 2, 3])
        self.assertEqual(report["reachable_open_face_ids"], [])
        self.assertIn("trapped water", report["limitations"])

    def test_point_touch_and_coincident_unrelated_sheet_do_not_connect(self):
        faces, state, parents = self.fixture()
        # Add an entirely coincident sheet with different original identities.
        count = len(faces.coordinates)
        other = PairedCrackFaces(
            np.vstack((faces.coordinates, faces.coordinates)),
            np.vstack((faces.minus, faces.minus + count)),
            np.vstack((faces.plus, faces.plus + count)), [[1., 0., 0.]] * 8,
        )
        report = self.access(other, np.vstack((state, state)), np.r_[parents, parents + 6])
        self.assertEqual(report["reachable_open_face_ids"], [0, 1, 2, 3])
        self.assertEqual(report["isolated_open_face_ids"], [4, 5, 6, 7])
        coordinates = np.array([(0., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                                (0., -1., 0.), (0., 0., -1.)] * 2)
        touching = PairedCrackFaces(coordinates, [[0, 1, 2], [0, 3, 4]],
                                    [[5, 6, 7], [5, 8, 9]], [[1., 0., 0.]] * 2)
        shift = np.zeros((10, 3))
        shift[5:, 0] = .001
        report = self.access(touching, shift, np.tile(np.arange(5), 2))
        self.assertEqual(report["reachable_open_face_ids"], [0])
        self.assertEqual(report["isolated_open_face_ids"], [1])

    def test_owner_reversal_keeps_edge_adjacency(self):
        faces, state, parents = self.fixture()
        minus, plus = faces.minus.copy(), faces.plus.copy()
        minus[1], plus[1] = plus[1].copy(), minus[1].copy()
        normals = faces.normals.copy()
        normals[1] *= -1
        reversed_faces = PairedCrackFaces(faces.coordinates, minus, plus, normals)
        self.assertEqual(self.access(reversed_faces, state, parents)["reachable_open_face_ids"], [0, 1, 2, 3])

    def test_explicit_cut_to_closed_then_open_hydraulic_access(self):
        points = np.array([(0., 0., -1.), (0., 0., 1.), (1., 0., 0.),
                           (-.5, .866, 0.), (-.5, -.866, 0.)])
        cells = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]]
        cut = split_interior_wall_faces(points, cells, [1] * 3, [[0, 1, 2]])
        accepted = .00001 * points
        state = cut.transfer_nodal_values(accepted)
        closed = self.access(cut.faces, state, cut.parent_node_ids, inlet=[[1, 2]])
        self.assertEqual(closed["reachable_open_face_ids"], [])
        self.assertEqual(closed["blocked_inlet_source_edges"], [[1, 2]])
        state[cut.faces.plus[0, 2]] += cut.faces.normals[0] * .001
        open_report = self.access(cut.faces, state, cut.parent_node_ids, inlet=[[1, 2]])
        self.assertEqual(open_report["reachable_open_face_ids"], [0])
        self.assertFalse(open_report["pressure_assigned"])
        # A painted crack-front marker is not automatically a hydraulic inlet:
        # the tied edge remains closed even though the face has opened.
        front = self.access(cut.faces, state, cut.parent_node_ids, inlet=[[0, 1]])
        self.assertEqual(front["reachable_open_face_ids"], [])
        self.assertEqual(front["blocked_inlet_source_edges"], [[0, 1]])

    def test_multiple_explicit_inlets_and_ambiguous_sheet_provenance(self):
        faces, state, parents = self.fixture()
        state[[8, 9], 0] = 0.
        report = self.access(faces, state, parents, inlet=[[0, 1], [4, 5]])
        self.assertEqual(report["reachable_open_face_ids"], [0, 1, 2, 3])
        duplicate = PairedCrackFaces(
            np.vstack((faces.coordinates, faces.coordinates)),
            np.vstack((faces.minus, faces.minus + 12)),
            np.vstack((faces.plus, faces.plus + 12)), [[1., 0., 0.]] * 8,
        )
        with self.assertRaisesRegex(ValueError, "multiple split sheets"):
            self.access(duplicate, np.vstack((state, state)), np.r_[parents, parents])

    def test_invalid_inlet_provenance_penetration_and_tolerance_fail(self):
        faces, state, parents = self.fixture()
        for inlet in ([], [[0, 8]], [[1, 2]], [[0, 0]], [[0, 1], [1, 0]], [[True, False]]):
            with self.subTest(inlet=inlet), self.assertRaises(ValueError):
                self.access(faces, state, parents, inlet=inlet)
        for tolerance in (-1., True, float("nan")):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                self.access(faces, state, parents, tolerance=tolerance)
        with self.assertRaisesRegex(ValueError, "parent"):
            self.access(faces, state, np.arange(12))
        state[6, 0] = -.001
        with self.assertRaisesRegex(ValueError, "contact solution"):
            self.access(faces, state, parents)


if __name__ == "__main__":
    unittest.main()
