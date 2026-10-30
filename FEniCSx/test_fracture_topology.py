"""Explicit crack cuts preserve native topology/provenance and accepted states."""
import unittest

import numpy as np

if __package__:
    from .fracture_topology import split_interior_wall_faces, interior_wall_candidates
else:
    from fracture_topology import split_interior_wall_faces, interior_wall_candidates


class FractureTopologyTests(unittest.TestCase):
    def fixture(self):
        points = np.array([(0., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                           (-1., 0., 0.), (1., 0., 0.)])
        cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
        return points, cells

    def test_explicit_cut_keeps_volume_regions_rows_and_creates_zero_gap(self):
        points, cells = self.fixture()
        original_points, original_cells = points.copy(), cells.copy()
        result = split_interior_wall_faces(points, cells, [1, 1], [[0, 1, 2]])
        self.assertEqual(len(result.coordinates_m), 8)
        np.testing.assert_array_equal(result.parent_cell_ids, [0, 1])
        np.testing.assert_array_equal(result.region_ids, [1, 1])
        np.testing.assert_array_equal(result.parent_node_ids[result.cells], cells)
        np.testing.assert_array_equal(result.coordinates_m[result.cells], points[cells])
        np.testing.assert_array_equal(result.faces.normals, [[1., 0., 0.]])
        state = np.arange(15).reshape(5, 3) * .001
        transferred = result.transfer_nodal_values(state)
        self.assertEqual(result.faces.measure(transferred)["maximum_opening_m"], 0.)
        self.assertEqual(result.faces.measure(transferred)["minimum_signed_gap_m"], 0.)
        np.testing.assert_array_equal(points, original_points)
        np.testing.assert_array_equal(cells, original_cells)
        pressures = np.full((1, 3), 1e5)
        loads = result.faces.pressure_forces(pressures)
        np.testing.assert_allclose(loads.sum(axis=0), 0., atol=1e-10)
        np.testing.assert_allclose(loads[result.faces.plus[0]].sum(axis=0), [5e4, 0., 0.])

    def test_existing_coincident_split_identities_are_not_merged(self):
        points, cells = self.fixture()
        coordinates = np.vstack((points, points))
        tetrahedra = np.vstack((cells, cells + 5))
        result = split_interior_wall_faces(coordinates, tetrahedra, [1] * 4, [[0, 1, 2]])
        np.testing.assert_array_equal(result.cells[2:], tetrahedra[2:])
        self.assertFalse(np.intersect1d(result.cells[:2], result.cells[2:]).size)

    def test_crack_front_node_remains_shared_when_connected_around_tip(self):
        # A triangular fan of tets around edge (0,1); cut a radial face.
        points = np.array([(0., 0., -1.), (0., 0., 1.), (1., 0., 0.),
                           (-.5, .866, 0.), (-.5, -.866, 0.)])
        cells = np.array([[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]])
        result = split_interior_wall_faces(points, cells, [1] * 3, [[0, 1, 2]])
        self.assertEqual(len(result.coordinates_m), len(points) + 1)
        np.testing.assert_array_equal(result.faces.minus[0, :2], result.faces.plus[0, :2])
        self.assertNotEqual(result.faces.minus[0, 2], result.faces.plus[0, 2])
        state = np.zeros_like(result.coordinates_m)
        state[result.faces.plus[0, 2]] = result.faces.normals[0] * .001
        np.testing.assert_allclose(result.faces.measure(state)["signed_vertex_gaps_m"], [[0., 0., .001]])
        load = result.faces.pressure_forces([[1e5] * 3])
        np.testing.assert_array_equal(load[:2], 0.)

    def test_isolated_fully_tied_interior_facet_requires_resolution(self):
        points = np.array([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.),
                           (0., 0., 1.), (.25, .25, .25)])
        cells = [[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]]
        with self.assertRaisesRegex(ValueError, "fully tied"):
            split_interior_wall_faces(points, cells, [1] * 4, [[0, 1, 4]])

    def test_opt_in_drop_reports_fully_tied_cuts_and_keeps_openable_ones(self):
        points = np.array([(0., 0., -1.), (0., 0., 1.), (1., 0., 0.),
                           (-.5, .866, 0.), (-.5, -.866, 0.)])
        cells = np.array([[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]])
        result = split_interior_wall_faces(points, cells, [1] * 3, [[0, 1, 2]], drop_fully_tied=True)
        self.assertEqual(result.tied_source_faces.shape, (0, 3))
        self.assertEqual(len(result.faces.minus), 1)
        points = np.array([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.),
                           (0., 0., 1.), (.25, .25, .25)])
        cells = [[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]]
        with self.assertRaisesRegex(ValueError, "Every cut is fully tied"):
            split_interior_wall_faces(points, cells, [1] * 4, [[0, 1, 4]], drop_fully_tied=True)
        result = split_interior_wall_faces(points, cells, [1] * 4, [[0, 1, 4], [0, 2, 4], [0, 3, 4]],
                                           drop_fully_tied=True)
        self.assertEqual(result.tied_source_faces.shape[1], 3)
        kept = {tuple(sorted(face)) for face in result.source_faces.tolist()}
        dropped = {tuple(face) for face in result.tied_source_faces.tolist()}
        self.assertFalse(kept & dropped)
        self.assertEqual(kept | dropped, {(0, 1, 4), (0, 2, 4), (0, 3, 4)})

    def test_node_star_completion_unlocks_seed_without_releasing_unlisted_faces(self):
        from itertools import combinations
        if __package__:
            from .fracture_faces import bonded_node_representatives
            from .fracture_topology import NodeStarPatchResolver
        else:
            from fracture_faces import bonded_node_representatives
            from fracture_topology import NodeStarPatchResolver
        points = np.array([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.),
                           (0., 0., 1.), (.25, .25, .25)])
        cells = np.array([[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]])
        candidates = interior_wall_candidates(points, cells, [1] * 4, [[0, 1]])
        split = split_interior_wall_faces(points, cells, [1] * 4, candidates)
        faces = split.faces
        resolver = NodeStarPatchResolver(faces, split.parent_node_ids)
        original_minus, original_plus = faces.minus.copy(), faces.plus.copy()
        for broken in ((), (1,)):
            completions = resolver(broken, (0,))
            self.assertTrue(completions)
            self.assertFalse(any(set(a) < set(b) for a in completions for b in completions))
            for patch in completions:
                self.assertIn(0, patch)
                self.assertFalse(set(patch) & set(broken))
                mask = np.ones(len(candidates), dtype=bool)
                mask[list(set(broken) | set(patch))] = False
                representatives = bonded_node_representatives(faces, mask)
                self.assertTrue(np.any(
                    representatives[faces.minus[0]] != representatives[faces.plus[0]]))
                for face in np.flatnonzero(mask):
                    np.testing.assert_array_equal(
                        representatives[faces.minus[face]], representatives[faces.plus[face]])
                self.assertTrue(resolver._connected_disk(patch))
            # Exhaustively check that a returned completion has minimum face count
            # among all seed-containing cuts unlocking at least one seed vertex.
            minimum = None
            available = sorted(set(range(len(candidates))) - set(broken) - {0})
            for size in range(len(available) + 1):
                for extra in combinations(available, size):
                    mask = np.ones(len(candidates), dtype=bool)
                    mask[list(set(broken) | {0} | set(extra))] = False
                    reps = bonded_node_representatives(faces, mask)
                    if np.any(reps[faces.minus[0]] != reps[faces.plus[0]]):
                        minimum = size + 1
                        break
                if minimum is not None:
                    break
            self.assertEqual(min(map(len, completions)), minimum)
        np.testing.assert_array_equal(faces.minus, original_minus)
        np.testing.assert_array_equal(faces.plus, original_plus)
        rotation = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        rotated = split_interior_wall_faces(points @ rotation.T, cells, [1] * 4, candidates)
        self.assertEqual(resolver((), (0,)),
                         NodeStarPatchResolver(rotated.faces, rotated.parent_node_ids)((), (0,)))
        copies = split_interior_wall_faces(
            np.vstack((points, points)), np.vstack((cells, cells + len(points))),
            [1] * 8, np.vstack((candidates, candidates + len(points))))
        self.assertEqual(
            resolver((), (0,)),
            NodeStarPatchResolver(copies.faces, copies.parent_node_ids)((), (0,)))

    def test_unknown_boundary_nonwall_nonmanifold_and_duplicate_cuts_fail(self):
        points, cells = self.fixture()
        for tetrahedra, regions, cuts in (
            (cells, [1, 2], [[0, 1, 2]]),
            (cells, [1, 1], [[0, 1, 3]]),
            (cells, [1, 1], [[0, 1, 2], [2, 1, 0]]),
            (cells, [1, 1], [[0, 1, 9]]),
            (cells, [1, 1], [[True, False, True]]),
            (np.vstack((cells, [0, 1, 2, 5])), [1] * 3, [[0, 1, 2]]),
        ):
            coordinates = np.vstack((points, [2., 0., 0.])) if len(tetrahedra) == 3 else points
            with self.subTest(cuts=cuts, regions=regions), self.assertRaises(ValueError):
                split_interior_wall_faces(coordinates, tetrahedra, regions, cuts)
        with self.assertRaises(ValueError):
            split_interior_wall_faces(points, cells, [1, 1], [])

    def test_accepted_state_transfer_validates_source_shape_and_values(self):
        points, cells = self.fixture()
        result = split_interior_wall_faces(points, cells, [1, 1], [[0, 1, 2]])
        for state in (np.zeros((4, 3)), np.full((5, 3), np.nan), np.ones((5, 3), dtype=bool)):
            with self.subTest(shape=state.shape), self.assertRaises(ValueError):
                result.transfer_nodal_values(state)

    def test_candidate_inventory_is_wall_only_source_bound_and_orientation_neutral(self):
        points, cells = self.fixture()
        np.testing.assert_array_equal(
            interior_wall_candidates(points, cells, [1, 1], [[0, 1]]), [[0, 1, 2]])
        # A rotated source retains identical candidate IDs, with no horizontal
        # or dominant-axis filter on eligible facets.
        rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        np.testing.assert_array_equal(
            interior_wall_candidates(points @ rotation.T, cells[::-1], [1, 1], [[1, 0]]), [[0, 1, 2]])
        for regions, edges in (([1, 2], [[0, 1]]), ([1, 1], [[0, 3]])):
            with self.subTest(regions=regions, edges=edges), self.assertRaises(ValueError):
                interior_wall_candidates(points, cells, regions, edges)
        # Coincident source identities remain separate, not welded by location.
        coordinates = np.vstack((points, points))
        tetrahedra = np.vstack((cells, cells + len(points)))
        np.testing.assert_array_equal(
            interior_wall_candidates(coordinates, tetrahedra, [1] * 4, [[0, 1]]),
            [[0, 1, 2], [5, 6, 7]])

    def test_candidate_extraction_rejects_nonmanifold_and_overlapping_owner_geometry(self):
        points, cells = self.fixture()
        with self.assertRaisesRegex(ValueError, "opposite sides"):
            interior_wall_candidates(np.vstack((points, [-2., 0., 0.])),
                                     np.array([cells[0], [0, 1, 2, 5]]), [1, 1], [[0, 1]])
        with self.assertRaisesRegex(ValueError, "Nonmanifold"):
            interior_wall_candidates(np.vstack((points, [2., 0., 0.])),
                                     np.vstack((cells, [0, 1, 2, 5])), [1] * 3, [[0, 1]])

    def test_external_boundary_labels_preserve_owner_orientation_and_geometry(self):
        points, cells = self.fixture()
        result = split_interior_wall_faces(points, cells, [1, 1], [[0, 1, 2]])
        source = np.array([[0, 1, 3], [0, 2, 3], [1, 2, 3],
                           [0, 1, 4], [0, 2, 4], [1, 2, 4]])
        tags = np.array([10, 11, 12, 13, 14, 15], dtype=np.int32)
        mapped, copied = result.transfer_boundary_faces(source, tags)
        np.testing.assert_array_equal(copied, tags)
        np.testing.assert_array_equal(result.parent_node_ids[mapped], source)
        np.testing.assert_array_equal(result.coordinates_m[mapped], points[source])
        for row, face in enumerate(mapped):
            owners = [cell for cell in result.cells if set(face).issubset(cell)]
            self.assertEqual(len(owners), 1)
            self.assertTrue(set(face).issubset(result.cells[0 if row < 3 else 1]))
        for source_faces, values in (([[0, 1, 2]], [10]), ([[0, 1, 3], [3, 1, 0]], [10, 11]),
                                     ([[0, 1, 3]], [-1]), ([[0, 1, 3]], [True])):
            with self.subTest(source_faces=source_faces), self.assertRaises(ValueError):
                result.transfer_boundary_faces(source_faces, values)


if __name__ == "__main__":
    unittest.main()
