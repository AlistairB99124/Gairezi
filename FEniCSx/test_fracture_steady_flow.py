"""Analytical steady surface head, cubic conductance and identity checks."""
import unittest

import numpy as np

if __package__:
    from .fracture_faces import PairedCrackFaces
    from .fracture_hydrostatic import HydrostaticBoundary, drained_crack_load
    from .fracture_steady_flow import steady_crack_load
else:
    from fracture_faces import PairedCrackFaces
    from fracture_hydrostatic import HydrostaticBoundary, drained_crack_load
    from fracture_steady_flow import steady_crack_load


class SteadySurfaceTests(unittest.TestCase):
    def fixture(self, n=4, aperture=.001, varying=False):
        points = np.array([(0., y / n, z / n) for z in range(n + 1) for y in range(n + 1)])
        triangles = []
        for j in range(n):
            for i in range(n):
                a = j * (n + 1) + i
                triangles.extend(((a, a + 1, a + n + 2), (a, a + n + 2, a + n + 1)))
        triangles = np.asarray(triangles)
        count = len(points)
        faces = PairedCrackFaces(np.vstack((points, points)), triangles, triangles + count,
                                 np.tile([1., 0., 0.], (len(triangles), 1)))
        state = np.zeros_like(faces.coordinates)
        state[count:, 0] = aperture * (1 + points[:, 2] if varying else np.ones(count))
        parents = np.tile(np.arange(count), 2)
        bottom = [[i, i + 1] for i in range(n)]
        top = [[n * (n + 1) + i, n * (n + 1) + i + 1] for i in range(n)]
        boundaries = [HydrostaticBoundary("reservoir", 4., bottom), HydrostaticBoundary("tailwater", 2., top)]
        return faces, state, parents, boundaries

    def solve(self, fixture, **overrides):
        faces, state, parents, boundaries = fixture
        args = dict(water_density_kg_m3=1000., gravity_m_s2=9.81,
                    saturated_face_mask=np.ones(len(faces.minus), dtype=bool),
                    closure_tolerance_m=1e-12, relative_tolerance=1e-10)
        args.update(overrides)
        return steady_crack_load(faces, state, parents, boundaries, **args)

    def test_constant_aperture_linear_head_exact_boundary_flow_and_forces(self):
        for n in (2, 8):
            fixture = self.fixture(n)
            faces = fixture[0]
            result = self.solve(fixture)
            z = faces.coordinates[faces.minus, 2]
            np.testing.assert_allclose(result.vertex_heads_m, 4 - 2 * z, atol=1e-13)
            np.testing.assert_allclose(result.vertex_pressures_pa, 9810. * (4 - 3 * z), atol=1e-9)
            np.testing.assert_allclose(result.nodal_forces_n.sum(axis=0), 0., atol=1e-10)
            count = len(faces.coordinates) // 2
            np.testing.assert_allclose(result.nodal_forces_n[count:].sum(axis=0), [9810. * 2.5, 0., 0.], atol=1e-9)
            reactions = np.array(result.report["dirichlet_reactions_scaled_m4"])
            self.assertAlmostEqual(reactions[reactions > 0].sum(), 2e-9, delta=1e-20)
            self.assertLess(result.report["relative_boundary_balance"], 1e-12)
            self.assertFalse(result.report["solver_ready"])

    def test_uniform_aperture_scaling_changes_conductance_not_head(self):
        first = self.solve(self.fixture(aperture=.001))
        second = self.solve(self.fixture(aperture=.002))
        np.testing.assert_allclose(second.vertex_heads_m, first.vertex_heads_m, atol=1e-13)
        np.testing.assert_allclose(second.report["dirichlet_reactions_scaled_m4"],
                                   8 * np.array(first.report["dirichlet_reactions_scaled_m4"]), atol=1e-21)

    def test_variable_aperture_converges_to_integrated_cubic_law(self):
        errors = []
        for n in (4, 16):
            fixture = self.fixture(n, varying=True)
            result = self.solve(fixture)
            z = fixture[0].coordinates[fixture[0].minus, 2]
            # H=4-2*integral_0^z (1+s)^-3 / integral_0^1 (1+s)^-3.
            exact = 4 - 2 * (1 - (1 + z)**-2) / .75
            errors.append(np.max(np.abs(result.vertex_heads_m - exact)))
            reactions = np.array(result.report["dirichlet_reactions_scaled_m4"])
            self.assertLess(abs(reactions[reactions > 0].sum() / (2e-9 / .375) - 1), .15)
        self.assertLess(errors[1], errors[0] / 8)
        self.assertLess(errors[1], .003)

    def test_equal_heads_reproduce_hydrostatic_component(self):
        fixture = self.fixture()
        faces, state, parents, boundaries = fixture
        boundaries[1] = HydrostaticBoundary("tailwater", 4., boundaries[1].inlet_source_edges)
        steady = self.solve(fixture)
        hydro = drained_crack_load(
            faces, state, parents, boundaries, water_density_kg_m3=1000., gravity_m_s2=9.81,
            saturated_face_mask=np.ones(len(faces.minus), dtype=bool),
            previously_wet_face_mask=np.zeros(len(faces.minus), dtype=bool), closure_tolerance_m=1e-12,
        )
        np.testing.assert_allclose(steady.vertex_pressures_pa, hydro.vertex_pressures_pa, atol=1e-9)
        np.testing.assert_allclose(steady.nodal_forces_n, hydro.nodal_forces_n, atol=1e-10)

    def test_owner_reversal_preserves_heads_pressure_and_forces(self):
        fixture = self.fixture()
        expected = self.solve(fixture)
        faces, state, parents, boundaries = fixture
        minus, plus, normals = faces.minus.copy(), faces.plus.copy(), faces.normals.copy()
        minus[::2], plus[::2] = plus[::2].copy(), minus[::2].copy()
        normals[::2] *= -1
        changed = PairedCrackFaces(faces.coordinates, minus, plus, normals)
        actual = self.solve((changed, state, parents, boundaries))
        np.testing.assert_allclose(actual.vertex_heads_m, expected.vertex_heads_m, atol=1e-13)
        np.testing.assert_allclose(actual.nodal_forces_n, expected.nodal_forces_n, atol=1e-10)

    def test_closed_dry_cavitating_and_bad_inputs_fail_explicitly(self):
        fixture = self.fixture()
        for key, value in (("relative_tolerance", True), ("relative_tolerance", 1.),
                           ("gravity_m_s2", -9.81), ("saturated_face_mask", [True]),
                           ("saturated_face_mask", np.zeros(len(fixture[0].minus), dtype=bool))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.solve(fixture, **{key: value})
        faces, state, parents, boundaries = fixture
        state[len(state) // 2, 0] = 0.
        with self.assertRaisesRegex(ValueError, "positive aperture"):
            self.solve(fixture)
        state[len(state) // 2, 0] = .001
        boundaries[1] = HydrostaticBoundary("tailwater", .5, boundaries[1].inlet_source_edges)
        with self.assertRaisesRegex(ValueError, "Negative pressure"):
            self.solve(fixture)

    def test_isolated_wet_sheet_and_conflicting_head_vertices_rejected(self):
        faces, state, parents, boundaries = self.fixture(2)
        count = len(faces.coordinates)
        duplicated = PairedCrackFaces(
            np.vstack((faces.coordinates, faces.coordinates)),
            np.vstack((faces.minus, faces.minus + count)),
            np.vstack((faces.plus, faces.plus + count)),
            np.vstack((faces.normals, faces.normals)),
        )
        new_parents = np.r_[parents, parents + 9]
        with self.assertRaisesRegex(ValueError, "Isolated wet"):
            self.solve((duplicated, np.vstack((state, state)), new_parents, boundaries))
        # A separate coincident sheet is allowed when independently connected;
        # neither coordinates nor shared elevations may weld its identities.
        extra = [
            HydrostaticBoundary("other_reservoir", 5., boundaries[0].inlet_source_edges + 9),
            HydrostaticBoundary("other_tailwater", 3., boundaries[1].inlet_source_edges + 9),
        ]
        result = self.solve((duplicated, np.vstack((state, state)), new_parents, boundaries + extra))
        np.testing.assert_allclose(result.vertex_heads_m[len(faces.minus):],
                                   result.vertex_heads_m[:len(faces.minus)] + 1., atol=1e-13)
        boundaries.append(HydrostaticBoundary("inconsistent", 5., boundaries[0].inlet_source_edges))
        with self.assertRaisesRegex(ValueError, "Conflicting hydraulic heads"):
            self.solve((faces, state, parents, boundaries))

    def test_point_touching_sheets_are_not_connected_by_a_pressure_dof(self):
        points = np.array([(0., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                           (0., -1., 0.), (0., 0., -1.)])
        triangles = np.array([[0, 1, 2], [0, 3, 4]])
        faces = PairedCrackFaces(np.vstack((points, points)), triangles, triangles + 5, [[1., 0., 0.]] * 2)
        state = np.zeros((10, 3))
        state[5:, 0] = .001
        boundaries = [HydrostaticBoundary("first", 3., [[1, 2]]),
                      HydrostaticBoundary("second", 4., [[3, 4]])]
        with self.assertRaisesRegex(ValueError, "Point-touching hydraulic sheets"):
            self.solve((faces, state, np.tile(np.arange(5), 2), boundaries))

    def test_very_small_aperture_and_head_translation_do_not_change_solution(self):
        fixture = self.fixture(aperture=1e-7)
        reference = self.solve(self.fixture())
        actual = self.solve(fixture, closure_tolerance_m=1e-12)
        np.testing.assert_allclose(actual.vertex_heads_m, reference.vertex_heads_m, atol=1e-13)
        np.testing.assert_allclose(actual.report["dirichlet_reactions_scaled_m4"],
                                   np.array(reference.report["dirichlet_reactions_scaled_m4"]) * 1e-12,
                                   rtol=1e-12, atol=1e-32)
        faces, state, parents, boundaries = fixture
        moved = PairedCrackFaces(faces.coordinates + [10., -7., 1000.], faces.minus, faces.plus, faces.normals)
        translated = [HydrostaticBoundary(b.name, b.elevation_m + 1000., b.inlet_source_edges)
                      for b in boundaries]
        result = self.solve((moved, state, parents, translated))
        np.testing.assert_allclose(result.vertex_pressures_pa, actual.vertex_pressures_pa, atol=2e-8)

    def tied_front_fixture(self, n=4, aperture=.001):
        faces, state, parents, _ = self.fixture(n, aperture)
        count = len(faces.coordinates) // 2
        top = np.arange(n * (n + 1), count)
        plus = faces.plus.copy()
        plus[np.isin(plus, top + count)] -= count
        faces = PairedCrackFaces(faces.coordinates, faces.minus, plus, faces.normals)
        state[count:, 0] = aperture * (1 - faces.coordinates[:count, 2])
        left = [[j * (n + 1), (j + 1) * (n + 1)] for j in range(n)]
        right = [[j * (n + 1) + n, (j + 1) * (n + 1) + n] for j in range(n)]
        return faces, state, parents, [
            HydrostaticBoundary("reservoir", 4., left),
            HydrostaticBoundary("tailwater", 3., right),
        ]

    def test_exact_zero_tied_front_linear_head_and_integrated_wedge_conductance(self):
        for n in (2, 8):
            fixture = self.tied_front_fixture(n)
            faces = fixture[0]
            with self.assertRaisesRegex(ValueError, "positive aperture"):
                self.solve(fixture)
            result = self.solve(fixture, allow_tied_fronts=True)
            y = faces.coordinates[faces.minus, 1]
            z = faces.coordinates[faces.minus, 2]
            np.testing.assert_allclose(result.vertex_heads_m, 4 - y, atol=1e-12)
            np.testing.assert_allclose(result.vertex_pressures_pa, 9810 * (4 - y - z), atol=1e-8)
            reactions = np.array(result.report["dirichlet_reactions_scaled_m4"])
            self.assertAlmostEqual(reactions[reactions > 0].sum(), .001**3 / 4, delta=1e-20)
            self.assertEqual(len(result.report["tied_front_vertex_ids"]), n + 1)
            self.assertEqual(result.report["minimum_vertex_aperture_m"], 0.)
            self.assertEqual(result.report["aperture_regularisation_m"], 0.)
            np.testing.assert_allclose(result.nodal_forces_n.sum(axis=0), 0., atol=1e-9)
            self.assertAlmostEqual(float(np.sum(result.nodal_forces_n * fixture[1])),
                                   9810 * .001 * 19 / 12, places=10)
            smaller = self.solve(self.tied_front_fixture(n, 1e-7), allow_tied_fronts=True)
            np.testing.assert_allclose(smaller.vertex_heads_m, result.vertex_heads_m, atol=1e-12)
            np.testing.assert_allclose(smaller.report["dirichlet_reactions_scaled_m4"],
                                       reactions * 1e-12, rtol=1e-10, atol=1e-30)
            reversed_faces = PairedCrackFaces(
                faces.coordinates, faces.plus, faces.minus, -faces.normals)
            reversed_load = self.solve((reversed_faces, *fixture[1:]), allow_tied_fronts=True)
            np.testing.assert_allclose(reversed_load.nodal_forces_n, result.nodal_forces_n, atol=1e-9)

    def test_tied_front_option_does_not_admit_contact_or_closed_inlets(self):
        fixture = self.fixture()
        faces, state, parents, boundaries = fixture
        state[len(state) // 2, 0] = 0.
        with self.assertRaisesRegex(ValueError, "positive aperture"):
            self.solve(fixture, allow_tied_fronts=True)
        fixture = self.tied_front_fixture()
        faces, state, parents, boundaries = fixture
        n = 4
        top = [[n * (n + 1) + i, n * (n + 1) + i + 1] for i in range(n)]
        boundaries.append(HydrostaticBoundary("closed_front", 3., top))
        with self.assertRaisesRegex(ValueError, "closed hydraulic inlet"):
            self.solve(fixture, allow_tied_fronts=True)
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            self.solve(self.fixture(), allow_tied_fronts=1)

    def test_closed_internal_tied_edge_cannot_short_circuit_distinct_fluid_sheets(self):
        points = np.array([[0., 0., 0.], [0., 1., 0.], [0., .5, 1.], [0., .5, -1.]])
        faces = PairedCrackFaces(
            np.vstack((points, points)), [[0, 1, 2], [1, 0, 3]],
            [[0, 1, 6], [1, 0, 7]], [[1., 0., 0.]] * 2)
        state = np.zeros_like(faces.coordinates)
        state[6:, 0] = .001
        boundaries = [HydrostaticBoundary("upper", 4., [[0, 2]]),
                      HydrostaticBoundary("lower", 3., [[0, 3]])]
        with self.assertRaisesRegex(ValueError, "Closed internal hydraulic edges"):
            self.solve((faces, state, np.tile(np.arange(4), 2), boundaries), allow_tied_fronts=True)

    def test_explicit_hydraulic_subset_preserves_inventory_rows_and_leaves_other_sheet_dry(self):
        from fracture_steady_flow import SteadyCrackHydraulics
        faces, state, parents, boundaries = self.fixture(2)
        count = len(faces.coordinates)
        inventory = PairedCrackFaces(
            np.vstack((faces.coordinates, faces.coordinates + [0., 0., 10.])),
            np.vstack((faces.minus, faces.minus + count)),
            np.vstack((faces.plus, faces.plus + count)),
            np.vstack((faces.normals, faces.normals)))
        # Reverse hydraulic row ordering deliberately; archive identities must
        # not silently assume that hydraulic rows are mechanical face rows.
        ids = np.arange(len(faces.minus))[::-1].copy()
        data = SteadyCrackHydraulics(
            np.r_[parents, parents + 100], tuple(boundaries), 1000., 9.81,
            np.ones(len(ids), dtype=bool), 1e-12, 1e-10, inventory_face_ids=ids)
        ids[:] = 0
        self.assertFalse(data.inventory_face_ids.flags.writeable)
        load = data.load(inventory, np.vstack((state, np.zeros_like(state))))
        expected = self.solve((faces, state, parents, boundaries))
        np.testing.assert_allclose(load.vertex_pressures_pa, expected.vertex_pressures_pa[::-1], atol=1e-8)
        np.testing.assert_allclose(load.nodal_forces_n[:count], expected.nodal_forces_n, atol=1e-9)
        np.testing.assert_array_equal(load.nodal_forces_n[count:], 0.)
        self.assertEqual(load.report["mechanical_inventory_face_ids"], list(range(len(ids)))[::-1])
        for invalid in ([0, 0], [-1], [1.], [], [True]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                SteadyCrackHydraulics(
                    parents, tuple(boundaries), 1000., 9.81, np.ones(len(invalid), dtype=bool),
                    1e-12, 1e-10, inventory_face_ids=invalid)
        foreign = SteadyCrackHydraulics(
            parents, tuple(boundaries), 1000., 9.81, np.ones(1, dtype=bool),
            1e-12, 1e-10, inventory_face_ids=[len(faces.minus)])
        with self.assertRaisesRegex(ValueError, "mechanical inventory"):
            foreign.load(faces, state)

    def test_saturated_common_head_trial_is_not_a_final_closed_or_disconnected_verdict(self):
        from fracture_steady_flow import HydrostaticEquilibriumHydraulics
        faces, state, parents, boundaries = self.fixture(2)
        with self.assertRaisesRegex(ValueError, "one common reservoir head"):
            HydrostaticEquilibriumHydraulics(
                parents, tuple(boundaries), 1000., 9.81, np.ones(len(faces.minus), dtype=bool),
                1e-12, 1e-10)
        boundaries = (boundaries[0],)
        data = HydrostaticEquilibriumHydraulics(
            parents, boundaries, 1000., 9.81, np.ones(len(faces.minus), dtype=bool), 1e-12, 1e-10)
        closed = np.zeros_like(state)
        trial = data.trial_load(faces, closed)
        self.assertTrue(trial.report["final_aperture_access_audit_pending"])
        self.assertFalse(data.requires_dry_opening)
        self.assertGreater(trial.vertex_pressures_pa.min(), 0.)
        with self.assertRaisesRegex(ValueError, "positive aperture"):
            data.load(faces, closed)
        accepted = data.load(faces, state)
        self.assertNotIn("final_aperture_access_audit_pending", accepted.report)
        np.testing.assert_array_equal(accepted.vertex_pressures_pa, trial.vertex_pressures_pa)
        np.testing.assert_array_equal(accepted.nodal_forces_n, trial.nodal_forces_n)


if __name__ == "__main__":
    unittest.main()
