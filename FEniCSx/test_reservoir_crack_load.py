"""Analytical clipped water load, reservoir work and open-access checks."""
import unittest

import numpy as np

from fracture_faces import PairedCrackFaces
from fracture_hydrostatic import HydrostaticBoundary
from reservoir_crack_load import ReservoirCrackHydraulics


class ReservoirLoadTests(unittest.TestCase):
    def fixture(self):
        points = np.array([(0., 0., 0.), (0., 1., 0.), (0., 0., 1.)])
        faces = PairedCrackFaces(
            np.vstack((points, points)), [[0, 1, 2]], [[3, 4, 5]], [[1., 0., 0.]])
        hydraulics = ReservoirCrackHydraulics(
            np.array([0, 1, 2, 0, 1, 2]),
            (HydrostaticBoundary("reservoir", .5, [[0, 1]]),), 1000., 10.,
            np.ones(1, dtype=bool), 1e-12, 1e-10)
        displacement = np.zeros((6, 3))
        displacement[3:, 0] = .001
        return faces, hydraulics, displacement

    def test_partial_triangle_has_exact_force_moment_and_virtual_work(self):
        faces, hydraulics, displacement = self.fixture()
        load = hydraulics.load(faces, displacement)
        # Integral_0^.5 (1-z)*10000*(.5-z) dz = 3125/3 N.
        force = load.nodal_forces_n[faces.plus[0], 0]
        self.assertAlmostEqual(force.sum(), 3125 / 3)
        # Integral_0^.5 z*(1-z)*10000*(.5-z) dz = 625/4 N m.
        self.assertAlmostEqual(force @ faces.coordinates[faces.plus[0], 2], 625 / 4)
        np.testing.assert_allclose(load.nodal_forces_n.sum(axis=0), 0., atol=1e-12)
        self.assertAlmostEqual(float(np.sum(load.nodal_forces_n * displacement)), .001 * 3125 / 3)
        np.testing.assert_array_equal(load.vertex_pressures_pa, [[5000., 5000., 0.]])
        self.assertEqual(load.report["partially_submerged_face_ids"], [0])
        # Interpolating clamped nodal pressures over the whole triangle is wrong.
        self.assertNotAlmostEqual(faces.pressure_forces(load.vertex_pressures_pa)[3:, 0].sum(), force.sum())

    def test_pressure_can_open_trial_but_closed_or_dry_only_access_is_not_accepted(self):
        faces, hydraulics, displacement = self.fixture()
        trial = hydraulics.trial_load(faces, np.zeros_like(displacement))
        self.assertGreater(np.linalg.norm(trial.nodal_forces_n), 0.)
        with self.assertRaisesRegex(ValueError, "no open submerged reservoir path"):
            hydraulics.load(faces, np.zeros_like(displacement))
        displacement[:] = 0.
        displacement[5, 0] = .001  # only the dry vertex opens; inlet stays closed
        with self.assertRaisesRegex(ValueError, "no open submerged reservoir path"):
            hydraulics.load(faces, displacement)

    def test_disconnected_face_never_passes_reservoir_access_audit(self):
        faces, hydraulics, displacement = self.fixture()
        points = np.vstack((faces.coordinates, faces.coordinates + [0., 2., 0.]))
        extended = PairedCrackFaces(
            points, [[0, 1, 2], [6, 7, 8]], [[3, 4, 5], [9, 10, 11]],
            [[1., 0., 0.], [1., 0., 0.]])
        hydraulics = ReservoirCrackHydraulics(
            np.array([0, 1, 2, 0, 1, 2, 6, 7, 8, 6, 7, 8]), hydraulics.boundaries,
            1000., 10., np.ones(2, dtype=bool), 1e-12, 1e-10)
        with self.assertRaisesRegex(ValueError, "no open submerged reservoir path"):
            hydraulics.load(extended, np.vstack((displacement, displacement)))


if __name__ == "__main__":
    unittest.main()
