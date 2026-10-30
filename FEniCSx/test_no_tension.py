"""Independent analytical/variational checks, not concrete calibration."""
import unittest

import numpy as np
from scipy.optimize import minimize

from FEniCSx.no_tension import elastic_constants, response, tensile_strength_response


class NoTensionTests(unittest.TestCase):
    young, poisson = 35e9, .2

    def test_selected_strength_keeps_elasticity_below_cap_and_limits_tension(self):
        strength = 1e6
        for stress in (.8e6, 1.2e6, -1.2e6):
            strain = np.diag([stress / self.young,
                              -self.poisson * stress / self.young,
                              -self.poisson * stress / self.young])
            result = tensile_strength_response(strain, self.young, self.poisson, strength)
            self.assertAlmostEqual(result.stress_pa[0, 0], min(stress, strength), places=7)
            if stress <= strength:
                np.testing.assert_allclose(result.relaxed_strain, 0, atol=1e-18)
        zero = tensile_strength_response(np.eye(3) * .001, self.young, self.poisson, 0.)
        np.testing.assert_array_equal(zero.stress_pa, response(
            np.eye(3) * .001, self.young, self.poisson).stress_pa)

    def test_finite_strength_energy_gradient_and_tangent(self):
        direction = np.array([[.2, .1, 0], [.1, -.4, .05], [0, .05, .3]])
        for strain in (np.diag([1e-6, -2e-6, 3e-6]), np.diag([-.003, .002, .004])):
            result = tensile_strength_response(strain, self.young, self.poisson, 1e6)
            step = 1e-9
            plus = tensile_strength_response(strain + step * direction, self.young, self.poisson, 1e6)
            minus = tensile_strength_response(strain - step * direction, self.young, self.poisson, 1e6)
            np.testing.assert_allclose(
                (plus.energy_j_m3 - minus.energy_j_m3) / (2 * step),
                np.sum(result.stress_pa * direction), rtol=1e-6, atol=.01,
            )
            np.testing.assert_allclose(
                (plus.stress_pa - minus.stress_pa) / (2 * step),
                result.tangent(direction), rtol=1e-6, atol=10,
            )
        for strength in (-1., True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                tensile_strength_response(np.eye(3), self.young, self.poisson, strength)

    def test_tension_is_stress_free_without_residual_stiffness(self):
        result = response(np.diag([.001, .002, .003]), self.young, self.poisson)
        np.testing.assert_array_equal(result.stress_pa, 0)
        self.assertEqual(result.energy_j_m3, 0)
        np.testing.assert_allclose(result.relaxed_strain, np.diag([.001, .002, .003]))

    def test_all_compressive_stress_states_keep_original_elastic_compliance(self):
        stress = np.array([np.diag([-1e6, -2e6, -3e6]), np.diag([-2e6, 0, 0])])
        strain = ((1 + self.poisson) * stress - self.poisson * np.trace(
            stress, axis1=-2, axis2=-1,
        )[:, None, None] * np.eye(3)) / self.young
        result = response(strain, self.young, self.poisson)
        np.testing.assert_allclose(result.stress_pa, stress, atol=1e-8)
        np.testing.assert_allclose(result.relaxed_strain, 0, atol=1e-18)

    def test_mixed_extension_keeps_uniaxial_compression_not_poisson_tension(self):
        strain = np.diag([-.001, .001, .002])
        result = response(strain, self.young, self.poisson)
        np.testing.assert_allclose(result.stress_pa, np.diag([-35e6, 0, 0]), atol=1e-8)
        np.testing.assert_allclose(result.relaxed_strain, np.diag([0, .0008, .0018]), atol=1e-18)

    def test_rotations_batches_and_complementarity(self):
        rng = np.random.default_rng(413)
        tensors = rng.normal(size=(100, 3, 3)) * .001
        tensors = (tensors + tensors.transpose(0, 2, 1)) / 2
        result = response(tensors, self.young, self.poisson)
        self.assertTrue(np.all(np.linalg.eigvalsh(result.stress_pa) <= 1e-7))
        self.assertTrue(np.all(np.linalg.eigvalsh(result.relaxed_strain) >= -1e-16))
        np.testing.assert_allclose(np.einsum("nij,nij->n", result.stress_pa,
                                           result.relaxed_strain), 0, atol=1e-9)
        self.assertTrue(np.all(result.energy_j_m3 >= -1e-9))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotated = response(rotation @ tensors @ rotation.T, self.young, self.poisson)
        np.testing.assert_allclose(rotated.stress_pa, rotation @ result.stress_pa @ rotation.T,
                                   atol=1e-6)

    def test_energy_gradient_and_tangent_in_each_smooth_branch(self):
        rng = np.random.default_rng(57)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        direction = np.array([[.2, .1, -.03], [.1, -.4, .05], [-.03, .05, .3]])
        for values in ((-.003, -.002, -.001), (-.003, -.002, .004),
                       (-.003, .003, .004), (.001, .002, .003),
                       (-.002, -.002, .004), (.002, .002, .002)):
            with self.subTest(values=values):
                strain = rotation @ np.diag(values) @ rotation.T
                result = response(strain, self.young, self.poisson)
                step = 1e-8
                minus = response(strain - step * direction, self.young, self.poisson)
                plus = response(strain + step * direction, self.young, self.poisson)
                np.testing.assert_allclose((plus.energy_j_m3 - minus.energy_j_m3) / (2 * step),
                                           np.sum(result.stress_pa * direction), rtol=1e-7, atol=.01)
                np.testing.assert_allclose((plus.stress_pa - minus.stress_pa) / (2 * step),
                                           result.tangent(direction), rtol=1e-6, atol=10)

    def test_independent_minimum_energy_oracle_including_auxetic_material(self):
        rng = np.random.default_rng(71)
        for poisson in (-.3, 0, .2, .45):
            mu, lam = elastic_constants(1., poisson)
            for _ in range(15):
                strain = np.diag(rng.normal(size=3))

                def energy(eta):
                    elastic = np.diag(strain) - eta
                    return mu * np.sum(elastic**2) + .5 * lam * np.sum(elastic)**2

                def gradient(eta):
                    elastic = np.diag(strain) - eta
                    return -2 * mu * elastic - lam * np.sum(elastic)

                optimum = minimize(energy, np.zeros(3), jac=gradient,
                                   bounds=[(0, None)] * 3, method="L-BFGS-B",
                                   options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 100})
                self.assertTrue(optimum.success, optimum.message)
                actual = response(strain, 1., poisson)
                self.assertAlmostEqual(actual.energy_j_m3, optimum.fun, places=10)
                np.testing.assert_allclose(np.diag(actual.relaxed_strain), optimum.x, atol=1e-7)

    def test_origin_uses_compressive_side_tangent(self):
        result = response(np.zeros((3, 3)), self.young, self.poisson)
        direction = -np.eye(3)
        mu, lam = elastic_constants(self.young, self.poisson)
        np.testing.assert_allclose(result.tangent(direction), 2 * mu * direction - 3 * lam * np.eye(3))

    def test_invalid_material_and_tensor_inputs_are_explicit(self):
        for young, poisson in ((0, .2), (True, .2), (1., .5), (1., -1), (np.nan, .2)):
            with self.subTest(young=young, poisson=poisson), self.assertRaises(ValueError):
                response(np.eye(3), young, poisson)
        for strain in (np.ones((2, 2)), np.full((3, 3), np.nan),
                       np.arange(9).reshape(3, 3), np.eye(3, dtype=complex), True):
            with self.subTest(strain=strain), self.assertRaises(ValueError):
                response(strain, self.young, self.poisson)


if __name__ == "__main__":
    unittest.main()
