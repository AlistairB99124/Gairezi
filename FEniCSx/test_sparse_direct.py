"""Tests for the MUMPS/SuperLU sparse direct solver with factorisation reuse."""
import unittest

import numpy as np
from scipy import sparse

from FEniCSx import sparse_direct


def laplacian(size):
    main = 2. * np.ones(size)
    off = -np.ones(size - 1)
    return sparse.diags([off, main, off], [-1, 0, 1], format="csr")


def saddle_point(size):
    constraint = sparse.csr_matrix(([1., -1.], ([0, 0], [0, size - 1])), shape=(1, size))
    return sparse.bmat([[laplacian(size), constraint.T], [constraint, None]], format="csc")


class SparseDirectTest(unittest.TestCase):
    def setUp(self):
        self.threshold = sparse_direct.MUMPS_MINIMUM_UNKNOWNS
        self.cache_size = sparse_direct.CACHE_SIZE
        sparse_direct.clear_cache()

    def tearDown(self):
        sparse_direct.MUMPS_MINIMUM_UNKNOWNS = self.threshold
        sparse_direct.CACHE_SIZE = self.cache_size
        sparse_direct.clear_cache()

    def use_mumps(self):
        if sparse_direct.backend() != "mumps":
            self.skipTest("petsc4py with MUMPS is not available")
        sparse_direct.MUMPS_MINIMUM_UNKNOWNS = 0

    def test_mumps_agrees_with_superlu_on_spd_and_saddle_point_systems(self):
        rhs_rng = np.random.default_rng(3)
        for matrix in (laplacian(40), saddle_point(40)):
            rhs = rhs_rng.standard_normal(matrix.shape[0])
            expected = sparse_direct.solve(matrix, rhs)
            self.use_mumps()
            actual = sparse_direct.solve(matrix, rhs)
            sparse_direct.MUMPS_MINIMUM_UNKNOWNS = self.threshold
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
            self.assertLess(np.linalg.norm(matrix @ actual - rhs), 1e-10 * np.linalg.norm(rhs))

    def test_symmetric_systems_use_ldlt_and_others_lu(self):
        self.use_mumps()
        rng = np.random.default_rng(5)
        unsymmetric = laplacian(40) + sparse.diags([0.3 * np.ones(39)], [1], format="csr")
        for matrix, symmetric in ((saddle_point(40), True), (unsymmetric, False)):
            rhs = rng.standard_normal(matrix.shape[0])
            x = sparse_direct.solve(matrix, rhs)
            self.assertLess(np.linalg.norm(matrix @ x - rhs), 1e-10 * np.linalg.norm(rhs))
            newest = sparse_direct._cache[next(reversed(sparse_direct._cache))]
            self.assertIs(newest.symmetric, symmetric)
            self.assertEqual(newest.ksp.getPC().getType(), "cholesky" if symmetric else "lu")

    def test_identical_matrix_reuses_factor_and_changed_matrix_refactors(self):
        self.use_mumps()
        matrix = laplacian(30)
        sparse_direct.solve(matrix, np.ones(30))
        first = list(sparse_direct._cache.values())
        sparse_direct.solve(matrix.copy(), np.arange(30.))
        self.assertEqual(list(sparse_direct._cache.values()), first)
        changed = matrix + sparse.identity(30, format="csr")
        x = sparse_direct.solve(changed, np.ones(30))
        np.testing.assert_allclose(changed @ x, np.ones(30), atol=1e-12)
        self.assertEqual(len(sparse_direct._cache), 2)
        newest = sparse_direct._cache[next(reversed(sparse_direct._cache))]
        self.assertTrue(sparse_direct._same(newest.source, sparse_direct._canonical(changed)))

    def test_cache_size_limits_kept_factors(self):
        self.use_mumps()
        for size in (10, 11, 12):
            sparse_direct.solve(laplacian(size), np.ones(size))
        self.assertEqual(len(sparse_direct._cache), sparse_direct.CACHE_SIZE)
        sparse_direct.clear_cache()
        sparse_direct.CACHE_SIZE = 0
        np.testing.assert_allclose(laplacian(8) @ sparse_direct.solve(laplacian(8), np.ones(8)), np.ones(8),
                                   atol=1e-12)
        self.assertEqual(len(sparse_direct._cache), 0)

    def test_singular_matrix_raises_on_both_backends(self):
        singular = sparse.csr_matrix(np.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., 1.]]))
        unsymmetric = sparse.csr_matrix(np.array([[1., -1., 0.], [-1., 1., 0.], [0., 2., 1.]]))
        with self.assertRaises(sparse_direct.SingularMatrixError):
            sparse_direct.solve(singular, np.array([1., -1., 1.]))
        self.use_mumps()
        for matrix in (singular, unsymmetric):
            with self.assertRaises(sparse_direct.SingularMatrixError):
                sparse_direct.solve(matrix, np.array([1., -1., 1.]))
        self.assertEqual(len(sparse_direct._cache), 0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            sparse_direct.solve(sparse.csr_matrix((3, 2)), np.ones(3))
        with self.assertRaises(ValueError):
            sparse_direct.solve(laplacian(3), np.ones(4))


if __name__ == "__main__":
    unittest.main()
