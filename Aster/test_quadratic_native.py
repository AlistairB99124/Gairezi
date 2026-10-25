"""Native characteristic contract failures without requiring native DLLs."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from Aster.native_discrete import characteristics
from Aster.quadratic_spring_benchmark import verify


class Mesh:
    def __init__(self):
        self.groups = {"QS_POINTS": [0, 1], "QS_LINKS": [2],
                       "QS_P_000000": [0], "QS_P_000001": [1], "QS_L_000000": [2]}

    def getConnectivity(self):
        return [[0], [1], [0, 1]]

    def getCells(self, group):
        return self.groups[group]


class NativeCharacteristicTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.archive = Path(self.temporary.name) / "characteristics.npz"
        self.data = {
            "point_node_ids": np.array([0, 1]), "link_node_ids": np.array([[0, 1]]),
            "point_matrices_n_per_m": np.array([np.eye(3), 2 * np.eye(3)]),
            "link_matrices_n_per_m": np.array([-.5 * np.eye(3)]),
            "point_group_matrices_n_per_m": np.array([np.eye(3), 2 * np.eye(3)]),
            "link_group_matrices_n_per_m": np.array([-.5 * np.eye(3)]),
            "point_group_ids": np.array([0, 1]), "link_group_ids": np.array([0]),
        }
        # Test only the contract/encoding here; actual native assembly is checked
        # separately by quadratic_spring_benchmark.
        self.modules = patch.dict("sys.modules", {
            "code_aster.Cata.Syntax": SimpleNamespace(_F=lambda **kwargs: kwargs),
            "code_aster.Commands": SimpleNamespace(AFFE_CARA_ELEM=lambda **kwargs: kwargs,
                                                    DEFI_GROUP=lambda **kwargs: kwargs),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def call(self):
        np.savez(self.archive, **self.data)
        return characteristics(Mesh(), "model", self.archive)

    def test_native_symmetric_encoding_reconstructs_diagonal_and_cross_blocks(self):
        result = self.call()
        self.assertEqual(result["MODELE"], "model")
        self.assertEqual(len(result["DISCRET"]), 3)
        for entry, expected in zip(result["DISCRET"], [
            np.eye(3), 2 * np.eye(3),
            np.block([[np.zeros((3, 3)), -.5 * np.eye(3)],
                      [-.5 * np.eye(3), np.zeros((3, 3))]]),
        ]):
            self.assertEqual(entry["SYME"], "OUI")
            rebuilt = np.zeros_like(expected)
            indices = [(row, column) for column in range(len(expected)) for row in range(column + 1)]
            for (row, column), value in zip(indices, entry["VALE"]):
                rebuilt[row, column] = rebuilt[column, row] = value
            np.testing.assert_array_equal(rebuilt, expected)

    def test_reordered_native_identities_are_rejected(self):
        self.data["point_node_ids"] = np.array([1, 0])
        with self.assertRaisesRegex(ValueError, "node identities/order"):
            self.call()

    def test_corrupt_grouped_coefficients_are_rejected(self):
        self.data["link_group_matrices_n_per_m"] *= 1.001
        with self.assertRaisesRegex(ValueError, "encoding error bound"):
            self.call()

    def test_invalid_group_mapping_is_rejected(self):
        self.data["link_group_ids"] = np.array([1])
        with self.assertRaisesRegex(ValueError, "Invalid grouped native"):
            self.call()

    def test_incomplete_identity_inventory_is_rejected(self):
        self.data["link_node_ids"] = np.array([[0, 1], [1, 0]])
        with self.assertRaisesRegex(ValueError, "identity archive is incomplete"):
            self.call()

    def test_incomplete_and_nonfinite_stiffness_fail_explicitly(self):
        for value in (np.zeros((1, 2, 2)), np.full((1, 3, 3), np.nan)):
            self.data["link_matrices_n_per_m"] = value
            with self.subTest(shape=value.shape), self.assertRaisesRegex(ValueError, "incomplete/nonfinite"):
                self.call()

    def test_changed_native_matrix_cannot_pass_benchmark(self):
        nodes = np.zeros((1, 3))
        pairs, blocks = np.array([[0, 0]]), np.eye(3)[None]
        native = {"coordinates": nodes.tolist(), "dofs": [[0, "DX"], [0, "DY"], [0, "DZ"]],
                  "matrix": np.eye(3).tolist()}
        self.assertEqual(verify(native, nodes, pairs, blocks)["relative_matrix_error"], 0.)
        native["matrix"][1][1] = 2.
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            verify(native, nodes, pairs, blocks)
        native["matrix"][1][1] = float("nan")
        with self.assertRaisesRegex(ValueError, "incomplete/nonfinite"):
            verify(native, nodes, pairs, blocks)
        native["matrix"] = np.zeros((3, 3)).tolist()
        with self.assertRaisesRegex(ValueError, "finite and nonzero"):
            verify(native, nodes, pairs, np.zeros((1, 3, 3)))


if __name__ == "__main__":
    unittest.main()
