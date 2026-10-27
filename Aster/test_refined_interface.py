"""Patch, energy and coverage checks for independently refined joint traces."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from Aster.quadratic_surface import surface_blocks
from Aster.refined_interface import interface_blocks
from Aster.test_quadratic_surface import TRIANGLE, dense
from Aster.source_contract import source_springs
from Aster.whole_model_crack_mesh import elevate_band, slice_plane


def trace(corners, triangles, quadratic):
    nodes = list(corners)
    edges = {}
    rows = []
    for triangle in triangles:
        row = list(triangle)
        if quadratic:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                key = tuple(sorted((triangle[a], triangle[b])))
                if key not in edges:
                    edges[key] = len(nodes)
                    nodes.append((corners[key[0]] + corners[key[1]]) / 2)
                row.append(edges[key])
        rows.append(row)
    return np.asarray(nodes), np.asarray(rows)


class RefinedInterfaceTests(unittest.TestCase):
    def test_both_wall_sides_retain_parent_faces_through_two_cuts_and_enrichment(self):
        tetra = np.array([[0., 0, -1.], [1., 0, 1.], [0., 1., 1.], [0., 0, 1.]])
        nodes = np.vstack([tetra, tetra[[2, 1, 0, 3]]])
        cells = np.array([[0, 1, 2, 3], [6, 5, 4, 7]])
        faces = {"slave": np.array([[0, 1, 2]]), "master": np.array([[6, 5, 4]])}
        origins = {"slave": np.array([0]), "master": np.array([0])}
        for plane in (-.1, .1):
            nodes, cells, _, faces = slice_plane(nodes, cells, np.ones(len(cells), bool),
                                                 faces, plane, origins)
        elevated = elevate_band(nodes, cells, np.ones(len(cells), bool), faces)
        area = np.linalg.norm(np.cross(tetra[1] - tetra[0], tetra[2] - tetra[0])) / 2
        pairs, blocks, audit = interface_blocks(
            elevated["nodes"], elevated["faces"]["slave"], elevated["faces"]["master"],
            origins["slave"], origins["master"], [[2., 3., 4.]], source_areas_m2=[area])
        matrix = dense(pairs, blocks, len(elevated["nodes"]))
        affine = elevated["nodes"] + 3.
        np.testing.assert_allclose(matrix @ affine.ravel(), 0, atol=1e-13)
        self.assertAlmostEqual(audit["area_m2"], area)
        self.assertGreater(audit["slave_faces"], 3)
        self.assertGreater(audit["master_faces"], 3)
        self.assertGreaterEqual(elevated["faces"]["slave"][:, 3:].min(), 0)
        self.assertGreaterEqual(elevated["faces"]["master"][:, 3:].min(), 0)

    def test_matching_operator_equals_exact_existing_mass_for_every_order(self):
        for slave_width, master_width in ((3, 3), (6, 3), (3, 6), (6, 6)):
            nodes = np.vstack([TRIANGLE[:slave_width], TRIANGLE[:master_width]])
            slave = np.arange(slave_width)[None]
            master = np.arange(slave_width, len(nodes))[None]
            coefficients = np.array([[2., 3., 4.]])
            expected = surface_blocks(nodes, slave, np.diag(coefficients[0]), master)
            actual = interface_blocks(nodes, slave, master, [0], [0], coefficients)
            np.testing.assert_allclose(dense(*expected, len(nodes)),
                                       dense(*actual[:2], len(nodes)), rtol=1e-12, atol=1e-13)

    def test_nonmatching_partitions_preserve_affine_motion_jump_energy_and_midside_modes(self):
        corners = TRIANGLE[:3]
        a = np.vstack([corners, (corners[0] + corners[1]) / 2])
        b = np.vstack([corners, (corners[1] + corners[2]) / 2])
        for quadratic_slave, quadratic_master in ((True, False), (False, True), (True, True)):
            with self.subTest(orders=(quadratic_slave, quadratic_master)):
                slave_nodes, slave = trace(a, [[0, 3, 2], [3, 1, 2]], quadratic_slave)
                master_nodes, master = trace(b, [[0, 1, 3], [0, 3, 2]], quadratic_master)
                nodes = np.vstack([slave_nodes, master_nodes])
                master += len(slave_nodes)
                pairs, blocks, audit = interface_blocks(
                    nodes, slave, master, [0, 0], [0, 0], [[2., 3., 4.]])
                matrix = dense(pairs, blocks, len(nodes))
                np.testing.assert_allclose(matrix, matrix.T, atol=1e-13)
                self.assertGreater(np.linalg.eigvalsh(matrix).min(), -1e-12)
                affine = nodes @ np.array([[.2, .3, -.1], [1., -2., 3.], [.7, .2, .9]]) + 5.
                np.testing.assert_allclose(matrix @ affine.ravel(), 0, atol=2e-13)
                jump = np.array([.2, -.3, .4])
                state = np.zeros_like(nodes)
                state[:len(slave_nodes)] = jump
                force = (matrix @ state.ravel()).reshape(-1, 3)
                np.testing.assert_allclose(force[:len(slave_nodes)].sum(0), 3 * jump * [2, 3, 4])
                np.testing.assert_allclose(force.sum(0), 0, atol=1e-14)
                self.assertAlmostEqual(state.ravel() @ matrix @ state.ravel(),
                                       3 * np.sum(jump ** 2 * [2, 3, 4]))
                self.assertAlmostEqual(audit["area_m2"], 3.)
                self.assertLess(audit["maximum_relative_coverage_error"], 1e-12)
                if quadratic_slave:
                    state[:] = 0
                    state[slave[0, 3], 0] = 1
                    self.assertGreater(state.ravel() @ matrix @ state.ravel(), 0)
                    state[:] = 0
                    state[:len(slave_nodes), 0] = slave_nodes[:, 0] ** 2
                    self.assertAlmostEqual(state.ravel() @ matrix @ state.ravel(), 2 * 3 * 2 ** 4 / 15)
                    if quadratic_master:
                        state[len(slave_nodes):, 0] = master_nodes[:, 0] ** 2
                        np.testing.assert_allclose(matrix @ state.ravel(), 0, atol=1e-13)

    def test_missing_duplicate_unmatched_and_welded_faces_fail(self):
        nodes = np.vstack([TRIANGLE, TRIANGLE])
        slave, master = np.arange(6)[None], np.arange(6, 12)[None]
        with self.assertRaisesRegex(ValueError, "split node"):
            interface_blocks(nodes, slave, slave, [0], [0], [[1., 2., 3.]])
        with self.assertRaisesRegex(ValueError, "coverage"):
            interface_blocks(nodes, slave, np.vstack([master, master]), [0], [0, 0], [[1., 2., 3.]])
        shifted = nodes.copy()
        shifted[6:, 0] += 1
        with self.assertRaisesRegex(ValueError, "coverage"):
            interface_blocks(shifted, slave, master, [0], [0], [[1., 2., 3.]])
        with self.assertRaisesRegex(ValueError, "coverage"):
            interface_blocks(nodes, slave, master, [1], [1], [[1., 2., 3.], [1., 2., 3.]])
        with self.assertRaisesRegex(ValueError, "source triangle area"):
            interface_blocks(nodes, slave, master, [0], [0], [[1., 2., 3.]], source_areas_m2=[6.])


class SourceInterfaceTests(unittest.TestCase):
    def test_source_faces_retain_split_identities_and_physical_coefficients(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            mesh = directory / "mesh.msh"
            mesh.write_bytes(b"fixture")
            (directory / "config.json").write_text("{}")
            (directory / "diagnostic.json").write_text(json.dumps({"rock_outer_support": {"faces": {}}}))
            (directory / "dam_model.sif").write_text(
                'Boundary Condition 1\n Name = "joint_spring"\n Target Boundaries(1) = 10\n'
                ' Mortar BC = Integer 2\n'
                ' Mortar BC Coefficient Displacement 1 = Real 2000\n'
                ' Mortar BC Coefficient Displacement 2 = Real 3000\n'
                ' Mortar BC Coefficient Displacement 3 = Real 4000\nEnd\n'
                'Boundary Condition 2\n Target Boundaries(1) = 20\nEnd\n')
            nodes = np.vstack([TRIANGLE[:3], TRIANGLE[:3]])
            faces, tags = np.array([[0, 1, 2], [3, 5, 4]]), np.array([10, 20])
            with patch("Aster.source_contract.saved_pressures", return_value=({}, 0.)):
                contract = source_springs(mesh, nodes, faces, tags)
                np.testing.assert_array_equal(contract["joint_slave_tri3"], [[0, 1, 2]])
                np.testing.assert_array_equal(contract["joint_master_tri3"], [[3, 4, 5]])
                np.testing.assert_array_equal(contract["joint_coefficients_n_per_m3"], [[2000, 3000, 4000]])
                np.testing.assert_allclose(contract["stiffness"].sum(0), [6000, 9000, 12000])
                with self.assertRaisesRegex(ValueError, "triangulations"):
                    source_springs(mesh, nodes, np.vstack([faces, faces[1]]), np.array([10, 20, 20]))
                nodes[5, 0] += 1
                with self.assertRaisesRegex(ValueError, "Nonmatching"):
                    source_springs(mesh, nodes, faces, tags)


if __name__ == "__main__":
    unittest.main()
