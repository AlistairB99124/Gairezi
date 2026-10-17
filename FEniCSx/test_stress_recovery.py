"""Independent consistent-mass checks for Elmer-compatible peak reporting."""
from pathlib import Path
import tempfile
import unittest

from dolfinx import fem, mesh
import numpy as np
import ufl

from dam_mesh import import_structure
from stress_recovery import recover_stress
from test_dam_mesh import DamMeshTests


class StressRecoveryTests(unittest.TestCase):
    def test_consistent_projection_and_tensor_average_match_dense_reference(self):
        imported = import_structure(DamMeshTests().cube(), 1e-12)
        domain = imported.domain
        space = fem.functionspace(domain, ("DG", 0, (3, 3)))
        stress = fem.Function(space)
        count = domain.topology.index_map(3).size_local
        values = np.zeros((count, 3, 3))
        for cell in range(count):
            values[cell] = [[cell + 1, 0.3 * cell, 0], [0.3 * cell, -2 * cell, 0], [0, 0, -1]]
            stress.x.array.reshape(-1, 3, 3)[space.dofmap.cell_dofs(cell)[0]] = values[cell]
        nodal_space = fem.functionspace(domain, ("Lagrange", 1, (3, 3)))
        dofs = np.asarray([nodal_space.dofmap.cell_dofs(cell) for cell in range(count)])
        xyz = domain.geometry.x[mesh.entities_to_geometry(
            domain, 3, np.arange(count, dtype=np.int32),
        )]
        volumes = np.abs(np.linalg.det(xyz[:, 1:] - xyz[:, :1])) / 6
        nodes = nodal_space.dofmap.index_map.size_local
        mass = np.zeros((nodes, nodes))
        rhs = np.zeros((nodes, 9))
        for cell, ids in enumerate(dofs):
            mass[np.ix_(ids, ids)] += volumes[cell] / 20 * (np.ones((4, 4)) + np.eye(4))
            rhs[ids] += volumes[cell] / 4 * values[cell].reshape(1, 9)
        expected_nodal = np.linalg.solve(mass, rhs)
        expected_elements = expected_nodal[dofs].mean(axis=1).reshape(-1, 3, 3)
        principal = np.linalg.eigvalsh(expected_elements)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = recover_stress(domain, stress, output)
            self.assertTrue((output / "recovered_stress.pvd").is_file())
        self.assertAlmostEqual(report["maximum_principal_tensile_stress_pa"], principal[:, -1].max(), places=9)
        self.assertAlmostEqual(report["maximum_principal_compressive_stress_pa"], principal[:, 0].min(), places=9)
        self.assertFalse(np.isclose(principal[:, -1].max(), np.linalg.eigvalsh(values)[:, -1].max()))

    def test_constant_stress_is_reproduced_exactly(self):
        imported = import_structure(DamMeshTests().cube(), 1e-12)
        tensor = np.array([[2e6, 0.3e6, 0], [0.3e6, -3e6, 0], [0, 0, 1e6]])
        with tempfile.TemporaryDirectory() as temporary:
            report = recover_stress(imported.domain, ufl.as_matrix(tensor.tolist()), Path(temporary))
        principal = np.linalg.eigvalsh(tensor)
        np.testing.assert_allclose(
            [report["maximum_principal_tensile_stress_pa"], report["maximum_principal_compressive_stress_pa"]],
            [principal[-1], principal[0]], rtol=1e-10,
        )

    def test_coincident_interface_nodes_do_not_share_recovered_stress(self):
        structure = DamMeshTests().cube()
        structure.nodes = np.vstack((structure.nodes, structure.nodes))
        structure.cells.append((5, tuple(range(8, 16))))
        structure.region_ids.append(1)
        structure.boundaries += [(9, tuple(i + 8 for i in face)) for _, face in structure.boundaries.copy()]
        imported = import_structure(structure, 1e-12)
        space = fem.functionspace(imported.domain, ("DG", 0, (3, 3)))
        stress = fem.Function(space)
        for cell, region in zip(imported.cell_tags.indices, imported.cell_tags.values):
            stress.x.array.reshape(-1, 3, 3)[space.dofmap.cell_dofs(int(cell))[0]] = (
                np.eye(3) * (10 if region == 1 else -20)
            )
        with tempfile.TemporaryDirectory() as temporary:
            report = recover_stress(imported.domain, stress, Path(temporary))
        self.assertAlmostEqual(report["maximum_principal_tensile_stress_pa"], 10)
        self.assertAlmostEqual(report["maximum_principal_compressive_stress_pa"], -20)


if __name__ == "__main__":
    unittest.main()
