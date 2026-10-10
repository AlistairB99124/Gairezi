"""Small grounded-spring and gravity benchmark independent of Elmer output."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from dam_elasticity import solve
from dam_mesh import import_structure, triangles
import test_dam_mesh


class DamElasticityTests(unittest.TestCase):
    def test_gravity_balanced_by_grounded_foundation_springs(self):
        structure = test_dam_mesh.DamMeshTests().cube()
        structure.boundaries = [(1 if i == 0 else 7, face) for i, (_, face) in enumerate(structure.boundaries)]
        structure.material = SimpleNamespace(youngs_modulus_pa=1e7, poissons_ratio=0.0, density_kg_m3=1000.0)
        structure.loads = SimpleNamespace(
            gravity_z_m_s2=-10.0, peak_water_pressure_pa=0.0, maximum_water_height_m=1.0,
            water_density_kg_m3=1000.0, tailwater_head_m=0.0, overflow_head_m=0.0
        )
        support = SimpleNamespace(spring_x_n_per_m3=1e6, spring_y_n_per_m3=1e6, spring_z_n_per_m3=1e6)
        structure.foundation_support = structure.plinth_support = support
        imported = import_structure(structure, 1e-12)
        with tempfile.TemporaryDirectory() as directory:
            displacement, _, report = solve(
                imported, structure, {"relative_tolerance": 1e-12, "maximum_iterations": 500}, Path(directory)
            )
        np.testing.assert_allclose(report["applied_force_n"], [0, 0, -10000], atol=1e-8)
        np.testing.assert_allclose(report["ground_spring_reactions_n"]["1"], [0, 0, 10000], atol=1e-6)
        self.assertLess(report["relative_force_balance_error"], 1e-10)
        self.assertTrue(np.all(displacement[:, 2] < 0))
        bottom = np.asarray(triangles(structure.boundaries[0][1]))
        xyz = structure.nodes[bottom]
        area = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1) / 2
        average = np.average(displacement[bottom, 2].mean(axis=1), weights=area)
        self.assertLess(abs(average + 0.01), 1e-10)
