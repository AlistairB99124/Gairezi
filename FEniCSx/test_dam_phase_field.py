"""Small end-to-end dam-load adapter check independent of the full dam mesh."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from dolfinx import mesh
from mpi4py import MPI
import numpy as np

from crack_field import CrackSpecification, PhaseFieldSolverOptions
from dam_phase_field import implied_at2_tensile_strength, solve


class DamPhaseFieldTests(unittest.TestCase):
    def model(self):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
        dimension = domain.topology.dim
        cells = np.arange(domain.topology.index_map(dimension).size_local, dtype=np.int32)
        cell_tags = mesh.meshtags(domain, dimension, cells, np.full(len(cells), 2, dtype=np.int32))
        domain.topology.create_connectivity(dimension - 1, dimension)
        facets = mesh.exterior_facet_indices(domain.topology)
        facet_tags = mesh.meshtags(
            domain,
            dimension - 1,
            facets,
            np.ones(len(facets), dtype=np.int32),
        )
        imported = SimpleNamespace(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
        material = SimpleNamespace(
            density_kg_m3=1.0,
            youngs_modulus_pa=1000.0,
            poissons_ratio=0.2,
            tensile_strength_pa=10.0,
        )
        loads = SimpleNamespace(
            gravity_z_m_s2=-1.0,
            water_density_kg_m3=1.0,
            tailwater_head_m=0.0,
            overflow_head_m=0.0,
        )
        support = SimpleNamespace(
            spring_x_n_per_m3=1000.0,
            spring_y_n_per_m3=1000.0,
            spring_z_n_per_m3=1000.0,
        )
        structure = SimpleNamespace(
            material=material,
            loads=loads,
            foundation_support=support,
            plinth_support=support,
        )
        crack = CrackSpecification(
            type="phase_field",
            shape="rectangular",
            centre_m=(0.5, 0.5, 0.5),
            size_m=(0.4, 0.4, 0.02),
            normal=(0.0, 0.0, 1.0),
            length_scale_m=0.3,
            critical_energy_release_rate_j_m2=1.0,
            internal_pressure_pa=0.0,
            residual_stiffness=1e-6,
            load_steps=1,
        )
        options = PhaseFieldSolverOptions(
            maximum_staggered_iterations=20,
            staggered_tolerance=1e-5,
            mechanics_relative_tolerance=1e-10,
            mechanics_maximum_iterations=500,
            damage_gradient_tolerance=1e-8,
            damage_maximum_iterations=500,
        )
        return imported, structure, crack, options

    def test_dam_adapter_writes_fields_force_balance_and_calibration_diagnostics(self):
        imported, structure, crack, options = self.model()
        with tempfile.TemporaryDirectory() as directory:
            result = solve(imported, structure, [crack], options, Path(directory))
            names = {path.name for path in Path(directory).iterdir()}
        self.assertTrue(
            {"degraded_stress.pvd", "strain.pvd", "crack_surface_density.pvd"}.issubset(names)
        )
        self.assertTrue(result.report["crack_placement"][0]["valid"])
        self.assertIn("relative_force_balance_error", result.report)
        self.assertIn("maximum_principal_tensile_stress_pa", result.report)
        self.assertGreater(result.report["crack_surface_measure_m2"], 0.0)
        expected = implied_at2_tensile_strength(1000.0, 1.0, 0.3)
        self.assertAlmostEqual(result.report["at2_implied_homogeneous_tensile_strength_pa"], expected)
        self.assertFalse(result.report["engineering_validated"])


if __name__ == "__main__":
    unittest.main()
