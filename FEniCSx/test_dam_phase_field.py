"""Small end-to-end dam-load adapter check independent of the full dam mesh."""
from pathlib import Path
from dataclasses import replace
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np

from crack_field import (
    CrackSpecification, PhaseFieldParameters, PhaseFieldSolverOptions,
    refine_for_phase_field, resolution_report,
)
from dam_phase_field import implied_at2_tensile_strength, solve


class DamPhaseFieldTests(unittest.TestCase):
    def test_initially_intact_baseline_evolves_damage_without_seeding_a_crack(self):
        imported, structure, crack, options = self.model()
        parameters = PhaseFieldParameters(
            length_scale_m=crack.length_scale_m,
            critical_energy_release_rate_j_m2=crack.critical_energy_release_rate_j_m2,
            internal_pressure_pa=0,
            load_steps=2,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "phase_field_solver.crack_phase_field",
            side_effect=AssertionError("An intact baseline must not seed a crack"),
        ):
            result = solve(imported, structure, [], options, Path(directory), parameters=parameters)
        self.assertEqual(result.report["initial_damage"], "zero everywhere; no seeded crack")
        self.assertEqual(result.report["crack_placement"], [])
        self.assertGreater(result.damage.x.array.max(), 0)
        self.assertLess(result.damage.x.array.max(), 1)
        self.assertEqual(len(result.report["steps"]), 2)
        self.assertEqual(result.report["internal_pressure_model"], "disabled (0 Pa)")

    def test_refinement_marks_only_wall_and_reports_both_regions(self):
        imported, _, crack, options = self.model()
        domain = imported.domain
        cells = imported.cell_tags.indices
        midpoints = mesh.compute_midpoints(domain, domain.topology.dim, cells)
        values = np.where(midpoints[:, 2] < 0.34, 1, 2).astype(np.int32)
        imported.cell_tags = mesh.meshtags(domain, domain.topology.dim, cells, values)
        crack = replace(crack, centre_m=(0.5, 0.5, 0.6))
        wall = imported.cell_tags.find(2)
        plinth = imported.cell_tags.find(1)
        report = resolution_report(domain, crack, wall)
        self.assertLessEqual(report["cells_in_damaged_band"], len(wall))
        options = replace(options, minimum_elements_per_length_scale=0.7, maximum_refinement_levels=2)
        with patch("crack_field.mesh.compute_incident_entities", wraps=mesh.compute_incident_entities) as marking:
            refined = refine_for_phase_field(imported, [crack], options)
        marked = marking.call_args_list[0].args[1]
        self.assertTrue(set(marked).issubset(set(wall)))
        self.assertFalse(set(marked) & set(plinth))
        self.assertGreater(refined.report["levels"][0]["plinth_cells"], 0)
        self.assertEqual(refined.report["levels"][0]["marked_plinth_cells"], 0)
        self.assertGreater(refined.report["levels"][-1]["wall_cells"], len(wall))

    def test_plinth_damage_stays_exactly_zero_during_real_solve(self):
        imported, structure, crack, options = self.model()
        domain = imported.domain
        dimension = domain.topology.dim
        cells = imported.cell_tags.indices
        midpoints = mesh.compute_midpoints(domain, dimension, cells)
        values = np.where(midpoints[:, 2] < 0.34, 1, 2).astype(np.int32)
        imported.cell_tags = mesh.meshtags(domain, dimension, cells, values)
        crack = replace(crack, centre_m=(0.5, 0.5, 0.6), length_scale_m=0.4)
        with tempfile.TemporaryDirectory() as directory:
            result = solve(imported, structure, [crack], options, Path(directory))
        plinth_dofs = fem.locate_dofs_topological(
            result.damage.function_space, dimension, imported.cell_tags.find(1),
        )
        self.assertGreater(len(plinth_dofs), 0)
        np.testing.assert_array_equal(result.damage.x.array[plinth_dofs], 0)
        self.assertGreater(result.damage.x.array.max(), 0)
        self.assertEqual(result.report["damage_region_id"], 2)

    def test_annular_sector_runs_real_phase_field_solver_and_writes_results(self):
        imported, structure, crack, options = self.model()
        crack = replace(
            crack, shape="annular_sector", size_m=(0.4, 0.6, 0.002),
            annular_sector=(0.5, 0.9, 0.3, 1.2),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = solve(imported, structure, [crack], options, Path(directory))
            self.assertTrue((Path(directory) / "damage.pvd").is_file())
            self.assertTrue((Path(directory) / "phase_field_report.json").is_file())
        self.assertEqual(len(result.report["steps"]), 1)
        self.assertFalse(result.report["engineering_validated"])

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
        self.assertIn("stress_recovery", result.report)
        self.assertIn("raw_element_stress", result.report)
        self.assertGreater(result.report["crack_surface_measure_m2"], 0.0)
        expected = implied_at2_tensile_strength(1000.0, 1.0, 0.3)
        self.assertAlmostEqual(result.report["at2_implied_homogeneous_tensile_strength_pa"], expected)
        self.assertFalse(result.report["engineering_validated"])


if __name__ == "__main__":
    unittest.main()
