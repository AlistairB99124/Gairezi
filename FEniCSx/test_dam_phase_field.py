"""Small end-to-end dam-load adapter check independent of the full dam mesh."""
from pathlib import Path
import json
import math
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
    def test_refined_baseline_is_zero_damage_same_mesh_before_cracking_and_preserved_on_failure(self):
        imported, structure, crack, options = self.model()
        with tempfile.TemporaryDirectory() as directory:
            result = solve(imported, structure, [crack], options, Path(directory), refined_baseline=True)
            baseline = result.report["refined_mesh_baseline"]
            self.assertEqual(baseline["scenario_name"], "Refined_NoPhase")
            self.assertEqual(baseline["crack_surface_measure_m2"], 0)
            self.assertEqual(baseline["mesh_nodes"], result.report["mesh_nodes"])
            self.assertEqual(baseline["mesh_cells"], result.report["mesh_cells"])
            self.assertEqual(
                result.report["matched_point_stresses"]["refined_baseline_tensile_peak"]["coordinate_xyz_m"],
                baseline["stress_recovery"]["tensile_peak_centroid_xyz_m"],
            )
            self.assertLessEqual(baseline["relative_true_residual"], options.mechanics_relative_tolerance)
            self.assertLess(baseline["relative_force_balance_error"], 1e-6)
            self.assertTrue((Path(directory) / "refined_baseline" / "baseline_report.json").is_file())
            self.assertFalse((Path(directory) / "refined_baseline" / "damage.pvd").exists())
        with tempfile.TemporaryDirectory() as directory, patch(
            "dam_phase_field.solve_staggered", side_effect=RuntimeError("cracking failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "cracking failed"):
                solve(imported, structure, [crack], options, Path(directory), refined_baseline=True)
            saved = json.loads((Path(directory) / "refined_baseline" / "baseline_report.json").read_text())
            self.assertEqual(saved["scenario_name"], "Refined_NoPhase")
            self.assertFalse((Path(directory) / "phase_field_report.json").exists())

    def test_experimental_amor_adapter_preserves_equilibrium_and_matching_stresses(self):
        from compression_preserving import stress
        from phase_field_solver import elastic_parameters

        imported, structure, crack, options = self.model()
        with tempfile.TemporaryDirectory() as directory:
            result = solve(
                imported, structure, [crack], options, Path(directory),
                mechanics_formulation="experimental_amor",
            )
        self.assertEqual(result.report["mechanics_formulation"], "experimental_amor")
        self.assertFalse(result.report["engineering_validated"])
        self.assertLess(result.report["relative_force_balance_error"], 1e-6)
        _, mu, bulk = elastic_parameters(
            structure.material.youngs_modulus_pa, structure.material.poissons_ratio,
        )
        tensor_space = fem.functionspace(imported.domain, ("DG", 0, (3, 3)))
        tensor = fem.Function(tensor_space)
        tensor.interpolate(fem.Expression(
            stress(result.displacement, result.damage, bulk, mu, crack.residual_stiffness),
            tensor_space.element.interpolation_points,
        ))
        self.assertAlmostEqual(
            result.report["raw_element_stress"]["maximum_principal_tensile_stress_pa"],
            np.linalg.eigvalsh(tensor.x.array.reshape(-1, 3, 3))[:, -1].max(),
        )

    def test_reused_mechanics_problem_updates_loads_at_every_step(self):
        from dolfinx.fem.petsc import LinearProblem

        imported, structure, crack, options = self.model()
        parameters = PhaseFieldParameters(
            length_scale_m=crack.length_scale_m,
            critical_energy_release_rate_j_m2=1e9,
            internal_pressure_pa=0,
            load_steps=3,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "phase_field_solver.LinearProblem", wraps=LinearProblem,
        ) as constructor:
            result = solve(imported, structure, [], options, Path(directory), parameters=parameters)
        self.assertEqual(constructor.call_count, 1)
        steps = result.report["steps"]
        final_displacement = steps[-1]["maximum_displacement_m"]
        self.assertGreater(final_displacement, 0)
        for step in steps:
            self.assertAlmostEqual(
                step["maximum_displacement_m"] / final_displacement,
                step["load_factor"], places=6,
            )
            for iteration in step["iterations"]:
                self.assertLessEqual(
                    iteration["mechanics_relative_true_residual"],
                    options.mechanics_relative_tolerance,
                )
                self.assertLess(iteration["damage_solver"]["gradient_norm"], 1e-6)

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
        values = np.where(midpoints[:, 2] < 0.34, 2, 1).astype(np.int32)
        imported.cell_tags = mesh.meshtags(domain, domain.topology.dim, cells, values)
        crack = replace(crack, centre_m=(0.5, 0.5, 0.6))
        wall = imported.cell_tags.find(1)
        bedrock = imported.cell_tags.find(2)
        report = resolution_report(domain, crack, wall)
        self.assertLessEqual(report["cells_in_damaged_band"], len(wall))
        options = replace(options, minimum_elements_per_length_scale=0.7, maximum_refinement_levels=2)
        with patch("crack_field.mesh.compute_incident_entities", wraps=mesh.compute_incident_entities) as marking:
            refined = refine_for_phase_field(imported, [crack], options)
        marked = marking.call_args_list[0].args[1]
        self.assertTrue(set(marked).issubset(set(wall)))
        self.assertFalse(set(marked) & set(bedrock))
        self.assertGreater(refined.report["levels"][0]["bedrock_cells"], 0)
        self.assertGreater(refined.report["levels"][0]["marked_wall_cells"], 0)
        self.assertGreater(refined.report["levels"][-1]["wall_cells"], len(wall))

    def test_bedrock_damage_stays_exactly_zero_during_real_solve(self):
        imported, structure, crack, options = self.model()
        domain = imported.domain
        dimension = domain.topology.dim
        cells = imported.cell_tags.indices
        midpoints = mesh.compute_midpoints(domain, dimension, cells)
        values = np.where(midpoints[:, 2] < 0.34, 2, 1).astype(np.int32)
        imported.cell_tags = mesh.meshtags(domain, dimension, cells, values)
        crack = replace(crack, centre_m=(0.5, 0.5, 0.6), length_scale_m=0.4)
        for formulation, acceleration in (("hybrid", False), ("experimental_amor", True)):
            with self.subTest(formulation=formulation), tempfile.TemporaryDirectory() as directory:
                result = solve(
                    imported, structure, [crack], options, Path(directory),
                    mechanics_formulation=formulation, damage_acceleration=acceleration,
                )
                bedrock_dofs = fem.locate_dofs_topological(
                    result.damage.function_space, dimension, imported.cell_tags.find(2),
                )
                self.assertGreater(len(bedrock_dofs), 0)
                np.testing.assert_array_equal(result.damage.x.array[bedrock_dofs], 0)
                self.assertGreater(result.damage.x.array.max(), 0)
                self.assertEqual(result.report["damage_region_ids"], [1])

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
        cell_tags = mesh.meshtags(domain, dimension, cells, np.full(len(cells), 1, dtype=np.int32))
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
            upstream_surface_z_m=0.0,
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
        self.assertAlmostEqual(result.report["implied_homogeneous_tensile_strength_pa"], expected)
        self.assertEqual(result.report["phase_field_model"], "AT2")
        self.assertFalse(result.report["engineering_validated"])

    def test_at1_dam_adapter_reports_model_specific_strength_and_crack_measure(self):
        imported, structure, crack, options = self.model()
        crack = replace(crack, model="AT1")
        with tempfile.TemporaryDirectory() as directory:
            result = solve(imported, structure, [crack], options, Path(directory))
        self.assertEqual(result.report["phase_field_model"], "AT1")
        self.assertAlmostEqual(
            result.report["implied_homogeneous_tensile_strength_pa"], math.sqrt(3 * 1000.0 * 1.0 / (8 * 0.3)),
        )
        self.assertNotIn("at2_implied_homogeneous_tensile_strength_pa", result.report)
        self.assertGreater(result.report["crack_surface_measure_m2"], 0.0)
        self.assertTrue(any("Hybrid AT1" in text for text in result.report["limitations"]))


if __name__ == "__main__":
    unittest.main()
