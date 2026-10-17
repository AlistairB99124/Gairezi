"""Tests for the generated civil-engineering phase-field report."""
import tempfile
from pathlib import Path
import unittest

from engineering_report import build_engineering_results, write_engineering_results


class EngineeringReportTests(unittest.TestCase):
    def setUp(self):
        self.configuration = {
            "material": {
                "concrete": {
                    "density": 2400,
                    "youngs_modulus": 35e9,
                    "poisson_ratio": 0.2,
                    "tensile_strength_pa": 2e6,
                }
            },
            "loads": {
                "gravity": -9.81,
                "water_height": 31.0,
                "tailwater_head": 3.0,
            },
            "crack_simulations": [
                {
                    "central_crack_pos_m": {"x": 76.743, "y": 13.947, "z": -25.0},
                    "size_m": {"x": 3.0, "y": 5.0, "z": 0.02},
                    "orientation": {
                        "normal_x": 0.0,
                        "normal_y": 0.0,
                        "normal_z": 1.0,
                    },
                    "phase_field_parameters": {
                        "critical_energy_release_rate_J_m2": 2700.0,
                        "length_scale_m": 0.1,
                        "residual_stiffness": 1e-8,
                    },
                    "loading": {"internal_pressure_pa": 0.0},
                }
            ],
        }
        self.result = {
            "model": "hybrid AT2 phase-field fracture",
            "energy_split": "volumetric_deviatoric",
            "internal_pressure_model": "disabled (0 Pa)",
            "steps": [
                {
                    "load_factor": 0.1,
                    "maximum_displacement_m": 0.000953,
                    "elastic_energy_j": 3535.0,
                    "fracture_energy_j": 45126.0,
                    "staggered_iterations": 2,
                },
                {
                    "load_factor": 1.0,
                    "maximum_displacement_m": 0.0095851,
                    "elastic_energy_j": 355040.0,
                    "fracture_energy_j": 45874.0,
                    "staggered_iterations": 3,
                },
            ],
            "configured_tensile_strength_pa": 2e6,
            "maximum_principal_tensile_stress_pa": 1.815e6,
            "maximum_degraded_stress_pa": 3.254e6,
            "at2_implied_homogeneous_tensile_strength_pa": 9.983e6,
            "implied_to_configured_strength_ratio": 4.9915,
            "crack_surface_measure_m2": 16.874,
            "applied_force_n": [-393.791e6, -115.593e6, -328.684e6],
            "ground_spring_reactions_n": {
                "1": [84.331e6, 22.280e6, 64.804e6],
                "8": [309.460e6, 93.313e6, 263.879e6],
            },
            "force_imbalance_n": [0.0737, -0.3063, -0.2756],
            "relative_force_balance_error": 7.96e-10,
            "limitations": [
                "No unilateral crack-face contact or friction is enforced."
            ],
        }
        self.refinement = {
            "status": "preliminary_under_resolved",
            "recommended_minimum_elements_per_length_scale": 2.0,
            "levels": [
                {
                    "cracks": [
                        {
                            "elements_per_length_scale": 0.915,
                            "maximum_cell_diameter_near_crack_m": 0.1093,
                        }
                    ]
                }
            ],
        }
        self.placement = [{"valid": True}]
        self.import_report = {
            "imported_nodes": 123337,
            "tetrahedra": 614824,
            "tetrahedral_volume_m3": 13097.91,
        }

    def build(self):
        return build_engineering_results(
            self.configuration,
            self.result,
            self.refinement,
            self.placement,
            self.import_report,
            configuration_sha256="abc123",
            environment={"python": "3.14", "dolfinx": "0.11.0", "petsc": [3, 24, 4]},
            started_at_utc="20261012T065918506145Z",
            completed_at_utc="2026-10-12T07:09:09+00:00",
        )

    def test_report_explains_key_results_units_and_limitations(self):
        report = self.build()
        self.assertIn("**9.59 mm**", report)
        self.assertIn("**90.8%**", report)
        self.assertIn("preliminary_under_resolved", report)
        self.assertIn("Damage is not crack opening", report)
        self.assertIn("dam-safety determination", report)
        self.assertIn("[damage.pvd](./damage.pvd)", report)
        self.assertIn("Configuration SHA-256: `abc123`", report)

    def test_writer_creates_results_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_engineering_results(
                Path(directory),
                self.configuration,
                self.result,
                self.refinement,
                self.placement,
                self.import_report,
                configuration_sha256="abc123",
                environment={},
                started_at_utc="start",
                completed_at_utc="finish",
            )
            self.assertEqual(path.name, "results.md")
            self.assertEqual(path.read_text(encoding="utf-8"), self.build().replace(
                "- Python: `3.14`.\n"
                "- DOLFINx: `0.11.0`.\n"
                "- PETSc: `[3, 24, 4]`.",
                "- Python: `not recorded`.\n"
                "- DOLFINx: `not recorded`.\n"
                "- PETSc: `not recorded`.",
            ).replace("`20261012T065918506145Z`", "`start`").replace(
                "`2026-10-12T07:09:09+00:00`", "`finish`"
            ))

    def test_mesh_controlled_comparison_and_matched_point_context(self):
        self.result["maximum_principal_compressive_stress_pa"] = -3e6
        self.result["refined_mesh_baseline"] = {
            "maximum_principal_tensile_stress_pa": 1.7e6,
            "maximum_principal_compressive_stress_pa": -2.9e6,
        }
        self.result["matched_point_stresses"] = {
            "refined_baseline_tensile_peak": {
                "coordinate_xyz_m": [78, 17, -27.875], "principal_stresses_pa": [-1e5, 0, 2e5],
            },
            "refined_baseline_compressive_peak": {
                "coordinate_xyz_m": [78, 15, -0.25], "principal_stresses_pa": [-3e6, -2e5, 0],
            },
        }
        report = self.build()
        self.assertIn("Mesh-controlled crack comparison", report)
        self.assertIn("| Tension | 1.7e+06 | 1.815e+06 | 115000 |", report)
        self.assertIn("(78.000, 17.000, -27.875)", report)
        self.assertIn("their difference is not local stress relief", report)


if __name__ == "__main__":
    unittest.main()
