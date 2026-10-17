"""Serial/MPI parity and failure propagation for the isolated mesh worker."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from dolfinx import io
import numpy as np

from dam_phase_field import solve
from parallel_fracture import solve_parallel
from run import LOAD_CASE, submitted_structure
import test_dam_phase_field


class ParallelFractureTests(unittest.TestCase):
    def test_two_process_worker_matches_serial_fields_and_recovered_stress(self):
        self.check_parity("hybrid")

    def test_experimental_amor_two_process_worker_matches_serial(self):
        self.check_parity("experimental_amor")

    def test_experimental_spectral_two_process_worker_matches_serial(self):
        self.check_parity("experimental_spectral")

    def check_parity(self, mechanics_formulation):
        imported, _, crack, options = test_dam_phase_field.DamPhaseFieldTests().model()
        structure = submitted_structure(LOAD_CASE)
        crack = replace(crack, critical_energy_release_rate_j_m2=2700, load_steps=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            serial = root / "serial"
            parallel = root / "parallel"
            parallel.mkdir()
            (parallel / "config.json").write_bytes(LOAD_CASE.read_bytes())
            with io.XDMFFile(imported.domain.comm, parallel / "mesh.xdmf", "w") as writer:
                writer.write_mesh(imported.domain)
                imported.cell_tags.name = "regions"
                imported.facet_tags.name = "boundaries"
                writer.write_meshtags(imported.cell_tags, imported.domain.geometry)
                writer.write_meshtags(imported.facet_tags, imported.domain.geometry)
            expected = solve(
                imported, structure, [crack], options, serial,
                mechanics_formulation=mechanics_formulation,
                damage_acceleration=mechanics_formulation != "hybrid",
                final_load_substeps=2,
                refined_baseline=True,
            ).report
            actual = solve_parallel(
                parallel, [crack], options, 2, mechanics_formulation=mechanics_formulation,
                damage_acceleration=mechanics_formulation != "hybrid",
                final_load_substeps=2,
            )
            self.assertEqual(actual["mechanics_formulation"], mechanics_formulation)
            self.assertEqual(actual["mechanics_solver"]["mpi_processes"], 2)
            for key in ("maximum_principal_tensile_stress_pa", "maximum_principal_compressive_stress_pa",
                        "relative_force_balance_error"):
                np.testing.assert_allclose(
                    actual["refined_mesh_baseline"][key], expected["refined_mesh_baseline"][key],
                    rtol=1e-5, atol=1e-8,
                )
            self.assertEqual(actual["refined_mesh_baseline"]["mesh_cells"], actual["mesh_cells"])
            for name in ("refined_baseline_tensile_peak", "refined_baseline_compressive_peak"):
                # Extrema may tie on symmetric test meshes; compare stresses rather than tie location.
                np.testing.assert_allclose(
                    actual["matched_point_stresses"][name]["principal_stresses_pa"],
                    expected["matched_point_stresses"][name]["principal_stresses_pa"], rtol=1e-4, atol=1e-5,
                )
            self.assertTrue((parallel / "refined_baseline" / "baseline_report.json").is_file())
            for key in (
                "maximum_principal_tensile_stress_pa",
                "maximum_principal_compressive_stress_pa",
                "maximum_degraded_stress_pa",
            ):
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-5, atol=1e-8)
            for left, right in zip(actual["steps"], expected["steps"]):
                for key in ("maximum_displacement_m", "maximum_damage", "total_internal_energy_j"):
                    np.testing.assert_allclose(left[key], right[key], rtol=1e-5, atol=1e-8)
                last = left["iterations"][-1]
                location = last["damage_change_location"]
                np.testing.assert_allclose(
                    location["maximum_absolute_change"], last["maximum_damage_change"], atol=1e-12,
                )
                self.assertEqual(len(location["coordinate_xyz_m"]), 3)
                self.assertGreaterEqual(location["candidate_damage"], location["previous_load_damage"] - 1e-12)
                self.assertLessEqual(location["candidate_damage"], 1.0 + 1e-12)
            self.assertTrue((parallel / "displacement.pvd").is_file())
            layouts = sorted(parallel.glob("checkpoint_layout_rank*.npz"))
            states = sorted(parallel.glob("iteration_checkpoint_rank*.npz"))
            self.assertEqual(len(layouts), 2)
            self.assertEqual(len(states), 2)
            for layout_path, state_path in zip(layouts, states):
                with np.load(layout_path) as layout, np.load(state_path) as state:
                    self.assertEqual(int(state["mpi_processes"]), 2)
                    self.assertEqual(int(state["step"]), 3)
                    np.testing.assert_array_equal(layout["coordinates"], state["coordinates"])
                    self.assertEqual(len(layout["cell_input_indices"]), len(layout["damage_cell_dofs"]))
            self.assertFalse(actual["engineering_validated"])

    def test_failed_rank_error_reaches_parent_without_loading_stale_report(self):
        _, _, crack, options = test_dam_phase_field.DamPhaseFieldTests().model()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "parallel_failure_rank1.json").write_text(json.dumps({
                "error": "RuntimeError: Mechanics solve failed at load step 10/10",
            }))
            (directory / "phase_field_report.json").write_text("{}")
            with patch(
                "parallel_fracture.subprocess.run", return_value=SimpleNamespace(returncode=1),
            ) as run, self.assertRaisesRegex(RuntimeError, "Mechanics solve failed at load step 10/10"):
                solve_parallel(directory, [crack], options, 2)
            for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                self.assertEqual(run.call_args.kwargs["env"][variable], "1")


if __name__ == "__main__":
    unittest.main()
