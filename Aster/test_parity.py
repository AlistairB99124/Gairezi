"""Fast parity-contract tests without a solver installation."""
from pathlib import Path
import io
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from Aster.parity_checks import (
    common_element_stress_checks, fixed_sample_checks, select_stress_samples,
    source_node_ids, surface_clearance,
)
from Aster.elastic_diagnostics import outward_faces, pressure_resultant, volume_energy
from Aster.source_contract import native_projector_audit, nodal_areas, real_value, saved_pressures, sif_boundaries


class SourceContractTests(unittest.TestCase):
    def test_sif_values_are_parsed_not_executed(self):
        blocks = sif_boundaries('Boundary Condition 1\n Name = "test"\n Spring 12 = Real -3e9\nEnd\n')
        self.assertEqual(real_value(blocks[1], "Spring 12"), -3e9)
        with self.assertRaisesRegex(ValueError, "Missing"):
            real_value("Spring 12 = Variable Coordinate 1", "Spring 12")

    def test_lumped_area_conserves_surface_and_constant_spring_force(self):
        xyz = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
        areas = nodal_areas(xyz, np.array([[0, 1, 2]]))
        np.testing.assert_allclose(list(areas.values()), [1., 1., 1.])
        normal = np.array([.6, .8, 0.])
        stiffness = 1e12 * np.outer(normal, normal)
        tangent = np.array([-.8, .6, 0.])
        np.testing.assert_allclose(stiffness @ tangent, 0., atol=1e-4)
        self.assertAlmostEqual(sum(areas.values()), 3.)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            nodal_areas(xyz, np.array([[0, 0, 1]]))

    def test_incident_cells_identify_coincident_joint_sides(self):
        # Nodes 1 and 2 coincide but belong to opposite tetrahedra.
        xyz = np.array([[9., 9., 9.], [0., 0., 0.], [0., 0., 0.],
                        [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., -1.]])
        source = np.array([[1, 3, 4, 5], [2, 3, 4, 6]])
        result_ids = np.array([2, 6, 3, 4, 1, 5])
        points = xyz[result_ids]
        cells = [(10, np.array([0, 2, 3, 1])), (10, np.array([4, 2, 3, 5]))]
        np.testing.assert_array_equal(source_node_ids(points, cells, xyz, source), result_ids)
        with self.assertRaisesRegex(ValueError, "match"):
            source_node_ids(points + 1., cells, xyz, source)

    def test_saved_pressure_datums_are_not_reconstructed_from_geometry(self):
        blocks = {
            1: 'Name = "UpstreamHydrostaticPressure"\nNormal Force = Variable Coordinate 3\n'
               'Real MATC "-9810 * (3 - tx) * (tx < 3)"',
            2: 'Name = "DownstreamTailwater"\nNormal Force = Variable Coordinate 3\n'
               'Real MATC "-9810 * (-26 - tx) * (tx < -26)"',
            3: 'Name = "CrestOverflowSurcharge"\nNormal Force = Real -29430',
        }
        pressures, crest = saved_pressures(blocks)
        self.assertEqual(pressures["DownstreamTailwater"]["surface_z_m"], -26.)
        self.assertEqual(pressures["UpstreamHydrostaticPressure"]["gradient_pa_per_m"], 9810.)
        self.assertEqual(crest, 29430.)
        blocks[2] = blocks[2].replace("tx < -26", "tx < -27")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            saved_pressures(blocks)

    def test_affine_strain_energy_and_rigid_rotation(self):
        points = np.array([[.2, .1, .4], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]])
        cells = np.array([[0, 1, 2, 3]])
        grad = np.array([[.01, .02, .03], [.04, -.01, .02], [.05, -.03, .01]])
        strain = .5 * (grad + grad.T)
        young, poisson = 35e9, .2
        mu = young / (2 * (1 + poisson))
        lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        density = .5 * lam * np.trace(strain) ** 2 + mu * (strain ** 2).sum()
        energy, volume = volume_energy(points, cells, points @ grad.T + 7., young, poisson)
        self.assertAlmostEqual(energy / volume, density, places=5)
        rotation = np.array([[0., -1., 2.], [1., 0., -3.], [-2., 3., 0.]])
        energy, _ = volume_energy(points, cells, points @ rotation.T + 5., young, poisson)
        self.assertLess(energy, 1e-16)

    def test_pressure_clipping_integrates_force_and_moment(self):
        points = np.array([[0., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
        force, moment = pressure_resultant(points, np.array([[0, 1, 2]]), 1., 1.)
        np.testing.assert_allclose(force, [-5/6, 0., 0.])
        np.testing.assert_allclose(moment, [0., -1/4, 17/24])

    def test_face_orientation_comes_from_incident_volume_not_input_winding(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        cells = np.array([[0, 1, 2, 3]])
        for face in ([[0, 1, 2]], [[0, 2, 1]]):
            oriented = outward_faces(points, np.array(face), cells)
            force, _ = pressure_resultant(points, oriented, 0., 0., constant=2.)
            np.testing.assert_allclose(force, [0., 0., 1.])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            outward_faces(points, np.array([[0, 1, 2]]), np.concatenate([cells, cells]))

    def test_common_element_stresses_ignore_nodal_recovery_and_use_volume_weights(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        cells = np.array([[0, 1, 2, 3]])
        displacement = points * .01
        result = common_element_stress_checks(
            points, cells, np.array([1]), displacement, 1.01 * displacement, 100., .2,
        )["wall"]
        self.assertAlmostEqual(result["relative_volume_weighted_l2_tensor_difference"], .01)
        np.testing.assert_allclose(result["volume_weighted_mean_stress_tensor_pa"]["elmer"],
                                   np.eye(3) * (100 * .01 / (1 - 2 * .2)))

    def test_timeout_stops_only_launched_solver_tree_and_preserves_logs(self):
        from Aster.run_case import run_solver
        process = MagicMock()
        process.pid = 12345
        process.communicate.side_effect = [subprocess.TimeoutExpired("solver", 1), ("out", "err")]
        with tempfile.TemporaryDirectory() as temp, \
             patch("Aster.run_case.subprocess.Popen") as popen, \
             patch("Aster.run_case.subprocess.run") as terminate, \
             patch("Aster.run_case.os.name", "nt"):
            popen.return_value.__enter__.return_value = process
            directory = Path(temp)
            with self.assertRaises(subprocess.TimeoutExpired):
                run_solver(["solver"], directory, 1)
            terminate.assert_called_once_with(
                ["taskkill", "/PID", "12345", "/T", "/F"], check=True, capture_output=True, timeout=30)
            self.assertEqual((directory / "stdout.log").read_text(), "out")

    def test_live_log_streams_output_before_the_solver_exits(self):
        from Aster.run_case import run_solver
        script = ("import sys, time\n"
                  "print('first', flush=True)\n"
                  "sys.stderr.write('warn\\n'); sys.stderr.flush()\n"
                  "time.sleep(3)\n"
                  "print('second', flush=True)\n")
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            live = directory / "solver.live.log"
            seen_before_exit = []

            def watch():
                import time
                deadline = time.monotonic() + 2.5
                while time.monotonic() < deadline:
                    if live.is_file() and "first" in live.read_text(encoding="utf-8"):
                        seen_before_exit.append(True)
                        return
                    time.sleep(0.05)

            import threading
            watcher = threading.Thread(target=watch)
            watcher.start()
            completed = run_solver([sys.executable, "-c", script], directory, 60, live_log=live)
            watcher.join()
            self.assertEqual(seen_before_exit, [True])
            self.assertEqual(completed.returncode, 0)
            self.assertEqual(completed.stdout.splitlines(), ["first", "second"])
            self.assertEqual(completed.stderr, "warn\n")
            self.assertEqual(sorted(live.read_text(encoding="utf-8").splitlines()), ["first", "second", "warn"])

    def test_live_log_follows_run_aster_work_file_without_duplicating_its_dump(self):
        from Aster.run_case import run_solver
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            work = directory / "run_aster_work"
            work.mkdir()
            # Mimics Windows run_aster: names its work dir, the solver appends to
            # fort.6, then the launcher prints the whole file once the solve ends.
            script = ("import sys, time\n"
                      f"work = {str(work)!r}\n"
                      "print('Prepare environment in ' + work, flush=True)\n"
                      "with open(work + '/fort.6', 'a') as f:\n"
                      "    f.write('row 1\\n'); f.flush(); time.sleep(3)\n"
                      "    f.write('row 2\\n'); f.flush()\n"
                      "print(open(work + '/fort.6').read(), end='', flush=True)\n"
                      "print('EXECUTION_CODE_ASTER_EXIT_1=0', flush=True)\n")
            live = directory / "solver.live.log"
            seen_before_exit = []

            def watch():
                import time
                deadline = time.monotonic() + 2.5
                while time.monotonic() < deadline:
                    if live.is_file() and "row 1" in live.read_text(encoding="utf-8"):
                        seen_before_exit.append(True)
                        return
                    time.sleep(0.05)

            import threading
            watcher = threading.Thread(target=watch)
            watcher.start()
            completed = run_solver([sys.executable, "-c", script], directory, 60, live_log=live)
            watcher.join()
            self.assertEqual(seen_before_exit, [True])
            self.assertEqual(live.read_text(encoding="utf-8").splitlines(),
                             [f"Prepare environment in {work}", "row 1", "row 2"])
            self.assertIn("row 2\nEXECUTION_CODE_ASTER_EXIT_1=0", completed.stdout)

    def test_live_log_timeout_stops_tree_and_preserves_logs(self):
        from Aster.run_case import run_solver
        script = "import time\nprint('started', flush=True)\ntime.sleep(60)\n"
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaises(subprocess.TimeoutExpired):
                run_solver([sys.executable, "-c", script], directory, 2, live_log=directory / "live.log")
            self.assertEqual((directory / "stdout.log").read_text(encoding="utf-8"), "started\n")
            self.assertEqual((directory / "live.log").read_text(encoding="utf-8"), "started\n")

    def test_verifier_starts_without_a_prior_freezer_import(self):
        completed = subprocess.run(
            [sys.executable, "-m", "Aster.verify_parity", "--help"],
            cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_surface_clearance_includes_face_interior_and_edges(self):
        triangle = np.array([[[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]])
        self.assertAlmostEqual(surface_clearance(np.array([.5, .5, 1.]), triangle), 1.)
        self.assertAlmostEqual(surface_clearance(np.array([2., 2., 0.]), triangle), np.sqrt(2.))
        self.assertAlmostEqual(surface_clearance(np.array([-1., -1., 0.]), triangle), np.sqrt(2.))

    def test_fixed_samples_use_original_positions_and_reject_insufficient_clearance(self):
        points = np.array([[0., 0., -2.], [4., 0., 0.], [0., 4., 0.], [0., 0., 2.]])
        cells, bodies = np.array([[0, 1, 2, 3]]), np.array([3])
        contract = select_stress_samples(points, cells, bodies)
        displacement = .01 * points
        checks = fixed_sample_checks(points, cells, bodies, displacement,
                                     1.01 * displacement, 100., .2, contract)
        sample = checks["foundation_interior_50"]
        self.assertAlmostEqual(sample["relative_tensor_difference"], .01)
        np.testing.assert_allclose(sample["xyz_m"], [1., 1., 0.])
        np.testing.assert_allclose(sample["elmer_displacement_m"], [.01, .01, 0.])
        contract["samples"][0]["xyz_m"] = [0., 0., 0.]
        with self.assertRaisesRegex(ValueError, "clearance"):
            fixed_sample_checks(points, cells, bodies, displacement, displacement, 100., .2, contract)
        with self.assertRaisesRegex(ValueError, "clearance"):
            select_stress_samples(points, cells, bodies, minimum_clearance_m=10.)

    def test_fixed_sample_on_internal_face_retains_both_element_tensors(self):
        points = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.],
                           [0., 0., 2.], [0., 0., -2.]])
        cells = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
        contract = {"schema_version": 1, "coordinate_frame": "undeformed", "units": "m",
                    "samples": [{"name": "test", "body_id": 3, "xyz_m": [.5, .5, 0.],
                                 "minimum_clearance_m": .1}]}
        displacement = .01 * points
        checks = fixed_sample_checks(points, cells, np.array([3, 3]), displacement,
                                     displacement, 100., .2, contract)
        self.assertEqual(checks["test"]["incident_cell_count"], 2)
        self.assertEqual(checks["test"]["relative_tensor_difference"], 0.)

    def test_native_projector_audit_distinguishes_consistent_mass_from_lumping(self):
        xyz = np.array([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            (model / "dam_model.sif").write_text(
                'Boundary Condition 1\nName = "test_spring"\nTarget Boundaries(1) = 1\n'
                'Mortar BC = Integer 2\nSave Projector = Logical True\nEnd\n'
                'Boundary Condition 2\nTarget Boundaries(1) = 2\nEnd\n')
            part = model / "mesh" / "partitioning.1"
            part.mkdir(parents=True)
            (part / "part.1.nodes").write_text("\n".join(
                f"{i} -1 {p[0]} {p[1]} {p[2]}" for i, p in enumerate(xyz[1:], 1)))
            entries = []
            for i in range(3):
                for j in range(3):
                    value = .5 / 12. * (1. + (i == j))
                    entries += [f"{i+1} {j+1} {value}", f"{i+1} {j+4} {-value}"]
            (model / "p1_part0.dat").write_text("\n".join(entries))
            report = native_projector_audit(model / "three_body.msh", xyz,
                                            np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 2]))
            self.assertEqual(report["exported_interface_count"], 1)
            self.assertLess(report["maximum_relative_consistent_mass_difference"], 1e-14)
            self.assertGreater(report["maximum_relative_lumped_mass_difference"], .5)
            (model / "p1_part0.dat").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                native_projector_audit(model / "three_body.msh", xyz,
                                      np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 2]))

    def test_verifier_does_not_accept_unverified_baseline_even_if_preliminary_checks_pass(self):
        from Aster import verify_parity
        with patch.object(verify_parity, "verify", return_value={
            "baseline_verified": False, "checks": {"force_balance": True},
        }), patch.object(sys, "argv", ["verify_parity", "test-output"]), \
                patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as error:
                verify_parity.main()
            self.assertEqual(error.exception.code, 2)

    def test_pipeline_exit_status_distinguishes_comparison_from_acceptance(self):
        from Aster import run_three_body_pipeline
        for status in ("requires_verification", "solve_failed", "verified"):
            with self.subTest(status=status), \
                    patch.object(run_three_body_pipeline, "run_pipeline", return_value={"status": status}), \
                    patch.object(sys, "argv", ["run_three_body_pipeline"]), \
                    patch("sys.stdout", new_callable=io.StringIO):
                if status == "verified":
                    run_three_body_pipeline.main()
                else:
                    with self.assertRaises(SystemExit) as error:
                        run_three_body_pipeline.main()
                    self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
