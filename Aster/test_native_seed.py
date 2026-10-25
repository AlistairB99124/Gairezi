"""Physical UI seed geometry tests independent of the native runtime."""
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from Aster.native_seed import seed_damage, stiffness_ratio
from Aster.seed_benchmark import finite_sector_metrics, history_checks, run, specimen_metrics
from Aster.seed_specimen_mesh import lateral_edges, normal_edges
from FEniCSx.crack_contract import CrackSpecification, freeze_selected_seed


def selected_seed():
    specification = CrackSpecification(
        type="phase_field", shape="annular_sector", centre_m=(1.5, 0., -2.),
        size_m=(1., 1., .002), normal=(0., 0., 1.), length_scale_m=.25,
        critical_energy_release_rate_j_m2=2700., internal_pressure_pa=0.,
        annular_sector=(1., 2., .5, 2.5),
    )
    selection = {"selected_location_m": {"y": 0., "z": -2.},
                 "seed_face": "upstream", "required_region_ids": [1]}
    return specification, selection


class NativeSeedTests(unittest.TestCase):
    def test_bundled_python_detection_preserves_windows_and_supports_linux_layout(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from Aster.run_case import bundled_python_for
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "bin" / "run_aster"
            launcher.parent.mkdir()
            launcher.touch()
            linux_python = root / "bin" / "python3"
            linux_python.touch()
            with patch("Aster.run_case.os", SimpleNamespace(name="posix")):
                self.assertEqual(bundled_python_for(str(launcher)), linux_python)
            windows_python = root / "external" / "Python3.10" / "python.exe"
            windows_python.parent.mkdir(parents=True)
            windows_python.touch()
            self.assertEqual(bundled_python_for(str(launcher)), windows_python)

    def test_shared_history_checks_reject_invalid_independent_results(self):
        import copy
        data = {
            "audit": {"shared_seed_sha256": "frozen", "gauss_points": 2,
                      "seeded_gauss_points": 1, "excluded_nonwall_points_inside_seed": 1,
                      "unsampled_seed_rejected": True, "damage_values": [.999, 0.]},
            "history": [{"time": t, "damage_values": [.999, value]}
                        for t, value in ((0., 0.), (.5, .1), (1., .2))],
        }
        result = history_checks(data, {"sha256": "frozen"}, [0., .5, 1.], False)
        self.assertEqual(result["final_initially_intact_damage_max"], .2)
        mutations = (
            lambda d: d["audit"].__setitem__("shared_seed_sha256", "changed"),
            lambda d: d["audit"].__setitem__("gauss_points", 3),
            lambda d: d["audit"].__setitem__("damage_values", [.999, float("nan")]),
            lambda d: d["history"][1].__setitem__("time", .25),
            lambda d: d["history"][-1].__setitem__("damage_values", [.999, .01]),
            lambda d: d["history"][0].__setitem__("damage_values", [.999, .1]),
            lambda d: d["history"][-1].__setitem__("damage_values", [.998, .2]),
            lambda d: d["history"][-1].__setitem__("damage_values", [.999, 1.1]),
        )
        for mutation in mutations:
            changed = copy.deepcopy(data)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                history_checks(changed, {"sha256": "frozen"}, [0., .5, 1.], False)

    def test_native_wider_specimen_med_geometry_volume_and_faces(self):
        import json
        import importlib.util
        import sys
        import tempfile
        from unittest.mock import patch
        native_library = (Path(sys.executable).parent.parent / "medcoupling-9.11.0"
                          / "lib" / "python3.10" / "site-packages" / "medcoupling.py")
        if not native_library.is_file() and importlib.util.find_spec("medcoupling") is None:
            self.skipTest("Requires a native MED interpreter")
        with patch.object(sys, "path", [str(Path(__file__).parent), *sys.path]):
            from Aster.seed_specimen_mesh import convert
            from convert_three_body_to_med import mc
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                inputs, mesh = directory / "inputs.json", directory / "mesh.med"
                centre = np.array([3., 4., -2.])
                inputs.write_text(json.dumps({
                    "specimen_divisions": 8, "normal_grading_power": 2,
                    "lateral_divisions": 16, "lateral_width_m": 2.,
                    "centre_m": centre.tolist(),
                }))
                convert(inputs, mesh)
                native = mc.MEDFileUMesh.New(str(mesh))
                volume = native.getMeshAtLevel(0)
                xyz = volume.getCoords().toNumPyArray()
                np.testing.assert_allclose(xyz.min(axis=0), centre - [1., 1., .5])
                np.testing.assert_allclose(xyz.max(axis=0), centre + [1., 1., .5])
                self.assertEqual(volume.getNumberOfCells(), 16 ** 2 * 9)
                self.assertAlmostEqual(volume.getMeasureField(True).getArray().toNumPyArray().sum(), 4.)
                surface = native.getMeshAtLevel(-1).computeCellCenterOfMass().toNumPyArray()
                for name, axis, offset in (("BACK", 0, -1.), ("LEFT", 1, -1.),
                                           ("BOTTOM", 2, -.5), ("TOP", 2, .5)):
                    ids = native.getGroupArr(-1, name).toNumPyArray()
                    self.assertGreater(len(ids), 0)
                    np.testing.assert_allclose(surface[ids, axis], centre[axis] + offset)
                values = json.loads(inputs.read_text())
                values["tetrahedral"] = True
                inputs.write_text(json.dumps(values))
                convert(inputs, mesh)
                native = mc.MEDFileUMesh.New(str(mesh))
                volume = native.getMeshAtLevel(0)
                self.assertEqual(volume.getNumberOfCells(), 6 * 16 ** 2 * 9)
                volumes = volume.getMeasureField(True).getArray().toNumPyArray()
                z = volume.computeCellCenterOfMass().toNumPyArray()[:, 2] - centre[2]
                edges = normal_edges(8, 2)
                layers = np.searchsorted(edges, z) - 1
                np.testing.assert_allclose(volumes, np.diff(edges)[layers] * 4 / (6 * 16 ** 2),
                                           rtol=1e-11, atol=0.)
                self.assertAlmostEqual(volumes.sum(), 4.)
                # Every tetrahedral face is either shared twice or on the exterior.
                self.assertEqual(native.getMeshAtLevel(-1).getNumberOfCells(),
                                 2 * (2 * 16 ** 2 + 4 * 16 * 9))

    def test_tetrahedral_integration_rule_and_mode_guard(self):
        import copy
        data = self.specimen()
        expanded = copy.deepcopy(data)
        expanded["audit"]["coordinates_m"] *= 3
        for row in expanded["history"]:
            row["damage_values"] *= 3
        result = specimen_metrics(expanded, 8, 0., tetrahedral=True)
        np.testing.assert_allclose(result["integrated_damage_volume_m3_by_state"], .002 * .999)
        asymmetric = copy.deepcopy(expanded)
        asymmetric["history"][-1]["damage_values"][0] = .1
        with self.assertRaisesRegex(ValueError, "asymmetric"):
            specimen_metrics(asymmetric, 8, 0., tetrahedral=True)
        with self.assertRaisesRegex(ValueError, "inventory"):
            specimen_metrics(data, 8, 0., tetrahedral=True)
        from unittest.mock import patch
        with patch("Aster.seed_benchmark.detect_installation") as detect:
            for value in (1, "true", None):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                    run(32, finite_sector=True, tetrahedral=value)
            with self.assertRaisesRegex(ValueError, "finite-sector"):
                run(32, tetrahedral=True)
            detect.assert_not_called()

    def test_wider_domain_preserves_spacing_and_physical_seed_integral(self):
        small, wide = lateral_edges(8), lateral_edges(16, 2.)
        np.testing.assert_array_equal(np.diff(wide), np.diff(small)[0])
        np.testing.assert_array_equal(wide[[0, -1]], [-1., 1.])
        data = self.specimen()
        result = specimen_metrics(data, 8, 0., lateral_width_m=2.)
        self.assertAlmostEqual(result["domain_volume_m3"], 4.)
        np.testing.assert_allclose(result["integrated_damage_volume_m3_by_state"], 4 * .002 * .999)
        for width in (True, 0., .5, float("nan"), float("inf"), "2"):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "width"):
                lateral_edges(16, width)
        from unittest.mock import patch
        with patch("Aster.seed_benchmark.detect_installation") as detect:
            with self.assertRaisesRegex(ValueError, "finite-sector"):
                run(32, lateral_width_m=2.)
            detect.assert_not_called()

    def test_finite_sector_volume_reports_sampling_error_and_boundary_damage(self):
        import math
        centre = [79 * math.sin(1.4), 79 * math.cos(1.4), -27.5]
        inputs = {"centre_m": centre, "shared_seed": {"payload": {"geometry": {
            "annular_sector": [78.8, 79.2, 1.3975, 1.4025], "seed_thickness_m": .002}}}}
        coordinates = np.array([centre, np.asarray(centre) + [.48, 0., 0.]])
        data = {"audit": {"coordinates_m": coordinates.tolist(), "initial_damage": .999},
                "history": [{"damage_values": [.999, .1]}]}
        metrics = {"integrated_damage_volume_m3_by_state": [.0003]}
        result = finite_sector_metrics(data, inputs, metrics)
        self.assertAlmostEqual(result["exact_physical_seed_volume_m3"], .000316)
        self.assertAlmostEqual(result["exact_initial_scalar_damage_integral_m3"], .000315684)
        self.assertAlmostEqual(result["relative_initial_sampling_error"], .0003 / .000315684 - 1)
        self.assertEqual(result["maximum_final_damage_in_outer_lateral_0_05_m"], .1)
        inputs["lateral_width_m"] = 2.
        data["audit"]["coordinates_m"][1] = (np.asarray(centre) + [.98, 0., 0.]).tolist()
        wider = finite_sector_metrics(data, inputs, metrics)
        self.assertEqual(wider, result)
        inputs["lateral_width_m"] = 1.
        inputs["shared_seed"]["payload"]["geometry"]["annular_sector"][1] = 80.
        with self.assertRaisesRegex(ValueError, "entirely inside"):
            finite_sector_metrics(data, inputs, metrics)

    def test_normal_grading_preserves_exact_seed_layer_and_domain(self):
        for n in (8, 32, 128, 256):
            uniform, graded = normal_edges(n), normal_edges(n, 2)
            self.assertEqual(len(uniform), len(graded))
            np.testing.assert_allclose(graded, -graded[::-1], atol=0, rtol=0)
            np.testing.assert_array_equal(graded[[0, -1]], [-.5, .5])
            np.testing.assert_array_equal(graded[n // 2:n // 2 + 2], [-.001, .001])
            self.assertTrue(np.all(np.diff(graded) > 0))
            self.assertLess(graded[n // 2 + 2] - .001, uniform[n // 2 + 2] - .001)
        for power in (True, 0, 1., 3):
            with self.subTest(power=power), self.assertRaisesRegex(ValueError, "grading"):
                normal_edges(64, power)

    def test_extended_native_budget_requires_explicit_approval(self):
        from unittest.mock import patch
        with patch("Aster.seed_benchmark.detect_installation", return_value={"available": False}) as detect:
            for budget in (True, 0, -1, 240., 481, 1800):
                with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "budget"):
                    run(128, budget)
            detect.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "requires native"):
                run(128, 1800, allow_extended_budget=True)
            detect.assert_called_once()
        with patch("Aster.seed_benchmark.detect_installation") as detect:
            with self.assertRaisesRegex(ValueError, "resolved"):
                run(finite_sector=True)
            detect.assert_not_called()

    def test_invalid_memory_budget_never_starts_native_work(self):
        from unittest.mock import patch
        with patch("Aster.seed_benchmark.detect_installation") as detect:
            for memory in (True, 0, 2047, 4096., "4096"):
                with self.subTest(memory=memory), self.assertRaisesRegex(ValueError, "memory"):
                    run(32, memory_mib=memory)
            detect.assert_not_called()

    def test_failed_specimen_conversion_is_persisted_and_does_not_start_native_solve(self):
        import json
        import subprocess
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with (
                patch("Aster.seed_benchmark.tempfile.mkdtemp", return_value=temporary),
                patch("Aster.seed_benchmark.detect_installation",
                      return_value={"available": True, "launcher_path": "fake"}),
                patch("Aster.seed_benchmark.bundled_python_for", return_value=Path("fake-python")),
                patch("Aster.seed_benchmark.subprocess.run",
                      return_value=subprocess.CompletedProcess([], 1, "conversion output", "native error")),
                patch("Aster.seed_benchmark.run_solver") as solver,
            ):
                with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                    run(8, memory_mib=4096)
                solver.assert_not_called()
            report = json.loads((directory / "benchmark_report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("conversion failed", report["error"])
            self.assertEqual(report["memory_mib"], 4096)
            self.assertEqual((directory / "conversion.stderr.log").read_text(), "native error")

    def test_invalid_specimen_resolution_never_starts_native_work(self):
        from unittest.mock import patch
        with patch("Aster.seed_benchmark.detect_installation") as detect:
            for divisions in (True, 0, 1, 12, 8., "16"):
                with self.subTest(divisions=divisions), self.assertRaises(ValueError):
                    run(divisions)
            detect.assert_not_called()
        with patch("Aster.seed_benchmark.detect_installation") as detect:
            for lateral in (True, 0, 3, 8., "8"):
                with self.subTest(lateral=lateral), self.assertRaisesRegex(ValueError, "Lateral"):
                    run(64, lateral_divisions=lateral)
            detect.assert_not_called()

    def specimen(self):
        edges = np.concatenate((np.linspace(-.5, -.001, 5), np.linspace(.001, .5, 5)))
        abscissa = np.array([.5 - .5 / np.sqrt(3), .5 + .5 / np.sqrt(3)])
        levels = (edges[:-1, None] + np.diff(edges)[:, None] * abscissa).ravel()
        z = np.repeat(levels, 256)
        xyz = np.column_stack((np.zeros(len(z)), np.zeros(len(z)), z))
        values = np.where(np.abs(z) < .001, .999, 0.)
        data = {"audit": {"coordinates_m": xyz.tolist()},
                "history": [{"damage_values": values.tolist()} for _ in range(5)]}
        return data

    def test_specimen_integrates_exact_cell_volumes_not_unweighted_point_average(self):
        data = self.specimen()
        report = specimen_metrics(data, 8, 0.)
        self.assertAlmostEqual(report["domain_volume_m3"], 1.)
        np.testing.assert_allclose(report["integrated_damage_volume_m3_by_state"], .002 * .999)
        self.assertEqual(report["maximum_zero_load_equilibrium_change"], 0.)
        self.assertEqual(report["profile_reflection_error"], 0.)
        self.assertEqual(report["maximum_damage_beyond_0_3_m"], 0.)
        reduced = self.specimen()
        reduced["audit"]["coordinates_m"] = reduced["audit"]["coordinates_m"][::64]
        for row in reduced["history"]:
            row["damage_values"] = row["damage_values"][::64]
        small = specimen_metrics(reduced, 8, 0., lateral_divisions=1)
        np.testing.assert_allclose(small["integrated_damage_volume_m3_by_state"],
                                   report["integrated_damage_volume_m3_by_state"])

    def test_specimen_rejects_missing_points_asymmetry_remote_damage_and_drift(self):
        import copy
        baseline = self.specimen()
        mutations = (
            lambda d: d["audit"]["coordinates_m"].pop(),
            lambda d: d["history"][-1]["damage_values"].__setitem__(0, .1),
            lambda d: d["history"][2]["damage_values"].__setitem__(0, .1),
        )
        for mutation in mutations:
            data = copy.deepcopy(baseline)
            mutation(data)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                specimen_metrics(data, 8, 0.)

    def test_native_case_syntax(self):
        case = Path(__file__).with_name("cases") / "seed_benchmark.comm"
        compile(case.read_text(), str(case), "exec")

    def test_exact_thin_seed_is_not_silently_widened(self):
        spec, selection = selected_seed()
        contract = freeze_selected_seed(spec, selection)
        points = np.array([
            [1.5, 0., -2.], [1.5, 0., -1.9991], [1.5, 0., -1.9989],
            [.9, 0., -2.], [2.1, 0., -2.], [0., 1.5, -2.], [0., -1.5, -2.],
        ])
        np.testing.assert_array_equal(seed_damage(points, contract, .999),
                                      [.999, .999, 0., 0., 0., 0., 0.])
        self.assertEqual(seed_damage(np.empty((0, 3)), contract, .999).shape, (0,))

    def test_seed_shift_changes_only_the_frozen_geometry(self):
        spec, selection = selected_seed()
        first = freeze_selected_seed(spec, selection)
        shifted = replace(spec, centre_m=(1.5, 0., -3.))
        selection["selected_location_m"]["z"] = -3.
        second = freeze_selected_seed(shifted, selection)
        points = np.array([[1.5, 0., -2.], [1.5, 0., -3.]])
        np.testing.assert_array_equal(seed_damage(points, first, .999), [.999, 0.])
        np.testing.assert_array_equal(seed_damage(points, second, .999), [0., .999])
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_invalid_damage_coordinates_and_tampered_contract_are_rejected(self):
        spec, selection = selected_seed()
        contract = freeze_selected_seed(spec, selection)
        for value in (0., 1., -.1, float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                seed_damage(np.zeros((1, 3)), contract, value)
        for points in (np.zeros(3), np.zeros((2, 2)), np.array([[0., 0., np.inf]])):
            with self.subTest(points=points), self.assertRaises(ValueError):
                seed_damage(points, contract, .999)
        contract["payload"]["geometry"]["elevation_m"] = -3.
        with self.assertRaisesRegex(ValueError, "hash"):
            seed_damage(np.zeros((1, 3)), contract, .999)

    def test_native_stiffness_not_at2_and_near_complete_remains_finite(self):
        values = stiffness_ratio(np.array([0., .1, .999]), 10., 5.)
        self.assertEqual(values[0], 1.)
        self.assertTrue(np.all(np.diff(values) < 0))
        self.assertGreater(values[-1], 0.)
        self.assertNotAlmostEqual(values[1], .9 ** 2)
        for damage, m, p in ((1., 10., 5.), (-.1, 10., 5.), (.1, 0., 5.),
                             (.1, 10., -1.), (.1, np.nan, 5.)):
            with self.subTest(damage=damage, m=m, p=p), self.assertRaises(ValueError):
                stiffness_ratio(damage, m, p)


if __name__ == "__main__":
    unittest.main()
