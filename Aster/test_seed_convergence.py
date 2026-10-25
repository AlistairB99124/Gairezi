import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Aster.seed_benchmark import specimen_metrics
from Aster.seed_convergence import refinement_checks, report, zero_load_planar_reference
from Aster.source_contract import sha256


class SeedConvergenceTests(unittest.TestCase):
    def test_zero_load_reference_satisfies_obstacle_equation_and_exact_integral(self):
        z = np.linspace(-.5, .5, 100001)
        profile, reference = zero_load_planar_reference(z, .999, .002, .5)
        self.assertAlmostEqual(float(np.trapezoid(profile, z)),
                               reference["scalar_damage_integral_m3_for_one_m2_area"], places=9)
        np.testing.assert_allclose(profile[np.abs(z) < .001], .999)
        self.assertEqual(float(profile[np.abs(z) > .251].max()), 0.)
        dz = z[1] - z[0]
        second = np.diff(profile, n=2) / dz ** 2
        active = (z[1:-1] > .01) & (z[1:-1] < .24)
        np.testing.assert_allclose(second[active], 2 / .25 ** 2, rtol=1e-6)
        for damage, thickness, band in ((0, .002, .5), (1, .002, .5), (.999, 0, .5), (.999, .002, 0)):
            with self.subTest(damage=damage, thickness=thickness, band=band), self.assertRaises(ValueError):
                zero_load_planar_reference(z, damage, thickness, band)

    def rows(self):
        return [{"normal_divisions": n, "equilibrium_integral_m3": .1,
                 "profile": [[-.5, 0.], [0., .999], [.5, 0.]]} for n in (64, 128, 256)]

    def test_requires_two_consecutive_integral_and_profile_checks(self):
        rows = self.rows()
        self.assertEqual(refinement_checks(rows[:2], .01)["status"], "normal_resolution_sensitivity_unresolved")
        self.assertEqual(refinement_checks(rows, .01)["status"], "successive_refinement_tolerance_met")
        rows[-1]["profile"][1][1] = .5
        result = refinement_checks(rows, .01)
        self.assertEqual(result["status"], "normal_resolution_sensitivity_unresolved")
        self.assertEqual(result["comparisons"][-1]["relative_integral_change"], 0)
        self.assertGreater(result["comparisons"][-1]["relative_profile_l2_change"], .4)

    def test_nonconvergent_integral_is_not_hidden_by_matching_profiles(self):
        rows = self.rows()
        rows[-1]["equilibrium_integral_m3"] = .11
        self.assertFalse(refinement_checks(rows, .01)["comparisons"][-1]["within_requested_tolerance"])

    def test_invalid_sequence_and_profiles_fail(self):
        for mutate in (
            lambda rows: rows[-1].update(normal_divisions=512),
            lambda rows: rows[-1].update(equilibrium_integral_m3=float("nan")),
            lambda rows: rows[-1].update(profile=[[0., .1], [0., .2]]),
            lambda rows: rows[-1].update(profile=[[0., .1], [1., 2.]]),
        ):
            rows = copy.deepcopy(self.rows())
            mutate(rows)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                refinement_checks(rows, .01)
        for value in (True, 0, 1, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                refinement_checks(self.rows(), value)

    def write_specimen(self, root, n):
        directory = root / str(n)
        directory.mkdir()
        inputs = {"shared_seed": {"sha256": "same-physical-seed",
                                  "payload": {"geometry": {"seed_thickness_m": .002}}},
                  "material": {"FT": 2e6, "parameters": {"LARG_BANDE": .5}},
                  "centre_m": [0., 0., 0.], "times": [0., .5, 1., 1.5, 2.]}
        inputs_path = directory / "inputs.json"
        inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
        edges = np.concatenate([np.linspace(-.5, -.001, n // 2 + 1),
                                np.linspace(.001, .5, n // 2 + 1)])
        gauss = np.array([.5 - .5 / np.sqrt(3), .5 + .5 / np.sqrt(3)])
        z = np.repeat((edges[:-1, None] + np.diff(edges)[:, None] * gauss).ravel(), 4)
        damage = np.where(np.abs(z) < .001, .999, 0.)
        native = {"audit": {"coordinates_m": np.column_stack([z * 0, z * 0, z]).tolist(),
                             "initial_damage": .999},
                  "history": [{"damage_values": damage.tolist()} for _ in range(5)]}
        (directory / "native_seed.json").write_text(json.dumps(native), encoding="utf-8")
        evidence = {"status": "seed_placement_and_irreversibility_passed", "specimen_divisions": n,
                    "lateral_divisions": 1, "inputs_sha256": sha256(inputs_path),
                    "specimen_metrics": specimen_metrics(native, n, 0., lateral_divisions=1)}
        (directory / "benchmark_report.json").write_text(json.dumps(evidence), encoding="utf-8")
        return directory

    def test_persisted_report_rechecks_native_metrics_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [self.write_specimen(root, n) for n in (8, 16)]
            output = root / "comparison.json"
            result = report(directories, output)
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertEqual(result["status"], "normal_resolution_sensitivity_unresolved")
            path = directories[0] / "inputs.json"
            path.write_text(path.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                report(directories, root / "invalid.json")
            self.assertFalse((root / "invalid.json").exists())

    def test_finite_sector_cannot_use_infinite_planar_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [self.write_specimen(root, n) for n in (8, 16)]
            path = directories[0] / "inputs.json"
            inputs = json.loads(path.read_text())
            inputs["finite_sector"] = True
            path.write_text(json.dumps(inputs), encoding="utf-8")
            evidence_path = directories[0] / "benchmark_report.json"
            evidence = json.loads(evidence_path.read_text())
            evidence["inputs_sha256"] = sha256(path)
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Finite-sector"):
                report(directories, root / "invalid.json")

    def test_changed_native_history_or_physical_contract_cannot_generate_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [self.write_specimen(root, n) for n in (8, 16)]
            native_path = directories[0] / "native_seed.json"
            original = native_path.read_text()
            native = json.loads(original)
            values = native["history"][0]["damage_values"]
            values[values.index(.999)] = .5
            native_path.write_text(json.dumps(native), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metrics differ"):
                report(directories, root / "invalid.json")
            native_path.write_text(original, encoding="utf-8")
            path = directories[0] / "inputs.json"
            inputs = json.loads(path.read_text())
            inputs["material"]["FT"] = 3e6
            path.write_text(json.dumps(inputs), encoding="utf-8")
            evidence_path = directories[0] / "benchmark_report.json"
            evidence = json.loads(evidence_path.read_text())
            evidence["inputs_sha256"] = sha256(path)
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "material"):
                report(directories, root / "invalid.json")
