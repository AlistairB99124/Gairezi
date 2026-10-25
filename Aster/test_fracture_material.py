"""Runtime-neutral fracture material contracts; native checks have a separate CLI."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from Aster.check_fracture_material import check, command_text, sensitivity_contracts, validate_responses
from Aster.fracture_material import FENICSX_DEFAULTS, material_contract
from Elmer.model_config import load_config


class FractureMaterialTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_nominal_inputs_keep_shared_material_and_energy(self):
        original = copy.deepcopy(self.config)
        contract = material_contract(self.config)
        p = contract["parameters"]
        self.assertEqual(p["FC"], 20e6)
        self.assertEqual(p["FT"], self.config["material"]["concrete"]["tensile_strength_pa"])
        self.assertEqual(p["E"], self.config["material"]["concrete"]["youngs_modulus"])
        self.assertEqual(p["GF"], json.loads(FENICSX_DEFAULTS.read_text())["critical_energy_release_rate_j_m2"])
        self.assertEqual((p["LARG_BANDE"], p["P"], p["Q"]), (.5, 5., 0.))
        self.assertEqual(contract["derived"]["half_band_width_m"], .25)
        self.assertEqual(contract["calibration_status"], "provisional_not_test_calibrated")
        self.assertAlmostEqual(contract["derived"]["equivalent_initial_fracture_energy_j_m2"],
                               3 * math.pi * p["GF"] / (4 * 7 ** 1.5))
        self.assertEqual(self.config, original)

    def test_historical_configuration_has_no_inferred_compressive_strength(self):
        del self.config["material"]["concrete"]["compressive_strength_pa"]
        with self.assertRaisesRegex(ValueError, "compressive_strength_pa is required"):
            material_contract(self.config)

    def test_native_strength_ratio_and_poisson_constraints(self):
        for key, value, message in (
            ("compressive_strength_pa", 10e6, "ratio"),
            ("poisson_ratio", -.1, "poisson_ratio"),
        ):
            config = copy.deepcopy(self.config)
            config["material"]["concrete"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                material_contract(config)

    def test_invalid_width_and_shape_are_rejected(self):
        for key, values in (("band_width_m", (0, -1, True, float("nan"), float("inf"))),
                            ("softening_p", (0, .9, True, float("nan")))):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    material_contract(self.config, **{key: value})

    def test_energy_is_read_from_the_shared_source_not_a_second_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            defaults = Path(directory) / "defaults.json"
            defaults.write_text('{"critical_energy_release_rate_j_m2": 150}')
            contract = material_contract(self.config, fenicsx_defaults_path=defaults)
        self.assertEqual(contract["parameters"]["GF"], 150)

    def test_sensitivity_grid_preserves_strength_and_energy(self):
        contracts = sensitivity_contracts(self.config)
        self.assertEqual(len(contracts), 9)
        self.assertEqual({c["parameters"]["LARG_BANDE"] for c in contracts}, {.25, .5, 1.})
        self.assertEqual({c["parameters"]["P"] for c in contracts}, {2., 5., 8.})
        self.assertEqual({c["parameters"]["FC"] for c in contracts}, {20e6})
        self.assertEqual(len({c["parameters"]["GF"] for c in contracts}), 1)
        compile(command_text(contracts), "material.comm", "exec")

    def test_incomplete_altered_and_nonfinite_native_results_are_rejected(self):
        contracts = sensitivity_contracts(self.config)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_responses(contracts, [])
        with self.assertRaisesRegex(ValueError, "malformed"):
            validate_responses(contracts, [None] * len(contracts))
        responses = []
        for contract in contracts:
            p = contract["parameters"]
            responses.append({"parameters": p.copy(), "internal": {
                "M": 1000. / p["LARG_BANDE"], "P": p["P"], "Q": 0., "SIG0": 1., "TAU": 1.,
                "K": 1.5 * p["GF"] / p["LARG_BANDE"],
                "C_GRAD_VARI": .1875 * p["GF"] * p["LARG_BANDE"],
            }})
        summary = validate_responses(contracts, responses)
        self.assertEqual((summary["accepted_cases"], summary["inadmissible_cases"]), (9, []))
        for key, value in (("M", float("nan")), ("M", 1.), ("K", 0.), ("M", 999.)):
            altered = copy.deepcopy(responses)
            altered[0]["internal"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_responses(contracts, altered)
        responses[0]["parameters"]["FC"] = 30e6
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_responses(contracts, responses)

    def test_only_predicted_band_width_rejections_are_admissible(self):
        contracts = sensitivity_contracts(self.config)
        invariant = 5.78  # Native M * LARG_BANDE at GF = 200 J/m2: 1 m bands with P >= 5 fail.

        def response(p):
            m = invariant / p["LARG_BANDE"]
            if m < p["P"] + 2:
                return {"parameters": p.copy(), "rejected": {
                    "message_id": "COMPOR1_94", "text": "m>p+2"}}
            return {"parameters": p.copy(), "internal": {
                "M": m, "P": p["P"], "Q": 0., "SIG0": 1., "TAU": 1.,
                "K": 1.5 * p["GF"] / p["LARG_BANDE"],
                "C_GRAD_VARI": .1875 * p["GF"] * p["LARG_BANDE"],
            }}

        responses = [response(c["parameters"]) for c in contracts]
        summary = validate_responses(contracts, responses)
        self.assertEqual(summary["accepted_cases"], 7)
        self.assertEqual({(c["LARG_BANDE"], c["P"]) for c in summary["inadmissible_cases"]},
                         {(1., 5.), (1., 8.)})
        self.assertAlmostEqual(summary["maximum_admissible_band_width_m_at_nominal_p"], invariant / 7)
        nominal = len(contracts) // 2
        for index, rejected, message in (
            (nominal, {"message_id": "COMPOR1_94", "text": ""}, "nominal"),
            (0, {"message_id": "COMPOR1_94", "text": ""}, "satisfies"),
            (7, {"message_id": "OTHER_1", "text": ""}, "unexpected"),
        ):
            altered = copy.deepcopy(responses)
            altered[index] = {"parameters": contracts[index]["parameters"].copy(), "rejected": rejected}
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, message):
                validate_responses(contracts, altered)

    def test_native_failure_and_missing_output_never_report_material_success(self):
        for code, stdout in ((1, "failed"), (0, "<EXCEPTION>"), (0, "no response")):
            with self.subTest(code=code, stdout=stdout), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config_path = root / "config.json"
                config_path.write_text(json.dumps(self.config))
                # An unrelated old response must never satisfy the fresh-run check.
                (root / "native_materials.json").write_text("[]")
                with patch("Aster.check_fracture_material.detect_installation", return_value={
                    "available": True, "launcher_path": "fake",
                }), patch("Aster.check_fracture_material.run_solver", return_value=
                          subprocess.CompletedProcess([], code, stdout, "diagnostic")):
                    with self.assertRaises(RuntimeError):
                        check(config_path, root)
                reports = list(root.glob("material-*/material_check.json"))
                self.assertEqual(len(reports), 1)
                report = json.loads(reports[0].read_text())
                self.assertEqual(report["status"], "failed")
                self.assertIn("error", report)
                self.assertEqual((reports[0].parent / "stderr.log").read_text(), "diagnostic")


if __name__ == "__main__":
    unittest.main()
