"""Native zero exit and partial output are not successful elastic verification."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Aster.quadratic_baseline import NativeBaselineError, run, verify_native_result


class NativeResultTests(unittest.TestCase):
    def test_extended_budget_requires_explicit_approval_before_detection(self):
        with patch("Aster.quadratic_baseline.detect_installation") as detect:
            for budget in (True, 0, -1, 540.5, 571, 3600):
                with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "budget"):
                    run(Path("unused"), budget_s=budget)
            detect.assert_not_called()

    def test_explicit_extended_budget_reaches_runtime_check(self):
        with patch("Aster.quadratic_baseline.detect_installation",
                   return_value={"available": False}) as detect:
            with self.assertRaisesRegex(RuntimeError, "requires Code_Aster"):
                run(Path("unused"), budget_s=3600, allow_extended_budget=True)
            detect.assert_called_once()

    def test_invalid_thread_counts_are_rejected_before_launch(self):
        with patch("Aster.quadratic_baseline.detect_installation") as detect:
            for threads in (True, 0, -1, 5, 1.5):
                with self.subTest(threads=threads), self.assertRaisesRegex(ValueError, "threads"):
                    run(Path("unused"), threads=threads)
            detect.assert_not_called()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.summary = self.directory / "native_summary.json"
        self.result = self.directory / "result.med"
        self.payload = {
            "status": "finite_native_elastic_solve_not_baseline_acceptance",
            "displacement_components": 36, "maximum_absolute_displacement_component_m": .001,
        }
        self.summary.write_text(json.dumps(self.payload), encoding="utf-8")
        self.result.write_bytes(b"synthetic output; no MED content verification in this unit test")

    def test_finite_output_is_explicitly_not_baseline_acceptance(self):
        self.assertEqual(verify_native_result(0, "", self.summary, self.result), self.payload)

    def test_zero_exit_memory_exception_is_explicit_failure_even_with_outputs(self):
        with self.assertRaises(NativeBaselineError) as raised:
            verify_native_result(0, "<EXCEPTION> <JEVEUX_62>", self.summary, self.result)
        self.assertEqual(raised.exception.code, "native_memory_allocation")

    def test_native_exception_or_nonzero_exit_cannot_report_success(self):
        for exitcode, logs in ((0, "<EXCEPTION>"), (0, "Traceback (most recent call last)"),
                               (0, "Fatal Python error"), (1, "ARRET NORMAL")):
            with self.subTest(exitcode=exitcode, logs=logs), self.assertRaises(NativeBaselineError):
                verify_native_result(exitcode, logs, self.summary, self.result)

    def test_missing_or_empty_output_fails(self):
        self.result.write_bytes(b"")
        with self.assertRaisesRegex(NativeBaselineError, "incomplete outputs"):
            verify_native_result(0, "", self.summary, self.result)
        self.result.unlink()
        with self.assertRaisesRegex(NativeBaselineError, "incomplete outputs"):
            verify_native_result(0, "", self.summary, self.result)

    def test_invalid_native_summary_fails(self):
        for key, value in (("maximum_absolute_displacement_component_m", float("nan")),
                           ("maximum_absolute_displacement_component_m", -.1),
                           ("displacement_components", 0), ("displacement_components", True),
                           ("status", "succeeded")):
            payload = dict(self.payload, **{key: value})
            self.summary.write_text(json.dumps(payload), encoding="utf-8")
            with self.subTest(key=key, value=value), self.assertRaisesRegex(NativeBaselineError, "invalid"):
                verify_native_result(0, "", self.summary, self.result)
