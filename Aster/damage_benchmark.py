"""Isolated native 3-D damage/irreversibility verification, not a dam analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from Aster.fracture_material import ROOT, material_contract
from Aster.run_case import bundled_python_for, detect_installation, file_hash, run_solver
from Elmer.model_config import CONFIG_PATH, load_config

CASE = Path(__file__).with_name("cases") / "damage_benchmark.comm"
SETTINGS = {
    "refinement": 1,
    "target_damage": .2,
    "initial_damage": .1,
    "near_complete_damage": .999,
    "relative_tolerance": 1e-8,
    "maximum_iterations": 40,
    "load_history": [[0., 0.], [1., 1.], [2., .5], [3., 1.05]],
    "times": np.linspace(0., 3., 31).tolist(),
}


def visualisation_warnings(series: dict, history_name: str = "native_history.json") -> list[str]:
    """Keep native recovery overshoots distinct from integration-point damage."""
    warnings = []
    for label, history in series.items():
        if (history["damage_field"] != "damage_recovered_nodal" or
                history["damage_recovery"] != "native_VARI_NOEU_unclipped" or
                history["damage_association"] != "nodes"):
            raise ValueError(f"{label}: unsupported damage recovery metadata")
        ranges = np.asarray([
            [row["damage_min"], row["damage_max"]] for row in history["snapshots"]
        ])
        if (ranges.ndim != 2 or ranges.shape[1] != 2 or not np.isfinite(ranges).all()
                or np.any(ranges[:, 0] > ranges[:, 1])):
            raise ValueError(f"{label}: invalid recovered nodal damage ranges")
        if ranges.min() < -1e-7 or ranges.max() > 1 + 1e-7:
            warnings.append(
                f"{label}: recovered nodal damage spans "
                f"[{ranges.min():.9g}, {ranges.max():.9g}], outside [0, 1]. "
                "Native nodal recovery is unbounded and is not clipped. "
                f"Use {history_name} integration-point values for damage verification."
            )
    return warnings


def verify(report: dict) -> dict:
    """Check the closed-form homogeneous tensile state and pointwise no healing."""
    expected_damage = SETTINGS["target_damage"]
    expected_stress = report["expected_target_stress_zz_pa"]
    if not np.isfinite(expected_stress) or expected_stress <= 0:
        raise ValueError("Benchmark expected stress must be finite and positive")
    checks = {}
    for label, initial in (("INTACT", 0.), ("SEEDED", SETTINGS["initial_damage"]),
                           ("LOCAL", SETTINGS["initial_damage"]), ("FREE", 0.),
                           ("NEAR", SETTINGS["near_complete_damage"]),
                           ("NEARLOC", SETTINGS["near_complete_damage"])):
        history = report["histories"][label]
        times = np.array([row["time"] for row in history])
        if times.shape != (31,) or not np.allclose(times, SETTINGS["times"], rtol=0., atol=1e-10):
            raise ValueError(f"{label}: missing converged increments")
        damage = np.array([row["damage_values"] for row in history])
        if damage.ndim != 2 or damage.shape[1] == 0 or not np.isfinite(damage).all():
            raise ValueError(f"{label}: invalid integration-point damage history")
        support = np.array([row["damage_point_ids"] for row in history])
        if (support.shape != (*damage.shape, 3) or
                not np.all(support == support[0]) or
                len(np.unique(support[0], axis=0)) != damage.shape[1]):
            raise ValueError(f"{label}: integration-point identities changed or are ambiguous")
        if damage.min() < -1e-7 or damage.max() > 1 + 1e-7:
            raise ValueError(f"{label}: damage outside [0, 1]")
        if label in ("LOCAL", "NEARLOC"):
            if (not np.isclose(damage[0].min(), 0., atol=1e-7) or
                    not np.isclose(damage[0].max(), initial, atol=1e-7)):
                raise ValueError(f"{label}: missing spatially localized initial damage")
        elif not np.allclose(damage[0], initial, atol=1e-7, rtol=0.):
            raise ValueError(f"{label}: initial damage was not retained")
        if np.diff(damage, axis=0).min() < -1e-7:
            raise ValueError(f"{label}: damage healed during loading/unloading")
        stress = np.array([row["stress_zz_mean_pa"] for row in history])
        if not np.isfinite(stress).all():
            raise ValueError(f"{label}: nonfinite stress")
        peak = int(np.flatnonzero(np.isclose(times, 1.))[0])
        unloaded = int(np.flatnonzero(np.isclose(times, 2.))[0])
        analytical = label in ("INTACT", "SEEDED")
        if analytical and not np.allclose(damage[peak], expected_damage, rtol=0., atol=1e-5):
            raise ValueError(f"{label}: nonlinear damage misses the closed-form target")
        stress_error = abs(stress[peak] - expected_stress) / expected_stress
        if analytical and stress_error > 1e-5:
            raise ValueError(f"{label}: nonlinear stress misses the closed-form target")
        if not np.allclose(damage[unloaded], damage[peak], rtol=0., atol=1e-7):
            raise ValueError(f"{label}: damage changed under tensile unloading")
        # Free contraction introduces compressive lateral strains; the native
        # crack-closure regularisation is nonlinear even at fixed damage.
        if label != "FREE" and not np.isclose(stress[unloaded], .5 * stress[peak], rtol=1e-5):
            raise ValueError(f"{label}: unloading stiffness is inconsistent")
        if label == "FREE" and not 0 < stress[unloaded] < stress[peak]:
            raise ValueError("FREE: tensile stress did not decrease on unloading")
        if label == "NEAR" and not np.allclose(damage, initial, rtol=0., atol=1e-7):
            raise ValueError("NEAR: near-complete initial damage was not retained throughout the history")
        if label == "NEARLOC":
            seeded = np.isclose(damage[0], initial, atol=1e-7, rtol=0.)
            if not np.allclose(damage[:, seeded], initial, atol=1e-7, rtol=0.):
                raise ValueError("NEARLOC: near-complete localized damage was not retained")
        if label != "NEAR" and not np.any(damage[-1] > damage[peak] + 1e-5):
            raise ValueError(f"{label}: damage did not grow on further loading")
        if label == "FREE":
            lateral = np.array([row["maximum_lateral_stress_pa"] for row in history])
            if not np.isfinite(lateral).all() or np.max(lateral) > 1e-5 * np.max(np.abs(stress)):
                raise ValueError("FREE: lateral stress is inconsistent with free contraction")
        checks[label] = {
            "increments": len(history) - 1,
            "gauss_points": damage.shape[1],
            "target_damage_max_absolute_error": (
                float(np.max(np.abs(damage[peak] - expected_damage))) if analytical else None
            ),
            "target_stress_relative_error": float(stress_error) if analytical else None,
            "minimum_damage_increment": float(np.diff(damage, axis=0).min()),
            "unloaded_to_peak_stress_ratio": float(stress[unloaded] / stress[peak]),
            "final_damage_range": [float(damage[-1].min()), float(damage[-1].max())],
        }
    return checks


def run(config_path: Path, output_dir: Path, timeout_s: int = 180) -> dict:
    config = load_config(config_path)
    material = material_contract(config)
    installation = detect_installation()
    if not installation["available"]:
        raise RuntimeError("Native damage benchmark requires Code_Aster")
    output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="damage-", dir=output_dir.resolve()))
    inputs_path = directory / "inputs.json"
    inputs_path.write_text(json.dumps({"material": material, "settings": SETTINGS}, indent=2), encoding="utf-8")
    export_path = directory / "benchmark.export"
    result_path = directory / "result.med"
    response_path = directory / "native_history.json"
    export_path.write_text(
        f"P time_limit {timeout_s}\nP memory_limit 1024\nP mpi_nbcpu 1\n"
        f"F comm {CASE.resolve()} D 1\nF libr {inputs_path} D 91\n"
        f"F libr {response_path} R 90\nF rmed {result_path} R 80\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1, "status": "running", "engineering_validated": False,
        "scope": "confined_and_free_contraction_tension_with_initial_damage_checks",
        "material": material, "settings": SETTINGS, "installation": installation,
        "inputs_sha256": file_hash(inputs_path), "case_sha256": file_hash(CASE),
        "directory": str(directory),
        "limitations": [
            "Not a dam solve or free crack propagation benchmark.",
            "INTACT/SEEDED/LOCAL prescribe affine confined strain; FREE solves lateral contraction.",
            "Initial and near-complete damage test native history retention, not AT2 seed equivalence.",
            "Local seed is one synthetic element, not a resolved dam crack or mesh convergence study.",
            "FREE tests native equilibrium/no healing, not a closed-form crack-closure response.",
            "Recovered nodal damage is an unbounded visualisation, not the verified integration-point field.",
        ],
    }
    started = time.monotonic()
    try:
        completed = run_solver([installation["launcher_path"], str(export_path)], directory, timeout_s)
        (directory / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (directory / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        logs = completed.stdout + completed.stderr
        if completed.returncode != 0 or any(
            marker in logs for marker in ("<EXCEPTION>", "Traceback (most recent call last)")
        ):
            raise RuntimeError(f"Native damage benchmark failed; see {directory}")
        if not response_path.is_file() or not result_path.is_file():
            raise RuntimeError(f"Native damage benchmark produced incomplete outputs; see {directory}")
        native = json.loads(response_path.read_text(encoding="utf-8"))
        report["checks"] = verify(native)
        python = bundled_python_for(installation["launcher_path"])
        if python is None:
            raise RuntimeError("Cannot export damage: bundled MED interpreter unavailable")
        extractor = Path(__file__).with_name("extract_damage_history.py")
        extracted = subprocess.run(
            [str(python), str(extractor), str(result_path), str(directory)],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if extracted.returncode != 0:
            raise RuntimeError(f"Damage export failed: {extracted.stdout}\n{extracted.stderr}")
        report["visualisation"] = json.loads((directory / "damage_series.json").read_text(encoding="utf-8"))
        report["warnings"] = visualisation_warnings(report["visualisation"])
        for label, history in native["histories"].items():
            series = report["visualisation"][label]
            snapshots = series["snapshots"]
            if ([row["time"] for row in snapshots] != [row["time"] for row in history] or
                    not (directory / series["pvd"]).is_file() or
                    any(not (directory / row["vtu"]).is_file() for row in snapshots)):
                raise ValueError(f"{label}: incomplete ParaView time series")
        report["status"] = "benchmark_passed"
    except (RuntimeError, ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        (directory / "benchmark_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Aster" / "results" / "damage_benchmark")
    args = parser.parse_args()
    report = run(args.config, args.output_dir)
    print(json.dumps({key: report[key] for key in
                      ("status", "directory", "checks", "warnings", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
