"""Bounded native physical-seed placement/irreversibility, not propagation accuracy."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from Aster.fracture_material import material_contract
from Aster.seed_specimen_mesh import lateral_edges, normal_edges
from Aster.run_case import bundled_python_for, detect_installation, file_hash, run_solver
from Elmer.model_config import load_config
from FEniCSx.crack_contract import CrackSpecification, freeze_selected_seed

CASE = Path(__file__).with_name("cases") / "seed_benchmark.comm"


def history_checks(data: dict, contract: dict, times: list[float], resolved: bool) -> dict:
    """Apply the same native seed-history gates to independent runtime results."""
    audit = data["audit"]
    if (audit["shared_seed_sha256"] != contract["sha256"]
            or not 0 < audit["seeded_gauss_points"] < audit["gauss_points"]
            or (not resolved and audit["excluded_nonwall_points_inside_seed"] <= 0)
            or audit["unsampled_seed_rejected"] is not True
            or [row["time"] for row in data["history"]] != times):
        raise ValueError("Native physical seed benchmark has incomplete evidence")
    initial = np.asarray(audit["damage_values"])
    history = np.asarray([row["damage_values"] for row in data["history"]])
    if (initial.shape != (audit["gauss_points"],) or not np.isfinite(initial).all()
            or np.count_nonzero(initial > 0) != audit["seeded_gauss_points"]
            or history.shape != (len(times), len(initial)) or not np.isfinite(history).all()
            or history.min() < -1e-7 or history.max() > 1 + 1e-7
            or np.any(np.diff(history, axis=0) < -1e-7)
            or not np.allclose(history[0], initial, rtol=0., atol=1e-7)
            or np.any(history[:, initial > 0] < .999 - 1e-7)):
        raise ValueError("Native physical seed history violates initial placement or irreversibility")
    return {
        "gauss_points": audit["gauss_points"],
        "seeded_gauss_points": audit["seeded_gauss_points"],
        "excluded_nonwall_points_inside_seed": audit["excluded_nonwall_points_inside_seed"],
        "retained_states": len(data["history"]),
        "unsampled_seed_rejected": audit["unsampled_seed_rejected"],
        "final_initially_intact_damage_max": float(history[-1, initial == 0].max()),
    }


def specimen_metrics(data: dict, divisions: int, elevation: float, lateral_divisions: int = 8,
                     normal_grading_power: int = 1, lateral_width_m: float = 1.,
                     tetrahedral: bool = False) -> dict:
    """Integrate the fixed Gauss-point history with exact structured cell volumes."""
    xyz = np.asarray(data["audit"]["coordinates_m"])
    history = np.asarray([row["damage_values"] for row in data["history"]])
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
            or history.shape != (5, len(xyz)) or not np.isfinite(history).all()):
        raise ValueError("Specimen coordinates and five-state damage history must match and be finite")
    z = xyz[:, 2] - elevation
    edges = normal_edges(divisions, normal_grading_power)
    lateral_edges(lateral_divisions, lateral_width_m)
    if not isinstance(tetrahedral, bool):
        raise ValueError("Tetrahedral specimen selection must be boolean")
    points_per_box = 24 if tetrahedral else 8
    layers = np.searchsorted(edges, z) - 1
    if np.any(layers < 0) or np.any(layers >= len(edges) - 1):
        raise ValueError("Specimen integration points lie outside its verified layers")
    counts = np.bincount(layers, minlength=len(edges) - 1)
    if not np.all(counts == lateral_divisions ** 2 * points_per_box):
        raise ValueError("Specimen integration-point inventory differs from its structured volume rule")
    weights = np.diff(edges)[layers] * lateral_width_m ** 2 / (points_per_box * lateral_divisions ** 2)
    if not np.isclose(weights.sum(), lateral_width_m ** 2, rtol=1e-12, atol=0.):
        raise ValueError("Specimen Gauss weights do not integrate its declared domain volume")
    integral = history @ weights
    unique_z = np.unique(np.round(z, 10))
    profile = np.array([history[-1, np.isclose(z, value, rtol=0., atol=1e-9)].mean()
                        for value in unique_z])
    symmetry_error = float(np.max(np.abs(profile - profile[::-1])))
    equilibrium_change = float(np.max(np.abs(history[2] - history[1])))
    remote = np.abs(z) > .3
    remote_damage = float(history[-1, remote].max())
    if symmetry_error > 1e-6 or equilibrium_change > 1e-7 or remote_damage > 1e-7:
        raise ValueError("Resolved seed specimen is asymmetric, unequilibrated or damages its remote boundary")
    return {
        "domain_volume_m3": float(weights.sum()),
        "integrated_damage_volume_m3_by_state": integral.tolist(),
        "maximum_zero_load_equilibrium_change": equilibrium_change,
        "profile_reflection_error": symmetry_error,
        "maximum_damage_beyond_0_3_m": remote_damage,
        "final_profile": [{"z_offset_m": float(z), "damage_mean": float(value)}
                          for z, value in zip(unique_z, profile)],
        "measure": "Gauss-integrated scalar damage volume; not crack volume or opening",
    }


def finite_sector_metrics(data: dict, inputs: dict, metrics: dict) -> dict:
    """Quantify finite seed sampling without treating Gauss hits as exact volume."""
    geometry = inputs["shared_seed"]["payload"]["geometry"]
    inner, outer, start, end = geometry["annular_sector"]
    angles = [start, end] + [n * math.pi / 2 for n in range(-4, 5) if start <= n * math.pi / 2 <= end]
    boundary = np.array([[radius * math.sin(angle), radius * math.cos(angle)]
                         for radius in (inner, outer) for angle in angles])
    centre = np.asarray(inputs["centre_m"])
    width = inputs.get("lateral_width_m", 1.)
    lateral_edges(inputs.get("lateral_divisions", 8), width)
    if np.any(np.abs(boundary - centre[:2]) >= width / 2):
        raise ValueError("Finite seed footprint must lie entirely inside the synthetic specimen")
    volume = .5 * (outer ** 2 - inner ** 2) * (end - start) * geometry["seed_thickness_m"]
    exact = volume * data["audit"]["initial_damage"]
    if not np.isfinite(exact) or exact <= 0:
        raise ValueError("Finite seed must have positive exact initial damage integral")
    xyz = np.asarray(data["audit"]["coordinates_m"])
    outer_shell = np.max(np.abs(xyz[:, :2] - centre[:2]), axis=1) > width / 2 - .05
    if not outer_shell.any():
        raise ValueError("Finite seed lateral-boundary check requires Gauss points in the outer 0.05 m")
    maximum = float(np.max(np.asarray(data["history"][-1]["damage_values"])[outer_shell]))
    return {
        "exact_physical_seed_volume_m3": volume,
        "exact_initial_scalar_damage_integral_m3": exact,
        "sampled_initial_scalar_damage_integral_m3": metrics["integrated_damage_volume_m3_by_state"][0],
        "relative_initial_sampling_error": float(
            metrics["integrated_damage_volume_m3_by_state"][0] / exact - 1),
        "maximum_final_damage_in_outer_lateral_0_05_m": maximum,
        "limitations": [
            "Gauss-point sampling does not exactly integrate a curved finite seed boundary.",
            "No infinite-planar analytic equilibrium applies to this finite-sector check.",
            "A passed nonlinear history does not establish finite-edge mesh convergence.",
        ],
    }


def run(specimen_divisions: int | None = None, native_budget_s: int = 240,
        allow_extended_budget: bool = False, lateral_divisions: int = 8,
        line_search: bool = False, normal_grading_power: int = 1,
        finite_sector: bool = False, lateral_width_m: float = 1.,
        memory_mib: int = 2048, tetrahedral: bool = False) -> dict:
    if not isinstance(tetrahedral, bool):
        raise ValueError("Tetrahedral specimen selection must be boolean")
    if tetrahedral and not finite_sector:
        raise ValueError("Tetrahedral diagnostic requires a resolved finite-sector specimen")
    if isinstance(memory_mib, bool) or not isinstance(memory_mib, int) or memory_mib < 2048:
        raise ValueError("Native memory budget must be an integer of at least 2048 MiB")
    if (isinstance(native_budget_s, bool) or not isinstance(native_budget_s, int)
            or native_budget_s <= 0 or (native_budget_s > 480 and not allow_extended_budget)):
        raise ValueError("Positive native budget required; more than 480 seconds needs explicit extended approval")
    if specimen_divisions is not None and (
        isinstance(specimen_divisions, bool) or not isinstance(specimen_divisions, int)
        or specimen_divisions not in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    ):
        raise ValueError("Specimen divisions must be a power of two from 8 to 4096")
    lateral_edges(lateral_divisions, lateral_width_m)
    if lateral_width_m != 1. and not finite_sector:
        raise ValueError("Wider lateral domains require the finite-sector specimen")
    if specimen_divisions is not None:
        normal_edges(specimen_divisions, normal_grading_power)
    if finite_sector and specimen_divisions is None:
        raise ValueError("Finite-sector check requires a resolved specimen mesh")
    installation = detect_installation()
    if not installation["available"]:
        raise RuntimeError("Physical seed benchmark requires native Code_Aster")
    material = material_contract(load_config())
    radius, theta, elevation = 79., 1.4, -27.5
    centre = (radius * math.sin(theta), radius * math.cos(theta), elevation)
    spec = CrackSpecification(
        type="phase_field", shape="annular_sector", centre_m=centre,
        size_m=(.4, .4, .002) if finite_sector else (1., 1., .002),
        normal=(0., 0., 1.), length_scale_m=.25,
        critical_energy_release_rate_j_m2=material["parameters"]["GF"], internal_pressure_pa=0.,
        annular_sector=(78.8, 79.2, 1.3975, 1.4025) if finite_sector else (78., 80., 1.3, 1.5),
    )
    contract = freeze_selected_seed(spec, {
        "selected_location_m": {"y": centre[1], "z": elevation},
        "seed_face": "upstream", "required_region_ids": [1],
    })
    root = Path(__file__).resolve().parent.parent
    output = root / "Aster" / "results" / "seed_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="seed-", dir=output))
    inputs = directory / "inputs.json"
    resolved = specimen_divisions is not None
    times = [0., .5, 1., 1.5, 2.] if resolved else [0., .5, 1.]
    input_values = {
        "repository_root": str(root), "shared_seed": contract,
        "translation_m": [centre[0] - .002, centre[1] - .002,
                          centre[2] - (.5 if resolved else .003)],
        "centre_m": list(centre), "specimen_divisions": specimen_divisions, "times": times,
        "lateral_divisions": lateral_divisions,
        "lateral_width_m": lateral_width_m,
        "line_search": line_search,
        "normal_grading_power": normal_grading_power,
        "finite_sector": finite_sector,
        "tetrahedral": tetrahedral,
        "material": material,
    }
    inputs.write_text(json.dumps(input_values, indent=2), encoding="utf-8")
    native = directory / "native_seed.json"
    result = directory / "result.med"
    export = directory / "benchmark.export"
    report = {
        "status": "running", "directory": str(directory), "engineering_validated": False,
        "scope": "synthetic_exact_UI_seed_placement_and_irreversibility_not_propagation_accuracy",
        "case_sha256": file_hash(CASE), "inputs_sha256": file_hash(inputs),
        "adapter_sha256": file_hash(Path(__file__).with_name("native_seed.py")),
        "shared_seed_sha256": contract["sha256"],
        "specimen_divisions": specimen_divisions,
        "lateral_divisions": lateral_divisions,
        "lateral_width_m": lateral_width_m,
        "line_search": line_search,
        "normal_grading_power": normal_grading_power,
        "finite_sector": finite_sector,
        "tetrahedral": tetrahedral,
        "native_budget_s": native_budget_s,
        "memory_mib": memory_mib,
        "extended_budget_authorized": allow_extended_budget,
    }
    started = time.monotonic()
    try:
        mesh_line = ""
        if resolved:
            python = bundled_python_for(installation["launcher_path"])
            if python is None:
                raise RuntimeError("Native specimen mesh requires the bundled MED interpreter")
            converter = Path(__file__).with_name("seed_specimen_mesh.py")
            mesh = directory / "mesh.med"
            report["converter_sha256"] = file_hash(converter)
            conversion = subprocess.run(
                [str(python), str(converter), str(inputs), str(mesh)], capture_output=True,
                text=True, timeout=30, check=False,
            )
            (directory / "conversion.stdout.log").write_text(conversion.stdout, encoding="utf-8")
            (directory / "conversion.stderr.log").write_text(conversion.stderr, encoding="utf-8")
            if conversion.returncode or not mesh.is_file() or mesh.stat().st_size == 0:
                raise RuntimeError(f"Native specimen conversion failed; see {directory}")
            report["mesh_sha256"] = file_hash(mesh)
            mesh_line = f"F mmed {mesh} D 20\n"
        export.write_text(
            f"P time_limit {native_budget_s}\nP memory_limit {memory_mib}\nP mpi_nbcpu 1\n"
            f"F comm {CASE} D 1\nF libr {inputs} D 91\n"
            f"F libr {native} R 90\nF rmed {result} R 80\n{mesh_line}", encoding="utf-8",
        )
        completed = run_solver([installation["launcher_path"], str(export)], directory, native_budget_s)
        (directory / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (directory / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if (completed.returncode or not native.is_file() or not result.is_file()
                or result.stat().st_size == 0 or any(marker in completed.stdout + completed.stderr
                    for marker in ("<EXCEPTION>", "Traceback (most recent call last)",
                                   "Windows fatal exception", "Fatal Python error"))):
            raise RuntimeError(f"Native physical seed benchmark failed; see {directory}")
        data = json.loads(native.read_text(encoding="utf-8"))
        audit = data["audit"]
        if tetrahedral:
            _, point_counts = np.unique(np.asarray(audit["damage_point_ids"])[:, 0],
                                        return_counts=True)
            if (len(point_counts) != 6 * lateral_divisions ** 2 * (specimen_divisions + 1)
                    or not np.all(point_counts == 4)):
                raise ValueError("Native tetrahedral specimen requires four Gauss points per tetrahedron")
        report["checks"] = history_checks(data, contract, times, resolved)
        surrounding = report["checks"]["final_initially_intact_damage_max"]
        if resolved:
            report["specimen_metrics"] = specimen_metrics(data, specimen_divisions, elevation,
                                                          lateral_divisions, normal_grading_power,
                                                          lateral_width_m, tetrahedral)
            if finite_sector:
                report["finite_sector_metrics"] = finite_sector_metrics(data, input_values,
                                                                        report["specimen_metrics"])
        python = bundled_python_for(installation["launcher_path"])
        if python is None:
            raise RuntimeError("Native seed visualisation requires the bundled MED interpreter")
        extractor = Path(__file__).with_name("extract_damage_history.py")
        extraction = subprocess.run(
            [str(python), str(extractor), str(result), str(directory), "--labels", "SEED"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if extraction.returncode:
            raise RuntimeError(f"Native seed visualisation failed: {extraction.stdout}\n{extraction.stderr}")
        visualisation = json.loads((directory / "damage_series.json").read_text(encoding="utf-8"))
        if ([row["time"] for row in visualisation["SEED"]["snapshots"]] != times
                or not (directory / visualisation["SEED"]["pvd"]).is_file()
                or any(not (directory / row["vtu"]).is_file() for row in visualisation["SEED"]["snapshots"])):
            raise ValueError("Native seed specimen has incomplete ParaView time history")
        report["visualisation"] = visualisation
        from Aster.damage_benchmark import visualisation_warnings
        report["warnings"] = [
            (f"The synthetic specimen is {lateral_width_m:g} x {lateral_width_m:g} x 1 m "
             if resolved else "The synthetic cube is 0.004 m across ") +
            "versus the 0.5 m native damage band. "
            f"Initially intact points reach damage {surrounding:.9g} at small imposed strain. "
            "This nonlocal response is not verified crack propagation or physical accuracy.",
            "The synthetic specimen does not validate full-dam seed equilibration, regularisation "
            "convergence or equivalence to the FEniCSx AT2 seed.",
        ]
        report["warnings"].extend(visualisation_warnings(visualisation, "native_seed.json"))
        if finite_sector and report["finite_sector_metrics"]["maximum_final_damage_in_outer_lateral_0_05_m"] > 1e-7:
            report["warnings"].append(
                "Finite-sector damage reaches the sampled lateral boundary shell; specimen extent needs checking.")
        report["status"] = "seed_placement_and_irreversibility_passed"
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specimen-divisions", type=int, choices=(8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
                        help="Localization-normal resolution of the planar specimen")
    parser.add_argument("--lateral-divisions", type=int, default=8, choices=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--lateral-width-m", type=float, default=1.,
                        help="Finite-sector domain width; retain physical spacing when enlarging")
    parser.add_argument("--line-search", action="store_true",
                        help="Use native CORDE line search for resolved specimen Newton iterations")
    parser.add_argument("--normal-grading-power", type=int, default=1, choices=(1, 2),
                        help="Grade normal cells towards the exact seed-layer boundary")
    parser.add_argument("--finite-sector", action="store_true",
                        help="Exercise finite radial/angular seed edges; not the infinite-planar reference")
    parser.add_argument("--tetrahedral", action="store_true",
                        help="Separate finite-sector TETRA10 element-family diagnostic")
    parser.add_argument("--native-budget-s", type=int, default=240)
    parser.add_argument("--memory-mib", type=int, default=2048,
                        help="Native allocation for this developer benchmark; check available host RAM")
    parser.add_argument("--allow-extended-budget", action="store_true",
                        help="Use only with explicit approval for longer developer verification")
    args = parser.parse_args()
    print(json.dumps(run(args.specimen_divisions, args.native_budget_s, args.allow_extended_budget,
                         args.lateral_divisions, args.line_search, args.normal_grading_power,
                         args.finite_sector, args.lateral_width_m, args.memory_mib,
                         args.tetrahedral), indent=2))
