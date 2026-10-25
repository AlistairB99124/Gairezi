"""Native Code_Aster cracking of a local dam-wall submodel at the frozen shared UI seed.

The submodel is a curved wall block around the seed. Its cut faces follow the
uncracked Elmer baseline displacement scaled with the load factor, while
gravity and water pressure act inside and on its free faces. Cracking only
softens the block; the surrounding dam stays at the uncracked baseline
stiffness, so damage near the cut faces is not trustworthy. The result is a
native ENDO_FISS_EXP response, not AT2 damage, crack opening or validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASE = Path(__file__).with_name("cases") / "crack_submodel.comm"
CONVERTER = Path(__file__).with_name("crack_submodel_mesh.py")
DEFAULT_ELEMENT_SIZE_M = 0.5
DEFAULT_MARGIN_M = 2.0
# Keeps the block off spring-jointed thickness steps, bedrock and foundation.
CLEARANCE_M = 0.25
MINIMUM_MARGIN_M = 0.5
SEED_DAMAGE = 0.999
LOAD_FACTORS = tuple(round(0.1 * step, 10) for step in range(1, 11))
# User jobs are not capped; the launcher still needs a finite number.
UNBOUNDED_S = 30 * 24 * 3600


def uniform_edges(lower: float, upper: float, size: float, required=()) -> list[float]:
    """Uniform spacing that keeps every required coordinate as an exact edge."""
    if not (math.isfinite(lower) and math.isfinite(upper) and upper > lower and size > 0):
        raise ValueError("Uniform edges need finite increasing bounds and a positive size")
    kept = [lower]
    for value in [*sorted(value for value in required if lower < value < upper), upper]:
        if value - kept[-1] > 1e-9:
            kept.append(value)
    kept[-1] = upper
    edges: list[float] = []
    for start, end in zip(kept, kept[1:]):
        count = max(1, math.ceil((end - start) / size - 1e-9))
        edges.extend(np.linspace(start, end, count + 1)[:-1].tolist())
    edges.append(upper)
    return edges


def graded_offsets(extent: float, first: float, size: float, ratio: float = 1.5) -> list[float]:
    """Positive offsets growing geometrically from ``first`` to ``size``, ending at ``extent``."""
    if not (extent > 0 and 0 < first <= size and ratio >= 1):
        raise ValueError("Graded edges need a positive extent and 0 < first <= size")
    offsets, position, step, used = [], 0.0, first, first
    while position + step < extent - 1e-9:
        position += step
        offsets.append(position)
        used, step = step, min(step * ratio, size)
    if offsets and extent - offsets[-1] < 0.5 * used:
        offsets.pop()
    offsets.append(extent)
    return offsets


def normal_edges(lower: float, elevation: float, upper: float, half_thickness: float,
                 size: float) -> list[float]:
    """Edges with one exact seed layer and graded layers either side."""
    if not lower < elevation - half_thickness < elevation + half_thickness < upper:
        raise ValueError("The seed layer must lie strictly inside the submodel height")
    first = min(max(4 * half_thickness, 0.02), size)
    below = [elevation - half_thickness - offset
             for offset in graded_offsets(elevation - half_thickness - lower, first, size)]
    above = [elevation + half_thickness + offset
             for offset in graded_offsets(upper - elevation - half_thickness, first, size)]
    below[-1], above[-1] = lower, upper
    return below[::-1] + [elevation - half_thickness, elevation + half_thickness] + above


def thickness_segment(wall: dict, elevation: float) -> dict:
    """Constant-thickness wall segment containing ``elevation`` (crest at z = 0)."""
    upstream = float(wall["upstream_radius_m"])
    thickness = float(wall["radial_width_m"])
    tops = sorted(float(step["top_z_m"]) for step in wall.get("thickness_steps") or ())
    for step in wall.get("thickness_steps") or ():
        if elevation < float(step["top_z_m"]):
            thickness = float(step["radial_width_m"])
    if any(math.isclose(elevation, top, abs_tol=1e-9) for top in tops):
        raise ValueError("The crack lies on a wall thickness step; choose a different elevation")
    above = [top for top in tops if top > elevation]
    below = [top for top in tops if top < elevation]
    return {
        "upstream_radius_m": upstream, "downstream_radius_m": upstream - thickness,
        "thickness_m": thickness,
        "upper_z_m": min(above) if above else 0.0, "upper_is_crest": not above,
        "lower_z_m": max(below) if below else -math.inf,
    }


def block_layout(config: dict, shared_seed: dict, coordinates: np.ndarray, regions: np.ndarray,
                 element_size_m: float = DEFAULT_ELEMENT_SIZE_M,
                 margin_m: float = DEFAULT_MARGIN_M) -> dict:
    """Choose a wall-only curved block around the frozen seed without moving the seed."""
    from FEniCSx.crack_contract import validate_selected_seed

    if not (math.isfinite(element_size_m) and 0.05 <= element_size_m <= 2.0):
        raise ValueError("Submodel element size must lie between 0.05 and 2 m")
    if not (math.isfinite(margin_m) and MINIMUM_MARGIN_M <= margin_m <= 10.0):
        raise ValueError("Submodel margin must lie between 0.5 and 10 m")
    geometry = validate_selected_seed(shared_seed)["geometry"]
    inner, outer, theta0, theta1 = (float(value) for value in geometry["annular_sector"])
    elevation = float(geometry["elevation_m"])
    half = float(geometry["seed_thickness_m"]) / 2
    wall = config["geometry"]["wall"]
    radius_c = float(wall["centerline_radius_m"])
    segment = thickness_segment(wall, elevation)
    r_in, r_out = segment["downstream_radius_m"], segment["upstream_radius_m"]
    if inner < r_in - 1e-9 or outer > r_out + 1e-9:
        raise ValueError("The frozen seed is not inside one constant-thickness wall segment")
    points = np.asarray(coordinates, dtype=float)
    regions = np.asarray(regions)
    radius = np.hypot(points[:, 0], points[:, 1])
    arc = radius_c * np.arctan2(points[:, 0], points[:, 1])
    z = points[:, 2]
    s0, s1 = radius_c * theta0, radius_c * theta1
    z_lo = max(elevation - margin_m, segment["lower_z_m"] + CLEARANCE_M)
    z_hi = min(elevation + margin_m, segment["upper_z_m"] if segment["upper_is_crest"]
               else segment["upper_z_m"] - CLEARANCE_M)
    crest = segment["upper_is_crest"] and z_hi > -CLEARANCE_M
    if crest:
        z_hi = 0.0
    s_lo, s_hi = s0 - margin_m, s1 + margin_m
    other = (regions != 1) & (radius >= r_in - 0.5) & (radius <= r_out + 0.5)
    # Clear abutments at the seed level first, then the bedrock/foundation below.
    level = other & (z >= elevation - MINIMUM_MARGIN_M) & (z <= z_hi)
    left, right = level & (arc < s0), level & (arc > s1)
    if np.any(left):
        s_lo = max(s_lo, float(arc[left].max()) + CLEARANCE_M)
    if np.any(right):
        s_hi = min(s_hi, float(arc[right].min()) - CLEARANCE_M)
    beneath = other & (arc >= s_lo) & (arc <= s_hi) & (z < elevation)
    if np.any(beneath):
        z_lo = max(z_lo, float(z[beneath].max()) + CLEARANCE_M)
    margins = {"below_m": elevation - z_lo, "above_m": z_hi - elevation,
               "start_m": s0 - s_lo, "end_m": s_hi - s1}
    short = {name: value for name, value in margins.items()
             if value < MINIMUM_MARGIN_M and not (name == "above_m" and crest)}
    if short:
        raise ValueError(
            "The submodel needs at least 0.5 m of wall concrete around the seed; the selected crack "
            f"is too close to a thickness step, bedrock or foundation ({short})")
    inside = ((radius >= r_in - 1e-6) & (radius <= r_out + 1e-6) & (arc >= s_lo - 1e-6)
              & (arc <= s_hi + 1e-6) & (z >= z_lo - 1e-6) & (z <= z_hi + 1e-6))
    if np.any(inside & (regions != 1)):
        raise ValueError("The submodel block contains baseline nodes outside the wall")
    if np.count_nonzero(inside) < 8:
        raise ValueError("The submodel block contains too few baseline wall nodes")
    arc_edges = uniform_edges(s_lo, s_hi, element_size_m, (s0, s1))
    return {
        "segment": segment, "chainage_radius_m": radius_c,
        "theta_edges_rad": [value / radius_c for value in arc_edges],
        "radius_edges_m": uniform_edges(r_in, r_out, element_size_m, (inner, outer)),
        "z_edges_m": normal_edges(z_lo, elevation, z_hi, half, element_size_m),
        "top_is_crest": crest,
        "bounds": {"chainage_m": [s_lo, s_hi], "radius_m": [r_in, r_out], "z_m": [z_lo, z_hi]},
        "margins_m": margins, "element_size_m": element_size_m, "margin_m": margin_m,
    }


def boundary_sources(layout: dict, coordinates: np.ndarray, regions: np.ndarray,
                     displacement: np.ndarray, reach_m: float = 1.5) -> dict:
    """Wall-only baseline nodes near the block, excluding nodes on spring-jointed steps."""
    points = np.asarray(coordinates, dtype=float)
    radius = np.hypot(points[:, 0], points[:, 1])
    arc = layout["chainage_radius_m"] * np.arctan2(points[:, 0], points[:, 1])
    (s_lo, s_hi), (r_in, r_out), (z_lo, z_hi) = (layout["bounds"][key] for key in
                                                 ("chainage_m", "radius_m", "z_m"))
    segment = layout["segment"]
    z = points[:, 2]
    upper = segment["upper_z_m"] + (1e-6 if segment["upper_is_crest"] else -1e-6)
    chosen = ((np.asarray(regions) == 1) & (radius >= r_in - 1e-6) & (radius <= r_out + 1e-6)
              & (arc >= s_lo - reach_m) & (arc <= s_hi + reach_m)
              & (z >= max(z_lo - reach_m, segment["lower_z_m"] + 1e-6)) & (z <= min(z_hi + reach_m, upper)))
    values = np.asarray(displacement, dtype=float)[chosen]
    if np.count_nonzero(chosen) < 20 or not np.isfinite(values).all():
        raise ValueError("Too few finite baseline wall displacements around the submodel")
    return {"coordinates_m": points[chosen].tolist(), "displacement_m": values.tolist()}


def moving_least_squares(sources: np.ndarray, values: np.ndarray, targets: np.ndarray,
                         neighbours: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Locally linear weighted fit; reproduces any linear field exactly."""
    sources, values, targets = (np.asarray(array, dtype=float) for array in (sources, values, targets))
    count = min(neighbours, len(sources))
    if count < 4:
        raise ValueError("Moving least squares needs at least four source points")
    result = np.empty((len(targets), values.shape[1]))
    nearest = np.empty(len(targets))
    for index, target in enumerate(targets):
        offsets = sources - target
        distance = np.linalg.norm(offsets, axis=1)
        chosen = np.argpartition(distance, count - 1)[:count]
        scale = max(float(distance[chosen].max()), 1e-12)
        weights = np.sqrt(np.exp(-(distance[chosen] / scale) ** 2))
        basis = np.column_stack([np.ones(count), offsets[chosen]]) * weights[:, None]
        solution, *_ = np.linalg.lstsq(basis, values[chosen] * weights[:, None], rcond=None)
        result[index] = solution[0]
        nearest[index] = distance[chosen].min()
    return result, nearest


def load_inputs(config_path: Path) -> dict:
    """Shared loads, matching the FEniCSx three-body definitions."""
    from Elmer.model_config import load_config
    from Elmer.regions.combined_structure import load_environment

    config = load_config(config_path)
    loads = load_environment(config_path)
    return {
        "concrete_density_kg_m3": float(config["material"]["concrete"]["density"]),
        "gravity_z_m_s2": loads.gravity_z_m_s2,
        "water_pressure_gradient_pa_per_m": loads.water_pressure_gradient_pa_per_m,
        "upstream_surface_z_m": loads.upstream_surface_z_m,
        "tailwater_head_m": loads.tailwater_head_m,
    }


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def crack_metrics(data: dict, layout: dict, seed_geometry: dict) -> dict:
    """Native damage growth beyond the frozen seed; counts Gauss points, not crack area."""
    audit = data["audit"]
    xyz = np.asarray(audit["coordinates_m"], dtype=float)
    seeded = np.asarray(audit["damage_values"], dtype=float) > 0
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    arc = layout["chainage_radius_m"] * np.arctan2(xyz[:, 0], xyz[:, 1])
    thickness = layout["segment"]["thickness_m"]
    (s_lo, s_hi), _, (z_lo, z_hi) = (layout["bounds"][key] for key in ("chainage_m", "radius_m", "z_m"))
    near_cut = ((arc - s_lo < 2 * layout["element_size_m"]) | (s_hi - arc < 2 * layout["element_size_m"])
                | (xyz[:, 2] - z_lo < 2 * layout["element_size_m"])
                | ((z_hi - xyz[:, 2] < 2 * layout["element_size_m"]) & (not layout["top_is_crest"])))
    history = []
    for state in data["history"]:
        damage = np.asarray(state["damage_values"], dtype=float)
        grown = (damage > 0.95) & ~seeded
        cracked = (damage > 0.95)
        entry = {
            "time": state["time"],
            "load_factor": max(0.0, min(1.0, state["time"] - 1.0)),
            "maximum_damage": float(damage.max()),
            "maximum_new_damage": float(damage[~seeded].max()) if np.any(~seeded) else 0.0,
            "new_points_above_0_95": int(np.count_nonzero(grown)),
            "new_points_above_0_5": int(np.count_nonzero((damage > 0.5) & ~seeded)),
            "new_damage_near_cut_faces_max": float(damage[~seeded & near_cut].max())
            if np.any(~seeded & near_cut) else 0.0,
            "maximum_principal_stress_pa": state["maximum_principal_stress_pa"],
        }
        if np.any(cracked):
            entry["cracked_radial_extent_m"] = float(radius[cracked].max() - radius[cracked].min())
            entry["cracked_through_thickness_fraction"] = min(1.0, entry["cracked_radial_extent_m"] / thickness)
            entry["cracked_chainage_extent_m"] = float(arc[cracked].max() - arc[cracked].min())
            entry["cracked_elevation_extent_m"] = [float(xyz[cracked, 2].min()), float(xyz[cracked, 2].max())]
        history.append(entry)
    inner, outer, theta0, theta1 = seed_geometry["annular_sector"]
    return {
        "basis": "native ENDO_FISS_EXP V1 at Gauss points; counts are not areas or crack openings",
        "seed_radial_extent_m": outer - inner,
        "seed_chainage_extent_m": layout["chainage_radius_m"] * (theta1 - theta0),
        "history": history,
        "final": history[-1],
    }


def run(dam_config: Path, prepared_seed: Path, baseline_fields: Path, output_root: Path, *,
        element_size_m: float = DEFAULT_ELEMENT_SIZE_M, margin_m: float = DEFAULT_MARGIN_M,
        memory_mib: int = 4096, time_limit_s: int = UNBOUNDED_S) -> dict:
    from Aster.fracture_material import material_contract
    from Aster.run_case import bundled_python_for, detect_installation, run_solver
    from Elmer.model_config import load_config
    from FEniCSx.selected_seed import fields_hash, validate_prepared_seed

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_root.resolve() / "aster_crack_submodel" / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    config_bytes = dam_config.read_bytes()
    (directory / "config.json").write_bytes(config_bytes)
    manifest = {
        "status": "running", "scenario_name": "Aster_Crack_Submodel",
        "crack_scenario_names": ["Aster_Crack_Submodel"], "engineering_validated": False,
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "started_at_utc": timestamp, "case_sha256": file_hash(CASE),
        "converter_sha256": file_hash(CONVERTER),
    }
    _write(directory / "manifest.json", manifest)
    started = time.monotonic()
    try:
        prepared = json.loads(prepared_seed.read_text(encoding="utf-8"))
        payload = validate_prepared_seed(prepared, config_bytes)
        shared = payload["generation"]["shared_seed"]
        with np.load(baseline_fields, allow_pickle=False) as archive:
            if "displacement_m" not in archive.files:
                raise ValueError("The baseline field archive has no Elmer displacement; rerun the baseline")
            coordinates, stress, regions, displacement = (
                archive[name] for name in ("coordinates_m", "stress_pa", "node_regions", "displacement_m"))
        if payload["baseline_fields_sha256"] != fields_hash((coordinates, stress, regions)):
            raise ValueError("Prepared UI seed baseline fields do not match this archive")
        if displacement.shape != coordinates.shape:
            raise ValueError("Baseline displacement does not match the baseline nodes")
        config = load_config(dam_config)
        material = material_contract(config)
        layout = block_layout(config, shared, coordinates, regions, element_size_m, margin_m)
        loads = load_inputs(dam_config)
        loads["concrete_minimum_z_m"] = float(coordinates[(regions & 5) > 0, 2].min())
        sources = boundary_sources(layout, coordinates, regions, displacement)
        manifest.update({
            "prepared_seed_sha256": prepared["sha256"], "crack_location": payload["generation"],
            "baseline_fields_sha256": file_hash(baseline_fields),
        })
        _write(directory / "manifest.json", manifest)
        times = [0.0, 1.0, *(1.0 + factor for factor in LOAD_FACTORS)]
        inputs = directory / "inputs.json"
        _write(inputs, {
            "repository_root": str(ROOT), "shared_seed": shared, "seed_damage": SEED_DAMAGE,
            "material": material, "layout": layout, "loads": loads, "times": times,
            "boundary_sources": sources,
        })
        installation = detect_installation()
        if not installation["available"]:
            raise RuntimeError("Native Code_Aster is not installed; the Aster cracking analysis cannot run")
        python = bundled_python_for(installation["launcher_path"])
        if python is None:
            raise RuntimeError("The Aster submodel mesh requires Code_Aster's bundled MED interpreter")
        mesh = directory / "mesh.med"
        conversion = subprocess.run([str(python), str(CONVERTER), str(inputs), str(mesh)],
                                    capture_output=True, text=True, timeout=600, check=False)
        (directory / "conversion.log").write_text(conversion.stdout + conversion.stderr, encoding="utf-8")
        if conversion.returncode or not mesh.is_file() or mesh.stat().st_size == 0:
            raise RuntimeError("Aster submodel mesh conversion failed; see conversion.log")
        native, result = directory / "native_crack.json", directory / "result.med"
        export = directory / "crack.export"
        export.write_text(
            f"P time_limit {time_limit_s}\nP memory_limit {memory_mib}\nP mpi_nbcpu 1\n"
            f"F comm {CASE} D 1\nF libr {inputs} D 91\nF mmed {mesh} D 20\n"
            f"F libr {native} R 90\nF rmed {result} R 80\n", encoding="utf-8")
        print(f"Running native Code_Aster cracking submodel in {directory}", flush=True)
        completed = run_solver([installation["launcher_path"], str(export)], directory, time_limit_s)
        (directory / "solver.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        markers = ("<EXCEPTION>", "Traceback (most recent call last)", "Windows fatal exception",
                   "Fatal Python error")
        if (completed.returncode or not native.is_file() or not result.is_file()
                or result.stat().st_size == 0
                or any(marker in completed.stdout + completed.stderr for marker in markers)):
            raise RuntimeError("Native Code_Aster cracking failed; see solver.log")
        data = json.loads(native.read_text(encoding="utf-8"))
        if [state["time"] for state in data["history"]] != times:
            raise ValueError("Native cracking did not converge every requested load factor")
        extraction = subprocess.run(
            [str(python), str(Path(__file__).with_name("extract_damage_history.py")), str(result),
             str(directory), "--labels", "CRACK"],
            capture_output=True, text=True, timeout=600, check=False)
        if extraction.returncode:
            raise RuntimeError(f"Aster ParaView export failed: {extraction.stdout}\n{extraction.stderr}")
        visualisation = json.loads((directory / "damage_series.json").read_text(encoding="utf-8"))["CRACK"]
        if (not (directory / visualisation["pvd"]).is_file()
                or [row["time"] for row in visualisation["snapshots"]] != times):
            raise ValueError("Aster cracking has incomplete ParaView output")
        metrics = crack_metrics(data, layout, shared["payload"]["geometry"])
        final = metrics["final"]
        report = {
            "scenario_name": "Aster_Crack_Submodel", "solver": "code_aster",
            "constitutive_law": "ENDO_FISS_EXP", "modelisation": "3D_GRAD_VARI",
            "engineering_validated": False,
            "material": material, "layout": layout, "loads": loads,
            "boundary_interpolation": data["boundary_interpolation"],
            "native_seed": {key: data["audit"][key] for key in
                            ("gauss_points", "seeded_gauss_points", "placement", "initial_damage")},
            "metrics": metrics,
            "maximum_damage": final["maximum_damage"],
            "maximum_principal_tensile_stress_pa": final["maximum_principal_stress_pa"],
            "stress_reporting_scope": "wall submodel only; no bedrock or foundation",
            "paraview": {"pvd": visualisation["pvd"], "med": result.name},
            "runtime_s": time.monotonic() - started,
            "limitations": [
                "Cut faces follow the uncracked Elmer baseline displacement, so the surrounding dam "
                "does not soften; damage near cut faces is boundary-driven.",
                "Native ENDO_FISS_EXP damage is not FEniCSx phase-field (AT1/AT2) damage or crack opening.",
                "Crack-face water pressure is not modelled.",
                "Mesh and band-width convergence of the dam submodel are unverified.",
            ],
        }
        if final["new_damage_near_cut_faces_max"] > 0.5:
            report["limitations"].insert(0, "Damage reached a cut-face boundary layer; enlarge the "
                                            "submodel margin before using this result.")
        _write(directory / "crack_report.json", report)
        manifest["status"] = "completed_not_engineering_validated"
    except (RuntimeError, ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
    manifest["runtime_s"] = time.monotonic() - started
    _write(directory / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dam-config", type=Path, required=True)
    parser.add_argument("--prepared-seed", type=Path, required=True)
    parser.add_argument("--baseline-fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--element-size-m", type=float, default=DEFAULT_ELEMENT_SIZE_M)
    parser.add_argument("--margin-m", type=float, default=DEFAULT_MARGIN_M)
    parser.add_argument("--memory-mib", type=int, default=4096)
    parser.add_argument("--time-limit-s", type=int, default=UNBOUNDED_S,
                        help="Developer runtime cap; user jobs are not capped")
    args = parser.parse_args()
    manifest = run(args.dam_config, args.prepared_seed, args.baseline_fields, args.output,
                   element_size_m=args.element_size_m, margin_m=args.margin_m,
                   memory_mib=args.memory_mib, time_limit_s=args.time_limit_s)
    print(json.dumps({key: manifest.get(key) for key in ("status", "error", "runtime_s")}, indent=2))
    return 0 if manifest["status"] == "completed_not_engineering_validated" else 1


if __name__ == "__main__":
    sys.exit(main())
