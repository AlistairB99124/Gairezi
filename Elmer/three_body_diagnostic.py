"""Isolated three-body Elmer check (bonded or stiff-spring joints); not frictional."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np

from .bedrock_preview import configured_rock_mesh, survey_coordinates
from .model_config import load_config, validate_config
from .regions.combined_structure import build_combined_structure, write_combined_gmsh
from .regions.tetrahedral import _CELL_FACES, tetrahedralize_structure
from .structure_assembly import (
    convert_with_elmergrid, mpi_launcher, write_combined_solver_input,
)


JOINT_MODELS = ("bonded", "spring")
# Boundary tags for the two sides of each spring joint (8/9 follow FACE_GROUP_IDS).
WALL_SIDE, PLINTH_TOP, PLINTH_SEAT, ROCK_SEAT = 8, 9, 11, 12
SPRING_PAIRS = {"wall_plinth": (WALL_SIDE, PLINTH_TOP), "plinth_bedrock": (PLINTH_SEAT, ROCK_SEAT)}
SPRING_SUPPORTS = {"wall_plinth": "plinth", "plinth_bedrock": "bedrock"}


def split_wall_from_plinth(concrete) -> tuple[list, list, list, int]:
    """Give wall cells their own copies of the wall/plinth joint nodes.

    Returns nodes, cells and boundaries with the wall side tagged WALL_SIDE and
    the plinth side tagged PLINTH_TOP, plus the number of joint faces.
    """
    faces: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = {}
    for (kind, cell), region in zip(concrete.cells, concrete.region_ids, strict=True):
        for pattern in _CELL_FACES[kind]:
            face = tuple(cell[i] for i in pattern)
            faces.setdefault(tuple(sorted(face)), []).append((region, face))
    joint = {key: dict(entries) for key, entries in faces.items()
             if sorted(region for region, _ in entries) == [1, 2]}
    joint_nodes = {node for key in joint for node in key}
    wall_nodes = {n for (_, c), r in zip(concrete.cells, concrete.region_ids) if r == 2 for n in c}
    plinth_nodes = {n for (_, c), r in zip(concrete.cells, concrete.region_ids) if r == 1 for n in c}
    if not joint or (wall_nodes & plinth_nodes) != joint_nodes:
        raise ValueError("Wall and plinth must meet only on conforming shared faces")
    nodes = list(concrete.nodes)
    copy = {}
    for node in sorted(joint_nodes):
        copy[node] = len(nodes)
        nodes.append(nodes[node])
    def wall_side(face: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(copy.get(n, n) for n in face)
    cells = [
        (kind, wall_side(cell) if region == 2 else cell)
        for (kind, cell), region in zip(concrete.cells, concrete.region_ids, strict=True)
    ]
    boundaries = []
    for tag, face in concrete.boundaries:
        owners = faces.get(tuple(sorted(face)))
        if owners is None or len(owners) != 1:
            raise ValueError(f"Boundary face {face} is not an exterior cell face")
        boundaries.append((tag, wall_side(face) if owners[0][0] == 2 else face))
    for sides in joint.values():
        boundaries.append((WALL_SIDE, wall_side(sides[2])))
        boundaries.append((PLINTH_TOP, sides[1]))
    return nodes, cells, boundaries, len(joint)


def interface_stiffness(config: dict) -> dict[str, dict[str, float]]:
    """Directional traction/jump coefficients in global Cartesian axes (N/m^3)."""
    validate_config(config)
    return {
        name: {
            axis: float(config["geometry"][region]["support"]["spring_n_per_m3"][axis])
            for axis in ("x", "y", "z")
        }
        for name, region in SPRING_SUPPORTS.items()
    }


def prepare(
    root: Path, output: Path,
    stream_margin_m: float | None = None, bank_margin_m: float | None = None,
    joints: str = "bonded",
    bottom_z_m: float | None = None,
    *,
    integrated: bool = False,
) -> dict[str, object]:
    if joints not in JOINT_MODELS:
        raise ValueError(f"joints must be one of {JOINT_MODELS}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory to exclude stale results")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "config.json", output / "config.json")
    config = load_config(output / "config.json")
    concrete = build_combined_structure(output)
    concrete.nodes = [tuple(p) for p in survey_coordinates(
        np.asarray(concrete.nodes),
        config["geometry"]["wall"]["centerline_radius_m"], 80.,
    )]
    if integrated:
        concrete = tetrahedralize_structure(concrete)
    rock = configured_rock_mesh(
        config, np.asarray(concrete.nodes),
        [face for tag, face in concrete.boundaries if tag == 1],
        stream_margin_m, bank_margin_m, bottom_z_m,
    )
    rock_points, rock_blocks, geometry = rock.points, rock.blocks, rock.report
    stiffness = interface_stiffness(config) if joints == "spring" else {}
    seat_faces = [face for tag, face in concrete.boundaries if tag == 1]
    if joints == "spring":
        nodes, concrete_cells, concrete_boundaries, wall_joint_faces = split_wall_from_plinth(concrete)
        # Rock gets its own seat nodes; the springs, not shared nodes, join it.
        index = {}
    else:
        nodes, concrete_cells, concrete_boundaries = (
            list(concrete.nodes), concrete.cells, concrete.boundaries,
        )
        wall_joint_faces = 0
        index = {tuple(round(v, 8) for v in p): i for i, p in enumerate(nodes)}
    mapping = []
    for coordinates in rock_points:
        point = tuple(coordinates)
        key = tuple(round(v, 8) for v in point)
        if key in index:
            mapping.append(index[key])
        else:
            mapping.append(len(nodes))
            nodes.append(point)
    rock_cells = [
        (kind, tuple(mapping[int(n)] for n in cell))
        for kind, block in rock_blocks.items() for cell in block
    ]
    cells = concrete_cells + rock_cells
    regions = concrete.region_ids + [6] * len(rock_cells)
    patterns = {**_CELL_FACES, 4: ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
                7: ((0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))}
    # Triangulate exterior/interface quads consistently for a topology audit.
    from .regions.tetrahedral import split_face
    face_counts = Counter()
    memberships: dict[tuple[int, ...], set[int]] = {}
    for (kind, cell), region in zip(concrete_cells, concrete.region_ids, strict=True):
        for pattern in patterns[kind]:
            face = tuple(cell[i] for i in pattern)
            for tri in split_face(face):
                key = tuple(sorted(tri))
                face_counts[key] += 1
                memberships.setdefault(key, set()).add(region)
    # Only rock-top faces can meet the concrete. The mesher audits rock volume.
    top_faces = [tuple(mapping[int(n)] for n in tri) for tri in rock.top_faces]
    for face in top_faces:
        key = tuple(sorted(face))
        face_counts[key] += 1
        memberships.setdefault(key, set()).add(6)
    if max(face_counts.values()) > 2:
        raise ValueError("Non-manifold three-body mesh")
    seat = [key for key, value in memberships.items() if value == {1, 6}]
    if joints == "spring":
        if seat or any(value == {1, 2} for value in memberships.values()):
            raise ValueError("Spring joints must not share nodes between bodies")
        def located(face: tuple[int, ...]) -> tuple:
            return tuple(sorted(tuple(round(v, 7) for v in nodes[n]) for n in face))
        plinth_seat = {located(tri) for face in seat_faces for tri in split_face(face)}
        rock_seat = [face for face in top_faces if located(face) in plinth_seat]
        if len(rock_seat) != geometry["seat_triangles"] or len(rock_seat) != len(plinth_seat):
            raise ValueError("Rock and plinth seat faces do not coincide")
        seat_count = len(rock_seat)
    elif len(seat) != geometry["seat_triangles"]:
        raise ValueError("Rock/plinth seat is not fully connected")
    else:
        seat_count = len(seat)
    boundaries = [(tag, face) for tag, face in concrete_boundaries if tag != 1]
    if joints == "spring":
        boundaries.extend((PLINTH_SEAT, face) for face in seat_faces)
        boundaries.extend((ROCK_SEAT, face) for face in rock_seat)
    bottom_faces = [tuple(mapping[int(n)] for n in tri) for tri in rock.bottom_faces]
    boundaries.extend((10, face) for face in bottom_faces)
    boundary_map = {tag: i+1 for i, tag in enumerate(sorted({tag for tag, _ in boundaries}))}
    mesh = replace(
        concrete, nodes=nodes, cells=cells,
        region_ids=[3 if region == 6 else region for region in regions],
        boundaries=[(boundary_map[tag], face) for tag, face in boundaries],
    )
    write_combined_gmsh(mesh, output / "three_body.msh")
    # Preserve concrete water datum; the added rock must not lower tailwater.
    path = output / ("dam_model.sif" if integrated else "diagnostic.sif")
    write_combined_solver_input(concrete, path)
    text = path.read_text()
    start = text.index("Boundary Condition 1\n")
    end = text.index("Boundary Condition 2\n", start)
    text = text[:start] + '''Boundary Condition 1
    Name = "DiagnosticFixedRockBottom"
    Target Boundaries(1) = 10
    Displacement 1 = Real 0
    Displacement 2 = Real 0
    Displacement 3 = Real 0
End

''' + text[end:]
    rock = config["material"]["bedrock"]
    text += f'''
Body 3
    Name = "BedrockBody"
    Equation = 1
    Material = 2
    Body Force = 2
End

Material 2
    Name = "Granite"
    Youngs Modulus = Real {rock["youngs_modulus"]:.12g}
    Poisson Ratio = Real {rock["poisson_ratio"]:.12g}
    Density = Real {rock["density"]:.12g}
End

Body Force 2
    Stress Bodyforce 3 = Real {rock["density"] * config["loads"]["gravity"]:.12g}
End
'''
    text = re.sub(
        r"Target Boundaries\(1\) = (\d+)",
        lambda match: f"Target Boundaries(1) = {boundary_map[int(match.group(1))]}", text,
    )
    if integrated:
        # Fixed-level fill avoids the runaway ILUT factors on the full rock mesh.
        text = text.replace(
            "Linear System Preconditioning = ILUT",
            "Linear System Preconditioning = ILU1\n"
            "    Linear System ILU Factor = Real 0.01",
        ).replace(
            "Linear System ILUT Tolerance = 1.0e-4",
            "Linear System Residual Output = 20",
        )
    if joints == "spring":
        text = text.replace(
            "    Variable DOFs = 3\n",
            "    Variable DOFs = 3\n    Apply Mortar BCs = Logical True\n", 1,
        )
        text += spring_conditions(text, boundary_map, stiffness)
    joint_note = (
        f"directional springs (global x/y/z) {stiffness} "
        "N/m^3 from geometry.plinth/bedrock.support at wall/plinth and plinth/rock"
        if joints == "spring" else "bonded interfaces"
    )
    result_name = "three_body_results" if integrated else "three_body_diagnostic"
    text = text.replace(
        "! Provisional bonded wall/plinth: shared nodes, no joint opening/sliding.\n"
        "! Foundation support only; geometry.plinth.support is inactive.\n",
        "",
    )
    path.write_text(
        f"! {'THREE-BODY ANALYSIS' if integrated else 'DIAGNOSTIC ONLY'}: "
        f"{joint_note}; fixed rock bottom; zero uplift.\n"
        "! Free outer rock sides. Configured graded rock mesh. Not a frictional/design result.\n"
        + text.replace("combined_results_mpi", result_name),
    )
    report = {
        "assumptions": [
            "linear directional springs in global x/y/z (carry tension) at wall/plinth and plinth/rock"
            if joints == "spring" else "bonded wall/plinth and plinth/rock",
            "zero uplift",
            f"fixed rock bottom z={geometry['bottom_z_m']:g} m", "free rock sides",
            "simultaneous gravity and existing concrete water loads",
            "80 m chainage reference",
            (
                f"distance-graded rock: {geometry['mesh_grading']}"
                if geometry.get("mesh_method") else
                f"nominal rock spacing {geometry['element_size_m']:g} m; "
                f"{geometry['vertical_layers']} vertical layers"
            ),
        ],
        "purpose": (
            "Initial three-body linear analysis, NOT engineering validation"
            if integrated else "Solver integration diagnostic, NOT engineering validation"
        ),
        "joints": joints,
        "interface_stiffness_n_per_m3": stiffness,
        "interface_stiffness_sources": {
            name: f"geometry.{region}.support.spring_n_per_m3"
            for name, region in SPRING_SUPPORTS.items()
        } if joints == "spring" else {},
        "nodes": len(nodes),
        "cells_by_body": dict(Counter(regions)),
        "solver_body_ids": {"plinth": 1, "wall": 2, "bedrock": 3},
        "boundary_id_map": boundary_map,
        "wall_plinth_joint_faces": wall_joint_faces,
        "plinth_rock_seat_triangles": seat_count,
        "geometry": geometry,
        "concrete_element_types": sorted({kind for kind, _ in concrete_cells}),
        "wall_plinth_cutback": concrete.wall_plinth_cutback,
        "linear_solver": (
            {"method": "BiCGStabl", "preconditioner": "ILU1",
             "ilu_factor": 0.01, "convergence_tolerance": 1e-10}
            if integrated else
            {"method": "BiCGStabl", "preconditioner": "ILUT",
             "ilut_tolerance": 1e-4, "convergence_tolerance": 1e-10}
        ),
        "solver_input": path.name,
        "result_name": result_name,
        "result": None,
    }
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2))
    return report


def spring_conditions(
    text: str, boundary_map: dict[int, int], stiffness: dict[str, dict[str, float]],
) -> str:
    """Mortar spring pairs: traction = k * (relative displacement), N/m^3."""
    first = max(int(n) for n in re.findall(r"^Boundary Condition (\d+)$", text, re.M)) + 1
    sections = []
    for offset, (name, (slave, master)) in enumerate(SPRING_PAIRS.items()):
        slave_bc, master_bc = first + 2 * offset, first + 2 * offset + 1
        coefficients = "".join(
            f"    Mortar BC Coefficient Displacement {i} = Real {stiffness[name][axis]:.12g}\n"
            for i, axis in enumerate(("x", "y", "z"), start=1)
        )
        sections.append(
            f"\nBoundary Condition {slave_bc}\n"
            f"    Name = \"{name}_spring\"\n"
            f"    Target Boundaries(1) = {boundary_map[slave]}\n"
            f"    Mortar BC = Integer {master_bc}\n"
            "    Galerkin Projector = Logical True\n"
            f"{coefficients}End\n"
            f"\nBoundary Condition {master_bc}\n"
            f"    Name = \"{name}_master\"\n"
            f"    Target Boundaries(1) = {boundary_map[master]}\n"
            "End\n"
        )
    return "".join(sections)


def spring_partition_options(boundary_map: dict[int, int]) -> tuple[str, ...]:
    # Elmer's mortar projector silently gives wrong answers when the two sides
    # of a joint sit on different ranks; keep every joint face on one rank.
    tags = [str(boundary_map[tag]) for pair in SPRING_PAIRS.values() for tag in pair]
    return ("-partdual", "-connect", *tags, "-halobc")


def check_spring_partitions(mesh: Path, partitions: int, boundary_map: dict[int, int]) -> None:
    """Fail unless both sides of every spring joint are on the same single rank."""
    owners: dict[int, set[int]] = {}
    for part in range(1, partitions + 1):
        with (mesh / f"partitioning.{partitions}" / f"part.{part}.boundary").open() as stream:
            for line in stream:
                owners.setdefault(int(line.split()[1]), set()).add(part)
    for name, pair in SPRING_PAIRS.items():
        ranks = [owners.get(boundary_map[tag], set()) for tag in pair]
        if len(ranks[0]) != 1 or ranks[0] != ranks[1]:
            raise RuntimeError(f"{name} spring sides were split across ranks {ranks}")


def solve(output: Path, timeout: int | None = 480, partitions: int = 4) -> Path:
    if partitions < 1 or (timeout is not None and timeout <= 0):
        raise ValueError("Partitions and optional solver timeout must be positive")
    report_path = output / "diagnostic.json"
    report = json.loads(report_path.read_text())
    if (output / "results").exists():
        raise ValueError("Results directory already exists; prepare a fresh run to exclude stale results")
    executable = shutil.which("ElmerSolver_mpi")
    if executable is None:
        raise RuntimeError("ElmerSolver_mpi unavailable")
    launcher, option = mpi_launcher()
    pvpython_executable()
    spring = report.get("joints") == "spring"
    boundary_map = {int(k): v for k, v in report["boundary_id_map"].items()}
    options = spring_partition_options(boundary_map) if spring else ()
    print(f"Converting and partitioning three-body mesh ({partitions} rank(s))...", flush=True)
    if not convert_with_elmergrid(
        output / "three_body.msh", output / "mesh",
        partitions if partitions > 1 else None, options if partitions > 1 else (),
    ):
        raise RuntimeError("ElmerGrid unavailable")
    if spring and partitions > 1:
        check_spring_partitions(output / "mesh", partitions, boundary_map)
    (output / "results").mkdir()
    environment = os.environ.copy()
    environment.setdefault("ELMER_HOME", str(Path(executable).resolve().parent.parent))
    log_path = output / "solver.log"
    print(f"Starting Elmer; live output below is also saved to {log_path}", flush=True)
    with log_path.open("w") as log:
        worker = subprocess.Popen(
            [launcher, option, str(partitions), executable,
             report.get("solver_input", "diagnostic.sif")], cwd=output,
            env=environment, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            code = wait_for_solver(worker, log_path, timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(worker.pid), "/T", "/F"], check=True)
            raise RuntimeError("Solver exceeded the requested time limit; inspect solver.log")
        except KeyboardInterrupt:
            subprocess.run(["taskkill", "/PID", str(worker.pid), "/T", "/F"], check=True)
            raise
    if code != 0:
        raise RuntimeError(f"Elmer exited {code}; inspect {output / 'solver.log'}")
    result = output / "results" / f"{report.get('result_name', 'three_body_diagnostic')}_t0001.pvtu"
    if not result.is_file() and not (partitions == 1 and result.with_suffix(".vtu").is_file()):
        raise RuntimeError("Elmer did not produce its expected PVTU")
    attribute_result(output, result, timeout=None if timeout is None else 90)
    gaps = measure_joint_gaps(result, timeout=None if timeout is None else 120)
    if not gaps["finite_displacement"]:
        raise RuntimeError("Solved displacement contains non-finite values")
    if not gaps["finite_stress"]:
        raise RuntimeError("Solved stress contains non-finite values")
    if spring:
        check_spring_gaps(gaps, report["interface_stiffness_n_per_m3"])
    report["result"] = str(result)
    report["solver_exit_code"] = code
    report["partitions"] = partitions
    report["joint_gaps"] = gaps
    report_path.write_text(json.dumps(report, indent=2))
    return result


def wait_for_solver(
    worker: subprocess.Popen, log_path: Path, timeout: int | None,
) -> int:
    """Relay solver output and heartbeats without imposing a default run limit."""
    started = time.monotonic()
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        while True:
            elapsed = time.monotonic() - started
            if timeout is not None and elapsed >= timeout:
                raise subprocess.TimeoutExpired(worker.args, timeout)
            interval = 30. if timeout is None else min(30., timeout - elapsed)
            try:
                code = worker.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                text = stream.read()
                if text:
                    print(text, end="" if text.endswith("\n") else "\n", flush=True)
                print(f"Elmer running: {time.monotonic() - started:.0f} s elapsed"
                      " (no automatic timeout unless requested).", flush=True)
            else:
                text = stream.read()
                if text:
                    print(text, end="" if text.endswith("\n") else "\n", flush=True)
                return code


def check_spring_gaps(gaps: dict, stiffness: dict[str, dict[str, float]]) -> None:
    """Reject the silent-garbage pattern: missing joints or implausible openings.

    Check each global-axis traction using its configured stiffness. The 100 MPa
    component ceiling is a plausibility check, not an engineering acceptance limit.
    """
    for name in SPRING_PAIRS:
        joint = gaps["joints"][name]
        if joint["locations"] == 0:
            raise RuntimeError(f"No {name} joint locations found in the result")
        maximum = np.asarray(joint["max_abs_gap_by_axis_m"], dtype=float)
        if maximum.shape != (3,) or not np.isfinite(maximum).all() or np.any(maximum < 0):
            raise RuntimeError(f"Invalid {name} directional joint gaps")
        coefficients = np.array([stiffness[name][axis] for axis in ("x", "y", "z")])
        if np.any(maximum * coefficients > 100e6):
            raise RuntimeError(f"{name} joint gap implies >100 MPa traction; result rejected")


def pvpython_executable() -> str:
    executable = shutil.which("pvpython")
    if executable is None and os.name == "nt":
        candidates = sorted(Path(os.environ.get("ProgramFiles", r"C:\Program Files")).glob(
            r"ParaView*\bin\pvpython.exe",
        ))
        if candidates:
            executable = str(candidates[-1])
    if executable is None:
        raise RuntimeError("ParaView pvpython is required to post-process the diagnostic result")
    return executable


def measure_joint_gaps(result: Path, timeout: int | None = 120) -> dict[str, object]:
    completed = subprocess.run(
        [pvpython_executable(), str(Path(__file__).with_name("measure_joint_gaps.py")), str(result)],
        check=True, timeout=timeout, capture_output=True, text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def attribute_result(output: Path, result: Path, timeout: int | None = 90) -> None:
    """Attach body labels by exact source-cell coordinates, not spatial guesses."""
    subprocess.run(
        [pvpython_executable(), str(Path(__file__).with_name("attribute_diagnostic_result.py")),
         str(output), str(result)], check=True, timeout=timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream-margin-m", type=float, help="Override the configured stream margin (m)")
    parser.add_argument("--bank-margin-m", type=float, help="Override the configured bank margin (m)")
    parser.add_argument("--bottom-z-m", type=float, help="Override the configured rock bottom elevation (m)")
    parser.add_argument("--joints", choices=JOINT_MODELS, default="bonded")
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=480)
    args = parser.parse_args()
    prepare(
        args.root, args.output, args.stream_margin_m, args.bank_margin_m,
        args.joints, args.bottom_z_m,
    )
    print(solve(args.output, args.timeout_s, args.partitions))


if __name__ == "__main__":
    main()
