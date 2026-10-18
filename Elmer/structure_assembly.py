"""Build and later assemble the independently approved structural regions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

ELMER_DIR = Path(__file__).resolve().parent
if str(ELMER_DIR) not in sys.path:
        sys.path.insert(0, str(ELMER_DIR))

from Elmer.structural_model import (
    audit_combined_structure,
    build_combined_structure,
)

from Elmer.regions.combined_structure import write_combined_gmsh, write_combined_vtu
from Elmer.regions.tetrahedral import tetrahedralize_structure
from Elmer.regions.plinth import audit_plinth, build_plinth, write_gmsh, write_vtu
from Elmer.model_config import CONFIG_PATH
from Elmer.regions.combined_structure import load_material, load_environment, load_foundation_support


OUTPUT_DIR = Path(__file__).resolve().parent / "structure_output"
COMBINED_SOLVER_DIR = OUTPUT_DIR / "combined_solver"
COMBINED_SOLVER_MPI_PROCESSES = 4
LOAD_CASES_PATH = CONFIG_PATH


def mpi_launcher() -> tuple[str, str]:
    names = ("mpiexec", "mpirun") if os.name == "nt" else ("mpirun", "mpiexec")
    for name in names:
        executable = shutil.which(name)
        if executable is not None:
            return executable, "-np" if name == "mpirun" else "-n"
    if os.name == "nt":
        directories = (
            os.environ.get("MSMPI_BIN"),
            str(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft MPI" / "Bin"),
        )
        for directory in directories:
            if directory:
                executable = Path(directory) / "mpiexec.exe"
                if executable.is_file():
                    return str(executable), "-n"
    raise RuntimeError(
        "MPI launcher not found. Install MPI and add mpirun or mpiexec to PATH; "
        "on Windows, install Microsoft MPI or set MSMPI_BIN to its Bin directory."
    )


def convert_with_elmergrid(
    mesh_path: Path,
    output_dir: Path,
    partition_count: int | None = None,
    partition_options: tuple[str, ...] = (),
) -> bool:
    executable = shutil.which("ElmerGrid")
    if executable is None:
        return False
    if output_dir.exists():
        shutil.rmtree(output_dir)
    command = [executable, "14", "2", str(mesh_path.resolve()), "-autoclean"]
    if partition_count is not None:
        command.extend(["-metiskway", str(partition_count), *partition_options])
    # ElmerGrid truncates absolute -out paths at a dotted ancestor such as .copilot.
    command.extend(["-out", output_dir.name])
    subprocess.run(
        command,
        check=True,
        cwd=output_dir.parent,
    )
    header = (
        output_dir / f"partitioning.{partition_count}" / "part.1.header"
        if partition_count is not None else output_dir / "mesh.header"
    )
    if not header.is_file():
        raise RuntimeError(f"ElmerGrid did not create the expected mesh header: {header}")
    return True


def build_plinth_region(root: Path = ROOT, output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    mesh = build_plinth(root)
    mesh_path = output_dir / "plinth.msh"
    preview_path = output_dir / "plinth.vtu"
    write_gmsh(mesh, mesh_path)
    write_vtu(mesh, preview_path)
    report = audit_plinth(mesh, root)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, output_dir / "plinth_mesh")
    return report


def build_combined_region(root: Path = ROOT, output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    mesh = build_combined_structure(root)
    mesh_path = output_dir / "combined_structure.msh"
    preview_path = output_dir / "combined_structure.vtu"
    write_combined_gmsh(mesh, mesh_path)
    write_combined_vtu(mesh, preview_path)
    report = audit_combined_structure(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, output_dir / "combined_structure_mesh")
    return report


def apply_load_case(mesh, path: Path) -> None:
    """Apply materials, loads and supports from the root configuration."""
    mesh.material = load_material(path)
    mesh.loads = load_environment(path)
    mesh.foundation_support = load_foundation_support(path)
    mesh.plinth_support = load_foundation_support(path, "plinth")


def write_combined_solver_input(mesh, path: Path) -> None:
    gravity_bodyforce = mesh.material.density_kg_m3 * mesh.loads.gravity_z_m_s2
    pressure_gradient = mesh.loads.water_pressure_gradient_pa_per_m
    foundation_support = mesh.foundation_support
    foundation_spring_x = foundation_support.spring_x_n_per_m3
    foundation_spring_y = foundation_support.spring_y_n_per_m3
    foundation_spring_z = foundation_support.spring_z_n_per_m3
    if any(tag in (8, 9) for tag, _ in mesh.boundaries):
        raise ValueError("Bonded solver input cannot contain separate wall/plinth spring faces; regenerate mesh")
    water_density = mesh.loads.water_density_kg_m3
    tailwater_head = mesh.loads.tailwater_head_m
    overflow_head = mesh.loads.overflow_head_m
    foundation_elevation = min(z_m for _, _, z_m in mesh.nodes)
    tailwater_elevation = foundation_elevation + tailwater_head
    water_pressure_gradient = water_density * abs(mesh.loads.gravity_z_m_s2)
    overflow_surcharge = water_pressure_gradient * overflow_head
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'''! Provisional bonded wall/plinth: shared nodes, no joint opening/sliding.
! Foundation support only; geometry.plinth.support is inactive.
Header
    CHECK KEYWORDS Warn
    Mesh DB "mesh" "/"
    Results Directory "results"
End

Simulation
    Max Output Level = 5
    Coordinate System = Cartesian 3D
    Simulation Type = Steady State
    Steady State Max Iterations = 1
    Output Intervals = 1
End

Constants
    Gravity(4) = Real 0 0 -1 {abs(mesh.loads.gravity_z_m_s2):.12g}
    Stefan Boltzmann = 5.670374419e-8
End

Body 1
    Name = "PlinthBody"
    Equation = 1
    Material = 1
    Body Force = 1
End

Body 2
    Name = "WallBody"
    Equation = 1
    Material = 1
    Body Force = 1
End

Body Force 1
    Stress Bodyforce 3 = Real {gravity_bodyforce:.12g}
End

Equation 1
    Name = "Elasticity"
    Active Solvers(2) = 1 2
End

Solver 1
    Equation = "Elasticity"
    Procedure = "StressSolve" "StressSolver"
    Variable = "Displacement"
    Variable DOFs = 3
    Calculate Stresses = Logical True
    Calculate Principal = Logical True
    Nonlinear System Max Iterations = 1
    Linear System Solver = Iterative
    Linear System Iterative Method = BiCGStabl
    Linear System Preconditioning = ILUT
    Linear System ILUT Tolerance = 1.0e-4
    Linear System Max Iterations = 2000
    Linear System Convergence Tolerance = 1.0e-10
    Linear System Abort Not Converged = Logical True
End

Solver 2
    Equation = "ResultOutput"
    Procedure = "ResultOutputSolve" "ResultOutputSolver"
    Output File Name = "combined_results_mpi"
    Vtu Format = Logical True
End

Material 1
    Name = "Concrete"
    Youngs Modulus = Real {mesh.material.youngs_modulus_pa:.12g}
    Poisson Ratio = Real {mesh.material.poissons_ratio:.12g}
    Density = Real {mesh.material.density_kg_m3:.12g}
End

Boundary Condition 1
    Name = "BedrockBaseSpring"
    Target Boundaries(1) = 1
    Spring 1 = Real {foundation_spring_x:.12g}
    Spring 2 = Real {foundation_spring_y:.12g}
    Spring 3 = Real {foundation_spring_z:.12g}
End

Boundary Condition 2
    Name = "UpstreamHydrostaticPressure"
    Target Boundaries(1) = 2
    Normal Force = Variable Coordinate 3
        Real MATC "-{pressure_gradient:.12g} * ({overflow_head:.12g} - tx) * (tx < {overflow_head:.12g})"
End

Boundary Condition 3
    Name = "DownstreamTailwater"
    Target Boundaries(1) = 3
    Normal Force = Variable Coordinate 3
        Real MATC "-{water_pressure_gradient:.12g} * ({tailwater_elevation:.12g} - tx) * (tx < {tailwater_elevation:.12g})"
End

Boundary Condition 4
    Name = "CrestOverflowSurcharge"
    Target Boundaries(1) = 4
    Normal Force = Real -{overflow_surcharge:.12g}
End

''')


def solve_combined_structure(
    root: Path = ROOT,
    solver_dir: Path = COMBINED_SOLVER_DIR,
    config_path: Path | None = None,
) -> dict[str, object]:
    executable = shutil.which("ElmerSolver_mpi")
    if executable is None:
        raise RuntimeError("ElmerSolver_mpi was not found on PATH; install an MPI-enabled Elmer build.")
    launcher, process_option = mpi_launcher()
    hex_mesh = build_combined_structure(root)
    apply_load_case(hex_mesh, config_path if config_path is not None else root / "config.json")
    report = audit_combined_structure(hex_mesh)
    mesh = tetrahedralize_structure(hex_mesh)
    report["solver_mesh"] = {
        "element_type": "tetrahedron",
        "tetrahedra": len(mesh.cells),
        "boundary_triangles": len(mesh.boundaries),
        "source_cells": len(hex_mesh.cells),
    }
    mesh_path = solver_dir / "combined_structure.msh"
    write_combined_gmsh(mesh, mesh_path)
    if not convert_with_elmergrid(
        mesh_path,
        solver_dir / "mesh",
        partition_count=COMBINED_SOLVER_MPI_PROCESSES,
    ):
        raise RuntimeError("ElmerGrid was not found on PATH")
    sif_path = solver_dir / "dam_model.sif"
    write_combined_solver_input(mesh, sif_path)
    (solver_dir / "results").mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("ELMER_HOME", str(Path(executable).resolve().parent.parent))
    subprocess.run(
        [launcher, process_option, str(COMBINED_SOLVER_MPI_PROCESSES), executable, sif_path.name],
        check=True,
        cwd=solver_dir,
        env=environment,
    )
    result_path = solver_dir / "results" / "combined_results_mpi_t0001.pvtu"
    if not result_path.is_file():
        raise RuntimeError(f"Elmer did not create {result_path}")
    report["solver_input"] = str(sif_path)
    report["stress_result"] = str(result_path)

    from analyze_stress import analyze_result

    analysis_dir = solver_dir / "results" / "analysis"
    try:
        stress_analysis = analyze_result(
            result_path,
            analysis_dir,
            mesh.material.youngs_modulus_pa,
            mesh.material.poissons_ratio,
            mesh.material.tensile_strength_pa,
        )
        report["stress_analysis"] = stress_analysis
    except Exception as error:  # noqa: BLE001 - analysis must not fail a succeeded solve
        report["stress_analysis_error"] = str(error)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "region",
        nargs="?",
        default="plinth",
        choices=("plinth", "combined", "combined-solve"),
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=None,
        help="Directory containing config.json (defaults to the repository root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write generated mesh/preview artifacts (defaults to Elmer/structure_output).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help=(
            "If given, write the structured JSON report to this file instead of stdout. "
            "Use this when stdout is shared with a subprocess (e.g. ElmerGrid/ElmerSolver_mpi) "
            "whose own console output would otherwise corrupt the report."
        ),
    )
    args = parser.parse_args()
    root = args.config_root if args.config_root is not None else ROOT
    output_dir = args.output_dir if args.output_dir is not None else OUTPUT_DIR
    solver_dir = output_dir / "combined_solver" if args.output_dir is not None else COMBINED_SOLVER_DIR
    if args.region == "plinth":
        report = build_plinth_region(root, output_dir)
    elif args.region == "combined":
        report = build_combined_region(root, output_dir)
    else:
        report = solve_combined_structure(root, solver_dir)
    if args.report_path is not None:
        args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
