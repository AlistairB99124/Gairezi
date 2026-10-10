"""Build and later assemble the independently approved structural regions."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from structural_model import (
    audit_combined_structure,
    build_combined_structure,
)

from Elmer.regions.combined_structure import write_combined_gmsh, write_combined_vtu
from Elmer.regions.plinth import audit_plinth, build_plinth, write_gmsh, write_vtu


OUTPUT_DIR = Path(__file__).resolve().parent / "structure_output"
COMBINED_SOLVER_DIR = OUTPUT_DIR / "combined_solver"
COMBINED_SOLVER_MPI_PROCESSES = 4
LOAD_CASES_PATH = Path(__file__).resolve().parent / "load_cases.json"


def convert_with_elmergrid(
    mesh_path: Path,
    output_dir: Path,
    partition_count: int | None = None,
) -> bool:
    executable = shutil.which("ElmerGrid")
    if executable is None:
        return False
    if output_dir.exists():
        shutil.rmtree(output_dir)
    command = [executable, "14", "2", str(mesh_path), "-autoclean"]
    if partition_count is not None:
        command.extend(["-metiskway", str(partition_count)])
    command.extend(["-out", str(output_dir)])
    subprocess.run(
        command,
        check=True,
        cwd=mesh_path.parent,
    )
    return True


def build_plinth_region() -> dict[str, object]:
    mesh = build_plinth(ROOT)
    mesh_path = OUTPUT_DIR / "plinth.msh"
    preview_path = OUTPUT_DIR / "plinth.vtu"
    write_gmsh(mesh, mesh_path)
    write_vtu(mesh, preview_path)
    report = audit_plinth(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, OUTPUT_DIR / "plinth_mesh")
    return report


def build_combined_region() -> dict[str, object]:
    mesh = build_combined_structure(ROOT)
    mesh_path = OUTPUT_DIR / "combined_structure.msh"
    preview_path = OUTPUT_DIR / "combined_structure.vtu"
    write_combined_gmsh(mesh, mesh_path)
    write_combined_vtu(mesh, preview_path)
    report = audit_combined_structure(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, OUTPUT_DIR / "combined_structure_mesh")
    return report


def apply_load_case(mesh, path: Path) -> None:
    """Override solver properties with the selected Elmer load case."""
    payload = json.loads(path.read_text())
    material = payload["material"]
    loads = payload["loads"]
    bedrock_support = loads["bedrock_support"]
    plinth_support = loads["plinth_support"]
    mesh.material = replace(
        mesh.material,
        density_kg_m3=float(material["density"]),
        youngs_modulus_pa=float(material["youngs_modulus"]),
        poissons_ratio=float(material["poisson_ratio"]),
        tensile_strength_pa=float(material["tensile_strength"]),
    )
    mesh.loads = replace(
        mesh.loads,
        gravity_z_m_s2=float(loads["gravity"]),
        maximum_water_height_m=float(loads["water_height"]),
        peak_water_pressure_pa=float(loads["water_density"]) * abs(float(loads["gravity"])) * float(loads["water_height"]),
        water_density_kg_m3=float(loads["water_density"]),
        tailwater_head_m=float(loads["tailwater_head"]),
        overflow_head_m=float(loads["overflow_head"]),
    )
    mesh.foundation_support = replace(
        mesh.foundation_support,
        spring_x_n_per_m3=float(bedrock_support["spring_x_n_per_m3"]),
        spring_y_n_per_m3=float(bedrock_support["spring_y_n_per_m3"]),
        spring_z_n_per_m3=float(bedrock_support["spring_z_n_per_m3"]),
    )
    mesh.plinth_support = replace(
        mesh.plinth_support,
        spring_x_n_per_m3=float(plinth_support["spring_x_n_per_m3"]),
        spring_y_n_per_m3=float(plinth_support["spring_y_n_per_m3"]),
        spring_z_n_per_m3=float(plinth_support["spring_z_n_per_m3"]),
    )


def write_combined_solver_input(mesh, path: Path) -> None:
    gravity_bodyforce = mesh.material.density_kg_m3 * mesh.loads.gravity_z_m_s2
    pressure_gradient = mesh.loads.peak_water_pressure_pa / mesh.loads.maximum_water_height_m
    foundation_support = mesh.foundation_support
    foundation_spring_x = foundation_support.spring_x_n_per_m3
    foundation_spring_y = foundation_support.spring_y_n_per_m3
    foundation_spring_z = foundation_support.spring_z_n_per_m3
    plinth_support = mesh.plinth_support
    plinth_spring_x = plinth_support.spring_x_n_per_m3
    plinth_spring_y = plinth_support.spring_y_n_per_m3
    plinth_spring_z = plinth_support.spring_z_n_per_m3
    water_density = mesh.loads.water_density_kg_m3
    tailwater_head = mesh.loads.tailwater_head_m
    overflow_head = mesh.loads.overflow_head_m
    foundation_elevation = min(z_m for _, _, z_m in mesh.nodes)
    tailwater_elevation = foundation_elevation + tailwater_head
    water_pressure_gradient = water_density * abs(mesh.loads.gravity_z_m_s2)
    overflow_surcharge = water_pressure_gradient * overflow_head
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'''Header
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
        Real MATC "-{pressure_gradient:.12g} * (0.0 - tx) * (tx < 0.0)"
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

Boundary Condition 5
    Name = "WallPlinthSpring"
    Target Boundaries(1) = 8
    Spring 1 = Real {plinth_spring_x:.12g}
    Spring 2 = Real {plinth_spring_y:.12g}
    Spring 3 = Real {plinth_spring_z:.12g}
End
''')


def solve_combined_structure() -> dict[str, object]:
    mesh = build_combined_structure(ROOT)
    apply_load_case(mesh, LOAD_CASES_PATH)
    mesh_path = COMBINED_SOLVER_DIR / "combined_structure.msh"
    write_combined_gmsh(mesh, mesh_path)
    if not convert_with_elmergrid(
        mesh_path,
        COMBINED_SOLVER_DIR / "mesh",
        partition_count=COMBINED_SOLVER_MPI_PROCESSES,
    ):
        raise RuntimeError("ElmerGrid was not found on PATH")
    sif_path = COMBINED_SOLVER_DIR / "dam_model.sif"
    write_combined_solver_input(mesh, sif_path)
    (COMBINED_SOLVER_DIR / "results").mkdir(exist_ok=True)
    executable = shutil.which("ElmerSolver_mpi")
    launcher = shutil.which("mpirun")
    if executable is None or launcher is None:
        raise RuntimeError("ElmerSolver_mpi and mpirun are required for the combined MPI solve")
    environment = os.environ.copy()
    environment.setdefault("ELMER_HOME", "/usr/local")
    subprocess.run(
        [launcher, "-np", str(COMBINED_SOLVER_MPI_PROCESSES), executable, sif_path.name],
        check=True,
        cwd=COMBINED_SOLVER_DIR,
        env=environment,
    )
    result_path = COMBINED_SOLVER_DIR / "results" / "combined_results_mpi_t0001.pvtu"
    if not result_path.is_file():
        raise RuntimeError(f"Elmer did not create {result_path}")
    report = audit_combined_structure(mesh)
    report["solver_input"] = str(sif_path)
    report["stress_result"] = str(result_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "region",
        nargs="?",
        default="plinth",
        choices=("plinth", "combined", "combined-solve"),
    )
    args = parser.parse_args()
    if args.region == "plinth":
        print(json.dumps(build_plinth_region(), indent=2))
    elif args.region == "combined":
        print(json.dumps(build_combined_region(), indent=2))
    elif args.region == "combined-solve":
        print(json.dumps(solve_combined_structure(), indent=2))


if __name__ == "__main__":
    main()
