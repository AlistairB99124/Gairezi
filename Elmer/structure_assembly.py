"""Build and later assemble the independently approved structural regions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from regions.center_wall import (
    audit_center_wall,
    build_center_wall,
    write_gmsh as write_center_wall_gmsh,
    write_vtu as write_center_wall_vtu,
)
from regions.combined_structure import (
    audit_combined_structure,
    build_combined_structure,
    write_combined_gmsh,
    write_combined_vtu,
)
from regions.left_wall import (
    audit_left_wall,
    build_left_wall,
    write_gmsh as write_left_wall_gmsh,
    write_vtu as write_left_wall_vtu,
)
from regions.left_wedged_wall import (
    audit_left_wedged_wall,
    build_left_wedged_wall,
    write_gmsh as write_left_wedged_wall_gmsh,
    write_vtu as write_left_wedged_wall_vtu,
)
from regions.left_wedged_wall_transition import (
    audit_left_wedged_wall_transition,
    build_left_wedged_wall_transition,
    write_gmsh as write_left_wedged_wall_transition_gmsh,
    write_vtu as write_left_wedged_wall_transition_vtu,
)
from regions.plinth import audit_plinth, build_plinth, write_gmsh, write_vtu
from regions.right_wall import (
    audit_right_wall,
    build_right_wall,
    write_gmsh as write_right_wall_gmsh,
    write_vtu as write_right_wall_vtu,
)
from regions.right_wedged_wall import (
    audit_right_wedged_wall,
    build_right_wedged_wall,
    write_gmsh as write_right_wedged_wall_gmsh,
    write_vtu as write_right_wedged_wall_vtu,
)
from regions.right_wedged_wall_transition import (
    audit_right_wedged_wall_transition,
    build_right_wedged_wall_transition,
    write_gmsh as write_right_wedged_wall_transition_gmsh,
    write_vtu as write_right_wedged_wall_transition_vtu,
)


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "structure_output"
COMBINED_SOLVER_DIR = OUTPUT_DIR / "combined_solver"


def convert_with_elmergrid(mesh_path: Path, output_dir: Path) -> bool:
    executable = shutil.which("ElmerGrid")
    if executable is None:
        return False
    if output_dir.exists():
        shutil.rmtree(output_dir)
    subprocess.run(
        [executable, "14", "2", str(mesh_path), "-autoclean", "-out", str(output_dir)],
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


def build_center_wall_region() -> dict[str, object]:
    mesh = build_center_wall(ROOT)
    mesh_path = OUTPUT_DIR / "center_wall.msh"
    preview_path = OUTPUT_DIR / "center_wall.vtu"
    write_center_wall_gmsh(mesh, mesh_path)
    write_center_wall_vtu(mesh, preview_path)
    report = audit_center_wall(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, OUTPUT_DIR / "center_wall_mesh")
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


def write_combined_solver_input(mesh, path: Path) -> None:
    gravity_bodyforce = mesh.material.density_kg_m3 * mesh.loads.gravity_z_m_s2
    pressure_gradient = mesh.loads.peak_water_pressure_pa / mesh.loads.maximum_water_height_m
    support = json.loads((Path(__file__).resolve().parent / "load_cases.json").read_text())["loads"]["bedrock_support"]
    horizontal_stiffness = float(support["horizontal_stiffness_n_per_m3"])
    vertical_stiffness = float(support["vertical_stiffness_n_per_m3"])
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
    Name = "CombinedDamBody"
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
    Output File Name = "combined_results"
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
    ! Undisturbed competent granite elastic half-space; stiffness in N/m^3.
    Spring 1 = Real {horizontal_stiffness:.12g}
    Spring 2 = Real {horizontal_stiffness:.12g}
    Spring 3 = Real {vertical_stiffness:.12g}
End

Boundary Condition 2
    Name = "UpstreamHydrostaticPressure"
    Target Boundaries(1) = 2
    Normal Force = Variable Coordinate 3
        Real MATC "-{pressure_gradient:.12g} * (0.0 - tx) * (tx < 0.0)"
End
''')


def solve_combined_structure() -> dict[str, object]:
        mesh = build_combined_structure(ROOT)
        mesh_path = COMBINED_SOLVER_DIR / "combined_structure.msh"
        write_combined_gmsh(mesh, mesh_path)
        if not convert_with_elmergrid(mesh_path, COMBINED_SOLVER_DIR / "mesh"):
                raise RuntimeError("ElmerGrid was not found on PATH")
        sif_path = COMBINED_SOLVER_DIR / "dam_model.sif"
        write_combined_solver_input(mesh, sif_path)
        (COMBINED_SOLVER_DIR / "results").mkdir(exist_ok=True)
        executable = shutil.which("ElmerSolver")
        if executable is None:
                raise RuntimeError("ElmerSolver was not found on PATH")
        environment = os.environ.copy()
        environment["ELMER_HOME"] = str(Path(executable).resolve().parent.parent)
        subprocess.run(
            [executable, sif_path.name],
            check=True,
            cwd=COMBINED_SOLVER_DIR,
            env=environment,
        )
        result_path = COMBINED_SOLVER_DIR / "results" / "combined_results_t0001.vtu"
        if not result_path.is_file():
                raise RuntimeError(f"Elmer did not create {result_path}")
        report = audit_combined_structure(mesh)
        report["solver_input"] = str(sif_path)
        report["stress_result"] = str(result_path)
        return report


def build_left_wall_region() -> dict[str, object]:
    mesh = build_left_wall(ROOT)
    mesh_path = OUTPUT_DIR / "left_wall.msh"
    preview_path = OUTPUT_DIR / "left_wall.vtu"
    write_left_wall_gmsh(mesh, mesh_path)
    write_left_wall_vtu(mesh, preview_path)
    report = audit_left_wall(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, OUTPUT_DIR / "left_wall_mesh")
    return report


def build_left_wedged_wall_region() -> dict[str, object]:
    mesh = build_left_wedged_wall(ROOT)
    mesh_path = OUTPUT_DIR / "left_wedged_wall.msh"
    preview_path = OUTPUT_DIR / "left_wedged_wall.vtu"
    write_left_wedged_wall_gmsh(mesh, mesh_path)
    write_left_wedged_wall_vtu(mesh, preview_path)
    report = audit_left_wedged_wall(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, OUTPUT_DIR / "left_wedged_wall_mesh")
    return report


def build_left_wedged_wall_transition_region() -> dict[str, object]:
    mesh = build_left_wedged_wall_transition(ROOT)
    mesh_path = OUTPUT_DIR / "left_wedged_wall_transition.msh"
    preview_path = OUTPUT_DIR / "left_wedged_wall_transition.vtu"
    write_left_wedged_wall_transition_gmsh(mesh, mesh_path)
    write_left_wedged_wall_transition_vtu(mesh, preview_path)
    report = audit_left_wedged_wall_transition(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(
        mesh_path, OUTPUT_DIR / "left_wedged_wall_transition_mesh"
    )
    return report


def build_right_wedged_wall_region() -> dict[str, object]:
    mesh = build_right_wedged_wall(ROOT)
    mesh_path = OUTPUT_DIR / "right_wedged_wall.msh"
    preview_path = OUTPUT_DIR / "right_wedged_wall.vtu"
    write_right_wedged_wall_gmsh(mesh, mesh_path)
    write_right_wedged_wall_vtu(mesh, preview_path)
    report = audit_right_wedged_wall(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, OUTPUT_DIR / "right_wedged_wall_mesh")
    return report


def build_right_wedged_wall_transition_region() -> dict[str, object]:
    mesh = build_right_wedged_wall_transition(ROOT)
    mesh_path = OUTPUT_DIR / "right_wedged_wall_transition.msh"
    preview_path = OUTPUT_DIR / "right_wedged_wall_transition.vtu"
    write_right_wedged_wall_transition_gmsh(mesh, mesh_path)
    write_right_wedged_wall_transition_vtu(mesh, preview_path)
    report = audit_right_wedged_wall_transition(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(
        mesh_path, OUTPUT_DIR / "right_wedged_wall_transition_mesh"
    )
    return report


def build_right_wall_region() -> dict[str, object]:
    mesh = build_right_wall(ROOT)
    mesh_path = OUTPUT_DIR / "right_wall.msh"
    preview_path = OUTPUT_DIR / "right_wall.vtu"
    write_right_wall_gmsh(mesh, mesh_path)
    write_right_wall_vtu(mesh, preview_path)
    report = audit_right_wall(mesh)
    report["gmsh_path"] = str(mesh_path)
    report["paraview_path"] = str(preview_path)
    report["elmer_conversion"] = convert_with_elmergrid(mesh_path, OUTPUT_DIR / "right_wall_mesh")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "region",
        nargs="?",
        default="plinth",
        choices=(
            "plinth", "center_wall", "combined", "combined-solve", "left_wall", "left_wedged_wall", "left_wedged_wall_transition",
            "right_wedged_wall", "right_wedged_wall_transition", "right_wall",
        ),
    )
    args = parser.parse_args()
    if args.region == "plinth":
        print(json.dumps(build_plinth_region(), indent=2))
    elif args.region == "center_wall":
        print(json.dumps(build_center_wall_region(), indent=2))
    elif args.region == "combined":
        print(json.dumps(build_combined_region(), indent=2))
    elif args.region == "combined-solve":
        print(json.dumps(solve_combined_structure(), indent=2))
    elif args.region == "left_wall":
        print(json.dumps(build_left_wall_region(), indent=2))
    elif args.region == "left_wedged_wall":
        print(json.dumps(build_left_wedged_wall_region(), indent=2))
    elif args.region == "left_wedged_wall_transition":
        print(json.dumps(build_left_wedged_wall_transition_region(), indent=2))
    elif args.region == "right_wedged_wall":
        print(json.dumps(build_right_wedged_wall_region(), indent=2))
    elif args.region == "right_wedged_wall_transition":
        print(json.dumps(build_right_wedged_wall_transition_region(), indent=2))
    elif args.region == "right_wall":
        print(json.dumps(build_right_wall_region(), indent=2))


if __name__ == "__main__":
    main()
