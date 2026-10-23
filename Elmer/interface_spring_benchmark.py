"""Hand-checkable Elmer benchmark for isotropic linear interface springs.

Two concrete blocks meet on a plane tilted 30 degrees. Roller supports and a
uniform 1 MPa downward traction give an exactly uniform uniaxial stress, so the
interface traction is t = -p * n_z * e_z and the spring gap must be
u_upper - u_lower = t / k (vertical only). Units: m, Pa, N/m^3.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .structure_assembly import convert_with_elmergrid, mpi_launcher

TILT_DEG = 30.
PRESSURE_PA = 1.0e6
DIVISIONS = 4


def interface_height(x: float) -> float:
    return 0.5 + (x - 0.5) * math.tan(math.radians(TILT_DEG))


def write_mesh(path: Path) -> None:
    """Two hex bodies with separate (duplicated) nodes on the tilted interface."""
    nodes: list[tuple[float, float, float]] = []
    elements: list[tuple[int, int, int, list[int]]] = []  # (gmsh type, physical, geometrical, nodes)
    n = DIVISIONS
    grids = []
    for body in (1, 2):
        grid = np.zeros((n + 1, n + 1, n + 1), dtype=int)
        for i in range(n + 1):
            x = i / n
            low, high = (0., interface_height(x)) if body == 1 else (interface_height(x), 1.)
            for j in range(n + 1):
                for k in range(n + 1):
                    nodes.append((x, j / n, low + (high - low) * k / n))
                    grid[i, j, k] = len(nodes)
        grids.append(grid)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    g = grid
                    elements.append((5, body, body, [
                        g[i, j, k], g[i+1, j, k], g[i+1, j+1, k], g[i, j+1, k],
                        g[i, j, k+1], g[i+1, j, k+1], g[i+1, j+1, k+1], g[i, j+1, k+1],
                    ]))
    # Contiguous boundary ids (ElmerGrid renumbers otherwise).
    lower, upper = grids
    faces = {
        1: [(lower, "k", 0)], 2: [(lower, "i", 0)], 3: [(lower, "j", 0)],
        4: [(lower, "k", n)], 5: [(upper, "k", 0)], 6: [(upper, "k", n)],
        7: [(upper, "i", 0)], 8: [(upper, "j", 0)],
    }
    for tag, entries in faces.items():
        for grid, axis, index in entries:
            for a in range(n):
                for b in range(n):
                    if axis == "k":
                        quad = [grid[a, b, index], grid[a+1, b, index],
                                grid[a+1, b+1, index], grid[a, b+1, index]]
                    elif axis == "i":
                        quad = [grid[index, a, b], grid[index, a+1, b],
                                grid[index, a+1, b+1], grid[index, a, b+1]]
                    else:
                        quad = [grid[a, index, b], grid[a+1, index, b],
                                grid[a+1, index, b+1], grid[a, index, b+1]]
                    elements.append((3, tag, tag, quad))
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(nodes))]
    lines += [f"{i+1} {x:.17g} {y:.17g} {z:.17g}" for i, (x, y, z) in enumerate(nodes)]
    lines += ["$EndNodes", "$Elements", str(len(elements))]
    lines += [
        f"{i+1} {kind} 2 {phys} {geom} " + " ".join(str(int(v)) for v in conn)
        for i, (kind, phys, geom, conn) in enumerate(elements)
    ]
    lines += ["$EndElements", ""]
    path.write_text("\n".join(lines))


def write_sif(path: Path, stiffness: float, concrete: dict[str, float]) -> None:
    path.write_text(f'''! Interface spring benchmark: isotropic linear springs between duplicated nodes.
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

Body 1
    Equation = 1
    Material = 1
End

Body 2
    Equation = 1
    Material = 1
End

Equation 1
    Active Solvers(2) = 1 2
End

Solver 1
    Equation = "Elasticity"
    Procedure = "StressSolve" "StressSolver"
    Variable = "Displacement"
    Variable DOFs = 3
    Displace Mesh = Logical False
    Apply Mortar BCs = Logical True
    Nonlinear System Max Iterations = 1
    Linear System Solver = Iterative
    Linear System Iterative Method = BiCGStabl
    Linear System Preconditioning = ILUT
    Linear System ILUT Tolerance = 1.0e-4
    Linear System Max Iterations = 2000
    Linear System Convergence Tolerance = 1.0e-12
    Linear System Abort Not Converged = Logical True
End

Solver 2
    Equation = "ResultOutput"
    Procedure = "ResultOutputSolve" "ResultOutputSolver"
    Output File Name = "benchmark"
    Vtu Format = Logical True
    Ascii Output = Logical True
End

Material 1
    Youngs Modulus = Real {concrete["youngs_modulus"]:.12g}
    Poisson Ratio = Real {concrete["poisson_ratio"]:.12g}
    Density = Real {concrete["density"]:.12g}
End

Boundary Condition 1
    Target Boundaries(1) = 1
    Displacement 3 = Real 0
End

Boundary Condition 2
    Target Boundaries(2) = 2 7
    Displacement 1 = Real 0
End

Boundary Condition 3
    Target Boundaries(2) = 3 8
    Displacement 2 = Real 0
End

Boundary Condition 4
    Name = "InterfaceSpringLower"
    Target Boundaries(1) = 4
    Mortar BC = Integer 5
    Galerkin Projector = Logical True
    Normal Projector = Logical True
    Mortar BC Coefficient Displacement 1 = Real {stiffness:.12g}
    Mortar BC Coefficient Displacement 2 = Real {stiffness:.12g}
    Mortar BC Coefficient Displacement 3 = Real {stiffness:.12g}
End

Boundary Condition 5
    Name = "InterfaceSpringUpper"
    Target Boundaries(1) = 5
End

Boundary Condition 6
    Target Boundaries(1) = 6
    Force 3 = Real -{PRESSURE_PA:.12g}
End
''')


def run(output: Path, stiffness: float, concrete: dict[str, float], partitions: int | None) -> Path:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_mesh(output / "benchmark.msh")
    write_sif(output / "benchmark.sif", stiffness, concrete)
    if not convert_with_elmergrid(output / "benchmark.msh", output / "mesh", partitions):
        raise RuntimeError("ElmerGrid unavailable")
    (output / "results").mkdir()
    environment = os.environ.copy()
    if partitions:
        executable = shutil.which("ElmerSolver_mpi")
        launcher, option = mpi_launcher()
        command = [launcher, option, str(partitions), executable, "benchmark.sif"]
    else:
        executable = shutil.which("ElmerSolver")
        command = [executable, "benchmark.sif"]
    environment["ELMER_HOME"] = str(Path(executable).resolve().parent.parent)
    with (output / "solver.log").open("w") as log:
        subprocess.run(command, cwd=output, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=120)
    results = sorted((output / "results").glob("benchmark_t0001.*vtu"))
    if not results:
        raise RuntimeError("Benchmark produced no VTU output")
    return results[-1]


def expected_gap(stiffness: float) -> np.ndarray:
    normal_z = math.cos(math.radians(TILT_DEG))
    return np.array([0., 0., -PRESSURE_PA * normal_z / stiffness])


def main() -> None:
    from .model_config import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stiffness", type=float, default=7.0e11)
    parser.add_argument("--partitions", type=int)
    args = parser.parse_args()
    config = load_config(args.root / "config.json")
    result = run(args.output, args.stiffness, config["material"]["concrete"], args.partitions)
    print(result)
    print(json.dumps({"expected_gap_m": expected_gap(args.stiffness).tolist()}))


if __name__ == "__main__":
    main()
