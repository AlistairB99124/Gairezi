"""Run only the already-imported fracture mesh in a local MPI worker group."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def solve_parallel(
    directory, specifications, options, processes, *,
    mechanics_formulation="hybrid", damage_acceleration=False, final_load_substeps=1,
):
    """Leave configuration/mesh preparation and final report generation serial."""
    request = {
        "specifications": [asdict(specification) for specification in specifications],
        "options": asdict(options),
        "mechanics_formulation": mechanics_formulation,
        "damage_acceleration": damage_acceleration,
        "final_load_substeps": final_load_substeps,
    }
    (directory / "parallel_solve.json").write_text(json.dumps(request, indent=2) + "\n")
    command = ["mpiexec"]
    if os.geteuid() == 0:
        command.append("--allow-run-as-root")
    command += ["-n", str(processes), sys.executable, "-u", str(Path(__file__).resolve()), str(directory)]
    print(f"Starting FEniCSx fracture mechanics on {processes} MPI processes", flush=True)
    environment = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = "1"
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode != 0:
        failures = sorted(directory.glob("parallel_failure_rank*.json"))
        detail = ""
        if failures:
            detail = ": " + json.loads(failures[0].read_text())["error"]
        raise RuntimeError(f"Parallel fracture solve exited with code {completed.returncode}{detail}")
    return json.loads((directory / "phase_field_report.json").read_text())


def main():
    from dolfinx import io
    from mpi4py import MPI
    from types import SimpleNamespace

    from crack_field import CrackSpecification, PhaseFieldSolverOptions
    from dam_phase_field import solve
    from run import assert_fracture_ready, submitted_structure

    comm = MPI.COMM_WORLD
    directory = Path(sys.argv[1]).resolve()
    try:
        request = json.loads((directory / "parallel_solve.json").read_text())
        specifications = [CrackSpecification(**value) for value in request["specifications"]]
        options = PhaseFieldSolverOptions(**request["options"])
        structure = submitted_structure(directory / "config.json")
        from three_body_model import is_three_body
        if is_three_body(structure):
            assert_fracture_ready(structure)
        with io.XDMFFile(comm, directory / "mesh.xdmf", "r") as reader:
            domain = reader.read_mesh(name="mesh")
            domain.topology.create_connectivity(3, 2)
            domain.topology.create_connectivity(2, 3)
            cell_tags = reader.read_meshtags(domain, name="regions")
            facet_tags = reader.read_meshtags(domain, name="boundaries")
        has_plinth = comm.allreduce(len(cell_tags.find(1)), op=MPI.SUM) > 0
        has_wall = comm.allreduce(len(cell_tags.find(2)), op=MPI.SUM) > 0
        has_legacy_wall_support = comm.allreduce(len(facet_tags.find(8)), op=MPI.SUM) > 0
        if has_plinth and has_wall and not has_legacy_wall_support:
            assert_fracture_ready(structure)
        if comm.allreduce(domain.topology.index_map(3).size_local == 0, op=MPI.LOR):
            raise ValueError("Each MPI worker requires owned mesh cells; reduce --fracture-processes")
        solve(
            SimpleNamespace(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags),
            structure, specifications, options, directory,
            mechanics_formulation=request["mechanics_formulation"],
            damage_acceleration=request["damage_acceleration"],
            final_load_substeps=request["final_load_substeps"],
            refined_baseline=True,
        )
    except Exception as error:
        failure = {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        (directory / f"parallel_failure_rank{comm.rank}.json").write_text(
            json.dumps(failure, indent=2) + "\n",
        )
        print(f"FEniCSx MPI rank {comm.rank} failed: {failure['error']}", file=sys.stderr, flush=True)
        comm.Abort(1)


if __name__ == "__main__":
    main()
