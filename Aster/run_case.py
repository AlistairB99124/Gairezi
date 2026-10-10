"""Run commands and Code_aster cases inside the Salome-Meca container."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTAINER_ROOT = Path("/workspace")
DOCKER_IMAGE = os.environ.get("CODE_ASTER_IMAGE", "jesusbill/code_aster:latest")
ASTER_BIN = "/opt/conda/bin/run_aster"


def docker_executable() -> str:
    """Find Docker Desktop's CLI, including its standard macOS location."""
    candidates = (
        os.environ.get("DOCKER_BIN"),
        shutil.which("docker"),
        str(Path.home() / ".docker" / "bin" / "docker"),
        "/Applications/Docker.app/Contents/Resources/bin/docker",
    )
    docker = next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), None)
    if docker is None:
        raise RuntimeError("Docker CLI not found. Start Docker Desktop or set DOCKER_BIN.")
    return docker


def container_path(path: Path) -> str:
    """Translate a workspace path into the mounted container path."""
    return "/workspace/" + path.resolve().relative_to(ROOT).as_posix()


def run_in_container(shell_command: str, cwd: Path) -> int:
    """Run a shell command inside Code_Aster's Docker image with this repo mounted."""
    script = "export PATH=/opt/conda/bin:$PATH; " + shell_command.replace(str(ROOT), str(CONTAINER_ROOT))
    return subprocess.run(
        [
            docker_executable(),
            "run",
            "--rm",
            "--mount",
            f"type=bind,src={ROOT},dst={CONTAINER_ROOT}",
            "--workdir",
            container_path(cwd),
            "--entrypoint",
            "/bin/sh",
            DOCKER_IMAGE,
            "-lc",
            script,
        ],
        cwd=cwd,
    ).returncode


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_docker_med_mesh(results: Path) -> Path:
    """Rebuild a case mesh using the Docker image's MED library when inputs exist."""
    mesh = results / "mesh.med"
    intermediate = results / "mesh_intermediate.npz"
    groups = results / "group_family_ids.json"
    if not (intermediate.exists() and groups.exists()):
        return mesh

    print(f"Rebuilding {mesh} with the Docker MED library")
    writer = ROOT / "Aster" / "write_med.py"
    returncode = run_in_container(
        f"python3 {container_path(writer)} {container_path(intermediate)} "
        f"{container_path(mesh)} {container_path(groups)}",
        results,
    )
    if returncode != 0:
        raise RuntimeError(f"MED mesh generation failed with exit code {returncode}")
    return mesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comm", type=Path, help="Command file, e.g. Aster/cases/validate_mesh.comm")
    parser.add_argument("--results", type=Path, default=ROOT / "Aster" / "results" / "00_elastic_baseline")
    parser.add_argument("--memory-mb", type=int, default=7000)
    parser.add_argument("--ncpus", type=int, default=1, help="OpenMP threads for the solver")
    parser.add_argument("--spec", type=Path, help="JSON spec attached as unit 32 (default: <comm stem>.json)")
    parser.add_argument("--name", help="Output file stem (default: comm stem)")
    args = parser.parse_args()

    comm = args.comm.resolve()
    results = args.results.resolve()
    mesh = ensure_docker_med_mesh(results)
    if not mesh.exists():
        sys.exit(f"{mesh} not found; run Aster/build_med_mesh.py first")

    stem = args.name or comm.stem
    spec = (args.spec or comm.with_suffix(".json")).resolve()
    export = results / f"{stem}.export"
    mess = results / f"{stem}.mess"
    result_med = results / f"{stem}_result.med"
    table = results / f"{stem}_table.txt"
    export.write_text(
        "P actions make_etude\n"
        "P mode interactif\n"
        f"P memory_limit {args.memory_mb}\n"
        f"P ncpus {args.ncpus}\n"
        f"F comm {container_path(comm)} D 1\n"
        f"F mmed {container_path(mesh)} D 20\n"
        f"F mess {container_path(mess)} R 6\n"
        f"F rmed {container_path(result_med)} R 80\n"
        f"F libr {container_path(table)} R 30\n"
        f"F libr {container_path(results / 'case_inputs.json')} D 31\n"
        + (f"F libr {container_path(spec)} D 32\n" if spec.exists() else "")
    )
    wrapper_returncode = run_in_container(f"{ASTER_BIN} {container_path(export)}", results)
    solver_succeeded = result_med.exists() and "EXECUTION_CODE_ASTER_EXIT_16=0" in mess.read_text(errors="replace")
    if solver_succeeded:
        from convert_med_to_vtu import convert

        convert(result_med, overwrite=True)
    manifest = {
        "case": stem,
        "returncode": 0 if solver_succeeded else wrapper_returncode,
        "wrapper_returncode": wrapper_returncode,
        "solver_succeeded": solver_succeeded,
        "comm_sha256": sha256(comm),
        "mesh_sha256": sha256(mesh),
        "export": str(export),
    }
    (results / f"{stem}_run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest["returncode"]


if __name__ == "__main__":
    sys.exit(main())
