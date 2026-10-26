"""Work Package 6 of ``Aster_Porting.md``: run the Code_Aster command file.

Detects a real Code_Aster/Salome-Meca installation (``KNOWN_INSTALL_PATHS`` /
``LAMB_CODE_ASTER_LAUNCHER``) and, if found, runs the case for real via its
``run_aster`` launcher and writes ``run_manifest.json`` per the plan's Work
Package 6 checklist (version/hashes/status/paths). If no installation is
found, it writes a clear "unavailable" status instead of crashing or
fabricating a result. Result interpretation stays in
``Aster/postprocess_baseline.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Aster.build_med_mesh import DEFAULT_OUTPUT_DIR

DEFAULT_CASE_PATH = ROOT / "Aster" / "cases" / "00_elastic_baseline.comm"
# Known launcher names across Code_Aster distributions/versions, checked on PATH.
LAUNCHER_CANDIDATES = ("run_aster", "as_run", "code_aster")
# Known fixed install locations (self-contained wrappers that set up their own
# PATH/PYTHONPATH; no environment variables need to be set to use them). Add
# more entries here, or set the LAMB_CODE_ASTER_LAUNCHER environment variable,
# if Code_Aster is installed somewhere else.
KNOWN_INSTALL_PATHS = (
    Path(r"C:\SALOME-MECA\2025\W64\CODE_ASTER_17.4\bin\run_aster.bat"),
)
MESH_UNIT = 20  # must match LIRE_MAILLAGE(UNITE=...) in the .comm file.
RESULT_UNIT = 80  # must match IMPR_RESU(UNITE=...) in the .comm file.
# Converting mesh_arrays.npz -> mesh.med takes well under a second (measured
# ~0.5s on the full ~101k-cell mesh); this timeout is just a generous safety
# margin, not a tuned budget.
MED_CONVERT_TIMEOUT_S = 120


def run_solver(command: list[str], run_dir: Path, timeout: int,
               live_log: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    """Keep a timeout from leaving a solver child running behind its launcher.

    With ``live_log``, output is also appended to that file line by line while
    the solver runs so progress can be followed; the returned output, timeout
    and process-tree kill are unchanged. ``env`` replaces the inherited
    environment (for example a relocated Linux Code_Aster runtime)."""
    if live_log is not None:
        return _run_solver_streaming(command, run_dir, timeout, live_log, env)
    with subprocess.Popen(command, cwd=run_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, env=env) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            _terminate_tree(process)
            stdout, stderr = process.communicate()
            (run_dir / "stdout.log").write_text(stdout, encoding="utf-8")
            (run_dir / "stderr.log").write_text(stderr, encoding="utf-8")
            raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _terminate_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=True,
                       capture_output=True, timeout=30)
    else:
        process.kill()


def _run_solver_streaming(command: list[str], run_dir: Path, timeout: int,
                          live_log: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    # Solver children write to a pipe, which Python and gfortran block-buffer
    # by default; ask them to flush per line so the live log is actually live.
    environment = {**(os.environ if env is None else env), "PYTHONUNBUFFERED": "1",
                   "GFORTRAN_UNBUFFERED_PRECONNECTED": "y"}
    captured: dict[str, list[str]] = {"stdout": [], "stderr": []}
    write_lock = threading.Lock()
    finished = threading.Event()
    tails: list[threading.Thread] = []
    with subprocess.Popen(command, cwd=run_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, env=environment) as process, \
            live_log.open("a", encoding="utf-8") as live:
        def write(line: str) -> None:
            with write_lock:
                live.write(line)
                live.flush()

        def pump(stream, name: str) -> None:
            for line in stream:
                captured[name].append(line)
                if tails:
                    # The solver's own output is being tailed; skip the launcher's end-of-run copy of it.
                    continue
                write(line)
                work = _RUN_ASTER_WORKDIR.search(line) if name == "stdout" else None
                # A Linux run_aster (natively or through wsl.exe) streams stdout live, and its
                # /tmp working directory is not a Windows path, so only Windows runs are tailed.
                if work and os.name == "nt" and Path(work.group(1).strip()).is_absolute():
                    tail = threading.Thread(target=_tail_file, daemon=True,
                                            args=(Path(work.group(1).strip()) / RUN_ASTER_MESSAGE, write, finished))
                    tails.append(tail)
                    tail.start()

        pumps = [threading.Thread(target=pump, args=(process.stdout, "stdout"), daemon=True),
                 threading.Thread(target=pump, args=(process.stderr, "stderr"), daemon=True)]
        for thread in pumps:
            thread.start()
        try:
            process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            _terminate_tree(process)
            process.wait()
            for thread in pumps:
                thread.join(timeout=30)
            finished.set()
            for thread in tails:
                thread.join(timeout=30)
            (run_dir / "stdout.log").write_text("".join(captured["stdout"]), encoding="utf-8")
            (run_dir / "stderr.log").write_text("".join(captured["stderr"]), encoding="utf-8")
            raise
        for thread in pumps:
            thread.join()
        finished.set()
        for thread in tails:
            thread.join(timeout=30)
    return subprocess.CompletedProcess(command, process.returncode, "".join(captured["stdout"]),
                                       "".join(captured["stderr"]))


# On Windows, run_aster sends the solver's output to ``fort.6`` in a temporary
# working directory and prints it only after the solve, so follow that file.
RUN_ASTER_MESSAGE = "fort.6"
_RUN_ASTER_WORKDIR = re.compile(r"Prepare environment in (.+)$")


def _tail_file(path: Path, write, finished: threading.Event, interval_s: float = 0.5) -> None:
    """Copy complete lines appended to ``path`` until ``finished``; a missing file is waited for."""
    offset, partial = 0, b""
    while True:
        done = finished.is_set()
        try:
            with path.open("rb") as stream:
                stream.seek(offset)
                data = stream.read()
        except OSError:
            data = b""
        if data:
            offset += len(data)
            lines = (partial + data).split(b"\n")
            partial = lines.pop()
            for raw in lines:
                write(raw.decode("utf-8", errors="replace").rstrip("\r") + "\n")
        if done:
            if partial:
                write(partial.decode("utf-8", errors="replace").rstrip("\r") + "\n")
            return
        finished.wait(interval_s)


def bundled_python_for(launcher_path: str) -> Path | None:
    """Derive the path to Code_Aster's own bundled Python interpreter from a
    detected ``run_aster``-style launcher path (``<install>/bin/run_aster.bat``
    -> ``<install>/external/Python3.10/python.exe``). Returns None if it
    doesn't exist at the expected location (e.g. a different distribution
    layout). Linux standalone distributions may provide ``<install>/bin/python3``;
    their relocated runtime environment must already be initialized."""
    install_root = Path(launcher_path).resolve().parent.parent
    candidates = [install_root / "external" / "Python3.10" / "python.exe"]
    if os.name != "nt":
        candidates.append(install_root / "bin" / "python3")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def build_med_mesh(
    launcher_path: str,
    mesh_dir: Path,
    arrays_filename: str = "mesh_arrays.npz",
    converter_filename: str = "convert_arrays_to_med.py",
) -> Path | None:
    """Convert <arrays_filename> to a real mesh.med via Code_Aster's bundled
    medcoupling module (<converter_filename>), regenerating it only when
    missing or older than the source array file. Returns the mesh.med path
    on success, or None if the bundled Python / medcoupling path isn't
    available (caller should fall back to the slow Gmsh reader). The
    defaults match the legacy single-WALL case
    (Aster/convert_arrays_to_med.py); Aster/run_three_body_case.py passes
    the three-body array file/converter (Aster/convert_three_body_to_med.py)
    instead, since that mesh has different cell geometry (all-tetrahedral)
    and groups."""
    arrays_path = mesh_dir / arrays_filename
    if not arrays_path.is_file():
        return None
    python_exe = bundled_python_for(launcher_path)
    if python_exe is None:
        return None
    med_path = mesh_dir / "mesh.med"
    if med_path.is_file() and med_path.stat().st_mtime >= arrays_path.stat().st_mtime:
        return med_path
    converter = Path(__file__).resolve().parent / converter_filename
    completed = subprocess.run(
        [str(python_exe), str(converter), str(arrays_path), str(med_path)],
        capture_output=True,
        text=True,
        timeout=MED_CONVERT_TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0 or not med_path.is_file():
        raise RuntimeError(
            f"mesh.med conversion failed (code {completed.returncode}):\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )
    return med_path


def extract_result_fields(launcher_path: str, result_med_path: Path, run_dir: Path) -> Path:
    """Extract DEPL/SIGM_NOEU from ``result.med`` into a plain ``.npz`` via
    Code_Aster's bundled medcoupling (Aster/extract_result_fields.py), so
    Aster/postprocess_baseline.py (which runs under this repo's .venv, not
    the bundled interpreter) can read the solve's result fields. Also
    exports ``result_stress.vtu`` for stock ParaView."""
    python_exe = bundled_python_for(launcher_path)
    if python_exe is None:
        raise RuntimeError("Code_Aster's bundled Python interpreter was not found; cannot extract result fields")
    out_path = run_dir / "aster_result_fields.npz"
    extractor = Path(__file__).resolve().parent / "extract_result_fields.py"
    completed = subprocess.run(
        [str(python_exe), str(extractor), str(result_med_path), str(out_path)],
        capture_output=True,
        text=True,
        timeout=MED_CONVERT_TIMEOUT_S,
        check=False,
    )
    vtu_path = run_dir / "result_stress.vtu"
    if completed.returncode != 0 or not out_path.is_file() or not vtu_path.is_file():
        raise RuntimeError(
            f"result field extraction failed (code {completed.returncode}):\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )
    return out_path


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def find_launcher() -> str | None:
    import os

    env_override = os.environ.get("LAMB_CODE_ASTER_LAUNCHER")
    if env_override and Path(env_override).is_file():
        return env_override
    for name in LAUNCHER_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    for candidate in KNOWN_INSTALL_PATHS:
        if candidate.is_file():
            return str(candidate)
    return None


def detect_installation() -> dict:
    launcher = find_launcher()
    return {
        "launcher_path": launcher,
        "launcher_name": Path(launcher).name if launcher else None,
        "available": launcher is not None,
    }


def run_case(
    case_path: Path = DEFAULT_CASE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    mesh_dir: Path = DEFAULT_OUTPUT_DIR,
    arrays_filename: str = "mesh_arrays.npz",
    converter_filename: str = "convert_arrays_to_med.py",
    export_name: str = "00_elastic_baseline",
    time_limit_s: int = 480,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    installation = detect_installation()
    manifest = {
        "schema_version": 1,
        "case_path": str(case_path.resolve()),
        "case_sha256": file_hash(case_path) if case_path.is_file() else None,
        "mesh_manifest_sha256": (
            file_hash(mesh_dir / "mesh_manifest.json") if (mesh_dir / "mesh_manifest.json").is_file() else None
        ),
        "installation": installation,
    }

    if not case_path.is_file():
        manifest["status"] = "case_file_missing"
        manifest["message"] = f"{case_path} does not exist; run Aster.generate_elastic_case first."
        _write(manifest, output_dir)
        return manifest

    if not installation["available"]:
        manifest["status"] = "unavailable"
        manifest["message"] = (
            "No Code_Aster launcher (run_aster/as_run/code_aster) was found on PATH in this "
            "environment. The case file has been generated but NOT executed. Run this script "
            "again from an environment with Code_Aster installed to produce a real result."
        )
        _write(manifest, output_dir)
        return manifest

    # Real execution path. The .comm file (Aster/generate_elastic_case.py)
    # hardcodes LIRE_MAILLAGE(FORMAT='MED', ...), since FORMAT='GMSH' was
    # found to not finish within 9+ minutes on this mesh (see
    # Aster_Porting.md amendment) while FORMAT='MED' takes well under a
    # second once mesh.med exists. So mesh.med must be built here, not
    # merely preferred, or the solve will fail with a format mismatch.
    try:
        med_path = build_med_mesh(installation["launcher_path"], mesh_dir, arrays_filename, converter_filename)
    except RuntimeError as error:
        manifest["status"] = "mesh_conversion_failed"
        manifest["message"] = str(error)
        _write(manifest, output_dir)
        return manifest
    if med_path is None:
        manifest["status"] = "mesh_missing"
        manifest["message"] = (
            f"{mesh_dir / arrays_filename} is missing, or Code_Aster's bundled Python "
            "(external/Python3.10/python.exe relative to the detected launcher) was not "
            "found; run Aster.build_med_mesh first and confirm the install layout."
        )
        _write(manifest, output_dir)
        return manifest

    run_dir = output_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_med_path = run_dir / "result.med"
    export_path = run_dir / f"{export_name}.export"
    export_path.write_text(
        "\n".join([
            f"P time_limit {time_limit_s}",
            "P memory_limit 4096",
            "P mpi_nbcpu 1",
            f"F comm {case_path.resolve()} D 1",
            f"F libr {med_path.resolve()} D {MESH_UNIT}",
            f"F rmed {result_med_path.resolve()} R {RESULT_UNIT}",
            "",
        ]),
        encoding="utf-8",
    )
    manifest["export_path"] = str(export_path.resolve())

    for old_output in (result_med_path, run_dir / "aster_result_fields.npz", run_dir / "result_stress.vtu",
                       run_dir / "_diagnostic_displacement.npz"):
        old_output.unlink(missing_ok=True)
    started = time.monotonic()
    try:
        completed = run_solver(
            [installation["launcher_path"], str(export_path.resolve())],
            run_dir, time_limit_s,
        )
        # Code_Aster's run_aster wrapper can report returncode 0 / "DIAGNOSTIC
        # JOB : OK" even when an internal command raised an uncaught Python
        # exception and aborted early -- the manifest's stdout_tail/stderr_tail
        # are truncated to the last 4000 chars, which can hide an exception
        # that occurred earlier in a long log. Persist the FULL stdout/stderr
        # next to the manifest, and scan it for Code_Aster's own exception
        # marker so a false "completed" status can't slip through silently.
        stdout_log_path = run_dir / "stdout.log"
        stderr_log_path = run_dir / "stderr.log"
        stdout_log_path.write_text(completed.stdout, encoding="utf-8")
        stderr_log_path.write_text(completed.stderr, encoding="utf-8")
        has_exception = "<EXCEPTION>" in completed.stdout or "<EXCEPTION>" in completed.stderr
        manifest["status"] = "completed" if completed.returncode == 0 and not has_exception else "failed"
        manifest["returncode"] = completed.returncode
        manifest["stdout_tail"] = completed.stdout[-4000:]
        manifest["stderr_tail"] = completed.stderr[-4000:]
        manifest["stdout_log_path"] = str(stdout_log_path.resolve())
        manifest["stderr_log_path"] = str(stderr_log_path.resolve())
        manifest["result_med_path"] = str(result_med_path) if result_med_path.is_file() else None
        manifest["result_fields_path"] = None
        manifest["result_vtu_path"] = None
        if manifest["status"] == "completed" and not manifest["result_med_path"]:
            manifest["status"] = "failed"
            manifest["message"] = "Code_Aster exited without producing result.med."
        if manifest["status"] == "completed" and manifest["result_med_path"]:
            manifest["result_fields_path"] = str(
                extract_result_fields(installation["launcher_path"], result_med_path, run_dir),
            )
            manifest["result_vtu_path"] = str((run_dir / "result_stress.vtu").resolve())
    except RuntimeError as error:
        manifest["status"] = "postprocessing_failed"
        manifest["message"] = str(error)
    except subprocess.TimeoutExpired as error:
        manifest["status"] = "timed_out"
        manifest["message"] = str(error)
    manifest["elapsed_s"] = time.monotonic() - started
    manifest["run_dir"] = str(run_dir.resolve())
    _write(manifest, output_dir)
    return manifest


def _write(manifest: dict, output_dir: Path) -> None:
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    manifest = run_case()
    print(json.dumps(manifest, indent=2))
    if manifest["status"] not in ("completed",):
        # Non-zero exit for "unavailable"/"failed"/etc. signals to callers (CI, scripts)
        # that no verified result exists yet, without raising a Python traceback.
        sys.exit(1 if manifest["status"] != "unavailable" else 2)


if __name__ == "__main__":
    main()
