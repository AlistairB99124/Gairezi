"""Compare native DIS_T assembly with exact quadratic surface integration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import subprocess

import numpy as np

from Aster.quadratic_surface import discrete_entries, group_stiffness_blocks, surface_blocks
from Aster.run_case import bundled_python_for, detect_installation, run_solver
from Aster.source_contract import sha256

CASE = Path(__file__).with_name("cases") / "quadratic_spring_benchmark.comm"


def verify(native: dict, nodes: np.ndarray, pairs: np.ndarray, blocks: np.ndarray) -> dict:
    matrix = np.asarray(native["matrix"])
    coordinates = np.asarray(native["coordinates"])
    dofs = native["dofs"]
    if (coordinates.shape != (len(nodes), 3) or not np.array_equal(coordinates, nodes)
            or matrix.shape != (3 * len(nodes), 3 * len(nodes)) or not np.isfinite(matrix).all()):
        raise ValueError("Native quadratic spring matrix or mesh is incomplete/nonfinite")
    expected = np.zeros_like(matrix)
    for (first, second), block in zip(pairs, blocks):
        expected[3 * first:3 * first + 3, 3 * second:3 * second + 3] += block
    components = {"DX": 0, "DY": 1, "DZ": 2}
    indices = [3 * node + components[component] for node, component in dofs]
    if sorted(indices) != list(range(len(matrix))):
        raise ValueError("Native quadratic spring numbering is not a complete displacement permutation")
    expected = expected[np.ix_(indices, indices)]
    scale = np.linalg.norm(expected)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Expected quadratic spring matrix must be finite and nonzero")
    relative = np.linalg.norm(matrix - expected) / scale
    if relative > 1e-10:
        raise ValueError(f"Native quadratic surface matrix mismatch: {relative:.6g}")
    return {"relative_matrix_error": relative, "dofs": len(matrix),
            "comparison": "every assembled entry, including all midside and split-interface blocks"}


def run(output_root: Path, timeout_s: int = 120, mixed_order: bool = False) -> dict:
    installation = detect_installation()
    if not installation["available"]:
        raise RuntimeError("Quadratic spring benchmark requires native Code_Aster")
    triangle = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.],
                         [1., 0., 0.], [1., 1.5, 0.], [0., 1.5, 0.]])
    nodes = np.vstack([triangle, triangle[:3] if mixed_order else triangle])
    master = np.arange(6, len(nodes)).reshape(1, -1)
    # Anisotropic cross-coupled stiffness also tests GLOBAL axes and flattening.
    stiffness = np.array([[4., 1., .3], [1., 3., .2], [.3, .2, 2.]]) * 1e9
    pairs, blocks = surface_blocks(nodes, np.arange(6).reshape(1, 6), stiffness,
                                   master)
    normal = np.array([.6, .8, 0])
    support_pairs, support_blocks = surface_blocks(
        nodes, master, 1e10 * np.outer(normal, normal),
    )
    pairs, blocks = np.concatenate([pairs, support_pairs]), np.concatenate([blocks, support_blocks])
    points, segments, point_entries, segment_entries = discrete_entries(pairs, blocks)
    output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="springs-", dir=output_root.resolve()))
    inputs = directory / "inputs.json"
    inputs.write_text(json.dumps({"repository_root": str(Path(__file__).resolve().parent.parent)}),
                      encoding="utf-8")
    discrete = directory / "quadratic_discrete.npz"
    point_matrices = np.array([entry["matrix"] for entry in point_entries])
    link_matrices = np.array([entry["matrix"] for entry in segment_entries])[:, :3, 3:]
    grouped_points, point_ids, _ = group_stiffness_blocks(point_matrices)
    grouped_links, link_ids, _ = group_stiffness_blocks(link_matrices)
    np.savez_compressed(
        discrete, point_node_ids=points, link_node_ids=segments,
        nodes=nodes,
        point_matrices_n_per_m=point_matrices, link_matrices_n_per_m=link_matrices,
        point_group_matrices_n_per_m=grouped_points, link_group_matrices_n_per_m=grouped_links,
        point_group_ids=point_ids, link_group_ids=link_ids,
    )
    mesh = directory / "mesh.med"
    python = bundled_python_for(installation["launcher_path"])
    if python is None:
        raise RuntimeError("Native matrix benchmark needs the bundled MED interpreter")
    subprocess.run(
        [str(python), str(Path(__file__).with_name("convert_quadratic_to_med.py")),
         "--discrete-benchmark", str(discrete), str(mesh)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    native_path = directory / "native_matrix.json"
    export = directory / "benchmark.export"
    export.write_text(
        f"P time_limit {timeout_s}\nP memory_limit 1024\nP mpi_nbcpu 1\n"
        f"F comm {CASE.resolve()} D 1\nF libr {inputs} D 91\nF libr {discrete} D 92\n"
        f"F libr {mesh} D 20\nF libr {native_path} R 90\n",
        encoding="utf-8",
    )
    report = {"status": "running", "directory": str(directory), "engineering_validated": False,
              "trace_nodes": [6, master.shape[1]],
              "case_sha256": sha256(CASE), "inputs_sha256": sha256(inputs),
              "discrete_sha256": sha256(discrete),
              "native_discrete_sha256": sha256(Path(__file__).with_name("native_discrete.py")),
              "scope": "native_quadratic_surface_operator_assembly_not_a_dam_solve"}
    try:
        completed = run_solver([installation["launcher_path"], str(export)], directory, timeout_s)
        (directory / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (directory / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if (completed.returncode or not native_path.is_file() or
                any(marker in completed.stdout + completed.stderr
                    for marker in ("<EXCEPTION>", "Traceback (most recent call last)"))):
            raise RuntimeError(f"Native quadratic spring assembly failed; see {directory}")
        report["checks"] = verify(json.loads(native_path.read_text(encoding="utf-8")), nodes, pairs, blocks)
        report["status"] = "benchmark_passed"
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        (directory / "benchmark_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "results" / "quadratic_spring_benchmark")
    parser.add_argument("--mixed-order", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, mixed_order=args.mixed_order), indent=2))


if __name__ == "__main__":
    main()
