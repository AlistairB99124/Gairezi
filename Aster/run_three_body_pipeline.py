"""Source-pinned export, solve, concrete comparison and elastic verification."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Aster.build_three_body_mesh_arrays import DEFAULT_OUTPUT_DIR, export, find_latest_source_mesh
from Aster.freeze_three_body_elmer_baseline import freeze
from Aster.generate_three_body_case import generate
from Aster.postprocess_baseline import compare
from Aster.run_case import run_case


def run_pipeline(source_mesh: Path | None = None, output_dir: Path = DEFAULT_OUTPUT_DIR,
                 time_limit_s: int = 1800) -> dict:
    output_dir = output_dir.resolve()
    source_mesh = source_mesh or find_latest_source_mesh()
    if source_mesh is None:
        return {
            "status": "no_elmer_run_found",
            "message": (
                "No Elmer.three_body_solver run was found under its default output root. "
                "Run the 'Elmer: Run combined solver' task first."
            ),
        }
    print(f"[1/6] Exporting mesh from {source_mesh} ...", file=sys.stderr)
    export(source_mesh=source_mesh, output_dir=output_dir)

    print("[2/6] Generating Code_Aster case from saved source inputs ...", file=sys.stderr)
    case_path = output_dir / "case.comm"
    generate(case_path=case_path, inputs_path=output_dir / "case_inputs.json",
             mesh_manifest_path=output_dir / "mesh_manifest.json")

    print("[3/6] Solving (this is the slow step, several minutes) ...", file=sys.stderr)
    run_manifest = run_case(
        case_path=case_path,
        output_dir=output_dir,
        mesh_dir=output_dir,
        arrays_filename="three_body_mesh_arrays.npz",
        converter_filename="convert_three_body_to_med.py",
        export_name="01_three_body_baseline",
        time_limit_s=time_limit_s,
    )
    if run_manifest.get("status") != "completed":
        return {
            "status": "solve_failed",
            "run_manifest_status": run_manifest.get("status"),
            "message": run_manifest.get("message"),
            "stderr_tail": run_manifest.get("stderr_tail"),
        }

    print("[4/6] Freezing the matching Elmer baseline for comparison ...", file=sys.stderr)
    freeze(output_dir=output_dir)

    print("[5/6] Comparing Elmer vs Code_Aster stresses ...", file=sys.stderr)
    report = compare(output_dir=output_dir)
    for name, peak in report["tensile_peak_comparison_by_region"].items():
        print(f"{name.capitalize()} recovered nodal peak tension: "
              f"Elmer {peak['elmer_mpa']:.6f} MPa; Aster {peak['aster_mpa']:.6f} MPa; "
              f"difference {peak['difference_percent']:+.2f}%", file=sys.stderr)
    from Aster.verify_parity import verify
    print("[6/6] Checking equilibrium, energy and element stress agreement ...", file=sys.stderr)
    report["verification"] = verify(output_dir)
    report["status"] = "verified" if report["verification"]["baseline_verified"] else "requires_verification"
    report["result_med_path"] = run_manifest["result_med_path"]
    report["result_vtu_path"] = run_manifest["result_vtu_path"]
    print(f"ParaView stress results: {report['result_vtu_path']}", file=sys.stderr)
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mesh", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--time-limit-s", type=int, default=1800)
    args = parser.parse_args()
    report = run_pipeline(args.source_mesh, args.output_dir, args.time_limit_s)
    print(json.dumps(report, indent=2))
    if report.get("status") != "verified":
        sys.exit(2)


if __name__ == "__main__":
    main()
