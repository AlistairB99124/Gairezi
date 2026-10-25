"""Generate and solve a case pinned to an already-exported source mesh."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Aster.run_case import run_case
from Aster.generate_three_body_case import generate

OUTPUT_DIR = ROOT / "Aster" / "results" / "01_three_body_baseline"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--time-limit-s", type=int, default=1800)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    case_path = output_dir / "case.comm"
    generate(case_path=case_path, inputs_path=output_dir / "case_inputs.json",
             mesh_manifest_path=output_dir / "mesh_manifest.json")
    manifest = run_case(
        case_path=case_path,
        output_dir=output_dir,
        mesh_dir=output_dir,
        arrays_filename="three_body_mesh_arrays.npz",
        converter_filename="convert_three_body_to_med.py",
        export_name="01_three_body_baseline",
        time_limit_s=args.time_limit_s,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("stdout_tail", "stderr_tail")}, indent=2))
    if manifest["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
