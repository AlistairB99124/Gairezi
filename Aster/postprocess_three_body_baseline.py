"""Three-body counterpart of ``Aster/postprocess_baseline.py``.

``Aster.postprocess_baseline.compare`` is already fully generic over
``output_dir`` (it only reads/writes ``elmer_baseline.json``,
``run_manifest.json`` and ``comparison_report.json`` inside whatever
directory it is given) -- no duplication is needed, just pointing it at the
three-body results directory instead of the legacy single-WALL one.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Aster.build_three_body_mesh_arrays import DEFAULT_OUTPUT_DIR
from Aster.postprocess_baseline import compare


def main() -> None:
    report = compare(output_dir=DEFAULT_OUTPUT_DIR)
    print(json.dumps(report, indent=2))
    if report.get("status") != "ok":
        sys.exit(2)


if __name__ == "__main__":
    main()
