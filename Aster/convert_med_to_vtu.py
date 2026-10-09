#!/usr/bin/env python3
"""Convert Code_Aster MED files to VTU files using meshio."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import meshio


def add_stress_fields(mesh: meshio.Mesh) -> None:
    """Expose Code_Aster's six-component nodal stress field for ParaView."""
    stress = mesh.point_data.get("result__SIGM_NOEU")
    if stress is None or stress.ndim != 2 or stress.shape[1] != 6:
        return

    component_names = ("stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_xz", "stress_yz")
    for index, name in enumerate(component_names):
        mesh.point_data[name] = stress[:, index]

    tensors = np.empty((len(stress), 3, 3), dtype=float)
    tensors[:, 0, 0] = stress[:, 0]
    tensors[:, 1, 1] = stress[:, 1]
    tensors[:, 2, 2] = stress[:, 2]
    tensors[:, 0, 1] = tensors[:, 1, 0] = stress[:, 3]
    tensors[:, 0, 2] = tensors[:, 2, 0] = stress[:, 4]
    tensors[:, 1, 2] = tensors[:, 2, 1] = stress[:, 5]
    principal = np.linalg.eigvalsh(tensors)
    mesh.point_data["principal_stress_min"] = principal[:, 0]
    mesh.point_data["principal_stress_mid"] = principal[:, 1]
    mesh.point_data["principal_stress_max"] = principal[:, 2]


def convert(source: Path, overwrite: bool) -> bool:
    """Convert one MED file, preserving compatible result fields for ParaView."""
    target = source.with_suffix(".vtu")
    if target.exists() and not overwrite:
        print(f"Skipping existing {target}")
        return True

    mesh = meshio.read(source)
    field_data = {}
    for name, value in mesh.field_data.items():
        try:
            field_data[name] = np.asarray(value)
        except ValueError:
            print(f"Omitting non-rectangular field metadata {name!r} from {source}")
    mesh.field_data = field_data
    add_stress_fields(mesh)
    meshio.write(target, mesh)
    print(f"Wrote {target}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="MED file or directory containing MED files (default: Aster/results)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing VTU files",
    )
    args = parser.parse_args()

    sources = [args.source] if args.source.is_file() else sorted(args.source.rglob("*.med"))
    if not sources:
        parser.error(f"no MED files found in {args.source}")

    failures = []
    for source in sources:
        try:
            convert(source, args.overwrite)
        except Exception as error:
            failures.append((source, error))
            print(f"Failed {source}: {error}")

    print(f"Converted {len(sources) - len(failures)}/{len(sources)} MED files")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())