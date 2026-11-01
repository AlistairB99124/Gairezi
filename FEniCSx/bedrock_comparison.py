"""Paired native static baselines with full rock and a fixed foundation seat."""
from __future__ import annotations

import argparse
from copy import copy
import json
from pathlib import Path

import numpy as np

from .run import submitted_structure
from .tensile_loss_event import EventSolverOptions, solve_source_event
from bedrock_comparison import rigid_seat, without_bedrock, write_fields


def run(config: Path, model: Path, arrays_path: Path, output: Path) -> dict:
    structure = submitted_structure(config, model)
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if not np.allclose(structure.nodes, arrays["nodes"], rtol=0, atol=1e-9):
        raise ValueError("Aster and native source-node identities differ")
    retained, mapping, seat = rigid_seat(arrays)
    reduced = without_bedrock(arrays)
    rigid = copy(structure)
    rigid.nodes = reduced["nodes"].tolist()
    rigid.cells = [(4, tuple(row)) for row in reduced["tetra4_concrete"].tolist()]
    rigid.region_ids = reduced["concrete_body_ids"].tolist()
    rigid.boundaries = [
        (tag, tuple(int(mapping[node]) for node in face))
        for tag, face in structure.boundaries
        if tag not in (10, 13, 14, 15, 16) and np.all(mapping[list(face)] >= 0)
    ] + [(10, tuple(int(mapping[node]) for node in face)) for face in seat]
    rigid.rock_support = {"model": "fixed_bottom"}
    if structure.spring_pairs is None:
        raise ValueError("Diagnostic requires explicit saved interface declarations")
    rigid.spring_pairs = [pair for pair in structure.spring_pairs
                          if not pair["name"].startswith(("foundation_bedrock", "wall_bedrock"))]
    material = json.loads(config.read_text(encoding="utf-8"))["material"]["concrete"]
    reports = {}
    for name, candidate, data in (("fenicsx_full", structure, arrays), ("fenicsx_rigid", rigid, reduced)):
        print(f"BEDROCK_COMPARISON {name}: solving elastic baseline", flush=True)
        directory = output / name
        result = solve_source_event(candidate, np.empty(0, dtype=int), maximum_volume_change=1e-6,
                                    options=EventSolverOptions(), output=directory, intact_only=True)
        u = result.intact_displacement
        domain = u.function_space.mesh
        cells = domain.geometry.dofmaps[0]
        owners = np.full(len(domain.geometry.x), len(cells), dtype=np.int32)
        np.minimum.at(owners, cells.ravel(), np.repeat(np.arange(len(cells), dtype=np.int32), 4))
        values = u.eval(domain.geometry.x, owners)
        displacement = np.empty_like(data["nodes"])
        source_ids = domain.geometry.input_global_indices
        if len(source_ids) != len(displacement) or len(np.unique(source_ids)) != len(displacement):
            raise ValueError("Diagnostic native node mapping is incomplete")
        displacement[source_ids] = values
        if not np.allclose(data["nodes"][source_ids], domain.geometry.x, rtol=0, atol=1e-9):
            raise ValueError("Diagnostic native coordinates changed")
        reports[name] = write_fields(directory, data, displacement, material)
        reports[name]["equilibrium"] = result.report["stages"][-1]
    (output / "fenicsx.json").write_text(json.dumps(reports, indent=2, allow_nan=False), encoding="utf-8")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "model", "arrays", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.model, args.arrays, args.output)


if __name__ == "__main__":
    main()
