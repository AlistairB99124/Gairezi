"""Export MED geometry and nodal arrays using MEDCoupling inside the container."""
from __future__ import annotations

import argparse
from pathlib import Path

import medcoupling as mc
import numpy as np


def export(source: Path, output_dir: Path) -> None:
    path = str(source)
    mesh_names = mc.GetMeshNames(path)
    if len(mesh_names) != 1:
        raise ValueError(f"Expected one mesh in {source}, found {mesh_names}")
    mesh_name = mesh_names[0]
    mesh = mc.ReadUMeshFromFile(path, mesh_name, 0)
    mesh.writeVTK(str(output_dir / "geometry"))

    arrays = {}
    nodal_names = mc.GetNodeFieldNamesOnMesh(path, mesh_name)
    for name in mc.GetAllFieldNamesOnMesh(path, mesh_name):
        if name not in nodal_names:
            print(f"Omitting non-nodal field {name!r} from the nodal VTU export")
            continue
        iteration, order = mc.GetAllFieldIterations(path, name)[-1][:2]
        field = mc.ReadFieldNode(path, mesh_name, 0, name, iteration, order)
        values = field.getArray().toNumPyArray()
        if values.shape[0] != mesh.getNumberOfNodes():
            raise ValueError(f"Field {name!r} does not cover all mesh nodes")
        if not np.array_equal(field.getMesh().getCoords().toNumPyArray(), mesh.getCoords().toNumPyArray()):
            raise ValueError(f"Field {name!r} uses a different node ordering")
        arrays[name] = values
    np.savez(output_dir / "fields.npz", **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    export(args.source, args.output_dir)


if __name__ == "__main__":
    main()
