"""Export the current partitioned Elmer plinth as a geometry-only PVTU dataset."""
from __future__ import annotations

import argparse
from pathlib import Path


ELMER_TO_VTK_CELL_TYPES = {504: 10, 706: 13, 808: 12}


def read_partition(path: Path) -> tuple[dict[int, tuple[float, float, float]], list[tuple[int, list[int]]]]:
  nodes_path = path.parent / f"{path.name}.nodes"
  elements_path = path.parent / f"{path.name}.elements"
    nodes = {
        int(node_id): (float(x_m), float(y_m), float(z_m))
        for node_id, _, x_m, y_m, z_m in (
      line.split() for line in nodes_path.read_text().splitlines()
        )
    }
    elements = []
  for line in elements_path.read_text().splitlines():
        _, body_id, element_type, *node_ids = (int(value) for value in line.split())
        if body_id == 1:
            elements.append((ELMER_TO_VTK_CELL_TYPES[element_type], node_ids))
    return nodes, elements


def write_piece(path: Path, nodes_by_id, elements) -> None:
    node_ids = sorted({node_id for _, cell_node_ids in elements for node_id in cell_node_ids})
    node_indices = {node_id: index for index, node_id in enumerate(node_ids)}
    connectivity = [node_indices[node_id] for _, cell_node_ids in elements for node_id in cell_node_ids]
    offsets = []
    offset = 0
    for _, cell_node_ids in elements:
        offset += len(cell_node_ids)
        offsets.append(offset)
    points = " ".join(f"{value:.12g}" for node_id in node_ids for value in nodes_by_id[node_id])
    path.write_text(f'''<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{len(node_ids)}" NumberOfCells="{len(elements)}">
      <PointData/>
      <CellData Scalars="BodyId">
        <DataArray type="Int32" Name="BodyId" format="ascii">{' '.join(['1'] * len(elements))}</DataArray>
      </CellData>
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">{points}</DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">{' '.join(map(str, connectivity))}</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">{' '.join(map(str, offsets))}</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">{' '.join(str(cell_type) for cell_type, _ in elements)}</DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
''')


def export_plinth(mesh_dir: Path, output_path: Path) -> None:
    partition_dirs = [path for path in mesh_dir.glob("partitioning.*") if path.is_dir()]
    if len(partition_dirs) != 1:
        raise ValueError(f"Expected one partitioning directory in {mesh_dir}")
    partition_dir = partition_dirs[0]
    partition_count = int(partition_dir.name.removeprefix("partitioning."))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    piece_names = []
    for partition_index in range(1, partition_count + 1):
        nodes, elements = read_partition(partition_dir / f"part.{partition_index}")
        piece_path = output_path.with_name(f"{output_path.stem}_{partition_count}np{partition_index}.vtu")
        write_piece(piece_path, nodes, elements)
        piece_names.append(piece_path.name)
    output_path.write_text(f'''<?xml version="1.0"?>
<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <PUnstructuredGrid>
    <PPointData/>
    <PCellData Scalars="BodyId"><PDataArray type="Int32" Name="BodyId"/></PCellData>
    <PPoints><PDataArray type="Float64" NumberOfComponents="3"/></PPoints>
{chr(10).join(f'    <Piece Source="{piece_name}"/>' for piece_name in piece_names)}
  </PUnstructuredGrid>
</VTKFile>
''')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_plinth(args.mesh_dir, args.output)


if __name__ == "__main__":
    main()