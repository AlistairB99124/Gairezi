"""Export the partitioned Elmer mesh with BodyId as its only cell field."""
from __future__ import annotations

import argparse
from pathlib import Path


ELMER_TO_VTK_CELL_TYPES = {
    504: 10,  # tetrahedron
    706: 13,  # triangular prism
    808: 12,  # hexahedron
}


def _partition_paths(mesh_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in mesh_dir.glob("partitioning.*")
        if path.is_dir() and path.name.removeprefix("partitioning.").isdigit()
    )


def _read_nodes(path: Path) -> tuple[dict[int, int], list[tuple[float, float, float]]]:
    node_indices: dict[int, int] = {}
    nodes: list[tuple[float, float, float]] = []
    for line in path.read_text().splitlines():
        node_id, _, x_m, y_m, z_m = line.split()
        node_indices[int(node_id)] = len(nodes)
        nodes.append((float(x_m), float(y_m), float(z_m)))
    return node_indices, nodes


def _read_elements(path: Path, body_id_filter: int | None) -> list[tuple[int, int, list[int]]]:
  elements: list[tuple[int, int, list[int]]] = []
    for line in path.read_text().splitlines():
        _, body_id, element_type, *node_ids = (int(value) for value in line.split())
    if body_id_filter is not None and body_id != body_id_filter:
      continue
        try:
            vtk_type = ELMER_TO_VTK_CELL_TYPES[element_type]
        except KeyError as error:
            raise ValueError(f"Unsupported Elmer bulk element type {element_type}") from error
    elements.append((body_id, vtk_type, node_ids))
  return elements


def _compact_elements(
    node_indices: dict[int, int],
    nodes: list[tuple[float, float, float]],
    elements: list[tuple[int, int, list[int]]],
) -> tuple[list[tuple[float, float, float]], list[int], list[int], list[int], list[int]]:
    used_node_ids = {node_id for _, _, node_ids in elements for node_id in node_ids}
    compact_indices = {node_id: index for index, node_id in enumerate(sorted(used_node_ids))}
    compact_nodes = [nodes[node_indices[node_id]] for node_id in sorted(used_node_ids)]
    connectivity = [compact_indices[node_id] for _, _, node_ids in elements for node_id in node_ids]
    offsets = []
    offset = 0
    for _, _, node_ids in elements:
        offset += len(node_ids)
        offsets.append(offset)
    return (
        compact_nodes,
        connectivity,
        offsets,
        [vtk_type for _, vtk_type, _ in elements],
        [body_id for body_id, _, _ in elements],
    )


def _write_piece(path: Path, nodes, connectivity, offsets, vtk_types, body_ids) -> None:
    points = " ".join(f"{coordinate:.12g}" for point in nodes for coordinate in point)
    connectivity_text = " ".join(map(str, connectivity))
    offsets_text = " ".join(map(str, offsets))
    types_text = " ".join(map(str, vtk_types))
    body_text = " ".join(map(str, body_ids))
    path.write_text(f'''<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{len(nodes)}" NumberOfCells="{len(body_ids)}">
      <PointData/>
      <CellData Scalars="BodyId">
        <DataArray type="Int32" Name="BodyId" format="ascii">{body_text}</DataArray>
      </CellData>
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">{points}</DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">{connectivity_text}</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">{offsets_text}</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">{types_text}</DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
''')


def export_body_id_pvtu(mesh_dir: Path, output_path: Path, body_id_filter: int | None = None) -> None:
    partition_dirs = _partition_paths(mesh_dir)
    if len(partition_dirs) != 1:
        raise ValueError(f"Expected one partitioning directory in {mesh_dir}, found {len(partition_dirs)}")
    partition_dir = partition_dirs[0]
    partition_count = int(partition_dir.name.removeprefix("partitioning."))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    piece_names = []
    for partition_index in range(1, partition_count + 1):
        node_indices, nodes = _read_nodes(partition_dir / f"part.{partition_index}.nodes")
        elements = _read_elements(
            partition_dir / f"part.{partition_index}.elements", body_id_filter
        )
        nodes, connectivity, offsets, vtk_types, body_ids = _compact_elements(
            node_indices, nodes, elements
        )
        piece_name = f"{output_path.stem}_{partition_count}np{partition_index}.vtu"
        piece_path = output_path.with_name(piece_name)
        _write_piece(piece_path, nodes, connectivity, offsets, vtk_types, body_ids)
        piece_names.append(piece_path.name)
    pieces = "\n".join(f'    <Piece Source="{piece_name}"/>' for piece_name in piece_names)
    output_path.write_text(f'''<?xml version="1.0"?>
<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <PUnstructuredGrid>
    <PPointData/>
    <PCellData Scalars="BodyId">
      <PDataArray type="Int32" Name="BodyId"/>
    </PCellData>
    <PPoints>
      <PDataArray type="Float64" NumberOfComponents="3"/>
    </PPoints>
{pieces}
  </PUnstructuredGrid>
</VTKFile>
''')


def main() -> None:
    parser = argparse.ArgumentParser(description="Export partitioned Elmer body IDs as a PVTU dataset.")
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--body-id", type=int, help="Export only this Elmer body ID.")
    args = parser.parse_args()
    export_body_id_pvtu(args.mesh_dir, args.output, args.body_id)


if __name__ == "__main__":
    main()