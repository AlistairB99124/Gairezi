from __future__ import annotations

import re
import struct
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent / "results"
SOURCE_PATH = RESULTS_DIR / "dam_results_t0001.vtu"
OUTPUT_PATH = RESULTS_DIR / "dam_results_volume_t0001.vtu"
VOLUME_CELL_TYPES = {12, 13}  # VTK_HEXAHEDRON and VTK_WEDGE


def read_chunk(appended: bytes, offset: int) -> bytes:
    length = struct.unpack("<I", appended[offset : offset + 4])[0]
    return appended[offset : offset + 4 + length]


def main() -> None:
    data = SOURCE_PATH.read_bytes()
    appended_tag = b'<AppendedData encoding="raw">'
    header_end = data.index(appended_tag)
    appended_start = data.index(b"_", header_end) + 1
    appended_end = data.index(b"</AppendedData>", appended_start)
    header = data[:header_end].decode("utf-8", errors="replace")
    appended = data[appended_start:appended_end]

    pattern = re.compile(
        r'<DataArray type="(?P<dtype>[^"]+)"(?: Name="(?P<name>[^"]+)")?'
        r'(?: NumberOfComponents="(?P<components>\d+)")? format="appended" '
        r'offset="(?P<offset>\d+)"/>'
    )
    arrays = [match.groupdict() for match in pattern.finditer(header)]
    arrays_by_name = {array["name"]: array for array in arrays if array["name"]}

    connectivity = arrays_by_name["connectivity"]
    offsets = arrays_by_name["offsets"]
    cell_types = arrays_by_name["types"]
    connectivity_chunk = read_chunk(appended, int(connectivity["offset"]))
    offsets_chunk = read_chunk(appended, int(offsets["offset"]))
    types_chunk = read_chunk(appended, int(cell_types["offset"]))

    source_connectivity = list(struct.unpack(f"<{(len(connectivity_chunk) - 4) // 4}i", connectivity_chunk[4:]))
    source_offsets = list(struct.unpack(f"<{(len(offsets_chunk) - 4) // 4}i", offsets_chunk[4:]))
    source_types = list(struct.unpack(f"<{(len(types_chunk) - 4) // 4}i", types_chunk[4:]))

    volume_connectivity: list[int] = []
    volume_offsets: list[int] = []
    volume_types: list[int] = []
    start = 0
    for end, cell_type in zip(source_offsets, source_types):
        if cell_type in VOLUME_CELL_TYPES:
            volume_connectivity.extend(source_connectivity[start:end])
            volume_offsets.append(len(volume_connectivity))
            volume_types.append(cell_type)
        start = end

    point_arrays = [array for array in arrays if array["name"] not in {"connectivity", "offsets", "types"}]
    points_array = next(array for array in point_arrays if array["name"] is None)
    point_data_arrays = [array for array in point_arrays if array["name"] is not None]

    output_chunks: list[bytes] = []
    offset = 0

    def add_chunk(chunk: bytes) -> int:
        nonlocal offset
        chunk_offset = offset
        output_chunks.append(chunk)
        offset += len(chunk)
        return chunk_offset

    point_data_xml = []
    for array in point_data_arrays:
        chunk = read_chunk(appended, int(array["offset"]))
        component_xml = f' NumberOfComponents="{array["components"]}"' if array["components"] else ""
        point_data_xml.append(
            f'        <DataArray type="{array["dtype"]}" Name="{array["name"]}"{component_xml} '
            f'format="appended" offset="{add_chunk(chunk)}"/>'
        )

    points_chunk = read_chunk(appended, int(points_array["offset"]))
    points_offset = add_chunk(points_chunk)
    connectivity_offset = add_chunk(struct.pack(f"<I{len(volume_connectivity)}i", 4 * len(volume_connectivity), *volume_connectivity))
    offsets_offset = add_chunk(struct.pack(f"<I{len(volume_offsets)}i", 4 * len(volume_offsets), *volume_offsets))
    types_offset = add_chunk(struct.pack(f"<I{len(volume_types)}i", 4 * len(volume_types), *volume_types))

    point_count = (len(points_chunk) - 4) // (8 * 3)
    xml = "\n".join([
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
        "  <UnstructuredGrid>",
        f'    <Piece NumberOfPoints="{point_count}" NumberOfCells="{len(volume_types)}">',
        "      <PointData>",
        *point_data_xml,
        "      </PointData>",
        "      <CellData/>",
        "      <Points>",
        f'        <DataArray type="{points_array["dtype"]}" NumberOfComponents="3" format="appended" offset="{points_offset}"/>',
        "      </Points>",
        "      <Cells>",
        f'        <DataArray type="Int32" Name="connectivity" format="appended" offset="{connectivity_offset}"/>',
        f'        <DataArray type="Int32" Name="offsets" format="appended" offset="{offsets_offset}"/>',
        f'        <DataArray type="Int32" Name="types" format="appended" offset="{types_offset}"/>',
        "      </Cells>",
        "    </Piece>",
        "  </UnstructuredGrid>",
        '  <AppendedData encoding="raw">',
    ]).encode("utf-8")
    OUTPUT_PATH.write_bytes(xml + b"_" + b"".join(output_chunks) + b"\n</AppendedData>\n</VTKFile>\n")
    print(f"Wrote {OUTPUT_PATH} with {len(volume_types)} volume cells")


if __name__ == "__main__":
    main()