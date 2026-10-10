from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import sys


mesh_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("curved_dam_mesh.msh")
maximum_bulk_edge_m = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
lines = mesh_path.read_text().splitlines()


def section_rows(name: str) -> list[list[str]]:
    start = lines.index(f"${name}") + 1
    count = int(lines[start])
    return [line.split() for line in lines[start + 1:start + 1 + count]]


nodes = {
    int(row[0]): tuple(map(float, row[1:]))
    for row in section_rows("Nodes")
}
elements = [
    (int(row[1]), tuple(map(int, row[3 + int(row[2]):])))
    for row in section_rows("Elements")
]

hex_edges = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

hex_lengths = []
worst_edge = (0.0, None, None)
for element_type, node_ids in elements:
    if element_type != 5:
        continue
    for first, second in hex_edges:
        length = math.dist(nodes[node_ids[first]], nodes[node_ids[second]])
        hex_lengths.append(length)
        if length > worst_edge[0]:
            worst_edge = (length, nodes[node_ids[first]], nodes[node_ids[second]])

if not hex_lengths:
    raise SystemExit("Mesh-size audit failed: no hexahedral bulk elements found")

oversized = [length for length in hex_lengths if length > maximum_bulk_edge_m + 1.0e-8]
element_types = Counter(element_type for element_type, _ in elements)
if oversized:
    raise SystemExit(
        "Mesh-size audit failed: "
        f"hex-edge-max={max(hex_lengths):.6f} m, "
        f"hex-edges-over-{maximum_bulk_edge_m:.3f}m={len(oversized)}, "
        f"worst-edge={worst_edge[1]} -> {worst_edge[2]}, "
        f"element-types={dict(sorted(element_types.items()))}"
    )

print(
    "Mesh-size audit passed: "
    f"hex-edge-range={min(hex_lengths):.6f} to {max(hex_lengths):.6f} m; "
    f"element-types={dict(sorted(element_types.items()))}."
)
