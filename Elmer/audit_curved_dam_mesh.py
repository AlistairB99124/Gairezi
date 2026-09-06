from collections import Counter
from pathlib import Path
import sys


mesh_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("curved_dam_mesh.msh")
lines = mesh_path.read_text().splitlines()


def section_rows(name):
    start = lines.index(f"${name}") + 1
    count = int(lines[start])
    return [line.split() for line in lines[start + 1:start + 1 + count]]


nodes = {int(row[0]): tuple(map(float, row[1:])) for row in section_rows("Nodes")}
elements = [
    (int(row[1]), tuple(map(int, row[3 + int(row[2]):])))
    for row in section_rows("Elements")
]

face_templates = {
    4: ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
    5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
    6: ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
        7: ((0, 1, 2, 3), (0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0)),
}


def tetra_volume(first, second, third, fourth):
    ax, ay, az = nodes[first]
    bx, by, bz = nodes[second]
    cx, cy, cz = nodes[third]
    dx, dy, dz = nodes[fourth]
    return abs(
        (bx - ax) * ((cy - ay) * (dz - az) - (cz - az) * (dy - ay))
        - (by - ay) * ((cx - ax) * (dz - az) - (cz - az) * (dx - ax))
        + (bz - az) * ((cx - ax) * (dy - ay) - (cy - ay) * (dx - ax))
    ) / 6.0


def volume(element_type, node_ids):
    if element_type == 4:
        return tetra_volume(*node_ids)
    if element_type == 6:
        return sum(tetra_volume(*(node_ids[index] for index in tetra)) for tetra in ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)))
        if element_type == 7:
            return sum(tetra_volume(*(node_ids[index] for index in tetra)) for tetra in ((0, 1, 2, 4), (0, 2, 3, 4)))
    if element_type == 5:
        return sum(tetra_volume(*(node_ids[index] for index in tetra)) for tetra in (
            (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
            (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
        ))
    raise ValueError(f"Unsupported volume element type {element_type}")


volume_faces = Counter()
zero_volume_elements = []
wedge_base_faces = []
for element_type, node_ids in elements:
    if element_type not in face_templates:
        continue
    if volume(element_type, node_ids) <= 1.0e-12:
        zero_volume_elements.append(node_ids)
    for template in face_templates[element_type]:
        volume_faces[tuple(sorted(node_ids[index] for index in template))] += 1
    if element_type == 6:
        base_face = (node_ids[1], node_ids[2], node_ids[5], node_ids[4])
        coordinates = [nodes[node_id] for node_id in base_face]
        radii = [(x_value * x_value + y_value * y_value) ** 0.5 for x_value, y_value, _ in coordinates]
        if (
            all(74.0 - 1.0e-8 <= radius <= 76.0 + 1.0e-8 for radius in radii)
            and abs(coordinates[0][2] - coordinates[1][2]) <= 1.0e-8
            and abs(coordinates[2][2] - coordinates[3][2]) <= 1.0e-8
        ):
            wedge_base_faces.append(tuple(sorted(base_face)))

boundary_faces = {
    tuple(sorted(node_ids))
    for element_type, node_ids in elements
    if element_type in (2, 3)
}
internal_boundary_faces = boundary_faces & {face for face, count in volume_faces.items() if count == 2}
collapsed_wall_edges = []
for element_type, node_ids in elements:
    if element_type != 5:
        continue
    for first_index, second_index in ((0, 4), (1, 5), (2, 6), (3, 7)):
        first = nodes[node_ids[first_index]]
        second = nodes[node_ids[second_index]]
        if sum((first[axis] - second[axis]) ** 2 for axis in range(3)) <= 1.0e-16:
            collapsed_wall_edges.append((node_ids[first_index], node_ids[second_index]))
unmatched_wedge_bases = [
    face for face in wedge_base_faces
    if volume_faces[face] == 1 and face not in boundary_faces
]
invalid_wedge_bases = [face for face in wedge_base_faces if volume_faces[face] not in (1, 2)]

if zero_volume_elements or internal_boundary_faces or collapsed_wall_edges or unmatched_wedge_bases or invalid_wedge_bases or not wedge_base_faces:
    raise SystemExit(
        "Mesh audit failed: "
        f"zero-volume={len(zero_volume_elements)}, "
        f"internal-2D-boundaries={len(internal_boundary_faces)}, "
        f"collapsed-wall-edges={len(collapsed_wall_edges)}, "
        f"unmatched-wedge-bases={len(unmatched_wedge_bases)}, "
        f"invalid-wedge-base-incidence={len(invalid_wedge_bases)}, "
        f"wedge-base-faces={len(wedge_base_faces)}"
    )

shared_wedge_bases = sum(volume_faces[face] == 2 for face in wedge_base_faces)
exposed_wedge_bases = len(wedge_base_faces) - shared_wedge_bases
print(
    "Mesh audit passed: "
    f"wedge base faces={len(wedge_base_faces)} "
    f"(shared plinth={shared_wedge_bases}, exposed bedrock={exposed_wedge_bases}); "
    f"zero-volume elements=0; internal 2D boundary faces=0."
)