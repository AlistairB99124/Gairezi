from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys


mesh_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("curved_dam_mesh.msh")
lines = mesh_path.read_text().splitlines()
markers = json.loads(mesh_path.with_name("curved_dam_mesh_meta.json").read_text())["topology_markers"]


def section_rows(name):
    start = lines.index(f"${name}") + 1
    count = int(lines[start])
    return [line.split() for line in lines[start + 1:start + 1 + count]]


nodes = {int(row[0]): tuple(map(float, row[1:])) for row in section_rows("Nodes")}
elements = {
    int(row[0]): (int(row[1]), int(row[3]), tuple(map(int, row[3 + int(row[2]):])))
    for row in section_rows("Elements")
}
face_templates = {
    4: ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
    5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
}


def tetra_volume(node_ids):
    (ax, ay, az), (bx, by, bz), (cx, cy, cz), (dx, dy, dz) = (nodes[node_id] for node_id in node_ids)
    return abs(
        (bx - ax) * ((cy - ay) * (dz - az) - (cz - az) * (dy - ay))
        - (by - ay) * ((cx - ax) * (dz - az) - (cz - az) * (dx - ax))
        + (bz - az) * ((cx - ax) * (dy - ay) - (cy - ay) * (dx - ax))
    ) / 6.0


def volume(element_type, node_ids):
    if element_type == 4:
        return tetra_volume(node_ids)
    return sum(tetra_volume(tuple(node_ids[index] for index in tetrahedron)) for tetrahedron in (
        (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
        (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
    ))


volume_faces = Counter()
zero_volume = []
for element_id, (element_type, _, node_ids) in elements.items():
    if element_type not in face_templates:
        continue
    if volume(element_type, node_ids) <= 1.0e-12:
        zero_volume.append(element_id)
    for template in face_templates[element_type]:
        volume_faces[tuple(sorted(node_ids[index] for index in template))] += 1

boundary_physical_ids = defaultdict(set)
for element_type, physical_id, node_ids in elements.values():
    if element_type in (2, 3):
        boundary_physical_ids[tuple(sorted(node_ids))].add(physical_id)

internal_boundaries = [face for face in boundary_physical_ids if volume_faces[face] > 1]
wedge_ids = markers["wedge_element_ids"]
transition_ids = markers["transition_element_ids"]
wedge_bottom_faces = [tuple(sorted(face)) for face in markers["wedge_bottom_faces"]]
wedge_wall_faces = [tuple(sorted(face)) for face in markers["wedge_wall_faces"]]
wedge_outer_faces = [tuple(sorted(face)) for face in markers.get("wedge_outer_faces", [])]
marker_errors = (
    not wedge_ids or not transition_ids or not wedge_bottom_faces or not wedge_wall_faces or not wedge_outer_faces
    or any(elements.get(element_id, (None, None, ()))[0] != 4 for element_id in wedge_ids + transition_ids)
)
shared_bottom_faces = [face for face in wedge_bottom_faces if volume_faces[face] == 2]
bedrock_bottom_faces = [
    face for face in wedge_bottom_faces
    if volume_faces[face] == 1 and 1 in boundary_physical_ids[face]
]
invalid_bottom_faces = [
    face for face in wedge_bottom_faces
    if face not in shared_bottom_faces and face not in bedrock_bottom_faces
]
invalid_wall_faces = [
    face for face in wedge_wall_faces
    if volume_faces[face] != 2
    or any(not math.isclose(math.hypot(nodes[node_id][0], nodes[node_id][1]), 76.0, abs_tol=1.0e-8) for node_id in face)
]
wall_ladder_errors = [
    marker for marker in markers.get("ordinary_wall_ladder", [])
    if not math.isclose(nodes[marker["node_id"]][2], marker["expected_z"], abs_tol=1.0e-8)
]
outer_node_ids = {node_id for face in wedge_outer_faces for node_id in face}
outer_nodes = [nodes[node_id] for node_id in outer_node_ids]
z_values = [node[2] for node in outer_nodes]
radii = [math.hypot(node[0], node[1]) for node in outer_nodes]
invalid_outer_nodes = [
    node for node in outer_nodes
    if not math.isclose(math.hypot(node[0], node[1]), 76.0 - (-25.0 - node[2]) / 2.0, abs_tol=1.0e-7)
]
dimension_errors = bool(
    not outer_nodes or not math.isclose(max(z_values), -25.0, abs_tol=1.0e-8)
    or min(radii) < 74.0 - 1.0e-8 or not outer_nodes or invalid_outer_nodes
)

if zero_volume or internal_boundaries or marker_errors or invalid_bottom_faces or invalid_wall_faces or wall_ladder_errors or dimension_errors:
    raise SystemExit(
        "Mesh audit failed: "
        f"zero-volume={len(zero_volume)}, internal-explicit-boundaries={len(internal_boundaries)}, "
        f"marker-errors={int(marker_errors)}, wedge-bottom-errors={len(invalid_bottom_faces)}, "
        f"wall-face-errors={len(invalid_wall_faces)}, wall-ladder-errors={len(wall_ladder_errors)}, "
        f"dimension-errors={int(dimension_errors)}"
    )

print(
    "Mesh audit passed: "
    f"wedge-tets={len(wedge_ids)}, transition-tets={len(transition_ids)}, "
    f"wedge-bottom-faces={len(wedge_bottom_faces)} (shared-plinth={len(shared_bottom_faces)}, bedrock={len(bedrock_bottom_faces)}), "
    f"wall-shared-faces={len(wedge_wall_faces)}, zero-volume=0, internal-explicit-boundaries=0; "
    f"ordinary-wall-ladder-errors=0; wedge-z={min(z_values):.6f}..{max(z_values):.6f}; wedge-r={min(radii):.6f}..{max(radii):.6f}."
)
'''
from collections import Counter
import json
import math
from pathlib import Path
import sys


mesh_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("curved_dam_mesh.msh")
lines = mesh_path.read_text().splitlines()
markers = json.loads(mesh_path.with_name("curved_dam_mesh_meta.json").read_text())["topology_markers"]


def section_rows(name):
    start = lines.index(f"${name}") + 1
    count = int(lines[start])
    return [line.split() for line in lines[start + 1:start + 1 + count]]


nodes = {int(row[0]): tuple(map(float, row[1:])) for row in section_rows("Nodes")}
elements = {
    int(row[0]): (int(row[1]), tuple(map(int, row[3 + int(row[2]):])))
    for row in section_rows("Elements")
}
face_templates = {
    4: ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
    5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
}


def tetra_volume(node_ids):
    (ax, ay, az), (bx, by, bz), (cx, cy, cz), (dx, dy, dz) = (nodes[node_id] for node_id in node_ids)
    return abs(
def tetra_volume(node_ids):
     (ax, ay, az), (bx, by, bz), (cx, cy, cz), (dx, dy, dz) = (nodes[node_id] for node_id in node_ids)
     return abs(
          (bx - ax) * ((cy - ay) * (dz - az) - (cz - az) * (dy - ay))
          - (by - ay) * ((cx - ax) * (dz - az) - (cz - az) * (dx - ax))
          + (bz - az) * ((cx - ax) * (dy - ay) - (cy - ay) * (dx - ax))
     ) / 6.0
        (bx - ax) * ((cy - ay) * (dz - az) - (cz - az) * (dy - ay))
        - (by - ay) * ((cx - ax) * (dz - az) - (cz - az) * (dx - ax))
def volume(element_type, node_ids):
     if element_type == 4:
          return tetra_volume(node_ids)
     return sum(tetra_volume(tuple(node_ids[index] for index in tetrahedron)) for tetrahedron in (
          (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
          (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
     ))
        + (bz - az) * ((cx - ax) * (dy - ay) - (cy - ay) * (dx - ax))
    ) / 6.0


def volume(element_type, node_ids):
    if element_type == 4:
        return tetra_volume(node_ids)
    return sum(tetra_volume(tuple(node_ids[index] for index in tetrahedron)) for tetrahedron in (
        (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
        (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
    ))


volume_faces = Counter()
zero_volume = []
for element_id, (element_type, node_ids) in elements.items():
    if element_type not in face_templates:
        continue
    if volume(element_type, node_ids) <= 1.0e-12:
        zero_volume.append(element_id)
    for template in face_templates[element_type]:
        volume_faces[tuple(sorted(node_ids[index] for index in template))] += 1

boundary_faces = {
    tuple(sorted(node_ids)) for element_type, node_ids in elements.values() if element_type in (2, 3)
}
internal_boundaries = [face for face in boundary_faces if volume_faces[face] > 1]
wedge_ids = markers["wedge_element_ids"]
transition_ids = markers["transition_element_ids"]
wedge_bottom_faces = [tuple(sorted(face)) for face in markers["wedge_bottom_faces"]]
wedge_wall_faces = [tuple(sorted(face)) for face in markers["wedge_wall_faces"]]
marker_errors = (
    not wedge_ids or not transition_ids or not wedge_bottom_faces
    or any(elements.get(element_id, (None, ()))[0] != 4 for element_id in wedge_ids + transition_ids)
)
unshared_bottom_faces = [face for face in wedge_bottom_faces if volume_faces[face] != 2]
invalid_wall_faces = [
    face for face in wedge_wall_faces
    if volume_faces[face] != 2
    or any(not math.isclose(math.hypot(nodes[node_id][0], nodes[node_id][1]), 76.0, abs_tol=1.0e-8) for node_id in face)
]
wedge_node_ids = {node_id for element_id in wedge_ids for node_id in elements[element_id][1]}
wedge_nodes = [nodes[node_id] for node_id in wedge_node_ids]
z_values = [node[2] for node in wedge_nodes]
radii = [math.hypot(node[0], node[1]) for node in wedge_nodes]
outer_nodes = [node for node in wedge_nodes if math.hypot(node[0], node[1]) < 76.0 - 1.0e-8]
invalid_outer_nodes = [
    node for node in outer_nodes
    if not math.isclose(math.hypot(node[0], node[1]), 76.0 - (-25.0 - node[2]) / 2.0, abs_tol=1.0e-7)
]
dimension_errors = bool(
    not wedge_nodes or not math.isclose(max(z_values), -25.0, abs_tol=1.0e-8)
    or min(radii) < 74.0 - 1.0e-8 or not outer_nodes or invalid_outer_nodes
)

if zero_volume or internal_boundaries or marker_errors or unshared_bottom_faces or invalid_wall_faces or dimension_errors:
    raise SystemExit(
        "Mesh audit failed: "
        f"zero-volume={len(zero_volume)}, internal-explicit-boundaries={len(internal_boundaries)}, "
        f"marker-errors={int(marker_errors)}, wedge-bottom-unshared={len(unshared_bottom_faces)}, "
        f"wall-face-errors={len(invalid_wall_faces)}, dimension-errors={int(dimension_errors)}"
    )

print(
    "Mesh audit passed: "
    f"wedge-tets={len(wedge_ids)}, transition-tets={len(transition_ids)}, "
    f"wedge-bottom-faces={len(wedge_bottom_faces)} (all incidence=2), "
    f"wall-shared-faces={len(wedge_wall_faces)}, zero-volume=0, internal-explicit-boundaries=0; "
    f"wedge-z={min(z_values):.6f}..{max(z_values):.6f}; wedge-r={min(radii):.6f}..{max(radii):.6f}."
)

"""Legacy audit retained by the editor after deletion; the active audit ends above.
from collections import Counter
from pathlib import Path
import math
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


def tetra_volume(node_ids):
    (ax, ay, az), (bx, by, bz), (cx, cy, cz), (dx, dy, dz) = (nodes[node_id] for node_id in node_ids)
    return abs(
        (bx - ax) * ((cy - ay) * (dz - az) - (cz - az) * (dy - ay))
        - (by - ay) * ((cx - ax) * (dz - az) - (cz - az) * (dx - ax))
        + (bz - az) * ((cx - ax) * (dy - ay) - (cy - ay) * (dx - ax))
    ) / 6.0


tetrahedra = [node_ids for element_type, node_ids in elements if element_type == 4]
zero_volume = [node_ids for node_ids in tetrahedra if tetra_volume(node_ids) <= 1.0e-12]
wedge_node_ids = {node_id for tetrahedron in tetrahedra for node_id in tetrahedron}
wedge_nodes = [nodes[node_id] for node_id in wedge_node_ids]
wall_nodes = [node for node in wedge_nodes if math.isclose(math.hypot(node[0], node[1]), 76.0, abs_tol=1.0e-8)]
outer_nodes = [node for node in wedge_nodes if math.hypot(node[0], node[1]) < 76.0 - 1.0e-8]

invalid_outer_nodes = []
for x_value, y_value, z_value in outer_nodes:
    radius = math.hypot(x_value, y_value)
    expected_radius = 76.0 - (-25.0 - z_value) / 2.0
    if not math.isclose(radius, expected_radius, abs_tol=1.0e-7):
        invalid_outer_nodes.append((radius, z_value, expected_radius))

face_counts = Counter()
for node_ids in tetrahedra:
    for face in ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)):
        face_counts[tuple(sorted(node_ids[index] for index in face))] += 1
internal_tetra_faces = sum(count == 2 for count in face_counts.values())

if not tetrahedra or zero_volume or not wall_nodes or not outer_nodes or invalid_outer_nodes:
    raise SystemExit(
        "Mesh audit failed: "
        f"wedge-tetrahedra={len(tetrahedra)}, zero-volume={len(zero_volume)}, "
        f"shared-wall-nodes={len(wall_nodes)}, outer-wedge-nodes={len(outer_nodes)}, "
        f"invalid-wedge-radii={len(invalid_outer_nodes)}"
    )

z_values = [node[2] for node in wedge_nodes]
radii = [math.hypot(node[0], node[1]) for node in wedge_nodes]
if not math.isclose(max(z_values), -25.0, abs_tol=1.0e-8) or min(radii) < 74.0 - 1.0e-8:
    raise SystemExit("Mesh audit failed: wedge top or radial extent is outside the specified limits")

print(
    "Mesh audit passed: "
    f"wedge tetrahedra={len(tetrahedra)}, internal tetra faces={internal_tetra_faces}; "
    f"wedge z={min(z_values):.6f} to {max(z_values):.6f}; "
    f"wedge radii={min(radii):.6f} to {max(radii):.6f}; "
    f"shared wall nodes={len(wall_nodes)}; zero-volume tetrahedra=0."
)

Legacy prism audit retained as inert text below.
from collections import Counter
from pathlib import Path
import math

mesh_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("curved_dam_mesh.msh")
lines = mesh_path.read_text().splitlines()

def section_rows(name):
    start = lines.index(f"${name}") + 1
    count = int(lines[start])

nodes = {int(row[0]): tuple(map(float, row[1:])) for row in section_rows("Nodes")}
elements = [
    (int(row[1]), int(row[3]), tuple(map(int, row[3 + int(row[2]):])))

face_templates = {
    4: ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
    5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),


def tetra_volume(first, second, third, fourth):
    ax, ay, az = nodes[first]
    bx, by, bz = nodes[second]

def volume(element_type, node_ids):
    if element_type == 4:
        return tetra_volume(*node_ids)

volume_faces = Counter()
boundary_faces = set()
zero_volume_elements = []

wedge_tetrahedra = []
for element_type, physical_id, node_ids in elements:
    if element_type in face_templates:

internal_boundary_faces = boundary_faces & {face for face, count in volume_faces.items() if count == 2}
wedge_node_ids = {node_id for tetrahedron in wedge_tetrahedra for node_id in tetrahedron}
wedge_nodes = [nodes[node_id] for node_id in wedge_node_ids]

invalid_wedge_nodes = []
for x_value, y_value, z_value in outer_nodes:
    radial_distance = math.hypot(x_value, y_value)

if not wedge_tetrahedra or zero_volume_elements or internal_boundary_faces or not wall_nodes or not outer_nodes or invalid_wedge_nodes:
    raise SystemExit(
        "Mesh audit failed: "

z_values = [node[2] for node in wedge_nodes]
radii = [math.hypot(node[0], node[1]) for node in wedge_nodes]
if not math.isclose(max(z_values), -25.0, abs_tol=1.0e-8) or min(radii) < 74.0 - 1.0e-8:

print(
    "Mesh audit passed: "
    f"wedge tetrahedra={len(wedge_tetrahedra)}; "

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
"""
'''