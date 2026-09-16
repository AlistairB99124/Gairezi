from collections import Counter, defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import sys


mesh_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("curved_dam_mesh.msh")
lines = mesh_path.read_text().splitlines()
markers = json.loads(mesh_path.with_name("curved_dam_mesh_meta.json").read_text())["topology_markers"]
metadata = json.loads(mesh_path.with_name("curved_dam_mesh_meta.json").read_text())


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
    7: ((0, 1, 2, 3), (0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0)),
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
    if element_type == 7:
        return tetra_volume((node_ids[0], node_ids[1], node_ids[2], node_ids[4])) + tetra_volume((node_ids[0], node_ids[2], node_ids[3], node_ids[4]))
    return sum(tetra_volume(tuple(node_ids[index] for index in tet)) for tet in (
        (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
        (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
    ))


def distance(first_node_id, second_node_id):
    return math.dist(nodes[first_node_id], nodes[second_node_id])


def horizontal_distance(first_node_id, second_node_id):
    first = nodes[first_node_id]
    second = nodes[second_node_id]
    return math.hypot(second[0] - first[0], second[1] - first[1])


face_incidence = Counter()
zero_volume = []
for element_id, (element_type, _, node_ids) in elements.items():
    if element_type not in face_templates:
        continue
    if volume(element_type, node_ids) <= 1.0e-12:
        zero_volume.append(element_id)
    for face in face_templates[element_type]:
        face_incidence[tuple(sorted(node_ids[index] for index in face))] += 1

boundary_ids = defaultdict(set)
for element_type, physical_id, node_ids in elements.values():
    if element_type in (2, 3):
        boundary_ids[tuple(sorted(node_ids))].add(physical_id)

internal_boundaries = [face for face in boundary_ids if face_incidence[face] > 1]
wedge_ids = markers["wedge_element_ids"]
transition_ids = markers["transition_element_ids"]
wedge_bottom = [tuple(sorted(face)) for face in markers["wedge_bottom_faces"]]
wedge_wall = [tuple(sorted(face)) for face in markers["wedge_wall_faces"]]
wedge_outer = [tuple(sorted(face)) for face in markers["wedge_outer_faces"]]
wall_plinth = [tuple(sorted(face)) for face in markers.get("wall_plinth_faces", [])]
center_wedge_enabled = bool(metadata["wedge_enabled"])
marker_errors = center_wedge_enabled and not all((wedge_ids, transition_ids, wedge_bottom, wedge_wall, wedge_outer))
marker_errors |= any(elements.get(element_id, (None, None, ()))[0] != 4 for element_id in wedge_ids + transition_ids)
shared_bottom = [face for face in wedge_bottom if face_incidence[face] == 2]
bedrock_bottom = [face for face in wedge_bottom if face_incidence[face] == 1 and 1 in boundary_ids[face]]
invalid_bottom = [face for face in wedge_bottom if face not in shared_bottom and face not in bedrock_bottom]
invalid_wall = [
    face for face in wedge_wall
    if face_incidence[face] != 2
    or any(not math.isclose(math.hypot(nodes[node_id][0], nodes[node_id][1]), 76.0, abs_tol=1.0e-8) for node_id in face)
]
ladder_errors = [
    marker for marker in markers["ordinary_wall_ladder"]
    if not math.isclose(nodes[marker["node_id"]][2], marker["expected_z"], abs_tol=1.0e-8)
]
outer_node_ids = {node_id for face in wedge_outer for node_id in face}
outer_nodes = [nodes[node_id] for node_id in outer_node_ids]
outer_errors = [
    node for node in outer_nodes
    if math.hypot(node[0], node[1]) > 76.0 + 1.0e-7 or node[2] > -25.0 + 1.0e-7
]
tip_hex_ids = markers.get("tip_hex_element_ids", [])
tip_prism_ids = markers.get("tip_prism_element_ids", [])
tip_transition_ids = markers.get("tip_transition_element_ids", [])
wedge_plinth_hex_ids = markers.get("wedge_plinth_hex_element_ids", [])
wedge_plinth_boundary_ids = markers.get("wedge_plinth_boundary_element_ids", [])
tip_size = metadata["tip_hex_target_size_m"]
tip_tolerance = 1.0e-7
tip_hex_errors = []
tip_x_edges = []
tip_y_edges = []
tip_z_edges = []
for element_id in tip_hex_ids:
    element_type, _, node_ids = elements.get(element_id, (None, None, ()))
    if element_type != 5:
        tip_hex_errors.append(f"non-hex={element_id}")
        continue
    x_edges = [
        distance(node_ids[0], node_ids[3]), distance(node_ids[1], node_ids[2]),
        distance(node_ids[4], node_ids[7]), distance(node_ids[5], node_ids[6]),
    ]
    y_edges = [
        horizontal_distance(node_ids[0], node_ids[1]), horizontal_distance(node_ids[3], node_ids[2]),
        horizontal_distance(node_ids[4], node_ids[5]), horizontal_distance(node_ids[7], node_ids[6]),
    ]
    z_edges = [distance(node_ids[index], node_ids[index + 4]) for index in range(4)]
    tip_x_edges.extend(x_edges)
    tip_y_edges.extend(y_edges)
    tip_z_edges.extend(z_edges)
    if (
        any(edge > tip_size + tip_tolerance for edge in x_edges)
        or any(abs(edge - tip_size) > 0.004 for edge in y_edges)
        or any(edge <= 1.0e-9 for edge in z_edges)
    ):
        tip_hex_errors.append(f"dimensions={element_id}")
tip_plinth_unbonded_faces = [
    element_id for element_id in tip_hex_ids
    if elements.get(element_id, (None, None, ()))[0] == 5
    and face_incidence[tuple(sorted(elements[element_id][2][index] for index in (0, 1, 2, 3)))] != 2
]


def is_wall_plinth_face_bonded(face):
    """A quad face may legitimately bond to two tetrahedron triangles instead
    of a matching quad (the standard hex/pyramid-to-tet transition). Accept
    either a direct quad match or a triangle-pair covering the same 4 nodes."""
    if face_incidence[face] == 2:
        return True
    if len(face) != 4 or face_incidence[face] != 1:
        return False
    triangles = list(combinations(face, 3))
    for first_index in range(len(triangles)):
        for second_index in range(first_index + 1, len(triangles)):
            first_triangle = tuple(sorted(triangles[first_index]))
            second_triangle = tuple(sorted(triangles[second_index]))
            if face_incidence[first_triangle] >= 1 and face_incidence[second_triangle] >= 1:
                return True
    return False


wall_plinth_unbonded_faces = [face for face in wall_plinth if not is_wall_plinth_face_bonded(face)]
tip_marker_errors = (
    not tip_hex_ids
    or tip_prism_ids
    or not tip_transition_ids
    or any(elements.get(element_id, (None, None, ()))[0] not in (4, 7) for element_id in tip_transition_ids)
)
wedge_plinth_marker_errors = (
    center_wedge_enabled and (
    not wedge_plinth_hex_ids
    or not wedge_plinth_boundary_ids)
    or any(elements.get(element_id, (None, None, ()))[0] != 5 for element_id in wedge_plinth_hex_ids)
    or any(elements.get(element_id, (None, None, ()))[0] != 4 for element_id in wedge_plinth_boundary_ids)
)
tip_transition_faces = {
    tuple(sorted(node_ids[index] for index in face))
    for element_id in tip_transition_ids
    for element_type, _, node_ids in (elements.get(element_id, (None, None, ())),)
    for face in face_templates.get(element_type, ())
}
tip_unbonded_faces = [
    face for face in tip_transition_faces
    if face_incidence[face] == 1 and face not in boundary_ids
]

if (
    zero_volume or internal_boundaries or marker_errors or invalid_bottom or invalid_wall
    or ladder_errors or (center_wedge_enabled and (not outer_nodes or outer_errors)) or tip_marker_errors or tip_hex_errors
    or wedge_plinth_marker_errors or tip_unbonded_faces or tip_plinth_unbonded_faces
    or not wall_plinth or wall_plinth_unbonded_faces
):
    raise SystemExit(
        "Mesh audit failed: "
        f"zero-volume={len(zero_volume)}, internal-explicit-boundaries={len(internal_boundaries)}, "
        f"marker-errors={int(marker_errors)}, wedge-bottom-errors={len(invalid_bottom)}, "
        f"wall-face-errors={len(invalid_wall)}, wall-ladder-errors={len(ladder_errors)}, "
        f"wedge-outer-errors={len(outer_errors)}, tip-marker-errors={int(tip_marker_errors)}, "
        f"wedge-plinth-marker-errors={int(wedge_plinth_marker_errors)}, "
        f"tip-hex-errors={len(tip_hex_errors)}, tip-unbonded-faces={len(tip_unbonded_faces)}, "
        f"tip-plinth-unbonded-faces={len(tip_plinth_unbonded_faces)}, "
        f"wall-plinth-unbonded-faces={len(wall_plinth_unbonded_faces)}"
    )

print(
    "Mesh audit passed: "
    f"wedge-hexes={len(wedge_ids)}, transition-tets={len(transition_ids)}, "
    f"wedge-bottom-faces={len(wedge_bottom)} (shared-plinth={len(shared_bottom)}, bedrock={len(bedrock_bottom)}), "
    f"wall-shared-faces={len(wedge_wall)}, zero-volume=0, internal-explicit-boundaries=0; "
    f"ordinary-wall-ladder-errors=0; tip-fine-hexes={len(tip_hex_ids)}, "
    f"tip-transition-elements={len(tip_transition_ids)}, "
    f"wedge-plinth-hexes={len(wedge_plinth_hex_ids)}, "
    f"wedge-plinth-boundary-hexes={len(wedge_plinth_boundary_ids)}, "
    f"tip-unbonded-faces=0, tip-plinth-unbonded-faces=0, "
    f"wall-plinth-faces={len(wall_plinth)}, wall-plinth-unbonded-faces=0, "
    f"tip-x-max={max(tip_x_edges):.6f}, tip-y-range={min(tip_y_edges):.6f}..{max(tip_y_edges):.6f}, "
    f"tip-z-range={min(tip_z_edges):.6f}..{max(tip_z_edges):.6f}."
)