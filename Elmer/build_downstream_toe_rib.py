"""Add a centre wedge and two joined side wedges along the
downstream wall/plinth corner. Every wedge has a 1.5m high by 2m wide
quadrilateral section with a 0.5m top shelf and a 1:1 outer slope.

Geometry per station:
  P0 = existing wall/plinth corner node (r=76.0, the shared wall-base/plinth-top node)
        P1 = point 1.5m up the wall face from P0 (r=76.0, z = P0.z + 1.5m) -- new node
    P2 = point 2m along the plinth top downstream of P0 (r=74.0, z = P0.z) -- existing
       plinth-top node, reused directly so the rib bonds to the plinth mesh

Right angle sits at P0. The centre wedge attaches directly to the 0.5m-thick
central plinth defined in Data/plinth.json. The side wedges run from the z=-10
outer stations to the corresponding centre-wedge end face, sharing all three
nodes at each join. Their z=-10 outer ends taper over 8m.
"""
from pathlib import Path
import json
import math
import shutil
import subprocess

MESH_PATH = Path(__file__).resolve().parent / "curved_dam_mesh.msh"
BACKUP_PATH = MESH_PATH.with_name(MESH_PATH.name + ".bak_pre_toe_rib_2x1")
ELMER_MESH_DIR = Path(__file__).resolve().parent / "mesh"

RADIUS_CENTERLINE = 78.0
WALL_R = 76.0
PLINTH_LEG_R = 74.0  # 2m downstream of the wall face (width leg)
HEIGHT_LEG_LENGTH = 1.5  # up the wall face
WIDTH_LEG_LENGTH = 2.0  # along the plinth top
TAPER_LENGTH = 8.0
STATION_STEP = 0.5
BEDROCK_Z = -29.0
WEDGE_ELEMENT_SIZE = 0.5
WEDGE_RADIAL_SUBDIVISIONS = 4
WEDGE_NODE_SNAP_TOLERANCE = 1.0e-8

METADATA_PATH = MESH_PATH.with_name("curved_dam_mesh_meta.json")


def resolve_elmergrid():
    for candidate in (
        shutil.which("ElmerGrid"),
        "/usr/local/bin/ElmerGrid",
        "/Users/alistairdavies/elmerfem/bin/ElmerGrid",
        "/Users/alistairdavies/Library/Application Support/Code/User/globalStorage/github.copilot-chat/debugCommand/ElmerGrid",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def convert_to_elmer_mesh():
    elmergrid = resolve_elmergrid()
    if not elmergrid:
        raise RuntimeError("ElmerGrid was not found; cannot update the solver mesh with the downstream wedge")
    if ELMER_MESH_DIR.exists():
        shutil.rmtree(ELMER_MESH_DIR)
    subprocess.run(
        [elmergrid, "14", "2", str(MESH_PATH), "-out", str(ELMER_MESH_DIR)],
        check=True,
        cwd=MESH_PATH.parent,
    )
    print(f"Converted {MESH_PATH} -> {ELMER_MESH_DIR}")


def parse_msh(path):
    with path.open() as fh:
        lines = fh.readlines()

    nodes = {}
    node_section = None  # (start_line_index_of_count, end_index_exclusive)
    element_section = None
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line == "$Nodes":
            count_index = i + 1
            count = int(lines[count_index].strip())
            start = count_index + 1
            for k in range(count):
                parts = lines[start + k].split()
                nid = int(parts[0])
                nodes[nid] = (float(parts[1]), float(parts[2]), float(parts[3]))
            node_section = (count_index, start + count)
            i = start + count
        elif line == "$Elements":
            count_index = i + 1
            count = int(lines[count_index].strip())
            start = count_index + 1
            element_section = (count_index, start + count)
            i = start + count
        else:
            i += 1

    return lines, nodes, node_section, element_section


def station_of(x, y):
    theta = math.atan2(x, y)
    return theta * RADIUS_CENTERLINE


def build_station_buckets(nodes):
    """Bucket nodes by rounded station for the wall and wedge-toe radii."""
    p2_candidates = {}
    p0_candidates = {}
    for nid, (x, y, z) in nodes.items():
        r = math.hypot(x, y)
        if PLINTH_LEG_R - 0.05 < r < PLINTH_LEG_R + 0.05:
            key = round(station_of(x, y), 6)
            cur = p2_candidates.get(key)
            if cur is None or z > cur[0]:
                p2_candidates[key] = (z, nid, x, y)
        elif 75.95 < r < 76.05:
            key = round(station_of(x, y), 6)
            p0_candidates.setdefault(key, []).append((z, nid, x, y))
    return p0_candidates, p2_candidates


def resolve_p0(p0_candidates, p2_buckets):
    """Disambiguate the true wall/plinth corner from the r=76 column by matching
    against the unambiguous plinth-top elevation at the wedge toe radius."""
    p0_buckets = {}
    for station, cands in p0_candidates.items():
        target = p2_buckets.get(station)
        if target is None:
            continue
        best = min(cands, key=lambda c: abs(c[0] - target[0]))
        p0_buckets[station] = best
    return p0_buckets


def rib_taper_fraction(station, taper_end_station, direction):
    """direction=+1 means the taper (outer, zero-width) end is at the LOW station
    of the range; direction=-1 means it is at the HIGH station of the range."""
    if direction > 0:
        if station <= taper_end_station:
            return 0.0
        if station >= taper_end_station + TAPER_LENGTH:
            return 1.0
        return (station - taper_end_station) / TAPER_LENGTH
    else:
        if station >= taper_end_station:
            return 0.0
        if station <= taper_end_station - TAPER_LENGTH:
            return 1.0
        return (taper_end_station - station) / TAPER_LENGTH


def tetra_volume(nodes, a, b, c, d):
    ax, ay, az = nodes[a]
    bx, by, bz = nodes[b]
    cx, cy, cz = nodes[c]
    dx, dy, dz = nodes[d]
    v1 = (bx - ax, by - ay, bz - az)
    v2 = (cx - ax, cy - ay, cz - az)
    v3 = (dx - ax, dy - ay, dz - az)
    cross = (
        v2[1] * v3[2] - v2[2] * v3[1],
        v2[2] * v3[0] - v2[0] * v3[2],
        v2[0] * v3[1] - v2[1] * v3[0],
    )
    dot = v1[0] * cross[0] + v1[1] * cross[1] + v1[2] * cross[2]
    return dot / 6.0


def hexahedron_volume(nodes, node_ids):
    tetrahedra = (
        (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
        (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
    )
    return sum(abs(tetra_volume(nodes, *(node_ids[index] for index in tetrahedron))) for tetrahedron in tetrahedra)


def find_or_add_node(coordinates, node_lookup, new_nodes, next_node_id):
    key = tuple(round(value, 7) for value in coordinates)
    node_id = node_lookup.get(key)
    if node_id is not None:
        return node_id, next_node_id
    node_id = next_node_id
    node_lookup[key] = node_id
    new_nodes.append((node_id, *coordinates))
    return node_id, node_id + 1


def build_triangle_section(
    p0,
    p1,
    p2,
    radial_subdivisions,
    node_lookup,
    new_nodes,
    next_node_id,
):
    """Build the 0.5m lattice in the specified four-sided wedge profile."""
    vertical_subdivisions = int(round(HEIGHT_LEG_LENGTH / WEDGE_ELEMENT_SIZE))
    if (
        vertical_subdivisions != 3
        or radial_subdivisions != 4
        or not math.isclose(vertical_subdivisions * WEDGE_ELEMENT_SIZE, HEIGHT_LEG_LENGTH, abs_tol=1.0e-9)
        or not math.isclose(radial_subdivisions * WEDGE_ELEMENT_SIZE, WIDTH_LEG_LENGTH, abs_tol=1.0e-9)
    ):
        raise ValueError("The quadrilateral wedge dimensions must align with the global 0.5m mesh")
    section_nodes = {}
    for vertical_index in range(vertical_subdivisions + 1):
        row_width_cells = radial_subdivisions - vertical_index
        for radial_index in range(row_width_cells + 1):
            radial_fraction = radial_index / radial_subdivisions
            vertical_fraction = vertical_index / vertical_subdivisions
            coordinates = tuple(
                p0[axis]
                + radial_fraction * (p2[axis] - p0[axis])
                + vertical_fraction * (p1[axis] - p0[axis])
                for axis in range(3)
            )
            section_nodes[radial_index, vertical_index], next_node_id = find_or_add_node(
                coordinates, node_lookup, new_nodes, next_node_id
            )
    triangles = []
    for vertical_index in range(vertical_subdivisions):
        row_width_cells = radial_subdivisions - vertical_index
        for radial_index in range(row_width_cells):
            lower_left = radial_index, vertical_index
            lower_right = radial_index + 1, vertical_index
            upper_left = radial_index, vertical_index + 1
            triangles.append((lower_left, lower_right, upper_left))
            if radial_index > 0:
                upper_right = radial_index - 1, vertical_index + 1
                triangles.append((lower_left, upper_left, upper_right))
    return section_nodes, triangles, next_node_id


def build_segment_tetrahedra(
    section_nodes,
    triangles,
    stations,
    nodes,
):
    tetrahedra = []
    for i in range(len(stations) - 1):
        s0, s1 = stations[i], stations[i + 1]
        start = section_nodes[s0]
        end = section_nodes[s1]
        for triangle in triangles:
            a, b, c = (start[index] for index in triangle)
            A, B, C = (end[index] for index in triangle)
            for tetrahedron in ((a, b, c, C), (a, b, B, C), (a, A, B, C)):
                if len(set(tetrahedron)) == 4 and abs(tetra_volume(nodes, *tetrahedron)) >= 1.0e-9:
                    tetrahedra.append(tetrahedron)
    return tetrahedra


def center_wedge_corners(stations, p0_buckets, p2_buckets, p0_candidates):
    """Return the three physical corners of each centre-plinth wedge section."""
    sections = {}
    for station in stations:
        p0_z, _, p0_x, p0_y = p0_buckets[station]
        p2_z, _, p2_x, p2_y = p2_buckets[station]
        if not math.isclose(p2_z, p0_z, abs_tol=1.0e-8):
            raise ValueError(f"No matching central plinth section at station {station:g}")
        p2 = (p2_x, p2_y, p2_z)
        wall_candidates = p0_candidates[station]
        p1_z, _, p1_x, p1_y = min(
            wall_candidates,
            key=lambda candidate: abs(candidate[0] - (p0_z + HEIGHT_LEG_LENGTH)),
        )
        if not math.isclose(p1_z, p0_z + HEIGHT_LEG_LENGTH, abs_tol=1.0e-8):
            raise ValueError(
                f"No wall node {HEIGHT_LEG_LENGTH:g}m above the bedrock wedge at station {station}"
            )
        sections[station] = (
            (p0_x, p0_y, p0_z),
            (p1_x, p1_y, p1_z),
            p2,
        )
    return sections


def side_wedge_corners(stations, tapered_end, direction, p0_buckets, p2_buckets, p0_candidates):
    sections = {}
    for station in stations:
        p0_z, _, p0_x, p0_y = p0_buckets[station]
        fraction = rib_taper_fraction(station, tapered_end, direction)
        _, _, full_p2_x, full_p2_y = p2_buckets[station]
        apex_z = p0_z + fraction * HEIGHT_LEG_LENGTH
        apex = min(
            p0_candidates[station],
            key=lambda candidate: abs(candidate[0] - apex_z),
        )
        if (
            math.isclose(fraction, 1.0, abs_tol=1.0e-8)
            and abs(apex[0] - apex_z) <= WEDGE_NODE_SNAP_TOLERANCE
        ):
            apex_x, apex_y, apex_z = apex[2], apex[3], apex[0]
        else:
            apex_x, apex_y = p0_x, p0_y
        sections[station] = (
            (p0_x, p0_y, p0_z),
            (apex_x, apex_y, apex_z),
            (
                p0_x + fraction * (full_p2_x - p0_x),
                p0_y + fraction * (full_p2_y - p0_y),
                p0_z,
            ),
        )
    return sections


def main():
    lines, nodes, node_section, element_section = parse_msh(MESH_PATH)
    max_node_id = max(nodes)
    metadata = json.loads(METADATA_PATH.read_text())
    join_station = float(metadata["side_wedge_join_station_m"])

    p0_candidates, p2_buckets = build_station_buckets(nodes)
    p0_buckets = resolve_p0(p0_candidates, p2_buckets)

    center_start = float(metadata["center_plinth_start_station_m"])
    center_end = float(metadata["center_plinth_end_station_m"])
    center_stations = [
        round(center_start + index * STATION_STEP, 9)
        for index in range(int(round((center_end - center_start) / STATION_STEP)) + 1)
    ]
    if any(station not in p0_buckets for station in center_stations):
        raise ValueError("The parent mesh is missing a central plinth corner section")

    # Locate the z=-10 crossing station on each side of the wedge by scanning the
    # true wall/plinth corner elevation profile (p0_buckets), moving away from the
    # wedge until the elevation profile passes -10 m.
    raised_center_start = center_stations[0]
    raised_center_end = center_stations[-1]
    wedge_lattice_stations = (
        station for station in p0_buckets
        if math.isclose(
            station / STATION_STEP,
            round(station / STATION_STEP),
            abs_tol=1.0e-8,
        )
    )
    low_candidates = sorted(s for s in wedge_lattice_stations if s <= raised_center_start)
    high_candidates = sorted(s for s in p0_buckets if (
        s >= raised_center_end
        and math.isclose(s / STATION_STEP, round(s / STATION_STEP), abs_tol=1.0e-8)
    ))

    def nearest_to_minus10(candidate_stations):
        return min(candidate_stations, key=lambda s: abs(p0_buckets[s][0] + 10.0))

    low_outer = nearest_to_minus10(low_candidates)
    high_outer = nearest_to_minus10(high_candidates)

    print(f"Low-side outer tip station: {low_outer} (corner z={p0_buckets[low_outer][0]:.3f})")
    print(f"High-side outer tip station: {high_outer} (corner z={p0_buckets[high_outer][0]:.3f})")
    print(
        f"Centre wedge plinth span: {center_stations[0]} to {center_stations[-1]} "
        f"(z={p0_buckets[center_stations[0]][0]:.1f})"
    )

    low_stations = [s for s in low_candidates if low_outer <= s <= raised_center_start]
    high_stations = [s for s in high_candidates if raised_center_end <= s <= high_outer]

    next_node_id = max_node_id + 1
    all_new_node_lines = []
    all_new_hexahedra = []
    all_new_tetrahedra = []

    vertical_subdivisions = int(round(HEIGHT_LEG_LENGTH / WEDGE_ELEMENT_SIZE))
    radial_subdivisions = WEDGE_RADIAL_SUBDIVISIONS
    if vertical_subdivisions < 1 or radial_subdivisions < 1 or not math.isclose(
        vertical_subdivisions * WEDGE_ELEMENT_SIZE, HEIGHT_LEG_LENGTH, abs_tol=1.0e-9
    ) or WIDTH_LEG_LENGTH / radial_subdivisions > WEDGE_ELEMENT_SIZE + 1.0e-9:
        raise ValueError("Wedge subdivisions must not exceed the configured element size")
    node_lookup = {
        tuple(round(value, 7) for value in coordinates): node_id
        for node_id, coordinates in nodes.items()
    }

    center_corners = center_wedge_corners(
        center_stations, p0_buckets, p2_buckets, p0_candidates
    )
    center_sections = {}
    wedge_triangles = None
    for station in center_stations:
        center_sections[station], section_triangles, next_node_id = build_triangle_section(
            *center_corners[station], radial_subdivisions,
            node_lookup, all_new_node_lines, next_node_id
        )
        if wedge_triangles is None:
            wedge_triangles = section_triangles
    # The side and centre sections are assembled into one ordered lattice below.
    # Centre nodes are reused at both joins so the resulting wedge has no
    # side/centre interface or separate wedge-volume boundary.
    low_corners = side_wedge_corners(
        [station for station in low_stations if station not in center_sections],
        low_outer,
        +1,
        p0_buckets,
        p2_buckets,
        p0_candidates,
    )
    low_sections = {}
    for station in low_stations:
        if station in center_sections:
            low_sections[station] = center_sections[station]
        else:
            low_sections[station], _, next_node_id = build_triangle_section(
                *low_corners[station], radial_subdivisions,
                node_lookup, all_new_node_lines, next_node_id
            )
    high_corners = side_wedge_corners(
        [station for station in high_stations if station not in center_sections],
        high_outer,
        -1,
        p0_buckets,
        p2_buckets,
        p0_candidates,
    )
    high_sections = {}
    for station in high_stations:
        if station in center_sections:
            high_sections[station] = center_sections[station]
        else:
            high_sections[station], _, next_node_id = build_triangle_section(
                *high_corners[station], radial_subdivisions,
                node_lookup, all_new_node_lines, next_node_id
            )
    nodes.update({node_id: (x, y, z) for node_id, x, y, z in all_new_node_lines})
    wedge_stations = [*low_stations, *center_stations[1:], *high_stations[1:]]
    wedge_sections = {**low_sections, **center_sections, **high_sections}
    if any(
        not math.isclose(end - start, STATION_STEP, abs_tol=1.0e-8)
        for start, end in zip(wedge_stations, wedge_stations[1:])
    ):
        raise ValueError("The unified wedge stations must form one contiguous half-metre grid")
    all_new_tetrahedra.extend(
        build_segment_tetrahedra(wedge_sections, wedge_triangles, wedge_stations, nodes)
    )

    all_coords = dict(nodes)
    all_coords.update({node_id: (x, y, z) for node_id, x, y, z in all_new_node_lines})

    used_new_node_ids = {
        node_id
        for element in [*all_new_hexahedra, *all_new_tetrahedra]
        for node_id in element
    }
    all_new_node_lines = [
        node for node in all_new_node_lines if node[0] in used_new_node_ids
    ]
    all_coords = dict(nodes)
    all_coords.update({node_id: (x, y, z) for node_id, x, y, z in all_new_node_lines})

    zero_volume = 0
    min_abs_vol = None
    for hexahedron in all_new_hexahedra:
        vol = hexahedron_volume(all_coords, hexahedron)
        if abs(vol) < 1.0e-9:
            zero_volume += 1
        if min_abs_vol is None or abs(vol) < min_abs_vol:
            min_abs_vol = abs(vol)
    for tetrahedron in all_new_tetrahedra:
        vol = abs(tetra_volume(all_coords, *tetrahedron))
        if vol < 1.0e-9:
            zero_volume += 1
        if min_abs_vol is None or vol < min_abs_vol:
            min_abs_vol = vol
    hexahedron_base_face = (0, 1, 5, 4)
    bedrock_face_counts = {}
    for hexahedron in all_new_hexahedra:
        face = tuple(hexahedron[index] for index in hexahedron_base_face)
        if all(math.isclose(all_coords[node_id][2], BEDROCK_Z, abs_tol=1.0e-8) for node_id in face):
            face_key = tuple(sorted(face))
            bedrock_face_counts[face_key] = bedrock_face_counts.get(face_key, 0) + 1
    center_bedrock_faces = [
        face for face, count in bedrock_face_counts.items()
        if count == 1
    ]

    print(
        f"Quadrilateral wedge: {HEIGHT_LEG_LENGTH:.1f}m wall height, 0.5m top shelf, "
        f"1:1 slope, {WIDTH_LEG_LENGTH:.1f}m base"
    )
    print(f"New nodes: {len(all_new_node_lines)}")
    print(
        f"New cells: {len(all_new_hexahedra)} hexahedra, {len(all_new_tetrahedra)} tetrahedra "
        f"(zero-volume: {zero_volume}, min |vol|: {min_abs_vol:.6g})"
    )
    print(f"Centre bedrock boundary triangles: {len(center_bedrock_faces)}")

    if not BACKUP_PATH.exists():
        shutil.copy2(MESH_PATH, BACKUP_PATH)
        print(f"Backed up original mesh to {BACKUP_PATH}")

    # --- Splice new nodes into the $Nodes block ---
    node_count_index, node_end_index = node_section
    old_node_count = int(lines[node_count_index].strip())
    new_node_count = old_node_count + len(all_new_node_lines)
    lines[node_count_index] = f"{new_node_count}\n"
    node_insert_lines = [f"{nid} {x:.12g} {y:.12g} {z:.12g}\n" for nid, x, y, z in all_new_node_lines]
    lines[node_end_index:node_end_index] = node_insert_lines
    shift = len(node_insert_lines)

    # --- Splice new elements into the $Elements block (indices shift by `shift`) ---
    element_count_index, element_end_index = element_section
    element_count_index += shift
    element_end_index += shift

    hexahedron_faces = (
        (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0),
    )
    tetrahedron_faces = ((0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 0, 3))
    new_wedge_face_counts = {}
    new_wedge_face_nodes = {}
    for hexahedron in all_new_hexahedra:
        for face in hexahedron_faces:
            node_ids = tuple(hexahedron[index] for index in face)
            face_key = tuple(sorted(node_ids))
            new_wedge_face_counts[face_key] = new_wedge_face_counts.get(face_key, 0) + 1
            new_wedge_face_nodes.setdefault(face_key, node_ids)
    for tetrahedron in all_new_tetrahedra:
        for face in tetrahedron_faces:
            node_ids = tuple(tetrahedron[index] for index in face)
            face_key = tuple(sorted(node_ids))
            new_wedge_face_counts[face_key] = new_wedge_face_counts.get(face_key, 0) + 1
            new_wedge_face_nodes.setdefault(face_key, node_ids)
    new_wedge_faces = set(new_wedge_face_counts)
    new_wedge_triangles = {
        face for face in new_wedge_faces
        if len(face) == 3 and new_wedge_face_counts[face] == 1
    }

    def parent_quad_is_covered_by_new_triangles(face_node_ids):
        if len(face_node_ids) != 4:
            return False
        return sum(
            tuple(sorted((face_node_ids[index] for index in indices))) in new_wedge_triangles
            for indices in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
        ) == 2

    old_element_lines = lines[element_count_index + 1:element_end_index]
    retained_element_lines = []
    superseded_boundary_count = 0
    for line in old_element_lines:
        fields = line.split()
        element_type = int(fields[1])
        tag_count = int(fields[2])
        node_ids = tuple(sorted(map(int, fields[3 + tag_count:])))
        if element_type in (2, 3) and (
            node_ids in new_wedge_faces
            or parent_quad_is_covered_by_new_triangles(node_ids)
        ):
            superseded_boundary_count += 1
            continue
        retained_element_lines.append(line)
    lines[element_count_index + 1:element_end_index] = retained_element_lines
    element_end_index = element_count_index + 1 + len(retained_element_lines)
    old_element_count = len(retained_element_lines)

    # Determine next element id by scanning the existing block's last id.
    last_element_line = lines[element_end_index - 1].split()
    next_element_id = int(last_element_line[0]) + 1

    element_insert_lines = []
    for tetrahedron in all_new_tetrahedra:
        element_insert_lines.append(
            f"{next_element_id} 4 2 1 1 {' '.join(map(str, tetrahedron))}\n"
        )
        next_element_id += 1
    for hexahedron in all_new_hexahedra:
        n1, n2, n3, n4, n5, n6, n7, n8 = hexahedron
        element_insert_lines.append(
            f"{next_element_id} 5 2 1 1 {n1} {n2} {n3} {n4} {n5} {n6} {n7} {n8}\n"
        )
        next_element_id += 1

    parent_volume_faces = set()
    volume_face_patterns = {
        4: ((0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
        5: hexahedron_faces,
        7: ((0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
    }
    for line in retained_element_lines:
        fields = list(map(int, line.split()))
        element_type, tag_count = fields[1], fields[2]
        if element_type not in volume_face_patterns:
            continue
        node_ids = fields[3 + tag_count:]
        for face in volume_face_patterns[element_type]:
            parent_volume_faces.add(tuple(sorted(node_ids[index] for index in face)))

    parent_volume_quads = {
        face for face in parent_volume_faces
        if len(face) == 4
    }
    parent_quad_triangles = {
        tuple(sorted((quad[index] for index in indices))): quad
        for quad in parent_volume_quads
        for indices in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    }

    def triangle_is_covered_by_parent_quad(face_node_ids):
        return len(face_node_ids) == 3 and face_node_ids in parent_quad_triangles

    for face_key, node_ids in new_wedge_face_nodes.items():
        if (
            new_wedge_face_counts[face_key] != 1
            or face_key in parent_volume_faces
            or triangle_is_covered_by_parent_quad(face_key)
        ):
            continue
        coordinates = [all_coords[node_id] for node_id in node_ids]
        radii = [math.hypot(x, y) for x, y, _ in coordinates]
        elevations = [z for _, _, z in coordinates]
        if all(math.isclose(z, BEDROCK_Z, abs_tol=1.0e-8) for z in elevations):
            physical_id = 1
        elif max(radii) <= PLINTH_LEG_R + 1.0e-8:
            physical_id = 3
        else:
            physical_id = 7
        element_type = 2 if len(node_ids) == 3 else 3
        element_insert_lines.append(
            f"{next_element_id} {element_type} 2 {physical_id} {physical_id} {' '.join(map(str, node_ids))}\n"
        )
        next_element_id += 1

    new_element_count = old_element_count + len(element_insert_lines)
    lines[element_count_index] = f"{new_element_count}\n"
    lines[element_end_index:element_end_index] = element_insert_lines

    MESH_PATH.write_text("".join(lines))
    print(f"Removed {superseded_boundary_count} base boundary faces covered by wedge hexahedra")
    print(f"Wrote {MESH_PATH} ({new_node_count} nodes, {new_element_count} elements)")
    convert_to_elmer_mesh()


if __name__ == "__main__":
    main()
