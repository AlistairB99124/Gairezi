"""Add a bedrock-founded centre wedge and two joined side wedges along the
downstream wall/plinth corner. Every wedge has a 2m height by 2m width 1:1
right-triangular section.

Geometry per station:
  P0 = existing wall/plinth corner node (r=76.0, the shared wall-base/plinth-top node)
    P1 = point 2m up the wall face from P0 (r=76.0, z = P0.z + 2m) -- new node
    P2 = point 2m along the plinth top downstream of P0 (r=74.0, z = P0.z) -- existing
       plinth-top node, reused directly so the rib bonds to the plinth mesh

Right angle sits at P0. The centre wedge exists only on the flat z=-29 bedrock
reach, where P0/P1 reuse wall nodes and P2 is a new bedrock node. The side wedges
run from the z=-10 outer stations to the corresponding centre-wedge end face,
sharing all three nodes at each join. Their z=-10 outer ends taper over 8m.
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
HEIGHT_LEG_LENGTH = 2.0  # up the wall face
WIDTH_LEG_LENGTH = 2.0  # along the plinth top
TAPER_LENGTH = 8.0
STATION_STEP = 0.5
BEDROCK_Z = -29.0
WEDGE_ELEMENT_SIZE = 0.5

METADATA_PATH = MESH_PATH.with_name("curved_dam_mesh_meta.json")


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
            key = round(station_of(x, y), 1)
            cur = p2_candidates.get(key)
            if cur is None or z > cur[0]:
                p2_candidates[key] = (z, nid, x, y)
        elif 75.95 < r < 76.05:
            key = round(station_of(x, y), 1)
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


def build_triangular_lattice(wedge_subdivisions):
    triangles = []
    for radial_index in range(wedge_subdivisions):
        for vertical_index in range(wedge_subdivisions - radial_index):
            lower = (radial_index, vertical_index)
            radial = (radial_index + 1, vertical_index)
            vertical = (radial_index, vertical_index + 1)
            triangles.append((lower, radial, vertical))
            if radial_index + vertical_index < wedge_subdivisions - 1:
                diagonal = (radial_index + 1, vertical_index + 1)
                triangles.append((radial, diagonal, vertical))
    return triangles


def find_or_add_node(coordinates, node_lookup, new_nodes, next_node_id):
    key = tuple(round(value, 9) for value in coordinates)
    node_id = node_lookup.get(key)
    if node_id is not None:
        return node_id, next_node_id
    node_id = next_node_id
    node_lookup[key] = node_id
    new_nodes.append((node_id, *coordinates))
    return node_id, node_id + 1


def build_wedge_section(p0, p1, p2, wedge_subdivisions, node_lookup, new_nodes, next_node_id):
    section_nodes = {}
    for radial_index in range(wedge_subdivisions + 1):
        for vertical_index in range(wedge_subdivisions + 1 - radial_index):
            radial_fraction = radial_index / wedge_subdivisions
            vertical_fraction = vertical_index / wedge_subdivisions
            coordinates = tuple(
                p0[axis]
                + radial_fraction * (p2[axis] - p0[axis])
                + vertical_fraction * (p1[axis] - p0[axis])
                for axis in range(3)
            )
            node_id, next_node_id = find_or_add_node(
                coordinates, node_lookup, new_nodes, next_node_id
            )
            section_nodes[radial_index, vertical_index] = node_id
    return section_nodes, next_node_id


def build_segment_tetras(section_nodes, stations, triangles):
    tetras = []
    for i in range(len(stations) - 1):
        s0, s1 = stations[i], stations[i + 1]
        for triangle in triangles:
            a0, a1, a2 = (section_nodes[s0][index] for index in triangle)
            b0, b1, b2 = (section_nodes[s1][index] for index in triangle)
            if len({a0, a1, a2, b0, b1, b2}) == 1:
                continue
            if a0 == a1 == a2:
                tetras.append((a0, b0, b1, b2))
            elif b0 == b1 == b2:
                tetras.append((a0, a1, a2, b0))
            else:
                tetras.append((a0, a1, a2, b0))
                tetras.append((a1, a2, b0, b1))
                tetras.append((a2, b0, b1, b2))
    return tetras


def center_wedge_corners(stations, p0_candidates, p0_buckets):
    """Return the three physical corners of each bedrock-founded centre section."""
    sections = {}
    for station in stations:
        p0_z, p0_id, p0_x, p0_y = p0_buckets[station]
        wall_candidates = p0_candidates[station]
        p1_z, _, p1_x, p1_y = min(
            wall_candidates,
            key=lambda candidate: abs(candidate[0] - (p0_z + HEIGHT_LEG_LENGTH)),
        )
        if not math.isclose(p1_z, p0_z + HEIGHT_LEG_LENGTH, abs_tol=1.0e-8):
            raise ValueError(f"No wall node 2m above the bedrock wedge at station {station}")
        sections[station] = (
            (p0_x, p0_y, p0_z),
            (p1_x, p1_y, p1_z),
            (p0_x * PLINTH_LEG_R / WALL_R, p0_y * PLINTH_LEG_R / WALL_R, p0_z),
        )
    return sections


def side_wedge_corners(stations, tapered_end, direction, p0_buckets, p2_buckets):
    sections = {}
    for station in stations:
        p0_z, _, p0_x, p0_y = p0_buckets[station]
        fraction = rib_taper_fraction(station, tapered_end, direction)
        _, _, full_p2_x, full_p2_y = p2_buckets[station]
        sections[station] = (
            (p0_x, p0_y, p0_z),
            (p0_x, p0_y, p0_z + fraction * HEIGHT_LEG_LENGTH),
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

    center_wall_nodes = {
        station: next(
            (candidate for candidate in candidates if math.isclose(candidate[0], BEDROCK_Z, abs_tol=1.0e-8)),
            None,
        )
        for station, candidates in p0_candidates.items()
    }
    center_stations = sorted(
        station for station, candidate in center_wall_nodes.items() if candidate is not None
    )
    if len(center_stations) < 2:
        raise ValueError("The centre wedge requires at least two wall sections at z=-29 bedrock")
    center_start, center_end = center_stations[0], center_stations[-1]
    if any(
        not math.isclose(end - start, STATION_STEP, abs_tol=1.0e-8)
        for start, end in zip(center_stations, center_stations[1:])
    ):
        raise ValueError("The z=-29 bedrock stations must form one contiguous half-metre grid")
    for station in center_stations:
        p0_buckets[station] = center_wall_nodes[station]

    # Locate the z=-10 crossing station on each side of the wedge by scanning the
    # true wall/plinth corner elevation profile (p0_buckets), moving away from the
    # wedge until the elevation profile passes -10 m.
    low_candidates = sorted(s for s in p0_buckets if s <= center_start)
    high_candidates = sorted(s for s in p0_buckets if s >= center_end)

    def nearest_to_minus10(candidate_stations):
        return min(candidate_stations, key=lambda s: abs(p0_buckets[s][0] + 10.0))

    low_outer = nearest_to_minus10(low_candidates)
    high_outer = nearest_to_minus10(high_candidates)

    print(f"Low-side outer tip station: {low_outer} (corner z={p0_buckets[low_outer][0]:.3f})")
    print(f"High-side outer tip station: {high_outer} (corner z={p0_buckets[high_outer][0]:.3f})")
    print(f"Centre wedge bedrock span: {center_start} to {center_end} (z={BEDROCK_Z:.1f})")

    low_stations = [s for s in low_candidates if low_outer <= s <= center_start]
    high_stations = [s for s in high_candidates if center_end <= s <= high_outer]

    next_node_id = max_node_id + 1
    all_new_node_lines = []
    all_new_tetras = []

    wedge_subdivisions = int(round(WIDTH_LEG_LENGTH / WEDGE_ELEMENT_SIZE))
    if wedge_subdivisions < 1 or not math.isclose(
        wedge_subdivisions * WEDGE_ELEMENT_SIZE, WIDTH_LEG_LENGTH, abs_tol=1.0e-9
    ) or not math.isclose(
        wedge_subdivisions * WEDGE_ELEMENT_SIZE, HEIGHT_LEG_LENGTH, abs_tol=1.0e-9
    ):
        raise ValueError("Wedge height and width must be whole multiples of the global element size")
    triangles = build_triangular_lattice(wedge_subdivisions)
    node_lookup = {
        tuple(round(value, 9) for value in coordinates): node_id
        for node_id, coordinates in nodes.items()
    }

    center_corners = center_wedge_corners(center_stations, p0_candidates, p0_buckets)
    center_sections = {}
    for station in center_stations:
        center_sections[station], next_node_id = build_wedge_section(
            *center_corners[station], wedge_subdivisions, node_lookup, all_new_node_lines, next_node_id
        )
    all_new_tetras.extend(build_segment_tetras(center_sections, center_stations, triangles))

    # The shared end sections use the same lattice node IDs as the centre wedge.
    low_corners = side_wedge_corners(
        [station for station in low_stations if station not in center_sections],
        low_outer,
        +1,
        p0_buckets,
        p2_buckets,
    )
    low_sections = {}
    for station in low_stations:
        if station in center_sections:
            low_sections[station] = center_sections[station]
        else:
            low_sections[station], next_node_id = build_wedge_section(
                *low_corners[station], wedge_subdivisions, node_lookup, all_new_node_lines, next_node_id
            )
    all_new_tetras.extend(build_segment_tetras(low_sections, low_stations, triangles))

    high_corners = side_wedge_corners(
        [station for station in high_stations if station not in center_sections],
        high_outer,
        -1,
        p0_buckets,
        p2_buckets,
    )
    high_sections = {}
    for station in high_stations:
        if station in center_sections:
            high_sections[station] = center_sections[station]
        else:
            high_sections[station], next_node_id = build_wedge_section(
                *high_corners[station], wedge_subdivisions, node_lookup, all_new_node_lines, next_node_id
            )
    all_new_tetras.extend(build_segment_tetras(high_sections, high_stations, triangles))

    all_coords = dict(nodes)
    all_coords.update({node_id: (x, y, z) for node_id, x, y, z in all_new_node_lines})

    zero_volume = 0
    min_abs_vol = None
    for tet in all_new_tetras:
        vol = tetra_volume(all_coords, *tet)
        if abs(vol) < 1.0e-9:
            zero_volume += 1
        if min_abs_vol is None or abs(vol) < min_abs_vol:
            min_abs_vol = abs(vol)

    center_bedrock_faces = []
    tetra_faces = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))
    for tetrahedron in all_new_tetras:
        for face_indices in tetra_faces:
            face = tuple(tetrahedron[index] for index in face_indices)
            if all(math.isclose(all_coords[node_id][2], BEDROCK_Z, abs_tol=1.0e-8) for node_id in face):
                center_bedrock_faces.append(face)

    print(f"Wedge lattice: {wedge_subdivisions} x {wedge_subdivisions} at {WEDGE_ELEMENT_SIZE:.1f}m")
    print(f"New nodes: {len(all_new_node_lines)}")
    print(f"New tetrahedra: {len(all_new_tetras)} (zero-volume: {zero_volume}, min |vol|: {min_abs_vol:.6g})")
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
    old_element_count = int(lines[element_count_index].strip())

    # Determine next element id by scanning the existing block's last id.
    last_element_line = lines[element_end_index - 1].split()
    next_element_id = int(last_element_line[0]) + 1

    element_insert_lines = []
    for tet in all_new_tetras:
        n1, n2, n3, n4 = tet
        element_insert_lines.append(
            f"{next_element_id} 4 2 1 1 {n1} {n2} {n3} {n4}\n"
        )
        next_element_id += 1

    for face in center_bedrock_faces:
        element_insert_lines.append(
            f"{next_element_id} 2 2 1 1 {' '.join(map(str, face))}\n"
        )
        next_element_id += 1

    new_element_count = old_element_count + len(element_insert_lines)
    lines[element_count_index] = f"{new_element_count}\n"
    lines[element_end_index:element_end_index] = element_insert_lines

    MESH_PATH.write_text("".join(lines))
    print(f"Wrote {MESH_PATH} ({new_node_count} nodes, {new_element_count} elements)")


if __name__ == "__main__":
    main()
