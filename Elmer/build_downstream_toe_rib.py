"""Add a 2m (height) x 1m (width) right-angle triangular-prism rib along the
downstream wall/plinth corner, running lengthwise (constant cross-section
extruded along the chainage), on each side of the main wedge (station
89.0-123.0). Ratio matches the main wedge's 2:1 (height:width).

Geometry per station:
  P0 = existing wall/plinth corner node (r=76.0, the shared wall-base/plinth-top node)
  P1 = point 2m up the wall face from P0 (r=76.0, z = P0.z + 2m) -- new node
  P2 = point 1m along the plinth top downstream of P0 (r=75.0, z = P0.z) -- existing
       plinth-top node, reused directly so the rib bonds to the plinth mesh

Right angle sits at P0. The two ribs run from the station where the wall/plinth
corner elevation first reaches z=-10 (moving away from the wedge) up to the wedge's
own start/end station, with the cross-section linearly tapered from zero to full
size over the first/last 1m (2 station steps) at the z=-10 (outer) end only. The
end that meets the main wedge stays at full size (abrupt transition into the wedge).
"""
from pathlib import Path
import math
import shutil
import subprocess

MESH_PATH = Path(__file__).resolve().parent / "curved_dam_mesh.msh"
BACKUP_PATH = MESH_PATH.with_name(MESH_PATH.name + ".bak_pre_toe_rib_2x1")
ELMER_MESH_DIR = Path(__file__).resolve().parent / "mesh"

RADIUS_CENTERLINE = 78.0
WALL_R = 76.0
PLINTH_LEG_R = 75.0  # 1m downstream of the wall face (width leg)
HEIGHT_LEG_LENGTH = 2.0  # up the wall face
WIDTH_LEG_LENGTH = 1.0  # along the plinth top
TAPER_LENGTH = 1.0
STATION_STEP = 0.5

# Must match curved_dam_mesh_meta.json's wedge_start_station_m/wedge_end_station_m
# (the center wedge now only activates where it is already a clean 2m x 1m
# triangle, so the ribs pick up exactly where the wedge leaves off).
WEDGE_START_STATION = 94.5
WEDGE_END_STATION = 120.0


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
    """Bucket nodes by rounded station for the two radii we need."""
    p2_candidates = {}
    p0_candidates = {}
    for nid, (x, y, z) in nodes.items():
        r = math.hypot(x, y)
        if 74.95 < r < 75.05:
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
    against the unambiguous plinth-top elevation (r=75) at the same station."""
    p0_buckets = {}
    for station, cands in p0_candidates.items():
        target = p2_buckets.get(station)
        if target is None:
            continue
        best = min(cands, key=lambda c: abs(c[0] - target[0]))
        p0_buckets[station] = best
    return p0_buckets


def station_range(start, end):
    steps = round((end - start) / STATION_STEP)
    return [round(start + STATION_STEP * i, 1) for i in range(steps + 1)]


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


def build_rib_stations(stations, tapered_end, direction, p0_buckets, p2_buckets, next_node_id):
    """Return per-station node id triples (p0, p1, p2) and the updated next_node_id,
    plus a list of new (id, x, y, z) node lines to append."""
    station_nodes = {}
    new_nodes = []
    for station in stations:
        p0_z, p0_id, p0_x, p0_y = p0_buckets[station]
        frac = rib_taper_fraction(station, tapered_end, direction)
        if frac <= 0.0:
            station_nodes[station] = (p0_id, p0_id, p0_id)
            continue

        p1_x, p1_y, p1_z = p0_x, p0_y, p0_z + frac * HEIGHT_LEG_LENGTH
        p1_id = next_node_id
        next_node_id += 1
        new_nodes.append((p1_id, p1_x, p1_y, p1_z))

        if frac >= 1.0:
            _, p2_id, _, _ = p2_buckets[station]
        else:
            full_z, _, full_x, full_y = p2_buckets[station]
            p2_x = p0_x + frac * (full_x - p0_x)
            p2_y = p0_y + frac * (full_y - p0_y)
            p2_z = p0_z + frac * (full_z - p0_z)
            p2_id = next_node_id
            next_node_id += 1
            new_nodes.append((p2_id, p2_x, p2_y, p2_z))

        station_nodes[station] = (p0_id, p1_id, p2_id)

    return station_nodes, new_nodes, next_node_id


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


def build_segment_tetras(all_nodes, station_nodes, stations):
    tetras = []
    for i in range(len(stations) - 1):
        s0, s1 = stations[i], stations[i + 1]
        a0, a1, a2 = station_nodes[s0]
        b0, b1, b2 = station_nodes[s1]
        start_degenerate = a0 == a1 == a2
        end_degenerate = b0 == b1 == b2
        if start_degenerate and end_degenerate:
            continue  # both ends collapsed to the same point, nothing to build
        if start_degenerate:
            tetras.append((a0, b0, b1, b2))
        elif end_degenerate:
            tetras.append((a0, a1, a2, b0))
        else:
            tetras.append((a0, a1, a2, b0))
            tetras.append((a1, a2, b0, b1))
            tetras.append((a2, b0, b1, b2))
    return tetras


def main():
    lines, nodes, node_section, element_section = parse_msh(MESH_PATH)
    max_node_id = max(nodes)

    p0_candidates, p2_buckets = build_station_buckets(nodes)
    p0_buckets = resolve_p0(p0_candidates, p2_buckets)

    # Locate the z=-10 crossing station on each side of the wedge by scanning the
    # true wall/plinth corner elevation profile (p0_buckets), moving away from the
    # wedge until the elevation profile passes -10 m.
    low_candidates = sorted(s for s in p0_buckets if s <= WEDGE_START_STATION)
    high_candidates = sorted(s for s in p0_buckets if s >= WEDGE_END_STATION)

    def nearest_to_minus10(candidate_stations):
        return min(candidate_stations, key=lambda s: abs(p0_buckets[s][0] + 10.0))

    low_outer = nearest_to_minus10([s for s in low_candidates if s >= WEDGE_START_STATION - 60])
    high_outer = nearest_to_minus10([s for s in high_candidates if s <= WEDGE_END_STATION + 60])

    print(f"Low-side outer tip station: {low_outer} (corner z={p0_buckets[low_outer][0]:.3f})")
    print(f"High-side outer tip station: {high_outer} (corner z={p0_buckets[high_outer][0]:.3f})")

    low_stations = station_range(low_outer, WEDGE_START_STATION)
    high_stations = station_range(WEDGE_END_STATION, high_outer)

    next_node_id = max_node_id + 1
    all_new_node_lines = []
    all_new_tetras = []

    # Low side: outer (taper) end is at the LOW station of the range.
    low_station_nodes, new_nodes, next_node_id = build_rib_stations(
        low_stations, low_outer, +1, p0_buckets, p2_buckets, next_node_id
    )
    all_new_node_lines.extend(new_nodes)
    low_tetras = build_segment_tetras(nodes, low_station_nodes, low_stations)
    all_new_tetras.extend(low_tetras)

    # High side: outer (taper) end is at the HIGH station of the range.
    high_station_nodes, new_nodes, next_node_id = build_rib_stations(
        high_stations, high_outer, -1, p0_buckets, p2_buckets, next_node_id
    )
    all_new_node_lines.extend(new_nodes)
    high_tetras = build_segment_tetras(nodes, high_station_nodes, high_stations)
    all_new_tetras.extend(high_tetras)

    # Build a coordinate lookup that includes the brand new nodes for volume checks.
    all_coords = dict(nodes)
    for nid, x, y, z in all_new_node_lines:
        all_coords[nid] = (x, y, z)

    zero_volume = 0
    min_abs_vol = None
    for tet in all_new_tetras:
        vol = tetra_volume(all_coords, *tet)
        if abs(vol) < 1.0e-9:
            zero_volume += 1
        if min_abs_vol is None or abs(vol) < min_abs_vol:
            min_abs_vol = abs(vol)

    print(f"New nodes: {len(all_new_node_lines)}")
    print(f"New tetrahedra: {len(all_new_tetras)} (zero-volume: {zero_volume}, min |vol|: {min_abs_vol:.6g})")

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

    new_element_count = old_element_count + len(element_insert_lines)
    lines[element_count_index] = f"{new_element_count}\n"
    lines[element_end_index:element_end_index] = element_insert_lines

    MESH_PATH.write_text("".join(lines))
    print(f"Wrote {MESH_PATH} ({new_node_count} nodes, {new_element_count} elements)")


if __name__ == "__main__":
    main()
