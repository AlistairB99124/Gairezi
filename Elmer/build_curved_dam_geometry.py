from pathlib import Path
from collections import Counter
import csv
import json
import math
import shutil
import subprocess
from typing import Optional


root = Path(__file__).resolve().parent.parent
input_path = root / "Data" / "Dam_Base_Contours.json"
plinth_path = root / "Data" / "plinth.json"
grid_controls_path = root / "Data" / "Computational_Grid_Controls.json"
out_dir = Path(__file__).resolve().parent

config_paths = [
    root / "v2" / "config.json",
    root / "config.json",
]
config = {}
for config_path in config_paths:
    if config_path.exists():
        config = json.loads(config_path.read_text())
        break

geometry_mode = str(config.get("geometry_mode", "v2")).lower()

if input_path.exists():
    with input_path.open() as fh:
        data = json.load(fh)
else:
    data = []

with grid_controls_path.open() as fh:
    grid_controls = json.load(fh)


def get_grid_control_value(control_name):
    for row in grid_controls:
        if row.get("Mesh Control") == control_name:
            return float(row["Client Value"])
    raise ValueError(f"Missing mesh control '{control_name}' in {grid_controls_path}")

wall_thickness = float(config.get("wall_thickness_m", 4.0))
upstream_radius = float(config.get("wall_upstream_radius_m", config.get("wall_radius_m", 80.0)))
radius = float(config.get("wall_centerline_radius_m", upstream_radius - 0.5 * wall_thickness))
downstream_radius = float(config.get("wall_downstream_radius_m", upstream_radius - wall_thickness))
if not math.isclose(upstream_radius - downstream_radius, wall_thickness, abs_tol=1.0e-8):
    raise ValueError("Upstream and downstream wall radii must differ by the wall thickness")
if not math.isclose(radius, 0.5 * (upstream_radius + downstream_radius), abs_tol=1.0e-8):
    raise ValueError("Wall centreline radius must lie midway between the wall-face radii")
wall_height_above_plinth_m = float(config.get("wall_height_above_plinth_m", 29.0))
wedge_enabled = bool(config.get("wedge_enabled", True))
wedge_anchor_radius = float(config.get("wedge_anchor_radius_m", downstream_radius))
wedge_start_below_crest_m = float(config.get("wedge_start_below_crest_m", 25.0))
wedge_ratio_horizontal_m = float(config.get("wedge_ratio_horizontal_m", 2.0))
wedge_ratio_vertical_m = float(config.get("wedge_ratio_vertical_m", 4.0))
if wedge_ratio_horizontal_m <= 0.0 or wedge_ratio_vertical_m <= 0.0:
    raise ValueError("Wedge horizontal and vertical ratio dimensions must be positive")
wedge_tangent = wedge_ratio_horizontal_m / wedge_ratio_vertical_m
wedge_angle_from_vertical_deg = math.degrees(math.atan(wedge_ratio_horizontal_m / wedge_ratio_vertical_m))
if not math.isclose(wedge_anchor_radius, downstream_radius, abs_tol=1.0e-8):
    raise ValueError("The wedge anchor radius must coincide with the downstream wall face")
if not 0.0 < wedge_start_below_crest_m:
    raise ValueError("The wedge start below crest must be positive")
if not 0.0 < wedge_angle_from_vertical_deg < 90.0:
    raise ValueError("The wedge angle from vertical must lie between 0 and 90 degrees")
plinth_upstream_offset_m = 1.0
plinth_downstream_offset_m = 2.0
plinth_base_width_m = wall_thickness + plinth_upstream_offset_m + plinth_downstream_offset_m
plinth_width_m = plinth_base_width_m

dam_height = 30.0
mesh_size = get_grid_control_value("Global Element Size")
target_block_size = mesh_size
local_corner_size = get_grid_control_value("Local Corner Refinement")
if not 0.0 < local_corner_size <= target_block_size:
    raise ValueError("Local Corner Refinement must be positive and no larger than Global Element Size")
arch_subdivisions = 3
vertical_layers = 4
thickness_layers = max(1, int(round(wall_thickness / target_block_size)))
crest_detail_height = 0.0
crest_extra_thickness = 0.0
abutment_segment_count = 6


def interpolate_profile_points(profile_points, subdivisions):
    refined = []
    for index in range(len(profile_points) - 1):
        start = profile_points[index]
        end = profile_points[index + 1]
        for step in range(subdivisions):
            fraction = step / subdivisions
            theta = start["theta"] + fraction * (end["theta"] - start["theta"])
            refined.append(
                {
                    "station": start["station"] + fraction * (end["station"] - start["station"]),
                    "theta": theta,
                    "x": radius * math.sin(theta),
                    "y": radius * math.cos(theta),
                    "base_z": start["base_z"] + fraction * (end["base_z"] - start["base_z"]),
                    "crest_z": start["crest_z"] + fraction * (end["crest_z"] - start["crest_z"]),
                    "ground_z": start.get("ground_z", start["base_z"]) + fraction * (end.get("ground_z", end["base_z"]) - start.get("ground_z", start["base_z"])),
                    "plinth_z": start.get("plinth_z", start["base_z"]) + fraction * (end.get("plinth_z", end["base_z"]) - start.get("plinth_z", start["base_z"])),
                    "wedge_offset_m": start.get("wedge_offset_m", 0.0) + fraction * (end.get("wedge_offset_m", 0.0) - start.get("wedge_offset_m", 0.0)),
                }
            )
    refined.append(profile_points[-1].copy())
    refined[-1]["x"] = radius * math.sin(refined[-1]["theta"])
    refined[-1]["y"] = radius * math.cos(refined[-1]["theta"])
    return refined


def interpolate_profile_points_by_target_spacing(profile_points, target_spacing_m):
    if len(profile_points) < 2:
        return profile_points

    start_station = profile_points[0]["station"]
    end_station = profile_points[-1]["station"]
    if end_station <= start_station:
        return profile_points

    resampled_stations = [start_station]
    station = start_station + target_spacing_m
    while station < end_station:
        resampled_stations.append(station)
        station += target_spacing_m
    if resampled_stations[-1] != end_station:
        resampled_stations.append(end_station)

    refined = []
    segment_index = 0
    for sample_station in resampled_stations:
        while (
            segment_index < len(profile_points) - 2
            and sample_station > profile_points[segment_index + 1]["station"]
        ):
            segment_index += 1

        start = profile_points[segment_index]
        end = profile_points[segment_index + 1]
        delta_station = end["station"] - start["station"]
        if abs(delta_station) < 1.0e-12:
            fraction = 0.0
        else:
            fraction = (sample_station - start["station"]) / delta_station

        theta = start["theta"] + fraction * (end["theta"] - start["theta"])
        refined.append(
            {
                "station": sample_station,
                "theta": theta,
                "x": radius * math.sin(theta),
                "y": radius * math.cos(theta),
                "base_z": start["base_z"] + fraction * (end["base_z"] - start["base_z"]),
                "crest_z": start["crest_z"] + fraction * (end["crest_z"] - start["crest_z"]),
                "ground_z": start.get("ground_z", start["base_z"]) + fraction * (end.get("ground_z", end["base_z"]) - start.get("ground_z", start["base_z"])),
                "plinth_z": start.get("plinth_z", start["base_z"]) + fraction * (end.get("plinth_z", end["base_z"]) - start.get("plinth_z", start["base_z"])),
                "wedge_offset_m": start.get("wedge_offset_m", 0.0) + fraction * (end.get("wedge_offset_m", 0.0) - start.get("wedge_offset_m", 0.0)),
            }
        )

    return refined


def assign_normals(profile_points):
    for point in profile_points:
        # The centreline is a 78 m circular arc. Use its exact inward radial normal so
        # positive downstream offsets follow concentric arcs, including the 76 m face.
        point["nx"] = -math.sin(point["theta"])
        point["ny"] = -math.cos(point["theta"])


def enforce_true_arc(profile_points):
    if len(profile_points) < 2:
        return profile_points

    start_theta = math.atan2(profile_points[0]["x"], profile_points[0]["y"])
    end_theta = math.atan2(profile_points[-1]["x"], profile_points[-1]["y"])
    span = end_theta - start_theta
    if span < 0.0:
        span += 2.0 * math.pi
    if abs(span) < 1.0e-8:
        span = 2.0 * math.pi

    start_station = profile_points[0]["station"]
    end_station = profile_points[-1]["station"]
    station_span = end_station - start_station
    if abs(station_span) < 1.0e-12:
        return profile_points

    for point in profile_points:
        fraction = (point["station"] - start_station) / station_span
        point["theta"] = start_theta + fraction * span
        point["x"] = radius * math.sin(point["theta"])
        point["y"] = radius * math.cos(point["theta"])

    return profile_points


def section_pair(point, vertical_fraction, thickness_fraction, body="combined"):
    plinth_top_z = max(point.get("plinth_z", point["base_z"]), point.get("ground_z", point["base_z"]))
    plinth_base_z = min(point.get("plinth_z", point["base_z"]), point.get("ground_z", point["base_z"]))

    if body == "plinth":
        z_value = plinth_base_z + vertical_fraction * (plinth_top_z - plinth_base_z)
        left_offset = -plinth_upstream_offset_m - 0.5 * wall_thickness
        right_offset = 0.5 * wall_thickness + plinth_downstream_offset_m
        effective_offset = left_offset + thickness_fraction * (right_offset - left_offset)
    elif body == "wedge":
        z_value = plinth_base_z + vertical_fraction * (plinth_top_z - plinth_base_z)
        left_offset = -plinth_upstream_offset_m - 0.5 * wall_thickness
        right_offset = 0.5 * wall_thickness + plinth_downstream_offset_m
        effective_offset = left_offset + thickness_fraction * (right_offset - left_offset)
    elif body == "wall":
        z_value = plinth_top_z + vertical_fraction * (point["crest_z"] - plinth_top_z)
        left_offset = -0.5 * wall_thickness
        right_offset = 0.5 * wall_thickness
        effective_offset = left_offset + thickness_fraction * (right_offset - left_offset)
    else:
        # A single bonded dam body is preferred here: the lower plinth block and the
        # upper wall block share the same interface nodes instead of being meshed as two
        # separate pieces. This avoids the non-conforming mesh and zero-area faces that
        # were causing the Elmer degeneracy error.
        plinth_transition_fraction = 0.5
        if vertical_fraction <= plinth_transition_fraction:
            local_fraction = vertical_fraction / max(plinth_transition_fraction, 1.0e-9)
            z_value = plinth_base_z + local_fraction * (plinth_top_z - plinth_base_z)
            left_offset = -plinth_upstream_offset_m - 0.5 * wall_thickness
            right_offset = 0.5 * wall_thickness + plinth_downstream_offset_m
        else:
            local_fraction = (vertical_fraction - plinth_transition_fraction) / max(1.0 - plinth_transition_fraction, 1.0e-9)
            z_value = plinth_top_z + local_fraction * (point["crest_z"] - plinth_top_z)
            left_offset = -0.5 * wall_thickness
            right_offset = 0.5 * wall_thickness
        effective_offset = left_offset + thickness_fraction * (right_offset - left_offset)

    x_value = point["x"] + point["nx"] * effective_offset
    y_value = point["y"] + point["ny"] * effective_offset
    return (x_value, y_value, z_value)


rows = []
seen = set()
for row in data:
    station = float(row["chainage"])
    if station in seen:
        continue
    seen.add(station)
    rows.append(row)

rows.sort(key=lambda row: float(row["chainage"]))
plinth_path = root / "Data" / "plinth.json"
if len(rows) < 3 and not plinth_path.exists():
    raise ValueError("Need at least three unique contour points to build the dam geometry")

profile_points = []
if geometry_mode == "v2":
    with plinth_path.open() as fh:
        plinth_data = json.load(fh)
    ground_by_station = {
        float(row["chainage"]): float(row.get("groundLevel", row.get("ground", 0.0)))
        for row in plinth_data
    }
    plinth_by_station = {
        float(row["chainage"]): float(row.get("plinth", row.get("plinthLevel", 0.0)))
        for row in plinth_data
    }

    if rows:
        candidate_stations = [float(row["chainage"]) for row in rows]
    else:
        candidate_stations = sorted(plinth_by_station)

    for station in candidate_stations:
        if station not in plinth_by_station:
            continue
        theta = station / radius
        ground_depth_below_crest = float(ground_by_station[station])
        plinth_depth_below_crest = float(plinth_by_station[station])
        # `plinth` is the local wall height from plinth to the crest. Keep the crest
        # at the project datum and place the wall base at that depth below it.
        plinth_elevation = -plinth_depth_below_crest
        crest_elevation = 0.0
        ground_elevation = -ground_depth_below_crest
        profile_points.append(
            {
                "station": station,
                "theta": theta,
                "x": radius * math.sin(theta),
                "y": radius * math.cos(theta),
                "base_z": plinth_elevation,
                "crest_z": crest_elevation,
                "ground_z": ground_elevation,
                "plinth_z": plinth_elevation,
                "rise_above_ground": ground_depth_below_crest - plinth_depth_below_crest,
                "wedge_offset_m": 0.0,
                "bedrock_boundary_z": ground_elevation,
            }
        )
    crest_elevation = max(point["crest_z"] for point in profile_points) if profile_points else dam_height
else:
    crest_elevation = dam_height
    for row in rows:
        station = float(row["chainage"])
        theta = station / radius
        base_z = crest_elevation - float(row["height"])
        profile_points.append(
            {
                "station": station,
                "theta": theta,
                "x": radius * math.sin(theta),
                "y": radius * math.cos(theta),
                "base_z": base_z,
                "crest_z": crest_elevation,
            }
        )

if not profile_points:
    raise ValueError(f"No compatible points available for geometry mode '{geometry_mode}'")

points = interpolate_profile_points_by_target_spacing(profile_points, target_block_size)
if len(points) <= len(profile_points):
    points = interpolate_profile_points(profile_points, arch_subdivisions)
points = enforce_true_arc(points)
points = [
    point
    for point in points
    if point["crest_z"] - point["base_z"] > 1.0e-8
]


def insert_station(points, station):
    for point in points:
        if math.isclose(point["station"], station, abs_tol=1.0e-9):
            return
    for index, start in enumerate(points[:-1]):
        end = points[index + 1]
        if start["station"] < station < end["station"]:
            fraction = (station - start["station"]) / (end["station"] - start["station"])
            inserted = {
                key: start[key] + fraction * (end[key] - start[key])
                for key in ("theta", "base_z", "crest_z", "ground_z", "plinth_z")
            }
            inserted["station"] = station
            inserted["x"] = radius * math.sin(inserted["theta"])
            inserted["y"] = radius * math.cos(inserted["theta"])
            points.insert(index + 1, inserted)
            return
    raise ValueError(f"Wedge station {station:.9f} lies outside the dam profile")


def raw_wedge_z_length(point):
    return point["crest_z"] - point["base_z"] - wedge_start_below_crest_m


def find_wedge_start_crossings(points):
    crossings = []
    for start, end in zip(points, points[1:]):
        start_delta = raw_wedge_z_length(start)
        end_delta = raw_wedge_z_length(end)
        if start_delta * end_delta < 0.0:
            fraction = -start_delta / (end_delta - start_delta)
            crossings.append(start["station"] + fraction * (end["station"] - start["station"]))
    return crossings


wedge_start_crossings = find_wedge_start_crossings(points) if wedge_enabled else []
if wedge_enabled:
    if len(wedge_start_crossings) != 2:
        raise ValueError("The plinth-derived wedge requires exactly two start-threshold intersections")
    raw_wedge_start_station_m, raw_wedge_end_station_m = wedge_start_crossings

    # Keep both wedge end caps inside the zero-height intersections, then use
    # existing wall-grid stations so the new wedge can share its Y-direction nodes.
    wedge_end_trim_m = 2.0 * target_block_size
    trimmed_start = raw_wedge_start_station_m + wedge_end_trim_m
    trimmed_end = raw_wedge_end_station_m - wedge_end_trim_m
    wedge_start_station_m = min(points, key=lambda point: abs(point["station"] - trimmed_start))["station"]
    wedge_end_station_m = min(points, key=lambda point: abs(point["station"] - trimmed_end))["station"]
    if wedge_end_station_m <= wedge_start_station_m:
        raise ValueError("The inward-trimmed wedge span must contain at least one wall-grid segment")

    insert_station(points, wedge_start_station_m)
    wedge_station = wedge_start_station_m + target_block_size
    while wedge_station < wedge_end_station_m:
        insert_station(points, wedge_station)
        wedge_station += target_block_size
    insert_station(points, wedge_end_station_m)
if len(points) < 2:
    raise ValueError("The plinth profile must contain at least two non-zero wall-height stations")

wedge_end_taper_length_m = 2.0
wedge_end_taper_station_spacing_m = 0.25
if wedge_enabled:
    for endpoint, direction in ((wedge_start_station_m, 1.0), (wedge_end_station_m, -1.0)):
        station = endpoint + direction * wedge_end_taper_station_spacing_m
        while abs(station - endpoint) < wedge_end_taper_length_m - 1.0e-9:
            insert_station(points, station)
            station += direction * wedge_end_taper_station_spacing_m


def wedge_z_length(point):
    raw_length = raw_wedge_z_length(point)
    if not wedge_enabled:
        return raw_length
    distance_to_end = min(point["station"] - wedge_start_station_m, wedge_end_station_m - point["station"])
    taper_fraction = max(0.0, min(1.0, distance_to_end / wedge_end_taper_length_m))
    return raw_length * math.sin(0.5 * math.pi * taper_fraction)

# Refine the retained 2 m wall tips only in X and Y. Each tip cell spans the
# entire local wall height in Z, avoiding thin cells at the sloping plinth.
wall_tip_refined_length_m = 4.0 * target_block_size
wall_tip_refined_cells = int(round(wall_tip_refined_length_m / local_corner_size))
if not math.isclose(wall_tip_refined_cells * local_corner_size, wall_tip_refined_length_m, abs_tol=1.0e-9):
    raise ValueError("Local Corner Refinement must divide the two-metre wall tip length exactly")
for endpoint, direction in ((points[0]["station"], 1.0), (points[-1]["station"], -1.0)):
    for cell_index in range(1, wall_tip_refined_cells + 1):
        insert_station(points, endpoint + direction * cell_index * local_corner_size)
assign_normals(points)
local_heights = [point["crest_z"] - point["base_z"] for point in points]
average_height = sum(local_heights) / len(local_heights)
maximum_wall_height_m = max(local_heights)
maximum_wedge_z_length = max(wedge_z_length(point) for point in points)
wedge_interface_layer = max(1, int(math.ceil(maximum_wedge_z_length / target_block_size)))
maximum_plinth_height_m = max(
    abs(point["base_z"] - point["ground_z"])
    for point in points
)
plinth_vertical_layers = max(1, int(math.ceil(maximum_plinth_height_m / target_block_size)))
vertical_layers = max(
    int(math.ceil(maximum_wall_height_m / target_block_size)),
    wedge_interface_layer + int(math.ceil(wedge_start_below_crest_m / target_block_size)),
)

csv_path = out_dir / "curved_dam_centerline.csv"
with csv_path.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["station", "x", "y", "z_base", "z_crest"])
    for point in points:
        writer.writerow([point["station"], point["x"], point["y"], point["base_z"], point["crest_z"]])

geo_path = out_dir / "curved_dam_geometry.geo"
mesh_path = out_dir / "curved_dam_mesh.msh"
meta_path = out_dir / "curved_dam_mesh_meta.json"
mesh_db_path = out_dir / "mesh"


def resolve_elmergrid() -> Optional[str]:
    candidate_paths = [
        shutil.which("ElmerGrid"),
        "/usr/local/bin/ElmerGrid",
        "/Users/alistairdavies/elmerfem/bin/ElmerGrid",
        "/Users/alistairdavies/Library/Application Support/Code/User/globalStorage/github.copilot-chat/debugCommand/ElmerGrid",
    ]
    for candidate in candidate_paths:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def point_on_local_section(point, local_offset_m, z_value):
    x_value = point["x"] + point["nx"] * local_offset_m
    y_value = point["y"] + point["ny"] * local_offset_m
    return (x_value, y_value, z_value)


def write_bedrock_illustration(points, mesh_size):
    bedrock_path = out_dir / "bedrock_illustration.geo"
    with bedrock_path.open("w") as fh:
        fh.write("// Temporary illustration geometry for the bedrock beneath the plinth.\n")
        fh.write("SetFactory(\"OpenCASCADE\");\n")
        fh.write(f"meshSize = {mesh_size};\n")

        upstream_point_ids = []
        downstream_point_ids = []
        point_counter = 0

        def add_point(xyz):
            nonlocal point_counter
            point_counter += 1
            fh.write(f"Point({point_counter}) = {{{xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}, meshSize}};\n")
            return point_counter

        for point in points:
            upstream_pos = point_on_local_section(point, -3.0, point["ground_z"])
            downstream_pos = point_on_local_section(point, 4.0, point["ground_z"])
            upstream_point_ids.append(add_point(upstream_pos))
            downstream_point_ids.append(add_point(downstream_pos))

        for index in range(len(upstream_point_ids) - 1):
            fh.write(f"Line({index + 1}) = {{{upstream_point_ids[index], upstream_point_ids[index + 1]}}};\n")

        for index in range(len(downstream_point_ids) - 1):
            offset = len(upstream_point_ids) - 1
            fh.write(f"Line({index + 1 + offset}) = {{{downstream_point_ids[index], downstream_point_ids[index + 1]}}};\n")

        fh.write(f"Line({len(upstream_point_ids) + len(downstream_point_ids) - 1}) = {{{upstream_point_ids[0], downstream_point_ids[0]}}};\n")
        fh.write(f"Line({len(upstream_point_ids) + len(downstream_point_ids)}) = {{{upstream_point_ids[-1], downstream_point_ids[-1]}}};\n")

    return bedrock_path


def write_rectangular_plinth_preview(points, mesh_size):
    preview_path = out_dir / "plinth_rectangle_preview.geo"
    with preview_path.open("w") as fh:
        fh.write("// Temporary section preview showing the rectangular plinth and wall footprint.\n")
        fh.write("SetFactory(\"OpenCASCADE\");\n")
        fh.write(f"meshSize = {mesh_size};\n")

        sample_point = points[0]
        plinth_top_z = max(sample_point.get("plinth_z", sample_point["base_z"]), sample_point.get("ground_z", sample_point["base_z"]))
        plinth_base_z = min(sample_point.get("plinth_z", sample_point["base_z"]), sample_point.get("ground_z", sample_point["base_z"]))
        wall_top_z = sample_point["crest_z"]

        # Rectangle representing the wall section above the plinth.
        wall_left_bottom = point_on_local_section(sample_point, -0.5 * wall_thickness, plinth_top_z)
        wall_left_top = point_on_local_section(sample_point, -0.5 * wall_thickness, wall_top_z)
        wall_right_top = point_on_local_section(sample_point, 0.5 * wall_thickness, wall_top_z)
        wall_right_bottom = point_on_local_section(sample_point, 0.5 * wall_thickness, plinth_top_z)

        # Rectangle representing the plinth section beneath the wall.
        plinth_left_bottom = point_on_local_section(sample_point, -plinth_upstream_offset_m - 0.5 * wall_thickness, plinth_base_z)
        plinth_left_top = point_on_local_section(sample_point, -plinth_upstream_offset_m - 0.5 * wall_thickness, plinth_top_z)
        plinth_right_top = point_on_local_section(sample_point, 0.5 * wall_thickness + plinth_downstream_offset_m, plinth_top_z)
        plinth_right_bottom = point_on_local_section(sample_point, 0.5 * wall_thickness + plinth_downstream_offset_m, plinth_base_z)

        points_list = [
            wall_left_bottom, wall_left_top, wall_right_top, wall_right_bottom,
            plinth_left_bottom, plinth_left_top, plinth_right_top, plinth_right_bottom,
        ]

        for index, xyz in enumerate(points_list, start=1):
            fh.write(f"Point({index}) = {{{xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}, meshSize}};\n")

    return preview_path


def convert_to_elmer_mesh(gmsh_path: Path, target_dir: Path) -> None:
    elmergrid = resolve_elmergrid()
    if not elmergrid:
        print("ElmerGrid not found on PATH or common install locations; skipping mesh DB conversion.")
        return

    if target_dir.exists():
        for child in target_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [elmergrid, "14", "2", str(gmsh_path), "-out", str(target_dir)],
        check=True,
        cwd=gmsh_path.parent,
    )
    print(f"Converted {gmsh_path} -> {target_dir}")


with geo_path.open("w") as fh:
    fh.write("// Curved dam wall generated as a swept solid with a rectangular cross-section\n")
    fh.write('SetFactory("OpenCASCADE");\n')
    fh.write(f"meshSize = {mesh_size};\n")
    fh.write(f"wallThick = {wall_thickness};\n")
    fh.write(f"wallHeightAbovePlinth = {wall_height_above_plinth_m};\n")
    fh.write(f"wallUpstreamRadius = {upstream_radius};\n")
    fh.write(f"wallCentrelineRadius = {radius};\n")
    fh.write(f"wallDownstreamRadius = {downstream_radius};\n")
    fh.write(f"damHeight = {dam_height};\n")
    fh.write(f"archSubdivisions = {arch_subdivisions};\n")
    fh.write(f"verticalLayers = {vertical_layers};\n")
    fh.write(f"thicknessLayers = {thickness_layers};\n")
    fh.write(f"crestDetailHeight = {crest_detail_height};\n")
    fh.write(f"crestExtraThickness = {crest_extra_thickness};\n")
    fh.write(f"abutmentSegmentCount = {abutment_segment_count};\n")
    fh.write(f"crestZ = {crest_elevation:.6f};\n")

    point_counter = 0
    for point in points:
        for vertical_fraction in (0.0, 1.0):
            for thickness_fraction in (0.0, 1.0):
                node = section_pair(point, vertical_fraction, thickness_fraction, body="wall")
                point_counter += 1
                fh.write(f"Point({point_counter}) = {{{node[0]:.6f}, {node[1]:.6f}, {node[2]:.6f}, meshSize}};\n")

node_list = []
station_node_ids = []
for point in points:
    level_pairs = []
    for layer_index in range(vertical_layers + 1):
        vertical_fraction = layer_index / vertical_layers
        thickness_ids = []
        for thickness_index in range(thickness_layers + 1):
            thickness_fraction = thickness_index / thickness_layers
            node = section_pair(point, vertical_fraction, thickness_fraction, body="combined")
            node_list.append(node)
            thickness_ids.append(len(node_list))
        level_pairs.append(thickness_ids)
    station_node_ids.append(level_pairs)



def generate_curved_wall_mesh(points, output_mesh: Path) -> tuple[int, int]:
    """Write a conforming MSH 2.2 mesh for the curved wall and its plinth."""
    def node_id(station_index, level_index, thickness_index):
        return (
            station_index * (vertical_layers + 1) * (thickness_layers + 1)
            + level_index * (thickness_layers + 1)
            + thickness_index
            + 1
        )

    plinth_thickness_layers = int(round(plinth_width_m / target_block_size))
    wall_start_in_plinth = int(round(plinth_upstream_offset_m / target_block_size))
    wall_end_in_plinth = wall_start_in_plinth + thickness_layers
    wedge_base_segments = max(
        1,
        int(math.ceil(maximum_wedge_z_length * wedge_tangent / target_block_size)),
    )
    downstream_plinth_segments = max(
        1,
        int(math.ceil(plinth_downstream_offset_m / target_block_size)),
    )
    plinth_active_segments = [
        points[index]["base_z"] - points[index]["ground_z"] > 1.0e-8
        and points[index + 1]["base_z"] - points[index + 1]["ground_z"] > 1.0e-8
        for index in range(len(points) - 1)
    ]
    plinth_active_stations = [False] * len(points)
    for index, active in enumerate(plinth_active_segments):
        if active:
            plinth_active_stations[index] = True
            plinth_active_stations[index + 1] = True

    wedge_active_stations = [
        wedge_enabled
        and wedge_start_station_m - 1.0e-9 <= point["station"] <= wedge_end_station_m + 1.0e-9
        and wedge_z_length(point) > 1.0e-8
        for point in points
    ]
    transition_wall_segments = [
        wedge_active_stations[index] != wedge_active_stations[index + 1]
        for index in range(len(points) - 1)
    ]

    def wedge_x_length(point):
        return wedge_z_length(point) * wedge_tangent

    def plinth_offsets():
        return [
            -0.5 * wall_thickness - plinth_upstream_offset_m + index * target_block_size
            for index in range(plinth_thickness_layers + 1)
        ]

    def ordinary_wall_z(point, level_index):
        # Use the global element-size ladder so every corresponding wall row
        # is horizontal. The last local boundary cell may be shorter at bedrock.
        global_z = point["crest_z"] - (vertical_layers - level_index) * target_block_size
        return max(point["base_z"], min(global_z, point["crest_z"]))

    def ordinary_wall_base_level(point):
        for level_index in range(vertical_layers):
            if ordinary_wall_z(point, level_index + 1) > ordinary_wall_z(point, level_index) + 1.0e-9:
                return level_index
        raise ValueError("Wall section has no positive-height element")

    def ordinary_wall_segment_base_level(start, end):
        for level_index in range(vertical_layers):
            if (
                ordinary_wall_z(start, level_index + 1) > ordinary_wall_z(start, level_index) + 1.0e-9
                and ordinary_wall_z(end, level_index + 1) > ordinary_wall_z(end, level_index) + 1.0e-9
            ):
                return level_index
        raise ValueError("Wall segment has no positive-height element")

    # The source end stations have no wall height, so the retained endpoints are
    # the first and last non-zero stations. Their first 2 m is fine in X/Y and
    # uses one full-height Z element.
    tip_start_station = points[0]["station"]
    tip_end_station = points[-1]["station"]
    tip_segment_indices = {
        index
        for index, (start, end) in enumerate(zip(points, points[1:]))
        if end["station"] <= tip_start_station + wall_tip_refined_length_m + 1.0e-9
        or start["station"] >= tip_end_station - wall_tip_refined_length_m - 1.0e-9
    }
    tip_transition_segments = {
        next(index for index, point in enumerate(points[:-1]) if math.isclose(point["station"], tip_start_station + wall_tip_refined_length_m, abs_tol=1.0e-9)),
        next(index for index, point in enumerate(points[:-1]) if math.isclose(points[index + 1]["station"], tip_end_station - wall_tip_refined_length_m, abs_tol=1.0e-9)),
    }
    tip_thickness_layers = int(round(wall_thickness / local_corner_size))
    if not math.isclose(tip_thickness_layers * local_corner_size, wall_thickness, abs_tol=1.0e-9):
        raise ValueError("Local Corner Refinement must divide the wall thickness exactly")
    tip_vertical_layers = 1

    def tip_wall_z(point, level_index):
        if level_index == 0:
            return point["base_z"]
        if level_index == tip_vertical_layers:
            return point["crest_z"]
        raise ValueError("The full-height tip mesh has exactly one Z layer")

    nodes = []
    for station_index, point in enumerate(points):
        for level_index in range(vertical_layers + 1):
            z_value = ordinary_wall_z(point, level_index)
            for thickness_index in range(thickness_layers + 1):
                offset = -0.5 * wall_thickness + wall_thickness * thickness_index / thickness_layers
                nodes.append((node_id(station_index, level_index, thickness_index), *point_on_local_section(point, offset, z_value)))

    next_node_id = len(nodes) + 1
    tip_node_ids = {}
    tip_station_indices = {
        station_index
        for segment_index in tip_segment_indices
        for station_index in (segment_index, segment_index + 1)
    }
    for station_index in sorted(tip_station_indices):
        point = points[station_index]
        for level_index in range(tip_vertical_layers + 1):
            z_value = tip_wall_z(point, level_index)
            for thickness_index in range(tip_thickness_layers + 1):
                offset = -0.5 * wall_thickness + thickness_index * local_corner_size
                if level_index == 0 and thickness_index % 5 == 0:
                    tip_node_ids[station_index, level_index, thickness_index] = node_id(
                        station_index, 0, thickness_index // 5
                    )
                    continue
                tip_node_ids[station_index, level_index, thickness_index] = next_node_id
                nodes.append((next_node_id, *point_on_local_section(point, offset, z_value)))
                next_node_id += 1
    plinth_node_ids = [None] * len(points)
    for station_index, point in enumerate(points):
        if not plinth_active_stations[station_index]:
            continue

        station_levels = []
        for vertical_index in range(plinth_vertical_layers + 1):
            fraction = vertical_index / plinth_vertical_layers
            z_value = point["ground_z"] + fraction * (point["base_z"] - point["ground_z"])
            level_ids = []
            for thickness_index, offset in enumerate(plinth_offsets()):
                if (
                    vertical_index == plinth_vertical_layers
                    and wall_start_in_plinth <= thickness_index <= wall_end_in_plinth
                ):
                    wall_level_index = (
                        0 if wedge_active_stations[station_index]
                        else ordinary_wall_base_level(point)
                    )
                    level_ids.append(node_id(
                        station_index, wall_level_index, thickness_index - wall_start_in_plinth,
                    ))
                    continue
                level_ids.append(next_node_id)
                nodes.append((next_node_id, *point_on_local_section(point, offset, z_value)))
                next_node_id += 1
            station_levels.append(level_ids)

        plinth_node_ids[station_index] = station_levels

    plinth_bottom_node_ids = [
        levels[0] if levels is not None else None
        for levels in plinth_node_ids
    ]
    plinth_top_node_ids = [
        levels[-1] if levels is not None else None
        for levels in plinth_node_ids
    ]

    wedge_outer_node_ids = [None] * len(points)
    wedge_wall_level_indices = [None] * len(points)
    wedge_transition_node_ids = [None] * len(points)
    for station_index, point in enumerate(points):
        if not wedge_active_stations[station_index]:
            continue
        # Carry the wedge's outer line through the plinth so its top node can
        # be shared exactly with both the wedge and its tetrahedral transition.
        transition_levels = []
        for vertical_index in range(plinth_vertical_layers + 1):
            fraction = vertical_index / plinth_vertical_layers
            z_value = point["ground_z"] + fraction * (point["base_z"] - point["ground_z"])
            transition_levels.append(next_node_id)
            nodes.append((
                next_node_id,
                *point_on_local_section(
                    point,
                    0.5 * wall_thickness + wedge_x_length(point),
                    z_value,
                ),
            ))
            next_node_id += 1
        wedge_transition_node_ids[station_index] = transition_levels
        wall_levels = [0]
        for level_index in range(1, wedge_interface_layer + 1):
            if ordinary_wall_z(point, level_index) > ordinary_wall_z(point, wall_levels[-1]) + 1.0e-9:
                wall_levels.append(level_index)
        if wall_levels[-1] != wedge_interface_layer:
            raise ValueError("The wedge top must be an ordinary, noncollapsed wall node")

        wedge_levels = [transition_levels[-1]]
        for level_index in wall_levels[1:-1]:
            z_value = ordinary_wall_z(point, level_index)
            fraction_to_top = (-25.0 - z_value) / (-25.0 - ordinary_wall_z(point, 0))
            wedge_levels.append(next_node_id)
            nodes.append((
                next_node_id,
                *point_on_local_section(
                    point,
                    0.5 * wall_thickness + wedge_x_length(point) * fraction_to_top,
                    z_value,
                ),
            ))
            next_node_id += 1
        wedge_levels.append(node_id(station_index, wedge_interface_layer, thickness_layers))
        wedge_outer_node_ids[station_index] = wedge_levels
        wedge_wall_level_indices[station_index] = wall_levels

    elements = []
    element_id = 1
    wedge_element_ids = []
    transition_element_ids = []
    tip_hex_element_ids = []
    tip_prism_element_ids = []
    tip_transition_element_ids = []
    wedge_plinth_hex_element_ids = []
    wedge_plinth_boundary_element_ids = []
    wedge_bottom_faces = []
    wedge_bedrock_faces = []
    wedge_wall_faces = []
    wedge_outer_faces = []
    wall_plinth_faces = []
    backed_wedge_wall_faces = set()

    def add_element(element_type, physical_id, node_ids):
        nonlocal element_id
        if element_type == 6:
            coordinates = [nodes[node_id - 1][1:] for node_id in node_ids]
            if len({tuple(round(value, 12) for value in coordinate) for coordinate in coordinates}) < 6:
                return
        elements.append((element_id, element_type, physical_id, node_ids))
        element_id += 1

    def add_prism_tetrahedra(lower_nodes, upper_nodes, transition=False):
        prism_nodes = [*lower_nodes, *upper_nodes]
        tetrahedron_ids = []
        for tetrahedron in (
            (0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5),
        ):
            add_element(4, 1, [prism_nodes[index] for index in tetrahedron])
            tetrahedron_ids.append(element_id - 1)
            if transition:
                transition_element_ids.append(element_id - 1)
        return tetrahedron_ids

    def add_tip_boundary(element_type, physical_id, node_ids):
        add_element(element_type, physical_id, node_ids)

    # The old all-axis fine-cell path is retained only as an empty compatibility
    # branch. Tip zones now use the structured wall hexes emitted below.
    for station_index in sorted(tip_segment_indices):
        start = points[station_index]
        end = points[station_index + 1]
        for level_index in range(tip_vertical_layers):
            start_lower_z = tip_wall_z(start, level_index)
            start_upper_z = tip_wall_z(start, level_index + 1)
            end_lower_z = tip_wall_z(end, level_index)
            end_upper_z = tip_wall_z(end, level_index + 1)
            if start_upper_z - start_lower_z <= 1.0e-9 or end_upper_z - end_lower_z <= 1.0e-9:
                continue
            for thickness_index in range(tip_thickness_layers):
                cell = [
                    tip_node_ids[station_index, level_index, thickness_index],
                    tip_node_ids[station_index + 1, level_index, thickness_index],
                    tip_node_ids[station_index + 1, level_index, thickness_index + 1],
                    tip_node_ids[station_index, level_index, thickness_index + 1],
                    tip_node_ids[station_index, level_index + 1, thickness_index],
                    tip_node_ids[station_index + 1, level_index + 1, thickness_index],
                    tip_node_ids[station_index + 1, level_index + 1, thickness_index + 1],
                    tip_node_ids[station_index, level_index + 1, thickness_index + 1],
                ]
                add_element(5, 1, cell)
                tip_hex_element_ids.append(element_id - 1)
                if level_index == tip_vertical_layers - 1:
                    add_tip_boundary(3, 4, [cell[4], cell[5], cell[6], cell[7]])
                if thickness_index == 0:
                    add_tip_boundary(3, 2, [cell[0], cell[1], cell[5], cell[4]])
                if thickness_index == tip_thickness_layers - 1:
                    add_tip_boundary(3, 3, [cell[3], cell[7], cell[6], cell[2]])
                if station_index == 0:
                    add_tip_boundary(3, 5, [cell[0], cell[4], cell[7], cell[3]])
                if station_index + 1 == len(points) - 1:
                    add_tip_boundary(3, 6, [cell[1], cell[2], cell[6], cell[5]])

    def triangle_has_area(triangle):
        first, second, third = (nodes[node_identifier - 1][1:] for node_identifier in triangle)
        first_edge = tuple(second[axis] - first[axis] for axis in range(3))
        second_edge = tuple(third[axis] - first[axis] for axis in range(3))
        cross_product = (
            first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
            first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
            first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
        )
        return math.sqrt(sum(component * component for component in cross_product)) > 1.0e-12

    def zipper_strip_triangles(fine_nodes, coarse_nodes):
        triangles = []
        fine_index = coarse_index = 0
        while fine_index < len(fine_nodes) - 1 or coarse_index < len(coarse_nodes) - 1:
            fine_next = (fine_index + 1) / (len(fine_nodes) - 1) if fine_index < len(fine_nodes) - 1 else math.inf
            coarse_next = (coarse_index + 1) / (len(coarse_nodes) - 1) if coarse_index < len(coarse_nodes) - 1 else math.inf
            if fine_next < coarse_next - 1.0e-9:
                triangles.append((fine_nodes[fine_index], coarse_nodes[coarse_index], fine_nodes[fine_index + 1]))
                fine_index += 1
            elif coarse_next < fine_next - 1.0e-9:
                triangles.append((fine_nodes[fine_index], coarse_nodes[coarse_index], coarse_nodes[coarse_index + 1]))
                coarse_index += 1
            else:
                triangles.extend(((fine_nodes[fine_index], coarse_nodes[coarse_index], coarse_nodes[coarse_index + 1]),
                                  (fine_nodes[fine_index], coarse_nodes[coarse_index + 1], fine_nodes[fine_index + 1])))
                fine_index += 1
                coarse_index += 1
        return [triangle for triangle in triangles if triangle_has_area(triangle)]

    def unique_wall_column(station_index, thickness_index):
        column = []
        for level_index in range(vertical_layers + 1):
            candidate = node_id(station_index, level_index, thickness_index)
            if not column or not math.isclose(nodes[candidate - 1][3], nodes[column[-1] - 1][3], abs_tol=1.0e-9):
                column.append(candidate)
        return column

    # Join each one-layer X/Y-fine tip to the regular wall with a closed zipper
    # interface. The fine face is a single element tall, while the adjoining
    # regular face may have multiple noncollapsed Z layers.
    for segment_index in sorted(tip_transition_segments):
        if segment_index == min(tip_transition_segments):
            fine_station_index, coarse_station_index = segment_index, segment_index + 1
        else:
            fine_station_index, coarse_station_index = segment_index + 1, segment_index
        fine_lower = [tip_node_ids[fine_station_index, 0, index] for index in range(tip_thickness_layers + 1)]
        fine_upper = [tip_node_ids[fine_station_index, 1, index] for index in range(tip_thickness_layers + 1)]
        coarse_columns = [unique_wall_column(coarse_station_index, index) for index in range(thickness_layers + 1)]
        surface_triangles = []
        for thickness_index in range(tip_thickness_layers):
            surface_triangles.extend(((fine_lower[thickness_index], fine_lower[thickness_index + 1], fine_upper[thickness_index + 1]),
                                      (fine_lower[thickness_index], fine_upper[thickness_index + 1], fine_upper[thickness_index])))
        for thickness_index in range(thickness_layers):
            for level_index in range(len(coarse_columns[thickness_index]) - 1):
                lower_left = coarse_columns[thickness_index][level_index]
                upper_left = coarse_columns[thickness_index][level_index + 1]
                lower_right = coarse_columns[thickness_index + 1][level_index]
                upper_right = coarse_columns[thickness_index + 1][level_index + 1]
                surface_triangles.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
        surface_triangles.extend(zipper_strip_triangles((fine_lower[0], fine_upper[0]), coarse_columns[0]))
        surface_triangles.extend(zipper_strip_triangles((fine_lower[-1], fine_upper[-1]), coarse_columns[-1]))
        surface_triangles.extend(zipper_strip_triangles(fine_lower, [column[0] for column in coarse_columns]))
        surface_triangles.extend(zipper_strip_triangles(fine_upper, [column[-1] for column in coarse_columns]))
        coordinates = [nodes[node_identifier - 1][1:] for triangle in surface_triangles for node_identifier in triangle]
        core_node_id = next_node_id
        nodes.append((core_node_id, *(sum(coordinate[axis] for coordinate in coordinates) / len(coordinates) for axis in range(3))))
        next_node_id += 1
        for triangle in surface_triangles:
            add_element(4, 1, [*triangle, core_node_id])
            tip_transition_element_ids.append(element_id - 1)

    for station_index in range(len(points) - 1):
        if station_index in tip_segment_indices or station_index in tip_transition_segments:
            continue
        for level_index in range(vertical_layers):
            for thickness_index in range(thickness_layers):
                wall_cell_node_ids = [
                    node_id(station_index, level_index, thickness_index),
                    node_id(station_index + 1, level_index, thickness_index),
                    node_id(station_index + 1, level_index, thickness_index + 1),
                    node_id(station_index, level_index, thickness_index + 1),
                    node_id(station_index, level_index + 1, thickness_index),
                    node_id(station_index + 1, level_index + 1, thickness_index),
                    node_id(station_index + 1, level_index + 1, thickness_index + 1),
                    node_id(station_index, level_index + 1, thickness_index + 1),
                ]
                start_lower_z = ordinary_wall_z(points[station_index], level_index)
                start_upper_z = ordinary_wall_z(points[station_index], level_index + 1)
                end_lower_z = ordinary_wall_z(points[station_index + 1], level_index)
                end_upper_z = ordinary_wall_z(points[station_index + 1], level_index + 1)
                if start_upper_z - start_lower_z <= 1.0e-9 or end_upper_z - end_lower_z <= 1.0e-9:
                    continue
                wedge_band_cell = (
                    thickness_index == thickness_layers - 1
                    and wedge_active_stations[station_index]
                    and wedge_active_stations[station_index + 1]
                    and level_index < wedge_interface_layer
                    and level_index in wedge_wall_level_indices[station_index]
                    and level_index + 1 in wedge_wall_level_indices[station_index]
                    and level_index in wedge_wall_level_indices[station_index + 1]
                    and level_index + 1 in wedge_wall_level_indices[station_index + 1]
                )
                if wedge_band_cell:
                    outer_lower_start = wall_cell_node_ids[3]
                    outer_lower_end = wall_cell_node_ids[2]
                    outer_upper_end = wall_cell_node_ids[6]
                    outer_upper_start = wall_cell_node_ids[7]
                    inner_lower_start = wall_cell_node_ids[0]
                    inner_lower_end = wall_cell_node_ids[1]
                    inner_upper_end = wall_cell_node_ids[5]
                    inner_upper_start = wall_cell_node_ids[4]
                    add_prism_tetrahedra(
                        (outer_lower_start, outer_lower_end, outer_upper_end),
                        (inner_lower_start, inner_lower_end, inner_upper_end),
                    )
                    add_prism_tetrahedra(
                        (outer_lower_start, outer_upper_end, outer_upper_start),
                        (inner_lower_start, inner_upper_end, inner_upper_start),
                    )
                    backed_wedge_wall_faces.update((
                        tuple(sorted((outer_lower_start, outer_lower_end, outer_upper_end))),
                        tuple(sorted((outer_lower_start, outer_upper_end, outer_upper_start))),
                    ))
                else:
                    add_element(5, 1, wall_cell_node_ids)

                if level_index == 0 and not plinth_active_segments[station_index]:
                    add_element(3, 1, [
                        node_id(station_index, level_index, thickness_index),
                        node_id(station_index, level_index, thickness_index + 1),
                        node_id(station_index + 1, level_index, thickness_index + 1),
                        node_id(station_index + 1, level_index, thickness_index),
                    ])
                if level_index == vertical_layers - 1:
                    add_element(3, 4, [
                        node_id(station_index, level_index + 1, thickness_index),
                        node_id(station_index + 1, level_index + 1, thickness_index),
                        node_id(station_index + 1, level_index + 1, thickness_index + 1),
                        node_id(station_index, level_index + 1, thickness_index + 1),
                    ])
                if thickness_index == 0:
                    add_element(3, 2, [
                        node_id(station_index, level_index, thickness_index),
                        node_id(station_index + 1, level_index, thickness_index),
                        node_id(station_index + 1, level_index + 1, thickness_index),
                        node_id(station_index, level_index + 1, thickness_index),
                    ])
                if thickness_index == thickness_layers - 1 and not (
                    wedge_active_stations[station_index]
                    and wedge_active_stations[station_index + 1]
                    and level_index < wedge_interface_layer
                ):
                    add_element(3, 3, [
                        node_id(station_index, level_index, thickness_index + 1),
                        node_id(station_index, level_index + 1, thickness_index + 1),
                        node_id(station_index + 1, level_index + 1, thickness_index + 1),
                        node_id(station_index + 1, level_index, thickness_index + 1),
                    ])

    def add_wall_plinth_transition(upper_faces, lower_quad):
        """Fill a wall-to-plinth transition volume with bonded pyramids."""
        nonlocal next_node_id
        coordinates = [nodes[node_identifier - 1][1:] for face in upper_faces for node_identifier in face]
        coordinates.extend(nodes[node_identifier - 1][1:] for node_identifier in lower_quad)
        core_node_id = next_node_id
        nodes.append((core_node_id, *(sum(coordinate[axis] for coordinate in coordinates) / len(coordinates) for axis in range(3))))
        next_node_id += 1
        for face in upper_faces:
            add_element(4 if len(face) == 3 else 7, 1, [*face, core_node_id])
            wall_plinth_faces.append(face)
        add_element(7, 1, [*lower_quad, core_node_id])

        perimeter = []
        for face in upper_faces:
            perimeter.extend(zip(face, (*face[1:], face[0])))
        boundary_edges = [edge for edge, count in Counter(tuple(sorted(edge)) for edge in perimeter).items() if count == 1]
        lower_edges = ((lower_quad[0], lower_quad[1]), (lower_quad[1], lower_quad[2]),
                       (lower_quad[2], lower_quad[3]), (lower_quad[3], lower_quad[0]))
        for upper_edge, lower_edge in zip(boundary_edges, lower_edges):
            add_element(4, 1, [upper_edge[0], upper_edge[1], lower_edge[1], core_node_id])
            add_element(4, 1, [upper_edge[0], lower_edge[1], lower_edge[0], core_node_id])

    def add_tip_plinth_transition(lower_start, lower_end, station_index, thickness_index):
        """Fill one coarse plinth-top cell below five fine wall-base quads."""
        fine_offset = (thickness_index - wall_start_in_plinth) * 5
        fine_start = [tip_node_ids[station_index, 0, fine_offset + index] for index in range(6)]
        fine_end = [tip_node_ids[station_index + 1, 0, fine_offset + index] for index in range(6)]
        upper_faces = [
            (fine_start[index], fine_end[index], fine_end[index + 1], fine_start[index + 1])
            for index in range(5)
        ]
        lower_quad = (
            lower_start[thickness_index], lower_end[thickness_index],
            lower_end[thickness_index + 1], lower_start[thickness_index + 1],
        )
        add_wall_plinth_transition(upper_faces, lower_quad)

    for station_index, active in enumerate(plinth_active_segments):
        if not active:
            continue

        start_levels = plinth_node_ids[station_index]
        end_levels = plinth_node_ids[station_index + 1]
        plinth_segment_count = min(len(start_levels[0]), len(end_levels[0])) - 1
        transition_segment = (
            wedge_active_stations[station_index]
            and wedge_active_stations[station_index + 1]
        )
        for vertical_index in range(plinth_vertical_layers):
            lower_start = start_levels[vertical_index]
            lower_end = end_levels[vertical_index]
            upper_start = start_levels[vertical_index + 1]
            upper_end = end_levels[vertical_index + 1]
            if transition_segment:
                # The wedge only intersects the downstream wall cell. Retain
                # the regular hexahedral plinth through the inner footprint.
                for thickness_index in range(wall_end_in_plinth - 1):
                    plinth_cell_node_ids = [
                        lower_start[thickness_index], lower_end[thickness_index],
                        lower_end[thickness_index + 1], lower_start[thickness_index + 1],
                        upper_start[thickness_index], upper_end[thickness_index],
                        upper_end[thickness_index + 1], upper_start[thickness_index + 1],
                    ]
                    add_element(5, 1, plinth_cell_node_ids)
                    wedge_plinth_hex_element_ids.append(element_id - 1)
                    if vertical_index == 0:
                        add_element(3, 1, [
                            lower_start[thickness_index], lower_start[thickness_index + 1],
                            lower_end[thickness_index + 1], lower_end[thickness_index],
                        ])
                    if thickness_index == 0:
                        add_element(3, 7, [
                            lower_start[thickness_index], lower_end[thickness_index],
                            upper_end[thickness_index], upper_start[thickness_index],
                        ])
                    if vertical_index == plinth_vertical_layers - 1 and thickness_index < wall_start_in_plinth:
                        add_element(3, 7, [
                            upper_start[thickness_index], upper_end[thickness_index],
                            upper_end[thickness_index + 1], upper_start[thickness_index + 1],
                        ])
                lower_outer_start = wedge_transition_node_ids[station_index][vertical_index]
                lower_outer_end = wedge_transition_node_ids[station_index + 1][vertical_index]
                upper_outer_start = wedge_transition_node_ids[station_index][vertical_index + 1]
                upper_outer_end = wedge_transition_node_ids[station_index + 1][vertical_index + 1]
                triangles = []
                for thickness_index in range(wall_end_in_plinth - 1, wall_end_in_plinth):
                    triangles.extend((
                        (
                            (lower_start[thickness_index], lower_end[thickness_index], lower_end[thickness_index + 1]),
                            (upper_start[thickness_index], upper_end[thickness_index], upper_end[thickness_index + 1]),
                        ),
                        (
                            (lower_start[thickness_index], lower_end[thickness_index + 1], lower_start[thickness_index + 1]),
                            (upper_start[thickness_index], upper_end[thickness_index + 1], upper_start[thickness_index + 1]),
                        ),
                    ))
                triangles.extend((
                    (
                        (lower_start[wall_end_in_plinth], lower_outer_start, lower_outer_end),
                        (upper_start[wall_end_in_plinth], upper_outer_start, upper_outer_end),
                    ),
                    (
                        (lower_start[wall_end_in_plinth], lower_outer_end, lower_end[wall_end_in_plinth]),
                        (upper_start[wall_end_in_plinth], upper_outer_end, upper_end[wall_end_in_plinth]),
                    ),
                    (
                        (lower_outer_start, lower_start[-1], lower_end[-1]),
                        (upper_outer_start, upper_start[-1], upper_end[-1]),
                    ),
                    (
                        (lower_outer_start, lower_end[-1], lower_outer_end),
                        (upper_outer_start, upper_end[-1], upper_outer_end),
                    ),
                ))
                for lower_triangle, upper_triangle in triangles:
                    coordinates = [nodes[node_identifier - 1][1:] for node_identifier in upper_triangle]
                    area_measure = math.dist(coordinates[0], coordinates[1]) * math.dist(coordinates[0], coordinates[2])
                    if area_measure <= 1.0e-12:
                        continue
                    wedge_plinth_boundary_element_ids.extend(
                        add_prism_tetrahedra(lower_triangle, upper_triangle, transition=True)
                    )
            else:
                for thickness_index in range(plinth_segment_count):
                    if (
                        vertical_index == plinth_vertical_layers - 1
                        and station_index in tip_segment_indices
                        and wall_start_in_plinth <= thickness_index < wall_end_in_plinth
                    ):
                        add_tip_plinth_transition(lower_start, lower_end, station_index, thickness_index)
                        continue
                    if (
                        vertical_index == plinth_vertical_layers - 1
                        and station_index in tip_transition_segments
                        and wall_start_in_plinth <= thickness_index < wall_end_in_plinth
                    ):
                        if station_index == min(tip_transition_segments):
                            fine_station_index, coarse_station_index = station_index, station_index + 1
                        else:
                            fine_station_index, coarse_station_index = station_index + 1, station_index
                        fine_offset = (thickness_index - wall_start_in_plinth) * 5
                        upper_faces = zipper_strip_triangles(
                            [tip_node_ids[fine_station_index, 0, fine_offset + index] for index in range(6)],
                            [
                                unique_wall_column(coarse_station_index, thickness_index - wall_start_in_plinth)[0],
                                unique_wall_column(coarse_station_index, thickness_index + 1 - wall_start_in_plinth)[0],
                            ],
                        )
                        add_wall_plinth_transition(
                            upper_faces,
                            (
                                lower_start[thickness_index], lower_end[thickness_index],
                                lower_end[thickness_index + 1], lower_start[thickness_index + 1],
                            ),
                        )
                        continue
                    if vertical_index == plinth_vertical_layers - 1 and wall_start_in_plinth <= thickness_index < wall_end_in_plinth:
                        wall_level_index = ordinary_wall_segment_base_level(
                            points[station_index], points[station_index + 1],
                        )
                        upper_quad = (
                            node_id(station_index, wall_level_index, thickness_index - wall_start_in_plinth),
                            node_id(station_index + 1, wall_level_index, thickness_index - wall_start_in_plinth),
                            node_id(station_index + 1, wall_level_index, thickness_index + 1 - wall_start_in_plinth),
                            node_id(station_index, wall_level_index, thickness_index + 1 - wall_start_in_plinth),
                        )
                        add_wall_plinth_transition(
                            [upper_quad],
                            (
                                lower_start[thickness_index], lower_end[thickness_index],
                                lower_end[thickness_index + 1], lower_start[thickness_index + 1],
                            ),
                        )
                        continue
                    plinth_cell_node_ids = [
                        lower_start[thickness_index], lower_end[thickness_index],
                        lower_end[thickness_index + 1], lower_start[thickness_index + 1],
                        upper_start[thickness_index], upper_end[thickness_index],
                        upper_end[thickness_index + 1], upper_start[thickness_index + 1],
                    ]
                    add_element(5, 1, plinth_cell_node_ids)
                    if vertical_index == 0:
                        add_element(3, 1, [
                            lower_start[thickness_index], lower_start[thickness_index + 1],
                            lower_end[thickness_index + 1], lower_end[thickness_index],
                        ])
                    if thickness_index == 0:
                        add_element(3, 7, [
                            lower_start[thickness_index], lower_end[thickness_index],
                            upper_end[thickness_index], upper_start[thickness_index],
                        ])
                    if thickness_index == plinth_segment_count - 1:
                        add_element(3, 7, [
                            lower_start[thickness_index + 1], upper_start[thickness_index + 1],
                            upper_end[thickness_index + 1], lower_end[thickness_index + 1],
                        ])
                    if vertical_index == plinth_vertical_layers - 1:
                        wall_face = wall_start_in_plinth <= thickness_index < wall_end_in_plinth
                        if not wall_face:
                            add_element(3, 7, [
                                upper_start[thickness_index], upper_end[thickness_index],
                                upper_end[thickness_index + 1], upper_start[thickness_index + 1],
                            ])
                if vertical_index == 0:
                    if transition_segment:
                        for lower_triangle, _ in triangles:
                            add_element(2, 1, list(lower_triangle))
                if vertical_index == plinth_vertical_layers - 1 and transition_segment:
                    for _, upper_triangle in triangles[2:]:
                        add_element(2, 7, list(upper_triangle))

        if station_index == 0 or not plinth_active_segments[station_index - 1]:
            for vertical_index in range(plinth_vertical_layers):
                lower_start = start_levels[vertical_index]
                upper_start = start_levels[vertical_index + 1]
                for thickness_index in range(plinth_segment_count):
                    add_element(3, 7, [
                        lower_start[thickness_index], upper_start[thickness_index],
                        upper_start[thickness_index + 1], lower_start[thickness_index + 1],
                    ])
        if station_index == len(plinth_active_segments) - 1 or not plinth_active_segments[station_index + 1]:
            for vertical_index in range(plinth_vertical_layers):
                lower_end = end_levels[vertical_index]
                upper_end = end_levels[vertical_index + 1]
                for thickness_index in range(plinth_segment_count):
                    add_element(3, 7, [
                        lower_end[thickness_index], lower_end[thickness_index + 1],
                        upper_end[thickness_index + 1], upper_end[thickness_index],
                    ])

    def zipper_triangles(start_nodes, end_nodes):
        triangles = []
        start_index = end_index = 0
        while start_index < len(start_nodes) - 1 or end_index < len(end_nodes) - 1:
            start_next_z = nodes[start_nodes[start_index + 1] - 1][3] if start_index < len(start_nodes) - 1 else math.inf
            end_next_z = nodes[end_nodes[end_index + 1] - 1][3] if end_index < len(end_nodes) - 1 else math.inf
            if start_next_z < end_next_z - 1.0e-9:
                triangles.append((start_nodes[start_index], end_nodes[end_index], start_nodes[start_index + 1]))
                start_index += 1
            elif end_next_z < start_next_z - 1.0e-9:
                triangles.append((start_nodes[start_index], end_nodes[end_index], end_nodes[end_index + 1]))
                end_index += 1
            else:
                triangles.extend(((start_nodes[start_index], end_nodes[end_index], end_nodes[end_index + 1]),
                                  (start_nodes[start_index], end_nodes[end_index + 1], start_nodes[start_index + 1])))
                start_index += 1
                end_index += 1
        return triangles

    for station_index in range(len(points) - 1):
        if not (wedge_active_stations[station_index] and wedge_active_stations[station_index + 1]):
            continue
        start_wall = [node_id(station_index, level_index, thickness_layers) for level_index in wedge_wall_level_indices[station_index]]
        end_wall = [node_id(station_index + 1, level_index, thickness_layers) for level_index in wedge_wall_level_indices[station_index + 1]]
        start_outer = wedge_outer_node_ids[station_index]
        end_outer = wedge_outer_node_ids[station_index + 1]
        wall_triangles = zipper_triangles(start_wall, end_wall)
        outer_triangles = zipper_triangles(start_outer, end_outer)
        bottom_triangles = (
            (start_wall[0], start_outer[0], end_outer[0]),
            (start_wall[0], end_outer[0], end_wall[0]),
        )
        boundary_triangles = [*wall_triangles, *outer_triangles, *bottom_triangles]
        boundary_coordinates = [nodes[node_identifier - 1][1:] for triangle in boundary_triangles for node_identifier in triangle]
        core_node_id = next_node_id
        nodes.append((core_node_id, *(sum(coordinate[axis] for coordinate in boundary_coordinates) / len(boundary_coordinates) for axis in range(3))))
        next_node_id += 1
        for triangle in boundary_triangles:
            add_element(4, 1, [*triangle, core_node_id])
            wedge_element_ids.append(element_id - 1)

        wedge_wall_faces.extend(wall_triangles)
        for wall_face in wall_triangles:
            if tuple(sorted(wall_face)) in backed_wedge_wall_faces:
                continue
            for thickness_index in range(thickness_layers):
                add_prism_tetrahedra(
                    tuple(node_identifier - thickness_index for node_identifier in wall_face),
                    tuple(node_identifier - thickness_index - 1 for node_identifier in wall_face),
                )
        wedge_bottom_faces.extend(bottom_triangles)
        if not plinth_active_segments[station_index]:
            wedge_bedrock_faces.extend(bottom_triangles)
            for face in bottom_triangles:
                add_element(2, 1, list(face))
        for face in outer_triangles:
            add_element(2, 3, list(face))
            wedge_outer_faces.append(face)

    for station_index, wedge_is_active in enumerate(wedge_active_stations):
        starts_wedge = wedge_is_active and (station_index == 0 or not wedge_active_stations[station_index - 1])
        ends_wedge = wedge_is_active and (
            station_index == len(wedge_active_stations) - 1 or not wedge_active_stations[station_index + 1]
        )
        if not (starts_wedge or ends_wedge):
            continue
        base = node_id(station_index, 0, thickness_layers)
        top = node_id(station_index, wedge_interface_layer, thickness_layers)
        outer = wedge_outer_node_ids[station_index][0]
        add_element(2, 7, [base, outer, top])

    for level_index in range(vertical_layers):
        for thickness_index in range(thickness_layers):
            if 0 in tip_station_indices or len(points) - 1 in tip_station_indices:
                continue
            add_element(3, 5, [
                node_id(0, level_index, thickness_index),
                node_id(0, level_index + 1, thickness_index),
                node_id(0, level_index + 1, thickness_index + 1),
                node_id(0, level_index, thickness_index + 1),
            ])
            add_element(3, 6, [
                node_id(len(points) - 1, level_index, thickness_index),
                node_id(len(points) - 1, level_index, thickness_index + 1),
                node_id(len(points) - 1, level_index + 1, thickness_index + 1),
                node_id(len(points) - 1, level_index + 1, thickness_index),
            ])

    volume_face_templates = {
        4: ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
        5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
        7: ((0, 1, 2, 3), (0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0)),
    }
    volume_face_incidence = {}
    for _, element_type, _, cell_node_ids in elements:
        for face in volume_face_templates.get(element_type, ()):
            face_node_ids = tuple(sorted(cell_node_ids[index] for index in face))
            volume_face_incidence[face_node_ids] = volume_face_incidence.get(face_node_ids, 0) + 1
    explicit_boundary_faces = {
        tuple(sorted(cell_node_ids))
        for _, element_type, _, cell_node_ids in elements
        if element_type in (2, 3)
    }

    def transition_boundary_id(face_node_ids):
        coordinates = [nodes[node_identifier - 1][1:] for node_identifier in face_node_ids]
        radii = [math.hypot(coordinate[0], coordinate[1]) for coordinate in coordinates]
        if all(math.isclose(radius_value, upstream_radius, abs_tol=1.0e-7) for radius_value in radii):
            return 2
        if all(math.isclose(radius_value, downstream_radius, abs_tol=1.0e-7) for radius_value in radii):
            return 3
        if all(math.isclose(coordinate[2], crest_elevation, abs_tol=1.0e-7) for coordinate in coordinates):
            return 4
        return 1

    # The sloping tip bases can leave triangular perimeter facets where the local
    # 0.1 m ladder clamps before its 0.5 m neighbour. Tag every exposed transition
    # face explicitly so it receives either its face load or the fixed bedrock BC.
    for transition_element_id in tip_transition_element_ids:
        _, element_type, _, cell_node_ids = elements[transition_element_id - 1]
        for face in volume_face_templates[element_type]:
            face_node_ids = tuple(sorted(cell_node_ids[index] for index in face))
            if volume_face_incidence[face_node_ids] != 1 or face_node_ids in explicit_boundary_faces:
                continue
            add_element(2 if len(face_node_ids) == 3 else 3, transition_boundary_id(face_node_ids), list(face_node_ids))
            explicit_boundary_faces.add(face_node_ids)


    used_node_ids = {
        node_identifier
        for _, _, _, node_ids in elements
        for node_identifier in node_ids
    }
    node_id_map = {
        node_identifier: new_identifier
        for new_identifier, node_identifier in enumerate(sorted(used_node_ids), start=1)
    }
    nodes = [
        (node_id_map[node_identifier], x_value, y_value, z_value)
        for node_identifier, x_value, y_value, z_value in nodes
        if node_identifier in used_node_ids
    ]
    elements = [
        (identifier, element_type, physical_id, [node_id_map[node_identifier] for node_identifier in node_ids])
        for identifier, element_type, physical_id, node_ids in elements
    ]
    topology_markers = {
        "wedge_element_ids": wedge_element_ids,
        "transition_element_ids": transition_element_ids,
        "wedge_bottom_faces": [
            [node_id_map[node_identifier] for node_identifier in face]
            for face in wedge_bottom_faces
        ],
        "wedge_bedrock_faces": [
            [node_id_map[node_identifier] for node_identifier in face]
            for face in wedge_bedrock_faces
        ],
        "wedge_wall_faces": [
            [node_id_map[node_identifier] for node_identifier in face]
            for face in wedge_wall_faces
        ],
        "wedge_outer_faces": [
            [node_id_map[node_identifier] for node_identifier in face]
            for face in wedge_outer_faces
        ],
        "wall_plinth_faces": [
            [node_id_map[node_identifier] for node_identifier in face]
            for face in wall_plinth_faces
        ],
        "ordinary_wall_ladder": [
            {
                "node_id": node_id_map[node_id(station_index, level_index, thickness_layers)],
                "expected_z": ordinary_wall_z(point, level_index),
            }
            for station_index, point in enumerate(points)
            for level_index in range(vertical_layers + 1)
            if node_id(station_index, level_index, thickness_layers) in node_id_map
        ],
        "tip_hex_element_ids": tip_hex_element_ids,
        "tip_prism_element_ids": tip_prism_element_ids,
        "tip_transition_element_ids": tip_transition_element_ids,
        "wedge_plinth_hex_element_ids": wedge_plinth_hex_element_ids,
        "wedge_plinth_boundary_element_ids": wedge_plinth_boundary_element_ids,
    }

    with output_mesh.open("w") as fh:
        fh.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")
        fh.write(f"{len(nodes)}\n")
        for identifier, x_value, y_value, z_value in nodes:
            fh.write(f"{identifier} {x_value:.12g} {y_value:.12g} {z_value:.12g}\n")
        fh.write("$EndNodes\n$Elements\n")
        fh.write(f"{len(elements)}\n")
        for identifier, element_type, physical_id, node_ids in elements:
            fh.write(f"{identifier} {element_type} 2 {physical_id} {physical_id} {' '.join(map(str, node_ids))}\n")
        fh.write("$EndElements\n")

    volume_element_count = sum(1 for _, element_type, _, _ in elements if element_type in (4, 5, 6))
    return volume_element_count, len(elements) - volume_element_count, topology_markers


volume_count, boundary_count, topology_markers = generate_curved_wall_mesh(points, mesh_path)

meta_path.write_text(
    json.dumps(
        {
            "station_count": len(points),
            "vertical_layers": vertical_layers,
            "arch_subdivisions": arch_subdivisions,
            "wall_thickness_m": wall_thickness,
            "wall_height_above_plinth_m": wall_height_above_plinth_m,
            "wall_upstream_radius_m": upstream_radius,
            "wall_centerline_radius_m": radius,
            "wall_downstream_radius_m": downstream_radius,
            "wedge_enabled": wedge_enabled,
            "wedge_anchor_radius_m": wedge_anchor_radius,
            "wedge_start_below_crest_m": wedge_start_below_crest_m,
            "wedge_ratio_horizontal_m": wedge_ratio_horizontal_m,
            "wedge_ratio_vertical_m": wedge_ratio_vertical_m,
            "wedge_max_sloping_length_m": math.hypot(wedge_ratio_horizontal_m, wedge_ratio_vertical_m),
            "wedge_angle_from_vertical_deg": wedge_angle_from_vertical_deg,
            "wedge_element_size_m": target_block_size,
            "wedge_start_station_m": wedge_start_station_m,
            "wedge_end_station_m": wedge_end_station_m,
            "wedge_end_taper_length_m": wedge_end_taper_length_m,
            "wedge_end_taper_station_spacing_m": wedge_end_taper_station_spacing_m,
            "wedge_transition_station_boundaries_m": [
                wedge_start_station_m,
                wedge_end_station_m,
            ],
            "plinth_upstream_offset_m": plinth_upstream_offset_m,
            "plinth_downstream_offset_m": plinth_downstream_offset_m,
            "plinth_width_m": plinth_width_m,
            "plinth_base_width_m": plinth_base_width_m,
            "plinth_vertical_layers": plinth_vertical_layers,
            "dam_height_m": dam_height,
            "mesh_size_m": mesh_size,
            "target_block_size_m": target_block_size,
            "local_corner_size_m": local_corner_size,
            "wall_tip_refined_length_m": wall_tip_refined_length_m,
            "wall_tip_cells_across_x": int(round(wall_thickness / local_corner_size)),
            "wall_tip_cells_vertical_z": 1,
            "wall_tip_retained_station_ranges_m": [
                [points[0]["station"], points[0]["station"] + wall_tip_refined_length_m],
                [points[-1]["station"] - wall_tip_refined_length_m, points[-1]["station"]],
            ],
            "maximum_bulk_hex_edge_m": target_block_size,
            "tip_hex_target_size_m": local_corner_size,
            "thickness_layers": thickness_layers,
            "crest_detail_height_m": crest_detail_height,
            "crest_extra_thickness_m": crest_extra_thickness,
            "abutment_segment_count": abutment_segment_count,
            "geometry_mode": geometry_mode,
            "topology_markers": topology_markers,
        },
        indent=2,
    )
)

bedrock_illustration_path = write_bedrock_illustration(points, mesh_size)
plinth_preview_path = write_rectangular_plinth_preview(points, mesh_size)
convert_to_elmer_mesh(mesh_path, mesh_db_path)

print(f"Wrote {csv_path}")
print(f"Wrote {geo_path}")
print(f"Wrote {bedrock_illustration_path}")
print(f"Wrote {plinth_preview_path}")
print(f"Wrote {mesh_path}")
print(f"Wrote {meta_path}")
print(f"Generated {len(points)} stations, {volume_count} volume elements, and {boundary_count} boundary faces")
