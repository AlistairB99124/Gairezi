from pathlib import Path
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

radius = float(config.get("wall_radius_m", 80.0))
wall_thickness = float(config.get("wall_thickness_m", 4.0))
wall_height_above_plinth_m = float(config.get("wall_height_above_plinth_m", 29.0))
# This revision is plinth-only. The wedge has been intentionally disabled to avoid the
# geometry ambiguity that was persisting in the solver output.
wedge_angle_deg = float(config.get("wedge_angle_deg", 60.0))
wedge_angle_rad = math.radians(wedge_angle_deg)
wedge_start_below_crest_m = float(config.get("wedge_start_below_crest_m", 25.54))
wedge_run_m = 0.0
wedge_vertical_height_m = 3.46
wedge_angle_from_wall_deg = float(config.get("wedge_angle_from_vertical_deg", 30.0))
wedge_angle_from_wall_rad = math.radians(wedge_angle_from_wall_deg)
wedge_angle_from_horizontal_deg = 0.0
wedge_angle_from_horizontal_rad = math.radians(wedge_angle_from_horizontal_deg)
plinth_upstream_offset_m = 1.0
plinth_downstream_offset_m = 2.0
plinth_base_width_m = wall_thickness + plinth_upstream_offset_m + plinth_downstream_offset_m
plinth_width_m = plinth_base_width_m

dam_height = 30.0
mesh_size = get_grid_control_value("Global Element Size")
target_block_size = mesh_size
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
    for index, point in enumerate(profile_points):
        if index == 0:
            tx = profile_points[1]["x"] - profile_points[0]["x"]
            ty = profile_points[1]["y"] - profile_points[0]["y"]
        elif index == len(profile_points) - 1:
            tx = profile_points[-1]["x"] - profile_points[-2]["x"]
            ty = profile_points[-1]["y"] - profile_points[-2]["y"]
        else:
            tx = profile_points[index + 1]["x"] - profile_points[index - 1]["x"]
            ty = profile_points[index + 1]["y"] - profile_points[index - 1]["y"]
        length = math.hypot(tx, ty)
        if length < 1.0e-8:
            point["nx"] = 1.0
            point["ny"] = 0.0
            continue

        # Define the local wall-normal from the arch tangent. Positive local offsets are the
        # project downstream direction, placing the wedge on the downstream wall face.
        point["nx"] = ty / length
        point["ny"] = -tx / length


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
if len(points) < 2:
    raise ValueError("The plinth profile must contain at least two non-zero wall-height stations")
assign_normals(points)
local_heights = [point["crest_z"] - point["base_z"] for point in points]
average_height = sum(local_heights) / len(local_heights)
junction_min_element_size_m = 0.1
junction_transition_distance_m = 4.0
main_body_element_size_m = 3.0
junction_layer_distances_m = [0.0, 0.1, 0.3, 0.7, 1.5, 2.7, 4.0]
maximum_wall_height_m = max(local_heights)
main_body_layer_count = max(
    1,
    int(math.ceil((maximum_wall_height_m - junction_transition_distance_m) / main_body_element_size_m)),
)
vertical_layers = len(junction_layer_distances_m) - 1 + main_body_layer_count

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
    fh.write(f"wallRadius = {radius};\n")
    fh.write(f"damHeight = {dam_height};\n")
    fh.write(f"archSubdivisions = {arch_subdivisions};\n")
    fh.write(f"verticalLayers = {vertical_layers};\n")
    fh.write(f"junctionMinElementSize = {junction_min_element_size_m};\n")
    fh.write(f"junctionTransitionDistance = {junction_transition_distance_m};\n")
    fh.write(f"mainBodyElementSize = {main_body_element_size_m};\n")
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

    def junction_distance(local_transition_height, level_index):
        target_distance = junction_layer_distances_m[level_index]
        if level_index == 0:
            return 0.0
        if level_index == 1:
            return min(junction_min_element_size_m, local_transition_height)
        return junction_min_element_size_m + (
            local_transition_height - junction_min_element_size_m
        ) * (target_distance - junction_min_element_size_m) / (
            junction_transition_distance_m - junction_min_element_size_m
        )

    plinth_thickness_layers = int(round(plinth_width_m / target_block_size))
    wall_start_in_plinth = int(round(plinth_upstream_offset_m / target_block_size))
    wall_end_in_plinth = wall_start_in_plinth + thickness_layers
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

    wedge_interface_layer = len(junction_layer_distances_m) - 1
    wedge_active_stations = [point["base_z"] <= -wedge_start_below_crest_m for point in points]

    nodes = []
    for station_index, point in enumerate(points):
        for level_index in range(vertical_layers + 1):
            if wedge_active_stations[station_index] and level_index <= wedge_interface_layer:
                local_transition_height = -wedge_start_below_crest_m - point["base_z"]
                z_value = point["base_z"] + junction_distance(local_transition_height, level_index)
            elif wedge_active_stations[station_index]:
                z_value = -wedge_start_below_crest_m + (level_index - wedge_interface_layer) / (vertical_layers - wedge_interface_layer) * wedge_start_below_crest_m
            else:
                transition_fraction = junction_transition_distance_m / maximum_wall_height_m
                if level_index <= wedge_interface_layer:
                    fraction = transition_fraction * junction_layer_distances_m[level_index] / junction_transition_distance_m
                else:
                    fraction = transition_fraction + (1.0 - transition_fraction) * (level_index - wedge_interface_layer) / (vertical_layers - wedge_interface_layer)
                z_value = point["base_z"] + fraction * (point["crest_z"] - point["base_z"])
            for thickness_index in range(thickness_layers + 1):
                offset = -0.5 * wall_thickness + wall_thickness * thickness_index / thickness_layers
                nodes.append((node_id(station_index, level_index, thickness_index), *point_on_local_section(point, offset, z_value)))

    next_node_id = len(nodes) + 1
    plinth_bottom_node_ids = [None] * len(points)
    plinth_top_node_ids = [None] * len(points)
    for station_index, point in enumerate(points):
        if not plinth_active_stations[station_index]:
            continue

        bottom_ids = []
        top_ids = []
        for thickness_index in range(plinth_thickness_layers + 1):
            offset = -0.5 * wall_thickness - plinth_upstream_offset_m + thickness_index * target_block_size
            bottom_ids.append(next_node_id)
            nodes.append((next_node_id, *point_on_local_section(point, offset, point["ground_z"])))
            next_node_id += 1

            if wall_start_in_plinth <= thickness_index <= wall_end_in_plinth:
                top_ids.append(node_id(station_index, 0, thickness_index - wall_start_in_plinth))
            else:
                top_ids.append(next_node_id)
                nodes.append((next_node_id, *point_on_local_section(point, offset, point["base_z"])))
                next_node_id += 1

        plinth_bottom_node_ids[station_index] = bottom_ids
        plinth_top_node_ids[station_index] = top_ids

    wedge_toe_node_ids = [None] * len(points)
    wedge_toe_bottom_node_ids = [None] * len(points)
    wedge_slope_node_ids = [[None] * (wedge_interface_layer + 1) for _ in points]
    for station_index, point in enumerate(points):
        if not wedge_active_stations[station_index]:
            continue
        wedge_height = -wedge_start_below_crest_m - point["base_z"]
        wedge_run = wedge_height / math.tan(wedge_angle_rad)
        wedge_toe_node_ids[station_index] = next_node_id
        nodes.append((
            next_node_id,
            *point_on_local_section(point, 0.5 * wall_thickness + wedge_run, point["base_z"]),
        ))
        next_node_id += 1
        wedge_toe_bottom_node_ids[station_index] = next_node_id
        nodes.append((
            next_node_id,
            *point_on_local_section(point, 0.5 * wall_thickness + wedge_run, point["ground_z"]),
        ))
        next_node_id += 1
        wedge_slope_node_ids[station_index][0] = wedge_toe_node_ids[station_index]
        wedge_slope_node_ids[station_index][-1] = node_id(station_index, wedge_interface_layer, thickness_layers)
        for layer_index in range(1, wedge_interface_layer):
            fraction = layer_index / wedge_interface_layer
            wedge_slope_node_ids[station_index][layer_index] = next_node_id
            nodes.append((
                next_node_id,
                *point_on_local_section(
                    point,
                    0.5 * wall_thickness + (1.0 - fraction) * wedge_run,
                    point["base_z"] + fraction * (-wedge_start_below_crest_m - point["base_z"]),
                ),
            ))
            next_node_id += 1

    elements = []
    element_id = 1

    def add_element(element_type, physical_id, node_ids):
        nonlocal element_id
        elements.append((element_id, element_type, physical_id, node_ids))
        element_id += 1

    for station_index in range(len(points) - 1):
        for level_index in range(vertical_layers):
            for thickness_index in range(thickness_layers):
                add_element(5, 1, [
                    node_id(station_index, level_index, thickness_index),
                    node_id(station_index + 1, level_index, thickness_index),
                    node_id(station_index + 1, level_index, thickness_index + 1),
                    node_id(station_index, level_index, thickness_index + 1),
                    node_id(station_index, level_index + 1, thickness_index),
                    node_id(station_index + 1, level_index + 1, thickness_index),
                    node_id(station_index + 1, level_index + 1, thickness_index + 1),
                    node_id(station_index, level_index + 1, thickness_index + 1),
                ])

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

    for station_index, active in enumerate(plinth_active_segments):
        if not active:
            continue

        bottom_start = plinth_bottom_node_ids[station_index]
        bottom_end = plinth_bottom_node_ids[station_index + 1]
        top_start = plinth_top_node_ids[station_index]
        top_end = plinth_top_node_ids[station_index + 1]
        for thickness_index in range(wall_end_in_plinth):
            add_element(5, 1, [
                bottom_start[thickness_index],
                bottom_end[thickness_index],
                bottom_end[thickness_index + 1],
                bottom_start[thickness_index + 1],
                top_start[thickness_index],
                top_end[thickness_index],
                top_end[thickness_index + 1],
                top_start[thickness_index + 1],
            ])
            add_element(3, 1, [
                bottom_start[thickness_index],
                bottom_start[thickness_index + 1],
                bottom_end[thickness_index + 1],
                bottom_end[thickness_index],
            ])
            if thickness_index in (0, plinth_thickness_layers - 1):
                add_element(3, 7, [
                    bottom_start[thickness_index],
                    bottom_end[thickness_index],
                    top_end[thickness_index],
                    top_start[thickness_index],
                ])
            if thickness_index < wall_start_in_plinth or thickness_index >= wall_end_in_plinth:
                add_element(3, 7, [
                    top_start[thickness_index],
                    top_end[thickness_index],
                    top_end[thickness_index + 1],
                    top_start[thickness_index + 1],
                ])

        if wedge_active_stations[station_index] and wedge_active_stations[station_index + 1]:
            toe_top_start = wedge_toe_node_ids[station_index]
            toe_top_end = wedge_toe_node_ids[station_index + 1]
            toe_bottom_start = wedge_toe_bottom_node_ids[station_index]
            toe_bottom_end = wedge_toe_bottom_node_ids[station_index + 1]
            downstream_bottom_start = bottom_start[-1]
            downstream_bottom_end = bottom_end[-1]
            downstream_top_start = top_start[-1]
            downstream_top_end = top_end[-1]
            wall_top_start = top_start[wall_end_in_plinth]
            wall_top_end = top_end[wall_end_in_plinth]
            wall_bottom_start = bottom_start[wall_end_in_plinth]
            wall_bottom_end = bottom_end[wall_end_in_plinth]

            add_element(5, 1, [
                wall_bottom_start, toe_bottom_start, toe_top_start, wall_top_start,
                wall_bottom_end, toe_bottom_end, toe_top_end, wall_top_end,
            ])
            add_element(5, 1, [
                toe_bottom_start, downstream_bottom_start, downstream_top_start, toe_top_start,
                toe_bottom_end, downstream_bottom_end, downstream_top_end, toe_top_end,
            ])
            add_element(3, 1, [
                wall_bottom_start, wall_top_start, wall_top_end, wall_bottom_end,
            ])
            add_element(3, 1, [
                toe_bottom_start, toe_bottom_end, downstream_bottom_end, downstream_bottom_start,
            ])
            add_element(3, 7, [
                toe_top_start, downstream_top_start, downstream_top_end, toe_top_end,
            ])
            add_element(3, 7, [
                wall_top_start, toe_top_start, toe_top_end, wall_top_end,
            ])
        else:
            for thickness_index in range(wall_end_in_plinth, plinth_thickness_layers):
                add_element(5, 1, [
                    bottom_start[thickness_index],
                    bottom_end[thickness_index],
                    bottom_end[thickness_index + 1],
                    bottom_start[thickness_index + 1],
                    top_start[thickness_index],
                    top_end[thickness_index],
                    top_end[thickness_index + 1],
                    top_start[thickness_index + 1],
                ])
                add_element(3, 1, [
                    bottom_start[thickness_index],
                    bottom_start[thickness_index + 1],
                    bottom_end[thickness_index + 1],
                    bottom_end[thickness_index],
                ])
                add_element(3, 7, [
                    top_start[thickness_index],
                    top_end[thickness_index],
                    top_end[thickness_index + 1],
                    top_start[thickness_index + 1],
                ])

        if station_index == 0 or not plinth_active_segments[station_index - 1]:
            for thickness_index in range(plinth_thickness_layers):
                add_element(3, 7, [
                    bottom_start[thickness_index],
                    top_start[thickness_index],
                    top_start[thickness_index + 1],
                    bottom_start[thickness_index + 1],
                ])
        if station_index == len(plinth_active_segments) - 1 or not plinth_active_segments[station_index + 1]:
            for thickness_index in range(plinth_thickness_layers):
                add_element(3, 7, [
                    bottom_end[thickness_index],
                    bottom_end[thickness_index + 1],
                    top_end[thickness_index + 1],
                    top_end[thickness_index],
                ])

    for station_index in range(len(points) - 1):
        if not (wedge_active_stations[station_index] and wedge_active_stations[station_index + 1]):
            continue
        wedge_toe_start = wedge_toe_node_ids[station_index]
        wedge_toe_end = wedge_toe_node_ids[station_index + 1]
        for layer_index in range(wedge_interface_layer):
            wall_base_start = node_id(station_index, layer_index, thickness_layers)
            wall_base_end = node_id(station_index + 1, layer_index, thickness_layers)
            wall_top_start = node_id(station_index, layer_index + 1, thickness_layers)
            wall_top_end = node_id(station_index + 1, layer_index + 1, thickness_layers)
            if layer_index == wedge_interface_layer - 1:
                add_element(6, 1, [
                    wall_base_start, wedge_slope_node_ids[station_index][layer_index], wall_top_start,
                    wall_base_end, wedge_slope_node_ids[station_index + 1][layer_index], wall_top_end,
                ])
                add_element(3, 3, [
                    wedge_slope_node_ids[station_index][layer_index],
                    wedge_slope_node_ids[station_index + 1][layer_index],
                    wall_top_end,
                    wall_top_start,
                ])
            else:
                add_element(5, 1, [
                    wall_base_start, wedge_slope_node_ids[station_index][layer_index], wedge_slope_node_ids[station_index][layer_index + 1], wall_top_start,
                    wall_base_end, wedge_slope_node_ids[station_index + 1][layer_index], wedge_slope_node_ids[station_index + 1][layer_index + 1], wall_top_end,
                ])
                add_element(3, 3, [
                    wedge_slope_node_ids[station_index][layer_index],
                    wedge_slope_node_ids[station_index + 1][layer_index],
                    wedge_slope_node_ids[station_index + 1][layer_index + 1],
                    wedge_slope_node_ids[station_index][layer_index + 1],
                ])

    for level_index in range(vertical_layers):
        for thickness_index in range(thickness_layers):
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

    volume_element_count = (len(points) - 1) * vertical_layers * thickness_layers
    volume_element_count += sum(plinth_active_segments) * plinth_thickness_layers
    return volume_element_count, len(elements) - volume_element_count


volume_count, boundary_count = generate_curved_wall_mesh(points, mesh_path)

meta_path.write_text(
    json.dumps(
        {
            "station_count": len(points),
            "vertical_layers": vertical_layers,
            "junction_min_element_size_m": junction_min_element_size_m,
            "junction_transition_distance_m": junction_transition_distance_m,
            "main_body_element_size_m": main_body_element_size_m,
            "arch_subdivisions": arch_subdivisions,
            "wall_thickness_m": wall_thickness,
            "wall_height_above_plinth_m": wall_height_above_plinth_m,
            "wall_radius_m": radius,
            "wedge_angle_deg": wedge_angle_deg,
            "plinth_upstream_offset_m": plinth_upstream_offset_m,
            "plinth_downstream_offset_m": plinth_downstream_offset_m,
            "plinth_width_m": plinth_width_m,
            "plinth_base_width_m": plinth_base_width_m,
            "dam_height_m": dam_height,
            "mesh_size_m": mesh_size,
            "target_block_size_m": target_block_size,
            "thickness_layers": thickness_layers,
            "crest_detail_height_m": crest_detail_height,
            "crest_extra_thickness_m": crest_extra_thickness,
            "abutment_segment_count": abutment_segment_count,
            "geometry_mode": geometry_mode,
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
print(f"Generated {len(points)} stations, {volume_count} hex elements, and {boundary_count} boundary faces")
