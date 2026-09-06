from __future__ import annotations

from pathlib import Path
import json
import math

import gmsh


root = Path(__file__).resolve().parent.parent
out_dir = Path(__file__).resolve().parent
plinth_path = root / "Data" / "plinth.json"
grid_controls_path = root / "Data" / "Computational_Grid_Controls.json"
config_path = root / "config.json"


def control_value(name: str) -> float:
    for row in json.loads(grid_controls_path.read_text()):
        if row["Mesh Control"] == name:
            return float(row["Client Value"])
    raise ValueError(f"Missing mesh control: {name}")


config = json.loads(config_path.read_text())
wall_thickness = float(config["wall_thickness_m"])
upstream_radius = float(config["wall_upstream_radius_m"])
centerline_radius = float(config["wall_centerline_radius_m"])
downstream_radius = float(config["wall_downstream_radius_m"])
wedge_threshold = float(config["wedge_start_below_crest_m"])
wedge_angle = math.radians(float(config["wedge_angle_from_vertical_deg"]))
mesh_size = control_value("Global Element Size")
corner_size = control_value("Local Corner Refinement")
minimum_transition_quality = 0.005
transition_bulk_size = mesh_size


def interpolate(rows: list[dict[str, float]], station: float, name: str) -> float:
    for start, end in zip(rows, rows[1:]):
        if start["chainage"] <= station <= end["chainage"]:
            fraction = (station - start["chainage"]) / (end["chainage"] - start["chainage"])
            return start[name] + fraction * (end[name] - start[name])
    raise ValueError(f"Station {station} lies outside the plinth profile")


def point_at(radius: float, station: float, z_value: float) -> tuple[float, float, float]:
    theta = station / centerline_radius
    return radius * math.sin(theta), radius * math.cos(theta), z_value


def wire(points: list[tuple[float, float, float]]) -> int:
    point_tags = [gmsh.model.occ.addPoint(*point) for point in points]
    line_tags = [
        gmsh.model.occ.addLine(point_tags[index], point_tags[(index + 1) % len(point_tags)])
        for index in range(len(point_tags))
    ]
    return gmsh.model.occ.addWire(line_tags)


def classify_boundary_surfaces(
    volumes: list[tuple[int, int]],
    start_station: float,
    end_station: float,
) -> dict[int, list[int]]:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False, recursive=False)
    groups = {index: [] for index in range(1, 8)}
    for _, surface_tag in boundaries:
        x_value, y_value, z_value = gmsh.model.occ.getCenterOfMass(2, surface_tag)
        radius = math.hypot(x_value, y_value)
        theta = math.atan2(x_value, y_value)
        if theta < 0.0:
            theta += 2.0 * math.pi
        bounds = gmsh.model.getBoundingBox(2, surface_tag)
        z_min, z_max = bounds[2], bounds[5]
        if z_min >= -1.0e-4 and z_max >= -1.0e-4:
            groups[4].append(surface_tag)
        elif theta <= (start_station / centerline_radius) + 0.02:
            groups[5].append(surface_tag)
        elif theta >= (end_station / centerline_radius) - 0.02:
            groups[6].append(surface_tag)
        elif radius >= upstream_radius - 0.1:
            groups[2].append(surface_tag)
        elif radius <= downstream_radius + 0.1:
            groups[3].append(surface_tag)
        elif z_max <= -1.0e-8:
            groups[1].append(surface_tag)
        else:
            groups[7].append(surface_tag)
    return groups


def add_convergence_size_field(
    crossings: list[float],
    profile_rows: list[dict[str, float]],
) -> None:
    field_id = gmsh.model.mesh.field.add("Distance")
    convergence_points = []
    for crossing in crossings:
        for radius in (downstream_radius, downstream_radius - 2.0):
            for z_value in (-wedge_threshold, -interpolate(profile_rows, crossing, "plinth")):
                convergence_points.append(gmsh.model.occ.addPoint(*point_at(radius, crossing, z_value)))
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.field.setNumbers(field_id, "PointsList", convergence_points)

    threshold_id = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_id, "InField", field_id)
    gmsh.model.mesh.field.setNumber(threshold_id, "SizeMin", corner_size)
    gmsh.model.mesh.field.setNumber(threshold_id, "SizeMax", transition_bulk_size)
    gmsh.model.mesh.field.setNumber(threshold_id, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold_id, "DistMax", 2.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_id)


def main() -> None:
    rows = [
        {"chainage": float(row["chainage"]), "ground": float(row["ground"]), "plinth": float(row["plinth"])}
        for row in json.loads(plinth_path.read_text())
    ]
    rows.sort(key=lambda row: row["chainage"])
    start_station = rows[0]["chainage"] + transition_bulk_size
    end_station = rows[-1]["chainage"] - transition_bulk_size

    # The wedge cannot be created at its zero-area convergence points. The local
    # fragment volumes stop one corner-size inward; Gmsh meshes their caps as the
    # conforming transition zone without moving either parent volume's surface.
    crossings = []
    for start, end in zip(rows, rows[1:]):
        start_height = start["plinth"] - wedge_threshold
        end_height = end["plinth"] - wedge_threshold
        if start_height * end_height < 0.0:
            crossings.append(start["chainage"] - start_height * (end["chainage"] - start["chainage"]) / (end_height - start_height))
    if len(crossings) != 2:
        raise ValueError("Expected two wedge convergence stations")

    stations = [start_station]
    station = start_station + transition_bulk_size
    while station < end_station:
        stations.append(station)
        station += transition_bulk_size
    stations.append(end_station)
    for crossing in crossings:
        for offset in (-corner_size, 0.0, corner_size):
            candidate = crossing + offset
            if start_station < candidate < end_station:
                stations.append(candidate)
    stations = sorted(set(stations))

    gmsh.initialize()
    gmsh.model.add("curved_dam_transition")
    try:
        wall_wires = []
        plinth_wire_runs: list[list[int]] = []
        plinth_wires: list[int] = []
        wedge_wires = []
        for station in stations:
            plinth_depth = interpolate(rows, station, "plinth")
            ground_depth = interpolate(rows, station, "ground")
            base_z = -plinth_depth
            ground_z = -ground_depth
            wall_wires.append(wire([
                point_at(upstream_radius, station, base_z),
                point_at(upstream_radius, station, 0.0),
                point_at(downstream_radius, station, 0.0),
                point_at(downstream_radius, station, base_z),
            ]))
            if abs(base_z - ground_z) > 1.0e-8:
                plinth_wires.append(wire([
                    point_at(upstream_radius + 1.0, station, ground_z),
                    point_at(upstream_radius + 1.0, station, base_z),
                    point_at(downstream_radius - 2.0, station, base_z),
                    point_at(downstream_radius - 2.0, station, ground_z),
                ]))
            elif plinth_wires:
                plinth_wire_runs.append(plinth_wires)
                plinth_wires = []
            wedge_height = plinth_depth - wedge_threshold
            within_wedge_span = crossings[0] + corner_size <= station <= crossings[1] - corner_size
            if within_wedge_span:
                wedge_run = wedge_height * math.tan(wedge_angle)
                if wedge_run > 1.0e-8:
                    wedge_wires.append(wire([
                        point_at(downstream_radius, station, base_z),
                        point_at(downstream_radius, station, -wedge_threshold),
                        point_at(downstream_radius - wedge_run, station, base_z),
                    ]))

        if plinth_wires:
            plinth_wire_runs.append(plinth_wires)
        wall_volume = gmsh.model.occ.addThruSections(wall_wires, makeSolid=True, makeRuled=True)
        plinth_volumes = []
        for wire_run in plinth_wire_runs:
            if len(wire_run) >= 2:
                plinth_volumes.extend(gmsh.model.occ.addThruSections(wire_run, makeSolid=True, makeRuled=True))
        wedge_volume = gmsh.model.occ.addThruSections(wedge_wires, makeSolid=True, makeRuled=True)
        gmsh.model.occ.fuse(wall_volume + plinth_volumes, wedge_volume)
        gmsh.model.occ.removeAllDuplicates()
        add_convergence_size_field(crossings, rows)
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(3)
        gmsh.model.addPhysicalGroup(3, [tag for _, tag in volumes], 1)
        gmsh.model.setPhysicalName(3, 1, "DamBody")
        boundary_groups = classify_boundary_surfaces(volumes, start_station, end_station)
        boundary_names = {
            1: "BedrockBase",
            2: "UpstreamHydrostaticPressure",
            3: "DownstreamTailwater",
            4: "CrestOverflowSurcharge",
            5: "LeftAbutmentSpring",
            6: "RightAbutmentSpring",
            7: "SideFaces",
        }
        for physical_id, surface_tags in boundary_groups.items():
            if not surface_tags and physical_id != 7:
                raise ValueError(f"Missing required Elmer boundary group {physical_id}: {boundary_names[physical_id]}")
            if surface_tags:
                gmsh.model.addPhysicalGroup(2, surface_tags, physical_id)
                gmsh.model.setPhysicalName(2, physical_id, boundary_names[physical_id])

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", transition_bulk_size)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(3)
        element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
        tetrahedron_index = list(element_types).index(4)
        qualities = gmsh.model.mesh.getElementQualities(element_tags[tetrahedron_index])
        minimum_quality = min(qualities) if len(qualities) else 0.0
        if minimum_quality < minimum_transition_quality:
            raise ValueError(
                f"Transition mesh quality {minimum_quality:.6g} is below {minimum_transition_quality:.6g}"
            )
        mesh_path = out_dir / "curved_dam_transition.msh"
        gmsh.write(str(mesh_path))
        print(f"Wrote {mesh_path}")
        print(f"Minimum tetrahedral quality: {minimum_quality:.6f}")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()