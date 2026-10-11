from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_gmsh_nodes(mesh_path: Path) -> list[tuple[float, float, float]]:
    lines = mesh_path.read_text().splitlines()
    start = lines.index("$Nodes")
    count = int(lines[start + 1])
    node_lines = lines[start + 2 : start + 2 + count]
    nodes: list[tuple[float, float, float]] = []
    for line in node_lines:
        parts = line.split()
        nodes.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return nodes


def read_mesh_metadata(mesh_path: Path) -> dict[str, int]:
    meta_path = mesh_path.with_name(f"{mesh_path.stem}_meta.json")
    if not meta_path.exists():
        return {"station_count": len(read_gmsh_nodes(mesh_path)) // 4, "vertical_layers": 1, "thickness_layers": 1}
    return json.loads(meta_path.read_text())


def split_stations(
    nodes: list[tuple[float, float, float]],
    station_count: int,
    vertical_layers: int = 1,
    thickness_layers: int = 1,
) -> list[dict[str, tuple[float, float, float]]]:
    nodes_per_level = thickness_layers + 1
    nodes_per_station = (vertical_layers + 1) * nodes_per_level
    wall_node_count = station_count * nodes_per_station
    if len(nodes) < wall_node_count:
        raise ValueError("Mesh does not contain the structured wall-node block")

    stations: list[dict[str, tuple[float, float, float]]] = []
    for i in range(0, wall_node_count, nodes_per_station):
        top_offset = vertical_layers * nodes_per_level
        stations.append(
            {
                "inner_base": nodes[i],
                "outer_base": nodes[i + thickness_layers],
                "inner_top": nodes[i + top_offset],
                "outer_top": nodes[i + top_offset + thickness_layers],
            }
        )
    return stations


def extract_downstream_face(mesh_path: Path) -> list[tuple[float, float, float]]:
    metadata = read_mesh_metadata(mesh_path)
    vertical_layers = int(metadata.get("vertical_layers", 1))
    face: list[tuple[float, float, float]] = []
    for station in load_station_edges(mesh_path):
        base_z = station["outer_base"][2]
        crest_z = station["outer_top"][2]
        for layer in range(vertical_layers + 1):
            fraction = layer / vertical_layers
            x_value, y_value, _ = station["outer_base"]
            face.append((x_value, y_value, base_z + fraction * (crest_z - base_z)))
    return face


def load_station_edges(mesh_path: Path) -> list[dict[str, tuple[float, float, float]]]:
    metadata = read_mesh_metadata(mesh_path)
    centerline_path = mesh_path.with_name("curved_dam_centerline.csv")
    if "wall_upstream_radius_m" not in metadata or "wall_downstream_radius_m" not in metadata:
        metadata = json.loads((Path(__file__).resolve().parent / "curved_dam_mesh_meta.json").read_text())
    if not centerline_path.exists():
        centerline_path = Path(__file__).resolve().parent / "curved_dam_centerline.csv"
    upstream_radius = float(metadata["wall_upstream_radius_m"])
    downstream_radius = float(metadata["wall_downstream_radius_m"])
    stations: list[dict[str, tuple[float, float, float]]] = []
    with centerline_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            theta = math.atan2(float(row["x"]), float(row["y"]))
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            base_z = float(row["z_base"])
            crest_z = float(row["z_crest"])
            stations.append(
                {
                    "inner_base": (upstream_radius * sin_theta, upstream_radius * cos_theta, base_z),
                    "outer_base": (downstream_radius * sin_theta, downstream_radius * cos_theta, base_z),
                    "inner_top": (upstream_radius * sin_theta, upstream_radius * cos_theta, crest_z),
                    "outer_top": (downstream_radius * sin_theta, downstream_radius * cos_theta, crest_z),
                }
            )
    return stations


def xyz(points: list[tuple[float, float, float]]) -> tuple[list[float], list[float], list[float]]:
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    return x, y, z


def plot_views(mesh_path: Path, out_path: Path) -> None:
    stations = load_station_edges(mesh_path)
    downstream_face = extract_downstream_face(mesh_path)

    inner_base = [s["inner_base"] for s in stations]
    outer_base = [s["outer_base"] for s in stations]
    inner_top = [s["inner_top"] for s in stations]
    outer_top = [s["outer_top"] for s in stations]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("Plan View (X-Y)", "Elevation View (X-Z)", "Elevation View (Y-Z)", "Isometric Sketch"),
    )

    x_ib, y_ib, _ = xyz(inner_base)
    x_ob, y_ob, _ = xyz(outer_base)
    fig.add_trace(go.Scatter(x=x_ib, y=y_ib, mode="lines", line=dict(color="black", width=1.8), name="Upstream wall face"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_ob, y=y_ob, mode="lines", line=dict(color="black", width=1.8), name="Downstream wall face"), row=1, col=1)
    for a, b in zip(inner_base[::5], outer_base[::5]):
        fig.add_trace(go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines", line=dict(color="rgba(90,90,90,0.9)", width=0.9), showlegend=False), row=1, col=1)

    x_it, _, z_it = xyz(inner_top)
    x_ot, _, z_ot = xyz(outer_top)
    x_ib, _, z_ib = xyz(inner_base)
    x_ob, _, z_ob = xyz(outer_base)
    downstream_x = [p[0] for p in downstream_face]
    downstream_z = [p[2] for p in downstream_face]
    fig.add_trace(go.Scatter(x=x_ib, y=z_ib, mode="lines", line=dict(color="black", width=1.5), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_ob, y=z_ob, mode="lines", line=dict(color="black", width=1.5), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_it, y=z_it, mode="lines", line=dict(color="black", width=1.2, dash="dash"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_ot, y=z_ot, mode="lines", line=dict(color="black", width=1.2, dash="dash"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=downstream_x, y=downstream_z, mode="lines", line=dict(color="red", width=1.8), name="Downstream wall face", showlegend=False), row=1, col=2)
    for ib, it, ob, ot in zip(inner_base[::5], inner_top[::5], outer_base[::5], outer_top[::5]):
        fig.add_trace(go.Scatter(x=[ib[0], it[0]], y=[ib[2], it[2]], mode="lines", line=dict(color="rgba(100,100,100,0.9)", width=0.8), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[ob[0], ot[0]], y=[ob[2], ot[2]], mode="lines", line=dict(color="rgba(100,100,100,0.9)", width=0.8), showlegend=False), row=1, col=2)

    _, y_it, z_it = xyz(inner_top)
    _, y_ot, z_ot = xyz(outer_top)
    _, y_ib, z_ib = xyz(inner_base)
    _, y_ob, z_ob = xyz(outer_base)
    downstream_y = [p[1] for p in downstream_face]
    fig.add_trace(go.Scatter(x=y_ib, y=z_ib, mode="lines", line=dict(color="black", width=1.5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=y_ob, y=z_ob, mode="lines", line=dict(color="black", width=1.5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=y_it, y=z_it, mode="lines", line=dict(color="black", width=1.2, dash="dash"), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=y_ot, y=z_ot, mode="lines", line=dict(color="black", width=1.2, dash="dash"), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=downstream_y, y=downstream_z, mode="lines", line=dict(color="red", width=1.8), showlegend=False), row=2, col=1)
    for ib, it, ob, ot in zip(inner_base[::5], inner_top[::5], outer_base[::5], outer_top[::5]):
        fig.add_trace(go.Scatter(x=[ib[1], it[1]], y=[ib[2], it[2]], mode="lines", line=dict(color="rgba(100,100,100,0.9)", width=0.8), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=[ob[1], ot[1]], y=[ob[2], ot[2]], mode="lines", line=dict(color="rgba(100,100,100,0.9)", width=0.8), showlegend=False), row=2, col=1)

    for edge in (inner_base, outer_base, inner_top, outer_top):
        x, y, z = xyz(edge)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="black", width=4), showlegend=False), row=2, col=2)
    for ib, ob, it, ot in zip(inner_base[::5], outer_base[::5], inner_top[::5], outer_top[::5]):
        fig.add_trace(go.Scatter3d(x=[ib[0], ob[0]], y=[ib[1], ob[1]], z=[ib[2], ob[2]], mode="lines", line=dict(color="rgba(90,90,90,0.9)", width=2), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter3d(x=[it[0], ot[0]], y=[it[1], ot[1]], z=[it[2], ot[2]], mode="lines", line=dict(color="rgba(90,90,90,0.9)", width=2), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter3d(x=[ib[0], it[0]], y=[ib[1], it[1]], z=[ib[2], it[2]], mode="lines", line=dict(color="rgba(90,90,90,0.9)", width=2), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter3d(x=[ob[0], ot[0]], y=[ob[1], ot[1]], z=[ob[2], ot[2]], mode="lines", line=dict(color="rgba(90,90,90,0.9)", width=2), showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="X (m)", row=1, col=1, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="Y (m)", row=1, col=1)
    fig.update_xaxes(title_text="X (m)", row=1, col=2)
    fig.update_yaxes(title_text="Z (m)", row=1, col=2)
    fig.update_xaxes(title_text="Y (m)", row=2, col=1)
    fig.update_yaxes(title_text="Z (m)", row=2, col=1)
    fig.update_scenes(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z (m)",
        camera=dict(eye=dict(x=1.4, y=-1.9, z=1.1)),
    )

    fig.update_layout(
        title=dict(text="Arch Dam Orientation Sketches", x=0.5, font=dict(size=20)),
        width=1400,
        height=1000,
        showlegend=True,
        template="plotly_white",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=1.2)

    html_path = out_path.with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs="cdn")

    print(f"Wrote {out_path}")
    print(f"Wrote {html_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create architect-style orientation sketches for the arch dam mesh.")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path(__file__).resolve().parent / "curved_dam_mesh.msh",
        help="Path to Gmsh mesh file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "arch_dam_orientations.png",
        help="Output image path",
    )
    args = parser.parse_args()

    plot_views(args.mesh, args.out)


if __name__ == "__main__":
    main()
