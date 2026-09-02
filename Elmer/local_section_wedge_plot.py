from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent.parent
mesh_path = Path(__file__).resolve().parent / "curved_dam_mesh.msh"
meta_path = mesh_path.with_name(f"{mesh_path.stem}_meta.json")
out_path = Path(__file__).resolve().parent / "results" / "local_section_wedge.png"


def read_gmsh_nodes(path: Path):
    lines = path.read_text().splitlines()
    start = lines.index("$Nodes")
    count = int(lines[start + 1])
    nodes = []
    for line in lines[start + 2 : start + 2 + count]:
        parts = line.split()
        if len(parts) >= 4:
            nodes.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return nodes


def read_metadata(path: Path):
    if not path.exists():
        return {"vertical_layers": 4, "thickness_layers": 1}
    return json.loads(path.read_text())


def station_slice(nodes, station_index, vertical_layers, thickness_layers):
    nodes_per_station = (vertical_layers + 1) * (thickness_layers + 1)
    start = station_index * nodes_per_station
    return nodes[start : start + nodes_per_station]


def main() -> None:
    nodes = read_gmsh_nodes(mesh_path)
    meta = read_metadata(meta_path)
    vertical_layers = int(meta.get("vertical_layers", 4))
    thickness_layers = int(meta.get("thickness_layers", 1))
    nodes_per_station = (vertical_layers + 1) * (thickness_layers + 1)

    # pick the station nearest the highest point of the dam where the wedge is most visible
    station_index = len(nodes) // nodes_per_station // 2
    station = station_slice(nodes, station_index, vertical_layers, thickness_layers)

    # Use the explicit local section geometry from the sketch, not the global arch coordinates.
    # Upstream face is the left edge of the wall. The wall thickness is 4.0 m. The
    # downstream wedge is defined by a 30° angle to the wall face, so the run varies with
    # the local base depth; the 2.0 m run is only the limiting case for a 3.46 m drop.
    crest_z = 0.0
    top_wedge_z = -25.54
    base_z = -29.0
    wall_thickness_m = 4.0
    wedge_run = math.tan(math.radians(30.0)) * (top_wedge_z - base_z)

    upstream_x = 0.0
    downstream_vertical_x = upstream_x + wall_thickness_m
    wedge_top = (downstream_vertical_x, top_wedge_z)
    wedge_base = (downstream_vertical_x + wedge_run, base_z)
    downstream_face_top = (downstream_vertical_x, crest_z)
    downstream_face_bottom = (downstream_vertical_x, top_wedge_z)
    upstream_face_top = (upstream_x, crest_z)
    upstream_face_bottom = (upstream_x, base_z)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=180)
    ax.set_title("Local section through downstream wedge")
    ax.set_xlabel("Offset from wall centreline (m)")
    ax.set_ylabel("Elevation (m)")
    ax.grid(True, color="0.85", linewidth=0.7)

    # Draw the wall, the 1 m upstream / 2 m downstream plinth, and the wedge.
    plinth_base_z = base_z - 3.46
    ax.fill(
        [upstream_x - 1.0, downstream_vertical_x + 2.0, downstream_vertical_x + 2.0, upstream_x - 1.0],
        [plinth_base_z, plinth_base_z, base_z, base_z],
        color="#c7c7c7",
        alpha=0.7,
        label="concrete plinth",
    )
    ax.plot([upstream_face_top[0], upstream_face_bottom[0]], [upstream_face_top[1], upstream_face_bottom[1]], "k-", linewidth=2.0, label="upstream face")
    ax.plot([downstream_face_top[0], downstream_face_bottom[0]], [downstream_face_top[1], downstream_face_bottom[1]], "b-", linewidth=2.0, label="downstream face")
    ax.plot([wedge_top[0], wedge_base[0], downstream_face_bottom[0], wedge_top[0]], [wedge_top[1], wedge_base[1], downstream_face_bottom[1], wedge_top[1]], "r-", linewidth=2.5, label="wedge face")

    # Add section labels.
    ax.axhline(0.0, color="0.6", linestyle="--", linewidth=0.8)
    ax.axhline(top_wedge_z, color="0.6", linestyle=":", linewidth=0.8)
    ax.axhline(base_z, color="0.6", linestyle=":", linewidth=0.8)
    ax.text((wedge_top[0] + wedge_base[0]) / 2.0 + 0.35, (wedge_top[1] + wedge_base[1]) / 2.0, "3.46 m", fontsize=9)
    ax.text((wedge_base[0] + downstream_face_bottom[0]) / 2.0 + 0.2, base_z - 0.4, f"{wedge_run:.2f} m", fontsize=9)
    ax.text((downstream_face_top[0] + downstream_face_bottom[0]) / 2.0 + 0.15, (downstream_face_top[1] + downstream_face_bottom[1]) / 2.0, "25.54 m", fontsize=9)

    ax.legend(frameon=False)
    ax.set_xlim(-1.5, 7.0)
    ax.set_ylim(plinth_base_z - 1.0, crest_z + 2.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
