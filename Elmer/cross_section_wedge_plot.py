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
out_path = Path(__file__).resolve().parent / "results" / "cross_section_wedge.png"


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
        return {"vertical_layers": 1, "thickness_layers": 1}
    return json.loads(path.read_text())


# Use the midpoint station of the dam to isolate the section where the wedge is visible.
# The section is built in local coordinates around the centerline; the relevant offset is along the normal direction.
def main() -> None:
    nodes = read_gmsh_nodes(mesh_path)
    meta = read_metadata(meta_path)
    vertical_layers = int(meta.get("vertical_layers", 1))
    thickness_layers = int(meta.get("thickness_layers", 1))
    nodes_per_station = (vertical_layers + 1) * (thickness_layers + 1)

    # choose a mid-arch station at roughly the center of the run
    station_index = max(1, len(nodes) // nodes_per_station // 2)
    station_offset = station_index * nodes_per_station
    station = nodes[station_offset : station_offset + nodes_per_station]

    xs = []
    zs = []
    for layer in range(vertical_layers + 1):
        for thickness in range(thickness_layers + 1):
            idx = layer * (thickness_layers + 1) + thickness
            x, y, z = station[idx]
            xs.append(x)
            zs.append(z)

    # isolate the downstream face in local section coordinates by taking the outer wall edge
    downstream = []
    for layer in range(vertical_layers + 1):
        idx = layer * (thickness_layers + 1) + thickness_layers
        downstream.append(station[idx])

    x_down = [p[0] for p in downstream]
    z_down = [p[2] for p in downstream]

    # create a simple section plot with the exact wedge geometry indicated by the drawing
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    ax.set_title("Dam cross-section: downstream wedge")
    ax.set_xlabel("Horizontal offset from wall centreline (m)")
    ax.set_ylabel("Elevation (m)")
    ax.grid(True, color="0.85", linewidth=0.7)

    # wall section outline
    wall_left = [station[i][0] for i in range(0, thickness_layers + 1)]
    wall_right = [station[i][0] for i in range(vertical_layers * (thickness_layers + 1), (vertical_layers + 1) * (thickness_layers + 1))]
    wall_z = [station[i][2] for i in range(0, thickness_layers + 1)]
    wall_top_z = [station[i][2] for i in range(vertical_layers * (thickness_layers + 1), (vertical_layers + 1) * (thickness_layers + 1))]
    ax.plot(wall_left, wall_z, "k-", linewidth=1.8, label="wall")
    ax.plot(wall_right, wall_top_z, "k-", linewidth=1.8)

    # highlight downstream face wedge
    ax.plot(x_down, z_down, "r-", linewidth=2.4, label="downstream wedge")

    # mark the exact section dimensions from the sketch
    crest_z = 0.0
    wedge_start_z = crest_z - 25.54
    base_z = crest_z - 29.0
    run = math.tan(math.radians(30.0)) * (wedge_start_z - base_z)
    ax.plot([x_down[0], x_down[-1]], [z_down[0], z_down[-1]], "r-", linewidth=2.4)
    ax.axhline(0.0, color="0.6", linestyle="--", linewidth=0.8)
    ax.text(x_down[0] + 0.5, (crest_z + wedge_start_z) / 2, "25.54 m", color="black", fontsize=9)
    ax.text(x_down[-1] + 0.7, (base_z + wedge_start_z) / 2, "3.46 m", color="black", fontsize=9)
    ax.text((x_down[-1] + x_down[0]) / 2, base_z - 0.7, f"{run:.2f} m run", color="black", fontsize=9)

    ax.legend(frameon=False)
    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.set_ylim(base_z - 2, crest_z + 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
