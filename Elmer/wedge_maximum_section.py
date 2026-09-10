from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
META_PATH = Path(__file__).resolve().parent / "curved_dam_mesh_meta.json"
OUT_PATH = Path(__file__).resolve().parent / "results" / "wedge_maximum_section.png"


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    metadata = json.loads(META_PATH.read_text())

    threshold = float(config["wedge_start_below_crest_m"])
    angle_from_vertical = float(config.get("wedge_angle_from_vertical_deg", 26.565051177077994))
    outer_radius = float(config["wedge_anchor_radius_m"])
    inner_radius = float(metadata.get("wedge_opposite_vertex_radius_m", 74.0))
    maximum_depth = threshold + float(metadata.get("maximum_wedge_height_m", 4.0))
    maximum_height = maximum_depth - threshold
    thickness = outer_radius - inner_radius
    slope_angle = 90.0 - angle_from_vertical

    wall_upstream_radius = float(config["wall_upstream_radius_m"])
    z_top = -threshold
    z_base = -maximum_depth
    bedrock_depth = 1.0

    fig, ax = plt.subplots(figsize=(8, 10), dpi=220)

    # The wall is shown only to give the wedge-facing wall boundary context.
    ax.add_patch(
        Polygon(
            [(wall_upstream_radius, 0.0), (outer_radius, 0.0), (outer_radius, z_base), (wall_upstream_radius, z_base)],
            closed=True,
            facecolor="0.86",
            edgecolor="0.2",
            linewidth=1.5,
            label="Wall",
        )
    )
    ax.axhspan(z_base - bedrock_depth, z_base, color="0.55", alpha=0.35, label="Bedrock below z = -29 m")

    # The wedge cross-section is the triangular region from r=76,z=-25 to r=74,z=-29.
    ax.add_patch(
        Polygon(
            [(outer_radius, z_top), (inner_radius, z_base), (outer_radius, z_base)],
            closed=True,
            facecolor="tab:orange",
            edgecolor="tab:red",
            linewidth=2.0,
            alpha=0.8,
            label="Wedge",
        )
    )
    ax.plot([outer_radius, outer_radius], [z_top, z_base], color="tab:blue", linewidth=3.0, label="Wall wedge face: r = 76 m")
    ax.plot([outer_radius, inner_radius], [z_top, z_base], color="tab:orange", linewidth=3.0, label=f"Wedge slope: {slope_angle:.2f}° to horizontal")
    ax.plot([inner_radius, outer_radius], [z_base, z_base], color="tab:red", linewidth=3.0, label="Wedge bottom face: r = 76 m to toe")
    ax.plot([inner_radius, inner_radius], [z_base - 0.08, z_base + 0.08], color="tab:red", linewidth=3.0, label="Toe vertex line: r = 74 m")

    # Dimension h is vertical; t is radial/horizontal.
    ax.annotate(
        f"h = {maximum_height:.0f} m",
        xy=(outer_radius - 0.15, z_top),
        xytext=(outer_radius - 0.15, z_base),
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "<->", "linewidth": 1.2},
    )
    ax.annotate(
        f"t = {thickness:.0f} m",
        xy=(outer_radius, z_base - 0.18),
        xytext=(inner_radius, z_base - 0.18),
        ha="center",
        va="top",
        arrowprops={"arrowstyle": "<->", "linewidth": 1.2},
    )
    ax.text((outer_radius + inner_radius) / 2.0 + 0.08, (z_top + z_base) / 2.0, f"{slope_angle:.2f}°", color="tab:red", rotation=-angle_from_vertical, va="center")
    ax.text(outer_radius + 0.05, z_top + 0.12, "z = -25 m", color="tab:blue", va="bottom")
    ax.text(inner_radius - 0.05, z_base - 0.05, "z = -29 m", color="tab:red", ha="right", va="top")
    ax.text((wall_upstream_radius + outer_radius) / 2.0, -12.5, "wall", color="0.25", ha="center", va="center", rotation=90)

    ax.set_title("Maximum Local Wedge Section\nEqual axis scale")
    ax.set_xlabel("Radius from dam centreline (m)")
    ax.set_ylabel("Signed mesh Z (m)")
    ax.set_xlim(inner_radius - 0.8, wall_upstream_radius + 0.8)
    ax.set_ylim(-30.0, -20.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.text(
        0.02,
        0.02,
        "Positive design depths: c = 25 m, p = 29 m; signed mesh elevations: z = -25 to -29 m",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )

    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.12, top=0.88)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
