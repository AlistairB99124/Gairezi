from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Data" / "plinth.json"
CONFIG_PATH = ROOT / "config.json"
META_PATH = Path(__file__).resolve().parent / "curved_dam_mesh_meta.json"
OUT_PATH = Path(__file__).resolve().parent / "results" / "wedge_orientation.png"


def load_inputs() -> tuple[list[dict[str, float]], dict, dict, dict]:
    plinth = json.loads(DATA_PATH.read_text())
    config = json.loads(CONFIG_PATH.read_text())
    metadata = json.loads(META_PATH.read_text())
    return plinth, config, metadata, {
        "threshold": float(config["wedge_start_below_crest_m"]),
        "angle_from_vertical": float(config["wedge_angle_from_vertical_deg"]),
        "outer_radius": float(config["wedge_anchor_radius_m"]),
        "inner_radius": float(metadata.get("wedge_opposite_vertex_radius_m", 74.0)),
    }


def main() -> None:
    plinth, config, metadata, inputs = load_inputs()
    threshold = inputs["threshold"]
    angle_from_vertical = inputs["angle_from_vertical"]
    outer_radius = inputs["outer_radius"]
    tangent = math.tan(math.radians(angle_from_vertical))

    chainage = [float(row["chainage"]) for row in plinth]
    depth = [float(row["plinth"]) for row in plinth]
    active = [value >= threshold for value in depth]
    start = float(metadata.get("wedge_start_station_m", 88.0))
    end = float(metadata.get("wedge_end_station_m", 124.0))
    width = end - start
    maximum_height = max(depth) - threshold
    maximum_thickness = maximum_height * tangent
    inner_radius = outer_radius - maximum_thickness

    fig = plt.figure(figsize=(14, 10), dpi=180)
    fig.suptitle("Wedge Orientation and Dimension Sanity Check", fontsize=16, fontweight="bold")

    # Plan view
    ax1 = fig.add_subplot(2, 2, 1)
    theta = [s / float(config["wall_centerline_radius_m"]) for s in chainage]
    wall_x = [outer_radius * math.sin(value) for value in theta]
    wall_y = [outer_radius * math.cos(value) for value in theta]
    toe_x = [inner_radius * math.sin(value) for value in theta]
    toe_y = [inner_radius * math.cos(value) for value in theta]
    ax1.plot(wall_x, wall_y, "k-", linewidth=1.8, label="Wedge face: r = 76 m")
    ax1.plot(toe_x, toe_y, "r--", linewidth=1.8, label="Maximum toe: r = 74 m")
    active_theta = [value for value, is_active in zip(theta, active) if is_active]
    if active_theta:
        active_wall_x = [outer_radius * math.sin(value) for value in active_theta]
        active_wall_y = [outer_radius * math.cos(value) for value in active_theta]
        active_toe_x = [inner_radius * math.sin(value) for value in active_theta]
        active_toe_y = [inner_radius * math.cos(value) for value in active_theta]
        ax1.plot(active_wall_x, active_wall_y, color="tab:blue", linewidth=3.0, label="Active wedge chainage")
        ax1.plot(active_toe_x, active_toe_y, color="tab:orange", linewidth=3.0)
    ax1.set_title("Plan View (X-Y)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.axis("equal")
    ax1.grid(True, color="0.9", linewidth=0.6)
    ax1.legend(frameon=False, fontsize=8)

    # Chainage logic
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(chainage, depth, "k-o", markersize=3, linewidth=1.3, label="Plinth depth p")
    ax2.axhline(threshold, color="tab:blue", linestyle="--", label="Threshold c = 25 m")
    ax2.axvspan(start, end, color="tab:orange", alpha=0.2, label=f"w = {width:.2f} m")
    ax2.set_title("Chainage Logic (positive depth below crest)")
    ax2.set_xlabel("Chainage (m)")
    ax2.set_ylabel("Depth below crest (m)")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, color="0.9", linewidth=0.6)
    ax2.legend(frameon=False, fontsize=8)

    # Maximum local section
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.add_patch(Polygon([(0, 0), (4, 0), (4, -29), (0, -29)], closed=True, facecolor="0.82", edgecolor="k", label="Wall"))
    ax3.axhspan(-32, -29, color="0.65", alpha=0.35, label="Bedrock below z = -29 m")
    wedge_points = [(4, -25), (4 + maximum_thickness, -29), (4, -29)]
    ax3.add_patch(Polygon(wedge_points, closed=True, facecolor="tab:orange", alpha=0.75, edgecolor="tab:red", label="Wedge"))
    ax3.plot([4, 4], [-25, -29], color="tab:blue", linewidth=2.2, label="Wall wedge face, r = 76 m")
    ax3.plot([4, 4 + maximum_thickness], [-29, -29], color="tab:red", linewidth=2.2, label="Wedge bottom face: r = 76 m to toe")
    ax3.plot([4, 4 + maximum_thickness], [-25, -29], color="tab:orange", linewidth=2.5, label="Wedge slope, 63.43° to horizontal")
    ax3.annotate("h = 4 m", xy=(3.5, -25.0), xytext=(3.5, -29.0), arrowprops={"arrowstyle": "<->"}, ha="right", va="center")
    ax3.annotate("t = 2 m", xy=(4.0, -30.2), xytext=(4 + maximum_thickness / 2, -30.2), arrowprops={"arrowstyle": "<->"}, ha="center")
    ax3.text(4.45, -27.0, "63.43°", rotation=-26.57, color="tab:red", va="center")
    ax3.text(4.15, -28.8, "bedrock interface", color="0.2", fontsize=8, va="top")
    ax3.set_title("Maximum Local Section")
    ax3.set_xlabel("Radial offset from wall face (m)")
    ax3.set_ylabel("Signed mesh Z (m)")
    ax3.set_xlim(-2, 7)
    ax3.set_ylim(-32, 2)
    ax3.grid(True, color="0.9", linewidth=0.6)
    ax3.legend(frameon=False, fontsize=8, loc="upper left")

    # Isometric
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    toe_radius = [outer_radius - max(0.0, value - threshold) * tangent for value in depth]
    plan_wall_x = [outer_radius * math.sin(value) for value in theta]
    plan_wall_y = [outer_radius * math.cos(value) for value in theta]
    ax4.plot(plan_wall_x, plan_wall_y, [-25.0] * len(chainage), color="tab:blue", linewidth=1.5)
    ax4.plot([r * math.sin(t) for r, t in zip(toe_radius, theta)], [r * math.cos(t) for r, t in zip(toe_radius, theta)], [-29.0] * len(chainage), color="tab:red", linewidth=1.5)
    for index in range(0, len(chainage), max(1, len(chainage) // 30)):
        ax4.plot(
            [plan_wall_x[index], toe_radius[index] * math.sin(theta[index])],
            [plan_wall_y[index], toe_radius[index] * math.cos(theta[index])],
            [-25.0, -29.0],
            color="0.35",
            linewidth=0.7,
        )
    ax4.set_title("Isometric Wedge Band")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.set_zlabel("Z (m)")
    ax4.view_init(elev=25, azim=-65)

    fig.text(0.5, 0.02, f"h = p - 25;  t = 0.5 * h;  w = {start:.2f} to {end:.2f} m = {width:.2f} m", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
