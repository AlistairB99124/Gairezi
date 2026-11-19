"""Pictures of a stepped wall's downstream face without and with its mortar fill.

    python -m Elmer.mortar_fill_pictures --config <config.json> --output <folder> --name option5

Writes ``<name>-downstream.png`` (the face seen from downstream, without and with the fill)
and ``<name>-section.png`` (the wall's section at its deepest point). The configuration must
have ``material.mortar`` and thickness steps. These show the model's geometry only; no solver
is involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .model_config import load_config, mortar_fill, wall_thickness_steps
from .regions.model_groups import FACE_GROUP_IDS
from .regions.mortar_fill import wall_with_fill

CONCRETE_COLOUR = np.array([0.74, 0.75, 0.76])
MORTAR_COLOUR = np.array([0.88, 0.74, 0.52])
LIGHT = np.array([-0.35, -0.55, 0.75]) / np.linalg.norm([-0.35, -0.55, 0.75])


def _shaded(triangles: np.ndarray, colour: np.ndarray, towards: np.ndarray) -> np.ndarray:
    """One colour per triangle, darker where it turns away from the light; normals face ``towards``."""
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    normals[np.einsum("ij,ij->i", normals, towards - triangles.mean(axis=1)) < 0] *= -1
    shade = 0.55 + 0.45 * np.clip(normals @ LIGHT, 0.0, 1.0)
    return np.clip(shade[:, None] * colour[None, :], 0.0, 1.0)


VIEW_ELEVATION_DEG = 12.0


def _downstream_view(axis, groups: list[tuple[np.ndarray, np.ndarray]], bounds: np.ndarray, title: str) -> None:
    """Parallel projection of the triangles as seen from downstream, drawn far to near."""
    from matplotlib.collections import PolyCollection

    centre = bounds.mean(axis=0)
    # The viewer stands downstream, on the arch-centre side, looking at the middle of the wall from a little above.
    bearing = np.arctan2(centre[1], centre[0])
    tilt = np.radians(VIEW_ELEVATION_DEG)
    forward = np.array([np.cos(tilt) * np.cos(bearing), np.cos(tilt) * np.sin(bearing), -np.sin(tilt)])
    right = np.array([np.sin(bearing), -np.cos(bearing), 0.0])
    up = np.cross(right, forward)
    if up[2] < 0:
        up = -up
    viewer = centre - 400.0 * forward
    triangles = np.vstack([part for part, _ in groups if len(part)])
    colours = np.vstack([_shaded(part, colour, viewer) for part, colour in groups if len(part)])
    order = np.argsort(-(triangles.mean(axis=1) @ forward))
    screen = np.stack([triangles @ right, triangles @ up], axis=2)[order]
    axis.add_collection(PolyCollection(screen, facecolors=colours[order], edgecolors=colours[order], linewidths=0.3))
    axis.set_xlim(screen[:, :, 0].min() - 2.0, screen[:, :, 0].max() + 2.0)
    axis.set_ylim(screen[:, :, 1].min() - 2.0, screen[:, :, 1].max() + 2.0)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title(title, fontsize=12)


def render(config_path: Path, output: Path, name: str, label: str | None = None) -> dict:
    """Write the two pictures for ``config_path`` and return what was drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as directory:
        shutil.copy2(config_path, Path(directory) / "config.json")
        config = load_config(Path(directory) / "config.json")
        definition = mortar_fill(config)
        if definition is None:
            raise ValueError("The configuration has no mortar fill to draw: it needs material.mortar and thickness steps")
        wall, fill = wall_with_fill(Path(directory))
    if fill is None:
        raise ValueError("No step of this wall has a ledge for the fill")
    nodes = np.asarray(wall.nodes)
    tags = np.asarray([tag for tag, _ in wall.boundaries])
    faces = np.asarray([face for _, face in wall.boundaries])
    seen = tags != FACE_GROUP_IDS["UPSTREAM"]  # the upstream face is hidden from downstream
    covered = np.zeros(len(faces), dtype=bool)
    covered[fill.wall_faces] = True
    bounds = np.array([nodes.min(axis=0), nodes.max(axis=0)])
    output.mkdir(parents=True, exist_ok=True)
    label = label or name
    slope = definition["slope_run_per_rise"]
    built = sorted({round(step["built_slope_run_per_rise"], 4) for step in fill.report["steps"]})

    figure = plt.figure(figsize=(12, 7.6), dpi=160)
    _downstream_view(figure.add_subplot(2, 1, 1), [(nodes[faces[seen]], CONCRETE_COLOUR)], bounds,
                     "Stepped wall as analysed")
    _downstream_view(figure.add_subplot(2, 1, 2), [
        (nodes[faces[seen & ~covered]], CONCRETE_COLOUR),
        (fill.points[np.vstack([fill.slope_faces, fill.end_faces])], MORTAR_COLOUR)], bounds,
        f"With mortar fill on the steps (1 in {1 / slope:g})")
    figure.suptitle(f"{label}: downstream face. Model geometry only; the fill is drawn in a different colour "
                    "to show where it is.", fontsize=11)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.91, hspace=0.08)
    downstream_path = output / f"{name}-downstream.png"
    figure.savefig(downstream_path)
    plt.close(figure)

    # Section through the wall at its deepest point: radius across, elevation up.
    wall_config = config["geometry"]["wall"]
    upstream = float(wall_config["upstream_radius_m"])
    bands = wall_thickness_steps(wall_config)
    base = float(nodes[:, 2].min())
    outline = [(upstream, 0.0), (upstream, base)]
    widths = [width for _, width in bands]
    tops = [top for top, _ in bands]
    for index in range(len(bands) - 1, -1, -1):
        radius = upstream - widths[index]
        bottom = base if index == len(bands) - 1 else tops[index + 1]
        outline += [(radius, bottom), (radius, tops[index])]
    figure, axes = plt.subplots(1, 2, figsize=(9, 7.5), dpi=160, sharey=True)
    for axis, with_fill in zip(axes, (False, True)):
        xs, ys = zip(*outline)
        axis.fill([upstream - x for x in xs], ys, color=CONCRETE_COLOUR, edgecolor="0.25", linewidth=1.0)
        if with_fill:
            for step in fill.report["steps"]:
                source = next(item for item in definition["steps"] if item["top_z_m"] == step["top_z_m"])
                face, edge = upstream - source["wall_face_radius_m"], upstream - source["ledge_edge_radius_m"]
                axis.fill([face, edge, face], [step["top_z_m"], step["top_z_m"], step["top_of_fill_z_m"]],
                          color=MORTAR_COLOUR, edgecolor="0.25", linewidth=1.0)
        axis.set_aspect("equal")
        axis.set_xlim(-1.0, max(widths) + 1.5)
        axis.set_ylim(base - 1.0, 1.5)
        axis.set_xlabel("Distance from the upstream face (m)")
        axis.set_title("As analysed" if not with_fill else f"With mortar fill (1 in {1 / slope:g})", fontsize=11)
        axis.grid(True, color="0.9", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Elevation relative to the crest (m)")
    figure.suptitle(f"{label}: section at the deepest point. Horizontal and vertical to the same scale.", fontsize=11)
    figure.tight_layout()
    section_path = output / f"{name}-section.png"
    figure.savefig(section_path)
    plt.close(figure)
    return {"downstream": str(downstream_path), "section": str(section_path), "mortar_volume_m3": fill.report["volume_m3"],
            "requested_slope_run_per_rise": slope, "built_slope_run_per_rise": built,
            "wall_element_size_m": wall.element_size_target_m, "steps": fill.report["steps"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True, help="file name stem of the pictures")
    parser.add_argument("--label", help="text at the top of the pictures (default: the name)")
    arguments = parser.parse_args()
    print(json.dumps(render(arguments.config, arguments.output, arguments.name, arguments.label), indent=2))


if __name__ == "__main__":
    main()
