"""Render the lower wall section with both outer columns flared."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from regions.left_wedged_wall import (
    DOWNSTREAM_WALL_RADIUS_M,
    DOWNSTREAM_WEDGE_INNER_RADIUS_M,
    DOWNSTREAM_WEDGE_TOE_RADIUS_M,
    UPSTREAM_RADIUS_M,
    UPSTREAM_WEDGE_INNER_RADIUS_M,
    WEDGE_HEIGHT_M,
    _wedge_section_faces,
)
from regions.plinth import DOWNSTREAM_RADIUS_M as PLINTH_DOWNSTREAM_RADIUS_M
from regions.plinth import UPSTREAM_RADIUS_M as PLINTH_UPSTREAM_RADIUS_M


OUTPUT_PATH = Path(__file__).resolve().parent / "structure_output" / "paired_wedge_4m_section.png"
ELEMENT_SIZE_M = 0.5


def main() -> None:
    wedge_faces = _wedge_section_faces(ELEMENT_SIZE_M)
    figure, axes = plt.subplots(figsize=(8.4, 6.2), dpi=180)

    axes.add_patch(Rectangle(
        (PLINTH_DOWNSTREAM_RADIUS_M, -ELEMENT_SIZE_M),
        PLINTH_UPSTREAM_RADIUS_M - PLINTH_DOWNSTREAM_RADIUS_M,
        ELEMENT_SIZE_M,
        facecolor="#8e9a9c",
        edgecolor="#263238",
        linewidth=1.4,
        label="Plinth",
    ))

    for index, face in enumerate(wedge_faces):
        downstream = max(radius_m for radius_m, _ in face) <= DOWNSTREAM_WEDGE_INNER_RADIUS_M
        axes.add_patch(Polygon(
            face,
            closed=True,
            facecolor="#d97745" if downstream else "#4f86a8",
            edgecolor="#263238",
            linewidth=0.75,
            label=("Downstream flare" if downstream else "Upstream flare") if index < 2 else None,
        ))

    radial_divisions = round((UPSTREAM_WEDGE_INNER_RADIUS_M - DOWNSTREAM_WEDGE_INNER_RADIUS_M) / ELEMENT_SIZE_M)
    vertical_divisions = round(WEDGE_HEIGHT_M / ELEMENT_SIZE_M)
    for radial_index in range(radial_divisions):
        for vertical_index in range(vertical_divisions):
            axes.add_patch(Rectangle(
                (
                    DOWNSTREAM_WEDGE_INNER_RADIUS_M + radial_index * ELEMENT_SIZE_M,
                    vertical_index * ELEMENT_SIZE_M,
                ),
                ELEMENT_SIZE_M,
                ELEMENT_SIZE_M,
                facecolor="#d8d5cb",
                edgecolor="#596064",
                linewidth=0.55,
            ))

    axes.add_patch(Rectangle(
        (DOWNSTREAM_WALL_RADIUS_M, WEDGE_HEIGHT_M),
        UPSTREAM_RADIUS_M - DOWNSTREAM_WALL_RADIUS_M,
        2.0,
        facecolor="#d8d5cb",
        edgecolor="#263238",
        linewidth=1.4,
        label="Wall above flare",
    ))
    upper_radial_divisions = round((UPSTREAM_RADIUS_M - DOWNSTREAM_WALL_RADIUS_M) / ELEMENT_SIZE_M)
    for radial_index in range(1, upper_radial_divisions):
        radius_m = DOWNSTREAM_WALL_RADIUS_M + radial_index * ELEMENT_SIZE_M
        axes.plot([radius_m, radius_m], [WEDGE_HEIGHT_M, WEDGE_HEIGHT_M + 2.0], color="#596064", linewidth=0.55)
    for vertical_index in range(1, 4):
        height_m = WEDGE_HEIGHT_M + vertical_index * ELEMENT_SIZE_M
        axes.plot([DOWNSTREAM_WALL_RADIUS_M, UPSTREAM_RADIUS_M], [height_m, height_m], color="#596064", linewidth=0.55)

    axes.annotate(
        "1.0 m downstream overhang",
        xy=((PLINTH_DOWNSTREAM_RADIUS_M + DOWNSTREAM_WEDGE_TOE_RADIUS_M) / 2.0, -0.25),
        xytext=(74.0, -1.15),
        arrowprops={"arrowstyle": "->", "color": "#263238"},
        fontsize=9,
    )
    axes.annotate(
        "3.0 m vertical core",
        xy=((DOWNSTREAM_WEDGE_INNER_RADIUS_M + UPSTREAM_WEDGE_INNER_RADIUS_M) / 2.0, 2.0),
        ha="center",
        va="center",
        fontsize=10,
    )
    axes.annotate(
        "4.0 m flare height",
        xy=(81.25, WEDGE_HEIGHT_M / 2.0),
        ha="left",
        va="center",
        rotation=90,
        fontsize=9,
    )

    axes.set_title("Lower wall section: paired outer-column flares")
    axes.set_xlabel("Radius (m), downstream to upstream")
    axes.set_ylabel("Height above plinth (m)")
    axes.set_xlim(73.7, 82.0)
    axes.set_ylim(-1.4, 6.35)
    axes.set_aspect("equal")
    axes.grid(False)
    axes.legend(loc="upper left", frameon=False, ncols=2)
    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH)
    plt.close(figure)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()