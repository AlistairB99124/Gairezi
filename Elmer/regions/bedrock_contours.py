"""Surveyed bedrock profile along the dam axis and the wall extent it implies.

Elevations are metres relative to the crest (z = 0). The wall sits directly on
the bedrock wherever the rock lies below the crest.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from ..model_config import load_config

CREST_Z_M = 0.0


@dataclass(frozen=True)
class ContourPoint:
    chainage_m: float
    bedrock_z_m: float


def bedrock_elevation(rows: list[dict], chainage_m: float) -> float:
    """Piecewise-linear surveyed rock elevation, without mesh-level snapping."""
    for left, right in zip(rows, rows[1:]):
        start, end = float(left["chainage"]), float(right["chainage"])
        if start - 1e-8 <= chainage_m <= end + 1e-8:
            fraction = min(1., max(0., (chainage_m - start) / (end - start)))
            return float(left["height"]) + fraction * (float(right["height"]) - float(left["height"]))
    raise ValueError(f"Chainage {chainage_m:g} is outside the bedrock contour range")


def wall_extent(rows: list[dict]) -> tuple[float, float]:
    """Chainages where the rock profile first and last reaches the crest.

    If an end station is still below the crest the wall stops there, with a
    vertical abutment face.
    """
    stations = [(float(row["chainage"]), float(row["height"])) for row in rows]
    below = [index for index, (_, height) in enumerate(stations) if height < CREST_Z_M]
    if not below:
        raise ValueError("Bedrock contours must dip below the crest (z = 0)")

    def crossing(inside: int, outside: int) -> float:
        (c_in, z_in), (c_out, z_out) = stations[inside], stations[outside]
        return c_in + (c_out - c_in) * (CREST_Z_M - z_in) / (z_out - z_in)

    first, last = below[0], below[-1]
    start = stations[0][0] if first == 0 else crossing(first, first - 1)
    end = stations[-1][0] if last == len(stations) - 1 else crossing(last, last + 1)
    if any(height > CREST_Z_M for chainage, height in stations if start < chainage < end):
        raise ValueError("Bedrock rises above the crest inside the wall; split valleys are unsupported")
    return start, end


def load_contours(path: Path, crest_raise_m: float = 0.0) -> list[ContourPoint]:
    """Rock profile over the wall extent, rebased to a raised crest datum (m)."""
    if not math.isfinite(crest_raise_m):
        raise ValueError("Crest raise must be finite")
    rows = load_config(path)["geometry"]["bedrock"]["contours"]
    rebased = [{"chainage": row["chainage"], "height": float(row["height"]) - crest_raise_m} for row in rows]
    start, end = wall_extent(rebased)
    chainages = [start, *(float(row["chainage"]) for row in rebased if start < float(row["chainage"]) < end), end]
    return [ContourPoint(value, bedrock_elevation(rebased, value)) for value in chainages]


def load_model_contours(root: Path) -> list[ContourPoint]:
    return load_contours(root / "config.json")


def load_global_element_size(path: Path) -> float:
    return float(load_config(path)["geometry"]["global_element_size_m"])
