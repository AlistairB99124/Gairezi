"""Element-count budgeting for the wall body's tetrahedral analysis mesh."""
from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from fractions import Fraction
import math
from math import gcd, lcm


TARGET_WALL_TETRAHEDRAL_ELEMENTS = 150_000
MINIMUM_WALL_TETRAHEDRAL_TARGET = 1_000
TETRAHEDRA_PER_SOURCE_CELL = {4: 1, 5: 6, 6: 3}


def wall_tetrahedral_target(geometry: dict) -> int:
    """Configured wall tetrahedron cap (geometry.target_wall_tetrahedral_elements), else the default."""
    value = geometry.get("target_wall_tetrahedral_elements", TARGET_WALL_TETRAHEDRAL_ELEMENTS)
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not float(value).is_integer() or value < MINIMUM_WALL_TETRAHEDRAL_TARGET):
        raise ValueError("geometry.target_wall_tetrahedral_elements must be an integer of at least "
                         f"{MINIMUM_WALL_TETRAHEDRAL_TARGET:,}")
    return int(value)


def config_for_wall_element_size(config: dict, size_m: float) -> dict:
    """Copy mesh settings so attached bodies use the wall's effective interface size."""
    if not math.isfinite(size_m) or size_m <= 0:
        raise ValueError("Element size must be finite and positive")
    result = deepcopy(config)
    geometry = result["geometry"]
    geometry["global_element_size_m"] = size_m
    bedrock_mesh = geometry["bedrock"].get("mesh")
    if bedrock_mesh is not None:
        bedrock_mesh["near_size_m"] = size_m
        bedrock_mesh["far_size_m"] = max(bedrock_mesh["far_size_m"], size_m)
    return result


def tetrahedral_element_count(cells: Iterable[tuple[int, tuple[int, ...]]]) -> int:
    """Count analysis tetrahedra without allocating the subdivided mesh."""
    try:
        return sum(TETRAHEDRA_PER_SOURCE_CELL[element_type] for element_type, _ in cells)
    except KeyError as error:
        raise ValueError(f"Unsupported source volume element type {error.args[0]}") from error


def size_for_element_target(current_size_m: float, current_count: int, target_count: int) -> float:
    """Estimate a uniform size using the cubic relation between size and cell count."""
    if (
        not math.isfinite(current_size_m)
        or current_size_m <= 0
        or type(current_count) is not int
        or current_count <= 0
        or type(target_count) is not int
        or target_count <= 0
    ):
        raise ValueError("Mesh size and element counts must be finite and positive")
    return current_size_m * (current_count / target_count) ** (1.0 / 3.0)


def course_compatible_size(
    size_m: float,
    step_levels_m: Iterable[float],
    *,
    not_coarser_than: bool,
) -> float:
    """Snap a size to a course spacing that also contains every wall step."""
    if not math.isfinite(size_m) or size_m <= 0:
        raise ValueError("Element size must be finite and positive")

    levels = [Fraction(str(abs(level))) for level in step_levels_m if level]
    if not levels:
        return size_m

    denominator = lcm(*(level.denominator for level in levels))
    common_units = [
        level.numerator * (denominator // level.denominator)
        for level in levels
    ]
    course_unit = Fraction(math.gcd(*common_units), denominator)
    ratio = float(course_unit) / size_m
    divisions = (
        math.ceil(ratio - 1e-12)
        if not_coarser_than
        else math.floor(ratio + 1e-12)
    )
    if divisions < 1:
        raise ValueError(
            "The wall element target cannot be met while keeping thickness steps "
            "on the wall course grid"
        )
    return float(course_unit / divisions)
