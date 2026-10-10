"""Validate scenarios before writing any simulation output."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re


def positive(value: object, label: str, *, allow_zero: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        raise ValueError(f"{label} must be a finite {'non-negative' if allow_zero else 'positive'} number")


def keys(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly: {', '.join(sorted(expected))}")
    return value


def load_scenarios(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("cracking.json must contain a nonempty scenario array")
    identifiers = set()
    for scenario in data:
        if not isinstance(scenario, dict):
            raise ValueError("Each scenario must be an object")
        identifier = scenario.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", identifier):
            raise ValueError("Scenario id must be a safe lowercase name of at most 64 characters")
        if identifier in identifiers:
            raise ValueError(f"Duplicate scenario id: {identifier}")
        identifiers.add(identifier)
        if not isinstance(scenario.get("enabled"), bool):
            raise ValueError(f"{identifier}: enabled must be true or false")
        kind = scenario.get("kind")
        if kind == "elastic_benchmark":
            keys(scenario, {"id", "enabled", "kind", "mesh_divisions", "axial_strain",
                            "maximum_relative_l2_error", "solver"}, identifier)
            divisions = scenario["mesh_divisions"]
            if isinstance(divisions, bool) or not isinstance(divisions, int) or not 2 <= divisions <= 64:
                raise ValueError(f"{identifier}: mesh_divisions must be an integer between 2 and 64")
            for name in ("axial_strain", "maximum_relative_l2_error"):
                positive(scenario[name], f"{identifier}.{name}")
            solver = keys(scenario["solver"], {"relative_tolerance", "maximum_iterations"}, "solver")
            positive(solver["relative_tolerance"], "solver.relative_tolerance")
            if solver["relative_tolerance"] >= 1:
                raise ValueError("solver.relative_tolerance must be less than 1")
            iterations = solver["maximum_iterations"]
            if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
                raise ValueError("solver.maximum_iterations must be a positive integer")
        elif kind == "contact_benchmark":
            keys(scenario, {"id", "enabled", "kind", "bar_lengths_m", "cross_section_m2", "initial_gap_m",
                            "mesh_divisions", "outer_closures_m", "verification"}, identifier)
            lengths = scenario["bar_lengths_m"]
            if not isinstance(lengths, list) or len(lengths) != 2:
                raise ValueError("bar_lengths_m must contain two positive lengths")
            for length in lengths:
                positive(length, "bar_lengths_m")
            positive(scenario["cross_section_m2"], "cross_section_m2")
            positive(scenario["initial_gap_m"], "initial_gap_m", allow_zero=True)
            divisions = scenario["mesh_divisions"]
            if not isinstance(divisions, list) or len(divisions) < 2 or any(
                isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 128 for n in divisions
            ) or len(set(divisions)) != len(divisions):
                raise ValueError("mesh_divisions must contain at least two distinct integers between 1 and 128")
            closures = scenario["outer_closures_m"]
            if not isinstance(closures, list) or not closures or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in closures
            ):
                raise ValueError("outer_closures_m must contain finite prescribed displacements")
            gap = scenario["initial_gap_m"]
            if not (any(c < gap for c in closures) and any(c > gap for c in closures)):
                raise ValueError("Contact benchmark sequence must test both open and closed states")
            if not any(c < gap for c in closures[closures.index(max(closures)) + 1:]):
                raise ValueError("Contact benchmark sequence must unload to an open state after maximum closure")
            verification = keys(scenario["verification"], {"gap_tolerance_m", "displacement_tolerance_m",
                                                            "relative_tolerance", "maximum_active_set_iterations"}, "verification")
            for name in ("gap_tolerance_m", "displacement_tolerance_m", "relative_tolerance"):
                positive(verification[name], f"verification.{name}")
            if verification["relative_tolerance"] >= 1:
                raise ValueError("verification.relative_tolerance must be less than 1")
            iterations = verification["maximum_active_set_iterations"]
            if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
                raise ValueError("maximum_active_set_iterations must be a positive integer")
        elif kind == "dam_elastic":
            keys(scenario, {"id", "enabled", "kind", "reference", "maximum_relative_volume_change",
                            "solver", "parity"}, identifier)
            if not isinstance(scenario["reference"], str) or not scenario["reference"].endswith(".pvtu"):
                raise ValueError("reference must name an Elmer .pvtu file relative to the repository")
            reference = Path(scenario["reference"])
            if reference.is_absolute() or ".." in reference.parts or "\\" in scenario["reference"]:
                raise ValueError("reference must be a repository-relative path using / separators")
            positive(scenario["maximum_relative_volume_change"], "maximum_relative_volume_change", allow_zero=True)
            solver = keys(scenario["solver"], {"relative_tolerance", "maximum_iterations"}, "solver")
            positive(solver["relative_tolerance"], "solver.relative_tolerance")
            if solver["relative_tolerance"] >= 1:
                raise ValueError("solver.relative_tolerance must be less than 1")
            iterations = solver["maximum_iterations"]
            if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
                raise ValueError("solver.maximum_iterations must be a positive integer")
            parity = keys(scenario["parity"], {"maximum_relative_l2_difference", "maximum_probe_relative_difference",
                                               "maximum_force_balance_error", "probes"}, "parity")
            for name in ("maximum_relative_l2_difference", "maximum_probe_relative_difference", "maximum_force_balance_error"):
                positive(parity[name], f"parity.{name}")
            if not isinstance(parity["probes"], list) or not parity["probes"]:
                raise ValueError("parity.probes must be a nonempty array")
            names = set()
            for probe in parity["probes"]:
                keys(probe, {"name", "region_id", "point_m", "maximum_distance_m"}, "probe")
                if not isinstance(probe["name"], str) or not probe["name"] or probe["name"] in names:
                    raise ValueError("Probe names must be nonempty and unique")
                names.add(probe["name"])
                if isinstance(probe["region_id"], bool) or probe["region_id"] not in (1, 2):
                    raise ValueError("Probe region_id must be 1 (plinth) or 2 (wall)")
                point = probe["point_m"]
                if not isinstance(point, list) or len(point) != 3 or any(
                    isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in point
                ):
                    raise ValueError("Probe point_m must be three finite coordinates in metres")
                positive(probe["maximum_distance_m"], "probe.maximum_distance_m")
        elif kind == "predefined_crack":
            keys(scenario, {"id", "enabled", "kind", "crack", "contact"}, identifier)
            crack = scenario["crack"]
            if not isinstance(crack, dict):
                raise ValueError(f"{identifier}: crack must be an object")
            orientation = crack.get("orientation")
            common = {"orientation", "chainage_m", "depth_m", "initial_opening_mm"}
            if orientation == "horizontal":
                keys(crack, common | {"z_m", "width_m"}, "crack")
                positive(crack["width_m"], "crack.width_m")
                elevations = ("z_m",)
            elif orientation == "vertical":
                keys(crack, common | {"z_lo_m", "z_hi_m"}, "crack")
                elevations = ("z_lo_m", "z_hi_m")
            else:
                raise ValueError("crack.orientation must be horizontal or vertical")
            for name in elevations:
                value = crack[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"crack.{name} must be a finite elevation in metres")
            if orientation == "vertical" and crack["z_lo_m"] >= crack["z_hi_m"]:
                raise ValueError("crack.z_lo_m must be below crack.z_hi_m")
            positive(crack["depth_m"], "crack.depth_m")
            positive(crack["chainage_m"], "crack.chainage_m", allow_zero=True)
            positive(crack["initial_opening_mm"], "crack.initial_opening_mm", allow_zero=True)
            contact = scenario["contact"]
            if not isinstance(contact, dict):
                raise ValueError("contact must be an object")
            model = contact.get("model")
            expected = {"model", "water_pressure_on_faces"}
            if model == "coulomb":
                expected.add("friction_coefficient")
            elif model != "frictionless":
                raise ValueError("contact.model must be frictionless or coulomb")
            keys(contact, expected, "contact")
            if model == "coulomb":
                positive(contact["friction_coefficient"], "contact.friction_coefficient", allow_zero=True)
            if not isinstance(contact["water_pressure_on_faces"], bool):
                raise ValueError("contact.water_pressure_on_faces must be true or false")
        else:
            raise ValueError(f"{identifier}: unknown scenario kind {kind!r}")
    return data
