"""Backend-neutral inputs and opening-limit assessment; not a fracture solver.

This contract deliberately cannot be consumed as a legacy damaged-volume seed.
Hydraulic boundary conditions and material-law parameters belong in the eventual
versioned backend preparation, not in guessed defaults here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from Elmer.model_config import ConfigurationValidationError, ValidationIssue


DEFAULT_OPENING_LIMIT_M = 0.002
SCENARIO_KIND = "intact_horizontal_cohesive"


@dataclass(frozen=True)
class DamageSelection:
    """Selected IDs in a job's mesh catalog; not a legacy single-point seed."""

    affected_element_ids: tuple[int, ...]
    crack_start_edge_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        issues = []
        for name, label in (
            ("affected_element_ids", "affected concrete element"),
            ("crack_start_edge_ids", "crack-start edge"),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                issues.append(ValidationIssue(name, "must be a tuple of mesh IDs"))
            elif not values:
                issues.append(ValidationIssue(name, f"select at least one {label}"))
            elif any(type(value) is not int or value < 0 for value in values):
                issues.append(ValidationIssue(name, "must contain nonnegative integer mesh IDs"))
            elif len(set(values)) != len(values):
                issues.append(ValidationIssue(name, "must not contain duplicate mesh IDs"))
        if issues:
            raise ConfigurationValidationError(issues)

    def to_dict(self) -> dict:
        return {
            "affected_element_ids": list(self.affected_element_ids),
            "crack_start_edge_ids": list(self.crack_start_edge_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> DamageSelection:
        keys = {"affected_element_ids", "crack_start_edge_ids"}
        if not isinstance(value, dict):
            raise ConfigurationValidationError([
                ValidationIssue("selection", "must be an object"),
            ])
        issues = []
        for name in sorted(keys - value.keys()):
            issues.append(ValidationIssue(name, "is required"))
        for name in sorted(value.keys() - keys):
            issues.append(ValidationIssue(str(name), "is not a supported selection field"))
        for name in sorted(keys & value.keys()):
            if not isinstance(value[name], list):
                issues.append(ValidationIssue(name, "must be an array of mesh IDs"))
        if issues:
            raise ConfigurationValidationError(issues)
        return cls(tuple(value["affected_element_ids"]), tuple(value["crack_start_edge_ids"]))


def _number(
    value: object, name: str, *, positive: bool = False, signed: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    if (not signed and value < 0) or (positive and value <= 0):
        requirement = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {requirement}")
    return float(value)


@dataclass(frozen=True)
class HorizontalFractureRequest:
    """A candidate plane, not an instruction to damage a finite seed volume."""

    selected_y_m: float
    selected_z_m: float
    friction_coefficient: float
    opening_limit_m: float = DEFAULT_OPENING_LIMIT_M

    def __post_init__(self) -> None:
        for name in ("selected_y_m", "selected_z_m"):
            _number(getattr(self, name), name, signed=True)
        _number(self.friction_coefficient, "friction_coefficient")
        _number(self.opening_limit_m, "opening_limit_m", positive=True)

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "scenario_kind": SCENARIO_KIND,
            "selected_location_m": {"y": self.selected_y_m, "z": self.selected_z_m},
            "friction_coefficient": self.friction_coefficient,
            "opening_limit_m": self.opening_limit_m,
            "initial_condition": "intact",
            "fracture_path": "horizontal_concrete",
            "hydraulic_model": "transient_fracture_flow",
            "bulk_seepage": False,
        }

    @classmethod
    def from_dict(cls, value: object) -> HorizontalFractureRequest:
        if not isinstance(value, dict):
            raise ValueError("Horizontal fracture request must be an object")
        constants = {
            "schema_version": 1,
            "scenario_kind": SCENARIO_KIND,
            "initial_condition": "intact",
            "fracture_path": "horizontal_concrete",
            "hydraulic_model": "transient_fracture_flow",
            "bulk_seepage": False,
        }
        keys = set(constants) | {
            "selected_location_m", "friction_coefficient", "opening_limit_m",
        }
        if set(value) != keys:
            raise ValueError(
                f"Horizontal fracture fields mismatch: missing={sorted(keys - set(value))}, "
                f"unsupported={sorted(set(value) - keys)}"
            )
        for name, expected in constants.items():
            if type(value[name]) is not type(expected) or value[name] != expected:
                raise ValueError(f"{name} must be {expected!r}")
        location = value["selected_location_m"]
        if not isinstance(location, dict) or set(location) != {"y", "z"}:
            raise ValueError("selected_location_m must contain only y and z coordinates")
        return cls(location["y"], location["z"], value["friction_coefficient"],
                   value["opening_limit_m"])


def _validate_opening_measurement(maximum_opening_m, location_m, fracture_initiated):
    _number(maximum_opening_m, "maximum_opening_m")
    if not isinstance(fracture_initiated, bool):
        raise ValueError("fracture_initiated must be a boolean")
    if not fracture_initiated and maximum_opening_m != 0:
        raise ValueError("A state with no fracture initiation cannot report crack opening")
    if not isinstance(location_m, tuple) or len(location_m) != 3:
        raise ValueError("location_m must be a three-coordinate tuple")
    for value in location_m:
        _number(value, "location_m coordinate", signed=True)


@dataclass(frozen=True)
class OpeningObservation:
    """A measured maximum opening from an accepted state, not scalar damage."""

    physical_time_s: float
    maximum_opening_m: float
    location_m: tuple[float, float, float]
    fracture_initiated: bool

    def __post_init__(self) -> None:
        _number(self.physical_time_s, "physical_time_s")
        _validate_opening_measurement(self.maximum_opening_m, self.location_m, self.fracture_initiated)


OpeningLimitStatus = Literal["below_limit", "limit_reached", "limit_exceeded", "not_assessed"]


@dataclass(frozen=True)
class EquilibriumOpeningObservation:
    """Measured opening at an accepted continuation state, not physical time."""

    state_id: str
    maximum_opening_m: float
    location_m: tuple[float, float, float]
    fracture_initiated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id.strip():
            raise ValueError("state_id must be a nonempty string")
        _validate_opening_measurement(self.maximum_opening_m, self.location_m, self.fracture_initiated)


def assess_equilibrium_opening(
    opening_limit_m: float,
    observations: tuple[EquilibriumOpeningObservation, ...],
    *,
    completed: bool,
) -> dict:
    """Warn on accepted-state peak; continuation IDs never imply seconds."""
    limit = _number(opening_limit_m, "opening_limit_m", positive=True)
    if not isinstance(completed, bool):
        raise ValueError("completed must be a boolean")
    if not observations or any(not isinstance(item, EquilibriumOpeningObservation) for item in observations):
        raise ValueError("Equilibrium assessment requires accepted EquilibriumOpeningObservation records")
    ids = [item.state_id for item in observations]
    if len(set(ids)) != len(ids):
        raise ValueError("Accepted equilibrium state IDs must be unique")
    initiated = False
    for item in observations:
        if initiated and not item.fracture_initiated:
            raise ValueError("Fracture initiation cannot be reversed in the accepted history")
        initiated = initiated or item.fracture_initiated
    peak = max(observations, key=lambda item: item.maximum_opening_m)
    return {
        **_opening_assessment(limit, peak.maximum_opening_m, peak.location_m, initiated,
                              completed, f"at accepted state {peak.state_id!r}"),
        "scenario_kind": "painted_tensile_loss",
        "state_axis": "accepted_equilibrium_states_not_physical_time",
        "peak_opening_state_id": peak.state_id,
        "final_state_id": observations[-1].state_id,
        "final_opening_m": observations[-1].maximum_opening_m,
    }


def _opening_assessment(limit, opening, location, initiated, completed, peak_description):
    reached = opening >= limit
    status: OpeningLimitStatus = (
        "not_assessed" if not completed else
        "limit_exceeded" if opening > limit else
        "limit_reached" if reached else "below_limit"
    )
    warnings = []
    if reached:
        warnings.append({
            "path": "opening_limit_m",
            "message": (
                f"Calculated peak crack opening {opening:.9g} m "
                f"reached or exceeded the opening limit {limit:.9g} m "
                f"{peak_description}. "
                "This is an opening-limit warning, not a solver error or a collapse verdict."
            ),
        })
    return {
        "completed": completed, "fracture_initiated": initiated,
        "opening_limit_m": limit, "peak_opening_m": opening,
        "peak_opening_location_m": list(location),
        "observed_limit_reached": reached, "opening_limit_status": status,
        "opening_criterion_satisfied": (not reached) if completed else None,
        "warnings": warnings, "engineering_validated": False,
    }


def assess_opening(
    request: HorizontalFractureRequest,
    observations: tuple[OpeningObservation, ...],
    *,
    completed: bool,
) -> dict:
    """Compatibility wrapper for the superseded horizontal request contract."""
    return assess_opening_history(
        request.opening_limit_m, observations, completed=completed, scenario_kind=SCENARIO_KIND,
    )


def assess_opening_history(
    opening_limit_m: float,
    observations: tuple[OpeningObservation, ...],
    *,
    completed: bool,
    scenario_kind: str = "painted_tensile_loss",
) -> dict:
    """Warn on peak opening at the limit without changing solver completion."""
    limit = _number(opening_limit_m, "opening_limit_m", positive=True)
    if not isinstance(scenario_kind, str) or not scenario_kind.strip():
        raise ValueError("scenario_kind must be a nonempty string")
    if not isinstance(completed, bool):
        raise ValueError("completed must be a boolean")
    if not observations:
        raise ValueError("Opening assessment requires at least one accepted observation")
    if any(not isinstance(item, OpeningObservation) for item in observations):
        raise ValueError("Opening history must contain accepted OpeningObservation records")
    times = [item.physical_time_s for item in observations]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("Accepted physical times must be strictly increasing")
    initiated = False
    for item in observations:
        if initiated and not item.fracture_initiated:
            raise ValueError("Fracture initiation cannot be reversed in the accepted history")
        initiated = initiated or item.fracture_initiated
    peak = max(observations, key=lambda item: item.maximum_opening_m)
    return {
        **_opening_assessment(limit, peak.maximum_opening_m, peak.location_m, initiated,
                              completed, f"at physical time {peak.physical_time_s:.9g} s"),
        "scenario_kind": scenario_kind,
        "peak_opening_time_s": peak.physical_time_s,
    }
