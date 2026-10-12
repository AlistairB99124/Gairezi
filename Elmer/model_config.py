"""Load and validate the root dam configuration shared by mesh and solver builders."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"

CONFIG_KEYS = {
    "geometry": {"global_element_size_m", "local_element_size_m", "wall", "plinth", "bedrock"},
    "material": {"concrete", "bedrock"},
    "loads": {
        "gravity",
        "water_height",
        "tailwater_head",
        "overflow_head",
        "overflow_water_level",
        "water_density",
    },
    "outputs": {"max_displacement", "max_tensile_stress", "cracking_risk"},
}
OPTIONAL_CONFIG_KEYS = {"crack_simulations"}
MATERIAL_KEYS = {"density", "youngs_modulus", "poisson_ratio", "tensile_strength_pa"}
RADIUS_KEYS = {"radial_width_m", "upstream_radius_m", "centerline_radius_m", "downstream_radius_m"}
SUPPORT_KEYS = {"model", "material", "spring_n_per_m3"}
CRACK_KEYS = {
    "type",
    "shape",
    "central_crack_pos_m",
    "size_m",
    "orientation",
    "phase_field_parameters",
    "loading",
    "solver",
}
VECTOR_KEYS = {"x", "y", "z"}
ORIENTATION_KEYS = {"normal_x", "normal_y", "normal_z"}
PHASE_FIELD_KEYS = {
    "model",
    "energy_split",
    "length_scale_m",
    "critical_energy_release_rate_J_m2",
    "residual_stiffness",
}
CRACK_LOADING_KEYS = {"internal_pressure_pa", "load_steps"}
PHASE_FIELD_SOLVER_KEYS = {
    "maximum_staggered_iterations",
    "staggered_tolerance",
    "mechanics_relative_tolerance",
    "mechanics_maximum_iterations",
    "damage_gradient_tolerance",
    "damage_maximum_iterations",
    "field_output_interval",
    "minimum_elements_per_length_scale",
    "maximum_refinement_levels",
}
CRACK_TYPES = {"phase_field"}
CRACK_SHAPES = {"rectangular"}
PHASE_FIELD_MODELS = {"AT2"}
ENERGY_SPLITS = {"volumetric_deviatoric"}
SUPPORT_MODELS = {
    "plinth": ("distributed_elastic_concrete", "concrete"),
    "bedrock": ("distributed_elastic_granite", "bedrock"),
}


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


class ConfigurationValidationError(ValueError):
    """A configuration error with paths suitable for mapping back to UI controls."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"{issue.path}: {issue.message}" for issue in issues))


class _ConfigValidator:
    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    def issue(self, path: str, message: str) -> None:
        self.issues.append(ValidationIssue(path, message))

    def object(
        self,
        value: object,
        path: str,
        keys: set[str],
        optional: set[str] | None = None,
    ) -> Mapping[str, object] | None:
        if not isinstance(value, dict):
            self.issue(path, "must be an object")
            return None
        for key in sorted(keys - value.keys()):
            self.issue(f"{path}.{key}" if path else key, "is required")
        for key in sorted(value.keys() - keys - (optional or set())):
            self.issue(f"{path}.{key}" if path else str(key), "is not a supported field")
        return value

    def number(
        self,
        value: object,
        path: str,
        *,
        positive: bool = False,
        nonnegative: bool = False,
    ) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.issue(path, "must be a number")
            return None
        if not math.isfinite(value):
            self.issue(path, "must be finite")
            return None
        result = float(value)
        if positive and result <= 0:
            self.issue(path, "must be positive")
        if nonnegative and result < 0:
            self.issue(path, "must be nonnegative")
        return result

    def field_number(
        self,
        values: Mapping[str, object] | None,
        key: str,
        path: str,
        **constraints: bool,
    ) -> float | None:
        if values is None or key not in values:
            return None
        return self.number(values[key], f"{path}.{key}", **constraints)

    def validate_radii(
        self,
        values: Mapping[str, object] | None,
        path: str,
    ) -> tuple[float | None, float | None]:
        if values is None:
            return None, None
        upstream = self.field_number(values, "upstream_radius_m", path, positive=True)
        downstream = self.field_number(values, "downstream_radius_m", path, positive=True)
        width = self.field_number(values, "radial_width_m", path, positive=True)
        centerline = self.field_number(values, "centerline_radius_m", path, positive=True)
        if upstream is not None and downstream is not None and width is not None:
            if not math.isclose(upstream - downstream, width, abs_tol=1e-8):
                self.issue(
                    f"{path}.radial_width_m",
                    "must equal upstream_radius_m - downstream_radius_m",
                )
        if upstream is not None and downstream is not None and centerline is not None:
            if not math.isclose(centerline, 0.5 * (upstream + downstream), abs_tol=1e-8):
                self.issue(
                    f"{path}.centerline_radius_m",
                    "must be midway between upstream_radius_m and downstream_radius_m",
                )
        return upstream, downstream

    def validate_material(self, material: Mapping[str, object] | None) -> None:
        for name in ("concrete", "bedrock"):
            path = f"material.{name}"
            values = self.object(material.get(name) if material is not None else None, path, MATERIAL_KEYS)
            if values is None:
                continue
            for key in ("density", "youngs_modulus", "tensile_strength_pa"):
                self.field_number(values, key, path, positive=True)
            poisson = self.field_number(values, "poisson_ratio", path)
            if poisson is not None and not -1 < poisson < 0.5:
                self.issue(f"{path}.poisson_ratio", "must lie between -1 and 0.5")

    def validate_contours(
        self,
        geometry: Mapping[str, object] | None,
        region: str,
    ) -> list[tuple[float, float]] | None:
        path = f"geometry.{region}.contours"
        region_values = geometry.get(region) if geometry is not None else None
        if not isinstance(region_values, dict):
            return None
        rows = region_values.get("contours")
        if not isinstance(rows, list):
            self.issue(path, "must be an array")
            return None
        if len(rows) < 2:
            self.issue(path, "must contain at least two stations")
        stations: list[tuple[float, float]] = []
        for index, row in enumerate(rows):
            row_path = f"{path}[{index}]"
            values = self.object(row, row_path, {"chainage", "height"})
            if values is None:
                continue
            chainage = self.field_number(values, "chainage", row_path)
            height = self.field_number(values, "height", row_path)
            if chainage is not None and height is not None:
                stations.append((chainage, height))
        if len(stations) != len(rows):
            return None
        chainages = [station[0] for station in stations]
        if any(left >= right for left, right in zip(chainages, chainages[1:])):
            self.issue(path, "chainages must be strictly increasing and unique")
        return stations

    def validate_support(
        self,
        geometry: Mapping[str, object] | None,
        region: str,
    ) -> None:
        path = f"geometry.{region}.support"
        region_values = geometry.get(region) if geometry is not None else None
        support_value = region_values.get("support") if isinstance(region_values, dict) else None
        values = self.object(support_value, path, SUPPORT_KEYS)
        if values is None:
            return
        expected_model, expected_material = SUPPORT_MODELS[region]
        for key, expected in (("model", expected_model), ("material", expected_material)):
            value = values.get(key)
            if key in values and value != expected:
                self.issue(f"{path}.{key}", f"must be {expected!r} for {region} support")
        springs = self.object(values.get("spring_n_per_m3"), f"{path}.spring_n_per_m3", {"x", "y", "z"})
        if springs is not None:
            for direction in ("x", "y", "z"):
                self.field_number(springs, direction, f"{path}.spring_n_per_m3", nonnegative=True)

    def validate_cracks(self, value: object) -> None:
        """Validate declared crack simulations without reinterpreting their physical inputs."""
        if not isinstance(value, list) or not value:
            self.issue("crack_simulations", "must be a nonempty array")
            return
        for index, entry in enumerate(value):
            path = f"crack_simulations[{index}]"
            crack = self.object(entry, path, CRACK_KEYS)
            if crack is None:
                continue
            if "type" in crack and crack["type"] not in CRACK_TYPES:
                self.issue(f"{path}.type", f"must be one of {sorted(CRACK_TYPES)}")
            if "shape" in crack and crack["shape"] not in CRACK_SHAPES:
                self.issue(f"{path}.shape", f"must be one of {sorted(CRACK_SHAPES)}")

            centre = self.object(crack.get("central_crack_pos_m"), f"{path}.central_crack_pos_m", VECTOR_KEYS)
            for axis in sorted(VECTOR_KEYS):
                self.field_number(centre, axis, f"{path}.central_crack_pos_m")

            # size_m.x and size_m.y span the crack plane; size_m.z is the opening measured along the normal.
            size = self.object(crack.get("size_m"), f"{path}.size_m", VECTOR_KEYS)
            for axis in sorted(VECTOR_KEYS):
                self.field_number(size, axis, f"{path}.size_m", positive=True)

            orientation = self.object(crack.get("orientation"), f"{path}.orientation", ORIENTATION_KEYS)
            components = [
                self.field_number(orientation, key, f"{path}.orientation")
                for key in sorted(ORIENTATION_KEYS)
            ]
            if all(component is not None for component in components):
                if math.hypot(*components) <= 0.0:
                    self.issue(f"{path}.orientation", "must define a nonzero crack normal")

            parameters = self.object(
                crack.get("phase_field_parameters"), f"{path}.phase_field_parameters", PHASE_FIELD_KEYS
            )
            for key in ("length_scale_m", "critical_energy_release_rate_J_m2"):
                self.field_number(parameters, key, f"{path}.phase_field_parameters", positive=True)
            residual = self.field_number(
                parameters, "residual_stiffness", f"{path}.phase_field_parameters", positive=True
            )
            if residual is not None and residual >= 1:
                self.issue(f"{path}.phase_field_parameters.residual_stiffness", "must be less than 1")
            if parameters is not None:
                if parameters.get("model") not in PHASE_FIELD_MODELS:
                    self.issue(
                        f"{path}.phase_field_parameters.model",
                        f"must be one of {sorted(PHASE_FIELD_MODELS)}",
                    )
                if parameters.get("energy_split") not in ENERGY_SPLITS:
                    self.issue(
                        f"{path}.phase_field_parameters.energy_split",
                        f"must be one of {sorted(ENERGY_SPLITS)}",
                    )

            loading = self.object(crack.get("loading"), f"{path}.loading", CRACK_LOADING_KEYS)
            self.field_number(loading, "internal_pressure_pa", f"{path}.loading", nonnegative=True)
            if loading is not None:
                steps = loading.get("load_steps")
                if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
                    self.issue(f"{path}.loading.load_steps", "must be a positive integer")

            solver = self.object(crack.get("solver"), f"{path}.solver", PHASE_FIELD_SOLVER_KEYS)
            if solver is not None:
                for key in (
                    "maximum_staggered_iterations",
                    "mechanics_maximum_iterations",
                    "damage_maximum_iterations",
                    "field_output_interval",
                    "maximum_refinement_levels",
                ):
                    value = solver.get(key)
                    minimum = 0 if key == "maximum_refinement_levels" else 1
                    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                        qualifier = "a non-negative" if minimum == 0 else "a positive"
                        self.issue(f"{path}.solver.{key}", f"must be {qualifier} integer")
                for key in (
                    "staggered_tolerance",
                    "mechanics_relative_tolerance",
                    "damage_gradient_tolerance",
                    "minimum_elements_per_length_scale",
                ):
                    value = self.field_number(solver, key, f"{path}.solver", positive=True)
                    if value is not None and key != "minimum_elements_per_length_scale" and value >= 1:
                        self.issue(f"{path}.solver.{key}", "must be less than 1")

    def validate(self, payload: object) -> list[ValidationIssue]:
        root = self.object(payload, "", set(CONFIG_KEYS), optional=OPTIONAL_CONFIG_KEYS)
        if root is None:
            return self.issues

        geometry = self.object(root.get("geometry"), "geometry", CONFIG_KEYS["geometry"])
        if geometry is not None:
            global_size = self.field_number(geometry, "global_element_size_m", "geometry", positive=True)
            local_size = self.field_number(geometry, "local_element_size_m", "geometry", positive=True)
            if global_size is not None and local_size is not None and local_size > global_size:
                self.issue("geometry.local_element_size_m", "must not exceed global_element_size_m")

            wall = self.object(
                geometry.get("wall"),
                "geometry.wall",
                RADIUS_KEYS | {"height_above_plinth_m"},
            )
            plinth = self.object(
                geometry.get("plinth"),
                "geometry.plinth",
                RADIUS_KEYS | {"contours", "support"},
            )
            bedrock = self.object(
                geometry.get("bedrock"),
                "geometry.bedrock",
                {"contours", "support"},
            )
            wall_upstream, wall_downstream = self.validate_radii(wall, "geometry.wall")
            plinth_upstream, plinth_downstream = self.validate_radii(plinth, "geometry.plinth")
            if wall is not None:
                self.field_number(wall, "height_above_plinth_m", "geometry.wall")
            if (
                wall_downstream is not None
                and plinth_downstream is not None
                and plinth_downstream > wall_downstream
            ):
                self.issue("geometry.plinth.downstream_radius_m", "the plinth must cover the wall radial width")
            if (
                wall_upstream is not None
                and plinth_upstream is not None
                and plinth_upstream < wall_upstream
            ):
                self.issue("geometry.plinth.upstream_radius_m", "the plinth must cover the wall radial width")

            self.validate_support(geometry, "plinth")
            self.validate_support(geometry, "bedrock")
            plinth_stations = self.validate_contours(geometry, "plinth")
            bedrock_stations = self.validate_contours(geometry, "bedrock")
            if plinth_stations is not None and bedrock_stations is not None:
                plinth_by_chainage = {chainage: height for chainage, height in plinth_stations}
                bedrock_by_chainage = {chainage: height for chainage, height in bedrock_stations}
                if plinth_by_chainage.keys() != bedrock_by_chainage.keys():
                    self.issue("geometry.bedrock.contours", "must use the same chainages as geometry.plinth.contours")
                else:
                    for index, (chainage, bedrock_height) in enumerate(bedrock_stations):
                        if bedrock_height > plinth_by_chainage[chainage]:
                            self.issue(
                                f"geometry.bedrock.contours[{index}].height",
                                "must not be above the plinth elevation at the same chainage",
                            )

        self.validate_material(self.object(root.get("material"), "material", CONFIG_KEYS["material"]))

        loads = self.object(root.get("loads"), "loads", CONFIG_KEYS["loads"])
        load_numbers: dict[str, float | None] = {}
        if loads is not None:
            for key in sorted(CONFIG_KEYS["loads"]):
                load_numbers[key] = self.field_number(
                    loads,
                    key,
                    "loads",
                    positive=(key == "water_density"),
                    nonnegative=(key in {"water_height", "tailwater_head", "overflow_head", "overflow_water_level"}),
                )
            water_height = load_numbers.get("water_height")
            overflow_head = load_numbers.get("overflow_head")
            overflow_level = load_numbers.get("overflow_water_level")
            if water_height is not None and overflow_head is not None and overflow_level is not None:
                if not math.isclose(overflow_level, water_height + overflow_head, abs_tol=1e-8):
                    self.issue("loads.overflow_water_level", "must equal water_height + overflow_head")

        outputs = self.object(root.get("outputs"), "outputs", CONFIG_KEYS["outputs"])
        if outputs is not None:
            for key in sorted(CONFIG_KEYS["outputs"]):
                value = outputs.get(key)
                if key in outputs and not isinstance(value, bool):
                    self.issue(f"outputs.{key}", "must be a boolean")
        if "crack_simulations" in root:
            self.validate_cracks(root["crack_simulations"])
        return self.issues


def validate_config(payload: object) -> dict[str, object]:
    """Validate an in-memory JSON object and return it without changing its values."""
    validator = _ConfigValidator()
    issues = validator.validate(payload)
    if issues:
        raise ConfigurationValidationError(issues)
    if not isinstance(payload, dict):
        raise TypeError("validated configuration must be a JSON object")
    return payload


def load_config(source: Path | dict[str, object] = CONFIG_PATH) -> dict[str, object]:
    """Load a configuration file or validate an already parsed JSON object."""
    if isinstance(source, Mapping):
        payload = source
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    return validate_config(payload)
