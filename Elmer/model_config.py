"""Load and validate the root dam configuration shared by mesh and solver builders."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"

CONFIG_KEYS = {
    "geometry": {"global_element_size_m", "local_element_size_m", "wall", "bedrock"},
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
OPTIONAL_CONFIG_KEYS = {"crack_simulations", "cracking", "fracture_debug"}
# Optional UI crack-seed settings (see FEniCSx/automatic_crack.cracking_settings).
CRACKING_KEYS = {"face", "depth_m", "opening_m", "full_width", "buffer_m", "length_m"}
# Optional fracture debug overrides (see FEniCSx/crack_contract.apply_fracture_debug).
FRACTURE_DEBUG_KEYS = {
    "model", "critical_energy_release_rate_j_m2", "length_scale_m", "residual_stiffness",
    "crack_water_pressure_factor", "load_steps", "solver", "aster",
}
FRACTURE_DEBUG_SOLVER_INTEGERS = {
    "maximum_staggered_iterations": 1, "mechanics_maximum_iterations": 1,
    "damage_maximum_iterations": 1, "field_output_interval": 1,
    "maximum_refinement_levels": 0, "maximum_refined_cells": 1,
}
FRACTURE_DEBUG_SOLVER_NUMBERS = {
    "staggered_tolerance", "mechanics_relative_tolerance", "damage_gradient_tolerance",
    "minimum_elements_per_length_scale",
}
FRACTURE_DEBUG_ASTER_KEYS = {"band_width_m", "softening_p"}
CRACK_SEED_FACES = {"upstream", "downstream"}
CRACK_OPENING_RANGE_M = (0.001, 0.01)
CRACK_BUFFER_MAX_M = 2.0
MATERIAL_KEYS = {"density", "youngs_modulus", "poisson_ratio", "tensile_strength_pa"}
RADIUS_KEYS = {"radial_width_m", "upstream_radius_m", "centerline_radius_m", "downstream_radius_m"}
SUPPORT_KEYS = {"model", "material", "spring_n_per_m3"}
CRACK_KEYS = {
    "name",
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
PHASE_FIELD_MODELS = {"AT1", "AT2"}
ENERGY_SPLITS = {"volumetric_deviatoric"}
# Outer rock-box supports. "fixed_bottom" (default when material.bedrock.support
# is absent) fixes every bottom DOF and leaves the sides free; "rollers" applies
# stiff normal-only penalty springs to the bottom and the four vertical sides.
ROCK_OUTER_SUPPORT_MODELS = {"fixed_bottom", "rollers"}
# Horizontal joints between the wall's thickness bands. "bonded" (default when
# geometry.wall.joint_support is absent) keeps one monolithic wall; "spring" makes
# each band a separate concrete block joined by linear x/y/z springs (N/m^3).
WALL_JOINT_MODELS = {"bonded", "spring"}
SUPPORT_MODELS = {
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
        self.warnings: list[ValidationIssue] = []

    def issue(self, path: str, message: str) -> None:
        self.issues.append(ValidationIssue(path, message))

    def warn(self, path: str, message: str) -> None:
        self.warnings.append(ValidationIssue(path, message))

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

    def validate_thickness_steps(
        self,
        value: object,
        upstream: float | None,
        crest_downstream: float | None,
    ) -> None:
        """Optional downstream thickening below given elevations.

        Each step sets the radial width (m) of the wall below ``top_z_m``; the
        upstream face stays at upstream_radius_m, so the extra thickness is on
        the downstream face. Steps must deepen and must not thin the wall.
        """
        path = "geometry.wall.thickness_steps"
        if not isinstance(value, list):
            self.issue(path, "must be an array")
            return
        previous_z = 0.0
        previous_width = (
            upstream - crest_downstream
            if upstream is not None and crest_downstream is not None else None
        )
        for index, row in enumerate(value):
            row_path = f"{path}[{index}]"
            values = self.object(row, row_path, {"top_z_m", "radial_width_m"})
            top_z = self.field_number(values, "top_z_m", row_path)
            width = self.field_number(values, "radial_width_m", row_path, positive=True)
            if top_z is not None:
                if top_z >= previous_z:
                    self.issue(
                        f"{row_path}.top_z_m",
                        "must be below the crest (z = 0) and below the previous step",
                    )
                previous_z = top_z
            if width is not None:
                if previous_width is not None and width < previous_width - 1e-9:
                    self.issue(
                        f"{row_path}.radial_width_m",
                        "must not be less than the wall width above it",
                    )
                if upstream is not None and width >= upstream:
                    self.issue(f"{row_path}.radial_width_m", "must be less than upstream_radius_m")
                previous_width = width

    def validate_wall_joint_support(self, value: object) -> None:
        """Joints between thickness bands: bonded, or linear springs in global x/y/z (N/m^3)."""
        path = "geometry.wall.joint_support"
        values = self.object(value, path, {"model"}, {"spring_n_per_m3"})
        if values is None:
            return
        model = values.get("model")
        if model not in WALL_JOINT_MODELS:
            self.issue(f"{path}.model", f"must be one of {sorted(WALL_JOINT_MODELS)}")
        if model == "spring" and "spring_n_per_m3" not in values:
            self.issue(f"{path}.spring_n_per_m3", "is required for spring joints")
        if "spring_n_per_m3" in values:
            springs = self.object(values["spring_n_per_m3"], f"{path}.spring_n_per_m3", VECTOR_KEYS)
            if springs is not None:
                for axis in ("x", "y", "z"):
                    # A zero spring would leave a block free to drift.
                    self.field_number(springs, axis, f"{path}.spring_n_per_m3", positive=True)

    def validate_material(self, material: Mapping[str, object] | None) -> None:
        for name in ("concrete", "bedrock"):
            path = f"material.{name}"
            values = self.object(
                material.get(name) if material is not None else None, path, MATERIAL_KEYS,
                optional={"support"} if name == "bedrock" else {
                    "compressive_strength_pa", "selected_tensile_strength_pa",
                },
            )
            if values is None:
                continue
            if name == "bedrock" and "support" in values:
                self.validate_rock_outer_support(values["support"], f"{path}.support")
            # Zero rock density is allowed so rock self-weight can be omitted.
            if name == "bedrock":
                self.field_number(values, "density", path, nonnegative=True)
            else:
                self.field_number(values, "density", path, positive=True)
            for key in ("youngs_modulus", "tensile_strength_pa"):
                self.field_number(values, key, path, positive=True)
            if name == "concrete" and "compressive_strength_pa" in values:
                self.field_number(values, "compressive_strength_pa", path, positive=True)
            if name == "concrete" and "selected_tensile_strength_pa" in values:
                self.field_number(values, "selected_tensile_strength_pa", path, nonnegative=True)
            poisson = self.field_number(values, "poisson_ratio", path)
            if poisson is not None and not -1 < poisson < 0.5:
                self.issue(f"{path}.poisson_ratio", "must lie between -1 and 0.5")
            if name == "bedrock":
                self.warn(
                    path,
                    "is retained in configuration; the bedrock volume is generated only for "
                    "the unsolved geometry preview and is not part of the solver mesh, so "
                    "these properties do not affect the analysis.",
                )

    def validate_rock_outer_support(self, value: object, path: str) -> None:
        """Outer rock-box supports; spring axes are the rock box's local frame.

        x: the two bank faces, y: the upstream/downstream faces, z: the bottom.
        """
        values = self.object(value, path, SUPPORT_KEYS)
        if values is None:
            return
        model = values.get("model")
        if "model" in values and model not in ROCK_OUTER_SUPPORT_MODELS:
            self.issue(f"{path}.model", f"must be one of {sorted(ROCK_OUTER_SUPPORT_MODELS)}")
        if "material" in values and values.get("material") != "bedrock":
            self.issue(f"{path}.material", "must be 'bedrock' for outer rock supports")
        springs = self.object(values.get("spring_n_per_m3"), f"{path}.spring_n_per_m3", VECTOR_KEYS)
        if springs is not None:
            for axis in ("x", "y", "z"):
                self.field_number(
                    springs, axis, f"{path}.spring_n_per_m3",
                    positive=model == "rollers", nonnegative=model != "rollers",
                )

    def validate_contours(
        self,
        geometry: Mapping[str, object] | None,
        region: str,
        key: str = "contours",
        minimum_stations: int = 2,
    ) -> list[tuple[float, float]] | None:
        path = f"geometry.{region}.{key}"
        region_values = geometry.get(region) if geometry is not None else None
        if not isinstance(region_values, dict):
            return None
        rows = region_values.get(key)
        if not isinstance(rows, list):
            self.issue(path, "must be an array")
            return None
        if len(rows) < minimum_stations:
            self.issue(path, f"must contain at least {minimum_stations} stations")
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
            if key == "model" and region == "bedrock" and value == "bonded":
                continue
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
            name = crack.get("name")
            if not isinstance(name, str) or not name.strip():
                self.issue(f"{path}.name", "must be a nonempty string")
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

            solver = self.object(
                crack.get("solver"), f"{path}.solver", PHASE_FIELD_SOLVER_KEYS,
                optional={"maximum_refined_cells"},
            )
            if solver is not None:
                for key in (
                    "maximum_staggered_iterations",
                    "mechanics_maximum_iterations",
                    "damage_maximum_iterations",
                    "field_output_interval",
                    "maximum_refinement_levels",
                    "maximum_refined_cells",
                ):
                    if key == "maximum_refined_cells" and key not in solver:
                        continue
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

        geometry = self.object(root.get("geometry"), "geometry", CONFIG_KEYS["geometry"], {"foundation"})
        if geometry is not None:
            if "foundation" in geometry:
                path = "geometry.foundation"
                foundation = self.object(geometry["foundation"], path, {"depth_m", "side_margin_m", "support"})
                if foundation is not None and "support" in foundation:
                    support_path = f"{path}.support"
                    interface = self.object(foundation["support"], support_path, {"model"}, {"spring_n_per_m3"})
                    if interface is not None:
                        model = interface.get("model")
                        if model not in ("bonded", "spring"):
                            self.issue(f"{support_path}.model", "must be 'bonded' or 'spring'; bonded means zero relative displacement")
                        if model == "spring":
                            springs = self.object(interface.get("spring_n_per_m3"), f"{support_path}.spring_n_per_m3", VECTOR_KEYS)
                            if springs is not None:
                                for axis in ("x", "y", "z"):
                                    self.field_number(springs, axis, f"{support_path}.spring_n_per_m3", positive=True)
                        elif "spring_n_per_m3" in interface:
                            self.issue(f"{support_path}.spring_n_per_m3", "is only valid for spring support")
                for key in ("depth_m", "side_margin_m"):
                    value = self.field_number(foundation, key, path, positive=True)
                    if value is not None and value > 5.:
                        self.issue(f"{path}.{key}", "must not exceed 5 m")
                bedrock_value = geometry.get("bedrock")
                support = bedrock_value.get("support") if isinstance(bedrock_value, Mapping) else None
                if not isinstance(support, Mapping):
                    support = {}
                if isinstance(bedrock_value, Mapping) and "mesh" not in bedrock_value:
                    self.issue(path, "requires geometry.bedrock.mesh (graded rock mesh)")
            global_size = self.field_number(geometry, "global_element_size_m", "geometry", positive=True)
            local_size = self.field_number(geometry, "local_element_size_m", "geometry", positive=True)
            if global_size is not None and local_size is not None and local_size > global_size:
                self.issue("geometry.local_element_size_m", "must not exceed global_element_size_m")

            wall = self.object(
                geometry.get("wall"),
                "geometry.wall",
                RADIUS_KEYS | {"height_m"},
                {"thickness_steps", "joint_support", "step_end_taper_ratio"},
            )
            bedrock = self.object(
                geometry.get("bedrock"),
                "geometry.bedrock",
                {"contours", "support"},
                {"bedrock_top_cross_section", "domain", "mesh"},
            )
            if bedrock is not None and "mesh" in bedrock:
                path = "geometry.bedrock.mesh"
                settings = self.object(
                    bedrock["mesh"], path,
                    {"near_size_m", "far_size_m", "near_distance_m", "far_distance_m"},
                )
                values = {
                    key: self.field_number(settings, key, path, positive=True)
                    for key in ("near_size_m", "far_size_m", "far_distance_m")
                }
                near_distance = self.field_number(settings, "near_distance_m", path, nonnegative=True)
                if all(values.get(key) is not None for key in ("near_size_m", "far_size_m")):
                    if values["far_size_m"] < values["near_size_m"]:
                        self.issue(f"{path}.far_size_m", "must not be smaller than near_size_m")
                if near_distance is not None and values["far_distance_m"] is not None:
                    if values["far_distance_m"] <= near_distance:
                        self.issue(f"{path}.far_distance_m", "must exceed near_distance_m")
            domain_bottom = None
            if bedrock is not None and "domain" in bedrock:
                domain = self.object(
                    bedrock["domain"], "geometry.bedrock.domain",
                    {"stream_margin_m", "bank_margin_m", "bottom_z_m"},
                )
                for key in ("stream_margin_m", "bank_margin_m"):
                    self.field_number(domain, key, "geometry.bedrock.domain", nonnegative=True)
                domain_bottom = self.field_number(domain, "bottom_z_m", "geometry.bedrock.domain")
            if bedrock is not None and "bedrock_top_cross_section" in bedrock:
                self.validate_contours(
                    geometry, "bedrock", "bedrock_top_cross_section", minimum_stations=1,
                )
                self.warn(
                    "geometry.bedrock.bedrock_top_cross_section",
                    "is a draft profile, not an active solver input. The geometry preview "
                    "derives its section from geometry.bedrock.contours.",
                )
            wall_upstream, wall_downstream = self.validate_radii(wall, "geometry.wall")
            if wall is not None and "thickness_steps" in wall:
                self.validate_thickness_steps(
                    wall["thickness_steps"], wall_upstream, wall_downstream,
                )
            if wall is not None and "joint_support" in wall:
                self.validate_wall_joint_support(wall["joint_support"])
            if wall is not None and "step_end_taper_ratio" in wall:
                self.field_number(wall, "step_end_taper_ratio", "geometry.wall", positive=True)
                if not wall.get("thickness_steps") or "foundation" not in geometry:
                    self.issue(
                        "geometry.wall.step_end_taper_ratio",
                        "requires geometry.wall.thickness_steps and geometry.foundation",
                    )
            if wall is not None:
                self.field_number(wall, "height_m", "geometry.wall")
                if "height_m" in wall:
                    self.warn(
                        "geometry.wall.height_m",
                        "is nominal metadata; it does not change the meshed geometry. "
                        "The wall runs from the crest (z = 0) down to geometry.bedrock.contours; "
                        "edit those heights instead.",
                    )

            self.validate_support(geometry, "bedrock")
            bedrock_stations = self.validate_contours(geometry, "bedrock")
            if domain_bottom is not None and bedrock_stations:
                if domain_bottom >= min(height for _, height in bedrock_stations):
                    self.issue(
                        "geometry.bedrock.domain.bottom_z_m",
                        "must be below every bedrock contour elevation",
                    )
            foundation_value = geometry.get("foundation")
            depth = foundation_value.get("depth_m") if isinstance(foundation_value, Mapping) else None
            if (domain_bottom is not None and bedrock_stations
                    and isinstance(depth, (int, float)) and not isinstance(depth, bool)
                    and math.isfinite(depth)
                    and domain_bottom >= min(height for _, height in bedrock_stations) - depth):
                self.issue("geometry.foundation.depth_m", "trench floor must remain above the rock-domain bottom")
            if bedrock_stations and min(height for _, height in bedrock_stations) >= 0:
                # The wall spans the chainages where rock lies below the crest (z = 0).
                self.issue("geometry.bedrock.contours", "must dip below the crest elevation (z = 0)")

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
            # Displacement and tensile stress are always reported; cracking_risk follows
            # whether a crack is defined in ``cracking`` (an empty array means no crack).
            cracks = root.get("cracking")
            if isinstance(cracks, list) and isinstance(outputs.get("cracking_risk"), bool):
                if cracks and not outputs["cracking_risk"]:
                    self.issue("outputs.cracking_risk", "must be true when a crack is defined in cracking")
                elif not cracks and outputs["cracking_risk"]:
                    self.issue("outputs.cracking_risk", "must be false when cracking defines no crack")
        if "crack_simulations" in root:
            self.validate_cracks(root["crack_simulations"])
        if "cracking" in root:
            geometry_value = root.get("geometry")
            wall = geometry_value.get("wall") if isinstance(geometry_value, dict) else None
            width = wall.get("radial_width_m") if isinstance(wall, dict) else None
            self.validate_cracking(root["cracking"], width)
        if root.get("fracture_debug") is not None:
            self.validate_fracture_debug(root["fracture_debug"])
        return self.issues

    def validate_fracture_debug(self, value: object) -> None:
        """Optional per-job overrides of FEniCSx/fracture_defaults.json (and Aster softening).

        Null (or an absent block) means the solvers use their shipped defaults.
        """
        path = "fracture_debug"
        debug = self.object(value, path, set(), optional=FRACTURE_DEBUG_KEYS)
        if debug is None:
            return
        if "model" in debug and debug["model"] not in ("AT1", "AT2"):
            self.issue(f"{path}.model", "must be 'AT1' or 'AT2'")
        self.field_number(debug, "critical_energy_release_rate_j_m2", path, positive=True)
        if debug.get("length_scale_m") is not None:
            self.field_number(debug, "length_scale_m", path, positive=True)
        stiffness = self.field_number(debug, "residual_stiffness", path, positive=True)
        if stiffness is not None and stiffness >= 1:
            self.issue(f"{path}.residual_stiffness", "must be less than 1")
        factor = self.field_number(debug, "crack_water_pressure_factor", path, nonnegative=True)
        if factor is not None and factor > 1:
            self.issue(f"{path}.crack_water_pressure_factor", "must lie between 0 and 1")
        if "load_steps" in debug:
            self.debug_integer(debug["load_steps"], f"{path}.load_steps", 1)
        if "solver" in debug:
            solver = self.object(debug["solver"], f"{path}.solver", set(),
                                 optional=set(FRACTURE_DEBUG_SOLVER_INTEGERS) | FRACTURE_DEBUG_SOLVER_NUMBERS)
            for key, minimum in FRACTURE_DEBUG_SOLVER_INTEGERS.items():
                if solver is not None and key in solver:
                    self.debug_integer(solver[key], f"{path}.solver.{key}", minimum)
            for key in FRACTURE_DEBUG_SOLVER_NUMBERS:
                self.field_number(solver, key, f"{path}.solver", positive=True)
        if "aster" in debug:
            aster = self.object(debug["aster"], f"{path}.aster", set(), optional=FRACTURE_DEBUG_ASTER_KEYS)
            self.field_number(aster, "band_width_m", f"{path}.aster", positive=True)
            softening = self.field_number(aster, "softening_p", f"{path}.aster", positive=True)
            if softening is not None and softening < 1:
                self.issue(f"{path}.aster.softening_p", "must be at least 1")

    def debug_integer(self, value: object, path: str, minimum: int) -> None:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not float(value).is_integer()):
            self.issue(path, "must be an integer")
        elif value < minimum:
            self.issue(path, f"must be at least {minimum}")

    def validate_cracking(self, value: object, wall_width: object) -> None:
        """Array of UI crack-seed definitions: empty (no crack; baselines only) or one crack."""
        if not isinstance(value, list):
            self.issue("cracking", "must be an array of crack objects")
            return
        if len(value) > 1:
            self.issue("cracking", "must contain at most one crack; multiple cracks are not supported yet")
        for index, crack in enumerate(value):
            self.validate_crack_seed(crack, f"cracking[{index}]", wall_width)

    def validate_crack_seed(self, value: object, path: str, wall_width: object) -> None:
        """One crack seed: face, radial depth, Z opening, along-wall extent, buffer."""
        cracking = self.object(value, path, CRACKING_KEYS)
        if cracking is None:
            return
        if "face" in cracking and cracking["face"] not in CRACK_SEED_FACES:
            self.issue(f"{path}.face", "must be 'upstream' or 'downstream'")
        depth = self.field_number(cracking, "depth_m", path, positive=True)
        if (depth is not None and isinstance(wall_width, (int, float)) and not isinstance(wall_width, bool)
                and math.isfinite(wall_width) and depth >= wall_width):
            self.issue(f"{path}.depth_m", f"must be less than the {wall_width:g} m wall radial width")
        opening = self.field_number(cracking, "opening_m", path, positive=True)
        if opening is not None and not CRACK_OPENING_RANGE_M[0] <= opening <= CRACK_OPENING_RANGE_M[1]:
            self.issue(f"{path}.opening_m", "must be between 0.001 and 0.01 m")
        buffer = self.field_number(cracking, "buffer_m", path, nonnegative=True)
        if buffer is not None and buffer > CRACK_BUFFER_MAX_M:
            self.issue(f"{path}.buffer_m", "must not exceed 2 m")
        full_width = cracking.get("full_width")
        if "full_width" in cracking and not isinstance(full_width, bool):
            self.issue(f"{path}.full_width", "must be a boolean")
        if "length_m" in cracking and cracking["length_m"] is not None:
            length = self.number(cracking["length_m"], f"{path}.length_m", positive=True)
            if length is not None and full_width is True:
                self.warn(f"{path}.length_m", "is ignored while full_width is true")
        elif full_width is False:
            self.issue(f"{path}.length_m", "is required when full_width is false")


def wall_thickness_steps(wall: Mapping[str, object]) -> list[tuple[float, float]]:
    """Return (top_z_m, radial_width_m) bands from the crest downwards.

    The first band is the crest section (top z = 0, radial_width_m); later
    bands come from the optional geometry.wall.thickness_steps.
    """
    bands = [(0.0, float(wall["upstream_radius_m"]) - float(wall["downstream_radius_m"]))]
    for step in wall.get("thickness_steps") or ():
        bands.append((float(step["top_z_m"]), float(step["radial_width_m"])))
    return bands


def wall_width_at(wall: Mapping[str, object], z: float) -> float:
    """Radial wall width (m) at elevation z; a step applies strictly below its top_z_m."""
    width = wall_thickness_steps(wall)[0][1]
    for top_z, step_width in wall_thickness_steps(wall)[1:]:
        if z < top_z:
            width = step_width
    return width


def wall_downstream_radius_at(wall: Mapping[str, object], z: float) -> float:
    """Downstream face radius (m) at elevation z; the upstream face is fixed."""
    return float(wall["upstream_radius_m"]) - wall_width_at(wall, z)


def tapered_wall_downstream_radius_at(wall: Mapping[str, object], z: float, seat_z: float) -> float:
    """Downstream radius (m) at elevation z above a seat at seat_z.

    With optional geometry.wall.step_end_taper_ratio, each step widens the band
    above it by at most ratio * (step top_z_m - seat_z): where a step ends against
    the abutment, its extra width grows with the seat depth below the step
    instead of starting at full width on a one-course shelf. Without it, this
    equals wall_downstream_radius_at.
    """
    ratio = wall.get("step_end_taper_ratio")
    if ratio is None:
        return wall_downstream_radius_at(wall, z)
    bands = wall_thickness_steps(wall)
    width = bands[0][1]
    for top_z, step_width in bands[1:]:
        if z < top_z:
            width = max(width, min(step_width, width + float(ratio) * max(0.0, top_z - seat_z)))
    return float(wall["upstream_radius_m"]) - width


def foundation_support(config: dict) -> dict:
    """Explicit wall/foundation connection, independent of internal wall joints."""
    return config["geometry"]["foundation"]["support"]


def wall_joint_support(wall: Mapping[str, object]) -> dict[str, object] | None:
    """Spring joints between thickness bands, or None for one bonded (monolithic) wall.

    Returns {"levels_m": step elevations (top to bottom), "spring_n_per_m3": {x, y, z}}.
    """
    support = wall.get("joint_support")
    levels = [top_z for top_z, _ in wall_thickness_steps(wall)[1:]]
    if not isinstance(support, Mapping) or support.get("model") != "spring" or not levels:
        return None
    springs = support["spring_n_per_m3"]
    return {
        "levels_m": levels,
        "spring_n_per_m3": {axis: float(springs[axis]) for axis in ("x", "y", "z")},  # type: ignore[index]
    }


def wall_block_index(wall: Mapping[str, object], z: float) -> int:
    """Thickness band (0 = crest block) containing elevation z; steps start strictly below top_z_m."""
    return sum(1 for top_z, _ in wall_thickness_steps(wall)[1:] if z < top_z)


def validate_config(payload: object) -> dict[str, object]:
    """Validate an in-memory JSON object and return it without changing its values."""
    validator = _ConfigValidator()
    issues = validator.validate(payload)
    if issues:
        raise ConfigurationValidationError(issues)
    if not isinstance(payload, dict):
        raise TypeError("validated configuration must be a JSON object")
    return payload


def collect_warnings(payload: object, *, three_body: bool = False) -> list[ValidationIssue]:
    """Return non-fatal warnings for nominal or currently inactive configuration
    properties (e.g. wall height_m, bedrock material, an ignored crack length_m)
    without reinterpreting or rejecting the submitted values. Intended to be called
    only once the configuration has already passed validate_config.
    """
    validator = _ConfigValidator()
    validator.validate(payload)
    return [
        warning for warning in validator.warnings
        if not (three_body and warning.path == "material.bedrock")
    ]


def rock_outer_support(config: Mapping[str, object]) -> dict[str, object]:
    """Outer rock-box support; absent configuration keeps the historical fixed bottom."""
    support = config["material"]["bedrock"].get("support")  # type: ignore[index, union-attr]
    if support is None:
        return {"model": "fixed_bottom", "spring_n_per_m3": None}
    springs = support["spring_n_per_m3"]
    return {
        "model": support["model"],
        "spring_n_per_m3": {axis: float(springs[axis]) for axis in ("x", "y", "z")},
    }


def load_config(source: Path | dict[str, object] = CONFIG_PATH) -> dict[str, object]:
    """Load a configuration file or validate an already parsed JSON object."""
    if isinstance(source, Mapping):
        payload = source
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    return validate_config(payload)
