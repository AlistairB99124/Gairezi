"""Weak construction (lift) joint facets.

A dam wall is poured in horizontal lifts. A poorly prepared lift surface (laitance or
debris left on the dried pour) weakens the *bond between layers* while the concrete on
either side is sound. Interface/cohesive-zone modelling (Goodman 1968; Hillerborg 1976;
Code_Aster R7.01.25 "lois de comportement des joints des barrages") represents such a
plane as zero-thickness joint elements with their own strength, inserted on the plane,
with the bulk elements left intact.

This module picks those planes out of the automatic selection: among the interior
wall-wall facets whose two owner cells are both selected, the near-horizontal ones
(unit normal within ``max_normal_deviation_deg`` of vertical) are lift-joint facets.
It is pure numpy so the Aster preparation (which has no dolfinx) and the FEniCSx runner
compute the same mask from the same geometry; Aster cross-checks the count against the
screening.

Poorly bonded lift joints keep roughly a tenth of the parent tensile strength (often taken
as zero) and a 27-35 degree friction angle (FERC Engineering Guidelines ch. III; EPRI
TR-100345, 1992).
"""
from __future__ import annotations

import math

import numpy as np

try:
    from Elmer.model_config import LIFT_JOINT_DEFAULTS, LIFT_JOINT_LAWS, LIFT_JOINT_UPLIFT_MODES
except ImportError:  # FEniCSx runs from its own directory under WSL
    LIFT_JOINT_DEFAULTS = {"max_normal_deviation_deg": 15.0, "include_base_contact": False,
                           "law": "JOINT_MECA_RUPT", "alpha": 1.0, "pena_contact": 1.0, "uplift": "none"}
    LIFT_JOINT_LAWS = {"JOINT_MECA_RUPT", "JOINT_MECA_FROT"}
    LIFT_JOINT_UPLIFT_MODES = {"none", "linear_upstream_to_downstream"}

RULE = ("interior wall-wall facets between two selected cells whose unit normal lies within "
        "{deviation:g} degrees of vertical (the horizontal lift planes of the selected band)")


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def lift_joint_settings(lift_joints: dict) -> dict:
    """The lift-joint block with the schema defaults filled in."""
    if not isinstance(lift_joints, dict):
        raise ValueError("lift_joints must be an object")
    settings = {**LIFT_JOINT_DEFAULTS, **lift_joints}
    for key in ("tensile_strength_pa", "cohesive_fracture_energy_j_m2"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"lift_joints.{key} must be a positive number")
    deviation = settings["max_normal_deviation_deg"]
    if (isinstance(deviation, bool) or not isinstance(deviation, (int, float))
            or not 0 < float(deviation) < 90):
        raise ValueError("lift_joints.max_normal_deviation_deg must lie in (0, 90)")
    if settings["law"] not in LIFT_JOINT_LAWS:
        raise ValueError(f"lift_joints.law must be one of {sorted(LIFT_JOINT_LAWS)}")
    alpha = settings["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 <= float(alpha) <= 2:
        raise ValueError("lift_joints.alpha must lie in [0, 2]")
    contact = settings["pena_contact"]
    if isinstance(contact, bool) or not isinstance(contact, (int, float)) or not float(contact) > 0:
        raise ValueError("lift_joints.pena_contact must be positive")
    if type(settings["include_base_contact"]) is not bool:
        raise ValueError("lift_joints.include_base_contact must be a boolean")
    friction = settings.get("friction_coefficient")
    if settings["law"] == "JOINT_MECA_FROT":
        if not _number(friction) or not 1e-3 <= float(friction) <= 10:
            raise ValueError("lift_joints.friction_coefficient must lie in [0.001, 10] for JOINT_MECA_FROT")
    elif friction is not None:
        raise ValueError("lift_joints.friction_coefficient applies only to JOINT_MECA_FROT")
    if "plane_elevation_m" in settings and not _number(settings["plane_elevation_m"]):
        raise ValueError("lift_joints.plane_elevation_m must be a finite number")
    if settings["uplift"] not in LIFT_JOINT_UPLIFT_MODES:
        raise ValueError(f"lift_joints.uplift must be one of {sorted(LIFT_JOINT_UPLIFT_MODES)}")
    return settings


def facet_unit_normals(coordinates_m: np.ndarray, faces: np.ndarray) -> np.ndarray:
    points = np.asarray(coordinates_m, dtype=float)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Degenerate facet with zero area")
    return normals / lengths[:, None]


def mean_facet_edge(coordinates_m: np.ndarray, faces: np.ndarray) -> float:
    """Mean edge length of the given triangles (the element size beside the joint)."""
    points = np.asarray(coordinates_m, dtype=float)[np.asarray(faces, dtype=np.int64).reshape(-1, 3)]
    if not len(points):
        raise ValueError("No facets to measure")
    return float(np.mean([np.linalg.norm(points[:, i] - points[:, (i + 1) % 3], axis=1) for i in range(3)]))


def lift_joint_facet_mask(coordinates_m: np.ndarray, faces: np.ndarray, owners: np.ndarray,
                          selected_cells: np.ndarray, max_normal_deviation_deg: float) -> np.ndarray:
    """Boolean mask over ``faces``: both owner cells selected and |n_z| >= cos(deviation).

    ``faces`` are node triples, ``owners`` the (m, 2) owner cells of each facet and
    ``selected_cells`` a boolean mask or an index array over the cells.
    """
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    owners = np.asarray(owners, dtype=np.int64).reshape(-1, 2)
    if len(owners) != len(faces):
        raise ValueError("Every facet needs exactly two owner cells")
    if not 0 < float(max_normal_deviation_deg) < 90:
        raise ValueError("max_normal_deviation_deg must lie in (0, 90)")
    selected = np.asarray(selected_cells)
    if selected.dtype != bool:
        cells = int(owners.max(initial=-1)) + 1
        member = np.zeros(cells, dtype=bool)
        member[selected.astype(np.int64)] = True
        selected = member
    if not len(faces):
        return np.zeros(0, dtype=bool)
    between = selected[owners[:, 0]] & selected[owners[:, 1]]
    normals = facet_unit_normals(coordinates_m, faces)
    horizontal = np.abs(normals[:, 2]) >= math.cos(math.radians(float(max_normal_deviation_deg))) - 1e-12
    return between & horizontal


def lift_joint_report(coordinates_m: np.ndarray, faces: np.ndarray, mask: np.ndarray,
                      settings: dict) -> dict:
    """Screening/summary record for the chosen lift facets."""
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(faces):
        raise ValueError("Lift mask does not match the facets")
    points = np.asarray(coordinates_m, dtype=float)[faces[mask]]
    areas = 0.5 * np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1) \
        if mask.any() else np.zeros(0)
    z = points[:, :, 2].mean(axis=1) if mask.any() else np.zeros(0)
    report = {
        "rule": RULE.format(deviation=float(settings["max_normal_deviation_deg"])),
        "law": settings["law"],
        "tensile_strength_pa": float(settings["tensile_strength_pa"]),
        "cohesive_fracture_energy_j_m2": float(settings["cohesive_fracture_energy_j_m2"]),
        "critical_opening_m": 2.0 * float(settings["cohesive_fracture_energy_j_m2"]) / float(settings["tensile_strength_pa"]),
        "max_normal_deviation_deg": float(settings["max_normal_deviation_deg"]),
        "alpha": float(settings["alpha"]),
        "pena_contact": float(settings["pena_contact"]),
        "include_base_contact": bool(settings["include_base_contact"]),
        "base_contact_model": "wall/foundation interface springs (a spring layer, not a cohesive facet set)",
        "uplift": settings["uplift"],
        "lift_joint_facets": int(mask.sum()),
        "candidate_facets": int(len(mask)),
        "lift_joint_area_m2": float(areas.sum()),
        "lift_plane_z_m": [float(z.min()), float(z.max())] if mask.any() else None,
        "mean_facet_edge_m": mean_facet_edge(coordinates_m, faces[mask]) if mask.any() else None,
    }
    if settings["law"] == "JOINT_MECA_FROT":
        # Mohr-Coulomb cone with apex at the configured tensile strength: c = mu * f_t.
        report["friction_coefficient"] = float(settings["friction_coefficient"])
        report["adhesion_pa"] = float(settings["friction_coefficient"]) * float(settings["tensile_strength_pa"])
    if "plane_elevation_m" in settings:
        report["plane_elevation_m"] = float(settings["plane_elevation_m"])
    return report


def horizontal_plane_survey(coordinates_m: np.ndarray, faces: np.ndarray, owners: np.ndarray,
                            cell_stress_pa: np.ndarray, max_normal_deviation_deg: float,
                            layer_height_m: float) -> list[dict]:
    """Intact-state demand on every horizontal plane of interior wall facets (possible pour planes).

    A weak lift joint fails by opening at its upstream edge and sliding, so the plane is
    chosen by the vertical stress and the shear it must transmit, not by the peak principal
    stress. Facets are grouped by elevation (gaps above a quarter of the layer height start a
    new plane); each facet's traction is the mean of its two owner cells' sigma.n with n = +z.
    The arch is centred on the origin, so the radial position is hypot(x, y); "upstream" is
    the outer third of each plane's radial extent.
    """
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    owners = np.asarray(owners, dtype=np.int64).reshape(-1, 2)
    if not len(faces):
        return []
    coordinates = np.asarray(coordinates_m, dtype=float)
    normals = facet_unit_normals(coordinates, faces)
    horizontal = np.abs(normals[:, 2]) >= math.cos(math.radians(float(max_normal_deviation_deg))) - 1e-12
    if not horizontal.any():
        return []
    faces, owners = faces[horizontal], owners[horizontal]
    points = coordinates[faces]
    areas = 0.5 * np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
    centroids = points.mean(axis=1)
    stress = np.asarray(cell_stress_pa, dtype=float)
    traction = 0.5 * (stress[owners[:, 0]][:, :, 2] + stress[owners[:, 1]][:, :, 2])  # (xz, yz, zz)
    radius = np.hypot(centroids[:, 0], centroids[:, 1])
    order = np.argsort(centroids[:, 2], kind="stable")
    breaks = np.flatnonzero(np.diff(centroids[order, 2]) > 0.25 * float(layer_height_m)) + 1
    planes = []
    for rows in np.split(order, breaks):
        area = float(areas[rows].sum())
        normal = traction[rows, 2]
        r_low, r_high = float(radius[rows].min()), float(radius[rows].max())
        upstream = radius[rows] >= r_high - (r_high - r_low) / 3.0
        shear = (traction[rows, :2] * areas[rows, None]).sum(axis=0)
        compression = float(-(normal * areas[rows]).sum())
        planes.append({
            "z_m": float(np.average(centroids[rows, 2], weights=areas[rows])),
            "facets": int(len(rows)),
            "area_m2": area,
            "mean_vertical_stress_pa": float((normal * areas[rows]).sum() / area),
            "peak_vertical_tension_pa": float(normal.max()),
            "upstream_peak_vertical_tension_pa": float(normal[upstream].max()),
            "tension_area_fraction": float(areas[rows][normal > 0].sum() / area),
            "normal_compression_n": compression,
            "shear_resultant_n": float(np.linalg.norm(shear)),
            # Friction coefficient the intact plane would need to carry its shear by friction alone.
            "shear_to_compression": float(np.linalg.norm(shear) / compression) if compression > 0 else None,
        })
    return planes
