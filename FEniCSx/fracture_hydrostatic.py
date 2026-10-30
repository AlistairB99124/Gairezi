"""Drained equilibrium loading of explicitly saturated connected crack faces.

No wetting kinetics, trapped-fluid drainage or crack growth is inferred.
Different connected heads require a separate steady-flow solution.
"""
from dataclasses import dataclass
import numbers

import numpy as np

if __package__:
    from .fracture_faces import PairedCrackFaces
    from .fracture_water_access import water_access
else:
    from fracture_faces import PairedCrackFaces
    from fracture_water_access import water_access


@dataclass(frozen=True)
class HydrostaticBoundary:
    name: str
    elevation_m: float
    inlet_source_edges: object

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Hydraulic boundary name must be nonempty")
        if (isinstance(self.elevation_m, bool) or not isinstance(self.elevation_m, numbers.Real)
                or not np.isfinite(self.elevation_m)):
            raise ValueError("Hydraulic boundary elevation must be finite")
        edges = np.asarray(self.inlet_source_edges)
        if (edges.dtype.kind not in "iu" or edges.ndim != 2 or edges.shape[1] != 2
                or not len(edges) or np.any(edges < 0) or np.any(edges[:, 0] == edges[:, 1])
                or len(np.unique(np.sort(edges, axis=1), axis=0)) != len(edges)):
            raise ValueError("Boundary inlet edges must be explicit unique nonnegative node pairs")
        edges = edges.copy()
        edges.flags.writeable = False
        object.__setattr__(self, "inlet_source_edges", edges)


@dataclass
class HydrostaticCrackLoad:
    vertex_pressures_pa: np.ndarray
    nodal_forces_n: np.ndarray
    report: dict


def drained_crack_load(faces: PairedCrackFaces, displacement_m, parent_node_ids,
                       boundaries, *, water_density_kg_m3, gravity_m_s2,
                       saturated_face_mask, previously_wet_face_mask,
                       closure_tolerance_m) -> HydrostaticCrackLoad:
    """Assign head only to explicitly saturated, open, boundary-connected faces.

    False saturation entries explicitly declare dry faces, not a drainage rule.
    Previously wet faces cannot be discarded on disconnection/reclosure.
    All reached faces must be entirely submerged; free-surface intersections
    require a partially saturated hydraulic connectivity treatment.
    """
    if not isinstance(faces, PairedCrackFaces):
        raise ValueError("Explicit PairedCrackFaces are required")
    for name, value in (("water_density_kg_m3", water_density_kg_m3), ("gravity_m_s2", gravity_m_s2)):
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive; gravity is a magnitude")
    masks = []
    for name, value in (("saturated_face_mask", saturated_face_mask),
                        ("previously_wet_face_mask", previously_wet_face_mask)):
        mask = np.asarray(value)
        if mask.dtype != np.dtype(bool) or mask.shape != (len(faces.minus),):
            raise ValueError(f"{name} must be an explicit boolean array in crack-face order")
        masks.append(mask.copy())
    saturated, previous = masks
    boundaries = tuple(boundaries)
    if (not boundaries or any(not isinstance(item, HydrostaticBoundary) for item in boundaries)
            or len({item.name for item in boundaries}) != len(boundaries)):
        raise ValueError("Explicit uniquely named HydrostaticBoundary entries are required")
    heads = np.full(len(faces.minus), np.nan)
    access_reports = {}
    heights = faces.coordinates[faces.minus, 2]
    for boundary in boundaries:
        access = water_access(
            faces, displacement_m, boundary.inlet_source_edges, parent_node_ids,
            closure_tolerance_m=closure_tolerance_m,
        )
        access_reports[boundary.name] = access
        reached = np.asarray(access["reachable_open_face_ids"], dtype=np.int64)
        if not len(reached):
            continue
        if np.any(~saturated[reached]):
            raise ValueError("Boundary-connected open faces require explicit saturation; no filling is inferred")
        if np.any(heights[reached].max(axis=1) > boundary.elevation_m):
            raise ValueError("Hydrostatic crack load requires fully submerged faces; partial saturation/free surfaces are not solved")
        assigned = heads[reached]
        if np.any(np.isfinite(assigned) & (assigned != boundary.elevation_m)):
            raise ValueError("Connected hydraulic boundaries have different heads; a steady-flow solution is required")
        heads[reached] = boundary.elevation_m
    connected = np.isfinite(heads)
    if np.any((saturated | previous) & ~connected):
        raise ValueError("Wet closed or isolated faces require a trapped-fluid/contact state; water cannot be silently drained")
    pressure = np.zeros_like(heights)
    pressure[connected] = water_density_kg_m3 * gravity_m_s2 * (
        heads[connected, None] - heights[connected]
    )
    forces = faces.pressure_forces(pressure)
    return HydrostaticCrackLoad(pressure, forces, {
        "status": "assigned_drained_hydrostatic_crack_load",
        "solver_ready": False, "engineering_validated": False,
        "mechanical_equilibrium_solved": False, "wetting_solved": False,
        "pressure_assigned": True,
        "pressurised_face_ids": np.flatnonzero(connected).tolist(),
        "explicitly_dry_face_ids": np.flatnonzero(~saturated).tolist(),
        "boundary_elevations_m": {item.name: float(item.elevation_m) for item in boundaries},
        "water_density_kg_m3": float(water_density_kg_m3),
        "gravity_m_s2": float(gravity_m_s2),
        "access": access_reports,
        "limitations": "Prescribed saturated geometry only; no crack growth, coupled contact solve, trapped fluid, free-surface flow or stability verdict.",
    })
