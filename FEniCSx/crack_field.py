"""Initial phase-field damage for the cracks declared in the root config.json.

Sign and damage conventions used throughout this module:

* ``signed_distance`` returns the exact signed distance, in metres, to the
  surface of the configured crack box: negative inside the crack, zero on its
  surface, positive in the surrounding intact material.
* The returned ``dolfinx.fem.Function`` is a damage field ``alpha`` with
  ``alpha = 1`` for fully cracked material and ``alpha = 0`` for intact
  material. This is the usual AT1/AT2 phase-field convention, but it is the
  opposite of codes that store an intact-material stiffness factor, so check
  the convention before coupling this field to a degradation function.

The field is a regularised *representation* of a pre-existing crack. It is an
initial condition for a phase-field solve, not a computed crack path and not a
physical crack opening displacement.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import warnings

from dolfinx import fem, mesh
from dolfinx.cpp.mesh import h as cell_diameters
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Elmer.model_config import load_config  # noqa: E402  (requires ROOT on sys.path)
from FEniCSx.crack_contract import CrackSpecification, PhaseFieldParameters, PhaseFieldSolverOptions

# Minimum elements across the phase-field length scale for a regularised solution.
MINIMUM_ELEMENTS_PER_LENGTH_SCALE = 2.0


def load_crack_specifications(config: Path | dict | None = None) -> list[CrackSpecification]:
    """Read and validate crack_simulations from the authoritative dam configuration."""
    payload = load_config(CONFIG_PATH if config is None else config)
    cracks = payload.get("crack_simulations")
    if not cracks:
        return []
    specifications = []
    for crack in cracks:
        centre = crack["central_crack_pos_m"]
        size = crack["size_m"]
        orientation = crack["orientation"]
        parameters = crack["phase_field_parameters"]
        specifications.append(
            CrackSpecification(
                name=crack.get("name", ""),
                type=crack["type"],
                shape=crack["shape"],
                centre_m=(float(centre["x"]), float(centre["y"]), float(centre["z"])),
                size_m=(float(size["x"]), float(size["y"]), float(size["z"])),
                normal=(
                    float(orientation["normal_x"]),
                    float(orientation["normal_y"]),
                    float(orientation["normal_z"]),
                ),
                length_scale_m=float(parameters["length_scale_m"]),
                critical_energy_release_rate_j_m2=float(parameters["critical_energy_release_rate_J_m2"]),
                internal_pressure_pa=float(crack["loading"]["internal_pressure_pa"]),
                model=parameters["model"],
                energy_split=parameters["energy_split"],
                residual_stiffness=float(parameters["residual_stiffness"]),
                load_steps=int(crack["loading"]["load_steps"]),
            )
        )
    return specifications


def load_phase_field_solver_options(config: Path | dict | None = None) -> PhaseFieldSolverOptions:
    """Read numerical controls shared by the configured crack simulation."""
    payload = load_config(CONFIG_PATH if config is None else config)
    cracks = payload.get("crack_simulations") or []
    if not cracks:
        raise ValueError("At least one crack_simulations entry is required")
    first = cracks[0]["solver"]
    if any(crack["solver"] != first for crack in cracks[1:]):
        raise ValueError("All crack simulations must use identical solver settings in one coupled solve")
    return PhaseFieldSolverOptions(**first)


def crack_frame(normal) -> np.ndarray:
    """Orthonormal crack axes as matrix columns: two in-plane axes then the unit normal.

    The frame is the shortest-arc rotation carrying global +z onto the configured
    normal, so a normal of (0, 0, 1) reproduces the global axes exactly and
    ``size_m.x``/``size_m.y`` keep their global meaning.
    """
    direction = np.asarray(normal, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("Crack normal must be three finite components")
    length = float(np.linalg.norm(direction))
    if length <= 0.0:
        raise ValueError("Crack normal must be a nonzero vector")
    direction = direction / length

    cosine = float(direction[2])
    axis = np.cross((0.0, 0.0, 1.0), direction)
    sine = float(np.linalg.norm(axis))
    if sine <= 1e-14:
        # Parallel or antiparallel to +z; the antiparallel case is a half turn about global x.
        return np.eye(3) if cosine > 0.0 else np.diag((1.0, -1.0, -1.0))
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def signed_distance(points, specification: CrackSpecification) -> np.ndarray:
    """Exact signed distance in metres from each point to the configured crack box.

    ``points`` may be an ``(n, 3)`` array or a single ``(3,)`` point. Distances are
    negative inside the crack, zero on its surface and positive outside it.
    """
    coordinates = np.asarray(points, dtype=float)
    single = coordinates.ndim == 1
    if single:
        coordinates = coordinates[np.newaxis, :]
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("points must have shape (n, 3) or (3,)")

    if specification.annular_sector is not None:
        from automatic_crack import sector_signed_distance

        distance = sector_signed_distance(
            coordinates, specification.annular_sector, specification.centre_m[2], specification.size_m[2],
        )
        return distance[0] if single else distance

    frame = crack_frame(specification.normal)
    centre = np.asarray(specification.centre_m, dtype=float)
    # Components of each offset along the crack axes.
    local = (coordinates - centre) @ frame
    overshoot = np.abs(local) - specification.half_extents_m
    outside = np.linalg.norm(np.maximum(overshoot, 0.0), axis=1)
    inside = np.minimum(overshoot.max(axis=1), 0.0)
    distance = outside + inside
    return distance[0] if single else distance


def damage_profile(distance, length_scale_m: float, model: str = "at2") -> np.ndarray:
    """Regularised damage from a signed distance, saturating to 1 inside the crack.

    ``at2`` uses the exponential optimal profile ``exp(-d / l)``, which decays but never
    reaches zero. ``at1`` uses ``(1 - d / (2 l))^2``, which reaches exactly zero at
    ``d = 2 l`` and so leaves a sharply bounded damaged band.
    """
    if not math.isfinite(length_scale_m) or length_scale_m <= 0.0:
        raise ValueError("length_scale_m must be a positive number")
    outside = np.maximum(np.asarray(distance, dtype=float), 0.0)
    if model == "at2":
        return np.exp(-outside / length_scale_m)
    if model == "at1":
        return np.where(outside < 2.0 * length_scale_m, (1.0 - outside / (2.0 * length_scale_m)) ** 2, 0.0)
    raise ValueError(f"Unknown phase-field profile {model!r}; expected 'at1' or 'at2'")


def resolution_report(domain, specification: CrackSpecification, eligible_cells=None) -> dict:
    """Report whether the mesh resolves the phase-field length scale near the crack."""
    dimension = domain.topology.dim
    domain.topology.create_connectivity(dimension, 0)
    cells = np.arange(domain.topology.index_map(dimension).size_local, dtype=np.int32)
    # Cell diameter: the largest distance between any two of its vertices.
    diameters = cell_diameters(domain._cpp_object, dimension, cells)
    midpoints = mesh.compute_midpoints(domain, dimension, cells)
    # Only cells reaching into the damaged band control regularisation. Inflate the band by
    # half a cell so a band thinner than one element still reports the elements that span it.
    distance = signed_distance(midpoints, specification)
    band = distance <= 2.0 * specification.length_scale_m + diameters / 2.0
    eligible = np.ones(len(cells), dtype=bool)
    if eligible_cells is not None:
        eligible[:] = False
        eligible[eligible_cells] = True
        if not eligible.any():
            raise ValueError("No damageable cells are available for crack resolution checks.")
        band &= eligible
    if not band.any():
        band = eligible & (distance <= distance[eligible].min())
    local = diameters[band]
    diameter = float(local.max())
    elements = specification.length_scale_m / diameter
    return {
        "length_scale_m": specification.length_scale_m,
        "maximum_cell_diameter_near_crack_m": diameter,
        "elements_per_length_scale": elements,
        "cells_in_damaged_band": int(band.sum()),
        "resolved": bool(elements >= MINIMUM_ELEMENTS_PER_LENGTH_SCALE),
    }


def crack_placement_report(
    imported, specification: CrackSpecification, required_regions=(1,), centre_regions=None,
) -> dict:
    """Locate a crack in the imported dam and verify which bodies its diffuse band reaches.

    Region 1 is the wall and region 2 is bedrock. The crack centre must lie in one of
    ``centre_regions`` (default: ``required_regions``) and its diffuse band must intersect
    at least one cell of every region in ``required_regions``.
    """
    from dolfinx import geometry

    required_regions = tuple(int(region) for region in required_regions)
    centre_regions = required_regions if centre_regions is None else tuple(int(r) for r in centre_regions)
    domain = imported.domain
    dimension = domain.topology.dim
    centre = np.asarray(specification.centre_m, dtype=domain.geometry.x.dtype).reshape(1, 3)
    tree = geometry.bb_tree(domain, dimension)
    candidates = geometry.compute_collisions_points(tree, centre)
    colliding = geometry.compute_colliding_cells(domain, candidates, centre)
    centre_cells = colliding.links(0)
    centre_cells = centre_cells[centre_cells < domain.topology.index_map(dimension).size_local]
    region_by_cell = {
        int(cell): int(value)
        for cell, value in zip(imported.cell_tags.indices, imported.cell_tags.values)
    }
    local_centre_regions = sorted(
        {region_by_cell[int(cell)] for cell in centre_cells if int(cell) in region_by_cell}
    )

    cells = np.arange(domain.topology.index_map(dimension).size_local, dtype=np.int32)
    diameters = cell_diameters(domain._cpp_object, dimension, cells)
    midpoints = mesh.compute_midpoints(domain, dimension, cells)
    distance = signed_distance(midpoints, specification)
    band = distance <= 2.0 * specification.length_scale_m + diameters / 2.0
    band_cells = cells[band]
    band_regions = np.asarray([region_by_cell.get(int(cell), -1) for cell in band_cells], dtype=np.int64)
    bounds = [domain.geometry.x.min(axis=0).tolist(), domain.geometry.x.max(axis=0).tolist()]
    from mpi4py import MPI

    bounds = [
        [float(domain.comm.allreduce(value, op=MPI.MIN)) for value in bounds[0]],
        [float(domain.comm.allreduce(value, op=MPI.MAX)) for value in bounds[1]],
    ]
    found_centre_regions = sorted({
        value for region_ids in domain.comm.allgather(local_centre_regions) for value in region_ids
    })
    band_counts = {
        str(region): int(domain.comm.allreduce(int(np.count_nonzero(band_regions == region)), op=MPI.SUM))
        for region in sorted(set(required_regions) | set(centre_regions))
    }
    report = {
        "centre_m": list(specification.centre_m),
        "mesh_bounds_m": bounds,
        "centre_cell_count": int(domain.comm.allreduce(len(centre_cells), op=MPI.SUM)),
        "centre_region_ids": found_centre_regions,
        "diffuse_band_cell_count": int(domain.comm.allreduce(len(band_cells), op=MPI.SUM)),
        "required_region_ids": list(required_regions),
        "allowed_centre_region_ids": list(centre_regions),
        "band_cell_count_by_region": band_counts,
        "required_region_band_cell_count": sum(band_counts[str(region)] for region in required_regions),
        "centre_in_required_region": any(region in centre_regions for region in found_centre_regions),
        "minimum_cell_midpoint_signed_distance_m": float(domain.comm.allreduce(distance.min(), op=MPI.MIN)),
    }
    report["valid"] = report["centre_in_required_region"] and all(
        band_counts[str(region)] > 0 for region in required_regions
    )
    return report


def validate_crack_placement(imported, specifications, required_regions=(1,), centre_regions=None) -> list[dict]:
    """Return placement reports or fail before constructing a meaningless damage field."""
    if isinstance(specifications, CrackSpecification):
        specifications = [specifications]
    reports = [
        crack_placement_report(imported, specification, required_regions, centre_regions)
        for specification in specifications
    ]
    invalid = [index for index, report in enumerate(reports) if not report["valid"]]
    if invalid:
        details = "; ".join(
            f"crack {index}: centre={reports[index]['centre_m']}, "
            f"mesh_bounds={reports[index]['mesh_bounds_m']}, "
            f"centre_regions={reports[index]['centre_region_ids']}, "
            f"band_cells_by_region={reports[index]['band_cell_count_by_region']}"
            for index in invalid
        )
        names = {1: "wall", 2: "bedrock"}
        bodies = " and ".join(f"concrete {names.get(region, region)}" for region in required_regions)
        raise ValueError(
            f"Configured crack does not intersect the {bodies} (required regions {list(required_regions)}) "
            "from an allowed centre region; "
            "check the dam global coordinate system. " + details
        )
    return reports

def _require_cell_budget(cells: int, options: PhaseFieldSolverOptions, level: int, levels: list, verb: str):
    if cells > options.maximum_refined_cells:
        raise RuntimeError(
            f"Crack-band refinement level {level} {verb} {cells} cells, above the "
            f"maximum_refined_cells budget of {options.maximum_refined_cells}. Stopped before running "
            "out of memory; reduce the crack extent or revisit the length scale. "
            f"Last resolution: {[c['elements_per_length_scale'] for c in levels[-1]['cracks']]}"
        )


@dataclass
class RefinedPhaseFieldMesh:
    domain: mesh.Mesh
    cell_tags: mesh.MeshTags
    facet_tags: mesh.MeshTags
    report: dict


def refine_for_phase_field(
    imported, specifications, options: PhaseFieldSolverOptions, damage_region_ids=(1,),
) -> RefinedPhaseFieldMesh:
    """Locally refine damageable cells in the diffuse crack bands while preserving mesh tags."""
    if isinstance(specifications, CrackSpecification):
        specifications = [specifications]
    damage_region_ids = tuple(int(region) for region in damage_region_ids)
    validate_crack_placement(imported, specifications, centre_regions=damage_region_ids)
    domain, cell_tags, facet_tags = imported.domain, imported.cell_tags, imported.facet_tags
    levels = []
    for level in range(options.maximum_refinement_levels + 1):
        damage_cells = np.concatenate([cell_tags.find(region) for region in damage_region_ids])
        reports = [resolution_report(domain, specification, damage_cells) for specification in specifications]
        levels.append({
            "level": level, "cracks": reports,
            "wall_cells": int(len(cell_tags.find(1))), "bedrock_cells": int(len(cell_tags.find(2))),
        })
        print(f"Refinement level {level}: wall={len(cell_tags.find(1))}, bedrock={len(cell_tags.find(2))}, "
              f"damage-region resolution={[r['elements_per_length_scale'] for r in reports]}", flush=True)
        if all(
            report["elements_per_length_scale"] >= options.minimum_elements_per_length_scale
            for report in reports
        ):
            return RefinedPhaseFieldMesh(
                domain,
                cell_tags,
                facet_tags,
                {
                    "status": (
                        "resolved"
                        if options.minimum_elements_per_length_scale >= MINIMUM_ELEMENTS_PER_LENGTH_SCALE
                        else "preliminary_under_resolved"
                    ),
                    "target_elements_per_length_scale": options.minimum_elements_per_length_scale,
                    "recommended_minimum_elements_per_length_scale": MINIMUM_ELEMENTS_PER_LENGTH_SCALE,
                    "meets_recommended_resolution": all(
                        report["elements_per_length_scale"] >= MINIMUM_ELEMENTS_PER_LENGTH_SCALE
                        for report in reports
                    ),
                    "levels": levels,
                    "refinement_region_ids": list(damage_region_ids),
                    "non_damage_region_refinement": "conformity only",
                },
            )
        if level == options.maximum_refinement_levels:
            break

        dimension = domain.topology.dim
        cells = np.arange(domain.topology.index_map(dimension).size_local, dtype=np.int32)
        diameters = cell_diameters(domain._cpp_object, dimension, cells)
        midpoints = mesh.compute_midpoints(domain, dimension, cells)
        marked = np.zeros(len(cells), dtype=bool)
        damageable = np.zeros(len(cells), dtype=bool)
        damageable[damage_cells] = True
        bedrock = np.zeros(len(cells), dtype=bool)
        bedrock[cell_tags.find(2)] = True
        for specification, report in zip(specifications, reports):
            if report["elements_per_length_scale"] < options.minimum_elements_per_length_scale:
                distance = signed_distance(midpoints, specification)
                marked |= (
                    damageable
                    & (distance <= 3.0 * specification.length_scale_m + diameters)
                    & (diameters > specification.length_scale_m / options.minimum_elements_per_length_scale)
                )
        marked_cells = cells[marked]
        levels[-1]["marked_wall_cells"] = int(np.count_nonzero(marked & ~bedrock))
        if not len(marked_cells):
            raise RuntimeError("No cells were selected for required crack-band refinement")
        from mpi4py import MPI

        total_cells = domain.topology.index_map(dimension).size_global
        # Each marked tetrahedron splits into 8, so this lower bound ignores conformity closure.
        predicted = total_cells + 7 * domain.comm.allreduce(len(marked_cells), op=MPI.SUM)
        _require_cell_budget(predicted, options, level + 1, levels, "predicted at least")
        domain.topology.create_connectivity(dimension, 1)
        edges = mesh.compute_incident_entities(domain.topology, marked_cells, dimension, 1)
        refined, parent_cell, parent_facet = mesh.refine(
            domain,
            edges,
            option=mesh.RefinementOption.parent_cell_and_facet,
        )
        _require_cell_budget(
            refined.topology.index_map(dimension).size_global, options, level + 1, levels, "reached",
        )
        refined.topology.create_connectivity(dimension, dimension - 1)
        refined.topology.create_connectivity(dimension - 1, dimension)
        cell_tags = mesh.transfer_meshtag(cell_tags, refined, parent_cell)
        facet_tags = mesh.transfer_meshtag(facet_tags, refined, parent_cell, parent_facet)
        domain = refined
    raise RuntimeError(
        f"Crack bands remain under-resolved after {options.maximum_refinement_levels} refinement levels: "
        f"{levels[-1]['cracks']}"
    )


def crack_phase_field(
    domain,
    specifications: CrackSpecification | list[CrackSpecification],
    *,
    degree: int = 1,
    model: str = "at2",
    name: str = "damage",
    check_resolution: bool = True,
):
    """Build a ``dolfinx.fem.Function`` holding the diffuse damage of the given cracks.

    Overlapping cracks are combined by taking the maximum damage, so no crack is
    erased by a neighbouring one. Returns damage with 1 fully cracked and 0 intact.
    """
    if isinstance(specifications, CrackSpecification):
        specifications = [specifications]
    if not specifications:
        raise ValueError("At least one crack specification is required")
    for specification in specifications:
        if specification.shape not in ("rectangular", "annular_sector"):
            raise ValueError(f"Unsupported crack shape {specification.shape!r}")
        if specification.shape == "annular_sector" and specification.annular_sector is None:
            raise ValueError("Annular-sector crack requires radial and angular bounds.")
        if specification.type != "phase_field":
            raise ValueError(f"Unsupported crack type {specification.type!r}; expected 'phase_field'")

    if check_resolution:
        for index, specification in enumerate(specifications):
            report = resolution_report(domain, specification)
            if not report["resolved"]:
                warnings.warn(
                    f"Crack {index} is under-resolved: {report['elements_per_length_scale']:.2f} elements per "
                    f"length scale (want at least {MINIMUM_ELEMENTS_PER_LENGTH_SCALE}). The damage band depends "
                    f"on mesh size until this is refined.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    space = fem.functionspace(domain, ("Lagrange", degree))
    damage = fem.Function(space, name=name)

    def evaluate(x: np.ndarray) -> np.ndarray:
        points = x[:3].T
        values = np.zeros(points.shape[0], dtype=np.float64)
        for specification in specifications:
            values = np.maximum(
                values,
                damage_profile(signed_distance(points, specification), specification.length_scale_m, model),
            )
        return values

    damage.interpolate(evaluate)
    damage.x.scatter_forward()
    return damage
