"""Read-only topology/checkpoint audit; no solve or physical-model changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def region_connection(points, cells, regions):
    """Distinguish shared node identity from merely coincident coordinates."""
    points = np.asarray(points)
    cells = np.asarray(cells)
    regions = np.asarray(regions)
    if cells.ndim != 2 or cells.shape[1] != 4 or len(regions) != len(cells):
        raise ValueError("Expected tetrahedral connectivity and one region ID per cell")
    if set(np.unique(regions)) != {1, 2}:
        raise ValueError("Audit requires exactly plinth (1) and wall (2) regions")
    nodes = {region: np.unique(cells[regions == region]) for region in (1, 2)}
    shared = np.intersect1d(nodes[1], nodes[2])
    coordinates = {
        region: {tuple(row) for row in np.round(points[ids], decimals=9)}
        for region, ids in nodes.items()
    }
    return {
        "wall_cells": int(np.count_nonzero(regions == 2)),
        "plinth_cells": int(np.count_nonzero(regions == 1)),
        "shared_vertex_ids": int(len(shared)),
        "coincident_coordinate_count_rounded_1e_9_m": len(coordinates[1] & coordinates[2]),
        "direct_continuous_p1_displacement_transfer": bool(len(shared)),
        "qualification": (
            "Shared nodes indicate some kinematic coupling, not proof of a fully bonded interface."
            if len(shared) else
            "No shared P1 displacement DOFs. Separate springs to ground do not couple the two bodies."
        ),
    }


def match_checkpoint_geometry(source_xyz, source_cells, target_xyz, target_cells, regions):
    """Match exported/reordered meshes without merging coincident body nodes."""
    from scipy.spatial import cKDTree

    tolerance = 1e-10
    distances, cell_map = cKDTree(source_xyz[source_cells].mean(axis=1)).query(
        target_xyz[target_cells].mean(axis=1), k=2,
    )
    if np.any(distances[:, 0] > tolerance) or np.any(distances[:, 1] <= tolerance):
        raise ValueError("Checkpoint cells have missing or ambiguous geometric matches")
    source_order = cell_map[:, 0]
    if len(np.unique(source_order)) != len(source_cells) or len(target_cells) != len(source_cells):
        raise ValueError("Checkpoint cell mapping is not bijective")
    node_map = np.full(len(target_xyz), -1, dtype=np.int64)
    for region in (1, 2):
        target_nodes = np.unique(target_cells[regions == region])
        source_nodes = np.unique(source_cells[source_order[regions == region]])
        distances, order = cKDTree(source_xyz[source_nodes]).query(target_xyz[target_nodes], k=2)
        if np.any(distances[:, 0] > tolerance) or np.any(distances[:, 1] <= tolerance):
            raise ValueError("Checkpoint nodes have missing or ambiguous matches within a body")
        mapped = source_nodes[order[:, 0]]
        already = node_map[target_nodes]
        if np.any((already >= 0) & (already != mapped)):
            raise ValueError("Checkpoint and target disagree about shared body nodes")
        node_map[target_nodes] = mapped
    if np.any(node_map < 0) or len(np.unique(node_map)) != len(source_xyz):
        raise ValueError("Checkpoint node mapping is not bijective")
    if not np.array_equal(
        np.sort(node_map[target_cells], axis=1), np.sort(source_cells[source_order], axis=1),
    ):
        raise ValueError("Checkpoint and exported mesh topology do not match")
    return node_map


def audit(run: Path) -> dict:
    from dolfinx import fem, io, mesh
    from mpi4py import MPI
    import ufl

    if MPI.COMM_WORLD.size != 1:
        raise ValueError("This read-only audit requires one MPI rank")
    from phase_field_solver import _strain_breakdown_details, strain

    with io.XDMFFile(MPI.COMM_WORLD, run / "mesh.xdmf", "r") as reader:
        domain = reader.read_mesh(name="mesh")
        domain.topology.create_connectivity(3, 2)
        domain.topology.create_connectivity(2, 3)
        tags = reader.read_meshtags(domain, name="regions")
        facets = reader.read_meshtags(domain, name="boundaries")
    cells = domain.geometry.dofmaps[0]
    regions = np.zeros(len(cells), dtype=np.int32)
    regions[tags.indices] = tags.values
    report = {
        "source_run": str(run),
        "read_only": True,
        "engineering_validated": False,
        "connection": region_connection(domain.geometry.x, cells, regions),
        "interface_facets": {},
    }
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facets)
    f_to_c = domain.topology.connectivity(2, 3)
    for tag in (1, 8, 9):
        ids = facets.find(tag)
        adjacent = [f_to_c.links(int(facet)) for facet in ids]
        report["interface_facets"][str(tag)] = {
            "count": int(len(ids)),
            "exterior_count": sum(len(values) == 1 for values in adjacent),
            "adjacent_regions": sorted({int(regions[cell]) for values in adjacent for cell in values}),
            "area_m2": float(fem.assemble_scalar(fem.form(1.0 * ds(tag)))),
        }
    report["mechanical_supports"] = {
        "tag_1": "plinth foundation: directional spring to ground",
        "tag_8": "wall base: directional spring to ground, not to plinth displacement",
        "tag_9": "plinth-side coincident surface: no coupling/contact term in dam adapter",
        "source": "dam_phase_field.support_form and dam_elasticity.solve",
    }
    baseline = json.loads((run / "refined_baseline" / "baseline_report.json").read_text())
    report["intact_spring_reactions_n"] = baseline["ground_spring_reactions_n"]
    manifest = json.loads((run / "manifest.json").read_text())
    report["failure"] = manifest.get("error")

    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    alpha_space = fem.functionspace(domain, ("Lagrange", 1))
    u, alpha = fem.Function(space), fem.Function(alpha_space)
    with np.load(run / "checkpoint_layout.npz") as layout, np.load(run / "iteration_checkpoint.npz") as state:
        if int(state["mpi_processes"]) != 1:
            raise ValueError("Audit requires a serial checkpoint; refusing to reinterpret MPI shards")
        source_xyz = layout["coordinates"]
        np.testing.assert_allclose(state["coordinates"], source_xyz, rtol=0, atol=1e-12)
        source_geometry = layout["geometry_cell_dofs"]
        node_map = match_checkpoint_geometry(
            source_xyz, source_geometry, domain.geometry.x, cells, regions,
        )
        for function, field, dof_key in (
            (u, "displacement", "displacement_cell_dofs"),
            (alpha, "damage", "damage_cell_dofs"),
        ):
            dofs = layout[dof_key]
            source_dofs = np.full(len(source_xyz), -1, dtype=np.int64)
            source_dofs[source_geometry] = dofs
            if np.any(source_dofs < 0):
                raise ValueError("Checkpoint DOF mapping is incomplete")
            values = state[field].reshape(-1, 3) if field == "displacement" else state[field]
            target = function.x.array.reshape(-1, 3) if field == "displacement" else function.x.array
            target[function.function_space.dofmap.list] = values[source_dofs[node_map[cells]]]
            if not np.isfinite(target).all():
                raise ValueError(f"Nonfinite or unmapped checkpoint field: {field}")
            function.x.scatter_forward()
        report["checkpoint"] = {
            "step": int(state["step"]), "iteration": int(state["iteration"]),
            "load_factor": float(state["load_factor"]),
            "meaning": "state saved BEFORE this iteration; not the rejected post-mechanics field or an accepted final solution",
        }
    domain.topology.create_connectivity(3, 3)
    domain.topology.create_connectivity(3, 3)
    wall_dofs = fem.locate_dofs_topological(alpha_space, 3, tags.find(2))
    plinth_dofs = fem.locate_dofs_topological(alpha_space, 3, tags.find(1))
    report["checkpoint"]["plinth_max_damage"] = float(alpha.x.array[plinth_dofs].max())
    report["checkpoint"]["shared_damage_dofs"] = int(len(np.intersect1d(wall_dofs, plinth_dofs)))
    for tag, name in ((8, "wall_base"), (9, "plinth_top")):
        dofs = fem.locate_dofs_topological(alpha_space, 2, facets.find(tag))
        report["checkpoint"][f"{name}_damage_range"] = [
            float(alpha.x.array[dofs].min()), float(alpha.x.array[dofs].max()),
        ]
    norms = fem.Function(fem.functionspace(domain, ("DG", 0)))
    norms.interpolate(fem.Expression(
        ufl.sqrt(ufl.inner(strain(u), strain(u))), norms.function_space.element.interpolation_points,
    ))
    report["checkpoint"]["strain_peak"] = _strain_breakdown_details(u, alpha, norms)
    residual = manifest["automatic_crack"]["specification"]["residual_stiffness"]
    peak_damage = np.asarray(report["checkpoint"]["strain_peak"]["damage_at_vertices"])
    report["checkpoint"]["strain_peak"]["hybrid_vertex_stiffness_fraction_range"] = [
        float(((1 - peak_damage)**2 + residual).min()),
        float(((1 - peak_damage)**2 + residual).max()),
    ]
    point = np.asarray(report["checkpoint"]["strain_peak"]["centroid_xyz_m"])
    for tag in (8, 9):
        centroids = mesh.compute_midpoints(domain, 2, facets.find(tag))
        distance = np.linalg.norm(centroids - point, axis=1)
        nearest = int(np.argmin(distance))
        report["interface_facets"][str(tag)]["nearest_centroid_to_checkpoint_strain_peak_m"] = centroids[nearest].tolist()
        report["interface_facets"][str(tag)]["centroid_distance_not_exact_surface_distance_m"] = float(distance[nearest])
    report["limitations"] = [
        "Topology and saved-state inspection only; no new equilibrium solve.",
        "Checkpoint can be an unconverged iterate and is not a dam-safety result.",
        "Ground-spring support does not implement wall/plinth contact or load redistribution.",
        "No inference that removing damage constraints or bonding nodes is physically justified.",
    ]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output}")
    result = audit(args.run)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
