"""Consistent L2 stress recovery with explicitly scoped nodal/element-mean peaks."""
from pathlib import Path

from dolfinx import fem, geometry, io, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import numpy as np
import ufl


def recover_stress(domain, expression, output: Path, *, sample_points: dict | None = None,
                   cell_tags=None, reference_stress=None) -> dict:
    """Match the Elmer reporting sequence without altering the mechanics solve."""
    space = fem.functionspace(domain, ("Lagrange", 1, (3, 3)))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    problem = LinearProblem(
        ufl.inner(trial, test) * ufl.dx,
        ufl.inner(expression, test) * ufl.dx,
        petsc_options_prefix="stress_recovery_",
        petsc_options={
            "ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12,
            "ksp_max_it": 2000, "ksp_error_if_not_converged": True,
        },
    )
    nodal = problem.solve()
    nodal.x.scatter_forward()
    reason = int(problem.solver.getConvergedReason())
    del problem
    if reason <= 0:
        raise RuntimeError(f"Nodal stress projection failed: PETSc reason {reason}")
    nodal.name = "recovered_nodal_stress_pa"
    with io.VTKFile(domain.comm, output / "recovered_stress.pvd", "w") as writer:
        writer.write_function(nodal)

    count = domain.topology.index_map(3).size_local
    cells = np.arange(count, dtype=np.int32)
    dofs = space.dofmap.list[:count]
    principal = np.empty((count, 3))
    for start in range(0, count, 65536):
        end = min(start + 65536, count)
        tensors = nodal.x.array.reshape(-1, 3, 3)[dofs[start:end]].mean(axis=1)
        if not np.isfinite(tensors).all():
            raise ValueError("Recovered stress contains nonfinite values")
        principal[start:end] = np.linalg.eigvalsh(tensors)
    body_peaks = {}
    bodies = [(1, "wall"), (2, "bedrock")]
    if cell_tags is not None and domain.comm.allreduce(len(cell_tags.find(3))) > 0:
        bodies.append((3, "foundation"))
    if cell_tags is not None:
        tags = np.zeros(count, dtype=np.int32)
        valid = cell_tags.indices < count
        tags[cell_tags.indices[valid]] = cell_tags.values[valid]
        for tag, name in bodies:
            selected = principal[tags == tag]
            records = domain.comm.allgather((
                float(selected[:, -1].max()) if len(selected) else -np.inf,
                float(selected[:, 0].min()) if len(selected) else np.inf,
            ))
            body_peaks[name] = {
                "maximum_principal_tensile_stress_pa": max(p[0] for p in records),
                "maximum_principal_compressive_stress_pa": min(p[1] for p in records),
            }
        principal[tags != 1, -1] = -np.inf
        principal[tags != 1, 0] = np.inf
    xyz = domain.geometry.x[mesh.entities_to_geometry(domain, 3, cells)].mean(axis=1)
    tensile = int(np.argmax(principal[:, -1]))
    compressive = int(np.argmin(principal[:, 0]))
    peaks = domain.comm.allgather((
        float(principal[tensile, -1]), xyz[tensile].tolist(),
        float(principal[compressive, 0]), xyz[compressive].tolist(),
    ))
    tensile_peak = max(peaks, key=lambda peak: peak[0])
    compressive_peak = min(peaks, key=lambda peak: peak[2])
    element_mean_stress = {
        "maximum_principal_tensile_stress_pa": tensile_peak[0],
        "maximum_principal_compressive_stress_pa": compressive_peak[2],
        "body_stress": body_peaks,
        "method": "principal eigenvalues of element-mean recovered nodal tensors",
    }
    peak_regions = {}
    stress_parity = {}
    if cell_tags is not None:
        owned_nodes = space.dofmap.index_map.size_local
        nodal_tensors = nodal.x.array.reshape(-1, 3, 3)[:owned_nodes]
        if not np.isfinite(nodal_tensors).all():
            raise ValueError("Recovered nodal stress contains nonfinite values")
        nodal_principal = np.linalg.eigvalsh(nodal_tensors)
        coordinates = space.tabulate_dof_coordinates()[:owned_nodes]
        if reference_stress is not None:
            if domain.comm.size != 1:
                raise ValueError("Source-order stress comparison requires a serial mesh")
            source_ids = domain.geometry.input_global_indices[domain.geometry.dofmaps[0]]
            reference = np.empty_like(nodal_tensors)
            reference[space.dofmap.list] = reference_stress[source_ids]
            concrete_ids = np.unique(space.dofmap.list[np.concatenate((
                cell_tags.find(1),
            ))])
            denominator = np.linalg.norm(reference[concrete_ids])
            if not np.isfinite(reference).all() or denominator <= 0:
                raise ValueError("Elmer nodal stress reference must be finite with nonzero concrete norm")
            relative_difference = float(
                np.linalg.norm((nodal_tensors - reference)[concrete_ids]) / denominator,
            )
            stress_parity = {
                "relative_nodal_stress_difference": relative_difference,
                "scope": "wall; matched source nodes; full recovered stress tensors",
                "maximum_relative_nodal_stress_difference": 1e-4,
            }
        nodal_peaks, body_peaks = [], {}
        for tag, name in bodies:
            ids = np.unique(space.dofmap.list[cell_tags.find(tag)])
            ids = ids[ids < owned_nodes]
            local = None
            if len(ids):
                tension = ids[np.argmax(nodal_principal[ids, -1])]
                compression = ids[np.argmin(nodal_principal[ids, 0])]
                local = (float(nodal_principal[tension, -1]), coordinates[tension].tolist(),
                         float(nodal_principal[compression, 0]), coordinates[compression].tolist(), tag)
            records = [record for record in domain.comm.allgather(local) if record is not None]
            if not records:
                raise ValueError(f"No recovered nodal stress for body {name}")
            maximum, minimum = max(records, key=lambda p: p[0]), min(records, key=lambda p: p[2])
            body_peaks[name] = {
                "maximum_principal_tensile_stress_pa": max(0., maximum[0]),
                "maximum_principal_compressive_stress_pa": min(0., minimum[2]),
            }
            if tag == 1:
                nodal_peaks.extend(records)
        tensile_peak = max(nodal_peaks, key=lambda p: p[0])
        compressive_peak = min(nodal_peaks, key=lambda p: p[2])
        peak_regions = {"tensile_peak_region_id": tensile_peak[4],
                        "compressive_peak_region_id": compressive_peak[4]}
    samples = {}
    if sample_points:
        # Peak coordinates are tabulated dof coordinates and may lie ~1e-14 m outside the
        # cells of their own body (wall base on the foundation, crest); pad the search.
        extent = domain.comm.allreduce(float(np.abs(domain.geometry.x).max(initial=0.)), op=MPI.MAX)
        tree = geometry.bb_tree(domain, 3, padding=1e-9 * max(extent, 1.))
        for name, point in sample_points.items():
            region = point.get("region_id") if isinstance(point, dict) else None
            point = point["coordinate_xyz_m"] if isinstance(point, dict) else point
            coordinates = np.asarray(point, dtype=domain.geometry.x.dtype).reshape(1, 3)
            collisions = geometry.compute_colliding_cells(
                domain, geometry.compute_collisions_points(tree, coordinates), coordinates,
            )
            local = [
                nodal.eval(coordinates, np.asarray([cell], dtype=np.int32)).reshape(3, 3).tolist()
                for cell in collisions.links(0) if cell < count
                and (cell_tags is None or tags[cell] == 1)
                and (region is None or (cell_tags is not None and tags[cell] == region))
            ]
            candidates = [tensor for rank_values in domain.comm.allgather(local) for tensor in rank_values]
            if not candidates:
                raise ValueError(f"Stress comparison point {name!r} is outside the mesh")
            tensors_at_point = np.asarray(candidates)
            if not np.isfinite(tensors_at_point).all() or not np.allclose(
                tensors_at_point, tensors_at_point[0], rtol=1e-8, atol=1e-5,
            ):
                raise ValueError(f"Stress comparison point {name!r} has ambiguous/nonfinite stress")
            values = np.linalg.eigvalsh(tensors_at_point[0])
            samples[name] = {
                "coordinate_xyz_m": coordinates[0].tolist(),
                "principal_stresses_pa": values.tolist(),
            }
    return {
        "element_mean_stress": element_mean_stress,
        "stress_parity": stress_parity,
        "body_stress": body_peaks,
        "maximum_principal_tensile_stress_pa": max(0., tensile_peak[0]) if cell_tags is not None else tensile_peak[0],
        "maximum_principal_compressive_stress_pa": min(0., compressive_peak[2]) if cell_tags is not None else compressive_peak[2],
        "matched_point_stresses": samples,
        "stress_recovery": {
            "method": ("consistent L2 projection to P1 nodal tensors; nodal eigenvalue extrema"
                       if cell_tags is not None else
                       "consistent L2 projection to P1 nodal tensors; element nodal-tensor mean; eigenvalues"),
            "peak_measure": "recovered_nodal" if cell_tags is not None else "recovered_element_mean",
            "units": "Pa",
            "averaging": "mesh connectivity only; separate coincident interface nodes remain separate",
            "tensile_peak_centroid_xyz_m": tensile_peak[1],
            "compressive_peak_centroid_xyz_m": compressive_peak[3],
            "peak_coordinate_location": "node" if cell_tags is not None else "element centroid",
            **peak_regions,
            "projection_convergence_reason": reason,
            "projection_relative_tolerance": 1e-12,
        },
    }
