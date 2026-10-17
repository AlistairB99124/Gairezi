"""Consistent L2 nodal stress recovery followed by element tensor averaging."""
from pathlib import Path

from dolfinx import fem, geometry, io, mesh
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import ufl


def recover_stress(domain, expression, output: Path, *, sample_points: dict | None = None) -> dict:
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
    if reason <= 0:
        raise RuntimeError(f"Nodal stress projection failed: PETSc reason {reason}")
    nodal.name = "recovered_nodal_stress_pa"
    with io.VTKFile(domain.comm, output / "recovered_stress.pvd", "w") as writer:
        writer.write_function(nodal)

    count = domain.topology.index_map(3).size_local
    cells = np.arange(count, dtype=np.int32)
    dofs = np.asarray([space.dofmap.cell_dofs(int(cell)) for cell in cells])
    tensors = nodal.x.array.reshape(-1, 3, 3)[dofs].mean(axis=1)
    if not np.isfinite(tensors).all():
        raise ValueError("Recovered stress contains nonfinite values")
    principal = np.linalg.eigvalsh(tensors)
    xyz = domain.geometry.x[mesh.entities_to_geometry(domain, 3, cells)].mean(axis=1)
    tensile = int(np.argmax(principal[:, -1]))
    compressive = int(np.argmin(principal[:, 0]))
    peaks = domain.comm.allgather((
        float(principal[tensile, -1]), xyz[tensile].tolist(),
        float(principal[compressive, 0]), xyz[compressive].tolist(),
    ))
    tensile_peak = max(peaks, key=lambda peak: peak[0])
    compressive_peak = min(peaks, key=lambda peak: peak[2])
    samples = {}
    if sample_points:
        tree = geometry.bb_tree(domain, 3)
        for name, point in sample_points.items():
            coordinates = np.asarray(point, dtype=domain.geometry.x.dtype).reshape(1, 3)
            collisions = geometry.compute_colliding_cells(
                domain, geometry.compute_collisions_points(tree, coordinates), coordinates,
            )
            local = [
                nodal.eval(coordinates, np.asarray([cell], dtype=np.int32)).reshape(3, 3).tolist()
                for cell in collisions.links(0) if cell < count
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
        "maximum_principal_tensile_stress_pa": tensile_peak[0],
        "maximum_principal_compressive_stress_pa": compressive_peak[2],
        "matched_point_stresses": samples,
        "stress_recovery": {
            "method": "consistent L2 projection to P1 nodal tensors; element nodal-tensor mean; eigenvalues",
            "units": "Pa",
            "averaging": "mesh connectivity only; separate coincident interface nodes remain separate",
            "tensile_peak_centroid_xyz_m": tensile_peak[1],
            "compressive_peak_centroid_xyz_m": compressive_peak[3],
            "projection_convergence_reason": reason,
            "projection_relative_tolerance": 1e-12,
        },
    }
