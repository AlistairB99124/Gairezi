"""Consistent L2 nodal stress recovery followed by element tensor averaging."""
from pathlib import Path

from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import ufl


def recover_stress(domain, expression, output: Path) -> dict:
    """Match the Elmer reporting sequence without altering the mechanics solve."""
    if domain.comm.size != 1:
        raise RuntimeError("Stress recovery currently requires one MPI process")
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
    return {
        "maximum_principal_tensile_stress_pa": float(principal[tensile, -1]),
        "maximum_principal_compressive_stress_pa": float(principal[compressive, 0]),
        "stress_recovery": {
            "method": "consistent L2 projection to P1 nodal tensors; element nodal-tensor mean; eigenvalues",
            "units": "Pa",
            "averaging": "mesh connectivity only; separate coincident interface nodes remain separate",
            "tensile_peak_centroid_xyz_m": xyz[tensile].tolist(),
            "compressive_peak_centroid_xyz_m": xyz[compressive].tolist(),
            "projection_convergence_reason": reason,
            "projection_relative_tolerance": 1e-12,
        },
    }
