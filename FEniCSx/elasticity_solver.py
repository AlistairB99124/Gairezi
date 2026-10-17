"""Rigid-body near-nullspace for three-dimensional P1 elasticity and GAMG."""
from dolfinx import la
from petsc4py import PETSc


def rigid_body_near_nullspace(space):
    coordinates = space.tabulate_dof_coordinates()
    modes = [la.vector(space.dofmap.index_map, bs=space.dofmap.index_map_bs) for _ in range(6)]
    for mode in modes:
        mode.array[:] = 0
    for component in range(3):
        modes[component].array[component::3] = 1
    x, y, z = coordinates.T
    modes[3].array[0::3], modes[3].array[1::3] = -y, x
    modes[4].array[0::3], modes[4].array[2::3] = z, -x
    modes[5].array[1::3], modes[5].array[2::3] = -z, y
    la.orthonormalize(modes)
    owned_size = space.dofmap.index_map.size_local * space.dofmap.index_map_bs
    vectors = [
        PETSc.Vec().createWithArray(mode.array[:owned_size], comm=space.mesh.comm)
        for mode in modes
    ]
    return PETSc.NullSpace().create(vectors=vectors, comm=space.mesh.comm)
