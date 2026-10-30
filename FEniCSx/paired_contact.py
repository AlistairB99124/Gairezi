"""Serial small-displacement frictionless contact on prescribed paired P1 faces.

Vertex gap constraints enforce nonpenetration everywhere on an affine face.
Multipliers are integrated nodal forces in N, not pointwise pressures.
No penalty stiffness, initial gap or tangential sticking is introduced.
"""
from dataclasses import dataclass
from dataclasses import asdict
import json
from pathlib import Path
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, MatrixRankWarning, minres, spsolve

if __package__:
    from .fracture_faces import PairedCrackFaces
else:
    from fracture_faces import PairedCrackFaces


@dataclass(frozen=True)
class ContactOptions:
    gap_tolerance_m: float
    force_tolerance_n: float
    relative_residual_tolerance: float
    maximum_iterations: int
    condense_bonds: bool = True
    linear_solver: str = "direct"
    linear_relative_tolerance: float = 1e-10
    linear_maximum_iterations: int = 2000

    def __post_init__(self):
        for name in ("gap_tolerance_m", "force_tolerance_n", "relative_residual_tolerance"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.relative_residual_tolerance >= 1:
            raise ValueError("relative_residual_tolerance must be less than one")
        if type(self.maximum_iterations) is not int or self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be a positive integer")
        if type(self.condense_bonds) is not bool:
            raise ValueError("condense_bonds must be an explicit boolean")
        if self.linear_solver not in ("direct", "minres", "mumps"):
            raise ValueError("linear_solver must be direct, minres or mumps")
        if (isinstance(self.linear_relative_tolerance, bool)
                or not isinstance(self.linear_relative_tolerance, (int, float))
                or not 0 < self.linear_relative_tolerance < 1):
            raise ValueError("linear_relative_tolerance must be finite and between zero and one")
        if type(self.linear_maximum_iterations) is not int or self.linear_maximum_iterations < 1:
            raise ValueError("linear_maximum_iterations must be a positive integer")


@dataclass
class ContactEquilibrium:
    displacement_m: np.ndarray
    contact_nodal_forces_n: np.ndarray
    support_reaction_n: np.ndarray
    signed_gaps_m: np.ndarray
    multipliers_n: np.ndarray
    report: dict


def gap_constraints(faces: PairedCrackFaces, source_to_dof, *, excluded_pairs=()):
    """Return unique vertex-normal rows in native scalar displacement order."""
    if not isinstance(faces, PairedCrackFaces):
        raise ValueError("Contact requires explicit paired faces")
    mapping = np.asarray(source_to_dof)
    count = len(faces.coordinates)
    if (mapping.dtype.kind not in "iu" or mapping.shape != (count,)
            or not np.array_equal(np.sort(mapping), np.arange(count))):
        raise ValueError("Source-to-DOF map must be a permutation of all vector block DOFs")
    rows, columns, values, identities = [], [], [], set()
    row = 0
    for minus, plus, normal in zip(faces.minus, faces.plus, faces.normals):
        for first, second in zip(minus, plus):
            if first == second:
                continue
            if tuple(sorted((int(first), int(second)))) in excluded_pairs:
                continue
            # Normal signs and lip exchange represent the same gap constraint.
            if first > second:
                first, second, direction = second, first, -normal
            else:
                direction = normal
            key = (int(first), int(second), *map(float, direction))
            if key in identities:
                continue
            identities.add(key)
            for node, sign in ((first, -1.), (second, 1.)):
                for component in range(3):
                    if direction[component] != 0.:
                        rows.append(row)
                        columns.append(3 * int(mapping[node]) + component)
                        values.append(sign * direction[component])
            row += 1
    return sparse.csr_matrix((values, (rows, columns)), shape=(row, 3 * count))


def bonded_constraints(faces, source_to_dof, *, face_mask=None):
    """Tie all jump components before an explicitly supplied cut is released."""
    gap_constraints(faces, source_to_dof)
    pairs = sorted({tuple(sorted((int(a), int(b))))
                    for index, (minus, plus) in enumerate(zip(faces.minus, faces.plus))
                    if face_mask is None or face_mask[index]
                    for a, b in zip(minus, plus) if a != b})
    # A spanning forest imposes the same equality space without redundant
    # multiplier rows when several candidate sheets share a node junction.
    parents = {}

    def root(node):
        parents.setdefault(node, node)
        if parents[node] != node:
            parents[node] = root(parents[node])
        return parents[node]

    independent = []
    for a, b in pairs:
        first, second = root(a), root(b)
        if first != second:
            parents[second] = first
            independent.append((a, b))
    rows, columns, values = [], [], []
    for pair_index, pair in enumerate(independent):
        for component in range(3):
            for node, sign in zip(pair, (-1., 1.)):
                rows.append(3 * pair_index + component)
                columns.append(3 * int(source_to_dof[node]) + component)
                values.append(sign)
    return sparse.csr_matrix((values, (rows, columns)),
                             shape=(3 * len(independent), 3 * len(source_to_dof)))


def mixed_constraints(faces, source_to_dof, bonded_face_mask):
    if __package__:
        from .fracture_faces import bonded_node_representatives
    else:
        from fracture_faces import bonded_node_representatives
    mask = np.asarray(bonded_face_mask)
    if mask.dtype != np.dtype(bool) or mask.shape != (len(faces.minus),):
        raise ValueError("Bonded face mask must be boolean and match the candidate inventory")
    ties = bonded_constraints(faces, source_to_dof, face_mask=mask)
    representatives = bonded_node_representatives(faces, mask)
    excluded = {tuple(sorted((int(a), int(b))))
                for minus, plus in zip(faces.minus, faces.plus) for a, b in zip(minus, plus)
                if representatives[a] == representatives[b]}
    normal = gap_constraints(faces, source_to_dof, excluded_pairs=excluded)
    return sparse.vstack((ties, normal), format="csr"), np.concatenate((
        np.ones(ties.shape[0], dtype=bool), np.zeros(normal.shape[0], dtype=bool)))


def solve_contact(stiffness, load_n, fixed_dofs, prescribed_m, faces, source_to_dof, *,
                  options: ContactOptions, iteration_callback=None, bonded=False,
                  bonded_face_mask=None) -> ContactEquilibrium:
    """Minimise elastic potential subject to nonnegative vertex-normal jumps.

    The supplied symmetric stiffness must admit the constrained equilibrium.
    Residual/contact checks do not prove positive definiteness or uniqueness.
    Contact is explicitly frictionless: both tangential components are free.
    Loads may include prescribed paired water tractions, but this function does
    not infer saturation, hydraulic access or trapped-fluid pressure.
    """
    if not isinstance(options, ContactOptions):
        raise ValueError("Explicit ContactOptions are required")
    if type(bonded) is not bool:
        raise ValueError("bonded must be an explicit boolean")
    if bonded_face_mask is not None:
        if bonded:
            raise ValueError("Choose all-bonded or an explicit bonded face mask")
        gap, equality = mixed_constraints(faces, source_to_dof, bonded_face_mask)
    else:
        gap = bonded_constraints(faces, source_to_dof) if bonded else gap_constraints(faces, source_to_dof)
        equality = np.full(gap.shape[0], bonded, dtype=bool)
    return _solve_constrained_contact(
        stiffness, load_n, fixed_dofs, prescribed_m, gap, equality,
        options=options, iteration_callback=iteration_callback, bonded=bonded)


def _solve_mumps_symmetric(operator, force):
    """Symmetric (indefinite) MUMPS LDL^T solve; avoids SuperLU fill on large 3D systems."""
    try:
        from petsc4py import PETSc
    except ImportError as error:
        raise RuntimeError("Contact linear_solver='mumps' requires petsc4py; no fallback solver used") from error
    matrix = sparse.csr_matrix(operator, dtype=float)
    matrix.sort_indices()
    size = matrix.shape[0]
    petsc_matrix = PETSc.Mat().createAIJ(
        size=(size, size), comm=PETSc.COMM_SELF,
        csr=(matrix.indptr.astype(PETSc.IntType), matrix.indices.astype(PETSc.IntType), matrix.data))
    petsc_matrix.assemble()
    petsc_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    solver = PETSc.KSP().create(PETSc.COMM_SELF)
    try:
        solver.setOperators(petsc_matrix)
        solver.setType("preonly")
        preconditioner = solver.getPC()
        # PETSc Cholesky with MUMPS on a matrix not flagged SPD is LDL^T (MUMPS SYM=2).
        preconditioner.setType("cholesky")
        preconditioner.setFactorSolverType("mumps")
        right_hand_side = petsc_matrix.createVecLeft()
        right_hand_side.array[:] = force
        result = petsc_matrix.createVecRight()
        try:
            solver.solve(right_hand_side, result)
        except PETSc.Error as error:
            raise RuntimeError("Contact MUMPS factorization failed; no equilibrium accepted") from error
        reason = solver.getConvergedReason()
        if reason < 0:
            raise RuntimeError(
                f"Contact MUMPS factorization failed (reason {reason}); "
                "the constraint system may be singular; no equilibrium accepted")
        return result.array.copy()
    finally:
        solver.destroy()
        petsc_matrix.destroy()


def _solve_constrained_contact(stiffness, load_n, fixed_dofs, prescribed_m, gap, equality, *,
                               options, iteration_callback=None, bonded=False, allow_fully_prescribed=False):
    size = gap.shape[1]
    matrix = sparse.csr_matrix(stiffness, dtype=float, copy=True)
    load = np.asarray(load_n)
    fixed = np.asarray(fixed_dofs)
    imposed = np.asarray(prescribed_m)
    if (matrix.shape != (size, size) or not np.isfinite(matrix.data).all()
            or load.shape != (size,) or load.dtype.kind not in "fiu" or not np.isfinite(load).all()
            or fixed.ndim != 1 or fixed.dtype.kind not in "iu"
            or np.any(fixed < 0) or np.any(fixed >= size) or len(np.unique(fixed)) != len(fixed)
            or imposed.shape != fixed.shape or imposed.dtype.kind not in "fiu"
            or not np.isfinite(imposed).all()):
        raise ValueError("Finite stiffness/load and unique prescribed scalar DOFs must match contact space")
    difference = matrix - matrix.T
    scale = max(float(np.max(np.abs(matrix.data), initial=0.)), np.finfo(float).tiny)
    if np.max(np.abs(difference.data), initial=0.) > 1e-12 * scale:
        raise ValueError("Contact requires symmetric elastic stiffness")
    if options.condense_bonds and np.any(equality):
        return _solve_condensed_bonds(
            matrix, load, fixed, imposed, gap, equality, options=options,
            iteration_callback=iteration_callback, bonded=bonded)
    free = np.setdiff1d(np.arange(size), fixed)
    if not len(free):
        if not allow_fully_prescribed:
            raise ValueError("Contact solve requires free displacement DOFs")
        displacement = np.zeros(size)
        displacement[fixed] = imposed
        signed = np.asarray(gap @ displacement).ravel()
        if signed.min(initial=0.) < -options.gap_tolerance_m:
            raise ValueError("Prescribed support displacements force crack-face penetration")
        return ContactEquilibrium(
            displacement, np.zeros(size), matrix @ displacement - load,
            signed, np.zeros(gap.shape[0]), {
                "status": "passed_reduced_frictionless_contact", "iterations": 0,
                "active_constraints": 0, "constraints": gap.shape[0],
                "bonded_equality_constraints": 0, "relative_free_residual": 0.,
                "minimum_signed_gap_m": float(signed.min(initial=0.)),
                "minimum_contact_multiplier_n": 0., "maximum_complementarity_work_j": 0.,
                "elastic_energy_j": float(.5 * displacement @ (matrix @ displacement)),
                "contact_resultant_n": [0., 0., 0.],
                "friction": "explicitly frictionless; tangential motion unconstrained",
                "displacement_uniqueness_verified": False, "stability_verified": False,
                "equilibrium_scope": "Exact prescribed quotient state; stability not assessed",
                "solver_ready": False, "engineering_validated": False})
    reduced = matrix[free][:, free] / scale
    rhs = (load[free] - matrix[free][:, fixed] @ imposed) / scale
    constraint = gap[:, free]
    constant = np.asarray(gap[:, fixed] @ imposed).ravel()
    movable = np.asarray(constraint.getnnz(axis=1)).ravel() > 0
    if (np.any(np.abs(constant[~movable & equality]) > options.gap_tolerance_m)
            or np.any(constant[~movable & ~equality] < -options.gap_tolerance_m)):
        raise ValueError("Prescribed support displacements force crack-face penetration")
    # Start with all movable zero-reference-gap constraints active. This also
    # preserves closure when the unconstrained split has large overlap.
    active = movable.copy()
    seen = set()
    linear_systems = []
    for iteration in range(1, options.maximum_iterations + 1):
        signature = tuple(np.flatnonzero(active))
        if signature in seen:
            raise RuntimeError("Contact active set cycled; no equilibrium accepted")
        seen.add(signature)
        ids = np.flatnonzero(active)
        if len(ids):
            block = constraint[ids]
            # MINRES and MUMPS LDL^T need the symmetric saddle-point form.
            sign = 1. if options.linear_solver == "direct" else -1.
            operator = sparse.bmat([[reduced, -block.T], [sign * block, None]], format="csc")
            force = np.concatenate((rhs, -sign * constant[ids]))
        else:
            operator, force = reduced.tocsc(), rhs
        linear_iterations = 0
        if options.linear_solver == "direct":
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    solution = spsolve(operator, force)
            except MatrixRankWarning as error:
                raise RuntimeError("Contact constraint system is singular; no equilibrium accepted") from error
        elif options.linear_solver == "mumps":
            solution = _solve_mumps_symmetric(operator, force)
        else:
            diagonal = reduced.diagonal()
            if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
                raise RuntimeError("MINRES requires positive mechanical diagonal entries; no regularization applied")
            if len(ids):
                schur_diagonal = np.asarray(block.power(2) @ (1. / diagonal)).ravel()
                diagonal = np.concatenate((diagonal, schur_diagonal))
            if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
                raise RuntimeError("MINRES constraint preconditioner is not positive; no equilibrium accepted")
            preconditioner = LinearOperator(operator.shape, matvec=lambda vector: vector / diagonal)

            def count_iteration(_):
                nonlocal linear_iterations
                linear_iterations += 1

            solution, info = minres(
                operator, force, rtol=options.linear_relative_tolerance,
                maxiter=options.linear_maximum_iterations, M=preconditioner,
                callback=count_iteration, check=True)
            if info != 0:
                raise RuntimeError(
                    f"Contact MINRES did not converge (info={info}, iterations={linear_iterations}); "
                    "no direct-solver fallback or equilibrium accepted")
        linear_systems.append({
            "active_set_iteration": iteration, "method": options.linear_solver,
            "system_dofs": operator.shape[0], "iterations": linear_iterations,
        })
        if not np.isfinite(solution).all():
            raise RuntimeError("Contact solve returned non-finite displacement or multipliers")
        displacement = np.zeros(size)
        displacement[fixed] = imposed
        displacement[free] = solution[:len(free)]
        multipliers = np.zeros(gap.shape[0])
        multipliers[ids] = solution[len(free):] * scale
        signed = np.asarray(gap @ displacement).ravel()
        if iteration_callback is not None:
            iteration_callback({
                "iteration": iteration, "active_constraints": int(active.sum()),
                "minimum_signed_gap_m": float(signed.min(initial=0.)),
                "minimum_contact_multiplier_n": float(multipliers[~equality].min(initial=0.)),
            })
        tensile = ids[~equality[ids] & (multipliers[ids] < -options.force_tolerance_n)]
        if len(tensile):
            active[tensile[np.argmin(multipliers[tensile])]] = False
            continue
        penetrating = np.flatnonzero(movable & ~active & (signed < -options.gap_tolerance_m))
        if len(penetrating):
            active[penetrating[np.argmin(signed[penetrating])]] = True
            continue
        contact = np.asarray(gap.T @ multipliers).ravel()
        residual = matrix @ displacement - load - contact
        force_scale = max(float(np.linalg.norm(load)), float(np.linalg.norm(contact)),
                          float(np.linalg.norm(matrix[:, fixed] @ imposed)),
                          options.force_tolerance_n)
        relative = float(np.linalg.norm(residual[free]) / force_scale)
        complementarity = float(np.max(np.abs(multipliers * signed), initial=0.))
        complementarity_limit = (options.gap_tolerance_m * max(
            float(np.max(np.abs(multipliers), initial=0.)), options.force_tolerance_n))
        invalid_constraint = (
            np.max(np.abs(signed[equality]), initial=0.) > options.gap_tolerance_m
            or signed[~equality].min(initial=0.) < -options.gap_tolerance_m
            or multipliers[~equality].min(initial=0.) < -options.force_tolerance_n)
        if (not np.isfinite(relative) or relative > options.relative_residual_tolerance
                or invalid_constraint or complementarity > complementarity_limit):
            raise RuntimeError(
                "Contact equilibrium failed independent residual/complementarity checks: "
                f"relative_residual={relative:.9g}, "
                f"minimum_gap={signed[~equality].min(initial=0.):.9g}, "
                f"maximum_bond_jump={np.max(np.abs(signed[equality]), initial=0.):.9g}, "
                f"minimum_normal_force={multipliers[~equality].min(initial=0.):.9g}, "
                f"complementarity={complementarity:.9g}, limit={complementarity_limit:.9g}")
        reaction = np.zeros(size)
        reaction[fixed] = residual[fixed]
        return ContactEquilibrium(displacement, contact, reaction, signed, multipliers, {
            "status": "passed_reduced_bonded_equilibrium" if bonded else "passed_reduced_frictionless_contact",
            "iterations": iteration, "active_constraints": int(active.sum()),
            "linear_systems": linear_systems,
            "displacement_uniqueness_verified": False, "stability_verified": False,
            "equilibrium_scope": (
                "Force balance and contact admissibility only; zero-energy modes "
                "may make displacement/tangential slip nonunique."),
            "constraints": gap.shape[0], "relative_free_residual": relative,
            "bonded_equality_constraints": int(equality.sum()),
            "minimum_signed_gap_m": float(signed.min(initial=0.)),
            "minimum_contact_multiplier_n": float(multipliers[~equality].min(initial=0.)),
            "maximum_complementarity_work_j": complementarity,
            "elastic_energy_j": float(.5 * displacement @ (matrix @ displacement)),
            "contact_resultant_n": contact.reshape(-1, 3).sum(axis=0).tolist(),
            "friction": "bonded before cut release" if bonded else "explicitly frictionless; tangential motion unconstrained",
            "solver_ready": False, "engineering_validated": False,
        })
    raise RuntimeError("Contact active-set iteration limit reached; no equilibrium accepted")


def _solve_condensed_bonds(matrix, load, fixed, imposed, gap, equality, *,
                           options, iteration_callback, bonded):
    """Exact quotient space for bond forests; normal contact stays unilateral."""
    ties, normal = gap[equality].tocsr(), gap[~equality].tocsr()
    if (np.any(np.diff(ties.indptr) != 2)
            or not np.all(np.sort(ties.data.reshape(-1, 2), axis=1) == [-1., 1.])):
        raise ValueError("Bond condensation requires exact two-node equality rows")
    endpoints = ties.indices.reshape(-1, 2)
    edge_graph = sparse.csr_matrix(
        (np.repeat(np.arange(len(endpoints)) + 1, 2),
         (endpoints.ravel(), endpoints[:, ::-1].ravel())),
        shape=matrix.shape)
    count, groups = connected_components(edge_graph, directed=False)
    prolongation = sparse.csr_matrix(
        (np.ones(len(groups)), (np.arange(len(groups)), groups)), shape=(len(groups), count))
    condensed_fixed, first = np.unique(groups[fixed], return_index=True)
    condensed_imposed = imposed[first]
    lookup = np.searchsorted(condensed_fixed, groups[fixed])
    if not np.array_equal(imposed, condensed_imposed[lookup]):
        raise ValueError("Prescribed support displacements conflict across an exact bonded identity")
    condensed_matrix = (prolongation.T @ matrix @ prolongation).tocsr()
    condensed_normal = (normal @ prolongation).tocsr()
    condensed_normal.eliminate_zeros()
    free = np.setdiff1d(np.arange(len(groups)), fixed)
    active_bonds = int(np.count_nonzero(np.diff(ties[:, free].indptr)))

    def callback(row):
        if iteration_callback is not None:
            iteration_callback({**row, "active_constraints": row["active_constraints"] + active_bonds})

    solved = _solve_constrained_contact(
        condensed_matrix, np.asarray(prolongation.T @ load).ravel(),
        condensed_fixed, condensed_imposed, condensed_normal, np.zeros(normal.shape[0], dtype=bool),
        options=options, iteration_callback=callback, bonded=False, allow_fully_prescribed=True)
    displacement = np.asarray(prolongation @ solved.displacement_m).ravel()
    normal_force = np.asarray(normal.T @ solved.multipliers_n).ravel()
    demand = matrix @ displacement - load - normal_force
    # Recover original integrated bond forces by subtree force balance, not a
    # second indefinite solve. Fixed nodes absorb their own subtree reactions.
    supported = np.zeros(len(groups), dtype=bool)
    supported[fixed] = True
    visited = np.zeros(len(groups), dtype=bool)
    parent = np.full(len(groups), -1, dtype=np.int64)
    parent_row = np.full(len(groups), -1, dtype=np.int64)
    order = []
    nodes = np.flatnonzero(np.diff(edge_graph.indptr))
    for root in np.concatenate((fixed, nodes)):
        if visited[root]:
            continue
        visited[root] = True
        stack = [int(root)]
        while stack:
            node = stack.pop()
            order.append(node)
            for entry in range(edge_graph.indptr[node], edge_graph.indptr[node + 1]):
                neighbour = edge_graph.indices[entry]
                if not visited[neighbour]:
                    visited[neighbour] = True
                    parent[neighbour] = node
                    parent_row[neighbour] = edge_graph.data[entry] - 1
                    stack.append(int(neighbour))
    multipliers = np.zeros(gap.shape[0])
    multipliers[~equality] = solved.multipliers_n
    bond_multipliers = np.zeros(ties.shape[0])
    subtree = demand.copy()
    for node in reversed(order):
        ancestor, row = parent[node], parent_row[node]
        if ancestor < 0:
            continue
        force = 0. if supported[node] else subtree[node]
        sign = ties.data[ties.indptr[row] + int(endpoints[row, 1] == node)]
        bond_multipliers[row] = force / sign
        subtree[ancestor] += force
    multipliers[equality] = bond_multipliers
    contact = np.asarray(gap.T @ multipliers).ravel()
    residual = matrix @ displacement - load - contact
    force_scale = max(float(np.linalg.norm(load)), float(np.linalg.norm(contact)),
                      float(np.linalg.norm(matrix[:, fixed] @ imposed)), options.force_tolerance_n)
    relative = float(np.linalg.norm(residual[free]) / force_scale)
    signed = np.asarray(gap @ displacement).ravel()
    complementarity = float(np.max(np.abs(multipliers * signed), initial=0.))
    complementarity_limit = options.gap_tolerance_m * max(
        float(np.max(np.abs(multipliers), initial=0.)), options.force_tolerance_n)
    if (not np.isfinite(relative) or relative > options.relative_residual_tolerance
            or np.max(np.abs(signed[equality]), initial=0.) > options.gap_tolerance_m
            or signed[~equality].min(initial=0.) < -options.gap_tolerance_m
            or multipliers[~equality].min(initial=0.) < -options.force_tolerance_n
            or complementarity > complementarity_limit):
        raise RuntimeError("Condensed bonds failed original-space force/jump/complementarity audit; no equilibrium accepted")
    reaction = np.zeros(len(groups))
    reaction[fixed] = residual[fixed]
    report = {
        **solved.report,
        "status": "passed_reduced_bonded_equilibrium" if bonded else "passed_reduced_frictionless_contact",
        "constraints": gap.shape[0], "bonded_equality_constraints": ties.shape[0],
        "active_constraints": solved.report["active_constraints"] + active_bonds,
        "relative_free_residual": relative,
        "minimum_signed_gap_m": float(signed.min(initial=0.)),
        "minimum_contact_multiplier_n": float(multipliers[~equality].min(initial=0.)),
        "maximum_complementarity_work_j": complementarity,
        "elastic_energy_j": float(.5 * displacement @ (matrix @ displacement)),
        "contact_resultant_n": contact.reshape(-1, 3).sum(axis=0).tolist(),
        "friction": "bonded before cut release" if bonded else "explicitly frictionless; tangential motion unconstrained",
        "bond_condensation": {
            "method": "exact identity quotient; original forest-force recovery",
            "original_displacement_dofs": len(groups), "condensed_displacement_dofs": count,
            "eliminated_bond_multipliers": ties.shape[0],
            "original_kkt_dofs_upper_bound": len(groups) + gap.shape[0],
            "condensed_kkt_dofs_upper_bound": count + normal.shape[0],
            "original_space_force_audit": True,
        },
    }
    return ContactEquilibrium(displacement, contact, reaction, signed, multipliers, report)


def solve_contact_equilibrium(
    space, faces, *, elastic_form, external_load, boundary_conditions,
    options: ContactOptions, output: Path, face_pressures_pa=None,
    interface_springs=None, additional_nodal_forces=None,
):
    """Persistent native 3-D fixed-cut contact with explicitly prescribed water.

    Water is an external traction only. Supplying it does not assert hydraulic
    reachability or solve flow, saturation, trapped-fluid state or fracture.
    """
    from dolfinx import fem, io
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector, set_bc
    from petsc4py import PETSc
    import ufl

    if __package__:
        from .fracture_faces import bulk_face_mapping, source_vector_values
    else:
        from fracture_faces import bulk_face_mapping, source_vector_values

    if not isinstance(options, ContactOptions):
        raise ValueError("Explicit ContactOptions are required")
    coordinates, cells, mapping = bulk_face_mapping(space, faces)
    if interface_springs is not None and interface_springs.space is not space:
        raise ValueError("Interface springs must belong to the supplied displacement space")
    field = fem.Function(space, name="contact_displacement_m")
    bcs = list(boundary_conditions)
    fixed = np.unique(np.concatenate([bc.dof_indices()[0] for bc in bcs])) if bcs else np.empty(0, dtype=int)
    extra = np.zeros_like(field.x.array)
    if additional_nodal_forces is not None:
        raw = np.asarray(additional_nodal_forces)
        if raw.shape != extra.shape or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
            raise ValueError("Additional nodal loads must match serial displacement DOFs")
        extra[:] = raw
    pressure = np.zeros(faces.minus.shape) if face_pressures_pa is None else np.asarray(face_pressures_pa)
    water_source = faces.pressure_forces(pressure)
    water = np.empty_like(water_source)
    water[mapping] = water_source
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    bilinear, load_form = elastic_form(trial, test), external_load(test)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running_reduced_native_contact", "solver_ready": False,
        "engineering_validated": False, "options": asdict(options),
        "physical_time_solved": False,
        "iterations": [],
        "limitations": "Prescribed fixed cut; frictionless, small-displacement contact. Water traction is explicitly prescribed, not transported. No irreversible crack initiation/growth or full-dam submission.",
    }

    def persist():
        (output / "contact_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def trace(row):
        report["iterations"].append(row)
        persist()

    operator, load = None, None
    persist()
    try:
        operator = assemble_matrix(fem.form(bilinear))
        operator.assemble()
        if interface_springs is not None:
            operator.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            if interface_springs.matrix is None:
                interface_springs.build_matrix(operator)
            interface_springs.add_to(operator)
            report["interface_springs"] = interface_springs.report
        load = assemble_vector(fem.form(load_form))
        load.array[:] += extra + water.ravel()
        prescribed = load.duplicate()
        try:
            prescribed.set(0.)
            set_bc(prescribed, bcs)
            imposed = prescribed.array[fixed].copy()
        finally:
            prescribed.destroy()
        offsets, columns, values = operator.getValuesCSR()
        stiffness = sparse.csr_matrix((values.copy(), columns.copy(), offsets.copy()), shape=operator.getSize())
        np.savez_compressed(
            output / "native_inputs.npz", coordinates_m=coordinates, cells=cells,
            minus=faces.minus, plus=faces.plus, normals=faces.normals,
            source_to_dof=mapping, fixed_dofs=fixed, prescribed_m=imposed,
            assembled_external_force_n=load.array, pressure_pa=pressure,
            nodal_water_forces_n=water_source, additional_nodal_forces_n=extra,
            stiffness_indptr=offsets, stiffness_indices=columns, stiffness_data=values,
        )
        result = solve_contact(stiffness, load.array, fixed, imposed, faces, mapping,
                               options=options, iteration_callback=trace)
        field.x.array[:] = result.displacement_m
        field.x.scatter_forward()
        report["contact"] = result.report
        if interface_springs is not None:
            report["interface_spring_energy_j"] = interface_springs.energy(field)
        _, displacement = source_vector_values(field)
        report["opening"] = faces.measure(displacement)
        np.savez_compressed(
            output / "contact_state.npz", displacement_m=displacement,
            nodal_contact_forces_n=result.contact_nodal_forces_n.reshape(-1, 3)[mapping],
            support_reaction_n=result.support_reaction_n.reshape(-1, 3)[mapping],
            signed_gaps_m=result.signed_gaps_m, contact_multipliers_n=result.multipliers_n,
        )
        with io.VTKFile(space.mesh.comm, output / "displacement.pvd", "w") as writer:
            writer.write_function(field, 0.)
        report["status"] = "passed_reduced_native_contact"
        persist()
        return field, result, report
    except (ValueError, RuntimeError, PETSc.Error) as error:
        report["status"] = "failed_reduced_native_contact"
        report["error"] = str(error)
        persist()
        raise RuntimeError(f"Native contact failed; no equilibrium accepted: {error}") from error
    finally:
        if operator is not None:
            operator.destroy()
        if load is not None:
            load.destroy()
