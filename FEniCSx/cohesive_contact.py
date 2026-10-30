"""Experimental exact-bond/cohesive/contact equilibrium on paired P1 lips.

Uninitiated vertex pairs are exactly tied (no penalty compliance), so the
baseline continuum stiffness and load paths are preserved. Initiation is an
explicit, audited state change: a tied pair whose recovered normal tie
traction reaches the tensile strength may be released to the extrinsic linear
Mode-I law in ``cohesive_mode_i``. Initiated pairs carry the cohesive traction
in opening and unilateral frictionless-normal contact in closure; closure never
heals tensile history. Tangential transfer after initiation is an explicit
choice: ``tied`` (no slip, Mode-I only) or ``frictionless`` (free slip).

Integration is lumped at paired vertices (tributary area of adjacent lip
triangles), which is exact for uniform traction on affine faces and avoids the
spurious traction oscillation of consistent cohesive integration. By default
paired nodes must belong to exactly one pair; ``allow_junctions`` admits
branching cracks and removes locally redundant junction constraint rows
(their multipliers are indeterminate and reported as zero).
This module does not decide hydraulic access, and nonconvex softening means
residual checks do not prove uniqueness or stability.
"""
from dataclasses import dataclass, field
import numbers

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import warnings

if __package__:
    from .cohesive_mode_i import LinearModeICohesiveLaw, ModeIHistory
    from .fracture_faces import PairedCrackFaces
    from .fracture_water_access import water_access
    from .hydrostatic_load import triangle_pressure_weights
else:
    from cohesive_mode_i import LinearModeICohesiveLaw, ModeIHistory
    from fracture_faces import PairedCrackFaces
    from fracture_water_access import water_access
    from hydrostatic_load import triangle_pressure_weights


TANGENTIAL_RESPONSES = ("tied", "frictionless")


def _positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _readonly(array):
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CohesiveInterface:
    """Vertex pairs with lumped tributary areas and unit minus-to-plus normals.

    ``allow_junctions`` admits a node in several distinct pairs (crack
    branches meeting at a vertex). Each unordered node pair may still occur
    only once; locally redundant junction constraints are eliminated by the
    solver and reported.
    """
    minus_nodes: np.ndarray
    plus_nodes: np.ndarray
    normals: np.ndarray
    areas_m2: np.ndarray
    coordinates_m: np.ndarray
    allow_junctions: bool = False

    def __post_init__(self):
        minus = np.asarray(self.minus_nodes)
        plus = np.asarray(self.plus_nodes)
        normals = np.asarray(self.normals, dtype=float)
        areas = np.asarray(self.areas_m2, dtype=float)
        coordinates = np.asarray(self.coordinates_m, dtype=float)
        count = len(minus)
        if type(self.allow_junctions) is not bool:
            raise ValueError("allow_junctions must be an explicit boolean")
        if (minus.dtype.kind not in "iu" or plus.dtype.kind not in "iu"
                or minus.shape != (count,) or plus.shape != (count,) or not count
                or normals.shape != (count, 3) or areas.shape != (count,)
                or coordinates.ndim != 2 or coordinates.shape[1] != 3):
            raise ValueError("Interface pairs, normals and areas must have matching shapes")
        if not (np.isfinite(normals).all() and np.isfinite(areas).all() and np.isfinite(coordinates).all()):
            raise ValueError("Interface geometry must be finite")
        if np.any(areas <= 0):
            raise ValueError("Interface tributary areas must be positive")
        if not np.allclose(np.linalg.norm(normals, axis=1), 1., rtol=0., atol=1e-12):
            raise ValueError("Interface normals must be unit vectors")
        nodes = np.concatenate((minus, plus))
        if np.any(minus == plus) or np.any(nodes < 0) or np.any(nodes >= len(coordinates)):
            raise ValueError("Each pair must join two distinct known source nodes")
        if self.allow_junctions:
            keys = np.sort(np.stack((minus, plus), axis=1), axis=1)
            if len(np.unique(keys, axis=0)) != count:
                raise ValueError("Each unordered junction node pair may occur only once")
        elif len(np.unique(nodes)) != len(nodes):
            raise ValueError("Each paired source node must belong to exactly one distinct pair; "
                             "crack junctions are outside this cohesive gate")
        for name, value in (("minus_nodes", minus), ("plus_nodes", plus), ("normals", normals),
                            ("areas_m2", areas), ("coordinates_m", coordinates)):
            object.__setattr__(self, name, _readonly(value))

    @property
    def count(self) -> int:
        return len(self.minus_nodes)

    @classmethod
    def from_faces(cls, faces: PairedCrackFaces, *, maximum_normal_deviation_rad: float,
                   allow_junctions: bool = False):
        """Aggregate face vertices into pairs; reject kinks beyond the stated deviation.

        With ``allow_junctions`` a node pair met in reversed owner order is the
        same pair with its face normal flipped, so orientation stays minus-to-plus.
        """
        if not isinstance(faces, PairedCrackFaces):
            raise ValueError("Cohesive interface requires explicit paired faces")
        deviation = _positive(maximum_normal_deviation_rad, "maximum_normal_deviation_rad")
        if deviation >= np.pi / 2:
            raise ValueError("maximum_normal_deviation_rad must be less than pi/2")
        lookup, minus, plus, weighted, areas, members = {}, [], [], [], [], []
        for face, (minus_row, plus_row) in enumerate(zip(faces.minus, faces.plus)):
            share = faces.areas[face] / 3
            for first, second in zip(minus_row, plus_row):
                if first == second:
                    continue  # tied crack-front vertex: continuous, not an interface pair
                key = (int(first), int(second))
                sign = 1.
                if allow_junctions and key not in lookup and key[::-1] in lookup:
                    key, sign = key[::-1], -1.
                if key not in lookup:
                    lookup[key] = len(minus)
                    minus.append(key[0])
                    plus.append(key[1])
                    weighted.append(np.zeros(3))
                    areas.append(0.)
                    members.append([])
                index = lookup[key]
                weighted[index] = weighted[index] + sign * share * faces.normals[face]
                areas[index] += share
                members[index].append((face, sign))
        if not minus:
            raise ValueError("Paired faces contain no untied vertex pairs")
        normals = np.asarray(weighted)
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 0):
            raise ValueError("Opposed face normals cancel at an interface vertex")
        normals /= lengths[:, None]
        limit = np.cos(deviation)
        for index, faces_at_pair in enumerate(members):
            ids = np.array([face for face, _ in faces_at_pair])
            signs = np.array([sign for _, sign in faces_at_pair])
            if np.min(signs * (faces.normals[ids] @ normals[index])) < limit:
                raise ValueError("Interface kink exceeds maximum_normal_deviation_rad at a vertex pair")
        return cls(np.asarray(minus), np.asarray(plus), normals, np.asarray(areas), faces.coordinates,
                   allow_junctions)

    def tangents(self):
        normals = self.normals
        helper = np.where(np.abs(normals[:, :1]) < .9, [[1., 0., 0.]], [[0., 1., 0.]])
        first = np.cross(normals, helper)
        first /= np.linalg.norm(first, axis=1)[:, None]
        return first, np.cross(normals, first)


@dataclass(frozen=True)
class InterfaceState:
    """Committed irreversible history; trial states never mutate it."""
    initiated: np.ndarray
    maximum_opening_m: np.ndarray

    def __post_init__(self):
        initiated = np.asarray(self.initiated)
        opening = np.asarray(self.maximum_opening_m, dtype=float)
        if (initiated.dtype != np.dtype(bool) or initiated.ndim != 1
                or opening.shape != initiated.shape or not np.isfinite(opening).all()
                or np.any(opening < 0) or np.any(opening[~initiated] != 0)):
            raise ValueError("State requires boolean initiation and nonnegative openings only on initiated pairs")
        object.__setattr__(self, "initiated", _readonly(initiated))
        object.__setattr__(self, "maximum_opening_m", _readonly(opening))

    @classmethod
    def intact(cls, interface: CohesiveInterface):
        return cls(np.zeros(interface.count, dtype=bool), np.zeros(interface.count))

    def initiate(self, pairs):
        ids = np.asarray(pairs)
        if ids.dtype.kind not in "iu" or ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(self.initiated)):
            raise ValueError("Initiation requires known pair indices")
        if np.any(self.initiated[ids]):
            raise ValueError("Pairs are already initiated; history cannot be reset")
        initiated = self.initiated.copy()
        initiated[ids] = True
        return InterfaceState(initiated, self.maximum_opening_m)


@dataclass(frozen=True)
class CohesiveOptions:
    gap_tolerance_m: float
    force_tolerance_n: float
    relative_residual_tolerance: float
    initiation_relative_tolerance: float
    tangential_response: str
    maximum_newton_iterations: int = 50
    maximum_active_set_iterations: int = 200

    def __post_init__(self):
        for name in ("gap_tolerance_m", "force_tolerance_n", "relative_residual_tolerance",
                     "initiation_relative_tolerance"):
            _positive(getattr(self, name), name)
        if self.relative_residual_tolerance >= 1 or self.initiation_relative_tolerance >= 1:
            raise ValueError("Relative tolerances must be less than one")
        if self.tangential_response not in TANGENTIAL_RESPONSES:
            raise ValueError("tangential_response must be explicitly 'tied' or 'frictionless'")
        for name in ("maximum_newton_iterations", "maximum_active_set_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class CohesiveEquilibrium:
    displacement_m: np.ndarray
    support_reaction_n: np.ndarray
    normal_opening_m: np.ndarray
    normal_traction_pa: np.ndarray
    trial_state: InterfaceState
    pair_status: np.ndarray
    report: dict = field(default_factory=dict)
    water_force_n: np.ndarray = None


def _pair_rows(interface, mapping, directions):
    count = interface.count
    rows = np.repeat(np.arange(count), 6)
    minus = 3 * mapping[interface.minus_nodes]
    plus = 3 * mapping[interface.plus_nodes]
    columns = np.stack([minus, minus + 1, minus + 2, plus, plus + 1, plus + 2], axis=1).ravel()
    values = np.concatenate((-directions, directions), axis=1).ravel()
    rows = sparse.csr_matrix((values, (rows, columns)), shape=(count, 3 * len(mapping)))
    # Stored zeros would make rows acting only on fixed DOFs look movable.
    rows.eliminate_zeros()
    return rows


def _cohesive_response(law, opening, committed, mask):
    traction = np.zeros(len(opening))
    tangent = np.zeros(len(opening))
    stored = np.zeros(len(opening))
    maximum = np.array(committed, copy=True)
    for index in np.flatnonzero(mask):
        response = law.response(max(float(opening[index]), 0.), ModeIHistory(committed[index]))
        traction[index] = response.tensile_traction_pa
        # The clamp makes traction constant for interpenetration; the active set then switches to contact.
        tangent[index] = response.tangent_pa_per_m if opening[index] >= 0 else 0.
        stored[index] = response.stored_energy_j_m2
        maximum[index] = response.trial_history.maximum_opening_m
    return traction, tangent, stored, maximum


def _independent_junction_rows(free_constraint, pair_of_row, movable, interface):
    """Mask of linearly independent constraint rows at crack junctions.

    Only connected components of the active pair graph containing a cycle can
    be redundant. Rows are kept greedily in stacking order (normal rows first),
    so dropped rows are implied by kept ones and carry an indeterminate,
    reported-as-zero multiplier.
    """
    keep = np.array(movable, copy=True)
    rows = np.flatnonzero(movable)
    if not len(rows):
        return keep
    pairs = np.unique(pair_of_row[rows])
    parent = {}

    def find(node):
        while parent.setdefault(node, node) != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for pair in pairs:
        a, b = find(int(interface.minus_nodes[pair])), find(int(interface.plus_nodes[pair]))
        if a != b:
            parent[a] = b
    edges, vertices = {}, {}
    for pair in pairs:
        root = find(int(interface.minus_nodes[pair]))
        edges[root] = edges.get(root, 0) + 1
    for node in list(parent):
        root = find(node)
        vertices[root] = vertices.get(root, 0) + 1
    cyclic = {root for root, count in edges.items() if count >= vertices[root]}
    if not cyclic:
        return keep
    row_root = np.array([find(int(interface.minus_nodes[pair_of_row[row]])) for row in rows])
    for root in cyclic:
        members = rows[row_root == root]
        block = free_constraint[members]
        columns = np.unique(block.indices)
        dense = block[:, columns].toarray()
        basis = []
        for row, values in zip(members, dense):
            norm = np.linalg.norm(values)
            residual = values.copy()
            for vector in basis:
                residual -= (vector @ residual) * vector
            remaining = np.linalg.norm(residual)
            if remaining > 1e-10 * norm:
                basis.append(residual / remaining)
            else:
                keep[row] = False
    return keep


def solve_cohesive_equilibrium(stiffness, load_n, fixed_dofs, prescribed_m,
                               interface: CohesiveInterface, source_to_dof,
                               law: LinearModeICohesiveLaw, committed: InterfaceState, *,
                               options: CohesiveOptions) -> CohesiveEquilibrium:
    """Equilibrate one fixed load with fixed initiation; never initiates pairs.

    The returned trial state updates maximum openings only. Bonded pairs whose
    recovered normal tie traction exceeds f_t*(1+tolerance) are reported as
    ``initiation_required``; the caller decides whether to bisect the load or
    initiate pairs. Failures raise instead of returning an unverified state.
    """
    if not isinstance(interface, CohesiveInterface) or not isinstance(law, LinearModeICohesiveLaw):
        raise ValueError("Explicit CohesiveInterface and LinearModeICohesiveLaw are required")
    if not isinstance(committed, InterfaceState) or committed.initiated.shape != (interface.count,):
        raise ValueError("Committed InterfaceState must match the interface")
    if not isinstance(options, CohesiveOptions):
        raise ValueError("Explicit CohesiveOptions are required")
    mapping = np.asarray(source_to_dof)
    nodes = len(interface.coordinates_m)
    if (mapping.dtype.kind not in "iu" or mapping.shape != (nodes,)
            or not np.array_equal(np.sort(mapping), np.arange(nodes))):
        raise ValueError("Source-to-DOF map must be a permutation of all vector block DOFs")
    size = 3 * nodes
    matrix = sparse.csr_matrix(stiffness, dtype=float, copy=True)
    load = np.asarray(load_n, dtype=float)
    fixed = np.asarray(fixed_dofs)
    imposed = np.asarray(prescribed_m, dtype=float)
    if (matrix.shape != (size, size) or not np.isfinite(matrix.data).all()
            or load.shape != (size,) or not np.isfinite(load).all()
            or fixed.ndim != 1 or fixed.dtype.kind not in "iu" or np.any(fixed < 0) or np.any(fixed >= size)
            or len(np.unique(fixed)) != len(fixed) or imposed.shape != fixed.shape
            or not np.isfinite(imposed).all()):
        raise ValueError("Finite stiffness/load and unique prescribed scalar DOFs must match the interface space")
    scale = max(float(np.max(np.abs(matrix.data), initial=0.)), np.finfo(float).tiny)
    if np.max(np.abs((matrix - matrix.T).data), initial=0.) > 1e-12 * scale:
        raise ValueError("Cohesive equilibrium requires symmetric elastic stiffness")
    free = np.setdiff1d(np.arange(size), fixed)
    if not len(free):
        raise ValueError("Cohesive equilibrium requires free displacement DOFs")

    normal_rows = _pair_rows(interface, mapping, interface.normals)
    first, second = interface.tangents()
    tangent_rows = (_pair_rows(interface, mapping, first), _pair_rows(interface, mapping, second))
    areas = interface.areas_m2
    initiated = committed.initiated
    history = committed.maximum_opening_m
    tied_tangent = options.tangential_response == "tied"
    strength = law.tensile_strength_pa
    # A just-initiated, never-opened pair can still carry f_t in closure-free tension.
    closed_tension_cap = np.where(history > 0, 0., strength) * areas

    displacement = np.zeros(size)
    displacement[fixed] = imposed
    contact = initiated.copy()
    seen = set()
    newton_total = 0
    newton_restarts = 0
    for active_iteration in range(1, options.maximum_active_set_iterations + 1):
        signature = tuple(np.flatnonzero(contact))
        if signature in seen:
            raise RuntimeError("Cohesive contact active set cycled; no equilibrium accepted")
        seen.add(signature)
        bonded = ~initiated
        normal_mask = bonded | contact
        tangential_mask = bonded | (initiated & tied_tangent)
        constraint = sparse.vstack(
            (normal_rows[normal_mask], tangent_rows[0][tangential_mask], tangent_rows[1][tangential_mask]),
            format="csr")
        tangential_pairs = np.flatnonzero(tangential_mask)
        pair_of_row = np.concatenate((np.flatnonzero(normal_mask), tangential_pairs, tangential_pairs))
        component_of_row = np.concatenate((
            np.zeros(int(normal_mask.sum()), dtype=int),
            np.ones(len(tangential_pairs), dtype=int), np.full(len(tangential_pairs), 2)))
        cohesive = initiated & ~contact
        free_constraint = constraint[:, free]
        movable = np.asarray(free_constraint.getnnz(axis=1)).ravel() > 0
        constant = np.asarray(constraint[:, fixed] @ imposed).ravel()
        if np.any(np.abs(constant[~movable]) > options.gap_tolerance_m):
            raise ValueError("Prescribed support displacements violate an interface tie or contact")
        redundant_rows = 0
        if interface.allow_junctions:
            independent = _independent_junction_rows(free_constraint, pair_of_row, movable, interface)
            redundant_rows = int(np.count_nonzero(movable & ~independent))
            movable &= independent
        free_constraint = free_constraint[movable]
        row_pair = pair_of_row[movable]
        row_component = component_of_row[movable]
        row_full = constraint[movable]
        multipliers = np.zeros(int(movable.sum()))
        converged = False
        # Full Newton first. If it stalls (e.g. a 2-cycle across a snap-back, where the softening slope
        # exceeds the structural stiffness so no equilibrium exists on the softening branch), restart once
        # from the fully separated state: one step with zero cohesive traction and stiffness, then Newton.
        restarted = False
        restart_step = False
        previous_openings = []
        limit = options.maximum_newton_iterations
        for newton in range(1, 2 * limit + 1):
            if newton > limit and not restarted:
                break
            newton_total += 1
            opening = np.asarray(normal_rows @ displacement).ravel()
            stalled = cohesive.any() and (newton == limit or (
                len(previous_openings) == 2
                and np.max(np.abs(opening - previous_openings[0]), initial=0.) <= options.gap_tolerance_m
                < np.max(np.abs(opening - previous_openings[1]), initial=0.)))
            restart_step = not restarted and stalled
            restarted |= restart_step
            newton_restarts += int(restart_step)
            previous_openings = (previous_openings + [opening])[-2:]
            traction, tangent, _, _ = _cohesive_response(law, opening, history, cohesive)
            if restart_step:
                traction = np.where(cohesive, 0., traction)
                tangent = np.where(cohesive, 0., tangent)
            internal = matrix @ displacement + normal_rows.T @ (areas * traction)
            residual = internal - load
            jacobian = matrix + normal_rows.T @ sparse.diags(areas * tangent) @ normal_rows
            reduced = (jacobian[free][:, free] / scale).tocsc()
            operator = sparse.bmat([[reduced, free_constraint.T], [free_constraint, None]], format="csc")
            force = np.concatenate((-residual[free] / scale, -np.asarray(row_full @ displacement).ravel()))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    solution = spsolve(operator, force)
            except MatrixRankWarning as error:
                raise RuntimeError(
                    "Cohesive tangent system is singular (e.g. a frictionless separated mechanism); "
                    "no equilibrium accepted") from error
            if not np.isfinite(solution).all():
                raise RuntimeError("Cohesive Newton step is non-finite; no equilibrium accepted")
            linear_error = np.linalg.norm(operator @ solution - force)
            if linear_error > 1e-8 * max(np.linalg.norm(force), np.finfo(float).tiny):
                raise RuntimeError(
                    "Cohesive tangent system is singular or ill-conditioned (e.g. a frictionless "
                    f"separated mechanism; linear residual {linear_error:.3g}); no equilibrium accepted")
            step = solution[:len(free)]
            multipliers = solution[len(free):] * scale
            displacement[free] += step
            step_norm = float(np.max(np.abs(step), initial=0.))
            if step_norm <= options.gap_tolerance_m * 1e-3 and not restart_step:
                converged = True
                break
        if not converged:
            raise RuntimeError("Cohesive Newton iteration limit reached; no equilibrium accepted")
        opening = np.asarray(normal_rows @ displacement).ravel()
        traction, tangent, stored, trial_maximum = _cohesive_response(law, opening, history, cohesive)
        constraint_force = -np.asarray(row_full.T @ multipliers).ravel()
        # Constraint force on the plus node, resolved per pair and component.
        pair_force = np.zeros((interface.count, 3))
        np.add.at(pair_force, (row_pair, row_component), -multipliers)
        normal_force = pair_force[:, 0]  # positive compresses the interface
        released = np.flatnonzero(contact & (-normal_force > closed_tension_cap + options.force_tolerance_n))
        if len(released):
            excess = -normal_force[released] - closed_tension_cap[released]
            contact[released[np.argmax(excess)]] = False
            continue
        penetrating = np.flatnonzero(cohesive & (opening < -options.gap_tolerance_m))
        if len(penetrating):
            contact[penetrating[np.argmin(opening[penetrating])]] = True
            continue
        cohesive_force = normal_rows.T @ (areas * traction)
        residual = matrix @ displacement + cohesive_force - load - constraint_force
        force_scale = max(float(np.linalg.norm(load)), float(np.linalg.norm(cohesive_force)),
                          float(np.linalg.norm(constraint_force)),
                          float(np.linalg.norm(matrix[:, fixed] @ imposed)), options.force_tolerance_n)
        relative = float(np.linalg.norm(residual[free]) / force_scale)
        tie_jump = np.max(np.abs(row_full @ displacement), initial=0.)
        if not np.isfinite(relative) or relative > options.relative_residual_tolerance or tie_jump > options.gap_tolerance_m:
            raise RuntimeError(
                "Cohesive equilibrium failed independent checks: "
                f"relative_residual={relative:.9g}, maximum_constrained_jump={tie_jump:.9g}")
        normal_traction = np.where(cohesive, traction, -normal_force / areas)
        clean_opening = np.where(initiated & ~contact, np.maximum(opening, 0.), 0.)
        trial = InterfaceState(initiated, np.where(cohesive, trial_maximum, history))
        overstressed = bonded & (normal_traction > strength * (1 + options.initiation_relative_tolerance))
        status = np.where(bonded, "bonded", np.where(contact, "closed", np.where(
            trial.maximum_opening_m >= law.critical_opening_m, "separated", "softening")))
        reaction = np.zeros(size)
        reaction[fixed] = residual[fixed]
        cohesive_stored = float(np.sum(areas * np.where(cohesive, stored, 0.)))
        dissipated = float(np.sum(areas * np.where(
            initiated, .5 * strength * np.minimum(trial.maximum_opening_m, law.critical_opening_m), 0.)))
        bonded_traction = normal_traction[bonded]
        return CohesiveEquilibrium(displacement, reaction, clean_opening, normal_traction, trial, status, {
            "status": "initiation_required" if overstressed.any() else "passed_cohesive_equilibrium",
            "active_set_iterations": active_iteration, "newton_iterations": newton_total, "newton_restarts": newton_restarts,
            "relative_free_residual": relative, "maximum_constrained_jump_m": float(tie_jump),
            "pairs": interface.count, "bonded_pairs": int(bonded.sum()),
            "junctions_allowed": interface.allow_junctions,
            "redundant_junction_constraint_rows": redundant_rows,
            "initiated_pairs": int(initiated.sum()), "closed_pairs": int((initiated & contact).sum()),
            "separated_pairs": int((status == "separated").sum()),
            "overstressed_bonded_pairs": int(overstressed.sum()),
            "maximum_bonded_tensile_ratio": float(np.max(bonded_traction, initial=-np.inf) / strength)
            if bonded.any() else None,
            "elastic_energy_j": float(.5 * displacement @ (matrix @ displacement)),
            "cohesive_stored_energy_j": cohesive_stored,
            "cohesive_dissipated_energy_j": dissipated,
            "tensile_strength_pa": strength, "fracture_energy_j_m2": law.fracture_energy_j_m2,
            "critical_opening_m": law.critical_opening_m,
            "tangential_response": options.tangential_response,
            "integration": "lumped vertex tributary area on affine paired lips",
            "initiation_criterion": "bonded tie normal traction >= f_t (reference normal, tension positive)",
            "legacy_griffith_area_charge": "not applied; cohesive law supplies fracture dissipation",
            "displacement_uniqueness_verified": False, "stability_verified": False,
            "solver_ready": False, "engineering_validated": False,
        })
    raise RuntimeError("Cohesive contact active-set iteration limit reached; no equilibrium accepted")


@dataclass(frozen=True)
class ReservoirCrackWater:
    """Instantaneous drained reservoir head on open, inlet-connected initiated faces.

    Water acts only on faces whose every untied vertex pair is initiated and
    which are reachable from an explicit reservoir inlet edge through faces
    with aperture above the closure tolerance. Bonded material, closed contact
    and isolated openings receive no pressure; no trapped water, filling
    kinetics, flow losses or pore pressure in intact concrete are modelled.
    """
    head_elevation_m: float
    unit_weight_n_m3: float
    inlet_source_edges: np.ndarray
    parent_node_ids: np.ndarray
    closure_tolerance_m: float
    maximum_coupling_iterations: int = 50

    def __post_init__(self):
        if (isinstance(self.head_elevation_m, bool) or not isinstance(self.head_elevation_m, numbers.Real)
                or not np.isfinite(self.head_elevation_m)):
            raise ValueError("head_elevation_m must be finite")
        _positive(self.unit_weight_n_m3, "unit_weight_n_m3")
        _positive(self.closure_tolerance_m, "closure_tolerance_m")
        inlets = np.asarray(self.inlet_source_edges)
        parents = np.asarray(self.parent_node_ids)
        if inlets.dtype.kind not in "iu" or inlets.ndim != 2 or inlets.shape[1] != 2 or not len(inlets):
            raise ValueError("Reservoir inlet source edges must be explicit node pairs")
        if parents.dtype.kind not in "iu" or parents.ndim != 1 or np.any(parents < 0):
            raise ValueError("Parent node IDs must be nonnegative integers")
        if type(self.maximum_coupling_iterations) is not int or self.maximum_coupling_iterations < 1:
            raise ValueError("maximum_coupling_iterations must be a positive integer")
        object.__setattr__(self, "inlet_source_edges", _readonly(inlets))
        object.__setattr__(self, "parent_node_ids", _readonly(parents))


def _initiated_faces(faces, interface, state):
    lookup = {(int(a), int(b)): index for index, (a, b) in enumerate(zip(interface.minus_nodes, interface.plus_nodes))}
    if interface.allow_junctions:
        lookup.update({(b, a): index for (a, b), index in list(lookup.items())})
    eligible = []
    for face, (minus, plus) in enumerate(zip(faces.minus, faces.plus)):
        pairs = [lookup.get((int(a), int(b))) for a, b in zip(minus, plus) if a != b]
        if any(pair is None for pair in pairs):
            raise ValueError("Paired faces contain vertex pairs absent from the cohesive interface")
        if all(state.initiated[pair] for pair in pairs):
            eligible.append(face)
    return np.asarray(eligible, dtype=int)


def _wet_faces(faces, interface, state, displacement, mapping, water):
    eligible = _initiated_faces(faces, interface, state)
    if not len(eligible):
        return frozenset(), "no fully initiated faces"
    parents = water.parent_node_ids
    if parents.shape != (len(faces.coordinates),):
        raise ValueError("Parent node IDs must match the paired mesh")
    subset = PairedCrackFaces(faces.coordinates, faces.minus[eligible], faces.plus[eligible],
                              faces.normals[eligible])
    edges = {tuple(sorted((int(row[i]), int(row[j])))) for row in parents[subset.minus]
             for i, j in ((0, 1), (1, 2), (2, 0))}
    inlets = [edge for edge in water.inlet_source_edges if tuple(sorted(map(int, edge))) in edges]
    if not inlets:
        return frozenset(), "no reservoir inlet edge on the initiated crack"
    source_displacement = displacement.reshape(-1, 3)[mapping]
    access = water_access(subset, source_displacement, np.asarray(inlets), parents,
                          closure_tolerance_m=water.closure_tolerance_m)
    reachable = eligible[np.asarray(access["reachable_open_face_ids"], dtype=int)]
    submerged = faces.coordinates[faces.minus[reachable], 2].min(axis=1) < water.head_elevation_m \
        if len(reachable) else np.zeros(0, dtype=bool)
    return frozenset(map(int, reachable[submerged])), "reservoir connected"


def _water_force(faces, wet, mapping, water):
    force = np.zeros(3 * len(faces.coordinates))
    for face in sorted(wet):
        points = faces.coordinates[faces.minus[face]]
        weights = triangle_pressure_weights(points, float(water.head_elevation_m), float(water.unit_weight_n_m3))
        for node_plus, node_minus, weight in zip(faces.plus[face], faces.minus[face], weights):
            force[3 * mapping[node_plus] + np.arange(3)] += weight * faces.normals[face]
            force[3 * mapping[node_minus] + np.arange(3)] -= weight * faces.normals[face]
    return force


def solve_wet_cohesive_equilibrium(stiffness, load_n, fixed_dofs, prescribed_m, interface, faces,
                                   source_to_dof, law, committed, *, options, water: ReservoirCrackWater):
    """Fixed-point coupling of cohesive equilibrium and reservoir crack access."""
    if not isinstance(water, ReservoirCrackWater) or not isinstance(faces, PairedCrackFaces):
        raise ValueError("Explicit ReservoirCrackWater and PairedCrackFaces are required")
    mapping = np.asarray(source_to_dof)
    wet, history = frozenset(), []
    for iteration in range(1, water.maximum_coupling_iterations + 1):
        water_force = _water_force(faces, wet, mapping, water)
        result = solve_cohesive_equilibrium(
            stiffness, np.asarray(load_n, dtype=float) + water_force, fixed_dofs, prescribed_m,
            interface, mapping, law, committed, options=options)
        # Access is judged on the committed initiation of this trial; initiation is the caller's decision.
        updated, reason = _wet_faces(faces, interface, result.trial_state, result.displacement_m, mapping, water)
        if updated == wet:
            result.report.update({
                "water_model": "instantaneous drained reservoir head on open inlet-connected initiated faces",
                "water_coupling_iterations": iteration, "wet_face_ids": sorted(wet),
                "wet_area_m2": float(faces.areas[sorted(wet)].sum()) if wet else 0.,
                "water_resultant_n": water_force.reshape(-1, 3).sum(axis=0).tolist(),
                "water_access_note": reason,
                "water_limitations": "no trapped water, filling kinetics, flow losses or pore pressure in intact concrete",
            })
            result.water_force_n = water_force
            return result
        if updated in history:
            raise RuntimeError("Crack water access cycled between wet sets; no equilibrium accepted")
        history.append(wet)
        wet = updated
    raise RuntimeError("Crack water coupling iteration limit reached; no equilibrium accepted")


def advance_with_initiation(problem_at, start_factor, end_factor, interface, source_to_dof, law,
                            committed: InterfaceState, *, options: CohesiveOptions,
                            maximum_bisections: int = 60, maximum_initiation_rounds: int = 1000,
                            faces: PairedCrackFaces = None, water: ReservoirCrackWater = None):
    """Advance a monotone scalar loading/displacement parameter with exact initiation.

    ``problem_at(factor)`` returns ``(stiffness, load_n, fixed_dofs, prescribed_m)``.
    If the end factor overstresses bonded pairs, bisection finds the factor
    where the peak bonded ratio is within tolerance of one; pairs within the
    tolerance initiate together and the same factor is re-equilibrated until no
    bonded pair is overstressed. Returns accepted equilibria in order; the last
    trial state is the new committed history. Never accepts a state with
    unresolved overstress.
    """
    for name, value in (("start_factor", start_factor), ("end_factor", end_factor)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if type(maximum_bisections) is not int or maximum_bisections < 1:
        raise ValueError("maximum_bisections must be a positive integer")
    if (water is None) != (faces is None):
        raise ValueError("Crack water coupling requires both paired faces and reservoir water")
    strength_band = options.initiation_relative_tolerance
    accepted = []
    state = committed
    low = float(start_factor)
    target = float(end_factor)

    def solve(factor, current):
        stiffness, load, fixed, prescribed = problem_at(factor)
        if water is not None:
            return solve_wet_cohesive_equilibrium(stiffness, load, fixed, prescribed, interface, faces,
                                                  source_to_dof, law, current, options=options, water=water)
        return solve_cohesive_equilibrium(stiffness, load, fixed, prescribed, interface, source_to_dof,
                                          law, current, options=options)

    rounds = 0
    while True:
        result = solve(target, state)
        if result.report["status"] == "passed_cohesive_equilibrium":
            result.report["load_factor"] = target
            accepted.append(result)
            return accepted
        factor = target
        lower, upper = low, target
        for _ in range(maximum_bisections):
            factor = .5 * (lower + upper)
            result = solve(factor, state)
            ratio = result.report["maximum_bonded_tensile_ratio"]
            if abs(ratio - 1) <= strength_band:
                break
            if ratio < 1:
                # Accept the sub-critical state so irreversible history follows the path.
                result.report["load_factor"] = factor
                accepted.append(result)
                state = result.trial_state
                lower = low = factor
            else:
                upper = factor
        else:
            raise RuntimeError("Initiation bisection did not resolve the strength crossing; no state accepted")
        # Re-equilibrate at the crossing; redistribution may make further pairs
        # critical at the same factor (unstable growth), which initiate in later rounds.
        while True:
            rounds += 1
            if rounds > maximum_initiation_rounds:
                raise RuntimeError("Initiation round limit reached; no state accepted")
            ready = np.flatnonzero(~result.trial_state.initiated & (
                result.normal_traction_pa >= law.tensile_strength_pa * (1 - strength_band)))
            if not len(ready):
                raise RuntimeError("Overstressed state has no initiable pair; no state accepted")
            state = result.trial_state.initiate(ready)
            result = solve(factor, state)
            if result.report["status"] == "passed_cohesive_equilibrium":
                break
        result.report["load_factor"] = factor
        result.report["initiated_pairs_since_step_start"] = int(np.count_nonzero(
            result.trial_state.initiated & ~committed.initiated))
        accepted.append(result)
        state = result.trial_state
        low = factor
        if factor == target:
            return accepted
