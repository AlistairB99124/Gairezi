"""Experimental whole-wall extrinsic cohesive network on pre-split P1 tetrahedra.

Every candidate interior wall facet is split once. Bonded facets are condensed
exactly (their lip copies share one displacement), so the intact continuum and
its load paths are reproduced without penalty compliance. A bonded facet
initiates when the facet-normal traction n.sigma.n of either adjacent P1 cell
reaches f_t (one-sided, so a one-element-thick tension layer can initiate).
Initiation is irreversible; initiated facets enter the
coupled cohesive/contact solver with junctions allowed. An initiated facet that
cannot open in P1 (all its vertices tied by surrounding bonded facets) is
expanded by the minimal node-star patch with the least kink; patch facets start
closed with zero opening history and carry up to f_t before softening.

Limitations (reported in every result): mesh-facet crack paths only; kinked
patches use an area-averaged pair normal; the initiation stress is the larger
owner-cell value, not a converged crack-tip quantity; growth increments are at least one
vertex fan; batch initiation within the strength tolerance; no uniqueness or
stability proof for softening.
"""
from dataclasses import dataclass, field
import numbers
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

if __package__:
    from .cohesive_contact import (CohesiveInterface, CohesiveOptions, InterfaceState, ReservoirCrackWater,
                                   solve_cohesive_equilibrium, solve_wet_cohesive_equilibrium)
    from .cohesive_mode_i import LinearModeICohesiveLaw
    from .fracture_faces import PairedCrackFaces, bonded_node_representatives
    from .fracture_topology import NodeStarPatchResolver, SplitTetrahedra, split_interior_wall_faces
else:
    from cohesive_contact import (CohesiveInterface, CohesiveOptions, InterfaceState, ReservoirCrackWater,
                                  solve_cohesive_equilibrium, solve_wet_cohesive_equilibrium)
    from cohesive_mode_i import LinearModeICohesiveLaw
    from fracture_faces import PairedCrackFaces, bonded_node_representatives
    from fracture_topology import NodeStarPatchResolver, SplitTetrahedra, split_interior_wall_faces


LIMITATIONS = (
    "Mesh-facet crack paths only; kinked patches use an area-averaged pair normal; initiation uses the "
    "larger owner-cell facet-normal stress (one-sided; conservative and mesh dependent); growth increments are at least "
    "one P1 vertex fan; facets within the strength tolerance initiate together; softening equilibria are "
    "not proven unique or stable; reservoir water is instantaneous drained head on connected open faces.")


@dataclass(frozen=True)
class NetworkWater:
    head_elevation_m: float
    unit_weight_n_m3: float
    inlet_source_edges: np.ndarray
    closure_tolerance_m: float
    # Proportional loading: crack-water unit weight follows the exterior water load factor.
    scale_with_load_factor: bool = False


@dataclass(frozen=True)
class NetworkOptions:
    cohesive: CohesiveOptions
    maximum_normal_deviation_rad: float = 1.0
    maximum_bisections: int = 60
    maximum_initiation_rounds: int = 500
    maximum_solves: int = 5000

    def __post_init__(self):
        if not isinstance(self.cohesive, CohesiveOptions):
            raise ValueError("Explicit CohesiveOptions are required")
        deviation = self.maximum_normal_deviation_rad
        if (isinstance(deviation, bool) or not isinstance(deviation, numbers.Real)
                or not 0 < deviation < np.pi / 2):
            raise ValueError("maximum_normal_deviation_rad must lie in (0, pi/2)")
        for name in ("maximum_bisections", "maximum_initiation_rounds", "maximum_solves"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class NetworkResult:
    status: str
    load_factor: float
    split: object
    initiated: np.ndarray
    face_status: np.ndarray
    face_opening_m: np.ndarray
    face_normal_stress_pa: np.ndarray
    maximum_opening_history_m: np.ndarray
    displacement_m: np.ndarray
    wet_faces: np.ndarray
    steps: list = field(default_factory=list)
    report: dict = field(default_factory=dict)


class _Stop(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _cell_gradients(points):
    return np.linalg.inv(np.hstack((np.ones((4, 1)), points)))[1:]


def facet_normal_stress_operator(split, elasticity):
    """Sparse T (2m x dofs) with (T @ u)[2f + k] = n.sigma_k.n for owner cell k of candidate facet f.

    A facet with one owner repeats it. ``elasticity`` is the 6x6 Voigt matrix
    [xx, yy, zz, xy, yz, zx] with engineering shear strains, matching the
    caller's stiffness assembly. See ``facet_normal_stress``.
    """
    matrix = np.asarray(elasticity, dtype=float)
    if matrix.shape != (6, 6) or not np.isfinite(matrix).all():
        raise ValueError("Elasticity must be a finite 6x6 Voigt matrix")
    faces = split.faces
    rows, columns, values = [], [], []
    for face, (normal, owners) in enumerate(zip(faces.normals, split.adjacent_cells)):
        owners = list(owners)
        if not 1 <= len(owners) <= 2:
            raise ValueError("Candidate facets must have one or two owner cells")
        nx, ny, nz = normal
        weight = np.array((nx * nx, ny * ny, nz * nz, 2 * nx * ny, 2 * ny * nz, 2 * nz * nx)) @ matrix
        for side, cell in enumerate(owners if len(owners) == 2 else owners * 2):
            nodes = split.cells[cell]
            gradients = _cell_gradients(split.coordinates_m[nodes])
            strain = np.zeros((6, 12))
            for local in range(4):
                gx, gy, gz = gradients[:, local]
                strain[:, 3 * local:3 * local + 3] = [
                    [gx, 0, 0], [0, gy, 0], [0, 0, gz], [gy, gx, 0], [0, gz, gy], [gz, 0, gx]]
            rows.append(np.full(12, 2 * face + side))
            columns.append((3 * nodes[:, None] + np.arange(3)).ravel())
            values.append(weight @ strain)
    return sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                             shape=(2 * len(faces.minus), 3 * len(split.coordinates_m)))


def facet_normal_stress(operator, displacement):
    """One-sided initiation measure: the larger owner-cell n.sigma.n per candidate facet."""
    return (operator @ displacement).reshape(-1, 2).max(axis=1)


class CohesiveNetwork:
    """Candidate inventory, irreversible face state and condensed solves.

    ``problem_at(split, factor)`` returns ``(stiffness, load, fixed, prescribed)``
    on the split mesh DOFs (3 per split node, node-major). Loads/supports at
    split copies of one source node must be assigned by the caller (for
    example per owning cell or face); copies with different prescribed values
    cannot be condensed and are rejected.
    """

    def __init__(self, coordinates_m, cells, region_ids, candidate_faces, *, elasticity, law,
                 options: NetworkOptions, problem_at, water: NetworkWater = None,
                 should_stop=None, patch_resolver=None, split=None, progress=None):
        if not isinstance(law, LinearModeICohesiveLaw):
            raise ValueError("Explicit LinearModeICohesiveLaw is required")
        if not isinstance(options, NetworkOptions):
            raise ValueError("Explicit NetworkOptions are required")
        if water is not None and not isinstance(water, NetworkWater):
            raise ValueError("Crack water must be explicit NetworkWater")
        self.split = split_interior_wall_faces(coordinates_m, cells, region_ids, candidate_faces,
                                               drop_fully_tied=True) if split is None else split
        self.faces = self.split.faces
        self.count = len(self.faces.minus)
        self.law = law
        self.options = options
        self.problem_at = problem_at
        self.water = water
        self.should_stop = should_stop
        self.stress_operator = facet_normal_stress_operator(self.split, elasticity)
        self.resolver = patch_resolver or NodeStarPatchResolver(self.faces, self.split.parent_node_ids)
        self.keys = [tuple(map(int, key)) for key in self.split.source_faces]
        self.solves = 0
        self.progress = progress
        self._stage = "full load"
        self._accepted_factor = 0.
        self._reduction_cache = {}

    @classmethod
    def from_split(cls, split, **kwargs):
        """Reuse an existing ``split_interior_wall_faces(..., drop_fully_tied=True)`` result."""
        if not isinstance(split, SplitTetrahedra):
            raise ValueError("A SplitTetrahedra inventory is required")
        return cls(None, None, None, None, split=split, **kwargs)

    def _check_stop(self):
        if self.should_stop is not None and self.should_stop():
            raise _Stop("cancelled", "Cancelled by request")
        if self.solves >= self.options.maximum_solves:
            raise _Stop("resource_limit", f"Solve limit {self.options.maximum_solves} reached")

    def _topology(self, initiated):
        key = initiated.tobytes()
        cached = self._reduction_cache.get(key)
        if cached is not None:
            return cached
        reps = bonded_node_representatives(self.faces, ~initiated)
        unique, index = np.unique(reps, return_inverse=True)
        size = len(self.split.coordinates_m)
        dof_rows = np.arange(3 * size)
        dof_columns = (3 * index[:, None] + np.arange(3)).ravel()
        projection = sparse.csr_matrix((np.ones(3 * size), (dof_rows, dof_columns)),
                                       shape=(3 * size, 3 * len(unique)))
        coordinates = self.split.coordinates_m[unique]
        parents = self.split.parent_node_ids[unique]
        minus, plus = index[self.faces.minus], index[self.faces.plus]
        representable = initiated & np.any(minus != plus, axis=1)
        topology = {"index": index, "projection": projection, "coordinates": coordinates,
                    "parents": parents, "minus": minus, "plus": plus, "representable": representable,
                    "interface": None, "faces": None, "face_pairs": None}
        ids = np.flatnonzero(representable)
        if len(ids):
            faces = PairedCrackFaces(coordinates, minus[ids], plus[ids], self.faces.normals[ids])
            try:
                interface = CohesiveInterface.from_faces(
                    faces, maximum_normal_deviation_rad=self.options.maximum_normal_deviation_rad,
                    allow_junctions=True)
            except ValueError as error:
                raise _Stop("unsupported_geometry", f"Crack interface not representable: {error}") from error
            lookup = {(int(a), int(b)): pair for pair, (a, b)
                      in enumerate(zip(interface.minus_nodes, interface.plus_nodes))}
            lookup.update({(b, a): pair for (a, b), pair in list(lookup.items())})
            face_pairs = np.full((self.count, 3), -1)
            for face in ids:
                for vertex in range(3):
                    a, b = int(minus[face, vertex]), int(plus[face, vertex])
                    if a != b:
                        face_pairs[face, vertex] = lookup[(a, b)]
            topology.update({"interface": interface, "faces": faces, "face_ids": ids, "face_pairs": face_pairs})
        if len(self._reduction_cache) > 8:
            self._reduction_cache.clear()
        self._reduction_cache[key] = topology
        return topology

    def _reduce(self, topology, stiffness, load, fixed, prescribed):
        projection = topology["projection"]
        stiffness = sparse.csr_matrix(stiffness)
        reduced_stiffness = (projection.T @ stiffness @ projection).tocsr()
        reduced_load = projection.T @ np.asarray(load, dtype=float)
        index = topology["index"]
        imposed = {}
        for dof, value in zip(np.asarray(fixed), np.asarray(prescribed, dtype=float)):
            reduced = 3 * int(index[dof // 3]) + int(dof % 3)
            if reduced in imposed and abs(imposed[reduced] - value) > 1e-15 + 1e-12 * abs(value):
                raise ValueError("Bonded copies of one node carry inconsistent prescribed displacements")
            imposed[reduced] = float(value)
        reduced_fixed = np.fromiter(imposed.keys(), dtype=np.int64)
        reduced_prescribed = np.fromiter(imposed.values(), dtype=float)
        order = np.argsort(reduced_fixed)
        return reduced_stiffness, reduced_load, reduced_fixed[order], reduced_prescribed[order]

    def _linear(self, stiffness, load, fixed, prescribed):
        size = stiffness.shape[0]
        free = np.setdiff1d(np.arange(size), fixed)
        displacement = np.zeros(size)
        displacement[fixed] = prescribed
        rhs = load[free] - stiffness[free][:, fixed] @ prescribed
        displacement[free] = spsolve(stiffness[free][:, free].tocsc(), rhs)
        if not np.isfinite(displacement).all():
            raise _Stop("nonconvergence", "Intact elastic solve is singular or non-finite")
        residual = stiffness @ displacement - load
        scale = max(float(np.linalg.norm(load)), float(np.linalg.norm(residual[fixed])), 1e-300)
        relative = float(np.linalg.norm(residual[free]) / scale)
        if relative > self.options.cohesive.relative_residual_tolerance:
            raise _Stop("nonconvergence", f"Elastic residual {relative:.3g} exceeds tolerance")
        return displacement, relative

    def solve(self, factor, initiated, history):
        """One equilibrium at fixed face state; returns a trial dict (no commit)."""
        self._check_stop()
        self.solves += 1
        started = time.monotonic()
        trial = self._solve(factor, initiated, history)
        self._emit_progress(trial, time.monotonic() - started)
        return trial

    def _emit_progress(self, trial, seconds):
        if self.progress is None:
            return
        ratio = "n/a" if trial["ratio"] is None else f"{trial['ratio']:.3f}"
        opened = int(np.isin(trial["status"], ("softening", "separated")).sum())
        self.progress(f"cohesive solve {self.solves}/{self.options.maximum_solves} ({self._stage}): "
                      f"trial load factor {trial['factor']:.4f}, accepted {self._accepted_factor:.4f}, "
                      f"max bonded stress/f_t {ratio}, initiated faces {int(trial['initiated'].sum())}, "
                      f"opened {opened}, wet {int(trial['wet'].sum())}; {seconds:.1f} s")

    def _solve(self, factor, initiated, history):
        topology = self._topology(initiated)
        stiffness, load, fixed, prescribed = self.problem_at(self.split, factor)
        K, f, fixed_r, prescribed_r = self._reduce(topology, stiffness, load, fixed, prescribed)
        interface = topology["interface"]
        trial_history = history.copy()
        pair_status = None
        report = {}
        wet = np.zeros(self.count, dtype=bool)
        if interface is None:
            displacement, relative = self._linear(K, f, fixed_r, prescribed_r)
            report = {"relative_free_residual": relative, "pairs": 0,
                      "redundant_junction_constraint_rows": 0, "cohesive_dissipated_energy_j": 0.,
                      "elastic_energy_j": float(.5 * displacement @ (K @ displacement))}
            pair_opening = pair_traction = None
        else:
            pairs = topology["face_pairs"]
            pair_history = np.zeros(interface.count)
            mask = pairs >= 0
            np.maximum.at(pair_history, pairs[mask], history[mask])
            committed = InterfaceState(np.ones(interface.count, dtype=bool), pair_history)
            mapping = np.arange(len(topology["coordinates"]))
            try:
                unit_weight = 0. if self.water is None else float(self.water.unit_weight_n_m3) * (
                    float(factor) if self.water.scale_with_load_factor else 1.)
                if unit_weight <= 0:
                    result = solve_cohesive_equilibrium(K, f, fixed_r, prescribed_r, interface, mapping,
                                                        self.law, committed, options=self.options.cohesive)
                else:
                    water = ReservoirCrackWater(
                        float(self.water.head_elevation_m), unit_weight,
                        np.asarray(self.water.inlet_source_edges), topology["parents"],
                        float(self.water.closure_tolerance_m))
                    try:
                        result = solve_wet_cohesive_equilibrium(
                            K, f, fixed_r, prescribed_r, interface, topology["faces"], mapping, self.law,
                            committed, options=self.options.cohesive, water=water)
                    except ValueError as error:
                        raise _Stop("unsupported_hydraulics", f"Crack water access is not representable: {error}") from error
                    wet[topology["face_ids"][sorted(result.report["wet_face_ids"])]] = True
            except RuntimeError as error:
                raise _Stop("nonconvergence", str(error)) from error
            displacement = result.displacement_m
            report = dict(result.report)
            pair_opening = result.normal_opening_m
            pair_traction = result.normal_traction_pa
            pair_status = result.pair_status
            trial_history[mask] = result.trial_state.maximum_opening_m[pairs[mask]]
        split_displacement = (topology["projection"] @ displacement)
        stress = facet_normal_stress(self.stress_operator, split_displacement)
        bonded_stress = np.where(initiated, -np.inf, stress)
        ratio = float(np.max(bonded_stress, initial=-np.inf) / self.law.tensile_strength_pa) \
            if (~initiated).any() else None
        face_opening = np.zeros(self.count)
        face_stress = stress.copy()
        status = np.where(initiated, "initiated_tied", "bonded").astype(object)
        if pair_opening is not None:
            pairs = topology["face_pairs"]
            rank = {"closed": 0, "softening": 1, "separated": 2}
            for face in topology["face_ids"]:
                members = pairs[face][pairs[face] >= 0]
                face_opening[face] = float(pair_opening[members].mean())
                face_stress[face] = float(pair_traction[members].mean())
                status[face] = max((str(pair_status[m]) for m in members), key=rank.get)
        return {"factor": float(factor), "initiated": initiated.copy(), "history": trial_history,
                "ratio": ratio, "stress": stress, "face_stress": face_stress, "face_opening": face_opening,
                "status": status.astype(str), "displacement": split_displacement, "wet": wet,
                "report": report, "topology": topology}

    def _expand(self, ready, initiated):
        """Initiate ready facets and minimal near-planar patches where they cannot open."""
        updated = initiated.copy()
        updated[ready] = True
        added = []
        unresolved = []
        topology = self._topology(updated)
        for face in sorted(ready, key=lambda f: self.keys[f]):
            if topology["representable"][face]:
                continue
            broken = set(np.flatnonzero(updated)) - {int(face)}
            patches = self.resolver(broken, [int(face)])
            if not patches:
                unresolved.append(int(face))
                continue
            normal = self.faces.normals[face]

            def score(patch):
                deviation = max(float(np.arccos(np.clip(abs(self.faces.normals[p] @ normal), -1, 1)))
                                for p in patch)
                return (deviation, len(patch), tuple(sorted(self.keys[p] for p in patch)))

            failure = None
            for chosen in sorted(patches, key=score):
                trial = updated.copy()
                trial[list(chosen)] = True
                try:
                    topology = self._topology(trial)
                except _Stop as error:
                    if error.status != "unsupported_geometry":
                        raise
                    failure = error
                    continue
                added.extend(p for p in chosen if not updated[p])
                updated = trial
                failure = None
                break
            if failure is not None:
                raise failure
        if unresolved:
            source = [list(self.keys[f]) for f in unresolved]
            raise _Stop("unresolved_initiation",
                        f"{len(unresolved)} critical facet(s) cannot open in P1 and have no node-star patch: {source[:5]}")
        return updated, added

    def run(self, start_factor=0., end_factor=1., *, initiated=None, history=None):
        """Proportional loading with bisection to each strength crossing.

        ``initiated``/``history`` continue from a previously committed result
        (irreversible: unloading never resets initiation or opening history).
        """
        tolerance = self.options.cohesive.initiation_relative_tolerance
        initiated = np.zeros(self.count, dtype=bool) if initiated is None else np.array(initiated, dtype=bool)
        history = np.zeros((self.count, 3)) if history is None else np.array(history, dtype=float)
        if initiated.shape != (self.count,) or history.shape != (self.count, 3) or np.any(history < 0):
            raise ValueError("Committed network state must match the candidate inventory")
        steps, last = [], None
        low, target = float(start_factor), float(end_factor)
        rounds = 0
        patch_faces = 0

        def accept(trial, note):
            nonlocal initiated, history, last
            initiated, history, last = trial["initiated"], trial["history"], trial
            self._accepted_factor = trial["factor"]
            steps.append({"load_factor": trial["factor"], "event": note,
                          "maximum_bonded_tensile_ratio": trial["ratio"],
                          "initiated_faces": int(initiated.sum()),
                          "opened_faces": int(np.isin(trial["status"], ("softening", "separated")).sum()),
                          "separated_faces": int((trial["status"] == "separated").sum()),
                          "wet_faces": int(trial["wet"].sum()),
                          "redundant_junction_constraint_rows": trial["report"].get("redundant_junction_constraint_rows", 0),
                          "cohesive_dissipated_energy_j": trial["report"].get("cohesive_dissipated_energy_j", 0.),
                          "elastic_energy_j": trial["report"].get("elastic_energy_j")})

        try:
            while True:
                self._stage = "target load"
                trial = self.solve(target, initiated, history)
                if trial["ratio"] is None or trial["ratio"] <= 1 + tolerance:
                    accept(trial, "target_reached")
                    break
                lower, upper = low, target
                for bisection in range(self.options.maximum_bisections):
                    self._stage = f"strength-crossing bisection {bisection + 1}"
                    factor = .5 * (lower + upper)
                    trial = self.solve(factor, initiated, history)
                    if abs(trial["ratio"] - 1) <= tolerance:
                        break
                    if trial["ratio"] < 1:
                        accept(trial, "subcritical")
                        lower = low = factor
                    else:
                        upper = factor
                else:
                    raise _Stop("nonconvergence", "Initiation bisection did not resolve the strength crossing")
                while True:
                    rounds += 1
                    if rounds > self.options.maximum_initiation_rounds:
                        raise _Stop("resource_limit", "Initiation round limit reached")
                    ratios = np.where(initiated, -np.inf, trial["stress"] / self.law.tensile_strength_pa)
                    ready = np.flatnonzero(ratios >= 1 - tolerance)
                    if not len(ready):
                        raise _Stop("nonconvergence", "Overstressed state has no initiable facet")
                    updated, added = self._expand(ready, initiated)
                    patch_faces += len(added)
                    self._stage = f"initiation round {rounds}, {len(ready)} critical facets"
                    trial = self.solve(factor, updated, history)
                    initiated = updated
                    if trial["ratio"] is None or trial["ratio"] <= 1 + tolerance:
                        break
                accept(trial, "initiation")
                low = factor
            status = "no_damage_initiated" if not initiated.any() else "accepted_equilibrium"
            message = "Target load reached"
        except _Stop as stop:
            status, message = stop.status, str(stop)
        if last is None:
            return NetworkResult(status, low, self.split, initiated, np.full(self.count, "bonded"),
                                 np.zeros(self.count), np.zeros(self.count), history,
                                 None, np.zeros(self.count, dtype=bool), steps,
                                 self._report(status, message, None, patch_faces))
        return NetworkResult(status, last["factor"], self.split, last["initiated"], last["status"],
                             last["face_opening"], last["face_stress"], last["history"],
                             last["displacement"], last["wet"], steps,
                             self._report(status, message, last, patch_faces))

    def _report(self, status, message, last, patch_faces):
        areas = self.faces.areas
        report = {
            "status": status, "message": message, "solves": self.solves,
            "candidate_faces": self.count,
            "dropped_fully_tied_candidates": int(len(self.split.tied_source_faces))
            if self.split.tied_source_faces is not None else 0,
            "tensile_strength_pa": self.law.tensile_strength_pa,
            "fracture_energy_j_m2": self.law.fracture_energy_j_m2,
            "critical_opening_m": self.law.critical_opening_m,
            "patch_faces_added": int(patch_faces),
            "initiation_criterion": "bonded facet: max over the two adjacent P1 cells of n.sigma.n >= f_t",
            "tangential_response": self.options.cohesive.tangential_response,
            "limitations": LIMITATIONS,
            "engineering_validated": False,
        }
        if last is not None:
            initiated = last["initiated"]
            report.update({
                "load_factor": last["factor"],
                "initiated_faces": int(initiated.sum()), "initiated_area_m2": float(areas[initiated].sum()),
                "opened_area_m2": float(areas[np.isin(last["status"], ("softening", "separated"))].sum()),
                "separated_area_m2": float(areas[last["status"] == "separated"].sum()),
                "wet_area_m2": float(areas[last["wet"]].sum()),
                "maximum_face_opening_m": float(last["face_opening"].max(initial=0.)),
                "maximum_bonded_tensile_ratio": last["ratio"],
                "cohesive_dissipated_energy_j": last["report"].get("cohesive_dissipated_energy_j", 0.),
                "elastic_energy_j": last["report"].get("elastic_energy_j"),
                "relative_free_residual": last["report"].get("relative_free_residual"),
            })
        return report
