"""Reduced saturated surface flow on a prescribed paired crack mesh.

Solve div(w^3 grad_surface H)=0 with explicit boundary heads. A uniform
rho*g/(12*mu) cancels from head; viscosity is needed only for physical discharge.
Optional shared-node tips retain exactly zero aperture without regularisation.
No fluid sources, closed-face contact, partial saturation, trapped pockets or growth.
"""
from dataclasses import dataclass
from itertools import combinations_with_replacement
import numbers
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

if __package__:
    from .fracture_faces import PairedCrackFaces, bonded_node_representatives
    from .fracture_hydrostatic import HydrostaticBoundary
    from .fracture_water_access import water_access
else:
    from fracture_faces import PairedCrackFaces, bonded_node_representatives
    from fracture_hydrostatic import HydrostaticBoundary
    from fracture_water_access import water_access


@dataclass
class SteadyCrackLoad:
    vertex_heads_m: np.ndarray
    vertex_pressures_pa: np.ndarray
    nodal_forces_n: np.ndarray
    report: dict


@dataclass(frozen=True)
class SteadyCrackHydraulics:
    """Explicit saturated cut or face subset; omitted faces are declared dry."""

    parent_node_ids: np.ndarray
    boundaries: tuple
    water_density_kg_m3: float
    gravity_m_s2: float
    saturated_face_mask: np.ndarray
    closure_tolerance_m: float
    relative_tolerance: float
    allow_tied_fronts: bool = False
    inventory_face_ids: object = None

    def __post_init__(self):
        parents = np.asarray(self.parent_node_ids)
        saturated = np.asarray(self.saturated_face_mask)
        if parents.ndim != 1 or parents.dtype.kind not in "iu" or np.any(parents < 0):
            raise ValueError("Hydraulic parent IDs must be a nonnegative integer vector")
        if saturated.ndim != 1 or saturated.dtype != np.dtype(bool) or not saturated.all():
            raise ValueError("Steady crack hydraulics requires explicitly saturated inventory faces")
        if self.inventory_face_ids is not None:
            ids = np.asarray(self.inventory_face_ids)
            if (ids.dtype.kind not in "iu" or ids.shape != saturated.shape or not len(ids)
                    or np.any(ids < 0) or len(np.unique(ids)) != len(ids)):
                raise ValueError("Hydraulic inventory face IDs must be unique nonnegative integers matching saturation")
            snapshot = ids.copy()
            snapshot.flags.writeable = False
            object.__setattr__(self, "inventory_face_ids", snapshot)
        boundaries = tuple(self.boundaries)
        if type(self.allow_tied_fronts) is not bool:
            raise ValueError("allow_tied_fronts must be an explicit boolean")
        if (not boundaries or any(not isinstance(item, HydrostaticBoundary) for item in boundaries)
                or len({item.name for item in boundaries}) != len(boundaries)):
            raise ValueError("Explicit uniquely named hydraulic boundaries are required")
        for name in ("water_density_kg_m3", "gravity_m_s2", "relative_tolerance", "closure_tolerance_m"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value)
                    or value < 0 or (name != "closure_tolerance_m" and value == 0)):
                raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'closure_tolerance_m' else 'positive'}")
        if self.relative_tolerance >= 1:
            raise ValueError("Hydraulic relative tolerance must be less than one")
        for name, value in (("parent_node_ids", parents), ("saturated_face_mask", saturated)):
            snapshot = value.copy()
            snapshot.flags.writeable = False
            object.__setattr__(self, name, snapshot)
        object.__setattr__(self, "boundaries", boundaries)

    def face_ids(self, faces):
        """Resolve hydraulic rows to the unchanged mechanical face inventory."""
        ids = (np.arange(len(faces.minus)) if self.inventory_face_ids is None
               else self.inventory_face_ids)
        if ids.shape != self.saturated_face_mask.shape or np.any(ids >= len(faces.minus)):
            raise ValueError("Hydraulic face IDs/saturation must match the mechanical inventory")
        return ids.copy()

    @property
    def requires_dry_opening(self):
        return True

    def trial_load(self, faces, displacement_m, *, bonded_face_mask=None):
        return self.load(faces, displacement_m, bonded_face_mask=bonded_face_mask)

    def load(self, faces, displacement_m, *, bonded_face_mask=None) -> SteadyCrackLoad:
        """Use bond equivalence for flow topology, original lips for solid loads."""
        ids = self.face_ids(faces)
        physical_subset = PairedCrackFaces(
            faces.coordinates, faces.minus[ids], faces.plus[ids], faces.normals[ids])
        subset = physical_subset
        representatives = np.arange(len(faces.coordinates))
        if bonded_face_mask is not None:
            representatives = bonded_node_representatives(faces, bonded_face_mask)
            mask = np.asarray(bonded_face_mask)
            if mask[ids].any():
                raise ValueError("Hydraulic faces cannot belong to the bonded inventory")
            parents = self.parent_node_ids
            state = np.asarray(displacement_m)
            if (parents.shape != (len(representatives),) or state.shape != faces.coordinates.shape
                    or not np.isfinite(state).all()
                    or not np.array_equal(parents, parents[representatives])):
                raise ValueError("Bonded hydraulic identities must retain source parents and finite displacement")
            if np.max(np.abs(state - state[representatives])) > self.closure_tolerance_m:
                raise ValueError("Hydraulic bond equivalence requires converged all-component displacement ties")
            subset = PairedCrackFaces(
                faces.coordinates, representatives[physical_subset.minus],
                representatives[physical_subset.plus], faces.normals[ids])
        load = steady_crack_load(
            subset, displacement_m, self.parent_node_ids, self.boundaries,
            water_density_kg_m3=self.water_density_kg_m3, gravity_m_s2=self.gravity_m_s2,
            saturated_face_mask=self.saturated_face_mask,
            closure_tolerance_m=self.closure_tolerance_m, relative_tolerance=self.relative_tolerance,
            allow_tied_fronts=self.allow_tied_fronts,
        )
        # Do not concentrate mechanical tractions on hydraulic representatives.
        load.nodal_forces_n = physical_subset.pressure_forces(load.vertex_pressures_pa)
        load.report["mechanical_inventory_face_ids"] = ids.tolist()
        load.report["hydraulic_row_order"] = "head/pressure rows follow mechanical_inventory_face_ids"
        load.report["bonded_node_representatives"] = representatives.tolist()
        return load


@dataclass(frozen=True)
class HydrostaticEquilibriumHydraulics(SteadyCrackHydraulics):
    """Explicit saturated common-head equilibrium branch, not a filling model.

    Trial pressure is conservative and independent of aperture. Acceptance
    still requires the inherited opening, access and flow audit on the final
    displacement. This permits reservoir pressure to open a closed trial cut.
    """

    def __post_init__(self):
        super().__post_init__()
        if len({boundary.elevation_m for boundary in self.boundaries}) != 1:
            raise ValueError("Saturated hydrostatic equilibrium requires one common reservoir head")

    @property
    def requires_dry_opening(self):
        return False

    def trial_load(self, faces, displacement_m, *, bonded_face_mask=None):
        ids = self.face_ids(faces)
        if bonded_face_mask is not None:
            bonded_node_representatives(faces, bonded_face_mask)
            if np.asarray(bonded_face_mask)[ids].any():
                raise ValueError("Hydraulic faces cannot belong to the bonded inventory")
        if (self.parent_node_ids.shape != (len(faces.coordinates),)
                or not np.array_equal(self.parent_node_ids[faces.minus[ids]],
                                      self.parent_node_ids[faces.plus[ids]])):
            raise ValueError("Hydrostatic equilibrium requires corresponding source-parent lip identities")
        points = faces.coordinates[faces.minus[ids]]
        heads = np.full((len(ids), 3), float(self.boundaries[0].elevation_m))
        pressure = self.water_density_kg_m3 * self.gravity_m_s2 * (heads - points[:, :, 2])
        if np.any(pressure < 0):
            raise ValueError("Negative trial pressure requires a free-surface/cavitation model")
        subset = PairedCrackFaces(
            faces.coordinates, faces.minus[ids], faces.plus[ids], faces.normals[ids])
        return SteadyCrackLoad(heads, pressure, subset.pressure_forces(pressure), {
            "status": "unaccepted_common_head_equilibrium_trial_load",
            "solver_ready": False, "engineering_validated": False,
            "final_aperture_access_audit_pending": True,
            "mechanical_inventory_face_ids": ids.tolist(),
        })


@dataclass(frozen=True)
class SteadyCouplingOptions:
    relative_tolerance: float
    absolute_aperture_tolerance_m: float
    maximum_iterations: int
    relaxation: float
    flow_relative_tolerance: float

    def __post_init__(self):
        for name in ("relative_tolerance", "absolute_aperture_tolerance_m",
                     "relaxation", "flow_relative_tolerance"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.relative_tolerance >= 1 or self.flow_relative_tolerance >= 1 or self.relaxation > 1:
            raise ValueError("Relative tolerances must be below one and relaxation at most one")
        if type(self.maximum_iterations) is not int or self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be a positive integer")


@dataclass
class CompliantSteadyCrack:
    displacement_m: np.ndarray
    vertex_apertures_m: np.ndarray
    hydraulic_load: SteadyCrackLoad
    report: dict


def compliant_steady_crack(
    faces: PairedCrackFaces, parent_node_ids, boundaries, *,
    reference_aperture_m, normal_compliance_m_pa, reference_pressure_pa,
    water_density_kg_m3, gravity_m_s2, saturated_face_mask,
    closure_tolerance_m, options: SteadyCouplingOptions,
) -> CompliantSteadyCrack:
    """Reduced local compliance w=w_ref+C*(p-p_ref), coupled to steady head.

    Uniform scalar properties, planar paired surfaces, small displacement.
    Local compliance is NOT bulk dam elasticity or a friction/contact law.
    The reference aperture describes an already open verification specimen,
    not an opening imposed on the user's intact dam.
    """
    if not isinstance(options, SteadyCouplingOptions):
        raise ValueError("Explicit SteadyCouplingOptions are required")
    if not isinstance(faces, PairedCrackFaces):
        raise ValueError("Explicit PairedCrackFaces are required")
    for name, value in (("reference_aperture_m", reference_aperture_m),
                        ("normal_compliance_m_pa", normal_compliance_m_pa),
                        ("reference_pressure_pa", reference_pressure_pa)):
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not np.isfinite(value) or value < 0
                or (name == "reference_aperture_m" and value == 0)):
            raise ValueError(f"{name} must be finite and nonnegative; reference aperture must be positive")
    pairs = np.sort(np.stack((faces.minus, faces.plus), axis=-1), axis=-1)
    keys, inverse = np.unique(pairs.reshape(-1, 2), axis=0, return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    if np.any(keys[:, 0] == keys[:, 1]) or len(np.unique(keys)) != keys.size:
        raise ValueError("Local compliance requires distinct unshared opposing lip nodes; no contact or junctions")
    normal = faces.normals[0]
    signs = np.einsum("ni,i->n", faces.normals, normal)
    if (np.any(np.abs(np.abs(signs) - 1.) > 1e-12)
            or np.ptp(faces.coordinates[faces.minus].reshape(-1, 3) @ normal) > 1e-10):
        raise ValueError("Local compliance verification requires one planar surface orientation")
    directions = np.zeros((len(keys), 3))
    assigned = np.zeros(len(keys), dtype=bool)
    for minus, plus, indices, face_normal in zip(faces.minus, faces.plus, triangles, faces.normals):
        for a, b, index in zip(minus, plus, indices):
            direction = face_normal if b == keys[index, 1] else -face_normal
            if assigned[index] and not np.allclose(directions[index], direction, rtol=0., atol=1e-12):
                raise ValueError("Inconsistent paired compliance normal directions")
            directions[index] = direction
            assigned[index] = True
    boundaries = tuple(boundaries)
    aperture = np.full(len(keys), float(reference_aperture_m))
    trace = []
    for iteration in range(1, options.maximum_iterations + 1):
        displacement = np.zeros_like(faces.coordinates)
        displacement[keys[:, 0]] = -.5 * aperture[:, None] * directions
        displacement[keys[:, 1]] = .5 * aperture[:, None] * directions
        load = steady_crack_load(
            faces, displacement, parent_node_ids, boundaries,
            water_density_kg_m3=water_density_kg_m3, gravity_m_s2=gravity_m_s2,
            saturated_face_mask=saturated_face_mask, closure_tolerance_m=closure_tolerance_m,
            relative_tolerance=options.flow_relative_tolerance,
        )
        pressure = np.zeros(len(keys))
        pressure[triangles] = load.vertex_pressures_pa
        if not np.allclose(pressure[triangles], load.vertex_pressures_pa, rtol=1e-12, atol=1e-12):
            raise RuntimeError("Shared hydraulic vertices returned inconsistent pressure")
        target = reference_aperture_m + normal_compliance_m_pa * (pressure - reference_pressure_pa)
        if not np.isfinite(target).all() or np.any(target <= closure_tolerance_m):
            raise ValueError("Compliant surface would close; a coupled contact/trapped-fluid model is required")
        error = float(np.max(np.abs(target - aperture)))
        threshold = options.absolute_aperture_tolerance_m + options.relative_tolerance * float(np.max(np.abs(target)))
        trace.append({"iteration": iteration, "maximum_aperture_residual_m": error,
                      "aperture_residual_limit_m": threshold,
                      "flow_relative_residual": load.report["relative_free_residual"]})
        if error <= threshold:
            return CompliantSteadyCrack(displacement, aperture[triangles], load, {
                "status": "passed_reduced_compliant_steady_crack",
                "solver_ready": False, "engineering_validated": False,
                "coupled_to_local_normal_compliance": True,
                "coupled_to_dam_mechanics": False, "physical_time_solved": False,
                "reference_aperture_m": float(reference_aperture_m),
                "normal_compliance_m_pa": float(normal_compliance_m_pa),
                "reference_pressure_pa": float(reference_pressure_pa),
                "iterations": trace,
                "limitations": "Already-open planar local-compliance specimen only; no dam elasticity, contact/friction, trapped fluid or growth.",
            })
        aperture += options.relaxation * (target - aperture)
    raise RuntimeError(
        f"Compliant steady crack did not converge in {options.maximum_iterations} iterations; "
        f"no coupled equilibrium accepted; last diagnostics: {trace[-1]}"
    )


def steady_crack_load(faces: PairedCrackFaces, displacement_m, parent_node_ids,
                      boundaries, *, water_density_kg_m3, gravity_m_s2,
                      saturated_face_mask, closure_tolerance_m,
                      relative_tolerance, allow_tied_fronts=False) -> SteadyCrackLoad:
    """Return steady hydraulic loads, with independently audited nodal balance.

    All inventory faces must explicitly be saturated. By default every vertex
    must be open. Opt-in tied fronts permit exact zero gaps at shared lip nodes
    only, with positive integrated face conductance and no closed internal edge.
    Hydraulic identity uses BOTH crack lip IDs; coordinates never weld sheets.
    Inlet edges are verified by water_access, not inferred from painted markers.
    No physical discharge is reported without a viscosity.
    """
    if not isinstance(faces, PairedCrackFaces):
        raise ValueError("Explicit PairedCrackFaces are required")
    if type(allow_tied_fronts) is not bool:
        raise ValueError("allow_tied_fronts must be an explicit boolean")
    for name, value in (("water_density_kg_m3", water_density_kg_m3),
                        ("gravity_m_s2", gravity_m_s2), ("relative_tolerance", relative_tolerance)):
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if relative_tolerance >= 1:
        raise ValueError("relative_tolerance must be less than one")
    saturated = np.asarray(saturated_face_mask)
    if (saturated.dtype != np.dtype(bool) or saturated.shape != (len(faces.minus),)
            or not saturated.all()):
        raise ValueError("Steady surface flow requires explicit saturation of every inventory face")
    boundaries = tuple(boundaries)
    if (not boundaries or any(not isinstance(item, HydrostaticBoundary) for item in boundaries)
            or len({item.name for item in boundaries}) != len(boundaries)):
        raise ValueError("Explicit uniquely named HydrostaticBoundary entries are required")
    access_reports = {}
    reached = set()
    for boundary in boundaries:
        access = water_access(faces, displacement_m, boundary.inlet_source_edges, parent_node_ids,
                              closure_tolerance_m=closure_tolerance_m)
        reached.update(access["reachable_open_face_ids"])
        access_reports[boundary.name] = access
    gaps = np.asarray(faces.measure(displacement_m)["signed_vertex_gaps_m"])
    tied = faces.minus == faces.plus
    if (np.any(gaps <= closure_tolerance_m) and (
            not allow_tied_fronts or np.any((gaps <= closure_tolerance_m) & ~tied))):
        raise ValueError("Steady surface flow requires positive aperture at every vertex; no contact or tied-front regularisation")
    if allow_tied_fronts:
        if np.any(np.max(gaps, axis=1) <= closure_tolerance_m):
            raise ValueError("Fully closed faces require a conserved fluid/contact state")
        if any(access["blocked_inlet_source_edges"] for access in access_reports.values()):
            raise ValueError("A closed hydraulic inlet cannot prescribe head on a tied front")
    if reached != set(range(len(faces.minus))):
        raise ValueError("Isolated wet faces require a trapped-fluid state, not a steady drained solve")

    pairs = np.sort(np.stack((faces.minus, faces.plus), axis=-1), axis=-1)
    keys, inverse = np.unique(pairs.reshape(-1, 2), axis=0, return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    edges = {}
    for face, triangle in enumerate(triangles):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edges.setdefault(tuple(sorted((int(triangle[i]), int(triangle[j])))), []).append(face)
    if any(len(owners) > 2 for owners in edges.values()):
        raise ValueError("Nonmanifold hydraulic edges require a branch-junction formulation")
    if allow_tied_fronts and any(
        len(owners) == 2 and all(keys[vertex, 0] == keys[vertex, 1] for vertex in edge)
        for edge, owners in edges.items()
    ):
        raise ValueError("Closed internal hydraulic edges require separate fluid-sheet identities")
    # A P1 vertex shared only at a point must not connect distinct fluid sheets.
    incident_faces = [set() for _ in keys]
    vertex_neighbors = [{} for _ in keys]
    for face, triangle in enumerate(triangles):
        for vertex in triangle:
            incident_faces[vertex].add(face)
    for edge, owners in edges.items():
        if len(owners) == 2:
            a, b = owners
            for vertex in edge:
                vertex_neighbors[vertex].setdefault(a, set()).add(b)
                vertex_neighbors[vertex].setdefault(b, set()).add(a)
    for vertex, incident in enumerate(incident_faces):
        connected = {next(iter(incident))}
        pending = list(connected)
        neighbors = vertex_neighbors[vertex]
        while pending:
            face = pending.pop()
            for other in neighbors.get(face, set()) - connected:
                connected.add(other)
                pending.append(other)
        if connected != incident:
            raise ValueError("Point-touching hydraulic sheets require distinct vertex identities")

    parents = np.asarray(parent_node_ids)
    edge_by_source = {}
    for edge, owners in edges.items():
        if len(owners) == 1:
            source = tuple(sorted(int(parents[keys[vertex, 0]]) for vertex in edge))
            edge_by_source.setdefault(source, []).append(edge)
    prescribed = {}
    for boundary in boundaries:
        for source in boundary.inlet_source_edges:
            matches = edge_by_source.get(tuple(sorted(map(int, source))), [])
            if len(matches) != 1:
                raise ValueError("Hydraulic boundary edge must identify one exterior split edge")
            for vertex in matches[0]:
                if vertex in prescribed and prescribed[vertex] != boundary.elevation_m:
                    raise ValueError("Conflicting hydraulic heads at a shared boundary vertex")
                prescribed[vertex] = float(boundary.elevation_m)

    points = faces.coordinates[faces.minus]
    rows, columns, entries = [], [], []
    for triangle, xyz, aperture, area in zip(triangles, points, gaps, faces.areas):
        basis = np.column_stack((xyz[1] - xyz[0], xyz[2] - xyz[0]))
        gradients = np.array([[-1., -1.], [1., 0.], [0., 1.]]) @ np.linalg.solve(basis.T @ basis, basis.T)
        # Exact integral of an affine aperture cubed: A/10 times the complete
        # homogeneous cubic polynomial in the three vertex aperture values.
        integral = area / 10 * sum(np.prod(aperture[list(ids)])
                                  for ids in combinations_with_replacement(range(3), 3))
        local = integral * (gradients @ gradients.T)
        rows.extend(np.repeat(triangle, 3))
        columns.extend(np.tile(triangle, 3))
        entries.extend(local.ravel())
    matrix = sparse.coo_matrix((entries, (rows, columns)), shape=(len(keys), len(keys))).tocsr()
    if not np.isfinite(matrix.data).all() or np.max(np.abs(matrix.data)) <= 0:
        raise ValueError("Steady hydraulic conductance must be finite and nonzero")
    fixed = np.array(sorted(prescribed), dtype=int)
    free = np.setdiff1d(np.arange(len(keys)), fixed)
    reference_head = min(prescribed.values())
    offset = np.zeros(len(keys))
    offset[fixed] = [prescribed[v] - reference_head for v in fixed]
    if len(free):
        scale = np.max(np.abs(matrix.data))
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                offset[free] = spsolve(matrix[free][:, free] / scale,
                                      -(matrix[free][:, fixed] @ offset[fixed]) / scale)
            except MatrixRankWarning as error:
                raise RuntimeError("Steady crack head solve is singular; no pressure accepted") from error
    if not np.isfinite(offset).all():
        raise RuntimeError("Steady crack head solve returned non-finite heads")
    residual = matrix @ offset
    flux_scale = max(float(np.linalg.norm(matrix.data) * np.ptp(offset)), np.finfo(float).tiny)
    relative_residual = float(np.linalg.norm(residual[free]) / flux_scale)
    relative_balance = float(abs(residual[fixed].sum()) / flux_scale)
    if (not np.isfinite(residual).all() or not np.isfinite(relative_residual)
            or not np.isfinite(relative_balance) or relative_residual > relative_tolerance
            or relative_balance > relative_tolerance):
        raise RuntimeError("Steady crack flow failed independent residual/boundary balance checks")
    face_heads = (offset + reference_head)[triangles]
    pressure_head = face_heads - points[:, :, 2]
    if np.any(pressure_head < 0):
        raise ValueError("Negative pressure requires a free-surface/cavitation model; no saturated load accepted")
    pressure = water_density_kg_m3 * gravity_m_s2 * pressure_head
    forces = faces.pressure_forces(pressure)
    return SteadyCrackLoad(face_heads, pressure, forces, {
        "status": "passed_reduced_steady_crack_flow",
        "solver_ready": False, "engineering_validated": False,
        "mechanical_equilibrium_solved": False, "wetting_solved": False,
        "relative_free_residual": relative_residual,
        "relative_boundary_balance": relative_balance,
        "dirichlet_vertex_ids": fixed.tolist(),
        "dirichlet_reactions_scaled_m4": residual[fixed].tolist(),
        "reaction_convention": "K H at prescribed vertices; multiply by rho*g/(12*mu) for boundary injection m3/s",
        "boundary_elevations_m": {item.name: float(item.elevation_m) for item in boundaries},
        "access": access_reports,
        "allow_tied_fronts": allow_tied_fronts,
        "tied_front_vertex_ids": np.flatnonzero(keys[:, 0] == keys[:, 1]).tolist(),
        "minimum_vertex_aperture_m": float(gaps.min()),
        "aperture_regularisation_m": 0.,
        "limitations": "Prescribed saturated open manifold surface with optional exact tied fronts only; no closed-face contact, trapped fluid, growth, physical discharge or coupled dam solve.",
    })
