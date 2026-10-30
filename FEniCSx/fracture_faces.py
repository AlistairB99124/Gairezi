"""Small-displacement paired P1 crack faces; no surface construction or growth.

Pairs are explicit corresponding source-node triples on initially coincident
faces, with distinct identities away from tied front vertices. Normals point
from minus to plus solid. Coordinates never establish identity. The caller
must supply validated surface provenance.
"""
import numbers
from itertools import combinations

import numpy as np


def _vectors(values, name):
    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a finite real (n, 3) array")
    return raw.astype(float)


class PairedCrackFaces:
    def __init__(self, coordinates_m, minus_triangles, plus_triangles, normals,
                 *, geometry_tolerance_m=1e-10):
        self.coordinates = _vectors(coordinates_m, "coordinates_m").copy()
        if (isinstance(geometry_tolerance_m, bool) or not isinstance(geometry_tolerance_m, numbers.Real)
                or not np.isfinite(geometry_tolerance_m) or geometry_tolerance_m <= 0):
            raise ValueError("geometry_tolerance_m must be finite and positive")
        pairs = []
        for name, values in (("minus_triangles", minus_triangles), ("plus_triangles", plus_triangles)):
            ids = np.asarray(values)
            if (ids.dtype.kind not in "iu" or ids.ndim != 2 or ids.shape[1] != 3 or not len(ids)
                    or np.any(ids < 0) or np.any(ids >= len(self.coordinates))
                    or any(len(set(row)) != 3 for row in ids)
                    or len(np.unique(np.sort(ids, axis=1), axis=0)) != len(ids)):
                raise ValueError(f"{name} must contain distinct known source-node triples without duplicate faces")
            pairs.append(ids.copy())
        self.minus, self.plus = pairs
        self.normals = _vectors(normals, "normals").copy()
        if self.minus.shape != self.plus.shape or self.normals.shape != (len(self.minus), 3):
            raise ValueError("Paired triangle and normal counts must match")
        if np.any(np.all(self.minus == self.plus, axis=1)):
            raise ValueError("Opposing crack lips cannot be fully tied; distinct identities are required away from the front")
        for minus_row, plus_row in zip(self.minus, self.plus):
            for node in np.intersect1d(minus_row, plus_row):
                if not np.array_equal(minus_row == node, plus_row == node):
                    raise ValueError("Shared crack-front IDs must occur only at matching paired vertices")
        if not np.allclose(np.linalg.norm(self.normals, axis=1), 1., rtol=0., atol=1e-12):
            raise ValueError("Crack normals must be explicit unit vectors from minus to plus")
        minus = self.coordinates[self.minus]
        plus = self.coordinates[self.plus]
        if np.any(np.linalg.norm(minus - plus, axis=2) > geometry_tolerance_m):
            raise ValueError("Corresponding crack nodes must be initially coincident; no initial gap is imposed")
        area_vectors = np.cross(minus[:, 1] - minus[:, 0], minus[:, 2] - minus[:, 0])
        double_area = np.linalg.norm(area_vectors, axis=1)
        if not np.isfinite(double_area).all() or np.any(double_area <= geometry_tolerance_m**2):
            raise ValueError("Crack triangles must have positive nondegenerate area")
        alignment = np.abs(np.einsum("ni,ni->n", area_vectors, self.normals)) / double_area
        if np.any(np.abs(alignment - 1.) > 1e-10):
            raise ValueError("Crack normals must be perpendicular to their reference triangles")
        self.areas = double_area / 2

    def measure(self, displacement_m) -> dict:
        """P1 signed opening extrema are attained at triangle vertices."""
        displacement = _vectors(displacement_m, "displacement_m")
        if displacement.shape != self.coordinates.shape:
            raise ValueError("Displacement must match source-node coordinate shape")
        jump = displacement[self.plus] - displacement[self.minus]
        if not np.isfinite(jump).all():
            raise RuntimeError("Crack displacement jump overflowed")
        signed = np.einsum("nvi,ni->nv", jump, self.normals)
        tangential = jump - signed[:, :, None] * self.normals[:, None, :]
        face, vertex = np.unravel_index(np.argmax(signed), signed.shape)
        return {
            "maximum_opening_m": max(0., float(signed[face, vertex])),
            "minimum_signed_gap_m": float(signed.min()),
            "peak_location_m": self.coordinates[self.minus[face, vertex]].tolist(),
            "peak_face_index": int(face), "peak_source_node_id": int(self.minus[face, vertex]),
            "signed_vertex_gaps_m": signed.tolist(),
            "tangential_vertex_jumps_m": tangential.tolist(),
            "reference_area_m2": float(self.areas.sum()),
            "method": "paired source-node displacement jump projected onto reference normal; affine P1 faces",
            "engineering_validated": False,
        }

    def pressure_forces(self, vertex_pressures_pa) -> np.ndarray:
        """Consistent nodal water tractions, integrated exactly for P1 pressure.

        Accept only explicitly supplied face pressure. This does not decide
        which faces are wet, nor allow flow through closed contact.
        """
        pressure = np.asarray(vertex_pressures_pa)
        if (pressure.dtype.kind not in "fiu" or pressure.shape != self.minus.shape
                or not np.isfinite(pressure).all() or np.any(pressure < 0)):
            raise ValueError("Face pressure must be a finite nonnegative (faces, 3) nodal array")
        integrated = self.areas[:, None] / 12 * (pressure + pressure.sum(axis=1)[:, None])
        vectors = integrated[:, :, None] * self.normals[:, None, :]
        force = np.zeros_like(self.coordinates)
        np.add.at(force, self.plus, vectors)
        np.add.at(force, self.minus, -vectors)
        if not np.isfinite(force).all():
            raise RuntimeError("Crack-face pressure load overflowed")
        return force


def write_surface_state(path, faces, displacement_m, *, released_face_mask,
                        wet_face_mask, vertex_pressures_pa, parent_node_ids=None):
    """Export both physical lips and measured jumps, retaining face/node IDs."""
    import meshio

    if not isinstance(faces, PairedCrackFaces):
        raise ValueError("Surface export requires explicit paired faces")
    released, wet = np.asarray(released_face_mask), np.asarray(wet_face_mask)
    if (released.dtype != np.dtype(bool) or wet.dtype != np.dtype(bool)
            or released.shape != (len(faces.minus),) or wet.shape != released.shape
            or np.any(wet & ~released)):
        raise ValueError("Released/wet masks must match faces; wet surfaces must be released")
    pressure = np.asarray(vertex_pressures_pa)
    if (pressure.dtype.kind not in "fiu" or pressure.shape != faces.minus.shape
            or not np.isfinite(pressure).all() or np.any(pressure < 0)
            or np.any(pressure[~wet] != 0)):
        raise ValueError("Surface pressures must be finite nonnegative vertex values, zero on dry faces")
    displacement = _vectors(displacement_m, "displacement_m")
    measured = faces.measure(displacement)
    nodes = np.stack((faces.minus, faces.plus), axis=1).ravel()
    parents = np.arange(len(faces.coordinates)) if parent_node_ids is None else np.asarray(parent_node_ids)
    if (parents.dtype.kind not in "iu" or parents.shape != (len(faces.coordinates),)
            or np.any(parents < 0)
            or parent_node_ids is not None and not np.array_equal(parents[faces.minus], parents[faces.plus])):
        raise ValueError("Surface parent IDs must match coordinates and corresponding lip identities")

    def paired_vertices(values):
        return np.repeat(np.asarray(values)[:, None], 2, axis=1).reshape(
            (len(nodes),) + np.asarray(values).shape[2:])

    signed = np.asarray(measured["signed_vertex_gaps_m"])
    meshio.write_points_cells(
        path, faces.coordinates[nodes],
        [("triangle", np.arange(len(nodes)).reshape(-1, 3))],
        point_data={
            "source_node_id": nodes, "original_parent_node_id": parents[nodes],
            "displacement_m": displacement[nodes],
            "signed_gap_m": paired_vertices(signed),
            "opening_m": paired_vertices(np.maximum(signed, 0.)),
            "tangential_jump_m": paired_vertices(measured["tangential_vertex_jumps_m"]),
            "water_pressure_pa": paired_vertices(pressure),
        },
        cell_data={
            "mechanical_inventory_face_id": [np.repeat(np.arange(len(faces.minus)), 2)],
            "lip_side": [np.tile([-1, 1], len(faces.minus))],
            "released": [np.repeat(released.astype(np.int32), 2)],
            "wet": [np.repeat(wet.astype(np.int32), 2)],
        },
        file_format="vtu",
    )


def bonded_node_representatives(faces, bonded_face_mask):
    """Canonical source identities implied by explicit all-component face ties."""
    if not isinstance(faces, PairedCrackFaces):
        raise ValueError("Bond identities require explicit paired faces")
    mask = np.asarray(bonded_face_mask)
    if mask.dtype != np.dtype(bool) or mask.shape != (len(faces.minus),):
        raise ValueError("Bonded face mask must be boolean and match the candidate inventory")
    representatives = np.arange(len(faces.coordinates))

    def root(node):
        while representatives[node] != node:
            representatives[node] = representatives[representatives[node]]
            node = representatives[node]
        return int(node)

    for minus, plus in zip(faces.minus[mask], faces.plus[mask]):
        for a, b in zip(minus, plus):
            first, second = root(int(a)), root(int(b))
            representatives[max(first, second)] = min(first, second)
    return np.array([root(node) for node in range(len(representatives))])


def source_vector_values(field) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates and vector values in input-node order on serial P1 tets."""
    space = field.function_space
    domain = space.mesh
    if (domain.comm.size != 1 or domain.topology.cell_name() != "tetrahedron"
            or domain.geometry.cmaps[0].degree != 1 or space.ufl_element().degree != 1
            or space.ufl_element().family_name != "P" or space.ufl_element().discontinuous
            or space.dofmap.index_map_bs != 3 or tuple(field.ufl_shape) != (3,)):
        raise ValueError("Source vector extraction requires serial blocked vector P1 on affine tetrahedra")
    source = domain.geometry.input_global_indices[domain.geometry.dofmaps[0]]
    dofs = space.dofmap.list
    count = len(domain.geometry.input_global_indices)
    if source.shape != dofs.shape or not np.array_equal(np.unique(source), np.arange(count)):
        raise ValueError("Input source-node identities are incomplete or not contiguous")
    coordinates = np.empty((count, 3))
    values = np.empty_like(coordinates)
    coordinates[source] = space.tabulate_dof_coordinates()[dofs]
    values[source] = field.x.array.reshape(-1, 3)[dofs]
    if not np.isfinite(values).all() or not np.isfinite(coordinates).all():
        raise ValueError("Source vector contains non-finite coordinates or displacement")
    return coordinates, values


def bulk_face_mapping(space, faces: PairedCrackFaces):
    """Validate exterior lips/orientation and return the input-node/DOF map."""
    from dolfinx import fem

    coordinates, _ = source_vector_values(fem.Function(space))
    if coordinates.shape != faces.coordinates.shape or not np.allclose(
        coordinates, faces.coordinates, rtol=0., atol=1e-12,
    ):
        raise ValueError("Crack inventory coordinates must match exact input-node identities of this mesh")
    source = space.mesh.geometry.input_global_indices[space.mesh.geometry.dofmaps[0]]
    targets = {tuple(sorted(map(int, row))): [] for row in np.vstack((faces.minus, faces.plus))}
    for cell_id, cell in enumerate(source):
        for triangle in combinations(cell, 3):
            key = tuple(sorted(map(int, triangle)))
            if key in targets:
                targets[key].append(cell_id)
    if any(len(owners) != 1 for owners in targets.values()):
        raise ValueError("Every crack lip must be an actual exterior triangle of the supplied bulk mesh")
    for minus, plus, normal in zip(faces.minus, faces.plus, faces.normals):
        for lip, outward in ((minus, normal), (plus, -normal)):
            owner = targets[tuple(sorted(map(int, lip)))][0]
            to_face = coordinates[lip].mean(axis=0) - coordinates[source[owner]].mean(axis=0)
            if np.dot(to_face, outward) <= 0:
                raise ValueError("Crack normal must point out of minus solid and into plus solid")
    mapping = np.empty(len(coordinates), dtype=np.int32)
    mapping[source] = space.dofmap.list
    return coordinates, source, mapping
