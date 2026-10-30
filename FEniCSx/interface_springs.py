"""Elmer-compatible mortar springs, including nonmatching refined interfaces.

Common intersections integrate the slave-weighted displacement jump B u.
Eliminating the lumped mortar multiplier gives B.T diag(k / weights) B,
matching the canonical Elmer Galerkin-projector spring discretisation.
"""
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from scipy.spatial import cKDTree
from scipy import sparse


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def intersection_mass(first, second):
    """Return the 6x6 scalar jump mass on the coplanar triangle intersection."""
    edge = first[1] - first[0]
    length = np.linalg.norm(edge)
    normal = np.cross(edge, first[2] - first[0])
    normal_length = np.linalg.norm(normal)
    if length <= 0 or normal_length <= 1e-14:
        raise ValueError("Degenerate spring surface triangle")
    normal /= normal_length
    if np.max(np.abs((second - first[0]) @ normal)) > 1e-7:
        return None, 0.
    basis = np.column_stack((edge / length, np.cross(normal, edge / length)))
    a, b = (first - first[0]) @ basis, (second - first[0]) @ basis
    polygon = list(b)
    for start, end in zip(a, np.roll(a, -1, axis=0)):
        output = []
        if not polygon:
            return None, 0.
        previous = polygon[-1]
        before = _cross(end - start, previous - start)
        for current in polygon:
            after = _cross(end - start, current - start)
            if (after >= 0) != (before >= 0):
                output.append(previous + before / (before - after) * (current - previous))
            if after >= 0:
                output.append(current)
            previous, before = current, after
        polygon = output
    if len(polygon) < 3:
        return None, 0.
    inverse_a = np.linalg.inv(np.column_stack((a, np.ones(3))).T)
    inverse_b = np.linalg.inv(np.column_stack((b, np.ones(3))).T)
    quadrature = np.asarray(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))
    matrix, area = np.zeros((6, 6)), 0.
    for index in range(1, len(polygon) - 1):
        triangle = np.asarray((polygon[0], polygon[index], polygon[index + 1]))
        weight = abs(_cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) / 2
        if weight <= 1e-15:
            continue
        points = np.column_stack((quadrature @ triangle, np.ones(3)))
        jump = np.column_stack((points @ inverse_a.T, -points @ inverse_b.T))
        matrix += weight / 3 * jump.T @ jump
        area += weight
    return matrix, area


def mortar_projector(blocks, node_count, *, multiplier_map=None):
    """Slave-weighted jump and positive slave mass row sums."""
    mapping = np.arange(node_count) if multiplier_map is None else np.asarray(multiplier_map)
    if (mapping.dtype.kind not in "iu" or mapping.shape != (node_count,)
            or np.any(mapping < 0) or np.any(mapping >= node_count)):
        raise ValueError("Mortar multiplier map must contain valid row identities for every displacement node")
    rows = np.concatenate([np.repeat(mapping[nodes[:3]], 6) for nodes, mass in blocks])
    columns = np.concatenate([np.tile(nodes, 3) for nodes, mass in blocks])
    values = np.concatenate([mass[:3, :].ravel() for nodes, mass in blocks])
    jump = sparse.coo_matrix((values, (rows, columns)), shape=(node_count, node_count)).tocsr()
    weights = np.zeros(node_count)
    for nodes, mass in blocks:
        np.add.at(weights, mapping[nodes[:3]], mass[:3, :3].sum(axis=1))
    return jump, weights


def projected_spring_matrix(blocks, node_count, *, multiplier_map=None):
    """Symmetric conservative scalar mortar operator; stiffness is applied later."""
    jump, weights = mortar_projector(blocks, node_count, multiplier_map=multiplier_map)
    inverse = np.zeros(node_count)
    active = weights > 0
    inverse[active] = 1 / weights[active]
    result = (jump.T @ sparse.diags(inverse) @ jump).tocsr()
    result.eliminate_zeros()
    return result


def interface_blocks(first, second, name):
    """Integrate on actual slave triangles, not the group's bookkeeping fit plane.

    Coincident curved sides have exact coplanar partners for each triangle;
    retaining their 3-D areas reproduces Elmer's exported [M,-M] projectors.
    """
    centres = np.asarray([p.mean(axis=0) for _, p in second])
    radii = np.asarray([np.linalg.norm(p - p.mean(axis=0), axis=1).max() for _, p in second])
    tree = cKDTree(centres)
    maximum_radius = radii.max()
    def key(points):
        return tuple(sorted(tuple(p) for p in np.round(points, 8)))
    matching = {key(points): index for index, (_, points) in enumerate(second)}
    if len(matching) != len(second):
        raise ValueError(f"{name} has duplicate spring surface faces")
    covered_second = np.zeros(len(second))
    total_area = 0.
    blocks = []
    for ids, points in first:
        centre = points.mean(axis=0)
        radius = np.linalg.norm(points - centre, axis=1).max()
        covered = 0.
        exact = matching.get(key(points))
        candidates = [exact] if exact is not None else tree.query_ball_point(
            centre, radius + maximum_radius + 1e-7)
        for candidate in candidates:
            other_ids, other_points = second[candidate]
            if np.linalg.norm(centre - centres[candidate]) > radius + radii[candidate] + 1e-7:
                continue
            mass, area = intersection_mass(points, other_points)
            if area == 0:
                continue
            covered += area
            covered_second[candidate] += area
            blocks.append((np.concatenate((ids, other_ids)), mass))
        expected = np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) / 2
        if not np.isclose(covered, expected, rtol=1e-6, atol=1e-10):
            raise ValueError(f"{name} interface coverage mismatch: {covered} versus {expected}")
        total_area += covered
    for covered, (_, points) in zip(covered_second, second):
        expected = np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) / 2
        if not np.isclose(covered, expected, rtol=1e-6, atol=1e-10):
            raise ValueError(f"{name} opposite interface coverage mismatch")
    return blocks, total_area


def spring_pairs(structure):
    """(name, (slave tag, master tag), coefficients or None when bonded) for each interface.

    Declared foundation interfaces use their validated source tags/stiffnesses.
    Without a foundation the wall/bedrock seat uses geometry.bedrock.support;
    spring-jointed wall thickness blocks use geometry.wall.joint_support.
    """
    declared = getattr(structure, "spring_pairs", None)
    if declared is not None:
        return [(pair["name"], tuple(pair["tags"]), pair["spring_n_per_m3"]) for pair in declared]
    support = structure.three_body_config["geometry"]["bedrock"]["support"]
    pairs = [("wall_bedrock", (11, 12),
              None if support.get("model") == "bonded" else support["spring_n_per_m3"])]
    pairs.extend((joint["name"], tuple(joint["tags"]), joint["spring_n_per_m3"])
                 for joint in getattr(structure, "wall_joints", []))
    return pairs


class InterfaceSprings:
    def __init__(self, imported, structure, space, *, multiplier_parent_node_ids=None):
        self.space = space
        domain = imported.domain
        multiplier_map = None
        if multiplier_parent_node_ids is not None:
            if domain.comm.size != 1:
                raise ValueError("Source-preserved mortar multipliers currently require a serial mesh")
            source = domain.geometry.input_global_indices[domain.geometry.dofmaps[0]]
            count = space.dofmap.index_map.size_local
            parents = np.asarray(multiplier_parent_node_ids)
            if (space.mesh is not domain or parents.dtype.kind not in "iu"
                    or parents.shape != (count,) or np.any(parents < 0)
                    or source.shape != space.dofmap.list.shape
                    or not np.array_equal(np.unique(source), np.arange(count))):
                raise ValueError("Mortar parent IDs must match complete serial source-node identities")
            _, first, groups = np.unique(parents, return_index=True, return_inverse=True)
            source_points = np.empty((count, 3))
            source_points[source] = space.tabulate_dof_coordinates()[space.dofmap.list]
            if not np.allclose(source_points, source_points[first][groups], rtol=0., atol=1e-12):
                raise ValueError("Mortar multiplier parents cannot merge different source coordinates")
            multiplier_map = np.empty(count, dtype=np.int64)
            multiplier_map[space.dofmap.list] = groups[source]
        self.multiplier_map = multiplier_map
        xyz = space.tabulate_dof_coordinates()
        local = {}
        facet_count = domain.topology.index_map(2).size_local
        pairs = spring_pairs(structure)
        for tag in sorted({tag for _, tags, _ in pairs for tag in tags}):
            records = []
            for facet in imported.facet_tags.find(tag):
                if facet >= facet_count:
                    continue
                dofs = fem.locate_dofs_topological(
                    space, 2, np.asarray([facet], dtype=np.int32), remote=False,
                )
                if len(dofs) != 3:
                    raise ValueError("Surface spring assembly requires vector P1 tetrahedra")
                global_dofs = space.dofmap.index_map.local_to_global(dofs)
                records.append((global_dofs, xyz[dofs]))
            local[tag] = records
        gathered = domain.comm.allgather(local)
        faces = {tag: [record for rank in gathered for record in rank[tag]] for tag in local}
        node_count = space.dofmap.index_map.size_global
        self._operator = sparse.csr_matrix((3 * node_count, 3 * node_count))
        self.report = {}
        declarations = {pair["name"]: pair for pair in getattr(structure, "spring_pairs", None) or []}
        for name, tags, coefficients in pairs:
            if coefficients is None:
                if any(faces[tag] for tag in tags):
                    raise ValueError(f"Bonded {name} interface must not be tagged as spring faces")
                cells = domain.geometry.dofmaps[0]
                regions = np.zeros(len(cells), dtype=np.int32)
                regions[imported.cell_tags.indices] = imported.cell_tags.values
                regions_by_interface = {
                    "wall_bedrock": (1, 2),
                    "wall_foundation": (1, 3),
                    "foundation_bedrock": (3, 2),
                }
                try:
                    first_region, second_region = regions_by_interface[name]
                except KeyError as exc:
                    raise ValueError(f"Unsupported bonded interface {name!r}") from exc
                geometry_map = domain.geometry.index_map()
                geometry_indices = np.arange(
                    geometry_map.size_local + geometry_map.num_ghosts, dtype=np.int32,
                )
                global_geometry_nodes = geometry_map.local_to_global(geometry_indices)
                global_cells = global_geometry_nodes[cells]
                faces_by_region = []
                for region in (first_region, second_region):
                    faces_by_region.append({
                        tuple(sorted(int(node) for node in face))
                        for cell in np.flatnonzero(regions == region)
                        for face in (global_cells[cell][[0, 1, 2]], global_cells[cell][[0, 1, 3]],
                                     global_cells[cell][[0, 2, 3]], global_cells[cell][[1, 2, 3]])
                    })
                gathered_bonded_faces = domain.comm.allgather(faces_by_region)
                global_faces = [
                    set().union(*(rank_faces[index] for rank_faces in gathered_bonded_faces))
                    for index in (0, 1)
                ]
                if not global_faces[0] & global_faces[1]:
                    raise ValueError(f"Bonded {name} interface must use shared-node faces")
                self.report[name] = {"connection": "bonded", "discretisation": "shared-node faces"}
                continue
            first, second = faces[tags[0]], faces[tags[1]]
            if not first or not second:
                raise ValueError(f"Missing {name} spring faces")
            blocks, total_area = interface_blocks(first, second, name)
            self.report[name] = {"area_m2": total_area, "spring_n_per_m3": coefficients,
                                 "faces": [len(first), len(second)],
                                 "discretisation": "Galerkin mortar; lumped multiplier compliance"}
            if multiplier_map is not None:
                self.report[name]["multiplier_identity"] = "original source parents retained across candidate node splits"
            scalar = projected_spring_matrix(blocks, node_count, multiplier_map=multiplier_map)
            self._operator += sparse.kron(
                scalar, sparse.diags([coefficients[axis] for axis in ("x", "y", "z")]),
                format="csr",
            )
        # Each plane's multiplier is eliminated above, before summing at shared corner nodes.
        for name in list(self.report):
            if "/" not in name:
                continue
            logical, part = name.split("/", 1)
            detail = self.report.pop(name)
            declaration = declarations[name]
            grouping = declaration.get("grouping", "exact_plane")
            group = self.report.setdefault(logical, {
                "area_m2": 0., "spring_n_per_m3": detail["spring_n_per_m3"],
                "grouping": grouping, "faces": {},
                "discretisation": "independent grouped Galerkin mortar; lumped multiplier compliance",
            })
            if grouping == "near_planar":
                group.update({key: declaration[key] for key in ("tolerance_m", "max_angle_deg", "continuity")})
                detail["source_plane"] = declaration["plane"]
                detail["tags"] = list(declaration["tags"])
            group["area_m2"] += detail["area_m2"]
            group["faces"][part] = detail
        self.matrix = None

    def add_to(self, matrix, *, operator=None):
        if self.matrix is None:
            raise RuntimeError("Spring matrix has not been assembled")
        matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        matrix.axpy(1., self.matrix if operator is None else operator,
                    structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        matrix.assemble()

    def build_matrix(self, template):
        start, end = template.getOwnershipRange()
        local = self._operator[start:end].tocsr()
        self.matrix = PETSc.Mat().createAIJ(
            size=template.getSizes(),
            csr=(local.indptr.astype(PETSc.IntType), local.indices.astype(PETSc.IntType),
                 local.data.astype(PETSc.ScalarType)),
            comm=self.space.mesh.comm,
        )
        self.matrix.assemble()
        return self.matrix

    def energy(self, displacement):
        if self.matrix is None:
            raise RuntimeError("Spring matrix has not been assembled")
        work = self.matrix.createVecLeft()
        try:
            self.matrix.mult(displacement.x.petsc_vec, work)
            return float(.5 * displacement.x.petsc_vec.dot(work))
        finally:
            work.destroy()


class SpringLinearProblem(LinearProblem):
    """DOLFINx 0.11 linear assembly with an additional nonlocal spring matrix."""
    def __init__(self, *args, interface_springs=None, additional_nodal_forces=None,
                 eliminate_interface_bcs=False, **kwargs):
        self._constrained_springs = None
        super().__init__(*args, **kwargs)
        self.interface_springs = interface_springs
        self.additional_nodal_forces = None
        if additional_nodal_forces is not None:
            forces = np.asarray(additional_nodal_forces)
            if (self.u.function_space.mesh.comm.size != 1 or forces.shape != self.b.array.shape
                    or forces.dtype.kind not in "fiu" or not np.isfinite(forces).all()):
                raise ValueError("Additional nodal forces require a finite serial vector matching the load")
            self.additional_nodal_forces = forces.astype(float).copy()
        if interface_springs is not None:
            self.A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            if interface_springs.matrix is None:
                interface_springs.build_matrix(self.A)
            if eliminate_interface_bcs and self.bcs:
                if self.u.function_space.mesh.comm.size != 1:
                    raise ValueError("Mortar/Dirichlet elimination currently requires a serial mesh")
                fixed = np.unique(np.concatenate([bc.dof_indices()[0] for bc in self.bcs])).astype(PETSc.IntType)
                self._constrained_springs = interface_springs.matrix.copy()
                self._constrained_springs.zeroRowsColumns(fixed, diag=0.)
                self._constrained_springs.assemble()

    @property
    def spring_jacobian(self):
        """Borrowed operator; original springs are retained for forces/reactions."""
        return (self._constrained_springs if self._constrained_springs is not None
                else self.interface_springs.matrix if self.interface_springs is not None else None)

    def __del__(self):
        if self._constrained_springs is not None:
            self._constrained_springs.destroy()
        super().__del__()

    def solve(self):
        if self.interface_springs is None and self.additional_nodal_forces is None:
            return super().solve()
        self.A.zeroEntries()
        fem.petsc.assemble_matrix(self.A, self.a, bcs=self.bcs)
        self.A.assemble()
        if self.interface_springs is not None:
            self.interface_springs.add_to(self.A, operator=self.spring_jacobian)
        with self.b.localForm() as local:
            local.set(0)
        fem.petsc.assemble_vector(self.b, self.L)
        fem.petsc.apply_lifting(self.b, [self.a], bcs=[self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        if self.additional_nodal_forces is not None:
            self.b.array[:] += self.additional_nodal_forces
        if self._constrained_springs is not None:
            imposed = self.b.duplicate()
            work = self.b.duplicate()
            try:
                imposed.set(0.)
                fem.petsc.set_bc(imposed, self.bcs)
                self.interface_springs.matrix.mult(imposed, work)
                self.b.axpy(-1., work)
            finally:
                imposed.destroy()
                work.destroy()
        fem.petsc.set_bc(self.b, self.bcs)
        self.solver.solve(self.b, self.x)
        self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        fem.petsc.assign(self.x, self.u)
        return self.u
