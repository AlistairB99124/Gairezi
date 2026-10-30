"""Split explicitly supplied interior wall faces, without choosing a crack path.

Node-star connectivity across uncut faces defines which corners remain tied.
No coordinates are merged and no initial opening or material change is imposed.
The caller supplies fracture initiation/growth decisions and all external tags.
"""
from dataclasses import dataclass, replace

import numpy as np

if __package__:
    from .fracture_faces import PairedCrackFaces, _vectors
else:
    from fracture_faces import PairedCrackFaces, _vectors


class _FullyTiedCut(ValueError):
    def __init__(self, keys):
        super().__init__("Cut is fully tied at its front; refine/enlarge the surface to represent a P1 opening")
        self.keys = list(keys)


@dataclass(frozen=True)
class SplitTetrahedra:
    coordinates_m: np.ndarray
    cells: np.ndarray
    region_ids: np.ndarray
    parent_node_ids: np.ndarray
    parent_cell_ids: np.ndarray
    source_faces: np.ndarray
    adjacent_cells: np.ndarray
    faces: PairedCrackFaces
    tied_source_faces: np.ndarray = None

    def transfer_nodal_values(self, source_values) -> np.ndarray:
        """Duplicate the accepted state; initiation creates no displacement jump."""
        values = np.asarray(source_values)
        source_count = int(self.parent_node_ids.max()) + 1
        if (values.dtype.kind not in "fiu" or values.ndim < 1 or values.shape[0] != source_count
                or not np.isfinite(values).all()):
            raise ValueError("Source nodal values must be finite and match the original node count")
        return values[self.parent_node_ids].copy()

    def transfer_boundary_faces(self, source_triangles, boundary_tags) -> tuple[np.ndarray, np.ndarray]:
        """Preserve explicit exterior labels by source connectivity, not location.

        The new crack lips are deliberately excluded; pressure/contact labels
        must come from the paired-face inventory, never an inherited reservoir
        label. Existing spring-interface split-node identities remain distinct.
        """
        triangles = np.asarray(source_triangles)
        tags = np.asarray(boundary_tags)
        if (triangles.dtype.kind not in "iu" or triangles.ndim != 2 or triangles.shape[1] != 3
                or not len(triangles) or np.any(triangles < 0)
                or np.any(triangles > self.parent_node_ids.max())
                or any(len(set(row)) != 3 for row in triangles)
                or len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles)
                or tags.dtype.kind not in "iu" or tags.shape != (len(triangles),)
                or np.any(tags <= 0)):
            raise ValueError("Exterior source triangles and positive integer tags must be valid and unique")
        source_cells = self.parent_node_ids[self.cells]
        owners = {}
        for cell_id, cell in enumerate(source_cells):
            for pattern in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                key = tuple(sorted(int(cell[i]) for i in pattern))
                owners.setdefault(key, []).append(cell_id)
        mapped = []
        for triangle in triangles:
            adjacent = owners.get(tuple(sorted(map(int, triangle))), [])
            if len(adjacent) != 1:
                raise ValueError("An exterior label must have exactly one original owner; crack lips need explicit new labels")
            cell_id = adjacent[0]
            mapped.append([int(self.cells[cell_id, np.flatnonzero(source_cells[cell_id] == node)[0]])
                           for node in triangle])
        return np.asarray(mapped, dtype=np.int64), tags.copy()


def _source_inventory(coordinates_m, cells, region_ids):
    points = _vectors(coordinates_m, "coordinates_m")
    tetrahedra = np.asarray(cells)
    regions = np.asarray(region_ids)
    if (tetrahedra.dtype.kind not in "iu" or tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4
            or not len(tetrahedra) or np.any(tetrahedra < 0) or np.any(tetrahedra >= len(points))
            or any(len(set(cell)) != 4 for cell in tetrahedra)
            or len(np.unique(np.sort(tetrahedra, axis=1), axis=0)) != len(tetrahedra)
            or regions.dtype.kind not in "iu" or regions.shape != (len(tetrahedra),)
            or np.any(regions <= 0)):
        raise ValueError("Source requires valid distinct tetrahedra and positive cell region IDs")
    determinants = np.linalg.det(points[tetrahedra[:, 1:]] - points[tetrahedra[:, :1]])
    if not np.isfinite(determinants).all() or np.any(np.abs(determinants) <= 1e-14):
        raise ValueError("Source tetrahedra must have finite nonzero volume")
    owners = {}
    for cell_id, cell in enumerate(tetrahedra):
        for pattern in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            key = tuple(sorted(int(cell[i]) for i in pattern))
            owners.setdefault(key, []).append(cell_id)
    if any(len(ids) > 2 for ids in owners.values()):
        raise ValueError("Nonmanifold tetrahedral face has more than two owners")
    return points, tetrahedra.astype(np.int64), regions, owners


def interior_wall_candidates(coordinates_m, cells, region_ids, marker_edges):
    """Canonical source facets for experimental mesh-constrained energy trials.

    All interior wall orientations are retained. This is not a continuous
    crack-path representation; native node-star resolution still has to be
    checked for every proposed cut, and a failed trial cannot mean arrest.
    """
    points, tetrahedra, regions, owners = _source_inventory(coordinates_m, cells, region_ids)
    edges = np.asarray(marker_edges)
    if (edges.dtype.kind not in "iu" or edges.ndim != 2 or edges.shape[1] != 2 or not len(edges)
            or np.any(edges < 0) or np.any(edges >= len(points)) or np.any(edges[:, 0] == edges[:, 1])
            or len(np.unique(np.sort(edges, axis=1), axis=0)) != len(edges)):
        raise ValueError("Markers must be unique source-node edge pairs")
    candidates = []
    for face, ids in sorted(owners.items()):
        if len(ids) != 2 or np.any(regions[ids] != 1):
            continue
        triangle = points[list(face)]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        sides = [(points[tetrahedra[i]].mean(axis=0) - triangle[0]) @ normal for i in ids]
        if sides[0] * sides[1] >= 0:
            raise ValueError("Interior wall candidate owners must lie on opposite sides")
        candidates.append(face)
    if not candidates:
        raise ValueError("No interior wall-concrete growth facets exist")
    inventory = {tuple(sorted((face[a], face[b]))) for face in candidates
                 for a, b in ((0, 1), (1, 2), (2, 0))}
    if any(tuple(sorted(map(int, edge))) not in inventory for edge in edges):
        raise ValueError("Every marker must touch an actual interior wall-concrete candidate facet")
    return np.asarray(candidates, dtype=np.int64)


class NodeStarPatchResolver:
    """Finite P1 surface resolution, not a prescribed gap or a fracture decision.

    At each seed vertex, an integer minimum bond cut separates its two lip
    identities in the remaining node-star graph. Unit capacities minimise the
    number of additional facets, not physical energy. Every resulting facet's
    actual area is charged by the subsequent Griffith equilibrium comparison.
    """

    def __init__(self, faces: PairedCrackFaces, parent_node_ids):
        parents = np.asarray(parent_node_ids)
        if (parents.dtype.kind not in "iu" or parents.shape != (len(faces.coordinates),)
                or not np.array_equal(parents[faces.minus], parents[faces.plus])):
            raise ValueError("Node-star patches require matching parent identities on paired lips")
        self.faces = faces
        self.triangles = parents[faces.minus]
        self.stars = {}

    def __call__(self, broken, seed_patch):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import maximum_flow, breadth_first_order

        broken = set(broken)
        seeds = set(seed_patch)
        patches = set()
        for seed in sorted(seeds):
            for corner, parent in enumerate(self.triangles[seed]):
                parent = int(parent)
                if parent not in self.stars:
                    self.stars[parent] = np.argwhere(self.triangles == parent)
                incident = self.stars[parent]
                incident = incident[~np.isin(incident[:, 0], list(broken))]
                face_ids, corners = incident.T
                left = self.faces.minus[face_ids, corners]
                right = self.faces.plus[face_ids, corners]
                nodes = np.unique(np.concatenate((left, right)))
                start = int(np.searchsorted(nodes, self.faces.minus[seed, corner]))
                end = int(np.searchsorted(nodes, self.faces.plus[seed, corner]))
                if start == end:
                    continue
                a, b = np.searchsorted(nodes, left), np.searchsorted(nodes, right)
                capacity = coo_matrix(
                    (np.ones(2 * len(a), dtype=np.int64),
                     (np.concatenate((a, b)), np.concatenate((b, a)))),
                    shape=(len(nodes), len(nodes))).tocsr()
                flow = maximum_flow(capacity, start, end)
                residual = capacity - flow.flow
                residual.data = (residual.data > 0).astype(np.int64)
                residual.eliminate_zeros()
                reachable = np.zeros(len(nodes), dtype=bool)
                reachable[breadth_first_order(residual, start, directed=True,
                                              return_predecessors=False)] = True
                cut = set(map(int, face_ids[reachable[a] != reachable[b]]))
                patch = tuple(sorted(seeds | cut))
                if seed not in cut or not self._connected_disk(patch):
                    continue
                patches.add(patch)
        ordered = sorted(patches, key=lambda ids: (len(ids), ids))
        return [patch for patch in ordered
                if not any(set(other) < set(patch) for other in ordered)]

    def _connected_disk(self, patch):
        edge_faces = {}
        for face in patch:
            row = self.triangles[face]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                edge_faces.setdefault(tuple(sorted(map(int, row[[a, b]]))), []).append(face)
        if any(len(owners) > 2 for owners in edge_faces.values()):
            return False
        neighbours = {face: set() for face in patch}
        for owners in edge_faces.values():
            for face in owners:
                neighbours[face].update(owners)
        reached, pending = set(), {patch[0]}
        while pending:
            face = pending.pop()
            reached.add(face)
            pending.update(neighbours[face] - reached)
        # A local disk has a single unbranched boundary and Euler characteristic 1.
        boundary = {}
        for edge, owners in edge_faces.items():
            if len(owners) == 1:
                for node in edge:
                    boundary[node] = boundary.get(node, 0) + 1
        vertices = set(map(int, self.triangles[list(patch)].ravel()))
        return (len(reached) == len(patch) and bool(boundary)
                and all(degree == 2 for degree in boundary.values())
                and len(vertices) - len(edge_faces) + len(patch) == 1)


def split_interior_wall_faces(coordinates_m, cells, region_ids, cut_faces, *,
                              drop_fully_tied: bool = False) -> SplitTetrahedra:
    """Split source-node triples; support shared tip nodes but reject fully tied cuts.

    All cell rows and regions are retained unchanged. Appended nodes map back to
    original nodes. Repeated growth must carry provenance through successive
    results; output node indices alone are not catalog/source identities.
    With ``drop_fully_tied`` such cuts are instead removed (repeatedly, as each
    removal ties more nodes) and listed in ``tied_source_faces``.
    """
    if drop_fully_tied:
        cuts = np.asarray(cut_faces)
        dropped = []
        while True:
            try:
                result = split_interior_wall_faces(coordinates_m, cells, region_ids, cuts)
            except _FullyTiedCut as error:
                tied = set(error.keys)
                keep = np.array([tuple(sorted(map(int, row))) not in tied for row in cuts])
                dropped.extend(sorted(tied))
                if not keep.any():
                    raise ValueError("Every cut is fully tied at its front; nothing can open") from None
                cuts = cuts[keep]
                continue
            return replace(result, tied_source_faces=np.asarray(dropped, dtype=np.int64).reshape(-1, 3))
    points, tetrahedra, regions, owners = _source_inventory(coordinates_m, cells, region_ids)
    cuts = np.asarray(cut_faces)
    if (cuts.dtype.kind not in "iu" or cuts.ndim != 2 or cuts.shape[1] != 3 or not len(cuts)
            or np.any(cuts < 0) or np.any(cuts >= len(points))
            or any(len(set(row)) != 3 for row in cuts)
            or len(np.unique(np.sort(cuts, axis=1), axis=0)) != len(cuts)):
        raise ValueError("Cut faces must be nonempty unique known source-node triples")
    vertices = points[tetrahedra]
    cut_keys = [tuple(sorted(map(int, row))) for row in cuts]
    for key in cut_keys:
        adjacent = owners.get(key, [])
        if len(adjacent) != 2:
            raise ValueError("Each cut must be an interior face with exactly two owners")
        if np.any(regions[adjacent] != 1):
            raise ValueError("Fracture cuts must lie between wall concrete cells (region 1)")
        normal = np.cross(points[key[1]] - points[key[0]], points[key[2]] - points[key[0]])
        sides = [(vertices[cell_id].mean(axis=0) - points[key[0]]) @ normal for cell_id in adjacent]
        if sides[0] * sides[1] >= 0:
            raise ValueError("Interior face owners must lie on opposite sides, not overlapping")
    cut_set = set(cut_keys)
    affected_nodes = set(map(int, cuts.ravel()))
    parents = np.arange(tetrahedra.size, dtype=np.int64)

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for face, adjacent in owners.items():
        if len(adjacent) != 2 or face in cut_set:
            continue
        left, right = adjacent
        for node in face:
            if node in affected_nodes:
                a = root(left * 4 + int(np.flatnonzero(tetrahedra[left] == node)[0]))
                b = root(right * 4 + int(np.flatnonzero(tetrahedra[right] == node)[0]))
                parents[max(a, b)] = min(a, b)

    result = tetrahedra.copy()
    components = {}
    node_parents = list(range(len(points)))
    for cell_id, cell in enumerate(tetrahedra):
        for corner, node in enumerate(cell):
            if node not in affected_nodes:
                continue
            component = root(cell_id * 4 + corner)
            by_component = components.setdefault(int(node), {})
            if component not in by_component:
                if not by_component:
                    by_component[component] = int(node)
                else:
                    by_component[component] = len(node_parents)
                    node_parents.append(int(node))
            result[cell_id, corner] = by_component[component]

    minus, plus, normals, adjacent_rows, tied_keys = [], [], [], [], []
    for key in cut_keys:
        left, right = owners[key]
        sides = []
        for cell_id in (left, right):
            sides.append([int(result[cell_id, np.flatnonzero(tetrahedra[cell_id] == node)[0]]) for node in key])
        if sides[0] == sides[1]:
            tied_keys.append(key)
            continue
        minus.append(sides[0])
        plus.append(sides[1])
        normal = np.cross(points[key[1]] - points[key[0]], points[key[2]] - points[key[0]])
        if normal @ (vertices[right].mean(axis=0) - vertices[left].mean(axis=0)) < 0:
            normal = -normal
        normals.append(normal / np.linalg.norm(normal))
        adjacent_rows.append([left, right])
    if tied_keys:
        raise _FullyTiedCut(tied_keys)
    node_parents = np.asarray(node_parents, dtype=np.int64)
    new_points = points[node_parents].copy()
    paired = PairedCrackFaces(new_points, minus, plus, normals)
    return SplitTetrahedra(
        new_points, result, regions.copy(), node_parents, np.arange(len(result), dtype=np.int64),
        np.asarray(cut_keys, dtype=np.int64), np.asarray(adjacent_rows, dtype=np.int64), paired,
    )
