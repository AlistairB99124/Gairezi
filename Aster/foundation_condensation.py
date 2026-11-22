"""Condense a linear, spring-connected foundation onto its wall interface."""
from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from Aster.support_trace import WEIGHTS
from bedrock_support import consistent_face_mass


@dataclass(frozen=True)
class CondensedFoundation:
    wall_nodes: np.ndarray
    retained_foundation_nodes: np.ndarray
    stiffness_n_per_m: np.ndarray
    gravity_load_n: np.ndarray


def _tetra_stiffness(points: np.ndarray, youngs_modulus: float, poisson_ratio: float) -> tuple[np.ndarray, float]:
    determinant = np.linalg.det(points[1:] - points[0])
    volume = abs(determinant) / 6.0
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Foundation contains a degenerate tetrahedron")
    gradients = np.linalg.inv(np.column_stack((np.ones(4), points)))[1:]
    strain = np.zeros((6, 12))
    for node in range(4):
        x, y, z = gradients[:, node]
        strain[:, 3 * node:3 * node + 3] = (
            (x, 0, 0), (0, y, 0), (0, 0, z),
            (y, x, 0), (0, z, y), (z, 0, x),
        )
    shear = youngs_modulus / (2 * (1 + poisson_ratio))
    lame = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    elasticity = np.zeros((6, 6))
    elasticity[:3, :3] = lame + 2 * shear * np.eye(3)
    elasticity[3:, 3:] = shear * np.eye(3)
    return volume * strain.T @ elasticity @ strain, volume


def _retained_nodes_for_link_budget(
    foundation_cells: np.ndarray, support_faces: np.ndarray,
    pairs: np.ndarray, max_links: int,
) -> np.ndarray:
    foundation_nodes = np.unique(foundation_cells)
    graph = {int(node): set() for node in np.unique(np.concatenate((foundation_nodes, pairs[:, 0])))}
    for cells in (foundation_cells, support_faces, pairs):
        for cell in cells:
            for node in cell:
                graph[int(node)].update(int(other) for other in cell if node != other)
    edges = sum(map(len, graph.values())) // 2
    # Native characteristic groups are costly to import; scale the fill budget
    # with the original graph while retaining the smaller budget for coarse meshes.
    budget = min(max_links, max(14000, (13 * edges + 9) // 10))
    remaining = set(foundation_nodes.tolist())
    heap = [(len(graph[node]), node) for node in remaining]
    heapq.heapify(heap)
    while heap:
        degree, node = heapq.heappop(heap)
        if node not in remaining or degree != len(graph[node]):
            continue
        neighbors = graph[node]
        fill = sum(len(neighbors - graph[other] - {other}) for other in neighbors) // 2
        next_edges = edges - degree + fill
        if next_edges > budget:
            break
        for other in neighbors:
            graph[other].update(neighbors - {other, node})
            graph[other].discard(node)
            if other in remaining:
                heapq.heappush(heap, (len(graph[other]), other))
        del graph[node]
        remaining.remove(node)
        edges = next_edges
    if edges > max_links:
        raise ValueError(
            f"Foundation condensation resource limit: the retained support graph needs "
            f"{edges} links, exceeding the limit of {max_links}."
        )
    return np.array(sorted(remaining), dtype=np.int64)


def condense_foundation(
    arrays: dict, *, youngs_modulus: float, poisson_ratio: float,
    density: float, gravity_m_s2: float,
    retained_foundation_nodes: np.ndarray | None = None,
    max_links: int | None = None,
    max_dense_bytes: int = 4 * 1024**3,
) -> CondensedFoundation:
    """Reduce the source Aster P1 foundation and its linear seat springs.

    The returned force is the foundation gravity contribution on the wall
    interface. Matrix and force use ``wall_nodes`` followed by any retained
    foundation nodes, whose linear response remains part of the support.
    """
    material = np.asarray((youngs_modulus, poisson_ratio, density, gravity_m_s2), dtype=float)
    if (not np.isfinite(material).all() or youngs_modulus <= 0 or density <= 0
            or not -1 < poisson_ratio < 0.5):
        raise ValueError("Foundation material and gravity must be finite and physically valid")
    nodes = np.asarray(arrays["nodes"], dtype=float)
    cells = np.asarray(arrays["tetra4_concrete"])
    bodies = np.asarray(arrays["concrete_body_ids"])
    pairs = np.asarray(arrays["seg2_wall_foundation_springs"])
    springs = np.asarray(arrays["interface_stiffness_n_per_m"], dtype=float)
    names = np.asarray(arrays["interface_names"])
    faces = np.asarray(arrays["support_faces"])
    tensors = np.asarray(arrays["support_face_tensors_n_per_m3"], dtype=float)
    if len(arrays.get("tetra4_bedrock", ())) or len(arrays.get("tetra4_mortar", ())):
        raise ValueError("Condensation does not support bedrock or mortar volume bodies")
    if (nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all()
            or cells.ndim != 2 or cells.shape[1] != 4
            or not np.issubdtype(cells.dtype, np.integer)
            or np.any(cells < 0) or np.any(cells >= len(nodes))
            or bodies.shape != (len(cells),)):
        raise ValueError("Invalid source nodes or concrete tetrahedra")
    foundation_cells = cells[bodies == 3]
    wall_cells = cells[bodies == 1]
    if not len(foundation_cells) or not len(wall_cells) or np.any(~np.isin(bodies, (1, 3))):
        raise ValueError("Condensation needs distinct wall and foundation tetrahedra")
    foundation_nodes = np.unique(foundation_cells)
    retained = np.asarray(
        np.empty(0, dtype=np.int64) if retained_foundation_nodes is None else retained_foundation_nodes)
    if max_links is not None and (retained_foundation_nodes is not None
                                  or not isinstance(max_links, int) or isinstance(max_links, bool)
                                  or max_links <= 0):
        raise ValueError("A positive link budget cannot be combined with explicitly retained nodes")
    if (retained.ndim != 1 or len(np.unique(retained)) != len(retained)
            or not np.issubdtype(retained.dtype, np.integer)
            or not np.isin(retained, foundation_nodes).all()):
        raise ValueError("Retained nodes must be distinct foundation nodes")
    retained = np.sort(retained)
    wall_volume_nodes = np.unique(wall_cells)
    if np.intersect1d(foundation_nodes, wall_volume_nodes).size:
        raise ValueError("Bonded wall/foundation nodes are not supported")
    if (np.isin(arrays.get("fixed_support_nodes", ()), foundation_nodes).any()
            or any(np.isin(arrays.get(name, ()), foundation_nodes).any()
                   for name in ("tri3_upstream", "tri3_downstream", "tri3_crest"))):
        raise ValueError("Foundation has unsupported constraints or external surface loads")
    if (pairs.ndim != 2 or pairs.shape[1] != 2
            or not np.issubdtype(pairs.dtype, np.integer)
            or np.any(pairs < 0) or np.any(pairs >= len(nodes))
            or springs.shape != (len(pairs), 3) or names.shape != (len(pairs),)
            or not np.isfinite(springs).all() or np.any(springs <= 0)):
        raise ValueError("Invalid interface spring inventory")
    selected = np.array([str(name).split("/")[0] == "wall_foundation" for name in names])
    foundation_pairs, foundation_springs = pairs[selected], springs[selected]
    if (not len(foundation_pairs)
            or not np.isin(foundation_pairs[:, 0], wall_volume_nodes).all()
            or not np.isin(foundation_pairs[:, 1], foundation_nodes).all()
            or np.isin(pairs[~selected], foundation_nodes).any()):
        raise ValueError("Foundation requires wall-to-foundation springs and no other interface connections")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer)
            or not len(faces) or not np.isin(faces, foundation_nodes).all()
            or tensors.shape != (len(faces), 3, 3) or not np.isfinite(tensors).all()
            or not np.allclose(tensors, tensors.transpose(0, 2, 1), rtol=1e-12, atol=1e-8)):
        raise ValueError("Foundation requires finite symmetric foundation-seat support tensors")
    if np.any(np.linalg.eigvalsh(tensors) < -1e-12 * np.maximum(1., np.linalg.norm(tensors, axis=(1, 2)))[:, None]):
        raise ValueError("Foundation-seat support tensors must be positive semidefinite")
    mass = consistent_face_mass(nodes, faces)
    support_nodes = np.asarray(arrays["support_nodes"])
    interpolation_nodes = np.asarray(arrays["support_interpolation_nodes"])
    parents = np.asarray(arrays["support_interpolation_parents"])
    weights = np.asarray(arrays["support_interpolation_weights"], dtype=float)
    support_matrices = np.asarray(arrays["support_matrices_n_per_m"], dtype=float)
    expected_support = np.repeat(tensors * mass.sum(axis=(1, 2))[:, None, None] / 3, 3, axis=0)
    if (support_nodes.shape != (3 * len(faces),)
            or not np.issubdtype(support_nodes.dtype, np.integer)
            or np.any(support_nodes < 0) or np.any(support_nodes >= len(nodes))
            or len(np.unique(support_nodes)) != len(support_nodes)
            or not np.array_equal(interpolation_nodes, support_nodes)
            or not np.array_equal(parents, np.repeat(faces, 3, axis=0))
            or weights.shape != (3 * len(faces), 3)
            or not np.allclose(weights, np.tile(WEIGHTS, (len(faces), 1)), rtol=0., atol=1e-14)
            or support_matrices.shape != expected_support.shape
            or not np.allclose(support_matrices, expected_support, rtol=1e-12, atol=1e-8)
            or not np.allclose(nodes[support_nodes], np.einsum("ni,nij->nj", weights, nodes[parents]),
                               rtol=0., atol=1e-9)):
        raise ValueError("Foundation support must use the recorded consistent P1 face trace")
    if max_links is not None:
        retained = _retained_nodes_for_link_budget(foundation_cells, faces, foundation_pairs, max_links)

    wall_nodes = np.unique(foundation_pairs[:, 0])
    if isinstance(max_dense_bytes, bool) or not isinstance(max_dense_bytes, int) or max_dense_bytes <= 0:
        raise ValueError("Condensation dense-memory limit must be a positive number of bytes")
    reduced_dofs = 3 * (len(wall_nodes) + len(retained))
    # Include the dense result and temporary arrays used for symmetry and block extraction.
    estimated_dense_bytes = 5 * reduced_dofs * reduced_dofs * np.dtype(np.float64).itemsize
    if estimated_dense_bytes > max_dense_bytes:
        raise ValueError(
            f"Foundation condensation resource limit: estimated dense memory "
            f"{estimated_dense_bytes / 1024**3:.2f} GiB for {reduced_dofs} reduced degrees of freedom "
            f"exceeds the {max_dense_bytes / 1024**3:.2f} GiB limit. "
            "Disable foundation condensation to retain the original foundation and supports, "
            "or increase the limit only when sufficient memory is available."
        )
    local = np.full(len(nodes), -1, dtype=np.int64)
    local[foundation_nodes] = np.arange(len(foundation_nodes))
    wall_local = np.full(len(nodes), -1, dtype=np.int64)
    wall_local[wall_nodes] = np.arange(len(wall_nodes))
    internal_size = 3 * len(foundation_nodes)
    boundary_size = 3 * len(wall_nodes)
    rows, columns, values = [], [], []
    gravity = np.zeros(internal_size)
    for cell in foundation_cells:
        matrix, volume = _tetra_stiffness(nodes[cell], youngs_modulus, poisson_ratio)
        dofs = (3 * local[cell, None] + np.arange(3)).ravel()
        rows.append(np.repeat(dofs, 12))
        columns.append(np.tile(dofs, 12))
        values.append(matrix.ravel())
        np.add.at(gravity, 3 * local[cell] + 2, density * gravity_m_s2 * volume / 4)
    face_dofs = (3 * local[faces][:, :, None] + np.arange(3)).reshape(len(faces), 9)
    face_matrices = (mass[:, :, :, None, None] * tensors[:, None, None, :, :]).transpose(0, 1, 3, 2, 4)
    rows.append(np.repeat(face_dofs, 9, axis=1).ravel())
    columns.append(np.tile(face_dofs, (1, 9)).ravel())
    values.append(face_matrices.reshape(-1))
    internal_dofs = 3 * local[foundation_pairs[:, 1], None] + np.arange(3)
    boundary_dofs = 3 * wall_local[foundation_pairs[:, 0], None] + np.arange(3)
    rows.append(internal_dofs.ravel())
    columns.append(internal_dofs.ravel())
    values.append(foundation_springs.ravel())
    internal = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(internal_size, internal_size),
    ).tocsc()
    coupling = sparse.coo_matrix(
        (-foundation_springs.ravel(), (internal_dofs.ravel(), boundary_dofs.ravel())),
        shape=(internal_size, boundary_size),
    ).tocsc()
    diagonal = np.bincount(boundary_dofs.ravel(), weights=foundation_springs.ravel(),
                           minlength=boundary_size)
    retained_dofs = (3 * local[retained, None] + np.arange(3)).ravel()
    eliminated_dofs = np.setdiff1d(np.arange(internal_size), retained_dofs, assume_unique=True)
    boundary = sparse.bmat(
        [[sparse.diags(diagonal), coupling[retained_dofs].T],
         [coupling[retained_dofs], internal[retained_dofs][:, retained_dofs]]],
        format="csc",
    )
    reduced = boundary.toarray()
    load = np.concatenate((np.zeros(boundary_size), gravity[retained_dofs]))
    if len(eliminated_dofs):
        cross = sparse.hstack(
            (coupling[eliminated_dofs], internal[eliminated_dofs][:, retained_dofs]),
            format="csc",
        )
        try:
            factor = splu(internal[eliminated_dofs][:, eliminated_dofs])
        except RuntimeError as error:
            raise ValueError("Foundation internal stiffness cannot be factorized") from error
        for start in range(0, reduced.shape[1], 32):
            end = min(start + 32, reduced.shape[1])
            reduced[:, start:end] -= cross.T @ factor.solve(cross[:, start:end].toarray())
        load -= cross.T @ factor.solve(gravity[eliminated_dofs])
    if not np.isfinite(reduced).all() or not np.isfinite(load).all():
        raise ValueError("Condensed foundation contains nonfinite stiffness or gravity")
    scale = np.max(np.abs(reduced))
    if scale <= 0 or not np.allclose(reduced, reduced.T, rtol=1e-10, atol=1e-10 * scale):
        raise ValueError("Condensed foundation stiffness is not symmetric")
    return CondensedFoundation(wall_nodes, retained, (reduced + reduced.T) / 2, np.asarray(load))


def prepare_condensed_support(
    arrays: dict, *, joint_cells: np.ndarray, youngs_modulus: float, poisson_ratio: float,
    density: float, gravity_m_s2: float, max_links: int, max_dense_bytes: int = 4 * 1024**3,
) -> tuple[dict, dict]:
    """Replace supported linear foundation cells with a discrete reduced support."""
    from Aster.quadratic_surface import group_stiffness_blocks

    names = np.asarray(arrays["interface_names"])
    if any(str(name).split("/")[0] != "wall_foundation" for name in names):
        raise ValueError("Condensed support cannot replace other interface springs")
    if len(arrays.get("fixed_support_nodes", ())):
        raise ValueError("Condensed support does not support additional fixed nodes")
    concrete = np.asarray(arrays["tetra4_concrete"])
    bodies = np.asarray(arrays["concrete_body_ids"])
    foundation_mask = bodies == 3
    foundation_nodes = np.unique(concrete[foundation_mask])
    joints = np.asarray(joint_cells)
    source_ids = np.asarray(arrays.get("source_node_ids", np.arange(len(arrays["nodes"]))))
    if (joints.ndim != 2 or joints.shape[1] != 6
            or not np.issubdtype(joints.dtype, np.integer)
            or np.any(joints < 0) or np.any(joints >= len(source_ids))):
        raise ValueError("Invalid split joint inventory")
    if len(joints) and np.isin(source_ids[joints], source_ids[foundation_nodes]).any():
        raise ValueError("Condensed support requires crack candidates outside the foundation")
    selected = np.asarray(arrays.get("damaged_cells", np.zeros(len(concrete), dtype=bool)))
    if selected.shape != (len(concrete),) or selected.dtype != np.dtype(bool):
        raise ValueError("Damaged cell inventory does not match the concrete mesh")
    if selected[foundation_mask].any():
        raise ValueError("Condensed support requires an undamaged linear foundation")
    reduced = condense_foundation(
        arrays, youngs_modulus=youngs_modulus, poisson_ratio=poisson_ratio,
        density=density, gravity_m_s2=gravity_m_s2, max_links=max_links,
        max_dense_bytes=max_dense_bytes,
    )
    nodes = np.r_[reduced.wall_nodes, reduced.retained_foundation_nodes]
    count = len(nodes)
    blocks = reduced.stiffness_n_per_m.reshape(count, 3, count, 3).transpose(0, 2, 1, 3)
    first, second = np.triu_indices(count, 1)
    selected_links = np.any(blocks[first, second] != 0, axis=(1, 2))
    links = np.column_stack((first[selected_links], second[selected_links]))
    if len(links) > max_links:
        raise ValueError(
            f"Foundation condensation resource limit: the reduced support contains "
            f"{len(links)} links, exceeding the limit of {max_links}."
        )
    point_matrices = blocks[np.arange(count), np.arange(count)]
    link_matrices = blocks[links[:, 0], links[:, 1]]
    point_groups, point_ids, _ = group_stiffness_blocks(point_matrices)
    link_groups, link_ids, _ = group_stiffness_blocks(link_matrices)
    result = {key: np.asarray(value).copy() for key, value in arrays.items()}
    result["tetra4_concrete"] = concrete[~foundation_mask]
    result["concrete_body_ids"] = bodies[~foundation_mask]
    if "source_cell_rows" in result:
        rows = result["source_cell_rows"]
        if rows.shape != (len(concrete),):
            raise ValueError("Source cell identities do not match the concrete mesh")
        result["source_cell_rows"] = rows[~foundation_mask]
    if "damaged_cells" in result:
        result["damaged_cells"] = selected[~foundation_mask]
    for key in ("seg2_wall_foundation_springs", "wall_foundation_tributary_area_m2",
                "interface_stiffness_n_per_m", "interface_names", "support_nodes",
                "support_matrices_n_per_m", "support_interpolation_nodes",
                "support_interpolation_parents", "support_interpolation_weights",
                "support_faces", "support_face_tensors_n_per_m3"):
        result[key] = result[key][:0]
    result.update(
        point_node_ids=nodes,
        link_node_ids=nodes[links],
        point_matrices_n_per_m=point_matrices,
        link_matrices_n_per_m=link_matrices,
        point_group_matrices_n_per_m=point_groups,
        link_group_matrices_n_per_m=link_groups,
        point_group_ids=point_ids,
        link_group_ids=link_ids,
        condensed_gravity_load_n=reduced.gravity_load_n.reshape(-1, 3),
    )
    active = [result["tetra4_concrete"].ravel(), nodes, joints.ravel()]
    for key in ("hexa8_joints", "tie_node_pairs", "interpolation_nodes", "interpolation_parents"):
        if key in result:
            active.append(result[key].ravel())
    result["condensed_active_nodes"] = np.unique(np.concatenate(active)).astype(np.int64)
    report = {
        "wall_interface_nodes": len(reduced.wall_nodes),
        "retained_foundation_nodes": len(reduced.retained_foundation_nodes),
        "eliminated_foundation_nodes": len(foundation_nodes) - len(reduced.retained_foundation_nodes),
        "foundation_cells_removed": int(foundation_mask.sum()),
        "discrete_links": len(links),
    }
    return result, report
