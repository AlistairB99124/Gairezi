"""Exact P1 face springs represented by native point springs at degree-two quadrature."""
import numpy as np

WEIGHTS = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))


def add_consistent_support(arrays: dict, faces, tensors) -> None:
    points = np.asarray(arrays["nodes"])
    faces = np.asarray(faces, dtype=np.int64)
    tensors = np.asarray(tensors, dtype=float)
    vertices = points[faces]
    area = np.linalg.norm(np.cross(vertices[:, 1] - vertices[:, 0],
                                   vertices[:, 2] - vertices[:, 0]), axis=1) / 2
    if tensors.shape != (len(faces), 3, 3) or not np.isfinite(tensors).all() or np.any(area <= 0):
        raise ValueError("Invalid consistent face support")
    coordinates = np.einsum("qa,faj->fqj", WEIGHTS, vertices).reshape(-1, 3)
    nodes = np.arange(len(points), len(points) + len(coordinates), dtype=np.int64)
    arrays.update(
        nodes=np.vstack((points, coordinates)),
        source_node_ids=np.concatenate((arrays["source_node_ids"], np.full(len(nodes), -1, dtype=np.int64))),
        support_nodes=nodes,
        support_matrices_n_per_m=np.repeat(tensors * area[:, None, None] / 3, 3, axis=0),
        support_interpolation_nodes=nodes,
        support_interpolation_parents=np.repeat(faces, 3, axis=0),
        support_interpolation_weights=np.tile(WEIGHTS, (len(faces), 1)),
        support_faces=faces,
        support_face_tensors_n_per_m3=tensors,
    )


def render_constraints(arrays: dict, *, prefix="support_", name="SUPPORT_TRACE") -> str:
    nodes = np.asarray(arrays.get(prefix + "interpolation_nodes", []), dtype=np.int64)
    if not len(nodes):
        return ""
    parents = arrays[prefix + "interpolation_parents"].tolist()
    weights = arrays[prefix + "interpolation_weights"].tolist()
    return (
        f"{name}_NODES = {nodes.tolist()!r}\n{name}_PARENTS = {parents!r}\n{name}_WEIGHTS = {weights!r}\n"
        f"{name} = AFFE_CHAR_MECA(MODELE=MODELE, LIAISON_DDL=[\n"
        "    _F(NOEUD=(f'N{child + 1}', *(f'N{node + 1}' for node, weight in zip(parents, weights) if weight)),\n"
        "       DDL=(dof,) * (1 + sum(weight != 0 for weight in weights)),\n"
        "       COEF_MULT=(1.0, *(-weight for weight in weights if weight)), COEF_IMPO=0.0)\n"
        f"    for child, parents, weights in zip({name}_NODES, {name}_PARENTS, {name}_WEIGHTS)\n"
        "    for dof in ('DX', 'DY', 'DZ')])\n\n")
