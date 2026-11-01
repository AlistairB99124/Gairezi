"""Local elastic-layer bedrock support for preliminary static assessment."""
from __future__ import annotations

import numpy as np


def elastic_layer_tensors(points, faces, *, youngs_modulus: float,
                          poisson_ratio: float, bottom_z_m: float,
                          spring_n_per_m3: dict | None = None) -> dict:
    """Distributed stiffness in N/m3, with depth measured vertically to the bottom."""
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Bedrock support points must be finite three-dimensional coordinates")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= len(points))):
        raise ValueError("Bedrock support requires nonempty valid triangular seat faces")
    if (isinstance(youngs_modulus, bool) or not np.isfinite(youngs_modulus)
            or youngs_modulus <= 0):
        raise ValueError("Bedrock Young's modulus must be finite and positive")
    if (isinstance(poisson_ratio, bool) or not np.isfinite(poisson_ratio)
            or not -1 < poisson_ratio < .5):
        raise ValueError("Bedrock Poisson ratio must lie strictly between -1 and 0.5")
    if isinstance(bottom_z_m, bool) or not np.isfinite(bottom_z_m):
        raise ValueError("Bedrock bottom elevation must be finite")
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        raise ValueError("Bedrock support seat contains duplicate faces")
    vertices = points[faces]
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    if np.any(twice_area <= 0):
        raise ValueError("Bedrock support seat contains a degenerate face")
    depth = vertices[:, :, 2].mean(axis=1) - bottom_z_m
    if np.any(depth <= 0):
        raise ValueError("Bedrock bottom must lie below every foundation-seat face centroid")
    normal = cross / twice_area[:, None]
    shear = youngs_modulus / (2 * (1 + poisson_ratio))
    constrained = youngs_modulus * (1 - poisson_ratio) / (
        (1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    tangential = shear / depth
    perpendicular = constrained / depth
    projection = normal[:, :, None] * normal[:, None, :]
    tensors = tangential[:, None, None] * np.eye(3) + (
        perpendicular - tangential)[:, None, None] * projection
    if not np.isfinite(tensors).all():
        raise ValueError("Bedrock layer stiffness overflowed")
    if spring_n_per_m3 is not None:
        if set(spring_n_per_m3) != {"x", "y", "z"} or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value <= 0 for value in spring_n_per_m3.values()
        ):
            raise ValueError("Bedrock distributed x/y/z stiffness must be finite and positive")
        tensors = np.repeat(
            np.diag([spring_n_per_m3[axis] for axis in ("x", "y", "z")])[None],
            len(faces), axis=0,
        )
        tangential = np.full(len(faces), spring_n_per_m3["x"])
        perpendicular = np.full(len(faces), spring_n_per_m3["z"])
    return {
        "face_area_m2": twice_area / 2,
        "effective_depth_m": depth,
        "normal_stiffness_n_per_m3": perpendicular,
        "tangential_stiffness_n_per_m3": tangential,
        "stiffness_n_per_m3": tensors,
    }


def lumped_layer_support(points, faces, **properties) -> dict:
    """Area-lumped nodal spring tensors; use the same discretisation in all solvers."""
    faces = np.asarray(faces)
    layer = elastic_layer_tensors(points, faces, **properties)
    nodes, inverse = np.unique(faces, return_inverse=True)
    matrices = np.zeros((len(nodes), 3, 3))
    per_vertex = layer["stiffness_n_per_m3"] * layer["face_area_m2"][:, None, None] / 3
    np.add.at(matrices, inverse.ravel(), np.repeat(per_vertex, 3, axis=0))
    return {**layer, "support_nodes": nodes, "support_matrices_n_per_m": matrices}


def consistent_face_mass(points, faces):
    """Exact P1 surface mass (m2): diagonal A/6, off-diagonal A/12."""
    vertices = np.asarray(points, dtype=float)[np.asarray(faces, dtype=np.int64)]
    area = np.linalg.norm(np.cross(vertices[:, 1] - vertices[:, 0],
                                   vertices[:, 2] - vertices[:, 0]), axis=1) / 2
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Consistent support requires finite nondegenerate triangles")
    return area[:, None, None] * (np.ones((3, 3)) + np.eye(3)) / 12


def reduced_layer_arrays(arrays: dict, config: dict) -> tuple[dict, dict]:
    """Concrete-only source arrays with compliant supports, never fixed-seat constraints."""
    from bedrock_comparison import rigid_seat, without_bedrock

    _, mapping, seat = rigid_seat(arrays)
    reduced = without_bedrock(arrays)
    rock = config["material"]["bedrock"]
    support = lumped_layer_support(
        reduced["nodes"], mapping[seat],
        youngs_modulus=rock["youngs_modulus"], poisson_ratio=rock["poisson_ratio"],
        bottom_z_m=config["geometry"]["bedrock"]["domain"]["bottom_z_m"],
        spring_n_per_m3=(config.get("bedrock_analysis") or {}).get("spring_n_per_m3"),
    )
    reduced["fixed_support_nodes"] = np.empty(0, dtype=np.int64)
    reduced["support_nodes"] = support["support_nodes"]
    reduced["support_matrices_n_per_m"] = support["support_matrices_n_per_m"]
    report = {
        "model": "elastic_layer",
        "depth_rule": "face-centroid vertical distance to geometry.bedrock.domain.bottom_z_m",
        "normal_stiffness_rule": "E*(1-nu)/((1+nu)*(1-2*nu)*h)",
        "tangential_stiffness_rule": "E/(2*(1+nu)*h)",
        "discretisation": "face-area/3 lumped nodal tensor springs",
        "effective_depth_min_m": float(support["effective_depth_m"].min()),
        "effective_depth_max_m": float(support["effective_depth_m"].max()),
        "seat_area_m2": float(support["face_area_m2"].sum()),
        "support_node_count": len(support["support_nodes"]),
        "limitations": [
            "Independent local springs omit cross-seat bedrock coupling",
            "Rock self-weight settlement is omitted",
            "Effective depth is a geometric preliminary-assessment assumption",
            "Linear springs transmit tension and shear; no rock separation or yielding",
        ],
    }
    return reduced, report
