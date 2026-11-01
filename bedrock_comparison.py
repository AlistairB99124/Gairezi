"""Rigid-seat sensitivity helpers; never an equivalent replacement for bedrock."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_vtu(path: Path, points, cells, displacement, bodies, principal, stress) -> None:
    from Elmer.regions.combined_structure import _packed_chunk

    chunks = []
    offset = 0

    def array(name, values, value_type="Float64", code="d", components=1):
        nonlocal offset
        chunk = _packed_chunk(code, np.asarray(values).ravel().tolist())
        field = (f'<DataArray type="{value_type}" Name="{name}" NumberOfComponents="{components}" '
                 f'format="appended" offset="{offset}"/>')
        chunks.append(chunk)
        offset += len(chunk)
        return field

    point_fields = array("displacement_m", displacement, components=3) + array(
        "displacement_magnitude_m", np.linalg.norm(displacement, axis=1))
    cell_fields = (array("body_id", bodies, "Int32", "i") +
                   array("principal_max_pa", principal[:, 2]) + array("principal_min_pa", principal[:, 0]) +
                   array("stress_pa", stress, components=9))
    coordinates = array("points", points, components=3)
    topology = (array("connectivity", cells, "Int64", "q") +
                array("offsets", np.arange(1, len(cells) + 1) * 4, "Int64", "q") +
                array("types", np.full(len(cells), 10), "UInt8", "B"))
    header = (
        '<?xml version="1.0"?><VTKFile type="UnstructuredGrid" version="0.1" '
        'byte_order="LittleEndian" header_type="UInt32"><UnstructuredGrid>'
        f'<Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(cells)}">'
        f'<PointData Vectors="displacement_m">{point_fields}</PointData>'
        f'<CellData Scalars="principal_max_pa">{cell_fields}</CellData>'
        f'<Points>{coordinates}</Points><Cells>{topology}</Cells>'
        '</Piece></UnstructuredGrid><AppendedData encoding="raw">').encode("ascii")
    path.write_bytes(header + b"_" + b"".join(chunks) + b"</AppendedData></VTKFile>")


def rigid_seat(arrays: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    concrete = np.asarray(arrays["tetra4_concrete"])
    rock = np.asarray(arrays["tetra4_bedrock"])
    bodies = np.asarray(arrays["concrete_body_ids"])
    if not len(rock) or not np.any(bodies == 3):
        raise ValueError("Bedrock comparison requires explicit rock and a concrete foundation")
    local = np.array(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
    foundation_faces = np.sort(concrete[bodies == 3][:, local].reshape(-1, 3), axis=1)
    rock_faces = np.sort(rock[:, local].reshape(-1, 3), axis=1)
    faces, counts = np.unique(np.vstack((foundation_faces, rock_faces)), axis=0, return_counts=True)
    shared = np.intersect1d(np.unique(concrete), np.unique(rock))
    seat = faces[(counts == 2) & np.isin(faces, shared).all(axis=1)]
    # Interior faces with three shared vertices must not become artificial supports.
    foundation_boundary, fc = np.unique(foundation_faces, axis=0, return_counts=True)
    rock_boundary, rc = np.unique(rock_faces, axis=0, return_counts=True)
    boundary_keys = {tuple(row) for row in foundation_boundary[fc == 1]}
    rock_keys = {tuple(row) for row in rock_boundary[rc == 1]}
    seat_keys = boundary_keys & rock_keys
    seat = np.asarray([row for row in seat if tuple(row) in seat_keys], dtype=np.int64)
    if seat.ndim != 2 or not len(seat) or not np.array_equal(np.unique(seat), shared):
        raise ValueError("Only a conforming bonded foundation/rock seat is supported by this diagnostic")
    retained = np.unique(concrete)
    mapping = np.full(len(arrays["nodes"]), -1, dtype=np.int64)
    mapping[retained] = np.arange(len(retained))
    return retained, mapping, seat


def without_bedrock(arrays: dict) -> dict:
    retained, mapping, seat = rigid_seat(arrays)
    result = {key: np.asarray(value).copy() for key, value in arrays.items()}
    result["nodes"] = result["nodes"][retained]
    result["source_node_ids"] = result["source_node_ids"][retained]
    for key in ("tetra4_concrete", "tri3_upstream", "tri3_downstream", "tri3_crest",
                "seg2_wall_foundation_springs", "joint_slave_tri3", "joint_master_tri3"):
        if key not in result:
            continue
        result[key] = mapping[result[key]]
        if np.any(result[key] < 0):
            raise ValueError(f"A retained concrete load or spring references removed rock: {key}")
    for key, width in (("tetra4_bedrock", 4), ("tri3_rock_bottom", 3), ("tri3_rock_sides", 3)):
        result[key] = np.empty((0, width), dtype=np.int64)
    result["support_nodes"] = np.empty(0, dtype=np.int64)
    result["support_matrices_n_per_m"] = np.empty((0, 3, 3))
    result["poi1_rock_bottom_nodes"] = np.empty(0, dtype=np.int64)
    result["fixed_support_nodes"] = np.unique(mapping[seat])
    return result


def write_fields(output: Path, arrays: dict, displacement: np.ndarray, material: dict) -> dict:
    """Common source-order P1 cell stress measure for all four diagnostic views."""
    coordinates = np.asarray(arrays["nodes"])
    cells = np.asarray(arrays["tetra4_concrete"])
    u = np.asarray(displacement)
    if u.shape != coordinates.shape or not np.isfinite(u).all():
        raise ValueError("Diagnostic displacement must be finite and source-node ordered")
    points = coordinates[cells]
    gradients = np.linalg.inv(points[:, 1:] - points[:, :1])
    deformation = np.einsum("nij,njk->nik", gradients, u[cells][:, 1:] - u[cells][:, :1])
    strain = .5 * (deformation + deformation.transpose(0, 2, 1))
    young, poisson = float(material["youngs_modulus"]), float(material["poisson_ratio"])
    mu, lam = young / (2 * (1 + poisson)), young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    stress = 2 * mu * strain + lam * np.trace(strain, axis1=1, axis2=2)[:, None, None] * np.eye(3)
    principal = np.linalg.eigvalsh(stress)
    volumes = np.abs(np.linalg.det(points[:, 1:] - points[:, :1])) / 6
    if not np.isfinite(stress).all() or np.any(volumes <= 0):
        raise ValueError("Diagnostic has invalid concrete stress or volume")
    used = np.unique(cells)
    mapping = np.full(len(coordinates), -1, dtype=int)
    mapping[used] = np.arange(len(used))
    path = output / f"{output.name}.vtu"
    _write_vtu(path, coordinates[used], mapping[cells], u[used], arrays["concrete_body_ids"], principal, stress)
    bodies = {}
    for tag, name in ((1, "wall"), (3, "foundation")):
        mask = arrays["concrete_body_ids"] == tag
        ids = np.unique(cells[mask])
        bodies[name] = {
            "peak_tension_pa": float(principal[mask, 2].max()),
            "peak_compression_pa": float(principal[mask, 0].min()),
            "maximum_displacement_m": float(np.linalg.norm(u[ids], axis=1).max()),
            "elastic_energy_j": float(.5 * np.einsum("nij,nij->n", stress[mask], strain[mask]) @ volumes[mask]),
        }
    report = {"bodies": bodies, "stress_measure": "constant P1 cell principal stress; no nodal recovery",
              "vtu": str(path.resolve()), "engineering_validated": False}
    (output / "fields.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
