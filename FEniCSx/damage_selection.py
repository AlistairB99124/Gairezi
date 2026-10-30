"""Job-bound native element/edge catalogs for painted damage previews.

Selection is not a constitutive model. Validating a preview must never launch
the legacy seeded solver or imply that hydraulic/fracture physics is ready.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from Aster.build_three_body_mesh_arrays import _read_gmsh_ascii
from Elmer.model_config import ConfigurationValidationError, ValidationIssue
from FEniCSx.crack_selection import principal_stresses
from FEniCSx.horizontal_fracture_contract import DamageSelection

DEPTH_FRACTION = 0.75


def read_saved_selection(path: Path, job_id: str, configuration_digest: str) -> dict:
    """Verify an app snapshot without trusting its stored job/config identities."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Saved selection must be an object")

    def digest(value):
        return hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

    content = {key: value for key, value in payload.items() if key != "selection_sha256"}
    if (payload.get("selection_sha256") != digest(content)
            or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1
            or payload.get("job_id") != job_id
            or payload.get("configuration_digest") != configuration_digest
            or payload.get("selection_id") != path.stem
            or payload.get("analysis_kind") != "painted_tensile_loss"
            or payload.get("status") != "saved_selection_only"
            or payload.get("solver_ready") is not False
            or not isinstance(payload.get("request"), dict)):
        raise ValueError("Saved selection integrity or job identity check failed")
    return payload


def build_catalog(source: Path, fields: Path, config: dict, config_bytes: bytes) -> dict:
    """Preserve native tetrahedron rows and source-node identities, including splits."""
    source_bytes = source.read_bytes()
    field_bytes = fields.read_bytes()
    points, cells, tags, _, _ = _read_gmsh_ascii(source)
    if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(source_bytes).digest():
        raise ValueError("Source mesh changed during damage catalog preparation")
    with np.load(fields, allow_pickle=False) as archive:
        baseline_points = archive["coordinates_m"]
        stress = archive["stress_pa"]
        regions = archive["node_regions"]
    if fields.read_bytes() != field_bytes:
        raise ValueError("Baseline fields changed during damage catalog preparation")
    wall = np.flatnonzero(tags == 1)
    if not len(wall):
        raise ValueError("Damage selection requires native wall tetrahedra")
    wall_nodes = np.unique(cells[wall])
    if (not np.isfinite(points[wall_nodes]).all() or baseline_points.ndim != 2
            or baseline_points.shape[1] != 3 or stress.shape != (len(baseline_points), 6)
            or regions.shape != (len(baseline_points),)
            or not np.isfinite(baseline_points).all() or not np.isfinite(stress).all()):
        raise ValueError("Damage selection requires finite native geometry and baseline stress")
    members = np.flatnonzero((regions & 1) != 0)
    principal = principal_stresses(stress[members])[:, -1]
    # Coincident wall nodes can have distinct identities/stresses. Display the
    # maximum recovered wall tension at that coordinate, without welding IDs.
    # Adding 0.0 merges -0.0 with 0.0, matching coordinate-tuple equality.
    member_keys = np.round(baseline_points[members], 8) + 0.0
    wall_keys = np.round(points[wall_nodes], 8) + 0.0
    _, groups = np.unique(np.concatenate((member_keys, wall_keys)), axis=0, return_inverse=True)
    groups = groups.reshape(-1)
    group_peak = np.full(groups.max() + 1, -np.inf)
    np.maximum.at(group_peak, groups[:len(members)], principal)
    matched = np.zeros(len(group_peak), dtype=bool)
    matched[groups[:len(members)]] = True
    wall_groups = groups[len(members):]
    if not matched[wall_groups].all():
        node = int(wall_nodes[np.flatnonzero(~matched[wall_groups])[0]])
        raise ValueError(f"Native wall node {node} has no matching Elmer baseline stress")
    nodal_peak = np.full(len(points), np.nan)
    nodal_peak[wall_nodes] = group_peak[wall_groups]

    wall_config = config["geometry"]["wall"]
    radius = float(wall_config["centerline_radius_m"])
    upstream = float(wall_config["upstream_radius_m"])
    if not np.isfinite([radius, upstream]).all() or radius <= 0 or upstream <= 0:
        raise ValueError("Damage selection requires positive finite wall radii")
    projected = np.column_stack((
        radius * np.cos(np.arctan2(points[:, 0], points[:, 1])), points[:, 2],
    ))
    steps = sorted(wall_config.get("thickness_steps") or (),
                   key=lambda item: item["top_z_m"], reverse=True)
    base_downstream = float(wall_config["downstream_radius_m"])

    def downstream_at(z: np.ndarray) -> np.ndarray:
        value = np.full(np.shape(z), base_downstream)
        for step in steps:
            value = np.where(z < step["top_z_m"], upstream - float(step["radial_width_m"]), value)
        return value

    # Use the same physical wall-course bins as the baseline map for automatic
    # face inference, not a global maximum that overrides every painted region.
    from FEniCSx.crack_selection import Y_BIN_M, course_spacing
    spacing = course_spacing(config)
    planar_radius = np.sqrt(points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1])
    distance_up = np.abs(upstream - planar_radius)
    distance_down = np.abs(planar_radius - downstream_at(points[:, 2]))
    # Face bits: 1 = equidistant/ambiguous, 2 = upstream, 4 = downstream.
    node_face_bit = np.where(np.abs(distance_up - distance_down) <= 1e-8, 1,
                             np.where(distance_up < distance_down, 2, 4)).astype(np.int8)
    node_bins = np.column_stack((
        np.floor(projected[wall_nodes, 0] / Y_BIN_M),
        np.floor(projected[wall_nodes, 1] / spacing),
    )).astype(np.int64)
    element_nodes = cells[wall]
    yz = projected[element_nodes].mean(axis=1)
    element_bins = np.column_stack((np.floor(yz[:, 0] / Y_BIN_M),
                                    np.floor(yz[:, 1] / spacing))).astype(np.int64)
    _, bin_ids = np.unique(np.concatenate((node_bins, element_bins)), axis=0, return_inverse=True)
    bin_ids = bin_ids.reshape(-1)
    node_bin, element_bin = bin_ids[:len(wall_nodes)], bin_ids[len(wall_nodes):]
    bin_peak = np.full(bin_ids.max() + 1, -np.inf)
    np.maximum.at(bin_peak, node_bin, nodal_peak[wall_nodes])
    bin_faces = np.zeros(len(bin_peak), dtype=np.int8)
    at_peak = nodal_peak[wall_nodes] == bin_peak[node_bin]
    np.bitwise_or.at(bin_faces, node_bin[at_peak], node_face_bit[wall_nodes[at_peak]])

    xyz = points[element_nodes]
    z_low = xyz[:, :, 2].min(axis=1)
    z_high = xyz[:, :, 2].max(axis=1)
    local_downstream = [downstream_at(z_low), downstream_at(z_high)]
    for step in steps:
        z = float(step["top_z_m"])
        inside = (z_low <= z) & (z <= z_high)
        for level in (z, np.nextafter(z, -np.inf)):
            value = downstream_at(np.float64(level))
            local_downstream.append(np.where(inside, value, np.nan))
    local_downstream = np.column_stack(local_downstream)
    if np.any(np.nanmin(local_downstream, axis=1) <= 0) or np.any(np.nanmax(local_downstream, axis=1) >= upstream):
        raise ValueError("Damage selection encountered invalid local wall thickness")
    depth = DEPTH_FRACTION * (upstream - local_downstream)
    centroid = xyz.mean(axis=1)
    planar_centroid = centroid[:, :2]
    direction = planar_centroid / np.sqrt(np.einsum("ij,ij->i", planar_centroid, planar_centroid))[:, None]
    # A supporting radial plane gives a conservative lower radius bound for
    # the entire linear tetrahedron, not just its vertices.
    lower_radius = np.einsum("eij,ej->ei", xyz[:, :, :2], direction).min(axis=1)
    upper_radius = np.linalg.norm(xyz[:, :, :2], axis=2).max(axis=1)
    up_ok = lower_radius >= np.nanmax(upstream - depth, axis=1) - 1e-9
    down_ok = upper_radius <= np.nanmin(local_downstream + depth, axis=1) + 1e-9
    element_peak = nodal_peak[element_nodes]
    own_peak = element_peak.max(axis=1)
    own_faces = np.bitwise_or.reduce(
        np.where(element_peak == own_peak[:, None], node_face_bit[element_nodes], 0), axis=1)
    has_nodes = np.zeros(len(bin_peak), dtype=bool)
    has_nodes[node_bin] = True
    face_bits = np.where(has_nodes[element_bin], bin_faces[element_bin], own_faces)
    face_names = {1: None, 2: "upstream", 4: "downstream"}
    stress_mpa = own_peak / 1e6
    elements = [{
        "id": cell_id, "node_ids": nodes,
        "face": face_names.get(bits), "upstream_eligible": up,
        "downstream_eligible": down, "stress_mpa": peak,
    } for cell_id, nodes, bits, up, down, peak in zip(
        wall.tolist(), element_nodes.tolist(), face_bits.tolist(), up_ok.tolist(),
        down_ok.tolist(), stress_mpa.tolist())]

    first, second = np.triu_indices(4, 1)
    pairs = np.sort(np.stack((element_nodes[:, first], element_nodes[:, second]), axis=2), axis=2).reshape(-1, 2)
    delta = projected[pairs[:, 0]] - projected[pairs[:, 1]]
    pairs = np.unique(pairs[np.sqrt(np.einsum("ij,ij->i", delta, delta)) > 1e-8], axis=0)
    payload = {
        "schema_version": 1,
        "source_mesh_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "baseline_fields_sha256": hashlib.sha256(field_bytes).hexdigest(),
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "depth_fraction": DEPTH_FRACTION,
        "nodes": [{"id": node, "y_m": y, "z_m": z} for node, (y, z) in zip(
            wall_nodes.tolist(), projected[wall_nodes].tolist())],
        "elements": elements,
        "edges": [{"id": index, "node_ids": pair} for index, pair in enumerate(pairs.tolist())],
        "solver_ready": False,
        "limitations": [
            "Selection preview only; no tensile-strength change or fracture solve is performed.",
            "Projected tetrahedra overlap through the wall depth; painting includes all eligible hits.",
            "Only whole native linear tetrahedra conservatively contained within 75% depth are eligible.",
        ],
    }
    # Identity excludes display-only floats (projected y and stress shading),
    # whose last bits differ between numpy builds on Windows and WSL. Topology,
    # face and eligibility, which the solver binds to, remain fully covered.
    identity = {
        "catalog_identity_version": 2,
        **{name: value for name, value in payload.items() if name not in ("nodes", "elements")},
        "node_ids": wall_nodes.tolist(),
        "elements": [[element["id"], element["node_ids"], element["face"], element["upstream_eligible"],
                      element["downstream_eligible"]] for element in elements],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    return {**payload, "catalog_sha256": digest}


def validate_selection(catalog: dict, request: object) -> dict:
    """Reject stale, foreign or over-depth selections without scheduling a solver."""
    if not isinstance(request, dict):
        raise ValueError("Damage selection must be an object")
    keys = {"catalog_sha256", "affected_element_ids", "crack_start_edge_ids",
            "face_overrides", "opening_limit_m"}
    if set(request) != keys:
        raise ValueError("Damage selection requires catalog hash, element IDs, edge IDs, face overrides and opening limit")
    if request["catalog_sha256"] != catalog["catalog_sha256"]:
        raise ValueError("Damage selection belongs to a different or stale mesh catalog; reload the preview")
    selected = DamageSelection.from_dict({name: request[name] for name in (
        "affected_element_ids", "crack_start_edge_ids",
    )})
    from FEniCSx.horizontal_fracture_contract import _number
    limit = _number(request["opening_limit_m"], "opening_limit_m", positive=True)
    overrides = request["face_overrides"]
    if not isinstance(overrides, dict) or any(
        not isinstance(key, str) or value not in ("upstream", "downstream")
        for key, value in overrides.items()
    ):
        raise ValueError("face_overrides must map element ID strings to upstream or downstream")
    elements = {item["id"]: item for item in catalog["elements"]}
    edges = {item["id"] for item in catalog["edges"]}
    issues = []
    faces = {}
    for element_id in selected.affected_element_ids:
        element = elements.get(element_id)
        path = f"affected_element_ids[{element_id}]"
        if element is None:
            issues.append(ValidationIssue(path, "is not a wall element in this job's mesh"))
            continue
        face = element["face"] or overrides.get(str(element_id))
        if face is None:
            issues.append(ValidationIssue(path, "baseline peak is equally close to both faces; choose a face"))
        elif not element[f"{face}_eligible"]:
            issues.append(ValidationIssue(path, "is not wholly contained in the selected face's 75% depth"))
        else:
            faces[str(element_id)] = face
    allowed_overrides = {str(item) for item in selected.affected_element_ids
                         if item in elements and elements[item]["face"] is None}
    if set(overrides) - allowed_overrides:
        issues.append(ValidationIssue("face_overrides", "only selected elements with ambiguous faces may be overridden"))
    for edge_id in selected.crack_start_edge_ids:
        if edge_id not in edges:
            issues.append(ValidationIssue("crack_start_edge_ids", f"edge {edge_id} is not in this job's mesh"))
    if issues:
        raise ConfigurationValidationError(issues)
    return {
        "status": "validated_selection_only", "solver_ready": False,
        "catalog_sha256": catalog["catalog_sha256"],
        **selected.to_dict(), "resolved_faces": faces, "opening_limit_m": limit,
        "message": "Selection is valid. Coupled tensile-loss/fracture/hydraulic solver integration is not yet available; no solve was started.",
    }


def bind_source_selection(source: Path, fields: Path, configuration: Path, structure, request: object) -> dict:
    """Rebuild the catalog and prove source rows describe this structure.

    Only unused nodes may have been compacted. Coincident identities, cell
    ordering and regions must match exactly; this is not baseline verification.
    """
    config_bytes = configuration.read_bytes()
    config = json.loads(config_bytes)
    if structure.three_body_config != config:
        raise ValueError("Selection configuration differs from the imported structure")
    catalog = build_catalog(source, fields, config, config_bytes)
    validation = validate_selection(catalog, request)
    points, cells, regions, _, _ = _read_gmsh_ascii(source)
    used = np.unique(cells)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    kinds = [kind for kind, _ in structure.cells]
    topology = np.asarray([nodes for _, nodes in structure.cells])
    if (any(kind != 4 for kind in kinds)
            or not np.array_equal(np.asarray(structure.nodes), points[used])
            or not np.array_equal(topology, remap[cells])
            or not np.array_equal(np.asarray(structure.region_ids), regions)):
        raise ValueError("Selection source row topology/coordinates/regions differ from the imported structure")
    for path, digest in (
        (source, catalog["source_mesh_sha256"]),
        (fields, catalog["baseline_fields_sha256"]),
        (configuration, catalog["configuration_sha256"]),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Selection inputs changed during source binding")
    return {
        "status": "bound_selection_only", "solver_ready": False,
        "catalog_sha256": catalog["catalog_sha256"],
        "source_mesh_sha256": catalog["source_mesh_sha256"],
        "baseline_fields_sha256": catalog["baseline_fields_sha256"],
        "configuration_sha256": catalog["configuration_sha256"],
        "affected_element_ids": validation["affected_element_ids"],
        "crack_start_edge_ids": validation["crack_start_edge_ids"],
        "marker_source_edges": [
            remap[catalog["edges"][edge_id]["node_ids"]].tolist()
            for edge_id in validation["crack_start_edge_ids"]
        ],
        "resolved_faces": validation["resolved_faces"],
        "opening_limit_m": validation["opening_limit_m"],
        "edge_role": "retained marker only; no initiation/growth law implemented",
    }
