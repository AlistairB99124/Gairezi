"""Exact saved Elmer mesh import and piecewise three-body material/load data."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from dolfinx import fem
import meshio
from mpi4py import MPI
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Elmer.model_config import foundation_support

# Wall (1) may crack; bedrock (2) and optional concrete foundation (3) stay intact.
DAMAGE_REGION_IDS = (1,)


def is_three_body(structure):
    return getattr(structure, "three_body_config", None) is not None


def physical_configuration(config):
    """The configuration without fracture_debug, which only tunes the cracking solvers.

    A stage rerun may change fracture_debug while reusing the meshes and the elastic
    baseline built from the originally submitted configuration.
    """
    return {key: value for key, value in config.items() if key != "fracture_debug"}


def require_fracture_resolution(options):
    from crack_field import MINIMUM_ELEMENTS_PER_LENGTH_SCALE

    if not options.minimum_elements_per_length_scale >= MINIMUM_ELEMENTS_PER_LENGTH_SCALE:
        raise ValueError(
            "Three-body fracture requires at least "
            f"{MINIMUM_ELEMENTS_PER_LENGTH_SCALE:g} elements per length scale; "
            f"configured target is {options.minimum_elements_per_length_scale:g}. "
            "Under-resolved research targets cannot be used for canonical dam jobs."
        )


def source_wall_joints(config, diagnostic):
    """Return the wall connection and its spring joints, checked against the submitted config.

    Thickness blocks stay body 1; they are separated only by duplicated nodes on
    the step levels and joined by linear springs (tags from the Elmer diagnostic).
    """
    connection = diagnostic.get("wall_connection", "single_monolithic_body")
    joints = diagnostic.get("wall_joints", [])
    wall = config["geometry"]["wall"]
    support = wall.get("joint_support", {"model": "bonded"})
    levels = sorted((float(step["top_z_m"]) for step in wall.get("thickness_steps", [])), reverse=True)
    if connection == "single_monolithic_body":
        if joints or (support["model"] == "spring" and levels):
            raise ValueError("Source wall is monolithic but the configuration declares spring wall joints")
        return connection, []
    if connection != "spring_jointed_blocks" or support["model"] != "spring":
        raise ValueError(f"Unsupported source wall connection {connection!r}")
    if [float(joint["z_m"]) for joint in joints] != levels or any(
        joint["spring_n_per_m3"] != support["spring_n_per_m3"] or len(joint["tags"]) != 2
        for joint in joints
    ):
        raise ValueError("Source wall joints differ from geometry.wall.joint_support/thickness_steps")
    return connection, [
        {"name": joint["name"], "z_m": float(joint["z_m"]),
         "tags": tuple(int(tag) for tag in joint["tags"]), "spring_n_per_m3": joint["spring_n_per_m3"]}
        for joint in joints
    ]


def source_foundation(config, diagnostic, wall_joints):
    """Validate the optional foundation and semantic interface pairs against the snapshot."""
    foundation = config["geometry"].get("foundation")
    metadata = diagnostic.get("foundation")
    if foundation is None:
        if metadata not in (None, {"present": False}) or diagnostic.get("solver_body_ids") != {"wall": 1, "bedrock": 2}:
            raise ValueError("Source foundation differs from the submitted configuration")
        return None, None
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in foundation.items()):
        raise ValueError("Source foundation dimensions differ from geometry.foundation")
    if diagnostic.get("solver_body_ids") != {"wall": 1, "bedrock": 2, "foundation": 3}:
        raise ValueError("Source foundation body tags differ from the submitted configuration")
    geometry = config["geometry"]
    support = foundation_support(config)
    bonded = support["model"] == "bonded"
    bedrock_bonded = geometry["bedrock"]["support"]["model"] == "bonded"
    bedrock_connection = "bonded" if bedrock_bonded else "spring"
    if (diagnostic.get("wall_bedrock_connection") != "not_in_contact"
                or diagnostic.get("wall_foundation_connection") != support["model"]
                or diagnostic.get("foundation_bedrock_connection") != bedrock_connection):
                raise ValueError("Source foundation connections differ from the submitted configuration")
    expected = {
                **({} if bonded else {"wall_foundation": [17, 18]}),
                **({} if bedrock_bonded else {"foundation_bedrock": [11, 12]}),
                **{joint["name"]: list(joint["tags"]) for joint in wall_joints},
    }
    declared = diagnostic.get("spring_pairs")
    if not isinstance(declared, dict) or set(declared) != set(expected):
                raise ValueError("Source foundation spring pairs differ from the configured interfaces")
    pairs = [
                *([] if bonded else [{
                    "name": "wall_foundation", "tags": (17, 18),
                    "spring_n_per_m3": support["spring_n_per_m3"],
                }]),
                *([] if bedrock_bonded else [{
                    "name": "foundation_bedrock", "tags": (11, 12),
                    "spring_n_per_m3": geometry["bedrock"]["support"]["spring_n_per_m3"],
                }]),
                *wall_joints,
    ]
    if diagnostic.get("interface_stiffness_n_per_m3") != {
                pair["name"]: pair["spring_n_per_m3"]
                for pair in pairs
    }:
                raise ValueError("Source foundation spring stiffness differs from the submitted configuration")
    expanded, used_tags = [], set()
    for pair in pairs:
        name = pair["name"]
        declaration = declared[name]
        if not isinstance(declaration, dict):
            if declaration != expected[name]:
                raise ValueError(f"Source {name} spring tags differ from the configured interface")
            parts = [(None, declaration, None)]
        else:
            faces, planes = declaration.get("faces"), declaration.get("planes")
            if (name not in ("wall_foundation", "foundation_bedrock")
                    or declaration.get("grouping") not in ("exact_plane", "near_planar")
                    or declaration.get("legacy_tags") != expected[name]
                    or not isinstance(faces, dict) or not faces or not isinstance(planes, dict)
                    or set(faces) != set(planes) or list(faces.values())[0] != expected[name]):
                raise ValueError(f"Invalid exact-plane source spring group {name}")
            if declaration["grouping"] == "near_planar":
                if (not finite_positive(declaration.get("tolerance_m"))
                        or not finite_positive(declaration.get("max_angle_deg"))
                        or declaration["max_angle_deg"] >= 90
                        or declaration.get("continuity") != "edge_connected; single orientation class"):
                    raise ValueError(f"Invalid near-planar grouping limits for {name}")
            parts = [(part, tags, planes[part]) for part, tags in faces.items()]
        for part, tags, plane in parts:
            if (not isinstance(tags, (list, tuple)) or len(tags) != 2
                    or any(type(tag) is not int or tag <= 0 or tag in used_tags for tag in tags)
                    or tags[0] == tags[1]
                    or (part is not None and tags != expected[name] and min(tags) < 1000)):
                raise ValueError(f"Invalid or reused source spring tags for {name}")
            used_tags.update(tags)
            item = dict(pair, name=f"{name}/{part}" if part is not None else name, tags=tuple(tags))
            if part is not None:
                if not isinstance(part, str) or not part or "/" in part or not isinstance(plane, dict):
                    raise ValueError(f"Invalid source spring plane for {name}")
                normal = np.asarray(plane.get("unit_normal"), dtype=float)
                offset = plane.get("offset_m")
                if (normal.shape != (3,) or not np.all(np.isfinite(normal))
                        or not np.isclose(np.linalg.norm(normal), 1., rtol=0, atol=1e-7)
                        or isinstance(offset, bool) or not isinstance(offset, (float, int))
                        or not np.isfinite(offset) or type(plane.get("triangles_per_side")) is not int
                        or plane["triangles_per_side"] <= 0):
                    raise ValueError(f"Invalid source spring plane metadata for {name}/{part}")
                item["plane"] = plane
                item["grouping"] = declaration["grouping"]
                if declaration["grouping"] == "near_planar":
                    item.update({key: declaration[key] for key in ("tolerance_m", "max_angle_deg", "continuity")})
                    if (plane.get("orientation_class") not in ("tread", "chamfer", "side_wall", "riser")
                            or any(isinstance(plane.get(key), bool) or not isinstance(plane.get(key), (int, float))
                                   or not np.isfinite(plane[key]) or not 0 <= plane[key] <= limit
                                   for key, limit in (("maximum_off_plane_distance_m", item["tolerance_m"]),
                                                      ("normal_spread_deg", item["max_angle_deg"] + 1e-6)))):
                        raise ValueError(f"Invalid near-planar metrics for {name}/{part}")
            expanded.append(item)
    if bedrock_bonded:
        expanded.append({
            "name": "foundation_bedrock", "tags": (),
            "spring_n_per_m3": None, "connection": "bonded",
        })
    return metadata, expanded


def finite_positive(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and np.isfinite(value) and value > 0)


def near_planar_geometry(xyz):
    """Recompute the source fit, pairwise normal spread and orientation classes."""
    import sys
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from Elmer.mortar_interfaces import fitted_plane, orientation_class

    # Elmer groups fit unique coordinate keys rounded to 8 decimals, not weighted triangle vertices.
    if not np.all(np.isfinite(xyz)):
        raise ValueError("Nonfinite near-planar spring faces")
    points = np.unique(np.round(xyz.reshape(-1, 3), 8), axis=0)
    normal, offset, distance = fitted_plane(points)
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if not np.all(np.isfinite(xyz)) or np.any(lengths <= 1e-14):
        raise ValueError("Nonfinite or degenerate near-planar spring faces")
    normals /= lengths[:, None]
    # Bound memory for large groups while retaining the full pairwise-angle check.
    cosine = min(np.abs(normals[start:start+256] @ normals.T).min()
                 for start in range(0, len(normals), 256))
    spread = float(np.degrees(np.arccos(np.clip(cosine, -1., 1.))))
    classes = {orientation_class(face) for face in xyz}
    edges = {}
    adjacency = [set() for _ in xyz]
    for index, face in enumerate(xyz):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((tuple(np.round(face[a], 8)), tuple(np.round(face[b], 8)))))
            for other in edges.setdefault(edge, []):
                adjacency[index].add(other)
                adjacency[other].add(index)
            edges[edge].append(index)
    reached, pending = set(), [0]
    while pending:
        index = pending.pop()
        if index not in reached:
            reached.add(index)
            pending.extend(adjacency[index] - reached)
    if len(reached) != len(xyz):
        raise ValueError("Near-planar spring group must be edge connected")
    return normal, offset, distance, spread, classes


def validate_near_planar_faces(xyz, pair):
    normal, offset, distance, spread, classes = near_planar_geometry(xyz)
    plane = pair["plane"]
    actual_distance = float(np.max(np.abs(xyz @ normal - offset)))
    centre = xyz.reshape(-1, 3).mean(axis=0)
    # Compare offsets locally: tiny normal rounding errors amplify at a remote global origin.
    declared_centre_offset = centre @ np.asarray(plane["unit_normal"]) - plane["offset_m"]
    if (classes != {plane["orientation_class"]} or actual_distance > pair["tolerance_m"] + 1e-8
            or spread > pair["max_angle_deg"] + 1e-6
            or not np.allclose(normal, plane["unit_normal"], rtol=0, atol=1e-7)
            or not np.isclose(declared_centre_offset, centre @ normal - offset, rtol=0, atol=1e-8)
            or not np.isclose(distance, plane["maximum_off_plane_distance_m"], rtol=0, atol=1e-8)
            or not np.isclose(spread, plane["normal_spread_deg"], rtol=0, atol=1e-5)):
        raise ValueError(f"Source {pair['name']} near-planar geometry/metrics differ from declaration")


def validate_foundation_faces(nodes, boundaries, body_nodes, pairs):
    """Check source plane tags, body membership and geometry before any refinement."""
    nodes = np.asarray(nodes)
    by_tag = {}
    for tag, face in boundaries:
        by_tag.setdefault(tag, []).append(face)
    for pair in pairs:
        if pair.get("spring_n_per_m3") is None:
            continue
        logical = pair["name"].split("/")[0]
        regions = {"wall_foundation": (1, 3), "foundation_bedrock": (3, 2)}.get(logical, (1, 1))
        reference = None
        for tag, region in zip(pair["tags"], regions, strict=True):
            faces = by_tag.get(tag, [])
            if not faces or any(not set(face).issubset(body_nodes[region]) for face in faces):
                raise ValueError(f"Source {pair['name']} spring face body/tags mismatch")
            plane = pair.get("plane")
            if plane is None:
                continue
            if len(faces) != plane["triangles_per_side"] or any(len(face) != 3 for face in faces):
                raise ValueError(f"Source {pair['name']} plane triangle count mismatch")
            xyz = nodes[np.asarray(faces)]
            if pair.get("grouping") == "near_planar":
                validate_near_planar_faces(xyz, pair)
                locations = {tuple(sorted(tuple(p) for p in np.round(face, 8))) for face in xyz}
                if len(locations) != len(faces) or (reference is not None and reference != locations):
                    raise ValueError(f"Source {pair['name']} near-planar sides are not coincident")
                reference = locations
                continue
            if reference is None:
                origin = xyz[0, 0]
                normal = np.cross(xyz[0, 1] - origin, xyz[0, 2] - origin)
                normal /= np.linalg.norm(normal)
                if normal[np.argmax(np.abs(normal))] < 0:
                    normal *= -1
                reference = origin, normal
                if (not np.allclose(np.round(normal, 7), plane["unit_normal"], rtol=0, atol=1e-7)
                        or not np.isclose(round(float(normal @ origin), 7), plane["offset_m"], rtol=0, atol=1e-7)):
                    raise ValueError(f"Source {pair['name']} plane metadata differs from mesh")
            origin, normal = reference
            if not np.all(np.isfinite(xyz)) or np.max(np.abs((xyz - origin) @ normal)) > 1e-7:
                raise ValueError(f"Source {pair['name']} includes off-plane spring faces")


def connection_report(structure):
    foundation = getattr(structure, "foundation", None)
    if foundation is not None:
        return {"foundation": foundation, "wall_bedrock_connection": "not_in_contact",
                "wall_foundation_connection": foundation_support(structure.three_body_config)["model"],
                "foundation_bedrock_connection": (
                    "bonded" if structure.three_body_config["geometry"]["bedrock"]["support"]["model"] == "bonded"
                    else "spring")}
    return {"wall_bedrock_connection": (
        "bonded" if structure.three_body_config["geometry"]["bedrock"]["support"].get("model") == "bonded"
        else "spring")}


def validate_bonded_interface(cells, regions, first_region, second_region, shared_nodes,
                              expected_faces, name):
    """Require a complete, conforming tetrahedral interface with shared displacement nodes."""
    patterns = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    exposed = []
    for region in (first_region, second_region):
        counts = {}
        for (_, cell), tag in zip(cells, regions, strict=True):
            if tag != region:
                continue
            for pattern in patterns:
                face = tuple(sorted(cell[index] for index in pattern))
                counts[face] = counts.get(face, 0) + 1
        exposed.append({face for face, count in counts.items() if count == 1})
    interface = exposed[0] & exposed[1]
    interface_nodes = {node for face in interface for node in face}
    if (not interface or interface_nodes != shared_nodes
            or len(interface) != expected_faces):
        raise ValueError(
            f"Bonded {name} interface must share all {expected_faces} boundary faces and nodes; "
            f"found {len(interface)} faces and {len(interface_nodes)} interface nodes"
        )


def source_wall_bedrock_seat(config, diagnostic):
    """Return whether the wall is bonded directly to bedrock, checking the source agrees.

    With a foundation, bedrock.support governs the foundation/bedrock interface
    (checked in source_foundation); the wall never touches the bedrock.
    """
    if config["geometry"].get("foundation") is not None:
        return False
    bonded_seat = config["geometry"]["bedrock"]["support"]["model"] == "bonded"
    if (diagnostic.get("wall_bedrock_connection", "spring") == "bonded") != bonded_seat:
        raise ValueError("Source wall/bedrock connection differs from the submitted configuration")
    return bonded_seat


def load_structure(config_path: Path, model: Path, structure):
    config = json.loads(config_path.read_text())
    if physical_configuration(json.loads((model / "config.json").read_text())) != physical_configuration(config):
        raise ValueError("Three-body source mesh configuration differs from the submitted snapshot")
    diagnostic = json.loads((model / "diagnostic.json").read_text())
    if diagnostic["joints"] != "spring":
        raise ValueError("Expected canonical spring-jointed three-body source")
    bonded_seat = source_wall_bedrock_seat(config, diagnostic)
    wall_connection, wall_joints = source_wall_joints(config, diagnostic)
    foundation, pairs = source_foundation(config, diagnostic, wall_joints)
    body_ids = (1, 2, 3) if foundation is not None else (1, 2)
    mesh_path = model / "three_body.msh"
    source = meshio.read(mesh_path)
    cells, regions, boundaries = [], [], []
    inverse_tags = {value: int(key) for key, value in diagnostic["boundary_id_map"].items()}
    for block, tags in zip(source.cells, source.cell_data["gmsh:physical"], strict=True):
        if block.type == "tetra":
            cells.extend((4, tuple(int(n) for n in cell)) for cell in block.data)
            regions.extend(int(tag) for tag in tags)
        elif block.type == "triangle":
            boundaries.extend((inverse_tags[int(tag)], tuple(int(n) for n in cell))
                              for cell, tag in zip(block.data, tags, strict=True))
        else:
            raise ValueError(f"Unsupported three-body source cell type {block.type}")
    if set(regions) != set(body_ids) or len(source.points) != diagnostic["nodes"]:
        raise ValueError("Canonical source body tags or node count do not match diagnostic")
    body_nodes = {
        region: {node for (_, cell), tag in zip(cells, regions, strict=True)
                 if tag == region for node in cell}
        for region in body_ids
    }
    shared_seat = body_nodes[1] & body_nodes[2]
    if bool(shared_seat) != bonded_seat:
        raise ValueError("Wall/bedrock mesh connectivity does not implement the configured connection")
    if bonded_seat:
        validate_bonded_interface(
            cells, regions, 1, 2, shared_seat,
            diagnostic["wall_rock_seat_triangles"], "wall/bedrock",
        )
    if foundation is not None:
        bonded_foundation = foundation_support(config)["model"] == "bonded"
        shared_wall_foundation = body_nodes[3] & body_nodes[1]
        if bool(shared_wall_foundation) != bonded_foundation:
            raise ValueError("Wall/foundation mesh connectivity does not implement the configured connection")
        if bonded_foundation:
            validate_bonded_interface(
                cells, regions, 1, 3, shared_wall_foundation,
                foundation["wall_seat_node_snap"]["matched_faces"], "wall/foundation",
            )
        bonded_foundation_bedrock = config["geometry"]["bedrock"]["support"]["model"] == "bonded"
        shared_foundation_bedrock = body_nodes[3] & body_nodes[2]
        if bool(shared_foundation_bedrock) != bonded_foundation_bedrock:
            raise ValueError("Foundation/bedrock mesh connectivity does not implement the configured connection")
        if bonded_foundation_bedrock:
            validate_bonded_interface(
                cells, regions, 3, 2, shared_foundation_bedrock,
                diagnostic["wall_rock_seat_triangles"], "foundation/bedrock",
            )
        validate_foundation_faces(source.points, boundaries, body_nodes, pairs)
    if __package__:
        from .dam_mesh import compact_unused_nodes
    else:
        from dam_mesh import compact_unused_nodes
    result = compact_unused_nodes(replace(
        structure, nodes=[tuple(p) for p in source.points], cells=cells,
        region_ids=regions, boundaries=boundaries,
    ))
    result.three_body_config = config
    result.wall_connection = wall_connection
    result.wall_joints = wall_joints
    result.foundation = foundation
    result.spring_pairs = pairs
    joint_tags = {
        tag for joint in (pairs or wall_joints)
        if joint["spring_n_per_m3"] is not None for tag in joint["tags"]
    }
    if joint_tags - {tag for tag, _ in result.boundaries}:
        raise ValueError("Source mesh lacks the declared wall-joint spring faces")
    result.rock_support = diagnostic.get("rock_outer_support", {"model": "fixed_bottom"})
    configured = config["material"]["bedrock"].get("support", {"model": "fixed_bottom"})
    if result.rock_support["model"] != configured["model"]:
        raise ValueError("Source outer rock support differs from material.bedrock.support")
    if configured["model"] == "rollers":
        springs = configured["spring_n_per_m3"]
        faces = result.rock_support["faces"]
        if sorted(int(tag) for tag in faces) != list(ROLLER_FACES) or any(
            face["normal_stiffness_n_per_m3"] != springs[face["local_axis"]] for face in faces.values()
        ):
            raise ValueError("Source rock roller faces/stiffness differ from material.bedrock.support")
    result.submitted_configuration_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    result.three_body_source = {
        "model_directory": str(model.resolve()),
        "unused_source_nodes_removed": int(len(source.points) - len(result.nodes)),
        "node_compaction": "Unused nodes only; seat connectivity follows the configured connection",
        **connection_report(result),
        "wall_connection": wall_connection,
        "wall_joints": wall_joints,
        "sha256": {name: hashlib.sha256((model / name).read_bytes()).hexdigest()
                   for name in ("three_body.msh", "config.json", "diagnostic.json")},
    }
    concrete = np.asarray([cell for (_, cell), tag in zip(result.cells, regions, strict=True) if tag == 1])
    result.concrete_minimum_z = float(np.asarray(result.nodes)[np.unique(concrete), 2].min())
    return result


def require_verified_baseline(structure, directory):
    """Reject missing, failed or stale elastic evidence before evolving damage."""
    if directory is None:
        raise ValueError("Three-body fracture requires --verified-baseline from a passed elastic comparison")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    report = json.loads((directory / "baseline_report.json").read_text())
    source = json.loads((directory / "three_body_source.json").read_text())
    if (manifest.get("status") != "completed_not_engineering_validated"
            or manifest.get("scenario_name") != "Baseline_NoPhase"):
        raise ValueError("Three-body baseline has not completed successfully")
    # The baseline's own config snapshot must be the one its evidence was hashed from and be
    # physically identical to this job's configuration (fracture_debug may differ on a rerun).
    baseline_bytes = (directory / "config.json").read_bytes()
    baseline_sha256 = hashlib.sha256(baseline_bytes).hexdigest()
    if (any(document.get("configuration_sha256") != baseline_sha256 for document in (manifest, report))
            or physical_configuration(json.loads(baseline_bytes)) != physical_configuration(structure.three_body_config)):
        raise ValueError("Verified baseline configuration does not match this fracture job")
    if source.get("sha256") != structure.three_body_source["sha256"]:
        raise ValueError("Verified baseline source mesh/configuration/diagnostic hashes have changed")
    gate = report.get("three_body_verification", {})
    error = gate.get("relative_l2_displacement_difference", np.inf)
    stress_error = gate.get("relative_nodal_stress_difference", np.inf)
    balance = report.get("relative_force_balance_error", np.inf)
    if (gate.get("passed") is not True
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not np.isfinite(value) for value in (error, stress_error, balance))
            or not 0 <= error <= 1e-4 or not 0 <= stress_error <= 1e-4
            or report.get("stress_recovery", {}).get("peak_measure") != "recovered_nodal"
            or not 0 <= balance <= 1e-6):
        raise ValueError("Three-body elastic parity/equilibrium gate did not pass")
    return {"directory": str(directory.resolve()), "configuration_sha256": baseline_sha256,
            "source_sha256": source["sha256"], "verification": gate}


def materials(imported, structure):
    """DG0 coefficients preserve discontinuous concrete/rock constitutive data."""
    config = structure.three_body_config
    space = fem.functionspace(imported.domain, ("DG", 0))
    fields = {}
    for key in ("density", "youngs_modulus", "poisson_ratio"):
        field = fem.Function(space, name=key)
        for tag, name in ((1, "concrete"), (2, "bedrock"), (3, "concrete")):
            cells = imported.cell_tags.find(tag)
            field.x.array[space.dofmap.list[cells, 0]] = config["material"][name][key]
        field.x.scatter_forward()
        fields[key] = field
    young, poisson = fields["youngs_modulus"], fields["poisson_ratio"]
    fields["mu"] = young / (2 * (1 + poisson))
    fields["lam"] = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    fields["bulk"] = fields["lam"] + 2 * fields["mu"] / 3
    concrete = fem.Function(space, name="concrete_damage_mask")
    for tag in DAMAGE_REGION_IDS:
        concrete.x.array[space.dofmap.list[imported.cell_tags.find(tag), 0]] = 1.
    concrete.x.scatter_forward()
    fields["damage_mask"] = concrete
    regions = fem.Function(space, name="body_id")
    regions.x.array[space.dofmap.list[imported.cell_tags.indices, 0]] = imported.cell_tags.values
    regions.x.scatter_forward()
    fields["body_id"] = regions
    return fields


def bottom_bcs(imported, space):
    dofs = fem.locate_dofs_topological(space, 2, imported.facet_tags.find(10))
    count = imported.domain.comm.allreduce(len(dofs), op=MPI.SUM)
    if count == 0:
        raise ValueError("Canonical mesh has no fixed rock-bottom DOFs")
    return [fem.dirichletbc(np.zeros(3, dtype=space.mesh.geometry.x.dtype), dofs, space)]


def bottom_reactions(imported, space, residual_form, *, interface_springs=None, displacement=None):
    from petsc4py import PETSc

    vector = fem.petsc.assemble_vector(fem.form(residual_form))
    try:
        vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        if interface_springs is not None:
            if displacement is None or interface_springs.matrix is None:
                raise ValueError("Spring support reactions require an assembled operator and displacement")
            work = interface_springs.matrix.createVecLeft()
            try:
                interface_springs.matrix.mult(displacement.x.petsc_vec, work)
                vector.array[:len(work.array)] += work.array
            finally:
                work.destroy()
        dofs = fem.locate_dofs_topological(space, 2, imported.facet_tags.find(10))
        dofs = dofs[dofs < space.dofmap.index_map.size_local]
        values = vector.array.reshape(-1, 3)[dofs].sum(axis=0)
        return imported.domain.comm.allreduce(values, op=MPI.SUM).tolist()
    finally:
        vector.destroy()


ROLLER_FACES = (10, 13, 14, 15, 16)


def rock_support(structure):
    """Outer rock support from the Elmer source report; legacy sources fix the bottom."""
    return getattr(structure, "rock_support", None) or {"model": "fixed_bottom"}


def _roller_faces(structure):
    return {int(tag): face for tag, face in rock_support(structure)["faces"].items()}


def support_bcs(imported, space, structure):
    """Dirichlet BCs only for the legacy fixed bottom; rollers are springs in the form."""
    return bottom_bcs(imported, space) if rock_support(structure)["model"] == "fixed_bottom" else []


def support_form(trial, test, ds, structure):
    """Normal-only roller penalty k (u.n)(v.n) on each outer rock face (k in N/m^3).

    Matches Elmer ``Spring ij = k n_i n_j``; the tangential directions stay free.
    """
    if rock_support(structure)["model"] != "rollers":
        return None
    import ufl

    form = 0
    for tag, face in _roller_faces(structure).items():
        normal = ufl.as_vector([float(v) for v in face["outward_normal"]])
        form += float(face["normal_stiffness_n_per_m3"]) * ufl.dot(trial, normal) * ufl.dot(test, normal) * ds(tag)
    return form


def require_support_tags(imported, structure):
    if rock_support(structure)["model"] != "rollers":
        return
    for tag in _roller_faces(structure):
        if imported.domain.comm.allreduce(len(imported.facet_tags.find(tag)), op=MPI.SUM) == 0:
            raise ValueError(f"Canonical mesh has no rock roller facets for boundary tag {tag}")


def support_reactions(imported, space, structure, displacement, residual_form, ds, *, interface_springs=None):
    """Rock support reactions (N) on the structure, per support face."""
    if rock_support(structure)["model"] == "fixed_bottom":
        return {"rock_bottom": bottom_reactions(
            imported, space, residual_form, interface_springs=interface_springs, displacement=displacement)}
    import ufl

    comm = imported.domain.comm
    reactions = {}
    for tag, face in _roller_faces(structure).items():
        normal = ufl.as_vector([float(v) for v in face["outward_normal"]])
        traction = -float(face["normal_stiffness_n_per_m3"]) * ufl.dot(displacement, normal) * normal
        reactions[f"rock_{face['name']}"] = [
            float(comm.allreduce(fem.assemble_scalar(fem.form(traction[i] * ds(tag))), op=MPI.SUM))
            for i in range(3)
        ]
    return reactions
