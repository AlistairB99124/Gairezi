"""Exact saved Elmer mesh import and piecewise three-body material/load data."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from dolfinx import fem
import meshio
from mpi4py import MPI
import numpy as np

# Wall (1) may crack; bedrock (2) stays intact.
DAMAGE_REGION_IDS = (1,)


def is_three_body(structure):
    return getattr(structure, "three_body_config", None) is not None


def require_fracture_resolution(options):
    from crack_field import MINIMUM_ELEMENTS_PER_LENGTH_SCALE

    if not options.minimum_elements_per_length_scale >= MINIMUM_ELEMENTS_PER_LENGTH_SCALE:
        raise ValueError(
            "Three-body fracture requires at least "
            f"{MINIMUM_ELEMENTS_PER_LENGTH_SCALE:g} elements per length scale; "
            f"configured target is {options.minimum_elements_per_length_scale:g}. "
            "Under-resolved research targets cannot be used for canonical dam jobs."
        )


def load_structure(config_path: Path, model: Path, structure):
    config = json.loads(config_path.read_text())
    if json.loads((model / "config.json").read_text()) != config:
        raise ValueError("Three-body source mesh configuration differs from the submitted snapshot")
    diagnostic = json.loads((model / "diagnostic.json").read_text())
    if diagnostic["joints"] != "spring" or diagnostic["solver_body_ids"] != {"wall": 1, "bedrock": 2}:
        raise ValueError("Expected canonical spring-jointed three-body source")
    bonded_seat = config["geometry"]["bedrock"]["support"]["model"] == "bonded"
    if (diagnostic.get("wall_bedrock_connection", "spring") == "bonded") != bonded_seat:
        raise ValueError("Source wall/bedrock connection differs from the submitted configuration")
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
    if set(regions) != {1, 2} or len(source.points) != diagnostic["nodes"]:
        raise ValueError("Canonical source body tags or node count do not match diagnostic")
    body_nodes = {
        region: {node for (_, cell), tag in zip(cells, regions, strict=True)
                 if tag == region for node in cell}
        for region in (1, 2)
    }
    shared_seat = body_nodes[1] & body_nodes[2]
    if bool(shared_seat) != bonded_seat:
        raise ValueError("Wall/bedrock mesh connectivity does not implement the configured connection")
    from dam_mesh import compact_unused_nodes
    result = compact_unused_nodes(replace(
        structure, nodes=[tuple(p) for p in source.points], cells=cells,
        region_ids=regions, boundaries=boundaries,
    ))
    result.three_body_config = config
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
        "wall_bedrock_connection": "bonded" if bonded_seat else "spring",
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
    if any(document.get("configuration_sha256") != structure.submitted_configuration_sha256
           for document in (manifest, report)):
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
    return {"directory": str(directory.resolve()), "configuration_sha256": manifest["configuration_sha256"],
            "source_sha256": source["sha256"], "verification": gate}


def materials(imported, structure):
    """DG0 coefficients preserve discontinuous concrete/rock constitutive data."""
    config = structure.three_body_config
    space = fem.functionspace(imported.domain, ("DG", 0))
    fields = {}
    for key in ("density", "youngs_modulus", "poisson_ratio"):
        field = fem.Function(space, name=key)
        for tag, name in ((1, "concrete"), (2, "bedrock")):
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


def bottom_reactions(imported, space, residual_form):
    from petsc4py import PETSc

    vector = fem.petsc.assemble_vector(fem.form(residual_form))
    try:
        vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
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


def support_reactions(imported, space, structure, displacement, residual_form, ds):
    """Rock support reactions (N) on the structure, per support face."""
    if rock_support(structure)["model"] == "fixed_bottom":
        return {"rock_bottom": bottom_reactions(imported, space, residual_form)}
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
