"""Global force/moment and strain-energy checks, independent of stress recovery."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def cell_kinematics(points, cells, displacement):
    xyz = points[cells]
    edges = xyz[:, 1:] - xyz[:, :1]
    volume = np.abs(np.linalg.det(edges)) / 6.
    if not np.isfinite(volume).all() or (volume <= 0).any():
        raise ValueError("Nonpositive or invalid tetrahedron volume")
    delta_u = displacement[cells[:, 1:]] - displacement[cells[:, :1]]
    grad = np.linalg.solve(edges, delta_u).transpose(0, 2, 1)
    return .5 * (grad + grad.transpose(0, 2, 1)), volume


def elastic_stress(strain, young, poisson):
    mu = young / (2. * (1. + poisson))
    lam = young * poisson / ((1. + poisson) * (1. - 2. * poisson))
    return 2. * mu * strain + lam * np.trace(strain, axis1=1, axis2=2)[:, None, None] * np.eye(3)


def volume_energy(points, cells, displacement, young, poisson):
    energy = 0.
    volume = 0.
    for start in range(0, len(cells), 20000):
        ids = cells[start:start + 20000]
        strain, vol = cell_kinematics(points, ids, displacement)
        density = .5 * (elastic_stress(strain, young, poisson) * strain).sum(axis=(1, 2))
        energy += float(vol @ density)
        volume += float(vol.sum())
    return energy, volume


def outward_faces(points, faces, cells):
    """Orient exterior triangles away from their unique incident tetrahedron."""
    dtype = np.dtype([(f"n{i}", np.int64) for i in range(3)])
    keys = np.ascontiguousarray(np.sort(faces, axis=1)).view(dtype).ravel()
    order = np.argsort(keys)
    if np.any(keys[order][1:] == keys[order][:-1]):
        raise ValueError("Duplicate pressure face")
    opposite = np.full(len(faces), -1, dtype=np.int64)
    hits = np.zeros(len(faces), dtype=np.int64)
    for vertex in range(4):
        candidate = np.ascontiguousarray(np.sort(np.delete(cells, vertex, axis=1), axis=1)).view(dtype).ravel()
        index = np.searchsorted(keys[order], candidate)
        valid = index < len(keys)
        rows = np.flatnonzero(valid)
        rows = rows[keys[order][index[rows]] == candidate[rows]]
        matched = order[index[rows]]
        np.add.at(hits, matched, 1)
        opposite[matched] = cells[rows, vertex]
    if not np.all(hits == 1):
        raise ValueError("Pressure face must have exactly one incident volume cell")
    oriented = faces.copy()
    xyz = points[oriented]
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    inward = np.einsum("ij,ij->i", normals, points[opposite] - xyz[:, 0]) > 0.
    oriented[inward] = oriented[inward][:, [0, 2, 1]]
    return oriented


def pressure_resultant(points, faces, surface, gradient, constant=None):
    force, moment = np.zeros(3), np.zeros(3)
    quadrature = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
    for xyz in points[faces]:
        polygon = [point for point in xyz]
        if constant is None:
            clipped = []
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                if a[2] <= surface:
                    clipped.append(a)
                if (a[2] < surface < b[2]) or (b[2] < surface < a[2]):
                    clipped.append(a + (b - a) * ((surface - a[2]) / (b[2] - a[2])))
            polygon = clipped
        for i in range(1, len(polygon) - 1):
            tri = np.array([polygon[0], polygon[i], polygon[i + 1]])
            normal_area = .5 * np.cross(tri[1] - tri[0], tri[2] - tri[0])
            samples = quadrature @ tri
            pressures = np.full(3, constant) if constant is not None else gradient * (surface - samples[:, 2])
            forces = -pressures[:, None] * normal_area / 3.
            force += forces.sum(axis=0)
            moment += np.cross(samples, forces).sum(axis=0)
    return force, moment


def diagnostics(mesh_dir: Path, full_displacement: np.ndarray) -> dict:
    with np.load(mesh_dir / "three_body_mesh_arrays.npz") as data:
        points = data["nodes"]
        if full_displacement.shape != points.shape or not np.isfinite(full_displacement).all():
            raise ValueError("Invalid diagnostic displacement field")
        manifest = json.loads((mesh_dir / "mesh_manifest.json").read_text())
        config = json.loads(Path(manifest["source_config"]).read_text())
        force, moment = np.zeros(3), np.zeros(3)
        energy = 0.
        for key, material in (("tetra4_concrete", "concrete"), ("tetra4_bedrock", "bedrock")):
            values = config["material"][material]
            cells = data[key]
            part, volume = volume_energy(points, cells, full_displacement,
                                         values["youngs_modulus"], values["poisson_ratio"])
            energy += part
            xyz = points[cells]
            vol = np.abs(np.linalg.det(xyz[:, 1:] - xyz[:, :1])) / 6.
            gravity = np.array([0., 0., config["loads"]["gravity"] * values["density"]])
            force += volume * gravity
            moment += np.cross((xyz.mean(axis=1) * vol[:, None]).sum(axis=0), gravity)
        for key, name in (("tri3_upstream", "UpstreamHydrostaticPressure"),
                          ("tri3_downstream", "DownstreamTailwater"), ("tri3_crest", None)):
            values = manifest["source_pressures"][name] if name else {}
            faces = outward_faces(points, data[key], data["tetra4_concrete"])
            f, m = pressure_resultant(points, faces, values.get("surface_z_m", 0.),
                                      values.get("gradient_pa_per_m", 0.),
                                      None if name else manifest["source_crest_pressure_pa"])
            force += f
            moment += m
        pairs = data["seg2_wall_foundation_springs"]
        relative = full_displacement[pairs[:, 0]] - full_displacement[pairs[:, 1]]
        energy += .5 * float((data["interface_stiffness_n_per_m"] * relative ** 2).sum())
        nodes = data["support_nodes"]
        displacement = full_displacement[nodes]
        reactions = -np.einsum("nij,nj->ni", data["support_matrices_n_per_m"], displacement)
        energy += -.5 * float((reactions * displacement).sum())
        reaction = reactions.sum(axis=0)
        reaction_moment = np.cross(points[nodes], reactions).sum(axis=0)
        return {
            "applied_force_n": force.tolist(), "support_reaction_n": reaction.tolist(),
            "applied_moment_nm": moment.tolist(), "support_reaction_moment_nm": reaction_moment.tolist(),
            "relative_force_balance_error": float(np.linalg.norm(force + reaction) / np.linalg.norm(force)),
            "relative_moment_balance_error": float(np.linalg.norm(moment + reaction_moment) / np.linalg.norm(moment)),
            "linear_strain_energy_j": energy,
        }
