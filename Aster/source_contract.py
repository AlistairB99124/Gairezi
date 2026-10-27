"""Read the saved Elmer case as data; never execute text from its SIF."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sif_boundaries(text: str) -> dict[int, str]:
    return {
        int(number): body
        for number, body in re.findall(r"(?ms)^Boundary Condition (\d+)\s*\n(.*?)^End\s*$", text)
    }


def real_value(block: str, key: str) -> float:
    match = re.search(rf"(?mi)^\s*{re.escape(key)}\s*=\s*Real\s+([-+\deE.]+)\s*$", block)
    if match is None:
        raise ValueError(f"Missing constant Real value for {key}")
    value = float(match[1])
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite {key}")
    return value


def target(block: str) -> int:
    match = re.search(r"Target Boundaries\(1\)\s*=\s*(\d+)", block)
    if match is None:
        raise ValueError("Expected exactly one target face tag")
    return int(match[1])


def nodal_areas(coords: np.ndarray, faces: np.ndarray) -> dict[int, float]:
    xyz = coords[faces]
    areas = .5 * np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    if not len(areas) or not np.isfinite(areas).all() or (areas <= 0).any():
        raise ValueError("Empty or degenerate spring surface")
    ids, inverse = np.unique(faces, return_inverse=True)
    lumped = np.bincount(inverse.ravel(), weights=np.repeat(areas / 3., 3))
    return dict(zip(ids.tolist(), lumped.tolist()))


def audit_source_materials(source_mesh: Path) -> dict:
    model = source_mesh.parent
    text = (model / "dam_model.sif").read_text(encoding="utf-8")
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))

    def block(kind, number):
        match = re.search(rf"(?ms)^{kind} {number}\s*\n(.*?)^End\s*$", text)
        if match is None:
            raise ValueError(f"Missing source {kind} {number}")
        return match[1]

    for body, material in ((1, 1), (2, 2), (3, 1)):
        value = block("Body", body)
        for key, number in (("Material", material), ("Body Force", material), ("Equation", 1)):
            if not re.search(rf"(?m)^\s*{key}\s*=\s*{number}\s*$", value):
                raise ValueError(f"Unexpected saved {key} assignment on body {body}")
    for number, name in ((1, "concrete"), (2, "bedrock")):
        values = config["material"][name]
        for key, expected in (("Youngs Modulus", values["youngs_modulus"]),
                              ("Poisson Ratio", values["poisson_ratio"]), ("Density", values["density"])):
            if not np.isclose(real_value(block("Material", number), key), expected, rtol=1e-12, atol=0.):
                raise ValueError(f"Saved material {number} differs from configuration: {key}")
        force = real_value(block("Body Force", number), "Stress Bodyforce 3")
        if not np.isclose(force, config["loads"]["gravity"] * values["density"], rtol=1e-12, atol=0.):
            raise ValueError(f"Saved body force {number} differs from configuration")
        if re.search(r"(?mi)^\s*Stress Bodyforce [12]\s*=", block("Body Force", number)):
            raise ValueError("Nonvertical saved gravity is not supported")
    return {"body_material_assignments": True, "material_constants": True, "gravity_force_density": True}


def saved_pressures(blocks: dict[int, str]) -> tuple[dict, float]:
    pressures = {}
    for name in ("UpstreamHydrostaticPressure", "DownstreamTailwater"):
        block = next((b for b in blocks.values() if f'Name = "{name}"' in b), None)
        match = re.search(
            r'Real MATC "\s*-([\d.eE+-]+)\s*\*\s*\(([-\d.eE+]+)\s*-\s*tx\)\s*\*\s*\(tx\s*<\s*([-\d.eE+]+)\)"',
            block or "",
        )
        if match is None or float(match[2]) != float(match[3]):
            raise ValueError(f"Unsupported saved hydrostatic law for {name}")
        gradient, surface = float(match[1]), float(match[2])
        if not np.isfinite([gradient, surface]).all() or gradient <= 0.:
            raise ValueError(f"Invalid saved hydrostatic law for {name}")
        pressures[name] = {"gradient_pa_per_m": gradient, "surface_z_m": surface}
    crest = next((b for b in blocks.values() if 'Name = "CrestOverflowSurcharge"' in b), None)
    if crest is None:
        raise ValueError("Missing saved crest surcharge")
    return pressures, -real_value(crest, "Normal Force")


def native_projector_audit(source_mesh: Path, coords, faces, tags) -> dict:
    """Compare saved native mortar rows with consistent and lumped triangle laws."""
    model = source_mesh.parent
    blocks = sif_boundaries((model / "dam_model.sif").read_text())
    partitions = list((model / "mesh").glob("partitioning.*"))
    if len(partitions) != 1:
        raise ValueError("Native projector audit requires one unambiguous partition node map")
    maps, hashes = {}, {}
    for path in partitions[0].glob("part.*.nodes"):
        data = np.loadtxt(path, ndmin=2)
        ids = data[:, 0].astype(np.int64)
        if (ids < 1).any() or (ids >= len(coords)).any() \
                or not np.allclose(data[:, 2:5], coords[ids], rtol=0., atol=1e-7):
            raise ValueError("Partition node map does not match saved source coordinates")
        maps[int(path.name.split(".")[1]) - 1] = ids
        hashes[str(path.relative_to(model))] = sha256(path)

    def compact(rows, cols, values):
        keys = np.asarray(rows, dtype=np.int64) * len(coords) + np.asarray(cols, dtype=np.int64)
        keys, inverse = np.unique(keys, return_inverse=True)
        return keys, np.bincount(inverse, weights=values)

    def difference(actual, expected):
        keys = np.union1d(actual[0], expected[0])
        delta = np.zeros(len(keys))
        delta[np.searchsorted(keys, actual[0])] += actual[1]
        delta[np.searchsorted(keys, expected[0])] -= expected[1]
        denominator = np.linalg.norm(expected[1])
        if denominator == 0.:
            raise ValueError("Zero native projector reference norm")
        return float(np.linalg.norm(delta) / denominator)

    checks, missing = {}, []
    for number, block in blocks.items():
        mortar = re.search(r"Mortar BC = Integer (\d+)", block)
        if mortar is None:
            continue
        name = re.search(r'Name\s*=\s*"([^"]+)"', block)[1].removesuffix("_spring")
        if not re.search(r"Save Projector = Logical True", block):
            missing.append(name)
            continue
        slave = faces[tags == target(block)]
        master = faces[tags == target(blocks[int(mortar[1])])]
        lookup = {tuple(np.round(coords[node], 8)): int(node) for node in np.unique(master)}
        corresponding = np.array([
            [lookup[tuple(np.round(coords[node], 8))] for node in face] for face in slave
        ])
        xyz = coords[slave]
        area = .5 * np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        mass = area[:, None, None] / 12. * (np.ones((3, 3)) + np.eye(3))
        rows = np.repeat(slave, 6, axis=1).ravel()
        cols = np.tile(np.column_stack([slave, corresponding]), (1, 3)).ravel()
        consistent = compact(rows, cols, np.concatenate([mass, -mass], axis=2).ravel())
        lumped = compact(np.repeat(slave, 2, axis=1).ravel(),
                         np.stack([slave, corresponding], axis=2).ravel(),
                         np.repeat(area / 3., 6) *
                         np.tile([1., -1.], len(slave) * 3))
        actual_rows, actual_cols, actual_values = [], [], []
        for rank, mapping in maps.items():
            path = model / f"p{number}_part{rank}.dat"
            if not path.is_file() or path.stat().st_size == 0:
                continue
            hashes[path.name] = sha256(path)
            data = np.loadtxt(path, ndmin=2)
            local = data[:, :2].astype(np.int64) - 1
            if not np.isfinite(data).all() or (local < 0).any() or (local >= len(mapping)).any() \
                    or not np.array_equal(data[:, :2], local + 1):
                raise ValueError(f"Invalid native projector entries: {path.name}")
            actual_rows.extend(mapping[local[:, 0]])
            actual_cols.extend(mapping[local[:, 1]])
            actual_values.extend(data[:, 2])
        if not actual_values:
            raise ValueError(f"Saved projector requested but missing for {name}")
        actual = compact(actual_rows, actual_cols, actual_values)
        checks[name] = {
            "relative_frobenius_difference_from_consistent_triangle_mass": difference(actual, consistent),
            "relative_frobenius_difference_from_lumped_triangle_mass": difference(actual, lumped),
        }
    if not checks:
        raise ValueError("No saved native projectors available for audit")
    return {
        "source_artifact_hashes": hashes, "exported_interface_count": len(checks),
        "interfaces_without_native_projector": missing, "interfaces": checks,
        "maximum_relative_consistent_mass_difference": max(
            item["relative_frobenius_difference_from_consistent_triangle_mass"] for item in checks.values()
        ),
        "maximum_relative_lumped_mass_difference": max(
            item["relative_frobenius_difference_from_lumped_triangle_mass"] for item in checks.values()
        ),
        "scope": "Saved native mortar projector rows only; not native whole-model reactions or energy.",
    }


def source_springs(source_mesh: Path, coords: np.ndarray, faces: np.ndarray, tags: np.ndarray) -> dict:
    model = source_mesh.parent
    config = model / "config.json"
    sif = model / "dam_model.sif"
    diagnostic = model / "diagnostic.json"
    for path in (config, sif, diagnostic):
        if not path.is_file():
            raise ValueError(f"Missing saved source-case artifact: {path}")
    blocks = sif_boundaries(sif.read_text(encoding="utf-8"))
    diag = json.loads(diagnostic.read_text(encoding="utf-8"))
    pairs, stiffness, names, areas, interface_names = [], [], [], [], []
    joint_slave, joint_master, joint_coefficients = [], [], []
    for block in blocks.values():
        mortar = re.search(r"Mortar BC = Integer (\d+)", block)
        if mortar is None:
            continue
        name_match = re.search(r'Name\s*=\s*"([^"]+)"', block)
        if name_match is None:
            raise ValueError("Spring boundary lacks a name")
        name = name_match[1].removesuffix("_spring")
        slave_faces = faces[tags == target(block)]
        master_faces = faces[tags == target(blocks[int(mortar[1])])]
        slave_area = nodal_areas(coords, slave_faces)
        master_ids = np.unique(master_faces)
        coordinate_to_master = {}
        for node in master_ids:
            key = tuple(np.round(coords[node], 8))
            if key in coordinate_to_master:
                raise ValueError(f"Ambiguous master coordinates on {name}")
            coordinate_to_master[key] = int(node)
        coefficients = np.array([real_value(block, f"Mortar BC Coefficient Displacement {i}") for i in (1, 2, 3)])
        if (coefficients <= 0).any():
            raise ValueError(f"Nonpositive interface stiffness on {name}")
        if any(tuple(np.round(coords[node], 8)) not in coordinate_to_master
               for node in np.unique(slave_faces)):
            raise ValueError(f"Nonmatching two-sided spring surface {name}")
        master_topology = {tuple(sorted(face)) for face in master_faces.tolist()}
        mapped = np.asarray([[coordinate_to_master[tuple(np.round(coords[node], 8))]
                              for node in face] for face in slave_faces], dtype=np.int64)
        mapped_topology = {tuple(sorted(face)) for face in mapped.tolist()}
        if (len(mapped) != len(master_faces) or len(master_topology) != len(master_faces)
                or len(mapped_topology) != len(mapped) or mapped_topology != master_topology):
            raise ValueError(f"Source interface triangulations do not match on {name}")
        joint_slave.extend(slave_faces)
        joint_master.extend(mapped)
        joint_coefficients.extend([coefficients] * len(slave_faces))
        for node, area in slave_area.items():
            other = coordinate_to_master.get(tuple(np.round(coords[node], 8)))
            if other is None or node == other or np.linalg.norm(coords[node] - coords[other]) > 1e-7:
                raise ValueError(f"Nonmatching two-sided spring surface {name}, node {node}")
            pairs.append((node, other))
            stiffness.append(coefficients * area)
            names.append(f"SPR_IF_{len(names):05d}")
            areas.append(area)
            interface_names.append(name)
    if not pairs:
        raise ValueError("Saved case contains no supported spring interfaces")

    support_by_node: dict[int, np.ndarray] = {}
    for face in diag["rock_outer_support"]["faces"].values():
        name = face["name"]
        semantic = next(key for key, value in diag["rock_outer_support"]["faces"].items() if value["name"] == name)
        saved_tag = diag["boundary_id_map"][semantic]
        matches = [block for block in blocks.values() if target(block) == saved_tag]
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous outer spring boundary {saved_tag}")
        block = matches[0]
        matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                key = f"Spring {i + 1}{j + 1}"
                if re.search(rf"(?mi)^\s*{key}\s*=", block):
                    matrix[i, j] = real_value(block, key)
        normal = np.asarray(face["outward_normal"])
        expected = face["normal_stiffness_n_per_m3"] * np.outer(normal, normal)
        if not np.allclose(matrix, expected, rtol=1e-10, atol=1e-3):
            raise ValueError(f"Saved support matrix differs from diagnostic on {name}")
        for node, area in nodal_areas(coords, faces[tags == saved_tag]).items():
            support_by_node[node] = support_by_node.get(node, np.zeros((3, 3))) + area * matrix
    nodes = np.array(sorted(support_by_node), dtype=np.int64)
    pressures, crest_pressure = saved_pressures(blocks)
    return {
        "pairs": np.asarray(pairs, dtype=np.int64),
        "stiffness": np.asarray(stiffness),
        "names": names,
        "areas": np.asarray(areas),
        "interfaces": interface_names,
        "joint_slave_tri3": np.asarray(joint_slave, dtype=np.int64),
        "joint_master_tri3": np.asarray(joint_master, dtype=np.int64),
        "joint_coefficients_n_per_m3": np.asarray(joint_coefficients),
        "support_nodes": nodes,
        "support_matrices": np.asarray([support_by_node[node] for node in nodes]),
        "source_hashes": {path.name: sha256(path) for path in (source_mesh, config, sif, diagnostic)},
        "source_config": str(config.resolve()),
        "pressures": pressures,
        "crest_pressure_pa": crest_pressure,
    }
