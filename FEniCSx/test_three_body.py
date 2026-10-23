"""Canonical wall-bedrock mesh, material and interface checks."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from dam_mesh import FACES, import_structure
from interface_springs import InterfaceSprings, intersection_mass, projected_spring_matrix


def column():
    nodes, cells, regions, boundaries = [], [], [], []
    for region, bottom in ((2, 0.0), (1, 1.0)):
        offset = len(nodes)
        nodes.extend([(x, y, z + bottom) for x, y, z in (
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        )])
        cell = tuple(range(offset, offset + 8))
        cells.append((5, cell))
        regions.append(region)
        tags = {2: (10, 12), 1: (11, 7)}[region]
        for index, face in enumerate(FACES[5]):
            tag = tags[index] if index < 2 else 7
            boundaries.append((tag, tuple(cell[node] for node in face)))
    support = {"model": "spring", "spring_n_per_m3": {"x": 1e6, "y": 2e6, "z": 3e6}}
    return SimpleNamespace(
        nodes=np.asarray(nodes, dtype=float), cells=cells, region_ids=regions, boundaries=boundaries,
        material=SimpleNamespace(youngs_modulus_pa=1e7, poissons_ratio=0.0, density_kg_m3=1000.0,
                                 tensile_strength_pa=2e6),
        loads=SimpleNamespace(gravity_z_m_s2=-10.0, water_density_kg_m3=1000.0,
                              tailwater_head_m=0.0, overflow_head_m=0.0,
                              upstream_surface_z_m=0.0),
        concrete_minimum_z=1.0, three_body_source={},
        three_body_config={
            "material": {
                "concrete": {"density": 1000.0, "youngs_modulus": 1e7, "poisson_ratio": 0.0},
                "bedrock": {"density": 2000.0, "youngs_modulus": 2e7, "poisson_ratio": 0.0},
            },
            "geometry": {"bedrock": {"support": support}},
        },
    )


def bonded_column():
    structure = column()
    structure.three_body_config["geometry"]["bedrock"]["support"]["model"] = "bonded"
    shared_nodes = dict(zip(range(8, 12), range(4, 8)))
    cells = [(kind, tuple(shared_nodes.get(node, node) for node in cell)) for kind, cell in structure.cells]
    boundaries = [(tag, tuple(shared_nodes.get(node, node) for node in face))
                  for tag, face in structure.boundaries if tag not in (11, 12)]
    used = sorted({node for _, cell in cells for node in cell})
    mapping = {node: index for index, node in enumerate(used)}
    structure.nodes = structure.nodes[used]
    structure.cells = [(kind, tuple(mapping[node] for node in cell)) for kind, cell in cells]
    structure.boundaries = [(tag, tuple(mapping[node] for node in face)) for tag, face in boundaries]
    return structure


JOINT_SPRINGS = {"x": 4e6, "y": 5e6, "z": 6e6}


def jointed_column():
    """Bedrock, then two wall blocks (both body 1) split at z = 2 with duplicated nodes."""
    structure = column()
    offset = len(structure.nodes)
    upper = structure.nodes[8:16] + [0, 0, 1]
    structure.nodes = np.vstack((structure.nodes, upper))
    cell = tuple(range(offset, offset + 8))
    structure.cells.append((5, cell))
    structure.region_ids.append(1)
    # The lower block's top face (index 1) becomes the joint master; the upper block's bottom its slave.
    lower_top = tuple(structure.cells[1][1][node] for node in FACES[5][1])
    structure.boundaries = [(22 if face == lower_top else tag, face) for tag, face in structure.boundaries]
    for index, face in enumerate(FACES[5]):
        structure.boundaries.append((21 if index == 0 else 7, tuple(cell[node] for node in face)))
    structure.wall_connection = "spring_jointed_blocks"
    structure.wall_joints = [{"name": "wall_joint_1", "z_m": -10.0, "tags": (21, 22),
                              "spring_n_per_m3": JOINT_SPRINGS}]
    return structure


def foundation_column():
    structure = jointed_column()
    structure.region_ids = [2, 3, 1]
    structure.wall_joints = []
    structure.wall_connection = "single_monolithic_body"
    structure.three_body_config["geometry"]["wall"] = {
        "joint_support": {"model": "spring", "spring_n_per_m3": JOINT_SPRINGS}}
    foundation = {"depth_m": 0.5, "side_margin_m": 0.5,
                  "support": {"model": "spring", "spring_n_per_m3": JOINT_SPRINGS}}
    structure.three_body_config["geometry"]["foundation"] = foundation
    from three_body_model import source_foundation
    diagnostic = {"foundation": foundation, "solver_body_ids": {"wall": 1, "bedrock": 2, "foundation": 3},
                  "wall_bedrock_connection": "not_in_contact", "wall_foundation_connection": "spring",
                  "foundation_bedrock_connection": "spring",
                  "spring_pairs": {"wall_foundation": [17, 18], "foundation_bedrock": [11, 12]}}
    diagnostic["interface_stiffness_n_per_m3"] = {
        "wall_foundation": JOINT_SPRINGS,
        "foundation_bedrock": structure.three_body_config["geometry"]["bedrock"]["support"]["spring_n_per_m3"]}
    structure.foundation, structure.spring_pairs = source_foundation(structure.three_body_config, diagnostic, [])
    structure.boundaries = [({21: 17, 22: 18}.get(tag, tag), face) for tag, face in structure.boundaries]
    return structure


def split_foundation_column():
    """Horizontal tread and vertical riser sharing an edge, with separate multipliers."""
    from three_body_model import source_foundation
    structure = foundation_column()
    structure.nodes = np.array([
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, -1),  # Foundation.
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (0, -1, -.5),
        (1, 0, 0), (0, 1, 0), (0, 0, -1), (1, 1, -1),  # Bedrock.
    ], dtype=float)
    structure.cells = [(4, (0, 1, 2, 3)), (4, (4, 5, 6, 8)),
                       (4, (4, 5, 7, 9)), (4, (10, 11, 12, 13))]
    structure.region_ids = [3, 1, 1, 2]
    face_tags = {tuple(sorted(face)): tag for face, tag in (
        ((0, 1, 2), 18), ((4, 5, 6), 17), ((0, 1, 3), 1001), ((4, 5, 7), 1000),
        ((1, 2, 3), 11), ((10, 11, 12), 12))}
    structure.boundaries = []
    for kind, cell in structure.cells:
        for pattern in FACES[kind]:
            face = tuple(sorted(cell[i] for i in pattern))
            structure.boundaries.append((face_tags.get(face, 7), face))
    planes = {
        "tread_0000": {"unit_normal": [0., 0., 1.], "offset_m": 0., "triangles_per_side": 1},
        "vertical_0001": {"unit_normal": [0., 1., 0.], "offset_m": 0., "triangles_per_side": 1},
    }
    diagnostic = {"foundation": structure.foundation,
                  "solver_body_ids": {"wall": 1, "bedrock": 2, "foundation": 3},
                  "wall_bedrock_connection": "not_in_contact", "wall_foundation_connection": "spring",
                  "foundation_bedrock_connection": "spring",
                  "spring_pairs": {"foundation_bedrock": [11, 12], "wall_foundation": {
                      "grouping": "exact_plane", "legacy_tags": [17, 18], "planes": planes,
                      "faces": dict(zip(planes, ([17, 18], [1000, 1001]))) }},
                  "interface_stiffness_n_per_m3": {
                      "wall_foundation": JOINT_SPRINGS,
                      "foundation_bedrock": structure.three_body_config["geometry"]["bedrock"]["support"]["spring_n_per_m3"]}}
    structure.foundation, structure.spring_pairs = source_foundation(structure.three_body_config, diagnostic, [])
    return structure, diagnostic


def near_planar_column():
    from dam_mesh import subdivide
    from three_body_model import near_planar_geometry, source_foundation
    structure = foundation_column()
    tetra, regions, _, faces, tags, _, points, _ = subdivide(
        structure.nodes, structure.cells, structure.region_ids, structure.boundaries)
    for index in range(12, 20):
        x, y, z = points[index]
        points[index, 2] += .2*x + .005*x*y
    structure.nodes = points
    structure.cells = [(4, tuple(cell)) for cell in tetra]
    structure.region_ids = regions.tolist()
    structure.boundaries = [(int(tag), tuple(face)) for tag, face in zip(tags, faces)]
    first = [face for tag, face in structure.boundaries if tag == 17]
    normal, offset, error, spread, classes = near_planar_geometry(points[np.asarray(first)])
    plane = {"unit_normal": normal.tolist(), "offset_m": offset,
             "maximum_off_plane_distance_m": error, "normal_spread_deg": spread,
             "orientation_class": next(iter(classes)), "triangles_per_side": len(first)}
    diagnostic = {"foundation": structure.foundation,
                  "solver_body_ids": {"wall": 1, "bedrock": 2, "foundation": 3},
                  "wall_bedrock_connection": "not_in_contact", "wall_foundation_connection": "spring",
                  "foundation_bedrock_connection": "spring",
                  "spring_pairs": {"foundation_bedrock": [11, 12], "wall_foundation": {
                      "grouping": "near_planar", "legacy_tags": [17, 18], "tolerance_m": .005,
                      "max_angle_deg": 3., "continuity": "edge_connected; single orientation class",
                      "faces": {"chamfer_0000": [17, 18]}, "planes": {"chamfer_0000": plane}}},
                  "interface_stiffness_n_per_m3": {
                      "wall_foundation": JOINT_SPRINGS,
                      "foundation_bedrock": structure.three_body_config["geometry"]["bedrock"]["support"]["spring_n_per_m3"]}}
    structure.foundation, structure.spring_pairs = source_foundation(structure.three_body_config, diagnostic, [])
    return structure, diagnostic


class ThreeBodyTests(unittest.TestCase):
    def test_foundation_keeps_wall_off_bedrock_when_bedrock_support_is_bonded(self):
        from three_body_model import source_wall_bedrock_seat
        def config(bedrock_model, foundation):
            geometry = {"bedrock": {"support": {"model": bedrock_model}}}
            if foundation:
                geometry["foundation"] = {"depth_m": 1.0}
            return {"geometry": geometry}
        # Foundation bonded to bedrock, wall on springs: the wall never touches bedrock.
        self.assertFalse(source_wall_bedrock_seat(
            config("bonded", True), {"wall_bedrock_connection": "not_in_contact"}))
        self.assertFalse(source_wall_bedrock_seat(
            config("spring", True), {"wall_bedrock_connection": "not_in_contact"}))
        self.assertTrue(source_wall_bedrock_seat(
            config("bonded", False), {"wall_bedrock_connection": "bonded"}))
        self.assertFalse(source_wall_bedrock_seat(config("spring", False), {}))
        with self.assertRaisesRegex(ValueError, "wall/bedrock connection differs"):
            source_wall_bedrock_seat(config("bonded", False), {"wall_bedrock_connection": "spring"})

    def test_bonded_interface_requires_complete_shared_faces_and_nodes(self):
        from three_body_model import validate_bonded_interface
        cells = [(4, (0, 1, 2, 3)), (4, (0, 2, 1, 4))]
        regions = [3, 2]
        shared = {0, 1, 2}
        validate_bonded_interface(cells, regions, 3, 2, shared, 1, "foundation/bedrock")
        with self.assertRaisesRegex(ValueError, "must share all 2 boundary faces"):
            validate_bonded_interface(cells, regions, 3, 2, shared, 2, "foundation/bedrock")
        with self.assertRaisesRegex(ValueError, "must share all 1 boundary faces"):
            validate_bonded_interface(cells, regions, 3, 2, {0, 1, 4}, 1, "foundation/bedrock")

    def test_near_planar_curved_slave_faces_report_and_rigid_translation(self):
        from dolfinx import fem
        from three_body_model import validate_foundation_faces
        structure, diagnostic = near_planar_column()
        body_nodes = {region: {node for (_, cell), tag in zip(structure.cells, structure.region_ids)
                              if tag == region for node in cell} for region in (1, 2, 3)}
        validate_foundation_faces(structure.nodes, structure.boundaries, body_nodes, structure.spring_pairs)
        imported = import_structure(structure, 1e-12)
        space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
        springs = InterfaceSprings(imported, structure, space)
        report = springs.report["wall_foundation"]
        declaration = diagnostic["spring_pairs"]["wall_foundation"]
        for key in ("grouping", "tolerance_m", "max_angle_deg", "continuity"):
            self.assertEqual(report[key], declaration[key])
        self.assertEqual(report["faces"]["chamfer_0000"]["source_plane"], declaration["planes"]["chamfer_0000"])
        self.assertEqual(report["spring_n_per_m3"], JOINT_SPRINGS)
        np.testing.assert_allclose(springs._operator @ np.tile([1., 2., 3.], space.dofmap.index_map.size_global),
                                   0, atol=1e-6)
        # The actual two triangles are not on one exact plane; do not flatten their areas.
        self.assertGreater(declaration["planes"]["chamfer_0000"]["maximum_off_plane_distance_m"], 1e-4)
        expected_area = sum(np.linalg.norm(np.cross(structure.nodes[f[1]]-structure.nodes[f[0]],
                                                   structure.nodes[f[2]]-structure.nodes[f[0]]))/2
                            for tag, f in structure.boundaries if tag == 17)
        self.assertAlmostEqual(report["area_m2"], expected_area)

    def test_near_planar_recomputes_limits_and_rejects_forged_geometry_metrics(self):
        from copy import deepcopy
        from three_body_model import source_foundation, validate_foundation_faces
        structure, diagnostic = near_planar_column()
        body_nodes = {region: {node for (_, cell), tag in zip(structure.cells, structure.region_ids)
                              if tag == region for node in cell} for region in (1, 2, 3)}
        for key, value in (("maximum_off_plane_distance_m", 0.), ("normal_spread_deg", 0.),
                           ("orientation_class", "tread"), ("offset_m", 20.)):
            pairs = deepcopy(structure.spring_pairs)
            pairs[0]["plane"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_foundation_faces(structure.nodes, structure.boundaries, body_nodes, pairs)
        for key, value in (("tolerance_m", 1e-5), ("max_angle_deg", .01)):
            pairs = deepcopy(structure.spring_pairs)
            pairs[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_foundation_faces(structure.nodes, structure.boundaries, body_nodes, pairs)
        for key, value in (("tolerance_m", -1.), ("max_angle_deg", float("nan")),
                           ("max_angle_deg", True), ("continuity", "disconnected")):
            changed = deepcopy(diagnostic)
            changed["spring_pairs"]["wall_foundation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                source_foundation(structure.three_body_config, changed, [])
        nodes = structure.nodes.copy()
        nodes[16:20, 2] += .02
        with self.assertRaises(ValueError):
            validate_foundation_faces(nodes, structure.boundaries, body_nodes, structure.spring_pairs)

    def test_near_planar_rejects_orientation_crossing_and_disconnected_groups(self):
        from three_body_model import near_planar_geometry, validate_near_planar_faces
        xyz = np.array([[[10., 0., 0.], [11., 0., 0.], [10., 1., 0.]],
                        [[11., 0., 0.], [11., 1., .01], [10., 1., 0.]]])
        normal, offset, error, spread, classes = near_planar_geometry(xyz)
        self.assertEqual(classes, {"tread", "chamfer"})
        pair = {"name": "mixed", "tolerance_m": .005, "max_angle_deg": 3.,
                "plane": {"unit_normal": normal, "offset_m": offset,
                          "maximum_off_plane_distance_m": error, "normal_spread_deg": spread,
                          "orientation_class": "tread"}}
        with self.assertRaises(ValueError):
            validate_near_planar_faces(xyz, pair)
        xyz[1] += [10., 0., 0.]
        with self.assertRaisesRegex(ValueError, "edge connected"):
            near_planar_geometry(xyz)

    def test_exact_plane_pairs_eliminate_independently_and_report_logically(self):
        from copy import deepcopy
        from unittest.mock import patch
        from dolfinx import fem
        from three_body_model import validate_foundation_faces
        structure, _ = split_foundation_column()
        body_nodes = {region: {node for (_, cell), tag in zip(structure.cells, structure.region_ids)
                              if tag == region for node in cell} for region in (1, 2, 3)}
        validate_foundation_faces(structure.nodes, structure.boundaries, body_nodes, structure.spring_pairs)
        imported = import_structure(structure, 1e-12)
        space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
        with patch("interface_springs.projected_spring_matrix", wraps=projected_spring_matrix) as projected:
            springs = InterfaceSprings(imported, structure, space)
        self.assertEqual(projected.call_count, 3)
        self.assertEqual(set(springs.report), {"wall_foundation", "foundation_bedrock"})
        group = springs.report["wall_foundation"]
        self.assertEqual(group["grouping"], "exact_plane")
        self.assertEqual(set(group["faces"]), {"tread_0000", "vertical_0001"})
        self.assertAlmostEqual(group["area_m2"], sum(p["area_m2"] for p in group["faces"].values()))
        np.testing.assert_allclose(springs._operator @ np.tile([1., 2., 3.], space.dofmap.index_map.size_global),
                                   0, atol=1e-6)
        pooled = deepcopy(structure)
        pooled.boundaries = [({1000: 17, 1001: 18}.get(tag, tag), face) for tag, face in pooled.boundaries]
        pooled.spring_pairs = [dict(structure.spring_pairs[0], name="wall_foundation", tags=(17, 18)),
                               structure.spring_pairs[-1]]
        pooled_imported = import_structure(pooled, 1e-12)
        pooled_space = fem.functionspace(pooled_imported.domain, ("Lagrange", 1, (3,)))
        combined = InterfaceSprings(pooled_imported, pooled, pooled_space)
        self.assertGreater(np.linalg.norm((springs._operator - combined._operator).data), 1e4)

    def test_exact_plane_source_rejects_changed_tags_metadata_or_geometry(self):
        from copy import deepcopy
        from three_body_model import source_foundation, validate_foundation_faces
        structure, diagnostic = split_foundation_column()
        for field, value in (("grouping", "all_risers"), ("legacy_tags", [11, 12]),
                             ("faces", {"tread_0000": [17, 18], "vertical_0001": [17, 1001]}),
                             ("planes", {})):
            changed = deepcopy(diagnostic)
            changed["spring_pairs"]["wall_foundation"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                source_foundation(structure.three_body_config, changed, [])
        body_nodes = {region: {node for (_, cell), tag in zip(structure.cells, structure.region_ids)
                              if tag == region for node in cell} for region in (1, 2, 3)}
        for field, value in (("offset_m", 999.), ("triangles_per_side", 9), ("unit_normal", [1., 0., 0.])):
            pairs = deepcopy(structure.spring_pairs)
            pairs[0]["plane"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_foundation_faces(structure.nodes, structure.boundaries, body_nodes, pairs)
        nodes = structure.nodes.copy()
        nodes[6, 2] += .1
        with self.assertRaisesRegex(ValueError, "off-plane|differs from mesh"):
            validate_foundation_faces(nodes, structure.boundaries, body_nodes, structure.spring_pairs)

    def test_staircase_tread_does_not_project_onto_adjacent_riser(self):
        tread = np.array(((0., 0., 0.), (1., -1., 0.), (1., 1., 0.)))
        riser = np.array(((0., 0., 1.), (0., 0., 0.), (1., 1., 0.)))
        point = np.array((2/3, 1/6, 1/6)) @ tread
        self.assertTrue(np.all(point >= riser.min(axis=0)))
        self.assertTrue(np.all(point <= riser.max(axis=0)))
        edges = (riser[1:] - riser[0]).T
        uv = np.linalg.lstsq(edges, point - riser[0], rcond=None)[0]
        self.assertGreaterEqual(min(1 - uv.sum(), *uv), -1e-14)
        self.assertGreater(np.linalg.norm(edges @ uv + riser[0] - point), 0.2)
        # A bounding-box and projected-barycentric test accepts this off-plane point.
        mass, area = intersection_mass(tread, riser)
        self.assertIsNone(mass)
        self.assertEqual(area, 0.)
        _, exact_area = intersection_mass(tread, tread)
        self.assertAlmostEqual(exact_area, 1.)

    def test_foundation_interfaces_materials_and_wall_only_damage(self):
        structure = foundation_column()
        imported = import_structure(structure, 1e-12)
        from dolfinx import fem
        from three_body_model import materials
        space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
        springs = InterfaceSprings(imported, structure, space)
        self.assertEqual(set(springs.report), {"wall_foundation", "foundation_bedrock"})
        self.assertEqual(springs.report["wall_foundation"]["spring_n_per_m3"], JOINT_SPRINGS)
        self.assertEqual(springs.report["foundation_bedrock"]["spring_n_per_m3"],
                         structure.three_body_config["geometry"]["bedrock"]["support"]["spring_n_per_m3"])
        translation = np.tile([1., 2., 3.], space.dofmap.index_map.size_global)
        np.testing.assert_allclose(springs._operator @ translation, 0, atol=1e-6)
        coefficients = materials(imported, structure)
        dg = coefficients["density"].function_space
        for tag, density, damage in ((1, 1000., 1.), (2, 2000., 0.), (3, 1000., 0.)):
            ids = dg.dofmap.list[imported.cell_tags.find(tag), 0]
            np.testing.assert_allclose(coefficients["density"].x.array[ids], density)
            np.testing.assert_allclose(coefficients["damage_mask"].x.array[ids], damage)

    def test_untagged_foundation_exposed_faces_are_traction_free(self):
        structure = foundation_column()
        foundation_nodes = set(structure.cells[1][1])
        structure.boundaries = [(tag, face) for tag, face in structure.boundaries
                                if tag != 7 or not set(face).issubset(foundation_nodes)]
        imported = import_structure(structure, 1e-12)
        self.assertGreater(len(imported.facet_tags.find(7)), 0)

    def test_bonded_foundation_keeps_only_the_rock_springs(self):
        from copy import deepcopy
        from three_body_model import connection_report, source_foundation
        from dolfinx import fem
        structure = foundation_column()
        support = {"model": "bonded"}
        structure.three_body_config["geometry"]["foundation"]["support"] = support
        structure.foundation["support"] = support
        diagnostic = {
            "foundation": deepcopy(structure.foundation),
            "solver_body_ids": {"wall": 1, "bedrock": 2, "foundation": 3},
            "wall_bedrock_connection": "not_in_contact",
            "wall_foundation_connection": "bonded", "foundation_bedrock_connection": "spring",
            "spring_pairs": {"foundation_bedrock": [11, 12]},
            "interface_stiffness_n_per_m3": {
                "foundation_bedrock": structure.three_body_config["geometry"]["bedrock"]["support"]["spring_n_per_m3"]},
        }
        structure.foundation, structure.spring_pairs = source_foundation(structure.three_body_config, diagnostic, [])
        # Tie only the coincident wall/foundation seat nodes, leaving the rock separate.
        wall_seat = {tuple(structure.nodes[n]): n for tag, face in structure.boundaries if tag == 17 for n in face}
        remap = {n: wall_seat[tuple(structure.nodes[n])]
                 for tag, face in structure.boundaries if tag == 18 for n in face}
        structure.cells = [(kind, tuple(remap.get(n, n) for n in cell)) for kind, cell in structure.cells]
        structure.boundaries = [(tag, tuple(remap.get(n, n) for n in face))
                                for tag, face in structure.boundaries if tag not in (17, 18)]
        used = sorted({n for _, cell in structure.cells for n in cell})
        compact = {old: new for new, old in enumerate(used)}
        structure.nodes = structure.nodes[used]
        structure.cells = [(kind, tuple(compact[n] for n in cell)) for kind, cell in structure.cells]
        structure.boundaries = [(tag, tuple(compact[n] for n in face)) for tag, face in structure.boundaries]
        imported = import_structure(structure, 1e-12)
        space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
        springs = InterfaceSprings(imported, structure, space)
        self.assertEqual(set(springs.report), {"foundation_bedrock"})
        self.assertEqual(connection_report(structure)["wall_foundation_connection"], "bonded")
        changed = deepcopy(diagnostic)
        changed["wall_foundation_connection"] = "spring"
        with self.assertRaisesRegex(ValueError, "connections differ"):
            source_foundation(structure.three_body_config, changed, [])

    def test_bonded_foundation_bedrock_uses_shared_faces_without_springs(self):
        from copy import deepcopy
        from three_body_model import connection_report, source_foundation
        from dolfinx import fem
        structure = foundation_column()
        structure.three_body_config["geometry"]["bedrock"]["support"]["model"] = "bonded"
        diagnostic = {
            "foundation": deepcopy(structure.foundation),
            "solver_body_ids": {"wall": 1, "bedrock": 2, "foundation": 3},
            "wall_bedrock_connection": "not_in_contact",
            "wall_foundation_connection": "spring",
            "foundation_bedrock_connection": "bonded",
            "spring_pairs": {"wall_foundation": [17, 18]},
            "interface_stiffness_n_per_m3": {"wall_foundation": JOINT_SPRINGS},
        }
        structure.foundation, structure.spring_pairs = source_foundation(
            structure.three_body_config, diagnostic, [],
        )
        self.assertEqual(
            next(pair for pair in structure.spring_pairs if pair["name"] == "foundation_bedrock"),
            {"name": "foundation_bedrock", "tags": (), "spring_n_per_m3": None, "connection": "bonded"},
        )
        self.assertEqual(connection_report(structure)["foundation_bedrock_connection"], "bonded")

        rock_nodes = {
            tuple(np.round(structure.nodes[node], 8)): node
            for tag, face in structure.boundaries if tag == 12 for node in face
        }
        remap = {
            node: rock_nodes[tuple(np.round(structure.nodes[node], 8))]
            for tag, face in structure.boundaries if tag == 11 for node in face
        }
        structure.cells = [
            (kind, tuple(remap.get(node, node) for node in cell))
            for kind, cell in structure.cells
        ]
        structure.boundaries = [
            (tag, tuple(remap.get(node, node) for node in face))
            for tag, face in structure.boundaries if tag not in (11, 12)
        ]
        used = sorted({node for _, cell in structure.cells for node in cell})
        compact = {old: new for new, old in enumerate(used)}
        structure.nodes = structure.nodes[used]
        structure.cells = [
            (kind, tuple(compact[node] for node in cell))
            for kind, cell in structure.cells
        ]
        structure.boundaries = [
            (tag, tuple(compact[node] for node in face))
            for tag, face in structure.boundaries
        ]

        imported = import_structure(structure, 1e-12)
        space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
        springs = InterfaceSprings(imported, structure, space)
        self.assertEqual(set(springs.report), {"wall_foundation", "foundation_bedrock"})
        self.assertEqual(springs.report["foundation_bedrock"]["connection"], "bonded")
        self.assertNotIn("spring_n_per_m3", springs.report["foundation_bedrock"])
        changed = deepcopy(diagnostic)
        changed["foundation_bedrock_connection"] = "spring"
        with self.assertRaisesRegex(ValueError, "connections differ"):
            source_foundation(structure.three_body_config, changed, [])

    def test_source_foundation_mismatch_is_rejected(self):
        from copy import deepcopy
        from three_body_model import source_foundation
        structure = foundation_column()
        diagnostic = {"foundation": structure.foundation,
                      "solver_body_ids": {"wall": 1, "bedrock": 2, "foundation": 3},
                      "wall_bedrock_connection": "not_in_contact", "wall_foundation_connection": "spring",
                      "foundation_bedrock_connection": "spring",
                      "spring_pairs": {"wall_foundation": [17, 18], "foundation_bedrock": [11, 12]}}
        diagnostic["interface_stiffness_n_per_m3"] = {
            "wall_foundation": JOINT_SPRINGS,
            "foundation_bedrock": structure.three_body_config["geometry"]["bedrock"]["support"]["spring_n_per_m3"]}
        source_foundation(structure.three_body_config, diagnostic, [])
        for key, value in (("interface_stiffness_n_per_m3", {}), ("foundation", None), ("foundation", {"depth_m": 1., "side_margin_m": 0.5}),
                           ("spring_pairs", {"wall_bedrock": [11, 12]}),
                           ("wall_bedrock_connection", "spring"), ("wall_foundation_connection", "bonded")):
            changed = deepcopy(diagnostic)
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                source_foundation(structure.three_body_config, changed, [])
        with self.assertRaises(ValueError):
            source_foundation(column().three_body_config, diagnostic, [])

    def test_jointed_wall_blocks_add_their_own_spring_pair(self):
        structure = jointed_column()
        imported = import_structure(structure, 1e-12)
        from dolfinx import fem
        space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
        springs = InterfaceSprings(imported, structure, space)
        self.assertEqual(set(springs.report), {"wall_bedrock", "wall_joint_1"})
        joint = springs.report["wall_joint_1"]
        self.assertEqual(joint["spring_n_per_m3"], JOINT_SPRINGS)
        self.assertAlmostEqual(joint["area_m2"], 1.0)
        self.assertEqual(joint["faces"][0], joint["faces"][1])
        # A rigid-body translation of the whole wall stores no spring energy in the joint.
        operator = springs._operator
        self.assertEqual(operator.shape[0], 3 * space.dofmap.index_map.size_global)
        translation = np.zeros(operator.shape[0])
        translation[2::3] = 1.0
        np.testing.assert_allclose(operator @ translation, 0, atol=1e-6)
        self.assertGreater(operator.nnz, 0)

    def test_source_wall_joints_must_match_configuration(self):
        from three_body_model import source_wall_joints
        wall = {"thickness_steps": [{"top_z_m": -10, "radial_width_m": 4.5},
                                    {"top_z_m": -20, "radial_width_m": 5.0}],
                "joint_support": {"model": "spring", "spring_n_per_m3": JOINT_SPRINGS}}
        config = {"geometry": {"wall": wall}}
        joints = [{"name": f"wall_joint_{k}", "z_m": z, "tags": [19 + 2 * k, 20 + 2 * k],
                   "spring_n_per_m3": JOINT_SPRINGS} for k, z in ((1, -10.0), (2, -20.0))]
        connection, parsed = source_wall_joints(
            config, {"wall_connection": "spring_jointed_blocks", "wall_joints": joints})
        self.assertEqual(connection, "spring_jointed_blocks")
        self.assertEqual([joint["tags"] for joint in parsed], [(21, 22), (23, 24)])
        for diagnostic in (
            {},  # monolithic source while the configuration asks for spring joints
            {"wall_connection": "spring_jointed_blocks", "wall_joints": joints[:1]},
            {"wall_connection": "spring_jointed_blocks", "wall_joints": [
                dict(joints[0], spring_n_per_m3={"x": 1, "y": 1, "z": 1}), joints[1]]},
            {"wall_connection": "unknown", "wall_joints": joints},
        ):
            with self.subTest(diagnostic=diagnostic), self.assertRaises(ValueError):
                source_wall_joints(config, diagnostic)
        bonded = {"geometry": {"wall": dict(wall, joint_support={"model": "bonded"})}}
        self.assertEqual(source_wall_joints(bonded, {}), ("single_monolithic_body", []))

    def test_spring_seat_uses_only_the_wall_bedrock_pair(self):
        structure = column()
        imported = import_structure(structure, 1e-12)
        from dolfinx import fem
        space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
        springs = InterfaceSprings(imported, structure, space)
        self.assertEqual(set(springs.report), {"wall_bedrock"})
        self.assertEqual(springs.report["wall_bedrock"]["spring_n_per_m3"],
                         {"x": 1e6, "y": 2e6, "z": 3e6})
        self.assertGreater(springs.report["wall_bedrock"]["area_m2"], 0)
        self.assertEqual(springs.report["wall_bedrock"]["faces"][0],
                         springs.report["wall_bedrock"]["faces"][1])

    def test_bonded_seat_uses_shared_node_faces(self):
        structure = bonded_column()
        imported = import_structure(structure, 1e-12)
        from dolfinx import fem
        space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
        springs = InterfaceSprings(imported, structure, space)
        self.assertEqual(springs.report["wall_bedrock"], {
            "connection": "bonded", "discretisation": "shared-node faces",
        })
        wall_nodes = {node for (_, cell), region in zip(structure.cells, structure.region_ids)
                      if region == 1 for node in cell}
        bedrock_nodes = {node for (_, cell), region in zip(structure.cells, structure.region_ids)
                         if region == 2 for node in cell}
        self.assertEqual(len(wall_nodes & bedrock_nodes), 4)

    def test_only_wall_cells_receive_damage_and_concrete_material(self):
        from three_body_model import DAMAGE_REGION_IDS, materials
        structure = column()
        imported = import_structure(structure, 1e-12)
        fields = materials(imported, structure)
        self.assertEqual(DAMAGE_REGION_IDS, (1,))
        self.assertEqual(set(np.unique(imported.cell_tags.values)), {1, 2})
        dofmap = fields["density"].function_space.dofmap.list[:, 0]
        wall = imported.cell_tags.find(1)
        bedrock = imported.cell_tags.find(2)
        np.testing.assert_allclose(fields["density"].x.array[dofmap[wall]], 1000.0)
        np.testing.assert_allclose(fields["density"].x.array[dofmap[bedrock]], 2000.0)
        np.testing.assert_allclose(fields["damage_mask"].x.array[dofmap[wall]], 1.0)
        np.testing.assert_allclose(fields["damage_mask"].x.array[dofmap[bedrock]], 0.0)

    def test_fracture_resolution_enforces_canonical_minimum(self):
        from crack_field import PhaseFieldSolverOptions
        from three_body_model import require_fracture_resolution
        options = PhaseFieldSolverOptions()
        for target in (0.9, 1.99, float("nan")):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "at least 2"):
                require_fracture_resolution(PhaseFieldSolverOptions(
                    minimum_elements_per_length_scale=target,
                ))
        for target in (2.0, 3.0):
            require_fracture_resolution(PhaseFieldSolverOptions(
                minimum_elements_per_length_scale=target,
            ))

    def test_baseline_peak_is_scoped_to_wall_nodes(self):
        from elmer_baseline import concrete_nodal_peak
        points = np.arange(36, dtype=float).reshape(12, 3)
        stress = np.zeros((12, 6))
        stress[:, 0] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200]
        regions = np.asarray([1] * 4 + [3] * 4 + [2] * 4)
        peak = concrete_nodal_peak(points, stress, regions)
        self.assertEqual(peak["region_id"], 1)
        self.assertEqual(peak["node_id"], 8)
        self.assertEqual(peak["value_pa"], 8.0)

    def test_projected_mortar_matches_multiplier_elimination(self):
        xyz = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        mass, area = intersection_mass(xyz, xyz)
        matrix = projected_spring_matrix([(np.arange(6), mass)], 6).toarray()
        jump = mass[:3]
        expected = jump.T @ np.diag(1 / mass[:3, :3].sum(axis=1)) @ jump
        np.testing.assert_allclose(matrix, expected)
        np.testing.assert_allclose(matrix @ np.ones(6), 0, atol=1e-16)
        self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-15)
        self.assertAlmostEqual(np.asarray((1.0, 1.0, 1.0, 0.0, 0.0, 0.0)) @ matrix
                               @ np.asarray((1.0, 1.0, 1.0, 0.0, 0.0, 0.0)), area)


if __name__ == "__main__":
    unittest.main()
