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


class ThreeBodyTests(unittest.TestCase):
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
