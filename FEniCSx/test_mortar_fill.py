"""The optional mortar fill (body 4) in the FEniCSx three-body import. Run inside WSL, from FEniCSx/."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from dam_mesh import FACES, import_structure
from interface_springs import InterfaceSprings
from test_three_body import JOINT_SPRINGS, foundation_column
from three_body_model import (
    MORTAR_FILL_REGION_ID, WALL_MORTAR_TAGS, materials, source_foundation, source_mortar_fill,
    validate_foundation_faces,
)

BOND = {"x": 1.25e9, "y": 1.25e9, "z": 1.25e9}
MORTAR = {"density": 2200.0, "youngs_modulus": 3.5e8, "poisson_ratio": 0.2, "bond_spring_n_per_m3": BOND}
RECORD = {"tetrahedra": 12, "volume_m3": 3.0, "bond_spring_n_per_m3": BOND,
          "material": {"density": 2200.0, "youngs_modulus": 3.5e8, "poisson_ratio": 0.2}}


def stepped_config(mortar=MORTAR) -> dict:
    """The checked-in configuration with a stepped wall on the elastic-layer bedrock."""
    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text(encoding="utf-8"))
    wall = config["geometry"]["wall"]
    wall.update(radial_width_m=4.0, centerline_radius_m=78.0, downstream_radius_m=76.0,
                thickness_steps=[{"top_z_m": -20.0, "radial_width_m": 4.5}], joint_support={"model": "bonded"})
    wall.pop("step_end_taper_ratio", None)
    config["bedrock_analysis"] = {**config.get("bedrock_analysis", {}), "mode": "elastic_layer"}
    config["material"].pop("mortar", None)
    if mortar is not None:
        config["material"]["mortar"] = copy.deepcopy(mortar)
    return config


def source_diagnostic(config: dict, fill: bool = True) -> dict:
    """What the Elmer model records for that configuration (only the parts the import checks)."""
    from Elmer.model_config import foundation_support

    support = foundation_support(config)
    diagnostic = {
        "foundation": dict(config["geometry"]["foundation"]),
        "source_body_ids": {"wall": 1, "foundation": 3},
        "wall_bedrock_connection": "not_in_contact", "wall_foundation_connection": support["model"],
        "foundation_bedrock_connection": "elastic_layer",
        "spring_pairs": {}, "interface_stiffness_n_per_m3": {},
    }
    if support["model"] != "bonded":
        diagnostic["spring_pairs"]["wall_foundation"] = [17, 18]
        diagnostic["interface_stiffness_n_per_m3"]["wall_foundation"] = support["spring_n_per_m3"]
    if fill:
        diagnostic["source_body_ids"]["mortar_fill"] = 4
        diagnostic["spring_pairs"]["wall_mortar"] = [19, 20]
        diagnostic["interface_stiffness_n_per_m3"]["wall_mortar"] = BOND
        diagnostic["mortar_fill"] = copy.deepcopy(RECORD)
    return diagnostic


class SourceContractTests(unittest.TestCase):
    def test_fill_is_recognised_only_when_source_and_configuration_agree(self):
        config = stepped_config()
        self.assertEqual(source_mortar_fill(config, source_diagnostic(config)), RECORD)
        plain = stepped_config(mortar=None)
        self.assertIsNone(source_mortar_fill(plain, source_diagnostic(plain, fill=False)))
        for submitted, recorded in ((config, source_diagnostic(config, fill=False)),
                                    (plain, source_diagnostic(config))):
            with self.assertRaisesRegex(ValueError, "mortar fill differs"):
                source_mortar_fill(submitted, recorded)
        for key, value in (("material", {**RECORD["material"], "density": 2400.0}),
                           ("bond_spring_n_per_m3", {**BOND, "z": 1.0e9}), ("tetrahedra", 0)):
            changed = source_diagnostic(config)
            changed["mortar_fill"][key] = value
            with self.assertRaisesRegex(ValueError, "material differs"):
                source_mortar_fill(config, changed)

    def test_a_mortar_block_without_steps_is_no_fill(self):
        config = stepped_config()
        config["geometry"]["wall"]["thickness_steps"] = []
        self.assertIsNone(source_mortar_fill(config, source_diagnostic(config, fill=False)))

    def test_the_fill_adds_one_spring_pair_with_its_bond_stiffness(self):
        config = stepped_config()
        _, pairs = source_foundation(config, source_diagnostic(config), [])
        fill = [pair for pair in pairs if pair["name"] == "wall_mortar"]
        self.assertEqual(fill, [{"name": "wall_mortar", "tags": WALL_MORTAR_TAGS, "spring_n_per_m3": BOND}])
        plain = stepped_config(mortar=None)
        _, plain_pairs = source_foundation(plain, source_diagnostic(plain, fill=False), [])
        self.assertEqual([pair for pair in pairs if pair["name"] != "wall_mortar"], plain_pairs)

    def test_a_source_that_ties_or_numbers_the_fill_differently_is_rejected(self):
        config = stepped_config()
        changes = (
            ("body tags", lambda d: d["source_body_ids"].pop("mortar_fill")),
            ("body tags", lambda d: d["source_body_ids"].update(mortar_fill=5)),
            ("spring pairs", lambda d: d["spring_pairs"].pop("wall_mortar")),
            ("spring tags", lambda d: d["spring_pairs"].update(wall_mortar=[21, 22])),
            ("spring stiffness", lambda d: d["interface_stiffness_n_per_m3"].update(wall_mortar={**BOND, "x": 1.0})),
        )
        for message, change in changes:
            diagnostic = source_diagnostic(config)
            change(diagnostic)
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                source_foundation(config, diagnostic, [])

    def test_tied_faces_must_be_wall_on_one_side_and_fill_on_the_other(self):
        nodes = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=float)
        pair = [{"name": "wall_mortar", "tags": WALL_MORTAR_TAGS, "spring_n_per_m3": BOND}]
        boundaries = [(19, (0, 1, 2)), (20, (3, 4, 5))]
        bodies = {1: {0, 1, 2}, 3: set(), MORTAR_FILL_REGION_ID: {3, 4, 5}}
        validate_foundation_faces(nodes, boundaries, bodies, pair)
        for wrong in ({1: {0, 1, 2, 3, 4, 5}, 3: set(), MORTAR_FILL_REGION_ID: set()},
                      {1: {3, 4, 5}, 3: set(), MORTAR_FILL_REGION_ID: {0, 1, 2}}):
            with self.assertRaisesRegex(ValueError, "body/tags mismatch"):
                validate_foundation_faces(nodes, boundaries, wrong, pair)


def filled_column():
    """test_three_body's bedrock/foundation/wall column with a fill block beside the wall, tied by springs."""
    structure = foundation_column()
    wall = structure.cells[2][1]
    offset = len(structure.nodes)
    structure.nodes = np.vstack((structure.nodes, structure.nodes[list(wall)] + [1.0, 0.0, 0.0]))
    cell = tuple(range(offset, offset + 8))
    structure.cells.append((5, cell))
    structure.region_ids.append(MORTAR_FILL_REGION_ID)
    # The wall's x = 1 face and the fill's x = 0 face coincide.
    on = lambda nodes, x: all(np.isclose(structure.nodes[node][0], x) for node in nodes)  # noqa: E731
    wall_face = next(tuple(wall[node] for node in face) for face in FACES[5]
                     if on([wall[node] for node in face], 1.0))
    structure.boundaries = [(19 if face == wall_face else tag, face) for tag, face in structure.boundaries]
    for face in FACES[5]:
        nodes = tuple(cell[node] for node in face)
        structure.boundaries.append((20 if on(nodes, 1.0) else 7, nodes))
    structure.three_body_config["material"]["mortar"] = copy.deepcopy(MORTAR)
    # The column's configuration is not a dam's: the fill and its tie are declared directly.
    structure.mortar_fill = copy.deepcopy(RECORD)
    structure.spring_pairs = [*structure.spring_pairs,
                              {"name": "wall_mortar", "tags": WALL_MORTAR_TAGS, "spring_n_per_m3": BOND}]
    return structure


class FillBodyTests(unittest.TestCase):
    def test_fill_has_its_own_material_no_damage_and_one_tie_to_the_wall(self):
        structure = filled_column()
        imported = import_structure(structure, 1e-12)
        from dolfinx import fem

        space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
        springs = InterfaceSprings(imported, structure, space)
        self.assertEqual(set(springs.report), {"wall_foundation", "foundation_bedrock", "wall_mortar"})
        self.assertEqual(springs.report["wall_mortar"]["spring_n_per_m3"], BOND)
        self.assertAlmostEqual(springs.report["wall_mortar"]["area_m2"], 1.0)
        self.assertEqual(springs.report["wall_foundation"]["spring_n_per_m3"], JOINT_SPRINGS)
        # Moving everything together stretches no spring.
        translation = np.tile([1., 2., 3.], space.dofmap.index_map.size_global)
        np.testing.assert_allclose(springs._operator @ translation, 0, atol=1e-3)
        # Moving the fill alone by d is resisted by bond stiffness x tied area x d on each axis.
        xyz = space.tabulate_dof_coordinates()
        fill_cells = imported.cell_tags.find(MORTAR_FILL_REGION_ID)
        fill_nodes = np.unique(space.dofmap.list[fill_cells])
        self.assertGreater(len(fill_cells), 0)
        self.assertTrue(np.all(xyz[fill_nodes][:, 0] >= 1.0 - 1e-12))
        moved = np.zeros((space.dofmap.index_map.size_global, 3))
        moved[fill_nodes] = [1e-3, 2e-3, 3e-3]
        force = (springs._operator @ moved.ravel()).reshape(-1, 3)
        np.testing.assert_allclose(force[fill_nodes].sum(axis=0), [1.25e6, 2.5e6, 3.75e6], rtol=1e-9)
        np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-3)
        coefficients = materials(imported, structure)
        dg = coefficients["density"].function_space
        for tag, density, young, damage in ((1, 1000., 1e7, 1.), (2, 2000., 2e7, 0.), (3, 1000., 1e7, 0.),
                                            (MORTAR_FILL_REGION_ID, 2200., 3.5e8, 0.)):
            ids = dg.dofmap.list[imported.cell_tags.find(tag), 0]
            np.testing.assert_allclose(coefficients["density"].x.array[ids], density)
            np.testing.assert_allclose(coefficients["youngs_modulus"].x.array[ids], young)
            np.testing.assert_allclose(coefficients["damage_mask"].x.array[ids], damage)
            np.testing.assert_allclose(coefficients["body_id"].x.array[ids], tag)

    def test_wall_displacement_is_scoped_to_the_wall(self):
        from dolfinx import fem
        from three_body_model import maximum_wall_displacement

        structure = filled_column()
        imported = import_structure(structure, 1e-12)
        space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
        displacement = fem.Function(space)
        fill_nodes = np.unique(space.dofmap.list[imported.cell_tags.find(MORTAR_FILL_REGION_ID)])
        displacement.x.array.reshape(-1, 3)[fill_nodes] = [0.0, 0.0, 5.0]
        self.assertEqual(maximum_wall_displacement(imported, displacement), 0.0)


if __name__ == "__main__":
    unittest.main()
