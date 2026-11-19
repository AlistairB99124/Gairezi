"""The mortar fill in the Code_Aster elastic case (no solver: export and case generation only)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Aster.build_three_body_mesh_arrays import MeshExportError, export
from Aster.elastic_diagnostics import diagnostics
from Aster.generate_three_body_case import generate
from Aster.source_contract import audit_source_materials
from Elmer.test_mortar_fill import MORTAR, TWO_STEPS, stepped_config, volumes


class MortarFillCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from Elmer.three_body_diagnostic import prepare

        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.manifests, cls.inputs = {}, {}
        for name, mortar in (("plain", None), ("filled", MORTAR)):
            folder = cls.root / name
            folder.mkdir()
            (folder / "config.json").write_text(json.dumps(stepped_config(TWO_STEPS, mortar=mortar)), encoding="utf-8")
            prepare(folder, folder / "model", joints="spring", integrated=True)
            cls.manifests[name] = export(source_mesh=folder / "model" / "three_body.msh", output_dir=folder / "aster")
            cls.inputs[name] = generate(case_path=folder / "aster" / "case.comm",
                                        inputs_path=folder / "aster" / "case_inputs.json",
                                        mesh_manifest_path=folder / "aster" / "mesh_manifest.json")

    def arrays(self, name: str) -> dict:
        with np.load(self.root / name / "aster" / "three_body_mesh_arrays.npz") as data:
            return {key: data[key] for key in data.files}

    def case(self, name: str) -> str:
        return (self.root / name / "aster" / "case.comm").read_text(encoding="utf-8")

    def test_the_fill_is_exported_as_its_own_block_and_never_as_concrete(self):
        plain, filled = self.arrays("plain"), self.arrays("filled")
        self.assertNotIn("tetra4_mortar", plain)
        self.assertNotIn("mortar_cells", self.manifests["plain"])
        self.assertNotIn("MORTAR", self.manifests["plain"]["groups"])
        # Concrete cells, their bodies and their coordinates are those of the wall without the fill.
        np.testing.assert_array_equal(filled["concrete_body_ids"], plain["concrete_body_ids"])
        np.testing.assert_array_equal(filled["nodes"][filled["tetra4_concrete"]], plain["nodes"][plain["tetra4_concrete"]])
        self.assertEqual(self.manifests["filled"]["concrete_cells"], self.manifests["plain"]["concrete_cells"])
        self.assertEqual(self.manifests["filled"]["groups"]["CONCRETE"], self.manifests["plain"]["groups"]["CONCRETE"])
        mortar = filled["tetra4_mortar"]
        self.assertEqual(len(mortar), self.manifests["filled"]["mortar_cells"])
        self.assertFalse(np.isin(mortar, filled["tetra4_concrete"]).any())
        diagnostic = json.loads((self.root / "filled" / "model" / "diagnostic.json").read_text())
        self.assertAlmostEqual(np.abs(volumes(filled["nodes"], mortar)).sum(), diagnostic["mortar_fill"]["volume_m3"], places=6)
        self.assertAlmostEqual(self.manifests["filled"]["groups"]["MORTAR"]["volume_m3"],
                               diagnostic["mortar_fill"]["volume_m3"], places=6)
        # Source rows: concrete first, then the fill.
        self.assertEqual(len(filled["source_cell_rows"]), len(filled["tetra4_concrete"]) + len(mortar))
        self.assertEqual(len(np.unique(filled["source_cell_rows"])), len(filled["source_cell_rows"]))

    def test_the_fill_is_joined_to_the_wall_by_springs_of_the_bond_stiffness(self):
        filled, plain = self.arrays("filled"), self.arrays("plain")
        names = filled["interface_names"]
        fill = np.array([name.startswith("wall_mortar") for name in names])
        self.assertTrue(fill.any())
        self.assertEqual(int((~fill).sum()), len(plain["interface_names"]))
        pairs = filled["seg2_wall_foundation_springs"][fill]
        wall_nodes, fill_nodes = np.unique(filled["tetra4_concrete"]), np.unique(filled["tetra4_mortar"])
        self.assertTrue(np.isin(pairs[:, 0], wall_nodes).all())
        self.assertTrue(np.isin(pairs[:, 1], fill_nodes).all())
        np.testing.assert_allclose(filled["nodes"][pairs[:, 0]], filled["nodes"][pairs[:, 1]], atol=1e-9)
        # Stiffness = bond spring (N/m3) times the node's share of the tied area, on every axis.
        area = filled["wall_foundation_tributary_area_m2"][fill]
        np.testing.assert_allclose(filled["interface_stiffness_n_per_m"][fill],
                                   np.repeat(1.25e9 * area[:, None], 3, axis=1), rtol=1e-12)
        coefficients = self.inputs["filled"]["source_interface_coefficients_n_per_m3"]
        for name in set(names[fill].tolist()):
            np.testing.assert_allclose(coefficients[name], [1.25e9] * 3, rtol=1e-9)

    def test_the_case_gives_the_fill_its_own_material_and_keeps_concrete_as_it_was(self):
        plain, filled = self.case("plain"), self.case("filled")
        self.assertNotIn("MORTAR", plain)
        self.assertIn("MORTAR_MAT = DEFI_MATERIAU(\n    ELAS=_F(\n        E=350000000,\n        NU=0.2,\n        RHO=2200,\n", filled)
        self.assertIn("_F(GROUP_MA='MORTAR', MATER=MORTAR_MAT),", filled)
        self.assertIn("'CONCRETE', 'MORTAR', 'UPSTREAM', 'DOWNSTREAM', 'CREST',", filled)
        self.assertIn("GROUP_MA=('CONCRETE', 'MORTAR'),", filled)
        concrete = lambda text: text[text.index("CONCRETE_MAT = DEFI_MATERIAU("):text.index("BEDROCK_MAT")]  # noqa: E731
        self.assertEqual(concrete(filled), concrete(plain))
        self.assertIn("_F(GROUP_MA='CONCRETE', MATER=CONCRETE_MAT),", filled)
        self.assertEqual(self.inputs["filled"]["material"]["mortar"],
                         {"density_kg_m3": 2200.0, "youngs_modulus_pa": 3.5e8, "poissons_ratio": 0.2})
        self.assertEqual(self.inputs["filled"]["material"]["concrete"], self.inputs["plain"]["material"]["concrete"])
        self.assertNotIn("mortar", self.inputs["plain"]["material"])
        self.assertNotIn("mortar_fill", self.inputs["plain"])
        self.assertEqual(self.inputs["filled"]["mortar_fill"]["cells"], self.manifests["filled"]["mortar_cells"])

    def test_the_saved_solver_input_is_audited_with_the_fill(self):
        mesh = self.root / "filled" / "model" / "three_body.msh"
        self.assertEqual(audit_source_materials(mesh),
                         {"body_material_assignments": True, "material_constants": True, "gravity_force_density": True})
        sif = mesh.parent / "dam_model.sif"
        original = sif.read_text(encoding="utf-8")
        self.addCleanup(sif.write_text, original, encoding="utf-8")
        for wrong, message in ((original.replace("Density = Real 2200", "Density = Real 2400"), "Saved material 2"),
                               (original.replace('Name = "MortarFillBody"\n    Equation = 1\n    Material = 2',
                                                 'Name = "MortarFillBody"\n    Equation = 1\n    Material = 1'),
                                "Unexpected saved Material assignment on body 3")):
            self.assertNotEqual(wrong, original)
            sif.write_text(wrong, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                audit_source_materials(mesh)

    def test_the_applied_load_grows_by_exactly_the_weight_of_the_fill(self):
        weights = {}
        for name in ("plain", "filled"):
            arrays = self.arrays(name)
            weights[name] = diagnostics(self.root / name / "aster", np.zeros_like(arrays["nodes"]))["applied_force_n"]
        volume = self.manifests["filled"]["groups"]["MORTAR"]["volume_m3"]
        self.assertAlmostEqual((weights["filled"][2] - weights["plain"][2]) / (-9.81 * 2200.0 * volume), 1.0, places=9)
        # Horizontal water load: the sloped face replaces the faces it covers; nothing is lost or doubled
        # (both models are dry downstream, so the horizontal resultant is the upstream water alone).
        np.testing.assert_allclose(weights["filled"][:2], weights["plain"][:2], rtol=1e-12)

    def test_a_fill_without_its_material_or_in_the_wrong_body_is_refused(self):
        folder = self.root / "filled"
        manifest_path = folder / "aster" / "mesh_manifest.json"
        config_path = folder / "model" / "config.json"
        original = config_path.read_text(encoding="utf-8")
        self.addCleanup(config_path.write_text, original, encoding="utf-8")
        config = json.loads(original)
        del config["material"]["mortar"]
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(ValueError):
            generate(case_path=folder / "refused.comm", inputs_path=folder / "refused.json", mesh_manifest_path=manifest_path)
        config_path.write_text(original, encoding="utf-8")
        mesh = folder / "model" / "three_body.msh"
        text = mesh.read_text()
        other = folder / "model" / "unknown_body.msh"
        self.addCleanup(other.unlink)
        lines = text.splitlines()
        start = lines.index("$Elements")
        for index in range(start + 2, len(lines)):
            row = lines[index].split()
            if len(row) > 4 and row[1] == "4" and row[3] == "4":
                row[3] = row[4] = "9"
                lines[index] = " ".join(row)
                break
        other.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(MeshExportError, "Unexpected tetra tags"):
            export(source_mesh=other, output_dir=folder / "refused")


if __name__ == "__main__":
    unittest.main()
