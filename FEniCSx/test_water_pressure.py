"""Pressure-profile regressions across input, Elmer, FEniCSx and Code_Aster."""
import ast
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Elmer.regions.combined_structure import (
    FoundationSupport, MaterialProperties, load_environment, write_combined_vtu
)
from Elmer.structure_assembly import apply_load_case, write_combined_solver_input
import test_dam_mesh
from dam_mesh import import_structure
from dam_elasticity import solve
from types import SimpleNamespace


class WaterPressureTests(unittest.TestCase):
    def loads(self):
        return load_environment(ROOT / "Data" / "Env_Boundaries_And_Loads.json")

    def test_calculation_and_profile(self):
        loads = self.loads()
        self.assertEqual(1000 * 9.81 * 34, 333540)
        self.assertEqual(loads.peak_water_pressure_pa, 333540)
        for z, expected in ((-31, 333540), (-29, 313920), (-25, 274680), (0, 29430), (3, 0), (4, 0)):
            self.assertAlmostEqual(loads.upstream_pressure_pa(z), expected)
        self.assertEqual(loads.water_pressure_gradient_pa_per_m, 9810)

    def test_inconsistent_peak_is_rejected(self):
        data = json.loads((ROOT / "Data" / "Env_Boundaries_And_Loads.json").read_text())
        next(row for row in data if row["Boundary"] == "Peak Water Pressure")["Value"] = 284490
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loads.json"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "MaximumWaterHeight\\+OverflowHead"):
                load_environment(path)

    def structure(self):
        support = FoundationSupport(1e9, 1e9, 5e9)
        return SimpleNamespace(
            loads=self.loads(), material=MaterialProperties(2400, 35e9, 0.2, 2e6),
            foundation_support=support, plinth_support=support,
            nodes=[(0, 0, -31)], cells=[], boundaries=[], region_ids=[]
        )

    def test_active_elmer_load_case_and_generated_sif(self):
        structure = self.structure()
        apply_load_case(structure, ROOT / "Elmer" / "load_cases.json")
        self.assertEqual(structure.loads.peak_water_pressure_pa, 333540)
        self.assertEqual(structure.loads.overflow_head_m, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.sif"
            write_combined_solver_input(structure, path)
            text = path.read_text()
        expression = re.search(r'Real MATC "([^"]+)"', text).group(1)
        for z, expected in ((-31, -333540), (0, -29430), (4, 0)):
            self.assertAlmostEqual(eval(expression, {"__builtins__": {}}, {"tx": z}), expected)
        self.assertIn("Normal Force = Real -29430", text)

    def test_preview_pressure_at_crest_and_base(self):
        structure = self.structure()
        structure.nodes = [(80, 0, -31), (80, 0, 0)]
        structure.boundaries = [(2, (0, 1))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.vtu"
            write_combined_vtu(structure, path)
            raw = path.read_bytes()
            start = raw.index(b"\n_", raw.index(b"<AppendedData")) + 2
            offset = int(re.search(
                rb'Name="HydrostaticPressure"[^>]*offset="(\d+)"', raw[:start]
            ).group(1))
            pressure = np.frombuffer(raw, dtype="<f8", count=2, offset=start + offset + 4)
            np.testing.assert_allclose(pressure, [333540, 29430])

    def test_code_aster_upstream_and_crack_expressions(self):
        loads = {
            "water_density_kg_m3": 1000, "peak_water_pressure_pa": 313920,
            "maximum_water_height_m": 29, "overflow_head_m": 3, "foundation_elevation_m": -29,
            "tailwater_head_m": 2
        }
        for name in ("00_elastic_baseline", "01_xfem_feasibility", "02_crack_stress_intensity",
                     "03_crack_scenarios", "04_crack_contact"):
            path = ROOT / "Aster" / "cases" / f"{name}.comm"
            with self.subTest(case=path.name):
                tree = ast.parse(path.read_text())
                env = {"loads": loads, "gravity_z": -9.81, "crack_z": -25,
                       "spec": {"z_m": -25}, "crack_spec": {},
                       "FORMULE": lambda **kwargs: kwargs, "DEFI_FONCTION": lambda **kwargs: kwargs}
                names = {"water_gradient", "reservoir_gradient", "tailwater_level", "upstream_pressure",
                         "crack_depth_below_surface", "crack_pressure"}
                for node in tree.body:
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
                        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
                upstream = env["upstream_pressure"]["VALE"]
                if isinstance(upstream, str):
                    for z, expected in ((-29, 313920), (0, 29430), (4, 0)):
                        self.assertAlmostEqual(eval(upstream, {"max": max}, {"Z": z}), expected)
                else:
                    self.assertEqual(upstream[:4], (-40, 421830, 3, 0))
                if "crack_pressure" in env:
                    self.assertAlmostEqual(env["crack_pressure"], 274680)

    def test_fenicsx_upstream_force_has_head_offset(self):
        structure = test_dam_mesh.DamMeshTests().cube()
        # Unit-height upstream plane x=1, crest plane z=1; no crest surcharge in this fixture.
        structure.boundaries = [(1 if i == 0 else 2 if i == 3 else 7, face)
                                for i, (_, face) in enumerate(structure.boundaries)]
        structure.material = SimpleNamespace(youngs_modulus_pa=35e9, poissons_ratio=0.2, density_kg_m3=2400)
        structure.loads = SimpleNamespace(gravity_z_m_s2=-9.81, maximum_water_height_m=29,
                                         peak_water_pressure_pa=313920, water_density_kg_m3=1000,
                                         overflow_head_m=3, tailwater_head_m=0)
        support = SimpleNamespace(spring_x_n_per_m3=1e9, spring_y_n_per_m3=1e9, spring_z_n_per_m3=5e9)
        structure.foundation_support = structure.plinth_support = support
        imported = import_structure(structure, 1e-12)
        with tempfile.TemporaryDirectory() as directory:
            _, _, report = solve(imported, structure, {"relative_tolerance": 1e-12, "maximum_iterations": 500}, Path(directory))
        self.assertAlmostEqual(report["applied_force_n"][0], -9810 * 2.5, places=6)


if __name__ == "__main__":
    unittest.main()
