"""Regression tests for platform-specific MPI launch commands."""
import os
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from Elmer import structure_assembly as assembly
from Elmer.regions.combined_structure import load_environment, load_material, load_foundation_support


class BondedTopologyTests(unittest.TestCase):
    def test_elmergrid_uses_local_output_name_under_dotted_ancestor(self):
        with tempfile.TemporaryDirectory(suffix=".case") as directory:
            root = Path(directory)
            output = root / "mesh"
            header = output / "partitioning.4" / "part.1.header"
            def conversion(*args, **kwargs):
                header.parent.mkdir(parents=True)
                header.write_text("test")
            with patch.object(assembly.shutil, "which", return_value="ElmerGrid"), patch.object(
                assembly.subprocess, "run", side_effect=conversion,
            ) as worker:
                self.assertTrue(assembly.convert_with_elmergrid(root / "source.msh", output, 4))
                self.assertEqual(worker.call_args.args[0][-2:], ["-out", "mesh"])
                self.assertEqual(worker.call_args.kwargs["cwd"], root)

    def test_elmergrid_missing_expected_output_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            assembly.shutil, "which", return_value="ElmerGrid",
        ), patch.object(assembly.subprocess, "run"):
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "expected mesh header"):
                assembly.convert_with_elmergrid(root / "source.msh", root / "mesh", 4)

    def test_generated_wall_plinth_are_one_body_before_and_after_tetrahedralization(self):
        from collections import Counter
        from Elmer.regions.combined_structure import audit_combined_structure, build_combined_structure
        from Elmer.regions.tetrahedral import _CELL_FACES, split_face, tetrahedralize_structure

        source = build_combined_structure(assembly.ROOT)
        audit = audit_combined_structure(source)
        self.assertEqual(audit["face_connected_components"], 1)
        self.assertGreater(audit["wall_plinth_interface_faces"], 0)
        self.assertEqual(audit["wall_plinth_connection"], "bonded_intact_provisional")
        self.assertEqual(len({tuple(round(x, 9) for x in xyz) for xyz in source.nodes}), len(source.nodes))
        self.assertFalse(any(tag in (8, 9) for tag, _ in source.boundaries))
        source_faces = {}
        for (kind, cell), region in zip(source.cells, source.region_ids, strict=True):
            for pattern in _CELL_FACES[kind]:
                oriented = tuple(cell[i] for i in pattern)
                source_faces.setdefault(tuple(sorted(oriented)), []).append((region, oriented))
        expected_triangles = sum(
            len(split_face(entries[0][1])) for entries in source_faces.values()
            if {region for region, _ in entries} == {1, 2}
        )
        mesh = tetrahedralize_structure(source)
        faces = Counter()
        regions = {}
        for (_, tet), region in zip(mesh.cells, mesh.region_ids):
            for pattern in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                face = tuple(sorted(tet[i] for i in pattern))
                faces[face] += 1
                regions.setdefault(face, set()).add(region)
        self.assertEqual(max(faces.values()), 2)
        interface = {face for face, membership in regions.items() if membership == {1, 2}}
        self.assertEqual(len(interface), expected_triangles)
        exterior = {tuple(sorted(face)) for _, face in mesh.boundaries}
        self.assertFalse(interface & exterior)
        self.assertEqual(exterior, {face for face, count in faces.items() if count == 1})


class RaisedReservoirTests(unittest.TestCase):
    def test_active_load_case_and_generated_pressure_law(self):
        root = assembly.ROOT
        environment = root / "config.json"
        support = load_foundation_support(environment)
        mesh = SimpleNamespace(
            loads=load_environment(environment),
            material=load_material(environment),
            foundation_support=support, plinth_support=support,
            nodes=[(80.0, 0.0, -31.0), (80.0, 0.0, 0.0)],
            boundaries=[],
        )
        payload = json.loads(assembly.LOAD_CASES_PATH.read_text())
        self.assertEqual(payload["loads"]["water_height"], 31)
        self.assertEqual(payload["loads"]["overflow_water_level"], 34)
        assembly.apply_load_case(mesh, assembly.LOAD_CASES_PATH)
        self.assertEqual(mesh.loads.peak_water_pressure_pa, 333540)
        self.assertEqual(mesh.loads.overflow_head_m, 3)
        self.assertEqual(mesh.loads.water_pressure_gradient_pa_per_m, 9810)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.sif"
            assembly.write_combined_solver_input(mesh, path)
            text = path.read_text()
        expression = re.search(r'Real MATC "([^"]+)"', text).group(1)
        for z, pressure in ((-31, -333540), (-30, -323730), (0, -29430), (3, 0)):
            self.assertAlmostEqual(eval(expression, {"__builtins__": {}}, {"tx": z}), pressure)
        self.assertIn("Normal Force = Real -29430", text)
        self.assertNotIn("WallPlinthSpring", text)
        self.assertNotIn("Target Boundaries(1) = 8", text)


class MpiLauncherTests(unittest.TestCase):
    def test_windows_prefers_mpiexec(self):
        environment = SimpleNamespace(name="nt", environ={})
        with patch.object(assembly, "os", environment), patch.object(
            assembly.shutil, "which", side_effect=lambda name: f"{name}.exe"
        ):
            self.assertEqual(assembly.mpi_launcher(), ("mpiexec.exe", "-n"))

    def test_unix_preserves_mpirun(self):
        environment = SimpleNamespace(name="posix", environ={})
        with patch.object(assembly, "os", environment), patch.object(
            assembly.shutil, "which", side_effect=lambda name: f"/bin/{name}"
        ):
            self.assertEqual(assembly.mpi_launcher(), ("/bin/mpirun", "-np"))

    def test_unix_mpiexec_fallback(self):
        environment = SimpleNamespace(name="posix", environ={})
        with patch.object(assembly, "os", environment), patch.object(
            assembly.shutil, "which", side_effect=lambda name: "/bin/mpiexec" if name == "mpiexec" else None
        ):
            self.assertEqual(assembly.mpi_launcher(), ("/bin/mpiexec", "-n"))

    def test_windows_searches_msmpi_bin_without_path(self):
        directory = str(Path("custom-mpi") / "Bin")
        expected = Path(directory) / "mpiexec.exe"
        environment = SimpleNamespace(name="nt", environ={"MSMPI_BIN": directory})
        with patch.object(assembly, "os", environment), patch.object(
            assembly.shutil, "which", return_value=None
        ), patch.object(Path, "is_file", autospec=True, side_effect=lambda path: path == expected):
            self.assertEqual(assembly.mpi_launcher(), (str(expected), "-n"))

    def test_windows_searches_standard_install_without_path(self):
        expected = Path(r"C:\Program Files") / "Microsoft MPI" / "Bin" / "mpiexec.exe"
        environment = SimpleNamespace(name="nt", environ={})
        with patch.object(assembly, "os", environment), patch.object(
            assembly.shutil, "which", return_value=None
        ), patch.object(Path, "is_file", autospec=True, side_effect=lambda path: path == expected):
            self.assertEqual(assembly.mpi_launcher(), (str(expected), "-n"))

    def test_missing_launcher_reports_actionable_error(self):
        environment = SimpleNamespace(name="nt", environ={})
        with patch.object(assembly, "os", environment), patch.object(
            assembly.shutil, "which", return_value=None
        ), patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "MSMPI_BIN"):
                assembly.mpi_launcher()

    def test_missing_solver_fails_before_mesh_build(self):
        with patch.object(assembly.shutil, "which", return_value=None), patch.object(
            assembly, "build_combined_structure"
        ) as build:
            with self.assertRaisesRegex(RuntimeError, "MPI-enabled Elmer"):
                assembly.solve_combined_structure()
            build.assert_not_called()

    def test_solve_passes_launcher_option_and_preserves_elmer_home(self):
        with patch.dict(os.environ, {"ELMER_HOME": "existing-elmer-home"}), patch.object(
            assembly.shutil, "which", return_value=str(Path("Elmer") / "bin" / "ElmerSolver_mpi.exe")
        ), patch.object(assembly, "mpi_launcher", return_value=("mpiexec.exe", "-n")), patch.object(
            assembly, "build_combined_structure"
        ), patch.object(assembly, "apply_load_case"), patch.object(
            assembly, "tetrahedralize_structure"
        ), patch.object(
            assembly, "write_combined_gmsh"
        ), patch.object(assembly, "convert_with_elmergrid", return_value=True), patch.object(
            assembly, "write_combined_solver_input"
        ), patch.object(Path, "mkdir"), patch.object(Path, "is_file", return_value=True), patch.object(
            assembly, "audit_combined_structure", return_value={}
        ), patch.object(assembly.subprocess, "run") as run:
            assembly.solve_combined_structure()
            args, kwargs = run.call_args
            self.assertEqual(args[0][:3], ["mpiexec.exe", "-n", "4"])
            self.assertEqual(kwargs["env"]["ELMER_HOME"], "existing-elmer-home")
            self.assertTrue(kwargs["check"])


if __name__ == "__main__":
    unittest.main()
