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

    def test_generated_wall_is_one_body_seated_on_bedrock_before_and_after_tetrahedralization(self):
        from collections import Counter
        from Elmer.regions.bedrock_contours import bedrock_elevation, load_model_contours
        from Elmer.regions.combined_structure import audit_combined_structure, build_combined_structure
        from Elmer.regions.tetrahedral import tetrahedralize_structure
        import math

        source = build_combined_structure(assembly.ROOT)
        audit = audit_combined_structure(source)
        self.assertEqual(audit["face_connected_components"], 1)
        self.assertEqual(audit["wall_connection"], "single_monolithic_body")
        self.assertGreater(audit["foundation_nodes"], 0)
        self.assertEqual(set(source.region_ids), {1})
        self.assertEqual(len({tuple(round(x, 9) for x in xyz) for xyz in source.nodes}), len(source.nodes))
        self.assertFalse(any(tag > 7 for tag, _ in source.boundaries))
        # Every foundation face sits on the piecewise-linear bedrock surface.
        rows = [{"chainage": p.chainage_m, "height": p.bedrock_z_m} for p in load_model_contours(assembly.ROOT)]
        seat_nodes = {node for tag, face in source.boundaries if tag == 1 for node in face}
        self.assertTrue(seat_nodes)
        for node in seat_nodes:
            x, y, z = source.nodes[node]
            chainage = 78.0 * math.atan2(x, y)
            self.assertLess(abs(z - bedrock_elevation(rows, chainage)), 1e-6)
        self.assertAlmostEqual(min(z for _, _, z in source.nodes), -29.0)
        mesh = tetrahedralize_structure(source)
        faces = Counter()
        for _, tet in mesh.cells:
            for pattern in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                faces[tuple(sorted(tet[i] for i in pattern))] += 1
        self.assertEqual(max(faces.values()), 2)
        self.assertEqual(set(mesh.region_ids), {1})
        exterior = {tuple(sorted(face)) for _, face in mesh.boundaries}
        self.assertEqual(exterior, {face for face, count in faces.items() if count == 1})


class ReservoirLevelTests(unittest.TestCase):
    """Upstream water surface = deepest seat (z = -height_m) + water_height + overflow_head."""

    def generated_sif(self, water_height, overflow_head, tailwater_head=0.0):
        payload = json.loads(assembly.CONFIG_PATH.read_text(encoding="utf-8-sig"))
        self.assertEqual(min(row["height"] for row in payload["geometry"]["bedrock"]["contours"]), -29.0)
        payload["loads"].update(
            water_height=water_height, overflow_head=overflow_head, tailwater_head=tailwater_head,
            overflow_water_level=water_height + overflow_head,
        )
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps(payload))
            mesh = SimpleNamespace(nodes=[(80.0, 0.0, -29.0), (80.0, 0.0, 0.0)], boundaries=[])
            assembly.apply_load_case(mesh, config)
            path = Path(directory) / "model.sif"
            assembly.write_combined_solver_input(mesh, path)
            text = path.read_text()
        upstream = re.search(r'Real MATC "([^"]+)"', text).group(1)
        crest = float(re.search(r"Normal Force = Real (\S+)", text).group(1))
        return mesh.loads, (lambda z: eval(upstream, {"__builtins__": {}}, {"tx": z})), crest

    def test_overflowing_reservoir_matches_legacy_crest_datum(self):
        loads, upstream, crest = self.generated_sif(29.0, 3.0)
        self.assertEqual(loads.upstream_surface_z_m, 3.0)
        self.assertEqual(loads.peak_water_pressure_pa, 9810 * 32)
        for z, pressure in ((-29, -313920), (0, -29430), (3, 0), (4, 0)):
            self.assertAlmostEqual(upstream(z), pressure)
        self.assertAlmostEqual(crest, -29430)

    def test_partial_reservoir_has_no_crest_surcharge(self):
        loads, upstream, crest = self.generated_sif(20.0, 0.0)
        self.assertEqual(loads.upstream_surface_z_m, -9.0)
        for z, pressure in ((-29, -196200), (-9, 0), (0, 0)):
            self.assertAlmostEqual(upstream(z), pressure)
        self.assertEqual(crest, 0.0)

    def test_empty_reservoir_applies_no_water_pressure(self):
        loads, upstream, crest = self.generated_sif(0.0, 0.0)
        self.assertEqual(loads.upstream_surface_z_m, -29.0)
        for z in (-29, -10, 0):
            self.assertEqual(upstream(z), 0)
        self.assertEqual(crest, 0.0)


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
