"""Elmer handover checks using real VTU partitions without a fracture solve."""
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elmer_baseline import baseline_files, import_baseline, read_reference


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.points = np.array([
            (0, 0, -26), (1, 0, -26), (1, 1, -26), (0, 1, -26),
            (0, 0, -25), (1, 0, -25), (1, 1, -25), (0, 1, -25),
        ], dtype=float)
        # Duplicate interface coordinates must remain separate for a spring-supported seat.
        self.points = np.vstack((self.points, self.points + [0, 0, 1]))
        self.structure = SimpleNamespace(
            nodes=self.points, cells=[(5, tuple(range(8))), (5, tuple(range(8, 16)))],
            region_ids=[1, 2],
        )
        self.displacement = np.vstack((np.tile([0, 0, 0.1], (8, 1)), np.tile([0.2, 0, 0], (8, 1))))
        self.config = self.root / "config.json"
        self.config.write_text('{"test": true}')
        self.reference = self.root / "result.pvtu"
        self.write_pieces()

    def write_pieces(self, missing_stress=False, nonfinite=False, shift=0, tetrahedral=False):
        for index in range(2):
            ids = slice(index * 8, (index + 1) * 8)
            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(self.points[ids] + self.displacement[ids] + shift, deep=True))
            grid = vtk.vtkUnstructuredGrid()
            grid.SetPoints(points)
            if tetrahedral:
                from Elmer.regions.tetrahedral import split_cell
                cells = split_cell(self.points[ids], 5, tuple(range(8)))
            else:
                cells = [tuple(range(8))]
            for cell in cells:
                nodes = vtk.vtkIdList()
                for node in cell:
                    nodes.InsertNextId(node)
                grid.InsertNextCell(vtk.VTK_TETRA if tetrahedral else vtk.VTK_HEXAHEDRON, nodes)
            array = numpy_to_vtk(self.displacement[ids], deep=True)
            array.SetName("displacement")
            grid.GetPointData().AddArray(array)
            names = ("stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_yz", "stress_xz")
            for name in names:
                if missing_stress and name == "stress_xz":
                    continue
                value = (index + 1) * 10.0 if name == "stress_xx" else 0.0
                values = np.full(8, value)
                if nonfinite and name == "stress_xx":
                    values[0] = np.nan
                array = numpy_to_vtk(values, deep=True)
                array.SetName(name)
                grid.GetPointData().AddArray(array)
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(str(self.root / f"piece{index}.vtu"))
            writer.SetInputData(grid)
            self.assertEqual(writer.Write(), 1)
        self.reference.write_text(
            '<VTKFile><PUnstructuredGrid><Piece Source="piece0.vtu"/>'
            '<Piece Source="piece1.vtu"/></PUnstructuredGrid></VTKFile>'
        )

    def test_real_partition_import_preserves_fields_coordinates_and_peak(self):
        result = import_baseline(self.structure, self.reference, self.config, self.root)
        self.assertEqual(result["max_principal_stress"]["value_pa"], 20.0)
        self.assertEqual(result["max_principal_stress"]["region_id"], 2)
        self.assertEqual(result["max_principal_stress"]["centroid_xyz_m"], [0.5, 0.5, -24.5])
        self.assertEqual(len(result["source_sha256"]), 3)
        self.assertIn("not fracture initial conditions", result["usage"])
        with np.load(self.root / result["field_archive"]) as fields:
            np.testing.assert_allclose(fields["coordinates_m"], self.points)
            np.testing.assert_allclose(fields["displacement_m"], self.displacement)
            np.testing.assert_array_equal(fields["node_regions"], [1] * 8 + [2] * 8)
        self.assertEqual(json.loads((self.root / "elmer_baseline.json").read_text()), result)

    def test_tetrahedral_solver_results_require_the_same_source_topology(self):
        from Elmer.regions.tetrahedral import split_cell

        self.write_pieces(tetrahedral=True)
        with self.assertRaisesRegex(ValueError, "cells do not match"):
            import_baseline(self.structure, self.reference, self.config, self.root)
        self.structure.cells = [
            (4, tetrahedron)
            for _, cell in self.structure.cells
            for tetrahedron in split_cell(self.points, 5, cell)
        ]
        self.structure.region_ids = [1] * 6 + [2] * 6
        report = import_baseline(self.structure, self.reference, self.config, self.root)
        self.assertEqual(report["element_count"], 12)
        self.assertEqual(report["max_principal_stress"]["value_pa"], 20)
        with np.load(self.root / report["field_archive"]) as fields:
            np.testing.assert_allclose(fields["coordinates_m"], self.points)
            np.testing.assert_allclose(fields["displacement_m"], self.displacement)
            np.testing.assert_array_equal(fields["node_regions"], [1] * 8 + [2] * 8)

    def test_missing_piece_is_rejected(self):
        (self.root / "piece1.vtu").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            import_baseline(self.structure, self.reference, self.config, self.root)

    def test_duplicate_partition_does_not_double_count_elements(self):
        text = self.reference.read_text().replace(
            "</PUnstructuredGrid>", '<Piece Source="piece1.vtu"/></PUnstructuredGrid>',
        )
        self.reference.write_text(text)
        report = import_baseline(self.structure, self.reference, self.config, self.root)
        self.assertEqual(report["element_count"], 2)
        self.assertEqual(report["max_principal_stress"]["value_pa"], 20)

    def test_inconsistent_ghost_stress_is_rejected(self):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(self.root / "piece1.vtu"))
        reader.Update()
        grid = reader.GetOutput()
        grid.GetPointData().GetArray("stress_xx").SetComponent(0, 0, 999)
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(self.root / "ghost.vtu"))
        writer.SetInputData(grid)
        self.assertEqual(writer.Write(), 1)
        self.reference.write_text(self.reference.read_text().replace(
            "</PUnstructuredGrid>", '<Piece Source="ghost.vtu"/></PUnstructuredGrid>',
        ))
        with self.assertRaisesRegex(ValueError, "ghost-node stress"):
            import_baseline(self.structure, self.reference, self.config, self.root)

    def test_no_tension_is_reported_without_inventing_a_tensile_peak(self):
        for index in range(2):
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(str(self.root / f"piece{index}.vtu"))
            reader.Update()
            grid = reader.GetOutput()
            for name in ("stress_xx", "stress_yy", "stress_zz"):
                grid.GetPointData().GetArray(name).FillComponent(0, -10.0)
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(str(self.root / f"piece{index}.vtu"))
            writer.SetInputData(grid)
            self.assertEqual(writer.Write(), 1)
        report = import_baseline(self.structure, self.reference, self.config, self.root)
        self.assertFalse(report["has_tension"])
        self.assertEqual(report["max_principal_stress"]["value_pa"], -10)

    def test_escaping_piece_is_rejected(self):
        self.reference.write_text('<VTKFile><Piece Source="../outside.vtu"/></VTKFile>')
        with self.assertRaisesRegex(ValueError, "escapes"):
            baseline_files(self.reference)

    def test_empty_wrapper_is_rejected(self):
        self.reference.write_text("<VTKFile/>")
        with self.assertRaisesRegex(ValueError, "no partition"):
            baseline_files(self.reference)

    def test_empty_source_mesh_is_rejected(self):
        self.structure.cells = []
        with self.assertRaisesRegex(ValueError, "no source volume"):
            import_baseline(self.structure, self.reference, self.config, self.root)

    def test_files_changing_during_read_are_rejected(self):
        def changing_reader(structure, reference):
            result = read_reference(structure, reference)
            reference.write_text(reference.read_text() + "\n")
            return result

        with patch("elmer_baseline.read_reference", side_effect=changing_reader):
            with self.assertRaisesRegex(ValueError, "changed during import"):
                import_baseline(self.structure, self.reference, self.config, self.root)
        self.assertFalse((self.root / "elmer_baseline_fields.npz").exists())

    def test_missing_stress_is_rejected(self):
        self.write_pieces(missing_stress=True)
        with self.assertRaisesRegex(ValueError, "missing stress"):
            import_baseline(self.structure, self.reference, self.config, self.root)

    def test_nonfinite_stress_is_rejected(self):
        self.write_pieces(nonfinite=True)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            import_baseline(self.structure, self.reference, self.config, self.root)

    def test_wrong_mesh_is_rejected(self):
        self.write_pieces(shift=10)
        with self.assertRaisesRegex(ValueError, "do not match"):
            import_baseline(self.structure, self.reference, self.config, self.root)


if __name__ == "__main__":
    unittest.main()
