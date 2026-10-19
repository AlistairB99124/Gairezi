"""Run with pvpython; no solver or repository geometry needed."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

from Elmer.stress_reporting_region import (
    DISTANCE_ARRAY, export_region, reference_coordinates, reporting_region, select_body, stress_summary,
)


def fixture():
    mesh = vtkUnstructuredGrid()
    points = vtkPoints()
    labels = []
    # Two concrete cubes; rock tetrahedra inside, straddling and outside 5 m.
    for body, x, z in ((1, 0., 0.), (2, 0., 1.), (3, 5.8, 0.), (3, 6.2, 0.),
                       (3, 5.9, 0.), (3, 0., -30.), (3, 4.9, 5.9)):
        start = points.GetNumberOfPoints()
        if body < 3:
            offsets = ((0,0,0),(1,0,0),(1,1,0),(0,1,0),
                       (0,0,1),(1,0,1),(1,1,1),(0,1,1))
            kind = 12
        else:
            offsets = ((0,0,0),(.4,0,0),(0,.2,0),(0,0,.2))
            kind = 10
        ids = vtkIdList()
        for dx, dy, dz in offsets:
            points.InsertNextPoint(x+dx, dy, z+dz)
            ids.InsertNextId(start + ids.GetNumberOfIds())
        mesh.InsertNextCell(kind, ids)
        labels.append(body)
    mesh.SetPoints(points)
    array = numpy_to_vtk(np.asarray(labels, dtype=np.int32), deep=True)
    array.SetName("GeometryIds")
    mesh.GetCellData().AddArray(array)
    for name, values in (
        ("displacement", np.zeros((points.GetNumberOfPoints(), 3))),
        ("stress_xx", np.full(points.GetNumberOfPoints(), 2e6)),
        ("stress_yy", np.full(points.GetNumberOfPoints(), -3e6)),
        ("stress_zz", np.zeros(points.GetNumberOfPoints())),
        ("stress_xy", np.zeros(points.GetNumberOfPoints())),
        ("stress_xz", np.zeros(points.GetNumberOfPoints())),
        ("stress_yz", np.zeros(points.GetNumberOfPoints())),
    ):
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        mesh.GetPointData().AddArray(array)
    return mesh


class StressReportingRegionTests(unittest.TestCase):
    def test_nearest_surface_not_bounds_or_centroid_and_no_remote_bottom(self):
        mesh = fixture()
        result = reporting_region(mesh)
        for body in (1, 2):
            before, after = select_body(mesh, body, body), select_body(result, body, body)
            self.assertEqual(before.GetNumberOfCells(), after.GetNumberOfCells())
            np.testing.assert_array_equal(reference_coordinates(before), reference_coordinates(after))
            for field in ("stress_xx", "stress_yy", "stress_zz", "displacement"):
                np.testing.assert_array_equal(
                    vtk_to_numpy(before.GetPointData().GetArray(field)),
                    vtk_to_numpy(after.GetPointData().GetArray(field)),
                )
        rock = select_body(result, 3, 3)
        points = reference_coordinates(rock)
        self.assertGreater(rock.GetNumberOfCells(), 0)
        self.assertGreaterEqual(points[:, 2].min(), 0.)
        self.assertLess(points[:, 2].max(), 1.)
        self.assertLessEqual(points[:, 0].max(), 6.000001)
        self.assertGreater(points[:, 0].max(), 5.99)
        self.assertLessEqual(vtk_to_numpy(rock.GetPointData().GetArray(DISTANCE_ARRAY)).max(), 5.000001)
        self.assertEqual(mesh.GetNumberOfCells(), 7)
        self.assertIsNone(mesh.GetPointData().GetArray(DISTANCE_ARRAY))
        summary = stress_summary(result)
        for body in summary.values():
            self.assertAlmostEqual(body["peak_tension_mpa"], 2.)
            self.assertAlmostEqual(body["peak_compression_mpa"], 3.)

    def test_invalid_distance_and_missing_body_fail(self):
        for distance in (0., -1., float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "distance"):
                reporting_region(fixture(), distance)
        with self.assertRaisesRegex(ValueError, "Expected attributed"):
            reporting_region(select_body(fixture(), 1, 2))

    def test_selection_uses_undeformed_coordinates(self):
        mesh = fixture()
        original = reporting_region(mesh)
        displacement = mesh.GetPointData().GetArray("displacement")
        values = vtk_to_numpy(displacement)
        coordinates = vtk_to_numpy(mesh.GetPoints().GetData())
        # Nonuniform body translations must not change reference-distance selection.
        rock = coordinates[:, 0] > 2.
        values[rock, 0] = 10.
        coordinates[:] += values
        shifted = reporting_region(mesh)
        self.assertEqual(original.GetNumberOfCells(), shifted.GetNumberOfCells())
        np.testing.assert_allclose(
            reference_coordinates(original), reference_coordinates(shifted), atol=1e-6,
        )

    def test_export_and_readback_preserve_summary_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.vtu"
            writer = vtkXMLUnstructuredGridWriter()
            writer.SetFileName(str(source))
            writer.SetInputData(fixture())
            self.assertEqual(writer.Write(), 1)
            original = source.read_bytes()
            report = export_region(source, root / "report")
            self.assertTrue(Path(report["result"]).is_file())
            self.assertEqual(report["rock_reporting_distance_m"], 5.)
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "empty"):
                export_region(source, root / "report")


if __name__ == "__main__":
    unittest.main()
