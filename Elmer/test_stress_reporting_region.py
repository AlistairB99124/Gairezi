"""Run with pvpython; no solver or repository geometry needed."""
import math
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

from Elmer.stress_reporting_region import (
    DISTANCE_ARRAY, EDGE_DISTANCE_ARRAY, add_edge_distance, export_region, reference_coordinates,
    reporting_region, select_body, stress_summary,
)


def block_fixture():
    """A 4 m wall cube of 1 m hexes (shared nodes) plus one remote rock tet."""
    mesh = vtkUnstructuredGrid()
    points = vtkPoints()
    index = {}
    for k in range(5):
        for j in range(5):
            for i in range(5):
                index[i, j, k] = points.InsertNextPoint(i, j, k)
    labels = []
    for k in range(4):
        for j in range(4):
            for i in range(4):
                ids = vtkIdList()
                for di, dj, dk in ((0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)):
                    ids.InsertNextId(index[i+di, j+dj, k+dk])
                mesh.InsertNextCell(12, ids)
                labels.append(1)
    ids = vtkIdList()
    for offset in ((10,0,0),(10.4,0,0),(10,.2,0),(10,0,.2)):
        ids.InsertNextId(points.InsertNextPoint(*offset))
    mesh.InsertNextCell(10, ids)
    labels.append(2)
    mesh.SetPoints(points)
    array = numpy_to_vtk(np.asarray(labels, dtype=np.int32), deep=True)
    array.SetName("GeometryIds")
    mesh.GetCellData().AddArray(array)
    count = points.GetNumberOfPoints()
    xx = np.full(count, 1e6)
    xx[index[0, 0, 0]] = 9e6
    xx[index[2, 2, 2]] = 3e6
    yy = np.full(count, -1e6)
    yy[index[4, 0, 2]] = -12e6
    yy[index[2, 2, 1]] = -4e6
    for name, values in (("displacement", np.zeros((count, 3))), ("stress_xx", xx), ("stress_yy", yy),
                         ("stress_zz", np.zeros(count)), ("stress_xy", np.zeros(count)),
                         ("stress_xz", np.zeros(count)), ("stress_yz", np.zeros(count))):
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        mesh.GetPointData().AddArray(array)
    return mesh, index


def fixture():
    mesh = vtkUnstructuredGrid()
    points = vtkPoints()
    labels = []
    # Two wall cubes; rock tetrahedra inside, straddling and outside 5 m.
    for body, x, z in ((1, 0., 0.), (1, 0., 1.), (2, 5.8, 0.), (2, 6.2, 0.),
                       (2, 5.9, 0.), (2, 0., -30.), (2, 4.9, 5.9)):
        start = points.GetNumberOfPoints()
        if body < 2:
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
    def test_only_concrete_is_retained_with_original_fields(self):
        mesh = fixture()
        result = reporting_region(mesh)
        for body in (1,):
            before, after = select_body(mesh, body, body), select_body(result, body, body)
            self.assertEqual(before.GetNumberOfCells(), after.GetNumberOfCells())
            np.testing.assert_array_equal(reference_coordinates(before), reference_coordinates(after))
            for field in ("stress_xx", "stress_yy", "stress_zz", "displacement"):
                np.testing.assert_array_equal(
                    vtk_to_numpy(before.GetPointData().GetArray(field)),
                    vtk_to_numpy(after.GetPointData().GetArray(field)),
                )
        rock = select_body(result, 2, 2)
        self.assertEqual(rock.GetNumberOfCells(), 0)
        self.assertEqual(result.GetNumberOfCells(), 2)
        self.assertEqual(mesh.GetNumberOfCells(), 7)
        self.assertIsNone(mesh.GetPointData().GetArray(DISTANCE_ARRAY))
        summary = stress_summary(result)
        for body in summary.values():
            self.assertAlmostEqual(body["peak_tension_mpa"], 2.)
            self.assertAlmostEqual(body["peak_compression_mpa"], 3.)

    def test_foundation_is_retained_and_peaks_stay_separate(self):
        mesh = fixture()
        vtk_to_numpy(mesh.GetCellData().GetArray("GeometryIds"))[0] = 3
        vtk_to_numpy(mesh.GetPointData().GetArray("stress_xx"))[:8] = 8e6
        region = reporting_region(mesh)
        summary = stress_summary(region)
        self.assertEqual(set(summary), {"wall", "foundation"})
        self.assertEqual(select_body(region, 3, 3).GetNumberOfCells(), 1)
        self.assertAlmostEqual(summary["foundation"]["peak_tension_mpa"], 8.)
        self.assertAlmostEqual(summary["wall"]["peak_tension_mpa"], 2.)

    def test_foundation_gap_pair_names_follow_present_bodies(self):
        from vtkmodules.vtkIOParallelXML import vtkXMLPUnstructuredGridWriter
        from Elmer.measure_joint_gaps import measure
        from vtkmodules.vtkFiltersCore import vtkAppendFilter
        mesh = fixture()
        vtk_to_numpy(mesh.GetCellData().GetArray("GeometryIds"))[0] = 3
        joined = vtkAppendFilter()
        joined.MergePointsOff()
        joined.AddInputData(mesh)
        for shift in (.002, .001):
            copy = vtkUnstructuredGrid()
            copy.DeepCopy(select_body(mesh, 1, 1))
            vtk_to_numpy(copy.GetPointData().GetArray("displacement"))[:, 0] = shift
            vtk_to_numpy(copy.GetPoints().GetData())[:, 0] += shift
            joined.AddInputData(copy)
        joined.Update()
        mesh = joined.GetOutput()
        root = Path(__file__).resolve().parents[1] / "Thickness_tests"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="foundation-gaps-", dir=root) as directory:
            path = Path(directory) / "result.pvtu"
            writer = vtkXMLPUnstructuredGridWriter()
            writer.SetFileName(str(path))
            writer.SetInputData(mesh)
            writer.SetNumberOfPieces(1)
            self.assertEqual(writer.Write(), 1)
            result = measure(path)
        self.assertEqual(set(result["joints"]), {"wall_foundation", "foundation_bedrock"})
        self.assertEqual(result["joints"]["wall_foundation"]["locations"], 12)
        self.assertAlmostEqual(result["joints"]["wall_foundation"]["max_abs_gap_m"], .002)

    def test_invalid_distance_and_missing_body_fail(self):
        for distance in (0., -1., float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "distance"):
                reporting_region(fixture(), distance)
        self.assertEqual(reporting_region(select_body(fixture(), 1, 1)).GetNumberOfCells(), 2)
        with self.assertRaisesRegex(ValueError, "wall"):
            reporting_region(select_body(fixture(), 2, 2))

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
            self.assertEqual(report["reporting_scope"], "concrete")
            self.assertEqual(report["sharp_edge_exclusion"]["distance_m"], .5)
            for body in report["bodies"].values():
                self.assertIn("away_from_sharp_edges", body)
            self.assertGreater(report["bodies"]["wall"]["away_from_sharp_edges"]["excluded_points"], 0)
            self.assertNotIn("bedrock", json.dumps(report).lower())
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "empty"):
                export_region(source, root / "report")


class SharpEdgeExclusionTests(unittest.TestCase):
    def test_edge_distance_uses_box_edges_not_coplanar_cell_edges(self):
        mesh, index = block_fixture()
        add_edge_distance(mesh)
        distance = vtk_to_numpy(mesh.GetPointData().GetArray(EDGE_DISTANCE_ARRAY))
        self.assertAlmostEqual(distance[index[0, 0, 0]], 0.)
        self.assertAlmostEqual(distance[index[2, 0, 0]], 0.)
        # Face centre: internal 1 m quad edges on a flat face are not sharp.
        self.assertAlmostEqual(distance[index[2, 2, 0]], 2.)
        self.assertAlmostEqual(distance[index[2, 2, 2]], math.sqrt(8.))
        self.assertGreater(distance[-1], 5.)

    def test_summary_keeps_raw_peaks_and_adds_edge_excluded_peaks(self):
        mesh, index = block_fixture()
        plain = stress_summary(mesh)
        self.assertNotIn("away_from_sharp_edges", plain["wall"])
        with self.assertRaisesRegex(ValueError, EDGE_DISTANCE_ARRAY):
            stress_summary(mesh, .5)
        add_edge_distance(mesh)
        wall = stress_summary(mesh, .5)["wall"]
        self.assertAlmostEqual(wall["peak_tension_mpa"], 9.)
        self.assertAlmostEqual(wall["peak_compression_mpa"], 12.)
        away = wall["away_from_sharp_edges"]
        self.assertAlmostEqual(away["peak_tension_mpa"], 3.)
        self.assertAlmostEqual(away["peak_compression_mpa"], 4.)
        self.assertEqual(away["tension_reference_xyz_m"], [2., 2., 2.])
        # 125 nodes, 27 interior, 54 face-interior (all >= 1 m from edges): 44 on edges.
        self.assertEqual(away["excluded_points"], 44)
        # Nodes exactly one exclusion distance away are kept.
        self.assertEqual(stress_summary(mesh, 1.)["wall"]["away_from_sharp_edges"]["excluded_points"], 44)
        # Face rings (48 nodes at 1 m) and interior nodes at sqrt(2) m (20) join the edges.
        self.assertEqual(stress_summary(mesh, 1.5)["wall"]["away_from_sharp_edges"]["excluded_points"], 112)
        none_left = stress_summary(mesh, 10.)["wall"]["away_from_sharp_edges"]
        self.assertIsNone(none_left["peak_tension_mpa"])
        for bad in (0., -1., float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "exclusion"):
                stress_summary(mesh, bad)


if __name__ == "__main__":
    unittest.main()
