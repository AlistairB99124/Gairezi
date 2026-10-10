"""Elmer deformed geometry, boundary-cell and interface-node regressions."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from elastic_parity import read_reference
import test_dam_mesh


class ReferenceMappingTests(unittest.TestCase):
    def test_deformed_elmer_geometry_preserves_separate_region_nodes(self):
        cube = test_dam_mesh.DamMeshTests().cube()
        points = np.vstack((cube.nodes, cube.nodes + [0, 0, 1]))
        structure = SimpleNamespace(nodes=points, cells=[(5, tuple(range(8))), (5, tuple(range(8, 16)))],
                                    region_ids=[1, 2])
        displacement = np.vstack((np.tile([0, 0, -0.001], (8, 1)), np.tile([0.002, 0, 0], (8, 1))))
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points + displacement, deep=True))
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(vtk_points)
        for _, nodes in structure.cells:
            ids = vtk.vtkIdList()
            for node in nodes:
                ids.InsertNextId(node)
            grid.InsertNextCell(vtk.VTK_HEXAHEDRON, ids)
        ids = vtk.vtkIdList()
        for node in (0, 1, 2, 3):
            ids.InsertNextId(node)
        grid.InsertNextCell(vtk.VTK_QUAD, ids)
        array = numpy_to_vtk(displacement, deep=True)
        array.SetName("displacement")
        grid.GetPointData().AddArray(array)
        for name in ("stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_yz", "stress_xz"):
            array = numpy_to_vtk(np.ones(16), deep=True)
            array.SetName(name)
            grid.GetPointData().AddArray(array)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(str(path / "piece.vtu"))
            writer.SetInputData(grid)
            self.assertEqual(writer.Write(), 1)
            reference = path / "result.pvtu"
            reference.write_text('<VTKFile><PUnstructuredGrid><Piece Source="piece.vtu"/></PUnstructuredGrid></VTKFile>')
            actual, stress, regions = read_reference(structure, reference)
        np.testing.assert_allclose(actual, displacement, atol=1e-15)
        np.testing.assert_allclose(stress, 1)
        np.testing.assert_array_equal(regions, [1] * 8 + [2] * 8)
