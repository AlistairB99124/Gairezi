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
        self.check_mapping(bonded=False)

    def test_deformed_elmer_geometry_preserves_bonded_shared_nodes(self):
        self.check_mapping(bonded=True)

    def test_jointed_wall_blocks_keep_coincident_same_region_copies(self):
        # Spring-jointed thickness blocks are both wall (region 1); only the owning cell
        # tells the coincident joint copies apart, even when Elmer reorders its points.
        self.check_mapping(bonded=False, regions=(1, 1), permute=True)

    def test_foundation_keeps_distinct_nodes_on_both_interfaces(self):
        self.check_mapping(bonded=False, regions=(1, 3), permute=True)
        self.check_mapping(bonded=False, regions=(3, 2), permute=True)

    def check_mapping(self, bonded, regions=(1, 2), permute=False):
        cube = test_dam_mesh.DamMeshTests().cube()
        points = np.vstack((cube.nodes, cube.nodes + [0, 0, 1]))
        structure = SimpleNamespace(nodes=points, cells=[(5, tuple(range(8))), (5, tuple(range(8, 16)))],
                                    region_ids=list(regions))
        displacement = np.vstack((np.tile([0, 0, -0.001], (8, 1)), np.tile([0.002, 0, 0], (8, 1))))
        if bonded:
            points = points[np.r_[0:8, 12:16]]
            structure.nodes = points
            structure.cells[1] = (5, (4, 5, 6, 7, 8, 9, 10, 11))
            displacement = points * 0.001 + [0, 0, -0.001]
        order = np.arange(len(points))[::-1] if permute else np.arange(len(points))
        position = np.argsort(order)
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk((points + displacement)[order], deep=True))
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(vtk_points)
        for _, nodes in structure.cells:
            ids = vtk.vtkIdList()
            for node in nodes:
                ids.InsertNextId(int(position[node]))
            grid.InsertNextCell(vtk.VTK_HEXAHEDRON, ids)
        ids = vtk.vtkIdList()
        for node in (0, 1, 2, 3):
            ids.InsertNextId(int(position[node]))
        grid.InsertNextCell(vtk.VTK_QUAD, ids)
        array = numpy_to_vtk(displacement[order], deep=True)
        array.SetName("displacement")
        grid.GetPointData().AddArray(array)
        for name in ("stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_yz", "stress_xz"):
            array = numpy_to_vtk(np.ones(len(points)), deep=True)
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
        expected_regions = ([1] * 4 + [3] * 4 + [2] * 4 if bonded
                            else [1 << (regions_id - 1) for regions_id in structure.region_ids for _ in range(8)])
        np.testing.assert_array_equal(regions, expected_regions)
