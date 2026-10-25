"""Stress visualization tests; run with Code_Aster's bundled Python for MED support."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

try:
    from Aster import extract_result_fields as exporter
except ModuleNotFoundError as error:
    if error.name != "MEDLoader":
        raise
    exporter = None


@unittest.skipIf(exporter is None, "Requires Code_Aster's bundled MEDLoader")
class StressExportTests(unittest.TestCase):
    def test_uniaxial_hydrostatic_and_shear_invariants(self):
        stress = np.array([
            [12., 0., 0., 0., 0., 0.],
            [-8., -8., -8., 0., 0., 0.],
            [0., 0., 0., 0., 5., 0.],
            [0., 0., 0., 0., 0., 7.],
        ])
        fields = exporter.stress_scalars(stress)
        np.testing.assert_allclose(fields["principal_min_pa"], [0., -8., -5., -7.])
        np.testing.assert_allclose(fields["principal_mid_pa"], [0., -8., 0., 0.])
        np.testing.assert_allclose(fields["principal_max_pa"], [12., -8., 5., 7.])
        np.testing.assert_allclose(fields["von_mises_pa"], [12., 0., np.sqrt(75), np.sqrt(147)])
        for i, component in enumerate(("xx", "yy", "zz", "xy", "xz", "yz")):
            np.testing.assert_array_equal(fields[f"stress_{component}_pa"], stress[:, i])

    def test_invalid_stress_is_rejected(self):
        for stress in (np.zeros((3, 5)), np.zeros(6), np.full((4, 6), np.nan),
                       np.full((4, 6), np.inf)):
            with self.subTest(shape=stress.shape):
                with self.assertRaisesRegex(ValueError, "finite nodal stresses"):
                    exporter.stress_scalars(stress)

    def test_vtu_contains_scalar_point_fields_on_original_mesh(self):
        ml = exporter.ml
        mesh = ml.MEDCouplingUMesh.New("stress_support", 3)
        mesh.setCoords(ml.DataArrayDouble(np.array([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        ])))
        mesh.allocateCells(1)
        mesh.insertNextCell(ml.NORM_TETRA4, [0, 1, 2, 3])
        mesh.finishInsertingCells()
        sigm = ml.MEDCouplingFieldDouble(ml.ON_NODES, ml.ONE_TIME)
        sigm.setMesh(mesh)
        sigm.setArray(ml.DataArrayDouble(np.tile([12., 0., 0., 0., 0., 0.], (4, 1))))
        sigm.setTime(1., 1, 1)
        with tempfile.TemporaryDirectory(suffix=".aster") as directory:
            target = Path(directory) / "result_stress.vtu"
            self.assertEqual(exporter.export_stress_vtu(sigm, target), target)
            # MEDCoupling appends raw binary after the XML metadata.
            header = target.read_bytes().split(b"<AppendedData", 1)[0]
            root = ET.fromstring(header + b"</VTKFile>")
            piece = root.find(".//Piece")
            self.assertEqual(piece.attrib["NumberOfPoints"], "4")
            self.assertEqual(piece.attrib["NumberOfCells"], "1")
            arrays = piece.findall("./PointData/DataArray")
            self.assertEqual({array.attrib["Name"] for array in arrays},
                             set(exporter.stress_scalars(np.zeros((4, 6)))))
            self.assertTrue(all(array.attrib.get("NumberOfComponents", "1") == "1" for array in arrays))

    def test_extract_excludes_reference_cells_and_preserves_coincident_sides(self):
        ml = exporter.ml
        tetra = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        mesh = ml.MEDCouplingUMesh.New("test_mesh", 3)
        mesh.setCoords(ml.DataArrayDouble(np.vstack([tetra, tetra, tetra + 20.])))
        mesh.allocateCells(3)
        for i in range(3):
            mesh.insertNextCell(ml.NORM_TETRA4, list(range(i * 4, (i + 1) * 4)))
        mesh.finishInsertingCells()
        med = ml.MEDFileUMesh.New()
        med.setMeshAtLevel(0, mesh)
        group = ml.DataArrayInt.New([0, 1])
        group.setName("CONCRETE")
        other = ml.DataArrayInt.New([2])
        other.setName("BEDROCK")
        med.setGroupsAtLevel(0, [group, other])
        with tempfile.TemporaryDirectory(suffix=".aster") as directory:
            path = Path(directory) / "result.med"
            archive = Path(directory) / "fields.npz"
            med.write(str(path), 2)
            for name, values in (
                ("DEPL", np.zeros((12, 3))),
                ("SIGM_NOEU", np.column_stack([
                    np.repeat([1e6, 2e6, 99e6], 4), np.zeros((12, 5)),
                ])),
            ):
                field = ml.MEDCouplingFieldDouble(ml.ON_NODES, ml.ONE_TIME)
                field.setName(exporter.RESULT_NAME_PREFIX + name)
                field.setMesh(mesh)
                field.setArray(ml.DataArrayDouble(values))
                field.setTime(0., 1, 1)
                ml.WriteFieldUsingAlreadyWrittenMesh(str(path), field)
            result = exporter.extract(path, archive)
            self.assertEqual(result["reporting_scope"], "concrete")
            with np.load(archive) as data:
                self.assertEqual(data["points_m"].shape, (8, 3))
                np.testing.assert_array_equal(data["points_m"][:4], data["points_m"][4:])
                np.testing.assert_array_equal(data["stress_pa"][:, 0], np.repeat([1e6, 2e6], 4))


if __name__ == "__main__":
    unittest.main()
