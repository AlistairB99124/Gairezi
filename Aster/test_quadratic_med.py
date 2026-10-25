"""Run with the native bundled Python/MED interpreter, not Windows .venv."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from Aster.quadratic_mesh import elevate
from Aster.quadratic_surface import surface_blocks

NATIVE_MED_AVAILABLE = (
    Path(sys.executable).parent.parent / "medcoupling-9.11.0" / "lib"
    / "python3.10" / "site-packages" / "medcoupling.py"
).is_file()
if NATIVE_MED_AVAILABLE:
    from Aster.convert_quadratic_to_med import convert, mc


@unittest.skipUnless(NATIVE_MED_AVAILABLE, "Requires Code_Aster's bundled MED interpreter")
class QuadraticMedTests(unittest.TestCase):
    def test_mixed_order_roundtrip_keeps_source_cells_body_groups_and_surface_orders(self):
        tetra = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]])
        nodes = np.vstack([tetra, tetra, tetra])
        cells = np.arange(12).reshape(3, 4)
        elevated = elevate(nodes, cells, cells[:, :3], np.array([True, False, False]))
        pairs, matrices = surface_blocks(
            elevated["nodes"], elevated["tri6"][:1], np.diag([2., 3., 4.]), cells[1:2, :3],
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive, output = directory / "mesh.npz", directory / "mesh.med"
            np.savez_compressed(archive, **elevated, body_ids=np.array([1, 3, 2]),
                                face_tags=np.array([1, 2, 3]),
                                surface_block_pairs=pairs, surface_block_matrices_n_per_m=matrices)
            convert(archive, output)
            native = mc.MEDFileUMesh.New(str(output))
            volume = native.getMeshAtLevel(0)
            self.assertEqual(volume.getNumberOfCells(), 3)
            self.assertEqual(volume.getNumberOfNodes(), 18)
            for name, row in (("WALL", 0), ("FOUNDATION", 1), ("REFERENCE", 2)):
                ids = native.getGroupArr(0, name).toNumPyArray()
                self.assertEqual(len(ids), 1)
                expected = elevated["tetra10"][row]
                np.testing.assert_array_equal(volume.getNodeIdsOfCell(int(ids[0])), expected[expected >= 0])
                face_ids = native.getGroupArr(-1, f"FACE_{row + 1}").toNumPyArray()
                expected = elevated["tri6"][row]
                np.testing.assert_array_equal(
                    native.getMeshAtLevel(-1).getNodeIdsOfCell(int(face_ids[0])), expected[expected >= 0],
                )
            self.assertEqual(len(native.getGroupArr(0, "VOLUMES")), 3)

    def test_native_topology_families_and_characteristics_preserve_every_identity(self):
        tetra = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]])
        nodes = np.vstack([tetra, tetra, tetra])
        cells = np.arange(12).reshape(3, 4)
        faces = cells[:, :3]
        elevated = elevate(nodes, cells, faces)
        pairs, matrices = surface_blocks(
            elevated["nodes"], elevated["tri6"][:1], np.diag([2., 3., 4.]),
            elevated["tri6"][1:2],
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive, output = directory / "mesh.npz", directory / "mesh.med"
            np.savez_compressed(archive, **elevated, body_ids=np.array([1, 3, 2]),
                                face_tags=np.array([1, 2, 3]),
                                surface_block_pairs=pairs, surface_block_matrices_n_per_m=matrices)
            report = convert(archive, output)
            self.assertEqual(report["volume_cells"], 3)
            native = mc.MEDFileUMesh.New(str(output))
            for name, cell in (("WALL", 0), ("FOUNDATION", 1), ("REFERENCE", 2)):
                ids = native.getGroupArr(0, name).toNumPyArray()
                np.testing.assert_array_equal(ids, [cell])
            np.testing.assert_array_equal(native.getGroupArr(0, "VOLUMES").toNumPyArray(), [0, 1, 2])
            np.testing.assert_array_equal(native.getMeshAtLevel(0).getCoords().toNumPyArray(), elevated["nodes"])
            for i, expected in enumerate(elevated["tetra10"]):
                np.testing.assert_array_equal(native.getMeshAtLevel(0).getNodeIdsOfCell(i), expected)
            for i in range(3):
                np.testing.assert_array_equal(native.getGroupArr(-1, f"FACE_{i + 1}").toNumPyArray(), [i])
            with np.load(output.with_name("quadratic_discrete.npz")) as data:
                self.assertEqual(len(data["point_matrices_n_per_m"]), report["point_elements"])
                self.assertEqual(len(data["link_matrices_n_per_m"]), report["link_elements"])
                for key, level in (("point_node_ids", -3), ("link_node_ids", -2)):
                    expected = data[key].reshape(len(data[key]), -1)
                    actual = np.array([native.getMeshAtLevel(level).getNodeIdsOfCell(i)
                                       for i in range(len(expected))])
                    np.testing.assert_array_equal(actual, expected)
                for kind in ("point", "link"):
                    matrices = data[f"{kind}_matrices_n_per_m"]
                    reconstructed = data[f"{kind}_group_matrices_n_per_m"][data[f"{kind}_group_ids"]]
                    error = np.max(np.abs(matrices - reconstructed), axis=(1, 2))
                    self.assertTrue(np.all(error <= 1e-10 * np.max(np.abs(matrices), axis=(1, 2))))
            encoding = json.loads(output.with_name("quadratic_discrete.json").read_text(encoding="utf-8"))
            self.assertLessEqual(encoding["point_encoding"]["maximum_block_relative_error"], 1e-10)
            self.assertLessEqual(encoding["link_encoding"]["maximum_block_relative_error"], 1e-10)
            np.testing.assert_array_equal(native.getGroupArr(-2, "QS_LINKS").toNumPyArray(),
                                          np.arange(report["link_elements"]))


if __name__ == "__main__":
    unittest.main()
