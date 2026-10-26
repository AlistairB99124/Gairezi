"""Whole-model crack mesh and orchestration helpers; no Code_Aster required."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Aster import whole_model_crack as wm
from Aster.whole_model_crack_mesh import archive_arrays, build, signed_volumes, slice_plane

KUHN = ((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7))
THETA = (1.30, 1.37)
SEED = {"annular_sector": [76.0, 79.0, 1.31, 1.36], "elevation_m": -10.3, "seed_thickness_m": 0.002}


def block(r, t, z, offset=0):
    grid = np.array(list(itertools.product(r, t, z)), dtype=float)
    nodes = np.column_stack([grid[:, 0] * np.sin(grid[:, 1]), grid[:, 0] * np.cos(grid[:, 1]), grid[:, 2]])
    index = lambda i, j, k: offset + (i * len(t) + j) * len(z) + k  # noqa: E731
    tets = []
    for i, j, k in itertools.product(range(len(r) - 1), range(len(t) - 1), range(len(z) - 1)):
        corner = [index(i + a, j + b, k + c) for a, b, c in itertools.product((0, 1), repeat=3)]
        tets += [[corner[v] for v in tet] for tet in KUHN]
    return nodes, np.array(tets)


def orient(nodes, tets):
    flip = signed_volumes(nodes, tets) < 0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    return tets


def boundary(tets):
    faces = np.sort(np.vstack([tets[:, list(face)] for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]), 1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    return unique[counts == 1]


def face_counts(cells):
    faces = np.sort(np.vstack([cells[:, list(face)] for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]), 1)
    return np.unique(faces, axis=0, return_counts=True)[1]


def three_body():
    """A small wall arc on a separate foundation (spring coupled) over node-sharing bedrock."""
    theta = np.linspace(*THETA, 5)
    wall_nodes, wall = block(np.arange(76.0, 81.0), theta, np.linspace(-14, -6, 9))
    found_nodes, found = block(np.arange(75.0, 82.0), theta, np.array([-16.0, -14.0]), len(wall_nodes))
    nodes = np.vstack([wall_nodes, found_nodes])
    wall, found = orient(nodes, wall), orient(nodes, found)
    rock_nodes, rock = block(np.arange(75.0, 82.0), theta, np.array([-20.0, -16.0]), len(nodes))
    nodes = np.vstack([nodes, rock_nodes])
    # Merge bedrock's top layer onto the foundation's bottom layer.
    lookup = {tuple(np.round(p, 6)): i for i, p in enumerate(found_nodes, start=len(wall_nodes))}
    remap = np.arange(len(nodes))
    for i, p in enumerate(rock_nodes, start=len(wall_nodes) + len(found_nodes)):
        remap[i] = lookup.get(tuple(np.round(p, 6)), i)
    rock = orient(nodes, remap[rock])
    radius = np.hypot(nodes[:, 0], nodes[:, 1])
    wall_faces = boundary(wall)
    r_mean, z_mean = radius[wall_faces].mean(1), nodes[wall_faces, 2].mean(1)
    bottom = np.flatnonzero(np.isclose(nodes[:len(wall_nodes), 2], -14))
    rock_faces = boundary(rock)
    return {
        "nodes": nodes, "tetra4_concrete": np.vstack([wall, found]), "tetra4_bedrock": rock,
        "concrete_body_ids": np.concatenate([np.ones(len(wall), int), np.full(len(found), 3)]),
        "tri3_upstream": wall_faces[np.isclose(r_mean, 80)], "tri3_downstream": wall_faces[np.isclose(r_mean, 76)],
        "tri3_crest": wall_faces[np.isclose(z_mean, -6)],
        "tri3_rock_bottom": rock_faces[np.isclose(nodes[rock_faces, 2], -20).all(1)],
        "tri3_rock_sides": rock_faces[np.isclose(np.arctan2(nodes[rock_faces, 0], nodes[rock_faces, 1]),
                                                 THETA[0]).all(1)],
        "seg2_wall_foundation_springs": np.array([[w, lookup[tuple(np.round(nodes[w], 6))]] for w in bottom]),
        "support_nodes": np.flatnonzero(np.isclose(nodes[:, 2], -20)),
    }


class SlicePlaneTest(unittest.TestCase):
    def test_cut_conserves_volume_and_stays_conforming(self):
        nodes, tets = block(np.arange(76.0, 79.0), np.linspace(*THETA, 4), np.linspace(-12, -8, 5))
        tets = orient(nodes, tets)
        selected = np.ones(len(tets), dtype=bool)
        cut_nodes, cells, parent, _ = slice_plane(nodes, tets, selected, {}, -10.3)
        volumes = signed_volumes(cut_nodes, cells)
        self.assertTrue(np.all(volumes > 0))
        self.assertAlmostEqual(volumes.sum(), signed_volumes(nodes, tets).sum(), places=8)
        self.assertGreater(len(cells), len(tets))
        self.assertTrue(np.all(np.isin(face_counts(cells), (1, 2))))
        np.testing.assert_allclose(np.bincount(parent, volumes, len(tets)), signed_volumes(nodes, tets),
                                   rtol=1e-9, atol=1e-12)
        side = np.sign(np.round(cut_nodes[cells, 2] + 10.3, 9))
        self.assertFalse(np.any((side.max(1) > 0) & (side.min(1) < 0)), "a cell still crosses the plane")

    def test_unselected_cells_are_untouched(self):
        nodes, tets = block(np.arange(76.0, 78.0), np.linspace(*THETA, 2), np.linspace(-12, -8, 2))
        tets = orient(nodes, tets)
        same_nodes, cells, parent, _ = slice_plane(nodes, tets, np.zeros(len(tets), bool), {}, -10.3)
        np.testing.assert_array_equal(cells, tets)
        self.assertEqual(len(same_nodes), len(nodes))


class BuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays = three_body()
        cls.built = build(cls.arrays, SEED, 0.75)
        cls.archive = archive_arrays(cls.arrays, cls.built)

    def test_band_contains_a_full_length_seed_layer_and_only_wall_cells(self):
        built, nodes = self.built, self.built["nodes"]
        self.assertTrue(built["seed_layer"].any())
        self.assertTrue(np.all(built["body_ids"][built["band"]] == 1))
        self.assertTrue(np.all(built["band"][built["seed_layer"]]))
        layer = nodes[built["cells"][built["seed_layer"], :4]]
        self.assertTrue(np.all(np.abs(layer[..., 2] + 10.3) <= 0.001 + 1e-9))
        angle = np.arctan2(layer[..., 0], layer[..., 1])
        # The layer spans the whole wall arc, not just the seeded sector.
        self.assertAlmostEqual(angle.min(), THETA[0], places=9)
        self.assertAlmostEqual(angle.max(), THETA[1], places=9)

    def test_band_cells_are_quadratic_with_edge_midpoints(self):
        cells, nodes, band = self.built["cells"], self.built["nodes"], self.built["band"]
        self.assertTrue(np.all(cells[band, 4:] >= 0) and np.all(cells[~band, 4:] == -1))
        edges = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
        for column, (a, b) in enumerate(edges, start=4):
            np.testing.assert_allclose(nodes[cells[band, column]],
                                       (nodes[cells[band, a]] + nodes[cells[band, b]]) / 2, atol=1e-9)
        self.assertGreater(len(self.built["hanging_nodes"]), 0)
        linear_nodes = np.unique(cells[~band, :4])
        self.assertFalse(np.isin(self.built["hanging_nodes"], linear_nodes).any())

    def test_archive_groups_partition_the_model(self):
        archive, count = self.archive, len(self.archive["cells"])
        for name in ("concrete", "bedrock", "wall", "foundation", "damage_band", "seed_layer",
                     "linear_volumes", "hanging_master"):
            self.assertEqual(archive[f"group_{name}"].shape, (count,))
        np.testing.assert_array_equal(archive["group_concrete"] ^ archive["group_bedrock"], np.ones(count, bool))
        np.testing.assert_array_equal(archive["group_damage_band"] ^ archive["group_linear_volumes"],
                                      np.ones(count, bool))
        self.assertFalse((archive["group_hanging_master"] & archive["group_damage_band"]).any())
        self.assertEqual(archive["faces_upstream"].shape[1], 6)
        self.assertEqual(len(archive["spring_pairs"]), len(self.arrays["seg2_wall_foundation_springs"]))

    def test_crack_faces_cover_the_seed_sector_and_point_into_the_slab(self):
        faces, nodes, report = self.built["crack_faces"], self.built["nodes"], self.built["report"]
        self.assertEqual(faces.shape[1], 6)
        np.testing.assert_array_equal(self.archive["faces_crack_faces"], faces)
        for column, (a, b) in enumerate(((0, 1), (1, 2), (2, 0)), start=3):
            np.testing.assert_allclose(nodes[faces[:, column]], (nodes[faces[:, a]] + nodes[faces[:, b]]) / 2,
                                       atol=1e-9)
        corners = nodes[faces[:, :3]]
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        top = np.isclose(corners[:, 0, 2], -10.299, atol=1e-9)
        bottom = np.isclose(corners[:, 0, 2], -10.301, atol=1e-9)
        self.assertTrue(np.all(top ^ bottom))
        self.assertTrue(np.all(np.isclose(corners[top, :, 2], -10.299, atol=1e-9)))
        # Right-hand normals point into the slab: down on the top plane, up on the bottom plane.
        self.assertTrue(np.all(normal[top, 2] < 0) and np.all(normal[bottom, 2] > 0))
        self.assertTrue(np.allclose(normal[:, :2], 0, atol=1e-9))
        centre = corners.mean(1)
        radius, angle = np.hypot(centre[:, 0], centre[:, 1]), np.arctan2(centre[:, 0], centre[:, 1])
        self.assertTrue(np.all((radius >= 76) & (radius <= 79) & (angle >= 1.31) & (angle <= 1.36)))
        self.assertAlmostEqual(report["top_area_m2"], report["bottom_area_m2"], delta=0.05 * report["top_area_m2"])
        self.assertGreater(report["top_area_m2"], 0.5 * report["nominal_seed_area_m2"])
        self.assertLess(report["top_area_m2"], 1.5 * report["nominal_seed_area_m2"])
        top_nodes, bottom_nodes = self.archive["crack_top_nodes"], self.archive["crack_bottom_nodes"]
        self.assertTrue(np.allclose(nodes[top_nodes, 2], -10.299) and np.allclose(nodes[bottom_nodes, 2], -10.301))
        np.testing.assert_array_equal(np.union1d(top_nodes, bottom_nodes), np.unique(faces))

    def test_wall_sharing_nodes_with_foundation_is_rejected(self):
        arrays = dict(self.arrays)
        concrete = arrays["tetra4_concrete"].copy()
        concrete[-1, 0] = arrays["tetra4_concrete"][0, 0]
        arrays["tetra4_concrete"] = concrete
        with self.assertRaisesRegex(ValueError, "shares nodes"):
            build(arrays, SEED, 0.75)

    def test_band_must_be_thicker_than_seed(self):
        with self.assertRaisesRegex(ValueError, "thicker than the seed"):
            build(self.arrays, SEED, 0.0005)


class OrchestrationTest(unittest.TestCase):
    def test_wsl_path_and_linux_command(self):
        self.assertEqual(wm.wsl_path(Path("C:/Users/a b/x.med")), "/mnt/c/Users/a b/x.med")
        command = wm.linux_command("Ubuntu", "/opt/ca", Path("D:/run"), ["/opt/ca/bin/run_aster", "c.export"], 6)
        self.assertEqual(command[:6], ["wsl.exe", "-d", "Ubuntu", "--cd", "/mnt/d/run", "--exec"])
        self.assertIn("OMP_NUM_THREADS=6", command)
        self.assertIn("PATH=/opt/ca/bin:/usr/bin:/bin", command)
        self.assertEqual(command[-2:], ["/opt/ca/bin/run_aster", "c.export"])

    def test_solver_works_on_disk_and_removes_its_work_directory_after_failure(self):
        import subprocess
        from unittest import mock

        def prepare(directory, *_):
            (directory / "whole_model_mesh.npz").write_bytes(b"x")
            return {"prepared_sha256": "s", "generation": {}, "mesh": {}, "inputs": {"times": [0.0]}}

        def run(command, **_):
            if "rm" in command:
                return subprocess.CompletedProcess(command, 0, "", "")
            (Path(self.output) / "mesh.med").write_bytes(b"med")
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "src"
            source.write_bytes(b"x")
            self.output = None

            def located(*_):
                self.output = next((Path(root) / "out" / wm.OUTPUT_ROOT_NAME).iterdir())
                return "/opt/ca"

            with mock.patch.object(wm, "prepare", side_effect=prepare), \
                    mock.patch.object(wm, "find_linux_installation", side_effect=located), \
                    mock.patch.object(wm.subprocess, "run", side_effect=run) as launched, \
                    mock.patch("Aster.run_case.run_solver", side_effect=OSError("solver died")) as solver:
                manifest = wm.run(source, source, source, Path(root) / "out")
        self.assertEqual(manifest["status"], "failed")
        work = manifest["linux_work_directory"]
        self.assertTrue(work.startswith("/var/tmp/") and not work.startswith("/tmp"))
        program = solver.call_args.args[0]
        self.assertEqual(program[program.index("--wrkdir") + 1], work)
        self.assertEqual(launched.call_args_list[-1].args[0][-3:], ["rm", "-rf", work])

    def test_support_matrix_is_column_major(self):
        manifest = {"wall_foundation_springs": [{"group": "SPR_WF_00000", "stiffness_n_per_m": [1, 2, 3],
                                                 "tributary_area_m2": 1.0}],
                    "support_springs": [{"group": "SPR_OUT_00000",
                                         "matrix_n_per_m": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}]}
        wall, support = wm.spring_inputs(manifest)
        self.assertEqual(wall[0]["stiffness_n_per_m"], [1.0, 2.0, 3.0])
        self.assertEqual(support[0]["matrix_column_major_n_per_m"], [1, 4, 7, 2, 5, 8, 3, 6, 9])
        manifest["support_springs"][0]["matrix_n_per_m"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            wm.spring_inputs(manifest)

    def test_export_uses_linux_paths_and_result_units(self):
        with tempfile.TemporaryDirectory() as folder:
            text = wm.export_file(Path(folder), 8, 18000, 3600).read_bytes().decode()
        self.assertNotIn("\r", text)
        self.assertIn("P ncpus 8\n", text)
        self.assertIn("P memory_limit 18000\n", text)
        for unit in ("D 1", "D 91", "D 20", "R 90", "R 89", "R 80"):
            self.assertRegex(text, rf"F \w+ /mnt/\S+ {unit}\n" if wm.os.name == "nt" else rf"F \w+ /\S+ {unit}\n")

    def test_crack_metrics_count_new_damage_outside_the_seed(self):
        xyz = np.array([[78 * np.sin(1.33), 78 * np.cos(1.33), -10.3],
                        [77 * np.sin(1.34), 77 * np.cos(1.34), -10.3],
                        [79.5 * np.sin(1.35), 79.5 * np.cos(1.35), -10.0]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.npz"
            np.savez(path, coordinates_m=xyz, initial_damage=np.array([0.999, 0, 0]),
                     damage=np.array([[0.999, 0, 0], [0.999, 0.97, 0.6]]), times=np.array([0.0, 1.0]))
            metrics = wm.crack_metrics(path, SEED, {}, 78.0, 4.0)
        final = metrics["final"]
        self.assertEqual((final["new_points_above_0_95"], final["new_points_above_0_5"]), (1, 2))
        self.assertAlmostEqual(final["cracked_radial_extent_m"], 1.0)
        self.assertAlmostEqual(final["cracked_through_thickness_fraction"], 0.25)
        self.assertEqual(metrics["seeded_gauss_points"], 1)
        self.assertEqual(metrics["history"][0]["cracked_radial_extent_m"], 0.0)

    def test_case_seeds_band_only_and_keeps_other_cells_elastic(self):
        case = wm.CASE.read_text(encoding="utf-8")
        self.assertIn('wall_group="DAMAGE_BAND", volume_group="DAMAGE_BAND"', case)
        self.assertIn("elastic_elsewhere=True", case)
        self.assertIn('_F(GROUP_MA="DAMAGE_BAND", RELATION="ENDO_FISS_EXP"', case)
        self.assertIn("LIAISON_MAIL", case)

    def test_case_applies_crack_water_pressure_without_reorienting_crack_faces(self):
        case = wm.CASE.read_text(encoding="utf-8")
        self.assertIn('PRES_REP=_F(GROUP_MA="CRACK_FACES"', case)
        self.assertIn('PRES=loads["crack_pressure_pa"]', case)
        orientation = case[case.index("ORIE_PEAU_3D"):case.index("MODEL = AFFE_MODELE")]
        self.assertNotIn("CRACK_FACES", orientation)
        self.assertIn("mean_crack_face_separation_m", case)

    def test_crack_water_pressure_matches_the_seeded_face_head(self):
        loads = {"upstream_water_surface_z_m": 3.0, "tailwater_elevation_m": -26.0,
                 "water_pressure_gradient_pa_per_m": 9810.0}

        def shared(face, pressure, z=-10.3):
            return {"payload": {"seed_face": face, "geometry": {"elevation_m": z},
                                "loading": {"internal_pressure_pa": pressure}}}

        water = wm.crack_water_pressure(shared("upstream", 9810.0 * 13.3), loads)
        self.assertAlmostEqual(water["pressure_pa"], 130473.0)
        self.assertEqual(water["water_surface_z_m"], 3.0)
        self.assertEqual(wm.crack_water_pressure(shared("downstream", 0.0), loads)["pressure_pa"], 0.0)
        self.assertAlmostEqual(wm.crack_water_pressure(shared("downstream", 9810.0, -27.0), loads)["pressure_pa"],
                               9810.0)
        with self.assertRaisesRegex(ValueError, "select the crack location again"):
            wm.crack_water_pressure(shared("upstream", 0.0), loads)
        half = wm.crack_water_pressure(shared("upstream", 0.5 * 9810.0 * 13.3), loads, 0.5)
        self.assertAlmostEqual(half["pressure_pa"], 0.5 * 130473.0)
        self.assertEqual(half["factor"], 0.5)
        dry = wm.crack_water_pressure(shared("upstream", 0.0), loads, 0.0)
        self.assertEqual(dry["pressure_pa"], 0.0)

    def test_wall_thickness_follows_steps(self):
        wall = {"radial_width_m": 6.0, "thickness_steps": [{"top_z_m": -5.0, "radial_width_m": 5.0},
                                                            {"top_z_m": -20.0, "radial_width_m": 4.0}]}
        self.assertEqual(wm.wall_thickness(wall, -2.0), 6.0)
        self.assertEqual(wm.wall_thickness(wall, -10.0), 5.0)


if __name__ == "__main__":
    unittest.main()
