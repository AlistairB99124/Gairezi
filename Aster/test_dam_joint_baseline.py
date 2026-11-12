"""Tests for the Code_Aster dam-joint baseline (no solver needed)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Aster import bundled_dam_tests  # noqa: E402
from Aster import dam_joint_baseline as djb  # noqa: E402
from Aster import dam_joint_fixture as fixture  # noqa: E402


class JointCellTests(unittest.TestCase):
    def setUp(self):
        self.arrays = fixture.fixture_arrays(across=4, along=2, wall_layers=2, block_layers=1)
        self.cells = djb.joint_cells(self.arrays)

    def test_each_contact_triangle_becomes_one_prism_from_foundation_to_wall(self):
        nodes, body = self.arrays["nodes"], self.arrays["concrete_body_ids"]
        wall = np.unique(self.arrays["tetra4_concrete"][body == 1])
        foundation = np.unique(self.arrays["tetra4_concrete"][body == 3])
        self.assertEqual(self.cells.shape, (len(self.arrays["joint_slave_tri3"]), 6))
        self.assertTrue(np.isin(self.cells[:, :3], foundation).all())
        self.assertTrue(np.isin(self.cells[:, 3:], wall).all())
        np.testing.assert_allclose(nodes[self.cells[:, :3]], nodes[self.cells[:, 3:]])
        # The same triangles as the baseline's, whatever their vertex order.
        self.assertEqual({tuple(sorted(row)) for row in self.cells[:, 3:].tolist()},
                         {tuple(sorted(row)) for row in self.arrays["joint_slave_tri3"].tolist()})

    def test_first_triangle_normal_points_away_from_the_wall(self):
        nodes = self.arrays["nodes"]
        a, b, c = (nodes[self.cells[:, k]] for k in range(3))
        self.assertTrue((np.cross(b - a, c - a)[:, 2] < 0).all())  # the wall is above the contact
        geometry = djb.joint_geometry(self.arrays, self.cells)
        np.testing.assert_allclose(geometry["normal"], np.tile([0.0, 0.0, 1.0], (len(self.cells), 1)), atol=1e-12)
        self.assertAlmostEqual(geometry["area_m2"].sum(), fixture.THICKNESS_M * fixture.WIDTH_M)

    def test_orientation_does_not_depend_on_the_vertex_order_given(self):
        shuffled = dict(self.arrays)
        shuffled["joint_slave_tri3"] = self.arrays["joint_slave_tri3"][:, [1, 0, 2]]
        shuffled["joint_master_tri3"] = self.arrays["joint_master_tri3"][:, [1, 0, 2]]
        geometry = djb.joint_geometry(shuffled, djb.joint_cells(shuffled))
        self.assertTrue((geometry["normal"][:, 2] > 0.999).all())

    def test_a_contact_that_does_not_match_is_refused(self):
        moved = dict(self.arrays)
        moved["nodes"] = self.arrays["nodes"].copy()
        moved["nodes"][self.arrays["joint_master_tri3"][0, 0], 0] += 0.01
        with self.assertRaisesRegex(djb.JointCaseError, "do not coincide"):
            djb.joint_cells(moved)
        missing = dict(self.arrays, joint_slave_tri3=np.empty((0, 3), dtype=np.int64),
                       joint_master_tri3=np.empty((0, 3), dtype=np.int64))
        with self.assertRaisesRegex(djb.JointCaseError, "no matched"):
            djb.joint_cells(missing)
        swapped = dict(self.arrays, joint_slave_tri3=self.arrays["joint_master_tri3"],
                       joint_master_tri3=self.arrays["joint_slave_tri3"])
        with self.assertRaisesRegex(djb.JointCaseError, "not a face of a wall"):
            djb.joint_cells(swapped)

    def test_configured_stiffness_must_be_one_value(self):
        self.assertEqual(djb.configured_stiffness(self.arrays), fixture.JOINT_STIFFNESS_PA_PER_M)
        mixed = dict(self.arrays, joint_coefficients_n_per_m3=np.array([[1.25e9, 1.25e9, 3e9]]))
        with self.assertRaisesRegex(djb.JointCaseError, "differs by direction"):
            djb.configured_stiffness(mixed)


class JointWaterTests(unittest.TestCase):
    def test_pressure_falls_linearly_from_reservoir_head_to_tailwater_head(self):
        position = {"kind": "x", "upstream_x_m": 0.0, "downstream_x_m": 2.0}
        points = np.array([[0.0, 0.0, -10.0], [1.0, 5.0, -10.0], [2.0, 0.0, -10.0], [-1.0, 0.0, -10.0], [0.0, 0.0, 5.0]])
        pressure = djb.joint_pressure(position, 9810.0, 0.0, -8.0, points)
        np.testing.assert_allclose(pressure, [98100.0, (98100.0 + 19620.0) / 2, 19620.0, 98100.0, 0.0])

    def test_arch_position_is_measured_radially_from_the_fitted_centre(self):
        angles = np.linspace(0.2, 1.2, 9)
        ring = lambda radius, z: np.column_stack([10 + radius * np.cos(angles), -5 + radius * np.sin(angles),  # noqa: E731
                                                  np.full(len(angles), z)])
        nodes = np.vstack([ring(80.0, 0.0), ring(80.0, -5.0), ring(75.5, 0.0), ring(75.5, -5.0), ring(77.0, -5.0)])
        strip = lambda start: np.array([[start + i, start + i + 1, start + 9 + i] for i in range(8)])  # noqa: E731
        arrays = {"nodes": nodes, "tri3_upstream": strip(0), "tri3_downstream": strip(18)}
        cells = np.array([[36, 37, 38, 36, 37, 38]])
        position = djb.arch_position(arrays, cells)
        np.testing.assert_allclose(position["centre_xy_m"], [10.0, -5.0], atol=1e-6)
        self.assertAlmostEqual(position["upstream_radius_m"], 80.0, places=6)
        self.assertAlmostEqual(position["downstream_radius_m"], 75.5, places=6)
        at = lambda radius: [10 + radius * np.cos(0.7), -5 + radius * np.sin(0.7), -5.0]  # noqa: E731
        pressure = djb.joint_pressure(position, 9810.0, 0.0, -5.0, np.array([at(80.0), at(77.75), at(75.5)]))
        np.testing.assert_allclose(pressure, [49050.0, 24525.0, 0.0], atol=1e-3)
        # A wall whose thickness varies has no single downstream radius.
        uneven = dict(arrays, nodes=nodes.copy())
        uneven["nodes"][27:36, :2] = ring(73.0, -5.0)[:, :2]
        with self.assertRaisesRegex(djb.JointCaseError, "uniform thickness"):
            djb.arch_position(uneven, cells)

    def test_water_levels_start_with_weight_alone_and_end_at_the_configured_levels(self):
        levels = djb.water_levels(fixture.fixture_loads() | {"tailwater_elevation_m": 1.4}, 1.0)
        self.assertEqual(levels["instants"], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(levels["upstream_level_m"][0], 1.0)
        self.assertEqual(levels["downstream_level_m"][0], 1.0)
        self.assertAlmostEqual(levels["upstream_level_m"][-1], 3.0)
        self.assertAlmostEqual(levels["downstream_level_m"][-1], 1.4)
        self.assertTrue(np.all(np.diff(levels["upstream_level_m"]) > 0))


class CaseFileTests(unittest.TestCase):
    def setUp(self):
        self.arrays = fixture.fixture_arrays(across=4, along=2, wall_layers=2, block_layers=1)
        self.base = fixture.fixture_base_comm(self.arrays)
        self.cells = djb.joint_cells(self.arrays)
        self.levels = djb.water_levels(fixture.fixture_loads(), 1.0)

    def render(self, case: dict) -> str:
        joint = djb.joint_parameters(djb.with_joints(self.arrays), self.cells, case)
        return djb.render_joint_comm(self.base, joint, fixture.fixture_loads(), self.levels, "/steps.npz", "/status.json")

    def test_springs_are_replaced_by_the_joint_layer_with_the_test_settings(self):
        comm = self.render(fixture.fixture_case(uplift=False))
        compile(comm.replace("from code_aster.Commands import *", "").replace("from code_aster import CA", ""),
                "case.comm", "exec")
        self.assertNotIn("WALL_FOUNDATION_SPRINGS", comm)
        self.assertNotIn("K_T_D_L", comm)
        self.assertNotIn("MECA_STATIQUE", comm)
        self.assertEqual(comm.count("_F(GROUP_MA='JOINTS', PHENOMENE='MECANIQUE', MODELISATION='3D_JOINT')"), 1)
        self.assertEqual(comm.count("K_T_N"), len(self.arrays["support_nodes"]))  # the bedrock support is untouched
        self.assertIn("JOINT_MECA_RUPT=_F(K_N=200000000, K_T=200000000, SIGM_MAX=0, ALPHA=1.0, PENA_RUPTURE=0.2)", comm)
        self.assertIn("_F(GROUP_MA='JOINTS', MATER=JOINT_MAT)", comm)
        # Nonlinear settings of ssnp142b.
        for setting in ("NEWTON=_F(MATRICE='TANGENTE', REAC_ITER=1)", "RECH_LINEAIRE=_F(METHODE='MIXTE', ITER_LINE_MAXI=5)",
                        "CONVERGENCE=_F(ITER_GLOB_MAXI=12, RESI_GLOB_RELA=1e-5)", "SOLVEUR=_F(METHODE='MUMPS')",
                        "ECHEC=_F(SUBD_NIVEAU=6, SUBD_PAS=10, SUBD_PAS_MINI=1.0e-10)",
                        "_F(RELATION='JOINT_MECA_RUPT', GROUP_MA='JOINTS')"):
            self.assertIn(setting, comm)
        self.assertNotIn("PRES_FLUIDE", comm)
        # The weight is at its full value from the first instant, before any water is raised.
        self.assertIn("RAMPE_POIDS = DEFI_FONCTION(NOM_PARA='INST', VALE=(0.0, 0.0, 1.0, 1.0),", comm)
        self.assertEqual(comm.count("FONC_MULT"), 1)
        self.assertIn("NIV_AM = DEFI_FONCTION(NOM_PARA='INST', VALE=(1, 1, 2, 1.5, 3, 2, 4, 2.5, 5, 3),", comm)

    def test_water_in_the_joint_is_an_imposed_pressure_as_in_the_test(self):
        comm = self.render(fixture.fixture_case(uplift=True))
        self.assertIn("PRES_FLUIDE=P_JOINT", comm)
        self.assertIn("NOM_PARA=('X', 'Y', 'Z', 'INST')", comm)
        self.assertIn("RECH_LINEAIRE=_F(ITER_LINE_MAXI=5)", comm)
        self.assertLess(comm.index("P_JOINT = FORMULE"), comm.index("JOINT_MAT = DEFI_MATERIAU"))
        self.assertLess(comm.index("JOINT_MAT = DEFI_MATERIAU"), comm.index("CHMAT = AFFE_MATERIAU("))

    def test_reference_case_uses_the_values_the_manual_recommends_for_a_concrete_dam(self):
        comm = self.render(dict(djb.CASES["reference_dry"]))
        self.assertIn("JOINT_MECA_RUPT=_F(K_N=3e+12, K_T=3e+12, SIGM_MAX=50000, ALPHA=1.0, PENA_RUPTURE=1)", comm)

    def test_friction_law_carries_the_recommended_values_and_the_settings_of_ssnp142d(self):
        comm = self.render(dict(djb.CASES["friction_reference_uplift"],
                                uplift_position=fixture.fixture_case(True)["uplift_position"]))
        self.assertIn("JOINT_MECA_FROT=_F(K_N=3e+12, K_T=3e+12, MU=1, ADHESION=1000, PENA_TANG=6000000, "
                      "PRES_FLUIDE=P_JOINT)", comm)
        self.assertIn("_F(RELATION='JOINT_MECA_FROT', GROUP_MA='JOINTS')", comm)
        self.assertIn("RECH_LINEAIRE=_F(ITER_LINE_MAXI=5)", comm)
        self.assertNotIn("JOINT_MECA_RUPT", comm)
        self.assertNotIn("SIGM_MAX", comm)
        with self.assertRaisesRegex(djb.JointCaseError, "Unknown joint law"):
            self.render(dict(djb.CASES["friction_reference_dry"], law="CZM_LIN_REG"))

    def test_a_changed_baseline_layout_is_refused(self):
        with self.assertRaisesRegex(djb.JointCaseError, "'solve' section"):
            djb.render_joint_comm(self.base.replace("RESU = MECA_STATIQUE(", "RESU = OTHER("), {}, {}, self.levels, "", "")
        without = "\n".join(line for line in self.base.split("\n") if "K_T_D_L" not in line)
        joint = djb.joint_parameters(djb.with_joints(self.arrays), self.cells, fixture.fixture_case(False))
        with self.assertRaisesRegex(djb.JointCaseError, "no wall/foundation spring"):
            djb.render_joint_comm(without, joint, fixture.fixture_loads(), self.levels, "", "")

    def test_written_case_records_its_inputs_and_departures(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case"
            record = djb.write_case(output, self.arrays, {"bedrock_analysis_mode": "elastic_layer"}, self.base,
                                    fixture.fixture_loads(), fixture.fixture_case(uplift=True))
            self.assertEqual(record["joint_cell_count"], len(self.cells))
            self.assertEqual(json.loads((output / djb.META_NAME).read_text())["departures_from_ssnp142"],
                             list(djb.DEPARTURES))
            with np.load(output / djb.ARRAYS_NAME) as stored:
                np.testing.assert_array_equal(stored["penta6_joints"], self.cells)
            self.assertTrue((output / djb.CASE_NAME).is_file())
            loads = fixture.fixture_loads() | {"overflow_surcharge_pa": 500.0}
            with self.assertRaisesRegex(djb.JointCaseError, "crest surcharge"):
                djb.write_case(output, self.arrays, {}, self.base, loads, fixture.fixture_case(uplift=False))


class ReportTests(unittest.TestCase):
    def write_run(self, directory: Path, opening_by_cell, state=2.0):
        arrays = fixture.fixture_arrays(across=4, along=2, wall_layers=2, block_layers=1)
        djb.write_case(directory, arrays, {"bedrock_analysis_mode": "elastic_layer"}, fixture.fixture_base_comm(arrays),
                       fixture.fixture_loads(), fixture.fixture_case(uplift=False))
        with np.load(directory / djb.ARRAYS_NAME) as stored:
            arrays = {key: stored[key] for key in stored.files}
        cells = arrays["penta6_joints"]
        geometry = djb.joint_geometry(arrays, cells)
        opening = opening_by_cell(geometry["centroid_m"])
        displacement = np.zeros_like(arrays["nodes"])
        # Lift each wall contact node by the mean opening of its cells, so nodal jumps match on average.
        displacement[np.unique(arrays["tetra4_concrete"][arrays["concrete_body_ids"] == 1]), 0] = 0.002
        ids = np.arange(100, 100 + len(cells))
        values = {"V7": opening, "V3": np.full(len(cells), state), "V8": np.full(len(cells), 3e-4),
                  "V9": np.full(len(cells), 4e-4), "V18": np.zeros(len(cells)),
                  "SIGN": np.where(opening > 0, 0.0, -1e5), "SITX": np.zeros(len(cells)), "SITY": np.zeros(len(cells))}
        np.savez(directory / djb.STEPS_NAME, instant=np.array([1.0, 2.0]),
                 displacement_m=np.array([displacement, displacement]), gauss_cells=ids, joint_cell_ids=ids,
                 **{name: np.array([value, value]) for name, value in values.items()})
        (directory / djb.STATUS_NAME).write_text(json.dumps({
            "status": "stopped", "message": "did not converge", "converged_instants": [1.0, 2.0]}))
        return geometry

    @staticmethod
    def write_results(directory: Path, instants=(1.0, 2.0)) -> None:
        """Results of a run with a closed, unloaded joint, for a case already written into ``directory``."""
        with np.load(directory / djb.ARRAYS_NAME) as stored:
            nodes, cells = stored["nodes"], stored["penta6_joints"]
        ids = np.arange(len(cells))
        zeros = np.zeros((len(instants), len(cells)))
        np.savez(directory / djb.STEPS_NAME, instant=np.array(instants),
                 displacement_m=np.zeros((len(instants), len(nodes), 3)), gauss_cells=ids, joint_cell_ids=ids,
                 **{name: zeros for name in ("V1", "V2", "V3", "V7", "V8", "V9", "V18", "SIGN", "SITX", "SITY")})
        (directory / djb.STATUS_NAME).write_text(json.dumps({"status": "completed", "converged_instants": list(instants)}))

    def test_report_gives_opened_area_largest_opening_and_sliding_per_water_level(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            geometry = self.write_run(output, lambda centroid: np.where(centroid[:, 0] < 0.25, 1e-3, -1e-5))
            report = djb.summarise(output)
        self.assertEqual((report["status"], report["message"]), ("stopped", "did not converge"))
        self.assertEqual([level["instant"] for level in report["levels"]], [1.0, 2.0])
        self.assertFalse(report["engineering_validated"])
        level = report["levels"][-1]
        expected_area = geometry["area_m2"][geometry["centroid_m"][:, 0] < 0.25].sum()
        self.assertAlmostEqual(level["opened_area_m2"], expected_area)
        self.assertAlmostEqual(level["opened_share"], 0.25)
        self.assertAlmostEqual(level["largest_opening_m"], 1e-3)
        self.assertLess(level["largest_opening_xyz_m"][0], 0.25)
        self.assertAlmostEqual(level["largest_closing_m"], 1e-5)
        self.assertAlmostEqual(level["largest_sliding_m"], 5e-4)
        self.assertAlmostEqual(level["largest_joint_compression_pa"], 1e5)
        self.assertAlmostEqual(level["maximum_wall_displacement_m"], 0.002)
        self.assertAlmostEqual(level["fill_fraction"], 0.25)
        self.assertEqual(report["gauss_points_per_joint_cell"], 1.0)

    def test_a_run_with_no_converged_level_reports_that_and_no_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.write_run(output, lambda centroid: np.zeros(len(centroid)))
            np.savez(output / djb.STEPS_NAME, instant=np.empty(0), displacement_m=np.empty(0))
            (output / djb.STATUS_NAME).write_text(json.dumps({"status": "failed", "converged_instants": []}))
            report = djb.summarise(output)
        self.assertEqual((report["status"], report["levels"]), ("failed", []))

    def test_principal_extremes_of_nodal_stress(self):
        stress = np.array([[2e6, 0, 0, 0, 0, 0], [0, -3e6, 0, 0, 0, 0], [0, 0, 0, 1e6, 0, 0]], dtype=float)
        tension, compression = djb.principal_extremes(stress)
        self.assertAlmostEqual(tension, 2e6)
        self.assertAlmostEqual(compression, -3e6)


class HandCalculationTests(unittest.TestCase):
    def test_cracked_base_satisfies_equilibrium(self):
        for uplift in (False, True):
            with self.subTest(uplift=uplift):
                expected = fixture.cracked_base(uplift)
                force = expected["peak_effective_compression_pa"] * expected["contact_length_m"] * fixture.WIDTH_M / 2
                self.assertAlmostEqual(force, expected["effective_normal_force_n"], places=6)
                self.assertAlmostEqual(expected["contact_length_m"] + expected["open_length_m"], fixture.THICKNESS_M)
                heel = fixture.analytic_opening(np.array([0.0, expected["open_length_m"], fixture.THICKNESS_M]), expected)
                self.assertAlmostEqual(heel[0], expected["heel_opening_m"])
                self.assertAlmostEqual(heel[1], 0.0)
                self.assertAlmostEqual(-heel[2] * fixture.JOINT_STIFFNESS_PA_PER_M,
                                       expected["peak_effective_compression_pa"], places=3)
        self.assertGreater(fixture.cracked_base(True)["heel_opening_m"], fixture.cracked_base(False)["heel_opening_m"])

    def test_a_small_tensile_strength_holds_the_heel_down_a_little(self):
        for uplift in (False, True):
            with self.subTest(uplift=uplift):
                free, held = fixture.cracked_base(uplift), fixture.cracked_base(uplift, 1000.0)
                self.assertLess(held["heel_opening_m"], free["heel_opening_m"])
                self.assertGreater(held["heel_opening_m"], 0.8 * free["heel_opening_m"])
                # Force balance of the held solution, integrated independently.
                x = np.linspace(0.0, fixture.THICKNESS_M, 200001)
                stress = np.minimum(fixture.JOINT_STIFFNESS_PA_PER_M * fixture.analytic_opening(x, held), 1000.0)
                self.assertAlmostEqual(-np.trapezoid(stress, x) * fixture.WIDTH_M / held["effective_normal_force_n"], 1.0,
                                       places=6)
        vanishing = fixture.cracked_base(False, 1e-6)
        self.assertAlmostEqual(vanishing["heel_opening_m"] / fixture.cracked_base(False)["heel_opening_m"], 1.0, places=4)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.levels = djb.water_levels(fixture.fixture_loads(), 1.0)
        self.addCleanup(djb.set_progress_log, None)

    def test_solver_output_is_turned_into_plain_progress_lines(self):
        message = lambda line: djb.progress_message(line, self.levels)  # noqa: E731
        self.assertEqual(message("DAM_JOINT_STEP start instant=1"),
                         "Solving level 1 of 5 (reservoir at 1.00 m, 0% of the head): self-weight only")
        self.assertEqual(message("DAM_JOINT_STEP done instant=4 seconds=36.5"),
                         "Converged level 4 of 5 (reservoir at 2.50 m, 75% of the head) in 36 s")
        self.assertEqual(message("DAM_JOINT_STEP failed instant=5"),
                         "Did NOT converge at level 5 of 5 (reservoir at 3.00 m, 100% of the head)")
        self.assertEqual(message("|     3        X | 2.11527E-02  X | 1.22687E+04    |     5          | 3.1E-02 |TANGENTE|"),
                         "    Newton iteration 3: relative residual 2.1e-02 (target 1e-05)")
        self.assertEqual(message("|     1          | 7.76915E-12    | 4.13396E-06    |     0   | 1.0E+00 |TANGENTE|"),
                         "    Newton iteration 1: relative residual 7.8e-12 (target 1e-05)")
        self.assertEqual(message("Instant de calcul:  3.100000000000e+00 - Niveau de découpe: 2"),
                         "    step cut (depth 2): trying up to instant 3.1")
        for quiet in ("Instant de calcul:  4.000000000000e+00 - Niveau de découpe: 1",
                      '    print(f"DAM_JOINT_STEP start instant={target:.6g}", flush=True)',
                      "|    ITERATION   |     RELATIF    |", "Critère(s) de convergence atteint(s)", ""):
            self.assertIsNone(message(quiet), quiet)

    def test_live_log_is_followed_into_the_progress_file_until_the_run_ends(self):
        import threading
        with tempfile.TemporaryDirectory() as directory:
            live, log = Path(directory) / "solver.live.log", Path(directory) / "progress.log"
            djb.set_progress_log(log)
            finished = threading.Event()
            follower = threading.Thread(target=djb._follow, args=(live, finished, self.levels))
            follower.start()
            with live.open("a", encoding="utf-8") as stream:
                stream.write("noise\nDAM_JOINT_STEP start instant=2\nDAM_JOINT_STEP done inst")
                stream.flush()
                stream.write("ant=2 seconds=12.0\n")  # a line that arrives in two writes is reported once
            finished.set()
            follower.join(timeout=10)
            djb.progress("after the run")
            lines = log.read_text(encoding="utf-8").splitlines()
        self.assertFalse(follower.is_alive())
        self.assertEqual([line.split("  ", 1)[1] for line in lines], [
            "Solving level 2 of 5 (reservoir at 1.50 m, 25% of the head)",
            "Converged level 2 of 5 (reservoir at 1.50 m, 25% of the head) in 12 s", "after the run"])
        self.assertRegex(lines[0], r"^\d\d:\d\d:\d\d  ")

    def test_no_progress_file_is_written_unless_one_is_set(self):
        djb.set_progress_log(None)
        djb.progress("nothing to see")  # must not raise or write anywhere

    def test_a_run_longer_than_the_agreed_limit_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(djb.JointCaseError, "not allowed more than 1200 s"):
                djb.solve(Path(directory), 1201)


class StageTests(unittest.TestCase):
    """The app's stage: dry joint, then water in the joint."""

    def run_stage(self, outcomes: dict):
        from unittest.mock import patch
        calls = []

        def fake_run(name, baseline_dir, output_root, time_limit_s):
            calls.append((name, time_limit_s))
            outcome = outcomes[name]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with tempfile.TemporaryDirectory() as directory, patch.object(djb, "run", side_effect=fake_run):
            stage = djb.run_stage(Path("baseline"), Path(directory), 480)
            written = json.loads((Path(directory) / djb.STAGE_REPORT_NAME).read_text(encoding="utf-8"))
        self.assertEqual(stage, written)
        return stage, calls

    def test_both_cases_run_in_order_and_are_reported_together(self):
        done = lambda opening: {"status": "completed", "levels": [{"largest_opening_m": opening}],  # noqa: E731
                                "elastic_baseline": {"maximum_wall_displacement_m": 0.0103}}
        stage, calls = self.run_stage({"friction_configured_dry": done(8.8e-4), "friction_configured_uplift": done(1.13e-3)})
        self.assertEqual(calls, [("friction_configured_dry", 480), ("friction_configured_uplift", 480)])
        self.assertEqual((stage["status"], stage["message"]), ("completed", None))
        self.assertEqual(list(stage["cases"]), ["dry", "water_in_joint"])
        self.assertEqual(stage["cases"]["water_in_joint"]["case"], "friction_configured_uplift")
        self.assertEqual(stage["cases"]["dry"]["levels"][-1]["largest_opening_m"], 8.8e-4)
        self.assertEqual(stage["elastic_baseline"], {"maximum_wall_displacement_m": 0.0103})
        self.assertFalse(stage["engineering_validated"])
        self.assertEqual(stage["departures_from_ssnp142"], list(djb.DEPARTURES))

    def test_a_case_that_does_not_complete_ends_the_stage_and_says_why(self):
        stage, calls = self.run_stage({
            "friction_configured_dry": {"status": "stopped", "message": "Code_Aster did not converge from instant 3 to 4:\nmore"},
            "friction_configured_uplift": {"status": "completed"}})
        self.assertEqual(calls, [("friction_configured_dry", 480)])  # the second case is not started
        self.assertEqual(stage["status"], "failed")
        self.assertEqual(stage["message"], "The dry case ended as 'stopped': Code_Aster did not converge from "
                                           "instant 3 to 4:. See its run/solver.live.log.")
        self.assertEqual(list(stage["cases"]), ["dry"])

    def test_a_baseline_the_method_cannot_use_is_reported_not_raised(self):
        stage, _ = self.run_stage({"friction_configured_dry": djb.JointCaseError("implemented for the elastic-layer bedrock mode")})
        self.assertEqual(stage["status"], "failed")
        self.assertIn("could not be prepared: implemented for the elastic-layer bedrock mode", stage["message"])


class CoolingTests(unittest.TestCase):
    """The provisional temperature load: the wall cooled in steps once the reservoir is full."""

    def setUp(self):
        self.arrays = fixture.fixture_arrays(across=4, along=2, wall_layers=4, block_layers=1)
        self.base = fixture.fixture_base_comm(self.arrays)
        self.cells = djb.joint_cells(self.arrays)
        self.case = fixture.cooling_case()  # closure at 20 C, then the wall at 10 C and 0 C
        self.levels = djb.water_levels(fixture.fixture_loads(), 1.0, [10.0, 20.0], 20.0)

    def render(self, case: dict, base: str | None = None) -> str:
        joint = djb.joint_parameters(djb.with_joints(self.arrays), self.cells, case)
        levels = self.levels if case.get("cooling") else djb.water_levels(fixture.fixture_loads(), 1.0)
        return djb.render_joint_comm(base or self.base, joint, fixture.fixture_loads(), levels, "/steps.npz", "/status.json")

    def test_each_cooling_step_is_one_more_instant_at_full_reservoir(self):
        self.assertEqual(self.levels["instants"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(self.levels["temperature_change_k"], [0.0, 0.0, 0.0, 0.0, 0.0, -10.0, -20.0])
        self.assertEqual(self.levels["wall_temperature_c"], [20.0, 20.0, 20.0, 20.0, 20.0, 10.0, 0.0])
        self.assertEqual(self.levels["closure_temperature_c"], 20.0)
        joint = djb.joint_parameters(djb.with_joints(self.arrays), self.cells, self.case)
        self.assertEqual(joint["cooling"]["steps_k"], [10.0, 20.0])  # what the case itself gives
        with self.assertRaisesRegex(djb.JointCaseError, "need the closure temperature"):
            djb.water_levels(fixture.fixture_loads(), 1.0, [10.0, 20.0])
        self.assertEqual(self.levels["fill_fraction"][4:], [1.0, 1.0, 1.0])
        self.assertEqual(self.levels["upstream_level_m"][4:], [3.0, 3.0, 3.0])
        # Without cooling nothing is added, so the app's cases are prepared exactly as before.
        self.assertNotIn("temperature_change_k", djb.water_levels(fixture.fixture_loads(), 1.0))

    def test_only_the_wall_is_given_expansion_and_a_temperature(self):
        comm = self.render(self.case)
        compile(comm.replace("from code_aster.Commands import *", "").replace("from code_aster import CA", ""),
                "case.comm", "exec")
        self.assertIn("WALL_MAT = DEFI_MATERIAU(ELAS=_F(E=30000000000.0, NU=0.2, RHO=2400.0, ALPHA=1e-05))", comm)
        self.assertIn("CONCRETE_MAT = DEFI_MATERIAU(ELAS=_F(E=30000000000.0, NU=0.2, RHO=2400.0))", comm)
        # The wall's assignment follows the one for all concrete, so it is the one that applies to the wall.
        self.assertIn("        _F(GROUP_MA='CONCRETE', MATER=CONCRETE_MAT),\n        _F(GROUP_MA='WALL', MATER=WALL_MAT),\n"
                      "        _F(GROUP_MA='JOINTS', MATER=JOINT_MAT),\n", comm)
        self.assertIn("CHMAT = AFFE_MATERIAU(\n    AFFE_VARC=_F(GROUP_MA='WALL', NOM_VARC='TEMP', EVOL=TEMP_EVOL, "
                      "NOM_CHAM='TEMP', VALE_REF=20.0),", comm)
        self.assertEqual(comm.count("AFFE_VARC"), 1)
        self.assertLess(comm.index("TEMP_EVOL = CREA_RESU("), comm.index("CHMAT = AFFE_MATERIAU("))
        self.assertIn("STRESS_EVERY_INSTANT = True", comm)

    def test_the_wall_stays_at_its_closure_temperature_until_the_reservoir_is_full(self):
        comm = self.render(self.case)
        history = comm[comm.index("TEMP_EVOL = CREA_RESU("):comm.index("CHMAT = AFFE_MATERIAU(")]
        self.assertEqual([line.strip() for line in history.splitlines()[1:-2]], [
            "_F(NOM_CHAM='TEMP', CHAM_GD=TEMP_0, INST=0.0),", "_F(NOM_CHAM='TEMP', CHAM_GD=TEMP_0, INST=1.0),",
            "_F(NOM_CHAM='TEMP', CHAM_GD=TEMP_0, INST=2.0),", "_F(NOM_CHAM='TEMP', CHAM_GD=TEMP_0, INST=3.0),",
            "_F(NOM_CHAM='TEMP', CHAM_GD=TEMP_0, INST=4.0),", "_F(NOM_CHAM='TEMP', CHAM_GD=TEMP_0, INST=5.0),",
            "_F(NOM_CHAM='TEMP', CHAM_GD=TEMP_1, INST=6.0),", "_F(NOM_CHAM='TEMP', CHAM_GD=TEMP_2, INST=7.0),"])
        for name, value in (("TEMP_0", "20.0"), ("TEMP_1", "10.0"), ("TEMP_2", "0.0")):
            self.assertIn(f"{name} = CREA_CHAMP(TYPE_CHAM='NOEU_TEMP_R', OPERATION='AFFE', MAILLAGE=MAIL,\n"
                          f"    AFFE=_F(TOUT='OUI', NOM_CMP='TEMP', VALE={value}))", comm)
        self.assertIn("INSTANTS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]", comm)

    def test_a_case_without_cooling_has_no_temperature_at_all(self):
        comm = self.render(fixture.fixture_case(False, "JOINT_MECA_FROT"))
        for absent in ("TEMP", "WALL_MAT", "AFFE_VARC", "ALPHA=1e-05", "GROUP_MA='WALL'"):
            self.assertNotIn(absent, comm)
        self.assertIn("STRESS_EVERY_INSTANT = False", comm)
        self.assertIn("INSTANTS = [1.0, 2.0, 3.0, 4.0, 5.0]", comm)

    def test_the_wall_takes_the_elastic_values_of_the_dam_case_layout(self):
        layout = "CONCRETE_MAT = DEFI_MATERIAU(\n    ELAS=_F(\n        E=35000000000,\n        NU=0.2,\n        RHO=2400,\n    ),\n)\n"
        base = self.base.replace("CONCRETE_MAT = DEFI_MATERIAU(ELAS=_F(E=30000000000.0, NU=0.2, RHO=2400.0))\n", layout)
        self.assertNotEqual(base, self.base)
        comm = self.render(self.case, base)
        self.assertIn("WALL_MAT = DEFI_MATERIAU(ELAS=_F(E=35000000000, NU=0.2, RHO=2400, ALPHA=1e-05))", comm)
        self.assertIn(layout, comm)
        with self.assertRaisesRegex(djb.JointCaseError, "'concrete material' section"):
            self.render(self.case, self.base.replace("CONCRETE_MAT = DEFI_MATERIAU(ELAS=", "CONCRETE_MAT = OTHER(ELAS="))

    def test_cooling_that_makes_no_sense_is_refused(self):
        # Closure at 20 C: no temperatures, a repeated one, a rising one, one above closure, one at closure.
        for temperatures, expansion in (([], 1e-5), ([10.0, 10.0], 1e-5), ([5.0, 10.0], 1e-5), ([25.0], 1e-5),
                                        ([20.0], 1e-5), ([15.0], 0.0)):
            with self.subTest(temperatures=temperatures, expansion=expansion):
                case = dict(self.case, cooling={"expansion_per_k": expansion, "closure_temperature_c": 20.0,
                                                "wall_temperatures_c": temperatures})
                with self.assertRaisesRegex(djb.JointCaseError, "below the closure temperature"):
                    djb.joint_parameters(djb.with_joints(self.arrays), self.cells, case)
        joint = djb.joint_parameters(djb.with_joints(self.arrays), self.cells, self.case)
        with self.assertRaisesRegex(djb.JointCaseError, "wall temperature for every instant"):
            djb.render_joint_comm(self.base, joint, fixture.fixture_loads(), djb.water_levels(fixture.fixture_loads(), 1.0),
                                  "", "")

    def test_the_dam_cooling_cases_are_the_app_cases_plus_provisional_cooling(self):
        for name, source in (("friction_configured_dry_cooling", "friction_configured_dry"),
                             ("friction_configured_uplift_cooling", "friction_configured_uplift")):
            case = dict(djb.CASES[name])
            self.assertEqual(case.pop("cooling"), {"expansion_per_k": 1e-5, "closure_temperature_c": 16.0,
                                                   "wall_temperatures_c": [12.0, 10.0, 8.0]})
            joint = djb.joint_parameters(djb.with_joints(self.arrays), self.cells, djb.CASES[name] | {"uplift": False})
            self.assertEqual(joint["cooling"]["steps_k"], [4.0, 6.0, 8.0])
            self.assertIn("PROVISIONAL", case.pop("purpose"))
            self.assertEqual(case, {key: value for key, value in djb.CASES[source].items() if key != "purpose"})
        # The app's stage does not run them.
        self.assertEqual([name for _, name in djb.STAGE_CASES], ["friction_configured_dry", "friction_configured_uplift"])
        self.assertFalse(any("cooling" in djb.CASES[name] for _, name in djb.STAGE_CASES))
        # The closure temperature given for the wall, and the coldest quoted air temperature as the last step.
        self.assertEqual(djb.CLOSURE_TEMPERATURE_C, 16.0)
        self.assertEqual(djb.WALL_TEMPERATURES_C[-1], djb.COLDEST_AIR_TEMPERATURE_C)
        self.assertEqual(djb.COLDEST_AIR_TEMPERATURE_C, 8.0)

    def test_progress_names_the_cooling_step(self):
        message = lambda line: djb.progress_message(line, self.levels)  # noqa: E731
        self.assertEqual(message("DAM_JOINT_STEP start instant=5"),
                         "Solving level 5 of 7 (reservoir at 3.00 m, 100% of the head)")
        self.assertEqual(message("DAM_JOINT_STEP done instant=6 seconds=20.2"),
                         "Converged level 6 of 7 (full reservoir, wall at 10 C, 10 K below closure) in 20 s")
        self.assertEqual(message("DAM_JOINT_STEP failed instant=7"),
                         "Did NOT converge at level 7 of 7 (full reservoir, wall at 0 C, 20 K below closure)")
        self.assertEqual(message("DAM_JOINT_STEP outoftime instant=7"),
                         "Ran out of the time limit at level 7 of 7 (full reservoir, wall at 0 C, 20 K below closure); "
                         "the levels before it are kept")

    def test_a_step_stopped_by_the_time_limit_is_not_called_a_convergence_failure(self):
        comm = self.render(self.case)
        self.assertIn('out_of_time = "MECANONLINE9_1" in detail', comm)
        self.assertIn('reason = "ran out of its time limit" if out_of_time else "did not converge"', comm)
        self.assertIn("""print(f"DAM_JOINT_STEP {'outoftime' if out_of_time else 'failed'} instant={target:.6g}", flush=True)""",
                      comm)

    def test_other_temperatures_can_be_given_for_a_cooling_case_only(self):
        case = djb.CASES["friction_configured_uplift_cooling"]
        self.assertIs(djb.with_wall_temperatures(case), case)
        changed = djb.with_wall_temperatures(case, wall_temperatures_c=[14, 12])
        self.assertEqual(changed["cooling"], {"expansion_per_k": 1e-5, "closure_temperature_c": 16.0,
                                              "wall_temperatures_c": [14.0, 12.0]})
        warmer = djb.with_wall_temperatures(case, closure_temperature_c=18)
        self.assertEqual(warmer["cooling"], {"expansion_per_k": 1e-5, "closure_temperature_c": 18.0,
                                             "wall_temperatures_c": [12.0, 10.0, 8.0]})
        # The stored case is not altered.
        self.assertEqual(case["cooling"], {"expansion_per_k": 1e-5, "closure_temperature_c": 16.0,
                                           "wall_temperatures_c": [12.0, 10.0, 8.0]})
        self.assertEqual({key: value for key, value in changed.items() if key != "cooling"},
                         {key: value for key, value in case.items() if key != "cooling"})
        with self.assertRaisesRegex(djb.JointCaseError, "only be given for a cooling case"):
            djb.with_wall_temperatures(djb.CASES["friction_configured_uplift"], wall_temperatures_c=[14, 12])
        # A closure temperature below the wall temperatures is refused when the case is prepared.
        with self.assertRaisesRegex(djb.JointCaseError, "below the closure temperature"):
            djb.joint_parameters(djb.with_joints(self.arrays), self.cells,
                                 djb.with_wall_temperatures(case, closure_temperature_c=10) | {"uplift": False})

    def write_run(self, directory: Path, contraction: float, block_contraction: float = 0.0) -> dict:
        """A run whose wall shrinks about its centre by ``contraction`` (a strain) after full water."""
        record = djb.write_case(directory, self.arrays, {"bedrock_analysis_mode": "elastic_layer"}, self.base,
                                fixture.fixture_loads(), self.case)
        with np.load(directory / djb.ARRAYS_NAME) as stored:
            arrays = {key: stored[key] for key in stored.files}
        nodes, body = arrays["nodes"], arrays["concrete_body_ids"]
        wall = np.unique(arrays["tetra4_concrete"][body == 1])
        block = np.unique(arrays["tetra4_concrete"][body == 3])
        full = np.zeros_like(nodes)
        full[wall, 0] = 0.001  # the state at full water, before any cooling
        centre = np.array([fixture.THICKNESS_M / 2, fixture.WIDTH_M / 2, fixture.BLOCK_HEIGHT_M])
        cooled = full.copy()
        cooled[wall] -= contraction * (nodes[wall] - centre)
        cooled[block] -= block_contraction * (nodes[block] - centre)
        cells = arrays["penta6_joints"]
        ids = np.arange(len(cells))
        zeros = np.zeros((3, len(cells)))
        stress = np.zeros((3, len(nodes), 6))
        stress[1, wall[0], 0], stress[2, wall[-1], 2], stress[2, wall[0], 1] = 1.5e6, 2.5e6, -4e6
        np.savez(directory / djb.STEPS_NAME, instant=np.array([5.0, 6.0, 7.0]),
                 displacement_m=np.array([full, (full + cooled) / 2, cooled]), gauss_cells=ids, joint_cell_ids=ids,
                 stress_by_instant_pa=stress, stress_pa=stress[-1], stress_instant=np.asarray(7.0),
                 **{name: zeros for name in ("V1", "V2", "V7", "V8", "V9", "V18", "SIGN", "SITX", "SITY")})
        (directory / djb.STATUS_NAME).write_text(json.dumps({"status": "completed", "converged_instants": [5.0, 6.0, 7.0]}))
        return {"record": record, "nodes": nodes, "wall": wall}

    def test_report_gives_the_temperature_and_the_wall_stress_at_each_level(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.write_run(Path(directory), 2e-4)
            report = djb.summarise(Path(directory))
        cooling = {"expansion_per_k": 1e-5, "closure_temperature_c": 20.0, "wall_temperatures_c": [10.0, 0.0],
                   "steps_k": [10.0, 20.0], "cooled_body": "wall", "provisional": True}
        note = djb.cooling_note(cooling)
        self.assertIn("from its closure temperature of 20 C to 0 C", note)
        self.assertIn("PROVISIONAL: the expansion coefficient (1e-05 per kelvin)", note)
        self.assertIn("not a construction record", note)
        self.assertEqual(run["record"]["provisional_inputs"], [note])
        self.assertEqual(report["provisional_inputs"], [note])
        self.assertEqual(report["joint"]["cooling"], cooling)
        self.assertFalse(report["engineering_validated"])
        self.assertEqual([level["wall_temperature_change_k"] for level in report["levels"]], [0.0, -10.0, -20.0])
        self.assertEqual([level["wall_temperature_c"] for level in report["levels"]], [20.0, 10.0, 0.0])
        self.assertEqual([level["fill_fraction"] for level in report["levels"]], [1.0, 1.0, 1.0])
        np.testing.assert_allclose([level["maximum_principal_tensile_stress_pa"] for level in report["levels"]],
                                   [0.0, 1.5e6, 2.5e6])
        self.assertAlmostEqual(report["levels"][2]["maximum_principal_compressive_stress_pa"], -4e6)
        self.assertEqual(report["levels"][1]["maximum_principal_tensile_stress_xyz_m"], run["nodes"][run["wall"][0]].tolist())
        self.assertEqual(report["levels"][2]["maximum_principal_tensile_stress_xyz_m"], run["nodes"][run["wall"][-1]].tolist())
        self.assertAlmostEqual(report["final_wall_stress"]["maximum_principal_tensile_stress_pa"], 2.5e6)

    def test_a_report_without_cooling_keeps_its_earlier_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            ReportTests().write_run(Path(directory), lambda centroid: np.zeros(len(centroid)))
            report = djb.summarise(Path(directory))
        self.assertNotIn("provisional_inputs", report)
        self.assertNotIn("cooling", report["joint"])
        for level in report["levels"]:
            self.assertNotIn("wall_temperature_change_k", level)
            self.assertNotIn("wall_temperature_c", level)
            self.assertNotIn("maximum_principal_tensile_stress_pa", level)

    def test_fixture_check_accepts_free_contraction_and_refuses_anything_else(self):
        free = 1e-5 * 20.0
        with tempfile.TemporaryDirectory() as directory:
            self.write_run(Path(directory), free)
            check = fixture.compare_cooling(Path(directory))
            self.assertTrue(check["passed"], check)
            self.assertAlmostEqual(check["measured"]["across_top_m"], -free * fixture.THICKNESS_M)
            self.assertAlmostEqual(check["measured"]["along_top_m"], -free * fixture.WIDTH_M)
            self.assertAlmostEqual(check["measured"]["upper_half_height_m"], -free * fixture.WALL_HEIGHT_M / 2)
            self.assertEqual(check["temperature_change_k"], -20.0)
            for strain, block in ((0.9 * free, 0.0), (0.0, 0.0), (-free, 0.0), (free, 0.2 * free)):
                with self.subTest(strain=strain, block=block):
                    self.write_run(Path(directory), strain, block)
                    self.assertFalse(fixture.compare_cooling(Path(directory))["passed"])
            # A run that stopped before the last cooling step is not a pass.
            self.write_run(Path(directory), free)
            with np.load(Path(directory) / djb.STEPS_NAME) as stored:
                data = {key: stored[key] for key in stored.files}
            np.savez(Path(directory) / djb.STEPS_NAME, **{**data, "instant": np.array([4.0, 5.0, 6.0])})
            stopped = fixture.compare_cooling(Path(directory))
            self.assertFalse(stopped["passed"])
            self.assertIn("did not reach the last cooling step", stopped["message"])


def split_wall(arrays: dict, z: float, compact: bool = False) -> dict:
    """The fixture's wall cut into two parts at height ``z`` and joined by springs, as a stepped wall's parts are."""
    nodes, cells, body = arrays["nodes"], arrays["tetra4_concrete"].copy(), arrays["concrete_body_ids"]
    wall = body == 1
    upper = wall & (nodes[cells][:, :, 2].mean(axis=1) > z)
    plane = np.intersect1d(np.flatnonzero(np.isclose(nodes[:, 2], z)), cells[wall])
    copy = np.arange(len(nodes))
    copy[plane] = len(nodes) + np.arange(len(plane))
    lower_faces = cells[wall & ~upper][:, djb._TET_FACES].reshape(-1, 3)
    lower_faces = lower_faces[np.isclose(nodes[lower_faces][:, :, 2], z).all(axis=1)]
    cells[upper] = copy[cells[upper]]
    pairs = np.column_stack([copy[plane], plane])
    stack = lambda key, rows: np.concatenate([arrays[key], rows])  # noqa: E731
    return dict(
        arrays, nodes=np.vstack([nodes, nodes[plane]]), tetra4_concrete=cells,
        joint_slave_tri3=stack("joint_slave_tri3", copy[lower_faces]),
        joint_master_tri3=stack("joint_master_tri3", lower_faces),
        joint_coefficients_n_per_m3=stack("joint_coefficients_n_per_m3", np.tile([2.0e8, 2.0e8, 5.0e8], (len(lower_faces), 1))),
        seg2_wall_foundation_springs=stack("seg2_wall_foundation_springs", pairs),
        wall_foundation_tributary_area_m2=stack("wall_foundation_tributary_area_m2", np.full(len(pairs), 0.5)),
        interface_stiffness_n_per_m=stack("interface_stiffness_n_per_m", np.tile([1.0e8, 1.0e8, 2.5e8], (len(pairs), 1))),
        interface_names=stack("interface_names", np.array(["wall_joint_1"] * len(pairs))),
        compact_spring_groups=np.array(compact))


class SteppedWallTests(unittest.TestCase):
    """A wall with thickness steps is meshed in parts with horizontal joints between them."""

    def setUp(self):
        self.arrays = fixture.fixture_arrays(across=4, along=2, wall_layers=4, block_layers=1)
        self.split = split_wall(self.arrays, 2.0)
        self.internal_pairs = len(self.split["seg2_wall_foundation_springs"]) - len(self.arrays["seg2_wall_foundation_springs"])

    def test_joint_cells_are_placed_on_the_foundation_contact_only(self):
        wall, other, on_foundation = djb.contact_sides(self.split)
        self.assertEqual(int(on_foundation.sum()), len(self.arrays["joint_slave_tri3"]))
        self.assertEqual(int((~on_foundation).sum()), 16)  # the 4 x 2 squares of the cut, two triangles each
        np.testing.assert_allclose(self.split["nodes"][wall[~on_foundation]][:, :, 2], 2.0)
        np.testing.assert_array_equal(djb.joint_cells(self.split), djb.joint_cells(self.arrays))
        # A joint between wall parts alone is not a contact to put the joint law on.
        only_internal = dict(self.split, joint_slave_tri3=wall[~on_foundation], joint_master_tri3=other[~on_foundation])
        with self.assertRaisesRegex(djb.JointCaseError, "no matched wall/foundation"):
            djb.joint_cells(only_internal)

    def test_stiffness_is_taken_from_the_foundation_contact(self):
        joined = djb.with_joints(self.split)
        self.assertEqual(djb.configured_stiffness(joined), fixture.JOINT_STIFFNESS_PA_PER_M)
        # Without knowing which triangles are on the foundation the two stiffnesses cannot be told apart.
        with self.assertRaisesRegex(djb.JointCaseError, "differs by direction"):
            djb.configured_stiffness(self.split)

    def test_only_the_springs_between_wall_parts_stay_in_the_mesh(self):
        joined = djb.with_joints(self.split)
        self.assertEqual(self.internal_pairs, 15)
        self.assertEqual(int(joined["internal_wall_joint_pairs"]), 15)
        np.testing.assert_array_equal(joined["seg2_wall_foundation_springs"],
                                      self.split["seg2_wall_foundation_springs"][-15:])
        np.testing.assert_array_equal(joined["interface_stiffness_n_per_m"], np.tile([1.0e8, 1.0e8, 2.5e8], (15, 1)))
        self.assertEqual(set(joined["interface_names"].tolist()), {"wall_joint_1"})
        self.assertEqual(len(joined["wall_foundation_tributary_area_m2"]), 15)
        # A wall in one part keeps its arrays as they were: its springs stay in the file without a model.
        whole = djb.with_joints(self.arrays)
        self.assertEqual(int(whole["internal_wall_joint_pairs"]), 0)
        np.testing.assert_array_equal(whole["seg2_wall_foundation_springs"], self.arrays["seg2_wall_foundation_springs"])
        self.assertEqual(djb.internal_joint_spring_lines(whole), [])

    def test_spring_lines_use_the_group_names_of_the_mesh_conversion(self):
        lines = djb.internal_joint_spring_lines(djb.with_joints(self.split))
        names = ", ".join(f"'SPR_WF_{row:05d}'" for row in range(15))
        self.assertEqual(lines, [f"        _F(GROUP_MA=({names}), REPERE='GLOBAL', CARA='K_T_D_L', "
                                 "VALE=(100000000, 100000000, 250000000)),"])
        compact = split_wall(self.arrays, 2.0, compact=True)
        compact["interface_stiffness_n_per_m"][-1] = [3.0e8, 3.0e8, 7.5e8]
        self.assertEqual(djb.internal_joint_spring_lines(djb.with_joints(compact)), [
            "        _F(GROUP_MA='K_IF_00000', REPERE='GLOBAL', CARA='K_T_D_L', VALE=(100000000, 100000000, 250000000)),",
            "        _F(GROUP_MA='K_IF_00001', REPERE='GLOBAL', CARA='K_T_D_L', VALE=(300000000, 300000000, 750000000)),"])

    def test_case_file_keeps_the_internal_springs_beside_the_joint_layer(self):
        joined = djb.with_joints(self.split)
        joint = djb.joint_parameters(joined, joined["penta6_joints"], fixture.fixture_case(False, "JOINT_MECA_FROT"))
        comm = djb.render_joint_comm(fixture.fixture_base_comm(self.split), joint, fixture.fixture_loads(),
                                     djb.water_levels(fixture.fixture_loads(), 1.0), "/steps.npz", "/status.json",
                                     djb.internal_joint_spring_lines(joined))
        compile(comm.replace("from code_aster.Commands import *", "").replace("from code_aster import CA", ""),
                "case.comm", "exec")
        self.assertIn("        _F(GROUP_MA='WALL_FOUNDATION_SPRINGS', PHENOMENE='MECANIQUE', MODELISATION='DIS_T'),\n"
                      "        _F(GROUP_MA='JOINTS', PHENOMENE='MECANIQUE', MODELISATION='3D_JOINT'),\n", comm)
        self.assertEqual(comm.count("CARA='K_T_D_L'"), 1)  # the baseline's own lines for every pair are gone
        self.assertIn("VALE=(100000000, 100000000, 250000000)),", comm)
        self.assertLess(comm.index("CARA_ROCK = AFFE_CARA_ELEM("), comm.index("CARA='K_T_D_L'"))
        self.assertLess(comm.index("CARA='K_T_D_L'"), comm.index("CARA='K_T_N'"))
        self.assertIn("JOINT_MECA_FROT=_F(K_N=200000000, K_T=200000000,", comm)

    def test_written_case_records_the_internal_joints(self):
        with tempfile.TemporaryDirectory() as directory:
            record = djb.write_case(Path(directory), self.split, {"bedrock_analysis_mode": "elastic_layer"},
                                    fixture.fixture_base_comm(self.split), fixture.fixture_loads(),
                                    fixture.fixture_case(False, "JOINT_MECA_FROT"))
            with np.load(Path(directory) / djb.ARRAYS_NAME) as stored:
                self.assertEqual(len(stored["seg2_wall_foundation_springs"]), 15)
                self.assertEqual(int(stored["joint_on_foundation"].sum()), len(stored["penta6_joints"]))
            comm = (Path(directory) / djb.CASE_NAME).read_text(encoding="utf-8")
            ReportTests.write_results(Path(directory))
            report = djb.summarise(Path(directory))
        expected = {"contact_triangles": 16, "spring_pairs": 15,
                    "treatment": "linear springs of the elastic baseline (they hold tension)"}
        self.assertEqual(record["internal_wall_joints"], expected)
        self.assertEqual(report["internal_wall_joints"], expected)
        self.assertEqual(comm.count("CARA='K_T_D_L'"), 1)
        self.assertEqual(record["joint_cell_count"], len(self.arrays["joint_slave_tri3"]))
        with tempfile.TemporaryDirectory() as directory:
            record = djb.write_case(Path(directory), self.arrays, {}, fixture.fixture_base_comm(self.arrays),
                                    fixture.fixture_loads(), fixture.fixture_case(False, "JOINT_MECA_FROT"))
        self.assertNotIn("internal_wall_joints", record)


class SteppedJointWaterTests(unittest.TestCase):
    """Water in the joint under a wall that is thicker below a step: the downstream edge moves out."""

    def setUp(self):
        angles = np.linspace(0.2, 1.2, 9)
        ring = lambda radius, z: np.column_stack([10 + radius * np.cos(angles), -5 + radius * np.sin(angles),  # noqa: E731
                                                  np.full(len(angles), z)])
        # Upstream at 80 m. Downstream at 76 m above z = -5 and at 75 m below it, with a ledge between.
        self.nodes = np.vstack([ring(80.0, 0.0), ring(80.0, -10.0), ring(76.0, 0.0), ring(76.0, -5.0),
                                ring(75.0, -5.0), ring(75.0, -10.0), ring(77.0, -2.0), ring(75.5, -8.0)])
        strip = lambda top, bottom: np.array([[top + i, top + i + 1, bottom + i] for i in range(8)]  # noqa: E731
                                             + [[top + i + 1, bottom + i + 1, bottom + i] for i in range(8)])
        self.arrays = {"nodes": self.nodes, "tri3_upstream": strip(0, 9),
                       "tri3_downstream": np.vstack([strip(18, 27), strip(27, 36), strip(36, 45)])}
        # Contact cells: one above the step (radius 77) and one below it (radius 75.5).
        self.cells = np.array([[54, 55, 56, 54, 55, 56], [63, 64, 65, 63, 64, 65]])
        self.at = lambda radius, z: [10 + radius * np.cos(0.7), -5 + radius * np.sin(0.7), z]

    def test_the_downstream_radius_follows_the_steps(self):
        position = djb.arch_position(self.arrays, self.cells)
        self.assertEqual(position["kind"], "radial_stepped")
        np.testing.assert_allclose(position["centre_xy_m"], [10.0, -5.0], atol=1e-6)
        self.assertAlmostEqual(position["upstream_radius_m"], 80.0, places=6)
        self.assertEqual([band["above_z_m"] for band in position["bands"]], [-5.0, None])
        np.testing.assert_allclose([band["downstream_radius_m"] for band in position["bands"]], [76.0, 75.0], atol=1e-6)
        points = np.array([self.at(80.0, -2.0), self.at(78.0, -2.0), self.at(76.0, -2.0),      # above the step: 4 m thick
                           self.at(80.0, -8.0), self.at(77.5, -8.0), self.at(75.0, -8.0),      # below it: 5 m thick
                           self.at(76.0, -5.0), self.at(76.0, -5.0 - 1e-3)])                   # at the step, and just under
        # Reservoir at 0 m, tailwater at -6 m.
        pressure = djb.joint_pressure(position, 9810.0, 0.0, -6.0, points)
        np.testing.assert_allclose(pressure, [
            19620.0, 9810.0, 0.0,
            78480.0, (78480.0 + 19620.0) / 2, 19620.0,
            0.0, 0.2 * 9810.0 * 5.001], rtol=1e-6, atol=1e-3)  # the tailwater is below both of the last two

    def test_the_formula_is_arithmetic_code_aster_can_evaluate_point_by_point(self):
        position = djb.arch_position(self.arrays, self.cells)
        expression = djb.position_expression(position)
        self.assertIn("* (Z > -5.000001)", expression)
        self.assertNotIn(" if ", expression)
        import math
        scalar = lambda point: eval(expression, {"sqrt": math.sqrt, "min": min, "max": max,  # noqa: E731
                                                 "X": point[0], "Y": point[1], "Z": point[2]})
        self.assertAlmostEqual(scalar(self.at(78.0, -2.0)), 0.5, places=6)
        self.assertAlmostEqual(scalar(self.at(77.5, -8.0)), 0.5, places=6)
        self.assertAlmostEqual(scalar(self.at(74.0, -8.0)), 1.0, places=6)

    def test_three_parts_and_shapes_that_are_not_steps(self):
        three = djb._downstream_radius({"bands": [{"above_z_m": -20.0, "downstream_radius_m": 76.0},
                                                  {"above_z_m": -25.0, "downstream_radius_m": 75.5},
                                                  {"above_z_m": None, "downstream_radius_m": 75.0}]},
                                       np.array([0.0, -20.0, -20.1, -25.0, -25.1, -29.0]))
        np.testing.assert_allclose(three, [76.0, 76.0, 75.5, 75.5, 75.0, 75.0])
        # A contact that reaches further downstream than the wall above it does not fit the steps.
        outside = np.array([[63, 64, 65, 63, 64, 65]])  # at radius 75.5, moved up beside the 4 m part
        moved = dict(self.arrays, nodes=self.nodes.copy())
        moved["nodes"][63:66, 2] = -2.0
        with self.assertRaisesRegex(djb.JointCaseError, "uniform thickness or with thickness steps"):
            djb.arch_position(moved, outside)
        # A downstream face that leans is not a step.
        leaning = dict(self.arrays, nodes=self.nodes.copy())
        leaning["nodes"][36:45, 2] = -6.0
        with self.assertRaisesRegex(djb.JointCaseError, "does not fit"):
            djb.arch_position(leaning, self.cells)


class BundledTestLogTests(unittest.TestCase):
    def test_log_outcomes(self):
        ok = " OK  NON_REGRESSION  XXXX\n OK  ANALYTIQUE  XXXX\n--- DIAGNOSTIC JOB : OK\n"
        self.assertEqual(bundled_dam_tests.parse_log(ok, 0)["checks_ok"], 2)
        self.assertTrue(bundled_dam_tests.parse_log(ok, 0)["passed"])
        alarm = bundled_dam_tests.parse_log(" OK  a\n <A> <MODELISA8_15>\n--- DIAGNOSTIC JOB : <A>_ALARM\n", 0)
        self.assertTrue(alarm["passed"])
        self.assertEqual(alarm["alarms"], ["MODELISA8_15"])
        self.assertFalse(bundled_dam_tests.parse_log(" OK  a\nNOOK  b\n--- DIAGNOSTIC JOB : NOOK_TEST_RESU\n", 0)["passed"])
        self.assertFalse(bundled_dam_tests.parse_log(ok, 1)["passed"])
        self.assertFalse(bundled_dam_tests.parse_log("--- DIAGNOSTIC JOB : OK\n", 0)["passed"])  # no check ran
        self.assertFalse(bundled_dam_tests.parse_log(" OK  a\n", 0)["passed"])  # no diagnostic: the run was cut off


if __name__ == "__main__":
    unittest.main()
