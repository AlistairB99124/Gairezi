"""Specimen verification of exact-bond/cohesive/contact interface equilibrium.

Two linear-elastic P1 tetrahedral blocks are joined by duplicated interface
lips. Lateral supports are statically determinate, so the uniaxial field is
exactly representable and closed-form force/opening relations apply.
"""
import dataclasses
import unittest

import numpy as np
from scipy import sparse

from FEniCSx.cohesive_contact import (
    CohesiveInterface, CohesiveOptions, InterfaceState, ReservoirCrackWater, advance_with_initiation,
    solve_cohesive_equilibrium, solve_wet_cohesive_equilibrium)
from FEniCSx.cohesive_mode_i import LinearModeICohesiveLaw
from FEniCSx.fracture_faces import PairedCrackFaces

E, NU = 35e9, .2
FT, GF = 2e6, 400.
WIDTH, HEIGHT = 1., .5
KUHN = ((0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7))


def elasticity():
    lame, shear = E * NU / ((1 + NU) * (1 - 2 * NU)), E / (2 * (1 + NU))
    matrix = np.full((3, 3), lame) + 2 * shear * np.eye(3)
    return np.block([[matrix, np.zeros((3, 3))], [np.zeros((3, 3)), shear * np.eye(3)]])


def tetra_stiffness(points):
    gradients = np.linalg.inv(np.hstack((np.ones((4, 1)), points)))[1:]
    volume = abs(np.linalg.det(points[1:] - points[0])) / 6
    strain = np.zeros((6, 12))
    for node in range(4):
        gx, gy, gz = gradients[:, node]
        strain[:, 3 * node:3 * node + 3] = [
            [gx, 0, 0], [0, gy, 0], [0, 0, gz], [gy, gx, 0], [0, gz, gy], [gz, 0, gx]]
    return volume * strain.T @ elasticity() @ strain


class Specimen:
    def __init__(self, divisions=1, layers=1):
        n = divisions
        xs = np.linspace(0, WIDTH, n + 1)
        coordinates, cells, lip = [], [], {}

        def grid(z_values, duplicate_bottom):
            ids = {}
            for k, z in enumerate(z_values):
                for j, y in enumerate(xs):
                    for i, x in enumerate(xs):
                        ids[i, j, k] = len(coordinates)
                        coordinates.append((x, y, z))
            return ids

        lower = grid(np.linspace(0, HEIGHT, layers + 1), False)
        upper = grid(np.linspace(HEIGHT, 2 * HEIGHT, layers + 1), True)
        for ids in (lower, upper):
            for k in range(layers):
                for j in range(n):
                    for i in range(n):
                        corners = [ids[i + a, j + b, k + c] for c in (0, 1) for b in (0, 1) for a in (0, 1)]
                        cells.extend([[corners[v] for v in tet] for tet in KUHN])
        self.coordinates = np.asarray(coordinates, float)
        self.cells = np.asarray(cells)
        minus, plus = [], []
        self.parents = np.arange(len(coordinates))
        for j in range(n + 1):
            for i in range(n + 1):
                self.parents[upper[i, j, 0]] = lower[i, j, layers]
        self.inlet = np.array([[lower[0, 0, layers], lower[0, 1, layers]]])
        for j in range(n):
            for i in range(n):
                quad_minus = [lower[i, j, layers], lower[i + 1, j, layers], lower[i + 1, j + 1, layers], lower[i, j + 1, layers]]
                quad_plus = [upper[i, j, 0], upper[i + 1, j, 0], upper[i + 1, j + 1, 0], upper[i, j + 1, 0]]
                for triangle in ((0, 1, 2), (0, 2, 3)):
                    minus.append([quad_minus[v] for v in triangle])
                    plus.append([quad_plus[v] for v in triangle])
        self.faces = PairedCrackFaces(self.coordinates, minus, plus, np.tile([0., 0., 1.], (len(minus), 1)))
        self.interface = CohesiveInterface.from_faces(self.faces, maximum_normal_deviation_rad=1e-3)
        size = 3 * len(self.coordinates)
        rows, columns, values = [], [], []
        for cell in self.cells:
            local = tetra_stiffness(self.coordinates[cell])
            dofs = (3 * cell[:, None] + np.arange(3)).ravel()
            rows.append(np.repeat(dofs, 12))
            columns.append(np.tile(dofs, 12))
            values.append(local.ravel())
        stiffness = sparse.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(size, size))
        self.stiffness = (.5 * (stiffness + stiffness.T)).tocsr()
        z = self.coordinates[:, 2]
        self.bottom = np.flatnonzero(np.isclose(z, 0))
        self.top = np.flatnonzero(np.isclose(z, 2 * HEIGHT))
        self.mapping = np.arange(len(self.coordinates))

    def pins(self, nodes):
        origin = nodes[np.argmin(np.linalg.norm(self.coordinates[nodes, :2], axis=1))]
        corner = nodes[np.argmin(np.linalg.norm(self.coordinates[nodes, :2] - [WIDTH, 0], axis=1))]
        return [3 * origin, 3 * origin + 1, 3 * corner + 1]

    def problem(self, pin_top=True, rotation=0.):
        def at(delta):
            fixed = [3 * node + 2 for node in np.concatenate((self.bottom, self.top))] + self.pins(self.bottom)
            if pin_top:
                fixed += self.pins(self.top)
            prescribed = np.zeros(len(fixed))
            top_x = self.coordinates[self.top, 0]
            prescribed[len(self.bottom):len(self.bottom) + len(self.top)] = delta + rotation * delta * (top_x - WIDTH / 2)
            return self.stiffness, np.zeros(3 * len(self.coordinates)), np.asarray(fixed), prescribed
        return at

    def force(self, result):
        return float(result.support_reaction_n[3 * self.top + 2].sum())


def options(tangential="tied"):
    return CohesiveOptions(gap_tolerance_m=1e-12, force_tolerance_n=1e-3, relative_residual_tolerance=1e-9,
                           initiation_relative_tolerance=1e-7, tangential_response=tangential)


LAW = LinearModeICohesiveLaw(FT, GF)
DELTA0 = 2 * HEIGHT * FT / E
AREA = WIDTH ** 2


def analytic_force(delta):
    """Monotone displacement-control response of the uniaxial specimen."""
    if delta <= DELTA0:
        return E * AREA * delta / (2 * HEIGHT)
    compliance = 2 * HEIGHT / E
    wc = LAW.critical_opening_m
    opening = (delta - compliance * FT) / (1 - compliance * FT / wc)
    return AREA * FT * max(1 - opening / wc, 0.) if opening < wc else 0.


def trace(specimen, deltas, state=None, tangential="tied", water=None, **problem):
    state = state or InterfaceState.intact(specimen.interface)
    path = [(0., 0., None)]
    at = specimen.problem(**problem)
    for start, end in zip(deltas[:-1], deltas[1:]):
        for result in advance_with_initiation(at, start, end, specimen.interface, specimen.mapping, LAW,
                                              state, options=options(tangential),
                                              faces=None if water is None else specimen.faces, water=water):
            path.append((result.report["load_factor"], specimen.force(result), result))
            state = result.trial_state
    return path, state


HEAD_ABOVE = 100.
UNIT_WEIGHT = 9810.
PRESSURE = UNIT_WEIGHT * HEAD_ABOVE


def reservoir(specimen, inlet=None):
    return ReservoirCrackWater(HEIGHT + HEAD_ABOVE, UNIT_WEIGHT,
                               specimen.inlet if inlet is None else inlet, specimen.parents, 1e-10)


def wet_analytic_force(delta):
    compliance, wc = 2 * HEIGHT / E, LAW.critical_opening_m
    opening = (delta - compliance * (FT - PRESSURE)) / (1 - compliance * FT / wc)
    if opening >= wc:
        return -PRESSURE * AREA
    return AREA * (FT * (1 - opening / wc) - PRESSURE)


class CohesiveSpecimenTests(unittest.TestCase):
    def test_bonded_interface_preserves_continuum_stiffness(self):
        specimen = Specimen()
        stiffness, load, fixed, prescribed = specimen.problem()(.5 * DELTA0)
        result = solve_cohesive_equilibrium(stiffness, load, fixed, prescribed, specimen.interface,
                                            specimen.mapping, LAW, InterfaceState.intact(specimen.interface),
                                            options=options())
        self.assertEqual(result.report["status"], "passed_cohesive_equilibrium")
        self.assertAlmostEqual(specimen.force(result), analytic_force(.5 * DELTA0), delta=1e-6 * FT)
        np.testing.assert_allclose(result.normal_traction_pa, FT / 2, rtol=1e-9)
        self.assertLess(result.report["maximum_constrained_jump_m"], 1e-15)
        self.assertEqual(set(result.pair_status), {"bonded"})

    def test_initiation_is_located_and_force_is_continuous(self):
        specimen = Specimen()
        path, state = trace(specimen, [0., 1.5 * DELTA0])
        loads = [row[0] for row in path]
        crossing = next(row for row in path[1:] if row[2].report["initiated_pairs"])
        self.assertAlmostEqual(crossing[0], DELTA0, delta=1e-6 * DELTA0)
        self.assertAlmostEqual(crossing[1], FT * AREA, delta=1e-5 * FT * AREA)
        before = [row for row in path[1:] if not row[2].report["initiated_pairs"]]
        self.assertAlmostEqual(before[-1][1], FT * AREA, delta=1e-5 * FT * AREA)
        self.assertTrue(state.initiated.all())
        self.assertEqual(loads[-1], 1.5 * DELTA0)
        for delta, force, _ in path[1:]:
            self.assertAlmostEqual(force, analytic_force(delta), delta=1e-6 * FT * AREA)

    def test_softening_dissipates_fracture_energy_exactly(self):
        specimen = Specimen()
        separation = LAW.critical_opening_m
        deltas = [0., 2 * DELTA0, .5 * separation, separation, 1.2 * separation]
        path, state = trace(specimen, deltas)
        work = sum(.5 * (f0 + f1) * (d1 - d0) for (d0, f0, _), (d1, f1, _) in zip(path[:-1], path[1:]))
        self.assertAlmostEqual(work, GF * AREA, delta=1e-5 * GF * AREA)
        final = path[-1][2]
        self.assertAlmostEqual(final.report["cohesive_dissipated_energy_j"], GF * AREA, delta=1e-9)
        self.assertAlmostEqual(path[-1][1], 0., delta=1e-6)
        self.assertEqual(set(final.pair_status), {"separated"})
        np.testing.assert_allclose(state.maximum_opening_m, 1.2 * separation, rtol=1e-9)

    def test_unloading_is_secant_and_reloading_does_not_heal(self):
        specimen = Specimen()
        peak = .5 * LAW.critical_opening_m
        path, state = trace(specimen, [0., 2 * DELTA0, peak])
        committed_opening = state.maximum_opening_m.copy()
        envelope_force = path[-1][1]
        path, unloaded = trace(specimen, [peak, .5 * peak], state=state)
        np.testing.assert_array_equal(unloaded.maximum_opening_m, committed_opening)
        secant_stiffness = envelope_force / AREA / committed_opening[0]
        compliance = 2 * HEIGHT / E
        expected = AREA * .5 * peak / (compliance + 1 / secant_stiffness)
        self.assertAlmostEqual(path[-1][1], expected, delta=1e-6 * FT * AREA)
        path, reloaded = trace(specimen, [.5 * peak, peak], state=unloaded)
        self.assertAlmostEqual(path[-1][1], envelope_force, delta=1e-6 * FT * AREA)
        self.assertTrue(reloaded.initiated.all())

    def test_compression_closes_crack_without_healing_history(self):
        specimen = Specimen()
        peak = .5 * LAW.critical_opening_m
        _, state = trace(specimen, [0., 2 * DELTA0, peak])
        stiffness, load, fixed, prescribed = specimen.problem()(-DELTA0)
        result = solve_cohesive_equilibrium(stiffness, load, fixed, prescribed, specimen.interface,
                                            specimen.mapping, LAW, state, options=options())
        self.assertEqual(set(result.pair_status), {"closed"})
        self.assertAlmostEqual(specimen.force(result), -E * AREA * DELTA0 / (2 * HEIGHT), delta=1e-6 * FT)
        np.testing.assert_allclose(result.normal_traction_pa, -FT, rtol=1e-8)
        np.testing.assert_array_equal(result.trial_state.maximum_opening_m, state.maximum_opening_m)
        self.assertTrue(result.trial_state.initiated.all())

    def test_mesh_refinement_gives_identical_global_response(self):
        deltas = [0., 2 * DELTA0, .6 * LAW.critical_opening_m]
        coarse, _ = trace(Specimen(1, 1), deltas)
        fine, _ = trace(Specimen(2, 2), deltas)
        self.assertAlmostEqual(coarse[-1][1], fine[-1][1], delta=1e-6 * FT * AREA)
        self.assertAlmostEqual(fine[-1][1], analytic_force(deltas[-1]), delta=1e-6 * FT * AREA)

    def test_tangential_choices_agree_in_mode_i_and_frictionless_mechanism_is_rejected(self):
        deltas = [0., 2 * DELTA0]
        tied, _ = trace(Specimen(), deltas, tangential="tied")
        free, _ = trace(Specimen(), deltas, tangential="frictionless")
        self.assertAlmostEqual(tied[-1][1], free[-1][1], delta=1e-6 * FT * AREA)
        with self.assertRaisesRegex(RuntimeError, "singular"):
            trace(Specimen(), deltas, tangential="frictionless", pin_top=False)

    def test_eccentric_loading_initiates_progressively_without_accepted_overstress(self):
        specimen = Specimen(2, 1)
        path, state = trace(specimen, [0., 1.4 * DELTA0], rotation=1.)
        accepted = [row[2] for row in path[1:]]
        for result in accepted:
            ratio = result.report["maximum_bonded_tensile_ratio"]
            if ratio is not None:
                self.assertLessEqual(ratio, 1 + 1e-7)
        counts = [result.report["initiated_pairs"] for result in accepted]
        self.assertEqual(counts, sorted(counts))
        self.assertGreater(state.initiated.sum(), 0)
        self.assertLess(state.initiated.sum(), specimen.interface.count)
        self.assertTrue(state.initiated[np.argmax(specimen.coordinates[specimen.interface.minus_nodes, 0])])

    def test_stalled_newton_restarts_from_the_separated_state(self):
        # A brittle law snaps back (w_c below the elastic stretch at f_t); with a tiny Newton budget the
        # solver must restart from the separated state and still accept only the true equilibrium.
        specimen = Specimen()
        brittle = LinearModeICohesiveLaw(FT, 1.)
        state = InterfaceState.intact(specimen.interface).initiate(np.arange(specimen.interface.count))
        stiffness, load, fixed, prescribed = specimen.problem()(2 * DELTA0)
        result = solve_cohesive_equilibrium(
            stiffness, load, fixed, prescribed, specimen.interface, specimen.mapping, brittle, state,
            options=dataclasses.replace(options(), maximum_newton_iterations=2))
        self.assertEqual(result.report["status"], "passed_cohesive_equilibrium")
        self.assertGreater(result.report["newton_restarts"], 0)
        self.assertEqual(set(result.pair_status), {"separated"})
        self.assertAlmostEqual(specimen.force(result), 0., delta=1e-6)

    def test_solve_does_not_mutate_committed_state(self):
        specimen = Specimen()
        state = InterfaceState.intact(specimen.interface).initiate(np.arange(specimen.interface.count))
        stiffness, load, fixed, prescribed = specimen.problem()(2 * DELTA0)
        result = solve_cohesive_equilibrium(stiffness, load, fixed, prescribed, specimen.interface,
                                            specimen.mapping, LAW, state, options=options())
        self.assertTrue(np.all(result.trial_state.maximum_opening_m > 0))
        np.testing.assert_array_equal(state.maximum_opening_m, 0.)
        with self.assertRaises(ValueError):
            state.maximum_opening_m[0] = 1.
        with self.assertRaisesRegex(ValueError, "already initiated"):
            state.initiate(np.array([0]))

    def test_reservoir_water_pressurises_open_connected_crack(self):
        specimen = Specimen(2, 1)
        water = reservoir(specimen)
        deltas = [0., 2 * DELTA0, .5 * LAW.critical_opening_m, 1.5 * LAW.critical_opening_m]
        path, _ = trace(specimen, deltas, water=water)
        dry_before = [row for row in path[1:] if not row[2].report["initiated_pairs"]]
        self.assertTrue(all(not row[2].report["wet_face_ids"] for row in dry_before))
        for delta in deltas[1:]:
            row = next(row for row in path[1:] if row[0] == delta)
            self.assertAlmostEqual(row[1], wet_analytic_force(delta), delta=1e-6 * FT * AREA)
            self.assertEqual(len(row[2].report["wet_face_ids"]), len(specimen.faces.minus))
            np.testing.assert_allclose(row[2].report["water_resultant_n"], 0., atol=1e-6)
        self.assertAlmostEqual(path[-1][1], -PRESSURE * AREA, delta=1e-6 * PRESSURE)
        self.assertAlmostEqual(path[-1][2].report["wet_area_m2"], AREA, delta=1e-12)

    def test_water_requires_reservoir_path_and_open_aperture(self):
        specimen = Specimen(2, 1)
        unrelated = reservoir(specimen, inlet=np.array([[0, 1]]))  # a bottom edge, not on the crack
        path, _ = trace(specimen, [0., 2 * DELTA0], water=unrelated)
        self.assertEqual(path[-1][2].report["wet_face_ids"], [])
        self.assertIn("no reservoir inlet", path[-1][2].report["water_access_note"])
        self.assertAlmostEqual(path[-1][1], analytic_force(2 * DELTA0), delta=1e-6 * FT * AREA)
        water = reservoir(specimen)
        _, state = trace(specimen, [0., 2 * DELTA0, .5 * LAW.critical_opening_m], water=water)
        stiffness, load, fixed, prescribed = specimen.problem()(-DELTA0)
        closed = solve_wet_cohesive_equilibrium(stiffness, load, fixed, prescribed, specimen.interface,
                                                specimen.faces, specimen.mapping, LAW, state,
                                                options=options(), water=water)
        self.assertEqual(closed.report["wet_face_ids"], [])
        self.assertEqual(set(closed.pair_status), {"closed"})
        self.assertAlmostEqual(specimen.force(closed), -E * AREA * DELTA0 / (2 * HEIGHT), delta=1e-6 * FT)

    def test_progressive_wet_crack_from_inlet_keeps_water_on_connected_initiated_faces(self):
        specimen = Specimen(3, 1)
        water = reservoir(specimen)
        path, state = trace(specimen, [0., 1.3 * DELTA0, 2. * DELTA0], water=water, rotation=-1.)
        lookup = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(specimen.interface.minus_nodes,
                                                                       specimen.interface.plus_nodes))}
        saw_partial_wet = False
        for _, _, result in path[1:]:
            ratio = result.report["maximum_bonded_tensile_ratio"]
            if ratio is not None:
                self.assertLessEqual(ratio, 1 + 1e-7)
            initiated = result.trial_state.initiated
            for face in result.report["wet_face_ids"]:
                pairs = [lookup[int(a), int(b)] for a, b in zip(specimen.faces.minus[face], specimen.faces.plus[face])]
                self.assertTrue(initiated[pairs].all())
            if 0 < len(result.report["wet_face_ids"]) < len(specimen.faces.minus):
                saw_partial_wet = True
        self.assertTrue(saw_partial_wet)
        self.assertTrue(state.initiated[np.argmin(specimen.coordinates[specimen.interface.minus_nodes, 0])])

    def test_invalid_interfaces_and_options_are_rejected(self):
        coordinates = np.zeros((4, 3))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            CohesiveInterface(np.array([0, 0]), np.array([1, 2]), np.tile([0., 0., 1.], (2, 1)),
                              np.ones(2), coordinates)
        with self.assertRaisesRegex(ValueError, "tangential_response"):
            CohesiveOptions(1e-12, 1e-3, 1e-9, 1e-7, "rough")
        with self.assertRaisesRegex(ValueError, "nonnegative openings only on initiated"):
            InterfaceState(np.array([False]), np.array([1e-4]))

    def test_junctions_are_opt_in_and_pairs_stay_unique(self):
        coordinates = np.zeros((4, 3))
        normals = np.tile([0., 0., 1.], (3, 1))
        junction = CohesiveInterface(np.array([0, 0, 1]), np.array([1, 2, 2]), normals, np.ones(3),
                                     coordinates, allow_junctions=True)
        self.assertTrue(junction.allow_junctions)
        with self.assertRaisesRegex(ValueError, "occur only once"):
            CohesiveInterface(np.array([0, 1]), np.array([1, 0]), normals[:2], np.ones(2), coordinates,
                              allow_junctions=True)
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            CohesiveInterface(np.array([0]), np.array([1]), normals[:1], np.ones(1), coordinates,
                              allow_junctions=1)


if __name__ == "__main__":
    unittest.main()
