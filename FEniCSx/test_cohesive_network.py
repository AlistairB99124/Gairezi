"""Whole-wall cohesive network verification on small P1 tetrahedral blocks.

Candidate facets are supplied as planes (no seeds); initiation, P1 patch
expansion, junctions, irreversibility, dissipation and reservoir water are
checked against closed-form uniaxial results where they exist.
"""
import unittest

import numpy as np
from scipy import sparse

from FEniCSx.cohesive_contact import CohesiveOptions
from FEniCSx.cohesive_mode_i import LinearModeICohesiveLaw
from FEniCSx.cohesive_network import CohesiveNetwork, NetworkOptions, NetworkWater
from FEniCSx.test_cohesive_contact import E, FT, GF, KUHN, elasticity, tetra_stiffness

LAW = LinearModeICohesiveLaw(FT, GF)


class Block:
    def __init__(self, divisions=(2, 1, 2), size=(1., 1., 1.), order=None):
        nx, ny, nz = divisions
        self.size = size
        axes = [np.linspace(0, length, n + 1) for length, n in zip(size, divisions)]
        ids = {}
        points = []
        for k, z in enumerate(axes[2]):
            for j, y in enumerate(axes[1]):
                for i, x in enumerate(axes[0]):
                    ids[i, j, k] = len(points)
                    points.append((x, y, z))
        cells = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    corners = [ids[i + a, j + b, k + c] for c in (0, 1) for b in (0, 1) for a in (0, 1)]
                    cells.extend([[corners[v] for v in tet] for tet in KUHN])
        self.coordinates = np.asarray(points, float)
        self.cells = np.asarray(cells)
        if order is not None:
            self.cells = self.cells[order(len(self.cells))]
        self.regions = np.ones(len(self.cells), dtype=np.int32)

    def plane(self, axis, value):
        faces = set()
        for cell in self.cells:
            for pattern in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                face = tuple(sorted(int(cell[i]) for i in pattern))
                if np.allclose(self.coordinates[list(face), axis], value):
                    faces.add(face)
        return np.asarray(sorted(faces))


class Loading:
    """Displacement control: z on bottom/top, optional x on lateral faces."""

    def __init__(self, block, delta_z, rotation=0., delta_x=None):
        self.block, self.delta_z, self.rotation, self.delta_x = block, delta_z, rotation, delta_x
        self.cache = {}

    def stiffness(self, split):
        key = id(split)
        if key not in self.cache:
            size = 3 * len(split.coordinates_m)
            rows, columns, values = [], [], []
            for cell in split.cells:
                local = tetra_stiffness(split.coordinates_m[cell])
                dofs = (3 * cell[:, None] + np.arange(3)).ravel()
                rows.append(np.repeat(dofs, 12))
                columns.append(np.tile(dofs, 12))
                values.append(local.ravel())
            matrix = sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                                       shape=(size, size))
            self.cache[key] = (.5 * (matrix + matrix.T)).tocsr()
        return self.cache[key]

    def sets(self, split):
        xyz = split.coordinates_m
        width, _, height = self.block.size
        bottom = np.flatnonzero(np.isclose(xyz[:, 2], 0))
        top = np.flatnonzero(np.isclose(xyz[:, 2], height))
        return xyz, width, height, bottom, top

    def __call__(self, split, factor):
        xyz, width, height, bottom, top = self.sets(split)

        def pin(nodes, at):
            return nodes[np.argmin(np.linalg.norm(xyz[nodes, :2] - at, axis=1))]

        fixed = [3 * n + 2 for n in bottom] + [3 * n + 2 for n in top]
        values = [0.] * len(bottom) + list(factor * (self.delta_z + self.rotation * (xyz[top, 0] - width / 2)))
        if self.delta_x is None:
            fixed += [3 * pin(bottom, (0, 0)), 3 * pin(bottom, (0, 0)) + 1, 3 * pin(bottom, (width, 0)) + 1,
                      3 * pin(top, (0, 0)), 3 * pin(top, (0, 0)) + 1, 3 * pin(top, (width, 0)) + 1]
            values += [0.] * 6
        else:
            left = np.flatnonzero(np.isclose(xyz[:, 0], 0))
            right = np.flatnonzero(np.isclose(xyz[:, 0], width))
            fixed += [3 * n for n in left] + [3 * n for n in right] + [3 * pin(bottom, (0, 0)) + 1]
            values += [0.] * len(left) + [factor * self.delta_x] * len(right) + [0.]
        return self.stiffness(split), np.zeros(3 * len(xyz)), np.asarray(fixed), np.asarray(values)

    def top_force(self, split, displacement):
        _, _, _, _, top = self.sets(split)
        return float((self.stiffness(split) @ displacement)[3 * top + 2].sum())


def options(**overrides):
    cohesive = CohesiveOptions(gap_tolerance_m=1e-12, force_tolerance_n=1e-3, relative_residual_tolerance=1e-9,
                               initiation_relative_tolerance=1e-6, tangential_response="tied")
    return NetworkOptions(cohesive, **overrides)


def network(block, candidates, loading, **kwargs):
    return CohesiveNetwork(block.coordinates, block.cells, block.regions, candidates,
                           elasticity=elasticity(), law=LAW, options=options(), problem_at=loading, **kwargs)


DELTA0 = FT / E  # uniaxial strength displacement of a unit-height block


def uniaxial_force(delta, area=1., height=1.):
    compliance = height / E
    wc = LAW.critical_opening_m
    if delta <= compliance * FT:
        return area * delta / compliance
    opening = (delta - compliance * FT) / (1 - compliance * FT / wc)
    return area * FT * max(1 - opening / wc, 0.)


class CohesiveNetworkTests(unittest.TestCase):
    def test_below_strength_no_initiation_and_exact_elastic_stress(self):
        block = Block()
        loading = Loading(block, .5 * DELTA0)
        result = network(block, block.plane(2, .5), loading).run()
        self.assertEqual(result.status, "no_damage_initiated")
        self.assertFalse(result.initiated.any())
        np.testing.assert_allclose(result.face_normal_stress_pa, .5 * FT, rtol=1e-9)
        self.assertAlmostEqual(loading.top_force(result.split, result.displacement_m), .5 * FT, delta=1e-3 * FT)

    def test_uniaxial_softening_matches_closed_form_and_dissipates_gf(self):
        block = Block()
        for delta in (2 * DELTA0, LAW.critical_opening_m * .5, 1.2 * LAW.critical_opening_m):
            loading = Loading(block, delta)
            result = network(block, block.plane(2, .5), loading).run()
            self.assertEqual(result.status, "accepted_equilibrium", result.report["message"])
            self.assertTrue(result.initiated.all())
            self.assertEqual(result.report["patch_faces_added"], 0)
            force = loading.top_force(result.split, result.displacement_m)
            self.assertAlmostEqual(force, uniaxial_force(delta), delta=1e-6 * FT)
        self.assertTrue(np.all(result.face_status == "separated"))
        self.assertAlmostEqual(result.report["cohesive_dissipated_energy_j"], GF * 1., delta=1e-9 * GF)

    def test_bending_initiates_without_seeds_and_expands_minimal_patches(self):
        block = Block((4, 1, 2), (1., .25, 1.))
        theta = 2 * FT / (E * 1.)
        loading = Loading(block, .2 * DELTA0, rotation=3 * theta)
        result = network(block, block.plane(2, .5), loading).run()
        self.assertEqual(result.status, "accepted_equilibrium", result.report["message"])
        self.assertGreater(result.report["patch_faces_added"], 0)
        centroids = result.split.coordinates_m[result.split.faces.minus].mean(axis=1)
        self.assertGreater(centroids[result.initiated, 0].mean(), .5)
        self.assertFalse(result.initiated[centroids[:, 0] < .2].any())
        self.assertTrue(np.isin(result.face_status[result.initiated], ("closed", "softening", "separated")).all())
        loads = [step["load_factor"] for step in result.steps]
        self.assertEqual(loads, sorted(loads))
        self.assertLessEqual(result.report["maximum_bonded_tensile_ratio"], 1 + 1e-6)

    def test_progress_reports_every_solve_with_stage_and_state(self):
        block = Block()
        lines = []
        result = network(block, block.plane(2, .5), Loading(block, 2 * DELTA0), progress=lines.append).run()
        self.assertEqual(result.status, "accepted_equilibrium", result.report["message"])
        self.assertEqual(len(lines), result.report["solves"])
        self.assertTrue(all(line.startswith(f"cohesive solve {i}/5000 (")
                            for i, line in enumerate(lines, 1)))
        self.assertIn("(target load)", lines[0])
        self.assertTrue(any("strength-crossing bisection" in line for line in lines))
        self.assertTrue(any("initiation round 1," in line for line in lines))
        self.assertRegex(lines[-1], r"initiated faces [1-9]\d*, opened \d+, wet 0; \d+\.\d s$")

    def test_candidate_and_cell_order_do_not_change_the_result(self):
        block = Block((4, 1, 2), (1., .25, 1.))
        theta = 2 * FT / E
        reference = network(block, block.plane(2, .5), Loading(block, .2 * DELTA0, rotation=3 * theta)).run()
        rng = np.random.default_rng(3)
        shuffled = Block((4, 1, 2), (1., .25, 1.), order=rng.permutation)
        candidates = shuffled.plane(2, .5)[rng.permutation(len(reference.initiated))]
        result = network(shuffled, candidates, Loading(shuffled, .2 * DELTA0, rotation=3 * theta)).run()

        def keys(outcome):
            return {tuple(face) for face, on in zip(outcome.split.source_faces, outcome.initiated) if on}

        self.assertEqual(result.status, reference.status)
        self.assertEqual(keys(result), keys(reference))
        self.assertAlmostEqual(result.load_factor, reference.load_factor, places=12)

    def test_unresolved_initiation_is_reported_not_accepted(self):
        block = Block((4, 1, 2), (1., .25, 1.))
        loading = Loading(block, .2 * DELTA0, rotation=3 * 2 * FT / E)
        result = network(block, block.plane(2, .5), loading, patch_resolver=lambda broken, seeds: []).run()
        self.assertEqual(result.status, "unresolved_initiation")
        self.assertFalse(result.initiated.any())
        self.assertLess(result.load_factor, 1.)

    def test_unloading_keeps_initiation_and_opening_history(self):
        block = Block()
        loading = Loading(block, .5 * LAW.critical_opening_m)
        model = network(block, block.plane(2, .5), loading)
        loaded = model.run()
        self.assertEqual(loaded.status, "accepted_equilibrium")
        unloaded = model.run(1., .4, initiated=loaded.initiated, history=loaded.maximum_opening_history_m)
        self.assertEqual(unloaded.status, "accepted_equilibrium")
        np.testing.assert_array_equal(unloaded.initiated, loaded.initiated)
        np.testing.assert_allclose(unloaded.maximum_opening_history_m, loaded.maximum_opening_history_m)
        self.assertLess(unloaded.face_opening_m.max(), loaded.face_opening_m.max())
        # Secant unloading of linear softening: no extra dissipation.
        self.assertAlmostEqual(unloaded.report["cohesive_dissipated_energy_j"],
                               loaded.report["cohesive_dissipated_energy_j"], delta=1e-9)

    def test_crossing_planes_form_a_junction_and_separate(self):
        block = Block((2, 1, 2), (1., .5, 1.))
        candidates = np.concatenate((block.plane(2, .5), block.plane(0, .5)))
        loading = Loading(block, 1.2 * LAW.critical_opening_m, delta_x=1.2 * LAW.critical_opening_m)
        result = network(block, candidates, loading).run()
        self.assertEqual(result.status, "accepted_equilibrium", result.report["message"])
        self.assertTrue(result.initiated.all())
        self.assertTrue(np.all(result.face_status == "separated"))
        self.assertGreater(max(step["redundant_junction_constraint_rows"] for step in result.steps), 0)
        self.assertAlmostEqual(result.report["cohesive_dissipated_energy_j"], GF * (1. * .5 + 1. * .5),
                               delta=1e-6 * GF)

    def test_reservoir_water_loads_a_connected_separated_crack(self):
        block = Block()
        height, unit_weight = 10., 9810.
        plane = block.plane(2, .5)
        xyz = block.coordinates
        inlet = np.asarray([[a, b] for a in range(len(xyz)) for b in range(a + 1, len(xyz))
                            if np.allclose(xyz[[a, b], 0], 0) and np.allclose(xyz[[a, b], 2], .5)
                            and any({a, b} <= set(face) for face in plane.tolist())])
        loading = Loading(block, 1.2 * LAW.critical_opening_m)
        water = NetworkWater(height, unit_weight, inlet, 1e-9)
        result = network(block, plane, loading, water=water).run()
        self.assertEqual(result.status, "accepted_equilibrium", result.report["message"])
        self.assertTrue(result.wet_faces.all())
        force = loading.top_force(result.split, result.displacement_m)
        self.assertAlmostEqual(abs(force), unit_weight * (height - .5) * 1., delta=1e-6 * unit_weight * height)


if __name__ == "__main__":
    unittest.main()
