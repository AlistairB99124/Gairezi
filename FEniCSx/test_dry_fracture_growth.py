"""Dry growth energy accounting, irreversibility and native equilibrium evidence."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dry_fracture_growth import DryGrowthEquilibrium, GrowthOptions, grow_dry_fracture


class DryGrowthTests(unittest.TestCase):
    def fixture(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [1., 1., 0.], [2., 1., 1.]])
        return points, np.array([[0, 1, 2], [1, 2, 3], [1, 3, 4]]), np.array([[0, 1]])

    def state(self, potential, digest="a" * 64):
        return DryGrowthEquilibrium(potential, digest,
                                    {"equilibrium_verified": True, "dry_conservative_problem": True})

    def run_growth(self, directory, callback, *, previous=(), limit=10):
        return grow_dry_fracture(*self.fixture(), equilibrate=callback,
                                 options=GrowthOptions(10., 1e-8, limit),
                                 output=Path(directory) / "growth", previously_broken=previous)

    def test_energy_cost_irreversibility_and_nonplanar_edge_front(self):
        calls = []

        def solve(ids, directory):
            calls.append(ids)
            # First two faces release 8J per face; third costs > its release.
            return self.state(-8. * len(set(ids) & {0, 1}) - (1. if 2 in ids else 0.))

        with tempfile.TemporaryDirectory() as directory:
            result = self.run_growth(directory, solve)
            self.assertEqual(result["accepted_surface_ids"], [0, 1])
            self.assertEqual([row["candidate"] for row in result["accepted_extensions"]], [0, 1])
            self.assertAlmostEqual(result["new_fracture_surface_energy_j"], 10.)
            self.assertEqual(calls, [(), (0,), (0, 1), (0, 1, 2)])
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_growth(directory, lambda ids, path: self.state(100. * len(ids)), previous=(1,))
            self.assertEqual(result["accepted_surface_ids"], [1])
            self.assertFalse(result["accepted_extensions"])
            self.assertEqual(result["new_fracture_surface_energy_j"], 0.)
            self.assertEqual(result["total_fracture_surface_energy_j"], 5.)

    def test_phase_lines_mark_initial_equilibrium_then_growth_once_with_utc_times(self):
        import contextlib
        import io
        from datetime import datetime
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(stream):
            self.run_growth(directory, lambda ids, path: self.state(-8. * len(set(ids) & {0, 1})))
        phases = [json.loads(line[len("Phase: "):]) for line in stream.getvalue().splitlines()
                  if line.startswith("Phase: ")]
        self.assertEqual([row["phase"] for row in phases], ["initial_equilibrium", "growth"])
        self.assertTrue(all(datetime.fromisoformat(row["time"]).tzinfo is not None for row in phases))

    def test_unknown_phase_is_rejected(self):
        from dry_fracture_growth import report_phase
        with self.assertRaises(ValueError):
            report_phase("painting")

    def test_tie_does_not_create_free_fracture_area(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_growth(directory, lambda ids, path: self.state(-5. * len(ids)))
            self.assertEqual(result["accepted_surface_ids"], [])
            self.assertEqual(result["trials"][0]["total_energy_change_j"], 0.)

    def test_locked_candidates_cannot_be_reported_as_arrest_and_are_not_solved(self):
        from dry_fracture_growth import _grow_conservative_fracture
        calls = []

        def equilibrium(ids, directory):
            calls.append(ids)
            return self.state(0.)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "growth"
            with self.assertRaisesRegex(RuntimeError, "NOT evidence of crack arrest"):
                _grow_conservative_fracture(
                    *self.fixture(), equilibrate=equilibrium,
                    options=GrowthOptions(10., 1e-8, 2), output=output,
                    previously_broken=(), equilibrium_type=DryGrowthEquilibrium, mode="dry",
                    trial_admissible=lambda current, proposed: False)
            report = json.loads((output / "growth_report.json").read_text())
            self.assertEqual(calls, [])
            self.assertEqual(report["failure_kind"], "no_kinematically_admissible_extension")
            self.assertEqual(report["status"], "failed_experimental_dry_growth")
            self.assertEqual(report["trials"], [])
            self.assertEqual(len(report["excluded_kinematically_locked_patches"]), 1)
            self.assertFalse(report["initial_equilibrium_started"])
            self.assertEqual(report["accepted_equilibria"], [])
            self.assertNotIn("initial_potential_j", report)
            self.assertFalse((output / "initial").exists())

    def test_admissible_connected_patch_is_still_tested_when_single_faces_are_locked(self):
        from dry_fracture_growth import _grow_conservative_fracture
        calls = []

        def equilibrium(ids, directory):
            calls.append(ids)
            return self.state(0.)

        with tempfile.TemporaryDirectory() as directory:
            result = _grow_conservative_fracture(
                *self.fixture(), equilibrate=equilibrium,
                options=GrowthOptions(10., 1e-8, 2, maximum_patch_faces=2),
                output=Path(directory) / "growth", previously_broken=(),
                equilibrium_type=DryGrowthEquilibrium, mode="dry",
                trial_admissible=lambda current, proposed: len(proposed) > 1)
            self.assertEqual(calls, [(), (0, 1)])
            self.assertEqual(result["status"], "arrested_in_tested_connected_patch_candidate_set")
            self.assertEqual(len(result["trials"]), 1)
            self.assertEqual(len(result["excluded_kinematically_locked_patches"]), 1)
            self.assertTrue(result["initial_equilibrium_started"])

    def test_native_locked_node_star_fails_before_dry_or_wet_equilibrium(self):
        from dolfinx import fem, mesh
        from mpi4py import MPI
        from dry_fracture_growth import solve_dry_contact_growth
        from fracture_hydrostatic import HydrostaticBoundary
        from fracture_topology import interior_wall_candidates, split_interior_wall_faces
        from hydrostatic_fracture_growth import solve_hydrostatic_contact_growth
        from paired_contact import ContactOptions
        from tensile_loss_event import EventSolverOptions

        points = np.array([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.),
                           (0., 0., 1.), (.25, .25, .25)])
        cells = np.array([[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]])
        markers = np.array([[0, 4]])
        candidates = interior_wall_candidates(points, cells, np.ones(4, dtype=int), markers)
        split = split_interior_wall_faces(points, cells, np.ones(4, dtype=int), candidates)
        reference = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        domain = mesh.create_mesh(MPI.COMM_WORLD, split.cells, reference.ufl_domain(), split.coordinates_m)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        args = dict(
            young_pa=1e8, poisson=.2, selected_cells=np.zeros(4, dtype=bool),
            boundary_conditions=(), external_load=None, event_options=EventSolverOptions(),
            contact_options=ContactOptions(1e-12, 1e-7, 1e-8, 100),
            growth_options=GrowthOptions(2700., 1e-8, 2))
        for mode, driver in (("dry", solve_dry_contact_growth),
                             ("common_head_hydrostatic", solve_hydrostatic_contact_growth)):
            hydraulic = {} if mode == "dry" else dict(
                boundaries=(HydrostaticBoundary("reservoir", 4., markers),),
                water_density_kg_m3=1000., gravity_m_s2=9.81,
                closure_tolerance_m=1e-12, flow_relative_tolerance=1e-10)
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "growth"
                with patch("tensile_loss_event.solve_contact_event") as solve:
                    with self.assertRaisesRegex(RuntimeError, "NOT evidence of crack arrest"):
                        driver(space, split.faces, split.parent_node_ids, markers,
                               **args, **hydraulic, output=output)
                    solve.assert_not_called()
                report = json.loads((output / "growth_report.json").read_text())
                self.assertEqual(report["failure_kind"], "no_kinematically_admissible_extension")
                self.assertFalse(report["initial_equilibrium_started"])
                self.assertEqual(report["accepted_equilibria"], [])
                self.assertEqual(len(report["excluded_kinematically_locked_patches"]), 3)
                self.assertFalse((output / "initial").exists())
                with np.load(output / "prepared_candidate_inputs.npz") as inputs:
                    np.testing.assert_array_equal(inputs["minus"], split.faces.minus)
                    np.testing.assert_array_equal(inputs["parent_node_ids"], split.parent_node_ids)

    def test_connected_patch_crosses_single_face_energy_barrier_without_free_area(self):
        for size in (1, 2):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                calls = []

                def solve(ids, path):
                    calls.append(ids)
                    return self.state(-12. if {0, 1}.issubset(ids) else -2. * len(ids))

                result = grow_dry_fracture(
                    *self.fixture(), equilibrate=solve,
                    options=GrowthOptions(10., 1e-8, 10, maximum_patch_faces=size),
                    output=Path(directory) / "growth")
                self.assertEqual(result["accepted_surface_ids"], [] if size == 1 else [0, 1])
                self.assertEqual(result["maximum_patch_faces"], size)
                self.assertEqual(len(calls), len(set(calls)))
                if size == 2:
                    accepted = result["accepted_extensions"]
                    self.assertEqual(len(accepted), 1)
                    self.assertEqual(accepted[0]["candidate_surface_ids"], [0, 1])
                    self.assertIsNone(accepted[0]["candidate"])
                    self.assertEqual(accepted[0]["surface_energy_cost_j"], 10.)
                    self.assertEqual(accepted[0]["total_energy_change_j"], -2.)
                    self.assertEqual(result["new_fracture_surface_energy_j"], 10.)
                    self.assertEqual(result["status"], "arrested_in_tested_connected_patch_candidate_set")

    def test_resolved_patch_charges_every_face_and_retains_irreversible_cut(self):
        from dry_fracture_growth import _grow_conservative_fracture
        calls = []

        def equilibrium(ids, directory):
            calls.append(ids)
            return self.state(-12. if {0, 1}.issubset(ids) else 0.)

        with tempfile.TemporaryDirectory() as directory:
            result = _grow_conservative_fracture(
                *self.fixture(), equilibrate=equilibrium,
                options=GrowthOptions(10., 1e-8, 2, resolve_locked_patches=True),
                output=Path(directory) / "growth", previously_broken=(),
                equilibrium_type=DryGrowthEquilibrium, mode="dry",
                trial_admissible=lambda current, proposed: len(proposed) > 1,
                resolve_patch=lambda current, seed: [(0, 1), (0, 1)])
            self.assertEqual(calls, [(), (0, 1), (0, 1, 2)])
            self.assertEqual(result["accepted_surface_ids"], [0, 1])
            self.assertEqual(result["accepted_extensions"][0]["surface_energy_cost_j"], 10.)
            self.assertEqual(result["accepted_extensions"][0]["total_energy_change_j"], -2.)
            self.assertEqual(len(result["generated_node_star_patches"]), 1)
            self.assertEqual(result["status"], "arrested_in_tested_node_star_patch_candidate_set")

    def test_node_star_option_requires_boolean_and_native_topology(self):
        for value in (1, None, "true"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "explicit boolean"):
                GrowthOptions(10., 1e-8, 2, resolve_locked_patches=value)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "native node-star topology"):
                grow_dry_fracture(
                    *self.fixture(), equilibrate=lambda ids, path: self.state(0.),
                    options=GrowthOptions(10., 1e-8, 2, resolve_locked_patches=True),
                    output=Path(directory) / "growth")

    def test_unaccepted_pressure_bound_can_only_reject_never_accept_growth(self):
        from dry_fracture_growth import UnacceptedTrialEnergyBound
        for bound, fail in ((-1., False), (-100., True)):
            with self.subTest(bound=bound), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "growth"

                def equilibrium(ids, path):
                    if not ids:
                        return self.state(0.)
                    return UnacceptedTrialEnergyBound(
                        bound, "a" * 64, {"mechanical_contact_equilibrium_verified": True,
                                         "hydraulic_equilibrium_accepted": False})

                if fail:
                    with self.assertRaisesRegex(RuntimeError, "coupled wet/contact state is unresolved"):
                        self.run_growth(directory, equilibrium)
                else:
                    result = self.run_growth(directory, equilibrium)
                    self.assertTrue(result["status"].startswith("arrested"))
                    self.assertEqual(result["accepted_extensions"], [])
                result = json.loads((root / "growth_report.json").read_text())
                self.assertEqual(result["accepted_surface_ids"], [])
                self.assertEqual(len(result["accepted_equilibria"]), 1)
                trial = result["trials"][0]
                self.assertEqual(trial["assessment"], "unresolved_hydraulic_trial_energy_bound" if fail
                                 else "rejected_by_optimistic_pressure_energy_bound")
                self.assertNotIn("conservative_energy_release_j", trial)
                self.assertEqual(trial["conservative_energy_release_upper_bound_j"], -bound)

    def test_patch_inventory_matches_exhaustive_connected_subsets_and_keeps_source_identity(self):
        from itertools import combinations
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.],
                           [2., 1., 0.], [2., 0., 0.],
                           [0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        triangles = np.array([[0, 1, 2], [1, 2, 3], [1, 3, 5], [3, 4, 5], [6, 7, 8]])
        edges = [{tuple(sorted(edge)) for edge in combinations(triangle, 2)} for triangle in triangles]
        marker = (0, 1)
        prior = {1}
        front = edges[1] | {marker}
        expected = set()
        for size in range(1, 4):
            for patch in combinations(sorted(set(range(len(triangles))) - prior), size):
                reachable = {patch[0]}
                while True:
                    extended = reachable | {
                        face for face in patch if any(edges[face] & edges[other] for other in reachable)}
                    if reachable == extended:
                        break
                    reachable = extended
                if len(reachable) == len(patch) and any(edges[face] & front for face in patch):
                    expected.add(tuple(sorted(prior | set(patch))))
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            result = grow_dry_fracture(
                points, triangles, np.array([marker]), previously_broken=(1,),
                equilibrate=lambda ids, path: calls.append(ids) or self.state(100. * len(ids)),
                options=GrowthOptions(10., 1e-8, 10, maximum_patch_faces=3),
                output=Path(directory) / "growth")
        self.assertEqual(set(calls[1:]), expected)
        self.assertEqual(len(calls[1:]), len(expected))
        self.assertTrue(all(4 not in ids for ids in calls))
        self.assertNotIn((0, 1, 3), calls)  # two disconnected new faces are not one patch
        self.assertEqual(result["accepted_surface_ids"], [1])
        self.assertEqual(result["new_fracture_surface_energy_j"], 0.)

    def test_failed_or_changed_multiface_trial_cannot_claim_arrest(self):
        for failure in ("failed", "changed"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                def solve(ids, path):
                    if len(ids) == 2:
                        if failure == "failed":
                            raise RuntimeError("patch equilibrium failed")
                        return self.state(-100., "b" * 64)
                    return self.state(0.)

                root = Path(directory) / "growth"
                with self.assertRaisesRegex(RuntimeError, "no arrested-growth result"):
                    grow_dry_fracture(
                        *self.fixture(), equilibrate=solve,
                        options=GrowthOptions(10., 1e-8, 10, maximum_patch_faces=2), output=root)
                report = json.loads((root / "growth_report.json").read_text())
                self.assertEqual(report["status"], "failed_experimental_dry_growth")
                self.assertEqual(report["accepted_surface_ids"], [])

    def test_patch_bound_rejects_nonpositive_noninteger_and_boolean_values(self):
        for value in (0, -1, True, 1.5, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "maximum_patch_faces"):
                GrowthOptions(10., 1e-8, 10, maximum_patch_faces=value)

    def test_patch_extension_limit_preserves_accepted_area_but_does_not_claim_arrest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "growth"
            with self.assertRaisesRegex(RuntimeError, "not arrested"):
                grow_dry_fracture(
                    *self.fixture(), equilibrate=lambda ids, path: self.state(-100. * len(ids)),
                    options=GrowthOptions(10., 1e-8, 1, maximum_patch_faces=2), output=root)
            report = json.loads((root / "growth_report.json").read_text())
            self.assertEqual(report["status"], "failed_experimental_dry_growth")
            self.assertEqual(report["accepted_surface_ids"], [0, 1])
            self.assertEqual(len(report["accepted_extensions"]), 1)
            self.assertEqual(report["accepted_extensions"][0]["surface_energy_cost_j"], 10.)

    def test_changed_problem_or_failed_trial_is_not_arrest(self):
        for failure in ("changed", "failed"):
            def solve(ids, path):
                if ids and failure == "failed":
                    raise RuntimeError("native equilibrium did not converge")
                return self.state(-100. * len(ids), "b" * 64 if ids else "a" * 64)
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, "no arrested-growth result"):
                    self.run_growth(directory, solve)
                report = json.loads((Path(directory) / "growth" / "growth_report.json").read_text())
                self.assertEqual(report["status"], "failed_experimental_dry_growth")

    def test_limit_with_favourable_remaining_extension_cannot_claim_arrest(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "not arrested"):
                self.run_growth(directory, lambda ids, path: self.state(-100. * len(ids)), limit=1)
            report = json.loads((Path(directory) / "growth" / "growth_report.json").read_text())
            self.assertEqual(report["accepted_surface_ids"], [0])
            self.assertEqual(report["status"], "failed_experimental_dry_growth")

    def test_point_contact_does_not_advance_front_or_weld_coincident_sheets(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [2., 0., 0.], [2., 1., 0.], [0., 0., 0.],
                           [1., 0., 0.], [0., 1., 0.]])
        triangles = np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7]])
        with tempfile.TemporaryDirectory() as directory:
            result = grow_dry_fracture(
                points, triangles, np.array([[0, 1]]),
                equilibrate=lambda ids, path: self.state(-100. * len(ids)),
                options=GrowthOptions(10., 1e-8, 10), output=Path(directory) / "growth")
            self.assertEqual(result["accepted_surface_ids"], [0])

    def test_growth_opening_peak_retains_accepted_reclosure_and_excludes_rejected_trials(self):
        from dry_fracture_growth import _grow_conservative_fracture
        from horizontal_fracture_contract import EquilibriumOpeningObservation, assess_equilibrium_opening

        def state(ids, path):
            opening = 99. if 2 in ids else .001 if 1 in ids else .003 if ids else 0.
            observation = EquilibriumOpeningObservation("event", opening, (0., 0., 0.), bool(ids))
            assessment = assess_equilibrium_opening(.002, (observation,), completed=True)
            return DryGrowthEquilibrium(
                -8. * len(set(ids) & {0, 1}) - (1. if 2 in ids else 0.), "a" * 64,
                {"equilibrium_verified": True, "dry_conservative_problem": True,
                 "opening_assessment": assessment, "final_opening_location_m": [0., 0., 0.]})

        for limit in (1, 2):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "growth"
                args = dict(
                    equilibrate=state, options=GrowthOptions(10., 1e-8, limit), output=root,
                    previously_broken=(), equilibrium_type=DryGrowthEquilibrium, mode="dry", opening_limit_m=.002)
                if limit == 1:
                    with self.assertRaisesRegex(RuntimeError, "not arrested"):
                        _grow_conservative_fracture(*self.fixture(), **args)
                else:
                    _grow_conservative_fracture(*self.fixture(), **args)
                report = json.loads((root / "growth_report.json").read_text())
                assessment = report["opening_assessment"]
                self.assertEqual(assessment["peak_opening_m"], .003)
                self.assertEqual(assessment["final_opening_m"], .003 if limit == 1 else .001)
                self.assertEqual(assessment["completed"], limit == 2)
                self.assertTrue(assessment["observed_limit_reached"])
                self.assertEqual(len(report["accepted_equilibria"]), 2 if limit == 1 else 3)
                self.assertNotIn(99., [item["maximum_opening_m"] for item in report["opening_history"]])

    def test_native_displacement_controlled_fracture_threshold(self):
        from dolfinx import fem, mesh
        from dolfinx.fem.petsc import assemble_matrix
        from mpi4py import MPI
        from scipy import sparse
        import ufl
        from fracture_faces import bulk_face_mapping
        from fracture_topology import split_interior_wall_faces
        from paired_contact import ContactOptions, solve_contact

        points = np.array([(0., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                           (-1., 0., 0.), (1., 0., 0.)])
        split = split_interior_wall_faces(points, np.array([[0, 1, 2, 3], [0, 1, 2, 4]]),
                                           [1, 1], [[0, 1, 2]])
        import basix.ufl
        domain = mesh.create_mesh(
            MPI.COMM_WORLD, split.cells,
            ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
            split.coordinates_m)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        xyz, _, mapping = bulk_face_mapping(space, split.faces)
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        operator = assemble_matrix(fem.form(1e8 * ufl.inner(
            ufl.sym(ufl.grad(trial)), ufl.sym(ufl.grad(test))) * ufl.dx))
        operator.assemble()
        try:
            offsets, columns, coefficients = operator.getValuesCSR()
            stiffness = sparse.csr_matrix((coefficients.copy(), columns.copy(), offsets.copy()), shape=operator.getSize())
        finally:
            operator.destroy()
        fixed = []
        for source, dof in enumerate(mapping):
            fixed.extend((3 * int(dof) + 1, 3 * int(dof) + 2))
            if abs(xyz[source, 0]) == 1.:
                fixed.append(3 * int(dof))
        fixed = np.array(sorted(fixed), dtype=np.int32)
        source_at_dof = np.argsort(mapping)
        for shift in (.001, .01):
            def equilibrium(ids, directory):
                prescribed = np.array([
                    shift * xyz[source_at_dof[dof // 3], 0] if dof % 3 == 0 else 0. for dof in fixed])
                state = solve_contact(
                    stiffness, np.zeros(stiffness.shape[0]), fixed, prescribed, split.faces, mapping,
                    options=ContactOptions(1e-12, 1e-7, 1e-9, 100), bonded=not bool(ids))
                energy = state.report["elastic_energy_j"]
                expected = 0. if ids else .5 * 1e8 * shift**2 / 3.
                self.assertAlmostEqual(energy, expected, places=7)
                return self.state(energy)

            with self.subTest(shift=shift), tempfile.TemporaryDirectory() as directory:
                result = grow_dry_fracture(
                    points, np.array([[0, 1, 2]]), np.array([[0, 1]]),
                    equilibrate=equilibrium, options=GrowthOptions(100., 1e-8, 2),
                    output=Path(directory) / "growth")
                self.assertEqual(result["accepted_surface_ids"], [] if shift == .001 else [0])
                self.assertAlmostEqual(result["trials"][0]["surface_energy_cost_j"], 50.)

    def test_native_selected_material_growth_automatically_releases_only_accepted_faces(self):
        from dolfinx import fem, mesh
        import ufl
        from dry_fracture_growth import solve_dry_contact_growth
        from paired_contact import ContactOptions
        from tensile_loss_event import EventSolverOptions
        import test_open_crack_equilibrium as bulk

        space, faces, parents, _, bcs = bulk.BulkFlowTests().fixture()
        parents = np.arange(len(faces.coordinates), dtype=np.int64)
        parents[faces.plus] = parents[faces.minus]
        vertices = np.arange(space.mesh.topology.index_map(0).size_local, dtype=np.int32)
        bcs = [fem.dirichletbc(0., fem.locate_dofs_topological(space.sub(i), 0, vertices), space.sub(i))
               for i in (1, 2)]
        for x, motion in ((-1., -.001), (1., .001)):
            ids = mesh.locate_entities(space.mesh, 0, lambda points: np.isclose(points[0], x))
            bcs.append(fem.dirichletbc(motion, fem.locate_dofs_topological(space.sub(0), 0, ids), space.sub(0)))
        selected = np.zeros(space.mesh.topology.index_map(3).size_local, dtype=bool)
        selected[0] = True
        # Every candidate touches a marker; energy, not marker membership,
        # decides whether it becomes a released surface.
        markers = np.unique(np.sort(faces.minus[:, :2], axis=1), axis=0)
        for gc in (1e-3, 1e9):
            with self.subTest(gc=gc), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "growth"
                def solve():
                    return solve_dry_contact_growth(
                        space, faces, parents, markers, young_pa=1e8, poisson=0.,
                        selected_cells=selected, boundary_conditions=bcs,
                        external_load=lambda v: ufl.dot(fem.Constant(space.mesh, np.zeros(3)), v) * ufl.dx,
                        event_options=EventSolverOptions(),
                        contact_options=ContactOptions(1e-12, 1e-7, 1e-9, 100),
                        growth_options=GrowthOptions(gc, 1e-8, len(faces.minus)), output=root,
                    )
                if gc < 1:
                    with self.assertRaisesRegex(RuntimeError, "NOT evidence of crack arrest"):
                        solve()
                    result = json.loads((root / "growth_report.json").read_text())
                    self.assertGreater(len(result["accepted_surface_ids"]), 0)
                    self.assertEqual(result["failure_kind"], "no_kinematically_admissible_extension")
                    self.assertTrue(result["excluded_kinematically_locked_patches"])
                    self.assertNotIn("accepted_event_directory", result)
                    continue
                result = solve()
                if gc < 1:
                    self.assertGreater(len(result["accepted_surface_ids"]), 0)
                else:
                    self.assertEqual(result["accepted_surface_ids"], [])
                final = root / result["accepted_event_directory"]
                saved = json.loads((final / "event_report.json").read_text())
                self.assertEqual(saved["released_face_ids"], result["accepted_surface_ids"])
                self.assertEqual(saved["dry_problem_sha256"], result["problem_sha256"])
                self.assertTrue((final / "post_event_state.npz").is_file())
                self.assertFalse(result["solver_ready"])
                with np.load(root / "prepared_candidate_inputs.npz") as inputs:
                    np.testing.assert_array_equal(inputs["parent_node_ids"], parents)
                    np.testing.assert_array_equal(inputs["marker_source_edges"], markers)
                    np.testing.assert_array_equal(inputs["candidate_parent_triangles"], parents[faces.minus])
                with np.load(final / "native_inputs.npz") as inputs:
                    mask = np.zeros(len(faces.minus), dtype=bool)
                    mask[result["accepted_surface_ids"]] = True
                    np.testing.assert_array_equal(inputs["released_face_mask"], mask)
                self.assertEqual(
                    json.loads((root / "growth_report.json").read_text())["accepted_event_directory"],
                    result["accepted_event_directory"])

    def test_source_preparation_preserves_rows_materials_and_exterior_tags(self):
        from dolfinx import fem
        from dam_mesh import import_structure, prepare_wall_candidates, source_cell_tags
        from interface_springs import InterfaceSprings
        from test_three_body import column
        from three_body_model import materials
        structure = column()
        original_nodes = structure.nodes.copy()
        original = import_structure(structure, 1e-12)
        prepared = prepare_wall_candidates(structure, [[8, 14]], 1e-12)
        native = prepared.imported
        tags = source_cell_tags(native)
        self.assertEqual(set(tags.values), {0, 1})
        space = fem.functionspace(native.domain, ("Lagrange", 1, (3,)))
        properties = materials(native, structure)
        young = properties["youngs_modulus"]
        np.testing.assert_array_equal(young.x.array[young.function_space.dofmap.list[:, 0]],
                                      np.where(tags.values == 1, 1e7, 2e7))
        springs = InterfaceSprings(native, structure, space,
                                   multiplier_parent_node_ids=prepared.parent_node_ids)
        for invalid in (np.zeros(len(native.points), dtype=np.int64),
                        prepared.parent_node_ids[:-1],
                        np.full(len(native.points), -1),
                        prepared.parent_node_ids.astype(float)):
            with self.assertRaises(ValueError):
                InterfaceSprings(native, structure, space, multiplier_parent_node_ids=invalid)
        self.assertIn("wall_bedrock", springs.report)
        self.assertEqual(
            {tag: len(native.facet_tags.find(tag)) for tag in (10, 11, 12)},
            {tag: len(original.facet_tags.find(tag)) for tag in (10, 11, 12)})
        np.testing.assert_array_equal(np.sort(prepared.parent_node_ids[native.tetrahedra], axis=1),
                                      np.sort(original.tetrahedra, axis=1))
        np.testing.assert_array_equal(structure.nodes, original_nodes)
        self.assertGreater(len(native.points), len(original.points))
        self.assertFalse(prepared.report["solver_ready"])

    def test_source_growth_runs_original_gravity_foundation_and_selected_rows(self):
        from dry_fracture_growth import solve_source_dry_growth
        from paired_contact import ContactOptions
        from tensile_loss_event import EventSolverOptions
        from test_three_body import column
        from dam_mesh import import_structure
        from dam_elasticity import solve as solve_elastic
        structure = column()
        original = import_structure(structure, 1e-12)
        structure.nodes = original.points
        structure.cells = [(4, tuple(map(int, cell))) for cell in original.tetrahedra]
        structure.region_ids = original.regions.tolist()
        affected = int(np.flatnonzero(original.regions == 1)[0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "growth"
            baseline_path = Path(directory) / "baseline"
            baseline_path.mkdir()
            baseline_u, _, _ = solve_elastic(
                original, structure, {"relative_tolerance": 1e-10, "maximum_iterations": 1000}, baseline_path)
            original_xyz, original_u = original.points, baseline_u
            result = solve_source_dry_growth(
                structure, [affected], [[8, 14]], maximum_volume_change=1e-12,
                event_options=EventSolverOptions(), contact_options=ContactOptions(1e-12, 1e-7, 1e-8, 100),
                growth_options=GrowthOptions(1e6, 1e-7, 20), output=root)
            self.assertEqual(result["accepted_surface_ids"], [])
            self.assertFalse(result["source_binding_verified"])
            final = root / result["accepted_event_directory"]
            report = json.loads((final / "event_report.json").read_text())
            np.testing.assert_allclose(report["support_reaction_n"], [0., 0., 30000.], atol=1e-5)
            condensed = report["intact_bonded"]["bond_condensation"]
            self.assertEqual(condensed["condensed_displacement_dofs"], 3 * len(original.points))
            self.assertGreater(condensed["original_displacement_dofs"],
                               condensed["condensed_displacement_dofs"])
            self.assertLess(condensed["condensed_kkt_dofs_upper_bound"],
                            condensed["original_kkt_dofs_upper_bound"])
            self.assertIn("wall_bedrock", report["interface_springs"])
            self.assertTrue((final / "interface_springs.npz").is_file())
            with np.load(final / "interface_springs.npz") as springs:
                self.assertIn("multiplier_row_map", springs.files)
            with np.load(root / "source_mapping.npz") as mapping:
                np.testing.assert_array_equal(mapping["selected_local_cells"],
                                               mapping["local_source_cell_rows"] == affected)
            with np.load(final / "native_inputs.npz") as inputs:
                np.testing.assert_allclose(inputs["assembled_exterior_force_n"].reshape(-1, 3).sum(axis=0),
                                           [0., 0., -30000.], atol=1e-8)
            with (np.load(root / "source_mapping.npz") as mapping,
                  np.load(final / "intact_state.npz") as intact,
                  np.load(final / "native_inputs.npz") as inputs):
                np.testing.assert_allclose(intact["displacement_m"],
                                           original_u[mapping["parent_node_ids"]], atol=1e-11, rtol=1e-8)
                np.testing.assert_allclose(original_xyz[mapping["parent_node_ids"]],
                                           inputs["coordinates_m"],
                                           atol=1e-12)
            self.assertEqual(json.loads((root / "source_report.json").read_text())["status"],
                             "completed_experimental_source_dry_growth")

    def test_source_growth_preserves_exact_partial_reservoir_tailwater_and_top_loads(self):
        from dolfinx import fem
        from dam_mesh import import_structure
        from dam_elasticity import water_load_parameters
        from hydrostatic_load import assemble_water_forces
        from dry_fracture_growth import solve_source_dry_growth
        from paired_contact import ContactOptions
        from tensile_loss_event import EventSolverOptions
        import test_tensile_loss_event as events
        structure = events.TensileLossEventTests().wet_source_fixture()
        structure.loads.upstream_surface_z_m = 1.5
        structure.loads.tailwater_head_m = .5
        imported = import_structure(structure, 1e-12)
        structure.nodes = imported.points
        structure.cells = [(4, tuple(map(int, cell))) for cell in imported.tetrahedra]
        structure.region_ids = imported.regions.tolist()
        space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
        expected = assemble_water_forces(imported, space, *water_load_parameters(imported, structure))
        total_water = expected.reshape(-1, 3).sum(axis=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "growth"
            result = solve_source_dry_growth(
                structure, [int(np.flatnonzero(imported.regions == 1)[0])], [[8, 14]],
                maximum_volume_change=1e-12, event_options=EventSolverOptions(),
                contact_options=ContactOptions(1e-12, 1e-7, 1e-8, 100),
                growth_options=GrowthOptions(1e9, 1e-7, 20), output=root)
            with np.load(root / "source_mapping.npz") as source:
                np.testing.assert_allclose(source["exterior_water_forces_n"].reshape(-1, 3).sum(axis=0),
                                           total_water, atol=1e-8)
            final = root / result["accepted_event_directory"]
            report = json.loads((final / "event_report.json").read_text())
            np.testing.assert_allclose(report["support_reaction_n"],
                                       -total_water + [0., 0., 30000.], atol=.01)
            self.assertIsNotNone(report["dry_mechanical_potential_j"])
            self.assertNotEqual(float(np.linalg.norm(total_water)), 0.)
            self.assertEqual(result["accepted_surface_ids"], [])

    def test_bound_growth_retains_verified_selection_and_exact_native_markers(self):
        from dry_fracture_growth import solve_bound_dry_growth
        from paired_contact import ContactOptions
        from tensile_loss_event import EventSolverOptions
        from test_three_body import column
        from dam_mesh import import_structure
        structure = column()
        original = import_structure(structure, 1e-12)
        structure.nodes = original.points
        structure.cells = [(4, tuple(map(int, cell))) for cell in original.tetrahedra]
        structure.region_ids = original.regions.tolist()
        structure.submitted_configuration_sha256 = "configuration"
        digest = hashlib.sha256(b"source").hexdigest()
        baseline = {"source_sha256": {"three_body.msh": digest}}
        binding = {
            "affected_element_ids": [int(np.flatnonzero(original.regions == 1)[0])],
            "marker_source_edges": [[8, 14]], "crack_start_edge_ids": [123],
            "opening_limit_m": .003, "resolved_faces": {"0": "upstream"},
            "source_mesh_sha256": digest, "configuration_sha256": "configuration",
            "baseline_fields_sha256": "fields", "catalog_sha256": "catalog",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "three_body.msh").write_bytes(b"source")
            (root / "manifest.json").write_text(json.dumps({"crack_seed_source": {"sha256": "fields"}}))
            with patch("run.submitted_structure", return_value=structure), patch(
                "three_body_model.require_verified_baseline", return_value=baseline
            ), patch("damage_selection.bind_source_selection", return_value=binding):
                result = solve_bound_dry_growth(
                    root / "config.json", root, root, root / "fields.npz", {},
                    maximum_volume_change=1e-12, event_options=EventSolverOptions(),
                    contact_options=ContactOptions(1e-12, 1e-7, 1e-8, 100),
                    growth_options=GrowthOptions(1e6, 1e-7, 20), output=root / "growth")
            self.assertTrue(result["source_binding_verified"])
            self.assertFalse(result["solver_ready"])
            saved = json.loads((root / "growth" / "growth_report.json").read_text())
            self.assertEqual(saved, result)
            self.assertEqual(saved["selection_provenance"]["verified_baseline"], baseline)
            self.assertEqual(saved["selection_provenance"]["crack_start_edge_ids"], [123])
            self.assertEqual(saved["selection_provenance"]["opening_limit_m"], .003)
            source = json.loads((root / "growth" / "source_report.json").read_text())
            self.assertTrue(source["source_binding_verified"])
            with np.load(root / "growth" / "trials" / "prepared_candidate_inputs.npz") as inputs:
                np.testing.assert_array_equal(inputs["marker_source_edges"], [[8, 14]])
            final = root / "growth" / result["accepted_event_directory"]
            report = json.loads((final / "event_report.json").read_text())
            self.assertEqual(report["opening_limit_m"], .003)

    def test_bound_growth_rejects_mismatched_evidence_before_preparation(self):
        from dry_fracture_growth import solve_bound_dry_growth
        from test_three_body import column
        for mismatch in ("fields", "mesh", "configuration", "diagnostic"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                digest = hashlib.sha256(b"source").hexdigest()
                (root / "three_body.msh").write_bytes(b"source")
                (root / "manifest.json").write_text(json.dumps({"crack_seed_source": {"sha256": "fields"}}))
                structure = column()
                structure.submitted_configuration_sha256 = "configuration"
                baseline = {"source_sha256": {"three_body.msh": digest}}
                binding = {"baseline_fields_sha256": "fields", "source_mesh_sha256": digest,
                           "configuration_sha256": "configuration"}
                if mismatch == "diagnostic":
                    baseline["source_sha256"]["diagnostic.json"] = digest
                    (root / "diagnostic.json").write_bytes(b"changed")
                else:
                    binding[{"fields": "baseline_fields_sha256", "mesh": "source_mesh_sha256",
                             "configuration": "configuration_sha256"}[mismatch]] = "foreign"
                with patch("run.submitted_structure", return_value=structure), patch(
                    "three_body_model.require_verified_baseline", return_value=baseline
                ), patch("damage_selection.bind_source_selection", return_value=binding), patch(
                    "dry_fracture_growth.solve_source_dry_growth"
                ) as solve:
                    with self.assertRaises(ValueError):
                        solve_bound_dry_growth(
                            root / "config.json", root, root, root / "fields.npz", {},
                            maximum_volume_change=1e-12, event_options=None, contact_options=None,
                            growth_options=None, output=root / "growth")
                    solve.assert_not_called()
                    self.assertFalse((root / "growth").exists())

    def test_bound_growth_failure_preserves_selection_evidence_without_accepted_report(self):
        from dry_fracture_growth import solve_bound_dry_growth
        from paired_contact import ContactOptions
        from tensile_loss_event import EventSolverOptions
        from test_three_body import column
        binding = {
            "affected_element_ids": [1], "marker_source_edges": [[8, 14]],
            "opening_limit_m": .003, "catalog_sha256": "catalog",
            "verified_baseline": {"source_sha256": {"three_body.msh": "mesh"}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "growth"
            with patch("tensile_loss_event.verified_source_selection", return_value=(column(), binding)), patch(
                "dry_fracture_growth.solve_dry_contact_growth", side_effect=RuntimeError("trial failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "no arrested-growth result"):
                    solve_bound_dry_growth(
                        Path(directory) / "config.json", Path(directory), Path(directory),
                        Path(directory) / "fields.npz", {}, maximum_volume_change=1e-12,
                        event_options=EventSolverOptions(),
                        contact_options=ContactOptions(1e-12, 1e-7, 1e-8, 100),
                        growth_options=GrowthOptions(1e6, 1e-7, 20), output=root)
            report = json.loads((root / "source_report.json").read_text())
            self.assertEqual(report["status"], "failed_experimental_source_dry_growth")
            self.assertEqual(report["selection_provenance"], binding)
            self.assertFalse(report["solver_ready"])
            self.assertTrue((root / "source_mapping.npz").is_file())
            self.assertFalse((root / "growth_report.json").exists())

    def test_source_failure_keeps_mapping_and_error_without_top_level_success(self):
        import dry_fracture_growth as growth
        from paired_contact import ContactOptions
        from tensile_loss_event import EventSolverOptions
        from test_three_body import column
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "growth"
            with patch.object(growth, "solve_dry_contact_growth", side_effect=RuntimeError("trial failed")):
                with self.assertRaisesRegex(RuntimeError, "no arrested-growth result"):
                    growth.solve_source_dry_growth(
                        column(), [1], [[8, 14]], maximum_volume_change=1e-12,
                        event_options=EventSolverOptions(),
                        contact_options=ContactOptions(1e-12, 1e-7, 1e-8, 100),
                        growth_options=GrowthOptions(1e6, 1e-7, 20), output=root)
            self.assertTrue((root / "source_mapping.npz").is_file())
            report = json.loads((root / "source_report.json").read_text())
            self.assertEqual(report["status"], "failed_experimental_source_dry_growth")
            self.assertIn("trial failed", report["error"])
            self.assertFalse((root / "growth_report.json").exists())


if __name__ == "__main__":
    unittest.main()
