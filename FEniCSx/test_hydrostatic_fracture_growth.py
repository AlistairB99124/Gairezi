"""Reservoir-work accounting and native common-head finite-candidate growth."""
from pathlib import Path
from dataclasses import asdict
from itertools import combinations
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
import ufl

from dry_fracture_growth import GrowthOptions
from fracture_faces import bulk_face_mapping
from fracture_hydrostatic import HydrostaticBoundary
from fracture_topology import split_interior_wall_faces
from hydrostatic_fracture_growth import (
    HydrostaticGrowthEquilibrium, solve_hydrostatic_contact_growth,
    solve_source_hydrostatic_growth, solve_bound_hydrostatic_growth, source_reservoir_boundaries,
)
from paired_contact import ContactOptions
from tensile_loss_event import EventSolverOptions


class HydrostaticGrowthTests(unittest.TestCase):
    def source_fixture(self):
        from test_three_body import column
        structure = column()
        structure.nodes = np.array([
            (0., 0., 0.), (0., 1., 0.), (0., 0., 1.), (-1., 0., 0.), (1., 0., 0.),
            (0., 0., 0.), (0., 1., 0.), (-1., 0., 0.), (1., 0., 0.), (0., 0., -1.)])
        cells = [(0, 1, 2, 3), (0, 1, 2, 4), (5, 6, 7, 9), (5, 6, 8, 9)]
        structure.cells = [(4, cell) for cell in cells]
        structure.region_ids = [1, 1, 2, 2]
        owners = {}
        for row, cell in enumerate(cells):
            for triangle in combinations(cell, 3):
                owners.setdefault(tuple(sorted(triangle)), []).append(row)
        structure.boundaries = []
        for triangle, rows in owners.items():
            if len(rows) == 1:
                tag = 2 if rows[0] < 2 else 10
                if triangle in ((0, 1, 3), (0, 1, 4)):
                    tag = 11
                if triangle in ((5, 6, 7), (5, 6, 8)):
                    tag = 12
                structure.boundaries.append((tag, triangle))
        structure.concrete_minimum_z = 0.
        structure.loads.upstream_surface_z_m = 4.
        return structure

    def source_options(self):
        return dict(
            maximum_volume_change=1e-12, event_options=EventSolverOptions(maximum_iterations=100),
            contact_options=ContactOptions(1e-12, 1e-7, 1e-8, 100),
            growth_options=GrowthOptions(1e9, 1e-7, 5),
            closure_tolerance_m=1e-12, flow_relative_tolerance=1e-10)

    def fixture(self):
        points = np.array([(0., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                           (-1., 0., 0.), (1., 0., 0.)])
        split = split_interior_wall_faces(
            points, np.array([[0, 1, 2, 3], [0, 1, 2, 4]]), [1, 1], [[0, 1, 2]])
        reference = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        domain = mesh.create_mesh(MPI.COMM_WORLD, split.cells, reference.ufl_domain(), split.coordinates_m)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        xyz, _, mapping = bulk_face_mapping(space, split.faces)
        bcs = [fem.dirichletbc(0., np.asarray(3 * mapping + i, dtype=np.int32), space.sub(i))
               for i in (1, 2)]
        for x, value in ((-1., -.001), (1., .001)):
            dofs = np.asarray(3 * mapping[np.isclose(xyz[:, 0], x)], dtype=np.int32)
            bcs.append(fem.dirichletbc(value, dofs, space.sub(0)))
        source_cells = domain.geometry.input_global_indices[domain.geometry.dofmaps[0]]
        selected = np.any(split.coordinates_m[source_cells, 0] < -.5, axis=1)
        args = dict(
            boundaries=(HydrostaticBoundary("reservoir", 4., [[0, 1]]),),
            water_density_kg_m3=1000., gravity_m_s2=9.81, closure_tolerance_m=1e-12,
            flow_relative_tolerance=1e-10, young_pa=1e8, poisson=0., selected_cells=selected,
            boundary_conditions=bcs,
            external_load=lambda v: ufl.dot(fem.Constant(domain, np.zeros(3)), v) * ufl.dx,
            linear_support=lambda u, v: 1e6 * ufl.dot(u, v) * ufl.dx,
            event_options=EventSolverOptions(maximum_iterations=100),
            contact_options=ContactOptions(1e-12, 1e-7, 1e-9, 100))
        return space, split, args

    def test_native_reservoir_work_drives_release_and_surface_cost_can_arrest(self):
        space, split, args = self.fixture()
        for resistance, expected in ((.01, [0]), (1e8, [])):
            with self.subTest(resistance=resistance), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "growth"
                result = solve_hydrostatic_contact_growth(
                    space, split.faces, split.parent_node_ids, [[0, 1]], **args,
                    growth_options=GrowthOptions(resistance, 1e-8, 2), output=root)
                self.assertEqual(result["accepted_surface_ids"], expected)
                self.assertEqual(result["potential_model"], "common_head_hydrostatic")
                trial = root / "step_0000_candidate_000000"
                report = json.loads((trial / "event_report.json").read_text())
                self.assertIsNone(report["dry_mechanical_potential_j"])
                with np.load(trial / "post_event_state.npz") as state:
                    displacement = state["displacement_m"]
                pressure = 9810 * (4 - split.faces.coordinates[split.faces.minus, 2])
                mass = split.faces.areas[0] / 12 * (np.ones((3, 3)) + np.eye(3))
                opening = np.asarray(split.faces.measure(displacement)["signed_vertex_gaps_m"])[0]
                work = float(pressure[0] @ mass @ opening)
                self.assertAlmostEqual(report["hydrostatic_reservoir_work_j"], work, places=9)
                self.assertGreater(work, 0.)
                potential = report["stored_mechanical_energy_j"] - report["exterior_dead_load_work_j"] - work
                self.assertAlmostEqual(report["conservative_hydrostatic_potential_j"], potential, places=9)
                initial = json.loads((root / "initial" / "event_report.json").read_text())
                release = initial["dry_mechanical_potential_j"] - potential
                self.assertAlmostEqual(result["trials"][0]["conservative_energy_release_j"], release, places=9)
                self.assertNotIn("mechanical_energy_release_j", result["trials"][0])
                self.assertGreater(release, .01 * split.faces.areas.sum())
                self.assertTrue((root / result["accepted_event_directory"] / "post_event_state.npz").is_file())
                self.assertFalse(result["solver_ready"])
                mechanical_only_release = initial["dry_mechanical_potential_j"] - (
                    report["stored_mechanical_energy_j"] - report["exterior_dead_load_work_j"])
                threshold = (max(0., mechanical_only_release) + release) / (2 * split.faces.areas.sum())
                self.assertGreater(threshold * split.faces.areas.sum(), mechanical_only_release)
                self.assertLess(threshold * split.faces.areas.sum(), release)
        with tempfile.TemporaryDirectory() as directory:
            result = solve_hydrostatic_contact_growth(
                space, split.faces, split.parent_node_ids, [[0, 1]], **args,
                growth_options=GrowthOptions(threshold, 1e-8, 2), output=Path(directory) / "growth")
            self.assertEqual(result["accepted_surface_ids"], [0])

    def test_actual_different_head_connection_and_foreign_inlets_are_rejected(self):
        space, split, args = self.fixture()
        for boundaries in (
            (HydrostaticBoundary("upstream", 4., [[0, 1]]),
             HydrostaticBoundary("tailwater", 3., [[1, 2]])),
            (HydrostaticBoundary("foreign", 4., [[0, 999]]),),
        ):
            with self.subTest(boundaries=boundaries), tempfile.TemporaryDirectory() as directory:
                args["boundaries"] = boundaries
                root = Path(directory) / "growth"
                with self.assertRaises((ValueError, RuntimeError)):
                    solve_hydrostatic_contact_growth(
                        space, split.faces, split.parent_node_ids, [[0, 1]], **args,
                        growth_options=GrowthOptions(1., 1e-8, 2), output=root)
                self.assertFalse((root / "growth_report.json").is_file() and
                                 json.loads((root / "growth_report.json").read_text())["status"].startswith("arrested"))

    def test_wet_growth_uses_selected_strength_and_records_surface_pressure(self):
        space, split, args = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "growth"
            result = solve_hydrostatic_contact_growth(
                space, split.faces, split.parent_node_ids, [[0, 1]], **args,
                selected_tensile_strength_pa=1000.,
                growth_options=GrowthOptions(.01, 1e-8, 2), output=root,
            )
            self.assertEqual(result["accepted_surface_ids"], [0])
            final = root / result["accepted_event_directory"]
            report = json.loads((final / "event_report.json").read_text())
            self.assertTrue(report["reused_bound_intact_equilibrium"])
            self.assertEqual(report["selected_tensile_strength_pa"], 1000.)
            self.assertLessEqual(
                report["iterations"][-1]["maximum_selected_principal_stress_pa"], 1000. + 1e-6,
            )
            with np.load(final / "hydraulic_state.npz") as water:
                np.testing.assert_allclose(
                    water["pressure_pa"], 9810 * (4 - split.faces.coordinates[split.faces.minus, 2]),
                )
                self.assertGreater(np.linalg.norm(water["nodal_water_forces_n"]), 0.)
            with np.load(final / "native_inputs.npz") as inputs:
                self.assertEqual(float(inputs["selected_tensile_strength_pa"]), 1000.)

    def test_supported_closed_trial_is_rejected_by_pressure_bound_not_accepted_as_wet(self):
        space, split, args = self.fixture()
        xyz, _, mapping = bulk_face_mapping(space, split.faces)
        bcs = [
            fem.dirichletbc(0., np.asarray(3 * mapping[~np.isclose(xyz[:, 0], 1.)] + 1,
                                          dtype=np.int32), space.sub(1)),
            fem.dirichletbc(0., np.asarray(3 * mapping + 2, dtype=np.int32), space.sub(2))]
        for x, value in ((-1., .001), (1., -.001)):
            dofs = np.asarray(3 * mapping[np.isclose(xyz[:, 0], x)], dtype=np.int32)
            bcs.append(fem.dirichletbc(value, dofs, space.sub(0)))
        mouth = np.asarray(3 * mapping[np.isclose(xyz[:, 0], 0.)], dtype=np.int32)
        bcs.append(fem.dirichletbc(0., mouth, space.sub(0)))
        args["boundary_conditions"] = bcs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "growth"
            result = solve_hydrostatic_contact_growth(
                space, split.faces, split.parent_node_ids, [[0, 1]], **args,
                selected_tensile_strength_pa=2e6,
                growth_options=GrowthOptions(.01, 1e-8, 2, resolve_locked_patches=True), output=root)
            self.assertEqual(result["accepted_surface_ids"], [])
            self.assertEqual(len(result["accepted_equilibria"]), 1)
            row = result["trials"][0]
            self.assertEqual(row["assessment"], "rejected_by_optimistic_pressure_energy_bound")
            self.assertGreaterEqual(row["total_energy_change_lower_bound_j"], -1e-8)
            trial = root / "step_0000_candidate_000000"
            self.assertFalse((trial / "post_event_state.npz").exists())
            self.assertFalse((trial / "hydraulic_state.npz").exists())
            self.assertTrue((trial / "unaccepted_pressure_bound_state.npz").exists())
            report = json.loads((trial / "event_report.json").read_text())
            self.assertEqual(report["status"], "failed_reduced_tensile_contact_event")
            self.assertEqual(report["failure_kind"], "unaccepted_hydraulic_access_energy_bound")
            self.assertLess(report["iterations"][-1]["scaled_free_residual"], 1e-8)
            self.assertLess(report["iterations"][-1]["scaled_force_imbalance"], 1e-8)
            self.assertNotIn("conservative_hydrostatic_potential_j", report)
            self.assertFalse(report["opening_assessment"]["completed"])

    def test_locked_seed_becomes_audited_wet_patch_with_irreversible_growth(self):
        from fracture_topology import interior_wall_candidates
        from hydrostatic_load import triangle_pressure_weights
        points = np.array([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.),
                           (0., 0., 1.), (.25, .25, .25)])
        cells = np.array([[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]])
        markers = np.array([[0, 1]])
        candidates = interior_wall_candidates(points, cells, [1] * 4, markers)
        split = split_interior_wall_faces(points, cells, [1] * 4, candidates)
        reference = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        domain = mesh.create_mesh(MPI.COMM_WORLD, split.cells, reference.ufl_domain(), split.coordinates_m)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        x = ufl.SpatialCoordinate(domain)
        args = dict(
            boundaries=(HydrostaticBoundary("reservoir", 4., markers),),
            water_density_kg_m3=1000., gravity_m_s2=9.81, closure_tolerance_m=1e-12,
            flow_relative_tolerance=1e-10, young_pa=1e8, poisson=.2,
            selected_cells=np.ones(4, dtype=bool), selected_tensile_strength_pa=2e6,
            boundary_conditions=(),
            external_load=lambda v: 1e5 * ufl.dot(x - ufl.as_vector((.25, .25, .25)), v) * ufl.dx,
            linear_support=lambda u, v: 1e9 * ufl.dot(u, v) * ufl.dx,
            event_options=EventSolverOptions(),
            contact_options=ContactOptions(1e-12, 1e-7, 1e-8, 100))
        accepted = ()
        saw_disconnected_released_face = False
        for energy, previous in ((1e9, ()), (.001, ()), (1e9, None)):
            previous = accepted if previous is None else previous
            with self.subTest(energy=energy, previous=previous), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "growth"
                result = solve_hydrostatic_contact_growth(
                    space, split.faces, split.parent_node_ids, markers, **args,
                    previously_broken=previous,
                    growth_options=GrowthOptions(energy, 1e-8, 10, resolve_locked_patches=True),
                    output=root)
                self.assertEqual(result["status"], "arrested_in_tested_node_star_patch_candidate_set")
                if not previous:
                    self.assertTrue(result["excluded_kinematically_locked_patches"])
                    self.assertTrue(result["generated_node_star_patches"])
                    self.assertEqual(len(result["trials"][0]["candidate_surface_ids"]), 2)
                    trial = json.loads((root / "step_0000_candidate_patch_000000/event_report.json").read_text())
                    self.assertGreater(trial["opening"]["maximum_opening_m"], 0.)
                    self.assertGreater(trial["hydrostatic_reservoir_work_j"], 0.)
                if energy < 1:
                    accepted = tuple(result["accepted_surface_ids"])
                    self.assertGreaterEqual(len(result["accepted_extensions"]), 2)
                    opening = result["opening_assessment"]["peak_opening_m"]
                    self.assertGreater(opening, 0.)
                    self.assertLess(opening, .002)
                else:
                    self.assertEqual(result["accepted_surface_ids"], list(previous))
                    self.assertEqual(result["accepted_extensions"], [])
                for row in result["trials"]:
                    np.testing.assert_allclose(
                        row["surface_energy_cost_j"],
                        energy * split.faces.areas[row["candidate_surface_ids"]].sum(), rtol=1e-14)
                states = result["accepted_equilibria"]
                for before, after in zip(states, states[1:]):
                    self.assertTrue(set(before["surface_ids"]) < set(after["surface_ids"]))
                for event_path in root.glob("*/event_report.json"):
                    report = json.loads(event_path.read_text())
                    self.assertLess(report["iterations"][-1]["scaled_free_residual"], 1e-8)
                    self.assertLess(report["iterations"][-1]["scaled_force_imbalance"], 1e-8)
                    self.assertGreaterEqual(report["opening"]["minimum_signed_gap_m"], -1e-12)
                    if report.get("hydraulics") is None:
                        continue
                    with (np.load(event_path.parent / "hydraulic_state.npz") as water,
                          np.load(event_path.parent / "post_event_state.npz") as state):
                        gaps = np.asarray(split.faces.measure(state["displacement_m"])["signed_vertex_gaps_m"])
                        ids = water["mechanical_inventory_face_ids"]
                        expected_work = sum(
                            triangle_pressure_weights(split.faces.coordinates[split.faces.minus[face]], 4., 9810.)
                            @ gaps[face] for face in ids)
                        self.assertAlmostEqual(report["hydrostatic_reservoir_work_j"], expected_work)
                        if set(report["released_face_ids"]) - set(ids):
                            self.assertTrue(report["explicitly_dry_released_face_ids"])
                            saw_disconnected_released_face = True
        self.assertTrue(saw_disconnected_released_face)

    def test_native_partially_submerged_release_includes_exact_reservoir_work(self):
        from hydrostatic_load import triangle_pressure_weights
        space, split, args = self.fixture()
        args["boundaries"] = (HydrostaticBoundary("reservoir", .5, [[0, 1]]),)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "growth"
            result = solve_hydrostatic_contact_growth(
                space, split.faces, split.parent_node_ids, [[0, 1]], **args,
                selected_tensile_strength_pa=1000.,
                growth_options=GrowthOptions(.01, 1e-8, 2), output=root)
            self.assertEqual(result["accepted_surface_ids"], [0])
            final = root / result["accepted_event_directory"]
            report = json.loads((final / "event_report.json").read_text())
            with np.load(final / "post_event_state.npz") as state:
                opening = np.asarray(split.faces.measure(state["displacement_m"])["signed_vertex_gaps_m"])[0]
            weights = triangle_pressure_weights(split.faces.coordinates[split.faces.minus[0]], .5, 9810.)
            self.assertAlmostEqual(report["hydrostatic_reservoir_work_j"], weights @ opening)
            self.assertGreater(report["hydrostatic_reservoir_work_j"], 0.)
            self.assertEqual(report["hydraulics"]["partially_submerged_face_ids"], [0])
            self.assertLess(report["iterations"][-1]["scaled_force_imbalance"], 1e-8)
            with (np.load(root / "initial" / "native_inputs.npz") as initial,
                  np.load(final / "native_inputs.npz") as released):
                np.testing.assert_array_equal(initial["assembled_exterior_force_n"], released["assembled_exterior_force_n"])
                self.assertEqual(float(released["selected_tensile_strength_pa"]), 1000.)

    def test_reservoir_inventory_keeps_unrelated_heads_and_waterline_edges(self):
        from dam_mesh import import_structure, wall_candidate_triangles
        from hydrostatic_fracture_growth import imported_reservoir_boundaries
        structure = self.source_fixture()
        structure.loads.upstream_surface_z_m = .5
        structure.loads.tailwater_head_m = .25
        structure.boundaries = [
            (3 if triangle == (0, 2, 4) else tag, triangle)
            for tag, triangle in structure.boundaries]
        imported = import_structure(structure, 1e-12)
        boundaries = imported_reservoir_boundaries(
            imported, structure, np.arange(len(imported.points)), wall_candidate_triangles(imported, [[0, 1]]))
        self.assertEqual({b.elevation_m for b in boundaries}, {.5, .25})
        self.assertTrue(any(
            imported.points[b.inlet_source_edges, 2].max() > b.elevation_m for b in boundaries))

    def test_equilibrium_seed_cannot_change_bound_strength_or_loads(self):
        from tensile_loss_event import solve_contact_event
        space, split, args = self.fixture()
        common = dict(
            young_pa=args["young_pa"], poisson=args["poisson"], selected_cells=args["selected_cells"],
            boundary_conditions=args["boundary_conditions"], external_load=args["external_load"],
            linear_support=args["linear_support"], options=args["event_options"],
            contact_options=args["contact_options"], selected_tensile_strength_pa=1000.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = solve_contact_event(
                space, split.faces, **common, released_face_mask=np.zeros(1, dtype=bool),
                output=root / "initial")
            for name, value in (("selected_tensile_strength_pa", 2000.), ("young_pa", 2e8)):
                with self.subTest(field=name), self.assertRaisesRegex(RuntimeError, "exact mechanical problem"):
                    solve_contact_event(
                        space, split.faces, **{**common, name: value},
                        equilibrium_seed=seed, output=root / name)
            plain = solve_contact_event(space, split.faces, **common, output=root / "plain")
            reused = solve_contact_event(
                space, split.faces, **common, equilibrium_seed=seed, output=root / "reused")
            np.testing.assert_allclose(
                reused.post_event_displacement.x.array, plain.post_event_displacement.x.array,
                rtol=1e-8, atol=1e-11)
            self.assertTrue(reused.report["reused_bound_intact_equilibrium"])
            self.assertLess(reused.report["iterations"][-1]["scaled_force_imbalance"], 1e-8)

    def test_unreleased_connection_to_another_head_does_not_block_trial(self):
        from test_open_crack_equilibrium import BulkFlowTests
        from fracture_faces import bonded_node_representatives
        space, faces, parents, _, _ = BulkFlowTests().fixture()
        parents = np.arange(len(faces.coordinates), dtype=np.int64)
        parents[faces.plus] = parents[faces.minus]
        released = np.zeros(len(faces.minus), dtype=bool)
        released[0] = True
        representative = bonded_node_representatives(faces, ~released)
        first_edges = {tuple(sorted(map(int, parents[row[[i, j]]])))
                       for row in representative[faces.minus[released]]
                       for i, j in ((0, 1), (1, 2), (2, 0))}
        other = next(tuple(sorted(map(int, parents[row[[i, j]]])))
                     for row in faces.minus for i, j in ((0, 1), (1, 2), (2, 0))
                     if tuple(sorted(map(int, parents[row[[i, j]]]))) not in first_edges)
        first = sorted(first_edges)[0]
        args = self.fixture()[2]
        args["boundaries"] = (HydrostaticBoundary("reservoir", 4., [first]),
                              HydrostaticBoundary("remote_tailwater", 3., [other]))
        def verify_factory(*unused, **kwargs):
            water = kwargs["hydraulic_factory"](released)
            self.assertEqual([b.name for b in water.boundaries], ["reservoir"])
            return {}
        with tempfile.TemporaryDirectory() as directory, patch(
                "hydrostatic_fracture_growth._solve_contact_growth", side_effect=verify_factory):
            solve_hydrostatic_contact_growth(
                space, faces, parents, [first], **args,
                growth_options=GrowthOptions(.01, 1e-8, 2), output=Path(directory))

    def test_closed_trial_is_failed_not_arrested_and_keeps_reservoir_identity(self):
        space, split, args = self.fixture()
        args["closure_tolerance_m"] = 1.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "growth"
            with self.assertRaisesRegex(RuntimeError, "no arrested-growth result"):
                solve_hydrostatic_contact_growth(
                    space, split.faces, split.parent_node_ids, [[0, 1]], **args,
                    growth_options=GrowthOptions(1., 1e-8, 2), output=root)
            report = json.loads((root / "growth_report.json").read_text())
            self.assertEqual(report["status"], "failed_experimental_common_head_hydrostatic_growth")
            self.assertEqual(report["accepted_surface_ids"], [])
            reservoir = report["problem_provenance"]["reservoir_problem"]
            self.assertEqual(reservoir["head_m"], 4.)
            self.assertEqual(reservoir["boundaries"], [{"name": "reservoir", "head_m": 4., "edges": [[0, 1]]}])
            trial = root / "step_0000_candidate_000000"
            self.assertTrue((trial / "intact_state.npz").is_file())
            self.assertFalse((trial / "post_event_state.npz").exists())

    def test_hydrostatic_equilibrium_cannot_omit_reservoir_evidence(self):
        evidence = {"equilibrium_verified": True, "common_head_reservoir_work_included": True}
        state = HydrostaticGrowthEquilibrium(-1., "a" * 64, evidence)
        self.assertEqual(state.potential_j, -1.)
        for energy, digest, record in ((np.nan, "a" * 64, evidence), (True, "a" * 64, evidence),
                                       (1., "foreign", evidence), (1., "a" * 64, {})):
            with self.assertRaises(ValueError):
                HydrostaticGrowthEquilibrium(energy, digest, record)

    def test_native_existing_wet_surface_extends_without_healing_prior_faces(self):
        self.check_native_existing_wet_surface_extension(patch_size=1)

    def test_native_connected_two_face_wet_jump_keeps_full_reservoir_work_and_prior_faces(self):
        self.check_native_existing_wet_surface_extension(patch_size=2)

    def test_native_iterative_condensed_two_face_wet_jump_passes_original_space_audits(self):
        self.check_native_existing_wet_surface_extension(patch_size=2, linear_solver="minres")

    def check_native_existing_wet_surface_extension(self, patch_size, linear_solver="direct"):
        from itertools import combinations
        from fracture_faces import bonded_node_representatives
        import test_open_crack_equilibrium as bulk
        space, faces, parents, _, bcs = bulk.BulkFlowTests().fixture()
        parents = np.arange(len(faces.coordinates), dtype=np.int64)
        parents[faces.plus] = parents[faces.minus]
        released = np.ones(len(faces.minus), dtype=bool)
        released[0] = False
        if patch_size == 2:
            neighbour = next(i for i in range(1, len(faces.minus))
                             if len(set(faces.minus[0]) & set(faces.minus[i])) == 2)
            released[neighbour] = False
        representatives = bonded_node_representatives(faces, ~released)
        owners = {}
        for triangle in representatives[faces.minus[released]]:
            for a, b in combinations(triangle, 2):
                edge = tuple(sorted((int(a), int(b))))
                owners[edge] = owners.get(edge, 0) + 1
        tied = set(representatives[faces.minus[~released]].ravel())
        inlet = next(edge for edge, count in owners.items()
                     if count == 1 and not any(node in tied for node in edge))
        count = space.mesh.topology.index_map(3).size_local
        centres = mesh.compute_midpoints(space.mesh, 3, np.arange(count, dtype=np.int32))
        x = ufl.SpatialCoordinate(space.mesh)
        for resistance, extend in ((.01, True), (1e8, False)):
            with self.subTest(resistance=resistance), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "growth"
                result = solve_hydrostatic_contact_growth(
                    space, faces, parents, [parents[list(inlet)]],
                    boundaries=(HydrostaticBoundary("reservoir", 4., [parents[list(inlet)]]),),
                    water_density_kg_m3=1000., gravity_m_s2=9.81, closure_tolerance_m=1e-12,
                    flow_relative_tolerance=1e-10, young_pa=1e8, poisson=0.,
                    selected_cells=centres[:, 0] < -.75, boundary_conditions=bcs,
                    external_load=lambda v: fem.Constant(space.mesh, 1e5) * ufl.conditional(
                        x[0] < 0, -1., 1.) * v[0] * ufl.dx,
                    event_options=EventSolverOptions(), contact_options=ContactOptions(
                        1e-12, 1e-7, 1e-9, 100, linear_solver=linear_solver,
                        linear_relative_tolerance=1e-13),
                    growth_options=GrowthOptions(resistance, 1e-8, 2, maximum_patch_faces=patch_size),
                    previously_broken=tuple(map(int, np.flatnonzero(released))), output=root)
                expected = list(range(len(faces.minus))) if extend else np.flatnonzero(released).tolist()
                self.assertEqual(result["accepted_surface_ids"], expected)
                self.assertEqual(len(result["accepted_extensions"]), int(extend))
                initial = json.loads((root / "initial" / "event_report.json").read_text())
                self.assertGreater(len(initial["hydraulics"]["tied_front_vertex_ids"]), 0)
                final = root / result["accepted_event_directory"]
                report = json.loads((final / "event_report.json").read_text())
                self.assertTrue(all(
                    item["method"] == linear_solver
                    for item in report["iterations"][-1]["contact"]["linear_systems"]))
                self.assertLess(report["iterations"][-1]["scaled_free_residual"], 1e-8)
                self.assertLess(report["iterations"][-1]["scaled_force_imbalance"], 1e-8)
                expected_cost = resistance * (faces.areas[~released].sum() if extend else 0.)
                self.assertAlmostEqual(result["new_fracture_surface_energy_j"], expected_cost)
                self.assertEqual(result["opening_assessment"]["completed"], True)
                if extend:
                    self.assertEqual(result["accepted_extensions"][0]["candidate_surface_ids"],
                                     np.flatnonzero(~released).tolist())
                    self.assertGreater(report["hydrostatic_reservoir_work_j"], 0.)
                    extension = result["accepted_extensions"][0]
                    release = (initial["conservative_hydrostatic_potential_j"]
                               - report["conservative_hydrostatic_potential_j"])
                    self.assertAlmostEqual(extension["conservative_energy_release_j"], release)
                    self.assertAlmostEqual(extension["total_energy_change_j"], expected_cost - release)

    def test_source_boundary_discovery_rejects_unmodelled_water_connections_and_keeps_failure(self):
        cases = {
                "crest": "Constant-pressure inlet must be horizontal",
                "no_inlet": "No submerged candidate edge",
        }
        for case, message in cases.items():
                with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                    structure = self.source_fixture()
                    if case == "crest":
                        structure.boundaries = [
                            (4 if triangle == (0, 2, 4) else tag, triangle)
                            for tag, triangle in structure.boundaries]
                    else:
                        structure.boundaries = [
                            (7 if tag == 2 else tag, triangle) for tag, triangle in structure.boundaries]
                    root = Path(directory) / "growth"
                    with patch("hydrostatic_fracture_growth.solve_hydrostatic_contact_growth") as solve:
                        with self.assertRaisesRegex(RuntimeError, message):
                            solve_source_hydrostatic_growth(
                                structure, [0], [[0, 1]], **self.source_options(), output=root)
                        solve.assert_not_called()
                    report = json.loads((root / "source_report.json").read_text())
                    self.assertEqual(report["status"], "failed_experimental_source_common_head_hydrostatic_growth")
                    self.assertTrue((root / "source_mapping.npz").is_file())
                    self.assertFalse((root / "growth_report.json").exists())

    def test_bound_source_wet_growth_retains_selection_limit_and_verified_evidence(self):
        structure = self.source_fixture()
        structure.submitted_configuration_sha256 = "configuration"
        digest = hashlib.sha256(b"source").hexdigest()
        baseline = {"source_sha256": {"three_body.msh": digest}}
        binding = {
                "affected_element_ids": [0], "marker_source_edges": [[0, 1]],
                "crack_start_edge_ids": [42], "opening_limit_m": .003,
                "source_mesh_sha256": digest, "configuration_sha256": "configuration",
                "baseline_fields_sha256": "fields", "catalog_sha256": "catalog",
                "resolved_faces": {"0": "upstream"},
        }
        with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "three_body.msh").write_bytes(b"source")
                (root / "manifest.json").write_text(json.dumps({"crack_seed_source": {"sha256": "fields"}}))
                with patch("run.submitted_structure", return_value=structure), patch(
                    "three_body_model.require_verified_baseline", return_value=baseline
                ), patch("damage_selection.bind_source_selection", return_value=binding):
                    result = solve_bound_hydrostatic_growth(
                        root / "config.json", root, root, root / "fields.npz", {},
                        **self.source_options(), output=root / "growth")
                self.assertTrue(result["source_binding_verified"])
                self.assertFalse(result["solver_ready"])
                self.assertEqual(result["opening_assessment"]["opening_limit_m"], .003)
                self.assertEqual(result["selection_provenance"], {**binding, "verified_baseline": baseline})
                saved = json.loads((root / "growth" / "growth_report.json").read_text())
                self.assertEqual(saved, result)
                source = json.loads((root / "growth" / "source_report.json").read_text())
                self.assertTrue(source["source_binding_verified"])
                self.assertEqual(source["selection_provenance"], result["selection_provenance"])
                for mismatch in ("fields", "mesh", "configuration", "diagnostic"):
                    with self.subTest(mismatch=mismatch):
                        foreign = dict(binding)
                        evidence = {"source_sha256": {"three_body.msh": digest}}
                        if mismatch == "diagnostic":
                            evidence["source_sha256"]["diagnostic.json"] = digest
                            (root / "diagnostic.json").write_bytes(b"changed")
                        else:
                            foreign[{"fields": "baseline_fields_sha256", "mesh": "source_mesh_sha256",
                                     "configuration": "configuration_sha256"}[mismatch]] = "foreign"
                        invalid = root / mismatch
                        with patch("run.submitted_structure", return_value=structure), patch(
                            "three_body_model.require_verified_baseline", return_value=evidence
                        ), patch("damage_selection.bind_source_selection", return_value=foreign), patch(
                            "hydrostatic_fracture_growth.solve_source_hydrostatic_growth"
                        ) as solve:
                            with self.assertRaises(ValueError):
                                solve_bound_hydrostatic_growth(
                                    root / "config.json", root, root, root / "fields.npz", {},
                                    **self.source_options(), output=invalid)
                            solve.assert_not_called()
                            self.assertFalse(invalid.exists())

    def test_source_discovery_keeps_matching_heads_and_never_uses_painted_marker(self):
        from dam_mesh import prepare_wall_candidates
        structure = self.source_fixture()
        structure.loads.tailwater_head_m = 4.
        structure.boundaries = [
            (3 if triangle == (0, 2, 4) else tag, triangle)
            for tag, triangle in structure.boundaries]
        prepared = prepare_wall_candidates(structure, [[0, 1]], 1e-12)
        boundaries = source_reservoir_boundaries(prepared, structure)
        self.assertEqual([boundary.name for boundary in boundaries], ["exterior_tag_2", "exterior_tag_3"])
        self.assertEqual({boundary.elevation_m for boundary in boundaries}, {4.})
        self.assertIn([0, 2], boundaries[1].inlet_source_edges.tolist())
        self.assertNotIn([0, 1], [edge for boundary in boundaries
                                      for edge in boundary.inlet_source_edges.tolist()])

    def test_native_unsplit_candidate_inventory_matches_independent_cell_owner_oracle(self):
        from dam_mesh import import_structure, prepare_wall_candidates, wall_candidate_triangles
        from fracture_topology import interior_wall_candidates
        from hydrostatic_fracture_growth import imported_reservoir_boundaries
        from FEniCSx.painted_event_runner import source_preflight
        from test_three_body import column, jointed_column
        for structure, marker in ((self.source_fixture(), [[0, 1]]),
                                  (column(), [[8, 14]]), (jointed_column(), [[8, 14]])):
            with self.subTest(cells=len(structure.cells)):
                imported = import_structure(structure, 1e-12)
                expected = interior_wall_candidates(
                    imported.points, imported.tetrahedra, imported.regions, marker)
                actual = wall_candidate_triangles(imported, marker)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(
                    prepare_wall_candidates(structure, marker, 1e-12).source_triangles, expected)
        structure = self.source_fixture()
        original = import_structure(structure, 1e-12)
        candidates = wall_candidate_triangles(original, [[0, 1]])
        original_boundaries = imported_reservoir_boundaries(
            original, structure, np.arange(len(original.points)), candidates)
        prepared = prepare_wall_candidates(structure, [[0, 1]], 1e-12)
        split_boundaries = source_reservoir_boundaries(prepared, structure)
        self.assertEqual(len(original_boundaries), len(split_boundaries))
        for before, after in zip(original_boundaries, split_boundaries):
            self.assertEqual((before.name, before.elevation_m), (after.name, after.elevation_m))
            np.testing.assert_array_equal(before.inlet_source_edges, after.inlet_source_edges)
        with patch("FEniCSx.fracture_topology.split_interior_wall_faces") as split:
            report = source_preflight(structure, [[0, 1]], 1e-12, "common_head_hydrostatic")
            split.assert_not_called()
        self.assertEqual(report["candidate_faces"], 1)
        self.assertFalse(report["candidate_mesh_split"])
        self.assertFalse(report["numerical_growth_solved"])
        for invalid in ([[0, 999]], [[0, 0]], [[0., 1.]], [[0, 1], [1, 0]], [[3, 4]]):
            with self.subTest(marker=invalid), self.assertRaises(ValueError):
                wall_candidate_triangles(original, invalid)
        structure.loads.upstream_surface_z_m = .5
        structure.boundaries = [
            (4 if triangle == (0, 2, 4) else tag, triangle)
            for tag, triangle in structure.boundaries]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            with self.assertRaisesRegex(ValueError, "Constant-pressure inlet must be horizontal"):
                source_preflight(structure, [[0, 1]], 1e-12, "common_head_hydrostatic", output=path)
            failed = json.loads(path.read_text())
            self.assertEqual(failed["candidate_faces"], 1)
            self.assertEqual(failed["status"], "failed_unsplit_candidate_preflight_not_solved")
            self.assertFalse(failed["candidate_mesh_split"])

    def test_bound_source_failure_retains_baseline_provenance_without_success(self):
        structure = self.source_fixture()
        structure.boundaries = [
            (4 if triangle == (0, 2, 4) else tag, triangle)
            for tag, triangle in structure.boundaries]
        binding = {
            "affected_element_ids": [0], "marker_source_edges": [[0, 1]], "opening_limit_m": .003,
            "catalog_sha256": "catalog", "verified_baseline": {"source_sha256": {"three_body.msh": "mesh"}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "growth"
            with patch("tensile_loss_event.verified_source_selection", return_value=(structure, binding)):
                with self.assertRaisesRegex(RuntimeError, "Constant-pressure inlet must be horizontal"):
                    solve_bound_hydrostatic_growth(
                            Path(directory) / "config.json", Path(directory), Path(directory),
                            Path(directory) / "fields.npz", {}, **self.source_options(), output=root)
            report = json.loads((root / "source_report.json").read_text())
            self.assertEqual(report["selection_provenance"], binding)
            self.assertFalse(report["source_binding_verified"])
            self.assertFalse(report["solver_ready"])
            self.assertTrue((root / "source_mapping.npz").is_file())
            self.assertFalse((root / "growth_report.json").exists())

    def test_actual_gmsh_catalog_selection_reaches_accepted_native_wet_growth(self):
        from FEniCSx.damage_selection import build_catalog
        structure = self.source_fixture()
        structure.three_body_config["material"]["concrete"].update({
            "tensile_strength_pa": 2e6, "selected_tensile_strength_pa": 0.})
        structure.nodes[:, 1] += 80.
        structure.three_body_config["geometry"].update({
            "global_element_size_m": .5,
            "wall": {"upstream_radius_m": 82., "downstream_radius_m": 78., "centerline_radius_m": 80.}})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configuration = root / "config.json"
            configuration.write_text(json.dumps(structure.three_body_config))
            structure.submitted_configuration_sha256 = hashlib.sha256(configuration.read_bytes()).hexdigest()
            source = root / "three_body.msh"
            lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(structure.nodes))]
            lines.extend(f"{i} {x} {y} {z}" for i, (x, y, z) in enumerate(structure.nodes, 1))
            lines.extend(["$EndNodes", "$Elements", str(len(structure.cells) + len(structure.boundaries))])
            index = 1
            for (_, cell), region in zip(structure.cells, structure.region_ids):
                lines.append(f"{index} 4 2 {region} {region} " + " ".join(str(node + 1) for node in cell))
                index += 1
            for tag, triangle in structure.boundaries:
                lines.append(f"{index} 2 2 {tag} {tag} " + " ".join(str(node + 1) for node in triangle))
                index += 1
            lines.append("$EndElements")
            source.write_text("\n".join(lines) + "\n")
            fields = root / "fields.npz"
            np.savez(fields, coordinates_m=structure.nodes, stress_pa=np.zeros((len(structure.nodes), 6)),
                     node_regions=np.array([1] * 5 + [2] * 5))
            catalog = build_catalog(source, fields, structure.three_body_config, configuration.read_bytes())
            marker = next(edge for edge in catalog["edges"] if edge["node_ids"] == [1, 3])
            request = {
                "catalog_sha256": catalog["catalog_sha256"], "affected_element_ids": [0],
                "crack_start_edge_ids": [marker["id"]], "face_overrides": {},
                "opening_limit_m": .003}
            structure.three_body_source = {
                "sha256": {"three_body.msh": hashlib.sha256(source.read_bytes()).hexdigest()}}
            gate = {"passed": True, "relative_l2_displacement_difference": 0.,
                    "relative_nodal_stress_difference": 0.,
                    "evidence_scope": "synthetic reduced-test gate, not independent solver parity"}
            manifest = {
                "status": "completed_not_engineering_validated", "scenario_name": "Baseline_NoPhase",
                "configuration_sha256": structure.submitted_configuration_sha256,
                "crack_seed_source": {"sha256": hashlib.sha256(fields.read_bytes()).hexdigest()}}
            baseline_report = {
                "configuration_sha256": structure.submitted_configuration_sha256,
                "three_body_verification": gate, "relative_force_balance_error": 0.,
                "stress_recovery": {"peak_measure": "recovered_nodal"}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "baseline_report.json").write_text(json.dumps(baseline_report))
            (root / "three_body_source.json").write_text(json.dumps(structure.three_body_source))
            options = self.source_options()
            options["growth_options"] = GrowthOptions(.001, 1e-7, 5)
            # Only structure rebuilding is substituted. The real baseline gate
            # consumes explicitly synthetic test evidence, not Elmer validation.
            with patch("run.submitted_structure", return_value=structure):
                result = solve_bound_hydrostatic_growth(
                    configuration, root, root, fields, request, **options, output=root / "growth")
            self.assertEqual(result["accepted_surface_ids"], [0])
            self.assertTrue(result["source_binding_verified"])
            provenance = result["selection_provenance"]
            self.assertEqual(provenance["marker_source_edges"], [[0, 2]])
            self.assertEqual(provenance["crack_start_edge_ids"], [marker["id"]])
            self.assertEqual(provenance["resolved_faces"], {"0": catalog["elements"][0]["face"]})
            self.assertEqual(result["opening_assessment"]["opening_limit_m"], .003)
            final = root / "growth" / result["accepted_event_directory"]
            with (np.load(root / "growth" / "source_mapping.npz") as mapping,
                  np.load(final / "native_inputs.npz") as inputs,
                  np.load(final / "hydraulic_state.npz") as water):
                np.testing.assert_array_equal(mapping["selected_local_cells"],
                                              mapping["local_source_cell_rows"] == 0)
                np.testing.assert_array_equal(inputs["released_face_mask"], [True])
                np.testing.assert_array_equal(water["mechanical_inventory_face_ids"], [0])
            report = json.loads((final / "event_report.json").read_text())
            self.assertGreater(report["hydrostatic_reservoir_work_j"], 0.)
            self.assertFalse(report["engineering_validated"])
            self.assertEqual(provenance["verified_baseline"]["verification"], gate)
            from FEniCSx.painted_event_runner import run_saved_event
            packet = {
                "schema_version": 1, "analysis_kind": "painted_tensile_loss", "job_id": "reduced-test",
                "configuration_digest": hashlib.sha256(json.dumps(
                    structure.three_body_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "selection_id": "saved-test", "status": "saved_selection_only", "solver_ready": False,
                "request": request}
            packet["selection_sha256"] = hashlib.sha256(json.dumps(
                packet, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            saved_path = root / "saved-test.json"
            saved_path.write_text(json.dumps(packet))
            numerical = {
                "maximum_source_cells": 4, "maximum_volume_change": 1e-12,
                "event": asdict(options["event_options"]), "contact": asdict(options["contact_options"]),
                "growth": asdict(options["growth_options"])}
            for mode in ("dry", "common_head_hydrostatic"):
                runner_options = dict(numerical)
                if mode != "dry":
                    runner_options["hydraulics"] = {
                        "closure_tolerance_m": 1e-12, "flow_relative_tolerance": 1e-10}
                runner_output = root / f"isolated_{mode}"
                with self.subTest(mode=mode), patch("FEniCSx.run.submitted_structure", return_value=structure):
                    completed = run_saved_event(
                        configuration, root, root, fields, saved_path, "reduced-test",
                        water_mode=mode, options=runner_options, output=runner_output)
                self.assertEqual(completed["status"], "completed_experimental_not_engineering_validated")
                self.assertFalse(completed["solver_ready"])
                self.assertEqual(completed["selection_id"], "saved-test")
                self.assertEqual(completed["accepted_surface_ids"], [0] if mode != "dry" else [])
                for artifact in completed["accepted_results"].values():
                    self.assertTrue((runner_output / artifact).is_file())
                self.assertEqual((runner_output / "inputs" / "saved-test.json").read_bytes(),
                                 saved_path.read_bytes())
                self.assertEqual(json.loads((runner_output / "manifest.json").read_text()),
                                 json.loads(json.dumps(completed)))
                import meshio
                surface = meshio.read(runner_output / completed["accepted_results"]["crack_surface"])
                np.testing.assert_array_equal(surface.cell_data["released"][0],
                                              [int(mode != "dry")] * 2)
                np.testing.assert_array_equal(surface.cell_data["wet"][0],
                                              [int(mode != "dry")] * 2)
                with np.load(runner_output / "growth" / "source_mapping.npz") as mapping:
                    np.testing.assert_array_equal(
                        surface.point_data["original_parent_node_id"],
                        mapping["parent_node_ids"][surface.point_data["source_node_id"]])
                if mode != "dry":
                    self.assertGreater(surface.point_data["water_pressure_pa"].min(), 0.)
                    self.assertAlmostEqual(surface.point_data["opening_m"].max(),
                                           completed["opening_assessment"]["final_opening_m"])
                else:
                    np.testing.assert_array_equal(surface.point_data["water_pressure_pa"], 0.)
                with self.assertRaises(FileExistsError):
                    run_saved_event(
                        configuration, root, root, fields, saved_path, "reduced-test",
                        water_mode=mode, options=runner_options, output=runner_output)
            for name, changes, message in (
                ("manifest.json", {"status": "running"}, "not completed"),
                ("baseline_report.json", {"relative_force_balance_error": 1.}, "did not pass"),
                ("three_body_source.json", {"sha256": {}}, "hashes have changed"),
            ):
                path = root / name
                original = path.read_bytes()
                path.write_text(json.dumps({**json.loads(original), **changes}))
                try:
                    with self.subTest(evidence=name), patch(
                        "run.submitted_structure", return_value=structure
                    ), patch("hydrostatic_fracture_growth.solve_source_hydrostatic_growth") as solve:
                        with self.assertRaisesRegex(ValueError, message):
                            solve_bound_hydrostatic_growth(
                                configuration, root, root, fields, request,
                                **options, output=root / "invalid")
                        solve.assert_not_called()
                        self.assertFalse((root / "invalid").exists())
                finally:
                    path.write_bytes(original)

    def test_native_source_growth_keeps_canonical_springs_loads_and_reservoir_edges(self):
        from dam_mesh import import_structure
        from dam_elasticity import solve as solve_elastic, water_load_parameters
        from hydrostatic_load import assemble_water_forces
        structure = self.source_fixture()
        original = import_structure(structure, 1e-12)
        space = fem.functionspace(original.domain, ("Lagrange", 1, (3,)))
        expected_water = assemble_water_forces(original, space, *water_load_parameters(original, structure))
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline"
            baseline_path.mkdir()
            baseline, _, baseline_report = solve_elastic(
                original, structure, {"relative_tolerance": 1e-10, "maximum_iterations": 1000}, baseline_path)
            self.assertLess(baseline_report["relative_force_balance_error"], 1e-8)
            np.testing.assert_array_equal(baseline[5:], 0.)
            root = Path(directory) / "growth"
            result = solve_source_hydrostatic_growth(
                structure, [0], [[0, 1]], **self.source_options(), output=root)
            self.assertEqual(result["accepted_surface_ids"], [])
            self.assertFalse(result["source_binding_verified"])
            self.assertFalse(result["solver_ready"])
            with (np.load(root / "source_mapping.npz") as source,
                  np.load(root / result["accepted_event_directory"] / "intact_state.npz") as intact):
                np.testing.assert_allclose(intact["displacement_m"], baseline[source["parent_node_ids"]],
                                           rtol=1e-8, atol=1e-11)
                np.testing.assert_allclose(source["exterior_water_forces_n"].reshape(-1, 3).sum(axis=0),
                                           expected_water.reshape(-1, 3).sum(axis=0), atol=1e-8)
                np.testing.assert_array_equal(source["selected_local_cells"], source["local_source_cell_rows"] == 0)
            trial = root / "trials" / "step_0000_candidate_000000"
            report = json.loads((trial / "event_report.json").read_text())
            self.assertIn("wall_bedrock", report["interface_springs"])
            self.assertGreater(report["hydrostatic_reservoir_work_j"], 0.)
            self.assertIsNone(report["dry_mechanical_potential_j"])
            reservoir = result["reservoir_problem"]
            self.assertEqual(reservoir["head_m"], 4.)
            self.assertEqual(reservoir["boundaries"], [{
                "name": "exterior_tag_2", "head_m": 4., "edges": [[0, 2], [1, 2]]}])
            self.assertLess(report["iterations"][-1]["scaled_free_residual"], 1e-8)
            low_options = self.source_options()
            low_options["growth_options"] = GrowthOptions(.001, 1e-7, 5)
            grown_root = Path(directory) / "grown"
            grown = solve_source_hydrostatic_growth(
                structure, [0], [[0, 1]], **low_options, output=grown_root)
            self.assertEqual(grown["accepted_surface_ids"], [0])
            self.assertTrue((grown_root / grown["accepted_event_directory"] / "hydraulic_state.npz").is_file())
            with (np.load(grown_root / "source_mapping.npz") as source,
                  np.load(root / "source_mapping.npz") as prior_source):
                np.testing.assert_array_equal(source["local_source_cell_rows"], prior_source["local_source_cell_rows"])
            iterative_options = dict(low_options)
            iterative_options["contact_options"] = ContactOptions(
                1e-12, 1e-7, 1e-8, 100, linear_solver="minres", linear_relative_tolerance=1e-13)
            iterative_root = Path(directory) / "iterative"
            iterative = solve_source_hydrostatic_growth(
                structure, [0], [[0, 1]], **iterative_options, output=iterative_root)
            self.assertEqual(iterative["accepted_surface_ids"], grown["accepted_surface_ids"])
            direct_event = grown_root / grown["accepted_event_directory"]
            iterative_event = iterative_root / iterative["accepted_event_directory"]
            with (np.load(direct_event / "post_event_state.npz") as direct_state,
                  np.load(iterative_event / "post_event_state.npz") as iterative_state):
                # Released frictionless specimen permits zero-energy modes;
                # do not require the two solvers to select the same gauge.
                np.testing.assert_allclose(iterative_state["support_reaction_n"], direct_state["support_reaction_n"],
                                           rtol=1e-8, atol=1e-5)
                np.testing.assert_allclose(iterative_state["stress_pa"], direct_state["stress_pa"],
                                           rtol=1e-8, atol=1e-5)
                np.testing.assert_allclose(iterative_state["signed_gaps_m"], direct_state["signed_gaps_m"],
                                           rtol=1e-8, atol=1e-11)
                np.testing.assert_allclose(iterative_state["energy_j_m3"], direct_state["energy_j_m3"],
                                           rtol=1e-8, atol=1e-7)
            with (np.load(direct_event / "intact_state.npz") as direct_intact,
                  np.load(iterative_event / "intact_state.npz") as iterative_intact):
                np.testing.assert_allclose(iterative_intact["displacement_m"], direct_intact["displacement_m"],
                                           rtol=1e-8, atol=1e-11)
            iterative_report = json.loads((iterative_event / "event_report.json").read_text())
            self.assertFalse(iterative_report["iterations"][-1]["contact"]["displacement_uniqueness_verified"])
            self.assertFalse(iterative_report["iterations"][-1]["contact"]["stability_verified"])
            self.assertAlmostEqual(iterative["final_potential_j"], grown["final_potential_j"], places=7)


if __name__ == "__main__":
    unittest.main()
