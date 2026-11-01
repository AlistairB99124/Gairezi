"""Automatic cracking runner: native adapter, bounded region and screening bound.

Run from FEniCSx with PYTHONPATH=.. (bare-module imports, as the native solvers use).
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from automatic_cracking import (
    ACCEPTED_OUTCOMES, NativeNetworkProblem, candidate_region, cell_max_principal_stress, crack_verdict,
    maximum_overload_factor,
    interior_wall_facet_normal_stress, isotropic_elasticity, opening_assessment, prepare_and_run,
    region_candidate_faces, region_cell_stresses, run_network_on_structure, screening_bound,
    screening_facet_bound, stress_comparison,
)
from automatic_screening import _wall_faces, config_digest, facet_check
from cohesive_contact import CohesiveOptions
from cohesive_mode_i import LinearModeICohesiveLaw
from cohesive_network import CohesiveNetwork, NetworkOptions
from test_three_body import column


def wet_column():
    from dam_mesh import import_structure

    structure = column()
    structure.wall_connection = "single_monolithic_body"
    boundaries = []
    for tag, nodes in structure.boundaries:
        xyz = structure.nodes[list(nodes)]
        if tag == 7:
            if np.all(xyz[:, 0] == 1.):
                tag = 2
            elif np.all(xyz[:, 0] == 0.):
                tag = 3
            elif np.all(xyz[:, 2] == 2.):
                tag = 4
        boundaries.append((tag, nodes))
    structure.boundaries = boundaries
    structure.loads.upstream_surface_z_m = 3.
    structure.loads.tailwater_head_m = 1.
    original = import_structure(structure, 1e-12)
    return structure, original


def bending_column(divisions=(2, 1, 3)):
    """Weightless refined wall under upstream reservoir head only: tensile facets near the heel."""
    from dam_mesh import FACES, import_structure

    structure = column()
    structure.wall_connection = "single_monolithic_body"
    nx, ny, nz = divisions
    nodes = [tuple(row) for row in structure.nodes[:8]]
    cells, regions = [structure.cells[0]], [2]
    boundaries = [(tag, face) for (tag, face) in structure.boundaries if max(face) < 8]
    ids = {}
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                ids[i, j, k] = len(nodes)
                nodes.append((i / nx, j / ny, 1. + k / nz))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                hexahedron = tuple(ids[i + a, j + b, k + c] for c in (0, 1) for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)))
                cells.append((5, hexahedron))
                regions.append(1)
                for face in FACES[5]:
                    xyz = np.asarray([nodes[hexahedron[n]] for n in face])
                    exterior = [np.allclose(xyz[:, axis], value) for axis, value in
                                ((2, 1.), (2, 2.), (0, 1.), (0, 0.), (1, 0.), (1, 1.))]
                    if not any(exterior):
                        continue
                    tag = 11 if exterior[0] else 2 if exterior[2] else 7
                    boundaries.append((tag, tuple(hexahedron[n] for n in face)))
    structure.nodes = np.asarray(nodes, dtype=float)
    structure.cells, structure.region_ids, structure.boundaries = cells, regions, boundaries
    structure.loads.upstream_surface_z_m = 3.
    structure.three_body_config["material"]["concrete"]["density"] = 0.
    return structure, import_structure(structure, 1e-12)


def write_archives(directory, stress):
    """Two wall tetrahedra sharing the z = 0 facet, as screening intact archives."""
    coordinates = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]])
    intact = Path(directory) / "intact"
    intact.mkdir(parents=True)
    np.savez(intact / "intact_cell_stress.npz", cell_stress_pa=stress, cell_region_tags=np.array([1, 1]),
             source_cell_rows=np.array([10, 11]))
    np.savez(intact / "native_inputs.npz", coordinates_m=coordinates,
             geometry_cells=np.array([[0, 1, 2, 3], [0, 1, 2, 4]]))
    return {name: hashlib.sha256((intact / name).read_bytes()).hexdigest()
            for name in ("intact_cell_stress.npz", "native_inputs.npz")}


class RegionTests(unittest.TestCase):
    def test_displacement_gauge_excludes_non_wall_nodes(self):
        from types import SimpleNamespace
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1]])
        split = SimpleNamespace(coordinates_m=points, cells=np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
                                region_ids=np.array([1, 3]))
        material = {"concrete": {"youngs_modulus": 1e7, "poisson_ratio": 0.},
                    "bedrock": {"youngs_modulus": 3e7, "poisson_ratio": 0.}}
        displacement = np.zeros((8, 3))
        displacement[:4, 0] = .025
        displacement[4:, 0] = 9.
        report, _, _ = stress_comparison(split, material, displacement, displacement)
        for name in ("intact", "cracked"):
            self.assertEqual(report[name]["maximum_wall_displacement_m"], .025)
            self.assertEqual(report[name]["maximum_displacement_m"], 9.)

    def test_rings_grow_through_shared_vertices_and_stay_in_the_wall(self):
        tetrahedra = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
        regions = np.array([1, 1, 1, 2])
        self.assertEqual(candidate_region(tetrahedra, regions, [0], 0).tolist(), [True, False, False, False])
        self.assertEqual(candidate_region(tetrahedra, regions, [0], 1).tolist(), [True, True, False, False])
        self.assertEqual(candidate_region(tetrahedra, regions, [0], 5).tolist(), [True, True, True, False])
        faces = region_candidate_faces(tetrahedra, regions, np.array([True, True, True, True]))
        self.assertEqual(faces.tolist(), [[1, 2, 3]])

    def test_cell_stress_is_exact_for_uniaxial_strain(self):
        coordinates = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        displacement = np.zeros((4, 3))
        displacement[:, 2] = 1e-3 * coordinates[:, 2]
        stress = cell_max_principal_stress(coordinates, np.array([[0, 1, 2, 3]]), displacement,
                                           isotropic_elasticity(1e7, 0.))
        np.testing.assert_allclose(stress, [1e4])

    def test_region_cell_stress_uses_each_body_material(self):
        coordinates = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        displacement = np.zeros((4, 3))
        displacement[:, 2] = 1e-3 * coordinates[:, 2]
        material = {"concrete": {"youngs_modulus": 1e7, "poisson_ratio": 0.},
                    "bedrock": {"youngs_modulus": 3e7, "poisson_ratio": 0.}}
        cells = np.array([[0, 1, 2, 3]] * 3)
        stress = region_cell_stresses(coordinates, cells, np.array([1, 2, 3]), displacement, material)
        np.testing.assert_allclose(stress[:, 2, 2], [1e4, 3e4, 1e4])
        with self.assertRaises(ValueError):
            region_cell_stresses(coordinates, cells[:1], np.array([9]), displacement, material)

    def test_screening_bound_uses_the_tolerance_conservatively(self):
        self.assertTrue(screening_bound({"peak_principal_stress_pa": 1.0e6}, 2e6, 1e-3))
        self.assertFalse(screening_bound({"peak_principal_stress_pa": 1.999e6}, 2e6, 1e-3))

    def test_facet_stress_takes_the_larger_owner_on_the_shared_normal(self):
        coordinates = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]])
        tetrahedra = np.array([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4]])
        regions = np.array([1, 1, 2])
        stress = np.zeros((3, 3, 3))
        stress[0] = np.diag([5e6, 0, 3e6])
        stress[1] = np.diag([5e6, 0, 1e6])
        stress[2] = np.diag([0, 0, 9e9])
        pairs, normal = interior_wall_facet_normal_stress(coordinates, tetrahedra, regions, stress)
        self.assertEqual(sorted(pairs[0].tolist()), [0, 1])
        np.testing.assert_allclose(normal, [3e6])
        _, one_sided, critical, check = facet_check(coordinates, tetrahedra, regions, stress, 2e6, 1e-3)
        np.testing.assert_allclose(one_sided, [3e6])
        self.assertEqual(critical.tolist(), [True])
        self.assertTrue(check["initiates"])
        self.assertEqual(check["critical_facets"], 1)
        self.assertAlmostEqual(check["peak_two_cell_average_facet_normal_stress_pa"], 2e6)
        self.assertAlmostEqual(check["peak_ratio_to_strength"], 1.5)

    def test_facet_bound_ignores_principal_peaks_parallel_to_shared_facets(self):
        stress = np.stack([np.diag([5e6, 0, 1e6])] * 2)
        with tempfile.TemporaryDirectory() as directory:
            hashes = write_archives(directory, stress)
            bound = screening_facet_bound(Path(directory), {"native_archive_sha256": hashes}, 2e6, 1e-3)
            self.assertTrue(bound["bounds_initiation"])
            self.assertEqual(bound["interior_wall_facets"], 1)
            self.assertAlmostEqual(bound["peak_facet_normal_stress_pa"], 1e6)
            self.assertEqual(bound["critical_source_rows"], [])
            self.assertIsNone(screening_facet_bound(Path(directory), {}, 2e6, 1e-3))
            with self.assertRaises(ValueError):
                screening_facet_bound(Path(directory), {"native_archive_sha256": dict(hashes, **{
                    "native_inputs.npz": "0" * 64})}, 2e6, 1e-3)

    def test_facet_bound_seeds_the_cells_adjacent_to_critical_facets(self):
        stress = np.stack([np.diag([0, 0, 3e6])] * 2)
        with tempfile.TemporaryDirectory() as directory:
            hashes = write_archives(directory, stress)
            bound = screening_facet_bound(Path(directory), {"native_archive_sha256": hashes}, 2e6, 1e-3)
            self.assertFalse(bound["bounds_initiation"])
            self.assertEqual(bound["critical_source_rows"], [10, 11])


class NativeAdapterTests(unittest.TestCase):
    def test_network_intact_state_matches_native_elastic_baseline(self):
        from dam_elasticity import solve as solve_elastic
        from dam_mesh import prepare_wall_candidates

        structure, original = wet_column()
        selected = original.regions == 1
        candidates = region_candidate_faces(original.tetrahedra, original.regions, selected)
        prepared = prepare_wall_candidates(structure, None, 1e-12, candidate_triangles=candidates,
                                           original=original)
        problem = NativeNetworkProblem(prepared, structure)
        options = NetworkOptions(CohesiveOptions(1e-12, 1e-6, 1e-10, 1e-6, "tied"))
        network = CohesiveNetwork.from_split(
            prepared.split, elasticity=isotropic_elasticity(1e7, 0.), law=LinearModeICohesiveLaw(1e12, 400.),
            options=options, problem_at=problem)
        result = network.run(0., 1.)
        self.assertEqual(result.status, "no_damage_initiated")
        with tempfile.TemporaryDirectory() as directory:
            baseline, _, _ = solve_elastic(original, structure,
                                           {"relative_tolerance": 1e-10, "maximum_iterations": 1000},
                                           Path(directory))
        np.testing.assert_allclose(result.displacement_m.reshape(-1, 3),
                                   np.asarray(baseline).reshape(-1, 3)[prepared.parent_node_ids],
                                   rtol=1e-7, atol=1e-12)


class RunnerTests(unittest.TestCase):
    def screening(self, original, peak, strength):
        wall = np.flatnonzero(original.regions == 1)
        return {"peak_principal_stress_pa": peak, "peak_source_row": int(original.parent_ids[wall[0]]),
                "overstressed_source_rows": sorted(set(map(int, original.parent_ids[wall]))),
                "tensile_strength_pa": strength, "relative_tolerance": 1e-3}

    def test_compressed_facets_do_not_initiate_even_with_low_strength(self):
        structure, original = wet_column()
        result, _, region = run_network_on_structure(
            structure, self.screening(original, 1e5, 2e3), gf_j_m2=1., tensile_strength_pa=2e3,
            relative_tolerance=1e-3, maximum_volume_change=1e-12, rings=1, progress=lambda _: None)
        self.assertEqual(result.status, "no_damage_initiated")
        self.assertEqual(region["status"], "region_sufficient")

    def test_bending_wall_initiates_softens_and_reports_growth_limit_honestly(self):
        # Hex-derived tet meshes force wedge patches the single-normal pair model cannot represent;
        # the runner must report that rather than claim an accepted equilibrium.
        structure, original = bending_column()
        strength = 1e3
        result, prepared, region = run_network_on_structure(
            structure, self.screening(original, 1e5, strength), gf_j_m2=1., tensile_strength_pa=strength,
            relative_tolerance=1e-3, maximum_volume_change=1e-12, rings=1, progress=lambda _: None)
        self.assertIn(result.status, ("accepted_equilibrium", "unsupported_geometry", "unsupported_hydraulics"),
                      result.report.get("message"))
        self.assertTrue(result.initiated.any())
        self.assertLess(result.steps[0]["load_factor"], 1.)
        self.assertGreater(result.steps[-1]["initiated_area_m2"], 0.)
        self.assertLessEqual(result.steps[-1]["opened_area_m2"], result.steps[-1]["initiated_area_m2"] + 1e-12)
        self.assertLess(result.load_factor, 1.)
        self.assertLessEqual(np.max(result.face_normal_stress_pa[~result.initiated]), strength * (1 + 1e-3))
        if result.status == "accepted_equilibrium":
            self.assertGreater(result.report["cohesive_dissipated_energy_j"], 0.)
            self.assertGreater(result.steps[-1]["wet_faces"], 0)
        else:
            self.assertNotIn(result.status, ACCEPTED_OUTCOMES)
            self.assertTrue(any(step["event"] == "initiation" for step in result.steps))

    def test_confined_run_opens_only_facets_between_selected_cells(self):
        structure, original = bending_column()
        strength = 1e3
        faces, owners = _wall_faces(np.asarray(original.tetrahedra), np.asarray(original.regions))
        pair = owners[6]  # a seed pair whose confined solve reaches an accepted equilibrium
        rows = sorted(int(original.parent_ids[cell]) for cell in pair)
        result, prepared, region = run_network_on_structure(
            structure, self.screening(original, 1e5, strength), gf_j_m2=1., tensile_strength_pa=strength,
            relative_tolerance=1e-3, maximum_volume_change=1e-12, rings=2, progress=lambda _: None,
            seed_rows=rows, confined=True)
        seeds = np.isin(original.parent_ids, rows) & (np.asarray(original.regions) == 1)
        self.assertEqual(len(region["attempts"]), 1)
        self.assertEqual(region["attempts"][0]["rings"], 0)
        self.assertEqual(region["attempts"][0]["region_cells"], int(seeds.sum()))
        expected = region_candidate_faces(original.tetrahedra, original.regions, seeds)
        self.assertEqual(region["attempts"][0]["candidate_faces"], len(expected))
        self.assertEqual(len(result.initiated), len(expected))
        self.assertEqual(result.status, "accepted_equilibrium")
        # Outside facets in this low-strength wall reach f_t: reported, not grown into.
        self.assertEqual(region["status"], "region_confined")
        self.assertGreater(region["outside_region_critical_facets"], 0)
        # The bonded reference is the same split mesh with every candidate tied at the same load.
        split = prepared.split
        self.assertEqual(result.intact_displacement_m.shape, result.displacement_m.shape)
        report, intact, cracked = stress_comparison(split, structure.three_body_config["material"],
                                                    result.intact_displacement_m, result.displacement_m)
        self.assertEqual(intact.shape, (len(split.cells), 3, 3))
        self.assertEqual(report["mesh_cells"], len(split.cells))
        self.assertGreater(report["intact"]["maximum_principal_tensile_stress_pa"], strength)
        self.assertLess(report["intact"]["maximum_principal_compressive_stress_pa"], 0.)
        same_cell = report["cracked"]["cracked_at_intact_tensile_peak"]
        self.assertEqual(same_cell["cell"], report["intact"]["tensile_peak_cell"])
        np.testing.assert_allclose(same_cell["principal_stresses_pa"], np.linalg.eigvalsh(cracked[same_cell["cell"]]))
        assessment = opening_assessment(result, LinearModeICohesiveLaw(strength, 1.), report)
        self.assertEqual(assessment["initiated_faces"], int(result.initiated.sum()))
        opened = int(np.isin(result.face_status, ("softening", "separated")).sum())
        self.assertEqual(assessment["opened_faces"], opened)
        self.assertEqual(assessment["cracked_equals_intact"], report["relative_displacement_change"] <= 1e-6)
        if opened:
            self.assertGreater(report["maximum_displacement_change_m"], 0.)

    def test_overload_search_finds_the_first_opening_beyond_the_design_load(self):
        # At f_t 20 kPa one facet releases at ~0.79 but stays closed at the design load.
        structure, original = bending_column()
        strength = 2e4
        faces, owners = _wall_faces(np.asarray(original.tetrahedra), np.asarray(original.regions))
        rows = sorted(int(original.parent_ids[cell]) for cell in owners[6])
        result, _, region = run_network_on_structure(
            structure, self.screening(original, 1e5, strength), gf_j_m2=1., tensile_strength_pa=strength,
            relative_tolerance=1e-3, maximum_volume_change=1e-12, rings=2, progress=lambda _: None,
            seed_rows=rows, confined=True, overload_factor=1.5)
        design, overload = region["design_load"], region["overload"]
        self.assertEqual(design["load_factor"], 1.)
        self.assertGreater(design["initiated_faces"], 0)
        self.assertEqual(design["opened_faces"], 0)
        self.assertLess(design["peak_stress_to_strength_ratio"], 1.)
        self.assertGreater(design["linear_first_opening_load_factor_estimate"], 1.)
        self.assertEqual(result.status, "accepted_equilibrium")
        self.assertGreater(result.load_factor, 1.)
        self.assertEqual(overload["reached_load_factor"], result.load_factor)
        low, high = overload["first_opening_bracket"]
        self.assertTrue(1. <= low < high == overload["first_opening_load_factor"] <= result.load_factor)
        self.assertGreater(overload["opened_area_m2"], 0.)
        if overload["stopped"] is None:
            self.assertEqual(result.load_factor, 1.5)
        phases = [step["phase"] for step in result.steps]
        self.assertEqual(phases, sorted(phases))  # "design" steps precede "overload" steps
        self.assertEqual(max(step["load_factor"] for step in result.steps if step["phase"] == "design"), 1.)
        self.assertEqual(result.intact_displacement_m.shape, result.displacement_m.shape)
        verdict = crack_verdict(design, overload, strength)
        self.assertIn("No crack opens at the design load", verdict)
        self.assertIn("first open between", verdict)
        # Without the overload search the design state is returned with a linear estimate.
        design_only, _, plain = run_network_on_structure(
            structure, self.screening(original, 1e5, strength), gf_j_m2=1., tensile_strength_pa=strength,
            relative_tolerance=1e-3, maximum_volume_change=1e-12, rings=2, progress=lambda _: None,
            seed_rows=rows, confined=True)
        self.assertEqual(design_only.load_factor, 1.)
        self.assertNotIn("overload", plain)
        self.assertIn("Linear estimate", crack_verdict(plain["design_load"], None, strength))

    def test_maximum_overload_factor_defaults_to_the_design_load(self):
        self.assertEqual(maximum_overload_factor({}), 1.)
        self.assertEqual(maximum_overload_factor({"damage_selection": {"max_damage_count": 4}}), 1.)
        self.assertEqual(maximum_overload_factor({"damage_selection": {"maximum_overload_factor": 2}}), 2.)
        for value in (0.5, 11, True):
            with self.assertRaises(ValueError):
                maximum_overload_factor({"damage_selection": {"maximum_overload_factor": value}})

    def test_runner_writes_bound_outcome_without_mesh_work(self):
        config = {"material": {"concrete": {"tensile_strength_pa": 2e6, "cohesive_fracture_energy_j_m2": 400.}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config))
            screening_dir = root / "screening"
            screening_dir.mkdir()
            screening = {"source_job_id": "job", "peak_principal_stress_pa": 1.05e6, "overstressed_cells": 0,
                         "tensile_strength_pa": 2e6, "relative_tolerance": 1e-3}
            (screening_dir / "screening.json").write_text(json.dumps(screening))
            import hashlib
            (screening_dir / "manifest.json").write_text(json.dumps({
                "status": "completed_automatic_screening", "source_job_id": "job",
                "config_digest": config_digest(config),
                "screening_sha256": hashlib.sha256((screening_dir / "screening.json").read_bytes()).hexdigest()}))
            summary = prepare_and_run(root / "config.json", root / "missing-model", screening_dir, root / "out",
                                      job_id="job", maximum_volume_change=1e-12)
            self.assertEqual(summary["outcome"], "no_damage_initiated")
            self.assertTrue(summary["accepted"])
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed_automatic_cracking")
            self.assertFalse(manifest["engineering_validated"])
            with self.assertRaises(ValueError):
                prepare_and_run(root / "config.json", root / "missing-model", screening_dir, root / "other",
                                job_id="different-job", maximum_volume_change=1e-12)
            self.assertEqual(json.loads((root / "other" / "manifest.json").read_text())["status"],
                             "failed_automatic_cracking")

    def test_runner_uses_the_intact_facet_bound_before_mesh_work(self):
        config = {"material": {"concrete": {"tensile_strength_pa": 2e6, "cohesive_fracture_energy_j_m2": 130.}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config))
            screening_dir = root / "screening"
            hashes = write_archives(screening_dir, np.stack([np.diag([5e6, 0, 1e6])] * 2))
            screening = {"source_job_id": "job", "peak_principal_stress_pa": 5e6, "overstressed_cells": 2,
                         "tensile_strength_pa": 2e6, "relative_tolerance": 1e-3, "maximum_volume_change": 1e-12,
                         "native_archive_sha256": hashes}
            (screening_dir / "screening.json").write_text(json.dumps(screening))
            import hashlib
            (screening_dir / "manifest.json").write_text(json.dumps({
                "status": "completed_automatic_screening", "source_job_id": "job",
                "config_digest": config_digest(config),
                "screening_sha256": hashlib.sha256((screening_dir / "screening.json").read_bytes()).hexdigest()}))
            summary = prepare_and_run(root / "config.json", root / "missing-model", screening_dir, root / "out",
                                      job_id="job", maximum_volume_change=1e-12)
            self.assertEqual(summary["outcome"], "no_damage_initiated")
            self.assertTrue(summary["accepted"])
            self.assertIn("intact facet bound", summary["outcome_basis"])
            self.assertAlmostEqual(summary["intact_facet_screening"]["peak_ratio_to_strength"], .5)


if __name__ == "__main__":
    unittest.main()
