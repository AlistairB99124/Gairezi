"""Small three-body mechanics/refinement benchmarks, independent of Elmer."""
from pathlib import Path
from dataclasses import replace
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from dam_mesh import import_structure
from interface_springs import intersection_mass, projected_spring_matrix


def column():
    from dam_mesh import FACES
    nodes, cells, regions, boundaries = [], [], [], []
    for region, bottom in ((3, 0.), (1, 1.), (2, 2.)):
        offset = len(nodes)
        nodes.extend([(x, y, z + bottom) for x, y, z in (
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))])
        cell = tuple(range(offset, offset + 8))
        cells.append((5, cell))
        regions.append(region)
        for index, face in enumerate(FACES[5]):
            tag = {3: (10, 12), 1: (11, 9), 2: (8, 7)}[region][index] if index < 2 else 7
            boundaries.append((tag, tuple(cell[n] for n in face)))
    return SimpleNamespace(
        nodes=np.asarray(nodes, dtype=float), cells=cells, region_ids=regions, boundaries=boundaries,
        material=SimpleNamespace(youngs_modulus_pa=1e7, poissons_ratio=0., density_kg_m3=1000.,
                                 tensile_strength_pa=2e6),
        loads=SimpleNamespace(gravity_z_m_s2=-10., water_density_kg_m3=1000.,
                              tailwater_head_m=0., overflow_head_m=0.),
        concrete_minimum_z=1., three_body_source={},
        three_body_config={
            "material": {
                "concrete": {"density": 1000., "youngs_modulus": 1e7, "poisson_ratio": 0.},
                "bedrock": {"density": 2000., "youngs_modulus": 2e7, "poisson_ratio": 0.},
            },
            "geometry": {name: {"support": {"spring_n_per_m3": {"x": 1e6, "y": 2e6, "z": k}}}
                         for name, k in (("plinth", 1e6), ("bedrock", 2e6))},
        },
    )


def bonded_column():
    structure = column()
    structure.three_body_config["geometry"]["bedrock"]["support"]["model"] = "bonded"
    seat = dict(zip(range(8, 12), range(4, 8)))
    cells = [(kind, tuple(seat.get(n, n) for n in cell)) for kind, cell in structure.cells]
    boundaries = [(tag, tuple(seat.get(n, n) for n in face))
                  for tag, face in structure.boundaries if tag not in (11, 12)]
    used = sorted({n for _, cell in cells for n in cell})
    mapping = {node: i for i, node in enumerate(used)}
    structure.nodes = structure.nodes[used]
    structure.cells = [(kind, tuple(mapping[n] for n in cell)) for kind, cell in cells]
    structure.boundaries = [(tag, tuple(mapping[n] for n in face)) for tag, face in boundaries]
    return structure


def roller_column(k=1e12):
    """Bonded column whose rock cube rests on normal roller springs on all five outer faces."""
    structure = bonded_column()
    rock = set(structure.cells[structure.region_ids.index(3)][1])
    planes = {13: (0, 0.), 14: (0, 1.), 15: (1, 0.), 16: (1, 1.)}
    boundaries = []
    for tag, face in structure.boundaries:
        if tag == 7 and set(face) <= rock:
            points = structure.nodes[list(face)]
            tag = next(t for t, (axis, value) in planes.items() if np.allclose(points[:, axis], value))
        boundaries.append((tag, face))
    structure.boundaries = boundaries
    bedrock = structure.three_body_config["material"]["bedrock"]
    bedrock.update(density=0., poisson_ratio=.25)
    normals = {10: ("bottom", "z", (0, 0, -1)), 13: ("bank_x_min", "x", (-1, 0, 0)),
               14: ("bank_x_max", "x", (1, 0, 0)), 15: ("stream_y_min", "y", (0, -1, 0)),
               16: ("stream_y_max", "y", (0, 1, 0))}
    structure.rock_support = {"model": "rollers", "faces": {
        str(tag): {"name": name, "local_axis": axis, "outward_normal": list(map(float, n)),
                   "normal_stiffness_n_per_m3": k}
        for tag, (name, axis, n) in normals.items()
    }}
    return structure


class ThreeBodyTests(unittest.TestCase):
    def test_rock_rollers_give_confined_compression_without_lateral_rock_motion(self):
        from dam_elasticity import solve
        structure = roller_column()
        imported = import_structure(structure, 1e-12)
        with tempfile.TemporaryDirectory() as directory:
            displacement, _, report = solve(imported, structure,
                {"relative_tolerance": 1e-10, "maximum_iterations": 1000}, Path(directory))
        reactions = report["ground_spring_reactions_n"]
        self.assertEqual(report["rock_outer_support"]["model"], "rollers")
        self.assertIn("roller", report["wall_plinth_model"])
        # Concrete 2 m x 1000 kg/m^3 x 10 m/s^2 over 1 m^2; massless rock.
        vertical, poisson = 20000., .25
        lateral = poisson / (1 - poisson) * vertical
        expected = {"rock_bottom": [0, 0, vertical],
                    "rock_bank_x_min": [lateral, 0, 0], "rock_bank_x_max": [-lateral, 0, 0],
                    "rock_stream_y_min": [0, lateral, 0], "rock_stream_y_max": [0, -lateral, 0]}
        self.assertEqual(set(reactions), set(expected))
        # Exact: vertical support and equal/opposite side pairs. The analytic K0 lateral
        # value is approximate because P1 consistent gravity loads on the tet-split
        # concrete do not make the rock-top traction exactly uniform.
        np.testing.assert_allclose(reactions["rock_bottom"], expected["rock_bottom"], rtol=1e-6, atol=1e-3)
        for low, high in (("rock_bank_x_min", "rock_bank_x_max"), ("rock_stream_y_min", "rock_stream_y_max")):
            np.testing.assert_allclose(reactions[low], -np.asarray(reactions[high]), rtol=1e-6, atol=1e-3)
        for name, value in expected.items():
            np.testing.assert_allclose(reactions[name], value, rtol=5e-2, atol=1e-2, err_msg=name)
        self.assertLess(report["relative_force_balance_error"], 1e-8)
        rock = sorted({n for (_, cell), region in zip(structure.cells, structure.region_ids)
                       if region == 3 for n in cell})
        np.testing.assert_allclose(displacement[rock, :2], 0., atol=1e-7)
        seat = rock[np.argmax(structure.nodes[rock, 2])]
        young = structure.three_body_config["material"]["bedrock"]["youngs_modulus"]
        confined = young * (1 - poisson) / ((1 + poisson) * (1 - 2 * poisson))
        self.assertAlmostEqual(displacement[seat, 2] / (-vertical / confined), 1., delta=5e-2)

    def test_bonded_seat_moves_with_rock_without_slip_and_retains_wall_spring(self):
        from dam_elasticity import solve
        structure = bonded_column()
        imported = import_structure(structure, 1e-12)
        with tempfile.TemporaryDirectory() as directory:
            displacement, _, report = solve(imported, structure,
                {"relative_tolerance": 1e-10, "maximum_iterations": 1000}, Path(directory))
        seat = np.flatnonzero(np.isclose(structure.nodes[:, 2], 1.))
        plinth_nodes = {n for (_, cell), region in zip(structure.cells, structure.region_ids)
                        if region == 1 for n in cell}
        rock_nodes = {n for (_, cell), region in zip(structure.cells, structure.region_ids)
                      if region == 3 for n in cell}
        self.assertEqual(plinth_nodes & rock_nodes, set(seat))
        self.assertTrue(np.all(displacement[seat, 2] < 0))
        np.testing.assert_allclose(report["ground_spring_reactions_n"]["rock_bottom"],
                                   [0, 0, 40000], atol=.001)
        self.assertLess(report["relative_force_balance_error"], 1e-8)
        self.assertEqual(report["plinth_bedrock_connection"], "bonded")
        self.assertEqual(report["interface_springs"]["plinth_bedrock"]["connection"], "bonded")
        self.assertEqual(report["interface_springs"]["wall_plinth"]["spring_n_per_m3"]["z"], 1e6)
        wall_base = {n for tag, face in structure.boundaries if tag == 8 for n in face}
        plinth_top = {n for tag, face in structure.boundaries if tag == 9 for n in face}
        self.assertFalse(wall_base & plinth_top)
        from dam_mesh import triangles
        def average(tag):
            faces = np.asarray([tri for value, face in structure.boundaries
                                if value == tag for tri in triangles(face)])
            points = structure.nodes[faces]
            area = np.linalg.norm(np.cross(points[:, 1] - points[:, 0],
                                            points[:, 2] - points[:, 0]), axis=1) / 2
            return np.average(displacement[faces, 2].mean(axis=1), weights=area)
        self.assertAlmostEqual(
            average(8) - average(9),
            -.01, places=9,
        )

    def test_canonical_fracture_cannot_opt_into_under_resolved_target(self):
        from crack_field import PhaseFieldSolverOptions
        from three_body_model import require_fracture_resolution

        options = PhaseFieldSolverOptions()
        for target in (.9, 1.99, float("nan")):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "at least 2"):
                require_fracture_resolution(replace(options, minimum_elements_per_length_scale=target))
        for target in (2., 3.):
            require_fracture_resolution(replace(options, minimum_elements_per_length_scale=target))

    def test_nonwall_stiffness_is_not_degraded_or_scaled_by_residual_stiffness(self):
        from dolfinx import fem
        from three_body_model import materials
        from compression_preserving import stress as amor_stress
        from phase_field_solver import undamaged_stress
        imported = import_structure(column(), 1e-12)
        coefficients = materials(imported, column())
        space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
        displacement = fem.Function(space)
        displacement.interpolate(lambda x: np.asarray((.01*x[0], .02*x[1], .03*x[2])))
        scalar = fem.functionspace(imported.domain, ("Lagrange", 1))
        damage = fem.Function(scalar)
        damage.x.array[:] = .6
        mask, lam, mu, bulk = (coefficients[key] for key in ("damage_mask", "lam", "mu", "bulk"))
        factor = mask * ((1 - damage)**2 + .2) + 1 - mask
        tensor_space = fem.functionspace(imported.domain, ("DG", 0, (3, 3)))
        expressions = (factor * undamaged_stress(displacement, lam, mu),
                       amor_stress(displacement, damage, bulk, mu, .2, damage_mask=mask))
        for expression in expressions:
            tensor = fem.Function(tensor_space)
            tensor.interpolate(fem.Expression(expression, tensor_space.element.interpolation_points))
            for tag, young in ((1, 1e7), (2, 1e7), (3, 2e7)):
                dofs = tensor_space.dofmap.list[imported.cell_tags.find(tag), 0]
                expected = young * np.diag((.01, .02, .03)) * (.36 if tag == 2 else 1.)
                np.testing.assert_allclose(tensor.x.array.reshape(-1, 3, 3)[dofs],
                                           np.broadcast_to(expected, (len(dofs), 3, 3)), atol=1e-8)

    def test_automatic_seed_ignores_larger_plinth_and_bedrock_peaks(self):
        from unittest.mock import patch
        from elmer_baseline import import_baseline
        structure = column()
        regions = np.asarray([4]*8 + [1]*8 + [2]*8)
        stress = np.zeros((24, 6))
        stress[:, 0] = np.repeat((100e6, 10e6, 1e6), 8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "result.pvtu"
            reference.write_text('<VTKFile><PUnstructuredGrid><Piece Source="piece.vtu"/></PUnstructuredGrid></VTKFile>')
            (root / "piece.vtu").write_text("fixture")
            config = root / "config.json"
            config.write_text("{}")
            with patch("elmer_baseline.read_reference", return_value=(np.zeros((24, 3)), stress, regions)):
                baseline = import_baseline(structure, reference, config, root)
            self.assertEqual(baseline["max_principal_stress"]["region_id"], 2)
            self.assertEqual(baseline["max_principal_stress"]["value_pa"], 1e6)

    def test_fracture_rejects_missing_failed_stale_and_nonfinite_baseline_evidence(self):
        from three_body_model import require_verified_baseline
        structure = column()
        structure.submitted_configuration_sha256 = "current-config"
        structure.three_body_source = {"sha256": {"three_body.msh": "current-mesh"}}
        with self.assertRaisesRegex(ValueError, "requires --verified-baseline"):
            require_verified_baseline(structure, None)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"status": "completed_not_engineering_validated",
                        "scenario_name": "Baseline_NoPhase", "configuration_sha256": "current-config"}
            report = {"configuration_sha256": "current-config",
                      "relative_force_balance_error": 1e-10,
                      "stress_recovery": {"peak_measure": "recovered_nodal"},
                      "three_body_verification": {"passed": True, "relative_l2_displacement_difference": 1e-8,
                                                  "relative_nodal_stress_difference": 1e-8}}
            source = {"sha256": {"three_body.msh": "current-mesh"}}
            def save():
                for name, value in (("manifest", manifest), ("baseline_report", report),
                                    ("three_body_source", source)):
                    (root / f"{name}.json").write_text(json.dumps(value))
            save()
            self.assertEqual(require_verified_baseline(structure, root)["source_sha256"], source["sha256"])
            for document, key, invalid in (
                (manifest, "status", "failed"), (manifest, "configuration_sha256", "old-config"),
                (source, "sha256", {"three_body.msh": "modified-mesh"}),
                (report, "relative_force_balance_error", float("nan")),
                (report, "stress_recovery", {"peak_measure": "recovered_element_mean"}),
                (report, "three_body_verification", {"passed": True, "relative_l2_displacement_difference": .001}),
            ):
                with self.subTest(key=key):
                    original = document[key]
                    document[key] = invalid
                    save()
                    with self.assertRaises(ValueError):
                        require_verified_baseline(structure, root)
                    document[key] = original

    def test_projected_mortar_matches_elmer_multiplier_elimination(self):
        xyz = np.asarray(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
        mass, area = intersection_mass(xyz, xyz)
        matrix = projected_spring_matrix([(np.arange(6), mass)], 6).toarray()
        jump = mass[:3]
        expected = jump.T @ np.diag(1 / mass[:3, :3].sum(axis=1)) @ jump
        np.testing.assert_allclose(matrix, expected)
        np.testing.assert_allclose(matrix @ np.ones(6), 0, atol=1e-16)
        self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-15)
        displacement = np.asarray((1., 1., 1., 0., 0., 0.))
        self.assertAlmostEqual(displacement @ matrix @ displacement, area)
        self.assertFalse(np.allclose(matrix, mass))

    def test_directional_joint_forces_are_equal_and_opposite_without_ground_bypass(self):
        from dolfinx import fem
        from interface_springs import InterfaceSprings
        structure = column()
        imported = import_structure(structure, 1e-12)
        space = fem.functionspace(imported.domain, ("Lagrange", 1, (3,)))
        springs = InterfaceSprings(imported, structure, space)
        displacement = np.zeros((space.dofmap.index_map.size_global, 3))
        wall = np.unique(space.dofmap.list[imported.cell_tags.find(2)])
        plinth = np.unique(space.dofmap.list[imported.cell_tags.find(1)])
        displacement[wall] = (1., 2., 3.)
        force = (springs._operator @ displacement.ravel()).reshape(-1, 3)
        np.testing.assert_allclose(force[wall].sum(axis=0), (1e6, 4e6, 3e6), rtol=1e-12)
        np.testing.assert_allclose(force[plinth].sum(axis=0), (-1e6, -4e6, -3e6), rtol=1e-12)
        np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-8)
        np.testing.assert_allclose(springs._operator @ np.ones(displacement.size), 0, atol=1e-8)

    def test_triangle_jump_mass_is_exact_symmetric_and_translation_invariant(self):
        xyz = np.asarray(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
        mass, area = intersection_mass(xyz, xyz)
        self.assertAlmostEqual(area, .5)
        np.testing.assert_allclose(mass, mass.T)
        np.testing.assert_allclose(mass @ np.ones(6), 0, atol=1e-16)
        np.testing.assert_allclose(mass[:3, :3], .5 / 12 * (np.ones((3, 3)) + np.eye(3)))
        self.assertGreaterEqual(np.linalg.eigvalsh(mass).min(), -1e-15)

    def test_nonmatching_triangle_common_refinement_conserves_constant_traction(self):
        xyz = np.asarray(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
        midpoint = (xyz[1] + xyz[2]) / 2
        total = np.zeros((3, 3))
        for other in (np.asarray((xyz[0], xyz[1], midpoint)), np.asarray((xyz[0], midpoint, xyz[2]))):
            mass, _ = intersection_mass(xyz, other)
            total += mass[:3, :3]
        expected, _ = intersection_mass(xyz, xyz)
        np.testing.assert_allclose(total, expected[:3, :3], atol=1e-16)

    def test_three_body_gravity_analytical_column_and_interface_jumps(self):
        from dam_elasticity import solve
        structure = column()
        imported = import_structure(structure, 1e-12)
        with tempfile.TemporaryDirectory() as directory:
            displacement, _, report = solve(imported, structure,
                {"relative_tolerance": 1e-10, "maximum_iterations": 1000}, Path(directory))
            diagnostics = json.loads((Path(directory) / "linear_solver_report.json").read_text())
            self.assertEqual(diagnostics["linear_solver"]["method"], "gmres")
            self.assertEqual(diagnostics["iterations"], report["iterations"])
            self.assertLessEqual(diagnostics["relative_true_residual"], 1e-10)
        self.assertLess(report["relative_force_balance_error"], 1e-8)
        np.testing.assert_allclose(report["applied_force_n"], [0, 0, -40000], atol=1e-7)
        np.testing.assert_allclose(report["ground_spring_reactions_n"]["rock_bottom"], [0, 0, 40000], atol=.001)
        from dam_mesh import triangles
        xyz = np.asarray(structure.nodes)
        def average(tag):
            faces = np.asarray([tri for value, face in structure.boundaries if value == tag for tri in triangles(face)])
            points = xyz[faces]
            area = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1) / 2
            return np.average(displacement[faces, 2].mean(axis=1), weights=area)
        for first, second, jump in ((8, 9, -.01), (11, 12, -.01)):
            np.testing.assert_allclose(average(first) - average(second), jump, atol=1e-9)
        np.testing.assert_allclose(displacement[:4], 0, atol=1e-12)
        self.assertEqual(report["model_kind"], "three_body")
        self.assertEqual(report["linear_solver"], {"method": "gmres", "preconditioner": "gamg"})
        self.assertEqual(set(report["body_stress"]), {"wall", "plinth", "bedrock"})

    def test_refined_nonmatching_interfaces_and_wall_only_damage(self):
        from crack_field import refine_for_phase_field
        from test_dam_phase_field import DamPhaseFieldTests
        structure = column()
        imported = import_structure(structure, 1e-12)
        _, _, crack, options = DamPhaseFieldTests().model()
        crack = replace(crack, centre_m=(.5, .5, 2.5), length_scale_m=1.2,
                        critical_energy_release_rate_j_m2=1e9)
        options = replace(options, minimum_elements_per_length_scale=2., maximum_refinement_levels=4)
        refined = refine_for_phase_field(imported, [crack], options)
        self.assertGreater(refined.domain.topology.index_map(3).size_global, len(imported.tetrahedra))
        for formulation in ("hybrid", "experimental_amor"):
            with self.subTest(formulation=formulation), tempfile.TemporaryDirectory() as directory:
                report = check_phase(refined, structure, Path(directory), formulation)
                self.assertLess(report["relative_force_balance_error"], 1e-6)
                self.assertGreater(report["steps"][-1]["interface_spring_energy_j"], 0)
                faces = report["refined_mesh_baseline"]["interface_springs"]["wall_plinth"]["faces"]
                self.assertNotEqual(faces[0], faces[1])

    def test_two_rank_three_body_refined_hybrid_and_amor_parity(self):
        self.check_two_rank(column)

    def test_phase_field_with_rock_rollers_balances_without_dirichlet_bottom(self):
        from crack_field import refine_for_phase_field
        from test_dam_phase_field import DamPhaseFieldTests
        imported = import_structure(roller_column(), 1e-12)
        _, _, crack, options = DamPhaseFieldTests().model()
        crack = replace(crack, centre_m=(.5, .5, 2.5), length_scale_m=1.2)
        refined = refine_for_phase_field(imported, [crack], replace(options, maximum_refinement_levels=4))
        for formulation in ("hybrid", "experimental_amor"):
            with self.subTest(formulation=formulation), tempfile.TemporaryDirectory() as directory:
                report = check_phase(refined, roller_column(), Path(directory), formulation)
                self.assertLess(report["relative_force_balance_error"], 1e-6)
                self.assertEqual(report["refined_mesh_baseline"]["rock_outer_support"]["model"], "rollers")
                self.assertIn("rock_bank_x_min", report["ground_spring_reactions_n"])

    def test_two_rank_bonded_seat_refined_hybrid_and_amor_parity(self):
        self.check_two_rank(bonded_column)

    def check_two_rank(self, factory):
        from dolfinx import io
        from test_dam_phase_field import DamPhaseFieldTests
        from crack_field import refine_for_phase_field
        imported = import_structure(factory(), 1e-12)
        _, _, crack, options = DamPhaseFieldTests().model()
        crack = replace(crack, centre_m=(.5, .5, 2.5), length_scale_m=1.2)
        refined = refine_for_phase_field(imported, [crack], replace(options, maximum_refinement_levels=4))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with io.XDMFFile(refined.domain.comm, root / "mesh.xdmf", "w") as writer:
                writer.write_mesh(refined.domain)
                refined.cell_tags.name, refined.facet_tags.name = "regions", "boundaries"
                writer.write_meshtags(refined.cell_tags, refined.domain.geometry)
                writer.write_meshtags(refined.facet_tags, refined.domain.geometry)
            expected = {law: check_phase(refined, factory(), root / f"serial_{law}", law)
                        for law in ("hybrid", "experimental_amor")}
            command = ["mpiexec"]
            if os.geteuid() == 0:
                command.append("--allow-run-as-root")
            command += ["-n", "2", sys.executable, "-B", str(Path(__file__).resolve()), "--mpi-worker", str(root)]
            if factory == bonded_column:
                command.append("bonded")
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            for law, reference in expected.items():
                actual = json.loads((root / f"mpi_{law}" / "phase_field_report.json").read_text())
                for key in ("maximum_principal_tensile_stress_pa", "maximum_principal_compressive_stress_pa"):
                    np.testing.assert_allclose(actual[key], reference[key], rtol=1e-5, atol=.01)
                np.testing.assert_allclose(
                    actual["steps"][-1]["maximum_displacement_m"],
                    reference["steps"][-1]["maximum_displacement_m"], rtol=1e-6,
                )


def check_phase(imported, structure, output, formulation):
    from dolfinx import fem
    from crack_field import PhaseFieldParameters, PhaseFieldSolverOptions
    from dam_phase_field import solve
    parameters = PhaseFieldParameters(length_scale_m=.5,
        critical_energy_release_rate_j_m2=1e6, internal_pressure_pa=0., load_steps=2)
    options = PhaseFieldSolverOptions(maximum_staggered_iterations=30, staggered_tolerance=1e-6,
        mechanics_relative_tolerance=1e-10, mechanics_maximum_iterations=1000,
        damage_gradient_tolerance=1e-8, damage_maximum_iterations=1000)
    result = solve(imported, structure, [], options, output, parameters=parameters,
                   mechanics_formulation=formulation, refined_baseline=True)
    nonwall = imported.cell_tags.indices[imported.cell_tags.values != 2]
    dofs = fem.locate_dofs_topological(result.damage.function_space, 3, nonwall)
    np.testing.assert_array_equal(result.damage.x.array[dofs], 0.)
    assert result.report["steps"][-1]["maximum_damage"] > 0
    return result.report


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mpi-worker":
        from dolfinx import io
        from mpi4py import MPI
        root = Path(sys.argv[2])
        try:
            with io.XDMFFile(MPI.COMM_WORLD, root / "mesh.xdmf", "r") as reader:
                domain = reader.read_mesh(name="mesh")
                domain.topology.create_connectivity(3, 2)
                domain.topology.create_connectivity(2, 3)
                imported = SimpleNamespace(domain=domain,
                    cell_tags=reader.read_meshtags(domain, name="regions"),
                    facet_tags=reader.read_meshtags(domain, name="boundaries"))
            for law in ("hybrid", "experimental_amor"):
                structure = bonded_column() if len(sys.argv) > 3 and sys.argv[3] == "bonded" else column()
                check_phase(imported, structure, root / f"mpi_{law}", law)
        except Exception:
            import traceback
            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)
    else:
        unittest.main()
