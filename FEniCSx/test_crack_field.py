"""Verify the configured crack geometry, its signed distance and the diffuse damage field."""
from pathlib import Path
import json
import math
import tempfile
from types import SimpleNamespace
import unittest
import warnings

from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
import ufl

from crack_field import (
    CONFIG_PATH,
    CrackSpecification,
    crack_frame,
    crack_placement_report,
    crack_phase_field,
    damage_profile,
    load_crack_specifications,
    PhaseFieldSolverOptions,
    refine_for_phase_field,
    resolution_report,
    signed_distance,
    validate_crack_placement,
)


def specification(**overrides) -> CrackSpecification:
    values = {
        "type": "phase_field",
        "shape": "rectangular",
        "centre_m": (0.0, 0.0, 0.0),
        "size_m": (3.0, 5.0, 0.02),
        "normal": (0.0, 0.0, 1.0),
        "length_scale_m": 0.1,
        "critical_energy_release_rate_j_m2": 2700.0,
        "internal_pressure_pa": 0.0,
    }
    values.update(overrides)
    return CrackSpecification(**values)


class CrackConfigurationTests(unittest.TestCase):
    def test_reads_the_configured_crack_from_the_root_config(self):
        cracks = load_crack_specifications()
        self.assertEqual(len(cracks), 1)
        crack = cracks[0]
        declared = json.loads(CONFIG_PATH.read_text())["crack_simulations"][0]
        self.assertEqual(crack.type, "phase_field")
        self.assertEqual(crack.shape, "rectangular")
        self.assertEqual(crack.centre_m, (76.7430082340184, 13.946708830164285, -25.0))
        self.assertEqual(crack.size_m, (3.0, 5.0, 0.02))
        self.assertEqual(crack.normal, (0.0, 0.0, 1.0))
        self.assertEqual(crack.length_scale_m, declared["phase_field_parameters"]["length_scale_m"])
        self.assertEqual(
            crack.critical_energy_release_rate_j_m2,
            declared["phase_field_parameters"]["critical_energy_release_rate_J_m2"],
        )
        self.assertEqual(crack.internal_pressure_pa, declared["loading"]["internal_pressure_pa"])

    def test_rejects_a_crack_whose_normal_has_no_direction(self):
        config = json.loads(CONFIG_PATH.read_text())
        config["crack_simulations"][0]["orientation"] = {"normal_x": 0.0, "normal_y": 0.0, "normal_z": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError) as error:
                load_crack_specifications(path)
        self.assertIn("nonzero crack normal", str(error.exception))

    def test_rejects_a_crack_with_a_nonpositive_size(self):
        config = json.loads(CONFIG_PATH.read_text())
        config["crack_simulations"][0]["size_m"]["z"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError) as error:
                load_crack_specifications(path)
        self.assertIn("size_m.z", str(error.exception))


class CrackFrameTests(unittest.TestCase):
    def test_a_z_normal_keeps_the_global_axes(self):
        np.testing.assert_allclose(crack_frame((0.0, 0.0, 1.0)), np.eye(3), atol=1e-15)

    def test_the_frame_is_orthonormal_and_carries_z_onto_the_normal(self):
        for normal in [(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (1, 2, 3), (-2, 0.5, -1)]:
            with self.subTest(normal=normal):
                frame = crack_frame(normal)
                np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(float(np.linalg.det(frame)), 1.0, places=12)
                expected = np.asarray(normal, dtype=float)
                expected = expected / np.linalg.norm(expected)
                np.testing.assert_allclose(frame[:, 2], expected, atol=1e-12)

    def test_a_zero_normal_is_rejected(self):
        with self.assertRaises(ValueError):
            crack_frame((0.0, 0.0, 0.0))


class SignedDistanceTests(unittest.TestCase):
    def test_the_centre_is_the_deepest_point_inside_the_crack(self):
        crack = specification()
        self.assertAlmostEqual(signed_distance((0.0, 0.0, 0.0), crack), -0.01, places=12)

    def test_the_distance_vanishes_on_each_face(self):
        crack = specification()
        for point in [(1.5, 0.0, 0.0), (0.0, 2.5, 0.0), (0.0, 0.0, 0.01), (0.0, 0.0, -0.01)]:
            with self.subTest(point=point):
                self.assertAlmostEqual(signed_distance(point, crack), 0.0, places=12)

    def test_the_distance_along_the_normal_measures_from_the_crack_face(self):
        crack = specification()
        np.testing.assert_allclose(
            signed_distance([(0.0, 0.0, 0.26), (0.0, 0.0, -0.51)], crack), [0.25, 0.50], atol=1e-12
        )

    def test_the_distance_beyond_a_corner_is_the_euclidean_corner_distance(self):
        crack = specification()
        point = (1.5 + 0.3, 2.5 + 0.4, 0.01)
        self.assertAlmostEqual(signed_distance(point, crack), 0.5, places=12)

    def test_the_crack_translates_with_its_configured_centre(self):
        crack = specification(centre_m=(0.0, 87.0, -28.0))
        self.assertAlmostEqual(signed_distance((0.0, 87.0, -28.0), crack), -0.01, places=12)
        self.assertAlmostEqual(signed_distance((0.0, 87.0, -27.5), crack), 0.49, places=12)

    def test_the_crack_rotates_with_its_configured_normal(self):
        crack = specification(normal=(1.0, 0.0, 0.0))
        # The opening now lies along global x, and the plane spans global y and z.
        self.assertAlmostEqual(signed_distance((0.0, 0.0, 0.0), crack), -0.01, places=12)
        self.assertAlmostEqual(signed_distance((0.5, 0.0, 0.0), crack), 0.49, places=12)
        self.assertAlmostEqual(signed_distance((0.0, 2.5, 0.0), crack), 0.0, places=12)
        self.assertAlmostEqual(signed_distance((0.0, 0.0, 1.5), crack), 0.0, places=12)

    def test_a_rotated_crack_matches_the_unrotated_crack_under_the_same_rotation(self):
        crack = specification()
        rotated = specification(normal=(1.0, 2.0, 3.0))
        frame = crack_frame(rotated.normal)
        generator = np.random.default_rng(20260101)
        local = generator.uniform(-4.0, 4.0, size=(200, 3))
        np.testing.assert_allclose(
            signed_distance(local @ frame.T, rotated), signed_distance(local, crack), atol=1e-12
        )

    def test_the_distance_is_one_lipschitz_as_a_true_distance_function(self):
        crack = specification(centre_m=(0.0, 87.0, -28.0), normal=(0.3, -0.7, 1.0))
        generator = np.random.default_rng(7)
        points = generator.uniform(-6.0, 6.0, size=(300, 3)) + np.array([0.0, 87.0, -28.0])
        other = generator.uniform(-6.0, 6.0, size=(300, 3)) + np.array([0.0, 87.0, -28.0])
        difference = np.abs(signed_distance(points, crack) - signed_distance(other, crack))
        self.assertTrue(np.all(difference <= np.linalg.norm(points - other, axis=1) + 1e-12))

    def test_points_outside_the_slab_edge_are_not_treated_as_cracked(self):
        crack = specification()
        # Level with the crack but well beyond its in-plane extent.
        self.assertGreater(signed_distance((10.0, 0.0, 0.0), crack), 0.0)

    def test_rejects_malformed_point_arrays(self):
        with self.assertRaises(ValueError):
            signed_distance([[0.0, 0.0]], specification())


class DamageProfileTests(unittest.TestCase):
    def test_damage_saturates_inside_the_crack(self):
        for model in ("at1", "at2"):
            with self.subTest(model=model):
                np.testing.assert_allclose(damage_profile([-0.5, -0.01, 0.0], 0.1, model), [1.0, 1.0, 1.0])

    def test_at2_follows_the_exponential_optimal_profile(self):
        distance = np.array([0.0, 0.1, 0.2, 0.5])
        np.testing.assert_allclose(damage_profile(distance, 0.1, "at2"), np.exp(-distance / 0.1), rtol=1e-12)

    def test_at1_reaches_exactly_zero_at_twice_the_length_scale(self):
        self.assertAlmostEqual(float(damage_profile(0.2, 0.1, "at1")), 0.0, places=12)
        self.assertAlmostEqual(float(damage_profile(0.3, 0.1, "at1")), 0.0, places=12)
        self.assertAlmostEqual(float(damage_profile(0.1, 0.1, "at1")), 0.25, places=12)

    def test_damage_decreases_away_from_the_crack_and_stays_bounded(self):
        distance = np.linspace(0.0, 2.0, 200)
        for model in ("at1", "at2"):
            with self.subTest(model=model):
                values = damage_profile(distance, 0.1, model)
                self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))
                self.assertTrue(np.all(np.diff(values) <= 1e-15))

    def test_rejects_an_unusable_length_scale_or_model(self):
        with self.assertRaises(ValueError):
            damage_profile(1.0, 0.0)
        with self.assertRaises(ValueError):
            damage_profile(1.0, 0.1, "unsupported")


class CrackPhaseFieldTests(unittest.TestCase):
    def box(self, divisions=24, half=1.2):
        return mesh.create_box(
            MPI.COMM_WORLD,
            [np.array([-half, -half, -half]), np.array([half, half, half])],
            [divisions, divisions, divisions],
            mesh.CellType.tetrahedron,
        )

    def test_builds_a_bounded_damage_function_centred_on_the_crack(self):
        domain = self.box()
        crack = specification(size_m=(1.0, 1.0, 0.02), length_scale_m=0.2)
        damage = crack_phase_field(domain, crack, check_resolution=False)
        self.assertIsInstance(damage, fem.Function)
        self.assertEqual(damage.name, "damage")
        values = damage.x.array
        self.assertTrue(np.all((values >= 0.0) & (values <= 1.0 + 1e-12)))
        self.assertAlmostEqual(float(values.max()), 1.0, places=6)

        coordinates = damage.function_space.tabulate_dof_coordinates()
        expected = damage_profile(signed_distance(coordinates, crack), crack.length_scale_m, "at2")
        np.testing.assert_allclose(values, expected, atol=1e-12)

    def test_damage_is_saturated_on_the_crack_and_decays_away_from_it(self):
        domain = self.box()
        crack = specification(size_m=(1.0, 1.0, 0.02), length_scale_m=0.2)
        damage = crack_phase_field(domain, crack, check_resolution=False)
        coordinates = damage.function_space.tabulate_dof_coordinates()
        distance = signed_distance(coordinates, crack)
        values = damage.x.array
        self.assertTrue(np.all(values[distance <= 0.0] >= 1.0 - 1e-12))
        far = distance > 5.0 * crack.length_scale_m
        if far.any():
            self.assertLess(float(values[far].max()), math.exp(-5.0) + 1e-6)

    def test_the_damaged_band_follows_a_rotated_normal(self):
        domain = self.box()
        crack = specification(size_m=(1.0, 1.0, 0.02), normal=(1.0, 0.0, 0.0), length_scale_m=0.2)
        damage = crack_phase_field(domain, crack, check_resolution=False)
        space = damage.function_space

        def value(point):
            evaluated = fem.Function(space)
            evaluated.interpolate(damage)
            index = int(np.argmin(np.linalg.norm(space.tabulate_dof_coordinates() - np.array(point), axis=1)))
            return float(damage.x.array[index])

        # On the crack plane the damage saturates; the same offset along the normal does not.
        self.assertGreater(value((0.0, 0.0, 0.0)), 0.99)
        self.assertLess(value((0.6, 0.0, 0.0)), value((0.0, 0.6, 0.0)))

    def test_at1_leaves_intact_material_outside_the_damaged_band(self):
        domain = self.box()
        crack = specification(size_m=(1.0, 1.0, 0.02), length_scale_m=0.1)
        damage = crack_phase_field(domain, crack, model="at1", check_resolution=False)
        coordinates = damage.function_space.tabulate_dof_coordinates()
        outside = signed_distance(coordinates, crack) > 2.0 * crack.length_scale_m
        self.assertTrue(outside.any())
        np.testing.assert_allclose(damage.x.array[outside], 0.0, atol=1e-15)

    def test_several_cracks_combine_without_erasing_each_other(self):
        domain = self.box()
        first = specification(centre_m=(0.0, -0.6, 0.0), size_m=(0.8, 0.4, 0.02), length_scale_m=0.15)
        second = specification(centre_m=(0.0, 0.6, 0.0), size_m=(0.8, 0.4, 0.02), length_scale_m=0.15)
        damage = crack_phase_field(domain, [first, second], check_resolution=False)
        coordinates = damage.function_space.tabulate_dof_coordinates()
        expected = np.maximum(
            damage_profile(signed_distance(coordinates, first), first.length_scale_m, "at2"),
            damage_profile(signed_distance(coordinates, second), second.length_scale_m, "at2"),
        )
        np.testing.assert_allclose(damage.x.array, expected, atol=1e-12)

    def test_higher_order_spaces_are_supported(self):
        domain = self.box(divisions=8)
        crack = specification(size_m=(1.0, 1.0, 0.02), length_scale_m=0.3)
        damage = crack_phase_field(domain, crack, degree=2, check_resolution=False)
        self.assertEqual(damage.function_space.ufl_element().degree, 2)
        self.assertTrue(np.all(damage.x.array <= 1.0 + 1e-12))

    def test_an_under_resolved_length_scale_is_reported_not_hidden(self):
        domain = self.box(divisions=4)
        crack = specification(size_m=(1.0, 1.0, 0.02), length_scale_m=0.01)
        report = resolution_report(domain, crack)
        self.assertFalse(report["resolved"])
        # A band thinner than one element must still report a usable number, not nan.
        self.assertTrue(math.isfinite(report["elements_per_length_scale"]))
        self.assertGreaterEqual(report["cells_in_damaged_band"], 1)
        self.assertLess(report["elements_per_length_scale"], 2.0)
        with self.assertWarns(RuntimeWarning):
            crack_phase_field(domain, crack)

    def test_a_resolved_length_scale_warns_nothing(self):
        domain = self.box(divisions=24)
        crack = specification(size_m=(1.0, 1.0, 0.02), length_scale_m=0.4)
        report = resolution_report(domain, crack)
        self.assertTrue(report["resolved"])
        self.assertGreaterEqual(report["elements_per_length_scale"], 2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            crack_phase_field(domain, crack)

    def test_unsupported_crack_descriptions_are_rejected(self):
        domain = self.box(divisions=4)
        with self.assertRaises(ValueError):
            crack_phase_field(domain, [])
        with self.assertRaises(ValueError):
            crack_phase_field(domain, specification(shape="elliptical"), check_resolution=False)
        with self.assertRaises(ValueError):
            crack_phase_field(domain, specification(type="cohesive"), check_resolution=False)


class CrackPlacementAndRefinementTests(unittest.TestCase):
    def imported_box(self):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
        dimension = domain.topology.dim
        cells = np.arange(domain.topology.index_map(dimension).size_local, dtype=np.int32)
        cell_tags = mesh.meshtags(domain, dimension, cells, np.full(len(cells), 2, dtype=np.int32))
        domain.topology.create_connectivity(dimension - 1, dimension)
        facets = mesh.exterior_facet_indices(domain.topology)
        facet_tags = mesh.meshtags(
            domain,
            dimension - 1,
            facets,
            np.ones(len(facets), dtype=np.int32),
        )
        return SimpleNamespace(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)

    def test_reports_a_crack_inside_the_required_material_region(self):
        imported = self.imported_box()
        crack = specification(
            centre_m=(0.5, 0.5, 0.5),
            size_m=(0.4, 0.4, 0.02),
            length_scale_m=0.2,
        )
        report = crack_placement_report(imported, crack)
        self.assertTrue(report["valid"])
        self.assertTrue(report["centre_in_required_region"])
        self.assertGreater(report["required_region_band_cell_count"], 0)

    def test_rejects_a_crack_outside_the_mesh_with_actionable_bounds(self):
        imported = self.imported_box()
        crack = specification(centre_m=(10.0, 10.0, 10.0))
        with self.assertRaises(ValueError) as context:
            validate_crack_placement(imported, crack)
        self.assertIn("mesh_bounds", str(context.exception))
        self.assertIn("concrete wall", str(context.exception))

    def test_local_refinement_preserves_region_and_boundary_tags(self):
        imported = self.imported_box()
        crack = specification(
            centre_m=(0.5, 0.5, 0.5),
            size_m=(0.4, 0.4, 0.02),
            length_scale_m=0.3,
        )
        before = resolution_report(imported.domain, crack)["elements_per_length_scale"]
        refined = refine_for_phase_field(
            imported,
            crack,
            PhaseFieldSolverOptions(
                minimum_elements_per_length_scale=0.6,
                maximum_refinement_levels=2,
            ),
        )
        after = refined.report["levels"][-1]["cracks"][0]["elements_per_length_scale"]
        self.assertGreater(after, before)
        self.assertGreater(len(refined.cell_tags.find(2)), 0)
        self.assertGreater(len(refined.facet_tags.find(1)), 0)
        self.assertEqual(refined.report["status"], "preliminary_under_resolved")
        self.assertFalse(refined.report["meets_recommended_resolution"])


class CrackSurfaceDensityTests(unittest.TestCase):
    """The AT2 surface density of the damage field must recover the crack area it represents."""

    THICKNESS = 0.02
    LENGTH_SCALE = 0.1

    def spanning_crack(self):
        # In-plane extents overrun the box, so no crack edge lies inside the domain.
        return specification(
            size_m=(10.0, 10.0, self.THICKNESS), length_scale_m=self.LENGTH_SCALE
        )

    def surface_density(self, divisions):
        domain = mesh.create_box(
            MPI.COMM_WORLD,
            [np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5])],
            [4, 4, divisions],
            mesh.CellType.tetrahedron,
        )
        damage = crack_phase_field(domain, self.spanning_crack(), check_resolution=False)
        density = (
            damage**2 / (2 * self.LENGTH_SCALE)
            + (self.LENGTH_SCALE / 2) * ufl.inner(ufl.grad(damage), ufl.grad(damage))
        ) * ufl.dx
        return fem.assemble_scalar(fem.form(density))

    def test_the_regularised_surface_approaches_the_represented_crack_area(self):
        # A unit cross-section slab of thickness t has exact density A * (1 + t / (2 l)).
        exact = 1.0 * (1.0 + self.THICKNESS / (2.0 * self.LENGTH_SCALE))
        errors = [abs(self.surface_density(n) - exact) / exact for n in (20, 40, 80)]
        self.assertTrue(
            all(later < earlier for earlier, later in zip(errors, errors[1:])),
            f"refinement must reduce the surface density error, got {errors}",
        )
        self.assertLess(errors[-1], 0.05)


if __name__ == "__main__":
    unittest.main()
