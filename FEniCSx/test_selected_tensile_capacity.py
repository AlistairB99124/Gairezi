"""Native load-controlled checks: applied tension is not compressive pressure."""
import json
from pathlib import Path
import tempfile
import unittest

from dolfinx import fem, mesh
import numpy as np
import ufl

from paired_contact import ContactOptions
from tensile_loss_event import EventSolverOptions, solve_contact_event
import test_open_crack_equilibrium as bulk


class SelectedTensileCapacityTests(unittest.TestCase):
    def specimen(self, traction, output):
        space, faces, _, _, _ = bulk.BulkFlowTests().fixture()
        domain = space.mesh
        vertices = np.arange(domain.topology.index_map(0).size_local, dtype=np.int32)
        bottom = mesh.locate_entities(domain, 0, lambda x: np.isclose(x[1], 0.))
        bcs = [fem.dirichletbc(
            0., fem.locate_dofs_topological(space.sub(i), 0, nodes), space.sub(i))
            for i, nodes in ((0, vertices), (1, bottom), (2, vertices))]
        top = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[1], 1.))
        tags = mesh.meshtags(domain, 2, top, np.ones(len(top), dtype=np.int32))
        ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
        count = domain.topology.index_map(3).size_local
        return solve_contact_event(
            space, faces, young_pa=35e9, poisson=0., selected_cells=np.ones(count, dtype=bool),
            selected_tensile_strength_pa=.5e6, boundary_conditions=bcs,
            external_load=lambda v: fem.Constant(domain, float(traction)) * v[1] * ds(1),
            options=EventSolverOptions(maximum_iterations=5),
            contact_options=ContactOptions(1e-12, 1e-7, 1e-8, 100),
            release_cut=False, output=output)

    def test_point_four_mpa_tension_stays_elastic_below_point_five_mpa_strength(self):
        with tempfile.TemporaryDirectory() as directory:
            event = self.specimen(.4e6, Path(directory) / "event")
            np.testing.assert_allclose(
                event.post_event_displacement.x.array, event.intact_displacement.x.array,
                rtol=1e-8, atol=1e-12)
            stress = event.report["iterations"][-1]["maximum_selected_principal_stress_pa"]
            self.assertAlmostEqual(stress / 1e6, .4, places=7)
            self.assertLess(event.report["iterations"][-1]["scaled_free_residual"], 1e-8)

    def test_one_point_two_mpa_tension_cannot_equilibrate_at_point_five_mpa_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "event"
            with self.assertRaisesRegex(RuntimeError, "no post-event equilibrium accepted"):
                self.specimen(1.2e6, output)
            report = json.loads((output / "event_report.json").read_text())
            self.assertEqual(report["status"], "failed_reduced_tensile_contact_event")
            self.assertTrue((output / "intact_state.npz").is_file())
            self.assertFalse((output / "post_event_state.npz").exists())
            self.assertIn("failed independent residual/complementarity checks", report["error"])

    def test_one_point_two_mpa_compression_does_not_test_tensile_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            event = self.specimen(-1.2e6, Path(directory) / "event")
            np.testing.assert_allclose(
                event.post_event_displacement.x.array, event.intact_displacement.x.array,
                rtol=1e-8, atol=1e-12)
            self.assertLess(event.report["iterations"][-1]["scaled_force_imbalance"], 1e-8)


if __name__ == "__main__":
    unittest.main()
