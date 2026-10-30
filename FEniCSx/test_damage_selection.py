"""Native selection containment, mesh identity and preview-only lifecycle."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from FEniCSx.damage_selection import build_catalog, validate_selection, bind_source_selection, read_saved_selection
from Elmer.model_config import ConfigurationValidationError


class DamageCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.msh"
        self.source.write_text("test mesh identity", encoding="utf-8")
        self.config = {"geometry": {
            "global_element_size_m": .5,
            "wall": {"upstream_radius_m": 80., "downstream_radius_m": 76.,
                     "centerline_radius_m": 78.},
        }}
        self.points = np.array([
            [0., 0., 0.],
            [.2, 79.8, -1.], [.4, 79.8, -1.], [.2, 77.2, -1.], [.2, 79.8, -.5],
            [.2, 76.1, -1.], [.4, 76.1, -1.], [.2, 76.5, -1.], [.2, 76.1, -.5],
        ])
        self.cells = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        self.fields = self.root / "fields.npz"
        stress = np.zeros((8, 6))
        stress[:, 0] = 1e5
        stress[0, 0] = 2e6
        np.savez(self.fields, coordinates_m=self.points[1:], stress_pa=stress,
                 node_regions=np.ones(8, dtype=int))
        self.arrays = (self.points, self.cells, np.array([1, 1]),
                       np.empty((0, 3), dtype=int), np.empty(0, dtype=int))

    def catalog(self):
        with patch("FEniCSx.damage_selection._read_gmsh_ascii", return_value=self.arrays):
            return build_catalog(self.source, self.fields, self.config,
                                 json.dumps(self.config).encode())

    def test_identity_ignores_last_bit_display_float_differences_between_platforms(self):
        reference = self.catalog()
        from FEniCSx import damage_selection
        cos, principal = np.cos, damage_selection.principal_stresses
        with patch("FEniCSx.damage_selection.np.cos", lambda value: np.nextafter(cos(value), np.inf)), \
                patch("FEniCSx.damage_selection.principal_stresses",
                      lambda value: np.nextafter(principal(value), np.inf)):
            nudged = self.catalog()
        self.assertNotEqual(nudged["nodes"], reference["nodes"])
        self.assertNotEqual([e["stress_mpa"] for e in nudged["elements"]],
                            [e["stress_mpa"] for e in reference["elements"]])
        self.assertEqual(nudged["catalog_sha256"], reference["catalog_sha256"],
                         "Display-only last-bit float differences must not make a saved selection stale")
        reordered = (self.arrays[0], np.array([[1, 2, 3, 4], [5, 6, 8, 7]]), *self.arrays[2:])
        with patch("FEniCSx.damage_selection._read_gmsh_ascii", return_value=reordered):
            moved = build_catalog(self.source, self.fields, self.config, json.dumps(self.config).encode())
        self.assertNotEqual(moved["catalog_sha256"], reference["catalog_sha256"],
                            "Native topology changes must still change catalog identity")

    def request(self, catalog):
        return {"catalog_sha256": catalog["catalog_sha256"],
                "affected_element_ids": [0], "crack_start_edge_ids": [0],
                "face_overrides": {}, "opening_limit_m": .002}

    def test_saved_snapshot_identity_and_integrity_shared_with_isolated_runner(self):
        config_digest = "b" * 64
        path = self.root / "selection-1.json"
        packet = {
            "schema_version": 1, "analysis_kind": "painted_tensile_loss", "job_id": "job",
            "configuration_digest": config_digest, "selection_id": path.stem,
            "status": "saved_selection_only", "solver_ready": False,
            "request": {"affected_element_ids": [1], "crack_start_edge_ids": [2]}}

        def signed(value):
            return {**value, "selection_sha256": hashlib.sha256(json.dumps(
                value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}

        path.write_text(json.dumps(signed(packet)))
        self.assertEqual(read_saved_selection(path, "job", config_digest), signed(packet))
        for change in (
            {"schema_version": True}, {"job_id": "foreign"}, {"configuration_digest": "foreign"},
            {"selection_id": "foreign"}, {"analysis_kind": "legacy"}, {"status": "solved"},
            {"solver_ready": True}, {"request": []},
        ):
            with self.subTest(change=change):
                path.write_text(json.dumps(signed({**packet, **change})))
                with self.assertRaisesRegex(ValueError, "identity"):
                    read_saved_selection(path, "job", config_digest)
        path.write_text(json.dumps({**signed(packet), "request": {}}))
        with self.assertRaisesRegex(ValueError, "integrity"):
            read_saved_selection(path, "job", config_digest)
        for content in ("[]", "not json"):
            path.write_text(content)
            with self.assertRaises(ValueError):
                read_saved_selection(path, "job", config_digest)

    def binding_fixture(self):
        configuration = self.root / "config.json"
        configuration.write_bytes(json.dumps(self.config).encode())
        structure = SimpleNamespace(
            nodes=self.points[1:].copy(),
            cells=[(4, tuple(row - 1)) for row in self.cells],
            region_ids=[1, 1], three_body_config=self.config,
        )
        return configuration, structure

    def test_source_binding_retains_markers_and_exact_compacted_identity(self):
        configuration, structure = self.binding_fixture()
        catalog = self.catalog()
        with patch("FEniCSx.damage_selection._read_gmsh_ascii", return_value=self.arrays):
            bound = bind_source_selection(self.source, self.fields, configuration, structure, self.request(catalog))
        self.assertEqual(bound["status"], "bound_selection_only")
        self.assertFalse(bound["solver_ready"])
        self.assertEqual(bound["affected_element_ids"], [0])
        self.assertEqual(bound["crack_start_edge_ids"], [0])
        self.assertEqual(bound["marker_source_edges"],
                         (np.asarray(catalog["edges"][0]["node_ids"]) - 1).reshape(1, 2).tolist())
        self.assertEqual(bound["resolved_faces"], {"0": "upstream"})
        self.assertEqual(bound["opening_limit_m"], .002)
        for key in ("source_mesh_sha256", "baseline_fields_sha256", "configuration_sha256", "catalog_sha256"):
            self.assertEqual(bound[key], catalog[key])

    def test_source_binding_reads_actual_gmsh_file_without_parser_substitution(self):
        configuration, structure = self.binding_fixture()
        lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(self.points) - 1)]
        lines.extend(f"{i} {x} {y} {z}" for i, (x, y, z) in enumerate(self.points[1:], 1))
        lines.extend(["$EndNodes", "$Elements", str(len(self.cells))])
        lines.extend(f"{i} 4 2 1 1 " + " ".join(map(str, nodes))
                     for i, nodes in enumerate(self.cells, 1))
        lines.append("$EndElements")
        self.source.write_text("\n".join(lines) + "\n", encoding="utf-8")
        catalog = build_catalog(self.source, self.fields, self.config, configuration.read_bytes())
        bound = bind_source_selection(self.source, self.fields, configuration, structure, self.request(catalog))
        self.assertEqual(bound["affected_element_ids"], [0])
        self.assertEqual(bound["marker_source_edges"],
                         (np.asarray(catalog["edges"][0]["node_ids"]) - 1).reshape(1, 2).tolist())
        self.assertEqual(bound["source_mesh_sha256"], catalog["source_mesh_sha256"])
        self.source.write_text(self.source.read_text().replace("1 4 2 1 1", "1 4 2 2 2"))
        with self.assertRaisesRegex(ValueError, "stale mesh catalog"):
            bind_source_selection(self.source, self.fields, configuration, structure, self.request(catalog))

    def test_source_binding_rejects_reordered_welded_or_changed_structures(self):
        for change in ("nodes", "cells", "regions", "kind", "config"):
            configuration, structure = self.binding_fixture()
            if change == "nodes":
                structure.nodes[0, 0] += .001
            elif change == "cells":
                structure.cells.reverse()
            elif change == "regions":
                structure.region_ids[0] = 2
            elif change == "kind":
                structure.cells[0] = (5, structure.cells[0][1])
            else:
                structure.three_body_config = {}
            with self.subTest(change=change), patch(
                "FEniCSx.damage_selection._read_gmsh_ascii", return_value=self.arrays,
            ), self.assertRaisesRegex(ValueError, "differ"):
                bind_source_selection(self.source, self.fields, configuration, structure, self.request(self.catalog()))

    def test_source_binding_rejects_files_changed_during_second_parse(self):
        configuration, structure = self.binding_fixture()
        catalog = self.catalog()
        calls = 0
        def parsing(path):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.source.write_text("changed after catalog rebuild", encoding="utf-8")
            return self.arrays
        with patch("FEniCSx.damage_selection._read_gmsh_ascii", side_effect=parsing):
            with self.assertRaisesRegex(ValueError, "changed during source binding"):
                bind_source_selection(self.source, self.fields, configuration, structure, self.request(catalog))

    def test_native_ids_face_and_whole_element_depth(self):
        catalog = self.catalog()
        self.assertEqual([e["id"] for e in catalog["elements"]], [0, 1])
        first, deep = catalog["elements"]
        self.assertEqual(first["face"], "upstream")
        self.assertTrue(first["upstream_eligible"])
        self.assertFalse(deep["upstream_eligible"])
        self.assertTrue(deep["downstream_eligible"])
        self.assertEqual(catalog["depth_fraction"], .75)
        self.assertFalse(catalog["solver_ready"])
        self.assertEqual(len({tuple(e["node_ids"]) for e in catalog["edges"]}),
                         len(catalog["edges"]))

    def test_valid_selection_is_not_solver_ready(self):
        catalog = self.catalog()
        result = validate_selection(catalog, self.request(catalog))
        self.assertEqual(result["status"], "validated_selection_only")
        self.assertFalse(result["solver_ready"])
        self.assertIn("no solve was started", result["message"])
        self.assertEqual(result["opening_limit_m"], .002)

    def test_missing_element_or_edge_rejected(self):
        catalog = self.catalog()
        for field in ("affected_element_ids", "crack_start_edge_ids"):
            request = self.request(catalog)
            request[field] = []
            with self.subTest(field=field), self.assertRaises(ConfigurationValidationError):
                validate_selection(catalog, request)

    def test_unknown_overdepth_stale_and_invalid_requests_rejected(self):
        catalog = self.catalog()
        for field, value in (
            ("affected_element_ids", [999]), ("crack_start_edge_ids", [999]),
            ("affected_element_ids", [1]), ("catalog_sha256", "0" * 64),
            ("opening_limit_m", 0), ("opening_limit_m", True),
            ("face_overrides", {"0": "downstream"}),
        ):
            request = self.request(catalog)
            request[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_selection(catalog, request)

    def test_ambiguous_peak_requires_face_and_depth_check(self):
        catalog = self.catalog()
        catalog["elements"][0]["face"] = None
        request = self.request(catalog)
        with self.assertRaisesRegex(ConfigurationValidationError, "choose a face"):
            validate_selection(catalog, request)
        request["face_overrides"] = {"0": "upstream"}
        self.assertEqual(validate_selection(catalog, request)["resolved_faces"], {"0": "upstream"})
        catalog["elements"][0]["upstream_eligible"] = False
        with self.assertRaisesRegex(ConfigurationValidationError, "75%"):
            validate_selection(catalog, request)

    def test_catalog_binding_changes_with_source_config_or_fields(self):
        first = self.catalog()
        self.assertEqual(first["catalog_sha256"], self.catalog()["catalog_sha256"])
        self.source.write_text("changed source", encoding="utf-8")
        self.assertNotEqual(first["catalog_sha256"], self.catalog()["catalog_sha256"])
        self.config["geometry"]["wall"]["centerline_radius_m"] = 77.9
        self.assertNotEqual(first["catalog_sha256"], self.catalog()["catalog_sha256"])

    def test_nonwall_elements_are_not_selectable(self):
        arrays = list(self.arrays)
        arrays[2] = np.array([1, 2])
        self.arrays = tuple(arrays)
        self.assertEqual([e["id"] for e in self.catalog()["elements"]], [0])

    def test_missing_stress_mapping_is_not_silently_defaulted(self):
        self.points = self.points.copy()
        self.points[1, 0] = 2
        self.arrays = (self.points, *self.arrays[1:])
        with self.assertRaisesRegex(ValueError, "no matching Elmer"):
            self.catalog()

    def test_equal_peaks_on_opposite_faces_require_explicit_choice(self):
        stress = np.zeros((8, 6))
        stress[0, 0] = stress[4, 0] = 2e6
        np.savez(self.fields, coordinates_m=self.points[1:], stress_pa=stress,
                 node_regions=np.ones(8, dtype=int))
        catalog = self.catalog()
        self.assertIsNone(catalog["elements"][0]["face"])
        with self.assertRaisesRegex(ConfigurationValidationError, "choose a face"):
            validate_selection(catalog, self.request(catalog))


if __name__ == "__main__":
    unittest.main()
