"""Geometry conservation and topology checks for plinth-only corner cutbacks."""
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Elmer.model_config import ConfigurationValidationError, load_config, validate_config
from Elmer.regions.combined_structure import build_combined_structure, audit_combined_structure
from Elmer.regions.interface_cutback import CELL_FACES, cutback_wall_plinth
from Elmer.regions.tetrahedral import split_cell, _signed_volume, split_face, tetrahedralize_structure
from Elmer.three_body_diagnostic import split_wall_from_plinth


ROOT = Path(__file__).resolve().parents[1]


def face_owners(mesh):
    owners = {}
    for (kind, cell), region in zip(mesh.cells, mesh.region_ids, strict=True):
        for pattern in CELL_FACES[kind]:
            face = tuple(sorted(cell[i] for i in pattern))
            owners.setdefault(face, []).append(region)
    return owners


def volumes(mesh):
    result = {1: 0., 2: 0.}
    for (kind, cell), region in zip(mesh.cells, mesh.region_ids, strict=True):
        for tet in split_cell(mesh.nodes, kind, cell):
            volume = _signed_volume(mesh.nodes, tet)
            if volume <= 0:
                raise AssertionError("Non-positive concrete tetrahedron")
            result[region] += volume
    return result


class InterfaceCutbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "config.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(json.dumps(cls.config))
            config["geometry"]["plinth"].pop("wall_interface", None)
            (root / "config.json").write_text(json.dumps(config))
            cls.original = build_combined_structure(root)
        cls.chamfered = cutback_wall_plinth(
            cls.original, cls.config["geometry"]["wall"]["centerline_radius_m"],
        )

    def test_cutback_transfers_volume_only_from_plinth_to_wall(self):
        before = volumes(self.original)
        after = volumes(self.chamfered)
        transfer = self.chamfered.wall_plinth_cutback["transferred_volume_m3"]
        self.assertGreater(transfer, 0.)
        removed = self.chamfered.wall_plinth_cutback["removed_shoulder_volume_m3"]
        self.assertGreater(removed, 0.)
        self.assertAlmostEqual(before[1] - after[1], transfer + removed, places=6)
        self.assertAlmostEqual(after[2] - before[2], transfer, places=6)
        self.assertAlmostEqual(sum(before.values()) - sum(after.values()), removed, places=6)
        self.assertAlmostEqual(removed / transfer, 0.5, places=6)
        self.assertEqual(self.original.nodes, self.chamfered.nodes)
        self.assertEqual(self.chamfered.wall_plinth_cutback["nodes_moved_or_added"], 0)

    def test_conforming_interface_has_45_degree_drops_and_no_longitudinal_risers(self):
        mesh = self.chamfered
        owners = face_owners(mesh)
        self.assertLessEqual(max(map(len, owners.values())), 2)
        self.assertEqual(
            {face for face, regions in owners.items() if len(regions) == 1},
            {tuple(sorted(face)) for _, face in mesh.boundaries},
        )
        radius = self.config["geometry"]["wall"]["centerline_radius_m"]
        inclined = 0
        ascending = descending = 0
        for face, regions in owners.items():
            if set(regions) != {1, 2}:
                continue
            p = [mesh.nodes[n] for n in face]
            stations = [radius * math.atan2(x, y) for x, y, _ in p]
            dz = max(v[2] for v in p) - min(v[2] for v in p)
            ds = max(stations) - min(stations)
            if dz > 1e-7:
                self.assertGreater(ds, 1e-7, "Vertical wall/plinth riser remains")
                if len(face) == 4:
                    self.assertAlmostEqual(ds, dz, places=7)
                    inclined += 1
                    ordered = sorted(zip(stations, (v[2] for v in p)))
                    ascending += ordered[-1][1] > ordered[0][1]
                    descending += ordered[-1][1] < ordered[0][1]
        self.assertEqual(inclined, mesh.wall_plinth_cutback["wall_wedges"])
        self.assertGreater(ascending, 0)
        self.assertGreater(descending, 0)
        self.assertEqual(audit_combined_structure(mesh)["face_connected_components"], 1)

    def test_shoulder_envelope_changes_but_bedrock_seat_is_preserved(self):
        def areas(mesh):
            result = Counter()
            for tag, face in mesh.boundaries:
                for tri in split_face(tuple(face)):
                    p = np.asarray([mesh.nodes[n] for n in tri])
                    result[tag] += np.linalg.norm(np.cross(p[1]-p[0], p[2]-p[0])) / 2
            return result
        before, after = areas(self.original), areas(self.chamfered)
        self.assertEqual(before.keys(), after.keys())
        self.assertGreater(abs(before[2] - after[2]), 1.)
        self.assertGreater(abs(before[3] - after[3]), 1.)
        original_seat = {tuple(sorted(f)) for tag, f in self.original.boundaries if tag == 1}
        revised_seat = {tuple(sorted(f)) for tag, f in self.chamfered.boundaries if tag == 1}
        self.assertEqual(original_seat, revised_seat)
        self.assertEqual(self.original.region_cell_counts["wall"] + 880,
                         self.chamfered.region_cell_counts["wall"])

    def test_cutbacks_extend_through_both_exposed_plinth_shoulders(self):
        mesh = self.chamfered
        report = mesh.wall_plinth_cutback
        self.assertEqual(report["radial_extent"], "full_plinth_width_including_shoulders")
        self.assertEqual(report["wall_wedges"], 880)
        self.assertEqual(report["shoulder_wall_wedges"], 0)
        self.assertEqual(report["removed_shoulder_wedges"], 220)
        self.assertEqual(report["shoulder_fill"], "empty_outside_original_wall_footprint")
        wall = self.config["geometry"]["wall"]
        plinth = self.config["geometry"]["plinth"]
        radius = wall["centerline_radius_m"]
        courses = {}
        shoulder_counts = Counter()
        for (kind, cell), region in zip(mesh.cells, mesh.region_ids, strict=True):
            if kind != 6 or region != 2:
                continue
            radii = [math.hypot(*mesh.nodes[n][:2]) for n in cell]
            stations = [radius * math.atan2(*mesh.nodes[n][:2]) for n in cell]
            heights = [mesh.nodes[n][2] for n in cell]
            key = tuple(round(v, 7) for v in (min(stations), max(stations), min(heights), max(heights)))
            courses.setdefault(key, []).extend(radii)
            if max(radii) <= wall["downstream_radius_m"] + 1e-7:
                shoulder_counts["downstream"] += 1
            if min(radii) >= wall["upstream_radius_m"] - 1e-7:
                shoulder_counts["upstream"] += 1
        self.assertEqual(len(courses), report["step_count"])
        self.assertEqual(shoulder_counts, {})
        wall_geometry = audit_combined_structure(mesh)["wall_geometry"]
        self.assertFalse(wall_geometry["locally_widened_at_cutbacks"])
        self.assertAlmostEqual(wall_geometry["uniform_thickness_m"], 4.)
        for radii in courses.values():
            self.assertAlmostEqual(min(radii), wall["downstream_radius_m"])
            self.assertAlmostEqual(max(radii), wall["upstream_radius_m"])
        owners = face_owners(mesh)
        for outer_radius in (plinth["downstream_radius_m"], plinth["upstream_radius_m"]):
            edges = set()
            for tag, face in mesh.boundaries:
                regions = owners[tuple(sorted(face))]
                if regions != [1]:
                    continue
                for a, b in zip(face, face[1:] + face[:1]):
                    if all(abs(math.hypot(*mesh.nodes[n][:2]) - outer_radius) < 1e-7
                           for n in (a, b)):
                        edges.add(tuple(sorted((a, b))))
            inclined = []
            for a, b in edges:
                ds = radius * (
                    math.atan2(*mesh.nodes[a][:2]) - math.atan2(*mesh.nodes[b][:2])
                )
                dz = mesh.nodes[a][2] - mesh.nodes[b][2]
                if abs(dz) > 1e-7 and abs(ds) > 1e-7:
                    stations = [radius * math.atan2(*mesh.nodes[n][:2]) for n in (a, b)]
                    heights = [mesh.nodes[n][2] for n in (a, b)]
                    key = tuple(round(v, 7) for v in (
                        min(stations), max(stations), min(heights), max(heights),
                    ))
                    if key not in courses:
                        continue
                    self.assertAlmostEqual(abs(ds), abs(dz), places=7)
                    inclined.append((a, b))
            self.assertEqual(len(inclined), report["step_count"])

    def test_tetrahedral_spring_interfaces_still_match_without_hanging_faces(self):
        mesh = tetrahedralize_structure(self.chamfered)
        owners = face_owners(mesh)
        self.assertLessEqual(max(map(len, owners.values())), 2)
        self.assertEqual(
            {face for face, regions in owners.items() if len(regions) == 1},
            {tuple(sorted(face)) for _, face in mesh.boundaries},
        )
        nodes, _, boundaries, count = split_wall_from_plinth(mesh)
        wall_faces = [f for tag, f in boundaries if tag == 8]
        plinth_faces = [f for tag, f in boundaries if tag == 9]
        self.assertEqual(len(wall_faces), count)
        self.assertEqual(len(plinth_faces), count)
        self.assertEqual(
            {tuple(sorted(nodes[n] for n in face)) for face in wall_faces},
            {tuple(sorted(nodes[n] for n in face)) for face in plinth_faces},
        )
        self.assertTrue(
            {n for face in wall_faces for n in face}.isdisjoint(
                {n for face in plinth_faces for n in face},
            ),
        )

    def test_exposed_slopes_are_loaded_by_side_and_wall_does_not_fill_shoulders(self):
        mesh = self.chamfered
        wall = self.config["geometry"]["wall"]
        owners = face_owners(mesh)
        slopes = Counter()
        for tag, face in mesh.boundaries:
            if tag == 1:
                continue
            radii = [math.hypot(*mesh.nodes[n][:2]) for n in face]
            heights = [mesh.nodes[n][2] for n in face]
            if len(face) != 4 or max(heights) - min(heights) < 1e-7:
                continue
            if min(radii) >= wall["upstream_radius_m"] - 1e-7:
                side = "upstream"
                expected_tag = 2
            elif max(radii) <= wall["downstream_radius_m"] + 1e-7:
                side = "downstream"
                expected_tag = 3
            else:
                continue
            stations = [wall["centerline_radius_m"] * math.atan2(*mesh.nodes[n][:2])
                        for n in face]
            if not math.isclose(max(stations)-min(stations), max(heights)-min(heights),
                                abs_tol=1e-7):
                continue
            if max(radii)-min(radii) < 1e-7:
                continue
            self.assertEqual(owners[tuple(sorted(face))], [1])
            self.assertEqual(tag, expected_tag)
            slopes[side] += 1
        self.assertEqual(slopes, {"upstream": 110, "downstream": 110})

    def test_configuration_enables_cutback_and_rejects_unknown_policy(self):
        self.assertEqual(self.config["geometry"]["plinth"]["wall_interface"], "cutback_45deg")
        config = json.loads(json.dumps(self.config))
        config["geometry"]["plinth"]["wall_interface"] = "add_material"
        with self.assertRaisesRegex(ConfigurationValidationError, "wall_interface"):
            validate_config(config)

    def test_explicit_stepped_mode_matches_omitted_policy(self):
        config = json.loads(json.dumps(self.config))
        config["geometry"]["plinth"]["wall_interface"] = "stepped"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config))
            stepped = build_combined_structure(root)
        self.assertIsNone(stepped.wall_plinth_cutback)
        self.assertEqual(stepped.nodes, self.original.nodes)
        self.assertEqual(stepped.cells, self.original.cells)
        self.assertEqual(stepped.region_ids, self.original.region_ids)
        self.assertEqual(stepped.boundaries, self.original.boundaries)

    def test_non_unit_step_run_fails_instead_of_silently_changing_angle(self):
        nodes = [(2*x, 2*y, z) for x, y, z in self.original.nodes]
        with self.assertRaisesRegex(ValueError, "rise equal"):
            cutback_wall_plinth(replace(self.original, nodes=nodes), 156.)
