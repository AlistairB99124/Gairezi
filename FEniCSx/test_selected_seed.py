"""Freeze UI placement before either solver, with source and consumer identity checks."""
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Elmer.model_config import load_config
from FEniCSx.selected_seed import consume, fields_hash, main, prepare, validate_prepared_seed


class PreparedSeedTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.config["geometry"]["wall"].pop("thickness_steps", None)
        self.config_bytes = json.dumps(self.config).encode("utf-8")
        theta, z = 108.5 / 78, -27.5
        radii = np.array([76., 77., 78., 79., 80.])
        self.location = (78 * math.cos(theta), z)
        points = np.column_stack((radii * math.sin(theta), radii * math.cos(theta),
                                  np.full(5, z)))
        stress = np.zeros((5, 6))
        stress[:, 0] = np.arange(1, 6) * 1e5
        self.fields = points, stress, np.ones(5, dtype=int)

    def test_consumption_does_not_rechoose_face_or_read_mutable_defaults(self):
        prepared = prepare(self.config_bytes, self.fields, self.location, 4)
        with patch("FEniCSx.selected_seed.locate_crack", side_effect=AssertionError("must not relocate")):
            spec, options, generation = consume(
                json.loads(json.dumps(prepared)), self.config_bytes, self.fields, 4,
            )
        self.assertEqual(list(spec.annular_sector), generation["shared_seed"]["payload"]["geometry"]["annular_sector"])
        self.assertEqual(generation["selected_location_m"], {"y": self.location[0], "z": self.location[1]})
        self.assertEqual(generation["shared_seed"]["payload"]["loading"]["load_factors"][-4:],
                         [.925, .95, .975, 1.])
        self.assertEqual(asdict(options), prepared["payload"]["solver_options"])
        self.assertEqual(prepare(self.config_bytes, self.fields, self.location, 4), prepared)

    def test_modified_config_fields_steps_or_prepared_payload_are_rejected(self):
        prepared = prepare(self.config_bytes, self.fields, self.location)
        with self.assertRaisesRegex(ValueError, "configuration"):
            consume(prepared, self.config_bytes + b" ", self.fields)
        fields = tuple(array.copy() for array in self.fields)
        fields[1][0, 0] += 1.
        with self.assertRaisesRegex(ValueError, "baseline fields"):
            consume(prepared, self.config_bytes, fields)
        with self.assertRaisesRegex(ValueError, "subdivision"):
            consume(prepared, self.config_bytes, self.fields, 2)
        changed = copy.deepcopy(prepared)
        changed["payload"]["generation"]["selected_location_m"]["z"] = -26.
        with self.assertRaisesRegex(ValueError, "hash"):
            consume(changed, self.config_bytes, self.fields)

    def test_rehashed_packet_with_inconsistent_solver_geometry_is_rejected(self):
        prepared = prepare(self.config_bytes, self.fields, self.location)
        prepared["payload"]["generation"]["specification"]["size_m"][2] *= 2
        prepared["sha256"] = hashlib.sha256(json.dumps(
            prepared["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            consume(prepared, self.config_bytes, self.fields)

    def test_source_hash_is_independent_of_archive_encoding_and_memory_layout(self):
        expected = fields_hash(self.fields)
        self.assertEqual(fields_hash(tuple(np.asfortranarray(array) for array in self.fields)), expected)
        self.assertEqual(fields_hash((self.fields[0], self.fields[1], self.fields[2].astype("<i4"))), expected)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fields.npz"
            np.savez_compressed(path, coordinates_m=self.fields[0], stress_pa=self.fields[1],
                                node_regions=self.fields[2])
            with np.load(path) as archive:
                restored = tuple(archive[name] for name in ("coordinates_m", "stress_pa", "node_regions"))
            self.assertEqual(fields_hash(restored), expected)

    def test_invalid_source_fields_and_configured_cracks_are_rejected(self):
        fields = tuple(array.copy() for array in self.fields)
        fields[1][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            prepare(self.config_bytes, fields, self.location)
        self.config["crack_simulations"] = [{"name": "not_a_UI_selection"}]
        with self.assertRaisesRegex(ValueError, "configured"):
            prepare(json.dumps(self.config).encode(), self.fields, self.location)

    def test_malformed_packet_is_explicitly_rejected(self):
        for malformed in (None, [], {}, {"payload": []}, {"payload": {}}):
            with self.subTest(packet=malformed), self.assertRaises(ValueError):
                validate_prepared_seed(malformed, self.config_bytes)

    def test_cli_writes_once_and_does_not_overwrite_a_seed_in_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, fields, output = root / "config.json", root / "fields.npz", root / "seed.json"
            config.write_bytes(self.config_bytes)
            np.savez_compressed(fields, coordinates_m=self.fields[0], stress_pa=self.fields[1],
                                node_regions=self.fields[2])
            arguments = ["selected_seed", "--config", str(config), "--baseline-fields", str(fields),
                         "--location", *map(str, self.location), "--output", str(output)]
            with patch("sys.argv", arguments):
                main()
                contents = output.read_bytes()
                consume(json.loads(contents), self.config_bytes, self.fields)
                with self.assertRaises(FileExistsError):
                    main()
                self.assertEqual(output.read_bytes(), contents)


if __name__ == "__main__":
    unittest.main()
