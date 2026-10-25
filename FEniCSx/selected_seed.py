"""Prepare one immutable UI selection before launching either cracking backend."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from FEniCSx.automatic_crack import locate_crack
from FEniCSx.crack_contract import (
    CrackSpecification, PhaseFieldSolverOptions, freeze_selected_seed, validate_selected_seed,
)


def _hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def fields_hash(fields: tuple[np.ndarray, np.ndarray, np.ndarray]) -> str:
    """Hash physical source arrays independently of NPZ compression and timestamps."""
    points, stress, regions = (np.asarray(array) for array in fields)
    if (points.ndim != 2 or points.shape[1] != 3 or stress.shape != (len(points), 6)
            or regions.shape != (len(points),) or not len(points)
            or not np.isfinite(points).all() or not np.isfinite(stress).all()
            or not np.issubdtype(regions.dtype, np.integer) or np.any(regions < 0)):
        raise ValueError("Selected seed needs finite baseline coordinates/stress and integer region bits")
    digest = hashlib.sha256()
    for name, array, dtype in (("coordinates_m", points, "<f8"), ("stress_pa", stress, "<f8"),
                                ("node_regions", regions, "<i8")):
        values = np.ascontiguousarray(array, dtype=dtype)
        digest.update(json.dumps([name, list(values.shape), dtype]).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def prepare(config_bytes: bytes, fields: tuple[np.ndarray, np.ndarray, np.ndarray],
            location: tuple[float, float], final_load_substeps: int = 1) -> dict:
    """Choose the face once from the original baseline; no solver runtime needed."""
    config = json.loads(config_bytes)
    if config.get("crack_simulations"):
        raise ValueError("A shared UI seed cannot be combined with configured crack_simulations")
    source_hash = fields_hash(fields)
    spec, options, generation = locate_crack(config, location, fields)
    generation["specification"] = asdict(spec)
    generation["shared_seed"] = freeze_selected_seed(spec, generation, final_load_substeps)
    payload = {
        "schema_version": 1,
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "baseline_fields_sha256": source_hash,
        "final_load_substeps": final_load_substeps,
        "solver_options": asdict(options),
        "generation": generation,
    }
    # Canonical JSON types keep Windows and WSL consumers identical.
    payload = json.loads(json.dumps(payload, allow_nan=False))
    return {"payload": payload, "sha256": _hash(payload)}


def validate_prepared_seed(prepared: dict, config_bytes: bytes) -> dict:
    """Check the packet's integrity and configuration binding before using its metadata."""
    if not isinstance(prepared, dict) or not isinstance(prepared.get("payload"), dict):
        raise ValueError("Prepared UI seed must contain a payload object")
    payload = prepared["payload"]
    if prepared.get("sha256") != _hash(payload):
        raise ValueError("Prepared UI seed hash mismatch")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported prepared UI seed schema")
    if payload.get("configuration_sha256") != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError("Prepared UI seed configuration does not match the submitted snapshot")
    if (not isinstance(payload.get("generation"), dict)
            or not isinstance(payload.get("solver_options"), dict)
            or not isinstance(payload.get("baseline_fields_sha256"), str)):
        raise ValueError("Prepared UI seed has missing solver or source metadata")
    return payload


def consume(prepared: dict, config_bytes: bytes, fields: tuple[np.ndarray, np.ndarray, np.ndarray],
            final_load_substeps: int = 1) -> tuple[CrackSpecification, PhaseFieldSolverOptions, dict]:
    """Reconstruct frozen inputs without locating a face or reading mutable defaults."""
    payload = validate_prepared_seed(prepared, config_bytes)
    if payload["baseline_fields_sha256"] != fields_hash(fields):
        raise ValueError("Prepared UI seed baseline fields do not match this job")
    if payload["final_load_substeps"] != final_load_substeps:
        raise ValueError("Prepared UI seed load subdivision does not match this run")
    generation = json.loads(json.dumps(payload["generation"], allow_nan=False))
    spec_data = dict(generation["specification"])
    for key in ("centre_m", "size_m", "normal", "annular_sector"):
        spec_data[key] = tuple(spec_data[key])
    specification = CrackSpecification(**spec_data)
    options = PhaseFieldSolverOptions(**payload["solver_options"])
    validate_selected_seed(generation["shared_seed"])
    if freeze_selected_seed(specification, generation, final_load_substeps) != generation["shared_seed"]:
        raise ValueError("Prepared solver specification disagrees with the shared physical seed")
    return specification, options, generation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-fields", type=Path, required=True)
    parser.add_argument("--location", nargs=2, type=float, required=True, metavar=("Y", "Z"))
    parser.add_argument("--final-load-substeps", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.baseline_fields, allow_pickle=False) as archive:
        fields = tuple(archive[name] for name in ("coordinates_m", "stress_pa", "node_regions"))
    prepared = prepare(args.config.read_bytes(), fields, tuple(args.location), args.final_load_substeps)
    # Never overwrite a seed that either backend may already be consuming.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(prepared, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Frozen UI seed {prepared['payload']['generation']['shared_seed']['sha256']} at {args.output}")


if __name__ == "__main__":
    main()
