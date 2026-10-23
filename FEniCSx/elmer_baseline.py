"""Consumer-owned Elmer baseline import; does not initialize fracture state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath
import xml.etree.ElementTree as ET

import numpy as np

from elastic_parity import read_reference


def baseline_files(reference: Path) -> list[Path]:
    """Only accept partition files inside the result directory."""
    if reference.suffix.lower() != ".pvtu":
        raise ValueError("Elmer baseline must be a PVTU wrapper.")
    root = reference.parent.resolve()
    pieces = ET.parse(reference).findall(".//Piece")
    if not pieces:
        raise ValueError("Elmer baseline PVTU references no partition pieces.")
    files = [reference]
    for piece in pieces:
        source = piece.get("Source", "")
        if not source or Path(source).is_absolute() or PureWindowsPath(source).drive:
            raise ValueError(f"Invalid Elmer partition source: {source!r}")
        partition = (root / source).resolve()
        if not partition.is_relative_to(root):
            raise ValueError(f"Elmer partition escapes result directory: {source!r}")
        if not partition.is_file() or partition.stat().st_size == 0:
            raise ValueError(f"Missing or empty Elmer partition: {partition}")
        files.append(partition)
    return files


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def concrete_nodal_peak(points: np.ndarray, stress: np.ndarray, node_regions: np.ndarray) -> dict:
    """Peak nodal principal stress over wall, matching the displayed Elmer field.

    Wall and bedrock nodes are distinct across a spring seat. Shared nodes on a bonded
    seat carry both wall and bedrock membership bits.
    """
    concrete = np.flatnonzero((node_regions & 1) > 0)
    if not len(concrete):
        raise ValueError("Elmer baseline has no wall nodes.")
    xx, yy, zz, xy, yz, xz = stress[concrete].T
    tensors = np.stack((
        np.stack((xx, xy, xz), axis=-1), np.stack((xy, yy, yz), axis=-1), np.stack((xz, yz, zz), axis=-1),
    ), axis=-2)
    principal = np.linalg.eigvalsh(tensors)[:, -1]
    index = int(concrete[int(np.argmax(principal))])
    return {
        "node_id": index + 1,
        "region_id": 1 if node_regions[index] & 1 else 2,
        "value_pa": float(principal.max()),
        "position_xyz_m": points[index].tolist(),
    }


def import_baseline(structure, reference: Path, config: Path, directory: Path) -> dict:
    """Map Elmer fields preserving either bonded or legacy split topology.

    Coordinates are undeformed metres; stress is Pa and tension is positive.
    The existing fracture solve remains independently loaded from config.
    """
    files = baseline_files(reference)
    if not structure.cells:
        raise ValueError("Elmer baseline contains no source volume elements.")
    hashes = {str(path.resolve()): file_hash(path) for path in files}
    displacement, stress, node_regions = read_reference(structure, reference)
    points = np.asarray(structure.nodes, dtype=float)
    if (
        points.shape != (len(points), 3)
        or displacement.shape != points.shape
        or stress.shape != (len(points), 6)
        or node_regions.shape != (len(points),)
        or not all(np.isfinite(array).all() for array in (points, displacement, stress))
    ):
        raise ValueError("Elmer baseline has invalid shapes or nonfinite coordinates/fields.")
    three_body = getattr(structure, "three_body_config", None) is not None
    peak = concrete_nodal_peak(points, stress, node_regions) if three_body else None
    for element_id, (_, nodes) in enumerate([] if three_body else structure.cells, start=1):
        ids = list(nodes)
        xx, yy, zz, xy, yz, xz = stress[ids].mean(axis=0)
        principal = float(np.linalg.eigvalsh(
            [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
        )[-1])
        if peak is None or principal > peak["value_pa"]:
            peak = {
                "element_id": element_id,
                "region_id": int(structure.region_ids[element_id - 1]),
                "value_pa": principal,
                "centroid_xyz_m": points[ids].mean(axis=0).tolist(),
            }

    if any(file_hash(path) != hashes[str(path.resolve())] for path in files):
        raise ValueError("Elmer baseline files changed during import.")
    report = {
        "schema_version": 1,
        "source_pvtu": str(reference.resolve()),
        "source_sha256": hashes,
        "configuration_sha256": file_hash(config),
        "node_count": len(points),
        "element_count": len(structure.cells),
        "coordinate_frame": "undeformed source mesh; crest z=0",
        "units": {"coordinates": "m", "displacement": "m", "stress": "Pa"},
        "stress_components": ["xx", "yy", "zz", "xy", "yz", "xz"],
        "max_principal_stress": peak,
        "has_tension": peak["value_pa"] > 0,
        "peak_method": (
            "maximum principal eigenvalue of the Elmer nodal stress tensor over wall nodes"
            if three_body else "maximum principal eigenvalue of element-mean nodal stress tensor"
        ),
        "field_archive": "elmer_baseline_fields.npz",
        "node_regions_encoding": "membership bits: 1=wall, 2=bedrock, 4=foundation; combined bits for shared nodes",
        "usage": "baseline fields are not fracture initial conditions; the user-selected crack location is recorded separately",
        "engineering_validated": False,
    }
    np.savez_compressed(
        directory / report["field_archive"],
        coordinates_m=points, displacement_m=displacement,
        stress_pa=stress, node_regions=node_regions,
    )
    (directory / "elmer_baseline.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report
