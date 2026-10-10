"""Compare the Code_aster baseline with the Elmer combined-structure result, node by node."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
ELMER_DIR = ROOT / "Elmer" / "structure_output" / "combined_solver" / "results"
MATCH_TOLERANCE_M = 1e-5
VTK_TYPES = {"Float64": "<f8", "Float32": "<f4", "Int32": "<i4", "UInt8": "u1"}


def read_vtu(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read Elmer's appended-raw VTU (UInt32 size prefix per array); meshio mis-sizes these files."""
    raw = path.read_bytes()
    marker = raw.index(b"<AppendedData")
    start = raw.index(b"_", marker) + 1
    header = raw[:marker].decode("ascii", errors="replace")
    point_count = int(re.search(r'NumberOfPoints="(\d+)"', header).group(1))
    arrays: dict[str, np.ndarray] = {}
    points = None
    for tag in re.findall(r"<DataArray[^>]*/>", header):
        attributes = dict(re.findall(r'(\w+)="([^"]*)"', tag))
        offset = start + int(attributes["offset"])
        size = int(np.frombuffer(raw, dtype="<u4", count=1, offset=offset)[0])
        dtype = np.dtype(VTK_TYPES[attributes["type"]])
        values = np.frombuffer(raw, dtype=dtype, count=size // dtype.itemsize, offset=offset + 4)
        name = attributes.get("Name")
        if name is None and attributes.get("NumberOfComponents") == "3":
            points = values.reshape(-1, 3)
        elif name and values.size % point_count == 0 and values.size // point_count in (1, 3):
            arrays[name] = values.reshape(point_count, -1)
    if points is None or len(points) != point_count:
        raise ValueError(f"Could not locate points in {path}")
    return points, arrays


def load_elmer(results_dir: Path, reference_points: np.ndarray) -> dict[str, np.ndarray]:
    """Merge the MPI partitions and return nodal fields in the order of reference_points."""
    tree = cKDTree(reference_points)
    count = len(reference_points)
    fields = {"displacement": np.full((count, 3), np.nan), "stress": np.full((count, 6), np.nan)}
    max_shared_mismatch = 0.0
    for path in sorted(results_dir.glob("combined_results_mpi_4np*_t0001.vtu")):
        points, data = read_vtu(path)
        # Elmer writes the displaced mesh; undo it to recover the undeformed node position.
        distance, index = tree.query(points - data["displacement"])
        if distance.max() > MATCH_TOLERANCE_M:
            raise ValueError(f"{path.name}: node match distance {distance.max():.3e} m")
        tensor = np.hstack([data[f"stress_{c}"] for c in ("xx", "yy", "zz", "xy", "xz", "yz")])
        for name, values in (("displacement", data["displacement"]), ("stress", tensor)):
            previous = fields[name][index]
            seen = ~np.isnan(previous[:, 0])
            if seen.any():
                max_shared_mismatch = max(max_shared_mismatch, float(np.abs(previous[seen] - values[seen]).max()))
            fields[name][index] = values
    if np.isnan(fields["displacement"]).any():
        raise ValueError("Some Aster nodes have no Elmer value")
    fields["shared_node_mismatch"] = max_shared_mismatch
    return fields


def principal_stresses(stress: np.ndarray) -> np.ndarray:
    xx, yy, zz, xy, xz, yz = stress.T
    tensors = np.stack([
        np.stack([xx, xy, xz], axis=1),
        np.stack([xy, yy, yz], axis=1),
        np.stack([xz, yz, zz], axis=1),
    ], axis=1)
    return np.linalg.eigvalsh(tensors)[:, ::-1]


def von_mises(stress: np.ndarray) -> np.ndarray:
    xx, yy, zz, xy, xz, yz = stress.T
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy**2 + xz**2 + yz**2))


def describe(diff: np.ndarray, scale: float) -> dict[str, float]:
    magnitude = np.abs(diff).ravel()
    return {
        "rms": float(np.sqrt(np.mean(diff**2))),
        "p99_abs": float(np.percentile(magnitude, 99)),
        "max_abs": float(magnitude.max()),
        "max_abs_over_reference_max": float(magnitude.max() / scale),
        "rms_over_reference_max": float(np.sqrt(np.mean(diff**2)) / scale),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aster", type=Path, default=ROOT / "Aster" / "results" / "00_elastic_baseline" / "aster_result.npz")
    parser.add_argument("--inputs", type=Path, default=ROOT / "Aster" / "results" / "00_elastic_baseline" / "case_inputs.json")
    parser.add_argument("--elmer-dir", type=Path, default=ELMER_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "Aster" / "results" / "00_elastic_baseline" / "comparison_summary.json")
    args = parser.parse_args()

    aster = np.load(args.aster)
    points = aster["points"]
    elmer = load_elmer(args.elmer_dir, points)

    u_aster = aster["displacement"]
    s_aster = aster["stress"]
    u_elmer, s_elmer = elmer["displacement"], elmer["stress"]
    p_aster, p_elmer = principal_stresses(s_aster), principal_stresses(s_elmer)

    summary: dict[str, object] = {
        "node_count": len(points),
        "elmer_shared_node_max_mismatch": elmer["shared_node_mismatch"],
    }
    u_scale = np.linalg.norm(u_elmer, axis=1).max()
    summary["displacement_m"] = {
        "elmer_max_magnitude": float(u_scale),
        "aster_max_magnitude": float(np.linalg.norm(u_aster, axis=1).max()),
        "difference": describe(u_aster - u_elmer, u_scale),
    }
    s_scale = np.abs(s_elmer).max()
    summary["stress_components_pa"] = {
        "elmer_max_abs": float(s_scale),
        "aster_max_abs": float(np.abs(s_aster).max()),
        "difference": describe(s_aster - s_elmer, s_scale),
    }
    summary["principal_stress_pa"] = {
        "elmer_max_tension": float(p_elmer[:, 0].max()),
        "aster_max_tension": float(p_aster[:, 0].max()),
        "elmer_max_compression": float(p_elmer[:, 2].min()),
        "aster_max_compression": float(p_aster[:, 2].min()),
        "elmer_location_of_max_tension": points[p_elmer[:, 0].argmax()].tolist(),
        "aster_location_of_max_tension": points[p_aster[:, 0].argmax()].tolist(),
    }
    summary["von_mises_pa"] = {
        "elmer_max": float(von_mises(s_elmer).max()),
        "aster_max": float(von_mises(s_aster).max()),
    }

    inputs = json.loads(args.inputs.read_text())
    spring_nodes = np.array(inputs["spring_nodes"])
    areas = np.array(inputs["spring_areas_m2"])
    k = np.array(inputs["foundation_spring_n_per_m3"])
    spring_elmer = -(u_elmer[spring_nodes] * k) * areas[:, None]
    spring_aster = -(aster["displacement"][spring_nodes] * k) * areas[:, None]
    summary["foundation_reaction_n"] = {
        "elmer_from_displacement": spring_elmer.sum(axis=0).tolist(),
        "aster_from_displacement": spring_aster.sum(axis=0).tolist(),
        "aster_reac_noda_on_spring_nodes": aster["reaction"][spring_nodes].sum(axis=0).tolist(),
        "aster_reac_noda_sum_all_nodes": aster["reaction"].sum(axis=0).tolist(),
    }

    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
