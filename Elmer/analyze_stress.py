from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import re
import struct

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from sketch_arch_dam_views import load_station_edges, plot_views, xyz


def latest_vtu_path(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("dam_results_t*.vtu"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return results_dir / "dam_results_t0001.vtu"


@dataclass
class DataArraySpec:
    name: str
    offset: int
    dtype: str
    components: int


def parse_vtu(path: Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    data = path.read_bytes()
    appended_tag = b'<AppendedData encoding="raw">'
    appended_start = data.index(appended_tag)
    appended_data_start = data.index(b"_", appended_start) + 1
    appended_data_end = data.index(b"</AppendedData>", appended_data_start)
    header = data[:appended_start].decode("utf-8", errors="replace")
    appended = data[appended_data_start:appended_data_end]

    pattern = re.compile(
        r'<DataArray type="(?P<dtype>[^"]+)"(?: Name="(?P<name>[^"]+)")?(?: NumberOfComponents="(?P<comps>\d+)")? format="appended" offset="(?P<offset>\d+)"/>'
    )
    arrays: dict[str, DataArraySpec] = {}
    unnamed_index = 0
    for match in pattern.finditer(header):
        name = match.group("name") or f"unnamed_{unnamed_index}"
        if match.group("name") is None:
            unnamed_index += 1
        arrays[name] = DataArraySpec(
            name=name,
            offset=int(match.group("offset")),
            dtype=match.group("dtype"),
            components=int(match.group("comps") or "1"),
        )

    def read_array(spec: DataArraySpec) -> np.ndarray:
        chunk_len = struct.unpack("<I", appended[spec.offset : spec.offset + 4])[0]
        start = spec.offset + 4
        chunk = appended[start : start + chunk_len]
        dtype_map = {
            "Float64": np.dtype("<f8"),
            "Int32": np.dtype("<i4"),
            "UInt8": np.dtype("u1"),
        }
        values = np.frombuffer(chunk, dtype=dtype_map[spec.dtype])
        if spec.components > 1:
            values = values.reshape((-1, spec.components))
        return values

    displacement = read_array(arrays["displacement"])
    points = read_array(arrays["unnamed_0"])
    connectivity = read_array(arrays["connectivity"])
    offsets = read_array(arrays["offsets"])
    cell_types = read_array(arrays["types"])

    cells: list[np.ndarray] = []
    start = 0
    for end, cell_type in zip(offsets.tolist(), cell_types.tolist()):
        if cell_type == 12:
            cells.append(connectivity[start:end])
        start = end

    return points, displacement, cells


def constitutive_matrix(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    lam = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))

    matrix = np.array(
        [
            [lam + 2.0 * mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2.0 * mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2.0 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ],
        dtype=float,
    )
    return matrix


def brick_center_stress(coords: np.ndarray, displacements: np.ndarray, material_matrix: np.ndarray) -> np.ndarray:
    natural_gradients = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    ) / 8.0

    jacobian = natural_gradients.T @ coords
    global_gradients = natural_gradients @ np.linalg.inv(jacobian)

    b_matrix = np.zeros((6, 24), dtype=float)
    for index, (dx, dy, dz) in enumerate(global_gradients):
        column = 3 * index
        b_matrix[0, column] = dx
        b_matrix[1, column + 1] = dy
        b_matrix[2, column + 2] = dz
        b_matrix[3, column] = dy
        b_matrix[3, column + 1] = dx
        b_matrix[4, column + 1] = dz
        b_matrix[4, column + 2] = dy
        b_matrix[5, column] = dz
        b_matrix[5, column + 2] = dx

    strain = b_matrix @ displacements.reshape(24)
    stress = material_matrix @ strain
    return stress


def affine_fit_stress(coords: np.ndarray, displacements: np.ndarray, material_matrix: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(coords)), coords])
    coeffs_x, _, _, _ = np.linalg.lstsq(design, displacements[:, 0], rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(design, displacements[:, 1], rcond=None)
    coeffs_z, _, _, _ = np.linalg.lstsq(design, displacements[:, 2], rcond=None)
    grad_u = np.array(
        [
            [coeffs_x[1], coeffs_x[2], coeffs_x[3]],
            [coeffs_y[1], coeffs_y[2], coeffs_y[3]],
            [coeffs_z[1], coeffs_z[2], coeffs_z[3]],
        ]
    )
    strain = np.array(
        [
            grad_u[0, 0],
            grad_u[1, 1],
            grad_u[2, 2],
            grad_u[0, 1] + grad_u[1, 0],
            grad_u[1, 2] + grad_u[2, 1],
            grad_u[0, 2] + grad_u[2, 0],
        ]
    )
    return material_matrix @ strain


def stress_tensor(voigt: np.ndarray) -> np.ndarray:
    sxx, syy, szz, sxy, syz, sxz = voigt.tolist()
    return np.array(
        [
            [sxx, sxy, sxz],
            [sxy, syy, syz],
            [sxz, syz, szz],
        ],
    )


def build_segment_collection(x_values: np.ndarray, y_values: np.ndarray, scalar_values: np.ndarray, cmap: str) -> LineCollection:
    points = np.column_stack([x_values, y_values]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_values = 0.5 * (scalar_values[:-1] + scalar_values[1:])
    collection = LineCollection(segments, cmap=cmap, linewidths=8)
    collection.set_array(segment_values)
    return collection


def plot_stress_distribution(element_rows: list[dict[str, float]], tensile_strength_pa: float, out_path: Path) -> None:
    centroids = np.array([[row["centroid_x_m"], row["centroid_y_m"], row["centroid_z_m"]] for row in element_rows])
    tension = np.array([row["principal_max_pa"] for row in element_rows]) / 1.0e6
    compression = -np.array([row["principal_min_pa"] for row in element_rows]) / 1.0e6
    tensile_strength_mpa = tensile_strength_pa / 1.0e6

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    fig.suptitle("Principal Stress Distribution", fontsize=16, fontweight="bold")

    plan_tension = build_segment_collection(centroids[:, 0], centroids[:, 1], tension, "Reds")
    axes[0, 0].add_collection(plan_tension)
    axes[0, 0].autoscale()
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_title("Plan: Tensile Principal Stress (MPa)")
    axes[0, 0].set_xlabel("X (m)")
    axes[0, 0].set_ylabel("Y (m)")
    axes[0, 0].grid(True, color="0.9", linewidth=0.6)
    fig.colorbar(plan_tension, ax=axes[0, 0], shrink=0.85)

    plan_compression = build_segment_collection(centroids[:, 0], centroids[:, 1], compression, "Blues")
    axes[0, 1].add_collection(plan_compression)
    axes[0, 1].autoscale()
    axes[0, 1].set_aspect("equal")
    axes[0, 1].set_title("Plan: Compressive Principal Stress Magnitude (MPa)")
    axes[0, 1].set_xlabel("X (m)")
    axes[0, 1].set_ylabel("Y (m)")
    axes[0, 1].grid(True, color="0.9", linewidth=0.6)
    fig.colorbar(plan_compression, ax=axes[0, 1], shrink=0.85)

    arc_step = np.linalg.norm(np.diff(centroids[:, :2], axis=0), axis=1)
    chainage = np.concatenate([[0.0], np.cumsum(arc_step)])
    axes[1, 0].plot(chainage, tension, color="#b22222", linewidth=2.2)
    axes[1, 0].fill_between(chainage, 0.0, tension, color="#e9967a", alpha=0.55)
    axes[1, 0].axhline(tensile_strength_mpa, color="#6a1b9a", linestyle="--", linewidth=1.6, label=f"Tensile strength {tensile_strength_mpa:.2f} MPa")
    axes[1, 0].set_title("Max Principal Tension by Arc Length")
    axes[1, 0].set_xlabel("Arc Length (m)")
    axes[1, 0].set_ylabel("Tension (MPa)")
    axes[1, 0].grid(True, color="0.9", linewidth=0.6)
    axes[1, 0].legend(frameon=False, loc="upper right")

    axes[1, 1].plot(chainage, compression, color="#0b5394", linewidth=2.2)
    axes[1, 1].fill_between(chainage, 0.0, compression, color="#9fc5e8", alpha=0.6)
    axes[1, 1].set_title("Min Principal Compression Magnitude by Arc Length")
    axes[1, 1].set_xlabel("Arc Length (m)")
    axes[1, 1].set_ylabel("Compression (MPa)")
    axes[1, 1].grid(True, color="0.9", linewidth=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def plot_critical_views(mesh_path: Path, tension_point: np.ndarray, compression_point: np.ndarray, out_path: Path) -> None:
    stations = load_station_edges(mesh_path)
    inner_base = [s["inner_base"] for s in stations]
    outer_base = [s["outer_base"] for s in stations]
    inner_top = [s["inner_top"] for s in stations]
    outer_top = [s["outer_top"] for s in stations]

    fig = plt.figure(figsize=(14, 10), dpi=180)
    fig.suptitle("Critical Stress Locations", fontsize=16, fontweight="bold")

    ax1 = fig.add_subplot(2, 2, 1)
    x_ib, y_ib, _ = xyz(inner_base)
    x_ob, y_ob, _ = xyz(outer_base)
    ax1.plot(x_ib, y_ib, "k-", linewidth=1.6)
    ax1.plot(x_ob, y_ob, "k-", linewidth=1.6)
    ax1.scatter(tension_point[0], tension_point[1], color="#d73027", s=70, label="Max tension")
    ax1.scatter(compression_point[0], compression_point[1], color="#08519c", s=70, label="Max compression")
    ax1.set_title("Plan View")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.axis("equal")
    ax1.grid(True, color="0.9", linewidth=0.6)
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(2, 2, 2)
    x_it, _, z_it = xyz(inner_top)
    x_ot, _, z_ot = xyz(outer_top)
    x_ib, _, z_ib = xyz(inner_base)
    x_ob, _, z_ob = xyz(outer_base)
    ax2.plot(x_ib, z_ib, "k-", linewidth=1.3)
    ax2.plot(x_ob, z_ob, "k-", linewidth=1.3)
    ax2.plot(x_it, z_it, "k--", linewidth=1.0)
    ax2.plot(x_ot, z_ot, "k--", linewidth=1.0)
    ax2.scatter(tension_point[0], tension_point[2], color="#d73027", s=70)
    ax2.scatter(compression_point[0], compression_point[2], color="#08519c", s=70)
    ax2.set_title("Elevation (X-Z)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.grid(True, color="0.9", linewidth=0.6)

    ax3 = fig.add_subplot(2, 2, 3)
    _, y_it, z_it = xyz(inner_top)
    _, y_ot, z_ot = xyz(outer_top)
    _, y_ib, z_ib = xyz(inner_base)
    _, y_ob, z_ob = xyz(outer_base)
    ax3.plot(y_ib, z_ib, "k-", linewidth=1.3)
    ax3.plot(y_ob, z_ob, "k-", linewidth=1.3)
    ax3.plot(y_it, z_it, "k--", linewidth=1.0)
    ax3.plot(y_ot, z_ot, "k--", linewidth=1.0)
    ax3.scatter(tension_point[1], tension_point[2], color="#d73027", s=70)
    ax3.scatter(compression_point[1], compression_point[2], color="#08519c", s=70)
    ax3.set_title("Elevation (Y-Z)")
    ax3.set_xlabel("Y (m)")
    ax3.set_ylabel("Z (m)")
    ax3.grid(True, color="0.9", linewidth=0.6)

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    for edge in (inner_base, outer_base, inner_top, outer_top):
        x_vals, y_vals, z_vals = xyz(edge)
        ax4.plot(x_vals, y_vals, z_vals, color="k", linewidth=1.2)
    ax4.scatter(tension_point[0], tension_point[1], tension_point[2], color="#d73027", s=80)
    ax4.scatter(compression_point[0], compression_point[1], compression_point[2], color="#08519c", s=80)
    ax4.set_title("Isometric")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.set_zlabel("Z (m)")
    ax4.view_init(elev=24, azim=-58)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def create_report_sheet(
    orientation_path: Path,
    distribution_path: Path,
    critical_path: Path,
    summary: dict,
    tensile_strength_pa: float,
    expected_tension_mpa: float | None,
    expected_compression_mpa: float | None,
    out_path: Path,
) -> None:
    orientation = plt.imread(orientation_path)
    distribution = plt.imread(distribution_path)
    critical = plt.imread(critical_path)

    fig = plt.figure(figsize=(16, 11), dpi=180)
    fig.suptitle("Arch Dam Stress Review Sheet", fontsize=18, fontweight="bold")

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(orientation)
    ax1.set_title("Geometry Orientation")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(distribution)
    ax2.set_title("Principal Stress Distribution")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(critical)
    ax3.set_title("Critical Stress Locations")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    max_tension_mpa = summary["max_tensile_principal_stress_pa"]["value_pa"] / 1.0e6
    max_compression_mpa = summary["max_compressive_principal_stress_pa"]["value_pa"] / 1.0e6
    utilization = max_tension_mpa / (tensile_strength_pa / 1.0e6)
    lines = [
        "Load Case Summary",
        "",
        f"Max tensile principal stress: {max_tension_mpa:.3f} MPa",
        f"Max compressive principal stress: {max_compression_mpa:.3f} MPa",
        f"Concrete tensile strength threshold: {tensile_strength_pa / 1.0e6:.3f} MPa",
        f"Tension utilization: {utilization:.1%}",
        "",
        "Critical centroid location (m):",
        f"X = {summary['max_tensile_principal_stress_pa']['centroid_xyz_m'][0]:.2f}",
        f"Y = {summary['max_tensile_principal_stress_pa']['centroid_xyz_m'][1]:.2f}",
        f"Z = {summary['max_tensile_principal_stress_pa']['centroid_xyz_m'][2]:.2f}",
    ]
    if expected_tension_mpa is not None:
        lines.extend([
            "",
            f"Client expected tension: {expected_tension_mpa:.2f} MPa",
            f"Delta: {max_tension_mpa - expected_tension_mpa:+.3f} MPa",
        ])
    if expected_compression_mpa is not None:
        lines.extend([
            f"Client expected compression: {expected_compression_mpa:.2f} MPa",
            f"Delta: {max_compression_mpa - expected_compression_mpa:+.3f} MPa",
        ])
    ax4.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=12, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def load_tensile_strength(load_cases_path: Path) -> float:
    payload = json.loads(load_cases_path.read_text())
    return float(payload["material"]["tensile_strength"])


def build_detailing_assessment(summary: dict, mesh_path: Path) -> dict:
    meta_path = mesh_path.with_name(f"{mesh_path.stem}_meta.json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    dam_height = float(metadata.get("dam_height_m", 30.0))
    wall_thickness = float(metadata.get("wall_thickness_m", 4.0))
    crest_extra_thickness = float(metadata.get("crest_extra_thickness_m", 0.0))
    crest_detail_height = float(metadata.get("crest_detail_height_m", 0.0))
    critical_z = float(summary["max_tensile_principal_stress_pa"]["centroid_xyz_m"][2])
    crest_zone = critical_z > dam_height - 0.2 * dam_height
    notes = [
        f"Model uses a base wall thickness of {wall_thickness:.2f} m.",
    ]
    if crest_extra_thickness > 0.0 and crest_detail_height > 0.0:
        notes.append(
            f"Model includes a crest-thickening surrogate: +{crest_extra_thickness:.2f} m over the top {crest_detail_height:.2f} m of wall height."
        )
    else:
        notes.append("Model does not include crest thickening.")
    notes.extend([
        "Model does not include galleries, fillets, or abutment transition stiffening.",
        "Model uses compliant base springs, explicit abutment spring restraints, and surrogate curved-wall geometry.",
    ])
    if crest_zone:
        notes.append("Critical tension lies in the upper 20% of the wall height, so crest-detail simplifications are likely influencing local stress concentration.")
    return {
        "critical_zone": "crest" if crest_zone else "body",
        "notes": notes,
    }


def analyze(
    vtu_path: Path,
    out_dir: Path,
    youngs_modulus: float,
    poisson_ratio: float,
    mesh_path: Path,
    load_cases_path: Path,
    expected_tension_mpa: float | None,
    expected_compression_mpa: float | None,
    exclude_below_z: float | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    points, displacement, hex_cells = parse_vtu(vtu_path)
    material_matrix = constitutive_matrix(youngs_modulus, poisson_ratio)

    element_rows: list[dict[str, float]] = []
    max_tension = {"element_id": -1, "value_pa": float("-inf"), "centroid_xyz_m": [0.0, 0.0, 0.0]}
    max_compression = {"element_id": -1, "value_pa": float("inf"), "centroid_xyz_m": [0.0, 0.0, 0.0]}
    max_tension_affine = {"element_id": -1, "value_pa": float("-inf")}
    max_compression_affine = {"element_id": -1, "value_pa": float("inf")}

    for element_id, node_ids in enumerate(hex_cells, start=1):
        coords = points[node_ids]
        element_displacements = displacement[node_ids]
        stress = brick_center_stress(coords, element_displacements, material_matrix)
        affine_stress = affine_fit_stress(coords, element_displacements, material_matrix)
        affine_principal = np.linalg.eigvalsh(stress_tensor(affine_stress))
        centroid = coords.mean(axis=0)

        row = {
            "element_id": element_id,
            "centroid_x_m": float(centroid[0]),
            "centroid_y_m": float(centroid[1]),
            "centroid_z_m": float(centroid[2]),
            "sigma_xx_pa": float(stress[0]),
            "sigma_yy_pa": float(stress[1]),
            "sigma_zz_pa": float(stress[2]),
            "tau_xy_pa": float(stress[3]),
            "tau_yz_pa": float(stress[4]),
            "tau_xz_pa": float(stress[5]),
            "principal_min_pa": float(affine_principal[0]),
            "principal_mid_pa": float(affine_principal[1]),
            "principal_max_pa": float(affine_principal[2]),
            "affine_principal_min_pa": float(affine_principal[0]),
            "affine_principal_mid_pa": float(affine_principal[1]),
            "affine_principal_max_pa": float(affine_principal[2]),
        }
        if exclude_below_z is not None and centroid[2] <= exclude_below_z:
            continue

        element_rows.append(row)

        if affine_principal[2] > max_tension["value_pa"]:
            max_tension = {
                "element_id": element_id,
            "value_pa": float(affine_principal[2]),
                "centroid_xyz_m": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
            }
        if affine_principal[0] < max_compression["value_pa"]:
            max_compression = {
                "element_id": element_id,
            "value_pa": float(affine_principal[0]),
                "centroid_xyz_m": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
            }
        if affine_principal[2] > max_tension_affine["value_pa"]:
            max_tension_affine = {"element_id": element_id, "value_pa": float(affine_principal[2])}
        if affine_principal[0] < max_compression_affine["value_pa"]:
            max_compression_affine = {"element_id": element_id, "value_pa": float(affine_principal[0])}

    if not element_rows:
        raise ValueError("The selected exclusion range removed every volume element")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "principal_stress_by_element.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(element_rows[0].keys()))
        writer.writeheader()
        writer.writerows(element_rows)

    summary = {
        "source_vtu": str(vtu_path),
        "element_count": len(element_rows),
        "max_tensile_principal_stress_pa": max_tension,
        "max_compressive_principal_stress_pa": max_compression,
        "independent_affine_check": {
            "max_tensile_principal_stress_pa": max_tension_affine,
            "max_compressive_principal_stress_pa": max_compression_affine,
        },
        "note": "Positive principal stress indicates tension. Negative principal stress indicates compression.",
    }
    if exclude_below_z is not None:
        summary["exclusion"] = {
            "excluded_centroid_z_m_lte": exclude_below_z,
            "description": "Excludes the wall, wedge, and plinth transition region at and below the stated elevation.",
        }
    tensile_strength = load_tensile_strength(load_cases_path)
    summary["tensile_strength_pa"] = tensile_strength
    summary["tension_utilization"] = max_tension["value_pa"] / tensile_strength
    if expected_tension_mpa is not None:
        summary["client_expected_tension_mpa"] = expected_tension_mpa
    if expected_compression_mpa is not None:
        summary["client_expected_compression_mpa"] = expected_compression_mpa
    summary["detailing_assessment"] = build_detailing_assessment(summary, mesh_path)
    summary_path = out_dir / "stress_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    stress_plot_path = out_dir / "principal_stress_distribution.png"
    plot_stress_distribution(element_rows, tensile_strength, stress_plot_path)

    critical_plot_path = out_dir / "critical_stress_locations.png"
    plot_critical_views(
        mesh_path,
        np.array(max_tension["centroid_xyz_m"]),
        np.array(max_compression["centroid_xyz_m"]),
        critical_plot_path,
    )

    orientation_path = out_dir / "arch_dam_orientations.png"
    plot_views(mesh_path, orientation_path)

    report_path = out_dir / "client_stress_report.png"
    create_report_sheet(
        orientation_path,
        stress_plot_path,
        critical_plot_path,
        summary,
        tensile_strength,
        expected_tension_mpa,
        expected_compression_mpa,
        report_path,
    )

    return csv_path, summary_path, stress_plot_path, critical_plot_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute tensile and compressive principal stress from Elmer VTU displacement output.")
    parser.add_argument(
        "--vtu",
        type=Path,
        default=latest_vtu_path(Path(__file__).resolve().parent / "results"),
        help="Path to the Elmer VTU result file",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for CSV and JSON outputs",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path(__file__).resolve().parent / "curved_dam_mesh.msh",
        help="Path to the Gmsh mesh file used for architecture-style overlays",
    )
    parser.add_argument(
        "--load-cases",
        type=Path,
        default=Path(__file__).resolve().parent / "load_cases.json",
        help="Path to load_cases.json for threshold overlays",
    )
    parser.add_argument("--youngs-modulus", type=float, default=3.5e10)
    parser.add_argument("--poisson-ratio", type=float, default=0.2)
    parser.add_argument("--expected-tension-mpa", type=float, default=1.2)
    parser.add_argument("--expected-compression-mpa", type=float, default=-3.33)
    parser.add_argument(
        "--exclude-below-z",
        type=float,
        help="Exclude volume elements whose centroid elevation is at or below this value in metres.",
    )
    args = parser.parse_args()

    csv_path, summary_path, stress_plot_path, critical_plot_path, report_path = analyze(
        args.vtu,
        args.out_dir,
        args.youngs_modulus,
        args.poisson_ratio,
        args.mesh,
        args.load_cases,
        args.expected_tension_mpa,
        args.expected_compression_mpa,
        args.exclude_below_z,
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {stress_plot_path}")
    print(f"Wrote {critical_plot_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()