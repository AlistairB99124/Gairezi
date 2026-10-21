from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import re
import struct

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import numpy as np

from kaleido_safety import write_image_with_timeout
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


def _parse_vtu_piece(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, np.ndarray]]]:
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
    stress_components = np.column_stack([
        read_array(arrays[name])
        for name in ("stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_yz", "stress_xz")
    ])
    points = read_array(arrays["unnamed_0"])
    connectivity = read_array(arrays["connectivity"])
    offsets = read_array(arrays["offsets"])
    cell_types = read_array(arrays["types"])

    cells: list[tuple[int, np.ndarray]] = []
    start = 0
    for end, cell_type in zip(offsets.tolist(), cell_types.tolist()):
        if cell_type in (10, 12, 13, 14):  # VTK tetrahedron, hexahedron, wedge, pyramid
            cells.append((cell_type, connectivity[start:end]))
        start = end

    return points, displacement, stress_components, cells


def parse_vtu(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, np.ndarray]]]:
    if path.suffix != ".pvtu":
        return _parse_vtu_piece(path)

    manifest = path.read_text()
    piece_paths = [path.parent / source for source in re.findall(r'<Piece Source="([^"]+)"/>', manifest)]
    if not piece_paths:
        raise ValueError(f"PVTU manifest has no result pieces: {path}")

    points_parts: list[np.ndarray] = []
    displacement_parts: list[np.ndarray] = []
    stress_parts: list[np.ndarray] = []
    cells: list[tuple[int, np.ndarray]] = []
    point_offset = 0
    for piece_path in piece_paths:
        points, displacement, stress_components, piece_cells = _parse_vtu_piece(piece_path)
        points_parts.append(points)
        displacement_parts.append(displacement)
        stress_parts.append(stress_components)
        cells.extend((cell_type, connectivity + point_offset) for cell_type, connectivity in piece_cells)
        point_offset += len(points)
    return np.vstack(points_parts), np.vstack(displacement_parts), np.vstack(stress_parts), cells


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


def plot_stress_distribution(element_rows: list[dict[str, float]], tensile_strength_pa: float, out_path: Path) -> None:
    centroids = np.array([[row["centroid_x_m"], row["centroid_y_m"], row["centroid_z_m"]] for row in element_rows])
    tension = np.array([row["principal_max_pa"] for row in element_rows]) / 1.0e6
    compression = -np.array([row["principal_min_pa"] for row in element_rows]) / 1.0e6
    tensile_strength_mpa = tensile_strength_pa / 1.0e6

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Plan: Tensile Principal Stress (MPa)",
            "Plan: Compressive Principal Stress Magnitude (MPa)",
            "Max Principal Tension by Arc Length",
            "Min Principal Compression Magnitude by Arc Length",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode="markers",
            marker=dict(color=tension, colorscale="Reds", size=5, colorbar=dict(title="MPa", x=0.44, len=0.42, y=0.79)),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode="markers",
            marker=dict(color=compression, colorscale="Blues", size=5, colorbar=dict(title="MPa", x=1.0, len=0.42, y=0.79)),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    arc_step = np.linalg.norm(np.diff(centroids[:, :2], axis=0), axis=1)
    chainage = np.concatenate([[0.0], np.cumsum(arc_step)])
    fig.add_trace(
        go.Scatter(x=chainage, y=tension, mode="lines", line=dict(color="#b22222", width=2.2), fill="tozeroy", fillcolor="rgba(233,150,122,0.55)", showlegend=False),
        row=2,
        col=1,
    )
    fig.add_hline(y=tensile_strength_mpa, line=dict(color="#6a1b9a", dash="dash", width=1.6), annotation_text=f"Tensile strength {tensile_strength_mpa:.2f} MPa", row=2, col=1)

    fig.add_trace(
        go.Scatter(x=chainage, y=compression, mode="lines", line=dict(color="#0b5394", width=2.2), fill="tozeroy", fillcolor="rgba(159,197,232,0.6)", showlegend=False),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="X (m)", row=1, col=1, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="Y (m)", row=1, col=1)
    fig.update_xaxes(title_text="X (m)", row=1, col=2)
    fig.update_yaxes(title_text="Y (m)", row=1, col=2)
    fig.update_xaxes(title_text="Arc Length (m)", row=2, col=1)
    fig.update_yaxes(title_text="Tension (MPa)", row=2, col=1)
    fig.update_xaxes(title_text="Arc Length (m)", row=2, col=2)
    fig.update_yaxes(title_text="Compression (MPa)", row=2, col=2)

    fig.update_layout(
        title=dict(text="Principal Stress Distribution", x=0.5, font=dict(size=20)),
        width=1400,
        height=1000,
        template="plotly_white",
    )

    write_image_with_timeout(fig, out_path, scale=1.2)


def plot_critical_views(mesh_path: Path, tension_point: np.ndarray, compression_point: np.ndarray, out_path: Path) -> None:
    stations = load_station_edges(mesh_path)
    inner_base = [s["inner_base"] for s in stations]
    outer_base = [s["outer_base"] for s in stations]
    inner_top = [s["inner_top"] for s in stations]
    outer_top = [s["outer_top"] for s in stations]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("Plan View", "Elevation (X-Z)", "Elevation (Y-Z)", "Isometric"),
    )

    x_ib, y_ib, _ = xyz(inner_base)
    x_ob, y_ob, _ = xyz(outer_base)
    fig.add_trace(go.Scatter(x=x_ib, y=y_ib, mode="lines", line=dict(color="black", width=1.6), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_ob, y=y_ob, mode="lines", line=dict(color="black", width=1.6), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[tension_point[0]], y=[tension_point[1]], mode="markers", marker=dict(color="#d73027", size=11), name="Max tension"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[compression_point[0]], y=[compression_point[1]], mode="markers", marker=dict(color="#08519c", size=11), name="Max compression"), row=1, col=1)

    x_it, _, z_it = xyz(inner_top)
    x_ot, _, z_ot = xyz(outer_top)
    x_ib, _, z_ib = xyz(inner_base)
    x_ob, _, z_ob = xyz(outer_base)
    fig.add_trace(go.Scatter(x=x_ib, y=z_ib, mode="lines", line=dict(color="black", width=1.3), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_ob, y=z_ob, mode="lines", line=dict(color="black", width=1.3), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_it, y=z_it, mode="lines", line=dict(color="black", width=1.0, dash="dash"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_ot, y=z_ot, mode="lines", line=dict(color="black", width=1.0, dash="dash"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[tension_point[0]], y=[tension_point[2]], mode="markers", marker=dict(color="#d73027", size=11), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[compression_point[0]], y=[compression_point[2]], mode="markers", marker=dict(color="#08519c", size=11), showlegend=False), row=1, col=2)

    _, y_it, z_it = xyz(inner_top)
    _, y_ot, z_ot = xyz(outer_top)
    _, y_ib, z_ib = xyz(inner_base)
    _, y_ob, z_ob = xyz(outer_base)
    fig.add_trace(go.Scatter(x=y_ib, y=z_ib, mode="lines", line=dict(color="black", width=1.3), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=y_ob, y=z_ob, mode="lines", line=dict(color="black", width=1.3), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=y_it, y=z_it, mode="lines", line=dict(color="black", width=1.0, dash="dash"), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=y_ot, y=z_ot, mode="lines", line=dict(color="black", width=1.0, dash="dash"), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[tension_point[1]], y=[tension_point[2]], mode="markers", marker=dict(color="#d73027", size=11), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[compression_point[1]], y=[compression_point[2]], mode="markers", marker=dict(color="#08519c", size=11), showlegend=False), row=2, col=1)

    for edge in (inner_base, outer_base, inner_top, outer_top):
        x_vals, y_vals, z_vals = xyz(edge)
        fig.add_trace(go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode="lines", line=dict(color="black", width=4), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter3d(x=[tension_point[0]], y=[tension_point[1]], z=[tension_point[2]], mode="markers", marker=dict(color="#d73027", size=6), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter3d(x=[compression_point[0]], y=[compression_point[1]], z=[compression_point[2]], mode="markers", marker=dict(color="#08519c", size=6), showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="X (m)", row=1, col=1, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="Y (m)", row=1, col=1)
    fig.update_xaxes(title_text="X (m)", row=1, col=2)
    fig.update_yaxes(title_text="Z (m)", row=1, col=2)
    fig.update_xaxes(title_text="Y (m)", row=2, col=1)
    fig.update_yaxes(title_text="Z (m)", row=2, col=1)
    fig.update_scenes(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z (m)",
        camera=dict(eye=dict(x=1.4, y=-1.9, z=1.1)),
    )

    fig.update_layout(
        title=dict(text="Critical Stress Locations", x=0.5, font=dict(size=20)),
        width=1400,
        height=1000,
        template="plotly_white",
    )

    write_image_with_timeout(fig, out_path, scale=1.2)


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
    orientation = Image.open(orientation_path)
    distribution = Image.open(distribution_path)
    critical = Image.open(critical_path)

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

    summary_fig = go.Figure()
    summary_fig.add_annotation(
        text="<br>".join(line if line else "&nbsp;" for line in lines),
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        align="left",
        font=dict(family="Courier New, monospace", size=16),
    )
    summary_fig.update_xaxes(visible=False)
    summary_fig.update_yaxes(visible=False)
    summary_fig.update_layout(width=800, height=700, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    summary_bytes = summary_fig.to_image(format="png", scale=1.2)
    import io
    summary_image = Image.open(io.BytesIO(summary_bytes))

    def fit(img: Image.Image, size: tuple[int, int]) -> Image.Image:
        return img.resize(size)

    cell_size = (800, 700)
    sheet = Image.new("RGB", (cell_size[0] * 2, cell_size[1] * 2 + 80), color="white")
    sheet.paste(fit(orientation, cell_size), (0, 80))
    sheet.paste(fit(distribution, cell_size), (cell_size[0], 80))
    sheet.paste(fit(critical, cell_size), (0, 80 + cell_size[1]))
    sheet.paste(fit(summary_image, cell_size), (cell_size[0], 80 + cell_size[1]))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(sheet)
    try:
        title_font = ImageFont.truetype("arial.ttf", 36)
    except OSError:
        title_font = ImageFont.load_default()
    draw.text((sheet.width // 2, 25), "Arch Dam Stress Review Sheet", font=title_font, fill="black", anchor="mm")

    sheet.save(out_path)


def plot_dam_outline_sketch(
    element_rows: list[dict[str, float]],
    tension_point: np.ndarray,
    compression_point: np.ndarray,
    out_path: Path,
) -> None:
    """Self-contained sketch of the dam highlighting stress extrema.

    Unlike `plot_critical_views`, this does not depend on a sidecar Gmsh mesh
    or station-edge metadata; it derives the dam silhouette directly from the
    solved element centroids, so it stays consistent with whatever geometry
    produced the current job's result.
    """
    centroids = np.array([[row["centroid_x_m"], row["centroid_y_m"], row["centroid_z_m"]] for row in element_rows])
    tension = np.array([row["principal_max_pa"] for row in element_rows]) / 1.0e6

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Plan View: Tensile Principal Stress (MPa)", "Elevation (X-Z): Tensile Principal Stress (MPa)"),
    )

    fig.add_trace(
        go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode="markers",
            marker=dict(color=tension, colorscale="RdYlBu_r", size=4, colorbar=dict(title="MPa", x=0.44, len=0.85)),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=[tension_point[0]], y=[tension_point[1]], mode="markers", marker=dict(color="black", symbol="x", size=14), name="Max tension"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[compression_point[0]], y=[compression_point[1]], mode="markers", marker=dict(color="black", symbol="cross", size=14), name="Max compression"), row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 2],
            mode="markers",
            marker=dict(color=tension, colorscale="RdYlBu_r", size=4, colorbar=dict(title="MPa", x=1.0, len=0.85)),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(go.Scatter(x=[tension_point[0]], y=[tension_point[2]], mode="markers", marker=dict(color="black", symbol="x", size=14), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[compression_point[0]], y=[compression_point[2]], mode="markers", marker=dict(color="black", symbol="cross", size=14), showlegend=False), row=1, col=2)

    fig.update_xaxes(title_text="X (m)", row=1, col=1, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="Y (m)", row=1, col=1)
    fig.update_xaxes(title_text="X (m)", row=1, col=2)
    fig.update_yaxes(title_text="Z (m)", row=1, col=2)

    fig.update_layout(
        title=dict(text="Dam Stress Sketch", x=0.5, font=dict(size=20)),
        width=1400,
        height=600,
        template="plotly_white",
    )

    write_image_with_timeout(fig, out_path, scale=1.2)


def analyze_result(
    vtu_path: Path,
    out_dir: Path,
    youngs_modulus: float,
    poisson_ratio: float,
    tensile_strength_pa: float,
) -> dict:
    """Lightweight, pipeline-facing analysis entry point used by the job worker.

    Unlike `analyze()`, this does not depend on a sidecar Gmsh mesh or legacy
    load-cases file; it derives everything from the solved VTU result and the
    material properties already known to the caller, so its output stays
    consistent with the job's own configuration snapshot.
    """
    points, _displacement, native_stress, volume_cells = parse_vtu(vtu_path)

    element_rows: list[dict[str, float]] = []
    max_tension = {"element_id": -1, "value_pa": float("-inf"), "centroid_xyz_m": [0.0, 0.0, 0.0]}
    max_compression = {"element_id": -1, "value_pa": float("inf"), "centroid_xyz_m": [0.0, 0.0, 0.0]}

    for element_id, (_cell_type, node_ids) in enumerate(volume_cells, start=1):
        coords = points[node_ids]
        stress = native_stress[node_ids].mean(axis=0)
        principal = np.linalg.eigvalsh(stress_tensor(stress))
        centroid = coords.mean(axis=0)
        row = {
            "element_id": element_id,
            "centroid_x_m": float(centroid[0]),
            "centroid_y_m": float(centroid[1]),
            "centroid_z_m": float(centroid[2]),
            "principal_min_pa": float(principal[0]),
            "principal_mid_pa": float(principal[1]),
            "principal_max_pa": float(principal[2]),
        }
        element_rows.append(row)
        if principal[2] > max_tension["value_pa"]:
            max_tension = {
                "element_id": element_id,
                "value_pa": float(principal[2]),
                "centroid_xyz_m": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
            }
        if principal[0] < max_compression["value_pa"]:
            max_compression = {
                "element_id": element_id,
                "value_pa": float(principal[0]),
                "centroid_xyz_m": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
            }

    if not element_rows:
        raise ValueError("Solved result contains no volume elements")

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
        "tensile_strength_pa": tensile_strength_pa,
        "tension_utilization": (max_tension["value_pa"] / tensile_strength_pa) if tensile_strength_pa else None,
        "note": (
            "Positive principal stress indicates tension. Negative principal stress indicates compression. "
            "Successful execution or convergence is not engineering validation or proof of dam safety."
        ),
    }
    summary_path = out_dir / "stress_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Kaleido (the headless-Chromium renderer behind fig.write_image) can
    # intermittently fail to cleanly close/kill its browser subprocess on
    # Windows after an image has already been rendered and written to disk.
    # Treat each plot as independent and non-fatal: the numeric stress
    # summary/CSV above is the authoritative result and must not be discarded
    # just because a report image failed to render.
    stress_plot_path = out_dir / "principal_stress_distribution.png"
    plot_error: str | None = None
    try:
        plot_stress_distribution(element_rows, tensile_strength_pa, stress_plot_path)
    except Exception as error:  # noqa: BLE001 - a plotting failure must not discard the stress analysis
        plot_error = f"principal_stress_distribution.png: {error}"
        if not stress_plot_path.is_file():
            stress_plot_path = None  # type: ignore[assignment]

    sketch_path = out_dir / "dam_stress_sketch.png"
    try:
        plot_dam_outline_sketch(
            element_rows,
            np.array(max_tension["centroid_xyz_m"]),
            np.array(max_compression["centroid_xyz_m"]),
            sketch_path,
        )
    except Exception as error:  # noqa: BLE001 - a plotting failure must not discard the stress analysis
        plot_error = f"{plot_error + '; ' if plot_error else ''}dam_stress_sketch.png: {error}"
        if not sketch_path.is_file():
            sketch_path = None  # type: ignore[assignment]

    result = {
        "stress_summary_path": str(summary_path),
        "stress_csv_path": str(csv_path),
        "stress_distribution_plot_path": str(stress_plot_path) if stress_plot_path else None,
        "stress_sketch_plot_path": str(sketch_path) if sketch_path else None,
        **summary,
    }
    if plot_error is not None:
        result["plot_error"] = plot_error
    return result


def load_tensile_strength(load_cases_path: Path) -> float:
    payload = json.loads(load_cases_path.read_text())
    return float(payload["material"]["concrete"]["tensile_strength_pa"])


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
    points, displacement, native_stress, volume_cells = parse_vtu(vtu_path)

    element_rows: list[dict[str, float]] = []
    max_tension = {"element_id": -1, "value_pa": float("-inf"), "centroid_xyz_m": [0.0, 0.0, 0.0]}
    max_compression = {"element_id": -1, "value_pa": float("inf"), "centroid_xyz_m": [0.0, 0.0, 0.0]}
    max_tension_affine = {"element_id": -1, "value_pa": float("-inf")}
    max_compression_affine = {"element_id": -1, "value_pa": float("inf")}

    for element_id, (cell_type, node_ids) in enumerate(volume_cells, start=1):
        coords = points[node_ids]
        stress = native_stress[node_ids].mean(axis=0)
        principal = np.linalg.eigvalsh(stress_tensor(stress))
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
            "principal_min_pa": float(principal[0]),
            "principal_mid_pa": float(principal[1]),
            "principal_max_pa": float(principal[2]),
            "native_nodal_principal_min_pa": float(principal[0]),
            "native_nodal_principal_mid_pa": float(principal[1]),
            "native_nodal_principal_max_pa": float(principal[2]),
        }
        if exclude_below_z is not None and centroid[2] <= exclude_below_z:
            continue

        element_rows.append(row)

        if principal[2] > max_tension["value_pa"]:
            max_tension = {
                "element_id": element_id,
            "value_pa": float(principal[2]),
                "centroid_xyz_m": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
            }
        if principal[0] < max_compression["value_pa"]:
            max_compression = {
                "element_id": element_id,
            "value_pa": float(principal[0]),
                "centroid_xyz_m": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
            }
        if principal[2] > max_tension_affine["value_pa"]:
            max_tension_affine = {"element_id": element_id, "value_pa": float(principal[2])}
        if principal[0] < max_compression_affine["value_pa"]:
            max_compression_affine = {"element_id": element_id, "value_pa": float(principal[0])}

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
        "native_nodal_stress_average": {
            "max_tensile_principal_stress_pa": max_tension_affine,
            "max_compressive_principal_stress_pa": max_compression_affine,
        },
        "note": "Positive principal stress indicates tension. Negative principal stress indicates compression.",
    }
    if exclude_below_z is not None:
        summary["exclusion"] = {
            "excluded_centroid_z_m_lte": exclude_below_z,
            "description": "Excludes the wall base transition region at and below the stated elevation.",
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
        "--config",
        "--material-properties",
        "--load-cases",
        dest="load_cases",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.json",
        help="Path to root config.json (legacy --material-properties and --load-cases options accepted)",
    )
    parser.add_argument("--youngs-modulus", type=float)
    parser.add_argument("--poisson-ratio", type=float)
    parser.add_argument("--expected-tension-mpa", type=float, default=1.2)
    parser.add_argument("--expected-compression-mpa", type=float, default=-3.33)
    parser.add_argument(
        "--exclude-below-z",
        type=float,
        help="Exclude volume elements whose centroid elevation is at or below this value in metres.",
    )
    args = parser.parse_args()
    material = json.loads(args.load_cases.read_text())["material"]["concrete"]
    youngs_modulus = float(material["youngs_modulus"]) if args.youngs_modulus is None else args.youngs_modulus
    poisson_ratio = float(material["poisson_ratio"]) if args.poisson_ratio is None else args.poisson_ratio

    csv_path, summary_path, stress_plot_path, critical_plot_path, report_path = analyze(
        args.vtu,
        args.out_dir,
        youngs_modulus,
        poisson_ratio,
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