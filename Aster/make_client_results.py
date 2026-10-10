"""Generate the client diagrams and comparison table for both solvers.

The stress diagrams come from Elmer/analyze_stress.py, run unchanged on each solver's
displacement field, so both solvers go through the same accepted post-processing.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile

import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Elmer"))
sys.path.insert(0, str(ROOT / "Aster"))

import analyze_stress  # noqa: E402
import compare_results as compare  # noqa: E402

RESULTS = ROOT / "Aster" / "results" / "00_elastic_baseline"
CLIENT_DIR = ROOT / "Client_Results"
DIAGRAMS = (
    "client_stress_report.png",
    "principal_stress_distribution.png",
    "critical_stress_locations.png",
    "arch_dam_orientations.png",
)
GMSH_TO_VTK = {4: 10, 5: 12, 6: 13, 7: 14}
NODE_COUNT = {4: 4, 5: 8, 6: 6, 7: 5}


def write_vtu(path: Path, points: np.ndarray, displacement: np.ndarray, types: np.ndarray, conn: np.ndarray) -> None:
    """Write the appended-raw layout that analyze_stress.parse_vtu reads."""
    connectivity = np.concatenate([conn[i, : NODE_COUNT[t]] for i, t in enumerate(types)]).astype("<i4")
    offsets = np.cumsum([NODE_COUNT[t] for t in types]).astype("<i4")
    vtk_types = np.array([GMSH_TO_VTK[t] for t in types], dtype="u1")
    chunks = [
        ("Float64", "displacement", 3, displacement.astype("<f8")),
        ("Float64", None, 3, points.astype("<f8")),
        ("Int32", "connectivity", 1, connectivity),
        ("Int32", "offsets", 1, offsets),
        ("UInt8", "types", 1, vtk_types),
    ]
    lines = ['<?xml version="1.0"?>', '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">', "<UnstructuredGrid>"]
    body = b""
    for dtype, name, components, values in chunks:
        label = f' Name="{name}"' if name else ""
        comps = f' NumberOfComponents="{components}"' if components > 1 else ""
        lines.append(f'<DataArray type="{dtype}"{label}{comps} format="appended" offset="{len(body)}"/>')
        raw = values.tobytes()
        body += struct.pack("<I", len(raw)) + raw
    lines += ["</UnstructuredGrid>", '<AppendedData encoding="raw">']
    path.write_bytes("\n".join(lines).encode("ascii") + b"\n_" + body + b"\n</AppendedData>\n</VTKFile>\n")


def run_pipeline(name: str, points: np.ndarray, displacement: np.ndarray, types, conn, work: Path):
    out_dir = work / name
    out_dir.mkdir(parents=True)
    vtu = out_dir / "input.vtu"
    # Elmer's VTU holds the displaced mesh, so both solvers are fed displaced coordinates.
    write_vtu(vtu, points + displacement, displacement, types, conn)
    analyze_stress.analyze(
        vtu,
        out_dir,
        3.5e10,
        0.2,
        ROOT / "Elmer" / "curved_dam_mesh.msh",
        ROOT / "Data" / "Concrete_Material_Properties.json",
        1.2,
        -3.33,
    )
    target = CLIENT_DIR / name
    for diagram in DIAGRAMS:
        shutil.copyfile(out_dir / diagram, target / diagram)
    summary = json.loads((out_dir / "stress_summary.json").read_text())
    with (out_dir / "principal_stress_by_element.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    tension = np.array([float(r["principal_max_pa"]) for r in rows])
    compression = np.array([float(r["principal_min_pa"]) for r in rows])
    return summary, tension, compression


def parity_plot(elmer_t, aster_t, elmer_c, aster_c, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)
    for axis, (a, b, title) in zip(axes, ((elmer_t, aster_t, "Max principal tension (MPa)"), (elmer_c, aster_c, "Min principal stress (MPa)"))):
        a, b = a / 1e6, b / 1e6
        axis.scatter(a, b, s=2, alpha=0.3, color="#1f4e79")
        low, high = min(a.min(), b.min()), max(a.max(), b.max())
        axis.plot([low, high], [low, high], "k--", linewidth=1)
        axis.set_xlabel("Elmer")
        axis.set_ylabel("Code_Aster")
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.grid(True, color="0.9")
    fig.suptitle("Element principal stress: Code_Aster vs Elmer")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    aster = np.load(RESULTS / "aster_result.npz")
    mesh = np.load(RESULTS / "mesh_intermediate.npz")
    inputs = json.loads((RESULTS / "case_inputs.json").read_text())
    points = aster["points"]
    elmer = compare.load_elmer(compare.ELMER_DIR, points)
    types, conn = mesh["orig_types"], mesh["orig_conn"]

    for name in ("Aster", "Elmer", "Comparison"):
        (CLIENT_DIR / name).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        s_elmer, t_elmer, c_elmer = run_pipeline("Elmer", points, elmer["displacement"], types, conn, work)
        s_aster, t_aster, c_aster = run_pipeline("Aster", points, aster["displacement"], types, conn, work)
    parity_plot(t_elmer, t_aster, c_elmer, c_aster, CLIENT_DIR / "Comparison" / "element_principal_stress_parity.png")

    spring_nodes = np.array(inputs["spring_nodes"])
    areas = np.array(inputs["spring_areas_m2"])
    k = np.array(inputs["foundation_spring_n_per_m3"])

    def reaction(u):
        return -(u[spring_nodes] * k * areas[:, None]).sum(axis=0) / 1e6

    nodal = {
        "Elmer": compare.principal_stresses(elmer["stress"]),
        "Aster": compare.principal_stresses(aster["stress"]),
    }
    u = {"Elmer": elmer["displacement"], "Aster": aster["displacement"]}
    r = {name: reaction(u[name]) for name in u}
    strength = s_elmer["tensile_strength_pa"] / 1e6

    def tension(s):
        return s["max_tensile_principal_stress_pa"]["value_pa"] / 1e6

    def compression(s):
        return s["max_compressive_principal_stress_pa"]["value_pa"] / 1e6

    table = [
        ("Max tensile principal stress, element-based (MPa)", tension(s_elmer), tension(s_aster), 1.2),
        ("Max compressive principal stress, element-based (MPa)", compression(s_elmer), compression(s_aster), -3.33),
        ("Tension utilisation vs %.1f MPa strength (-)" % strength, tension(s_elmer) / strength, tension(s_aster) / strength, ""),
        ("Max tensile principal stress, nodal output (MPa)", nodal["Elmer"][:, 0].max() / 1e6, nodal["Aster"][:, 0].max() / 1e6, ""),
        ("Max compressive principal stress, nodal output (MPa)", nodal["Elmer"][:, 2].min() / 1e6, nodal["Aster"][:, 2].min() / 1e6, ""),
        ("Max displacement magnitude (mm)", *(np.linalg.norm(u[n], axis=1).max() * 1e3 for n in ("Elmer", "Aster")), ""),
        ("Foundation reaction Fx (MN)", r["Elmer"][0], r["Aster"][0], ""),
        ("Foundation reaction Fy (MN)", r["Elmer"][1], r["Aster"][1], ""),
        ("Foundation reaction Fz (MN)", r["Elmer"][2], r["Aster"][2], ""),
    ]
    with (CLIENT_DIR / "Comparison" / "comparison_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Quantity", "Elmer", "Code_Aster", "Difference (Aster - Elmer)", "Difference (%)", "Client expected"])
        for label, e_value, a_value, expected in table:
            diff = a_value - e_value
            writer.writerow([label, f"{e_value:.4f}", f"{a_value:.4f}", f"{diff:.4f}", f"{100 * diff / e_value:.2f}", expected])
        writer.writerow([
            "Location of max tension, element centroid (m)",
            " ".join(f"{v:.2f}" for v in s_elmer["max_tensile_principal_stress_pa"]["centroid_xyz_m"]),
            " ".join(f"{v:.2f}" for v in s_aster["max_tensile_principal_stress_pa"]["centroid_xyz_m"]),
            "", "", "",
        ])
    print((CLIENT_DIR / "Comparison" / "comparison_table.csv").read_text())


if __name__ == "__main__":
    main()
