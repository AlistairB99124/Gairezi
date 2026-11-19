import meshio
import numpy as np
from pathlib import Path

def convert_all_steps(base_path: Path, output_dir: Path, prefix: str = "uplift"):
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = np.load(base_path / "joint_steps.npz", allow_pickle=True)
    mesh_arrays = np.load(base_path / "three_body_mesh_arrays.npz", allow_pickle=True)

    points = np.asarray(mesh_arrays["nodes"], dtype=float)
    instants = steps["instant"]

    # Check if node indices are 1-based (Code_Aster/MED style) or 0-based
    tet_cells = np.asarray(mesh_arrays["tetra4_concrete"], dtype=int)
    joint_tri = np.asarray(mesh_arrays["joint_slave_tri3"], dtype=int)
    if tet_cells.min() == 1:
        tet_cells = tet_cells - 1
    if joint_tri.min() == 1:
        joint_tri = joint_tri - 1

    # --- 1. Export 3D Concrete Body Time Series (.pvd + .vtu) ---
    body_pvd_entries = []
    stress_pa = steps["stress_pa"]  # shape (56963, 6) -> [SXX, SYY, SZZ, SXY, SXZ, SYZ]
    sxx, syy, szz, sxy, sxz, syz = stress_pa.T
    von_mises = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 6 * (sxy**2 + sxz**2 + syz**2)))

    for i, t in enumerate(instants):
        vtu_name = f"{prefix}_dam_body_step_{i+1}.vtu"
        vtu_path = output_dir / vtu_name
        point_data = {
            "displacement_m": steps["displacement_m"][i],
            "displacement_mag_mm": np.linalg.norm(steps["displacement_m"][i], axis=1) * 1000.0,
            "stress_pa_final": stress_pa,
            "von_mises_pa_final": von_mises,
        }
        body_mesh = meshio.Mesh(points=points, cells=[("tetra", tet_cells)], point_data=point_data)
        meshio.write(vtu_path, body_mesh)
        body_pvd_entries.append((t, vtu_name))

    write_pvd(output_dir / f"{prefix}_dam_body.pvd", body_pvd_entries)

    # --- 2. Export Joint Interface Time Series (.pvd + .vtu) ---
    joint_keys = ["V1", "V2", "V3", "V5", "V7", "V8", "V9", "V18", "SIGN", "SITX", "SITY"]
    joint_pvd_entries = []

    for i, t in enumerate(instants):
        vtu_name = f"{prefix}_joint_interface_step_{i+1}.vtu"
        vtu_path = output_dir / vtu_name

        # Attach nodal displacement so you can Warp By Vector on the joint too
        point_data = {"displacement_m": steps["displacement_m"][i]}

        # Attach all joint element variables (2268 triangles)
        cell_data = {k: [steps[k][i]] for k in joint_keys if k in steps.files}
        # Compute resultant shear stress magnitude
        cell_data["SITU_shear_mag_pa"] = [np.hypot(steps["SITX"][i], steps["SITY"][i])]

        joint_mesh = meshio.Mesh(
            points=points,
            cells=[("triangle", joint_tri)],
            point_data=point_data,
            cell_data=cell_data,
        )
        meshio.write(vtu_path, joint_mesh)
        joint_pvd_entries.append((t, vtu_name))

    write_pvd(output_dir / f"{prefix}_joint_interface.pvd", joint_pvd_entries)
    print(f"Done! Open '{prefix}_dam_body.pvd' and '{prefix}_joint_interface.pvd' in ParaView.")

def write_pvd(pvd_path: Path, entries: list[tuple[float, str]]):
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for timestep, filename in entries:
        lines.append(f'    <DataSet timestep="{timestep}" group="" part="0" file="{filename}"/>')
    lines.extend(['  </Collection>', '</VTKFile>'])
    pvd_path.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    joint_base_path = Path(r"C:\Users\alist\AppData\Local\Lamb\DamAnalysisWorkbench\jobs\9b744803e09c4ea9a8862f5f2de649e1\joint_opening\friction_configured_uplift")
    output_folder = Path(r"C:\Users\alist\OneDrive\Desktop\paraview_joint_results")
    convert_all_steps(base_path=joint_base_path, output_dir=output_folder, prefix="uplift")
