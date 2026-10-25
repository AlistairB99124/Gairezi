"""Extract the nodal result fields Code_Aster wrote into ``result.med``
(``DEPL``, ``SIGM_NOEU``) into a plain ``.npz`` archive that the rest of the
toolchain (``.venv``, Python 3.15) can read without needing ``medcoupling``.
Also writes ``result_stress.vtu`` beside the archive for stock ParaView,
which does not include Salome's MED reader. This contains nodal stress
components, principal stresses and von Mises stress in Pa on the stress
field's concrete volume mesh (wall and foundation only).

Must be run with Code_Aster's own bundled Python interpreter
(``<install>/external/Python3.10/python.exe``), exactly like
``convert_arrays_to_med.py``, because ``medcoupling``/``MEDLoader`` are only
importable from that interpreter.

Field name/mesh name notes learned by inspecting a real ``result.med``
(not documented anywhere obvious):

* The MED mesh is renamed internally by Code_Aster to a generic name (e.g.
  ``"00000001"``), not the ``NOM_MED`` used at ``LIRE_MAILLAGE`` -- use
  ``MEDLoader.GetMeshNames`` to discover it rather than assuming ``"WALL"``.
* ``IMPR_RESU(... RESU=_F(RESULTAT=RESU, NOM_CHAM=('DEPL', 'SIGM_NOEU', ...)))``
  writes field names prefixed with the result-object name:
  ``"RESU____DEPL"``, ``"RESU____SIGM_NOEU"`` (the prefix is whatever
  variable name receives ``MECA_STATIQUE`` in the ``.comm`` file, padded
  with underscores).
* ``SIGM_NOEU``'s 6 Voigt components are ordered
  ``(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ)`` -- note ``xz``/``yz`` are swapped
  relative to ``Elmer.analyze_stress.stress_tensor``'s
  ``(xx, yy, zz, xy, yz, xz)`` convention; this script reorders to match
  Elmer's convention so the rest of the comparison code can be shared.

Usage::

    <code_aster_install>\\external\\Python3.10\\python.exe extract_result_fields.py \\
        <result.med> <output aster_result_fields.npz>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_medcoupling() -> None:
    python_exe = Path(sys.executable).resolve()
    external_dir = python_exe.parent.parent
    dll_dirs = [
        external_dir / "hdf51.10.5" / "bin",
        external_dir / "hdf51.10.5" / "lib",
        external_dir / "MED-4.4.1" / "lib",
        external_dir / "medcoupling-9.11.0" / "lib",
        Path(r"C:\Windows\System32"),
        external_dir,
    ]
    for d in dll_dirs:
        if d.is_dir() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(d))
    site_packages = external_dir / "medcoupling-9.11.0" / "lib" / "python3.10" / "site-packages"
    if site_packages.is_dir():
        sys.path.insert(0, str(site_packages))


_bootstrap_medcoupling()

import MEDLoader as ml  # noqa: E402
import numpy as np  # noqa: E402

# RESU is the variable name MECA_STATIQUE/CALC_CHAMP assign to in
# Aster/generate_elastic_case.py; IMPR_RESU pads it to 8 chars with
# underscores to build the on-disk field name.
RESULT_NAME_PREFIX = "RESU____"
# SIGM_NOEU order (SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ) -> Elmer's
# (xx, yy, zz, xy, yz, xz): swap the last two components.
_SIGM_NOEU_TO_ELMER_ORDER = (0, 1, 2, 3, 5, 4)


def stress_scalars(stress_raw: np.ndarray) -> dict[str, np.ndarray]:
    """Scalar fields from Code_Aster's (xx, yy, zz, xy, xz, yz) stresses, in Pa."""
    if stress_raw.ndim != 2 or stress_raw.shape[1] != 6 or not np.isfinite(stress_raw).all():
        raise ValueError("Expected finite nodal stresses with shape (node_count, 6)")
    fields = {
        f"stress_{component}_pa": stress_raw[:, i]
        for i, component in enumerate(("xx", "yy", "zz", "xy", "xz", "yz"))
    }
    tensor = np.empty((len(stress_raw), 3, 3), dtype=np.float64)
    tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2] = stress_raw[:, :3].T
    tensor[:, 0, 1] = tensor[:, 1, 0] = stress_raw[:, 3]
    tensor[:, 0, 2] = tensor[:, 2, 0] = stress_raw[:, 4]
    tensor[:, 1, 2] = tensor[:, 2, 1] = stress_raw[:, 5]
    principal = np.linalg.eigvalsh(tensor)
    for i, name in enumerate(("min", "mid", "max")):
        fields[f"principal_{name}_pa"] = principal[:, i]
    fields["von_mises_pa"] = np.sqrt(
        ((principal[:, 2] - principal[:, 1]) ** 2
         + (principal[:, 1] - principal[:, 0]) ** 2
         + (principal[:, 0] - principal[:, 2]) ** 2) / 2.0,
    )
    return fields


def export_stress_vtu(sigm, out_path: Path) -> Path:
    """Preserve the MED stress support and node order, including coincident joint nodes."""
    stress_raw = np.array(sigm.getArray().toNumPyArray(), dtype=np.float64)
    fields = []
    for name, values in stress_scalars(stress_raw).items():
        field = ml.MEDCouplingFieldDouble(ml.ON_NODES, ml.ONE_TIME)
        field.setMesh(sigm.getMesh())
        field.setName(name)
        field.setArray(ml.DataArrayDouble(values.reshape(-1, 1).copy()))
        field.setTime(*sigm.getTime())
        field.checkConsistencyLight()
        fields.append(field)
    written = Path(ml.MEDCouplingFieldDouble.WriteVTK(str(out_path.with_suffix("")), fields))
    if not written.is_file() or written.stat().st_size == 0:
        raise RuntimeError(f"Stress VTU export did not produce a nonempty file: {written}")
    return written


def extract(result_med_path: Path, out_path: Path) -> dict:
    mesh_names = ml.GetMeshNames(str(result_med_path))
    if not mesh_names:
        raise RuntimeError(f"{result_med_path} contains no mesh")
    mesh_name = mesh_names[0]

    depl = ml.ReadFieldNode(str(result_med_path), mesh_name, 0, RESULT_NAME_PREFIX + "DEPL", 1, 1)
    sigm = ml.ReadFieldNode(str(result_med_path), mesh_name, 0, RESULT_NAME_PREFIX + "SIGM_NOEU", 1, 1)
    mesh_arrays_path = result_med_path.parent.parent / "three_body_mesh_arrays.npz"
    if mesh_arrays_path.is_file():
        with np.load(mesh_arrays_path) as arrays:
            if "source_node_ids" in arrays:
                full_points = np.asarray(depl.getMesh().getCoords().toNumPyArray())
                np.testing.assert_allclose(full_points, arrays["nodes"], rtol=0., atol=1e-9)
                np.savez_compressed(out_path.parent / "_diagnostic_displacement.npz",
                                    displacement_m=depl.getArray().toNumPyArray())

    med_mesh = ml.MEDFileUMesh(str(result_med_path))
    groups = med_mesh.getGroupsOnSpecifiedLev(0)
    group = "CONCRETE" if "CONCRETE" in groups else "WALL"
    if group not in groups:
        raise RuntimeError("Result MED has no concrete volume group (CONCRETE or WALL)")
    scope = "concrete" if group == "CONCRETE" else "legacy_wall"
    cell_ids = med_mesh.getGroupArr(0, group)
    # Subset by cell identity, not coordinates: opposite interface sides
    # have coincident nodes with different stresses.
    depl = depl[cell_ids]
    sigm = sigm[cell_ids]
    points = np.array(depl.getMesh().getCoords().toNumPyArray(), dtype=np.float64)
    displacement = np.array(depl.getArray().toNumPyArray(), dtype=np.float64)
    sigm_points = np.array(sigm.getMesh().getCoords().toNumPyArray(), dtype=np.float64)
    if not np.array_equal(points, sigm_points) or not np.array_equal(
        depl.getMesh().getNodalConnectivity().toNumPyArray(),
        sigm.getMesh().getNodalConnectivity().toNumPyArray(),
    ):
        raise RuntimeError("Concrete displacement and stress supports do not share node/cell ordering")
    stress = np.array(sigm.getArray().toNumPyArray(), dtype=np.float64)[:, _SIGM_NOEU_TO_ELMER_ORDER]

    if points.shape != displacement.shape or stress.shape != (len(points), 6):
        raise RuntimeError(
            f"Unexpected field shapes: points={points.shape}, "
            f"displacement={displacement.shape}, stress={stress.shape}",
        )

    result_vtu_path = export_stress_vtu(sigm, out_path.parent / "result_stress.vtu")
    extra = {}
    if mesh_arrays_path.is_file():
        with np.load(mesh_arrays_path) as arrays:
            if "source_node_ids" in arrays:
                cells = np.asarray(sigm.getMesh().getNodalConnectivity().toNumPyArray()).reshape(-1, 5)[:, 1:]
                source_cells = arrays["tetra4_concrete"]
                if cells.shape != source_cells.shape:
                    raise RuntimeError("Concrete result topology differs from exported source")
                mapped = np.full(len(points), -1, dtype=np.int64)
                mapped[cells.ravel()] = arrays["source_node_ids"][source_cells].ravel()
                if not np.array_equal(mapped[cells], arrays["source_node_ids"][source_cells]):
                    raise RuntimeError("Result node identity is inconsistent")
                np.testing.assert_allclose(points[cells], arrays["nodes"][source_cells], rtol=0., atol=1e-9)
                extra["source_node_ids"] = mapped
                for body in (1, 3):
                    mask = np.zeros(len(points), dtype=bool)
                    mask[np.unique(cells[arrays["concrete_body_ids"] == body])] = True
                    extra[f"body_{body}_mask"] = mask
    np.savez_compressed(
        out_path,
        points_m=points,
        displacement_m=displacement,
        stress_pa=stress,
        reporting_scope=scope,
        **extra,
    )
    return {
        "mesh_name": mesh_name,
        "node_count": int(len(points)),
        "reporting_scope": scope,
        "out_path": str(out_path.resolve()),
        "result_vtu_path": str(result_vtu_path.resolve()),
    }


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <result.med> <aster_result_fields.npz>", file=sys.stderr)
        sys.exit(2)
    result = extract(Path(sys.argv[1]), Path(sys.argv[2]))
    print(result)


if __name__ == "__main__":
    main()
