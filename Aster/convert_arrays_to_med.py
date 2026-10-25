"""Convert ``mesh_arrays.npz`` (written by ``build_med_mesh.py``) into a real
``.med`` mesh using Code_Aster's bundled ``medcoupling`` module.

Why this script exists
-----------------------
``LIRE_MAILLAGE(FORMAT='GMSH', ...)`` was observed to not complete within
9+ minutes of continuous ~100% CPU on this mesh (~101k cells / 117k nodes;
see ``Aster_Porting.md`` amendment). Code_Aster's native ``FORMAT='MED'``
reader is expected to be far faster since MED is its preferred binary format.

This script must be run with Code_Aster's *own bundled* Python interpreter
(``<install>/external/Python3.10/python.exe``), not this repo's ``.venv``,
because:

* ``medcoupling`` (which wraps the MED C library directly, no ``h5py``
  needed) is only importable from that interpreter.
* This repo's ``.venv`` (Python 3.15) has no ``h5py`` wheel available, and
  Code_Aster's bundled Python has no ``h5py`` module at all -- so a true MED
  write is only possible via ``medcoupling``, not ``meshio``+``h5py``.

Usage::

    <code_aster_install>\\external\\Python3.10\\python.exe convert_arrays_to_med.py \\
        <mesh_arrays.npz> <output mesh.med>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Code_Aster's bundled interpreter does not have these directories on its
# default DLL search path or sys.path; both are required before ``import
# medcoupling`` will succeed. Paths are relative to this script's install,
# discovered relative to the running interpreter's own location so this
# works regardless of which Code_Aster version/drive it is installed under.
def _bootstrap_medcoupling() -> None:
    python_exe = Path(sys.executable).resolve()
    # python.exe lives at <install>/external/Python3.10/python.exe
    external_dir = python_exe.parent.parent
    install_dir = external_dir.parent
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
    del install_dir  # only used for clarity above; not otherwise needed


_bootstrap_medcoupling()

import numpy as np  # noqa: E402
import medcoupling as mc  # noqa: E402

MESH_NAME = "WALL"


def _reorient(norm_type: int, row: np.ndarray) -> list[int]:
    """Flip node winding so volumes come out positive under MED's convention.

    Gmsh/VTK HEXA8 and PENTA6 node ordering, when fed to medcoupling as-is,
    was found (by checking ``getMeasureField(False)``) to produce a
    systematically negative signed volume for every single cell -- i.e. a
    consistent chirality mismatch between the two conventions, not a mesh
    defect. Reversing each face's winding (while keeping the same
    bottom/top face grouping) flips the handedness without changing which
    physical face is "bottom" vs "top".
    """
    if norm_type == mc.NORM_HEXA8:
        n0, n1, n2, n3, n4, n5, n6, n7 = (int(v) for v in row)
        return [n0, n3, n2, n1, n4, n7, n6, n5]
    if norm_type == mc.NORM_PENTA6:
        n0, n1, n2, n3, n4, n5 = (int(v) for v in row)
        return [n0, n2, n1, n3, n5, n4]
    return [int(v) for v in row]


def _build_level(name: str, dim: int, node_coords: "mc.DataArrayDouble",
                  cell_blocks: list[tuple[int, np.ndarray, np.ndarray]]) -> tuple["mc.MEDCouplingUMesh", list["mc.DataArrayInt"]]:
    """Build one MEDCouplingUMesh level from one or more
    ``(norm_type, connectivity, group_names)`` blocks, inserting all cells in
    block order (so the Nth cell of block K lands at global index
    ``sum(len(block) for earlier blocks) + N``), and return the mesh plus the
    group ``DataArrayInt`` list (one per distinct non-empty group name,
    collecting global indices) ready for ``setGroupsAtLevel``.
    """
    total_cells = sum(len(conn) for _norm, conn, _groups in cell_blocks)
    mesh = mc.MEDCouplingUMesh.New(name, dim)
    mesh.allocateCells(total_cells)
    indices_by_group: dict[str, list[int]] = {}
    global_index = 0
    for norm_type, conn, group_names in cell_blocks:
        for row, group_name in zip(conn, group_names):
            mesh.insertNextCell(norm_type, _reorient(norm_type, row))
            if group_name:
                indices_by_group.setdefault(str(group_name), []).append(global_index)
            global_index += 1
    mesh.setCoords(node_coords)
    mesh.finishInsertingCells()
    # MED file format requires cells grouped/ordered by geometric type; our
    # insertion order is block-by-block (hex8 then wedge6, or tri3 then
    # quad4) which already satisfies that for exactly two types, but
    # sortCellsInMEDFileFrmt() is still required to produce the internal
    # per-type numbering MEDFileUMesh expects, and it returns the old->new
    # permutation so our group indices (computed in insertion order) can be
    # remapped to match.
    old_to_new = mesh.sortCellsInMEDFileFrmt()
    group_arrays = []
    for group_name, indices in indices_by_group.items():
        remapped = [old_to_new[i] for i in indices]
        arr = mc.DataArrayInt.New(remapped)
        arr.setName(group_name)
        group_arrays.append(arr)
    return mesh, group_arrays


def convert(npz_path: Path, med_path: Path) -> None:
    data = np.load(npz_path, allow_pickle=True)
    nodes = data["nodes"]
    coords = mc.DataArrayDouble(nodes.tolist(), nodes.shape[0], nodes.shape[1])
    coords.setInfoOnComponents(["X [m]", "Y [m]", "Z [m]"])

    volume_blocks = [
        (mc.NORM_HEXA8, data["hex8"], data["hex8_group"]),
        (mc.NORM_PENTA6, data["wedge6"], data["wedge6_group"]),
    ]
    volume_mesh, volume_group_arrays = _build_level(MESH_NAME, 3, coords, volume_blocks)

    face_blocks = [
        (mc.NORM_TRI3, data["tri3"], data["tri3_group"]),
        (mc.NORM_QUAD4, data["quad4"], data["quad4_group"]),
    ]
    face_mesh, face_group_arrays = _build_level(MESH_NAME, 2, coords, face_blocks)

    # One POI1 (single-node) element per FOUNDATION surface node, needed by
    # RIGI_PARASOL's GROUP_MA_POI1 argument (generate_elastic_case.py).
    poi1_nodes = data["foundation_poi1_nodes"]
    poi1_group_names = np.full(len(poi1_nodes), "FOUNDATION_POI1", dtype=object)
    point_blocks = [(mc.NORM_POINT1, poi1_nodes.reshape(-1, 1), poi1_group_names)]
    point_mesh, point_group_arrays = _build_level(MESH_NAME, 0, coords, point_blocks)

    mfu = mc.MEDFileUMesh.New()
    mfu.setName(MESH_NAME)
    mfu.setMeshAtLevel(0, volume_mesh)
    mfu.setMeshAtLevel(-1, face_mesh)
    mfu.setMeshAtLevel(-3, point_mesh)

    if volume_group_arrays:
        mfu.setGroupsAtLevel(0, volume_group_arrays)
    if face_group_arrays:
        mfu.setGroupsAtLevel(-1, face_group_arrays)
    if point_group_arrays:
        mfu.setGroupsAtLevel(-3, point_group_arrays)

    med_path.parent.mkdir(parents=True, exist_ok=True)
    mfu.write(str(med_path), 2)
    print(f"WROTE {med_path} ({med_path.stat().st_size} bytes)")
    print("volume groups:", sorted(a.getName() for a in volume_group_arrays))
    print("face groups:", sorted(a.getName() for a in face_group_arrays))
    print("level0 cells:", volume_mesh.getNumberOfCells(), "level-1 cells:", face_mesh.getNumberOfCells())


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <mesh_arrays.npz> <output mesh.med>", file=sys.stderr)
        raise SystemExit(2)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
