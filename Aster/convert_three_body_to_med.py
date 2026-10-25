"""Convert ``three_body_mesh_arrays.npz`` (written by
``build_three_body_mesh_arrays.py``) into a real ``.med`` mesh using
Code_Aster's bundled ``medcoupling`` module.

Same rationale and bootstrap as ``convert_arrays_to_med.py`` (must run under
Code_Aster's own bundled Python, not this repo's ``.venv``); this script only
differs in cell geometry (all-tetrahedral, no hex/wedge) and group set
(CONCRETE/BEDROCK volumes; UPSTREAM/DOWNSTREAM/CREST/ROCK_BOTTOM/ROCK_SIDES
faces; ROCK_BOTTOM_POI1 points for RIGI_PARASOL; WALL_FOUNDATION_SPRINGS +
per-element SPR_WF_NNNNN SEG2 groups at level -2 for the wall/foundation
interface's discrete DIS_T springs).

Usage::

    <code_aster_install>\\external\\Python3.10\\python.exe convert_three_body_to_med.py \\
        <three_body_mesh_arrays.npz> <output mesh.med>
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

import numpy as np  # noqa: E402
import medcoupling as mc  # noqa: E402

MESH_NAME = "THREE_BODY"


def _build_level(name: str, dim: int, node_coords: "mc.DataArrayDouble",
                  cell_blocks: list[tuple[int, np.ndarray, str]]) -> tuple["mc.MEDCouplingUMesh", list["mc.DataArrayInt"]]:
    """Build one mesh level from ``(norm_type, connectivity, group_name)``
    blocks (every row of a block shares the same group), same ordering/group
    convention as ``convert_arrays_to_med.py``."""
    total_cells = sum(len(conn) for _norm, conn, _group in cell_blocks)
    mesh = mc.MEDCouplingUMesh.New(name, dim)
    mesh.allocateCells(total_cells)
    indices_by_group: dict[str, list[int]] = {}
    global_index = 0
    for norm_type, conn, group_name in cell_blocks:
        for row in conn:
            mesh.insertNextCell(norm_type, [int(v) for v in row])
            if group_name:
                indices_by_group.setdefault(group_name, []).append(global_index)
            global_index += 1
    mesh.setCoords(node_coords)
    mesh.finishInsertingCells()
    old_to_new = mesh.sortCellsInMEDFileFrmt()
    group_arrays = []
    for group_name, indices in indices_by_group.items():
        remapped = [old_to_new[i] for i in indices]
        arr = mc.DataArrayInt.New(remapped)
        arr.setName(group_name)
        group_arrays.append(arr)
    return mesh, group_arrays


def _build_seg2_spring_level(name: str, coords: "mc.DataArrayDouble", pairs: np.ndarray) \
        -> tuple["mc.MEDCouplingUMesh", list["mc.DataArrayInt"]]:
    """Build the level-(-2) 1-D mesh of wall/foundation interface spring
    elements: one zero-length SEG2 per (wall_node, foundation_node) pair,
    each in its OWN group (``SPR_WF_00000``, ...) so
    ``generate_three_body_case.py`` can give every element its own
    ``AFFE_CARA_ELEM``/``DISCRET`` stiffness (tributary area varies per
    node), plus one combined group spanning every spring element for
    ``AFFE_MODELE``."""
    mesh = mc.MEDCouplingUMesh.New(name, 1)
    mesh.allocateCells(len(pairs))
    for row in pairs:
        mesh.insertNextCell(mc.NORM_SEG2, [int(row[0]), int(row[1])])
    mesh.setCoords(coords)
    mesh.finishInsertingCells()
    old_to_new = mesh.sortCellsInMEDFileFrmt()

    all_arr = mc.DataArrayInt.New(sorted(int(old_to_new[i]) for i in range(len(pairs))))
    all_arr.setName("WALL_FOUNDATION_SPRINGS")
    per_element_arrays = [all_arr]
    for i in range(len(pairs)):
        arr = mc.DataArrayInt.New([int(old_to_new[i])])
        arr.setName(f"SPR_WF_{i:05d}")
        per_element_arrays.append(arr)
    return mesh, per_element_arrays


def convert(npz_path: Path, med_path: Path) -> None:
    data = np.load(npz_path, allow_pickle=True)
    nodes = data["nodes"]
    coords = mc.DataArrayDouble(nodes.tolist(), nodes.shape[0], nodes.shape[1])
    coords.setInfoOnComponents(["X [m]", "Y [m]", "Z [m]"])

    volume_blocks = [
        (mc.NORM_TETRA4, data["tetra4_concrete"], "CONCRETE"),
        (mc.NORM_TETRA4, data["tetra4_bedrock"], "BEDROCK"),
    ]
    volume_mesh, volume_group_arrays = _build_level(MESH_NAME, 3, coords, volume_blocks)
    if "concrete_body_ids" in data:
        for body, name in ((1, "WALL"), (3, "FOUNDATION")):
            ids = np.flatnonzero(data["concrete_body_ids"] == body)
            if len(ids):
                group = mc.DataArrayInt.New(ids.tolist())
                group.setName(name)
                volume_group_arrays.append(group)

    face_blocks = [
        (mc.NORM_TRI3, data["tri3_upstream"], "UPSTREAM"),
        (mc.NORM_TRI3, data["tri3_downstream"], "DOWNSTREAM"),
        (mc.NORM_TRI3, data["tri3_crest"], "CREST"),
        (mc.NORM_TRI3, data["tri3_rock_bottom"], "ROCK_BOTTOM"),
        (mc.NORM_TRI3, data["tri3_rock_sides"], "ROCK_SIDES"),
    ]
    face_mesh, face_group_arrays = _build_level(MESH_NAME, 2, coords, face_blocks)

    poi1_nodes = data["support_nodes"]
    point_blocks = [(mc.NORM_POINT1, poi1_nodes.reshape(-1, 1), "OUTER_SPRINGS")]
    point_mesh, point_group_arrays = _build_level(MESH_NAME, 0, coords, point_blocks)
    for i in range(len(poi1_nodes)):
        group = mc.DataArrayInt.New([i])
        group.setName(f"SPR_OUT_{i:05d}")
        point_group_arrays.append(group)

    spring_pairs = data["seg2_wall_foundation_springs"]
    spring_mesh, spring_group_arrays = _build_seg2_spring_level(MESH_NAME, coords, spring_pairs)

    mfu = mc.MEDFileUMesh.New()
    mfu.setName(MESH_NAME)
    mfu.setMeshAtLevel(0, volume_mesh)
    mfu.setMeshAtLevel(-1, face_mesh)
    mfu.setMeshAtLevel(-2, spring_mesh)
    mfu.setMeshAtLevel(-3, point_mesh)
    mfu.setGroupsAtLevel(0, volume_group_arrays)
    mfu.setGroupsAtLevel(-1, face_group_arrays)
    mfu.setGroupsAtLevel(-2, spring_group_arrays)
    mfu.setGroupsAtLevel(-3, point_group_arrays)

    med_path.parent.mkdir(parents=True, exist_ok=True)
    mfu.write(str(med_path), 2)
    print(f"WROTE {med_path} ({med_path.stat().st_size} bytes)")
    print("volume groups:", sorted(a.getName() for a in volume_group_arrays))
    print("face groups:", sorted(a.getName() for a in face_group_arrays))
    print("spring elements:", spring_mesh.getNumberOfCells())
    print("level0 cells:", volume_mesh.getNumberOfCells(), "level-1 cells:", face_mesh.getNumberOfCells())


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <three_body_mesh_arrays.npz> <output mesh.med>", file=sys.stderr)
        raise SystemExit(2)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
