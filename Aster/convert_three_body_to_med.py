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
    # Both native joint types sort after TETRA4 in MED order, preserving body indices.
    if "hexa8_joints" in data and len(data["hexa8_joints"]):
        volume_blocks.append((mc.NORM_HEXA8, data["hexa8_joints"], "JOINTS"))
    elif "penta6_joints" in data and len(data["penta6_joints"]):
        volume_blocks.append((mc.NORM_PENTA6, data["penta6_joints"], "JOINTS"))
    volume_mesh, volume_group_arrays = _build_level(MESH_NAME, 3, coords, volume_blocks)
    if "concrete_body_ids" in data:
        for body, name in ((1, "WALL"), (3, "FOUNDATION")):
            ids = np.flatnonzero(data["concrete_body_ids"] == body)
            if len(ids):
                group = mc.DataArrayInt.New(ids.tolist())
                group.setName(name)
                volume_group_arrays.append(group)
    if "damaged_cells" in data:
        selected = np.asarray(data["damaged_cells"])
        if (selected.dtype != np.dtype(bool)
                or selected.shape != (len(data["tetra4_concrete"]) + len(data["tetra4_bedrock"]),)
                or selected[len(data["tetra4_concrete"]):].any()
                or np.any(data["concrete_body_ids"][selected[:len(data["tetra4_concrete"])]] != 1)):
            raise ValueError("Damaged material group must contain only selected wall tetrahedra")
        if selected.any():
            group = mc.DataArrayInt.New(np.flatnonzero(selected).tolist())
            group.setName("DAMAGED_ELEMENTS")
            volume_group_arrays.append(group)
    if "damaged_joint_faces" in data and "hexa8_joints" in data:
        for selected, name in ((True, "DAMAGED_JOINTS"), (False, "INTACT_JOINTS")):
            mask = data["damaged_joint_faces"][data["joint_parent_faces"]] == selected
            ids = np.flatnonzero(mask) + len(data["tetra4_concrete"]) + len(data["tetra4_bedrock"])
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
    compact = bool(data.get("compact_spring_groups", False))
    if compact:
        tensors = data["support_matrices_n_per_m"]
        groups = {}
        for row, tensor in enumerate(tensors):
            key = tuple(tensor[i, j] for j in range(3) for i in range(3))
            groups.setdefault(key, []).append(row)
        for index, rows in enumerate(groups.values()):
            group = mc.DataArrayInt.New(rows)
            group.setName(f"K_OUT_{index:05d}")
            point_group_arrays.append(group)
    else:
        for i in range(len(poi1_nodes)):
            group = mc.DataArrayInt.New([i])
            group.setName(f"SPR_OUT_{i:05d}")
            point_group_arrays.append(group)

    spring_pairs = data["seg2_wall_foundation_springs"]
    spring_mesh, spring_group_arrays = _build_seg2_spring_level(MESH_NAME, coords, spring_pairs)
    if compact:
        # The per-element groups already preserve the MED sort permutation.
        sorted_ids = [int(group[0]) for group in spring_group_arrays[1:]]
        groups = {}
        for row, stiffness in enumerate(data["interface_stiffness_n_per_m"]):
            groups.setdefault(tuple(stiffness), []).append(sorted_ids[row])
        spring_group_arrays = spring_group_arrays[:1]
        for index, rows in enumerate(groups.values()):
            group = mc.DataArrayInt.New(rows)
            group.setName(f"K_IF_{index:05d}")
            spring_group_arrays.append(group)

    mfu = mc.MEDFileUMesh.New()
    mfu.setName(MESH_NAME)
    mfu.setMeshAtLevel(0, volume_mesh)
    mfu.setMeshAtLevel(-1, face_mesh)
    mfu.setMeshAtLevel(-2, spring_mesh)
    if len(poi1_nodes):
        mfu.setMeshAtLevel(-3, point_mesh)
    mfu.setGroupsAtLevel(0, volume_group_arrays)
    mfu.setGroupsAtLevel(-1, face_group_arrays)
    mfu.setGroupsAtLevel(-2, spring_group_arrays)
    if len(poi1_nodes):
        mfu.setGroupsAtLevel(-3, point_group_arrays)

    med_path.parent.mkdir(parents=True, exist_ok=True)
    node_groups = []
    if "fixed_support_nodes" in data and len(data["fixed_support_nodes"]):
        group = mc.DataArrayInt.New(data["fixed_support_nodes"].tolist())
        group.setName("FIXED_SEAT")
        node_groups.append(group)
    if "tie_node_pairs" in data and len(data["tie_node_pairs"]):
        # One single-node group per side of each tie, so the case pairs nodes explicitly
        # (coincident split copies make geometric pairing ambiguous).
        for index, (node, copy) in enumerate(data["tie_node_pairs"].tolist()):
            for name, value in ((f"TIEA{index:05d}", node), (f"TIEB{index:05d}", copy)):
                group = mc.DataArrayInt.New([int(value)])
                group.setName(name)
                node_groups.append(group)
    if node_groups:
        mfu.setGroupsAtLevel(1, node_groups)
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
