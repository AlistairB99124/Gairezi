"""Bundled-MED interpreter entry point for source-pinned TETRA10 candidates."""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Script execution must also support the bundled native interpreter.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Aster.convert_three_body_to_med import mc, np
from Aster.quadratic_surface import compact_blocks, group_stiffness_blocks


def _level(coordinates, connectivity, norm, names):
    """Build a uniform cell level and explicit families in linear time."""
    count, width = connectivity.shape
    if (count + 1) * (width + 1) > np.iinfo(np.int32).max or len(coordinates) > np.iinfo(np.int32).max:
        raise ValueError("Quadratic candidate exceeds installed MED 32-bit connectivity capacity")
    mesh = mc.MEDCouplingUMesh.New("THREE_BODY", {mc.NORM_TETRA10: 3, mc.NORM_TRI6: 2,
                                               mc.NORM_SEG2: 1, mc.NORM_POINT1: 0}[norm])
    linear_width = {mc.NORM_TETRA10: 4, mc.NORM_TRI6: 3}.get(norm)
    linear = np.any(connectivity < 0, axis=1)
    if linear.any():
        if (linear_width is None or np.any(connectivity[:, :linear_width] < 0)
                or np.any(connectivity[linear, linear_width:] != -1)):
            raise ValueError("Invalid mixed-order padded MED connectivity")
        # MED requires cell types to be contiguous, with families reordered too.
        order = np.argsort(~linear, kind="stable")
        connectivity, linear = connectivity[order], linear[order]
        names = names[order]
        types = np.where(linear, {10: mc.NORM_TETRA4, 6: mc.NORM_TRI3}[width], norm)
        padded = np.column_stack([types, connectivity])
        flat = padded[padded >= 0].astype(np.int32)
        sizes = np.where(linear, linear_width + 1, width + 1)
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
    else:
        flat = np.column_stack([np.full(count, norm), connectivity]).astype(np.int32).ravel()
        offsets = np.arange(0, (count + 1) * (width + 1), width + 1, dtype=np.int32)
    mesh.setConnectivity(mc.DataArrayInt(flat.ravel()), mc.DataArrayInt(offsets), True)
    mesh.setCoords(coordinates)
    return mesh, names


def convert(archive: Path, output: Path) -> dict:
    started = time.monotonic()
    def progress(message):
        print(f"{time.monotonic() - started:.3f}s {message}", flush=True)
    with np.load(archive) as data:
        coordinates = mc.DataArrayDouble(data["nodes"])
        coordinates.setInfoOnComponents(["X [m]", "Y [m]", "Z [m]"])
        bodies, cells, faces, tags = data["body_ids"], data["tetra10"], data["tri6"], data["face_tags"]
        pairs, matrices = compact_blocks(data["surface_block_pairs"], data["surface_block_matrices_n_per_m"])
    progress("loaded quadratic archive")
    point_mask, link_mask = pairs[:, 0] == pairs[:, 1], pairs[:, 0] < pairs[:, 1]
    points, links = pairs[point_mask, 0], pairs[link_mask]
    # compact_blocks sorts by (row, column); verify every lower block explicitly.
    stride = len(coordinates)
    keys = pairs[:, 0] * stride + pairs[:, 1]
    transpose = np.searchsorted(keys, pairs[link_mask, 1] * stride + pairs[link_mask, 0])
    if (np.any(transpose >= len(keys)) or
            not np.array_equal(pairs[transpose], links[:, ::-1]) or
            not np.allclose(matrices[transpose], matrices[link_mask].transpose(0, 2, 1),
                            rtol=1e-12, atol=1e-8)):
        raise ValueError("Quadratic spring archive has missing/nonsymmetric transpose blocks")
    med = mc.MEDFileUMesh.New()
    family_info, group_info, next_family = {}, {}, -1
    for level, conn, norm, names, aggregate in (
        (0, cells, mc.NORM_TETRA10,
         np.array([{1: "WALL", 3: "FOUNDATION", 2: "REFERENCE"}[int(tag)] for tag in bodies]), "VOLUMES"),
        (-1, faces, mc.NORM_TRI6, np.array([f"FACE_{int(tag)}" for tag in tags]), "SURFACES"),
        (-2, links, mc.NORM_SEG2, np.full(len(links), "QS_LINKS"), "QS_LINKS"),
        (-3, points.reshape(-1, 1), mc.NORM_POINT1,
         np.full(len(points), "QS_POINTS"), "QS_POINTS"),
    ):
        mesh, names = _level(coordinates, conn, norm, names)
        med.setMeshAtLevel(level, mesh)
        distinct, inverse = np.unique(names, return_inverse=True)
        family_ids = np.arange(next_family, next_family - len(distinct), -1, dtype=np.int32)
        families = [f"F{-int(value)}" for value in family_ids]
        family_info.update(zip(families, family_ids.tolist()))
        group_info.update({str(name): [family] for name, family in zip(distinct, families)})
        if aggregate not in group_info:
            group_info[aggregate] = families
        med.setFamilyFieldArr(level, mc.DataArrayInt(family_ids[inverse]))
        next_family -= len(distinct)
        progress(f"built level {level}: {len(conn)} elements, {len(distinct)} groups")
    for name, identifier in family_info.items():
        med.setFamilyId(name, identifier)
    for name, families in group_info.items():
        med.setFamiliesOnGroup(name, families)
    progress("assigned explicit MED families")
    med.write(str(output), 2)
    progress("wrote MED mesh")
    save_discrete(output.with_name("quadratic_discrete.npz"), coordinates.toNumPyArray(),
                  points, links, matrices[point_mask], matrices[link_mask])
    progress("wrote discrete stiffness archive")
    return {"nodes": len(coordinates), "volume_cells": len(cells), "point_elements": len(points),
            "link_elements": len(links)}


def convert_discrete_benchmark(archive: Path, output: Path) -> None:
    with np.load(archive) as data:
        coordinates = mc.DataArrayDouble(data["nodes"])
        med = mc.MEDFileUMesh.New()
        for level, key, norm, group in (
            (0, "link_node_ids", mc.NORM_SEG2, "QS_LINKS"),
            (-1, "point_node_ids", mc.NORM_POINT1, "QS_POINTS"),
        ):
            connectivity = data[key].reshape(len(data[key]), -1)
            mesh, _ = _level(coordinates, connectivity, norm, None)
            med.setMeshAtLevel(level, mesh)
            ids = mc.DataArrayInt.New(list(range(len(connectivity))))
            ids.setName(group)
            med.setGroupsAtLevel(level, [ids])
        med.write(str(output), 2)


def save_discrete(path, nodes, points, links, point_matrices, link_matrices):
    grouped_points, point_ids, point_audit = group_stiffness_blocks(point_matrices)
    grouped_links, link_ids, link_audit = group_stiffness_blocks(link_matrices)
    np.savez_compressed(
        path, nodes=nodes, point_node_ids=points, link_node_ids=links,
        point_matrices_n_per_m=point_matrices, link_matrices_n_per_m=link_matrices,
        point_group_matrices_n_per_m=grouped_points, link_group_matrices_n_per_m=grouped_links,
        point_group_ids=point_ids, link_group_ids=link_ids,
    )
    import json
    path.with_suffix(".json").write_text(
        json.dumps({"point_encoding": point_audit, "link_encoding": link_audit}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--discrete-benchmark":
        convert_discrete_benchmark(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) == 3:
        print(convert(Path(sys.argv[1]), Path(sys.argv[2])))
    else:
        raise SystemExit("Expected quadratic_mesh_arrays.npz and output.med")
