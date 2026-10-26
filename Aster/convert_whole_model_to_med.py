"""Bundled-MED writer for the whole-model crack mesh (mixed TETRA4/TETRA10)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Aster.convert_three_body_to_med import MESH_NAME, _build_seg2_spring_level, mc, np  # noqa: E402

# Linux MEDCoupling builds use 64-bit identifiers; the Windows bundle uses 32-bit.
IDS = np.int64 if mc.MEDCouplingSizeOfIDs() == 64 else np.int32
VOLUME_GROUPS = ("CONCRETE", "BEDROCK", "WALL", "FOUNDATION", "DAMAGE_BAND", "SEED_LAYER",
                 "LINEAR_VOLUMES", "HANGING_MASTER")
FACE_GROUPS = ("UPSTREAM", "DOWNSTREAM", "CREST", "ROCK_BOTTOM", "ROCK_SIDES", "CRACK_FACES")


def _mixed_level(coordinates, connectivity, quadratic_type, linear_type, linear_width, dimension):
    """One level with linear cells first; returns the mesh and the new index of each input row."""
    connectivity = np.asarray(connectivity, dtype=np.int64)
    linear = connectivity[:, linear_width] < 0
    if np.any(connectivity[:, :linear_width] < 0) or np.any(connectivity[linear, linear_width:] != -1) \
            or np.any(connectivity[~linear] < 0):
        raise ValueError("Invalid mixed-order connectivity")
    order = np.argsort(~linear, kind="stable")
    rows, kinds = connectivity[order], linear[order]
    sizes = np.where(kinds, linear_width + 1, connectivity.shape[1] + 1)
    types = np.where(kinds, linear_type, quadratic_type)
    padded = np.column_stack([types, rows])
    flat = padded[padded >= 0]
    if len(coordinates) >= np.iinfo(np.int32).max or flat.size >= np.iinfo(np.int32).max:
        raise ValueError("Whole-model crack mesh exceeds 32-bit MED connectivity")
    mesh = mc.MEDCouplingUMesh.New(MESH_NAME, dimension)
    mesh.setConnectivity(mc.DataArrayInt(flat.astype(IDS)),
                         mc.DataArrayInt(np.concatenate([[0], np.cumsum(sizes)]).astype(IDS)), True)
    mesh.setCoords(coordinates)
    position = np.empty(len(order), dtype=np.int64)
    position[order] = np.arange(len(order))
    return mesh, position


def _group(name, indices):
    array = mc.DataArrayInt(np.sort(np.asarray(indices, dtype=np.int64)).astype(IDS))
    array.setName(name)
    return array


def convert(archive: Path, output: Path) -> dict:
    with np.load(archive, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    coordinates = mc.DataArrayDouble(np.ascontiguousarray(arrays["nodes"]))
    coordinates.setInfoOnComponents(["X [m]", "Y [m]", "Z [m]"])
    volume, volume_position = _mixed_level(coordinates, arrays["cells"], mc.NORM_TETRA10,
                                           mc.NORM_TETRA4, 4, 3)
    volume_groups = []
    for name in VOLUME_GROUPS:
        mask = arrays[f"group_{name.lower()}"]
        if mask.shape != (len(arrays["cells"]),) or mask.dtype != np.bool_:
            raise ValueError(f"Invalid volume group mask {name}")
        if not mask.any():
            if name == "HANGING_MASTER":
                continue
            raise ValueError(f"Empty volume group {name}")
        volume_groups.append(_group(name, volume_position[np.flatnonzero(mask)]))
    faces = np.vstack([arrays[f"faces_{name.lower()}"] for name in FACE_GROUPS])
    face_mesh, face_position = _mixed_level(coordinates, faces, mc.NORM_TRI6, mc.NORM_TRI3, 3, 2)
    face_groups, start = [], 0
    for name in FACE_GROUPS:
        count = len(arrays[f"faces_{name.lower()}"])
        if count:
            face_groups.append(_group(name, face_position[start:start + count]))
        start += count
    spring_mesh, spring_groups = _build_seg2_spring_level(MESH_NAME, coordinates,
                                                          arrays["spring_pairs"])
    support = arrays["support_nodes"].reshape(-1, 1)
    point_mesh = mc.MEDCouplingUMesh.New(MESH_NAME, 0)
    point_mesh.setConnectivity(
        mc.DataArrayInt(np.column_stack([np.full(len(support), mc.NORM_POINT1), support])
                        .astype(IDS).ravel()),
        mc.DataArrayInt(np.arange(0, 2 * len(support) + 1, 2, dtype=IDS)), True)
    point_mesh.setCoords(coordinates)
    point_groups = [_group("OUTER_SPRINGS", np.arange(len(support)))]
    point_groups += [_group(f"SPR_OUT_{i:05d}", [i]) for i in range(len(support))]
    med = mc.MEDFileUMesh.New()
    med.setName(MESH_NAME)
    med.setMeshAtLevel(0, volume)
    med.setMeshAtLevel(-1, face_mesh)
    med.setMeshAtLevel(-2, spring_mesh)
    med.setMeshAtLevel(-3, point_mesh)
    med.setGroupsAtLevel(0, volume_groups)
    med.setGroupsAtLevel(-1, face_groups)
    med.setGroupsAtLevel(-2, spring_groups)
    med.setGroupsAtLevel(-3, point_groups)
    node_groups = [_group(name, arrays[key]) for name, key in (
        ("HANGING_NODES", "hanging_nodes"), ("CRACK_TOP", "crack_top_nodes"),
        ("CRACK_BOTTOM", "crack_bottom_nodes")) if len(arrays[key])]
    med.setGroupsAtLevel(1, node_groups)
    output.parent.mkdir(parents=True, exist_ok=True)
    med.write(str(output), 2)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Whole-model crack MED export produced no mesh")
    summary = {"nodes": int(len(arrays["nodes"])), "volume_cells": int(len(arrays["cells"])),
               "quadratic_cells": int(np.count_nonzero(arrays["cells"][:, 4] >= 0)),
               "face_cells": int(len(faces)), "springs": int(len(arrays["spring_pairs"])),
               "support_points": int(len(support)), "hanging_nodes": int(len(arrays["hanging_nodes"]))}
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: convert_whole_model_to_med.py <whole_model_mesh.npz> <mesh.med>")
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
