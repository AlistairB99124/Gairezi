"""Source-pinned quadratic topology and consistently integrated spring operators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tempfile
import time

import numpy as np

from Aster.build_three_body_mesh_arrays import _read_gmsh_ascii
from Aster.quadratic_surface import compact_blocks, surface_blocks
from Aster.source_contract import real_value, sha256, sif_boundaries, target

TET_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
TRI_EDGES = ((0, 1), (1, 2), (2, 0))


def elevate(nodes: np.ndarray, cells: np.ndarray, faces: np.ndarray,
            quadratic_cells: np.ndarray | None = None) -> dict:
    """Elevate selected, topologically split bodies; -1 pads retained linear cells."""
    nodes, cells, faces = np.asarray(nodes), np.asarray(cells), np.asarray(faces)
    if (nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all()
            or cells.ndim != 2 or cells.shape[1] != 4 or not len(cells)
            or faces.ndim != 2 or faces.shape[1] != 3
            or not np.issubdtype(cells.dtype, np.integer)
            or not np.issubdtype(faces.dtype, np.integer)
            or cells.min() < 0 or cells.max() >= len(nodes)
            or (faces.size and (faces.min() < 0 or faces.max() >= len(nodes)))):
        raise ValueError("Expected finite nodes, TETRA4 cells and TRI3 faces")
    edges = np.sort(np.concatenate([cells[:, pair] for pair in TET_EDGES]), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    if quadratic_cells is not None:
        mask = np.asarray(quadratic_cells)
        if mask.dtype != np.bool_ or mask.shape != (len(cells),) or not mask.any():
            raise ValueError("Quadratic cell mask must select at least one volume cell")
        if np.intersect1d(cells[mask], cells[~mask]).size:
            raise ValueError("Mixed-order bodies require split node identities to avoid hanging nodes")
        selected = np.unique(inverse.reshape(6, len(cells))[:, mask])
        mapping = np.full(len(unique), -1, dtype=np.int64)
        mapping[selected] = np.arange(len(selected)) + len(nodes)
    else:
        selected = np.arange(len(unique))
        mapping = selected + len(nodes)
    tet10 = np.column_stack((cells, mapping[inverse.reshape(6, len(cells)).T]))
    # Sorted integer keys permit vectorized surface lookup without coordinate welding.
    stride = len(nodes)
    edge_keys = unique[:, 0] * stride + unique[:, 1]
    surface_edges = np.sort(np.stack([faces[:, pair] for pair in TRI_EDGES], axis=1), axis=2)
    keys = surface_edges[..., 0] * stride + surface_edges[..., 1]
    indices = np.searchsorted(edge_keys, keys)
    if indices.size and (np.any(indices >= len(unique)) or
                         not np.array_equal(edge_keys[np.minimum(indices, len(unique) - 1)], keys)):
        raise ValueError("Surface edge does not belong to the volume topology")
    return {
        "nodes": np.vstack([nodes, (nodes[unique[selected, 0]] + nodes[unique[selected, 1]]) / 2]),
        "tetra10": tet10, "tri6": np.column_stack([faces, mapping[indices]]),
        "midpoint_parent_edges": unique[selected],
    }


def operators(
    nodes: np.ndarray, faces: np.ndarray, tags: np.ndarray, blocks: dict[int, str], diagnostic: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Integrate each saved spring coefficient on matching quadratic traces."""
    contributions, operators, interfaces, supports = [], [], [], []
    def trace(tag):
        selected = faces[tags == tag]
        if selected.shape[1] == 6 and np.all(selected[:, 3:] == -1):
            return selected[:, :3]
        if np.any(selected < 0):
            raise ValueError(f"Inconsistent surface trace order on source tag {tag}")
        return selected
    for number, block in blocks.items():
        mortar = re.search(r"Mortar BC = Integer (\d+)", block)
        if mortar is None:
            continue
        slave = trace(target(block))
        master = trace(target(blocks[int(mortar[1])]))
        if not len(slave) or not len(master):
            raise ValueError(f"Empty source spring interface {number}")
        lookup = {}
        for node in np.unique(master):
            key = tuple(np.round(nodes[node], 8))
            if key in lookup:
                raise ValueError(f"Ambiguous quadratic master node on interface {number}")
            lookup[key] = node
        try:
            xyz = nodes[slave[:, :3]]
            required = (np.concatenate([xyz, (xyz + xyz[:, [1, 2, 0]]) / 2], axis=1)
                        if master.shape[1] == 6 else xyz)
            corresponding = np.array([[lookup[tuple(np.round(point, 8))] for point in face]
                                      for face in required], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"Nonmatching quadratic interface {number}") from error
        # Matching points are insufficient: the other side must contain these faces.
        master_faces = {tuple(sorted(face)) for face in master[:, :3]}
        if any(tuple(sorted(face)) not in master_faces for face in corresponding[:, :3]):
            raise ValueError(f"Nonconforming opposite face topology on interface {number}")
        coefficients = np.array([real_value(block, f"Mortar BC Coefficient Displacement {i}")
                                 for i in (1, 2, 3)])
        if np.any(coefficients <= 0):
            raise ValueError("Nonpositive source spring interface coefficient")
        pairs, matrix = surface_blocks(nodes, slave, np.diag(coefficients), corresponding)
        contributions.append(pairs)
        operators.append(matrix)
        interfaces.append({"source_boundary": number, "faces": len(slave),
                           "slave_trace_nodes": slave.shape[1], "master_trace_nodes": master.shape[1],
                           "coefficient_n_per_m3": coefficients.tolist()})
    for semantic, face in diagnostic["rock_outer_support"]["faces"].items():
        saved_tag = diagnostic["boundary_id_map"][semantic]
        matches = [block for block in blocks.values() if target(block) == saved_tag]
        if len(matches) != 1:
            raise ValueError("Missing/ambiguous quadratic outer support boundary")
        block = matches[0]
        matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                key = f"Spring {i + 1}{j + 1}"
                if re.search(rf"(?mi)^\s*{key}\s*=", block):
                    matrix[i, j] = real_value(block, key)
        normal = np.asarray(face["outward_normal"])
        expected = face["normal_stiffness_n_per_m3"] * np.outer(normal, normal)
        if not np.allclose(matrix, expected, rtol=1e-10, atol=1e-3):
            raise ValueError("Quadratic support coefficient differs from source diagnostic")
        surface = trace(saved_tag)
        pairs, values = surface_blocks(nodes, surface, matrix)
        contributions.append(pairs)
        operators.append(values)
        supports.append({"source_face_tag": saved_tag, "faces": len(surface),
                         "trace_nodes": surface.shape[1],
                         "matrix_n_per_m3": matrix.tolist()})
    if not interfaces or not supports:
        raise ValueError("Source quadratic mesh requires audited joints and outer supports")
    pairs, matrix = compact_blocks(np.concatenate(contributions), np.concatenate(operators))
    return pairs, matrix, {"interfaces": interfaces, "supports": supports}


def prepare(source_directory: Path, output_root: Path, wall_only: bool = False) -> dict:
    """Build a unique quadratic candidate; preparation never reports a solve."""
    started = time.monotonic()
    manifest_path = source_directory / "mesh_manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(source_manifest["source_mesh"])
    for name, digest in source_manifest["source_hashes"].items():
        if sha256(source.parent / name) != digest:
            raise ValueError(f"Source artifact changed: {name}")
    arrays_path = source_directory / "three_body_mesh_arrays.npz"
    with np.load(arrays_path) as data:
        nodes = data["nodes"]
        original_ids = data["source_node_ids"]
        concrete = data["tetra4_concrete"]
        other = data["tetra4_bedrock"]
        body_ids = np.concatenate((data["concrete_body_ids"], np.full(len(other), 2)))
        cells = np.concatenate((concrete, other))
    raw_nodes, raw_cells, raw_body_ids, raw_faces, tags = _read_gmsh_ascii(source)
    dense = np.full(len(raw_nodes), -1, dtype=np.int64)
    dense[original_ids] = np.arange(len(original_ids))
    if not np.array_equal(nodes, raw_nodes[original_ids]):
        raise ValueError("Saved linear archive coordinates differ from source identities")
    if len(cells) != len(raw_cells) or np.any(dense[raw_faces] < 0):
        raise ValueError("Quadratic candidate requires every original volume cell and surface node")
    # Check the entire volume inventory, not merely its size.
    original = np.sort(dense[raw_cells], axis=1)
    candidate = np.sort(cells, axis=1)
    order = np.lexsort(original.T[::-1])
    other_order = np.lexsort(candidate.T[::-1])
    if not np.array_equal(original[order], candidate[other_order]):
        raise ValueError("Saved linear archive altered the source volume topology")
    if not np.array_equal(raw_body_ids[order], body_ids[other_order]):
        raise ValueError("Saved linear archive altered source material/body assignments")
    elevated = elevate(nodes, cells, dense[raw_faces], body_ids == 1 if wall_only else None)
    blocks = sif_boundaries((source.parent / "dam_model.sif").read_text())
    diagnostic = json.loads((source.parent / "diagnostic.json").read_text())
    pairs, values, audit = operators(elevated["nodes"], elevated["tri6"], tags, blocks, diagnostic)
    output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="quadratic-", dir=output_root.resolve()))
    path = directory / "quadratic_mesh_arrays.npz"
    np.savez_compressed(path, **elevated, body_ids=body_ids, face_tags=tags,
                        surface_block_pairs=pairs, surface_block_matrices_n_per_m=values,
                        original_source_node_ids=original_ids)
    report = {
        "schema_version": 1,
        "status": "quadratic_candidate_prepared_not_solved",
        "directory": str(directory),
        "source_mesh_manifest": str(manifest_path.resolve()),
        "source_mesh_manifest_sha256": sha256(manifest_path),
        "source_array_sha256": sha256(arrays_path),
        "preparation_code_sha256": sha256(Path(__file__)),
        "surface_code_sha256": sha256(Path(__file__).with_name("quadratic_surface.py")),
        "source_hashes": source_manifest["source_hashes"],
        "quadratic_array_sha256": sha256(path),
        "original_nodes": len(nodes), "quadratic_nodes": len(elevated["nodes"]),
        "volume_cells": len(cells), "surface_faces": len(tags),
        "quadratic_faces": int(np.count_nonzero(elevated["tri6"][:, 3] >= 0)),
        "linear_faces": int(np.count_nonzero(elevated["tri6"][:, 3] < 0)),
        "element_order_policy": "quadratic_wall_linear_elastic_domains" if wall_only else "all_quadratic",
        "quadratic_volume_cells": int(np.count_nonzero(elevated["tetra10"][:, 4] >= 0)),
        "linear_volume_cells": int(np.count_nonzero(elevated["tetra10"][:, 4] < 0)),
        "wall_cells": int(np.count_nonzero(body_ids == 1)),
        "foundation_cells": int(np.count_nonzero(body_ids == 3)),
        "surface_operator_node_pairs": len(pairs),
        "minimum_displacement_dofs_before_constraints": 3 * len(elevated["nodes"]),
        "surface_law": "exact consistent TRI3/TRI6 trace integration, not vertex-only or diagonal lumping",
        "audit": audit,
        "engineering_validated": False,
        "baseline_verified": False,
        "elapsed_s": time.monotonic() - started,
    }
    (directory / "quadratic_mesh_manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wall-only", action="store_true",
                        help="Elevate only the damageable wall, preserving split elastic bodies at source order")
    args = parser.parse_args()
    report = prepare(args.source_dir, args.output_dir, args.wall_only)
    print(json.dumps({key: value for key, value in report.items() if key != "audit"}, indent=2))


if __name__ == "__main__":
    main()
