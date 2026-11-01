"""Independent Code_Aster corroboration of the FEniCSx automatic cohesive cracking run.

The FEniCSx run (FEniCSx/automatic_cracking.py) releases mesh facets of the wall
through a bespoke cohesive network solver. This module re-solves the same problem
with Code_Aster's own nonlinear machinery so the two codes can be compared:

* the job's Aster elastic-baseline mesh is split along exactly the FEniCSx
  candidate facets (same parent nodes, verified), and each split facet becomes a
  three zero-thickness, fully integrated ``3D_JOINT`` HEXA8 patches with exact
  P1 trace interpolation and the ``CZM_LIN_REG`` law
  (linear softening, ``SIGM_C`` = f_t, ``GC`` = G_f, so the opening at full
  separation is 2 G_f / f_t, the FEniCSx critical opening);
* every load (self weight, reservoir/tailwater pressure, crest surcharge) is
  ramped proportionally with ``STAT_NON_LINE`` through the same load factors
  FEniCSx used, recording joint damage and opening at each converged factor;
* the corroboration report compares first initiation, initiated/opened facets
  (count, area, overlap), peak opening and the cracked wall stress state.

Known modelling differences are reported, not hidden: ``CZM_LIN_REG`` uses a
mixed-mode jump norm (FEniCSx is mode I with tied tangential response), it is
an intrinsic law with a stiff initial penalty (FEniCSx is rigid until
initiation), Aster evaluates joint traction at integration points (FEniCSx
initiates on the larger one-cell facet stress), and no crack-face water pressure
is applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = 1
REPORT_NAME = "aster_cohesive_report.json"
STEPS_NAME = "aster_cohesive_steps.npz"
STATUS_NAME = "aster_cohesive_status.json"
SPLIT_ARRAYS_NAME = "three_body_mesh_arrays.npz"
MESH_MANIFEST_NAME = "mesh_manifest.json"
CASE_NAME = "case.comm"
# Initial joint stiffness = factor * E / h, so the pre-initiation joint compliance
# is a 0.1 % perturbation of the surrounding concrete over one element.
JOINT_PENALTY_FACTOR = 1000.0
FIRST_LOAD_FACTOR = 0.5
LOAD_INCREMENT = 0.1
MEMORY_LIMIT_MB = 8192
LOADED_TRI_KEYS = ("tri3_upstream", "tri3_downstream", "tri3_crest", "tri3_rock_bottom", "tri3_rock_sides")
# Code_Aster refuses joint models with any cell above 200 vertex neighbours (VOLUFINI_3);
# the margin covers any difference in how Code_Aster counts them.
NEIGHBOUR_TARGET = 150
DIFFERENCES = (
    "CZM_LIN_REG damages on the mixed-mode jump norm; FEniCSx opens in mode I with a tied tangential response.",
    "Aster joints are intrinsic: a stiff initial penalty (1000 E/h) instead of FEniCSx's rigid tie before initiation.",
    "Aster evaluates traction and mixed-mode initiation at fully integrated joint points; "
    "FEniCSx initiates on the larger one-cell facet stress. Initiation factors need not match.",
    "No water pressure is applied inside opened cracks in Aster.",
)


class CorroborationError(RuntimeError):
    """The Aster corroboration cannot be prepared or run for this job."""


# ---------------------------------------------------------------- mesh split

def _codes(triples: np.ndarray, base: int) -> np.ndarray:
    ordered = np.sort(np.asarray(triples, dtype=np.int64), axis=1)
    return (ordered[:, 0] * base + ordered[:, 1]) * base + ordered[:, 2]


_TET_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


def _tri_area(points: np.ndarray, tri) -> float:
    a, b, c = points[list(tri)]
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def split_for_joints(arrays: dict, faces: np.ndarray) -> dict:
    """Duplicate the nodes of ``faces`` (parent node triples) so each face becomes a
    zero-thickness joint between its two tetrahedra.

    Around each crack node, tetrahedra stay attached through every shared face that
    is not a candidate face; each further connected group gets a copy of the node.
    Crack-tip nodes (one group) are not split, so the tip stays tied as in FEniCSx;
    their joints use a joint-only node copy listed in ``tie_node_pairs`` (node, copy)
    that the case ties back to the node. Returns the new arrays plus ``joints``
    (one PENTA6 row per face, in ``faces`` order: minus triangle then plus triangle).
    """
    nodes = np.asarray(arrays["nodes"], dtype=float)
    concrete = np.asarray(arrays["tetra4_concrete"], dtype=np.int64)
    bedrock = np.asarray(arrays["tetra4_bedrock"], dtype=np.int64)
    tets = np.vstack([concrete, bedrock])
    faces = np.asarray(faces, dtype=np.int64)
    base = len(nodes) + 1
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise CorroborationError("Candidate facets require parent-node triples")
    if not len(faces):
        result = {key: np.asarray(value).copy() for key, value in arrays.items()}
        result["tie_node_pairs"] = np.empty((0, 2), dtype=np.int64)
        return {"arrays": result, "joints": np.empty((0, 6), dtype=np.int64),
                "split_nodes": 0, "added_nodes": 0, "tip_ties": 0, "spring_shares": []}
    if faces.min() < 0 or faces.max() >= len(nodes):
        raise CorroborationError("Candidate facets reference nodes outside the Aster mesh")

    tet_face_codes = _codes(tets[:, _TET_FACES].reshape(-1, 3), base)
    order = np.argsort(tet_face_codes, kind="stable")
    sorted_codes = tet_face_codes[order]

    def owners(code: int) -> np.ndarray:
        lo, hi = np.searchsorted(sorted_codes, [code, code + 1])
        return order[lo:hi] // 4

    candidate_codes = _codes(faces, base)
    if len(np.unique(candidate_codes)) != len(faces):
        raise CorroborationError("Candidate facets are repeated")
    candidate_set = set(candidate_codes.tolist())
    face_tets = np.empty((len(faces), 2), dtype=np.int64)
    for index, code in enumerate(candidate_codes):
        found = owners(int(code))
        if len(found) != 2 or np.any(found >= len(concrete)):
            raise CorroborationError(f"Candidate facet {index} is not shared by exactly two concrete tetrahedra")
        face_tets[index] = found

    crack_nodes = np.unique(faces)
    rows, cols = np.nonzero(np.isin(tets, crack_nodes))
    hit = tets[rows, cols]
    by_node = np.argsort(hit, kind="stable")
    starts = np.searchsorted(hit[by_node], crack_nodes)
    ends = np.searchsorted(hit[by_node], crack_nodes, side="right")

    new_tets = tets.copy()
    new_points = [nodes]
    new_sources = []
    source_ids = np.asarray(arrays.get("source_node_ids", np.arange(len(nodes))), dtype=np.int64)
    copies: dict[int, list[tuple[int, np.ndarray]]] = {}
    next_id = len(nodes)
    for node, lo, hi in zip(crack_nodes.tolist(), starts, ends):
        incident = np.unique(rows[by_node[lo:hi]])
        parent = {int(t): int(t) for t in incident}

        def find(item: int) -> int:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        shared: dict[int, list[int]] = {}
        for tet in incident.tolist():
            others = [int(v) for v in tets[tet] if v != node]
            for drop in range(3):
                triple = [node] + [others[k] for k in range(3) if k != drop]
                shared.setdefault(int(_codes(np.array([triple]), base)[0]), []).append(tet)
        for code, members in shared.items():
            if len(members) == 2 and code not in candidate_set:
                a, b = find(members[0]), find(members[1])
                if a != b:
                    parent[max(a, b)] = min(a, b)
        groups: dict[int, list[int]] = {}
        for tet in incident.tolist():
            groups.setdefault(find(tet), []).append(tet)
        components = [np.array(sorted(group)) for _, group in sorted(groups.items())]
        if len(components) < 2:
            continue
        node_copies = [(node, components[0])]
        for component in components[1:]:
            for tet in component:
                new_tets[tet][tets[tet] == node] = next_id
            node_copies.append((next_id, component))
            new_points.append(nodes[node][None, :])
            new_sources.append(source_ids[node])
            next_id += 1
        copies[node] = node_copies

    support = np.unique(np.concatenate((
        np.asarray(arrays["support_nodes"], dtype=np.int64),
        np.asarray(arrays.get("support_interpolation_parents", []), dtype=np.int64).ravel(),
    )))
    if np.any(np.isin(support, list(copies))):
        raise CorroborationError("A candidate facet node carries an outer support spring")
    points = np.vstack(new_points)

    def remap(tri, tet: int) -> list[int]:
        return [int(new_tets[tet][list(tets[tet]).index(v)]) for v in tri]

    joints = np.empty((len(faces), 6), dtype=np.int64)
    for index, (tri, (minus, plus)) in enumerate(zip(faces.tolist(), face_tets.tolist())):
        a, b, c = nodes[tri]
        normal = np.cross(b - a, c - a)
        # Code_Aster's 3D_JOINT measures an opening as a positive normal jump only when this
        # right-hand normal points away from the plus side (verified on a fixture run).
        if np.dot(points[new_tets[plus]].mean(axis=0) - (a + b + c) / 3.0, normal) > 0:
            tri = [tri[0], tri[2], tri[1]]
        joints[index] = remap(tri, minus) + remap(tri, plus)

    # Code_Aster rejects joints with repeated nodes, so the plus face of a joint uses a
    # joint-only copy of each crack-front node, tied back to it in the case (tie_node_pairs).
    tip_copy: dict[int, int] = {}
    for row in joints:
        for k in range(3):
            if row[k] == row[k + 3]:
                node = int(row[k])
                if node not in tip_copy:
                    tip_copy[node] = len(points)
                    points = np.vstack([points, points[node][None, :]])
                    new_sources.append(int(source_ids[node] if node < len(source_ids)
                                           else new_sources[node - len(source_ids)]))
                row[k + 3] = tip_copy[node]

    result = {key: np.asarray(value) for key, value in arrays.items()}
    result["tie_node_pairs"] = np.array(sorted(tip_copy.items()), dtype=np.int64).reshape(-1, 2)
    result["nodes"] = points
    result["tetra4_concrete"] = new_tets[:len(concrete)]
    result["tetra4_bedrock"] = new_tets[len(concrete):]
    result["source_node_ids"] = np.concatenate([source_ids, np.asarray(new_sources, dtype=np.int64)])
    loaded_codes = set()
    for key in LOADED_TRI_KEYS:
        tris = np.asarray(arrays[key], dtype=np.int64).copy()
        loaded_codes.update(_codes(tris, base).tolist()) if len(tris) else None
        touched = np.flatnonzero(np.isin(tris, list(copies)).any(axis=1))
        for row in touched:
            found = owners(int(_codes(tris[row:row + 1], base)[0]))
            if len(found) != 1:
                raise CorroborationError(f"Loaded face {key}[{row}] is not a boundary face")
            tris[row] = remap(tris[row].tolist(), int(found[0]))
        result[key] = tris

    pairs = np.asarray(arrays["seg2_wall_foundation_springs"], dtype=np.int64)
    areas = np.asarray(arrays["wall_foundation_tributary_area_m2"], dtype=float)
    stiffness = np.asarray(arrays["interface_stiffness_n_per_m"], dtype=float)
    names = np.asarray(arrays["interface_names"])
    new_pairs, new_areas, new_stiffness, new_names, spring_shares = [pairs.copy()], [areas.copy()], \
        [stiffness.copy()], [names.copy()], []
    side_nodes = [set(pairs[:, 0].tolist()), set(pairs[:, 1].tolist())]
    added = 0
    for row in np.flatnonzero(np.isin(pairs, list(copies)).any(axis=1)):
        split_ends = [end for end in (0, 1) if int(pairs[row, end]) in copies]
        if len(split_ends) != 1:
            raise CorroborationError(f"Interface spring {row} joins two split nodes")
        end = split_ends[0]
        node = int(pairs[row, end])
        weights = []
        for _, component in copies[node]:
            weight = 0.0
            for tet in component.tolist():
                others = [int(v) for v in tets[tet] if v != node]
                for drop in range(3):
                    tri = [node] + [others[k] for k in range(3) if k != drop]
                    code = int(_codes(np.array([tri]), base)[0])
                    if code in loaded_codes or len(owners(code)) != 1:
                        continue
                    if all(v in side_nodes[end] for v in tri):
                        weight += _tri_area(nodes, tri) / 3.0
            weights.append(weight)
        weights = np.asarray(weights)
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(weights), 1.0 / len(weights))
        # Copies with no interface area keep no spring; the original row moves to the first sharing copy.
        sharing = [(copy_id, float(weight)) for (copy_id, _), weight in zip(copies[node], weights) if weight > 0]
        new_pairs[0][row, end] = sharing[0][0]
        share_rows = [int(row)]
        for copy_id, weight in sharing[1:]:
            pair = pairs[row].copy()
            pair[end] = copy_id
            new_pairs.append(pair[None, :])
            new_areas.append(np.array([areas[row] * weight]))
            new_stiffness.append(stiffness[row][None, :] * weight)
            new_names.append(names[row:row + 1])
            share_rows.append(len(pairs) + added)
            added += 1
        new_areas[0][row] = areas[row] * sharing[0][1]
        new_stiffness[0][row] = stiffness[row] * sharing[0][1]
        spring_shares.append({"row": int(row), "rows": share_rows, "fractions": [w for _, w in sharing]})
    result["seg2_wall_foundation_springs"] = np.vstack(new_pairs)
    result["wall_foundation_tributary_area_m2"] = np.concatenate(new_areas)
    result["interface_stiffness_n_per_m"] = np.vstack(new_stiffness)
    result["interface_names"] = np.concatenate(new_names)
    return {
        "arrays": result,
        "joints": joints,
        "split_nodes": len(copies),
        "added_nodes": next_id - len(nodes),
        "tip_ties": len(tip_copy),
        "spring_shares": spring_shares,
    }


def split_manifest(manifest: dict, split: dict) -> dict:
    """Mesh manifest for the split mesh: interface springs follow the split rows."""
    result = json.loads(json.dumps(manifest))
    springs = result["wall_foundation_springs"]
    arrays = split["arrays"]
    if len(springs) + sum(len(share["rows"]) - 1 for share in split["spring_shares"]) != \
            len(arrays["seg2_wall_foundation_springs"]):
        raise CorroborationError("Mesh manifest springs do not match the mesh arrays")
    template = {}
    for share in split["spring_shares"]:
        for row, fraction in zip(share["rows"], share["fractions"]):
            template[row] = (share["row"], fraction)
    for row, (source, fraction) in sorted(template.items()):
        entry = dict(manifest["wall_foundation_springs"][source])
        entry["group"] = f"SPR_WF_{row:05d}"
        entry["tributary_area_m2"] = float(entry["tributary_area_m2"]) * fraction
        entry["stiffness_n_per_m"] = (np.asarray(entry["stiffness_n_per_m"], dtype=float) * fraction).tolist()
        if row < len(springs):
            springs[row] = entry
        else:
            springs.append(entry)
    result["joints"] = {"group": "JOINTS", "cells": int(len(arrays.get("hexa8_joints", split["joints"]))),
                        "candidate_faces": int(len(split["joints"])),
                        "split_nodes": split["split_nodes"], "added_nodes": split["added_nodes"]}
    return result


def integrate_joint_faces(split: dict) -> None:
    """Three fully integrated native quadrilateral patches per original P1 triangle.

    Edge/centroid displacements are exact interpolations, not new physical freedoms.
    HEXA8 3D_JOINT uses FPG4; unlike centroid PENTA6 it integrates the intact affine
    jump energy exactly, including rotations of otherwise isolated tetrahedra.
    """
    arrays = split["arrays"]
    points = np.asarray(arrays["nodes"], dtype=float)
    added_points, interpolation_nodes, interpolation_parents, interpolation_weights = [], [], [], []
    node_cache: dict[tuple[int, ...], int] = {}

    def interpolate(parents: tuple[int, ...]) -> int:
        parents = tuple(sorted(parents))
        if parents not in node_cache:
            node = len(points) + len(added_points)
            node_cache[parents] = node
            added_points.append(points[list(parents)].mean(axis=0))
            interpolation_nodes.append(node)
            interpolation_parents.append((*parents, *([parents[-1]] * (3 - len(parents)))))
            interpolation_weights.append((*([1.0 / len(parents)] * len(parents)),
                                          *([0.0] * (3 - len(parents)))))
        return node_cache[parents]

    patches = []
    for joint in split["joints"]:
        sides = []
        for vertices in (joint[:3], joint[3:]):
            a, b, c = map(int, vertices)
            ab, bc, ca = (interpolate(pair) for pair in ((a, b), (b, c), (c, a)))
            centre = interpolate((a, b, c))
            sides.append(((a, ab, centre, ca), (ab, b, bc, centre), (centre, bc, c, ca)))
        patches.extend((*minus, *plus) for minus, plus in zip(*sides))
    arrays["nodes"] = np.vstack((points, np.asarray(added_points).reshape(-1, 3)))
    # Interpolation nodes have no single source vertex identity.
    arrays["source_node_ids"] = np.concatenate((arrays["source_node_ids"],
                                               np.full(len(added_points), -1, dtype=np.int64)))
    arrays["hexa8_joints"] = np.asarray(patches, dtype=np.int64).reshape(-1, 8)
    arrays["joint_parent_faces"] = np.repeat(np.arange(len(split["joints"]), dtype=np.int64), 3)
    arrays["interpolation_nodes"] = np.asarray(interpolation_nodes, dtype=np.int64)
    arrays["interpolation_parents"] = np.asarray(interpolation_parents, dtype=np.int64).reshape(-1, 3)
    arrays["interpolation_weights"] = np.asarray(interpolation_weights, dtype=float).reshape(-1, 3)


def _neighbour_counts(cells: list[np.ndarray], node_count: int, threshold: int) -> tuple[np.ndarray, list]:
    """Cells sharing at least one node with each cell, for every cell that may exceed ``threshold``.

    Returns (cell indices over the threshold, per-node incidence lists)."""
    flat = np.concatenate([c.ravel() for c in cells])
    owner = np.concatenate([np.repeat(np.arange(len(c)) + offset, c.shape[1])
                            for c, offset in zip(cells, np.cumsum([0] + [len(c) for c in cells[:-1]]))])
    order = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order], np.arange(node_count + 1))
    valence = np.diff(starts)
    over = []
    for c, offset in zip(cells, np.cumsum([0] + [len(c) for c in cells[:-1]])):
        bound = valence[c].sum(axis=1) - c.shape[1]
        for row in np.flatnonzero(bound > threshold):
            neighbours = np.unique(np.concatenate([owner[order[starts[v]:starts[v + 1]]] for v in c[row]]))
            if len(neighbours) - 1 > threshold:
                over.append(int(row + offset))
    return np.asarray(over, dtype=np.int64), (order, starts, owner, valence)


def relieve_neighbourhoods(split: dict, threshold: int = NEIGHBOUR_TARGET) -> int:
    """Keep every cell's vertex-neighbour count within Code_Aster's ``3D_JOINT`` limit.

    Joint elements need a cell neighbourhood, which Code_Aster refuses above 200
    neighbours per cell; badly graded bedrock tetrahedra can exceed that. Each
    offending node used only by tetrahedra is given a copy that takes half of its
    tetrahedra and is tied back to it (``tie_node_pairs``), which leaves the
    displacement field and stiffness unchanged. Returns the number of copies."""
    arrays = split["arrays"]
    concrete_count = len(arrays["tetra4_concrete"])
    tets = np.vstack([arrays["tetra4_concrete"], arrays["tetra4_bedrock"]]).astype(np.int64)
    points = np.asarray(arrays["nodes"], dtype=float)
    sources = list(np.asarray(arrays["source_node_ids"], dtype=np.int64))
    others = [np.asarray(split["joints"], dtype=np.int64)]
    others += [np.asarray(arrays[key], dtype=np.int64).reshape(-1, 3) for key in LOADED_TRI_KEYS]
    others += [np.asarray(arrays["seg2_wall_foundation_springs"], dtype=np.int64).reshape(-1, 2),
               np.asarray(arrays["support_nodes"], dtype=np.int64).reshape(-1, 1)]
    pinned = set(np.unique(np.concatenate([o.ravel() for o in others])).tolist())
    pairs = [tuple(pair) for pair in np.asarray(arrays["tie_node_pairs"], dtype=np.int64).reshape(-1, 2).tolist()]
    root = {copy: node for node, copy in pairs}
    pinned.update(root)
    pinned.update(node for node, _ in pairs)
    added = 0
    for _ in range(64):
        over, (order, starts, owner, valence) = _neighbour_counts([tets] + others, len(points), threshold)
        if not len(over):
            break
        chosen = set()
        for cell in over.tolist():
            if cell >= len(tets):
                raise CorroborationError("A joint, face or spring cell has too many neighbours for Code_Aster")
            free = [int(v) for v in tets[cell] if int(v) not in pinned]
            if not free:
                raise CorroborationError(f"Mesh cell {cell} has too many neighbours for Code_Aster joints")
            chosen.add(max(free, key=lambda v: valence[v]))
        for node in sorted(chosen):
            incident = owner[order[starts[node]:starts[node + 1]]]
            centroids = points[tets[incident]].mean(axis=1) - points[node]
            axis = np.linalg.svd(centroids - centroids.mean(axis=0), full_matrices=False)[2][0]
            projection = centroids @ axis
            moved = incident[projection > np.median(projection)]
            if not len(moved) or len(moved) == len(incident):
                raise CorroborationError(f"Cannot relieve the neighbourhood of mesh node {node}")
            copy = len(points)
            points = np.vstack([points, points[node][None, :]])
            sources.append(sources[node])
            tets[moved] = np.where(tets[moved] == node, copy, tets[moved])
            node_root = root.get(node, node)
            pairs.append((node_root, copy))
            root[copy] = node_root
            added += 1
    else:
        raise CorroborationError("Could not bring the mesh within Code_Aster's joint neighbour limit")
    arrays["nodes"] = points
    arrays["source_node_ids"] = np.asarray(sources, dtype=np.int64)
    arrays["tetra4_concrete"] = tets[:concrete_count]
    arrays["tetra4_bedrock"] = tets[concrete_count:]
    arrays["tie_node_pairs"] = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    split["relief_copies"] = added
    return added


# ---------------------------------------------------------------- inputs

def find_fenicsx_output(job_dir: Path) -> Path:
    """The single successful automatic cracking output of a job."""
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    run_id = job.get("automatic_cracking_run_id")
    if not run_id:
        raise CorroborationError("The job has no automatic cracking run")
    output = job_dir / "automatic_cracking" / run_id / "output"
    if not (output / "summary.json").is_file() or not (output / "cohesive_state.npz").is_file():
        raise CorroborationError("The automatic cracking run has no saved cohesive state")
    return output


def candidate_faces(state) -> np.ndarray:
    """FEniCSx candidate facets as Aster parent-node triples (FEniCSx face order)."""
    parents = np.asarray(state["parent_node_ids"], dtype=np.int64)
    minus = np.sort(parents[np.asarray(state["minus"])], axis=1)
    plus = np.sort(parents[np.asarray(state["plus"])], axis=1)
    if not np.array_equal(minus, plus):
        raise CorroborationError("FEniCSx candidate facet sides do not share parent nodes")
    return minus


def load_factors(summary: dict) -> list[float]:
    """The proportional load factors to solve: FEniCSx's first step, then fixed increments
    to the larger of 1 and the factor FEniCSx finished at."""
    final = max(1.0, float(summary.get("load_factor") or 1.0),
                float(summary.get("maximum_overload_factor") or 1.0))
    factors = [FIRST_LOAD_FACTOR]
    while factors[-1] < final - 1e-9:
        factors.append(round(min(final, factors[-1] + LOAD_INCREMENT), 10))
    return factors


def joint_parameters(arrays: dict, faces: np.ndarray, summary: dict, concrete_e_pa: float,
                     concrete_nu: float) -> dict:
    f_t = float(summary["tensile_strength_pa"])
    g_f = float(summary["fracture_energy_j_m2"])
    nodes = np.asarray(arrays["nodes"], dtype=float)
    edge_faces = faces if len(faces) else np.asarray(arrays["tetra4_concrete"])[:, :3]
    edges = np.concatenate([np.linalg.norm(nodes[edge_faces[:, i]] - nodes[edge_faces[:, (i + 1) % 3]], axis=1)
                            for i in range(3)])
    h = float(edges.mean())
    stiffness = JOINT_PENALTY_FACTOR * concrete_e_pa / h
    critical = 2.0 * g_f / f_t
    # CZM_LIN_REG leaves its elastic branch at the jump PENA_ADHERENCE * critical opening
    # (checked on a Code_Aster run), so its initial stiffness is f_t / (PENA_ADHERENCE * critical).
    adherence = float(np.clip(f_t / (stiffness * critical), 1e-12, 1.0))
    return {
        "law": "CZM_LIN_REG",
        "tensile_strength_pa": f_t,
        "fracture_energy_j_m2": g_f,
        "critical_opening_m": critical,
        "pena_adherence": adherence,
        "initial_stiffness_pa_per_m": f_t / (adherence * critical),
        "elastic_limit_opening_m": adherence * critical,
        "mean_facet_edge_m": h,
        "concrete_youngs_modulus_pa": float(concrete_e_pa),
        "concrete_poissons_ratio": float(concrete_nu),
    }


# ---------------------------------------------------------------- case

_ANCHORS = {
    "debut": "DEBUT(LANG='EN')",
    "model":    "MODELE = AFFE_MODELE(\n    MAILLAGE=MAIL,\n    AFFE=(\n",
    "chmat": "CHMAT = AFFE_MATERIAU(",
    "concrete_mat": "        _F(GROUP_MA='CONCRETE', MATER=CONCRETE_MAT),\n",
    "solve": "RESU = MECA_STATIQUE(",
}


def _comm_string(value) -> str:
    return repr(str(value))


def render_cohesive_comm(base_comm: str, joint: dict, factors: list[float], steps_path: Path,
                         status_path: Path, tie_pairs: int = 0,
                         interpolation: dict | None = None) -> str:
    """Turn the job's linear baseline case into the stepped cohesive case."""
    for name, anchor in _ANCHORS.items():
        if base_comm.count(anchor) != 1:
            raise CorroborationError(f"The baseline case no longer has its '{name}' section")
    comm = base_comm[:base_comm.index(_ANCHORS["solve"])]
    comm = comm.replace(_ANCHORS["debut"], (
        "import json\nimport time\n\nimport numpy as np\n\n"
        "from code_aster.Commands import *\nfrom code_aster import CA\n\n"
        "DEBUT(LANG='EN', ERREUR=_F(ERREUR_F='EXCEPTION'))"))
    has_joints = joint.get("has_joints", True)
    if has_joints:
        comm = comm.replace(_ANCHORS["model"], _ANCHORS["model"] + (
        "        _F(GROUP_MA='JOINTS', PHENOMENE='MECANIQUE', MODELISATION='3D_JOINT'),\n"))
        comm = comm.replace(_ANCHORS["chmat"], (
        "JOINT_MAT = DEFI_MATERIAU(\n"
        f"    ELAS=_F(E={joint['concrete_youngs_modulus_pa']:.12g}, NU={joint['concrete_poissons_ratio']:.12g}),\n"
        f"    RUPT_FRAG=_F(GC={joint['fracture_energy_j_m2']:.12g}, SIGM_C={joint['tensile_strength_pa']:.12g},\n"
        f"                 PENA_ADHERENCE={joint['pena_adherence']:.12g}, PENA_CONTACT=1.0),\n"
        ")\n"
        "# Doc R7.02.11 CZM_LIN_REG: linear softening, full separation at 2 GC / SIGM_C\n"
        "# (the FEniCSx critical opening); the elastic branch ends at PENA_ADHERENCE times that\n"
        "# opening, so the initial stiffness is SIGM_C**2 / (2 * GC * PENA_ADHERENCE).\n"
        "# ELAS carries the concrete values; the joint response is RUPT_FRAG.\n\n"
        + _ANCHORS["chmat"]))
        comm = comm.replace(_ANCHORS["concrete_mat"], _ANCHORS["concrete_mat"] + (
        "        _F(GROUP_MA='JOINTS', MATER=JOINT_MAT),\n"))
    assignment = joint.get("material_assignment")
    if has_joints and assignment is not None:
        original = joint["intact_joint"]
        comm = comm.replace(_ANCHORS["chmat"], (
            "INTACT_JOINT_MAT = DEFI_MATERIAU(\n"
            f"    ELAS=_F(E={original['concrete_youngs_modulus_pa']:.12g}, NU={original['concrete_poissons_ratio']:.12g}),\n"
            f"    RUPT_FRAG=_F(GC={original['fracture_energy_j_m2']:.12g}, SIGM_C={original['tensile_strength_pa']:.12g}, "
            f"PENA_ADHERENCE={original['pena_adherence']:.12g}, PENA_CONTACT=1.0),\n"
            ")\n" + _ANCHORS["chmat"]))
        groups = ""
        if joint["damaged_joint_count"]:
            groups += "        _F(GROUP_MA='DAMAGED_JOINTS', MATER=JOINT_MAT),\n"
        if joint["damaged_joint_count"] < joint["candidate_face_count"]:
            groups += "        _F(GROUP_MA='INTACT_JOINTS', MATER=INTACT_JOINT_MAT),\n"
        comm = comm.replace("        _F(GROUP_MA='JOINTS', MATER=JOINT_MAT),\n", groups)
    opening_threshold = repr(joint.get("elastic_limit_opening_m", 0.))
    if assignment is not None:
        thresholds = np.where(interpolation["damaged_joint_faces"][interpolation["joint_parent_faces"]],
                              joint["elastic_limit_opening_m"], joint["intact_joint"]["elastic_limit_opening_m"])
        opening_threshold = "np.asarray(" + repr(thresholds.tolist()) + ")"
    if assignment is not None and assignment["selected_source_rows"]:
        properties = assignment["properties"]
        comm = comm.replace(_ANCHORS["chmat"], (
            "DAMAGED_MAT = DEFI_MATERIAU(\n"
            f"    ELAS=_F(E={properties['youngs_modulus']:.12g}, NU={properties['poisson_ratio']:.12g}, "
            f"RHO={properties['density']:.12g}),\n"
            ")\n" + _ANCHORS["chmat"]))
        comm = comm.replace(_ANCHORS["concrete_mat"], _ANCHORS["concrete_mat"] +
                            "        _F(GROUP_MA='DAMAGED_ELEMENTS', MATER=DAMAGED_MAT),\n")
    targets = ", ".join(f"{value:.10g}" for value in factors)
    tie_load = ""
    tie_excit = ""
    if tie_pairs:
        tie_load = (
            f"TIE_PAIRS = {int(tie_pairs)}\n"
            "TIE = AFFE_CHAR_MECA(MODELE=MODELE, LIAISON_DDL=[\n"
            "    _F(GROUP_NO=(f'TIEA{index:05d}', f'TIEB{index:05d}'), DDL=(dof, dof), COEF_MULT=(1.0, -1.0),\n"
            "       COEF_IMPO=0.0)\n"
            "    for index in range(TIE_PAIRS) for dof in ('DX', 'DY', 'DZ')])\n"
            "# Ties each joint-only crack-front node copy (the crack front stays closed) and each\n"
            "# neighbourhood-relief node copy to its node: same displacement, exactly.\n\n")
        tie_excit = "    _F(CHARGE=TIE),\n"
    comm += tie_load
    if "SUPPORT_TRACE = AFFE_CHAR_MECA" in comm:
        tie_excit += "    _F(CHARGE=SUPPORT_TRACE),\n"
    if interpolation is not None and len(interpolation["interpolation_nodes"]):
        parents = interpolation["interpolation_parents"].tolist()
        weights = interpolation["interpolation_weights"].tolist()
        nodes = interpolation["interpolation_nodes"].tolist()
        comm += (
            f"INTERPOLATION_NODES = {nodes!r}\nINTERPOLATION_PARENTS = {parents!r}\n"
            f"INTERPOLATION_WEIGHTS = {weights!r}\n"
            "INTERPOLATION = AFFE_CHAR_MECA(MODELE=MODELE, LIAISON_DDL=[\n"
            "    _F(NOEUD=(f'N{child + 1}', *(f'N{node + 1}' for node, weight in zip(parents, weights) if weight)),\n"
            "       DDL=(dof,) * (1 + sum(weight != 0 for weight in weights)),\n"
            "       COEF_MULT=(1.0, *(-weight for weight in weights if weight)), COEF_IMPO=0.0)\n"
            "    for child, parents, weights in zip(INTERPOLATION_NODES, INTERPOLATION_PARENTS,\n"
            "                                      INTERPOLATION_WEIGHTS)\n"
            "    for dof in ('DX', 'DY', 'DZ')])\n\n")
        tie_excit += "    _F(CHARGE=INTERPOLATION),\n"
    comm += f'''RAMP = DEFI_FONCTION(NOM_PARA='INST', VALE=(0.0, 0.0, 100.0, 100.0), PROL_DROITE='LINEAIRE')
# Every load is scaled by the load factor (= INST), as in the FEniCSx proportional ramp.

STEPS_PATH = {_comm_string(steps_path)}
STATUS_PATH = {_comm_string(status_path)}
TARGETS = [{targets}]
JOINT_CELLS = list(MAIL.getCells('JOINTS')) if {has_joints!r} else []


def joint_component(field, component):
    if not JOINT_CELLS:
        return np.empty(0)
    values, description = field.toSimpleFieldOnCells().getValuesWithDescription(
        [component], ['JOINTS'])
    values = np.asarray(values, dtype=float)
    cells = np.asarray(description[0], dtype=np.int64)
    index = {{cell: row for row, cell in enumerate(JOINT_CELLS)}}
    peak = np.full(len(JOINT_CELLS), -np.inf)
    np.maximum.at(peak, np.array([index[cell] for cell in cells.tolist()], dtype=np.int64), values)
    return peak


def nodal_displacement(field):
    result = np.full((MAIL.getNumberOfNodes(), 3), np.nan)
    for column, component in enumerate(('DX', 'DY', 'DZ')):
        values, description = field.getValuesWithDescription([component])
        result[np.asarray(description[0], dtype=np.int64), column] = np.asarray(values, dtype=float)
    return result


history = {{"load_factor": [], "damage_state": [], "normal_jump_m": [], "normal_traction_pa": [],
           "dissipated_energy_j_m2": []}}
status = {{"schema_version": 1, "status": "running", "targets": TARGETS, "converged_load_factors": [],
          "joint_cells": len(JOINT_CELLS), "started_at": time.time()}}


def save(final=None):
    if final is not None:
        status.update(final)
    np.savez(STEPS_PATH, **{{key: np.asarray(value, dtype=float) for key, value in history.items()}})
    with open(STATUS_PATH, 'w', encoding='utf-8') as stream:
        json.dump(status, stream, indent=2)


save()
EXCIT = (
    _F(CHARGE=GRAVITE, FONC_MULT=RAMP),
    _F(CHARGE=PRESSIONS, FONC_MULT=RAMP),
    _F(CHARGE=CREST_SURCHARGE, FONC_MULT=RAMP),
{tie_excit})
COMPORT = (
    _F(RELATION='ELAS', TOUT='OUI'),
{ "    _F(RELATION='CZM_LIN_REG', GROUP_MA='JOINTS')," if has_joints else "" }
)
RESU = None
reached = 0.0
for target in TARGETS:
    print(f"ASTER_COHESIVE_STEP start load_factor={{target:.6g}}", flush=True)
    times = DEFI_LIST_INST(
        DEFI_LIST=_F(LIST_INST=DEFI_LIST_REEL(DEBUT=reached, INTERVALLE=_F(JUSQU_A=target, NOMBRE=1))),
        ECHEC=_F(EVENEMENT='ERREUR', ACTION='DECOUPE', SUBD_METHODE='MANUEL', SUBD_PAS=4, SUBD_NIVEAU=4),
    )
    options = dict(
        MODELE=MODELE, CHAM_MATER=CHMAT, CARA_ELEM=CARA_ROCK, EXCIT=EXCIT, COMPORTEMENT=COMPORT,
        INCREMENT=_F(LIST_INST=times), NEWTON=_F(MATRICE='TANGENTE', REAC_ITER=1),
        CONVERGENCE=_F(ITER_GLOB_MAXI=30, RESI_GLOB_RELA=1.0e-6),
        SOLVEUR=_F(METHODE='MULT_FRONT'),
    )
    try:
        if RESU is None:
            RESU = STAT_NON_LINE(**options)
        else:
            RESU = STAT_NON_LINE(reuse=RESU, ETAT_INIT=_F(EVOL_NOLI=RESU), **options)
    except CA.AsterError as error:
        detail = str(error).strip() or type(error).__name__
        save({{"status": "stopped", "stopped_at_load_factor": target,
               "message": f"Code_Aster did not converge from load factor {{reached:.6g}} to {{target:.6g}}: {{detail}}"}})
        print(f"ASTER_COHESIVE_STEP failed load_factor={{target:.6g}}", flush=True)
        break
    reached = target
    last = RESU.getLastIndex()
    vari = RESU.getField('VARI_ELGA', last) if JOINT_CELLS else None
    sief = RESU.getField('SIEF_ELGA', last) if JOINT_CELLS else None
    history["load_factor"].append(target)
    history["damage_state"].append(joint_component(vari, 'V3'))
    history["normal_jump_m"].append(joint_component(vari, 'V7'))
    history["dissipated_energy_j_m2"].append(joint_component(vari, 'V5'))
    history["normal_traction_pa"].append(joint_component(sief, 'SIGN'))
    if len(history["load_factor"]) == 1:
        first_depl = nodal_displacement(RESU.getField('DEPL', last))
    status["converged_load_factors"].append(target)
    save()
    print(f"ASTER_COHESIVE_STEP done load_factor={{target:.6g}} damaged_joints="
          f"{{int(np.sum(history['damage_state'][-1] >= 1.0))}}", flush=True)
    if {joint.get("stop_overload_on_opening", False)!r} and target >= 1.0 and np.any(
            (history["damage_state"][-1] >= 1.0) &
            (history["normal_jump_m"][-1] > {opening_threshold})):
        save({{"overload_stopped_on_opening": True}})
        break

if history["load_factor"]:
    final_depl = nodal_displacement(RESU.getField('DEPL', RESU.getLastIndex()))
    np.savez(STEPS_PATH.replace('.npz', '_displacement.npz'), first=first_depl, final=final_depl,
             coordinates=np.asarray(MAIL.getCoordinates().getValues(), dtype=float).reshape(-1, 3))
if status["status"] == "running":
    save({{"status": "completed"}})
elif not history["load_factor"]:
    save({{"status": "failed"}})
status["finished_at"] = time.time()
save()

FIN()
'''
    return comm


# ---------------------------------------------------------------- preparation and run

def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(baseline_dir: Path, fenicsx_output: Path, output_dir: Path) -> dict:
    """Write the split mesh arrays, mesh manifest and cohesive case into ``output_dir``."""
    from Aster.generate_three_body_case import render_comm

    summary = json.loads((fenicsx_output / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") == "prepared_aster_candidates":
        manifest = json.loads((fenicsx_output / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("status") != "prepared_aster_candidates"
                or manifest.get("summary_sha256") != file_hash(fenicsx_output / "summary.json")
                or manifest.get("candidate_state_sha256") != file_hash(fenicsx_output / "cohesive_state.npz")
                or any(file_hash(baseline_dir / name) != digest
                       for name, digest in summary["baseline_sha256"].items())):
            raise CorroborationError("Aster candidate preparation does not match the current accepted baseline")
    with np.load(fenicsx_output / "cohesive_state.npz") as state:
        faces = candidate_faces(state)
        fenicsx_nodes = len(state["coordinates_m"])
        parent_nodes = np.asarray(state["parent_node_ids"], dtype=np.int64)
        fenicsx_coordinates = np.asarray(state["coordinates_m"], dtype=float)
    with np.load(baseline_dir / SPLIT_ARRAYS_NAME, allow_pickle=True) as stored:
        arrays = {key: stored[key] for key in stored.files}
    if (parent_nodes.shape != (fenicsx_nodes,) or parent_nodes.min() < 0
            or parent_nodes.max() >= len(arrays["nodes"])
            or not np.allclose(arrays["nodes"][parent_nodes], fenicsx_coordinates, rtol=0., atol=1e-8)):
        raise CorroborationError("Aster source node identities/coordinates differ from the candidate mesh")
    split = split_for_joints(arrays, faces)
    ties = len(split["arrays"]["tie_node_pairs"])
    support_aux = len(arrays.get("support_interpolation_nodes", []))
    if len(split["arrays"]["nodes"]) - ties - support_aux != fenicsx_nodes:
        raise CorroborationError(
            f"The Aster split mesh has {len(split['arrays']['nodes']) - ties - support_aux} nodes but the candidate mesh has "
            f"{fenicsx_nodes}; the two crack topologies differ")
    relieve_neighbourhoods(split)
    ties = len(split["arrays"]["tie_node_pairs"])
    integrate_joint_faces(split)
    assignment = summary.get("material_assignment")
    if assignment is not None:
        selected_rows = np.asarray(assignment["selected_source_rows"], dtype=np.int64)
        volume_rows = split["arrays"]["source_cell_rows"]
        selected = np.isin(volume_rows, selected_rows)
        concrete_count = len(split["arrays"]["tetra4_concrete"])
        if (len(np.unique(selected_rows)) != len(selected_rows) or selected.sum() != len(selected_rows)
                or selected[concrete_count:].any()
                or np.any(split["arrays"]["concrete_body_ids"][selected[:concrete_count]] != 1)):
            raise CorroborationError("Damaged material assignment must match exact selected wall source cells")
        split["arrays"]["damaged_cells"] = selected
        selected_faces = {tuple(sorted(face)) for cell in arrays["tetra4_concrete"][selected[:concrete_count]]
                          for face in (cell[[0, 1, 2]], cell[[0, 1, 3]], cell[[0, 2, 3]], cell[[1, 2, 3]])}
        split["arrays"]["damaged_joint_faces"] = np.array(
            [tuple(sorted(face)) in selected_faces for face in faces], dtype=bool)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / SPLIT_ARRAYS_NAME, penta6_joints=split["joints"], **split["arrays"])
    baseline_manifest = json.loads((baseline_dir / MESH_MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest = split_manifest(baseline_manifest, split)
    manifest_path = output_dir / MESH_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    base_comm, base_inputs = render_comm(mesh_manifest_path=manifest_path)
    concrete = base_inputs["material"]["concrete"]
    joint = joint_parameters(arrays, faces, summary, concrete["youngs_modulus_pa"], concrete["poissons_ratio"])
    joint["has_joints"] = bool(len(faces))
    joint["stop_overload_on_opening"] = summary.get("status") == "prepared_aster_candidates"
    joint["material_assignment"] = assignment
    if assignment is not None:
        config = json.loads(Path(baseline_manifest["source_config"]).read_text(encoding="utf-8"))
        original = config["material"]["concrete"]
        joint["intact_joint"] = joint_parameters(
            arrays, faces, {"tensile_strength_pa": original["tensile_strength_pa"],
                            "fracture_energy_j_m2": original["cohesive_fracture_energy_j_m2"]},
            concrete["youngs_modulus_pa"], concrete["poissons_ratio"])
        joint["damaged_joint_count"] = int(split["arrays"]["damaged_joint_faces"].sum())
        joint["candidate_face_count"] = len(faces)
    factors = load_factors(summary)
    run_dir = output_dir / "run"
    run_dir.mkdir(exist_ok=True)
    comm = render_cohesive_comm(base_comm, joint, factors, (run_dir / STEPS_NAME).resolve(),
                                (run_dir / STATUS_NAME).resolve(), tie_pairs=ties, interpolation=split["arrays"])
    (output_dir / CASE_NAME).write_text(comm, encoding="utf-8", newline="\n")
    inputs = {
        "schema_version": SCHEMA_VERSION,
        "joint": joint,
        "load_factors": factors,
        "candidate_faces": int(len(faces)),
        "split_nodes": split["split_nodes"],
        "added_nodes": split["added_nodes"],
        "tip_ties": split["tip_ties"],
        "neighbourhood_relief_copies": split["relief_copies"],
        "joint_integration": "three_HEXA8_FPG4_patches_exact_P1_trace",
        "native_joint_cells": len(split["arrays"]["hexa8_joints"]),
        "interpolation_nodes": len(split["arrays"]["interpolation_nodes"]),
        "tangent_update_interval": 1,
        "shared_interface_springs": len(split["spring_shares"]),
        "fenicsx_summary_sha256": file_hash(fenicsx_output / "summary.json"),
        "fenicsx_state_sha256": file_hash(fenicsx_output / "cohesive_state.npz"),
        "baseline_arrays_sha256": file_hash(baseline_dir / SPLIT_ARRAYS_NAME),
        "case_sha256": file_hash(output_dir / CASE_NAME),
        "baseline_manifest_sha256": file_hash(baseline_dir / MESH_MANIFEST_NAME),
        "source_configuration_sha256": file_hash(Path(baseline_manifest["source_config"])),
    }
    if assignment is not None:
        baseline_fields = baseline_dir / "aster_full_displacement.npz"
        inputs.update(material_assignment=assignment, baseline_displacement_path=str(baseline_fields.resolve()),
                      baseline_displacement_sha256=file_hash(baseline_fields),
                      intact_concrete_properties=concrete)
    (output_dir / "case_inputs.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    return inputs


def solve(output_dir: Path, time_limit_s: int | None, memory_limit_mb: int = MEMORY_LIMIT_MB,
          runtime: str = "wsl") -> dict:
    """Run the prepared case with the local Code_Aster; returns the case's own status."""
    from Aster.run_case import build_med_mesh, detect_installation, MESH_UNIT, run_solver, run_wsl_solver

    run_dir = output_dir / "run"
    run_dir.mkdir(exist_ok=True)
    if runtime == "wsl" and os.name == "nt":
        from Aster.whole_model_crack import (
            DEFAULT_WSL_DISTRIBUTION, DISTRIBUTION_ENVIRONMENT,
            find_linux_installation, wsl_path,
        )
        distribution = os.environ.get(DISTRIBUTION_ENVIRONMENT, DEFAULT_WSL_DISTRIBUTION)
        root = find_linux_installation(distribution)
        med_path = output_dir / "mesh.med"
        med_path.unlink(missing_ok=True)
        converter = ROOT / "Aster" / "convert_three_body_to_med.py"
        converted = run_wsl_solver(distribution, root, output_dir, [
            f"{root}/bin/python3", wsl_path(converter), wsl_path(output_dir / SPLIT_ARRAYS_NAME),
            wsl_path(med_path)], min(time_limit_s, 120) if time_limit_s is not None else 120, run_dir / "conversion.log")
        if converted.returncode in (124, 137):
            raise subprocess.TimeoutExpired(converted.args, min(time_limit_s, 120) if time_limit_s is not None else 120)
        if converted.returncode != 0 or not med_path.is_file():
            raise CorroborationError(f"WSL Aster mesh conversion failed: {converted.stdout}\n{converted.stderr}")
        case = output_dir / CASE_NAME
        text = case.read_text(encoding="utf-8")
        for name in (STEPS_NAME, STATUS_NAME):
            path = (run_dir / name).resolve()
            text = text.replace(_comm_string(path), repr(wsl_path(path)))
        text = text.replace("SOLVEUR=_F(METHODE='MULT_FRONT')", "SOLVEUR=_F(METHODE='MUMPS')")
        case.write_text(text, encoding="utf-8")
        case_path, mesh_path = wsl_path(case), wsl_path(med_path)
    else:
        installation = detect_installation()
        if not installation["available"]:
            raise CorroborationError("Code_Aster is not installed (no run_aster launcher was found)")
        med_path = build_med_mesh(installation["launcher_path"], output_dir, SPLIT_ARRAYS_NAME,
                                 "convert_three_body_to_med.py")
        if med_path is None:
            raise CorroborationError("Code_Aster's bundled Python was not found; the mesh could not be converted")
        case_path, mesh_path = str((output_dir / CASE_NAME).resolve()), str(med_path.resolve())
    for name in (STEPS_NAME, STATUS_NAME, STEPS_NAME.replace(".npz", "_displacement.npz")):
        (run_dir / name).unlink(missing_ok=True)
    export_path = run_dir / "aster_cohesive.export"
    export_path.write_text("\n".join([
        *([f"P time_limit {int(time_limit_s)}"] if time_limit_s is not None else []),
        f"P memory_limit {int(memory_limit_mb)}",
        "P mpi_nbcpu 1",
        f"F comm {case_path} D 1",
        f"F libr {mesh_path} D {MESH_UNIT}",
        "",
    ]), encoding="utf-8")
    if runtime == "wsl" and os.name == "nt":
        completed = run_wsl_solver(distribution, root, run_dir,
                                   [f"{root}/bin/run_aster", wsl_path(export_path)],
                                   time_limit_s, run_dir / "solver.live.log")
    else:
        completed = run_solver([installation["launcher_path"], str(export_path.resolve())],
                               run_dir, time_limit_s, live_log=run_dir / "solver.live.log")
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    try:
        status = json.loads((run_dir / STATUS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        tail = (completed.stdout + completed.stderr)[-1500:]
        raise CorroborationError(f"Code_Aster exited with code {completed.returncode} before the cohesive "
                                 f"solve started:\n{tail}") from None
    status["returncode"] = completed.returncode
    status["runtime"] = "wsl" if runtime == "wsl" and os.name == "nt" else "native"
    if (completed.returncode != 0 or "<EXCEPTION>" in completed.stdout + completed.stderr
            or status.get("status") == "running"):
        status["status"] = "failed"
        status["message"] = f"Code_Aster stopped unexpectedly (exit code {completed.returncode})"
    return status


# ---------------------------------------------------------------- post-processing

def wall_cell_stress(points: np.ndarray, tets: np.ndarray, displacement: np.ndarray, e_pa: float,
                     nu: float) -> np.ndarray:
    """Constant P1 tetrahedron stress tensors (Pa) from nodal displacement."""
    x = points[tets]
    u = displacement[tets]
    edges = np.transpose(x[:, 1:] - x[:, :1], (0, 2, 1))
    jumps = np.transpose(u[:, 1:] - u[:, :1], (0, 2, 1))
    gradient = jumps @ np.linalg.inv(edges)
    strain = 0.5 * (gradient + np.transpose(gradient, (0, 2, 1)))
    lam = e_pa * nu / ((1 + nu) * (1 - 2 * nu))
    mu = e_pa / (2 * (1 + nu))
    trace = np.trace(strain, axis1=1, axis2=2)
    return lam * trace[:, None, None] * np.eye(3) + 2 * mu * strain


def _stress_summary(points, tets, stress, displacement) -> dict:
    principal = np.linalg.eigvalsh(stress)
    tensile, compressive = int(np.argmax(principal[:, 2])), int(np.argmin(principal[:, 0]))
    wall_nodes = np.unique(tets)
    return {
        "maximum_principal_tensile_stress_pa": float(principal[tensile, 2]),
        "maximum_principal_compressive_stress_pa": float(principal[compressive, 0]),
        "tensile_peak_centroid_xyz_m": points[tets[tensile]].mean(axis=0).tolist(),
        "compressive_peak_centroid_xyz_m": points[tets[compressive]].mean(axis=0).tolist(),
        "maximum_displacement_m": float(np.linalg.norm(displacement[wall_nodes], axis=1).max()),
        "maximum_wall_displacement_m": float(np.linalg.norm(displacement[wall_nodes], axis=1).max()),
    }


def _jaccard(a: np.ndarray, b: np.ndarray) -> float | None:
    union = int(np.sum(a | b))
    return None if union == 0 else float(np.sum(a & b)) / union


def _ratio(a: float, b: float) -> float | None:
    return None if b == 0 else float(a) / float(b)


def postprocess(output_dir: Path, fenicsx_output: Path, status: dict) -> dict:
    inputs = json.loads((output_dir / "case_inputs.json").read_text(encoding="utf-8"))
    joint = inputs["joint"]
    summary = json.loads((fenicsx_output / "summary.json").read_text(encoding="utf-8"))
    preparation_only = summary.get("status") == "prepared_aster_candidates"
    with np.load(fenicsx_output / "cohesive_state.npz") as state:
        areas = np.asarray(state["areas_m2"], dtype=float)
        if not preparation_only:
            f_initiated = np.asarray(state["initiated"], dtype=bool)
            f_opened = np.isin(np.asarray(state["face_status"]), ("softening", "separated"))
            f_factor = float(state["load_factor"])
    run_dir = output_dir / "run"
    steps_path = run_dir / STEPS_NAME
    report = {
        "schema_version": SCHEMA_VERSION,
        "solver": "Code_Aster STAT_NON_LINE",
        "engineering_validated": False,
        "joint": joint,
        "requested_load_factors": inputs["load_factors"],
        "candidate_faces": inputs["candidate_faces"],
        "split_nodes": inputs["split_nodes"],
        "added_nodes": inputs["added_nodes"],
        "solve_status": status.get("status"),
        "returncode": status.get("returncode"),
        "solve_message": status.get("message"),
        "differences_from_fenicsx": list(DIFFERENCES),
        "joint_integration": inputs.get("joint_integration", "PENTA6_FPG1"),
        "native_joint_cells": inputs.get("native_joint_cells", inputs["candidate_faces"]),
        "tangent_update_interval": inputs.get("tangent_update_interval", 1),
    }
    if not steps_path.is_file():
        report.update(status="failed", steps=[])
        return report
    with np.load(steps_path) as steps:
        factors = np.asarray(steps["load_factor"], dtype=float)
        if len(factors) == 0:
            report.update(status="failed", steps=[])
            return report
        native_faces = report["native_joint_cells"]
        damage = np.asarray(steps["damage_state"], dtype=float).reshape(len(factors), native_faces)
        jump = np.asarray(steps["normal_jump_m"], dtype=float).reshape(len(factors), native_faces)
        traction = np.asarray(steps["normal_traction_pa"], dtype=float).reshape(len(factors), native_faces)
    if (jump.shape != damage.shape or traction.shape != damage.shape
            or not all(np.isfinite(values).all() for values in (factors, damage, jump, traction))
            or np.any(np.diff(factors) <= 0)):
        raise CorroborationError("Aster converged step fields are non-finite, unordered or inconsistent")
    if inputs.get("joint_integration") == "three_HEXA8_FPG4_patches_exact_P1_trace":
        with np.load(output_dir / SPLIT_ARRAYS_NAME, allow_pickle=False) as stored:
            mapping = np.asarray(stored["joint_parent_faces"], dtype=np.int64)
        if (mapping.shape != (damage.shape[1],) or not np.array_equal(
                mapping, np.repeat(np.arange(len(areas)), 3))):
            raise CorroborationError("Aster quadrilateral patches do not match the original candidate facets")
        # Report peaks over each original candidate facet, not invented extra crack faces.
        damage, jump, traction = (values.reshape(len(factors), len(areas), 3).max(axis=2)
                                  for values in (damage, jump, traction))
        report["facet_state_definition"] = (
            "Peak over three FPG4 quadrilateral patches per original facet; initiated/opened areas "
            "are the areas of candidate facets touched, not fully separated surface areas.")
    if damage.shape[1] != len(areas):
        raise CorroborationError("Aster joint results do not match the FEniCSx candidate facets")
    initiated = damage >= 1.0 - 1e-9
    opening_limit = joint["elastic_limit_opening_m"]
    if inputs.get("material_assignment") is not None:
        with np.load(output_dir / SPLIT_ARRAYS_NAME) as arrays:
            opening_limit = np.where(arrays["damaged_joint_faces"], opening_limit,
                                     joint["intact_joint"]["elastic_limit_opening_m"])
        report["material_assignment"] = inputs["material_assignment"]
    opened = initiated & (jump > opening_limit * (1.0 + 1e-6))
    separated = damage >= 2.0 - 1e-9
    step_rows = [{
        "load_factor": float(factor),
        "initiated_faces": int(initiated[i].sum()),
        "initiated_area_m2": float(areas[initiated[i]].sum()),
        "opened_faces": int(opened[i].sum()),
        "opened_area_m2": float(areas[opened[i]].sum()),
        "separated_faces": int(separated[i].sum()),
        "peak_opening_m": float(max(0.0, jump[i].max())) if len(areas) else 0.0,
        "peak_joint_normal_traction_pa": float(traction[i].max()) if len(areas) else 0.0,
    } for i, factor in enumerate(factors)]
    first = next((i for i in range(len(factors)) if initiated[i].any()), None)
    bracket = None if first is None else [float(factors[first - 1]) if first else 0.0, float(factors[first])]
    linear_estimate = None
    if len(areas) and not initiated[0].any() and traction[0].max() > 0:
        strengths = np.full(len(areas), joint["tensile_strength_pa"], dtype=float)
        if inputs.get("material_assignment") is not None:
            with np.load(output_dir / SPLIT_ARRAYS_NAME) as arrays:
                strengths[~arrays["damaged_joint_faces"]] = joint["intact_joint"]["tensile_strength_pa"]
        positive = traction[0] > 0
        linear_estimate = float(np.min(factors[0] * strengths[positive] / traction[0, positive]))

    def state_at(target: float) -> int | None:
        matches = np.flatnonzero(np.isclose(factors, target, atol=1e-9, rtol=0.))
        return int(matches[0]) if len(matches) else None

    design = state_at(1.0)
    final = len(factors) - 1
    complete = status.get("status") == "completed" and (
        np.isclose(factors[-1], inputs["load_factors"][-1], rtol=0., atol=1e-9)
        or (status.get("overload_stopped_on_opening") is True and factors[-1] >= 1.0))
    report.update(
        status="completed" if complete else "partial",
        reached_load_factor=float(factors[-1]),
        steps=step_rows,
        first_initiation_bracket=bracket,
        linear_first_initiation_estimate=linear_estimate,
        design_load=None if design is None else step_rows[design],
    )
    report["opening_assessment"] = {
        "peak_opening_m": step_rows[-1]["peak_opening_m"],
        "initiated_faces": step_rows[-1]["initiated_faces"],
        "opened_faces": step_rows[-1]["opened_faces"],
        "separated_faces": step_rows[-1]["separated_faces"],
        "note": ("CZM_LIN_REG initiation is not full separation. Opened faces have passed the elastic "
                 "opening threshold; separated faces have native damage state 2. Areas count original "
                 "facets touched, not a fully separated surface. Full-separation opening is 2 Gf / ft "
                 "for each assigned cohesive law. Lower peak wall displacement is not evidence of "
                 "greater strength; inspect matched locations, stress redistribution and separation."),
    }

    displacement_path = run_dir / STEPS_NAME.replace(".npz", "_displacement.npz")
    if displacement_path.is_file():
        with np.load(output_dir / SPLIT_ARRAYS_NAME, allow_pickle=True) as arrays:
            nodes = np.asarray(arrays["nodes"], dtype=float)
            wall_mask = np.asarray(arrays["concrete_body_ids"]) == 1
            wall = np.asarray(arrays["tetra4_concrete"])[wall_mask]
            source_ids = arrays["source_node_ids"].copy()
            damaged_wall = (arrays["damaged_cells"][:len(wall_mask)][wall_mask]
                            if "damaged_cells" in arrays else np.zeros(len(wall), dtype=bool))
        with np.load(displacement_path) as fields:
            coordinates, first_u, final_u = fields["coordinates"], fields["first"], fields["final"]
        if (first_u.shape != nodes.shape or final_u.shape != nodes.shape
                or not np.isfinite(first_u).all() or not np.isfinite(final_u).all()):
            raise CorroborationError("Aster displacement fields are missing, non-finite or have incorrect shape")
        if coordinates.shape != nodes.shape or np.abs(coordinates - nodes).max() > 1e-6:
            raise CorroborationError("Aster node order does not match the split mesh arrays")
        e_pa, nu = joint["concrete_youngs_modulus_pa"], joint["concrete_poissons_ratio"]
        final_stress = wall_cell_stress(nodes, wall, final_u, e_pa, nu)
        assignment = inputs.get("material_assignment")
        if assignment is not None:
            properties = assignment["properties"]
            final_stress[damaged_wall] = wall_cell_stress(
                nodes, wall[damaged_wall], final_u, properties["youngs_modulus"], properties["poisson_ratio"])
        cracked = _stress_summary(nodes, wall, final_stress, final_u)
        comparison = {"load_factor": float(factors[-1]), "scope": "wall (body 1) cells, constant P1 stress",
                      "cracked": cracked, "intact": None}
        if assignment is not None:
            path = Path(inputs["baseline_displacement_path"])
            if file_hash(path) != inputs["baseline_displacement_sha256"]:
                raise CorroborationError("Intact baseline displacement changed since material assignment")
            with np.load(path) as baseline:
                ids = baseline["source_node_ids"]
                valid = ids >= 0
                ids, first = np.unique(ids[valid], return_index=True)
                physical = source_ids >= 0
                positions = np.searchsorted(ids, source_ids[physical])
                if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], source_ids[physical]):
                    raise CorroborationError("Selected-material result cannot map to intact baseline nodes")
                intact_u = np.zeros_like(final_u)
                intact_u[physical] = baseline["displacement_m"][valid][first][positions] * factors[-1]
                if not np.allclose(baseline["coordinates_m"][valid][first][positions], nodes[physical], atol=1e-9, rtol=0.):
                    raise CorroborationError("Intact baseline coordinates do not match selected-material result")
            comparison["intact"] = _stress_summary(nodes, wall, wall_cell_stress(nodes, wall, intact_u, e_pa, nu), intact_u)
            comparison["intact_definition"] = "original concrete Aster baseline scaled to final load factor; differences include selected material changes and cracking"
            np.savez(displacement_path, coordinates=coordinates, first=first_u, final=final_u, intact=intact_u)
        elif not initiated[0].any():
            # Before initiation the joints are linear, so the bonded state scales with the load.
            scale = factors[-1] / factors[0]
            intact_u = first_u * scale
            comparison["intact"] = _stress_summary(nodes, wall, wall_cell_stress(nodes, wall, intact_u, e_pa, nu),
                                                   intact_u)
            comparison["intact_definition"] = (
                f"bonded joints, same mesh and loads: the elastic load-factor {factors[0]:g} state scaled linearly")
        report["stress_comparison"] = comparison
        report.update(cracked)
        report["peak_scope"] = "wall"
        report["mesh_nodes"] = len(nodes)
        report["mesh_cells"] = len(wall)
        report["configured_tensile_strength_pa"] = joint["tensile_strength_pa"]
        report["crack_surface_measure_m2"] = step_rows[-1]["opened_area_m2"]
        report["stress_recovery"] = {
            "peak_measure": "raw_cellwise",
            "tensile_peak_centroid_xyz_m": cracked["tensile_peak_centroid_xyz_m"],
            "compressive_peak_centroid_xyz_m": cracked["compressive_peak_centroid_xyz_m"],
        }
        from Aster.export_paraview import export_mesh
        # Preserve all bodies in the view, while assessments use only wall nodes.
        with np.load(output_dir / SPLIT_ARRAYS_NAME, allow_pickle=True) as stored:
            export_arrays = {key: stored[key] for key in stored.files}
        paraview = export_mesh(export_arrays, final_u, output_dir / "paraview", damage[-1], jump[-1])
        report["paraview"] = paraview

    if preparation_only:
        report["corroboration"] = {
            "verdict": "not_available",
            "message": "Candidates came from accepted Aster displacement; no supporting nonlinear solve was run.",
        }
        return report
    corroboration = {"fenicsx_load_factor": f_factor, "aster_load_factor": float(factors[-1])}
    fenicsx_design = summary.get("design_load") or {}
    corroboration["first_initiation"] = {
        "fenicsx_load_factor": fenicsx_design.get("first_initiation_load_factor"),
        "aster_bracket": bracket,
        "aster_linear_estimate": linear_estimate,
    }
    at_final = state_at(f_factor)
    if at_final is not None:
        a_init, a_open = initiated[at_final], opened[at_final]
        corroboration.update(
            fenicsx_initiated_faces=int(f_initiated.sum()), aster_initiated_faces=int(a_init.sum()),
            fenicsx_initiated_area_m2=float(areas[f_initiated].sum()),
            aster_initiated_area_m2=float(areas[a_init].sum()),
            fenicsx_opened_faces=int(f_opened.sum()), aster_opened_faces=int(a_open.sum()),
            fenicsx_opened_area_m2=float(areas[f_opened].sum()), aster_opened_area_m2=float(areas[a_open].sum()),
            initiated_overlap=_jaccard(f_initiated, a_init),
            opened_overlap=_jaccard(f_opened, a_open),
            fenicsx_peak_opening_m=float((summary.get("network") or {}).get("maximum_face_opening_m") or 0.0),
            aster_peak_opening_m=step_rows[at_final]["peak_opening_m"],
        )
    corroboration["verdict"], corroboration["message"] = _verdict(corroboration, complete)
    wet = float((summary.get("network") or {}).get("wet_area_m2") or 0.0)
    if wet > 0:
        corroboration["message"] += (f" FEniCSx applied reservoir water on {wet:.3g} m2 of open crack; "
                                     "Aster has no crack-face water, so this part is not corroborated.")
    report["corroboration"] = corroboration
    return report


def _verdict(c: dict, complete: bool) -> tuple[str, str]:
    """Agreement category with a one-line explanation (both codes at FEniCSx's final load factor)."""
    if "aster_initiated_faces" not in c:
        return "incomplete", (f"Code_Aster stopped at load factor {c['aster_load_factor']:g} before FEniCSx's "
                              f"{c['fenicsx_load_factor']:g}; only the steps it reached can be compared.")
    f_open, a_open = c["fenicsx_opened_faces"] > 0, c["aster_opened_faces"] > 0
    f_init, a_init = c["fenicsx_initiated_faces"] > 0, c["aster_initiated_faces"] > 0
    counts = (f"FEniCSx {c['fenicsx_initiated_faces']} initiated / {c['fenicsx_opened_faces']} opened; "
              f"Code_Aster {c['aster_initiated_faces']} initiated / {c['aster_opened_faces']} opened")
    if f_open != a_open or f_init != a_init:
        return "not_corroborated", f"The codes disagree on whether the wall cracks: {counts}."
    if not f_init:
        return "corroborated", f"Neither code initiates a crack: {counts}."
    overlap = c["initiated_overlap"] or 0.0
    area_ratio = _ratio(c["aster_opened_area_m2"], c["fenicsx_opened_area_m2"]) if f_open else 1.0
    if overlap >= 0.5 and area_ratio is not None and 0.5 <= area_ratio <= 2.0:
        return "corroborated", f"Both codes crack the same facets: {counts}; initiated overlap {overlap:.0%}."
    return "partially_corroborated", (f"Both codes crack, but the extent differs: {counts}; "
                                      f"initiated overlap {overlap:.0%}.")


def run(baseline_dir: Path, fenicsx_output: Path, output_dir: Path, time_limit_s: int) -> dict:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_NAME).unlink(missing_ok=True)
    print("[1/3] Splitting the Aster mesh along the FEniCSx candidate facets ...", flush=True)
    inputs = prepare(baseline_dir, fenicsx_output, output_dir)
    print(f"      {inputs['candidate_faces']} candidate facets, {inputs['native_joint_cells']} integrated joints, "
          f"{inputs['interpolation_nodes']} interpolation nodes; load factors "
          f"{', '.join(f'{v:g}' for v in inputs['load_factors'])}", flush=True)
    print("[2/3] Solving the cohesive model with Code_Aster (one nonlinear solve per load factor) ...", flush=True)
    status = solve(output_dir, time_limit_s)
    print("[3/3] Comparing with FEniCSx ...", flush=True)
    report = postprocess(output_dir, fenicsx_output, status)
    report["runtime_s"] = time.time() - started
    inputs["case_sha256"] = file_hash(output_dir / CASE_NAME)
    (output_dir / "case_inputs.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    report["input_hashes"] = {key: value for key, value in inputs.items() if key.endswith("_sha256")}
    report["accepted"] = (report.get("status") == "completed" and status.get("returncode") == 0
                          and report.get("maximum_wall_displacement_m") is not None
                          and report.get("paraview") is not None)
    report["analysis_scope"] = "whole_model"
    report["assessment_authority"] = "code_aster"
    report["runtime"] = status.get("runtime")
    (output_dir / REPORT_NAME).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    corroboration = report.get("corroboration") or {}
    print(f"{report['status']}: {corroboration.get('message', report.get('solve_message'))}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-dir", type=Path, required=True,
                        help="the job's Code_Aster elastic baseline directory")
    parser.add_argument("--fenicsx-output", type=Path, required=True,
                        help="the automatic cracking output directory (summary.json, cohesive_state.npz)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-s", type=int, default=30 * 24 * 3600)
    args = parser.parse_args()
    try:
        report = run(args.baseline_dir.resolve(), args.fenicsx_output.resolve(), args.output_dir.resolve(),
                     args.time_limit_s)
    except CorroborationError as error:
        print(f"Code_Aster corroboration failed: {error}", file=sys.stderr, flush=True)
        sys.exit(2)
    if report.get("accepted") is not True:
        sys.exit(2)


if __name__ == "__main__":
    main()
