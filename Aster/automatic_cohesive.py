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
initiates on the larger one-cell facet stress). Historical cases without the
explicit crack_water_pressure setting retain dry cracks. New configurations
can enable reservoir-connected opened-crack pressure without uplift.
Isolated hydraulic campaigns opt in through the
mesh manifest, adding undrained base uplift and reservoir-connected crack water
with a converged active set at each explicit accepted load instant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Elmer.model_config import residual_target_tensile_stress_pa

SCHEMA_VERSION = 1
# run_aster needs a numeric CPU budget even when its external wall-clock timeout is disabled.
UNBOUNDED_CPU_TIME_LIMIT_S = 2_147_483_647
REPORT_NAME = "aster_cohesive_report.json"
STEPS_NAME = "aster_cohesive_steps.npz"
STATUS_NAME = "aster_cohesive_status.json"
SPLIT_ARRAYS_NAME = "three_body_mesh_arrays.npz"
MESH_MANIFEST_NAME = "mesh_manifest.json"
CASE_NAME = "case.comm"
RUN_ASTER_EXIT_MARKER = "EXECUTION_CODE_ASTER_EXIT_"
# Initial joint stiffness = factor * E / h, so the pre-initiation joint compliance
# is a 0.1 % perturbation of the surrounding concrete over one element.
JOINT_PENALTY_FACTOR = 1000.0
# DEFI_MATERIAU / JOINT_MECA_RUPT catalogue bounds on PENA_RUPTURE (Cata/Commands/defi_materiau.py).
LIFT_PENA_RUPTURE_MIN = 1e-12
LIFT_PENA_RUPTURE_MAX = 10.0
# A Mohr-Coulomb joint has no cohesive elastic branch: any opening beyond this counts as open.
FRICTION_OPENING_LIMIT_M = 1e-9
FIRST_LOAD_FACTOR = 0.5
LOAD_INCREMENT = 0.1
# Friction solves climb in fine targets first: every completed STAT_NON_LINE target is
# an archived checkpoint (a later stall still reports the reached capacity instead of
# discarding converged equilibria), and the smaller target gaps bound the DECOUPE
# window, giving a ~10x finer minimum substep through the sliding-onset zone.
FRICTION_FINE_INCREMENT = 0.05
FRICTION_FINE_LIMIT = 0.3
# Code_Aster's budget is per MPI rank. Leave headroom within the 24 GiB WSL
# allocation for Python, native solver workspace and the operating system.
MEMORY_LIMIT_MB = 16384
LOADED_TRI_KEYS = ("tri3_upstream", "tri3_downstream", "tri3_crest", "tri3_rock_bottom", "tri3_rock_sides")
# Code_Aster refuses joint models with any cell above 200 vertex neighbours (VOLUFINI_3);
# the margin covers any difference in how Code_Aster counts them.
NEIGHBOUR_TARGET = 150
# OpenMP threads for the cohesive solve (export ncpus; run_aster ignores the caller's
# OMP_NUM_THREADS). Matches whole_model_crack.DEFAULT_THREADS.
SOLVER_THREADS = 8
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

    support = np.asarray(arrays["support_nodes"], dtype=np.int64)
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

    if len(arrays.get("support_faces", [])):
        support_faces = np.asarray(arrays["support_faces"], dtype=np.int64).copy()
        for row in np.flatnonzero(np.isin(support_faces, list(copies)).any(axis=1)):
            found = owners(int(_codes(support_faces[row:row + 1], base)[0]))
            if len(found) != 1:
                raise CorroborationError(
                    f"Wall-base support face {row} cannot be assigned to one crack-side component")
            support_faces[row] = remap(support_faces[row].tolist(), int(found[0]))
        result["support_faces"] = support_faces
        if len(arrays.get("support_interpolation_parents", [])):
            parents = np.asarray(arrays["support_interpolation_parents"], dtype=np.int64)
            if parents.shape != (3 * len(support_faces), 3):
                raise CorroborationError("Wall-base support interpolation topology is inconsistent")
            result["support_interpolation_parents"] = np.repeat(support_faces, 3, axis=0)

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


def apply_support_degradation(manifest: dict, arrays: dict, degradation: dict) -> dict:
    """Scenario boundary-condition degradation: scale spring stiffnesses.

    ``interface_scale`` multiplies every wall/foundation interface spring
    (foundation contact sliding/debonding scenarios); ``abutment`` scales the
    elastic-layer support matrices whose point springs lie within an angular
    fraction of the seat measured from one end of the arch (abutment support-loss
    scenarios, Design_Docs/failure_scenarios.md). The manifest entries and the
    mesh-array stiffnesses are scaled identically: the MED conversion derives the
    compact K_OUT/K_IF group partition from the arrays while the rendered comm
    derives it from the manifest, and the two must agree. Returns applied counts."""
    unknown = set(degradation) - {"interface_scale", "abutment"}
    if unknown:
        raise CorroborationError(f"Unknown support degradation keys: {sorted(unknown)}")
    applied = dict(degradation)
    interface = degradation.get("interface_scale")
    if interface is not None:
        if isinstance(interface, bool) or not np.isfinite(interface) or not 0 < float(interface) <= 1:
            raise CorroborationError("interface_scale must lie in (0, 1]")
        stiffness = np.asarray(arrays["interface_stiffness_n_per_m"], dtype=float)
        if len(stiffness) != len(manifest["wall_foundation_springs"]):
            raise CorroborationError("Interface springs do not match the mesh arrays")
        arrays["interface_stiffness_n_per_m"] = stiffness * float(interface)
        for entry in manifest["wall_foundation_springs"]:
            entry["stiffness_n_per_m"] = (np.asarray(entry["stiffness_n_per_m"], dtype=float)
                                          * float(interface)).tolist()
        applied["interface_springs_scaled"] = len(manifest["wall_foundation_springs"])
    abutment = degradation.get("abutment")
    if abutment is not None:
        scale, fraction, end = abutment.get("scale"), abutment.get("fraction"), abutment.get("end")
        if (any(isinstance(v, bool) or not np.isfinite(v) for v in (scale, fraction))
                or not 0 < float(scale) <= 1 or not 0 < float(fraction) <= .5
                or end not in ("positive", "negative")):
            raise CorroborationError(
                "Abutment degradation requires scale in (0, 1], fraction in (0, 0.5] and end positive/negative")
        springs = manifest["support_springs"]
        matrices = np.asarray(arrays["support_matrices_n_per_m"], dtype=float)
        points = np.asarray(arrays["nodes"])[np.asarray(arrays["support_nodes"], dtype=np.int64)]
        if len(springs) != len(points) or len(matrices) != len(points):
            raise CorroborationError("Support spring groups do not match the support geometry")
        # The source arch is centred on (0, 0); normalise seat angles to [0, 1]
        # relative to an anchor on the seat so the sector is wraparound-safe.
        theta = np.arctan2(points[:, 1], points[:, 0])
        relative = np.angle(np.exp(1j * (theta - theta[0])))
        span = float(relative.max() - relative.min())
        if span <= 0:
            raise CorroborationError("Support seat has no angular extent")
        position = (relative - relative.min()) / span
        selected = position >= 1 - float(fraction) if end == "positive" else position <= float(fraction)
        if not selected.any():
            raise CorroborationError("Abutment sector selected no support springs")
        matrices[selected] *= float(scale)
        arrays["support_matrices_n_per_m"] = matrices
        for index in np.flatnonzero(selected):
            springs[index]["matrix_n_per_m"] = (np.asarray(springs[index]["matrix_n_per_m"], dtype=float)
                                                * float(scale)).tolist()
        applied["abutment"] = {**abutment, "support_springs_scaled": int(selected.sum())}
    return applied


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


def preload_factor(controls: dict) -> float:
    value = controls.get("load_factor_start", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise CorroborationError("load_factor_start must be a finite number between 0 and 1")
    return float(value)


def load_factors(summary: dict) -> list[float]:
    """Schedule proportional cracking loads, optionally after same-load intact preload.

    Zero start preserves the existing wet/dry first targets and overload behaviour.

    Friction runs instead start at FRICTION_FINE_INCREMENT and climb in fine steps up to
    FRICTION_FINE_LIMIT: the first dam-scale run stalled inside a single 0->0.5 target at
    load factor ~0.23 and lost every converged equilibrium with it."""
    loading = summary.get("reservoir_loading")
    if loading is not None:
        return [depth / loading["final_water_depth_m"] for depth in loading["water_depth_targets_m"]]
    if (summary.get("solver_controls") or {}).get("loading_mode") == "reservoir_filling":
        raise CorroborationError("Reservoir filling requires validated physical depth metadata")
    final = max(1.0, float(summary.get("load_factor") or 1.0),
                float(summary.get("maximum_overload_factor") or 1.0))
    lift = (summary.get("material_assignment") or {}).get("joint_properties") or {}
    friction = summary.get("damaged_joint_friction") is not None or lift.get("law") == "JOINT_MECA_FROT"
    controls = summary.get("solver_controls") or {}
    start = preload_factor(controls)
    increment = controls.get("load_increment", 0.025 if summary.get("crack_water_pressure") is True else LOAD_INCREMENT)
    if isinstance(increment, bool) or not isinstance(increment, (int, float)) or not math.isfinite(increment) or not 0 < increment <= 1:
        raise CorroborationError("load_increment must be finite, positive and no greater than 1")
    if start > 0:
        if final > 1:
            raise CorroborationError("Intact preload supports design loading through factor 1, not overload")
        values = [start]
        while values[-1] < 1:
            following = round(min(1., values[-1] + increment), 10)
            if following <= values[-1]:
                raise CorroborationError("load_increment is too small to advance the load schedule")
            values.append(following)
        return values
    if summary.get("crack_water_pressure") is True:
        configured_increment = float(controls.get("load_increment", 0.025))
        values = [0.01]
        while values[-1] < final - 1e-9:
            following = round(min(final, values[-1] + configured_increment), 10)
            if following <= values[-1]:
                raise CorroborationError("load_increment is too small to advance the load schedule")
            values.append(following)
        return values
    configured_increment = float(controls.get("load_increment", LOAD_INCREMENT))
    factors = [FRICTION_FINE_INCREMENT if friction else FIRST_LOAD_FACTOR]
    while factors[-1] < final - 1e-9:
        increment = (FRICTION_FINE_INCREMENT if friction and factors[-1] < FRICTION_FINE_LIMIT - 1e-9
                     else configured_increment)
        following = round(min(final, factors[-1] + increment), 10)
        if following <= factors[-1]:
            raise CorroborationError("load_increment is too small to advance the load schedule")
        factors.append(following)
    return factors


def joint_parameters(arrays: dict, faces: np.ndarray, summary: dict, concrete_e_pa: float,
                     concrete_nu: float) -> dict:
    f_t = float(summary["tensile_strength_pa"])
    g_f = float(summary["fracture_energy_j_m2"])
    if not all(np.isfinite(value) and value > 0 for value in (f_t, g_f, concrete_e_pa)):
        raise CorroborationError("Cohesive strength, fracture energy and elastic modulus must be finite and positive")
    nodes = np.asarray(arrays["nodes"], dtype=float)
    edge_faces = faces if len(faces) else np.asarray(arrays["tetra4_concrete"])[:, :3]
    edges = np.concatenate([np.linalg.norm(nodes[edge_faces[:, i]] - nodes[edge_faces[:, (i + 1) % 3]], axis=1)
                            for i in range(3)])
    h = float(edges.mean())
    stiffness = JOINT_PENALTY_FACTOR * concrete_e_pa / h
    critical = 2.0 * g_f / f_t
    # CZM_LIN_REG leaves its elastic branch at the jump PENA_ADHERENCE * critical opening
    # (checked on a Code_Aster run), so its initial stiffness is f_t / (PENA_ADHERENCE * critical).
    adherence = float(f_t / (stiffness * critical))
    if not 1e-12 <= adherence < 1.0:
        raise CorroborationError(
            f"CZM_LIN_REG elastic/softening branches require 1e-12 <= PENA_ADHERENCE < 1; "
            f"computed {adherence:.6g}. Material inputs and the mesh-derived joint penalty are incompatible; "
            "refusing to clip the value to a degenerate cohesive law.")
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


def vertical_joint_strength(critical_opening_m: float, fracture_energy_j_m2: float) -> float:
    """Tensile strength giving the target critical opening: w_c = 2 G_F / f_t, so f_t = 2 G_F / w_c."""
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value > 0
               for value in (critical_opening_m, fracture_energy_j_m2)):
        raise CorroborationError("Vertical joint critical opening and fracture energy must be finite and positive")
    return 2.0 * float(fracture_energy_j_m2) / float(critical_opening_m)


def save_split_arrays(output_dir: Path, split: dict) -> None:
    """Write the split mesh arrays; the MED converter reads every group flag from this file."""
    np.savez(output_dir / SPLIT_ARRAYS_NAME, penta6_joints=split["joints"], **split["arrays"])


def vertical_crack_facets(coordinates_m: np.ndarray, faces: np.ndarray, peak_xyz_m, band_m: float,
                          max_normal_deviation_deg: float) -> np.ndarray:
    """Boolean mask over ``faces`` on the radial vertical plane through the tensile peak.

    The arch is centred on the origin, so the plane contains the z-axis and the peak's radial
    direction and passes through the origin. Its normal is the horizontal tangent at the peak.
    A facet qualifies when it is near-vertical, its normal lies along that tangent, and its
    centroid is within ``band_m`` of the plane.
    """
    from FEniCSx.lift_joints import facet_unit_normals
    if not 0 < float(max_normal_deviation_deg) < 90:
        raise CorroborationError("Vertical joint max_normal_deviation_deg must lie in (0, 90)")
    if not np.isfinite(band_m) or band_m <= 0:
        raise CorroborationError("Vertical joint band half-width must be finite and positive")
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(faces):
        return np.zeros(0, dtype=bool)
    coordinates = np.asarray(coordinates_m, dtype=float)
    peak = np.asarray(peak_xyz_m, dtype=float)
    radius = float(np.hypot(peak[0], peak[1]))
    if radius <= 0:
        raise CorroborationError("The tensile peak lies on the arch axis, so the radial plane is undefined")
    tangent = np.array([-peak[1], peak[0], 0.0]) / radius
    normals = facet_unit_normals(coordinates, faces)
    angle = math.radians(float(max_normal_deviation_deg))
    vertical = np.abs(normals[:, 2]) <= math.sin(angle) + 1e-12
    along = np.abs(normals @ tangent) >= math.cos(angle) - 1e-12
    centroids = coordinates[faces].mean(axis=1)
    within = np.abs((centroids - peak) @ tangent) <= float(band_m)
    return vertical & along & within


def vertical_joint_settings(config: dict) -> dict | None:
    """Return the enabled ``damage_selection.vertical_joint`` block of the source configuration, or None."""
    selection = config.get("damage_selection")
    block = selection.get("vertical_joint") if isinstance(selection, dict) else None
    if not block or not block.get("enabled", True):
        return None
    band = block.get("band_m")
    return {
        "critical_opening_m": float(block["critical_opening_m"]),
        "fracture_energy_j_m2": float(block["fracture_energy_j_m2"]),
        "max_normal_deviation_deg": float(block.get("max_normal_deviation_deg", 15.0)),
        "band_m": None if band is None else float(band),
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


def friction_joint_parameters(friction: dict, stiffness_pa_per_m: float) -> dict:
    """Mohr-Coulomb (JOINT_MECA_FROT) parameters for pre-cracked damaged joints.

    The damaged joints keep the mesh-derived penalty stiffness as K_N/K_T; the
    friction coefficient and adhesion come from the configuration. A friction
    joint carries compression and Coulomb shear but no tension or cohesive
    softening — the correct model for an existing crack, which CZM_LIN_REG with a
    near-zero strength cannot represent (it loses all shear transfer and leaves
    the block above a full-section cut as a mechanism)."""
    unknown = set(friction) - {"friction_coefficient", "adhesion_pa"}
    if unknown:
        raise CorroborationError(f"Unknown damaged joint friction keys: {sorted(unknown)}")
    mu = friction.get("friction_coefficient")
    adhesion = friction.get("adhesion_pa", 0.)
    if (isinstance(mu, bool) or not isinstance(mu, (int, float)) or not np.isfinite(mu)
            or not 1e-3 <= float(mu) <= 10.):
        raise CorroborationError("Damaged joint friction coefficient must lie in [0.001, 10]")
    if (isinstance(adhesion, bool) or not isinstance(adhesion, (int, float))
            or not np.isfinite(adhesion) or adhesion < 0):
        raise CorroborationError("Damaged joint adhesion must be finite and nonnegative")
    return {"law": "JOINT_MECA_FROT", "friction_coefficient": float(mu), "adhesion_pa": float(adhesion),
            "k_n_pa_per_m": float(stiffness_pa_per_m),
            # The bond-mimicking tangential penalty (1000 E/h) makes frictional
            # stick-slip Newton iterations chatter; E/h regularises stick with
            # micrometre-scale pre-slip elasticity and converges. The normal penalty
            # must stay stiff: softening it leaves near-singular tangential pivots
            # wherever the contact pressure is small.
            "k_t_pa_per_m": float(stiffness_pa_per_m) / JOINT_PENALTY_FACTOR}


def lift_joint_parameters(joint_properties: dict, stiffness_pa_per_m: float,
                          neighbour_stiffness_pa_per_m: float | None = None) -> dict:
    """Aster parameters for weak construction (lift) joints (R7.01.25).

    ``stiffness_pa_per_m`` is the shared mesh penalty (1000 E/h) and
    ``neighbour_stiffness_pa_per_m`` the stiffness E/h of the volume elements beside the
    joint (default: the penalty / 1000). U2.05.07 4.1.2: "the initial stiffness of the joint
    elements must be greater than the stiffness of the neighbouring volume elements"; a softer
    joint is an elastic layer that sheds its load before it can reach its strength.

    JOINT_MECA_FROT (Mohr-Coulomb friction with contact): K_N is the mesh penalty, K_T the
    regularised E/h used for the damaged friction joints, MU the configured coefficient and
    ADHESION = MU * f_t, so the cone's apex R_t = c / MU is the configured tensile strength.
    The law is perfectly plastic: it does not lose that tensile capacity after opening
    (R7.01.25 1.2.2), so a poorly bonded surface should be given a small strength. It is the
    law that lets the plane slide; the fracture energy is not used.

    JOINT_MECA_RUPT (cohesive rupture with contact, no slip limit): linear elastic to
    SIGM_MAX, then linear softening with slope -K_N / PENA_RUPTURE, full separation at
    SIGM_MAX (1 + PENA_RUPTURE) / K_N, so Gf = SIGM_MAX^2 (1 + PENA_RUPTURE) / (2 K_N)
    (ssnp162b). The catalogue caps PENA_RUPTURE at 10, so K_N follows from the strength and
    energy at that cap, K_N = SIGM_MAX^2 (1 + 10) / (2 Gf); when that would exceed the mesh
    penalty the penalty is kept and PENA_RUPTURE solved instead. A fracture energy so large
    that K_N would fall below the neighbouring elements' stiffness is refused: this law
    cannot dissipate it without becoming a soft layer."""
    f_t = joint_properties.get("tensile_strength_pa")
    g_f = joint_properties.get("cohesive_fracture_energy_j_m2")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value > 0
               for value in (f_t, g_f, stiffness_pa_per_m)):
        raise CorroborationError("Lift joint strength, fracture energy and joint stiffness must be finite and positive")
    alpha = joint_properties.get("alpha", 1.0)
    contact = joint_properties.get("pena_contact", 1.0)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 <= float(alpha) <= 2:
        raise CorroborationError("Lift joint ALPHA must lie in [0, 2]")
    if isinstance(contact, bool) or not isinstance(contact, (int, float)) or not float(contact) > 0:
        raise CorroborationError("Lift joint PENA_CONTACT must be positive")
    law = joint_properties.get("law", "JOINT_MECA_RUPT")
    f_t, g_f, penalty = float(f_t), float(g_f), float(stiffness_pa_per_m)
    neighbour = (penalty / JOINT_PENALTY_FACTOR if neighbour_stiffness_pa_per_m is None
                 else float(neighbour_stiffness_pa_per_m))
    if not np.isfinite(neighbour) or neighbour <= 0:
        raise CorroborationError("Neighbouring element stiffness must be finite and positive")
    if law == "JOINT_MECA_FROT":
        mu = joint_properties.get("friction_coefficient")
        if (isinstance(mu, bool) or not isinstance(mu, (int, float)) or not np.isfinite(mu)
                or not 1e-3 <= float(mu) <= 10.):
            raise CorroborationError("Lift joint friction coefficient must lie in [0.001, 10]")
        return {
            "law": "JOINT_MECA_FROT",
            "tensile_strength_pa": f_t,
            "friction_coefficient": float(mu),
            "adhesion_pa": float(mu) * f_t,
            "fracture_energy_j_m2": 0.0,
            "fracture_energy_note": "not used: JOINT_MECA_FROT is perfectly plastic (no softening)",
            "k_n_pa_per_m": penalty,
            # As for the damaged friction joints: E/h regularises stick-slip, the normal
            # penalty stays stiff.
            "k_t_pa_per_m": penalty / JOINT_PENALTY_FACTOR,
            "stiffness_basis": "shared mesh penalty (normal) and neighbouring element stiffness (tangential)",
            "mesh_penalty_pa_per_m": penalty,
            "neighbour_stiffness_pa_per_m": neighbour,
            # A friction joint has no cohesive elastic branch; any opening beyond the
            # hydraulic tolerance counts as open.
            "elastic_limit_opening_m": FRICTION_OPENING_LIMIT_M,
            "critical_opening_m": None,
        }
    if law != "JOINT_MECA_RUPT":
        raise CorroborationError("Lift joint law must be JOINT_MECA_RUPT or JOINT_MECA_FROT")
    k_n = f_t ** 2 * (1.0 + LIFT_PENA_RUPTURE_MAX) / (2.0 * g_f)
    pena_rupture = LIFT_PENA_RUPTURE_MAX
    stiffness_basis = "joint strength and fracture energy at the catalogue PENA_RUPTURE cap"
    if k_n < neighbour:
        limit = f_t ** 2 * (1.0 + LIFT_PENA_RUPTURE_MAX) / (2.0 * neighbour)
        raise CorroborationError(
            f"JOINT_MECA_RUPT cannot dissipate Gf = {g_f:g} J/m2 at f_t = {f_t:g} Pa without a joint softer "
            f"than its neighbouring elements (K_N {k_n:.4g} Pa/m < E/h {neighbour:.4g} Pa/m; Code_Aster "
            f"U2.05.07 4.1.2). The largest fracture energy this law can carry here is {limit:.3g} J/m2. "
            "Lower lift_joints.cohesive_fracture_energy_j_m2 or use the JOINT_MECA_FROT law.")
    if k_n > penalty:
        k_n = penalty
        pena_rupture = 2.0 * penalty * g_f / f_t ** 2 - 1.0
        stiffness_basis = "shared mesh penalty (the joint energy would need a stiffer K_N)"
        if not pena_rupture > 0:
            raise CorroborationError(
                f"JOINT_MECA_RUPT needs PENA_RUPTURE > 0 but Gf={g_f:g}, f_t={f_t:g} and K_N={k_n:.6g} give "
                f"{pena_rupture:.6g}: the elastic opening f_t/K_N already exceeds the critical opening 2Gf/f_t")
    if not LIFT_PENA_RUPTURE_MIN <= pena_rupture <= LIFT_PENA_RUPTURE_MAX:
        raise CorroborationError(f"JOINT_MECA_RUPT PENA_RUPTURE {pena_rupture:.6g} is outside the catalogue range")
    return {
        "law": "JOINT_MECA_RUPT",
        "tensile_strength_pa": f_t,
        "fracture_energy_j_m2": g_f,
        "k_n_pa_per_m": float(k_n),
        "k_t_pa_per_m": float(k_n),
        "stiffness_basis": stiffness_basis,
        "mesh_penalty_pa_per_m": penalty,
        "neighbour_stiffness_pa_per_m": neighbour,
        "pena_rupture": float(pena_rupture),
        "pena_contact": float(contact),
        "alpha": float(alpha),
        "elastic_limit_opening_m": f_t / float(k_n),
        "critical_opening_m": 2.0 * g_f / f_t,
    }


def render_cohesive_comm(base_comm: str, joint: dict, factors: list[float], steps_path: Path,
                         status_path: Path, tie_pairs: int = 0,
                         interpolation: dict | None = None) -> str:
    """Turn the job's linear baseline case into the stepped cohesive case."""
    if joint.get("continuation") is not None and (
            not factors or factors[-1] != 1.0 or any(factor > 1.0 for factor in factors)):
        raise CorroborationError("Fixed-mesh continuation requires a ramp ending at design load factor 1.0")
    for name, anchor in _ANCHORS.items():
        if base_comm.count(anchor) != 1:
            raise CorroborationError(f"The baseline case no longer has its '{name}' section")
    comm = base_comm[:base_comm.index(_ANCHORS["solve"])]
    from inspect import getsource
    from Aster.run_case import resource_failure
    comm = comm.replace(_ANCHORS["debut"], (
        "import json\nimport re\nimport time\nimport os\n\nimport numpy as np\n\n"
        "from code_aster.Commands import *\nfrom code_aster import CA\n\n"
        "from code_aster.Utilities import MPI\n\n"
        "from contextlib import contextmanager\n\n"
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
    friction_law = joint.get("damaged_friction")
    lift_law = joint.get("lift_joint")
    lift_frictional = lift_law is not None and lift_law.get("law") == "JOINT_MECA_FROT"
    any_friction = joint.get("damaged_friction") is not None or lift_frictional
    lift_count = int(joint.get("lift_joint_count", 0))
    damaged_count = int(joint.get("damaged_joint_count", 0))
    vertical_count = int(joint.get("vertical_joint_count", 0))
    intact_count = int(joint.get("candidate_face_count", 0)) - lift_count - damaged_count - vertical_count
    if has_joints and assignment is not None:
        # Four facet classes: LIFT_JOINTS (weak construction joint, JOINT_MECA_RUPT),
        # DAMAGED_JOINTS (facets of degraded elements: their CZM or Mohr-Coulomb friction),
        # VERTICAL_JOINTS (CZM on the radial plane through the tensile peak) and INTACT_JOINTS
        # (the intact concrete's CZM). JOINT_MAT already carries the summary law, i.e. the
        # degraded material's CZM when elements are degraded.
        materials = ""
        if lift_frictional:
            materials += (
                "LIFT_MAT = DEFI_MATERIAU(\n"
                f"    JOINT_MECA_FROT=_F(K_N={lift_law['k_n_pa_per_m']:.12g}, K_T={lift_law['k_t_pa_per_m']:.12g},\n"
                f"                       MU={lift_law['friction_coefficient']:.12g}, "
                f"ADHESION={lift_law['adhesion_pa']:.12g}),\n"
                ")\n"
                "# Doc R7.01.25 JOINT_MECA_FROT: weak construction (lift) joint as a Mohr-Coulomb joint with\n"
                "# contact. |tau| <= ADHESION - MU sigma_n; the cone's apex ADHESION / MU = "
                f"{lift_law['tensile_strength_pa']:.6g} Pa is the\n"
                "# configured tensile strength (perfectly plastic: not lost after opening). The plane can slide.\n\n")
        elif lift_law is not None:
            materials += (
                "LIFT_MAT = DEFI_MATERIAU(\n"
                f"    JOINT_MECA_RUPT=_F(K_N={lift_law['k_n_pa_per_m']:.12g}, K_T={lift_law['k_t_pa_per_m']:.12g},\n"
                f"                       SIGM_MAX={lift_law['tensile_strength_pa']:.12g}, "
                f"PENA_RUPTURE={lift_law['pena_rupture']:.12g},\n"
                f"                       PENA_CONTACT={lift_law['pena_contact']:.12g}, ALPHA={lift_law['alpha']:.12g}),\n"
                ")\n"
                "# Doc R7.01.25 JOINT_MECA_RUPT: weak construction (lift) joint. Cohesive rupture at\n"
                "# SIGM_MAX with linear softening -K_N/PENA_RUPTURE, full separation at\n"
                f"# SIGM_MAX (1 + PENA_RUPTURE) / K_N = {lift_law['critical_opening_m']:.6g} m = 2 Gf / f_t\n"
                f"# (Gf = {lift_law['fracture_energy_j_m2']:.6g} J/m2); contact penalty PENA_CONTACT * K_N;\n"
                "# ALPHA sets how the shear stiffness decays with opening. No Coulomb slip limit.\n\n")
        if friction_law is not None:
            materials += (
                "FROT_MAT = DEFI_MATERIAU(\n"
                f"    JOINT_MECA_FROT=_F(K_N={friction_law['k_n_pa_per_m']:.12g}, "
                f"K_T={friction_law['k_t_pa_per_m']:.12g},\n"
                f"                       MU={friction_law['friction_coefficient']:.12g}, "
                f"ADHESION={friction_law['adhesion_pa']:.12g}),\n"
                ")\n"
                "# Doc R7.01.25 JOINT_MECA_FROT: Mohr-Coulomb joint with contact. Pre-cracked damaged\n"
                "# joints carry compression and friction but no tension or cohesive softening.\n\n")
        original = joint["intact_joint"]
        materials += (
            "INTACT_JOINT_MAT = DEFI_MATERIAU(\n"
            f"    ELAS=_F(E={original['concrete_youngs_modulus_pa']:.12g}, NU={original['concrete_poissons_ratio']:.12g}),\n"
            f"    RUPT_FRAG=_F(GC={original['fracture_energy_j_m2']:.12g}, SIGM_C={original['tensile_strength_pa']:.12g}, "
            f"PENA_ADHERENCE={original['pena_adherence']:.12g}, PENA_CONTACT=1.0),\n"
            ")\n"
            "# The intact joints keep the CZM_LIN_REG law built from the intact concrete.\n\n")
        if vertical_count:
            vertical = joint["vertical_joint"]
            materials += (
                "VERTICAL_JOINT_MAT = DEFI_MATERIAU(\n"
                f"    ELAS=_F(E={vertical['concrete_youngs_modulus_pa']:.12g}, NU={vertical['concrete_poissons_ratio']:.12g}),\n"
                f"    RUPT_FRAG=_F(GC={vertical['fracture_energy_j_m2']:.12g}, SIGM_C={vertical['tensile_strength_pa']:.12g}, "
                f"PENA_ADHERENCE={vertical['pena_adherence']:.12g}, PENA_CONTACT=1.0),\n"
                ")\n"
                f"# Radial vertical crack path through the tensile peak: critical opening "
                f"{vertical['critical_opening_m']:.6g} m = 2 Gf / f_t (f_t calibrated, not measured).\n\n")
        comm = comm.replace(_ANCHORS["chmat"], materials + _ANCHORS["chmat"])
        groups = ""
        if lift_count:
            groups += "        _F(GROUP_MA='LIFT_JOINTS', MATER=LIFT_MAT),\n"
        if damaged_count:
            groups += ("        _F(GROUP_MA='DAMAGED_JOINTS', MATER=FROT_MAT),\n" if friction_law is not None
                       else "        _F(GROUP_MA='DAMAGED_JOINTS', MATER=JOINT_MAT),\n")
        if vertical_count:
            groups += "        _F(GROUP_MA='VERTICAL_JOINTS', MATER=VERTICAL_JOINT_MAT),\n"
        if intact_count > 0:
            groups += "        _F(GROUP_MA='INTACT_JOINTS', MATER=INTACT_JOINT_MAT),\n"
        comm = comm.replace("        _F(GROUP_MA='JOINTS', MATER=JOINT_MAT),\n", groups)
    opening_threshold = repr(joint.get("elastic_limit_opening_m", 0.))
    friction_cells = "None"
    lift_cells = "None"
    if assignment is not None:
        parents = interpolation["joint_parent_faces"]
        damaged_faces = np.asarray(interpolation["damaged_joint_faces"], dtype=bool)[parents]
        lift_faces = (np.asarray(interpolation["lift_joint_faces"], dtype=bool)[parents]
                      if "lift_joint_faces" in interpolation else np.zeros_like(damaged_faces))
        vertical_faces = (np.asarray(interpolation["vertical_joint_faces"], dtype=bool)[parents]
                          if "vertical_joint_faces" in interpolation else np.zeros_like(damaged_faces))
        # Friction joints have no cohesive elastic limit; any opening beyond the
        # hydraulic tolerance counts as open.
        damaged_limit = FRICTION_OPENING_LIMIT_M if friction_law is not None else joint["elastic_limit_opening_m"]
        vertical_limit = joint["vertical_joint"]["elastic_limit_opening_m"] if vertical_count else 0.
        thresholds = np.where(lift_faces, lift_law["elastic_limit_opening_m"] if lift_law is not None else 0.,
                              np.where(vertical_faces, vertical_limit,
                                       np.where(damaged_faces, damaged_limit,
                                                joint["intact_joint"]["elastic_limit_opening_m"])))
        opening_threshold = "np.asarray(" + repr(thresholds.tolist()) + ")"
        # Mohr-Coulomb cells: the damaged friction joints and a frictional lift plane share
        # JOINT_MECA_FROT's internal variables; a rupture lift plane has its own mapping.
        frictional = ((damaged_faces if friction_law is not None else np.zeros_like(damaged_faces))
                      | (lift_faces if lift_frictional else np.zeros_like(lift_faces)))
        if frictional.any():
            friction_cells = "np.asarray(" + repr([bool(value) for value in frictional]) + ")"
        if lift_law is not None and not lift_frictional:
            lift_cells = "np.asarray(" + repr([bool(value) for value in lift_faces]) + ")"
    if assignment is not None and assignment["selected_source_rows"] and assignment.get("properties") is not None:
        properties = assignment["properties"]
        comm = comm.replace(_ANCHORS["chmat"], (
            "DAMAGED_MAT = DEFI_MATERIAU(\n"
            f"    ELAS=_F(E={properties['youngs_modulus']:.12g}, NU={properties['poisson_ratio']:.12g}, "
            f"RHO={properties['density']:.12g}),\n"
            ")\n" + _ANCHORS["chmat"]))
        comm = comm.replace(_ANCHORS["concrete_mat"], _ANCHORS["concrete_mat"] +
                            "        _F(GROUP_MA='DAMAGED_ELEMENTS', MATER=DAMAGED_MAT),\n")
    targets = ", ".join(f"{value:.10g}" for value in factors)
    if not has_joints:
        comport_joints = ""
    elif assignment is not None:
        comport_joints = ""
        if lift_count:
            comport_joints += (f"    _F(RELATION='{lift_law['law']}', GROUP_MA='LIFT_JOINTS'),\n")
        if damaged_count:
            comport_joints += ("    _F(RELATION='JOINT_MECA_FROT', GROUP_MA='DAMAGED_JOINTS'),\n"
                               if friction_law is not None else
                               "    _F(RELATION='CZM_LIN_REG', GROUP_MA='DAMAGED_JOINTS'),\n")
        if vertical_count:
            comport_joints += "    _F(RELATION='CZM_LIN_REG', GROUP_MA='VERTICAL_JOINTS'),\n"
        if intact_count > 0:
            comport_joints += "    _F(RELATION='CZM_LIN_REG', GROUP_MA='INTACT_JOINTS'),\n"
    else:
        comport_joints = "    _F(RELATION='CZM_LIN_REG', GROUP_MA='JOINTS'),\n"
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
    if "UPLIFT = AFFE_CHAR_MECA" in comm:
        tie_excit += "    _F(CHARGE=UPLIFT, FONC_MULT=RAMP),\n"
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
    # Friction joints need self-weight established before significant water load: ramping
    # gravity proportionally with the water leaves every friction face at the Mohr-Coulomb
    # apex (zero normal stress, indeterminate stick/slip) at small load factors, while a
    # constant gravity cannot be eased by time-step subdivision. Ramping gravity to full
    # by load factor 0.05 keeps compression ahead of shear at every instant and stays
    # subdividable; every solve target (the first is also 0.05) carries full self-weight.
    gravity_excit = ("    _F(CHARGE=GRAVITE, FONC_MULT=GRAVITY_RAMP),\n" if any_friction
                     else "    _F(CHARGE=GRAVITE, FONC_MULT=RAMP),\n")
    if "CONDENSED_GRAVITY = AFFE_CHAR_MECA" in comm:
        gravity_excit += (
            "    _F(CHARGE=CONDENSED_GRAVITY, FONC_MULT=GRAVITY_RAMP),\n" if any_friction
            else "    _F(CHARGE=CONDENSED_GRAVITY, FONC_MULT=RAMP),\n")
    gravity_ramp = ("GRAVITY_RAMP = DEFI_FONCTION(NOM_PARA='INST', VALE=(0.0, 0.0, 0.05, 1.0), "
                    "PROL_DROITE='CONSTANT')\n" if any_friction else "")
    # Softening laws (friction stick-slip, lift-joint rupture of many facets at once) need the
    # line search: a full tangent Newton step over a snap-through diverges from iteration 0.
    controls = joint.get("solver_controls") or {}
    line_iterations = int(controls.get("line_search_iterations", 8))
    maximum_iterations = int(controls.get("maximum_iterations", 40))
    subdivision_levels = int(controls.get("subdivision_levels", 6))
    residual = float(controls.get("relative_residual_tolerance", 1e-6))
    line_search = (f"\n        RECH_LINEAIRE=_F(METHODE='MIXTE', ITER_LINE_MAXI={line_iterations}),"
                   if has_joints else "")
    # Friction and lift-joint steps empirically converge by iteration ~30 or oscillate
    # forever on toggling faces; a 100-iteration budget made every failed bite cost ~30
    # wall-clock minutes before DECOUPE was allowed to subdivide. 40 keeps margin over the
    # observed convergence tail while making failures 2.5x cheaper.
    toggling_law = friction_law if friction_law is not None else lift_law
    convergence = (f"CONVERGENCE=_F(ITER_GLOB_MAXI={maximum_iterations}, "
                   f"RESI_GLOB_MAXI={toggling_law['residual_tolerance_n']:.6g}),"
                   if toggling_law is not None else
                   f"CONVERGENCE=_F(ITER_GLOB_MAXI={maximum_iterations}, RESI_GLOB_RELA={residual:.12g}),")
    inactive_displacements = ""
    if interpolation is not None and "condensed_active_nodes" in interpolation:
        active = np.asarray(interpolation["condensed_active_nodes"], dtype=np.int64)
        inactive_displacements = (
            f"    active = np.asarray({active.tolist()!r}, dtype=np.int64)\n"
            "    inactive = np.ones(len(result), dtype=bool)\n"
            "    inactive[active] = False\n"
            "    result[inactive] = 0.0\n")
    comm += getsource(resource_failure) + "\n"
    comm += f'''{gravity_ramp}RAMP = DEFI_FONCTION(NOM_PARA='INST', VALE=(0.0, 0.0, 100.0, 100.0), PROL_DROITE='LINEAIRE')
# Every load is scaled by the load factor (= INST), as in the FEniCSx proportional ramp.

STEPS_PATH = {_comm_string(steps_path)}
STATUS_PATH = {_comm_string(status_path)}
IS_WRITER = MPI.ASTER_COMM_WORLD.Get_rank() == 0
TARGETS = [{targets}]
TARGET_LEVELS = [0] * len(TARGETS)
JOINT_CELLS = list(MAIL.getCells('JOINTS')) if {has_joints!r} else []
# Per-joint-cell mask of Mohr-Coulomb friction cells (None = all cohesive).
FRICTION_CELLS = {friction_cells}
# Per-joint-cell mask of JOINT_MECA_RUPT lift-joint cells (None = none).
LIFT_CELLS = {lift_cells}
LIFT_GF = {(lift_law or {}).get('fracture_energy_j_m2', 0.0)!r}


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


def joint_gauss(field, component):
    values, description = field.toSimpleFieldOnCells().getValuesWithDescription(
        [component], ['JOINTS'])
    return np.asarray(values, dtype=float), np.asarray(description[0], dtype=np.int64)


def joint_mean(values, cells):
    # Mean over each joint cell's Gauss points: plane resultants need means, not peaks.
    index = {{cell: row for row, cell in enumerate(JOINT_CELLS)}}
    rows = np.array([index[cell] for cell in cells.tolist()], dtype=np.int64)
    total, count = np.zeros(len(JOINT_CELLS)), np.zeros(len(JOINT_CELLS))
    np.add.at(total, rows, values)
    np.add.at(count, rows, 1.0)
    return total / np.maximum(count, 1.0)


def nodal_displacement(field):
    result = np.full((MAIL.getNumberOfNodes(), 3), np.nan)
    for column, component in enumerate(('DX', 'DY', 'DZ')):
        values, description = field.getValuesWithDescription([component])
        result[np.asarray(description[0], dtype=np.int64), column] = np.asarray(values, dtype=float)
{inactive_displacements}\
    return result


history = {{"load_factor": [], "damage_state": [], "normal_jump_m": [], "normal_traction_pa": [],
           "dissipated_energy_j_m2": [], "mean_normal_traction_pa": [], "mean_tangential_traction_pa": [],
           "sliding": [], "cumulative_slip_m": []}}
status = {{"schema_version": 1, "status": "running", "targets": TARGETS, "converged_load_factors": [],
          "joint_cells": len(JOINT_CELLS), "started_at": time.time(), "step_timings": []}}


@contextmanager
def root_archive_write():
    failure = None
    try:
        yield IS_WRITER
    except OSError as error:
        failure = (error.errno, str(error))
    failure = MPI.ASTER_COMM_WORLD.bcast(failure, root=0)
    if failure is not None:
        raise OSError(failure[0], failure[1])


def save(final=None, record_timing=False):
    if final is not None:
        status.update(final)
    with root_archive_write() as writer:
        if writer:
            archive_started = time.perf_counter()
            with open(STEPS_PATH + '.tmp', 'wb') as stream:
                np.savez(stream, **{{key: np.asarray(value, dtype=float) for key, value in history.items()}})
            os.replace(STEPS_PATH + '.tmp', STEPS_PATH)
            if record_timing:
                status["step_timings"][-1]["archive_s"] = time.perf_counter() - archive_started
            with open(STATUS_PATH + '.tmp', 'w', encoding='utf-8') as stream:
                json.dump(status, stream, indent=2)
            os.replace(STATUS_PATH + '.tmp', STATUS_PATH)


def stop_for_resource_failure(error):
    detail = (type(error).__name__ + ': ' + str(error) if isinstance(error, (OSError, MemoryError))
              else str(error))
    failure = resource_failure(detail)
    if failure is None:
        return False
    status.update(status='failed', stop_reason='resource_limit', resource_failure=failure,
                  stopped_at_load_factor=min(target, 1.0), reached_load_factor=min(reached, 1.0),
                  message=failure['message'])
    print('LAMB_SOLVER_RESOURCE_ERROR ' + json.dumps(failure), flush=True)
    save()
    return True


save()
EXCIT = (
{gravity_excit}    _F(CHARGE=PRESSIONS, FONC_MULT=RAMP),
    _F(CHARGE=CREST_SURCHARGE, FONC_MULT=RAMP),
{tie_excit})
COMPORT = (
    _F(RELATION='ELAS', TOUT='OUI'),
{comport_joints})
RESU = None
reached = 0.0
mesh_coordinates = np.asarray(MAIL.getCoordinates().getValues(), dtype=float).reshape(-1, 3)
accepted_depl = None
target_index = 0
while target_index < len(TARGETS):
    target = TARGETS[target_index]
    target_level = TARGET_LEVELS[target_index]
    step_started = time.perf_counter()
    mechanical_s = 0.0
    print(f"ASTER_COHESIVE_STEP start load_factor={{target:.6g}}", flush=True)
    times = DEFI_LIST_INST(
        DEFI_LIST=_F(LIST_INST=DEFI_LIST_REEL(DEBUT=reached, INTERVALLE=_F(JUSQU_A=target, NOMBRE=1))),
        ECHEC=_F(EVENEMENT='ERREUR', ACTION='DECOUPE', SUBD_METHODE='MANUEL', SUBD_PAS=4, SUBD_NIVEAU={subdivision_levels}),
    )
    options = dict(
        MODELE=MODELE, CHAM_MATER=CHMAT, CARA_ELEM=CARA_ROCK, EXCIT=EXCIT, COMPORTEMENT=COMPORT,
        INCREMENT=_F(LIST_INST=times), NEWTON=_F(MATRICE='TANGENTE', REAC_ITER=1),{line_search}
        {convergence}
        SOLVEUR=_F(METHODE='MULT_FRONT'),
    )
    try:
        mechanical_started = time.perf_counter()
        if RESU is None:
            RESU = STAT_NON_LINE(**options)
        else:
            RESU = STAT_NON_LINE(reuse=RESU, ETAT_INIT=_F(EVOL_NOLI=RESU), **options)
        mechanical_s += time.perf_counter() - mechanical_started
    except CA.AsterError as error:
        if stop_for_resource_failure(error):
            break
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
    # CZM_LIN_REG: V3=INDIENDO (damage), V5=DISSIP, V7=SAUT_N.
    # JOINT_MECA_FROT: V2=INDIPLAS (sliding), V5=INDIOUV (open), V7=SAUT_N.
    # JOINT_MECA_RUPT (lib64/aster/code_aster/Behaviours/joint_meca_rupt.py, 20 variables;
    # documented in share/aster/tests/ssnp162b.comm): V1=SEUILDEP, V2=INDIDISS,
    # V3=INDIENDN (0 sain, 1 endommagé, 2 cassé), V4=PCENERDI (% normal damage),
    # V5=INDIENDT, V6=PCENDOT, V7=SAUT_N. V3/V7 therefore share CZM_LIN_REG's slots; the
    # law has no DISSIP variable so dissipated energy is rebuilt from V4 and the joint Gf.
    damage = joint_component(vari, 'V3')
    dissipated = joint_component(vari, 'V5')
    sliding_flag = np.zeros(len(damage))
    slip = np.zeros(len(damage))
    if FRICTION_CELLS is not None and len(damage):
        opened = joint_component(vari, 'V5')
        sliding = joint_component(vari, 'V2')
        damage = np.where(FRICTION_CELLS,
                          np.where(opened > 0.5, 2.0, np.where(sliding > 0.5, 1.0, 0.0)), damage)
        dissipated = np.where(FRICTION_CELLS, 0.0, dissipated)
        # JOINT_MECA_FROT (R7.01.25 3.3): V1 = cumulative plastic tangential displacement,
        # V2 = sliding indicator.
        sliding_flag = np.where(FRICTION_CELLS, (sliding > 0.5).astype(float), 0.0)
        slip = np.where(FRICTION_CELLS, joint_component(vari, 'V1'), 0.0)
    if LIFT_CELLS is not None and len(damage):
        # PCENERDI is the fraction of the normal softening branch consumed; the linear
        # softening dissipates Gf * fraction up to full separation.
        fraction = np.clip(joint_component(vari, 'V4'), 0.0, 1.0)
        dissipated = np.where(LIFT_CELLS, LIFT_GF * fraction, dissipated)
    history["damage_state"].append(damage)
    history["normal_jump_m"].append(joint_component(vari, 'V7'))
    history["dissipated_energy_j_m2"].append(dissipated)
    history["normal_traction_pa"].append(joint_component(sief, 'SIGN'))
    if JOINT_CELLS:
        normal_gauss, gauss_cells = joint_gauss(sief, 'SIGN')
        shear_gauss = np.hypot(joint_gauss(sief, 'SITX')[0], joint_gauss(sief, 'SITY')[0])
        history["mean_normal_traction_pa"].append(joint_mean(normal_gauss, gauss_cells))
        history["mean_tangential_traction_pa"].append(joint_mean(shear_gauss, gauss_cells))
    else:
        history["mean_normal_traction_pa"].append(np.empty(0))
        history["mean_tangential_traction_pa"].append(np.empty(0))
    history["sliding"].append(sliding_flag)
    history["cumulative_slip_m"].append(slip)
    accepted_depl = nodal_displacement(RESU.getField('DEPL', last))
    if len(history["load_factor"]) == 1:
        first_depl = accepted_depl.copy()
    status["step_timings"].append({{"load_factor": target, "mechanical_s": mechanical_s,
                                  "step_s": time.perf_counter() - step_started}})
    status["converged_load_factors"].append(target)
    save(record_timing=True)
    print(f"ASTER_COHESIVE_STEP done load_factor={{target:.6g}} damaged_joints="
          f"{{int(np.sum(history['damage_state'][-1] >= 1.0))}}", flush=True)
    if {joint.get("stop_overload_on_opening", False)!r} and target >= 1.0 and np.any(
            (history["damage_state"][-1] >= 1.0) &
            (history["normal_jump_m"][-1] > {opening_threshold})):
        save({{"overload_stopped_on_opening": True}})
        break
    target_index += 1

if history["load_factor"]:
    with root_archive_write() as writer:
        if writer:
            path = STEPS_PATH.replace('.npz', '_displacement.npz')
            with open(path + '.tmp', 'wb') as stream:
                np.savez(stream, first=first_depl, final=accepted_depl, coordinates=mesh_coordinates)
            os.replace(path + '.tmp', path)
if status["status"] == "running":
    save({{"status": "completed"}})
elif not history["load_factor"]:
    save({{"status": "failed"}})
status["finished_at"] = time.time()
save()

if status.get('resource_failure') is not None:
    raise SystemExit(1)
FIN()
'''
    if "hydraulic_loads" in joint:
        from Aster import hydraulic_loads
        settings = hydraulic_loads.validate_settings(joint["hydraulic_loads"]["settings"])
        metadata = joint["hydraulic_loads"]
        runtime_source = Path(hydraulic_loads.__file__).read_text(encoding="utf-8")
        archive = steps_path.parent.parent / SPLIT_ARRAYS_NAME
        plane_uplift = metadata.get("lift_plane_uplift") or {}
        pore = ""
        if plane_uplift.get("mode", "none") != "none":
            # Pore pressure inside the lift plane (closed faces); wet faces keep the full head.
            pore = (f", pore_faces=np.asarray(hydraulic_arrays['lift_joint_faces'], dtype=bool), "
                    f"downstream_z={float(plane_uplift['downstream_z_m'])!r}")
        comm = comm.replace("RESU = None\n", (
            f"exec(compile({runtime_source!r}, 'hydraulic_loads.py', 'exec'))\n"
            f"hydraulic_settings = {settings!r}\n"
            f"hydraulic_subdivision_levels = {hydraulic_loads.MAX_LOAD_INTERVAL_BISECTIONS}\n"
            f"with np.load({_comm_string(archive.resolve())}, allow_pickle=False) as stored:\n"
            "    hydraulic_arrays = {key: stored[key] for key in stored.files}\n"
            f"crack_water = CrackWater(hydraulic_arrays, {metadata['upstream_z_m']!r}, "
            f"{metadata['gradient_pa_per_m']!r}, hydraulic_settings['opening_tolerance_m']{pore})\n"
            "status['lift_plane_uplift'] = crack_water.pore_report\n"
            f"hydraulic_uplift_resultant = np.asarray({metadata['uplift']['resultant_n']!r})\n"
            f"HYDRAULIC_OPENING_LIMIT = {opening_threshold}\n"
            "def hydraulic_open_faces(result):\n"
            "    if result is None:\n"
            "        return np.zeros(len(hydraulic_arrays['penta6_joints']), dtype=bool)\n"
            "    if not JOINT_CELLS:\n"
            "        return np.empty(0, dtype=bool)\n"
            "    field = result.getField('VARI_ELGA', result.getLastIndex())\n"
            "    damage = joint_component(field, 'V3')\n"
            "    jump = joint_component(field, 'V7')\n"
            "    if FRICTION_CELLS is not None:\n"
            "        opened = joint_component(field, 'V5')\n"
            "        damage = np.where(FRICTION_CELLS, np.where(opened > 0.5, 2., 0.), damage)\n"
            "    return ((damage >= 1. - 1e-9) & (jump > HYDRAULIC_OPENING_LIMIT)).reshape(-1, 3).any(axis=1)\n"
            "hydraulic_history = []\nhydraulic_masks = []\nRESU = None\n"))
        start = comm.index("    try:\n        mechanical_started = time.perf_counter()\n        if RESU is None:")
        end = comm.index("    reached = target\n", start)
        comm = comm[:start] + hydraulic_loads.COUPLED_SOLVE + comm[end:]
        # The hydraulic trials keep the DEFI_LIST_INST / DECOUPE sub-stepping: each trial is a
        # fresh STAT_NON_LINE whose last archived instant is still the target, and chaining via
        # ETAT_INIT(INST=reached) is unaffected by archived sub-instants. Without it one
        # divergent Newton step (softening snap-through of many joints) ended the whole run.
        comm = comm.replace('    status["converged_load_factors"].append(target)',
                            '    status["hydraulic_steps"] = hydraulic_history\n'
                            '    status["converged_load_factors"].append(target)')
    if joint.get("continuation") is not None:
        from Aster.cohesive_continuation import render, render_preload
        comm = render(comm, dict(joint["continuation"], opening_limit_m=joint["elastic_limit_opening_m"]))
        if joint.get("reservoir_loading") is not None:
            from Aster.reservoir_continuation import render as render_reservoir
            comm = render_reservoir(comm, joint["reservoir_loading"], joint["tensile_strength_pa"],
                                    joint["reservoir_loading"]["selection_enabled"])
        else:
            comm = render_preload(comm, preload_factor(joint.get("solver_controls") or {}),
                                  joint["tensile_strength_pa"])
    elif preload_factor(joint.get("solver_controls") or {}) > 0:
        raise CorroborationError("Intact preload requires the fixed wall-facet bank")
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
    if summary.get("continuation") is not None:
        with np.load(fenicsx_output / "cohesive_state.npz", allow_pickle=False) as state:
            split["arrays"]["continuation_active_faces"] = state["continuation_active_faces"].copy()
            split["arrays"]["continuation_face_owners"] = state["continuation_face_owners"].copy()
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
    lift_properties = assignment.get("joint_properties") if isinstance(assignment, dict) else None
    degraded = assignment is not None and assignment.get("properties") is not None
    if assignment is not None:
        selected_rows = np.asarray(assignment["selected_source_rows"], dtype=np.int64)
        volume_rows = split["arrays"]["source_cell_rows"]
        selected = np.isin(volume_rows, selected_rows)
        concrete_count = len(split["arrays"]["tetra4_concrete"])
        if (len(np.unique(selected_rows)) != len(selected_rows) or selected.sum() != len(selected_rows)
                or selected[concrete_count:].any()
                or np.any(~np.isin(split["arrays"]["concrete_body_ids"][selected[:concrete_count]], (1, 3)))):
            raise CorroborationError("Damaged material assignment must match exact selected wall/foundation source cells")
    # Three facet classes. Lift facets (the recorded near-horizontal planes) take JOINT_MECA_RUPT;
    # the remaining facets between degraded elements take the degraded-material CZM (or
    # friction); everything else is intact. A lift facet is never also a damaged facet.
    lift_faces = np.zeros(len(faces), dtype=bool)
    if lift_properties is not None:
        with np.load(fenicsx_output / "cohesive_state.npz") as state:
            if "lift_joint_faces" not in state.files:
                raise CorroborationError("Lift joints are configured but the prepared candidates carry no lift facet mask")
            lift_faces = np.asarray(state["lift_joint_faces"], dtype=bool)
        if lift_faces.shape != (len(faces),) or not lift_faces.any():
            raise CorroborationError("The lift facet mask does not match the candidate facets or selects none")
        if int(lift_faces.sum()) != int(lift_properties.get("lift_joint_facets", -1)):
            raise CorroborationError("The lift facet mask disagrees with the screening's lift-joint facet count")
        from FEniCSx.lift_joints import facet_unit_normals
        limit = math.cos(math.radians(float(lift_properties["max_normal_deviation_deg"]))) - 1e-12
        if np.any(np.abs(facet_unit_normals(arrays["nodes"], faces[lift_faces])[:, 2]) < limit):
            raise CorroborationError("A recorded lift-joint facet is not near-horizontal in the Aster mesh")
    if assignment is not None:
        split["arrays"]["lift_joint_faces"] = lift_faces
        if degraded:
            split["arrays"]["damaged_cells"] = selected
            selected_faces = {tuple(sorted(face)) for cell in arrays["tetra4_concrete"][selected[:concrete_count]]
                              for face in (cell[[0, 1, 2]], cell[[0, 1, 3]], cell[[0, 2, 3]], cell[[1, 2, 3]])}
            split["arrays"]["damaged_joint_faces"] = np.array(
                [tuple(sorted(face)) in selected_faces for face in faces], dtype=bool) & ~lift_faces
        else:
            split["arrays"]["damaged_cells"] = np.zeros(len(split["arrays"]["source_cell_rows"]), dtype=bool)
            split["arrays"]["damaged_joint_faces"] = np.zeros(len(faces), dtype=bool)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_manifest = json.loads((baseline_dir / MESH_MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest = split_manifest(baseline_manifest, split)
    if summary.get("crack_water_pressure") is True:
        from Aster.hydraulic_loads import CRACK_WATER_SETTINGS
        if "hydraulic_loads" in manifest and manifest["hydraulic_loads"] != CRACK_WATER_SETTINGS:
            raise CorroborationError("Crack-water-only configuration conflicts with hydraulic campaign settings")
        manifest["hydraulic_loads"] = dict(CRACK_WATER_SETTINGS)
    degradation = summary.get("support_degradation")
    if degradation is not None:
        degradation = apply_support_degradation(manifest, split["arrays"], degradation)
    save_split_arrays(output_dir, split)
    manifest_path = output_dir / MESH_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    base_comm, base_inputs = render_comm(mesh_manifest_path=manifest_path)
    concrete = base_inputs["material"]["concrete"]
    friction = summary.get("damaged_joint_friction")
    if friction is not None and assignment is None:
        raise CorroborationError("Damaged joint friction requires a damaged material assignment")
    if friction is not None and not degraded:
        raise CorroborationError("Damaged joint friction requires degraded element properties")
    law_summary = summary
    source_concrete = json.loads(
        Path(baseline_manifest["source_config"]).read_text(encoding="utf-8"))["material"]["concrete"]
    if friction is not None:
        original_concrete = source_concrete
        # The damaged joints use Mohr-Coulomb friction; the CZM parameters then
        # describe the intact joints and must not be derived from the (possibly
        # strengthless) damaged properties.
        law_summary = {**summary, "tensile_strength_pa": original_concrete["tensile_strength_pa"],
                       "fracture_energy_j_m2": original_concrete["cohesive_fracture_energy_j_m2"]}
    joint = joint_parameters(arrays, faces, law_summary, concrete["youngs_modulus_pa"], concrete["poissons_ratio"])
    # A handful of marginal joint faces (friction stick-slip, or lift joints at the
    # elastic/softening and open/closed thresholds) toggle forever at machine precision,
    # so those solves converge on an absolute residual scaled by the structure weight;
    # RESI_GLOB_RELA normalises by the largest nodal load and is unreachable while the
    # faces flip. Dam-scale runs plateau at ~1e-4 of self-weight (a 0.01% imbalance
    # localised at the toggling faces; the lift-joint run cycled at 2e-4 relative with
    # a 21 N absolute residual), so that is the acceptance level.
    corners = np.asarray(arrays["nodes"], dtype=float)[np.asarray(arrays["tetra4_concrete"], dtype=np.int64)]
    volume = float(np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])).sum() / 6.)
    residual_tolerance_n = max(1e-4 * volume * float(source_concrete.get("density", 2400.)) * 9.81, 1.)
    if friction is not None:
        joint["damaged_friction"] = friction_joint_parameters(friction, joint["initial_stiffness_pa_per_m"])
        joint["damaged_friction"]["residual_tolerance_n"] = residual_tolerance_n
    if lift_properties is not None:
        # ``joint`` is the CZM of the summary (degraded-material values when elements are
        # degraded, else the intact concrete's); the lift planes take their own law. The
        # neighbouring element stiffness is E/h over the lift facets themselves, the same
        # measure the element selection checks before any solve is spent.
        from FEniCSx.lift_joints import mean_facet_edge
        neighbour = float(concrete["youngs_modulus_pa"]) / mean_facet_edge(arrays["nodes"], faces[lift_faces])
        joint["lift_joint"] = lift_joint_parameters(lift_properties, joint["initial_stiffness_pa_per_m"], neighbour)
        joint["lift_joint"]["residual_tolerance_n"] = residual_tolerance_n
    joint["has_joints"] = bool(len(faces))
    joint["stop_overload_on_opening"] = summary.get("status") == "prepared_aster_candidates"
    joint["material_assignment"] = assignment
    joint["solver_controls"] = summary.get("solver_controls") or {}
    if summary.get("continuation") is not None:
        joint["continuation"] = {
            **summary["continuation"],
            "arrays_path": str((output_dir / SPLIT_ARRAYS_NAME).resolve()),
            "youngs_modulus_pa": concrete["youngs_modulus_pa"],
            "poissons_ratio": concrete["poissons_ratio"],
        }
        joint["stop_overload_on_opening"] = False
    if "hydraulic_loads" in base_inputs:
        joint["hydraulic_loads"] = {
            **base_inputs["hydraulic_loads"],
            "upstream_z_m": base_inputs["upstream_water_surface_z_m"],
            "gradient_pa_per_m": base_inputs["water_pressure_gradient_pa_per_m"],
        }
    if lift_properties is not None and lift_properties.get("uplift", "none") != "none":
        if "hydraulic_loads" not in joint:
            raise CorroborationError("Lift-plane uplift requires crack_water_pressure (the coupled hydraulic solve)")
        joint["hydraulic_loads"]["lift_plane_uplift"] = {
            "mode": lift_properties["uplift"],
            "downstream_z_m": float(base_inputs["tailwater_elevation_m"]),
        }
    if assignment is not None:
        config = json.loads(Path(baseline_manifest["source_config"]).read_text(encoding="utf-8"))
        original = config["material"]["concrete"]
        joint["intact_joint"] = joint_parameters(
            arrays, faces, {"tensile_strength_pa": original["tensile_strength_pa"],
                            "fracture_energy_j_m2": original["cohesive_fracture_energy_j_m2"]},
            concrete["youngs_modulus_pa"], concrete["poissons_ratio"])
        joint["damaged_joint_count"] = int(split["arrays"]["damaged_joint_faces"].sum())
        joint["lift_joint_count"] = int(split["arrays"]["lift_joint_faces"].sum())
        joint["candidate_face_count"] = len(faces)
        vertical_cfg = vertical_joint_settings(config)
        vertical = np.zeros(len(faces), dtype=bool)
        if vertical_cfg is not None:
            peak = summary.get("peak_location_m")
            if peak is None:
                raise CorroborationError("The vertical joint path needs the tensile peak (peak_location_m) from screening")
            band = (vertical_cfg["band_m"] if vertical_cfg["band_m"] is not None
                    else joint["intact_joint"]["mean_facet_edge_m"])
            vertical = (vertical_crack_facets(arrays["nodes"], faces, peak, band,
                                              vertical_cfg["max_normal_deviation_deg"])
                        & ~np.asarray(split["arrays"]["lift_joint_faces"], dtype=bool)
                        & ~np.asarray(split["arrays"]["damaged_joint_faces"], dtype=bool))
            if not vertical.any():
                raise CorroborationError("No facets lie on the vertical crack plane through the tensile peak")
            strength = vertical_joint_strength(vertical_cfg["critical_opening_m"], vertical_cfg["fracture_energy_j_m2"])
            joint["vertical_joint"] = joint_parameters(
                arrays, faces[vertical], {"tensile_strength_pa": strength,
                                          "fracture_energy_j_m2": vertical_cfg["fracture_energy_j_m2"]},
                concrete["youngs_modulus_pa"], concrete["poissons_ratio"])
        joint["vertical_joint_count"] = int(vertical.sum())
        split["arrays"]["vertical_joint_faces"] = vertical
        save_split_arrays(output_dir, split)
    source_config = json.loads(Path(baseline_manifest["source_config"]).read_text(encoding="utf-8"))
    condense = source_config.get("bedrock_analysis", {}).get("condense_foundation", False)
    if condense:
        if degradation is not None:
            raise CorroborationError("Condensed foundation does not support support-degradation scenarios")
        hydraulic = manifest.get("hydraulic_loads", {})
        if hydraulic.get("base_uplift", "none") != "none":
            raise CorroborationError("Condensed foundation does not support base uplift")
        from Aster.foundation_condensation import prepare_condensed_support
        split["arrays"], condensation = prepare_condensed_support(
            split["arrays"], joint_cells=split["joints"],
            youngs_modulus=concrete["youngs_modulus_pa"],
            poisson_ratio=concrete["poissons_ratio"],
            density=concrete["density_kg_m3"],
            gravity_m_s2=-float(base_inputs["gravity_m_s2"]), max_links=40000,
            max_dense_bytes=int(((source_config.get("fracture_debug") or {}).get("aster") or {}).get(
                "maximum_condensation_memory_mb", 4096)) * 1024**2,
        )
        save_split_arrays(output_dir, split)
        manifest["condensed_foundation"] = condensation
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        base_comm, base_inputs = render_comm(mesh_manifest_path=manifest_path)
    if summary.get("reservoir_loading") is not None:
        from Elmer.reservoir_loading import reservoir_loading
        loading_config = dict(source_config, fracture_debug=dict(
            source_config.get("fracture_debug") or {}, aster=joint["solver_controls"]))
        loading = reservoir_loading(loading_config, base_inputs)
        if loading is None:
            raise CorroborationError("Prepared reservoir loading has no explicit source configuration mode")
        if joint.get("continuation") is None or joint.get("damaged_friction") or joint.get("lift_joint"):
            raise CorroborationError("Reservoir filling requires the fixed wall cohesive bank and CZM_LIN_REG")
        controls = joint["solver_controls"]
        joint["reservoir_loading"] = dict(
            loading.metadata(), water_depth_targets_m=loading.targets(),
            selection_enabled=summary["reservoir_loading"]["selection_enabled"],
            maximum_iterations=int(controls.get("maximum_iterations", 60)),
            relative_residual_tolerance=float(controls.get("relative_residual_tolerance", 5e-6)),
            line_search_iterations=int(controls.get("line_search_iterations", 12)))
        if "hydraulic_loads" in joint:
            if joint["hydraulic_loads"]["settings"]["base_uplift"] != "none":
                raise CorroborationError("Reservoir filling does not enable base uplift")
            joint["reservoir_loading"]["hydraulic_settings"] = joint["hydraulic_loads"]["settings"]
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
        "support_degradation": degradation,
        "fenicsx_summary_sha256": file_hash(fenicsx_output / "summary.json"),
        "fenicsx_state_sha256": file_hash(fenicsx_output / "cohesive_state.npz"),
        "baseline_arrays_sha256": file_hash(baseline_dir / SPLIT_ARRAYS_NAME),
        "case_sha256": file_hash(output_dir / CASE_NAME),
        "baseline_manifest_sha256": file_hash(baseline_dir / MESH_MANIFEST_NAME),
        "source_configuration_sha256": file_hash(Path(baseline_manifest["source_config"])),
    }
    if condense:
        inputs["condensed_foundation"] = condensation
    if assignment is not None or joint.get("reservoir_loading") is not None:
        baseline_fields = baseline_dir / "aster_full_displacement.npz"
        inputs.update(baseline_displacement_path=str(baseline_fields.resolve()),
                      baseline_displacement_sha256=file_hash(baseline_fields),
                      intact_concrete_properties=concrete)
        if assignment is not None:
            inputs["material_assignment"] = assignment
    (output_dir / "case_inputs.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    return inputs


def solve(output_dir: Path, time_limit_s: int | None, memory_limit_mb: int = MEMORY_LIMIT_MB,
          runtime: str = "wsl") -> dict:
    """Run the prepared case with the local Code_Aster; returns the case's own status."""
    from Aster.run_case import (build_med_mesh, detect_installation, MESH_UNIT, run_solver,
                               run_wsl_solver, native_command, resource_failure)

    run_dir = output_dir / "run"
    run_dir.mkdir(exist_ok=True)
    with np.load(output_dir / SPLIT_ARRAYS_NAME, allow_pickle=False) as stored:
        condensed = "condensed_gravity_load_n" in stored.files
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
        arrays_path = (output_dir / SPLIT_ARRAYS_NAME).resolve()
        text = text.replace(_comm_string(arrays_path), repr(wsl_path(arrays_path)))
        if condensed:
            text = text.replace(_comm_string(ROOT.resolve()), repr(wsl_path(ROOT.resolve())))
        text = text.replace("SOLVEUR=_F(METHODE='MULT_FRONT')", "SOLVEUR=_F(METHODE='MUMPS', NPREC=12)")
        text = text.replace("SOLVEUR=_F(METHODE='MUMPS')", "SOLVEUR=_F(METHODE='MUMPS', NPREC=12)")
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
    for name in (STEPS_NAME, STATUS_NAME, STEPS_NAME.replace(".npz", "_displacement.npz"),
                 STEPS_NAME.replace(".npz", "_hydraulic.npz")):
        (run_dir / name).unlink(missing_ok=True)

    export_path = run_dir / "aster_cohesive.export"
    cpu_time_limit_s = (max(int(time_limit_s) * max(1, int(SOLVER_THREADS)), 3600)
                        if time_limit_s is not None else UNBOUNDED_CPU_TIME_LIMIT_S)
    archive_path = ((wsl_path((output_dir / SPLIT_ARRAYS_NAME).resolve())
                     if runtime == "wsl" and os.name == "nt"
                     else str((output_dir / SPLIT_ARRAYS_NAME).resolve())) if condensed else None)
    export_path.write_text("\n".join([
        f"P time_limit {int(cpu_time_limit_s)}",
        f"P memory_limit {int(memory_limit_mb)}",
        "P mpi_nbcpu 1",
        # run_aster overrides OMP_NUM_THREADS with the export's ncpus (default 1);
        # without this line the whole nonlinear solve factorizes single-threaded.
        f"P ncpus {SOLVER_THREADS}",
        f"F comm {case_path} D 1",
        f"F libr {mesh_path} D {MESH_UNIT}",
        *([f"F libr {archive_path} D 92"] if archive_path else []),
        "",
    ]), encoding="utf-8")
    if runtime == "wsl" and os.name == "nt":
        completed = run_wsl_solver(distribution, root, run_dir,
                                   [f"{root}/bin/run_aster", wsl_path(export_path)],
                                   time_limit_s, run_dir / "solver.live.log")
    else:
        completed = run_solver(native_command(installation["launcher_path"],
                                              [installation["launcher_path"], str(export_path.resolve())]),
                               run_dir, time_limit_s, live_log=run_dir / "solver.live.log")
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    failure = resource_failure(completed.stdout + completed.stderr)
    try:
        status = json.loads((run_dir / STATUS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        if failure is not None:
            raise CorroborationError(f"Code_Aster resource limit: {failure['message']}") from None
        tail = (completed.stdout + completed.stderr)[-1500:]
        raise CorroborationError(f"Code_Aster exited with code {completed.returncode} before the cohesive "
                                 f"solve started:\n{tail}") from None
    status["returncode"] = completed.returncode
    status["runtime"] = "wsl" if runtime == "wsl" and os.name == "nt" else "native"
    scratch = re.search(r"^LAMB_SOLVER_SCRATCH (.+)$", completed.stdout, re.MULTILINE)
    if scratch:
        status["solver_scratch"] = json.loads(scratch.group(1))
        status["solver_scratch"]["cleaned"] = "LAMB_SOLVER_SCRATCH_CLEANED " in completed.stdout
    # A comm-handled convergence stop writes status "stopped" and exits 0; Code_Aster still
    # prints the caught exception box, so the <EXCEPTION> scan must not reclassify it.
    # run_aster always prints its exit marker, even after a solver error. A run that was still
    # stepping and ends without it (and not through the time limit's 124/137) was killed from
    # outside, e.g. its WSL session was shut down; that says nothing about the model.
    interrupted = (status.get("status") == "running" and completed.returncode not in (0, 124, 137)
                   and RUN_ASTER_EXIT_MARKER not in completed.stdout)
    native_kill = re.search(
        r"(?m)^(?:\[\d+,\d+\]<(?:stdout|stderr)>:\s*)?\s*EXECUTION_CODE_ASTER_EXIT_\d+=137\s*$",
        completed.stdout + completed.stderr)
    native_exception = re.search(
        r"(?m)^(?:\[\d+,\d+\]<(?:stdout|stderr)>:\s*)?\s*(?:\u2551\s*)?<EXCEPTION>",
        completed.stdout + completed.stderr)
    if (completed.returncode != 0 or status.get("status") == "running"
            or (native_exception and status.get("status") != "stopped")):
        status["status"] = "failed"
        status["message"] = f"Code_Aster stopped unexpectedly (exit code {completed.returncode})"
        if interrupted:
            started = re.findall(r"ASTER_COHESIVE_STEP start load_factor=(\S+)", completed.stdout)
            status["interrupted"] = True
            status["message"] = (
                "Code_Aster was terminated from outside"
                + (f" while solving load factor {started[-1]}" if started else "")
                + f" (exit code {completed.returncode}): its log ends mid-solve with no solver error. "
                  "The process was killed or its WSL session was shut down or restarted (for example by "
                  "a WSL update); this is not a model result - rerun the stage")
    if native_kill is not None:
        started = re.findall(r"ASTER_COHESIVE_STEP start load_factor=(\S+)", completed.stdout)
        message = (
            "Code_Aster was killed by the operating system (SIGKILL, native exit code 137)"
            + (f" while solving load factor {started[-1]}" if started else "")
            + ". WSL memory exhaustion or an external kill can cause this; "
              "the exit code alone does not prove which. This is not nonlinear nonconvergence. "
              "Check the WSL kernel journal for an out-of-memory kill and reduce MPI memory replication.")
        status.update(status="failed", stop_reason="process_killed", message=message,
                      process_failure={"kind": "process_killed", "signal": 9, "message": message})
    if failure is not None:
        status.update(status="failed", stop_reason="resource_limit", resource_failure=failure,
                      message=failure["message"])
    if status.get("status") == "failed" and (status.get("preload") or {}).get("status") in ("pending", "running"):
        status["preload"].update(status="failed", message=status["message"], finished_at=time.time())
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


def open_face_traction_summary(areas, opened, peak_traction, mean_traction) -> dict:
    """Describe effective normal traction on open cohesive faces at one load state."""
    areas = np.asarray(areas, dtype=float)
    opened = np.asarray(opened, dtype=bool)
    peak_traction = np.asarray(peak_traction, dtype=float)
    if (areas.ndim != 1 or opened.shape != areas.shape or peak_traction.shape != areas.shape
            or not np.isfinite(areas).all() or np.any(areas <= 0)
            or not np.isfinite(peak_traction).all()):
        raise CorroborationError("Invalid open-face traction fields or facet areas")
    if mean_traction is not None:
        mean_traction = np.asarray(mean_traction, dtype=float)
        if mean_traction.shape != areas.shape or not np.isfinite(mean_traction).all():
            raise CorroborationError("Invalid mean open-face traction field")
    result = {
        "scope": "all candidate facets open beyond the elastic cohesive branch",
        "definition": ("Effective cohesive normal traction, tension positive; water pressure is an "
                       "external face load and is not included in this traction. The maximum is a "
                       "patch/Gauss-point peak; weighted statistics use positive facet-mean traction. "
                       "An opened facet is touched by opening, not necessarily open over its entire area."),
        "opened_faces": int(opened.sum()),
        "opened_area_m2": float(areas[opened].sum()),
        "maximum_tension_pa": None,
        "area_weighted_mean_tension_pa": None,
        "area_weighted_p95_tension_pa": None,
    }
    if not opened.any():
        return result
    result["maximum_tension_pa"] = float(max(0.0, peak_traction[opened].max()))
    if mean_traction is not None:
        values = np.maximum(mean_traction[opened], 0.0)
        weights = areas[opened]
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order])
        index = min(int(np.searchsorted(cumulative, 0.95 * cumulative[-1], side="left")), len(order) - 1)
        result["area_weighted_mean_tension_pa"] = float(np.average(values, weights=weights))
        result["area_weighted_p95_tension_pa"] = float(values[order[index]])
    return result


def joint_class_summary(mask, areas, initiated, opened, separated, traction, jump, strength_pa, law) -> dict:
    """Final state of one joint class (lift / damaged / intact).

    ``traction`` and ``jump`` are (steps, facets) peaks; the class peak is over the whole run.
    ``utilisation`` is the largest tension the class saw over its tensile strength: far
    below 1 with no initiated faces means the class never took load.
    """
    mask = np.asarray(mask, dtype=bool)
    peak = float(traction[:, mask].max())
    return {
        "law": law,
        "faces": int(mask.sum()),
        "area_m2": float(areas[mask].sum()),
        "initiated_faces": int(initiated[mask].sum()),
        "opened_faces": int(opened[mask].sum()),
        "separated_faces": int(separated[mask].sum()),
        "peak_normal_traction_pa": peak,
        "tensile_strength_pa": None if strength_pa is None else float(strength_pa),
        "utilisation": (None if not strength_pa else float(max(peak, 0.) / float(strength_pa))),
        "peak_opening_m": float(max(0., jump[:, mask].max())),
    }


def lift_plane_sliding(factors, areas, normal, shear, sliding, slip, jump, lift_law) -> dict:
    """Shear demand against Mohr-Coulomb capacity on the lift plane, per converged step.

    ``normal`` (tension positive) and ``shear`` (magnitude) are facet means of the joint
    tractions; the joint carries the effective stress because water pressure is applied
    on its two lips as external forces. For JOINT_MECA_FROT the facet capacity is
    max(c - mu * sigma_n, 0) and ``mobilisation`` = sum(|tau| A) / sum(capacity A): 1 means
    the whole plane is at its friction limit. The demand sums shear magnitudes, so it is an
    upper bound on the shear resultant when shear directions differ across the plane.
    JOINT_MECA_RUPT has no slip limit; the friction coefficient the plane would need to
    carry its shear by friction alone is reported instead.
    """
    areas = np.asarray(areas, dtype=float)
    total = float(areas.sum())
    frictional = lift_law.get("law") == "JOINT_MECA_FROT"
    mu, cohesion = (float(lift_law["friction_coefficient"]), float(lift_law["adhesion_pa"])) if frictional \
        else (None, None)
    rows = []
    for i, factor in enumerate(factors):
        compression = float((np.maximum(-normal[i], 0.) * areas).sum())
        demand = float((shear[i] * areas).sum())
        row = {
            "load_factor": float(factor),
            "normal_compression_n": compression,
            "shear_demand_n": demand,
            "tension_area_fraction": float(areas[normal[i] > 0].sum() / total),
            "open_area_fraction": float(areas[jump[i] > FRICTION_OPENING_LIMIT_M].sum() / total),
            "required_friction_coefficient": demand / compression if compression > 0 else None,
        }
        if frictional:
            capacity = float((np.maximum(cohesion - mu * normal[i], 0.) * areas).sum())
            row["shear_capacity_n"] = capacity
            row["mobilisation"] = demand / capacity if capacity > 0 else None
            row["sliding_area_fraction"] = (None if sliding is None
                                            else float(areas[sliding[i] > 0.5].sum() / total))
            row["peak_cumulative_slip_m"] = None if slip is None else float(slip[i].max())
        rows.append(row)
    return {
        "law": lift_law.get("law"),
        "friction_coefficient": mu,
        "adhesion_pa": cohesion,
        "area_m2": total,
        "definition": ("facet-mean joint tractions (effective stress); demand = sum of shear magnitudes x area "
                       "(an upper bound on the resultant); " +
                       ("capacity = sum of max(c - mu sigma_n, 0) x area; mobilisation 1 = the whole plane at "
                        "its friction limit" if frictional else
                        "the law has no slip limit, so only the friction coefficient the plane would need is given")),
        "steps": rows,
        "final": rows[-1] if rows else None,
    }


def unjointed_overstress(points, wall, stress, joints, strength_pa, *, residual_target_pa=None) -> dict:
    """Wall cells without any cohesive joint whose final principal tension exceeds the intact strength.

    A cohesive model cracks only along the joints it was given (Code_Aster U2.05.07 3.1).
    Tension above the strength in cells that have no joint on any face means the crack path
    the structure wants lies outside the modelled one, and the result does not predict it.
    """
    wall = np.asarray(wall, dtype=np.int64)
    scale = int(len(points)) + 1

    def keys(triples):
        ordered = np.sort(np.asarray(triples, dtype=np.int64).reshape(-1, 3), axis=1)
        return (ordered[:, 0] * scale + ordered[:, 1]) * scale + ordered[:, 2]

    joints = np.asarray(joints, dtype=np.int64).reshape(-1, 6)
    lips = np.unique(np.concatenate((keys(joints[:, :3]), keys(joints[:, 3:])))) if len(joints) else np.empty(0, np.int64)
    faces = keys(wall[:, _TET_FACES])
    jointed = np.isin(faces, lips).reshape(len(wall), 4).any(axis=1)
    principal = np.linalg.eigvalsh(0.5 * (stress + np.transpose(stress, (0, 2, 1))))[:, 2]
    threshold = float(strength_pa if residual_target_pa is None else residual_target_pa)
    criterion = "intact tensile strength" if residual_target_pa is None else "residual tensile-stress target"
    over = ~jointed & (principal > threshold)
    corners = points[wall]
    volumes = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6.
    report = {
        "tensile_strength_pa": float(strength_pa),
        "threshold_pa": threshold,
        "jointed_wall_cells": int(jointed.sum()),
        "unjointed_wall_cells": int((~jointed).sum()),
        "cells": int(over.sum()),
        "volume_m3": float(volumes[over].sum()),
        "peak_principal_stress_pa": None,
        "peak_centroid_xyz_m": None,
        "wall_cell_indices": np.flatnonzero(over).tolist(),
    }
    if (~jointed).any():
        candidates = np.flatnonzero(~jointed)
        peak = int(candidates[np.argmax(principal[candidates])])
        report["peak_principal_stress_pa"] = float(principal[peak])
        report["peak_centroid_xyz_m"] = corners[peak].mean(axis=0).tolist()
    report["note"] = (
        f"{report['cells']} wall cells with no cohesive joint exceed the {criterion} at the final "
        "state. They cannot crack in this model, so cracking outside the selected band is not predicted."
        if report["cells"] else
        f"No wall cell without a cohesive joint exceeds the {criterion} at the final state.")
    return report


def displacement_attribution(first_u, final_u, intact_u, wall_nodes, factors, first_initiated: bool,
                             proportional: bool, softening_sources: list[str]) -> dict:
    """Split the displacement change into elastic softening and cracking.

    While no joint has initiated the damaged model is linear, so its first step scaled to
    the final load factor is the "softened but uncracked" response: the same softened
    supports, materials and joint stiffness with no cracking. The difference to the final
    state is what cracking added; the difference to the intact baseline is what the softened
    inputs alone did. The split needs proportional loading (not the friction runs, whose
    self-weight is applied first).
    """
    def peak(field):
        return float(np.linalg.norm(field[wall_nodes], axis=1).max())

    report = {
        "final_peak_m": peak(final_u),
        "intact_peak_m": None if intact_u is None else peak(intact_u),
        "softened_uncracked_peak_m": None,
        "elastic_softening_change_m": None,
        "cracking_change_m": None,
        "softening_sources": list(softening_sources),
    }
    if proportional and not first_initiated:
        softened = first_u * (factors[-1] / factors[0])
        report["softened_uncracked_peak_m"] = peak(softened)
        report["cracking_change_m"] = peak(final_u - softened)
        if intact_u is not None:
            report["elastic_softening_change_m"] = peak(softened - intact_u)
        report["basis"] = (f"first converged step (load factor {factors[0]:g}, no joint initiated) scaled "
                           f"linearly to {factors[-1]:g}; changes are the largest wall-node difference between fields")
    elif intact_u is not None:
        report["total_change_m"] = peak(final_u - intact_u)
        report["basis"] = (
            ("self-weight is applied before the water in friction runs, so the first step cannot be scaled; "
             if not proportional else "a joint had already initiated at the first step; ")
            + "only the total change from the intact baseline is available"
            + ("" if softening_sources else
               " (no elastic softening is configured, so that total is the effect of the joints)"))
    return report


def assessment_notes(report: dict) -> list[str]:
    """Plain-language findings for the results view: what each joint class did, the sliding
    margin of the lift plane, where the displacement change came from, and where the model
    could not crack. Each note is one self-contained sentence or two."""
    notes = []
    names = {"lift": "Lift joints", "damaged": "Joints between degraded elements", "intact": "Intact-concrete joints"}
    for key, label in names.items():
        item = (report.get("joint_classes") or {}).get(key)
        if item is None:
            continue
        strength, utilisation = item["tensile_strength_pa"], item["utilisation"]
        text = (f"{label} ({item['law']}, {item['faces']} faces, {item['area_m2']:.4g} m2): "
                f"{item['initiated_faces']} initiated, {item['opened_faces']} opened, "
                f"{item['separated_faces']} fully separated; largest opening {item['peak_opening_m'] * 1e3:.3g} mm")
        if strength:
            text += (f"; highest tension {max(item['peak_normal_traction_pa'], 0.) / 1e3:.4g} kPa, "
                     f"{utilisation:.1%} of the {strength / 1e3:.4g} kPa strength")
            if item["initiated_faces"] == 0 and utilisation < 0.5:
                text += ". These joints never approached their strength: they were not loaded in tension"
        notes.append(text + ".")
    sliding = (report.get("lift_plane_sliding") or {}).get("final")
    if sliding is not None:
        law = report["lift_plane_sliding"]["law"]
        if sliding.get("mobilisation") is not None:
            text = (f"Lift plane sliding ({law}, friction coefficient {report['lift_plane_sliding']['friction_coefficient']:g}): "
                    f"shear demand {sliding['shear_demand_n'] / 1e6:.4g} MN against a friction capacity of "
                    f"{sliding['shear_capacity_n'] / 1e6:.4g} MN, mobilisation {sliding['mobilisation']:.2f} "
                    "(1 means the whole plane is at its friction limit)")
            if sliding.get("sliding_area_fraction") is not None:
                text += (f"; {sliding['sliding_area_fraction']:.0%} of the plane is sliding and "
                         f"{sliding['open_area_fraction']:.0%} is open")
            if sliding.get("peak_cumulative_slip_m") is not None:
                text += f"; largest slip {sliding['peak_cumulative_slip_m'] * 1e3:.3g} mm"
            notes.append(text + ".")
        elif law == "JOINT_MECA_FROT":
            notes.append(f"Lift plane sliding ({law}): the plane has no friction capacity left at the final state "
                         f"({sliding['open_area_fraction']:.0%} of it is open), so it carries no shear.")
        elif sliding.get("required_friction_coefficient") is not None:
            notes.append(f"Lift plane shear ({law} has no slip limit): the plane carries {sliding['shear_demand_n'] / 1e6:.4g} MN "
                         f"of shear under {sliding['normal_compression_n'] / 1e6:.4g} MN of compression, which friction "
                         f"alone could carry only with a coefficient of {sliding['required_friction_coefficient']:.2f}.")
    attribution = report.get("displacement_attribution")
    if attribution is not None:
        final, intact = attribution["final_peak_m"], attribution["intact_peak_m"]
        text = f"Peak wall displacement {final * 1e3:.4g} mm"
        if intact is not None:
            text += f" (intact dam at the same load: {intact * 1e3:.4g} mm)"
        if attribution.get("cracking_change_m") is not None:
            text += (f". Cracking changed the displacement field by at most {attribution['cracking_change_m'] * 1e3:.3g} mm")
            if attribution.get("elastic_softening_change_m") is not None:
                text += (f"; the softened inputs alone, before any cracking, changed it by "
                         f"{attribution['elastic_softening_change_m'] * 1e3:.3g} mm")
        elif attribution.get("total_change_m") is not None:
            text += (f". The largest change from the intact dam is {attribution['total_change_m'] * 1e3:.3g} mm; "
                     "it could not be split into softening and cracking for this run")
        if attribution["softening_sources"]:
            text += ". Elastic softening configured: " + "; ".join(attribution["softening_sources"])
        notes.append(text + ".")
    overstress = report.get("unjointed_overstress")
    if overstress is not None and overstress["cells"]:
        notes.append(
            f"{overstress['cells']} wall cells outside the jointed band ({overstress['volume_m3']:.4g} m3) exceed the "
            f"intact tensile strength of {overstress['tensile_strength_pa'] / 1e6:.4g} MPa at the final state, up to "
            f"{overstress['peak_principal_stress_pa'] / 1e6:.4g} MPa. They have no cohesive joint and cannot crack in "
            "this model, so cracking there is not predicted and the result is not a crack-path prediction.")
    return notes


def postprocess(output_dir: Path, fenicsx_output: Path, status: dict) -> dict:
    inputs = json.loads((output_dir / "case_inputs.json").read_text(encoding="utf-8"))
    joint = inputs["joint"]
    loading = joint.get("reservoir_loading")
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
        "resource_failure": status.get("resource_failure"),
        "process_failure": status.get("process_failure"),
        "preload": (dict(status["preload"], result_directory=str((run_dir / "preload").resolve()))
                    if status.get("preload") is not None else None),
        "solver_scratch": status.get("solver_scratch"),
        "differences_from_fenicsx": list(DIFFERENCES),
        "joint_integration": inputs.get("joint_integration", "PENTA6_FPG1"),
        "native_joint_cells": inputs.get("native_joint_cells", inputs["candidate_faces"]),
        "tangent_update_interval": inputs.get("tangent_update_interval", 1),
    }
    if loading is not None:
        report.update(
            loading_mode="reservoir_filling", reservoir_loading=loading,
            load_factor_definition="normalised_water_depth_not_force_multiplier",
            reached_water_depth_m=status.get("reached_water_depth_m"),
            water_surface_z_m=status.get("water_surface_z_m"),
            activation_history_complete=status.get("activation_history_complete"),
            activation_warnings=status.get("activation_warnings", []))
    if joint.get("continuation") is not None:
        report["continuation_rounds"] = [
            dict(row, result_directory=str((run_dir / f"round_{row['iteration']:04d}").resolve()))
            for row in status.get("continuation_rounds", [])]
        report["continuation_stop_reason"] = status.get("continuation_stop_reason")
        if status.get("failed_continuation_instant") is not None:
            report["failed_continuation_instant"] = status["failed_continuation_instant"]
    if "step_timings" in status:
        report["step_timings"] = status["step_timings"]
    if "hydraulic_loads" in joint:
        report["hydraulic_loads"] = joint["hydraulic_loads"]
        report["hydraulic_steps"] = status.get("hydraulic_steps", [])
        report["differences_from_fenicsx"] = [
            note for note in report["differences_from_fenicsx"]
            if "water" not in note.lower() and "uplift" not in note.lower()]
        uplift = joint["hydraulic_loads"]["settings"]["base_uplift"] != "none"
        plane_uplift = (joint["hydraulic_loads"].get("lift_plane_uplift") or {}).get("mode", "none") != "none"
        if plane_uplift:
            report["lift_plane_uplift"] = status.get("lift_plane_uplift")
        report["differences_from_fenicsx"].append(
            "Aster applies reservoir-connected opened crack-face pressure"
            + (" and undrained heel-to-toe uplift." if uplift else "; base uplift is disabled.")
            + (" The lift plane's closed faces carry a pore pressure linear from the reservoir head upstream "
               "to the tailwater head downstream." if plane_uplift else "")
            + " Small-strain hydrostatic active-set pressure assumes instantaneous reservoir head, "
            "not coupled flow/permeability or transient drainage; supporting loads may differ.")
    if not steps_path.is_file():
        report.update(status="failed", steps=[])
        return report
    with np.load(steps_path) as steps:
        factors = np.asarray(steps["load_factor"], dtype=float)
        instants = np.asarray(steps["solver_instant"], dtype=float) if joint.get("continuation") is not None else factors
        depths = np.asarray(steps["water_depth_m"], dtype=float) if loading is not None else None
        if len(factors) == 0:
            report.update(status="failed", steps=[])
            return report
        native_faces = report["native_joint_cells"]
        damage = np.asarray(steps["damage_state"], dtype=float).reshape(len(factors), native_faces)
        jump = np.asarray(steps["normal_jump_m"], dtype=float).reshape(len(factors), native_faces)
        traction = np.asarray(steps["normal_traction_pa"], dtype=float).reshape(len(factors), native_faces)
        # Gauss-point means and friction state (absent in runs made before they were recorded).
        extras = {name: (np.asarray(steps[name], dtype=float).reshape(len(factors), native_faces)
                         if name in steps.files else None)
                  for name in ("mean_normal_traction_pa", "mean_tangential_traction_pa", "sliding",
                               "cumulative_slip_m")}
    if (jump.shape != damage.shape or traction.shape != damage.shape
            or not all(np.isfinite(values).all() for values in (factors, damage, jump, traction))
            or any(values is not None and not np.isfinite(values).all() for values in extras.values())
            or instants.shape != factors.shape or not np.isfinite(instants).all()
            or np.any(np.diff(instants) <= 0)
            or (joint.get("continuation") is not None and loading is None
                and not np.array_equal(factors, np.minimum(instants, 1.0)))):
        raise CorroborationError("Aster converged step fields are non-finite, unordered or inconsistent")
    if loading is not None and (
            depths.shape != factors.shape or not np.isfinite(depths).all()
            or np.any(np.diff(depths) < 0) or np.any(depths < 0)
            or np.any(depths > loading["final_water_depth_m"])
            or not np.allclose(factors, depths / loading["final_water_depth_m"], rtol=0., atol=1e-12)
            or status.get("reached_water_depth_m") != float(depths[-1])):
        raise CorroborationError("Reservoir steps have inconsistent accepted physical water depths")
    if inputs.get("joint_integration") == "three_HEXA8_FPG4_patches_exact_P1_trace":
        with np.load(output_dir / SPLIT_ARRAYS_NAME, allow_pickle=False) as stored:
            mapping = np.asarray(stored["joint_parent_faces"], dtype=np.int64)
        if (mapping.shape != (damage.shape[1],) or not np.array_equal(
                mapping, np.repeat(np.arange(len(areas)), 3))):
            raise CorroborationError("Aster quadrilateral patches do not match the original candidate facets")
        # Report peaks over each original candidate facet, not invented extra crack faces.
        damage, jump, traction = (values.reshape(len(factors), len(areas), 3).max(axis=2)
                                  for values in (damage, jump, traction))
        # The three patches of a facet have equal area, so facet means are patch means.
        for name, reduce in (("mean_normal_traction_pa", np.mean), ("mean_tangential_traction_pa", np.mean),
                             ("sliding", np.max), ("cumulative_slip_m", np.max)):
            if extras[name] is not None:
                extras[name] = reduce(extras[name].reshape(len(factors), len(areas), 3), axis=2)
        report["facet_state_definition"] = (
            "Peak over three FPG4 quadrilateral patches per original facet; initiated/opened areas "
            "are the areas of candidate facets touched, not fully separated surface areas.")
    if damage.shape[1] != len(areas):
        raise CorroborationError("Aster joint results do not match the FEniCSx candidate facets")
    initiated = damage >= 1.0 - 1e-9
    opening_limit = joint["elastic_limit_opening_m"]
    lift_law = joint.get("lift_joint")
    lift_faces = damaged_faces = None
    if inputs.get("material_assignment") is not None:
        with np.load(output_dir / SPLIT_ARRAYS_NAME) as arrays:
            damaged_faces = np.asarray(arrays["damaged_joint_faces"], dtype=bool)
            lift_faces = (np.asarray(arrays["lift_joint_faces"], dtype=bool) if "lift_joint_faces" in arrays.files
                          else np.zeros_like(damaged_faces))
        friction = joint.get("damaged_friction") is not None
        opening_limit = np.where(
            lift_faces, lift_law["elastic_limit_opening_m"] if lift_law is not None else 0.,
            np.where(damaged_faces, FRICTION_OPENING_LIMIT_M if friction else opening_limit,
                     joint["intact_joint"]["elastic_limit_opening_m"]))
        if lift_law is not None:
            degraded = inputs["material_assignment"].get("properties") is not None
            report["lift_joints"] = {
                **lift_law,
                "lift_joint_faces": int(lift_faces.sum()),
                "lift_joint_area_m2": float(areas[lift_faces].sum()),
                "elements": ("the selected band's elements carry the degraded material; the other facets "
                             "between them take its cohesive law" if degraded else
                             "intact concrete everywhere; only the lift-plane joint law is weakened"),
                "limitation": (
                    "JOINT_MECA_FROT is perfectly plastic: the plane can slide and open, but its tensile "
                    "capacity (ADHESION / MU) and adhesion are not lost afterwards, so give a poorly bonded "
                    "surface a small strength (R7.01.25 1.2.2)"
                    if lift_law["law"] == "JOINT_MECA_FROT" else
                    "JOINT_MECA_RUPT keeps an opening-dependent elastic shear stiffness after "
                    "rupture and has no Coulomb slip limit, so sliding on the open joint is "
                    "not predicted (use the JOINT_MECA_FROT law for sliding)"),
            }
        report["material_assignment"] = inputs["material_assignment"]
    opened = initiated & (jump > opening_limit * (1.0 + 1e-6))
    separated = damage >= 2.0 - 1e-9
    if damaged_faces is not None and len(areas):
        # Which class of joint did what: a weak plane that never approaches its strength was
        # bypassed, however many other facets initiated.
        strengths = {"lift": lift_law["tensile_strength_pa"] if lift_law is not None else None,
                     "damaged": (None if joint.get("damaged_friction") is not None
                                 else joint["tensile_strength_pa"]),
                     "intact": joint["intact_joint"]["tensile_strength_pa"]}
        laws = {"lift": lift_law["law"] if lift_law is not None else None,
                "damaged": "JOINT_MECA_FROT" if joint.get("damaged_friction") is not None else "CZM_LIN_REG",
                "intact": "CZM_LIN_REG"}
        report["joint_classes"] = {
            name: joint_class_summary(mask, areas, initiated[-1], opened[-1], separated[-1], traction, jump,
                                      strengths[name], laws[name])
            for name, mask in (("lift", lift_faces), ("damaged", damaged_faces),
                               ("intact", ~lift_faces & ~damaged_faces)) if mask.any()}
        if lift_law is not None and lift_faces.any() and extras["mean_tangential_traction_pa"] is not None:
            report["lift_plane_sliding"] = lift_plane_sliding(
                factors, areas[lift_faces], extras["mean_normal_traction_pa"][:, lift_faces],
                extras["mean_tangential_traction_pa"][:, lift_faces],
                None if extras["sliding"] is None else extras["sliding"][:, lift_faces],
                None if extras["cumulative_slip_m"] is None else extras["cumulative_slip_m"][:, lift_faces],
                jump[:, lift_faces], lift_law)
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
    if joint.get("continuation") is not None:
        for row, instant in zip(step_rows, instants):
            row["solver_instant"] = float(instant)
    if loading is not None:
        for row, depth in zip(step_rows, depths):
            row.update(water_depth_m=float(depth),
                       water_surface_z_m=loading["depth_reference_z_m"] + float(depth))
    first = next((i for i in range(len(factors)) if initiated[i].any()), None)
    bracket = None if first is None else [float(factors[first - 1]) if first else 0.0, float(factors[first])]
    linear_estimate = None
    if loading is None and len(areas) and not initiated[0].any() and traction[0].max() > 0:
        strengths = np.full(len(areas), joint["tensile_strength_pa"], dtype=float)
        if damaged_faces is not None:
            strengths[~damaged_faces & ~lift_faces] = joint["intact_joint"]["tensile_strength_pa"]
            if lift_law is not None:
                strengths[lift_faces] = lift_law["tensile_strength_pa"]
        positive = traction[0] > 0
        linear_estimate = float(np.min(factors[0] * strengths[positive] / traction[0, positive]))

    def state_at(target: float) -> int | None:
        matches = np.flatnonzero(np.isclose(factors, target, atol=1e-9, rtol=0.))
        return int(matches[-1] if joint.get("continuation") is not None else matches[0]) if len(matches) else None

    design = state_at(1.0)
    final = len(factors) - 1
    complete = status.get("status") == "completed" and (
        np.isclose(factors[-1], inputs["load_factors"][-1], rtol=0., atol=1e-9)
        or (status.get("overload_stopped_on_opening") is True and factors[-1] >= 1.0))
    if loading is not None:
        complete = (complete and depths[-1] == loading["final_water_depth_m"]
                    and status.get("continuation_stop_reason") != "iteration_limit")
        report["first_initiation_water_depth_bracket_m"] = (
            None if first is None else [float(depths[first - 1]) if first else loading["water_depth_start_m"],
                                       float(depths[first])])
    report.update(
        status="completed" if complete else "partial",
        reached_load_factor=float(factors[-1]),
        steps=step_rows,
        first_initiation_bracket=bracket,
        linear_first_initiation_estimate=linear_estimate,
        design_load=None if design is None else step_rows[design],
    )
    if (not complete and status.get("status") == "stopped"
            and status.get("stopped_at_load_factor") is not None):
        # Newton stopped converging. That may be the structure's capacity, or the solver
        # failing to step past a local softening event: Code_Aster's cohesive-zone notice
        # (U2.05.07 4.3) calls load control essential for these laws and this solve has none.
        # Without an independent force balance the stop is reported, not called a failure.
        reached, target = float(factors[-1]), float(status["stopped_at_load_factor"])
        last = step_rows[-1]
        report["outcome"] = "capacity_limit"
        report["capacity_assessment"] = {
            "reached_load_factor": reached,
            "failed_load_factor": target,
            "last_converged_state": last,
            "established": False,
            "basis": "Newton non-convergence after time-step subdivision; no load control, no independent force balance",
            "message": (
                f"Code_Aster converged every load step up to load factor {reached:g} and then did not converge "
                f"between {reached:g} and {target:g} despite repeated time-step subdivision. This is not, by "
                "itself, evidence that the dam fails there: softening joint laws can stop a Newton solve at a "
                "local snap-back that the structure would pass, and this solve has no load control to step "
                f"through one. At the last converged state {last['initiated_faces']} joints had initiated, "
                f"{last['separated_faces']} had fully separated and the largest opening was "
                f"{last['peak_opening_m'] * 1e3:.3g} mm. Treat {reached:g} as the highest load this model "
                "reached; a capacity claim needs the sliding assessment or a force balance on the failing "
                "surface to support it. The last converged state is archived and viewable."),
        }
    if status.get("failed_continuation_instant") is not None and report.get("capacity_assessment") is not None:
        report["capacity_assessment"]["message"] = (
            "Code_Aster reached the full design load, but could not re-establish equilibrium "
            "after activating additional wall facets. Physical loads remained at factor 1.0; "
            f"the failed solver instant {status['failed_continuation_instant']:g} is a continuation "
            "label, not an increased reservoir level or physical time. The previous accepted "
            "round and its fracture history are preserved. This is a solver failure, not "
            "evidence of dam capacity or achievement of the residual tensile-stress target.")
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
    report["open_face_traction"] = open_face_traction_summary(
        areas, opened[-1], traction[-1],
        None if extras["mean_normal_traction_pa"] is None else extras["mean_normal_traction_pa"][-1])
    displacement_path = run_dir / STEPS_NAME.replace(".npz", "_displacement.npz")
    if displacement_path.is_file():
        with np.load(output_dir / SPLIT_ARRAYS_NAME, allow_pickle=True) as arrays:
            nodes = np.asarray(arrays["nodes"], dtype=float)
            wall_mask = np.asarray(arrays["concrete_body_ids"]) == 1
            wall = np.asarray(arrays["tetra4_concrete"])[wall_mask]
            source_ids = arrays["source_node_ids"].copy()
            damaged_wall = (arrays["damaged_cells"][:len(wall_mask)][wall_mask]
                            if "damaged_cells" in arrays else np.zeros(len(wall), dtype=bool))
            joint_lips = (np.asarray(arrays["penta6_joints"], dtype=np.int64) if "penta6_joints" in arrays
                          else np.empty((0, 6), dtype=np.int64))
            if joint.get("continuation") is not None:
                active_faces = arrays["continuation_active_faces"].astype(bool)
                rounds = report["continuation_rounds"]
                if rounds:
                    with np.load(run_dir / f"round_{rounds[-1]['iteration']:04d}" / "accepted_state.npz") as checkpoint:
                        active_faces = checkpoint["active_faces"].astype(bool)
                report["active_candidate_faces"] = int(active_faces.sum())
                joint_lips = joint_lips[active_faces]
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
        if assignment is not None and "properties" in assignment:
            # Degraded elements only; lift joints leave every element intact concrete.
            properties = assignment["properties"]
            final_stress[damaged_wall] = wall_cell_stress(
                nodes, wall[damaged_wall], final_u, properties["youngs_modulus"], properties["poisson_ratio"])
        cracked = _stress_summary(nodes, wall, final_stress, final_u)
        target = residual_target_tensile_stress_pa(summary) if "residual_target_tensile_stress_pa" in summary else float(
            (summary.get("solver_controls") or {}).get("whole_wall_tension_tolerance_pa", 1.0))
        if joint.get("continuation") is not None:
            target = float(joint["continuation"]["target_pa"])
        wall_principal = np.linalg.eigvalsh(final_stress)[:, -1]
        report["residual_tension_assessment"] = {
            "threshold_pa": target,
            "maximum_principal_tensile_stress_pa": float(wall_principal.max()),
            "peak_centroid_xyz_m": cracked["tensile_peak_centroid_xyz_m"],
            "wall_cells_above_threshold": int(np.count_nonzero(wall_principal > target)),
            "scope": "all wall P1 cells at the last accepted mechanical load factor",
        }
        comparison = {"load_factor": float(factors[-1]), "scope": "wall (body 1) cells, constant P1 stress",
                      "cracked": cracked, "intact": None}
        intact_u = None
        has_reference = assignment is not None or (loading is not None and "baseline_displacement_path" in inputs)
        if has_reference and (loading is None or depths[-1] == loading["final_water_depth_m"]):
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
                intact_u[physical] = baseline["displacement_m"][valid][first][positions] * (
                    factors[-1] if loading is None else 1.)
                if not np.allclose(baseline["coordinates_m"][valid][first][positions], nodes[physical], atol=1e-9, rtol=0.):
                    raise CorroborationError("Intact baseline coordinates do not match selected-material result")
            comparison["intact"] = _stress_summary(nodes, wall, wall_cell_stress(nodes, wall, intact_u, e_pa, nu), intact_u)
            comparison["intact_definition"] = (
                ("original concrete full-depth Aster baseline; differences include "
                 if loading is not None else
                 "original concrete Aster baseline scaled to final load factor; differences include ")
                + ("selected material changes and cracking" if assignment is not None and "properties" in assignment
                   else "cohesive cracking (every volume element keeps the intact concrete)"))
            np.savez(displacement_path, coordinates=coordinates, first=first_u, final=final_u, intact=intact_u)
        elif loading is None and not initiated[0].any():
            # Before initiation the joints are linear, so the bonded state scales with the load.
            scale = factors[-1] / factors[0]
            intact_u = first_u * scale
            comparison["intact"] = _stress_summary(nodes, wall, wall_cell_stress(nodes, wall, intact_u, e_pa, nu),
                                                   intact_u)
            comparison["intact_definition"] = (
                f"bonded joints, same mesh and loads: the elastic load-factor {factors[0]:g} state scaled linearly")
        elif loading is not None:
            comparison["intact_definition"] = (
                "No same-depth intact reference is available; reservoir filling is not proportional loading.")
        report["stress_comparison"] = comparison
        # Where the model cannot crack although the stress says it would.
        intact_strength = (joint["intact_joint"]["tensile_strength_pa"] if "intact_joint" in joint
                           else joint["tensile_strength_pa"])
        report["unjointed_overstress"] = unjointed_overstress(
            nodes, wall, final_stress, joint_lips, intact_strength)
        report["unjointed_residual_tension"] = unjointed_overstress(
            nodes, wall, final_stress, joint_lips, intact_strength, residual_target_pa=target)
        with np.load(output_dir / SPLIT_ARRAYS_NAME, allow_pickle=False) as stored:
            source_rows = np.asarray(stored["source_cell_rows"], dtype=np.int64)
        growth_assessment = (report["unjointed_residual_tension"]
                             if "residual_target_tensile_stress_pa" in summary or joint.get("continuation") is not None
                             else report["unjointed_overstress"])
        indices = growth_assessment.get("wall_cell_indices", [])
        report["growth_source_rows"] = source_rows[np.asarray(indices, dtype=np.int64)].tolist()
        # How much of the displacement change is elastic softening and how much is cracking.
        sources = []
        degradation = inputs.get("support_degradation") or {}
        if isinstance(degradation.get("interface_scale"), (int, float)) and degradation["interface_scale"] < 1:
            sources.append(f"wall/foundation seat springs scaled by {degradation['interface_scale']:g}")
        if (assignment is not None and assignment.get("properties") is not None
                and float(assignment["properties"]["youngs_modulus"]) < float(e_pa)):
            sources.append(f"selected elements' Young's modulus {float(assignment['properties']['youngs_modulus']):.4g} Pa "
                           f"(intact {float(e_pa):.4g} Pa)")
        if lift_law is not None and lift_law["k_n_pa_per_m"] < lift_law["mesh_penalty_pa_per_m"]:
            sources.append(f"lift-joint normal stiffness {lift_law['k_n_pa_per_m']:.4g} Pa/m (as compliant as "
                           f"{float(e_pa) / lift_law['k_n_pa_per_m']:.3g} m of concrete)")
        proportional = loading is None and joint.get("damaged_friction") is None and not (
            lift_law is not None and lift_law.get("law") == "JOINT_MECA_FROT")
        report["displacement_attribution"] = displacement_attribution(
            first_u, final_u, intact_u, np.unique(wall), factors,
            bool(initiated[0].any()), proportional, sources)
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
        from Aster.export_paraview import export_joint_series, export_mesh
        # Preserve all bodies in the view, while assessments use only wall nodes.
        with np.load(output_dir / SPLIT_ARRAYS_NAME, allow_pickle=True) as stored:
            export_arrays = {key: stored[key] for key in stored.files}
        # Final-state stress for every concrete (wall + foundation) cell, with the
        # damaged cells' assigned material, so ParaView shows post-cracking stresses.
        all_concrete = np.asarray(export_arrays["tetra4_concrete"], dtype=np.int64)
        all_stress = wall_cell_stress(nodes, all_concrete, final_u, e_pa, nu)
        if assignment is not None and "properties" in assignment and "damaged_cells" in export_arrays:
            damaged_all = np.asarray(export_arrays["damaged_cells"], dtype=bool)[:len(all_concrete)]
            properties = assignment["properties"]
            all_stress[damaged_all] = wall_cell_stress(
                nodes, all_concrete[damaged_all], final_u,
                properties["youngs_modulus"], properties["poisson_ratio"])
        principals = np.linalg.eigvalsh(0.5 * (all_stress + np.transpose(all_stress, (0, 2, 1))))
        deviator = all_stress - np.trace(all_stress, axis1=1, axis2=2)[:, None, None] / 3.0 * np.eye(3)
        concrete_cell_stress = {
            "max_principal_stress_pa": principals[:, 2],
            "min_principal_stress_pa": principals[:, 0],
            "von_mises_stress_pa": np.sqrt(1.5 * np.einsum("cij,cij->c", deviator, deviator)),
        }
        paraview = export_mesh(export_arrays, final_u, output_dir / "paraview", damage[-1], jump[-1],
                               concrete_cell_stress=concrete_cell_stress)
        report["paraview"] = paraview
        report["paraview"]["joint_history"] = export_joint_series(
            export_arrays, instants if loading is not None else factors, damage, jump,
            output_dir / "paraview")
        report["paraview"]["stress_arrays"] = {
            "names": sorted(concrete_cell_stress),
            "scope": "concrete (wall + foundation) cells; rock cells are NaN",
            "measure": "final-state constant P1 cell stress from the solved displacement, "
                       "using each cell's assigned material (damaged or intact)",
        }

    report["assessment_notes"] = assessment_notes(report)
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
    if wet > 0 and "hydraulic_loads" not in joint:
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


def run(baseline_dir: Path, fenicsx_output: Path, output_dir: Path, time_limit_s: int | None) -> dict:
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
    if (inputs["joint"].get("hydraulic_loads", {}).get("settings", {}).get("base_uplift") == "none"
            and report.get("status") == "completed"):
        from Aster.hydraulic_loads import validate_crack_water_evidence
        validate_crack_water_evidence(report)
    report["runtime_s"] = time.time() - started
    inputs["case_sha256"] = file_hash(output_dir / CASE_NAME)
    (output_dir / "case_inputs.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    report["input_hashes"] = {key: value for key, value in inputs.items() if key.endswith("_sha256")}
    report["accepted"] = (report.get("status") in ("completed",)
                          or (report.get("status") == "partial"
                              and report.get("outcome") == "capacity_limit")) and (
        status.get("returncode") == 0
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
