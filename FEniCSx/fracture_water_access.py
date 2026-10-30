"""Geometric water-access eligibility on explicit paired crack surfaces.

This is not saturation, a pressure prescription or a transient wetting solver.
Only explicitly supplied exterior inlet edges seed reachability. Aperture is
the measured normal displacement jump; closed paths receive no artificial gap.
"""
import numbers

import numpy as np

if __package__:
    from .fracture_faces import PairedCrackFaces
else:
    from fracture_faces import PairedCrackFaces


def water_access(faces: PairedCrackFaces, displacement_m, inlet_source_edges,
                 parent_node_ids, *, closure_tolerance_m: float) -> dict:
    """Report reachable open faces, without assigning water to reachable faces.

    An inlet is an exterior edge of this surface inventory, selected from
    verified hydraulic boundary geometry, NOT a painted crack-start marker.
    Edge adjacency uses both lip identities, invariant under owner reversal.
    Source parents are used only for matching explicit boundary inlet IDs.
    """
    if not isinstance(faces, PairedCrackFaces):
        raise ValueError("Explicit PairedCrackFaces are required for water access")
    if (isinstance(closure_tolerance_m, bool)
            or not isinstance(closure_tolerance_m, numbers.Real)
            or not np.isfinite(closure_tolerance_m) or closure_tolerance_m < 0):
        raise ValueError("closure_tolerance_m must be explicitly supplied, finite and nonnegative")
    parents = np.asarray(parent_node_ids)
    if (parents.dtype.kind not in "iu" or parents.shape != (len(faces.coordinates),)
            or np.any(parents < 0)):
        raise ValueError("Parent node IDs must match the paired mesh and be nonnegative integers")
    if not np.array_equal(parents[faces.minus], parents[faces.plus]):
        raise ValueError("Corresponding lips must retain identical original parent node IDs")
    inlets = np.asarray(inlet_source_edges)
    if (inlets.dtype.kind not in "iu" or inlets.ndim != 2 or inlets.shape[1] != 2
            or not len(inlets) or np.any(inlets < 0)
            or np.any(inlets[:, 0] == inlets[:, 1])
            or len(np.unique(np.sort(inlets, axis=1), axis=0)) != len(inlets)):
        raise ValueError("Hydraulic inlet source edges must be explicit, unique source-node pairs")
    source_triangles = parents[faces.minus]
    if any(len(set(row)) != 3 for row in source_triangles):
        raise ValueError("Parent node IDs collapse a crack triangle")
    measurement = faces.measure(displacement_m)
    gaps = np.asarray(measurement["signed_vertex_gaps_m"])
    if np.min(gaps) < -closure_tolerance_m:
        raise ValueError("Penetrating crack faces require a converged contact solution before water-access checks")
    open_faces = np.max(gaps, axis=1) > closure_tolerance_m
    edges = {}
    source_to_edges = {}
    for face_id, (minus, plus, source) in enumerate(zip(faces.minus, faces.plus, source_triangles)):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            # A different pair of coincident lips is not the same hydraulic edge.
            key = tuple(sorted((tuple(sorted((int(minus[k]), int(plus[k])))) for k in (i, j))))
            source_key = tuple(sorted((int(source[i]), int(source[j]))))
            entry = (face_id, bool(max(gaps[face_id, i], gaps[face_id, j]) > closure_tolerance_m))
            edges.setdefault(key, []).append(entry)
            source_to_edges.setdefault(source_key, set()).add(key)
    adjacency = [set() for _ in faces.minus]
    for owners in edges.values():
        passing = [face for face, is_open in owners if is_open]
        for face in passing:
            adjacency[face].update(other for other in passing if other != face)
    seeds = set()
    blocked = []
    for inlet in inlets:
        source_key = tuple(sorted(map(int, inlet)))
        matched = source_to_edges.get(source_key, set())
        if not matched:
            raise ValueError(f"Hydraulic inlet edge {source_key} is absent from the crack inventory")
        # A coincident unrelated sheet must not make an inlet ambiguous.
        if len(matched) != 1:
            raise ValueError(f"Hydraulic inlet edge {source_key} matches multiple split sheets")
        owners = edges[next(iter(matched))]
        if len(owners) != 1:
            raise ValueError(f"Hydraulic inlet edge {source_key} must be an exterior crack edge")
        face, is_open = owners[0]
        if is_open:
            seeds.add(face)
        else:
            blocked.append(list(source_key))
    reachable = set(seeds)
    pending = list(seeds)
    while pending:
        face = pending.pop()
        for other in adjacency[face] - reachable:
            reachable.add(other)
            pending.append(other)
    return {
        "status": "geometric_water_access_only",
        "solver_ready": False, "engineering_validated": False,
        "wetting_solved": False, "pressure_assigned": False,
        "inlet_face_ids": sorted(seeds),
        "reachable_open_face_ids": sorted(reachable),
        "isolated_open_face_ids": np.flatnonzero(
            open_faces & ~np.isin(np.arange(len(open_faces)), list(reachable)),
        ).tolist(),
        "closed_face_ids": np.flatnonzero(~open_faces).tolist(),
        "blocked_inlet_source_edges": blocked,
        "closure_tolerance_m": float(closure_tolerance_m),
        "limitations": "Connectivity is eligibility only; saturation, trapped water and transient wetting/pressure require a separate conserved-fluid state.",
    }
