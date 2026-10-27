"""Export a saved Elmer mesh with every volume cell and interface node intact.

Saved SIF coefficients define all joint springs and directional outer supports.
Their surface integration is nodally lumped, not Elmer's consistent Galerkin
assembly. See Aster_Porting.md for the remaining elastic acceptance gates.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "01_three_body_baseline"


def find_latest_source_mesh() -> Path | None:
    """Locate the most-recently-solved ``Elmer.three_body_solver`` run's
    ``model/three_body.msh`` under its ephemeral per-run job directory
    (``Elmer.three_body_solver.default_output_root()``,
    ``%LOCALAPPDATA%\\Lamb\\three-body-solver\\<run-id>\\`` on Windows), so
    this exporter can be run with no arguments (e.g. from a VS Code task)
    right after running that solver, instead of requiring the run-id to be
    typed in by hand each time. Returns None if no such run exists yet."""
    elmer_dir = ROOT / "Elmer"
    if str(elmer_dir) not in sys.path:
        sys.path.insert(0, str(elmer_dir))
    from Elmer.three_body_solver import default_output_root

    root = default_output_root()
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*/model/three_body.msh"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None

# Gmsh physical tags actually written into three_body.msh by
# Elmer.three_body_diagnostic/write_combined_gmsh. NOTE: these do NOT match
# Elmer/regions/model_groups.py's FACE_GROUP_IDS numbering -- that module's
# own comment says "IDs in diagnostic.json are semantic source IDs;
# boundary_id_map maps them to saved mesh/SIF IDs", i.e. the numbers below
# are the renumbered *saved-mesh* tags, empirically identified by matching
# each candidate tag's geometry (area, z-range) against diagnostic.json's
# per-face area_m2 values (rock_outer_support.faces) and the legacy
# single-wall mesh_manifest.json's UPSTREAM/DOWNSTREAM/CREST areas -- do not
# assume these numbers are stable across other three-body runs/mesh
# revisions without re-verifying the same way.
WALL_VOLUME_TAG = 1
BEDROCK_VOLUME_TAG = 2
FOUNDATION_VOLUME_TAG = 3

UPSTREAM_FACE_TAG = 1  # area ~3053 m^2, z in [-29, 0]; matches legacy UPSTREAM (~3132 m^2)
DOWNSTREAM_FACE_TAG = 2  # area ~2986 m^2, z in [-29, 0]; matches legacy DOWNSTREAM (~3063 m^2)
CREST_FACE_TAG = 3  # area ~655 m^2, z == 0 exactly (flat top); matches legacy CREST (~672 m^2)
ROCK_BOTTOM_FACE_TAG = 6  # area 41801.38 m^2, z == -87 exactly; matches diagnostic.json "bottom"
ROCK_SIDE_FACE_TAGS = (7, 8, 9, 10)  # areas 14770.65 (banks) x2 + 20572.98 (streams) x2

GRID_CELL_SIZE_M = 0.01
WELD_TOLERANCE_M = 1e-6

# MED/Code_Aster group names used for the wall/foundation discrete DIS_T
# spring elements (one SEG2 per node pair). Must match the names
# convert_three_body_to_med.py assigns to the level-(-2) cells it creates
# from the ``seg2_wall_foundation_springs``/``wall_foundation_tributary_area_m2``
# npz arrays below.
WALL_FOUNDATION_SPRING_GROUP = "WALL_FOUNDATION_SPRINGS"
WALL_FOUNDATION_SPRING_PREFIX = "SPR_WF_"

# The bedrock volume (inherited directly from Elmer's graded_bedrock.py mesh
# grading near the outer domain-truncation surface, far laterally from the
# dam) contains a small number of severely sliver tetrahedra -- element
# shape quality here is max_edge_length^3 / volume, which is <= ~93 for
# every CONCRETE element but reaches ~4800 for the worst BEDROCK elements
# (max/min edge ratio up to ~190:1). Code_Aster's MECA_STATIQUE solves
# "successfully" (ARRET NORMAL, no pivot/singularity diagnostic) on the full
# mesh but silently produces an all-NaN DEPL field -- confirmed by removing
# all loading except gravity and fully fixing every outer bedrock boundary,
# which still produced all-NaN displacements, ruling out the load functions
# and RIGI_PARASOL spring magnitude as the cause. Dropping bedrock
# tetrahedra above this quality threshold (~0.34% of bedrock cells; verified
# to leave the mesh a single connected component) is the documented fix;
# the dropped elements sit at the free bedrock top surface >50 m laterally
# from the dam, in the semi-infinite-domain truncation region, not the
# region being compared against Elmer.
BEDROCK_SLIVER_QUALITY_THRESHOLD = 50.0

# Even after dropping the worst-shaped slivers above, MECA_STATIQUE's MUMPS
# factorization still hits a "pivot nul ou presque nul" (FACTOR_11) exception
# at a handful of individual bedrock nodes. Root cause (confirmed by direct
# inspection of the mesh_arrays.npz): these are *free* bedrock nodes -- not on
# ROCK_BOTTOM/ROCK_SIDES (so they get no explicit DDL_IMPO/RIGI_PARASOL) and
# not shared with CONCRETE -- that are supported by only 1-2 tetrahedra in the
# whole mesh. A node whose entire stiffness connection to the rest of the
# structure runs through a single (or pair of) tetrahedra sits at the tip of
# Elmer's graded_bedrock.py mesh-grading transition and produces a
# near-singular local stiffness block regardless of that tet's own shape
# quality (raising BEDROCK_SLIVER_QUALITY_THRESHOLD alone just moves the
# failure to a different such node -- confirmed empirically: threshold 300
# failed at node 140520, threshold 150 failed at a different node, 135762).
# MIN_FREE_NODE_SUPPORT below drops any bedrock tetrahedra that are the
# sole/near-sole support of such a node; the node itself then has zero
# remaining elements and is pruned by the existing re-densification pass.
MIN_FREE_NODE_SUPPORT = 3


class MeshExportError(ValueError):
    """The source Gmsh file is missing an expected section or group."""


def _read_gmsh_ascii(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a Gmsh 2.2 ASCII mesh with tetrahedra (type 4) and triangles
    (type 2) only. Returns (node_coords[N,3] 1-indexed-to-row-0 array,
    tet_node_ids[T,4], tet_tags[T], tri_node_ids[F,3], tri_tags[F])."""
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0

    def find(tag: str) -> int:
        nonlocal index
        while lines[index].strip() != tag:
            index += 1
        return index

    find("$Nodes")
    index += 1
    node_count = int(lines[index])
    index += 1
    coords = np.zeros((node_count + 1, 3), dtype=np.float64)  # row 0 unused (Gmsh IDs are 1-based)
    for row in range(node_count):
        parts = lines[index + row].split()
        node_id = int(parts[0])
        coords[node_id, 0] = float(parts[1])
        coords[node_id, 1] = float(parts[2])
        coords[node_id, 2] = float(parts[3])
    index += node_count

    find("$Elements")
    index += 1
    element_count = int(lines[index])
    index += 1
    tet_rows: list[list[int]] = []
    tet_tags: list[int] = []
    tri_rows: list[list[int]] = []
    tri_tags: list[int] = []
    for row in range(element_count):
        parts = lines[index + row].split()
        element_type = int(parts[1])
        tag_count = int(parts[2])
        tag = int(parts[3])
        node_ids = [int(value) for value in parts[3 + tag_count:]]
        if element_type == 4:
            tet_rows.append(node_ids)
            tet_tags.append(tag)
        elif element_type == 2:
            tri_rows.append(node_ids)
            tri_tags.append(tag)
    return coords, np.asarray(tet_rows, dtype=np.int64), np.asarray(tet_tags, dtype=np.int64), \
        np.asarray(tri_rows, dtype=np.int64), np.asarray(tri_tags, dtype=np.int64)


def _weld_coincident_nodes(
    coords: np.ndarray, used_node_ids: np.ndarray,
    touches_wall: np.ndarray | None = None, touches_foundation: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return a remap array (indexed by original 1-based node id) sending every
    node to the lowest-numbered node id at the same coordinate (grid-bucket
    search, same technique as Aster/postprocess_baseline.py), plus the list
    of (wall_node_id, foundation_node_id) pairs that were deliberately left
    UNMERGED because they sit on the wall/foundation spring interface (see
    module docstring's "known, documented simplifications" -- this is the
    carve-out that lets that one interface be modelled with real DIS_T
    springs instead of being welded bonded like everything else).

    A coincident-node group is treated as a wall/foundation interface
    (unmerged, paired) only when it splits cleanly into nodes touched
    exclusively by WALL tetrahedra and nodes touched exclusively by
    FOUNDATION tetrahedra, with no third kind of member (e.g. a node also
    touching BEDROCK, or touching both WALL and FOUNDATION already). Any
    other group (including wall-internal-joint groups, where every member is
    WALL-only) is merged/bonded exactly as before.
    """
    remap = np.arange(coords.shape[0], dtype=np.int64)
    interface_pairs: list[tuple[int, int]] = []
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for node_id in used_node_ids:
        xyz = coords[node_id]
        key = tuple(np.floor(xyz / GRID_CELL_SIZE_M).astype(np.int64))
        buckets.setdefault(key, []).append(int(node_id))

    def neighbours(key: tuple[int, int, int]):
        kx, ky, kz = key
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbour = (kx + dx, ky + dy, kz + dz)
                    if neighbour in buckets:
                        yield from buckets[neighbour]

    visited: set[int] = set()
    for node_id in used_node_ids:
        node_id = int(node_id)
        if node_id in visited:
            continue
        xyz = coords[node_id]
        key = tuple(np.floor(xyz / GRID_CELL_SIZE_M).astype(np.int64))
        group = [
            other for other in neighbours(key)
            if other not in visited and np.linalg.norm(coords[other] - xyz) <= WELD_TOLERANCE_M
        ]
        if not group:
            continue
        group = group + [node_id]
        for member in group:
            visited.add(member)

        if touches_wall is not None and touches_foundation is not None:
            wall_only = [n for n in group if touches_wall[n] and not touches_foundation[n]]
            foundation_only = [n for n in group if touches_foundation[n] and not touches_wall[n]]
            if wall_only and foundation_only and len(wall_only) + len(foundation_only) == len(group):
                canonical_wall = min(wall_only)
                canonical_foundation = min(foundation_only)
                for other in wall_only:
                    remap[other] = canonical_wall
                for other in foundation_only:
                    remap[other] = canonical_foundation
                interface_pairs.append((canonical_wall, canonical_foundation))
                continue

        canonical = min(group)
        for other in group:
            remap[other] = canonical
    return remap, interface_pairs


def _extract_boundary_triangles(tet_nodes: np.ndarray) -> np.ndarray:
    """Faces (as sorted node-id triples) that belong to exactly one
    tetrahedron in ``tet_nodes`` -- i.e. the exposed boundary surface of that
    (sub)set of tetrahedra, independent of any face-tag information."""
    if tet_nodes.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    faces = np.concatenate([
        tet_nodes[:, [0, 1, 2]], tet_nodes[:, [0, 1, 3]], tet_nodes[:, [0, 2, 3]], tet_nodes[:, [1, 2, 3]],
    ], axis=0)
    sorted_faces = np.sort(faces, axis=1)
    unique_faces, counts = np.unique(sorted_faces, axis=0, return_counts=True)
    return unique_faces[counts == 1]


def _wall_foundation_tributary_areas(
    coords: np.ndarray, wall_tet_nodes: np.ndarray, wall_interface_nodes: set[int],
) -> dict[int, float]:
    """Lumped nodal tributary area (m^2) for each wall-side interface node,
    computed from the WALL body's own exposed boundary faces restricted to
    the wall/foundation interface (i.e. excluding UPSTREAM/DOWNSTREAM/CREST/
    wall-internal-joint faces, which don't touch any foundation-only node).
    This mirrors how Elmer's Mortar BC integrates a surface traction
    coefficient (N/m^3) over the boundary mesh to get a nodal force; here we
    lump it the same way medcoupling/Code_Aster's own mass-lumped spring
    elements would, so that DIS_T stiffness = spring_n_per_m3 * tributary_area_m2."""
    boundary_faces = _extract_boundary_triangles(wall_tet_nodes)
    areas: dict[int, float] = {}
    for face in boundary_faces:
        if not all(int(n) in wall_interface_nodes for n in face):
            continue
        p0, p1, p2 = coords[face[0]], coords[face[1]], coords[face[2]]
        area = 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
        for n in face:
            areas[int(n)] = areas.get(int(n), 0.0) + area / 3.0
    return areas


def _drop_sliver_tetrahedra(coords: np.ndarray, cells: np.ndarray, quality_threshold: float) -> tuple[np.ndarray, int]:
    """Remove tetrahedra whose shape quality (max edge length cubed / volume,
    a scale-invariant degeneracy measure) exceeds ``quality_threshold``.
    Returns the filtered cell array and the number of cells dropped. See
    ``BEDROCK_SLIVER_QUALITY_THRESHOLD``'s docstring for why this is needed."""
    p0, p1, p2, p3 = coords[cells[:, 0]], coords[cells[:, 1]], coords[cells[:, 2]], coords[cells[:, 3]]
    edge_pairs = ((p0, p1), (p0, p2), (p0, p3), (p1, p2), (p1, p3), (p2, p3))
    max_edge = np.max([np.linalg.norm(a - b, axis=1) for a, b in edge_pairs], axis=0)
    volume = np.abs(np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0
    quality = max_edge**3 / np.maximum(volume, 1e-300)
    keep = quality <= quality_threshold
    return cells[keep], int((~keep).sum())


def _prune_weakly_supported_free_nodes(
    bedrock_cells: np.ndarray, concrete_cells: np.ndarray, protected_mask: np.ndarray,
    min_support: int, node_count: int,
) -> tuple[np.ndarray, int]:
    """Iteratively drop bedrock tetrahedra that are the sole/near-sole
    support of a free (non-boundary, non-concrete) node. See
    ``MIN_FREE_NODE_SUPPORT``'s docstring for why this is needed. Converges
    quickly in practice (a handful of iterations) since each pass only
    removes elements, never adds them."""
    concrete_mask = np.zeros(node_count, dtype=bool)
    concrete_mask[np.unique(concrete_cells.ravel())] = True
    total_dropped = 0
    while True:
        counts = np.zeros(node_count, dtype=np.int64)
        for col in range(4):
            np.add.at(counts, bedrock_cells[:, col], 1)
        free = (counts > 0) & ~concrete_mask & ~protected_mask
        weak_nodes = free & (counts < min_support)
        if not weak_nodes.any():
            return bedrock_cells, total_dropped
        touches_weak = np.any(weak_nodes[bedrock_cells], axis=1)
        total_dropped += int(touches_weak.sum())
        bedrock_cells = bedrock_cells[~touches_weak]


def export(output_dir: Path = DEFAULT_OUTPUT_DIR, source_mesh: Path | None = None) -> dict:
    if source_mesh is None:
        raise MeshExportError(
            "source_mesh is required: point it at an already-solved "
            "Elmer.three_body_solver run's <run>/model/three_body.msh",
        )
    if not source_mesh.is_file():
        raise MeshExportError(f"Missing three-body combined mesh: {source_mesh}")

    coords, tet_nodes, tet_tags, tri_nodes, tri_tags = _read_gmsh_ascii(source_mesh)
    from Aster.source_contract import source_springs
    contract = source_springs(source_mesh, coords, tri_nodes, tri_tags)
    if tet_nodes.size == 0:
        raise MeshExportError(f"{source_mesh} contains no tetrahedra")

    is_wall = tet_tags == WALL_VOLUME_TAG
    is_foundation = tet_tags == FOUNDATION_VOLUME_TAG
    is_bedrock = tet_tags == BEDROCK_VOLUME_TAG
    unknown_tet = ~(is_wall | is_foundation | is_bedrock)
    if unknown_tet.any():
        raise MeshExportError(f"Unexpected tetra tags: {sorted(set(tet_tags[unknown_tet].tolist()))}")

    kept_ids = np.unique(tet_nodes)
    if not np.isin(tri_nodes, kept_ids).all():
        raise MeshExportError("Surface references a node outside the volume mesh")
    dense_of = np.full(coords.shape[0], -1, dtype=np.int64)
    dense_of[kept_ids] = np.arange(len(kept_ids), dtype=np.int64)
    dense_coords = coords[kept_ids]
    tet_nodes_dense = dense_of[tet_nodes]
    tri_nodes_dense = dense_of[tri_nodes]

    concrete_cells = tet_nodes_dense[is_wall | is_foundation]
    bedrock_cells = tet_nodes_dense[is_bedrock]
    n_slivers_dropped = 0

    def face_group(tag: int) -> np.ndarray:
        return tri_nodes_dense[tri_tags == tag]

    upstream_faces = face_group(UPSTREAM_FACE_TAG)
    downstream_faces = face_group(DOWNSTREAM_FACE_TAG)
    crest_faces = face_group(CREST_FACE_TAG)
    rock_bottom_faces = face_group(ROCK_BOTTOM_FACE_TAG)
    rock_side_faces = np.concatenate([face_group(tag) for tag in ROCK_SIDE_FACE_TAGS], axis=0) \
        if any((tri_tags == tag).any() for tag in ROCK_SIDE_FACE_TAGS) else np.empty((0, 3), dtype=np.int64)

    for name, faces in (
        ("UPSTREAM", upstream_faces), ("DOWNSTREAM", downstream_faces), ("CREST", crest_faces),
        ("ROCK_BOTTOM", rock_bottom_faces), ("ROCK_SIDES", rock_side_faces),
    ):
        if faces.size == 0:
            raise MeshExportError(f"Required face group {name} is empty after export")

    n_weak_dropped = 0
    wall_foundation_springs = dense_of[contract["pairs"]]
    wall_foundation_tributary_area_m2 = contract["areas"]
    support_nodes = dense_of[contract["support_nodes"]]

    rock_bottom_nodes = np.unique(rock_bottom_faces.ravel())
    rock_side_nodes = np.unique(rock_side_faces.ravel())

    def bounds(indices: np.ndarray) -> dict:
        xyz = dense_coords[indices]
        return {"min_xyz_m": xyz.min(axis=0).tolist(), "max_xyz_m": xyz.max(axis=0).tolist()}

    def tri_area_m2(faces: np.ndarray) -> float:
        p0, p1, p2 = dense_coords[faces[:, 0]], dense_coords[faces[:, 1]], dense_coords[faces[:, 2]]
        cross = np.cross(p1 - p0, p2 - p0)
        return float(0.5 * np.linalg.norm(cross, axis=1).sum())

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "three_body_mesh_arrays.npz",
        nodes=dense_coords,
        tetra4_concrete=concrete_cells,
        tetra4_bedrock=bedrock_cells,
        tri3_upstream=upstream_faces,
        tri3_downstream=downstream_faces,
        tri3_crest=crest_faces,
        tri3_rock_bottom=rock_bottom_faces,
        tri3_rock_sides=rock_side_faces,
        poi1_rock_bottom_nodes=rock_bottom_nodes,
        seg2_wall_foundation_springs=wall_foundation_springs,
        wall_foundation_tributary_area_m2=wall_foundation_tributary_area_m2,
        interface_stiffness_n_per_m=contract["stiffness"],
        interface_names=np.asarray(contract["interfaces"]),
        joint_slave_tri3=dense_of[contract["joint_slave_tri3"]],
        joint_master_tri3=dense_of[contract["joint_master_tri3"]],
        joint_coefficients_n_per_m3=contract["joint_coefficients_n_per_m3"],
        support_nodes=support_nodes,
        support_matrices_n_per_m=contract["support_matrices"],
        concrete_body_ids=tet_tags[is_wall | is_foundation],
        source_node_ids=kept_ids,
    )

    manifest = {
        "schema_version": 1,
        "source_mesh": str(source_mesh.resolve()),
        "source_config": contract["source_config"],
        "source_hashes": contract["source_hashes"],
        "source_pressures": contract["pressures"],
        "source_crest_pressure_pa": contract["crest_pressure_pa"],
        "support_springs": [
            {"group": f"SPR_OUT_{i:05d}", "matrix_n_per_m": matrix.tolist()}
            for i, matrix in enumerate(contract["support_matrices"])
        ],
        "simplifications_vs_elmer": [
            "Surface spring laws use nodal area lumping rather than Elmer's consistent Galerkin projector; "
            "all source volume cells and interface node identities are preserved.",
        ],
        "node_count": int(dense_coords.shape[0]),
        "concrete_cells": int(concrete_cells.shape[0]),
        "bedrock_cells": int(bedrock_cells.shape[0]),
        "bedrock_slivers_dropped": n_slivers_dropped,
        "bedrock_weakly_supported_dropped": n_weak_dropped,
        "wall_foundation_spring_pairs": int(wall_foundation_springs.shape[0]),
        "wall_foundation_spring_total_area_m2": float(wall_foundation_tributary_area_m2.sum()),
        "wall_foundation_spring_group_all": WALL_FOUNDATION_SPRING_GROUP,
        "wall_foundation_springs": [
            {"group": f"{WALL_FOUNDATION_SPRING_PREFIX}{i:05d}", "tributary_area_m2": float(area),
             "interface": contract["interfaces"][i], "stiffness_n_per_m": contract["stiffness"][i].tolist()}
            for i, area in enumerate(wall_foundation_tributary_area_m2)
        ],
        "groups": {
            "CONCRETE": {"cells": int(concrete_cells.shape[0]), **bounds(np.unique(concrete_cells.ravel()))},
            "BEDROCK": {"cells": int(bedrock_cells.shape[0]), **bounds(np.unique(bedrock_cells.ravel()))},
            "UPSTREAM": {"faces": int(upstream_faces.shape[0]), "area_m2": tri_area_m2(upstream_faces), **bounds(np.unique(upstream_faces.ravel()))},
            "DOWNSTREAM": {"faces": int(downstream_faces.shape[0]), "area_m2": tri_area_m2(downstream_faces), **bounds(np.unique(downstream_faces.ravel()))},
            "CREST": {"faces": int(crest_faces.shape[0]), "area_m2": tri_area_m2(crest_faces), **bounds(np.unique(crest_faces.ravel()))},
            "ROCK_BOTTOM": {"faces": int(rock_bottom_faces.shape[0]), "area_m2": tri_area_m2(rock_bottom_faces), **bounds(rock_bottom_nodes)},
            "ROCK_SIDES": {"faces": int(rock_side_faces.shape[0]), "area_m2": tri_area_m2(rock_side_faces), **bounds(rock_side_nodes)},
        },
        "coordinate_bounds": bounds(np.arange(dense_coords.shape[0])),
    }
    (output_dir / "mesh_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    if len(sys.argv) == 2:
        source_mesh = Path(sys.argv[1])
    elif len(sys.argv) == 1:
        source_mesh = find_latest_source_mesh()
        if source_mesh is None:
            print(
                "usage: "
                f"{sys.argv[0]} [path to three_body.msh] -- no explicit path was given and no "
                "Elmer.three_body_solver run was found under its default output root "
                "(run Elmer.three_body_solver / the 'Elmer: Run combined solver' task first)",
                file=sys.stderr,
            )
            raise SystemExit(2)
        print(f"No path given; using the latest Elmer.three_body_solver run: {source_mesh}", file=sys.stderr)
    else:
        print(f"usage: {sys.argv[0]} [path to three_body.msh]", file=sys.stderr)
        raise SystemExit(2)
    manifest = export(source_mesh=source_mesh)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
