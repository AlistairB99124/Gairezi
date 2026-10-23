"""Wall-on-bedrock Elmer check (bonded or stiff-spring seat); not frictional."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np

from .bedrock_preview import configured_rock_mesh, survey_coordinates
from .model_config import foundation_support, load_config, rock_outer_support, validate_config, wall_joint_support
from .regions.model_groups import joint_boundary_ids
from .regions.combined_structure import build_combined_structure, write_combined_gmsh
from .regions.tetrahedral import _CELL_FACES, tetrahedralize_structure
from .structure_assembly import (
    convert_with_elmergrid, mpi_launcher, write_combined_solver_input,
)


JOINT_MODELS = ("bonded", "spring")
# Boundary tags for the two sides of the wall/bedrock spring seat.
WALL_SEAT, ROCK_SEAT = 11, 12
ROCK_BOTTOM = 10
# Outer rock-box sides: local axis (x = across valley/banks, y = stream depth) and
# outward sign along that rock-local unit vector.
ROCK_SIDES = {13: ("x", -1.), 14: ("x", 1.), 15: ("y", -1.), 16: ("y", 1.)}
WALL_BODY, BEDROCK_BODY = 1, 2
SPRING_PAIRS = {"wall_bedrock": (WALL_SEAT, ROCK_SEAT)}
SPRING_SUPPORTS = {"wall_bedrock": "bedrock"}
WALL_JOINT_SOURCE = "geometry.wall.joint_support.spring_n_per_m3"
_ROCK_CELL_FACES = {**_CELL_FACES, 7: ((0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))}


def rock_side_faces(rock) -> dict[int, list[tuple[int, ...]]]:
    """Exterior rock faces on the four vertical box planes, in rock-local node ids.

    Every unpaired rock-cell face must be a top, bottom or side face, so a side
    support cannot silently miss part of the rock perimeter.
    """
    report = rock.report
    origin = np.asarray(report["section_origin_xy_m"], dtype=float)
    axes = {"x": np.asarray(report["across_unit_vector_xy"], dtype=float),
            "y": np.asarray(report["depth_unit_vector_xy"], dtype=float)}
    limits = {"x": report["rock_across_limits_m"], "y": report["rock_depth_limits_m"]}
    points = np.asarray(rock.points, dtype=float)
    local = {axis: (points[:, :2] - origin) @ vector for axis, vector in axes.items()}
    span = max(abs(value) for pair in limits.values() for value in pair)
    tolerance = 1e-7 * max(span, 1.)
    exterior: list[np.ndarray] = []
    for width in (3, 4):
        faces = [
            np.asarray(block, dtype=np.int64)[:, pattern]
            for kind, block in rock.blocks.items() for pattern in _ROCK_CELL_FACES[kind]
            if len(pattern) == width and len(block)
        ]
        if not faces:
            continue
        faces = np.vstack(faces)
        _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("Rock volume has non-manifold faces")
        exterior.extend(faces[first[counts == 1]])
    known = {tuple(sorted(int(n) for n in face)) for face in (*rock.top_faces, *rock.bottom_faces)}
    sides: dict[int, list[tuple[int, ...]]] = {tag: [] for tag in ROCK_SIDES}
    unclassified = 0
    for face in exterior:
        if tuple(sorted(int(n) for n in face)) in known:
            continue
        for tag, (axis, sign) in ROCK_SIDES.items():
            plane = limits[axis][0 if sign < 0 else 1]
            if np.all(np.abs(local[axis][face] - plane) <= tolerance):
                sides[tag].append(tuple(int(n) for n in face))
                break
        else:
            unclassified += 1
    if unclassified or not all(sides.values()):
        raise ValueError(
            f"Rock exterior is not a closed box: {unclassified} unclassified faces; "
            f"side face counts {{tag: len(faces)}} = { {tag: len(f) for tag, f in sides.items()} }"
        )
    return sides


def rock_outer_support_report(config: dict, rock, sides: dict[int, list[tuple[int, ...]]]) -> dict[str, object]:
    """Face-wise outer rock supports shared by the Elmer SIF and FEniCSx."""
    support = rock_outer_support(config)
    if support["model"] == "fixed_bottom":
        return {"model": "fixed_bottom", "fixed_boundary_tag": ROCK_BOTTOM,
                "description": "all bottom DOFs fixed; free outer rock sides"}
    springs = support["spring_n_per_m3"]
    geometry = rock.report
    points = np.asarray(rock.points, dtype=float)
    across = np.asarray(geometry["across_unit_vector_xy"], dtype=float)
    depth = np.asarray(geometry["depth_unit_vector_xy"], dtype=float)
    faces = {str(ROCK_BOTTOM): {
        "name": "bottom", "local_axis": "z", "outward_normal": [0., 0., -1.],
        "normal_stiffness_n_per_m3": springs["z"],
        "area_m2": face_area(points, [tuple(face) for face in rock.bottom_faces]),
    }}
    for tag, (axis, sign) in ROCK_SIDES.items():
        vector = sign * (across if axis == "x" else depth)
        faces[str(tag)] = {
            "name": {13: "bank_x_min", 14: "bank_x_max", 15: "stream_y_min", 16: "stream_y_max"}[tag],
            "local_axis": axis, "outward_normal": [float(vector[0]), float(vector[1]), 0.],
            "normal_stiffness_n_per_m3": springs[axis],
            "area_m2": face_area(points, sides[tag]),
        }
    return {
        "model": "rollers",
        "description": (
            "normal-only penalty springs k*(n.u) on the rock bottom and four vertical sides; "
            "tangential motion free (roller approximation)"
        ),
        "spring_axes": "rock-box local frame: x banks, y upstream/downstream, z bottom",
        "source": "material.bedrock.support.spring_n_per_m3",
        "faces": faces,
    }


def face_area(points: np.ndarray, faces: list[tuple[int, ...]]) -> float:
    total = 0.
    for face in faces:
        p = points[list(face)]
        for i in range(1, len(face) - 1):
            total += np.linalg.norm(np.cross(p[i] - p[0], p[i + 1] - p[0])) / 2
    return float(total)


def roller_condition(number: int, tag: int, face: dict[str, object], boundary_map: dict[int, int]) -> str:
    """Elmer BC: traction = -k (n.u) n via Spring ij = k n_i n_j (Cartesian, no N-T)."""
    normal = np.asarray(face["outward_normal"], dtype=float)
    tensor = face["normal_stiffness_n_per_m3"] * np.outer(normal, normal)
    entries = "".join(
        f"    Spring {i + 1}{j + 1} = Real {tensor[i, j] + 0.:.12g}\n"
        for i in range(3) for j in range(3) if tensor[i, j] != 0.
    )
    return (
        f"Boundary Condition {number}\n"
        f"    Name = \"RockRoller_{face['name']}\"\n"
        f"    Target Boundaries(1) = {boundary_map[tag]}\n"
        f"{entries}End\n\n"
    )


def wall_joints(config: dict) -> list[dict[str, object]]:
    """Spring joints between wall thickness blocks, top to bottom (empty when bonded)."""
    support = wall_joint_support(config["geometry"]["wall"])
    if support is None:
        return []
    return [
        {
            "name": f"wall_joint_{index + 1}",
            "z_m": level_m,
            "tags": list(joint_boundary_ids(index)),
            "spring_n_per_m3": dict(support["spring_n_per_m3"]),
        }
        for index, level_m in enumerate(support["levels_m"])
    ]


def interface_stiffness(config: dict) -> dict[str, dict[str, float]]:
    """Directional traction/jump coefficients in global Cartesian axes (N/m^3)."""
    validate_config(config)
    bedrock_bonded = config["geometry"]["bedrock"]["support"]["model"] == "bonded"
    values = {
        **({
            name: {
                axis: float(config["geometry"][region]["support"]["spring_n_per_m3"][axis])
                for axis in ("x", "y", "z")
            }
            for name, region in SPRING_SUPPORTS.items()
        } if not bedrock_bonded else {}),
        **{joint["name"]: dict(joint["spring_n_per_m3"]) for joint in wall_joints(config)},
    }
    if "foundation" in config["geometry"]:
        if not bedrock_bonded:
            values["foundation_bedrock"] = values.pop("wall_bedrock")
        support = foundation_support(config)
        if support["model"] == "spring":
            values["wall_foundation"] = dict(support["spring_n_per_m3"])
    return values


def _merge_bonded_seat_copies(concrete, tags=(1,)):
    """Merge block copies of joint nodes lying on a bonded seat.

    Both blocks are bonded to the same support point there, so their copies must move
    together; the joint springs still act everywhere above the seat line.
    """
    seat_nodes = {n for tag, face in concrete.boundaries if tag in tags for n in face}
    canonical: dict[tuple, int] = {}
    remap = list(range(len(concrete.nodes)))
    for i in sorted(seat_nodes):
        remap[i] = canonical.setdefault(tuple(round(v, 8) for v in concrete.nodes[i]), i)
    if remap == list(range(len(concrete.nodes))):
        return concrete
    kept = sorted(set(remap))
    compact = {old: new for new, old in enumerate(kept)}
    renumber = [compact[remap[i]] for i in range(len(remap))]
    return replace(
        concrete,
        nodes=[concrete.nodes[i] for i in kept],
        cells=[(kind, tuple(renumber[n] for n in cell)) for kind, cell in concrete.cells],
        boundaries=[(tag, tuple(renumber[n] for n in face)) for tag, face in concrete.boundaries],
    )


def prepare(
    root: Path, output: Path,
    stream_margin_m: float | None = None, bank_margin_m: float | None = None,
    joints: str = "bonded",
    bottom_z_m: float | None = None,
    *,
    integrated: bool = False,
) -> dict[str, object]:
    if joints not in JOINT_MODELS:
        raise ValueError(f"joints must be one of {JOINT_MODELS}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory to exclude stale results")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "config.json", output / "config.json")
    config = load_config(output / "config.json")
    # Thickness blocks are only split where springs join them; otherwise the wall stays monolithic.
    concrete = build_combined_structure(output, split_wall_joints=joints == "spring")
    joint_list = wall_joints(config) if concrete.wall_joint_levels_m else []
    foundation_present = "foundation" in config["geometry"]
    bonded_foundation = foundation_present and foundation_support(config)["model"] == "bonded"
    bonded_bedrock = config["geometry"]["bedrock"]["support"]["model"] == "bonded"
    seat_pairs = (
        {"foundation_bedrock": (11, 12), "wall_foundation": (17, 18)}
        if foundation_present else dict(SPRING_PAIRS)
    )
    if bonded_foundation:
        seat_pairs.pop("wall_foundation")
    if bonded_bedrock:
        seat_pairs.pop("foundation_bedrock" if foundation_present else "wall_bedrock")
    pairs = {**seat_pairs, **{joint["name"]: tuple(joint["tags"]) for joint in joint_list}}
    joint_levels_text = ", ".join(f"{joint['z_m']:g}" for joint in joint_list)
    concrete.nodes = [tuple(p) for p in survey_coordinates(
        np.asarray(concrete.nodes),
        config["geometry"]["wall"]["centerline_radius_m"], 80.,
    )]
    if integrated or foundation_present:
        concrete = tetrahedralize_structure(concrete)
    if joint_list and (bonded_foundation or (bonded_bedrock and not foundation_present)):
        concrete = _merge_bonded_seat_copies(concrete)
    # Joint nodes have one copy per wall block; the rock seat sees one node per location.
    canonical: dict[tuple, int] = {}
    seat_index = [canonical.setdefault(tuple(round(v, 8) for v in p), i)
                  for i, p in enumerate(concrete.nodes)]
    solver_concrete = concrete
    foundation = None
    if foundation_present:
        if joints != "spring":
            raise ValueError("Foundation trench requires joints='spring' (configured interface workflow)")
        from .foundation_trench import build_foundation
        from .graded_bedrock import build_graded_rock
        foundation = build_foundation(config, concrete)
        rock = build_graded_rock(
            config, foundation.points, [tuple(f) for f in foundation.terrain_faces],
            stream_margin_m, bank_margin_m, bottom_z_m, foundation=foundation,
        )
        offset = len(concrete.nodes)
        convert = lambda face: tuple(offset + int(n) for n in face)
        concrete = replace(
            concrete, nodes=concrete.nodes + [tuple(p) for p in foundation.points],
            cells=concrete.cells + [(4, convert(cell)) for cell in foundation.tetrahedra],
            region_ids=concrete.region_ids + [3] * len(foundation.tetrahedra),
            boundaries=[(17 if tag == 1 else tag, face) for tag, face in concrete.boundaries]
            + [(18, convert(face)) for face in foundation.wall_faces]
            + [(1, convert(face)) for face in foundation.contact_faces],
        )
        if bonded_foundation:
            concrete = _merge_bonded_seat_copies(concrete, tags=(17, 18))
            concrete = replace(concrete, boundaries=[
                (tag, face) for tag, face in concrete.boundaries if tag not in (17, 18)
            ])
        foundation.report["support"] = dict(foundation_support(config))
    else:
        rock = configured_rock_mesh(
            config, np.asarray(concrete.nodes),
            [tuple(seat_index[n] for n in face) for tag, face in concrete.boundaries if tag == 1],
            stream_margin_m, bank_margin_m, bottom_z_m,
        )
    rock_points, rock_blocks, geometry = rock.points, rock.blocks, rock.report
    bonded_seat = joints == "spring" and bonded_bedrock
    stiffness = interface_stiffness(config) if joints == "spring" else {}
    stiffness = {name: value for name, value in stiffness.items() if name in pairs}
    seat_faces = [face for tag, face in concrete.boundaries if tag == 1]
    nodes, concrete_cells, concrete_boundaries = (
        list(concrete.nodes), concrete.cells, concrete.boundaries,
    )
    spring_seat = joints == "spring" and not bonded_seat
    # A spring seat gives the rock its own copies of the seat nodes.
    index = {} if spring_seat else {tuple(round(v, 8) for v in p): i for i, p in enumerate(nodes)}
    mapping = []
    for coordinates in rock_points:
        point = tuple(coordinates)
        key = tuple(round(v, 8) for v in point)
        if key in index:
            mapping.append(index[key])
        else:
            mapping.append(len(nodes))
            nodes.append(point)
    rock_cells = [
        (kind, tuple(mapping[int(n)] for n in cell))
        for kind, block in rock_blocks.items() for cell in block
    ]
    cells = concrete_cells + rock_cells
    regions = concrete.region_ids + [BEDROCK_BODY] * len(rock_cells)
    patterns = {**_CELL_FACES, 4: ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
                7: ((0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))}
    # Triangulate exterior/interface quads consistently for a topology audit.
    from .regions.tetrahedral import split_face
    face_counts = Counter()
    memberships: dict[tuple[int, ...], set[int]] = {}
    for (kind, cell), region in zip(concrete_cells, concrete.region_ids, strict=True):
        for pattern in patterns[kind]:
            face = tuple(cell[i] for i in pattern)
            for tri in split_face(face):
                key = tuple(sorted(tri))
                face_counts[key] += 1
                memberships.setdefault(key, set()).add(region)
    # Only rock-top faces can meet the concrete. The mesher audits rock volume.
    top_faces = [tuple(mapping[int(n)] for n in tri) for tri in rock.top_faces]
    for face in top_faces:
        key = tuple(sorted(face))
        face_counts[key] += 1
        memberships.setdefault(key, set()).add(BEDROCK_BODY)
    if max(face_counts.values()) > 2:
        raise ValueError("Non-manifold wall/bedrock mesh")
    seat = [key for key, value in memberships.items() if value == {3 if foundation_present else WALL_BODY, BEDROCK_BODY}]
    if spring_seat:
        if seat:
            raise ValueError("Spring seat must not share nodes between wall and bedrock")
        def located(face: tuple[int, ...]) -> tuple:
            return tuple(sorted(tuple(round(v, 7) for v in nodes[n]) for n in face))
        wall_seat = {located(tri) for face in seat_faces for tri in split_face(face)}
        rock_seat = [face for face in top_faces if located(face) in wall_seat]
        if len(rock_seat) != geometry["seat_triangles"] or len(rock_seat) != len(wall_seat):
            raise ValueError("Rock and wall seat faces do not coincide")
        seat_count = len(rock_seat)
    elif len(seat) != geometry["seat_triangles"]:
        raise ValueError("Rock/wall seat is not fully connected")
    else:
        seat_count = len(seat)
    boundaries = [(tag, face) for tag, face in concrete_boundaries if tag != 1]
    if spring_seat:
        boundaries.extend((WALL_SEAT, face) for face in seat_faces)
        boundaries.extend((ROCK_SEAT, face) for face in rock_seat)
    bottom_faces = [tuple(mapping[int(n)] for n in tri) for tri in rock.bottom_faces]
    boundaries.extend((ROCK_BOTTOM, face) for face in bottom_faces)
    sides = rock_side_faces(rock) if rock_outer_support(config)["model"] == "rollers" else {}
    outer_support = rock_outer_support_report(config, rock, sides)
    for tag, faces in sides.items():
        boundaries.extend((tag, tuple(mapping[n] for n in face)) for face in faces)
    from .mortar_interfaces import split_planar_interfaces, expand_spring_pairs
    pair_contract = pairs
    if foundation_present:
        boundaries, pair_contract = split_planar_interfaces(nodes, boundaries, pairs)
    solver_pairs = expand_spring_pairs(pair_contract)
    solver_stiffness = {name: stiffness[name.split("/")[0]] for name in solver_pairs if name.split("/")[0] in stiffness}
    boundary_map = {tag: i+1 for i, tag in enumerate(sorted({tag for tag, _ in boundaries}))}
    mesh = replace(
        concrete, nodes=nodes, cells=cells,
        region_ids=regions,
        boundaries=[(boundary_map[tag], face) for tag, face in boundaries],
    )
    write_combined_gmsh(mesh, output / "three_body.msh")
    # Preserve concrete water datum; the added rock must not lower tailwater.
    path = output / ("dam_model.sif" if integrated else "diagnostic.sif")
    write_combined_solver_input(solver_concrete, path)
    text = path.read_text()
    start = text.index("Boundary Condition 1\n")
    end = text.index("Boundary Condition 2\n", start)
    if outer_support["model"] == "rollers":
        # Tags are remapped below with the concrete BCs, so write source tags here.
        identity = {tag: tag for tag in boundary_map}
        faces = outer_support["faces"]
        bottom = roller_condition(1, ROCK_BOTTOM, faces[str(ROCK_BOTTOM)], identity)
        last = max(int(n) for n in re.findall(r"^Boundary Condition (\d+)$", text, re.M))
        side_text = "\n" + "".join(
            roller_condition(last + offset, tag, faces[str(tag)], identity)
            for offset, tag in enumerate(ROCK_SIDES, start=1)
        )
        text = text[:start] + bottom + text[end:] + side_text
    else:
        text = text[:start] + '''Boundary Condition 1
    Name = "DiagnosticFixedRockBottom"
    Target Boundaries(1) = 10
    Displacement 1 = Real 0
    Displacement 2 = Real 0
    Displacement 3 = Real 0
End

''' + text[end:]
    rock = config["material"]["bedrock"]
    text += f'''
Body 2
    Name = "BedrockBody"
    Equation = 1
    Material = 2
    Body Force = 2
End

Material 2
    Name = "Granite"
    Youngs Modulus = Real {rock["youngs_modulus"]:.12g}
    Poisson Ratio = Real {rock["poisson_ratio"]:.12g}
    Density = Real {rock["density"]:.12g}
End

Body Force 2
    Stress Bodyforce 3 = Real {rock["density"] * config["loads"]["gravity"] + 0.:.12g}
End
'''
    if foundation_present:
        text += '\nBody 3\n    Name = "FoundationBody"\n    Equation = 1\n    Material = 1\n    Body Force = 1\nEnd\n'
    text = re.sub(
        r"Target Boundaries\(1\) = (\d+)",
        lambda match: f"Target Boundaries(1) = {boundary_map[int(match.group(1))]}", text,
    )
    if integrated:
        # Fixed-level fill avoids the runaway ILUT factors on the full rock mesh.
        # The foundation's many grouped spring interfaces stall ILU1 (residual ~1e-6
        # after 2000 iterations); level-2 fill reaches ~6e-9 within ~360 iterations.
        text = text.replace(
            "Linear System Preconditioning = ILUT",
            f"Linear System Preconditioning = {'ILU2' if foundation_present else 'ILU1'}\n"
            "    Linear System ILU Factor = Real 0.01",
        ).replace(
            "Linear System ILUT Tolerance = 1.0e-4",
            "Linear System Residual Output = 20",
        )
    if stiffness:
        text = text.replace(
            "    Variable DOFs = 3\n",
            "    Variable DOFs = 3\n    Apply Mortar BCs = Logical True\n", 1,
        )
        text += spring_conditions(text, boundary_map, solver_stiffness, solver_pairs)
    if foundation_present:
        wall_foundation_note = (
            "perfectly bonded wall/foundation (shared nodes; zero relative displacement)"
            if bonded_foundation else
            "wall/foundation directional springs (global x/y/z) from "
            "geometry.foundation.support.spring_n_per_m3"
        )
        foundation_bedrock_note = (
            "perfectly bonded foundation/bedrock (shared nodes; zero relative displacement)"
            if bonded_bedrock else
            "foundation/bedrock directional springs from geometry.bedrock.support.spring_n_per_m3"
        )
        joint_note = f"{wall_foundation_note}; {foundation_bedrock_note}"
    elif spring_seat:
        joint_note = (
            f"directional springs (global x/y/z) {stiffness} N/m^3 "
            "from geometry.bedrock.support at the wall/rock seat"
        )
    elif bonded_seat:
        joint_note = "perfectly bonded wall/rock seat (shared nodes; zero relative displacement)"
    else:
        joint_note = "bonded wall/rock seat"
    joint_note += (
        f"; wall blocks joined at z={joint_levels_text} m by "
        f"directional springs from {WALL_JOINT_SOURCE}"
        if joint_list else ""
    )
    result_name = "three_body_results" if integrated else "three_body_diagnostic"
    text = text.replace(
        "! Single wall body seated on the bedrock; foundation springs from geometry.bedrock.support.\n",
        "",
    )
    rollers = outer_support["model"] == "rollers"
    if rollers:
        k = rock_outer_support(config)["spring_n_per_m3"]
        support_header = (
            "rock rollers (normal penalty springs bottom/sides)",
            f"! Rock box rollers: normal-only springs x={k['x']:g}, y={k['y']:g}, z={k['z']:g} N/m^3 "
            "(local banks/stream/bottom). Configured graded rock mesh. Not a frictional/design result.\n",
        )
        support_assumptions = [
            f"rock bottom z={geometry['bottom_z_m']:g} m and four vertical sides on normal-only "
            "roller springs (tangential motion free)",
            "rock self-weight and dam loads applied together (no geostatic initial stage)",
        ]
    else:
        support_header = (
            "fixed rock bottom",
            "! Free outer rock sides. Configured graded rock mesh. Not a frictional/design result.\n",
        )
        support_assumptions = [f"fixed rock bottom z={geometry['bottom_z_m']:g} m", "free rock sides"]
    path.write_text(
        f"! {'WALL-FOUNDATION-BEDROCK ANALYSIS' if foundation_present and integrated else 'WALL-BEDROCK ANALYSIS' if integrated else 'DIAGNOSTIC ONLY'}: "
        f"{joint_note}; {support_header[0]}; zero uplift.\n"
        + support_header[1]
        + text.replace("combined_results_mpi", result_name),
    )
    report = {
        "foundation": foundation.report if foundation_present else {"present": False},
        "foundation_bedrock_connection": ("bonded" if bonded_bedrock else "spring") if foundation_present else None,
        "wall_foundation_connection": "bonded" if bonded_foundation else "spring" if foundation_present else None,
        "assumptions": [
            (
                f"{wall_foundation_note}; {foundation_bedrock_note}"
                if foundation_present else
                "linear directional springs in global x/y/z (carry tension) at the wall/rock seat"
                if spring_seat else
                "perfectly bonded wall/rock seat (zero relative displacement)"
            ),
            *([] if not foundation_present else [
                "both concrete interfaces share displacement nodes"
                if bonded_foundation and bonded_bedrock else
                "the wall/foundation interface shares displacement nodes; foundation/bedrock interface uses springs"
                if bonded_foundation else
                "the wall/foundation interface uses springs; foundation/bedrock interface shares displacement nodes"
                if bonded_bedrock else
                "both concrete interfaces use directional springs"
            ]),
            *([
                f"wall split into thickness blocks at z={joint_levels_text}"
                " m; linear directional springs in global x/y/z (carry tension, no opening "
                "or sliding limit) join the blocks; damage cannot cross a joint"
            ] if joint_list else []),
            (f"wall seated exclusively on a structured staircase concrete foundation; horizontal treads and station-plane risers; cut floor and sides "
             f"{'bonded' if bonded_bedrock else 'spring-connected'} to bedrock"
             if foundation_present else "the wall is seated directly on the bedrock contours"),
            *(["constant-width radial shoulders and crest-end extensions separate all wall/rock coordinates; no shared displacement nodes or wall/rock interface faces"]
              if foundation_present else []),
            "zero uplift",
            *support_assumptions,
            "simultaneous gravity and existing concrete water loads",
            "80 m chainage reference",
            (
                f"distance-graded rock: {geometry['mesh_grading']}"
                if geometry.get("mesh_method") else
                f"nominal rock spacing {geometry['element_size_m']:g} m; "
                f"{geometry['vertical_layers']} vertical layers"
            ),
        ],
        "purpose": (
            "Initial wall-on-bedrock linear analysis, NOT engineering validation"
            if integrated else "Solver integration diagnostic, NOT engineering validation"
        ),
        "joints": joints,
        "wall_bedrock_connection": "not_in_contact" if foundation_present else "spring" if spring_seat else "bonded",
        "interface_stiffness_n_per_m3": stiffness,
        "interface_stiffness_sources": {
            **{
                name: f"geometry.{region}.support.spring_n_per_m3"
                for name, region in SPRING_SUPPORTS.items() if name in stiffness
            },
            **{joint["name"]: WALL_JOINT_SOURCE for joint in joint_list},
            **({"wall_foundation": "geometry.foundation.support.spring_n_per_m3"
                } if foundation_present and not bonded_foundation else {}),
            **({"foundation_bedrock": "geometry.bedrock.support.spring_n_per_m3"}
               if foundation_present and not bonded_bedrock else {}),
        } if joints == "spring" else {},
        "spring_pairs": {name: value if isinstance(value, dict) else list(value)
                         for name, value in pair_contract.items() if name in stiffness},
        "wall_connection": "spring_jointed_blocks" if joint_list else "single_monolithic_body",
        "wall_joints": joint_list,
        "nodes": len(nodes),
        "cells_by_body": dict(Counter(regions)),
        "solver_body_ids": {"wall": WALL_BODY, "bedrock": BEDROCK_BODY, **({"foundation": 3} if foundation_present else {})},
        "boundary_id_map": boundary_map,
        "rock_outer_support": outer_support,
        "wall_rock_seat_triangles": seat_count,
        "wall_underside": {
            "profile": "stepped_45deg_chamfer" if foundation_present else "piecewise_linear_bedrock_contours",
            "elevation_quantization": foundation_present,
            "radial_extent": "full_wall_thickness",
        },
        "geometry": geometry,
        "concrete_element_types": sorted({kind for kind, _ in concrete_cells}),
        "linear_solver": (
            {"method": "BiCGStabl", "preconditioner": "ILU2" if foundation_present else "ILU1",
             "ilu_factor": 0.01, "convergence_tolerance": 1e-10}
            if integrated else
            {"method": "BiCGStabl", "preconditioner": "ILUT",
             "ilut_tolerance": 1e-4, "convergence_tolerance": 1e-10}
        ),
        "solver_input": path.name,
        "result_name": result_name,
        "result": None,
    }
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2))
    return report


def spring_conditions(
    text: str, boundary_map: dict[int, int], stiffness: dict[str, dict[str, float]],
    pairs: dict[str, tuple[int, int]] | None = None,
) -> str:
    """Mortar spring pairs: traction = k * (relative displacement), N/m^3."""
    first = max(int(n) for n in re.findall(r"^Boundary Condition (\d+)$", text, re.M)) + 1
    sections = []
    active = [(name, pair) for name, pair in (pairs or SPRING_PAIRS).items() if name in stiffness]
    for offset, (name, (slave, master)) in enumerate(active):
        slave_bc, master_bc = first + 2 * offset, first + 2 * offset + 1
        coefficients = "".join(
            f"    Mortar BC Coefficient Displacement {i} = Real {stiffness[name][axis]:.12g}\n"
            for i, axis in enumerate(("x", "y", "z"), start=1)
        )
        sections.append(
            f"\nBoundary Condition {slave_bc}\n"
            f"    Name = \"{name}_spring\"\n"
            f"    Target Boundaries(1) = {boundary_map[slave]}\n"
            f"    Mortar BC = Integer {master_bc}\n"
            "    Galerkin Projector = Logical True\n"
            "    Normal Projector = Logical True\n"
            + ("    Save Projector = Logical True\n" if "/" in name else "")
            + f"{coefficients}End\n"
            f"\nBoundary Condition {master_bc}\n"
            f"    Name = \"{name}_master\"\n"
            f"    Target Boundaries(1) = {boundary_map[master]}\n"
            "End\n"
        )
    return "".join(sections)


def spring_partition_options(
    boundary_map: dict[int, int], pairs: dict[str, tuple[int, int]] | None = None,
) -> tuple[str, ...]:
    # Elmer's mortar projector silently gives wrong answers when the two sides
    # of a joint sit on different ranks; keep every joint face on one rank.
    tags = [str(boundary_map[tag]) for pair in (pairs or SPRING_PAIRS).values()
            if all(tag in boundary_map for tag in pair) for tag in pair]
    # A partition-only source mesh coalesces large planar inventories to one
    # tag; original face labels are restored before solver launch.
    if len(tags) > 128:
        return ("-partdual", "-connect", str(min(map(int, tags))), "-halobc")
    return ("-partdual", "-connect", *tags, "-halobc")


def check_spring_partitions(
    mesh: Path, partitions: int, boundary_map: dict[int, int],
    pairs: dict[str, tuple[int, int]] | None = None,
) -> None:
    """Fail unless both sides of every spring joint are on the same single rank."""
    owners: dict[int, set[int]] = {}
    for part in range(1, partitions + 1):
        with (mesh / f"partitioning.{partitions}" / f"part.{part}.boundary").open() as stream:
            for line in stream:
                owners.setdefault(int(line.split()[1]), set()).add(part)
    for name, pair in (pairs or SPRING_PAIRS).items():
        if not any(tag in boundary_map for tag in pair):
            continue
        ranks = [owners.get(boundary_map[tag], set()) for tag in pair]
        if len(ranks[0]) != 1 or ranks[0] != ranks[1]:
            raise RuntimeError(f"{name} spring sides were split across ranks {ranks}")


def solve(output: Path, timeout: int | None = 480, partitions: int = 4) -> Path:
    if partitions < 1 or (timeout is not None and timeout <= 0):
        raise ValueError("Partitions and optional solver timeout must be positive")
    report_path = output / "diagnostic.json"
    report = json.loads(report_path.read_text())
    if (output / "results").exists():
        raise ValueError("Results directory already exists; prepare a fresh run to exclude stale results")
    executable = shutil.which("ElmerSolver_mpi")
    if executable is None:
        raise RuntimeError("ElmerSolver_mpi unavailable")
    launcher, option = mpi_launcher()
    pvpython_executable()
    spring = report.get("joints") == "spring"
    boundary_map = {int(k): v for k, v in report["boundary_id_map"].items()}
    from .mortar_interfaces import expand_spring_pairs
    pairs = expand_spring_pairs(report.get("spring_pairs", SPRING_PAIRS))
    options = spring_partition_options(boundary_map, pairs) if spring else ()
    print(f"Converting and partitioning three-body mesh ({partitions} rank(s))...", flush=True)
    partition_source = output / "three_body.msh"
    original_boundary_tags = None
    if sum(len(pair) for pair in pairs.values()) > 128:
        from .mortar_interfaces import coalesced_partition_input
        partition_source = output / "partition_input.msh"
        original_boundary_tags = coalesced_partition_input(
            output / "three_body.msh", partition_source,
            {boundary_map[tag] for pair in pairs.values() for tag in pair},
        )
    if not convert_with_elmergrid(
        partition_source, output / "mesh",
        partitions if partitions > 1 else None, options if partitions > 1 else (),
    ):
        raise RuntimeError("ElmerGrid unavailable")
    if original_boundary_tags is not None:
        from .mortar_interfaces import restore_partition_boundary_tags
        report["partition_boundary_restoration"] = restore_partition_boundary_tags(
            output / "mesh", original_boundary_tags,
        )
        report_path.write_text(json.dumps(report, indent=2))
    if spring and partitions > 1:
        check_spring_partitions(output / "mesh", partitions, boundary_map, pairs)
    (output / "results").mkdir()
    environment = os.environ.copy()
    environment.setdefault("ELMER_HOME", str(Path(executable).resolve().parent.parent))
    log_path = output / "solver.log"
    print(f"Starting Elmer; live output below is also saved to {log_path}", flush=True)
    with log_path.open("w") as log:
        worker = subprocess.Popen(
            [launcher, option, str(partitions), executable,
             report.get("solver_input", "diagnostic.sif")], cwd=output,
            env=environment, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            code = wait_for_solver(worker, log_path, timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(worker.pid), "/T", "/F"], check=True)
            raise RuntimeError("Solver exceeded the requested time limit; inspect solver.log")
        except KeyboardInterrupt:
            subprocess.run(["taskkill", "/PID", str(worker.pid), "/T", "/F"], check=True)
            raise
    if code != 0:
        raise RuntimeError(f"Elmer exited {code}; inspect {output / 'solver.log'}")
    result = output / "results" / f"{report.get('result_name', 'three_body_diagnostic')}_t0001.pvtu"
    if not result.is_file() and not (partitions == 1 and result.with_suffix(".vtu").is_file()):
        raise RuntimeError("Elmer did not produce its expected PVTU")
    attribute_result(output, result, timeout=None if timeout is None else 90)
    gaps = measure_joint_gaps(
        result, timeout=None if timeout is None else 120,
        joint_levels_m=[joint["z_m"] for joint in report.get("wall_joints", [])],
    )
    if not gaps["finite_displacement"]:
        raise RuntimeError("Solved displacement contains non-finite values")
    if not gaps["finite_stress"]:
        raise RuntimeError("Solved stress contains non-finite values")
    if spring:
        check_spring_gaps(gaps, report["interface_stiffness_n_per_m3"])
    for interface in ("wall_bedrock", "wall_foundation", "foundation_bedrock"):
        if report.get(f"{interface}_connection") == "bonded":
            seat = gaps["joints"][interface]
            if (seat["locations"] <= 0 or not np.isfinite(seat["max_abs_gap_m"])
                    or seat["max_abs_gap_m"] > 1e-10):
                raise RuntimeError(f"Bonded {interface.replace('_', '/')} displacement continuity check failed")
    report["result"] = str(result)
    report["solver_exit_code"] = code
    report["partitions"] = partitions
    report["joint_gaps"] = gaps
    report_path.write_text(json.dumps(report, indent=2))
    return result


def wait_for_solver(
    worker: subprocess.Popen, log_path: Path, timeout: int | None,
) -> int:
    """Relay solver output and heartbeats without imposing a default run limit."""
    started = time.monotonic()
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        while True:
            elapsed = time.monotonic() - started
            if timeout is not None and elapsed >= timeout:
                raise subprocess.TimeoutExpired(worker.args, timeout)
            interval = 30. if timeout is None else min(30., timeout - elapsed)
            try:
                code = worker.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                text = stream.read()
                if text:
                    print(text, end="" if text.endswith("\n") else "\n", flush=True)
                print(f"Elmer running: {time.monotonic() - started:.0f} s elapsed"
                      " (no automatic timeout unless requested).", flush=True)
            else:
                text = stream.read()
                if text:
                    print(text, end="" if text.endswith("\n") else "\n", flush=True)
                return code


def check_spring_gaps(gaps: dict, stiffness: dict[str, dict[str, float]]) -> None:
    """Reject the silent-garbage pattern: missing joints or implausible openings.

    Check each global-axis traction using its configured stiffness. The 100 MPa
    component ceiling is a plausibility check, not an engineering acceptance limit.
    """
    for name in stiffness:
        joint = gaps["joints"][name]
        if joint["locations"] == 0:
            raise RuntimeError(f"No {name} joint locations found in the result")
        maximum = np.asarray(joint["max_abs_gap_by_axis_m"], dtype=float)
        if maximum.shape != (3,) or not np.isfinite(maximum).all() or np.any(maximum < 0):
            raise RuntimeError(f"Invalid {name} directional joint gaps")
        coefficients = np.array([stiffness[name][axis] for axis in ("x", "y", "z")])
        if np.any(maximum * coefficients > 100e6):
            raise RuntimeError(f"{name} joint gap implies >100 MPa traction; result rejected")


def pvpython_executable() -> str:
    executable = shutil.which("pvpython")
    if executable is None and os.name == "nt":
        candidates = sorted(Path(os.environ.get("ProgramFiles", r"C:\Program Files")).glob(
            r"ParaView*\bin\pvpython.exe",
        ))
        if candidates:
            executable = str(candidates[-1])
    if executable is None:
        raise RuntimeError("ParaView pvpython is required to post-process the diagnostic result")
    return executable


def measure_joint_gaps(
    result: Path, timeout: int | None = 120, joint_levels_m: list[float] | None = None,
) -> dict[str, object]:
    """Seat gaps, plus wall_joint_k gaps at each wall block joint elevation (m)."""
    completed = subprocess.run(
        [pvpython_executable(), str(Path(__file__).with_name("measure_joint_gaps.py")), str(result),
         *([f"levels={','.join(f'{level:.12g}' for level in joint_levels_m)}"] if joint_levels_m else [])],
        check=True, timeout=timeout, capture_output=True, text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def attribute_result(output: Path, result: Path, timeout: int | None = 90) -> None:
    """Attach body labels by exact source-cell coordinates, not spatial guesses."""
    subprocess.run(
        [pvpython_executable(), str(Path(__file__).with_name("attribute_diagnostic_result.py")),
         str(output), str(result)], check=True, timeout=timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream-margin-m", type=float, help="Override the configured stream margin (m)")
    parser.add_argument("--bank-margin-m", type=float, help="Override the configured bank margin (m)")
    parser.add_argument("--bottom-z-m", type=float, help="Override the configured rock bottom elevation (m)")
    parser.add_argument("--joints", choices=JOINT_MODELS, default="bonded")
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=480)
    args = parser.parse_args()
    prepare(
        args.root, args.output, args.stream_margin_m, args.bank_margin_m,
        args.joints, args.bottom_z_m,
    )
    print(solve(args.output, args.timeout_s, args.partitions))


if __name__ == "__main__":
    main()
