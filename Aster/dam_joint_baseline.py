"""Cracking baseline for the dam, following Code_Aster's own dam-joint test SSNP142.

SSNP142 ("filling of a dam", share/aster/tests/ssnp142b.comm) is EDF's validation
case for the dam joint laws of R7.01.25: a dam on its ground with a layer of joint
elements at the contact, self-weight applied in full, the reservoir raised in
steps, solved dry and with water pressure in the joint. This module transfers that
recipe to the dam model and changes only what the dam forces:

* the joint layer replaces the linear springs at the wall / concrete-foundation
  contact (the contact is already meshed on both sides with matching triangles);
* volumes stay linear tetrahedra and the joints are prism (PENTA6) ``3D_JOINT``
  cells with ``JOINT_MECA_RUPT``, the element types of ssnp142b itself;
* nonlinear solver settings are those of ssnp142b.

Every departure from the test is listed in ``DEPARTURES`` and written into each
run's report. Nothing here is engineering-validated: a converged run shows how
this model responds, not that the dam is safe or unsafe.

Run from the Gairezi directory (PowerShell)::

    .venv\\Scripts\\python.exe -m Aster.dam_joint_baseline fixture
    .venv\\Scripts\\python.exe -m Aster.dam_joint_baseline run reference_dry --job <job directory>
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_ROOT = ROOT / "Aster" / "results" / "dam_joint_baseline"
ARRAYS_NAME = "three_body_mesh_arrays.npz"
MANIFEST_NAME = "mesh_manifest.json"
CASE_NAME = "case.comm"
META_NAME = "joint_case.json"
STEPS_NAME = "joint_steps.npz"
STATUS_NAME = "joint_status.json"
REPORT_NAME = "joint_report.json"
# The app's stage: these cases in this order, summarised together in STAGE_REPORT_NAME.
STAGE_CASES = (("dry", "friction_configured_dry"), ("water_in_joint", "friction_configured_uplift"))
STAGE_REPORT_NAME = "joint_opening_report.json"
# R7.01.25, table "Valeur conseillee pour le barrage en beton".
RECOMMENDED_K_N_PA_PER_M = 3.0e12
RECOMMENDED_SIGM_MAX_PA = 5.0e4
# The same table gives 1 as the default rupture penalty; Code_Aster 17.4 requires it to be given.
RECOMMENDED_PENA_RUPTURE = 1.0
# ssnp142b.comm: zero tensile strength with this rupture penalty.
TEST_PENA_RUPTURE = 0.2
# R7.01.25, same table, for the friction law JOINT_MECA_FROT (the law of ssnp142d): friction
# coefficient 1 and a tangential hardening of (K_N + K_T) * 1e-6.
RECOMMENDED_FRICTION = 1.0
RECOMMENDED_PENA_TANG_FACTOR = 1.0e-6
# The table's default adhesion is 0, but a joint with no adhesion has no stiffness in its
# unloaded state and the first solve is singular (checked on the two-block case). ssnp142d
# uses 1000 Pa; with friction 1 that is a tensile strength of 1 kPa.
TEST_ADHESION_PA = 1000.0
# ssnp142d.comm hardens a sliding joint by a tenth of its tangential stiffness ("ecrouissage").
TEST_PENA_TANG_FACTOR = 0.1
# A joint that never breaks, for the control run (peak elastic contact tension is about 1 MPa).
UNBREAKABLE_SIGM_MAX_PA = 0.3e6
# Reservoir filling: weight alone at instant 1, then the head raised in equal steps.
FILL_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
# PROVISIONAL, not site data. Thermal expansion of concrete: the generic 1e-5 per kelvin used
# where no measurement exists.
PROVISIONAL_EXPANSION_PER_K = 1.0e-5
# Closure temperature of the wall: the temperature at which it is taken to be free of thermal
# stress. Set by the project owner on 12 November 2026; the project holds no construction record.
CLOSURE_TEMPERATURE_C = 16.0
# Lowest figure of the climate statistics for Nyanga quoted by the project owner (8 minimum,
# 26 maximum, degrees Celsius). It is an air temperature, so it bounds how cold the wall can get.
COLDEST_AIR_TEMPERATURE_C = 8.0
# Wall temperatures solved once the reservoir is full, from the closure temperature down to
# that figure. Each is about a minute of solving on the dam; three fit a ten-minute limit with
# water in the joint.
WALL_TEMPERATURES_C = (12.0, 10.0, COLDEST_AIR_TEMPERATURE_C)


def cooling_note(cooling: dict) -> str:
    """What the temperature load of a run is, and which of its inputs are not established."""
    return (
        "Temperature load, not part of ssnp142: once the reservoir is full the wall alone is cooled uniformly "
        f"through its thickness, in steps, from its closure temperature of {cooling['closure_temperature_c']:g} C "
        f"to {cooling['wall_temperatures_c'][-1]:g} C. The closure temperature is an input set by the project "
        "owner, not a construction record. Whether the wall as a whole reaches the lowest temperature is not "
        f"established. PROVISIONAL: the expansion coefficient ({cooling['expansion_per_k']:g} per kelvin) is the "
        "generic value for concrete. The foundation keeps its temperature and no gradient through the wall is "
        "applied.")


OPEN_TOLERANCE_M = 1.0e-6
MEMORY_LIMIT_MB = 8192

CASES = {
    "control": {
        "purpose": "Joint elements with the configured spring stiffness and a joint that cannot break: "
                   "must reproduce the elastic baseline, so the element swap itself changes nothing",
        "stiffness": "configured", "sigm_max_pa": UNBREAKABLE_SIGM_MAX_PA, "pena_rupture": RECOMMENDED_PENA_RUPTURE,
        "uplift": False},
    "reference_dry": {
        "purpose": "Code_Aster's recommended joint values for a concrete dam (R7.01.25), no water in the joint",
        "stiffness": RECOMMENDED_K_N_PA_PER_M, "sigm_max_pa": RECOMMENDED_SIGM_MAX_PA,
        "pena_rupture": RECOMMENDED_PENA_RUPTURE, "uplift": False},
    "reference_uplift": {
        "purpose": "Code_Aster's recommended joint values with water pressure in the joint, as ssnp142b",
        "stiffness": RECOMMENDED_K_N_PA_PER_M, "sigm_max_pa": RECOMMENDED_SIGM_MAX_PA,
        "pena_rupture": RECOMMENDED_PENA_RUPTURE, "uplift": True},
    "selected_dry": {
        "purpose": "Zero tensile strength as ssnp142b, with the stiffness configured in config.json, no water in the joint",
        "stiffness": "configured", "sigm_max_pa": 0.0, "pena_rupture": TEST_PENA_RUPTURE, "uplift": False},
    "selected_uplift": {
        "purpose": "Zero tensile strength as ssnp142b, configured stiffness, water pressure in the joint",
        "stiffness": "configured", "sigm_max_pa": 0.0, "pena_rupture": TEST_PENA_RUPTURE, "uplift": True},
    # The friction law of ssnp142d: no tension (zero adhesion) and Coulomb friction, so the shear a
    # closed contact carries falls continuously to zero as it opens.
    "friction_reference_dry": {
        "purpose": "Friction law with Code_Aster's recommended values for a concrete dam (R7.01.25), no water in the joint",
        "law": "JOINT_MECA_FROT", "stiffness": RECOMMENDED_K_N_PA_PER_M, "friction_coefficient": RECOMMENDED_FRICTION,
        "adhesion_pa": TEST_ADHESION_PA, "uplift": False},
    "friction_reference_uplift": {
        "purpose": "Friction law with Code_Aster's recommended values, water pressure in the joint",
        "law": "JOINT_MECA_FROT", "stiffness": RECOMMENDED_K_N_PA_PER_M, "friction_coefficient": RECOMMENDED_FRICTION,
        "adhesion_pa": TEST_ADHESION_PA, "uplift": True},
    "friction_configured_dry": {
        "purpose": "Friction law with the stiffness configured in config.json and the sliding hardening of "
                   "ssnp142d, no water in the joint",
        "law": "JOINT_MECA_FROT", "stiffness": "configured", "friction_coefficient": RECOMMENDED_FRICTION,
        "adhesion_pa": TEST_ADHESION_PA, "pena_tang_factor": TEST_PENA_TANG_FACTOR, "uplift": False},
    "friction_configured_uplift": {
        "purpose": "Friction law with the configured stiffness and the sliding hardening of ssnp142d, "
                   "water pressure in the joint",
        "law": "JOINT_MECA_FROT", "stiffness": "configured", "friction_coefficient": RECOMMENDED_FRICTION,
        "adhesion_pa": TEST_ADHESION_PA, "pena_tang_factor": TEST_PENA_TANG_FACTOR, "uplift": True},
    # Not part of ssnp142 and not run by the app: the two cases above, then the wall cooled at full reservoir.
    "friction_configured_dry_cooling": {
        "purpose": "As friction_configured_dry, then at full reservoir the wall cooled uniformly in steps below "
                   "its closure temperature (expansion coefficient PROVISIONAL, not site data)",
        "law": "JOINT_MECA_FROT", "stiffness": "configured", "friction_coefficient": RECOMMENDED_FRICTION,
        "adhesion_pa": TEST_ADHESION_PA, "pena_tang_factor": TEST_PENA_TANG_FACTOR, "uplift": False,
        "cooling": {"expansion_per_k": PROVISIONAL_EXPANSION_PER_K, "closure_temperature_c": CLOSURE_TEMPERATURE_C,
                    "wall_temperatures_c": list(WALL_TEMPERATURES_C)}},
    "friction_configured_uplift_cooling": {
        "purpose": "As friction_configured_uplift, then at full reservoir the wall cooled uniformly in steps below "
                   "its closure temperature (expansion coefficient PROVISIONAL, not site data)",
        "law": "JOINT_MECA_FROT", "stiffness": "configured", "friction_coefficient": RECOMMENDED_FRICTION,
        "adhesion_pa": TEST_ADHESION_PA, "pena_tang_factor": TEST_PENA_TANG_FACTOR, "uplift": True,
        "cooling": {"expansion_per_k": PROVISIONAL_EXPANSION_PER_K, "closure_temperature_c": CLOSURE_TEMPERATURE_C,
                    "wall_temperatures_c": list(WALL_TEMPERATURES_C)}},
}

DEPARTURES = (
    "Joint cells have zero thickness (the contact is meshed with coincident nodes); ssnp142b's are 1 mm thick. "
    "Their node order is set explicitly, so ORIE_FISSURE is not used.",
    "Weight alone is solved first, then the reservoir is raised in four equal steps of head; ssnp142b applies "
    "weight and the first water level in the same step. The weight is ramped up during that first instant "
    "(and is constant afterwards) so that Code_Aster's step cutting can reduce the increment; a constant "
    "weight, as in ssnp142b, makes a cut step identical to the full one.",
    "Each water level is its own STAT_NON_LINE call continuing the previous one, so a level that does not "
    "converge still leaves the earlier levels as results; ssnp142b uses one call.",
    "Water pressure in the joint varies across the base from reservoir head at the upstream edge to tailwater "
    "head at the downstream edge at each elevation; ssnp142b's varies along one axis to zero.",
    "The concrete foundation rests on the existing distributed bedrock springs, which still hold tension; "
    "ssnp142b's ground is a meshed block with a fixed boundary.",
    "The spring cells of the elastic baseline stay in the mesh file but are given no model, so they carry nothing.",
    "A wall with thickness steps is meshed in parts with horizontal joints between them. Those joints keep the "
    "linear springs of the elastic baseline, which hold tension; only the wall/foundation contact can open.",
    "Friction-law runs use the manual's friction coefficient of 1 with ssnp142d's adhesion of 1000 Pa "
    "(a 1 kPa tensile strength); the manual's default adhesion of 0 makes the first solve singular.",
)


class JointCaseError(RuntimeError):
    """The joint case cannot be prepared from these inputs."""


# ------------------------------------------------------------------ progress

# A cracking run that needs longer than this is not worth having (project decision, 11 Nov 2026).
MAXIMUM_TIME_LIMIT_S = 1200
_progress_log: Path | None = None
_progress_echo = False
_NEWTON_ROW = re.compile(r"^\|\s+(\d+)\s+[XE]?\s*\|\s*([0-9.]+E[+-]\d+)")
_STEP_LINE = re.compile(r"^DAM_JOINT_STEP (start|done|failed|outoftime) instant=(\S+)(?: seconds=(\S+))?")
_CUT_LINE = re.compile(r"^Instant de calcul:\s*(\S+) - Niveau de d.coupe:\s*(\d+)")


def set_progress_log(path: Path | None, echo: bool = False) -> None:
    """Append a timestamped line to ``path`` for every stage of a run (None switches it off).

    With ``echo`` the same lines go to standard output, which is the stage log when the app runs this.
    """
    global _progress_log, _progress_echo
    _progress_log, _progress_echo = path, echo


def progress(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {message}"
    if _progress_log is not None:
        with _progress_log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    if _progress_echo:
        print(line, flush=True)


def progress_message(line: str, levels: dict) -> str | None:
    """What a line of Code_Aster's live output means for someone watching the run, or None."""
    line = line.strip()
    step = _STEP_LINE.match(line)
    if step:
        state, instant, seconds = step.group(1), float(step.group(2)), step.group(3)
        index = levels["instants"].index(instant)
        cooled = -(levels.get("temperature_change_k") or [0.0] * len(levels["instants"]))[index]
        label = f"level {index + 1} of {len(levels['instants'])} " + (
            f"(full reservoir, wall at {levels['wall_temperature_c'][index]:g} C, {cooled:g} K below closure)"
            if cooled else
            f"(reservoir at {levels['upstream_level_m'][index]:.2f} m, {levels['fill_fraction'][index]:.0%} of the head)")
        if state == "start":
            return f"Solving {label}" + (": self-weight only" if index == 0 else "")
        if state == "done":
            return f"Converged {label} in {float(seconds):.0f} s"
        if state == "outoftime":
            return f"Ran out of the time limit at {label}; the levels before it are kept"
        return f"Did NOT converge at {label}"
    cut = _CUT_LINE.match(line)
    if cut and int(cut.group(2)) > 1:
        return f"    step cut (depth {cut.group(2)}): trying up to instant {float(cut.group(1)):g}"
    row = _NEWTON_ROW.match(line)
    if row:
        return f"    Newton iteration {row.group(1)}: relative residual {float(row.group(2)):.1e} (target 1e-05)"
    return None


def _follow(path: Path, finished: threading.Event, levels: dict) -> None:
    """Copy the meaningful lines of the solver's live log into the progress log while it runs."""
    position, pending = 0, ""
    while True:
        last_pass = finished.is_set()
        if path.is_file():
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(position)
                pending += stream.read()
                position = stream.tell()
            *lines, pending = pending.split("\n")
            for line in lines:
                message = progress_message(line, levels)
                if message:
                    progress(message)
        if last_pass:
            return
        finished.wait(1.0)


# ------------------------------------------------------------------ mesh

_TET_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


def _body_nodes(arrays: dict) -> tuple[np.ndarray, np.ndarray]:
    """Which nodes belong to wall tetrahedra and which to foundation tetrahedra."""
    cells = np.asarray(arrays["tetra4_concrete"], dtype=np.int64)
    is_wall = np.asarray(arrays["concrete_body_ids"]) == 1
    in_wall = np.zeros(len(arrays["nodes"]), dtype=bool)
    in_foundation = np.zeros(len(arrays["nodes"]), dtype=bool)
    in_wall[cells[is_wall]] = True
    in_foundation[cells[~is_wall]] = True
    return in_wall, in_foundation


def contact_sides(arrays: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The baseline's matched contact triangles (wall side, other side) and which of them lie on the foundation.

    A wall with thickness steps is meshed in parts. The horizontal joints between the parts are
    listed with the wall/foundation contact, but have wall concrete on both sides.
    """
    nodes = np.asarray(arrays["nodes"], dtype=float)
    wall = np.asarray(arrays["joint_slave_tri3"], dtype=np.int64)
    other = np.asarray(arrays["joint_master_tri3"], dtype=np.int64)
    if wall.shape != other.shape or wall.ndim != 2 or wall.shape[1] != 3 or not len(wall):
        raise JointCaseError("The baseline mesh has no matched wall/foundation contact triangles")
    if not np.allclose(nodes[wall], nodes[other], atol=1e-9, rtol=0.0):
        raise JointCaseError("Wall and foundation contact triangles do not coincide vertex by vertex")
    in_wall, in_foundation = _body_nodes(arrays)
    # A mortar fill on the wall's steps is tied to the wall face it covers: wall on one side, fill on the other.
    in_fill = np.zeros(len(nodes), dtype=bool)
    if "tetra4_mortar" in arrays:
        in_fill[np.asarray(arrays["tetra4_mortar"], dtype=np.int64)] = True
    on_foundation = in_foundation[other].all(axis=1)
    beside_wall = other[~on_foundation]
    if not in_wall[wall].all() or not (in_wall[beside_wall].all(axis=1) | in_fill[beside_wall].all(axis=1)).all():
        raise JointCaseError("A wall contact triangle is not a face of a wall tetrahedron")
    if not on_foundation.any():
        raise JointCaseError("The baseline mesh has no matched wall/foundation contact triangles")
    return wall, other, on_foundation


def joint_cells(arrays: dict) -> np.ndarray:
    """One PENTA6 row per matched contact triangle: foundation triangle, then wall triangle.

    Code_Aster's 3D_JOINT reports an opening as a positive normal jump when the
    first triangle's right-hand normal points away from the second side (the same
    convention as Aster.automatic_cohesive, checked here by the two-block run).
    """
    nodes = np.asarray(arrays["nodes"], dtype=float)
    wall, other, on_foundation = contact_sides(arrays)
    wall, foundation = wall[on_foundation], other[on_foundation]
    if np.intersect1d(wall, foundation).size:
        raise JointCaseError("Wall and foundation contact triangles share nodes")
    cells = np.asarray(arrays["tetra4_concrete"], dtype=np.int64)
    body = np.asarray(arrays["concrete_body_ids"])
    wall_cells = cells[body == 1]
    base = len(nodes) + 1

    def codes(triples: np.ndarray) -> np.ndarray:
        ordered = np.sort(triples, axis=1)
        return (ordered[:, 0] * base + ordered[:, 1]) * base + ordered[:, 2]

    face_codes = codes(wall_cells[:, _TET_FACES].reshape(-1, 3))
    order = np.argsort(face_codes, kind="stable")
    position = np.searchsorted(face_codes[order], codes(wall))
    if np.any(position >= len(order)) or np.any(face_codes[order][np.minimum(position, len(order) - 1)] != codes(wall)):
        raise JointCaseError("A wall contact triangle is not a face of a wall tetrahedron")
    owners = wall_cells[order[position] // 4]
    a, b, c = (nodes[foundation[:, k]] for k in range(3))
    normal = np.cross(b - a, c - a)
    toward_wall = nodes[owners].mean(axis=1) - (a + b + c) / 3.0
    flip = np.einsum("ij,ij->i", toward_wall, normal) > 0
    foundation, wall = foundation.copy(), wall.copy()
    foundation[flip] = foundation[flip][:, [0, 2, 1]]
    wall[flip] = wall[flip][:, [0, 2, 1]]
    return np.hstack([foundation, wall])


def joint_geometry(arrays: dict, cells: np.ndarray) -> dict:
    """Area, centroid and unit normal (pointing from the foundation into the wall) of each joint cell."""
    nodes = np.asarray(arrays["nodes"], dtype=float)
    a, b, c = (nodes[cells[:, k]] for k in range(3))
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1) / 2.0
    if np.any(area <= 0):
        raise JointCaseError("A contact triangle has no area")
    return {"area_m2": area, "centroid_m": (a + b + c) / 3.0, "normal": -cross / (2.0 * area[:, None])}


def configured_stiffness(arrays: dict) -> float:
    """The contact stiffness per area configured for the elastic baseline (N/m3), if it is the same in all directions."""
    coefficients = np.asarray(arrays["joint_coefficients_n_per_m3"], dtype=float)
    on_foundation = arrays.get("joint_on_foundation")
    if on_foundation is not None and len(on_foundation) == len(coefficients):
        coefficients = coefficients[np.asarray(on_foundation, dtype=bool)]
    values = np.unique(coefficients)
    if len(values) != 1 or not np.isfinite(values[0]) or values[0] <= 0:
        raise JointCaseError(
            "The configured wall/foundation spring stiffness differs by direction or location; "
            "a joint needs one normal and one tangential value")
    return float(values[0])


_SPRING_PAIR_KEYS = ("seg2_wall_foundation_springs", "wall_foundation_tributary_area_m2",
                     "interface_stiffness_n_per_m", "interface_names")


def with_joints(arrays: dict) -> dict:
    """The mesh arrays of a joint run: joint cells on the wall/foundation contact.

    Spring pairs between the parts of a stepped wall are kept, and are then the only spring pairs
    left: the pairs of the wall/foundation contact are replaced by the joint cells. The pairs that
    tie a mortar fill to the wall face it covers are kept in the same way.
    """
    result = {key: np.asarray(value) for key, value in arrays.items()}
    result["penta6_joints"] = joint_cells(arrays)
    result["joint_on_foundation"] = contact_sides(arrays)[2]
    pairs = np.asarray(arrays["seg2_wall_foundation_springs"], dtype=np.int64)
    internal = ~_body_nodes(arrays)[1][pairs].any(axis=1)
    result["internal_wall_joint_pairs"] = np.asarray(int(internal.sum()))
    if "tetra4_mortar" in result:
        fill_nodes = np.zeros(len(result["nodes"]), dtype=bool)
        fill_nodes[np.asarray(result["tetra4_mortar"], dtype=np.int64)] = True
        result["mortar_fill_tie_pairs"] = np.asarray(int((internal & fill_nodes[pairs].any(axis=1)).sum()))
        result["joint_on_mortar_fill"] = fill_nodes[np.asarray(arrays["joint_master_tri3"], dtype=np.int64)].all(axis=1)
    if internal.any():
        for key in _SPRING_PAIR_KEYS:
            if key in result:
                result[key] = result[key][internal]
    return result


def internal_joint_spring_lines(arrays: dict) -> list[str]:
    """Case-file lines giving the springs between the parts of a stepped wall their baseline stiffness.

    ``arrays`` are those of the joint run (see with_joints). The group names follow
    Aster.convert_three_body_to_med: one group per spring, or one per stiffness value when compact.
    """
    if not int(arrays.get("internal_wall_joint_pairs", 0)):
        return []
    groups: dict[tuple, list[str]] = {}
    for row, stiffness in enumerate(np.asarray(arrays["interface_stiffness_n_per_m"], dtype=float)):
        groups.setdefault(tuple(float(value) for value in stiffness), []).append(f"SPR_WF_{row:05d}")
    compact = bool(arrays.get("compact_spring_groups", False))
    lines = []
    for index, (value, names) in enumerate(groups.items()):
        group = f"K_IF_{index:05d}" if compact else names[0] if len(names) == 1 else tuple(names)
        lines.append(f"        _F(GROUP_MA={group!r}, REPERE='GLOBAL', CARA='K_T_D_L', "
                     f"VALE=({value[0]:.12g}, {value[1]:.12g}, {value[2]:.12g})),")
    return lines


# ------------------------------------------------------------------ water in the joint

def _fit_circle(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    matrix = np.column_stack([2 * points, np.ones(len(points))])
    solution, *_ = np.linalg.lstsq(matrix, (points ** 2).sum(axis=1), rcond=None)
    centre = solution[:2]
    radius = math.sqrt(solution[2] + centre @ centre)
    return centre, radius, float(np.sqrt(np.mean((np.linalg.norm(points - centre, axis=1) - radius) ** 2)))


# Node angles closer than this (radians) lie on the same radial line of the contact mesh.
STRIP_ANGLE_TOLERANCE = 1.0e-6


def _strip_position(arrays: dict, cells: np.ndarray, centre: np.ndarray, r_up: float, tolerance: float) -> dict | None:
    """The position rule of a wall whose thickness is not uniform, read from the contact itself, or None.

    The contact is meshed in radial strips between the upstream and the downstream face. Each
    strip's own downstream end is the downstream edge of the contact there, whatever the shape of
    the wall above it (thickness steps, with or without tapered step ends).
    """
    nodes = np.asarray(arrays["nodes"], dtype=float)
    offset = nodes[cells[:, :3]][..., :2] - centre
    reference = offset.reshape(-1, 2).mean(axis=0)
    if np.linalg.norm(reference) < 1.0e-9:
        return None
    reference = reference / np.linalg.norm(reference)
    angle = np.arctan2(reference[0] * offset[..., 1] - reference[1] * offset[..., 0],
                       reference[0] * offset[..., 0] + reference[1] * offset[..., 1])
    radius = np.linalg.norm(offset, axis=2)
    ordered = np.sort(angle.ravel())
    lines = ordered[np.concatenate([[True], np.diff(ordered) > STRIP_ANGLE_TOLERANCE])]
    if len(lines) < 2 or lines[-1] - lines[0] >= math.pi:
        return None
    strip = np.searchsorted(lines, angle.mean(axis=1)) - 1
    if strip.min() < 0 or strip.max() > len(lines) - 2:
        return None
    # Every cell lies between two neighbouring radial lines.
    if (np.any(angle < lines[strip][:, None] - STRIP_ANGLE_TOLERANCE)
            or np.any(angle > lines[strip + 1][:, None] + STRIP_ANGLE_TOLERANCE)):
        return None
    inner = np.full(len(lines) - 1, np.inf)
    outer = np.zeros(len(lines) - 1)
    np.minimum.at(inner, strip, radius.min(axis=1))
    np.maximum.at(outer, strip, radius.max(axis=1))
    if not np.all(np.isfinite(inner)) or np.any(np.abs(outer - r_up) > tolerance) or np.any(r_up - inner < 0.5):
        return None
    # Each strip must end on the downstream face.
    face = np.linalg.norm(nodes[np.unique(arrays["tri3_downstream"])][:, :2] - centre, axis=1)
    ends = np.unique(np.round(inner, 6))
    if np.any(np.abs(face[:, None] - ends[None, :]).min(axis=0) > tolerance):
        return None
    starts = np.concatenate([[0], np.flatnonzero(np.abs(np.diff(inner)) > tolerance) + 1])
    values = [float(np.median(group)) for group in np.split(inner, starts[1:])]
    return {"kind": "radial_strips", "centre_xy_m": [float(centre[0]), float(centre[1])],
            "upstream_radius_m": float(r_up), "reference_direction": [float(reference[0]), float(reference[1])],
            "downstream_radius_m": values[0],
            "changes": [{"angle_rad": float(lines[start]), "downstream_radius_m": value}
                        for start, value in zip(starts[1:], values[1:])]}


def arch_position(arrays: dict, cells: np.ndarray) -> dict:
    """Where a point of the contact lies between the upstream and downstream edges of a circular arch wall."""
    nodes = np.asarray(arrays["nodes"], dtype=float)
    upstream = nodes[np.unique(arrays["tri3_upstream"])][:, :2]
    downstream = nodes[np.unique(arrays["tri3_downstream"])][:, :2]
    centre, r_up, misfit = _fit_circle(upstream)
    r_down_all = np.linalg.norm(downstream - centre, axis=1)
    r_down = float(np.median(r_down_all))
    joint_r = np.linalg.norm(nodes[np.unique(cells)][:, :2] - centre, axis=1)
    tolerance = 0.05
    if misfit <= 0.02 and np.ptp(r_down_all) > tolerance:
        strips = _strip_position(arrays, cells, centre, r_up, tolerance)
        if strips is not None:
            return strips
    if (misfit > 0.02 or np.ptp(r_down_all) > tolerance or r_up - r_down < 0.5
            or joint_r.min() < r_down - tolerance or joint_r.max() > r_up + tolerance):
        raise JointCaseError(
            "Water pressure in the joint is implemented for a wall on a circular arc, of uniform thickness or "
            "with a contact meshed in radial strips from the upstream to the downstream face; "
            f"this mesh does not fit (upstream misfit {misfit:.3f} m, downstream radius spread "
            f"{np.ptp(r_down_all):.3f} m, contact radii {joint_r.min():.2f} to {joint_r.max():.2f} m "
            f"against {r_down:.2f} to {r_up:.2f} m)")
    return {"kind": "radial", "centre_xy_m": [float(centre[0]), float(centre[1])],
            "upstream_radius_m": float(r_up), "downstream_radius_m": r_down}


def position_expression(position: dict) -> str:
    """Formula text for the fraction of the way from the upstream (0) to the downstream (1) edge."""
    if position["kind"] == "radial":
        xc, yc = position["centre_xy_m"]
        r_up, r_down = position["upstream_radius_m"], position["downstream_radius_m"]
        raw = f"({r_up!r} - sqrt((X - ({xc!r}))**2 + (Y - ({yc!r}))**2)) / {r_up - r_down!r}"
    elif position["kind"] == "radial_strips":
        xc, yc = position["centre_xy_m"]
        ux, uy = position["reference_direction"]
        r_up = position["upstream_radius_m"]
        # Angle around the arch centre, measured from the middle of the contact.
        angle = (f"atan2(({ux!r}) * (Y - ({yc!r})) - ({uy!r}) * (X - ({xc!r})), "
                 f"({ux!r}) * (X - ({xc!r})) + ({uy!r}) * (Y - ({yc!r})))")
        # The downstream radius of the first strip, plus each change beyond the angle where it happens.
        r_down, previous = repr(position["downstream_radius_m"]), position["downstream_radius_m"]
        for change in position["changes"]:
            r_down += f" + ({change['downstream_radius_m'] - previous!r}) * ({angle} > {change['angle_rad']!r})"
            previous = change["downstream_radius_m"]
        raw = f"({r_up!r} - sqrt((X - ({xc!r}))**2 + (Y - ({yc!r}))**2)) / ({r_up!r} - ({r_down}))"
    elif position["kind"] == "x":
        raw = f"(X - ({position['upstream_x_m']!r})) / {position['downstream_x_m'] - position['upstream_x_m']!r}"
    else:
        raise JointCaseError(f"Unknown joint position rule {position['kind']!r}")
    return f"min(max({raw}, 0.0), 1.0)"


def joint_pressure(position: dict, gradient: float, upstream_level: float, downstream_level: float,
                   points: np.ndarray) -> np.ndarray:
    """The joint water pressure the case formula gives at ``points`` (Pa), for checks and reports."""
    x, y, z = np.asarray(points, dtype=float).T
    fraction = eval(position_expression(position), {"sqrt": np.sqrt, "atan2": np.arctan2, "min": np.minimum,
                                                    "max": np.maximum, "X": x, "Y": y, "Z": z})
    return ((1.0 - fraction) * np.maximum(gradient * (upstream_level - z), 0.0)
            + fraction * np.maximum(gradient * (downstream_level - z), 0.0))


# ------------------------------------------------------------------ case file

_ANCHORS = {
    "debut": "DEBUT(LANG='EN')",
    "springs_model": "        _F(GROUP_MA='WALL_FOUNDATION_SPRINGS', PHENOMENE='MECANIQUE', MODELISATION='DIS_T'),\n",
    "chmat": "CHMAT = AFFE_MATERIAU(",
    "concrete_mat": "        _F(GROUP_MA='CONCRETE', MATER=CONCRETE_MAT),\n",
    "solve": "RESU = MECA_STATIQUE(",
}


def water_levels(loads: dict, base_z: float, cooling_steps_k: tuple | list = (),
                 closure_temperature_c: float | None = None) -> dict:
    """Reservoir and tailwater level at each instant: instant 1 is weight alone.

    Each cooling step adds one instant at full reservoir, with the wall that many kelvin below
    its closure temperature. Until then the wall is at its closure temperature.
    """
    fractions = list(FILL_FRACTIONS) + [FILL_FRACTIONS[-1]] * len(cooling_steps_k)
    instants = [float(i + 1) for i in range(len(fractions))]
    upstream = [base_z + f * (loads["upstream_water_surface_z_m"] - base_z) for f in fractions]
    downstream = [base_z + f * max(loads["tailwater_elevation_m"] - base_z, 0.0) for f in fractions]
    levels = {"instants": instants, "fill_fraction": fractions, "upstream_level_m": upstream,
              "downstream_level_m": downstream, "base_z_m": base_z}
    if cooling_steps_k:
        if closure_temperature_c is None:
            raise JointCaseError("Cooling steps need the closure temperature of the wall")
        levels["temperature_change_k"] = [0.0] * len(FILL_FRACTIONS) + [-float(step) for step in cooling_steps_k]
        levels["closure_temperature_c"] = float(closure_temperature_c)
        levels["wall_temperature_c"] = [float(closure_temperature_c) + change
                                        for change in levels["temperature_change_k"]]
    return levels


_CONCRETE_ELASTICITY = re.compile(r"CONCRETE_MAT = DEFI_MATERIAU\(\s*ELAS=_F\((.*?)\),?\s*\)\n", re.S)


def cooling_sections(comm: str, cooling: dict, levels: dict) -> tuple[str, str, str]:
    """Case-file text for the wall's cooling: definitions, the wall's material line and the AFFE_VARC line."""
    temperatures = levels.get("wall_temperature_c")
    if not temperatures or len(temperatures) != len(levels["instants"]):
        raise JointCaseError("A cooling case needs a wall temperature for every instant")
    closure = float(cooling["closure_temperature_c"])
    elastic = _CONCRETE_ELASTICITY.search(comm)
    if elastic is None or "ALPHA" in elastic.group(1):
        raise JointCaseError("The baseline case no longer has its 'concrete material' section")
    properties = ", ".join(part.strip() for part in elastic.group(1).split(",") if part.strip())
    values = sorted(set(temperatures), reverse=True)
    name = {value: f"TEMP_{index}" for index, value in enumerate(values)}
    fields = "".join(
        f"{name[value]} = CREA_CHAMP(TYPE_CHAM='NOEU_TEMP_R', OPERATION='AFFE', MAILLAGE=MAIL,\n"
        f"    AFFE=_F(TOUT='OUI', NOM_CMP='TEMP', VALE={value!r}))\n" for value in values)
    history = "".join(f"    _F(NOM_CHAM='TEMP', CHAM_GD={name[value]}, INST={instant!r}),\n"
                      for instant, value in zip([0.0] + levels["instants"], [temperatures[0]] + temperatures))
    definitions = (
        "# Temperature load (not part of ssnp142): once the reservoir is full the wall is cooled uniformly\n"
        f"# below its closure temperature of {closure:g} C, at which it is free of thermal stress. TEMP is in\n"
        "# degrees Celsius. The expansion coefficient is PROVISIONAL (generic value for concrete).\n"
        f"WALL_MAT = DEFI_MATERIAU(ELAS=_F({properties}, ALPHA={cooling['expansion_per_k']:.12g}))\n"
        + fields +
        "TEMP_EVOL = CREA_RESU(OPERATION='AFFE', TYPE_RESU='EVOL_THER', AFFE=(\n" + history + "))\n\n")
    material = "        _F(GROUP_MA='WALL', MATER=WALL_MAT),\n"
    variable = ("\n    AFFE_VARC=_F(GROUP_MA='WALL', NOM_VARC='TEMP', EVOL=TEMP_EVOL, NOM_CHAM='TEMP', "
                f"VALE_REF={closure!r}),")
    return definitions, material, variable


def render_joint_comm(base_comm: str, joint: dict, loads: dict, levels: dict, steps_path: str,
                      status_path: str, internal_spring_lines: list[str] | tuple = ()) -> str:
    """Turn the linear baseline case into the staged joint case (see the module docstring).

    ``internal_spring_lines`` (see internal_joint_spring_lines) keep the springs between the parts
    of a stepped wall; without them no spring of the baseline's contact is modelled.
    """
    for name, anchor in _ANCHORS.items():
        if base_comm.count(anchor) != 1:
            raise JointCaseError(f"The baseline case no longer has its '{name}' section")
    comm = base_comm[:base_comm.index(_ANCHORS["solve"])]
    comm = comm.replace(_ANCHORS["debut"], (
        "import json\nimport time\n\nimport numpy as np\n\n"
        "from code_aster.Commands import *\nfrom code_aster import CA\n\n"
        "DEBUT(LANG='EN', ERREUR=_F(ERREUR_F='EXCEPTION'))"))
    comm = comm.replace(_ANCHORS["springs_model"],
                        (_ANCHORS["springs_model"] if internal_spring_lines else "")
                        + "        _F(GROUP_MA='JOINTS', PHENOMENE='MECANIQUE', MODELISATION='3D_JOINT'),\n")
    lines = comm.split("\n")
    first = next((index for index, line in enumerate(lines) if "CARA='K_T_D_L'" in line), None)
    if first is None:
        raise JointCaseError("The baseline case has no wall/foundation spring stiffness lines to remove")
    kept = [line for line in lines if "CARA='K_T_D_L'" not in line]
    kept[first:first] = list(internal_spring_lines)
    comm = "\n".join(kept)

    gradient = float(loads["water_pressure_gradient_pa_per_m"])
    pairs = lambda values: ", ".join(f"{t:.12g}, {v:.12g}" for t, v in zip(levels["instants"], values))  # noqa: E731
    law = joint["law"]
    material = [f"K_N={1.25e11:.12g}", f"K_T={1.25e+11:.12g}"]
    if law == "JOINT_MECA_RUPT":
        material += [f"SIGM_MAX={joint['sigm_max_pa']:.12g}", "ALPHA=1.0",
                     f"PENA_RUPTURE={joint['pena_rupture']:.12g}"]
    elif law == "JOINT_MECA_FROT":
        material += [f"MU={0.8:.12g}", f"ADHESION={1000:.12g}",
                     f"PENA_TANG={1.25e+10:.12g}"]
    else:
        raise JointCaseError(f"Unknown joint law {law!r}")
    uplift = ""
    if joint.get("uplift_position") is not None:
        fraction = position_expression(joint["uplift_position"])
        uplift = (
            "P_JOINT = FORMULE(\n"
            "    NOM_PARA=('X', 'Y', 'Z', 'INST'),\n"
            f"    VALE='(1.0 - ({fraction})) * max({gradient!r} * (NIV_AM(INST) - Z), 0.0) '\n"
            f"         '+ ({fraction}) * max({gradient!r} * (NIV_AV(INST) - Z), 0.0)',\n"
            "    NIV_AM=NIV_AM, NIV_AV=NIV_AV,\n"
            ")\n"
            "# As ssnp142b's PRES_FLUIDE: imposed water pressure in the joint, reservoir head at the\n"
            "# upstream edge falling linearly to tailwater head at the downstream edge.\n")
        material.append("PRES_FLUIDE=P_JOINT")
    cooling = joint.get("cooling")
    thermal, wall_material, thermal_variable = cooling_sections(comm, cooling, levels) if cooling else ("", "", "")
    comm = comm.replace(_ANCHORS["chmat"], (
        "# Reservoir and tailwater level against the instant, as niv_am / niv_av in ssnp142b.\n"
        f"NIV_AM = DEFI_FONCTION(NOM_PARA='INST', VALE=({pairs(levels['upstream_level_m'])}),\n"
        "                       PROL_GAUCHE='CONSTANT', PROL_DROITE='CONSTANT')\n"
        f"NIV_AV = DEFI_FONCTION(NOM_PARA='INST', VALE=({pairs(levels['downstream_level_m'])}),\n"
        "                       PROL_GAUCHE='CONSTANT', PROL_DROITE='CONSTANT')\n"
        + uplift +
        "JOINT_MAT = DEFI_MATERIAU(\n"
        f"    {law}=_F({', '.join(material)}),\n"
        ")\n"
        f"# Doc R7.01.25 {law}, the dam joint law of "
        f"{'ssnp142b' if law == 'JOINT_MECA_RUPT' else 'ssnp142d'}.\n\n"
        + thermal + _ANCHORS["chmat"] + thermal_variable))
    # A later assignment replaces an earlier one, so the wall's cells take the material with expansion.
    comm = comm.replace(_ANCHORS["concrete_mat"],
                        _ANCHORS["concrete_mat"] + wall_material + "        _F(GROUP_MA='JOINTS', MATER=JOINT_MAT),\n")
    # ssnp142b's dry solve uses the mixed line search; its solve with joint water and ssnp142d use the default one.
    line_search = ("RECH_LINEAIRE=_F(METHODE='MIXTE', ITER_LINE_MAXI=5)"
                   if law == "JOINT_MECA_RUPT" and joint.get("uplift_position") is None
                   else "RECH_LINEAIRE=_F(ITER_LINE_MAXI=5)")
    has_trace = "SUPPORT_TRACE = AFFE_CHAR_MECA(" in comm
    comm += f'''
P_AMONT = FORMULE(NOM_PARA=('Z', 'INST'), VALE='max({gradient!r} * (NIV_AM(INST) - Z), 0.0)', NIV_AM=NIV_AM)
P_AVAL = FORMULE(NOM_PARA=('Z', 'INST'), VALE='max({gradient!r} * (NIV_AV(INST) - Z), 0.0)', NIV_AV=NIV_AV)
RETENUE = AFFE_CHAR_MECA_F(
    MODELE=MODELE,
    PRES_REP=(
        _F(GROUP_MA=('UPSTREAM', 'CREST'), PRES=P_AMONT),
        _F(GROUP_MA='DOWNSTREAM', PRES=P_AVAL),
    ),
)
# Hydrostatic pressure of the current level, as p_am / p_av in ssnp142b. On the crest it is
# the overflow surcharge once the level is above the crest.

STEPS_PATH = {steps_path!r}
STATUS_PATH = {status_path!r}
INSTANTS = {levels["instants"]!r}
JOINT_CELLS = list(MAIL.getCells('JOINTS'))
VARIABLES = ('V1', 'V2', 'V3', 'V5', 'V7', 'V8', 'V9', 'V18')
STRESSES = ('SIGN', 'SITX', 'SITY')
STRESS_COMPONENTS = ('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ')
# Wall stress is recovered at the last instant; a cooling case needs it at every instant.
STRESS_EVERY_INSTANT = {bool(cooling)!r}


def gauss(field, component):
    values, description = field.toSimpleFieldOnCells().getValuesWithDescription([component], ['JOINTS'])
    return np.asarray(values, dtype=float), np.asarray(description[0], dtype=np.int64)


def nodal(field, components):
    result = np.full((MAIL.getNumberOfNodes(), len(components)), np.nan)
    for column, component in enumerate(components):
        values, description = field.getValuesWithDescription([component])
        result[np.asarray(description[0], dtype=np.int64), column] = np.asarray(values, dtype=float)
    return result


history = {{"instant": [], "displacement_m": []}}
history.update({{name: [] for name in VARIABLES + STRESSES}})
extra = {{}}
status = {{"schema_version": 1, "status": "running", "instants": INSTANTS, "converged_instants": [],
          "joint_cells": len(JOINT_CELLS), "started_at": time.time()}}


def save(final=None):
    if final is not None:
        status.update(final)
    np.savez(STEPS_PATH, **{{key: np.asarray(value, dtype=float) for key, value in history.items()}}, **extra)
    with open(STATUS_PATH, 'w', encoding='utf-8') as stream:
        json.dump(status, stream, indent=2)


save()
# The weight reaches its full value at the first instant, before any water, and stays there.
RAMPE_POIDS = DEFI_FONCTION(NOM_PARA='INST', VALE=(0.0, 0.0, {levels["instants"][0]!r}, 1.0),
                            PROL_GAUCHE='CONSTANT', PROL_DROITE='CONSTANT')
EXCIT = (
    _F(CHARGE=GRAVITE, FONC_MULT=RAMPE_POIDS),
    _F(CHARGE=RETENUE),
{"    _F(CHARGE=SUPPORT_TRACE)," if has_trace else ""}
)
RESU = None
reached = 0.0
indexes = []
for target in INSTANTS:
    print(f"DAM_JOINT_STEP start instant={{target:.6g}}", flush=True)
    started = time.time()
    # Step cutting as ssnp142b.
    times = DEFI_LIST_INST(
        DEFI_LIST=_F(LIST_INST=DEFI_LIST_REEL(DEBUT=reached, INTERVALLE=_F(JUSQU_A=target, NOMBRE=1))),
        ECHEC=_F(SUBD_NIVEAU=6, SUBD_PAS=10, SUBD_PAS_MINI=1.0e-10),
    )
    # Nonlinear settings of ssnp142b / ssnp142d.
    options = dict(
        MODELE=MODELE, CHAM_MATER=CHMAT, CARA_ELEM=CARA_ROCK, EXCIT=EXCIT,
        COMPORTEMENT=(_F(RELATION='ELAS', TOUT='OUI'), _F(RELATION={law!r}, GROUP_MA='JOINTS')),
        INCREMENT=_F(LIST_INST=times), NEWTON=_F(MATRICE='TANGENTE', REAC_ITER=1),
        {line_search},
        CONVERGENCE=_F(ITER_GLOB_MAXI=12, RESI_GLOB_RELA=1e-5),
        SOLVEUR=_F(METHODE='MUMPS'),
        ARCHIVAGE=_F(LIST_INST=DEFI_LIST_REEL(VALE=(target,))),
    )
    try:
        if RESU is None:
            RESU = STAT_NON_LINE(**options)
        else:
            RESU = STAT_NON_LINE(reuse=RESU, ETAT_INIT=_F(EVOL_NOLI=RESU), **options)
    except CA.AsterError as error:
        detail = str(error).strip() or type(error).__name__
        # MECANONLINE9_1: Code_Aster stops a step it cannot finish inside its time limit.
        out_of_time = "MECANONLINE9_1" in detail
        reason = "ran out of its time limit" if out_of_time else "did not converge"
        save({{"status": "stopped", "stopped_before_instant": target,
               "message": f"Code_Aster {{reason}} from instant {{reached:.6g}} to {{target:.6g}}: {{detail}}"}})
        print(f"DAM_JOINT_STEP {{'outoftime' if out_of_time else 'failed'}} instant={{target:.6g}}", flush=True)
        break
    reached = target
    last = RESU.getLastIndex()
    indexes.append(last)
    vari = RESU.getField('VARI_ELGA', last)
    sief = RESU.getField('SIEF_ELGA', last)
    for name in VARIABLES:
        values, cells = gauss(vari, name)
        history[name].append(values)
    for name in STRESSES:
        history[name].append(gauss(sief, name)[0])
    extra["gauss_cells"] = cells
    extra["joint_cell_ids"] = np.asarray(JOINT_CELLS, dtype=np.int64)
    history["instant"].append(target)
    history["displacement_m"].append(nodal(RESU.getField('DEPL', last), ('DX', 'DY', 'DZ')))
    status["converged_instants"].append(target)
    status.setdefault("seconds_per_instant", []).append(round(time.time() - started, 1))
    save()
    print(f"DAM_JOINT_STEP done instant={{target:.6g}} seconds={{time.time() - started:.1f}}", flush=True)

if history["instant"]:
    try:
        wanted = tuple(history["instant"]) if STRESS_EVERY_INSTANT else (reached,)
        RESU = CALC_CHAMP(reuse=RESU, RESULTAT=RESU, GROUP_MA='CONCRETE', CONTRAINTE=('SIGM_NOEU',), INST=wanted)
        IMPR_RESU(
            FORMAT='MED',
            UNITE=80,
            RESU=_F(RESULTAT=RESU, NOM_CHAM=('DEPL', 'SIGM_NOEU', 'SIEQ_NOEU')),
        )
        extra["stress_pa"] = nodal(RESU.getField('SIGM_NOEU', indexes[-1]), STRESS_COMPONENTS)
        extra["stress_instant"] = np.asarray(reached)
        if STRESS_EVERY_INSTANT:
            extra["stress_by_instant_pa"] = np.stack(
                [nodal(RESU.getField('SIGM_NOEU', index), STRESS_COMPONENTS) for index in indexes])
    except CA.AsterError as error:
        status["stress_error"] = str(error).strip() or type(error).__name__
    extra["coordinates_m"] = np.asarray(MAIL.getCoordinates().getValues(), dtype=float).reshape(-1, 3)
if status["status"] == "running":
    save({{"status": "completed"}})
elif not history["instant"]:
    save({{"status": "failed"}})
status["finished_at"] = time.time()
save()

FIN()
'''
    return comm


# ------------------------------------------------------------------ prepare and run

def _wsl(path: Path) -> str:
    from Aster.whole_model_crack import wsl_path
    return wsl_path(path)


def joint_parameters(arrays: dict, cells: np.ndarray, case: dict) -> dict:
    stiffness = configured_stiffness(arrays) if case["stiffness"] == "configured" else float(case["stiffness"])
    stiffness = stiffness * 100
    joint = {"law": case.get("law", "JOINT_MECA_FROT"), "k_n_pa_per_m": stiffness, "k_t_pa_per_m": stiffness,
             "uplift_position": None}
    if joint["law"] == "JOINT_MECA_FROT":
        joint.update(
                    friction_coefficient=0.8,
                    adhesion_pa=1000,
                    pena_tang_pa_per_m=(stiffness / 10))
    elif joint["law"] == "JOINT_MECA_RUPT":
        joint.update(sigm_max_pa=float(case["sigm_max_pa"]), pena_rupture=float(case["pena_rupture"]))
    else:
        raise JointCaseError(f"Unknown joint law {joint['law']!r}")
    if case.get("uplift"):
        joint["uplift_position"] = case.get("uplift_position") or arch_position(arrays, cells)
    if case.get("cooling"):
        closure = float(case["cooling"]["closure_temperature_c"])
        temperatures = [float(value) for value in case["cooling"]["wall_temperatures_c"]]
        expansion = float(case["cooling"]["expansion_per_k"])
        if (not temperatures or expansion <= 0
                or any(later >= earlier for earlier, later in zip([closure] + temperatures, temperatures))):
            raise JointCaseError("Cooling needs a positive expansion coefficient and wall temperatures that fall, "
                                 "step by step, below the closure temperature")
        joint["cooling"] = {"expansion_per_k": expansion, "closure_temperature_c": closure,
                            "wall_temperatures_c": temperatures,
                            "steps_k": [closure - value for value in temperatures],
                            "cooled_body": "wall", "provisional": True}
    return joint


def write_case(output_dir: Path, arrays: dict, manifest: dict, base_comm: str, loads: dict, case: dict,
               meta: dict | None = None) -> dict:
    """Write the mesh arrays, manifest and case file of one joint run into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = with_joints(arrays)
    cells = arrays["penta6_joints"]
    joint = joint_parameters(arrays, cells, case)
    nodes = np.asarray(arrays["nodes"], dtype=float)
    wet = np.unique(np.concatenate([np.asarray(arrays[key]).ravel() for key in ("tri3_upstream", "tri3_downstream")]))
    cooling = joint.get("cooling") or {}
    levels = water_levels(loads, float(nodes[wet][:, 2].min()), cooling.get("steps_k", ()),
                          cooling.get("closure_temperature_c"))
    crest_z = nodes[np.unique(arrays["tri3_crest"])][:, 2]
    surcharge = loads["water_pressure_gradient_pa_per_m"] * max(loads["upstream_water_surface_z_m"] - crest_z.max(), 0.0)
    if np.ptp(crest_z) > 1e-6 or abs(surcharge - loads["overflow_surcharge_pa"]) > 1e-6 * max(1.0, surcharge):
        raise JointCaseError("The crest surcharge of the baseline is not the hydrostatic head above a level crest")
    np.savez_compressed(output_dir / ARRAYS_NAME, **arrays)
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    comm = render_joint_comm(base_comm, joint, loads, levels, _wsl(output_dir / STEPS_NAME),
                             _wsl(output_dir / STATUS_NAME), internal_joint_spring_lines(arrays))
    (output_dir / CASE_NAME).write_text(comm, encoding="utf-8", newline="\n")
    record = {"schema_version": 1, "case": case, "joint": joint, "levels": levels, "loads": loads,
              "joint_cell_count": int(len(cells)), "departures_from_ssnp142": list(DEPARTURES), **(meta or {})}
    if joint.get("cooling"):
        record["provisional_inputs"] = [cooling_note(joint["cooling"])]
    ties = int(arrays.get("mortar_fill_tie_pairs", 0))
    if int(arrays["internal_wall_joint_pairs"]) > ties:
        between_wall_parts = ~arrays["joint_on_foundation"]
        if "joint_on_mortar_fill" in arrays:
            between_wall_parts &= ~arrays["joint_on_mortar_fill"]
        record["internal_wall_joints"] = {
            "contact_triangles": int(between_wall_parts.sum()),
            "spring_pairs": int(arrays["internal_wall_joint_pairs"]) - ties,
            "treatment": "linear springs of the elastic baseline (they hold tension)"}
    if "tetra4_mortar" in arrays:
        record["mortar_fill"] = {
            "cells": int(len(arrays["tetra4_mortar"])),
            "volume_m3": manifest.get("groups", {}).get("MORTAR", {}).get("volume_m3"),
            "material": (meta or {}).get("material", {}).get("mortar"),
            "tie_triangles": int(arrays["joint_on_mortar_fill"].sum()),
            "tie_spring_pairs": ties,
            "treatment": "its own elastic body with its own weight, tied to the wall face it covers by the "
                         "elastic baseline's linear springs; excluded from wall stress and displacement, "
                         "and not cooled in a cooling case"}
    (output_dir / META_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for name in (STEPS_NAME, STATUS_NAME, REPORT_NAME):
        (output_dir / name).unlink(missing_ok=True)
    return record


def prepare(baseline_dir: Path, output_dir: Path, case: dict) -> dict:
    """Prepare a joint run from a finished job's Code_Aster elastic baseline directory."""
    from Aster.generate_three_body_case import render_comm

    baseline_dir = baseline_dir.resolve()
    with np.load(baseline_dir / ARRAYS_NAME, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    manifest = json.loads((baseline_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("bedrock_analysis_mode") != "elastic_layer":
        raise JointCaseError("The joint baseline is implemented for the elastic-layer bedrock mode")
    output_dir.mkdir(parents=True, exist_ok=True)
    # render_comm reads the arrays beside the manifest, so both are written before the case is rendered.
    np.savez_compressed(output_dir / ARRAYS_NAME, **arrays)
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    base_comm, inputs = render_comm(mesh_manifest_path=output_dir / MANIFEST_NAME)
    loads = {key: float(inputs[key]) for key in (
        "water_pressure_gradient_pa_per_m", "upstream_water_surface_z_m", "tailwater_elevation_m",
        "overflow_surcharge_pa", "gravity_m_s2")}
    meta = {"baseline_dir": str(baseline_dir), "material": inputs["material"]}
    report_path = baseline_dir / "aster_baseline_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        meta["elastic_baseline"] = {key: report.get(key) for key in (
            "maximum_wall_displacement_m", "maximum_principal_tensile_stress_pa",
            "maximum_principal_compressive_stress_pa")}
    return write_case(output_dir, arrays, manifest, base_comm, loads, case, meta)


def solve(output_dir: Path, time_limit_s: int) -> dict:
    """Convert the mesh and run the prepared case with the Linux Code_Aster in WSL."""
    from Aster.run_case import MESH_UNIT, SOLVER_THREADS, run_wsl_solver
    from Aster.whole_model_crack import DEFAULT_WSL_DISTRIBUTION, DISTRIBUTION_ENVIRONMENT, find_linux_installation
    import subprocess

    if time_limit_s > MAXIMUM_TIME_LIMIT_S:
        raise JointCaseError(f"The time limit is {time_limit_s} s; runs are not allowed more than "
                             f"{MAXIMUM_TIME_LIMIT_S} s")
    distribution = os.environ.get(DISTRIBUTION_ENVIRONMENT, DEFAULT_WSL_DISTRIBUTION)
    root = find_linux_installation(distribution)
    run_dir = output_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    med_path = output_dir / "mesh.med"
    med_path.unlink(missing_ok=True)
    live_log = run_dir / "solver.live.log"
    live_log.unlink(missing_ok=True)
    levels = json.loads((output_dir / META_NAME).read_text(encoding="utf-8"))["levels"]
    started = time.monotonic()
    outcome = {"installation": root, "distribution": distribution, "time_limit_s": time_limit_s}
    finished = threading.Event()
    follower = threading.Thread(target=_follow, args=(live_log, finished, levels), daemon=True)
    try:
        progress(f"Converting the mesh for Code_Aster ({output_dir.name})")
        conversion = run_wsl_solver(distribution, root, output_dir, [
            f"{root}/bin/python3", _wsl(Path(__file__).with_name("convert_three_body_to_med.py")),
            _wsl(output_dir / ARRAYS_NAME), _wsl(med_path)], 300, run_dir / "conversion.log")
        if conversion.returncode != 0 or not med_path.is_file():
            raise RuntimeError("Mesh conversion failed; see run/conversion.log")
        export = run_dir / "dam_joint.export"
        export.write_text("\n".join((
            f"P time_limit {time_limit_s}", f"P memory_limit {MEMORY_LIMIT_MB}", "P mpi_nbcpu 1",
            f"P ncpus {SOLVER_THREADS}", f"F comm {_wsl(output_dir / CASE_NAME)} D 1",
            f"F libr {_wsl(med_path)} D {MESH_UNIT}", "")), encoding="utf-8")
        progress(f"Starting Code_Aster on {SOLVER_THREADS} cores, limit {time_limit_s} s; it reads the model first "
                 "(about a minute for the dam)")
        follower.start()
        completed = run_wsl_solver(distribution, root, run_dir, [f"{root}/bin/run_aster", _wsl(export)],
                                   time_limit_s, live_log)
        outcome["exit_code"] = completed.returncode
        if completed.returncode in (124, 137):
            outcome["status"] = "timed_out"
        elif completed.returncode != 0:
            outcome["status"] = "failed"
        else:
            outcome["status"] = "completed"
    except subprocess.TimeoutExpired:
        outcome["status"] = "timed_out"
    except RuntimeError as error:
        outcome.update(status="failed", message=str(error))
    finally:
        finished.set()
        if follower.is_alive():
            follower.join(timeout=10)
    outcome["elapsed_s"] = round(time.monotonic() - started, 1)
    (run_dir / "run_manifest.json").write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    progress(f"Code_Aster ended: {outcome['status']} after {outcome['elapsed_s']:.0f} s ({output_dir.name})")
    return outcome


# ------------------------------------------------------------------ report

def _per_cell(values: np.ndarray, gauss_cells: np.ndarray, cell_ids: np.ndarray, reduce) -> np.ndarray:
    rows = np.searchsorted(cell_ids, gauss_cells)
    if np.any(cell_ids[rows] != gauss_cells):
        raise JointCaseError("Joint results refer to cells outside the joint group")
    result = np.full(len(cell_ids), np.nan)
    order = np.argsort(rows, kind="stable")
    boundaries = np.flatnonzero(np.diff(rows[order])) + 1
    for group in np.split(order, boundaries):
        result[rows[group[0]]] = reduce(values[group])
    return result


def principal_stresses(stress: np.ndarray) -> np.ndarray:
    """Principal stresses per node in ascending order, from (xx, yy, zz, xy, xz, yz) rows."""
    tensor = np.empty((len(stress), 3, 3))
    tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2] = stress[:, :3].T
    tensor[:, 0, 1] = tensor[:, 1, 0] = stress[:, 3]
    tensor[:, 0, 2] = tensor[:, 2, 0] = stress[:, 4]
    tensor[:, 1, 2] = tensor[:, 2, 1] = stress[:, 5]
    return np.linalg.eigvalsh(tensor)


def principal_extremes(stress: np.ndarray) -> tuple[float, float]:
    """Largest and smallest principal stress over nodes, from (xx, yy, zz, xy, xz, yz) rows."""
    principal = principal_stresses(stress)
    return float(principal[:, 2].max()), float(principal[:, 0].min())


def summarise(output_dir: Path) -> dict:
    """Read a finished (or stopped) run and report what the contact did at each water level."""
    meta = json.loads((output_dir / META_NAME).read_text(encoding="utf-8"))
    status = json.loads((output_dir / STATUS_NAME).read_text(encoding="utf-8"))
    with np.load(output_dir / ARRAYS_NAME, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    cells = arrays["penta6_joints"]
    geometry = joint_geometry(arrays, cells)
    area, centroid, normal = geometry["area_m2"], geometry["centroid_m"], geometry["normal"]
    report = {"schema_version": 1, "case": meta["case"], "joint": meta["joint"], "status": status["status"],
              "message": status.get("message"), "converged_instants": status["converged_instants"],
              "seconds_per_instant": status.get("seconds_per_instant"), "contact_area_m2": float(area.sum()),
              "elastic_baseline": meta.get("elastic_baseline"), "departures_from_ssnp142": meta["departures_from_ssnp142"],
              "engineering_validated": False, "levels": []}
    if meta.get("provisional_inputs"):
        report["provisional_inputs"] = meta["provisional_inputs"]
    if meta.get("internal_wall_joints"):
        report["internal_wall_joints"] = meta["internal_wall_joints"]
    if meta.get("mortar_fill"):
        report["mortar_fill"] = meta["mortar_fill"]
    with np.load(output_dir / STEPS_NAME, allow_pickle=False) as steps:
        data = {key: steps[key] for key in steps.files}
    if not len(data["instant"]):
        (output_dir / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    cell_ids, gauss_cells = data["joint_cell_ids"].astype(np.int64), data["gauss_cells"].astype(np.int64)
    if len(cell_ids) != len(cells) or np.any(np.diff(cell_ids) <= 0):
        raise JointCaseError("Code_Aster's joint cells do not match the prepared joint cells")
    report["gauss_points_per_joint_cell"] = float(len(gauss_cells) / len(cell_ids))
    wall_nodes = np.unique(arrays["tetra4_concrete"][arrays["concrete_body_ids"] == 1])
    support = arrays["support_nodes"].astype(np.int64)
    support_k = arrays["support_matrices_n_per_m"][:, 2, 2]
    levels = meta["levels"]
    for row, instant in enumerate(data["instant"].tolist()):
        index = levels["instants"].index(instant)
        mean = lambda name: _per_cell(data[name][row], gauss_cells, cell_ids, np.mean)  # noqa: E731
        peak = lambda name: _per_cell(data[name][row], gauss_cells, cell_ids, np.max)  # noqa: E731
        opening = mean("V7")
        frictional = meta["joint"]["law"] == "JOINT_MECA_FROT"
        # JOINT_MECA_RUPT: V3 is the damage state (0 sound, 1 damaged, 2 broken).
        # JOINT_MECA_FROT: V2 marks sliding points and V1 is their cumulative plastic slip.
        state = np.zeros(len(cells)) if frictional else peak("V3")
        sliding = peak("V2") > 0.5 if frictional and "V2" in data else np.zeros(len(cells), dtype=bool)
        plastic_slip = peak("V1") if frictional and "V1" in data else np.zeros(len(cells))
        traction = mean("SIGN")
        slip = np.hypot(mean("V8"), mean("V9"))
        shear = np.hypot(mean("SITX"), mean("SITY"))
        displacement = data["displacement_m"][row]
        jump = np.einsum("ckj,cj->ck", displacement[cells[:, 3:]] - displacement[cells[:, :3]], normal).mean(axis=1)
        opened = opening > OPEN_TOLERANCE_M
        widest = int(np.argmax(opening))
        lifting = displacement[support, 2] > 0
        level = {
            "instant": instant, "fill_fraction": levels["fill_fraction"][index],
            "upstream_level_m": levels["upstream_level_m"][index],
            "downstream_level_m": levels["downstream_level_m"][index],
            "maximum_wall_displacement_m": float(np.linalg.norm(displacement[wall_nodes], axis=1).max()),
            "opened_area_m2": float(area[opened].sum()), "opened_share": float(area[opened].sum() / area.sum()),
            "largest_opening_m": float(opening.max()), "largest_opening_xyz_m": centroid[widest].tolist(),
            "largest_closing_m": float(-opening.min()),
            "damaged_area_m2": float(area[state >= 1].sum()), "broken_area_m2": float(area[state >= 2].sum()),
            "sliding_area_m2": float(area[sliding].sum()), "largest_plastic_slip_m": float(plastic_slip.max()),
            "largest_sliding_m": float(slip.max()), "largest_sliding_xyz_m": centroid[int(np.argmax(slip))].tolist(),
            "largest_joint_tension_pa": float(traction.max()), "largest_joint_compression_pa": float(-traction.min()),
            "largest_joint_shear_pa": float(shear.max()),
            "joint_normal_force_n": float((traction * area).sum()),
            "joint_tension_force_n": float((np.clip(traction, 0, None) * area).sum()),
            "largest_joint_water_pressure_pa": float(mean("V18").max()),
            # Independent of the joint law: the same opening rebuilt from nodal displacements.
            "opening_check_largest_difference_m": float(np.abs(jump - opening).max()),
            "support_tension_force_n": float((support_k[lifting] * displacement[support[lifting], 2]).sum()),
            "support_compression_force_n": float(-(support_k[~lifting] * displacement[support[~lifting], 2]).sum()),
        }
        if "temperature_change_k" in levels:
            level["wall_temperature_change_k"] = levels["temperature_change_k"][index]
            level["wall_temperature_c"] = levels["wall_temperature_c"][index]
        if "stress_by_instant_pa" in data:
            principal = principal_stresses(data["stress_by_instant_pa"][row][wall_nodes])
            highest = int(np.argmax(principal[:, 2]))
            level.update(
                maximum_principal_tensile_stress_pa=float(principal[highest, 2]),
                maximum_principal_tensile_stress_xyz_m=arrays["nodes"][wall_nodes[highest]].tolist(),
                maximum_principal_compressive_stress_pa=float(principal[:, 0].min()))
        report["levels"].append(level)
    if "stress_pa" in data:
        tension, compression = principal_extremes(data["stress_pa"][wall_nodes])
        report["final_wall_stress"] = {"instant": float(data["stress_instant"]),
                                       "maximum_principal_tensile_stress_pa": tension,
                                       "maximum_principal_compressive_stress_pa": compression,
                                       "measure": "principal extremes of recovered nodal stress over wall nodes"}
    (output_dir / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def with_wall_temperatures(case: dict, closure_temperature_c: float | None = None,
                           wall_temperatures_c: list[float] | None = None) -> dict:
    """The cooling case with another closure temperature or other wall temperatures (None keeps each)."""
    if closure_temperature_c is None and not wall_temperatures_c:
        return case
    if "cooling" not in case:
        raise JointCaseError("Temperatures can only be given for a cooling case")
    cooling = dict(case["cooling"])
    if closure_temperature_c is not None:
        cooling["closure_temperature_c"] = float(closure_temperature_c)
    if wall_temperatures_c:
        cooling["wall_temperatures_c"] = [float(value) for value in wall_temperatures_c]
    return {**case, "cooling": cooling}


def run(name: str, baseline_dir: Path, output_root: Path = DEFAULT_OUTPUT_ROOT, time_limit_s: int = 600,
        closure_temperature_c: float | None = None, wall_temperatures_c: list[float] | None = None) -> dict:
    output_dir = output_root / name
    case = with_wall_temperatures(CASES[name], closure_temperature_c, wall_temperatures_c)
    progress(f"=== Dam joint case '{name}': {case['purpose']}")
    if "cooling" in case:
        progress(f"Closure temperature of the wall {case['cooling']['closure_temperature_c']:g} C; wall temperatures "
                 f"solved at full reservoir: {', '.join(f'{value:g}' for value in case['cooling']['wall_temperatures_c'])} C")
    progress("Preparing the case from the job's elastic baseline (no solver yet)")
    prepare(baseline_dir, output_dir, case)
    outcome = solve(output_dir, time_limit_s)
    if not (output_dir / STATUS_NAME).is_file():
        return {"status": outcome["status"], "message": outcome.get("message", "The solver wrote no status; see run/solver.live.log")}
    report = summarise(output_dir)
    report["run"] = outcome
    (output_dir / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    cooled_levels = [level for level in report["levels"] if "wall_temperature_change_k" in level
                     and level["fill_fraction"] == FILL_FRACTIONS[-1]]
    for level in cooled_levels:
        progress(f"Full reservoir, wall at {level['wall_temperature_c']:g} C "
                 f"({abs(level['wall_temperature_change_k']):g} K below closure): largest "
                 f"opening {level['largest_opening_m'] * 1e3:.3f} mm, opened {level['opened_area_m2']:.0f} of "
                 f"{report['contact_area_m2']:.0f} m2, peak wall displacement "
                 f"{level['maximum_wall_displacement_m'] * 1e3:.2f} mm"
                 + (f", peak wall tension {level['maximum_principal_tensile_stress_pa'] / 1e6:.2f} MPa"
                    if "maximum_principal_tensile_stress_pa" in level else ""))
    if report["levels"] and not cooled_levels:
        level = report["levels"][-1]
        progress(f"Result at {level['fill_fraction']:.0%} of the head: largest opening "
                 f"{level['largest_opening_m'] * 1e3:.3f} mm, opened {level['opened_area_m2']:.0f} of "
                 f"{report['contact_area_m2']:.0f} m2, peak wall displacement "
                 f"{level['maximum_wall_displacement_m'] * 1e3:.2f} mm. Report: {output_dir / REPORT_NAME}")
    elif cooled_levels:
        progress(f"Report: {output_dir / REPORT_NAME}")
    progress(f"=== Finished '{name}': {report['status']}")
    return report


def run_stage(baseline_dir: Path, output_dir: Path, time_limit_s: int = 600) -> dict:
    """The app's joint-opening stage: the dry joint, then water in the joint, each from the same baseline.

    A case that does not complete ends the stage; the cases that did complete stay in the report.
    """
    stage = {"schema_version": 1, "status": "completed", "message": None, "cases": {},
             "departures_from_ssnp142": list(DEPARTURES), "engineering_validated": False}
    for label, name in STAGE_CASES:
        try:
            report = run(name, baseline_dir, output_dir, time_limit_s)
        except (JointCaseError, OSError, ValueError, KeyError) as error:
            report = {"status": "failed", "message": f"The case could not be prepared: {error}"}
            progress(f"=== '{name}' could not be prepared: {error}")
        stage["cases"][label] = {"case": name, **report}
        stage.setdefault("elastic_baseline", report.get("elastic_baseline"))
        if report["status"] != "completed":
            reason = str(report.get("message") or "").strip().splitlines()
            stage.update(status="failed", message=(
                f"The {label.replace('_', ' ')} case ended as '{report['status']}'"
                + (f": {reason[0]}" if reason else "") + ". See its run/solver.live.log."))
            break
    (output_dir / STAGE_REPORT_NAME).write_text(json.dumps(stage, indent=2) + "\n", encoding="utf-8")
    raise RuntimeError("Stage did not complete successfully")


def print_report(report: dict) -> None:
    print(f"status: {report['status']}" + (f" ({report['message']})" if report.get("message") else ""))
    for level in report.get("levels", []):
        print(f"  level {level['upstream_level_m']:8.2f} m ({level['fill_fraction']:.0%})"
              + (f", wall {level['wall_temperature_c']:g} C ({level['wall_temperature_change_k']:+g} K)"
                 if "wall_temperature_c" in level else "")
              + (f", wall tension {level['maximum_principal_tensile_stress_pa'] / 1e6:.3f} MPa"
                 if "maximum_principal_tensile_stress_pa" in level else "") + ": "
              f"opened {level['opened_area_m2']:7.1f} m2 ({level['opened_share']:.0%}), "
              f"largest opening {level['largest_opening_m'] * 1e3:8.4f} mm, "
              f"sliding {level['largest_sliding_m'] * 1e3:7.4f} mm, "
              f"wall displacement {level['maximum_wall_displacement_m'] * 1e3:7.3f} mm, "
              f"check {level['opening_check_largest_difference_m'] * 1e3:.2e} mm")
    if "final_wall_stress" in report:
        stress = report["final_wall_stress"]
        print(f"  wall peak tension {stress['maximum_principal_tensile_stress_pa'] / 1e6:.3f} MPa, "
              f"peak compression {stress['maximum_principal_compressive_stress_pa'] / 1e6:.3f} MPa")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="prepare, solve and report one joint case on the dam")
    runner.add_argument("case", choices=sorted(CASES))
    source = runner.add_mutually_exclusive_group(required=True)
    source.add_argument("--job", type=Path, help="finished job directory (uses its cracking/aster_elastic_baseline)")
    source.add_argument("--baseline-dir", type=Path)
    runner.add_argument("--closure-temperature-c", type=float, metavar="CELSIUS",
                        help="closure temperature of the wall for a cooling case, at which it is free of thermal "
                             f"stress (default {CLOSURE_TEMPERATURE_C:g})")
    runner.add_argument("--wall-temperatures-c", type=float, nargs="+", metavar="CELSIUS",
                        help="wall temperatures of a cooling case, each lower than the one before and all below "
                             "the closure temperature (each is about a minute of solving on the dam; default "
                             f"{' '.join(f'{value:g}' for value in WALL_TEMPERATURES_C)})")
    for sub in (runner,):
        sub.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        sub.add_argument("--time-limit-s", type=int, default=600,
                         help=f"solver time limit in seconds (at most {MAXIMUM_TIME_LIMIT_S})")
    fixture = commands.add_parser("fixture", help="two-block check against a hand calculation")
    fixture.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    fixture.add_argument("--time-limit-s", type=int, default=300)
    fixture.add_argument("--cooling", action="store_true",
                         help="run only the check of the temperature load against free thermal contraction")
    stage = commands.add_parser("app", help="the app's stage: the dry joint, then water in the joint")
    stage.add_argument("--baseline-dir", type=Path, required=True)
    stage.add_argument("--output-dir", type=Path, required=True)
    stage.add_argument("--time-limit-s", type=int, default=600, help="limit of each of the two solves")
    for sub in (runner, fixture, stage):
        sub.add_argument("--progress-log", type=Path, help="append a timestamped line per stage to this file")
    arguments = parser.parse_args()
    set_progress_log(arguments.progress_log, echo=arguments.command == "app")
    if arguments.time_limit_s > MAXIMUM_TIME_LIMIT_S:
        parser.error(f"--time-limit-s may not exceed {MAXIMUM_TIME_LIMIT_S} s")
    if arguments.command == "app":
        outcome = run_stage(arguments.baseline_dir, arguments.output_dir, arguments.time_limit_s)
        sys.exit(0 if outcome["status"] == "completed" else 2)
    if arguments.command == "fixture":
        from Aster.dam_joint_fixture import run_cooling_fixture, run_fixture
        check = run_cooling_fixture if arguments.cooling else run_fixture
        sys.exit(0 if check(arguments.output_root, arguments.time_limit_s)["passed"] else 2)
    if ((arguments.closure_temperature_c is not None or arguments.wall_temperatures_c)
            and "cooling" not in CASES[arguments.case]):
        parser.error(f"Temperatures need a cooling case; '{arguments.case}' is not one")
    baseline = arguments.baseline_dir or arguments.job / "cracking" / "aster_elastic_baseline"
    try:
        report = run(arguments.case, baseline, arguments.output_root, arguments.time_limit_s,
                     arguments.closure_temperature_c, arguments.wall_temperatures_c)
    except JointCaseError as error:
        parser.error(str(error))
    print_report(report)
    sys.exit(0 if report["status"] == "completed" else 2)


if __name__ == "__main__":
    main()
