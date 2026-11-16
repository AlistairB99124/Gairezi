"""Two-block check of the dam joint case against a hand calculation.

A 1 m thick, 2 m high concrete wall stands on a concrete block through the same
joint layer the dam case uses (prism ``3D_JOINT`` cells, ``JOINT_MECA_RUPT`` with
zero tensile strength). Water is raised against one face. With a joint much
softer than the blocks, the blocks move as rigid bodies and the contact follows
the textbook "cracked base" solution: a linear compressive stress over the part
still in contact and a linearly growing opening at the heel. The run is accepted
when Code_Aster reproduces that opening profile, dry and with water pressure in
the joint.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Aster import dam_joint_baseline as djb  # noqa: E402

THICKNESS_M, WIDTH_M, WALL_HEIGHT_M, BLOCK_HEIGHT_M = 1.0, 1.0, 2.0, 1.0
DENSITY_KG_M3, YOUNG_PA, POISSON = 2400.0, 30.0e9, 0.2
GRAVITY_M_S2, WATER_GRADIENT_PA_PER_M = 9.81, 9810.0
# Soft against the blocks (E / height is 1.5e10 Pa/m), so they stay nearly rigid.
JOINT_STIFFNESS_PA_PER_M = 2.0e8
SUPPORT_STIFFNESS_N_PER_M3 = 1.0e13
ACCEPTED_PROFILE_ERROR = 0.05
# Cooling check: the wall, away from the contact, must shorten by expansion * temperature drop.
COOLING_CHECK_CLOSURE_C = 20.0
COOLING_CHECK_TEMPERATURES_C = (10.0, 0.0)
ACCEPTED_CONTRACTION_ERROR = 0.02


def cracked_base(uplift: bool, tensile_strength_pa: float = 0.0) -> dict:
    """Rigid wall on an elastic joint under its weight and full water on one face.

    The joint carries no tension by default. With ``tensile_strength_pa`` it carries tension
    elastically up to that value and holds it where it has opened further, which is how the
    friction law treats its small adhesion.
    """
    expected = _no_tension_base(uplift)
    if tensile_strength_pa <= 0:
        return expected
    b, w, k = THICKNESS_M, WIDTH_M, JOINT_STIFFNESS_PA_PER_M
    normal, moment = expected["effective_normal_force_n"], expected["overturning_moment_nm"]
    x = np.linspace(0.0, b, 20001)

    def unbalance(slope: float, neutral: float) -> np.ndarray:
        stress = np.minimum(k * slope * (neutral - x), tensile_strength_pa)
        return np.array([w * np.trapezoid(stress, x) + normal, w * np.trapezoid(stress * (x - b / 2), x) + moment])

    unknowns = np.array([expected["opening_slope"], expected["open_length_m"]])
    for _ in range(50):
        residual = unbalance(*unknowns)
        if np.abs(residual).max() < 1e-9 * normal:
            break
        steps = unknowns * 1e-6
        jacobian = np.column_stack([(unbalance(*(unknowns + np.eye(2)[i] * steps[i])) - residual) / steps[i]
                                    for i in range(2)])
        unknowns = unknowns - np.linalg.solve(jacobian, residual)
    else:
        raise ValueError("The cracked-base hand calculation did not converge")
    slope, neutral = unknowns
    return {**expected, "contact_length_m": b - neutral, "open_length_m": neutral,
            "peak_effective_compression_pa": k * slope * (b - neutral), "heel_opening_m": slope * neutral,
            "opening_slope": slope, "tensile_strength_pa": tensile_strength_pa}


def _no_tension_base(uplift: bool) -> dict:
    b, w, h = THICKNESS_M, WIDTH_M, WALL_HEIGHT_M
    weight = DENSITY_KG_M3 * GRAVITY_M_S2 * b * w * h
    thrust = WATER_GRADIENT_PA_PER_M * h ** 2 / 2 * w
    moment = thrust * h / 3
    normal = weight
    if uplift:
        # Triangular pressure in the joint: full head at the heel, zero at the toe.
        lift = WATER_GRADIENT_PA_PER_M * h * b * w / 2
        normal -= lift
        moment += lift * b / 6
    eccentricity = moment / normal
    if eccentricity <= b / 6:
        raise ValueError("The fixture is meant to lift at the heel")
    contact = 3 * (b / 2 - eccentricity)
    peak = 2 * normal / (contact * w)
    slope = peak / (JOINT_STIFFNESS_PA_PER_M * contact)
    return {"contact_length_m": contact, "open_length_m": b - contact, "peak_effective_compression_pa": peak,
            "heel_opening_m": slope * (b - contact), "opening_slope": slope, "effective_normal_force_n": normal,
            "overturning_moment_nm": moment}


def analytic_opening(x: np.ndarray, expected: dict) -> np.ndarray:
    """Opening (positive) or closing (negative) at distance ``x`` from the heel."""
    return expected["opening_slope"] * (expected["open_length_m"] - x)


def fixture_arrays(across: int = 24, along: int = 2, wall_layers: int = 24, block_layers: int = 8) -> dict:
    xs = np.linspace(0.0, THICKNESS_M, across + 1)
    ys = np.linspace(0.0, WIDTH_M, along + 1)
    nodes, index = [], {}

    def node(block: str, i: int, j: int, z: float) -> int:
        key = (block, i, j, round(z, 9))
        if key not in index:
            index[key] = len(nodes)
            nodes.append((xs[i], ys[j], z))
        return index[key]

    cells = {"wall": [], "block": []}
    levels = {"block": np.linspace(0.0, BLOCK_HEIGHT_M, block_layers + 1),
              "wall": np.linspace(BLOCK_HEIGHT_M, BLOCK_HEIGHT_M + WALL_HEIGHT_M, wall_layers + 1)}
    for block, zs in levels.items():
        for k, i, j in itertools.product(range(len(zs) - 1), range(across), range(along)):
            corner = {(a, c, d): node(block, i + a, j + c, zs[k + d]) for a in (0, 1) for c in (0, 1) for d in (0, 1)}
            for order in itertools.permutations(range(3)):
                at = [0, 0, 0]
                path = [corner[tuple(at)]]
                for axis in order:
                    at[axis] = 1
                    path.append(corner[tuple(at)])
                cells[block].append(path)
    points = np.array(nodes, dtype=float)
    concrete = np.array(cells["wall"] + cells["block"], dtype=np.int64)
    corners = points[concrete]
    negative = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0
    concrete[negative] = concrete[negative][:, [0, 2, 1, 3]]
    body = np.array([1] * len(cells["wall"]) + [3] * len(cells["block"]), dtype=np.int64)

    def boundary(rows: np.ndarray) -> np.ndarray:
        faces = np.sort(rows[:, djb._TET_FACES].reshape(-1, 3), axis=1)
        unique, counts = np.unique(faces, axis=0, return_counts=True)
        return unique[counts == 1]

    def on(faces: np.ndarray, axis: int, value: float) -> np.ndarray:
        return faces[np.all(np.isclose(points[faces][:, :, axis], value), axis=1)]

    wall_faces, block_faces = boundary(concrete[body == 1]), boundary(concrete[body == 3])
    wall_bottom, block_top = on(wall_faces, 2, BLOCK_HEIGHT_M), on(block_faces, 2, BLOCK_HEIGHT_M)
    partner = {tuple(np.round(points[n], 9)): n for n in np.unique(block_top)}
    master = np.array([[partner[tuple(np.round(points[n], 9))] for n in row] for row in wall_bottom], dtype=np.int64)
    pairs = np.array(sorted({(int(a), int(b)) for a, b in zip(wall_bottom.ravel(), master.ravel())}), dtype=np.int64)
    support_faces = on(block_faces, 2, 0.0)
    support = np.unique(support_faces)
    tributary = np.zeros(len(points))
    for face in support_faces:
        a, b, c = points[face]
        tributary[face] += np.linalg.norm(np.cross(b - a, c - a)) / 6
    return {
        "nodes": points,
        "tetra4_concrete": concrete,
        "tetra4_bedrock": np.empty((0, 4), dtype=np.int64),
        "concrete_body_ids": body,
        "tri3_upstream": on(wall_faces, 0, 0.0),
        "tri3_downstream": on(wall_faces, 0, THICKNESS_M),
        "tri3_crest": on(wall_faces, 2, BLOCK_HEIGHT_M + WALL_HEIGHT_M),
        "tri3_rock_bottom": np.empty((0, 3), dtype=np.int64),
        "tri3_rock_sides": np.empty((0, 3), dtype=np.int64),
        "seg2_wall_foundation_springs": pairs,
        "wall_foundation_tributary_area_m2": np.ones(len(pairs)),
        "interface_stiffness_n_per_m": np.full((len(pairs), 3), JOINT_STIFFNESS_PA_PER_M),
        "interface_names": np.array(["wall_foundation"] * len(pairs)),
        "joint_slave_tri3": wall_bottom,
        "joint_master_tri3": master,
        "joint_coefficients_n_per_m3": np.full((len(wall_bottom), 3), JOINT_STIFFNESS_PA_PER_M),
        "support_nodes": support,
        "support_matrices_n_per_m": np.eye(3)[None] * (tributary[support] * SUPPORT_STIFFNESS_N_PER_M3)[:, None, None],
        "source_node_ids": np.arange(len(points), dtype=np.int64),
    }


def fixture_loads() -> dict:
    return {"water_pressure_gradient_pa_per_m": WATER_GRADIENT_PA_PER_M,
            "upstream_water_surface_z_m": BLOCK_HEIGHT_M + WALL_HEIGHT_M, "tailwater_elevation_m": BLOCK_HEIGHT_M,
            "overflow_surcharge_pa": 0.0, "gravity_m_s2": GRAVITY_M_S2}


def fixture_base_comm(arrays: dict) -> str:
    """A linear case with the section layout of Aster.generate_three_body_case.render_comm (elastic-layer mode)."""
    springs = ",\n        ".join(
        f"_F(GROUP_MA='SPR_WF_{i:05d}', REPERE='GLOBAL', CARA='K_T_D_L', VALE=({k[0]:.12g}, {k[1]:.12g}, {k[2]:.12g}))"
        for i, k in enumerate(arrays["interface_stiffness_n_per_m"]))
    outer = ",\n        ".join(
        f"_F(GROUP_MA='SPR_OUT_{i:05d}', REPERE='GLOBAL', CARA='K_T_N', SYME='NON', "
        f"VALE={tuple(float(matrix[r][c]) for c in range(3) for r in range(3))!r})"
        for i, matrix in enumerate(arrays["support_matrices_n_per_m"]))
    return f'''DEBUT(LANG='EN')

MAIL = LIRE_MAILLAGE(FORMAT='MED', UNITE=20, NOM_MED='THREE_BODY', VERI_MAIL=_F(VERIF='NON'))

MAIL = MODI_MAILLAGE(
    reuse=MAIL,
    MAILLAGE=MAIL,
    ORIE_PEAU_3D=_F(GROUP_MA=('UPSTREAM', 'DOWNSTREAM', 'CREST')),
)

MODELE = AFFE_MODELE(
    MAILLAGE=MAIL,
    AFFE=(
        _F(
            GROUP_MA=(
                'CONCRETE', 'UPSTREAM', 'DOWNSTREAM', 'CREST',
            ),
            PHENOMENE='MECANIQUE',
            MODELISATION='3D',
        ),
        _F(GROUP_MA='OUTER_SPRINGS', PHENOMENE='MECANIQUE', MODELISATION='DIS_T'),
        _F(GROUP_MA='WALL_FOUNDATION_SPRINGS', PHENOMENE='MECANIQUE', MODELISATION='DIS_T'),
    ),
)

CONCRETE_MAT = DEFI_MATER_GC(
    ENDO_LOCA_TC=_F(
        CODIFICATION='ESSAI',
        E={YOUNG_PA:.12g},
        NU={POISSON:.12g},
        FT=0.5e6,               # Caps solid concrete tension at 0.5 MPa
        FC=30.0e6,              # 30 MPa compressive strength
        SIGM_COMP_SEUIL=15.0e6, # Compressive damage threshold (~0.5 * FC)
        GF=150.0,               # Fracture energy (N/m)
        DIST_FISSURE=1.0,       # Average element size (1.0 m) for fracture energy regularization
        COEF_ECRO_TRAC=1.5,     # Mandatory tensile softening shape factor
    ),
)


CHMAT = AFFE_MATERIAU(
    MAILLAGE=MAIL,
    AFFE=(
        _F(GROUP_MA='CONCRETE', MATER=CONCRETE_MAT),
    ),
)

GRAVITE = AFFE_CHAR_MECA(MODELE=MODELE, PESANTEUR=_F(GRAVITE={GRAVITY_M_S2!r}, DIRECTION=(0.0, 0.0, -1.0)))

CARA_ROCK = AFFE_CARA_ELEM(
    MODELE=MODELE,
    DISCRET=(
        {springs},
        {outer}
    ),
)

RESU = MECA_STATIQUE(
    MODELE=MODELE,
    CHAM_MATER=CHMAT,
    CARA_ELEM=CARA_ROCK,
    EXCIT=(_F(CHARGE=GRAVITE),),
    SOLVEUR=_F(METHODE='MUMPS'),
)

FIN()
'''


def fixture_case(uplift: bool, law: str = "JOINT_MECA_RUPT") -> dict:
    case = {"purpose": "Two-block check against the cracked-base hand calculation", "law": law,
            "stiffness": "configured", "uplift": uplift,
            "uplift_position": {"kind": "x", "upstream_x_m": 0.0, "downstream_x_m": THICKNESS_M}}
    if law == "JOINT_MECA_FROT":
        # Friction 1 holds the wall: the water thrust is at most 0.53 of the effective normal force.
        case.update(friction_coefficient=djb.RECOMMENDED_FRICTION, adhesion_pa=djb.TEST_ADHESION_PA)
    else:
        case.update(sigm_max_pa=0.0, pena_rupture=djb.TEST_PENA_RUPTURE)
    return case


def compare(output_dir: Path, uplift: bool) -> dict:
    """Code_Aster's joint opening at full water against the hand calculation."""
    joint = json.loads((output_dir / djb.META_NAME).read_text(encoding="utf-8"))["joint"]
    strength = joint["adhesion_pa"] / joint["friction_coefficient"] if joint["law"] == "JOINT_MECA_FROT" else 0.0
    expected = cracked_base(uplift, strength)
    with np.load(output_dir / djb.ARRAYS_NAME, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    with np.load(output_dir / djb.STEPS_NAME, allow_pickle=False) as steps:
        data = {key: steps[key] for key in steps.files}
    cells = arrays["penta6_joints"]
    geometry = djb.joint_geometry(arrays, cells)
    result = {"uplift": uplift, "expected": expected, "converged_instants": data["instant"].tolist()}
    if not len(data["instant"]) or data["instant"][-1] != len(djb.FILL_FRACTIONS):
        return {**result, "passed": False, "message": "The run did not reach full water"}
    cell_ids, gauss_cells = data["joint_cell_ids"].astype(np.int64), data["gauss_cells"].astype(np.int64)
    opening = djb._per_cell(data["V7"][-1], gauss_cells, cell_ids, np.mean)
    wanted = analytic_opening(geometry["centroid_m"][:, 0], expected)
    displacement = data["displacement_m"][-1]
    jump = np.einsum("ckj,cj->ck", displacement[cells[:, 3:]] - displacement[cells[:, :3]],
                     geometry["normal"]).mean(axis=1)
    opened = opening > djb.OPEN_TOLERANCE_M
    traction = djb._per_cell(data["SIGN"][-1], gauss_cells, cell_ids, np.mean)
    result.update(
        gauss_points_per_joint_cell=len(gauss_cells) / len(cell_ids),
        largest_opening_m=float(opening.max()), expected_largest_opening_at_cell_m=float(wanted.max()),
        opened_length_m=float(geometry["area_m2"][opened].sum() / WIDTH_M),
        largest_joint_compression_pa=float(-traction.min()),
        profile_error=float(np.abs(opening - wanted).max() / expected["heel_opening_m"]),
        opening_sign_and_order_error_m=float(np.abs(jump - opening).max()),
        joint_normal_force_n=float((traction * geometry["area_m2"]).sum()))
    result["passed"] = bool(result["profile_error"] <= ACCEPTED_PROFILE_ERROR
                            and result["opening_sign_and_order_error_m"] <= 1e-3 * expected["heel_opening_m"])
    return result


def cooling_case() -> dict:
    """The dry friction case, then the wall cooled at full water."""
    return {**fixture_case(False, "JOINT_MECA_FROT"),
            "purpose": "Two-block check of the temperature load against free thermal contraction",
            "cooling": {"expansion_per_k": djb.PROVISIONAL_EXPANSION_PER_K,
                        "closure_temperature_c": COOLING_CHECK_CLOSURE_C,
                        "wall_temperatures_c": list(COOLING_CHECK_TEMPERATURES_C)}}


def compare_cooling(output_dir: Path) -> dict:
    """Change of the wall's size between full water and the last cooling step against free contraction.

    Measured on the top face and the upper half of the wall, away from the contact that restrains
    it. The block below is not cooled and must keep its size.
    """
    meta = json.loads((output_dir / djb.META_NAME).read_text(encoding="utf-8"))
    levels, cooling = meta["levels"], meta["joint"]["cooling"]
    with np.load(output_dir / djb.ARRAYS_NAME, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    with np.load(output_dir / djb.STEPS_NAME, allow_pickle=False) as steps:
        instants, displacement = steps["instant"].tolist(), steps["displacement_m"]
    full_water = float(len(djb.FILL_FRACTIONS))
    if not instants or instants[-1] != levels["instants"][-1] or full_water not in instants:
        return {"passed": False, "message": "The run did not reach the last cooling step",
                "converged_instants": instants}
    nodes, body = arrays["nodes"], arrays["concrete_body_ids"]
    wall = np.unique(arrays["tetra4_concrete"][body == 1])
    block = np.unique(arrays["tetra4_concrete"][body == 3])
    change = displacement[-1] - displacement[instants.index(full_water)]
    strain = cooling["expansion_per_k"] * -levels["temperature_change_k"][-1]
    top, middle = BLOCK_HEIGHT_M + WALL_HEIGHT_M, BLOCK_HEIGHT_M + WALL_HEIGHT_M / 2

    def at(ids: np.ndarray, axis: int, value: float) -> np.ndarray:
        return ids[np.isclose(nodes[ids, axis], value)]

    def lengthening(ids: np.ndarray, axis: int, low: float, high: float) -> float:
        return float(change[at(ids, axis, high), axis].mean() - change[at(ids, axis, low), axis].mean())

    wall_top = at(wall, 2, top)
    measured = {"across_top_m": lengthening(wall_top, 0, 0.0, THICKNESS_M),
                "along_top_m": lengthening(wall_top, 1, 0.0, WIDTH_M),
                "upper_half_height_m": lengthening(wall, 2, middle, top)}
    expected = {"across_top_m": -strain * THICKNESS_M, "along_top_m": -strain * WIDTH_M,
                "upper_half_height_m": -strain * WALL_HEIGHT_M / 2}
    errors = {name: abs(measured[name] - expected[name]) / abs(expected[name]) for name in expected}
    block_change = lengthening(at(block, 2, BLOCK_HEIGHT_M), 0, 0.0, THICKNESS_M)
    return {"temperature_change_k": levels["temperature_change_k"][-1], "expansion_per_k": cooling["expansion_per_k"],
            "measured": measured, "expected": expected, "relative_error": errors,
            "block_top_lengthening_m": block_change, "converged_instants": instants,
            "passed": bool(max(errors.values()) <= ACCEPTED_CONTRACTION_ERROR
                           and abs(block_change) <= 0.1 * strain * THICKNESS_M)}


def run_cooling_fixture(output_root: Path = djb.DEFAULT_OUTPUT_ROOT, time_limit_s: int = 300) -> dict:
    name = "fixture_friction_dry_cooling"
    output_dir = output_root / name
    arrays = fixture_arrays()
    djb.progress(f"=== Two-block check '{name}': the wall's thermal contraction against expansion x temperature drop")
    djb.write_case(output_dir, arrays, {"schema_version": 1, "bedrock_analysis_mode": "elastic_layer", "fixture": True},
                   fixture_base_comm(arrays), fixture_loads(), cooling_case())
    outcome = djb.solve(output_dir, time_limit_s)
    if not (output_dir / djb.STEPS_NAME).is_file():
        check = {"passed": False, "message": f"No results ({outcome.get('status')}); see run/solver.live.log"}
    else:
        check = compare_cooling(output_dir)
        djb.summarise(output_dir)
    check["run"] = outcome
    (output_dir / "fixture_check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    line = f"{name}: {'PASS' if check['passed'] else 'FAIL'}" + (
        "  contraction of the wall for {:g} K: ".format(-check["temperature_change_k"])
        + ", ".join(f"{key.removesuffix('_m').replace('_', ' ')} {check['measured'][key] * 1e3:.4f} mm "
                    f"(expected {check['expected'][key] * 1e3:.4f} mm, {check['relative_error'][key]:.1%} off)"
                    for key in check["expected"])
        + f"; uncooled block {check['block_top_lengthening_m'] * 1e3:.5f} mm"
        if "measured" in check else f"  {check.get('message')}")
    print(line, flush=True)
    djb.progress(line)
    return {"passed": check["passed"], "results": {name: check}}


def run_fixture(output_root: Path = djb.DEFAULT_OUTPUT_ROOT, time_limit_s: int = 300) -> dict:
    arrays = fixture_arrays()
    manifest = {"schema_version": 1, "bedrock_analysis_mode": "elastic_layer", "fixture": True}
    results = {}
    for name, uplift, law in (("fixture_dry", False, "JOINT_MECA_RUPT"), ("fixture_uplift", True, "JOINT_MECA_RUPT"),
                              ("fixture_friction_dry", False, "JOINT_MECA_FROT"),
                              ("fixture_friction_uplift", True, "JOINT_MECA_FROT")):
        output_dir = output_root / name
        djb.progress(f"=== Two-block check '{name}' ({law}, {'water in the joint' if uplift else 'dry joint'})")
        djb.write_case(output_dir, arrays, manifest, fixture_base_comm(arrays), fixture_loads(),
                       fixture_case(uplift, law))
        outcome = djb.solve(output_dir, time_limit_s)
        if not (output_dir / djb.STEPS_NAME).is_file():
            results[name] = {"passed": False, "message": f"No results ({outcome.get('status')}); see run/solver.live.log"}
        else:
            results[name] = compare(output_dir, uplift)
            djb.summarise(output_dir)
        results[name]["run"] = outcome
        (output_dir / "fixture_check.json").write_text(json.dumps(results[name], indent=2) + "\n", encoding="utf-8")
        check = results[name]
        line = (f"{name}: {'PASS' if check['passed'] else 'FAIL'}"
                + (f"  profile error {check['profile_error']:.1%} of the heel opening "
                   f"({check['expected']['heel_opening_m'] * 1e3:.4f} mm expected), opened length "
                   f"{check['opened_length_m']:.3f} m (expected {check['expected']['open_length_m']:.3f} m)"
                   if "profile_error" in check else f"  {check.get('message')}"))
        print(line, flush=True)
        djb.progress(line)
    return {"passed": all(result["passed"] for result in results.values()), "results": results}
