"""Tests for the Code_Aster corroboration of FEniCSx automatic cohesive cracking."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Aster import automatic_cohesive as ac  # noqa: E402

XS = (0.0, 0.5, 1.0)
YS = (0.0, 1.0)
SPRING_N_PER_M3 = 1.0e12


def column_fixture() -> dict:
    """A 1 m x 1 m column: bedrock z -1..0 and foundation 0..1 share nodes; the wall
    (z 1..3) has its own z = 1 nodes joined to the foundation by interface springs."""
    nodes, index = [], {}

    def node(block, i, j, k, z):
        key = (block, i, j, k)
        if key not in index:
            index[key] = len(nodes)
            nodes.append((XS[i], YS[j], z))
        return index[key]

    blocks = {"lower": (-1.0, 0.0, 1.0), "wall": (1.0, 2.0, 3.0)}
    tets = {"bedrock": [], "foundation": [], "wall": []}
    for block, levels in blocks.items():
        for k in range(len(levels) - 1):
            region = "wall" if block == "wall" else ("bedrock" if levels[k] < 0 else "foundation")
            for i in range(len(XS) - 1):
                for j in range(len(YS) - 1):
                    v = {(a, b, c): node(block, i + a, j + b, k + c, levels[k + c])
                         for a in (0, 1) for b in (0, 1) for c in (0, 1)}
                    for order in itertools.permutations(range(3)):
                        corner = [0, 0, 0]
                        path = [v[tuple(corner)]]
                        for axis in order:
                            corner[axis] = 1
                            path.append(v[tuple(corner)])
                        tets[region].append(path)
    points = np.array(nodes, dtype=float)

    def oriented(rows):
        rows = np.array(rows, dtype=np.int64)
        x = points[rows]
        negative = np.linalg.det(x[:, 1:] - x[:, :1]) < 0
        rows[negative] = rows[negative][:, [0, 2, 1, 3]]
        return rows

    concrete = oriented(tets["wall"] + tets["foundation"])
    bedrock = oriented(tets["bedrock"])
    body = np.array([1] * len(tets["wall"]) + [3] * len(tets["foundation"]), dtype=np.int64)

    def boundary(rows):
        faces = np.sort(rows[:, ac._TET_FACES].reshape(-1, 3), axis=1)
        unique, counts = np.unique(faces, axis=0, return_counts=True)
        return unique[counts == 1]

    concrete_faces = boundary(concrete)
    bedrock_faces = boundary(bedrock)

    def on(faces, axis, value):
        return faces[np.all(np.isclose(points[faces][:, :, axis], value), axis=1)]

    sides = np.vstack([on(bedrock_faces, 0, 0.0), on(bedrock_faces, 0, 1.0),
                       on(bedrock_faces, 1, 0.0), on(bedrock_faces, 1, 1.0)])
    wall_bottom = sorted({index[("wall", i, j, 0)] for i in range(3) for j in range(2)})
    pairs, areas = [], []
    for i in range(3):
        for j in range(2):
            pairs.append((index[("wall", i, j, 0)], index[("lower", i, j, 2)]))
            areas.append((0.25 if i in (0, 2) else 0.5) * 0.5)
    support = np.array(sorted({index[("lower", i, j, 0)] for i in range(3) for j in range(2)}), dtype=np.int64)
    areas = np.array(areas)
    assert len(wall_bottom) == 6
    return {
        "nodes": points,
        "tetra4_concrete": concrete,
        "tetra4_bedrock": bedrock,
        "tri3_upstream": on(concrete_faces, 0, 0.0),
        "tri3_downstream": on(concrete_faces, 0, 1.0),
        "tri3_crest": on(concrete_faces, 2, 3.0),
        "tri3_rock_bottom": on(bedrock_faces, 2, -1.0),
        "tri3_rock_sides": sides,
        "poi1_rock_bottom_nodes": support,
        "seg2_wall_foundation_springs": np.array(pairs, dtype=np.int64),
        "wall_foundation_tributary_area_m2": areas,
        "interface_stiffness_n_per_m": np.repeat(areas[:, None] * SPRING_N_PER_M3, 3, axis=1),
        "interface_names": np.array(["wall_foundation"] * len(pairs)),
        "support_nodes": support,
        "support_matrices_n_per_m": np.repeat(np.eye(3)[None] * 1.0e13, len(support), axis=0),
        "concrete_body_ids": body,
        "source_node_ids": np.arange(len(points), dtype=np.int64),
    }


def plane_faces(arrays: dict, axis: int, value: float, keep) -> np.ndarray:
    points = arrays["nodes"]
    faces = np.unique(np.sort(arrays["tetra4_concrete"][:, ac._TET_FACES].reshape(-1, 3), axis=1), axis=0)
    selected = [face for face in faces
                if np.allclose(points[face][:, axis], value) and keep(points[face])]
    return np.array(selected, dtype=np.int64)


def crack_faces(arrays: dict) -> np.ndarray:
    """Upstream half of the z = 2 wall section (x <= 0.5): its x = 0.5 edge is a crack tip."""
    return plane_faces(arrays, 2, 2.0, lambda p: p[:, 0].max() <= 0.5 + 1e-9)


def fixture_manifest(arrays: dict) -> dict:
    springs = [{"group": f"SPR_WF_{i:05d}", "interface": "wall_foundation",
                "tributary_area_m2": float(area), "stiffness_n_per_m": stiffness.tolist()}
               for i, (area, stiffness) in enumerate(zip(arrays["wall_foundation_tributary_area_m2"],
                                                         arrays["interface_stiffness_n_per_m"]))]
    return {"compact_spring_groups": bool(arrays.get("compact_spring_groups", False)),
            "wall_foundation_springs": springs,
            "support_springs": [{"group": f"SPR_OUT_{i:05d}", "matrix_n_per_m": matrix.tolist()}
                                for i, matrix in enumerate(arrays["support_matrices_n_per_m"])]}


def fixture_base_comm(manifest: dict, gradient_pa_per_m: float) -> str:
    """The same section layout as Aster.generate_three_body_case.render_comm."""
    from Aster.generate_three_body_case import grouped_spring_entries
    compact = manifest.get("compact_spring_groups", False)
    springs = ",\n        ".join(
        f"_F(GROUP_MA={s['group']!r}, REPERE='GLOBAL', CARA='K_T_D_L', VALE={s['vale']!r})"
        for s in grouped_spring_entries([
            {"group": s["group"], "vale": tuple(s["stiffness_n_per_m"])}
            for s in manifest["wall_foundation_springs"]], "K_IF_" if compact else None))
    outer = ",\n        ".join(
        f"_F(GROUP_MA={s['group']!r}, REPERE='GLOBAL', CARA='K_T_N', SYME='NON', VALE={s['vale']!r})"
        for s in grouped_spring_entries([
            {"group": s["group"], "vale": tuple(v for row in s["matrix_n_per_m"] for v in row)}
            for s in manifest["support_springs"]], "K_OUT_" if compact else None))
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
            GROUP_MA=('CONCRETE', 'BEDROCK', 'UPSTREAM', 'DOWNSTREAM', 'CREST', 'ROCK_BOTTOM', 'ROCK_SIDES'),
            PHENOMENE='MECANIQUE',
            MODELISATION='3D',
        ),
        _F(GROUP_MA='OUTER_SPRINGS', PHENOMENE='MECANIQUE', MODELISATION='DIS_T'),
        _F(GROUP_MA='WALL_FOUNDATION_SPRINGS', PHENOMENE='MECANIQUE', MODELISATION='DIS_T'),
    ),
)

CONCRETE_MAT = DEFI_MATERIAU(ELAS=_F(E=30e9, NU=0.2, RHO=2400.0))
BEDROCK_MAT = DEFI_MATERIAU(ELAS=_F(E=30e9, NU=0.25, RHO=2600.0))

CHMAT = AFFE_MATERIAU(
    MAILLAGE=MAIL,
    AFFE=(
        _F(GROUP_MA='CONCRETE', MATER=CONCRETE_MAT),
        _F(GROUP_MA='BEDROCK', MATER=BEDROCK_MAT),
    ),
)

GRAVITE = AFFE_CHAR_MECA(MODELE=MODELE, PESANTEUR=_F(GRAVITE=9.81, DIRECTION=(0.0, 0.0, -1.0)))
UPSTREAM_PRES = FORMULE(NOM_PARA='Z', VALE='{gradient_pa_per_m:.12g} * max(3.0 - Z, 0.0)')
PRESSIONS = AFFE_CHAR_MECA_F(MODELE=MODELE, PRES_REP=_F(GROUP_MA='UPSTREAM', PRES=UPSTREAM_PRES))
CREST_SURCHARGE = AFFE_CHAR_MECA(MODELE=MODELE, PRES_REP=_F(GROUP_MA='CREST', PRES=0.0))

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
    EXCIT=(_F(CHARGE=GRAVITE), _F(CHARGE=PRESSIONS), _F(CHARGE=CREST_SURCHARGE)),
    SOLVEUR=_F(METHODE='MULT_FRONT'),
)

FIN()
'''


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.arrays = column_fixture()

    def test_partial_plane_splits_only_interior_crack_nodes(self):
        faces = crack_faces(self.arrays)
        split = ac.split_for_joints(self.arrays, faces)
        points = self.arrays["nodes"]
        crack_nodes = np.unique(faces)
        interior = crack_nodes[points[crack_nodes][:, 0] < 0.25]
        self.assertEqual(split["split_nodes"], len(interior))
        self.assertEqual(split["added_nodes"], len(interior))
        joints = split["joints"]
        self.assertEqual(joints.shape, (len(faces), 6))
        new_points = split["arrays"]["nodes"]
        np.testing.assert_allclose(new_points[joints[:, :3]], new_points[joints[:, 3:]])
        tip = np.isclose(new_points[joints[:, :3]][:, :, 0], 0.5)
        ties = split["arrays"]["tie_node_pairs"]
        self.assertEqual(split["tip_ties"], len(ties))
        self.assertTrue(np.all(joints[:, :3] != joints[:, 3:]))
        self.assertTrue(np.array_equal(np.isin(joints[:, 3:], ties[:, 1]), tip))
        self.assertTrue(np.all(np.isin(joints[:, :3][tip], ties[:, 0])))
        # Tie copies belong to no tetrahedron.
        self.assertFalse(np.any(np.isin(split["arrays"]["tetra4_concrete"], ties[:, 1])))
        # Minus is below the section and plus above. Code_Aster measures a positive (opening)
        # normal jump when the first triangle's right-hand normal points away from the plus side.
        for row in joints:
            a, b, c = new_points[row[:3]]
            self.assertLess(np.cross(b - a, c - a)[2], 0)
        below = split["arrays"]["tetra4_concrete"][
            points[self.arrays["tetra4_concrete"]].mean(axis=1)[:, 2] < 2.0]
        self.assertTrue(np.all(np.isin(joints[:, :3], below)))
        self.assertEqual(len(split["arrays"]["source_node_ids"]), len(new_points))

    def test_loaded_faces_follow_their_tetrahedron(self):
        split = ac.split_for_joints(self.arrays, crack_faces(self.arrays))
        upstream = split["arrays"]["tri3_upstream"]
        concrete = split["arrays"]["tetra4_concrete"]
        tet_nodes = [set(row) for row in concrete.tolist()]
        for tri in upstream.tolist():
            self.assertTrue(any(set(tri) <= nodes for nodes in tet_nodes))
        self.assertGreater(len(np.setdiff1d(upstream, self.arrays["tri3_upstream"])), 0)

    def test_interface_springs_are_shared_between_node_copies(self):
        faces = plane_faces(self.arrays, 0, 0.5, lambda p: p[:, 2].min() >= 1.0 - 1e-9 and p[:, 2].max() <= 2.0 + 1e-9)
        split = ac.split_for_joints(self.arrays, faces)
        arrays = split["arrays"]
        self.assertEqual(len(split["spring_shares"]), 2)
        for share in split["spring_shares"]:
            # The Kuhn diagonals give the node one interface triangle on one side and two on the other.
            np.testing.assert_allclose(sorted(share["fractions"]), [1 / 3, 2 / 3])
        self.assertEqual(len(arrays["seg2_wall_foundation_springs"]), 8)
        np.testing.assert_allclose(arrays["wall_foundation_tributary_area_m2"].sum(),
                                   self.arrays["wall_foundation_tributary_area_m2"].sum())
        np.testing.assert_allclose(arrays["interface_stiffness_n_per_m"].sum(axis=0),
                                   self.arrays["interface_stiffness_n_per_m"].sum(axis=0))
        manifest = ac.split_manifest(fixture_manifest(self.arrays), split)
        self.assertEqual([s["group"] for s in manifest["wall_foundation_springs"]],
                         [f"SPR_WF_{i:05d}" for i in range(8)])
        np.testing.assert_allclose([s["stiffness_n_per_m"] for s in manifest["wall_foundation_springs"]],
                                   arrays["interface_stiffness_n_per_m"])

    def test_faces_outside_the_concrete_are_rejected(self):
        faces = np.unique(np.sort(self.arrays["tetra4_bedrock"][:, ac._TET_FACES].reshape(-1, 3), axis=1),
                          axis=0)
        inner = [f for f in faces if np.allclose(self.arrays["nodes"][f][:, 0], 0.5)]
        with self.assertRaisesRegex(ac.CorroborationError, "two concrete tetrahedra"):
            ac.split_for_joints(self.arrays, np.array(inner))


def star_split(subdivisions: int = 4) -> dict:
    """A bedrock cube cut into tetrahedra that all share its centre node."""
    grid = np.linspace(-1.0, 1.0, subdivisions + 1)
    points, index = [np.zeros(3)], {}

    def node(p):
        key = tuple(np.round(p, 9))
        if key not in index:
            index[key] = len(points)
            points.append(np.asarray(p, dtype=float))
        return index[key]

    tets = []
    for axis in range(3):
        for side in (-1.0, 1.0):
            u, v = [a for a in range(3) if a != axis]
            for i in range(subdivisions):
                for j in range(subdivisions):
                    corners = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.zeros(3)
                        p[axis], p[u], p[v] = side, grid[i + di], grid[j + dj]
                        corners.append(node(p))
                    for tri in ((corners[0], corners[1], corners[2]), (corners[0], corners[2], corners[3])):
                        tets.append([0, *tri])
    empty3 = np.zeros((0, 3), dtype=np.int64)
    arrays = {key: empty3 for key in ac.LOADED_TRI_KEYS}
    arrays.update({
        "nodes": np.asarray(points), "tetra4_concrete": np.zeros((0, 4), dtype=np.int64),
        "tetra4_bedrock": np.asarray(tets, dtype=np.int64), "source_node_ids": np.arange(len(points)),
        "seg2_wall_foundation_springs": np.zeros((0, 2), dtype=np.int64),
        "support_nodes": np.zeros(0, dtype=np.int64), "tie_node_pairs": np.zeros((0, 2), dtype=np.int64),
    })
    return {"arrays": arrays, "joints": np.zeros((0, 6), dtype=np.int64)}


class NeighbourhoodTests(unittest.TestCase):
    def test_high_valence_node_is_split_into_tied_copies(self):
        split = star_split()
        before = split["arrays"]["tetra4_bedrock"].copy()
        points = split["arrays"]["nodes"].copy()
        self.assertEqual(len(before), 192)
        added = ac.relieve_neighbourhoods(split, threshold=150)
        arrays = split["arrays"]
        self.assertGreater(added, 0)
        over, _ = ac._neighbour_counts([arrays["tetra4_bedrock"]], len(arrays["nodes"]), 150)
        self.assertEqual(len(over), 0)
        pairs = arrays["tie_node_pairs"]
        self.assertEqual(len(pairs), added)
        self.assertTrue(np.all(pairs[:, 0] == 0))
        np.testing.assert_allclose(arrays["nodes"][pairs[:, 1]], 0.0)
        # Same tetrahedra in the same places: only the centre node id changed.
        np.testing.assert_allclose(arrays["nodes"][arrays["tetra4_bedrock"]], points[before])
        self.assertEqual(len(arrays["source_node_ids"]), len(arrays["nodes"]))
        self.assertTrue(np.all(arrays["source_node_ids"][pairs[:, 1]] == 0))

    def test_meshes_within_the_limit_are_unchanged(self):
        split = ac.split_for_joints(column_fixture(), crack_faces(column_fixture()))
        nodes = len(split["arrays"]["nodes"])
        self.assertEqual(ac.relieve_neighbourhoods(split), 0)
        self.assertEqual(len(split["arrays"]["nodes"]), nodes)

    def test_pinned_nodes_are_never_split(self):
        split = star_split()
        split["arrays"]["support_nodes"] = np.array([0])
        with self.assertRaisesRegex(ac.CorroborationError, "too many neighbours"):
            ac.relieve_neighbourhoods(split, threshold=150)


class CaseTests(unittest.TestCase):
    def test_reduced_case_does_not_require_a_bedrock_assignment(self):
        base = (ROOT / "Aster" / "cases" / "01_three_body_baseline.comm").read_text(encoding="utf-8")
        base = base.replace("        _F(GROUP_MA='BEDROCK', MATER=BEDROCK_MAT),\n", "")
        joint = {"tensile_strength_pa": 8e5, "fracture_energy_j_m2": 200., "pena_adherence": 1e-4,
                 "concrete_youngs_modulus_pa": 3e10, "concrete_poissons_ratio": .2}
        comm = ac.render_cohesive_comm(base, joint, [.5, 1.], Path("s.npz"), Path("s.json"))
        self.assertIn("_F(GROUP_MA='JOINTS', MATER=JOINT_MAT)", comm)
        compile(comm, "reduced.comm", "exec")

    def test_load_factors_follow_fenicsx(self):
        self.assertEqual(ac.load_factors({"load_factor": 1.0}), [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.assertEqual(ac.load_factors({"load_factor": 1.25})[-2:], [1.2, 1.25])

    def test_joint_law_matches_the_fenicsx_softening(self):
        arrays = column_fixture()
        joint = ac.joint_parameters(arrays, crack_faces(arrays), {"tensile_strength_pa": 8e5,
                                                                 "fracture_energy_j_m2": 200.0}, 3e10, 0.2)
        self.assertAlmostEqual(joint["critical_opening_m"], 5e-4)
        self.assertAlmostEqual(joint["initial_stiffness_pa_per_m"],
                               ac.JOINT_PENALTY_FACTOR * 3e10 / joint["mean_facet_edge_m"], delta=1.0)

    def test_committed_baseline_case_has_every_patched_section(self):
        base = (ROOT / "Aster" / "cases" / "01_three_body_baseline.comm").read_text(encoding="utf-8")
        joint = {"tensile_strength_pa": 8e5, "fracture_energy_j_m2": 200.0, "pena_adherence": 1e-4,
                 "concrete_youngs_modulus_pa": 3.5e10, "concrete_poissons_ratio": 0.2}
        comm = ac.render_cohesive_comm(base, joint, [0.5, 1.0], Path("s.npz"), Path("s.json"), tie_pairs=3)
        self.assertNotIn("MECA_STATIQUE", comm)
        for text in ("MODELISATION='3D_JOINT'", "_F(CHARGE=TIE)", "RELATION='CZM_LIN_REG'",
                     "_F(GROUP_MA='JOINTS', MATER=JOINT_MAT)", "TARGETS = [0.5, 1]", "FONC_MULT=RAMP"):
            self.assertIn(text, comm)
        compile(comm, "case.comm", "exec")
        with self.assertRaisesRegex(ac.CorroborationError, "solve"):
            ac.render_cohesive_comm(base.replace("RESU = MECA_STATIQUE(", "RESU = X("), joint, [1.0],
                                    Path("s.npz"), Path("s.json"))


class IntegratedJointTests(unittest.TestCase):
    def test_quadrilateral_patches_preserve_affine_energy_forces_and_moments(self):
        arrays = column_fixture()
        split = ac.split_for_joints(arrays, crack_faces(arrays))
        ac.integrate_joint_faces(split)
        stored = split["arrays"]
        points = stored["nodes"]
        np.testing.assert_allclose(
            points[stored["interpolation_nodes"]],
            np.einsum("ni,nij->nj", stored["interpolation_weights"],
                      points[stored["interpolation_parents"]]))
        self.assertEqual(len(stored["hexa8_joints"]), 3 * len(split["joints"]))
        # Consistent P1 surface mass matrix fixes all three affine jump modes.
        for face, triangle in enumerate(split["joints"][:, :3]):
            vertices = points[triangle]
            area = ac._tri_area(points, triangle)
            mass = np.zeros((3, 3))
            integrated_area = 0.0
            for quad in stored["hexa8_joints"][3 * face:3 * face + 3, :4]:
                xy = points[quad]
                barycentric = np.linalg.lstsq(
                    np.vstack((vertices.T, np.ones(3))),
                    np.vstack((xy.T, np.ones(4))), rcond=None)[0].T
                for r, s in itertools.product((-1 / np.sqrt(3), 1 / np.sqrt(3)), repeat=2):
                    shape = np.array([(1-r)*(1-s), (1+r)*(1-s), (1+r)*(1+s), (1-r)*(1+s)]) / 4
                    dr = np.array([-(1-s), 1-s, 1+s, -(1+s)]) / 4
                    ds = np.array([-(1-r), -(1+r), 1+r, 1-r]) / 4
                    weight = np.linalg.norm(np.cross(dr @ xy, ds @ xy))
                    trace = shape @ barycentric
                    mass += weight * np.outer(trace, trace)
                    integrated_area += weight
            expected = area * (np.ones((3, 3)) + np.eye(3)) / 12
            np.testing.assert_allclose(mass, expected, atol=1e-14)
            self.assertAlmostEqual(integrated_area, area)
            jump = vertices @ np.array([[.2, -.3, .1], [.1, .4, -.2], [-.3, .2, .5]]) + .01
            np.testing.assert_allclose(mass @ jump, expected @ jump, atol=1e-14)
            np.testing.assert_allclose(np.cross(vertices, mass @ jump).sum(axis=0),
                                       np.cross(vertices, expected @ jump).sum(axis=0), atol=1e-14)
            self.assertGreater(np.linalg.eigvalsh(mass)[0], 0)

    def test_interpolation_case_compiles_and_does_not_add_supports(self):
        arrays = column_fixture()
        split = ac.split_for_joints(arrays, crack_faces(arrays))
        ac.integrate_joint_faces(split)
        joint = ac.joint_parameters(arrays, crack_faces(arrays),
                                   {"tensile_strength_pa": 3e5, "fracture_energy_j_m2": 100}, 30e9, .2)
        comm = ac.render_cohesive_comm(fixture_base_comm(fixture_manifest(arrays), 4e5), joint,
                                      [.5, 1.], Path("s.npz"), Path("s.json"),
                                      interpolation=split["arrays"])
        compile(comm, "case.comm", "exec")
        self.assertIn("_F(CHARGE=INTERPOLATION)", comm)
        self.assertNotIn("DDL_IMPO", comm)
        np.testing.assert_array_equal(split["arrays"]["support_nodes"], arrays["support_nodes"])


def fake_fenicsx(directory: Path, faces_count: int, *, initiated=None, opened=None, factor=1.0,
                 first=0.75) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    initiated = np.zeros(faces_count, bool) if initiated is None else initiated
    opened = np.zeros(faces_count, bool) if opened is None else opened
    status = np.where(opened, "softening", np.where(initiated, "closed", "bonded"))
    np.savez(directory / "cohesive_state.npz", areas_m2=np.full(faces_count, 0.125), initiated=initiated,
             face_status=status, load_factor=np.array(factor))
    (directory / "summary.json").write_text(json.dumps({
        "tensile_strength_pa": 8e5, "fracture_energy_j_m2": 200.0, "load_factor": factor,
        "design_load": {"first_initiation_load_factor": first},
        "network": {"maximum_face_opening_m": 1e-5, "wet_area_m2": 0.0}}), encoding="utf-8")
    return directory


class PostprocessTests(unittest.TestCase):
    def test_wsl_launcher_failure_overrides_completed_status(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run").mkdir()
            (root / ac.CASE_NAME).write_text("SOLVEUR=_F(METHODE='MULT_FRONT')")
            calls = []

            def execute(distribution, installation, directory, program, timeout, log):
                calls.append(program)
                if "convert_three_body_to_med.py" in " ".join(program):
                    (root / "mesh.med").write_text("mesh")
                    return subprocess.CompletedProcess(program, 0, "converted", "")
                (root / "run" / ac.STATUS_NAME).write_text('{"status":"completed"}')
                return subprocess.CompletedProcess(program, 1, "launcher failed", "")

            with patch("Aster.whole_model_crack.find_linux_installation", return_value="/opt/aster"), \
                    patch("Aster.whole_model_crack.wsl_path", side_effect=str), \
                    patch("Aster.run_case.run_wsl_solver", side_effect=execute), \
                    patch("Aster.automatic_cohesive.os.name", "nt"):
                status = ac.solve(root, 60)
            self.assertEqual(len(calls), 2)
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["returncode"], 1)
            self.assertEqual(status["runtime"], "wsl")

    def test_aster_only_preparation_has_no_fabricated_supporting_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = self.write_case(root, [[0] * 4, [0] * 4], [[0] * 4, [0] * 4],
                                     [[1e5] * 4, [2e5] * 4], [.5, 1.])
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "summary.json").write_text(json.dumps({"status": "prepared_aster_candidates"}))
            np.savez(candidate / "cohesive_state.npz", areas_m2=np.full(4, .125))
            report = ac.postprocess(root, candidate, status)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["corroboration"]["verdict"], "not_available")
            self.assertNotIn("fenicsx_initiated_faces", report["corroboration"])

    def write_case(self, root: Path, damage, jump, traction, factors, status="completed"):
        joint = {"elastic_limit_opening_m": 1e-8, "tensile_strength_pa": 8e5,
                 "concrete_youngs_modulus_pa": 3e10, "concrete_poissons_ratio": 0.2}
        (root / "run").mkdir(parents=True)
        (root / "case_inputs.json").write_text(json.dumps({
            "joint": joint, "load_factors": [0.5, 1.0], "candidate_faces": 4, "split_nodes": 1,
            "added_nodes": 1}), encoding="utf-8")
        np.savez(root / "run" / ac.STEPS_NAME, load_factor=np.array(factors), damage_state=np.array(damage),
                 normal_jump_m=np.array(jump), normal_traction_pa=np.array(traction),
                 dissipated_energy_j_m2=np.zeros_like(np.array(damage)))
        return {"status": status}

    def test_matching_crack_is_corroborated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = self.write_case(root, [[0, 0, 0, 0], [1, 1, 0, 0]], [[0, 0, 0, 0], [2e-5, 0, 0, 0]],
                                     [[4e5, 2e5, 1e5, 0], [8e5, 8e5, 3e5, 0]], [0.5, 1.0])
            fenicsx = fake_fenicsx(root / "f", 4, initiated=np.array([1, 1, 0, 0], bool),
                                   opened=np.array([1, 0, 0, 0], bool))
            report = ac.postprocess(root, fenicsx, status)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["first_initiation_bracket"], [0.5, 1.0])
        self.assertAlmostEqual(report["linear_first_initiation_estimate"], 1.0)
        self.assertEqual(report["design_load"]["opened_faces"], 1)
        corroboration = report["corroboration"]
        self.assertEqual(corroboration["verdict"], "corroborated")
        self.assertEqual(corroboration["initiated_overlap"], 1.0)

    def test_disagreement_and_early_stop_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = self.write_case(root, [[0, 0, 0, 0], [0, 0, 0, 0]], [[0] * 4, [0] * 4],
                                     [[1e5] * 4, [2e5] * 4], [0.5, 1.0])
            fenicsx = fake_fenicsx(root / "f", 4, initiated=np.array([1, 0, 0, 0], bool),
                                   opened=np.array([1, 0, 0, 0], bool))
            self.assertEqual(ac.postprocess(root, fenicsx, status)["corroboration"]["verdict"],
                             "not_corroborated")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = self.write_case(root, [[0, 0, 0, 0]], [[0] * 4], [[1e5] * 4], [0.5], status="stopped")
            report = ac.postprocess(root, fake_fenicsx(root / "f", 4), status)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["corroboration"]["verdict"], "incomplete")

    def test_integrated_patch_results_keep_original_facet_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            damage = np.zeros((2, 12))
            jump = np.zeros_like(damage)
            damage[1, 1], jump[1, 1] = 1., 2e-5
            status = self.write_case(root, damage, jump, np.ones_like(damage) * 1e5, [.5, 1.])
            path = root / "case_inputs.json"
            inputs = json.loads(path.read_text())
            inputs.update(joint_integration="three_HEXA8_FPG4_patches_exact_P1_trace",
                          native_joint_cells=12)
            path.write_text(json.dumps(inputs))
            np.savez(root / ac.SPLIT_ARRAYS_NAME, joint_parent_faces=np.repeat(np.arange(4), 3))
            candidate = fake_fenicsx(root / "f", 4)
            report = ac.postprocess(root, candidate, status)
            self.assertEqual(report["design_load"]["initiated_faces"], 1)
            self.assertEqual(report["design_load"]["opened_faces"], 1)
            self.assertAlmostEqual(report["design_load"]["opened_area_m2"], .125)
            np.savez(root / ac.SPLIT_ARRAYS_NAME, joint_parent_faces=np.zeros(12, dtype=int))
            with self.assertRaisesRegex(ac.CorroborationError, "original candidate facets"):
                ac.postprocess(root, candidate, status)


def _aster_launcher():
    from Aster.run_case import detect_installation
    return detect_installation()["launcher_path"]


@unittest.skipUnless(_aster_launcher(), "Code_Aster is not installed")
class AsterFixtureRun(unittest.TestCase):
    """Real Code_Aster run of the column with a partial cohesive section (about a minute)."""

    def test_no_candidate_facets_runs_real_native_equilibrium(self):
        arrays = column_fixture()
        split = ac.split_for_joints(arrays, np.empty((0, 3), dtype=np.int64))
        ac.integrate_joint_faces(split)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / ac.SPLIT_ARRAYS_NAME, penta6_joints=split["joints"], **split["arrays"])
            joint = ac.joint_parameters(arrays, np.empty((0, 3), dtype=np.int64),
                                       {"tensile_strength_pa": 3e5, "fracture_energy_j_m2": 100}, 30e9, .2)
            joint["has_joints"] = False
            (root / "run").mkdir()
            comm = ac.render_cohesive_comm(fixture_base_comm(fixture_manifest(arrays), 4e3), joint, [.5, 1.],
                                          (root / "run" / ac.STEPS_NAME).resolve(),
                                          (root / "run" / ac.STATUS_NAME).resolve(),
                                          interpolation=split["arrays"])
            (root / ac.CASE_NAME).write_text(comm, encoding="utf-8")
            (root / "case_inputs.json").write_text(json.dumps({
                "joint": joint, "load_factors": [.5, 1.], "candidate_faces": 0,
                "native_joint_cells": 0, "split_nodes": 0, "added_nodes": 0,
                "joint_integration": "three_HEXA8_FPG4_patches_exact_P1_trace"}))
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "summary.json").write_text(json.dumps({"status": "prepared_aster_candidates"}))
            np.savez(candidate / "cohesive_state.npz", areas_m2=np.empty(0))
            status = ac.solve(root, time_limit_s=120, memory_limit_mb=2048)
            self.assertEqual(status["status"], "completed", status)
            with np.load(root / "run" / ac.STEPS_NAME) as fields:
                self.assertEqual(fields["damage_state"].shape, (2, 0))
            report = ac.postprocess(root, candidate, status)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["design_load"]["opened_faces"], 0)
            self.assertEqual(report["corroboration"]["verdict"], "not_available")
            self.assertTrue(np.isfinite(report["maximum_wall_displacement_m"]))
            with np.load(root / "run" / ac.STEPS_NAME.replace(".npz", "_displacement.npz")) as fields:
                self.assertTrue(np.isfinite(fields["final"]).all())
                self.assertGreater(np.linalg.norm(fields["final"]), 0.)

    def test_cohesive_section_cracks_under_reservoir_load(self):
        arrays = column_fixture()
        faces = crack_faces(arrays)
        split = ac.split_for_joints(arrays, faces)
        ac.integrate_joint_faces(split)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / ac.SPLIT_ARRAYS_NAME, penta6_joints=split["joints"], **split["arrays"])
            manifest = ac.split_manifest(fixture_manifest(arrays), split)
            summary = {"tensile_strength_pa": 3.0e5, "fracture_energy_j_m2": 100.0, "load_factor": 1.0}
            joint = ac.joint_parameters(arrays, faces, summary, 30e9, 0.2)
            # Full integration sees nonuniform traction missed at the triangle centroid.
            # Capture a lower intact step for the bonded-state comparison.
            factors = [0.1, *ac.load_factors(summary)]
            (root / "run").mkdir()
            comm = ac.render_cohesive_comm(fixture_base_comm(manifest, 4.0e5), joint, factors,
                                           (root / "run" / ac.STEPS_NAME).resolve(),
                                           (root / "run" / ac.STATUS_NAME).resolve(),
                                           tie_pairs=len(split["arrays"]["tie_node_pairs"]),
                                           interpolation=split["arrays"])
            (root / ac.CASE_NAME).write_text(comm, encoding="utf-8")
            (root / "case_inputs.json").write_text(json.dumps({
                "joint": joint, "load_factors": factors, "candidate_faces": len(faces),
                "split_nodes": split["split_nodes"], "added_nodes": split["added_nodes"],
                "joint_integration": "three_HEXA8_FPG4_patches_exact_P1_trace",
                "native_joint_cells": len(split["arrays"]["hexa8_joints"])}), encoding="utf-8")
            status = ac.solve(root, time_limit_s=480, memory_limit_mb=2048)
            log = (root / "run" / "solver.live.log").read_text(encoding="utf-8", errors="replace")
            self.assertEqual(status["status"], "completed", f"{status}\n{log[-4000:]}")
            fenicsx = fake_fenicsx(root / "f", len(faces))
            report = ac.postprocess(root, fenicsx, status)
        self.assertEqual(report["reached_load_factor"], 1.0)
        first, last = report["steps"][0], report["steps"][-1]
        self.assertEqual(first["initiated_faces"], 0, report)
        self.assertGreater(last["initiated_faces"], 0, report)
        self.assertGreater(last["peak_opening_m"], 0.0)
        self.assertGreater(report["linear_first_initiation_estimate"], 0.1)
        stress = report["stress_comparison"]
        self.assertEqual(report["peak_scope"], "wall")
        self.assertTrue(np.isfinite(report["maximum_wall_displacement_m"]))
        self.assertEqual(report["maximum_wall_displacement_m"], stress["cracked"]["maximum_displacement_m"])
        self.assertGreater(stress["intact"]["maximum_principal_tensile_stress_pa"], 0.0)
        self.assertNotEqual(stress["cracked"]["maximum_displacement_m"], stress["intact"]["maximum_displacement_m"])

    def test_isolated_tetrahedron_has_no_initial_joint_mechanism(self):
        arrays = column_fixture()
        wall = arrays["tetra4_concrete"][arrays["concrete_body_ids"] == 1]
        unique, counts = np.unique(np.sort(wall[:, ac._TET_FACES].reshape(-1, 3), axis=1),
                                   axis=0, return_counts=True)
        faces = np.array([face for face in np.sort(wall[3][ac._TET_FACES], axis=1)
                          if np.any(np.all(unique[counts == 2] == face, axis=1))])
        self.assertEqual(len(faces), 2)
        split = ac.split_for_joints(arrays, faces)
        ac.integrate_joint_faces(split)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / ac.SPLIT_ARRAYS_NAME, penta6_joints=split["joints"], **split["arrays"])
            manifest = ac.split_manifest(fixture_manifest(arrays), split)
            joint = ac.joint_parameters(arrays, faces,
                                       {"tensile_strength_pa": 3e5, "fracture_energy_j_m2": 100}, 30e9, .2)
            (root / "run").mkdir()
            # Isolate the intact iteration-zero mechanism, not full physical separation.
            comm = ac.render_cohesive_comm(fixture_base_comm(manifest, 4e3), joint, [.5, 1.],
                                          (root / "run" / ac.STEPS_NAME).resolve(),
                                          (root / "run" / ac.STATUS_NAME).resolve(),
                                          tie_pairs=len(split["arrays"]["tie_node_pairs"]),
                                          interpolation=split["arrays"])
            (root / ac.CASE_NAME).write_text(comm, encoding="utf-8")
            status = ac.solve(root, time_limit_s=120, memory_limit_mb=2048)
            log = (root / "run" / "solver.live.log").read_text(encoding="utf-8", errors="replace")
            self.assertEqual(status["status"], "completed", f"{status}\n{log[-4000:]}")
            self.assertEqual(status["returncode"], 0)
            with np.load(root / "run" / ac.STEPS_NAME) as fields:
                np.testing.assert_allclose(fields["load_factor"], [.5, 1.])
                self.assertEqual(fields["damage_state"].shape, (2, 6))
                self.assertTrue(np.all(fields["damage_state"] < 1))
            with np.load(root / "run" / ac.STEPS_NAME.replace(".npz", "_displacement.npz")) as fields:
                self.assertTrue(np.isfinite(fields["final"]).all())

    def test_consistent_support_trace_survives_native_joint_run(self):
        from Aster.support_trace import add_consistent_support, render_constraints
        arrays = column_fixture()
        arrays["compact_spring_groups"] = np.array(True)
        faces = crack_faces(arrays)
        seat = arrays["tri3_rock_bottom"]
        add_consistent_support(arrays, seat, np.repeat((np.eye(3) * 1e13)[None], len(seat), axis=0))
        split = ac.split_for_joints(arrays, faces)
        ac.relieve_neighbourhoods(split)
        ac.integrate_joint_faces(split)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / ac.SPLIT_ARRAYS_NAME, penta6_joints=split["joints"], **split["arrays"])
            manifest = ac.split_manifest(fixture_manifest(arrays), split)
            base = fixture_base_comm(manifest, 4e3)
            base = base.replace("RESU = MECA_STATIQUE(", render_constraints(split["arrays"]) + "RESU = MECA_STATIQUE(")
            joint = ac.joint_parameters(arrays, faces,
                                       {"tensile_strength_pa": 3e5, "fracture_energy_j_m2": 100}, 30e9, .2)
            (root / "run").mkdir()
            comm = ac.render_cohesive_comm(base, joint, [.5, 1.],
                                          (root / "run" / ac.STEPS_NAME).resolve(),
                                          (root / "run" / ac.STATUS_NAME).resolve(),
                                          tie_pairs=len(split["arrays"]["tie_node_pairs"]),
                                          interpolation=split["arrays"])
            self.assertIn("_F(CHARGE=SUPPORT_TRACE)", comm)
            (root / ac.CASE_NAME).write_text(comm, encoding="utf-8")
            status = ac.solve(root, time_limit_s=120, memory_limit_mb=2048)
            self.assertEqual(status["status"], "completed", status)
            with np.load(root / "run" / ac.STEPS_NAME.replace(".npz", "_displacement.npz")) as fields:
                final = fields["final"]
                self.assertTrue(np.isfinite(final).all())
                stored = split["arrays"]
                np.testing.assert_allclose(final[stored["support_interpolation_nodes"]],
                    np.einsum("ni,nij->nj", stored["support_interpolation_weights"],
                              final[stored["support_interpolation_parents"]]), atol=1e-12)

if __name__ == "__main__":
    unittest.main()
