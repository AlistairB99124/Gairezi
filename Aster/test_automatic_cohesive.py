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

    def test_friction_load_factors_climb_in_fine_checkpointed_steps(self):
        summary = {"load_factor": 1.0, "damaged_joint_friction": {"friction_coefficient": 1.0}}
        factors = ac.load_factors(summary)
        self.assertEqual(factors[:7], [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
        self.assertEqual(factors[-1], 1.0)
        self.assertEqual(factors, sorted(set(factors)))
        overload = ac.load_factors({**summary, "maximum_overload_factor": 2.0})
        self.assertEqual(overload[:7], factors[:7])
        self.assertEqual(overload[-1], 2.0)

    def test_joint_law_matches_the_fenicsx_softening(self):
        arrays = column_fixture()
        joint = ac.joint_parameters(arrays, crack_faces(arrays), {"tensile_strength_pa": 8e5,
                                                                 "fracture_energy_j_m2": 200.0}, 3e10, 0.2)
        self.assertAlmostEqual(joint["critical_opening_m"], 5e-4)
        self.assertAlmostEqual(joint["initial_stiffness_pa_per_m"],
                               ac.JOINT_PENALTY_FACTOR * 3e10 / joint["mean_facet_edge_m"], delta=1.0)

    def test_strength_only_changes_do_not_reduce_initial_joint_stiffness(self):
        arrays = column_fixture()
        faces = crack_faces(arrays)
        intact = ac.joint_parameters(arrays, faces,
                                     {"tensile_strength_pa": 2e6, "fracture_energy_j_m2": 100.}, 35e9, .2)
        weak = ac.joint_parameters(arrays, faces,
                                   {"tensile_strength_pa": 1.2e6, "fracture_energy_j_m2": 60.}, 35e9, .2)
        self.assertAlmostEqual(weak["initial_stiffness_pa_per_m"],
                               intact["initial_stiffness_pa_per_m"], delta=1.)
        self.assertAlmostEqual(weak["critical_opening_m"], 1e-4)
        self.assertAlmostEqual(weak["elastic_limit_opening_m"] /
                               intact["elastic_limit_opening_m"], .6)
        tougher = ac.joint_parameters(arrays, faces,
                                      {"tensile_strength_pa": 1.2e6, "fracture_energy_j_m2": 150.}, 35e9, .2)
        self.assertGreater(tougher["critical_opening_m"], weak["critical_opening_m"])

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

    def test_joint_parameters_refuse_degenerate_branches_instead_of_clipping(self):
        arrays = column_fixture()
        faces = crack_faces(arrays)
        with self.assertRaisesRegex(ac.CorroborationError, "refusing to clip"):
            ac.joint_parameters(arrays, faces,
                                {"tensile_strength_pa": 1e12, "fracture_energy_j_m2": 1e10}, 30e9, .2)
        joint = ac.joint_parameters(arrays, faces,
                                   {"tensile_strength_pa": 8e5, "fracture_energy_j_m2": 100.}, 30e9, .2)
        self.assertGreater(joint["pena_adherence"], 0.)
        self.assertLess(joint["pena_adherence"], 1.)
        self.assertAlmostEqual(joint["critical_opening_m"], .00025)
        for strength in (0., -1., float("nan"), float("inf")):
            with self.assertRaisesRegex(ac.CorroborationError, "finite and positive"):
                ac.joint_parameters(arrays, faces,
                                    {"tensile_strength_pa": strength, "fracture_energy_j_m2": 100.}, 30e9, .2)


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
            self.assertNotIn("interrupted", status)

    def test_solver_killed_mid_solve_is_reported_as_interrupted_not_as_a_model_failure(self):
        import subprocess
        marker = f"{ac.RUN_ASTER_EXIT_MARKER}502=1\n"
        progress = ("ASTER_COHESIVE_STEP done load_factor=0.175 damaged_joints=0\n"
                    "ASTER_COHESIVE_STEP start load_factor=0.2\n")
        for stdout, returncode, interrupted in ((progress, 1, True), (progress + marker, 1, False),
                                                (progress, 137, False)):
            with self.subTest(returncode=returncode, marker=marker in stdout), \
                    tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "run").mkdir()
                (root / ac.CASE_NAME).write_text("SOLVEUR=_F(METHODE='MULT_FRONT')")

                def execute(distribution, installation, directory, program, timeout, log):
                    if "convert_three_body_to_med.py" in " ".join(program):
                        (root / "mesh.med").write_text("mesh")
                        return subprocess.CompletedProcess(program, 0, "converted", "")
                    (root / "run" / ac.STATUS_NAME).write_text('{"status":"running"}')
                    return subprocess.CompletedProcess(program, returncode, stdout, "")

                with patch("Aster.whole_model_crack.find_linux_installation", return_value="/opt/aster"), \
                        patch("Aster.whole_model_crack.wsl_path", side_effect=str), \
                        patch("Aster.run_case.run_wsl_solver", side_effect=execute), \
                        patch("Aster.automatic_cohesive.os.name", "nt"):
                    status = ac.solve(root, 60)
                self.assertEqual(status["status"], "failed")
                self.assertEqual(status.get("interrupted", False), interrupted)
                if interrupted:
                    self.assertIn("terminated from outside while solving load factor 0.2", status["message"])
                    self.assertIn("not a model result", status["message"])
                else:
                    self.assertIn("stopped unexpectedly", status["message"])

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
        self.assertEqual(report["opening_assessment"]["initiated_faces"], 2)
        self.assertEqual(report["opening_assessment"]["separated_faces"], 0)
        self.assertIn("not full separation", report["opening_assessment"]["note"])
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

    def test_convergence_stop_is_reported_without_claiming_structural_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = self.write_case(root, [[0, 0, 0, 0], [1, 1, 0, 0]], [[0] * 4, [2e-5, 0, 0, 0]],
                                     [[4e5] * 4, [8e5] * 4], [0.5, 0.9], status="stopped")
            status["stopped_at_load_factor"] = 1.0
            report = ac.postprocess(root, fake_fenicsx(root / "f", 4), status)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["outcome"], "capacity_limit")
        capacity = report["capacity_assessment"]
        self.assertEqual(capacity["reached_load_factor"], 0.9)
        self.assertEqual(capacity["failed_load_factor"], 1.0)
        # Newton non-convergence without load control is not established failure (U2.05.07 4.3).
        self.assertIs(capacity["established"], False)
        self.assertIn("did not converge between 0.9 and 1", capacity["message"])
        self.assertIn("not, by itself, evidence that the dam fails", capacity["message"])
        self.assertIn("2 joints had initiated, 0 had fully separated", capacity["message"])
        for overclaim in ("no static equilibrium", "cannot carry that load", "brings the dam to failure"):
            self.assertNotIn(overclaim, capacity["message"])
        self.assertEqual(capacity["last_converged_state"]["load_factor"], 0.9)

    def test_a_genuine_crash_is_not_reported_as_a_capacity_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = self.write_case(root, [[0, 0, 0, 0]], [[0] * 4], [[1e5] * 4], [0.5], status="failed")
            status["stopped_at_load_factor"] = 1.0
            report = ac.postprocess(root, fake_fenicsx(root / "f", 4), status)
        self.assertEqual(report["status"], "partial")
        self.assertNotIn("outcome", report)
        self.assertNotIn("capacity_assessment", report)

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
class ReportDiagnosticsTests(unittest.TestCase):
    """The checks that showed the first lift-joint run had not cracked where it claimed."""

    def test_joint_class_summary_reports_a_bypassed_weak_plane(self):
        areas = np.array([1., 2., 3., 4.])
        traction = np.array([[1e4, 2e4, 1e5, 4e5], [1.7e4, 1.5e4, 5e5, 5e5]])
        jump = np.array([[1e-6, 2e-6, 0., 1e-5], [8e-6, 3e-6, 1e-4, 2e-4]])
        initiated = np.array([False, False, True, True])
        lift = ac.joint_class_summary(np.array([True, True, False, False]), areas, initiated, initiated,
                                      np.zeros(4, bool), traction, jump, 2e5, "JOINT_MECA_RUPT")
        self.assertEqual((lift["faces"], lift["area_m2"], lift["initiated_faces"]), (2, 3., 0))
        self.assertAlmostEqual(lift["utilisation"], 0.1)          # 20 kPa of a 200 kPa strength
        self.assertEqual(lift["peak_opening_m"], 8e-6)
        damaged = ac.joint_class_summary(np.array([False, False, True, True]), areas, initiated, initiated,
                                         np.zeros(4, bool), traction, jump, 5e5, "CZM_LIN_REG")
        self.assertEqual((damaged["initiated_faces"], damaged["utilisation"]), (2, 1.))
        friction = ac.joint_class_summary(np.array([True, False, False, False]), areas, initiated, initiated,
                                          np.zeros(4, bool), traction, jump, None, "JOINT_MECA_FROT")
        self.assertIsNone(friction["utilisation"])

    def test_lift_plane_sliding_compares_shear_demand_with_mohr_coulomb_capacity(self):
        areas = np.array([2., 2.])
        normal = np.array([[-1e5, -1e5], [-1e5, 5e4]])           # second step: one face in tension
        shear = np.array([[3e4, 5e4], [8e4, 0.]])
        sliding, slip = np.array([[0., 0.], [1., 0.]]), np.array([[0., 0.], [2e-4, 0.]])
        jump = np.array([[0., 0.], [0., 1e-5]])
        law = {"law": "JOINT_MECA_FROT", "friction_coefficient": 0.7, "adhesion_pa": 1e4}
        report = ac.lift_plane_sliding([0.5, 1.0], areas, normal, shear, sliding, slip, jump, law)
        first, final = report["steps"]
        self.assertAlmostEqual(first["normal_compression_n"], 4e5)
        self.assertAlmostEqual(first["shear_demand_n"], 1.6e5)
        self.assertAlmostEqual(first["shear_capacity_n"], (1e4 + 0.7 * 1e5) * 4.)
        self.assertAlmostEqual(first["mobilisation"], 1.6e5 / 3.2e5)
        self.assertAlmostEqual(first["required_friction_coefficient"], 0.4)
        # A face in tension has capacity c - mu sigma_n = 1e4 - 0.7 * 5e4 < 0, i.e. none.
        self.assertAlmostEqual(final["shear_capacity_n"], (1e4 + 0.7 * 1e5) * 2.)
        self.assertAlmostEqual(final["mobilisation"], 1.0)
        self.assertEqual((final["tension_area_fraction"], final["open_area_fraction"],
                          final["sliding_area_fraction"]), (0.5, 0.5, 0.5))
        self.assertEqual(final["peak_cumulative_slip_m"], 2e-4)
        self.assertIs(report["final"], final)
        rupture = ac.lift_plane_sliding([1.0], areas, normal[:1], shear[:1], None, None, jump[:1],
                                        {"law": "JOINT_MECA_RUPT"})
        self.assertNotIn("mobilisation", rupture["final"])
        self.assertAlmostEqual(rupture["final"]["required_friction_coefficient"], 0.4)
        self.assertIn("no slip limit", rupture["definition"])

    def test_unjointed_overstress_counts_only_cells_that_cannot_crack(self):
        # Three tetrahedra on a line: the first two share a jointed face (split nodes), the
        # third has no joint. Only the third counts, however high the jointed cells' stress.
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1., 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
                           [5., 0, 0], [6, 0, 0], [5, 1, 0], [5, 0, 1]])
        wall = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        joints = np.array([[1, 2, 3, 4, 5, 6]])
        stress = np.stack([np.diag([3e6, 0., 0.]), np.diag([3e6, 0., 0.]), np.diag([2.5e6, 0., 0.])])
        report = ac.unjointed_overstress(points, wall, stress, joints, 2e6)
        self.assertEqual((report["jointed_wall_cells"], report["unjointed_wall_cells"], report["cells"]), (2, 1, 1))
        self.assertAlmostEqual(report["volume_m3"], 1 / 6)
        self.assertEqual(report["peak_principal_stress_pa"], 2.5e6)
        np.testing.assert_allclose(report["peak_centroid_xyz_m"], [5.25, .25, .25])
        self.assertIn("cannot crack", report["note"])
        calm = ac.unjointed_overstress(points, wall, stress * 0.5, joints, 2e6)
        self.assertEqual(calm["cells"], 0)
        self.assertEqual(calm["peak_principal_stress_pa"], 1.25e6)
        self.assertIn("No wall cell", calm["note"])

    def test_displacement_attribution_separates_softening_from_cracking(self):
        wall_nodes = np.array([0, 1])
        intact = np.array([[6.62e-3, 0, 0], [1e-3, 0, 0], [9., 9, 9]])      # node 2 is not wall
        first = np.array([[15.88e-5, 0, 0], [2e-5, 0, 0], [0., 0, 0]])       # load factor 0.01
        final = np.array([[15.90e-3, 0, 0], [2e-3, 0, 0], [0., 0, 0]])       # load factor 1
        sources = ["wall/foundation seat springs scaled by 0.1"]
        report = ac.displacement_attribution(first, final, intact, wall_nodes, [0.01, 0.5, 1.0], False, True, sources)
        self.assertAlmostEqual(report["softened_uncracked_peak_m"], 15.88e-3)
        self.assertAlmostEqual(report["elastic_softening_change_m"], 15.88e-3 - 6.62e-3)
        self.assertAlmostEqual(report["cracking_change_m"], 0.02e-3)
        self.assertEqual(report["softening_sources"], sources)
        # Friction runs load self-weight first, and a first step that already cracked is not linear.
        for first_initiated, proportional, text in ((False, False, "self-weight is applied before the water"),
                                                    (True, True, "already initiated at the first step")):
            report = ac.displacement_attribution(first, final, intact, wall_nodes, [0.01, 1.0],
                                                 first_initiated, proportional, [])
            self.assertIsNone(report["softened_uncracked_peak_m"])
            self.assertAlmostEqual(report["total_change_m"], 15.90e-3 - 6.62e-3)
            self.assertIn(text, report["basis"])
            self.assertIn("no elastic softening is configured", report["basis"])

    def test_assessment_notes_say_what_each_part_of_the_model_did(self):
        report = {
            "joint_classes": {
                "lift": {"law": "JOINT_MECA_RUPT", "faces": 372, "area_m2": 160.4, "initiated_faces": 0,
                         "opened_faces": 0, "separated_faces": 0, "peak_normal_traction_pa": 1.7e4,
                         "tensile_strength_pa": 2e5, "utilisation": 0.085, "peak_opening_m": 7.75e-6},
                "damaged": {"law": "CZM_LIN_REG", "faces": 4036, "area_m2": 2000., "initiated_faces": 2318,
                            "opened_faces": 1342, "separated_faces": 2, "peak_normal_traction_pa": 5e5,
                            "tensile_strength_pa": 5e5, "utilisation": 1.0, "peak_opening_m": 1.46e-4}},
            "lift_plane_sliding": {"law": "JOINT_MECA_FROT", "friction_coefficient": 0.7, "final": {
                "shear_demand_n": 6.1e7, "shear_capacity_n": 7.4e7, "mobilisation": 0.82,
                "normal_compression_n": 9e7, "sliding_area_fraction": 0.4, "open_area_fraction": 0.1,
                "peak_cumulative_slip_m": 3e-4, "required_friction_coefficient": 0.68}},
            "displacement_attribution": {
                "final_peak_m": 15.90e-3, "intact_peak_m": 6.62e-3, "softened_uncracked_peak_m": 15.88e-3,
                "elastic_softening_change_m": 9.26e-3, "cracking_change_m": 0.02e-3,
                "softening_sources": ["wall/foundation seat springs scaled by 0.1"]},
            "unjointed_overstress": {"cells": 12, "volume_m3": 1.5, "tensile_strength_pa": 9e5,
                                     "peak_principal_stress_pa": 3.44e6},
        }
        notes = ac.assessment_notes(report)
        self.assertEqual(len(notes), 5)
        self.assertIn("8.5% of the 200 kPa strength", notes[0])
        self.assertIn("never approached their strength", notes[0])
        self.assertIn("2318 initiated, 1342 opened, 2 fully separated", notes[1])
        self.assertNotIn("never approached", notes[1])
        self.assertIn("mobilisation 0.82", notes[2])
        self.assertIn("40% of the plane is sliding and 10% is open", notes[2])
        self.assertIn("Cracking changed the displacement field by at most 0.02 mm", notes[3])
        self.assertIn("the softened inputs alone, before any cracking, changed it by 9.26 mm", notes[3])
        self.assertIn("seat springs scaled by 0.1", notes[3])
        self.assertIn("12 wall cells outside the jointed band", notes[4])
        self.assertIn("up to 3.44 MPa", notes[4])
        self.assertEqual(ac.assessment_notes({}), [])


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

    def test_lift_joint_plane_cracks_and_postprocesses(self):
        """Weak lift plane (JOINT_MECA_RUPT) at z = 2 with the intact CZM_LIN_REG elsewhere: the
        material passes the catalogue, the weight-scaled residual criterion closes the solve
        and the post-processing accepts an assignment without degraded element properties."""
        arrays = column_fixture()
        # Upstream half of the z = 2 plane as weak lift joints; the tied x > 0.5 ligament keeps
        # the upper block from becoming a mechanism once the weak plane ruptures.
        faces = crack_faces(arrays)
        split = ac.split_for_joints(arrays, faces)
        ac.integrate_joint_faces(split)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split["arrays"]["damaged_cells"] = np.zeros(
                len(split["arrays"]["tetra4_concrete"]) + len(split["arrays"]["tetra4_bedrock"]), dtype=bool)
            split["arrays"]["damaged_joint_faces"] = np.zeros(len(faces), dtype=bool)
            split["arrays"]["lift_joint_faces"] = np.ones(len(faces), dtype=bool)
            np.savez(root / ac.SPLIT_ARRAYS_NAME, penta6_joints=split["joints"], **split["arrays"])
            manifest = ac.split_manifest(fixture_manifest(arrays), split)
            summary = {"tensile_strength_pa": 3.0e5, "fracture_energy_j_m2": 100.0, "load_factor": 1.0}
            joint = ac.joint_parameters(arrays, faces, summary, 30e9, 0.2)
            joint["intact_joint"] = dict(joint)
            lift = {"tensile_strength_pa": 5.0e3, "cohesive_fracture_energy_j_m2": 0.003, "alpha": 1.0,
                    "pena_contact": 1.0, "law": "JOINT_MECA_RUPT", "max_normal_deviation_deg": 15.0,
                    "include_base_contact": False, "lift_joint_facets": len(faces)}
            joint["lift_joint"] = ac.lift_joint_parameters(lift, joint["initial_stiffness_pa_per_m"])
            joint["lift_joint"]["residual_tolerance_n"] = 1.0
            joint["material_assignment"] = {"joint_properties": lift, "selected_source_rows": [0]}
            joint["lift_joint_count"] = len(faces)
            joint["damaged_joint_count"] = 0
            joint["candidate_face_count"] = len(faces)
            factors = [0.1, *ac.load_factors(summary)]
            (root / "run").mkdir()
            comm = ac.render_cohesive_comm(fixture_base_comm(manifest, 4.0e5), joint, factors,
                                           (root / "run" / ac.STEPS_NAME).resolve(),
                                           (root / "run" / ac.STATUS_NAME).resolve(),
                                           tie_pairs=len(split["arrays"]["tie_node_pairs"]),
                                           interpolation=split["arrays"])
            self.assertIn("JOINT_MECA_RUPT=_F(", comm)
            self.assertIn("PENA_RUPTURE=10,", comm)
            self.assertIn("RESI_GLOB_MAXI=1", comm)
            self.assertIn("RECH_LINEAIRE", comm)
            (root / ac.CASE_NAME).write_text(comm, encoding="utf-8")
            # A zero intact baseline over the original nodes is enough for the comparison path
            # that crashed on the live job (it assumed degraded element properties).
            baseline = root / "aster_full_displacement.npz"
            np.savez(baseline, source_node_ids=arrays["source_node_ids"], coordinates_m=arrays["nodes"],
                     displacement_m=np.zeros_like(arrays["nodes"]))
            (root / "case_inputs.json").write_text(json.dumps({
                "joint": joint, "load_factors": factors, "candidate_faces": len(faces),
                "split_nodes": split["split_nodes"], "added_nodes": split["added_nodes"],
                "joint_integration": "three_HEXA8_FPG4_patches_exact_P1_trace",
                "material_assignment": joint["material_assignment"],
                "baseline_displacement_path": str(baseline),
                "baseline_displacement_sha256": ac.file_hash(baseline),
                "native_joint_cells": len(split["arrays"]["hexa8_joints"])}), encoding="utf-8")
            status = ac.solve(root, time_limit_s=480, memory_limit_mb=2048)
            log = (root / "run" / "solver.live.log").read_text(encoding="utf-8", errors="replace")
            self.assertEqual(status["status"], "completed", f"{status}\n{log[-4000:]}")
            fenicsx = fake_fenicsx(root / "f", len(faces))
            report = ac.postprocess(root, fenicsx, status)
        self.assertEqual(report["reached_load_factor"], 1.0)
        self.assertEqual(report["lift_joints"]["law"], "JOINT_MECA_RUPT")
        self.assertEqual(report["lift_joints"]["lift_joint_faces"], len(faces))
        self.assertGreater(report["steps"][-1]["initiated_faces"], 0, report)
        self.assertNotIn("properties", report["material_assignment"])
        self.assertIn("lift-joint cracking only", report["stress_comparison"]["intact_definition"])
        # The report says which joint class did the work and what the plane's shear demand is.
        lift_class = report["joint_classes"]["lift"]
        self.assertEqual((lift_class["law"], lift_class["faces"]), ("JOINT_MECA_RUPT", len(faces)))
        self.assertGreater(lift_class["initiated_faces"], 0)
        self.assertGreater(lift_class["utilisation"], 0.5)
        self.assertNotIn("damaged", report["joint_classes"])
        sliding = report["lift_plane_sliding"]
        self.assertEqual(sliding["law"], "JOINT_MECA_RUPT")
        self.assertEqual(len(sliding["steps"]), len(report["steps"]))
        # The plane carries shear until it ruptures; a broken JOINT_MECA_RUPT face carries none.
        self.assertGreater(max(step["shear_demand_n"] for step in sliding["steps"]), 0.)
        self.assertNotIn("mobilisation", sliding["final"])
        # The weak plane has already initiated at the first load step, so no linear step exists
        # to scale: only the total change from the intact baseline can be reported, and since
        # a stiff joint (K_N >= E/h) is the only softening source that total is the joints'.
        attribution = report["displacement_attribution"]
        self.assertEqual(attribution["softening_sources"],
                         [source for source in attribution["softening_sources"] if "lift-joint" in source])
        self.assertIsNone(attribution["softened_uncracked_peak_m"])
        self.assertGreater(attribution["total_change_m"], 0.)
        self.assertIn("already initiated at the first step", attribution["basis"])
        overstress = report["unjointed_overstress"]
        self.assertEqual(overstress["jointed_wall_cells"] + overstress["unjointed_wall_cells"], report["mesh_cells"])
        self.assertGreater(overstress["jointed_wall_cells"], 0)

    def test_frictional_lift_plane_carries_uplift_as_effective_stress(self):
        """JOINT_MECA_FROT lift plane with pore pressure inside it, through a real Code_Aster run.

        Only self-weight and the plane's pore pressure act (the reservoir load on the face is
        negligible), so the plane stays closed and its effective compression must fall by
        exactly the uplift resultant as the load factor rises."""
        arrays = column_fixture()
        # The whole z = 2 section: no crack front inside the wall, so every pore force acts
        # through the joint (front nodes are tied and would absorb their share).
        faces = plane_faces(arrays, 2, 2.0, lambda points: True)
        straight = arrays["nodes"].copy()
        # Curve the straight column onto an arch centred on the origin (the uplift profile is
        # radial): the upstream face x = 0 becomes the outer radius, as on the dam.
        radius, angle = 81. - straight[:, 0], -straight[:, 1] / 80.
        arrays["nodes"][:, 0], arrays["nodes"][:, 1] = radius * np.cos(angle), radius * np.sin(angle)
        unit_weight = 2e4
        split = ac.split_for_joints(arrays, faces)
        ac.integrate_joint_faces(split)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split["arrays"]["damaged_cells"] = np.zeros(
                len(split["arrays"]["tetra4_concrete"]) + len(split["arrays"]["tetra4_bedrock"]), dtype=bool)
            split["arrays"]["damaged_joint_faces"] = np.zeros(len(faces), dtype=bool)
            split["arrays"]["lift_joint_faces"] = np.ones(len(faces), dtype=bool)
            np.savez(root / ac.SPLIT_ARRAYS_NAME, penta6_joints=split["joints"], **split["arrays"])
            manifest = ac.split_manifest(fixture_manifest(arrays), split)
            summary = {"tensile_strength_pa": 3.0e5, "fracture_energy_j_m2": 100.0, "load_factor": 1.0}
            joint = ac.joint_parameters(arrays, faces, summary, 30e9, 0.2)
            joint["intact_joint"] = dict(joint)
            lift = {"tensile_strength_pa": 5.0e3, "cohesive_fracture_energy_j_m2": 1.0, "alpha": 1.0,
                    "pena_contact": 1.0, "law": "JOINT_MECA_FROT", "friction_coefficient": 0.7,
                    "uplift": "linear_upstream_to_downstream", "max_normal_deviation_deg": 15.0,
                    "include_base_contact": False, "lift_joint_facets": len(faces)}
            joint["lift_joint"] = ac.lift_joint_parameters(lift, joint["initial_stiffness_pa_per_m"])
            joint["lift_joint"]["residual_tolerance_n"] = 1.0
            joint["material_assignment"] = {"joint_properties": lift, "selected_source_rows": [0]}
            joint["lift_joint_count"] = len(faces)
            joint["damaged_joint_count"] = 0
            joint["candidate_face_count"] = len(faces)
            from Aster.hydraulic_loads import CRACK_WATER_SETTINGS
            joint["hydraulic_loads"] = {
                "settings": CRACK_WATER_SETTINGS, "upstream_z_m": 3., "gradient_pa_per_m": unit_weight,
                "uplift": {"mode": "none", "resultant_n": [0., 0., 0.]},
                "lift_plane_uplift": {"mode": "linear_upstream_to_downstream", "downstream_z_m": 1.}}
            factors = [.05, .1, .2, .3, .4, .5]
            (root / "run").mkdir()
            comm = ac.render_cohesive_comm(fixture_base_comm(manifest, 1e-3), joint, factors,
                                           (root / "run" / ac.STEPS_NAME).resolve(),
                                           (root / "run" / ac.STATUS_NAME).resolve(),
                                           tie_pairs=len(split["arrays"]["tie_node_pairs"]),
                                           interpolation=split["arrays"])
            for text in ("JOINT_MECA_FROT=_F(K_N=", "MU=0.7, ADHESION=3500)",
                         "_F(RELATION='JOINT_MECA_FROT', GROUP_MA='LIFT_JOINTS')", "FRICTION_CELLS = np.asarray(",
                         "LIFT_CELLS = None", "GRAVITY_RAMP = DEFI_FONCTION",
                         "pore_faces=np.asarray(hydraulic_arrays['lift_joint_faces'], dtype=bool), downstream_z=1.0",
                         "status['lift_plane_uplift'] = crack_water.pore_report"):
                self.assertIn(text, comm)
            self.assertNotIn("JOINT_MECA_RUPT", comm.split("COMPORT = (")[1].split(")")[0])
            (root / ac.CASE_NAME).write_text(comm, encoding="utf-8")
            baseline = root / "aster_full_displacement.npz"
            np.savez(baseline, source_node_ids=arrays["source_node_ids"], coordinates_m=arrays["nodes"],
                     displacement_m=np.zeros_like(arrays["nodes"]))
            (root / "case_inputs.json").write_text(json.dumps({
                "joint": joint, "load_factors": factors, "candidate_faces": len(faces),
                "split_nodes": split["split_nodes"], "added_nodes": split["added_nodes"],
                "joint_integration": "three_HEXA8_FPG4_patches_exact_P1_trace",
                "material_assignment": joint["material_assignment"],
                "baseline_displacement_path": str(baseline),
                "baseline_displacement_sha256": ac.file_hash(baseline),
                "native_joint_cells": len(split["arrays"]["hexa8_joints"])}), encoding="utf-8")
            status = ac.solve(root, time_limit_s=480, memory_limit_mb=2048)
            log = (root / "run" / "solver.live.log").read_text(encoding="utf-8", errors="replace")
            self.assertEqual(status["status"], "completed", f"{status}\n{log[-4000:]}")
            # The stand-in candidate state carries placeholder areas; resultants need the real ones.
            points = arrays["nodes"][faces]
            areas = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1) / 2
            candidates = fake_fenicsx(root / "f", len(faces))
            with np.load(candidates / "cohesive_state.npz") as state:
                stored = {key: state[key] for key in state.files}
            np.savez(candidates / "cohesive_state.npz", **{**stored, "areas_m2": areas})
            report = ac.postprocess(root, candidates, status)
        self.assertEqual(report["reached_load_factor"], factors[-1])
        # Independent expectation: the wall is 1 m wide, so the radial fraction is the straight
        # x and the pore pressure at the plane (reservoir 1 m above it, tailwater below it) is
        # unit_weight * (1 - x) at full load, integrated over the curved facets.
        pore_force = float((areas * unit_weight * (1. - straight[faces][:, :, 0].mean(axis=1))).sum())
        uplift = report["lift_plane_uplift"]
        self.assertEqual((uplift["mode"], uplift["faces"]), ("linear_upstream_to_downstream", len(faces)))
        self.assertAlmostEqual(uplift["resultant_on_one_lip_n"], pore_force, delta=1e-3 * pore_force)
        self.assertGreater(uplift["maximum_pressure_pa"], uplift["minimum_pressure_pa"])
        # The plane stays closed: every lift face carries pore pressure and none is wet.
        history = report["hydraulic_steps"]
        for step in history:
            self.assertEqual((step["wet_faces"], step["pore_pressure_faces"]), (0, len(faces)))
            np.testing.assert_allclose(step["paired_resultant_n"], 0., atol=1e-6)
        self.assertEqual(report["lift_joints"]["law"], "JOINT_MECA_FROT")
        self.assertIn("perfectly plastic", report["lift_joints"]["limitation"])
        sliding = report["lift_plane_sliding"]
        self.assertEqual((sliding["law"], sliding["friction_coefficient"], sliding["adhesion_pa"]),
                         ("JOINT_MECA_FROT", .7, 3500.))
        steps = {round(step["load_factor"], 6): step for step in sliding["steps"]}
        # Effective stress: self-weight is complete from load factor 0.05, the pore pressure
        # scales with the load factor, so the compression falls by the uplift resultant.
        drop = steps[.1]["normal_compression_n"] - steps[.5]["normal_compression_n"]
        self.assertAlmostEqual(drop, .4 * pore_force, delta=.02 * .4 * pore_force)
        weight = 2400. * 9.81 * 1. * areas.sum()           # 1 m of wall above the plane
        for factor in (.1, .5):
            # Vertical equilibrium of the block above the plane: joint compression = weight - uplift.
            self.assertAlmostEqual(steps[factor]["normal_compression_n"], weight - factor * pore_force,
                                   delta=.02 * weight)
        for step in sliding["steps"]:
            self.assertEqual((step["tension_area_fraction"], step["open_area_fraction"]), (0., 0.))
            # All faces compressed, so capacity = c A + mu N; nothing shears the plane here.
            self.assertAlmostEqual(step["shear_capacity_n"],
                                   3500. * areas.sum() + .7 * step["normal_compression_n"], delta=1e-6 * weight)
            self.assertLess(step["mobilisation"], .05)
            self.assertEqual(step["sliding_area_fraction"], 0.)
        # Self-weight is applied before the water, so the first step cannot be scaled.
        attribution = report["displacement_attribution"]
        self.assertIsNone(attribution["softened_uncracked_peak_m"])
        self.assertEqual(attribution["softening_sources"], [])
        self.assertIn("self-weight is applied before the water", attribution["basis"])
        lift_class = report["joint_classes"]["lift"]
        self.assertEqual((lift_class["law"], lift_class["initiated_faces"]), ("JOINT_MECA_FROT", 0))

    def test_lift_and_damaged_joints_combine_in_one_native_run(self):
        """Three facet classes in one Code_Aster case: the upstream quarter of the z = 2 half-plane
        as weak lift joints, the rest of it as the degraded elements' CZM (DAMAGED_JOINTS with
        DAMAGED_MAT on the adjacent cells), and the intact law everywhere else."""
        arrays = column_fixture()
        faces = crack_faces(arrays)
        centroids = arrays["nodes"][faces].mean(axis=1)
        lift_mask = centroids[:, 0] < np.median(centroids[:, 0])
        damaged_mask = ~lift_mask
        self.assertTrue(lift_mask.any() and damaged_mask.any())
        split = ac.split_for_joints(arrays, faces)
        ac.integrate_joint_faces(split)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concrete = split["arrays"]["tetra4_concrete"]
            damaged_faces = {tuple(sorted(face)) for face in faces[damaged_mask]}
            damaged_cells = np.array([any(tuple(sorted(cell[list(face)])) in damaged_faces
                                          for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
                                      for cell in concrete], dtype=bool)
            self.assertTrue(damaged_cells.any())
            split["arrays"]["damaged_cells"] = np.r_[damaged_cells,
                                                    np.zeros(len(split["arrays"]["tetra4_bedrock"]), dtype=bool)]
            split["arrays"]["damaged_joint_faces"] = damaged_mask
            split["arrays"]["lift_joint_faces"] = lift_mask
            np.savez(root / ac.SPLIT_ARRAYS_NAME, penta6_joints=split["joints"], **split["arrays"])
            manifest = ac.split_manifest(fixture_manifest(arrays), split)
            properties = {"density": 2400.0, "youngs_modulus": 20e9, "poisson_ratio": 0.2,
                          "tensile_strength_pa": 1.5e5, "compressive_strength_pa": 30e6,
                          "cohesive_fracture_energy_j_m2": 50.0}
            summary = {"tensile_strength_pa": properties["tensile_strength_pa"],
                       "fracture_energy_j_m2": properties["cohesive_fracture_energy_j_m2"], "load_factor": 1.0}
            joint = ac.joint_parameters(arrays, faces, summary, 30e9, 0.2)
            joint["intact_joint"] = ac.joint_parameters(
                arrays, faces, {"tensile_strength_pa": 3.0e5, "fracture_energy_j_m2": 100.0}, 30e9, 0.2)
            lift = {"tensile_strength_pa": 5.0e3, "cohesive_fracture_energy_j_m2": 0.003, "alpha": 1.0,
                    "pena_contact": 1.0, "law": "JOINT_MECA_RUPT", "max_normal_deviation_deg": 15.0,
                    "include_base_contact": False, "lift_joint_facets": int(lift_mask.sum())}
            joint["lift_joint"] = ac.lift_joint_parameters(lift, joint["initial_stiffness_pa_per_m"])
            joint["lift_joint"]["residual_tolerance_n"] = 1.0
            joint["material_assignment"] = {"properties": properties, "joint_properties": lift,
                                            "selected_source_rows": np.flatnonzero(damaged_cells).tolist()}
            joint["lift_joint_count"] = int(lift_mask.sum())
            joint["damaged_joint_count"] = int(damaged_mask.sum())
            joint["candidate_face_count"] = len(faces)
            factors = [0.1, *ac.load_factors(summary)]
            (root / "run").mkdir()
            comm = ac.render_cohesive_comm(fixture_base_comm(manifest, 4.0e5), joint, factors,
                                           (root / "run" / ac.STEPS_NAME).resolve(),
                                           (root / "run" / ac.STATUS_NAME).resolve(),
                                           tie_pairs=len(split["arrays"]["tie_node_pairs"]),
                                           interpolation=split["arrays"])
            for text in ("LIFT_MAT = DEFI_MATERIAU", "DAMAGED_MAT = DEFI_MATERIAU", "INTACT_JOINT_MAT = DEFI_MATERIAU",
                         "_F(GROUP_MA='LIFT_JOINTS', MATER=LIFT_MAT)", "_F(GROUP_MA='DAMAGED_JOINTS', MATER=JOINT_MAT)",
                         "_F(GROUP_MA='DAMAGED_ELEMENTS', MATER=DAMAGED_MAT)",
                         "_F(RELATION='JOINT_MECA_RUPT', GROUP_MA='LIFT_JOINTS')",
                         "_F(RELATION='CZM_LIN_REG', GROUP_MA='DAMAGED_JOINTS')"):
                self.assertIn(text, comm)
            # Every candidate facet is lift or damaged here, so no (empty) intact group is written.
            self.assertNotIn("GROUP_MA='INTACT_JOINTS'", comm)
            (root / ac.CASE_NAME).write_text(comm, encoding="utf-8")
            baseline = root / "aster_full_displacement.npz"
            np.savez(baseline, source_node_ids=arrays["source_node_ids"], coordinates_m=arrays["nodes"],
                     displacement_m=np.zeros_like(arrays["nodes"]))
            (root / "case_inputs.json").write_text(json.dumps({
                "joint": joint, "load_factors": factors, "candidate_faces": len(faces),
                "split_nodes": split["split_nodes"], "added_nodes": split["added_nodes"],
                "joint_integration": "three_HEXA8_FPG4_patches_exact_P1_trace",
                "material_assignment": joint["material_assignment"],
                "baseline_displacement_path": str(baseline),
                "baseline_displacement_sha256": ac.file_hash(baseline),
                "native_joint_cells": len(split["arrays"]["hexa8_joints"])}), encoding="utf-8")
            status = ac.solve(root, time_limit_s=480, memory_limit_mb=2048)
            log = (root / "run" / "solver.live.log").read_text(encoding="utf-8", errors="replace")
            self.assertEqual(status["status"], "completed", f"{status}\n{log[-4000:]}")
            fenicsx = fake_fenicsx(root / "f", len(faces))
            report = ac.postprocess(root, fenicsx, status)
        self.assertEqual(report["reached_load_factor"], 1.0)
        self.assertEqual(report["lift_joints"]["lift_joint_faces"], int(lift_mask.sum()))
        self.assertEqual(report["material_assignment"]["properties"], properties)
        self.assertEqual(report["material_assignment"]["joint_properties"]["law"], "JOINT_MECA_RUPT")
        self.assertIn("selected material changes and cracking", report["stress_comparison"]["intact_definition"])
        self.assertTrue(np.isfinite(report["maximum_wall_displacement_m"]))

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

class SupportDegradationTest(unittest.TestCase):
    def fixture(self):
        theta = np.linspace(0., np.pi / 2, 8)
        points = np.column_stack((10 * np.cos(theta), 10 * np.sin(theta), np.zeros(8)))
        arrays = {"nodes": points, "support_nodes": np.arange(8),
                  "support_matrices_n_per_m": np.repeat(np.eye(3)[None], 8, axis=0),
                  "interface_stiffness_n_per_m": np.array([[2., 4., 6.]])}
        manifest = {"wall_foundation_springs": [{"stiffness_n_per_m": [2., 4., 6.]}],
                    "support_springs": [{"matrix_n_per_m": np.eye(3).tolist()} for _ in range(8)]}
        return manifest, arrays

    def test_interface_and_abutment_sector_scaling(self):
        manifest, arrays = self.fixture()
        applied = ac.apply_support_degradation(manifest, arrays, {
            "interface_scale": .5, "abutment": {"scale": .1, "fraction": .25, "end": "positive"}})
        self.assertEqual(manifest["wall_foundation_springs"][0]["stiffness_n_per_m"], [1., 2., 3.])
        np.testing.assert_allclose(arrays["interface_stiffness_n_per_m"], [[1., 2., 3.]])
        self.assertEqual(applied["interface_springs_scaled"], 1)
        scaled = [i for i, s in enumerate(manifest["support_springs"]) if s["matrix_n_per_m"][0][0] != 1.]
        self.assertEqual(scaled, [6, 7])
        self.assertEqual(applied["abutment"]["support_springs_scaled"], 2)
        for index in range(8):
            expected = .1 if index in scaled else 1.
            self.assertAlmostEqual(manifest["support_springs"][index]["matrix_n_per_m"][0][0], expected)
            # The MED compact grouping partitions by the array tensors; they must
            # track the manifest exactly or the comm references missing groups.
            np.testing.assert_allclose(arrays["support_matrices_n_per_m"][index],
                                       np.asarray(manifest["support_springs"][index]["matrix_n_per_m"]))
        manifest, arrays = self.fixture()
        ac.apply_support_degradation(manifest, arrays,
                                     {"abutment": {"scale": .1, "fraction": .25, "end": "negative"}})
        scaled = [i for i, s in enumerate(manifest["support_springs"]) if s["matrix_n_per_m"][0][0] != 1.]
        self.assertEqual(scaled, [0, 1])

    def test_degradation_validation(self):
        for degradation in ({"bogus": 1}, {"interface_scale": 0}, {"interface_scale": 1.5},
                            {"abutment": {"scale": .1, "fraction": .75, "end": "positive"}},
                            {"abutment": {"scale": 0., "fraction": .25, "end": "positive"}},
                            {"abutment": {"scale": .1, "fraction": .25, "end": "sideways"}}):
            manifest, arrays = self.fixture()
            with self.assertRaises(ac.CorroborationError):
                ac.apply_support_degradation(manifest, arrays, degradation)


class VerticalCrackPathTests(unittest.TestCase):
    def setUp(self):
        # Peak at radius 10 m on the +x axis, so the radial plane is y = 0 and its normal is +y.
        self.peak = np.array([10.0, 0.0, 5.0])

    def triangles(self):
        """Facets: vertical on the plane (in band), vertical but far off, horizontal, and radial-normal."""
        coords = np.array([
            [9.0, 0.1, 0.0], [11.0, 0.1, 0.0], [9.0, 0.1, 1.0],        # 0 vertical, normal y, near
            [9.0, 5.0, 0.0], [11.0, 5.0, 0.0], [9.0, 5.0, 1.0],        # 1 vertical, normal y, far
            [9.0, 0.0, 5.0], [11.0, 0.0, 5.0], [9.0, 1.0, 5.0],        # 2 horizontal (normal z)
            [10.0, 0.0, 4.0], [10.0, 1.0, 4.0], [10.0, 0.0, 6.0],      # 3 vertical, normal x (radial)
        ], dtype=float)
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int64)
        return coords, faces

    def test_selects_only_near_vertical_facets_on_the_radial_plane(self):
        coords, faces = self.triangles()
        mask = ac.vertical_crack_facets(coords, faces, self.peak, band_m=0.5, max_normal_deviation_deg=15.0)
        self.assertEqual(mask.tolist(), [True, False, False, False])

    def test_band_widens_the_plane(self):
        coords, faces = self.triangles()
        mask = ac.vertical_crack_facets(coords, faces, self.peak, band_m=6.0, max_normal_deviation_deg=15.0)
        self.assertEqual(mask.tolist(), [True, True, False, False])

    def test_peak_on_axis_is_refused(self):
        coords, faces = self.triangles()
        with self.assertRaises(ac.CorroborationError):
            ac.vertical_crack_facets(coords, faces, [0.0, 0.0, 5.0], band_m=0.5, max_normal_deviation_deg=15.0)

    def test_strength_gives_the_target_critical_opening(self):
        f_t = ac.vertical_joint_strength(critical_opening_m=2.0e-3, fracture_energy_j_m2=150.0)
        self.assertAlmostEqual(f_t, 150.0e3)
        self.assertAlmostEqual(2.0 * 150.0 / f_t, 2.0e-3)

    def test_settings_read_the_damage_selection_block(self):
        config = {"damage_selection": {"vertical_joint": {
            "fracture_energy_j_m2": 150.0, "critical_opening_m": 0.002, "band_m": None}}}
        settings = ac.vertical_joint_settings(config)
        self.assertEqual(settings["critical_opening_m"], 0.002)
        self.assertEqual(settings["max_normal_deviation_deg"], 15.0)
        self.assertIsNone(settings["band_m"])
        config["damage_selection"]["vertical_joint"]["enabled"] = False
        self.assertIsNone(ac.vertical_joint_settings(config))
        self.assertIsNone(ac.vertical_joint_settings({}))

    def test_vertical_mask_is_saved_for_the_mesh_converter(self):
        mask = np.array([True, False, True])
        split = {"joints": np.zeros((2, 6), dtype=np.int64),
                 "arrays": {"vertical_joint_faces": mask, "lift_joint_faces": np.zeros(3, dtype=bool)}}
        with tempfile.TemporaryDirectory() as temp:
            ac.save_split_arrays(Path(temp), split)
            with np.load(Path(temp) / ac.SPLIT_ARRAYS_NAME) as saved:
                self.assertEqual(saved["vertical_joint_faces"].tolist(), mask.tolist())


if __name__ == "__main__":
    unittest.main()
