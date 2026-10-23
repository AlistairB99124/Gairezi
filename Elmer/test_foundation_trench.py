"""Focused regressions for the optional normal-depth concrete trench."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import meshio
import numpy as np

from Elmer.model_config import ConfigurationValidationError, load_config, validate_config
from Elmer.three_body_diagnostic import face_area, interface_stiffness, prepare

ROOT = Path(__file__).resolve().parents[1]


class FoundationConfigTests(unittest.TestCase):
    def test_near_planar_groups_never_join_orientation_changes_or_disconnected_faces(self):
        from Elmer.mortar_interfaces import near_planar_groups
        points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.],
                         [1.,0.,1.],[1.,1.,1.],[4.,0.,0.],[5.,0.,0.],[4.,1.,0.]])
        points[:,0]+=10.  # Away from arch axis: both vertical faces are radial sides.
        faces=[(0,1,2),(1,3,2),(1,4,3),(4,5,3),(6,7,8)]
        groups=near_planar_groups(points,faces)
        self.assertEqual(sorted(len(a) for a,b in groups),[1,2,2])
        self.assertEqual(len({b["orientation_class"] for a,b in groups}),2)

    def test_near_planar_fit_tolerance_prevents_curved_strip_overgrowth(self):
        from Elmer.mortar_interfaces import near_planar_groups
        points=np.array([[i,j,.002*i*i] for i in range(8) for j in (0.,1.)])
        faces=[f for i in range(7) for f in ((2*i,2*i+2,2*i+1),(2*i+2,2*i+3,2*i+1))]
        groups=near_planar_groups(points,faces)
        self.assertGreater(len(groups),1)
        self.assertTrue(all(b["maximum_off_plane_distance_m"]<=.005+1e-12 for a,b in groups))
        self.assertTrue(all(b["normal_spread_deg"]<=3.+1e-8 for a,b in groups))

    def test_many_planar_pairs_use_compact_partition_connectivity(self):
        from Elmer.three_body_diagnostic import spring_partition_options
        pairs = {str(i): (1000+2*i, 1001+2*i) for i in range(1000)}
        mapping = {tag: tag for pair in pairs.values() for tag in pair}
        self.assertEqual(spring_partition_options(mapping, pairs), ("-partdual", "-connect", "1000", "-halobc"))

    def test_partition_only_coalescing_restores_exact_boundary_labels(self):
        from Elmer.mortar_interfaces import coalesced_partition_input, restore_partition_boundary_tags
        with tempfile.TemporaryDirectory(dir=ROOT / "Thickness_tests") as directory:
            root = Path(directory)
            source = root / "source.msh"
            source.write_text("$Elements\n1\n1 2 2 1000 1000 1 2 3\n$EndElements\n")
            original = coalesced_partition_input(source, root / "partition.msh", {1000})
            boundary = root / "mesh.boundary"
            boundary.write_text("1 1 1 0 303 1 2 3\n")
            audit = restore_partition_boundary_tags(root, original)
            self.assertEqual(boundary.read_text().split()[1], "1000")
            self.assertEqual(audit["restored_boundary_records"], 1)
            boundary.write_text("1 1 1 0 303 1 2 4\n")
            with self.assertRaisesRegex(ValueError, "changed boundary node IDs"):
                restore_partition_boundary_tags(root, original)

    def test_absent_block_preserves_legacy_spring_pair(self):
        config = load_config(ROOT / "config.json")
        config["geometry"].pop("foundation", None)
        config["geometry"]["bedrock"]["support"]["model"] = "distributed_elastic_granite"
        self.assertIn("wall_bedrock", interface_stiffness(config))
        self.assertNotIn("wall_foundation", interface_stiffness(config))

    def test_dimensions_and_unknown_keys_are_validated(self):
        config = load_config(ROOT / "config.json")
        for key in ("depth_m", "side_margin_m"):
            for value in (0, -1, True, float("nan"), 6):
                invalid = copy.deepcopy(config)
                invalid["geometry"]["foundation"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ConfigurationValidationError):
                    validate_config(invalid)
        config["geometry"]["foundation"]["unknown"] = 1
        with self.assertRaisesRegex(ConfigurationValidationError, "not a supported field"):
            validate_config(config)

    def test_bonded_foundation_bedrock_interface_is_valid(self):
        config = load_config(ROOT / "config.json")
        config["geometry"]["bedrock"]["support"]["model"] = "bonded"
        validate_config(config)
        self.assertNotIn("foundation_bedrock", interface_stiffness(config))

    def test_trench_floor_must_stay_above_domain_bottom(self):
        config = load_config(ROOT / "config.json")
        config["geometry"]["bedrock"]["domain"]["bottom_z_m"] = -29.25
        with self.assertRaisesRegex(ConfigurationValidationError, "trench floor must remain above"):
            validate_config(config)

    def test_trench_requires_the_graded_rock_mesher(self):
        config = load_config(ROOT / "config.json")
        del config["geometry"]["bedrock"]["mesh"]
        with self.assertRaisesRegex(ConfigurationValidationError, "graded rock mesh"):
            validate_config(config)

    def test_spring_sources_follow_the_two_interfaces(self):
        config = load_config(ROOT / "config.json")
        config["geometry"]["bedrock"]["support"]["model"] = "distributed_elastic_granite"
        springs = {"x": 4.e9, "y": 5.e9, "z": 6.e9}
        config["geometry"]["foundation"]["support"] = {"model": "spring", "spring_n_per_m3": springs}
        values = interface_stiffness(config)
        self.assertEqual(values["wall_foundation"], springs)
        self.assertEqual(values["foundation_bedrock"], config["geometry"]["bedrock"]["support"]["spring_n_per_m3"])
        self.assertNotIn("wall_bedrock", values)

    def test_missing_support_never_inherits_wall_joints(self):
        config = load_config(ROOT / "config.json")
        del config["geometry"]["foundation"]["support"]
        for joint in (None, {"model": "bonded"},
                      {"model": "spring", "spring_n_per_m3": {"x": 1.e9, "y": 2.e9, "z": 3.e9}}):
            if joint is None:
                config["geometry"]["wall"].pop("joint_support", None)
            else:
                config["geometry"]["wall"]["joint_support"] = joint
            with self.subTest(joint=joint), self.assertRaises(ConfigurationValidationError) as error:
                validate_config(config)
            self.assertIn("geometry.foundation.support: is required", str(error.exception))

    def test_explicit_support_is_independent_of_internal_wall_joints(self):
        config = load_config(ROOT / "config.json")
        config["geometry"]["bedrock"]["support"]["model"] = "distributed_elastic_granite"
        springs = {"x": 1.e9, "y": 2.e9, "z": 3.e9}
        config["geometry"]["foundation"]["support"] = {"model": "spring", "spring_n_per_m3": springs}
        self.assertEqual(interface_stiffness(config)["wall_foundation"], springs)
        config["geometry"]["foundation"]["support"] = {"model": "bonded"}
        values = interface_stiffness(config)
        self.assertNotIn("wall_foundation", values)
        self.assertIn("foundation_bedrock", values)
        self.assertIn("wall_joint_1", values)
        config["geometry"]["wall"]["joint_support"] = {"model": "bonded"}
        validate_config(config)
        self.assertEqual(set(interface_stiffness(config)), {"foundation_bedrock"})

    def test_invalid_interface_support_is_rejected(self):
        config = load_config(ROOT / "config.json")
        for support in (None, {}, {"model": "displacement", "displacement": {"x": 0, "y": 0, "z": 0}},
                        {"model": "spring"}, {"model": "bonded", "spring_n_per_m3": {"x": 1, "y": 1, "z": 1}},
                        {"model": "spring", "spring_n_per_m3": {"x": 0, "y": 1, "z": 1}}):
            config["geometry"]["foundation"]["support"] = support
            with self.subTest(support=support), self.assertRaises(ConfigurationValidationError):
                validate_config(config)


class BondedFoundationTopologyTests(unittest.TestCase):
    def test_spring_wall_seat_with_bonded_foundation_bedrock(self):
        from collections import Counter
        from Elmer.regions.tetrahedral import _CELL_FACES
        config = load_config(ROOT / "config.json")
        springs = {"x": 1.25e9, "y": 1.25e9, "z": 3.e9}
        config["geometry"]["foundation"]["support"] = {
            "model": "spring", "spring_n_per_m3": springs}
        config["geometry"]["bedrock"]["support"]["model"] = "bonded"
        validate_config(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config))
            output = root / "model"
            report = prepare(root, output, joints="spring", integrated=True)
            mesh = meshio.read(output / "three_body.msh")
            tetra = mesh.cells_dict["tetra"]
            regions = mesh.cell_data_dict["gmsh:physical"]["tetra"]
            used = {body: set(tetra[regions == body].ravel()) for body in (1, 2, 3)}
            self.assertFalse(used[1] & used[3])
            self.assertFalse(used[1] & used[2])
            self.assertTrue(used[2] & used[3])
            faces = {}
            for body in (2, 3):
                counts = Counter(tuple(sorted(cell[list(pattern)]))
                                 for cell in tetra[regions == body] for pattern in _CELL_FACES[4])
                faces[body] = {face for face, count in counts.items() if count == 1}
            shared = faces[2] & faces[3]
            self.assertEqual(len(shared), report["wall_rock_seat_triangles"])
            self.assertEqual(set().union(*map(set, shared)), used[2] & used[3])
            self.assertEqual(report["wall_foundation_connection"], "spring")
            self.assertEqual(report["foundation_bedrock_connection"], "bonded")
            self.assertEqual(report["interface_stiffness_n_per_m3"]["wall_foundation"], springs)
            self.assertEqual(report["interface_stiffness_sources"]["wall_foundation"],
                             "geometry.foundation.support.spring_n_per_m3")
            self.assertIn("wall_foundation", report["spring_pairs"])
            self.assertNotIn("foundation_bedrock", report["spring_pairs"])
            self.assertNotIn("foundation_bedrock", report["interface_stiffness_n_per_m3"])
            text = (output / "dam_model.sif").read_text()
            self.assertIn('Name = "wall_foundation', text)
            self.assertNotIn('Name = "foundation_bedrock', text)

    def test_bonded_seat_shares_all_interface_faces_without_grounding(self):
        from collections import Counter
        from Elmer.regions.tetrahedral import _CELL_FACES
        config = load_config(ROOT / "config.json")
        config["geometry"]["foundation"]["support"] = {"model": "bonded"}
        config["geometry"]["bedrock"]["support"]["model"] = "bonded"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config))
            output = root / "model"
            report = prepare(root, output, joints="spring", integrated=True)
            mesh = meshio.read(output / "three_body.msh")
            tetra = mesh.cells_dict["tetra"]
            regions = mesh.cell_data_dict["gmsh:physical"]["tetra"]
            used = {body: set(tetra[regions == body].ravel()) for body in (1, 2, 3)}
            self.assertTrue(used[1] & used[3])
            self.assertTrue(used[3] & used[2])
            self.assertFalse(used[1] & used[2])
            faces = {}
            for body in (1, 2, 3):
                counts = Counter(tuple(sorted(cell[list(pattern)]))
                                 for cell in tetra[regions == body] for pattern in _CELL_FACES[4])
                faces[body] = {face for face, count in counts.items() if count == 1}
            wall_foundation_faces = faces[1] & faces[3]
            foundation_bedrock_faces = faces[2] & faces[3]
            self.assertEqual(
                len(wall_foundation_faces),
                report["foundation"]["wall_seat_node_snap"]["matched_faces"],
            )
            self.assertEqual(len(foundation_bedrock_faces), report["wall_rock_seat_triangles"])
            self.assertEqual(set().union(*map(set, wall_foundation_faces)), used[1] & used[3])
            self.assertEqual(set().union(*map(set, foundation_bedrock_faces)), used[2] & used[3])
            self.assertEqual(report["wall_foundation_connection"], "bonded")
            self.assertEqual(report["foundation_bedrock_connection"], "bonded")
            self.assertNotIn("wall_foundation", report["spring_pairs"])
            self.assertNotIn("foundation_bedrock", report["spring_pairs"])
            self.assertNotIn("wall_foundation", report["interface_stiffness_n_per_m3"])
            self.assertNotIn("foundation_bedrock", report["interface_stiffness_n_per_m3"])
            self.assertNotIn("wall_foundation", report["interface_stiffness_sources"])
            self.assertNotIn("foundation_bedrock", report["interface_stiffness_sources"])
            self.assertIn("wall_joint_1", report["spring_pairs"])
            text = (output / "dam_model.sif").read_text()
            self.assertNotIn('Name = "wall_foundation', text)
            self.assertNotIn('Name = "foundation_bedrock', text)
            expected_fixed = 1 if config["material"]["bedrock"].get("support") is None else 0
            self.assertEqual(text.count("Displacement 1 = Real 0"), expected_fixed)
            self.assertGreaterEqual(text.count("zero relative displacement"), 2)
            self.assertEqual(text.count("Mortar BC Coefficient Displacement"), 3)


class FoundationTopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "Thickness_tests").mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(prefix="foundation-test-", dir=ROOT / "Thickness_tests")
        root = Path(cls.directory.name)
        config = load_config(ROOT / "config.json")
        config["geometry"]["foundation"]["support"] = {
            "model": "spring", "spring_n_per_m3": {"x": 4.e9, "y": 5.e9, "z": 6.e9}}
        config["geometry"]["bedrock"]["support"]["model"] = "distributed_elastic_granite"
        (root / "config.json").write_text(json.dumps(config))
        cls.output = root / "model"
        cls.report = prepare(root, cls.output, joints="spring", integrated=True)
        cls.mesh = meshio.read(cls.output / "three_body.msh")
        cls.tetra = cls.mesh.cells_dict["tetra"]
        cls.regions = cls.mesh.cell_data_dict["gmsh:physical"]["tetra"]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_separate_concrete_foundation_body(self):
        self.assertEqual(self.report["solver_body_ids"], {"wall": 1, "bedrock": 2, "foundation": 3})
        self.assertEqual(self.report["foundation"]["material"], "concrete")
        text = (self.output / "dam_model.sif").read_text()
        self.assertIn('Name = "FoundationBody"', text)
        self.assertIn("Linear System Preconditioning = ILU2\n",text)
        self.assertEqual(self.report["linear_solver"]["preconditioner"],"ILU2")
        self.assertIn('Body 3\n    Name = "FoundationBody"\n    Equation = 1\n    Material = 1\n    Body Force = 1', text)
        self.assertEqual(self.report["wall_connection"], "spring_jointed_blocks")
        self.assertEqual(self.report["wall_bedrock_connection"], "not_in_contact")

    def test_wall_never_touches_bedrock(self):
        used = {body: set(self.tetra[self.regions == body].ravel()) for body in (1, 2, 3)}
        self.assertTrue(all(used.values()))
        for a, b in ((1, 2), (1, 3), (2, 3)):
            self.assertFalse(used[a] & used[b])
        wall = {tuple(np.round(p, 7)) for p in self.mesh.points[sorted(used[1])]}
        rock = {tuple(np.round(p, 7)) for p in self.mesh.points[sorted(used[2])]}
        self.assertFalse(wall & rock)

    def test_pairs_have_coincident_faces_and_equal_areas(self):
        faces = self.mesh.cells_dict["triangle"]
        tags = self.mesh.cell_data_dict["gmsh:physical"]["triangle"]
        from Elmer.mortar_interfaces import expand_spring_pairs
        for name, pair in expand_spring_pairs(self.report["spring_pairs"]).items():
            with self.subTest(pair=name):
                selected = [faces[tags == self.report["boundary_id_map"][tag]] for tag in pair]
                self.assertGreater(len(selected[0]), 0)
                self.assertEqual(len(selected[0]), len(selected[1]))
                self.assertAlmostEqual(face_area(self.mesh.points, selected[0]), face_area(self.mesh.points, selected[1]), places=7)
                locations = [{tuple(sorted(tuple(np.round(self.mesh.points[n], 7)) for n in face)) for face in side} for side in selected]
                self.assertEqual(*locations)

    def test_every_mortar_subpair_is_connected_orientation_safe_near_planar(self):
        from Elmer.mortar_interfaces import fitted_plane, orientation_class
        faces = self.mesh.cells_dict["triangle"]
        tags = self.mesh.cell_data_dict["gmsh:physical"]["triangle"]
        for name in ("wall_foundation","foundation_bedrock"):
            entry = self.report["spring_pairs"][name]
            self.assertEqual(entry["grouping"],"near_planar")
            self.assertEqual(entry["tolerance_m"],.005)
            self.assertEqual(entry["max_angle_deg"],3.)
            for part,pair in entry["faces"].items():
                selected = faces[tags==self.report["boundary_id_map"][pair[0]]]
                p = self.mesh.points[np.unique(selected)]
                normal,offset,error = fitted_plane(p)
                self.assertLessEqual(error,.005+1e-8,(name,part))
                self.assertEqual({orientation_class(self.mesh.points[f]) for f in selected},
                                 {entry["planes"][part]["orientation_class"]})
                self.assertLessEqual(entry["planes"][part]["normal_spread_deg"],3.+1e-8)
                # Connectivity is by location, as in near_planar_groups.
                located = [{tuple(np.round(self.mesh.points[n], 8)) for n in face} for face in selected]
                remaining=set(range(len(selected))); connected={remaining.pop()}
                nodes=set(located[next(iter(connected))])
                while remaining:
                    found={i for i in remaining if len(nodes & located[i])>=2}
                    self.assertTrue(found,(name,part))
                    remaining-=found
                    nodes.update(*(located[i] for i in found))

    def test_constant_dimensions_and_course_depth(self):
        points = self.mesh.points[np.unique(self.tetra[self.regions == 3])]
        for z, inner in ((-5., 74.5), (-15., 74.5), (-29., 74.5)):
            with self.subTest(thickness_band_z=z):
                top = points[np.abs(points[:, 2] - z) < 1e-8]
                radii = np.linalg.norm(top[:, :2], axis=1)
                self.assertAlmostEqual(radii.min(), inner, delta=.001)
                self.assertAlmostEqual(radii.max(), 80.5, delta=.001)
        self.assertAlmostEqual(points[:, 2].min(), -29.-self.report["foundation"]["minimum_vertical_thickness_m"], places=6)
        self.assertGreaterEqual(self.report["foundation"]["minimum_vertical_thickness_m"], .5)
        self.assertGreaterEqual(self.report["foundation"]["minimum_riser_backing_m"], .5)
        self.assertEqual(self.report["foundation"]["construction_cell_type"], "hexahedron")
        self.assertEqual(self.report["foundation"]["element_size_m"], .5)
        self.assertEqual(self.report["foundation"]["width_m"], 6.)
        self.assertEqual(self.report["foundation"]["radial_limits_m"], [74.5, 80.5])

    def test_all_wall_seat_nodes_snap_to_foundation_with_separate_ids(self):
        faces = self.mesh.cells_dict["triangle"]
        tags = self.mesh.cell_data_dict["gmsh:physical"]["triangle"]
        pairs = self.report["spring_pairs"]["wall_foundation"]["faces"].values()
        wall_ids, foundation_ids = [np.unique(np.concatenate([
            faces[tags == self.report["boundary_id_map"][pair[side]]].ravel() for pair in pairs
        ])) for side in (0, 1)]
        self.assertFalse(set(wall_ids) & set(foundation_ids))
        points = self.mesh.points[foundation_ids]
        distances = np.array([np.linalg.norm(points - p, axis=1).min() for p in self.mesh.points[wall_ids]])
        self.assertTrue(np.all(distances <= 1e-6))
        audit = self.report["foundation"]["wall_seat_node_snap"]
        self.assertEqual(audit["wall_nodes"], len(wall_ids))
        self.assertEqual(audit["matched_nodes"], len(wall_ids))
        self.assertEqual(audit["unmatched_nodes"], 0)
        self.assertLessEqual(audit["maximum_coincidence_distance_m"], 1e-6)

    def test_trench_outline_has_no_thickness_band_corners(self):
        self.assertEqual(self.report["foundation"]["outline_radial_step_edges"], 0)
        # Top perimeter in source coordinates is a constant-radius arc on
        # either side, with radial caps only at the two chainage endpoints.
        faces = self.mesh.cells_dict["triangle"]
        tags = self.mesh.cell_data_dict["gmsh:physical"]["triangle"]
        contact = np.vstack([faces[tags == self.report["boundary_id_map"][pair[0]]]
                             for pair in self.report["spring_pairs"]["foundation_bedrock"]["faces"].values()])
        # On the flat valley floor, both cut side walls retain constant radii.
        bottom = self.mesh.points[np.unique(contact)]
        flat = bottom[np.abs(bottom[:, 2] - bottom[:,2].min()) < 1e-8]
        radii = np.linalg.norm(flat[:, :2], axis=1)
        self.assertAlmostEqual(radii.min(), 74.5, delta=.001)
        self.assertAlmostEqual(radii.max(), 80.5, delta=.001)


class StaircaseSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from Elmer.regions.combined_structure import build_combined_structure
        from Elmer.regions.tetrahedral import tetrahedralize_structure
        from Elmer.three_body_diagnostic import survey_coordinates
        from Elmer.foundation_trench import build_foundation
        cls.wall = build_combined_structure(ROOT, split_wall_joints=True)
        cls.wall.nodes = [tuple(p) for p in survey_coordinates(np.asarray(cls.wall.nodes), 78., 80.)]
        cls.foundation = build_foundation(load_config(ROOT / "config.json"), tetrahedralize_structure(cls.wall))

    def test_chamfer_wall_uses_prisms_and_foundation_uses_hexes(self):
        self.assertEqual({kind for kind, _ in self.wall.cells}, {5, 6})
        self.assertEqual(self.foundation.hexahedra.shape[1], 8)
        self.assertEqual(len(self.foundation.hexahedra), self.foundation.report["construction_hexahedra"])

    def test_quadrilateral_wall_interfaces_are_accepted(self):
        from Elmer.foundation_trench import build_foundation
        foundation = build_foundation(load_config(ROOT / "config.json"), self.wall)
        self.assertEqual(len(foundation.wall_faces), len(self.foundation.wall_faces))
        self.assertEqual(foundation.report["wall_seat_node_snap"]["unmatched_nodes"], 0)

    def test_rock_lip_is_excavation_only(self):
        from Elmer.regions.bedrock_contours import bedrock_elevation
        f = self.foundation
        lip = f.points[np.unique(f.terrain_faces)]
        rows = load_config(ROOT / "config.json")["geometry"]["bedrock"]["contours"]
        surveyed = np.array([bedrock_elevation(rows,80*np.arctan2(x,y)) for x,y,z in lip])
        self.assertTrue(np.all(lip[:,2] <= surveyed + 1e-7))

    def test_no_vertical_seat_faces_and_chamfers_near_45_degrees(self):
        from Elmer.regions.tetrahedral import split_face
        points = np.asarray(self.wall.nodes)
        angles = []
        for tag, face in self.wall.boundaries:
            if tag != 1:
                continue
            for tri in split_face(tuple(face)):
                p = points[list(tri)]
                normal = np.cross(p[1]-p[0],p[2]-p[0])
                angle = np.degrees(np.arccos(abs(normal[2])/np.linalg.norm(normal)))
                self.assertLess(angle, 47.1)
                if angle > 1e-5:
                    self.assertGreater(angle,44.)
                    angles.append(angle)
        self.assertTrue(angles)

    def test_foundation_normal_thickness_at_least_configured_depth(self):
        f = self.foundation
        for cell in f.points[f.hexahedra]:
            for tri in ((4,5,6),(4,6,7)):
                p=cell[list(tri)]
                n=np.cross(p[1]-p[0],p[2]-p[0]);n/=np.linalg.norm(n)
                layers = int(np.ceil(f.report["minimum_vertical_thickness_m"]/.5))
                self.assertGreaterEqual(layers*abs(np.dot(cell[0]-cell[4],n)),.5-1e-6)

    def test_transferred_wedge_volume_is_conserved(self):
        from Elmer.regions.tetrahedral import split_cell
        from Elmer.foundation_trench import tet_volumes
        p=np.asarray(self.wall.nodes)
        wedge=[tet for kind,cell in self.wall.cells if kind==6
               for tet in split_cell(p,kind,cell)]
        gain=float(tet_volumes(p,np.asarray(wedge)).sum())
        self.assertGreater(gain,0.)
        self.assertAlmostEqual(gain,self.foundation.report["cut_wedge_foundation_loss_m3"],places=7)
        self.assertAlmostEqual(gain,self.foundation.report["cut_wedge_wall_gain_m3"],places=7)

    def test_chamfer_cells_never_cross_wall_joints(self):
        points=np.asarray(self.wall.nodes)
        for kind,cell in self.wall.cells:
            z=points[list(cell),2]
            for joint in (-10.,-20.):
                self.assertFalse(z.min()<joint-1e-8 and z.max()>joint+1e-8)
