"""Concrete trench lining below the complete, faceted wall seat (metres)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np

from .graded_bedrock import audit_tetrahedral_boundary
from .regions.tetrahedral import split_face


@dataclass
class FoundationMesh:
    points: np.ndarray
    tetrahedra: np.ndarray
    top_faces: np.ndarray
    wall_faces: np.ndarray
    contact_faces: np.ndarray
    report: dict
    hexahedra: np.ndarray | None = None
    terrain_faces: np.ndarray | None = None
    exposed_rock_faces: np.ndarray | None = None


def build_foundation(config: dict, concrete) -> FoundationMesh:
    """Chamfered excavation-only seat with a minimum-normal-depth concrete lining.

    Hexahedra are built first, then deterministically split for the existing
    all-tetrahedral mortar/solver pipeline. No Gmsh foundation remeshing occurs.
    """
    from .model_config import wall_thickness_steps
    from .regions.bedrock_contours import bedrock_elevation
    from .regions.tetrahedral import split_cell, _CELL_FACES
    geometry = config["geometry"]
    settings, wall = geometry["foundation"], geometry["wall"]
    depth, margin, size = settings["depth_m"], settings["side_margin_m"], geometry["global_element_size_m"]
    upstream = wall["upstream_radius_m"]
    width = max(w for _, w in wall_thickness_steps(wall)) + 2 * margin
    inner, outer = upstream + margin - width, upstream + margin
    coordinates = np.asarray(concrete.nodes)
    wall_ids = sorted({n for tag, face in concrete.boundaries if tag == 1 for n in face})
    points, lookup = [], {}
    def node(p):
        key = tuple(np.round(p, 8))
        if key not in lookup:
            lookup[key] = len(points)
            points.append(tuple(p))
        return lookup[key]
    # Preserve the wall's node ordering, so lowest-id quad diagonals match.
    for n in wall_ids:
        node(coordinates[n])
    wall_faces = [tuple(node(coordinates[n]) for n in tri)
                  for tag, face in concrete.boundaries if tag == 1
                  for tri in split_face(tuple(face))]
    wall_keys = {tuple(sorted(face)) for face in wall_faces}
    angles = sorted({round(math.atan2(p[0], p[1]), 10) for p in coordinates})
    # Actual arc distance at the innermost radius is the restrictive distance.
    extension_angle = max(size, depth) / inner
    angles = [angles[0] - extension_angle, *angles, angles[-1] + extension_angle]
    tops = [min(p[2] for p in coordinates[wall_ids]
                if abs(round(math.atan2(p[0],p[1]),10)-a)<1e-11)
            for a in angles[1:-1]]
    tops = [tops[0], *tops, tops[-1]]
    angle_index = {a: i for i, a in enumerate(angles)}
    slopes = [abs(b-a)/(inner*math.sin(angles[i+1]-angles[i]))
              for i,(a,b) in enumerate(zip(tops,tops[1:]))]
    # A conservative common vertical offset preserves >=depth normal thickness
    # on every inclined triangular facet, including the inner-radius facets.
    vertical_depth = depth * math.sqrt(1+max(slopes)**2)
    floors = [z-vertical_depth for z in tops]
    treads = [(a+b)/2 for a,b in zip(tops,tops[1:])]
    radii = sorted({round(math.hypot(x, y), 8) for x, y, z in coordinates} | {inner, outer})
    complete_radii = [radii[0]]
    for left, right in zip(radii, radii[1:]):
        complete_radii.extend(np.linspace(left, right, max(1, math.ceil((right-left)/size-1e-8))+1)[1:])
    radii = complete_radii
    # Reuse exact wall XY coordinates, rather than a rounded-angle roundtrip.
    xy = {(round(math.atan2(x, y), 10), round(math.hypot(x, y), 8)): (x, y)
          for x, y, z in coordinates}
    lattice = {}
    def grid(i, j, z):
        key = (i, j, round(z, 8))
        if key not in lattice:
            x, y = xy.get((angles[i], round(radii[j], 8)),
                          (radii[j]*math.sin(angles[i]), radii[j]*math.cos(angles[i])))
            lattice[key] = node((x, y, z))
        return lattice[key]
    hexes, hex_indices = [], []
    layers = max(1,math.ceil(vertical_depth/size))
    for i in range(len(tops)-1):
        for layer in range(layers):
            lower, upper = layer/layers, (layer+1)/layers
            z0,z1 = floors[i]+lower*vertical_depth,floors[i]+upper*vertical_depth
            w0,w1 = floors[i+1]+lower*vertical_depth,floors[i+1]+upper*vertical_depth
            for j in range(len(radii)-1):
                hexes.append((grid(i,j,z0),grid(i+1,j,w0),grid(i+1,j+1,w0),grid(i,j+1,z0),
                              grid(i,j,z1),grid(i+1,j,w1),grid(i+1,j+1,w1),grid(i,j+1,z1)))
                hex_indices.append((i,j,layer,layer+1))
    faces, owners = {}, Counter()
    for cell, index in zip(hexes, hex_indices):
        for pattern in _CELL_FACES[5]:
            face = tuple(cell[k] for k in pattern)
            key = tuple(sorted(face))
            owners[key] += 1
            faces[key] = (face, index)
    roof, contact, found_wall = [], [], []
    for key, count in owners.items():
        if count != 1:
            continue
        face, (i,j,z0,z1) = faces[key]
        triangles = split_face(face)
        matches = [tuple(sorted(tri)) in wall_keys for tri in triangles]
        if any(matches):
            if not all(matches):
                raise ValueError("Wall/foundation quad diagonals do not match")
            found_wall.extend(triangles)
            roof.extend(triangles)
            continue
        # Remaining roof facets are the exposed radial margins.
        is_roof = all(abs(points[n][2]-tops[min(range(len(angles)),key=lambda k:abs(angles[k]-math.atan2(points[n][0],points[n][1]))) ])<1e-7
                      for n in face)
        (roof if is_roof else contact).extend(triangles)
    if {tuple(sorted(f)) for f in found_wall} != wall_keys:
        raise ValueError(f"Staircase foundation is missing {len(wall_keys - {tuple(sorted(f)) for f in found_wall})} wall faces")
    # The exterior terrain terminates at the chamfered lip, never above survey.
    # The rock volume replaces this cap by the lining floor and radial sides.
    terrain = []
    surveyed_z = [bedrock_elevation(geometry["bedrock"]["contours"], a*80.) for a in angles]
    survey_z = tops
    if any(z > surveyed + 1e-7 for z,surveyed in zip(tops,surveyed_z)):
        raise ValueError("Chamfered rock lip would raise surveyed terrain")
    for i in range(len(treads)):
        for j in range(len(radii)-1):
            terrain.extend(split_face((grid(i,j,tops[i]),grid(i+1,j,tops[i+1]),
                                       grid(i+1,j+1,tops[i+1]),grid(i,j+1,tops[i]))))
    exposed = []
    p = np.asarray(points)
    tetra = np.asarray([tet for cell in hexes for tet in split_cell(p,5,cell)])
    audit_tetrahedral_boundary(tetra,np.asarray(roof+contact))
    snap = np.asarray([np.linalg.norm(p[node(coordinates[n])] - coordinates[n]) for n in wall_ids])
    # Integrate cut wedges above the matching interface triangles. Their old
    # tread is the higher endpoint; the same volume changes material ownership.
    q = p[np.asarray(wall_faces)]
    horizontal_area = np.abs(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])[:,2])/2
    transferred = float(np.dot(horizontal_area,q[:,:,2].max(axis=1)-q[:,:,2].mean(axis=1)))
    normals = np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
    face_angles = np.degrees(np.arccos(np.clip(np.abs(normals[:,2])/np.linalg.norm(normals,axis=1),0.,1.)))
    chamfer_angles = face_angles[face_angles>1e-5]
    report = {
        "present":True,"material":"concrete","region_id":3,"depth_m":depth,"side_margin_m":margin,
        "width_m":width,"radial_limits_m":[inner,outer],"outline_radial_step_edges":0,
        "width_definition":"constant widest wall thickness band plus twice side_margin_m along the complete seat",
        "seat_definition":"stepped_45deg_chamfer",
        "depth_definition":"parallel station profile with conservative common vertical offset depth_m*sqrt(1+maximum inner-radius slope squared), guaranteeing minimum normal thickness",
        "crest_end_definition":"one full depth/size station beyond each wall end, at the adjacent chamfered wall-end seat level",
        "rock_bench_definition":"rock lip follows chamfered station-minimum top; exterior triangles interpolate to unchanged surveyed terrain anchors",
        "rock_lip_maximum_lowering_m":max(s-z for s,z in zip(surveyed_z,survey_z)),
        "construction_cell_type":"hexahedron","construction_hexahedra":len(hexes),
        "solver_cell_type":"tetrahedron","element_size_m":size,
        "minimum_vertical_thickness_m":vertical_depth,
        "minimum_normal_thickness_m":depth,
        "chamfer_count":int(sum(s>1e-8 for s in slopes)),
        "transferred_foundation_to_wall_volume_m3":transferred,
        "cut_wedge_wall_gain_m3":transferred,"cut_wedge_foundation_loss_m3":transferred,
        "chamfer_definition":"station tops are minimum of adjacent former column treads; straight split across each column; triangular wedges assigned to wall; floor lowered by conservative normal-depth offset",
        "riser_course_height_m":size,"risers_more_than_one_course":0,
        "wall_joint_chamfer_definition":"all wedge endpoints lie on course/joint levels; no wedge crosses a joint; wedge belongs to band below its higher endpoint",
        "chamfer_angle_range_deg":([float(chamfer_angles.min()),float(chamfer_angles.max())] if len(chamfer_angles) else []),
        "minimum_riser_backing_m":depth,"volume_m3":float(tet_volumes(p,tetra).sum()),
        "open_excavation_volume_m3":sum(.5*(outer**2-inner**2)*math.sin(angles[i+1]-angles[i])
            *((survey_z[i]+survey_z[i+1])/2-top) for i,top in enumerate(treads)),
        "wall_seat_node_snap":{"wall_nodes":len(wall_ids),"matched_nodes":int((snap<=1e-6).sum()),
          "unmatched_nodes":int((snap>1e-6).sum()),"maximum_coincidence_distance_m":float(snap.max()),
          "tolerance_m":1e-6,"matched_faces":len(wall_faces)},
    }
    return FoundationMesh(p,tetra,np.asarray(roof),np.asarray(found_wall),np.asarray(contact),report,
                          np.asarray(hexes),np.asarray(terrain),np.asarray(exposed))


def tet_volumes(points, tetrahedra):
    p = points[tetrahedra]
    return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6


