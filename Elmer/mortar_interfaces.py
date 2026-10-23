"""Connected near-planar mortar groups for coincident spring interfaces."""
from collections import defaultdict

import numpy as np

from .regions.model_groups import FIRST_PLANAR_MORTAR_BOUNDARY_ID


def expand_spring_pairs(pairs):
    """Flatten logical face groups; legacy two-tag pairs remain accepted."""
    return {
        **{name: tuple(value) for name, value in pairs.items() if not isinstance(value, dict)},
        **{f"{name}/{part}": tuple(tags) for name, value in pairs.items() if isinstance(value, dict)
           for part, tags in value["faces"].items()},
    }


def plane_key(points):
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError("Degenerate mortar face")
    normal /= norm
    if normal[np.argmax(np.abs(normal))] < 0:
        normal *= -1
    return tuple(np.round(normal, 7)) + (round(float(normal @ points[0]), 7),)


NEAR_PLANAR_TOLERANCE_M = 0.005
NEAR_PLANAR_MAX_ANGLE_DEG = 3.0


def orientation_class(p):
    normal = np.cross(p[1]-p[0],p[2]-p[0])
    normal /= np.linalg.norm(normal)
    if abs(normal[2]) > .999999:
        return "tread"
    if abs(normal[2]) > .1:
        return "chamfer"
    radial = p[:,:2].mean(axis=0)
    radial /= np.linalg.norm(radial)
    return "side_wall" if abs(normal[:2] @ radial) > .9 else "riser"


def fitted_plane(p):
    centre = p.mean(axis=0)
    normal = np.linalg.svd(p-centre, full_matrices=False)[2][-1]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal *= -1
    return normal, float(normal @ centre), float(np.max(np.abs((p-centre) @ normal)))


def near_planar_groups(points, faces, tolerance_m=NEAR_PLANAR_TOLERANCE_M,
                       max_angle_deg=NEAR_PLANAR_MAX_ANGLE_DEG):
    """Edge-connected groups, one orientation class, bounded fit and normal spread."""
    xyz = points[np.asarray(faces)]
    normals = np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    normals /= np.linalg.norm(normals,axis=1)[:,None]
    classes = [orientation_class(p) for p in xyz]
    locations = [[tuple(p) for p in np.round(q,8)] for q in xyz]
    edges = defaultdict(list)
    for i,triangle in enumerate(locations):
        for a,b in ((0,1),(1,2),(2,0)):
            edges[tuple(sorted((triangle[a],triangle[b])))].append(i)
    adjacency = [set() for _ in faces]
    for owners in edges.values():
        for i in owners:
            adjacency[i].update(j for j in owners if i!=j)
    remaining = set(range(len(faces)))
    result = []
    cosine = np.cos(np.radians(max_angle_deg))
    for seed in range(len(faces)):
        if seed not in remaining:
            continue
        members = [seed]
        remaining.remove(seed)
        nodes = set(locations[seed])
        pending = set(adjacency[seed]) & remaining
        while pending:
            candidate = min(pending)
            pending.remove(candidate)
            if candidate not in remaining or classes[candidate]!=classes[seed]:
                continue
            if np.min(np.abs(normals[members] @ normals[candidate])) < cosine-1e-12:
                continue
            proposed = nodes | set(locations[candidate])
            normal,offset,error = fitted_plane(np.asarray(sorted(proposed)))
            if error > tolerance_m+1e-12:
                continue
            members.append(candidate)
            remaining.remove(candidate)
            nodes = proposed
            pending.update(adjacency[candidate] & remaining)
        normal,offset,error = fitted_plane(np.asarray(sorted(nodes)))
        cosines = np.abs(normals[members] @ normals[members].T)
        spread = float(np.degrees(np.arccos(np.clip(cosines.min(),-1,1))))
        result.append((members,{"unit_normal":normal.tolist(),"offset_m":offset,
                       "maximum_off_plane_distance_m":error,"normal_spread_deg":spread,
                       "orientation_class":classes[seed],"triangles_per_side":len(members)}))
    return result


def split_planar_interfaces(points, boundaries, pairs):
    """Retag matching sides into connected, orientation-safe near-planar groups."""
    points = np.asarray(points)
    replacement,contract = {},dict(pairs)
    next_tag = FIRST_PLANAR_MORTAR_BOUNDARY_ID
    def location(face):
        return tuple(sorted(tuple(p) for p in np.round(points[list(face)],8)))
    for name in ("foundation_bedrock","wall_foundation"):
        if name not in pairs:
            continue
        slave,master = pairs[name]
        slave_ids = [i for i,(tag,face) in enumerate(boundaries) if tag==slave]
        master_ids = {location(face):i for i,(tag,face) in enumerate(boundaries) if tag==master}
        if len(slave_ids)!=len(master_ids):
            raise ValueError(f"Unmatched near-planar mortar interface: {name}")
        source_faces = [boundaries[i][1] for i in slave_ids]
        faces,planes = {},{}
        for number,(members,metadata) in enumerate(near_planar_groups(points,source_faces)):
            tags = (slave,master) if number==0 else (next_tag,next_tag+1)
            if number:
                next_tag += 2
            part = f"{metadata['orientation_class']}_{number:04d}"
            faces[part] = list(tags)
            planes[part] = metadata
            for member in members:
                i = slave_ids[member]
                key = location(boundaries[i][1])
                if key not in master_ids:
                    raise ValueError(f"Non-coincident mortar face: {name}")
                replacement[i] = tags[0]
                replacement[master_ids[key]] = tags[1]
        contract[name] = {"faces":faces,"planes":planes,"grouping":"near_planar",
                          "legacy_tags":[slave,master],"tolerance_m":NEAR_PLANAR_TOLERANCE_M,
                          "max_angle_deg":NEAR_PLANAR_MAX_ANGLE_DEG,
                          "continuity":"edge_connected; single orientation class"}
    return [(replacement.get(i,tag),face) for i,(tag,face) in enumerate(boundaries)],contract


def coalesced_partition_input(source, destination, mortar_tags):
    """Coalesce tags for ElmerGrid ONLY; retain exact original face labels."""
    original = {}
    connected_tag = min(mortar_tags)
    in_elements = False
    with source.open() as incoming, destination.open("w") as outgoing:
        for line in incoming:
            if line.strip() == "$Elements":
                in_elements = True
            elif line.strip() == "$EndElements":
                in_elements = False
            elif in_elements:
                fields = line.split()
                if len(fields) > 3 and fields[1] == "2":
                    ntags = int(fields[2])
                    key = tuple(sorted(int(n) for n in fields[3+ntags:]))
                    tag = int(fields[3])
                    if key in original:
                        raise ValueError("Duplicate source boundary face")
                    original[key] = tag
                    if tag in mortar_tags:
                        fields[3:3+ntags] = [str(connected_tag)] * ntags
                        line = " ".join(fields) + "\n"
            outgoing.write(line)
    return original


def restore_partition_boundary_tags(mesh, original):
    """Restore physical tags by source node IDs, failing on any altered face."""
    files = [path for path in [mesh / "mesh.boundary"] if path.is_file()]
    files += sorted(mesh.glob("partitioning.*/part.*.boundary"))
    if not files:
        raise ValueError("No converted boundary files to restore")
    total = 0
    for path in files:
        lines = []
        with path.open() as stream:
            for line in stream:
                fields = line.split()
                if int(fields[4]) != 303:
                    raise ValueError("Planar mortar partition restoration requires triangle boundaries")
                key = tuple(sorted(int(n) for n in fields[5:]))
                if key not in original:
                    raise ValueError(f"ElmerGrid changed boundary node IDs: {key}")
                fields[1] = str(original[key])
                lines.append(" ".join(fields) + "\n")
                total += 1
        path.write_text("".join(lines))
    return {"restored_boundary_records": total, "boundary_files": len(files),
            "method": "coalesced mortar tag for partitioning only; source-node face-key restoration before solve"}
