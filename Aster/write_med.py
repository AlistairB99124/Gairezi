"""Write mesh.med from mesh_intermediate.npz with MEDCoupling (run inside the container)."""
import json
import sys

import numpy as np
import medcoupling as mc

MED_TYPES = {
    "tetra": mc.NORM_TETRA4,
    "pyramid": mc.NORM_PYRA5,
    "wedge": mc.NORM_PENTA6,
    "hexahedron": mc.NORM_HEXA8,
    "triangle": mc.NORM_TRI3,
    "quad": mc.NORM_QUAD4,
}
VOLUME_ORDER = ("tetra", "pyramid", "wedge", "hexahedron")
FACE_ORDER = ("triangle", "quad")
# Gmsh -> MED node order; MED volume cells have the opposite base-face orientation.
GMSH_TO_MED = {
    "hexahedron": [0, 3, 2, 1, 4, 7, 6, 5],
    "wedge": [0, 2, 1, 3, 5, 4],
}


def build_level(points, npz, order, dimension):
    coords = mc.DataArrayDouble(points.tolist(), len(points), 3)
    coords.setInfoOnComponents(["X", "Y", "Z"])
    mesh = mc.MEDCouplingUMesh("dam", dimension)
    mesh.setCoords(coords)
    types = [name for name in order if f"conn_{name}" in npz]
    cell_count = sum(len(npz[f"conn_{name}"]) for name in types)
    mesh.allocateCells(cell_count)
    families = []
    for name in types:
        connectivity = npz[f"conn_{name}"]
        if name in GMSH_TO_MED:
            connectivity = connectivity[:, GMSH_TO_MED[name]]
        for row in connectivity:
            mesh.insertNextCell(MED_TYPES[name], row.tolist())
        families.append(npz[f"fam_{name}"])
    mesh.finishInsertingCells()
    return mesh, np.concatenate(families)


def main(npz_path, med_path, groups_path):
    npz = np.load(npz_path)
    groups = json.load(open(groups_path))
    points = npz["points"]
    volumes, volume_families = build_level(points, npz, VOLUME_ORDER, 3)
    faces, face_families = build_level(points, npz, FACE_ORDER, 2)
    faces.setCoords(volumes.getCoords())

    signed = volumes.getMeasureField(False).getArray().toNumPyArray()
    negative = int((signed <= 0).sum())
    print(f"volume cells: {len(signed)}, non-positive measures: {negative}")
    if negative:
        sys.exit("Non-positive cell volumes: node ordering does not match MED convention")

    mesh = mc.MEDFileUMesh()
    mesh.setName("dam")
    mesh.setCoords(volumes.getCoords())
    mesh.setMeshAtLevel(0, volumes)
    mesh.setMeshAtLevel(-1, faces)
    mesh.setFamilyFieldArr(0, mc.DataArrayInt(volume_families.astype(np.int64)))
    mesh.setFamilyFieldArr(-1, mc.DataArrayInt(face_families.astype(np.int64)))
    for name, family_id in groups.items():
        mesh.addFamily(name, family_id)
        mesh.setFamiliesOnGroup(name, [name])

    # One POI1 cell per foundation node, each in its own group SPR<k>, for discrete springs.
    spring_nodes = npz["spring_nodes"]
    if len(spring_nodes) == 0:
        mesh.write(med_path, 2)
        print("wrote", med_path)
        return
    spring_cells = mc.MEDCouplingUMesh("dam", 0)
    spring_cells.setCoords(volumes.getCoords())
    spring_cells.allocateCells(len(spring_nodes))
    for node in spring_nodes:
        spring_cells.insertNextCell(mc.NORM_POINT1, [int(node)])
    spring_cells.finishInsertingCells()
    mesh.setMeshAtLevel(-3, spring_cells)
    spring_family_ids = -(1000 + np.arange(len(spring_nodes), dtype=np.int64))
    mesh.setFamilyFieldArr(-3, mc.DataArrayInt(spring_family_ids))
    for index, family_id in enumerate(spring_family_ids):
        mesh.addFamily(f"SPR{index}", int(family_id))
        mesh.setFamiliesOnGroup(f"SPR{index}", [f"SPR{index}"])
    mesh.setFamiliesOnGroup("SPRINGS", [f"SPR{index}" for index in range(len(spring_nodes))])
    mesh.write(med_path, 2)
    print("wrote", med_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
