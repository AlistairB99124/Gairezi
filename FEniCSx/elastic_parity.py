"""Compare nodal interpolants at identical source nodes, without merging interfaces."""
import hashlib
from pathlib import Path
import tempfile
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def reference_structure(root: Path, load_case: Path, reference: Path):
    sys.path.insert(0, str(root))
    from structural_model import build_combined_structure, audit_combined_structure
    from Elmer.structure_assembly import apply_load_case, write_combined_solver_input
    from Elmer.regions.combined_structure import write_combined_gmsh
    from Elmer.regions.tetrahedral import tetrahedralize_structure

    structure = build_combined_structure(root)
    apply_load_case(structure, load_case)
    audit_combined_structure(structure)
    structure = tetrahedralize_structure(structure)
    directory = reference.parent.parent
    source_mesh = directory / "combined_structure.msh"
    source_sif = directory / "dam_model.sif"
    with tempfile.TemporaryDirectory() as temporary:
        generated_mesh = Path(temporary) / "current.msh"
        generated_sif = Path(temporary) / "current.sif"
        write_combined_gmsh(structure, generated_mesh)
        write_combined_solver_input(structure, generated_sif)
        for current, source in ((generated_mesh, source_mesh), (generated_sif, source_sif)):
            if current.read_text().splitlines() != source.read_text().splitlines():
                raise ValueError(f"Elmer reference input {source} differs from the current model; regenerate the reference.")
    if not reference.is_file():
        raise FileNotFoundError(f"Elmer reference results missing: {reference}")
    files = [source_mesh, source_sif, reference]
    files.extend(reference.parent / piece.attrib["Source"] for piece in ET.parse(reference).findall(".//Piece"))
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("An Elmer reference PVTU piece is missing")
    if any(path.stat().st_mtime < max(source_mesh.stat().st_mtime, source_sif.stat().st_mtime) for path in files[2:]):
        raise ValueError("Elmer reference results predate their mesh/SIF inputs")
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    return structure, hashes


def read_reference(structure, reference: Path):
    """Map each Elmer point through its parent source cell, not coordinates alone.

    Spring seats and spring-jointed wall blocks duplicate coincident nodes; only
    the owning cell identifies which copy an Elmer point belongs to.
    """
    points = np.asarray(structure.nodes)
    node_regions = np.zeros(len(points), dtype=np.int32)
    for (kind, cell), region in zip(structure.cells, structure.region_ids):
        if region not in (1, 2, 3):
            raise ValueError(f"Unexpected structural region {region}")
        node_regions[list(cell)] |= 1 << (region - 1)
    source_centres = np.asarray([points[list(cell)].mean(axis=0) for _, cell in structure.cells])
    source_sizes = np.asarray([len(cell) for _, cell in structure.cells])
    tree = cKDTree(source_centres)
    values = np.zeros((len(points), 3))
    stresses = np.zeros((len(points), 6))
    counts = np.zeros(len(points), dtype=int)
    cell_seen = np.zeros(len(structure.cells), dtype=bool)
    for piece in ET.parse(reference).findall(".//Piece"):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(reference.parent / piece.attrib["Source"]))
        reader.Update()
        grid = reader.GetOutput()
        if reader.GetErrorCode() or grid.GetNumberOfCells() == 0:
            raise ValueError(f"Cannot read Elmer piece {piece.attrib['Source']}")
        displacement_array = grid.GetPointData().GetArray("displacement")
        if displacement_array is None:
            raise ValueError("Elmer reference has no displacement field")
        displacement = vtk_to_numpy(displacement_array)
        # Elmer ResultOutput writes deformed coordinates and includes surface cells.
        xyz = vtk_to_numpy(grid.GetPoints().GetData()) - displacement
        connectivity = vtk_to_numpy(grid.GetCells().GetConnectivityArray())
        offsets = vtk_to_numpy(grid.GetCells().GetOffsetsArray())
        sizes = np.diff(offsets)
        centres = np.add.reduceat(xyz[connectivity], offsets[:-1], axis=0) / sizes[:, None]
        volume_cells = np.isin(vtk_to_numpy(grid.GetCellTypesArray()), [10, 12, 13, 14])
        distance, parents = tree.query(centres[volume_cells])
        if distance.max() > 1e-7:
            raise ValueError("Elmer result cells do not match the current source mesh")
        cell_seen[parents] = True
        volume_sizes = sizes[volume_cells]
        if np.any(volume_sizes != source_sizes[parents]):
            raise ValueError("Elmer result cell types do not match the current source mesh")
        point_source = np.full(len(xyz), -1, dtype=np.int64)
        volume_offsets = offsets[:-1][volume_cells]
        for size in np.unique(volume_sizes):
            chosen = volume_sizes == size
            elmer_ids = connectivity[volume_offsets[chosen, None] + np.arange(size)]
            source_ids = np.asarray([structure.cells[parent][1] for parent in parents[chosen]])
            gaps = np.linalg.norm(xyz[elmer_ids][:, :, None] - points[source_ids][:, None, :], axis=3)
            nearest = gaps.argmin(axis=2)
            if np.take_along_axis(gaps, nearest[:, :, None], axis=2).max() > 1e-7:
                raise ValueError("Elmer point does not match a node of its source cell")
            mapped = np.take_along_axis(source_ids, nearest, axis=1).ravel()
            flat = elmer_ids.ravel()
            previous = point_source[flat]
            if np.any((previous >= 0) & (previous != mapped)):
                raise ValueError("Elmer piece merged nodes that must remain separate (spring seat or wall joint)")
            point_source[flat] = mapped
            # Repeated Elmer points within one batch must also map to a single source node.
            if np.any(point_source[flat] != mapped):
                raise ValueError("Elmer piece merged nodes that must remain separate (spring seat or wall joint)")
        stress_names = ("stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_yz", "stress_xz")
        if any(grid.GetPointData().GetArray(name) is None for name in stress_names):
            raise ValueError("Elmer reference is missing stress components")
        stress = np.column_stack([vtk_to_numpy(grid.GetPointData().GetArray(name)) for name in stress_names])
        selected = np.flatnonzero(point_source >= 0)
        if len(selected):
            source_ids = point_source[selected]
            existing = counts[source_ids] > 0
            if np.any(existing) and not np.allclose(
                values[source_ids[existing]] / counts[source_ids[existing], None],
                displacement[selected[existing]], rtol=1e-6, atol=1e-10
            ):
                raise ValueError("Elmer ghost-node displacement values disagree")
            if np.any(existing) and not np.allclose(
                stresses[source_ids[existing]] / counts[source_ids[existing], None],
                stress[selected[existing]], rtol=1e-6, atol=1e-6,
            ):
                raise ValueError("Elmer ghost-node stress values disagree")
            np.add.at(values, source_ids, displacement[selected])
            np.add.at(stresses, source_ids, stress[selected])
            np.add.at(counts, source_ids, 1)
    if not cell_seen.all() or np.any(counts == 0):
        raise ValueError("Elmer reference does not cover every source cell/node")
    return values / counts[:, None], stresses / counts[:, None], node_regions


def compare(imported, structure, displacement, sigma, reference, settings):
    expected, expected_stress, node_regions = read_reference(structure, reference)
    original_node_count = len(expected)
    if imported.interpolation_parents:
        expected = np.vstack((expected, [expected[list(ids)].mean(axis=0) for ids in imported.interpolation_parents]))
        expected_stress = np.vstack((expected_stress, [
            expected_stress[list(ids)].mean(axis=0) for ids in imported.interpolation_parents
        ]))
        node_regions = np.concatenate((node_regions, [
            node_regions[ids[0]] for ids in imported.interpolation_parents
        ]))
    tets, volume = imported.tetrahedra, imported.volumes

    def norm_squared(values):
        nodal = values[tets]
        return float(np.sum(volume / 20 * (
            np.sum(nodal.sum(axis=1)**2, axis=1) + np.sum(nodal**2, axis=(1, 2))
        )))

    error = np.sqrt(norm_squared(displacement - expected) / norm_squared(expected))
    elmer_reactions = {}
    from dam_elasticity import ground_supports

    for tag, coefficients in ground_supports(imported, structure)[0].items():
        coefficients = np.asarray(coefficients)
        from dolfinx import mesh
        facets = imported.facet_tags.find(tag)
        geometry_nodes = mesh.entities_to_geometry(imported.domain, 2, facets)
        faces = imported.domain.geometry.input_global_indices[geometry_nodes]
        xyz = imported.points[faces]
        area = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1) / 2
        elmer_reactions[str(tag)] = (-coefficients * (
            expected[faces].mean(axis=1) * area[:, None]
        ).sum(axis=0)).tolist()
    probes = []
    for probe in settings["probes"]:
        candidates = np.flatnonzero((node_regions[:original_node_count] & probe["region_id"]) != 0)
        target = np.asarray(probe["point_m"], dtype=float)
        distances = np.linalg.norm(imported.points[candidates] - target, axis=1)
        node = int(candidates[np.argmin(distances)])
        distance = float(distances.min())
        if distance > probe["maximum_distance_m"]:
            raise ValueError(f"Probe {probe['name']} has no node within its configured distance")
        magnitude = np.linalg.norm(expected[node])
        if magnitude <= 1e-12:
            raise ValueError(f"Probe {probe['name']} has near-zero reference displacement; relative comparison is undefined")
        probes.append({
            "name": probe["name"], "region_id": probe["region_id"], "source_node_id": node,
            "requested_point_m": target.tolist(), "actual_point_m": imported.points[node].tolist(),
            "distance_m": distance, "elmer_displacement_m": expected[node].tolist(),
            "fenicsx_displacement_m": displacement[node].tolist(),
            "relative_vector_difference": float(np.linalg.norm(displacement[node] - expected[node]) / magnitude),
        })
    tensors = sigma.x.array.reshape(-1, 9)
    original_tensors = np.empty_like(tensors)
    original_tensors[imported.domain.topology.original_cell_index] = tensors
    components = original_tensors[:, [0, 4, 8, 1, 5, 2]]
    region_stress = {}
    for region in np.unique(imported.regions):
        selected = imported.regions == region
        weights = volume[selected]
        fenics_mean = np.average(components[selected], axis=0, weights=weights)
        elmer_mean = np.average(expected_stress[tets[selected]].mean(axis=1), axis=0, weights=weights)
        region_stress[str(region)] = {
            "component_order": ["xx", "yy", "zz", "xy", "yz", "xz"],
            "fenicsx_mean_stress_pa": fenics_mean.tolist(), "elmer_mean_stress_pa": elmer_mean.tolist(),
            "relative_mean_tensor_difference": float(np.linalg.norm(fenics_mean - elmer_mean) / np.linalg.norm(elmer_mean)),
        }
    return {
        "relative_l2_displacement_difference": float(error),
        "norm_definition": "exact tetrahedral L2 norm of P1 interpolants of both nodal solutions; mixed-cell references are not their native hex fields",
        "displacement_limit": settings["maximum_relative_l2_difference"],
        "probe_limit": settings["maximum_probe_relative_difference"],
        "displacement_checks_passed": bool(error <= settings["maximum_relative_l2_difference"]
            and all(p["relative_vector_difference"] <= settings["maximum_probe_relative_difference"] for p in probes)),
        "probes": probes, "region_mean_stress_comparison": region_stress,
        "elmer_ground_spring_reactions_n": elmer_reactions,
        "reaction_note": "Integrated from Elmer nodal displacements on the same tagged triangles; not a native Elmer reaction output.",
        "stress_note": "Diagnostic only: Elmer recovered nodal stresses versus tetrahedral DG0 stresses.",
    }
