"""Run with pvpython: retain concrete and a near-structure rock reporting region."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import reference as vtk_reference, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkStaticCellLocator, vtkUnstructuredGrid
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm
from vtkmodules.vtkFiltersCore import (
    vtkAppendFilter, vtkFeatureEdges, vtkImplicitPolyDataDistance, vtkThreshold,
)
from vtkmodules.vtkFiltersGeneral import vtkClipDataSet
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
from vtkmodules.vtkIOXML import (
    vtkXMLPUnstructuredGridReader, vtkXMLUnstructuredGridReader, vtkXMLUnstructuredGridWriter,
)


DISTANCE_ARRAY = "DistanceToConcrete_m"
EDGE_DISTANCE_ARRAY = "DistanceToSharpEdge_m"
BODY_NAMES = {1: "wall", 2: "bedrock", 3: "foundation"}
# Concrete-surface edges whose faces meet at least this far from flat are
# treated as stress singularities (90-degree steps, joint/face and rim edges);
# 45-degree seat chamfers fall below it.
SHARP_EDGE_ANGLE_DEG = 60.
# Default exclusion: one wall element (0.5 m); nodal peaks on the edge itself
# are mesh-dependent in linear elasticity.
EDGE_EXCLUSION_M = .5


def select_body(mesh, lower: int, upper: int):
    threshold = vtkThreshold()
    threshold.SetInputData(mesh)
    threshold.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, "GeometryIds")
    threshold.SetLowerThreshold(lower)
    threshold.SetUpperThreshold(upper)
    threshold.SetThresholdFunction(vtkThreshold.THRESHOLD_BETWEEN)
    threshold.Update()
    result = vtkUnstructuredGrid()
    result.DeepCopy(threshold.GetOutput())
    return result


def reference_coordinates(mesh) -> np.ndarray:
    displacement = mesh.GetPointData().GetArray("displacement")
    if displacement is None:
        raise ValueError("Missing solved displacement")
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    values = vtk_to_numpy(displacement)
    if not np.isfinite(points).all() or not np.isfinite(values).all():
        raise ValueError("Non-finite coordinates or displacement")
    return points - values


def reporting_region(mesh, distance_m: float = 5.):
    if not math.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("Reporting distance must be finite and positive")
    labels = mesh.GetCellData().GetArray("GeometryIds")
    if labels is None or set(np.unique(vtk_to_numpy(labels))) not in ({1, 2}, {1, 2, 3}):
        raise ValueError("Expected attributed wall/bedrock (1/2), optionally foundation (3)")
    concrete = select_body(mesh, 1, 1)
    foundation = select_body(mesh, 3, 3)
    if foundation.GetNumberOfCells():
        joined = vtkAppendFilter()
        joined.MergePointsOff()
        joined.AddInputData(concrete)
        joined.AddInputData(foundation)
        joined.Update()
        concrete = vtkUnstructuredGrid()
        concrete.DeepCopy(joined.GetOutput())
    rock = select_body(mesh, 2, 2)
    reference = vtkUnstructuredGrid()
    reference.DeepCopy(concrete)
    points = vtkPoints()
    points.SetData(numpy_to_vtk(reference_coordinates(concrete), deep=True))
    reference.SetPoints(points)
    surface = vtkDataSetSurfaceFilter()
    surface.SetInputData(reference)
    surface.Update()
    distance = vtkImplicitPolyDataDistance()
    distance.SetInput(surface.GetOutput())
    rock_distances = np.asarray([
        abs(distance.EvaluateFunction(point)) for point in reference_coordinates(rock)
    ])
    for body, values in ((concrete, np.zeros(concrete.GetNumberOfPoints())), (rock, rock_distances)):
        array = numpy_to_vtk(values, deep=True)
        array.SetName(DISTANCE_ARRAY)
        body.GetPointData().AddArray(array)
    clip = vtkClipDataSet()
    clip.SetInputData(rock)
    clip.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, DISTANCE_ARRAY)
    clip.SetValue(distance_m)
    clip.SetOutputPointsPrecision(vtkAlgorithm.DOUBLE_PRECISION)
    clip.InsideOutOn()
    clip.Update()
    if not clip.GetOutput().GetNumberOfCells():
        raise ValueError("No bedrock remains in reporting region")
    append = vtkAppendFilter()
    append.MergePointsOff()
    append.AddInputData(concrete)
    append.AddInputData(clip.GetOutput())
    append.Update()
    result = vtkUnstructuredGrid()
    result.DeepCopy(append.GetOutput())
    retained_distances = vtk_to_numpy(result.GetPointData().GetArray(DISTANCE_ARRAY))
    # VTK clip-edge weights have sub-micrometre interpolation roundoff even
    # with double-precision output coordinates.
    if not np.isfinite(retained_distances).all() or retained_distances.max() > distance_m + 1e-6:
        raise ValueError(
            f"Reporting-region distance check failed: range "
            f"{retained_distances.min()} to {retained_distances.max()}, threshold {distance_m}"
        )
    return result


def sharp_concrete_edges(mesh, angle_deg: float = SHARP_EDGE_ANGLE_DEG):
    """Feature edges of wall/foundation surfaces in undeformed coordinates.

    Bodies and wall blocks are joined by springs with duplicated nodes, so
    joint and interface faces are surfaces here and their perimeters count.
    """
    if not math.isfinite(angle_deg) or not 0. < angle_deg < 180.:
        raise ValueError("Sharp-edge angle must be between 0 and 180 degrees")
    concrete_parts = vtkAppendFilter()
    concrete_parts.MergePointsOff()
    concrete_parts.AddInputData(select_body(mesh, 1, 1))
    concrete_parts.AddInputData(select_body(mesh, 3, 3))
    concrete_parts.Update()
    concrete = vtkUnstructuredGrid()
    concrete.DeepCopy(concrete_parts.GetOutput())
    if not concrete.GetNumberOfCells():
        raise ValueError("No concrete cells for sharp-edge detection")
    undeformed = vtkUnstructuredGrid()
    undeformed.DeepCopy(concrete)
    points = vtkPoints()
    points.SetData(numpy_to_vtk(reference_coordinates(concrete), deep=True))
    undeformed.SetPoints(points)
    surface = vtkDataSetSurfaceFilter()
    surface.SetInputData(undeformed)
    surface.Update()
    edges = vtkFeatureEdges()
    edges.SetInputData(surface.GetOutput())
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.ManifoldEdgesOff()
    edges.SetFeatureAngle(angle_deg)
    edges.Update()
    result = edges.GetOutput()
    if not result.GetNumberOfCells():
        raise ValueError("No sharp concrete edges found")
    return result


def add_edge_distance(region, angle_deg: float = SHARP_EDGE_ANGLE_DEG):
    """Add each point's undeformed distance to the nearest sharp concrete edge."""
    edges = sharp_concrete_edges(region, angle_deg)
    locator = vtkStaticCellLocator()
    locator.SetDataSet(edges)
    locator.BuildLocator()
    closest = [0., 0., 0.]
    cell, sub, squared = vtk_reference(0), vtk_reference(0), vtk_reference(0.)
    values = np.empty(region.GetNumberOfPoints())
    for i, point in enumerate(reference_coordinates(region)):
        locator.FindClosestPoint(point, closest, cell, sub, squared)
        values[i] = math.sqrt(float(squared))
    array = numpy_to_vtk(values, deep=True)
    array.SetName(EDGE_DISTANCE_ARRAY)
    region.GetPointData().AddArray(array)
    return region


def _principal_peaks(principal, coordinates, mask) -> dict:
    if not mask.any():
        return {"peak_tension_mpa": None, "peak_compression_mpa": None,
                "tension_reference_xyz_m": None, "compression_reference_xyz_m": None}
    ids = np.flatnonzero(mask)
    tension = int(ids[np.argmax(principal[ids, 2])])
    compression = int(ids[np.argmin(principal[ids, 0])])
    return {
        "peak_tension_mpa": max(0., float(principal[tension, 2])) / 1e6,
        "peak_compression_mpa": max(0., -float(principal[compression, 0])) / 1e6,
        "tension_reference_xyz_m": coordinates[tension].tolist(),
        "compression_reference_xyz_m": coordinates[compression].tolist(),
    }


def stress_summary(mesh, edge_exclusion_m: float | None = None) -> dict:
    """Raw peaks per body; with edge_exclusion_m, also peaks at least that far
    (undeformed) from sharp concrete edges, which needs EDGE_DISTANCE_ARRAY."""
    if edge_exclusion_m is not None and (not math.isfinite(edge_exclusion_m) or edge_exclusion_m <= 0):
        raise ValueError("Edge exclusion distance must be finite and positive")
    report = {}
    for body, name in BODY_NAMES.items():
        selected = select_body(mesh, body, body)
        if not selected.GetNumberOfCells():
            if body == 3:
                continue
            raise ValueError(f"Missing reporting body: {name}")
        data = selected.GetPointData()
        tensor = np.empty((selected.GetNumberOfPoints(), 3, 3))
        for field, i, j in (
            ("stress_xx", 0, 0), ("stress_yy", 1, 1), ("stress_zz", 2, 2),
            ("stress_xy", 0, 1), ("stress_xz", 0, 2), ("stress_yz", 1, 2),
        ):
            array = data.GetArray(field)
            if array is None:
                raise ValueError(f"Missing solved field: {field}")
            tensor[:, i, j] = tensor[:, j, i] = vtk_to_numpy(array)
        if not np.isfinite(tensor).all():
            raise ValueError(f"Non-finite reporting stresses: {name}")
        principal = np.linalg.eigvalsh(tensor)
        coordinates = reference_coordinates(selected)
        report[name] = {
            "reporting_cells": selected.GetNumberOfCells(),
            **_principal_peaks(principal, coordinates, np.ones(len(principal), dtype=bool)),
        }
        if edge_exclusion_m is not None:
            edge_distance = data.GetArray(EDGE_DISTANCE_ARRAY)
            if edge_distance is None:
                raise ValueError(f"Missing {EDGE_DISTANCE_ARRAY}; call add_edge_distance first")
            # Small tolerance keeps nodes exactly one exclusion distance away.
            kept = vtk_to_numpy(edge_distance) >= edge_exclusion_m - 1e-6
            report[name]["away_from_sharp_edges"] = {
                "excluded_points": int((~kept).sum()),
                **_principal_peaks(principal, coordinates, kept),
            }
    return report


def export_region(
    result: Path, output: Path, distance_m: float = 5., edge_exclusion_m: float = EDGE_EXCLUSION_M,
) -> dict:
    if not result.is_file():
        raise FileNotFoundError(result)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty reporting output directory")
    reader = vtkXMLPUnstructuredGridReader() if result.suffix.lower() == ".pvtu" else vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(result))
    reader.Update()
    source = reader.GetOutput()
    region = add_edge_distance(reporting_region(source, distance_m))
    summary = stress_summary(region, edge_exclusion_m)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "structure_and_near_rock.vtu"
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(region)
    if writer.Write() != 1:
        raise RuntimeError("Failed to export reporting region")
    check = vtkXMLUnstructuredGridReader()
    check.SetFileName(str(path))
    check.Update()
    saved = check.GetOutput()
    if saved.GetNumberOfCells() != region.GetNumberOfCells() or stress_summary(saved, edge_exclusion_m) != summary:
        raise RuntimeError("Reporting VTU readback mismatch")
    np.testing.assert_array_equal(
        vtk_to_numpy(saved.GetPoints().GetData()), vtk_to_numpy(region.GetPoints().GetData()),
    )
    for original, restored in (
        (region.GetPointData(), saved.GetPointData()), (region.GetCellData(), saved.GetCellData()),
    ):
        for i in range(original.GetNumberOfArrays()):
            array = original.GetArray(i)
            saved_array = restored.GetArray(array.GetName())
            if saved_array is None:
                raise RuntimeError(f"Reporting export lost field {array.GetName()}")
            np.testing.assert_array_equal(vtk_to_numpy(array), vtk_to_numpy(saved_array))
    report = {
        "source_full_domain_result": str(result.resolve()),
        "result": str(path.resolve()), "rock_reporting_distance_m": distance_m,
        "selection": "All wall and optional foundation; rock within nearest concrete-surface distance in undeformed coordinates",
        "boundary_approximation": "Exact surface distance at rock vertices; linearly interpolated within cells, clipped at threshold",
        "stress_method": "Principal eigenvalues of recovered nodal stress tensor; interpolated tensors at clip intersections",
        "full_domain_cells": source.GetNumberOfCells(),
        "maximum_exported_distance_m": float(
            vtk_to_numpy(saved.GetPointData().GetArray(DISTANCE_ARRAY)).max()
        ),
        "bodies": summary,
        "sharp_edge_exclusion": {
            "distance_m": edge_exclusion_m,
            "feature_angle_deg": SHARP_EDGE_ANGLE_DEG,
            "edges": "Wall/foundation surface edges (undeformed) where faces meet at or beyond the feature angle, "
                     "including step corners, joint/interface perimeters and rims; 45-degree seat chamfers are not edges",
            "purpose": "Raw peaks at sharp re-entrant edges are mesh-dependent singularities in linear elasticity; "
                       "'away_from_sharp_edges' peaks are an additional view, not a replacement",
        },
        "limitations": [
            "Reporting filter only: full solver domain and restraints are unchanged",
            "Hiding remote stresses does not eliminate boundary influence or prove domain independence",
            "Rock gravity/geostatic stress is not separated from dam-induced stress",
            "Clip-boundary stresses are interpolated; mesh refinement is needed for a more exact 5 m surface",
            "Edge-excluded peaks depend on the chosen exclusion distance and are not a design stress",
        ],
    }
    (output / "stress_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance-m", type=float, default=5.)
    parser.add_argument("--edge-exclusion-m", type=float, default=EDGE_EXCLUSION_M)
    args = parser.parse_args()
    print(json.dumps(export_region(args.result, args.output, args.distance_m, args.edge_exclusion_m), indent=2))
