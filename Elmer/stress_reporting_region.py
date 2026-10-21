"""Run with pvpython: retain concrete and a near-structure rock reporting region."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkUnstructuredGrid
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm
from vtkmodules.vtkFiltersCore import vtkAppendFilter, vtkImplicitPolyDataDistance, vtkThreshold
from vtkmodules.vtkFiltersGeneral import vtkClipDataSet
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
from vtkmodules.vtkIOXML import (
    vtkXMLPUnstructuredGridReader, vtkXMLUnstructuredGridReader, vtkXMLUnstructuredGridWriter,
)


DISTANCE_ARRAY = "DistanceToConcrete_m"
BODY_NAMES = {1: "wall", 2: "bedrock"}


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
    if labels is None or set(np.unique(vtk_to_numpy(labels))) != set(BODY_NAMES):
        raise ValueError("Expected attributed wall and bedrock (GeometryIds 1/2)")
    concrete = select_body(mesh, 1, 1)
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


def stress_summary(mesh) -> dict:
    report = {}
    for body, name in BODY_NAMES.items():
        selected = select_body(mesh, body, body)
        if not selected.GetNumberOfCells():
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
        tension = int(np.argmax(principal[:, 2]))
        compression = int(np.argmin(principal[:, 0]))
        coordinates = reference_coordinates(selected)
        report[name] = {
            "reporting_cells": selected.GetNumberOfCells(),
            "peak_tension_mpa": max(0., float(principal[tension, 2])) / 1e6,
            "peak_compression_mpa": max(0., -float(principal[compression, 0])) / 1e6,
            "tension_reference_xyz_m": coordinates[tension].tolist(),
            "compression_reference_xyz_m": coordinates[compression].tolist(),
        }
    return report


def export_region(result: Path, output: Path, distance_m: float = 5.) -> dict:
    if not result.is_file():
        raise FileNotFoundError(result)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty reporting output directory")
    reader = vtkXMLPUnstructuredGridReader() if result.suffix.lower() == ".pvtu" else vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(result))
    reader.Update()
    source = reader.GetOutput()
    region = reporting_region(source, distance_m)
    summary = stress_summary(region)
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
    if saved.GetNumberOfCells() != region.GetNumberOfCells() or stress_summary(saved) != summary:
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
        "selection": "All wall; rock within nearest concrete-surface distance in undeformed coordinates",
        "boundary_approximation": "Exact surface distance at rock vertices; linearly interpolated within cells, clipped at threshold",
        "stress_method": "Principal eigenvalues of recovered nodal stress tensor; interpolated tensors at clip intersections",
        "full_domain_cells": source.GetNumberOfCells(),
        "maximum_exported_distance_m": float(
            vtk_to_numpy(saved.GetPointData().GetArray(DISTANCE_ARRAY)).max()
        ),
        "bodies": summary,
        "limitations": [
            "Reporting filter only: full solver domain and restraints are unchanged",
            "Hiding remote stresses does not eliminate boundary influence or prove domain independence",
            "Rock gravity/geostatic stress is not separated from dam-induced stress",
            "Clip-boundary stresses are interpolated; mesh refinement is needed for a more exact 5 m surface",
        ],
    }
    (output / "stress_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance-m", type=float, default=5.)
    args = parser.parse_args()
    print(json.dumps(export_region(args.result, args.output, args.distance_m), indent=2))
