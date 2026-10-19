"""Run with ParaView's pvpython to preserve Elmer arrays while adding body IDs."""
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

from vtkmodules.vtkCommonCore import vtkIntArray, vtkIdList
from vtkmodules.vtkFiltersCore import vtkExtractCells
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader, vtkXMLUnstructuredGridWriter
from vtkmodules.vtkIOParallelXML import vtkXMLPUnstructuredGridWriter


def attribute(output: Path, result: Path) -> None:
    if not result.is_file():
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(result.with_suffix(".vtu")))
        reader.Update()
        writer = vtkXMLPUnstructuredGridWriter()
        writer.SetFileName(str(result))
        writer.SetInputData(reader.GetOutput())
        writer.SetNumberOfPieces(1)
        writer.SetStartPiece(0)
        writer.SetEndPiece(0)
        if writer.Write() != 1 or not result.is_file():
            raise RuntimeError("Failed to wrap serial result in a PVTU")
    with (output / "three_body.msh").open() as stream:
        while next(stream).strip() != "$Nodes":
            pass
        nodes = {}
        reference_nodes = {}
        for _ in range(int(next(stream))):
            row = next(stream).split()
            node = int(row[0])
            location = tuple(round(float(v), 7) for v in row[1:])
            nodes[node] = reference_nodes.setdefault(location, node)
        while next(stream).strip() != "$Elements":
            pass
        bodies = {}
        for _ in range(int(next(stream))):
            row = [int(v) for v in next(stream).split()]
            if row[1] not in (4, 5, 6, 7):
                continue
            members = sorted(nodes[n] for n in row[3+row[2]:])
            key = struct.pack(f"<{len(members)}Q", *members)
            if key in bodies:
                raise ValueError("Duplicate volume cell")
            bodies[key] = row[3]
    del nodes
    tree = ET.parse(result)
    grid = tree.getroot().find("PUnstructuredGrid")
    if grid is None:
        raise ValueError("Invalid PVTU")
    remaining = len(bodies)
    for piece in grid.findall("Piece"):
        path = result.parent / piece.attrib["Source"]
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(path))
        reader.Update()
        mesh = reader.GetOutput()
        volume_ids = vtkIdList()
        for i in range(mesh.GetNumberOfCells()):
            if mesh.GetCell(i).GetCellDimension() == 3:
                volume_ids.InsertNextId(i)
        extract = vtkExtractCells()
        extract.SetInputData(mesh)
        extract.SetCellList(volume_ids)
        extract.Update()
        mesh = extract.GetOutput()
        labels = vtkIntArray()
        labels.SetName("GeometryIds")
        displacement = mesh.GetPointData().GetArray("displacement")
        if displacement is None:
            raise ValueError("Missing solved displacement field")
        point_nodes = [
            reference_nodes[tuple(round(v - u, 7) for v, u in zip(
                mesh.GetPoint(i), displacement.GetTuple(i), strict=True,
            ))]
            for i in range(mesh.GetNumberOfPoints())
        ]
        for i in range(mesh.GetNumberOfCells()):
            ids = mesh.GetCell(i).GetPointIds()
            members = sorted(point_nodes[ids.GetId(j)] for j in range(ids.GetNumberOfIds()))
            key = struct.pack(f"<{len(members)}Q", *members)
            body = bodies[key]
            labels.InsertNextValue(abs(body))
            if body > 0:
                remaining -= 1
                bodies[key] = -body
        mesh.GetCellData().AddArray(labels)
        writer = vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(mesh)
        if writer.Write() != 1:
            raise RuntimeError(f"Failed writing {path}")
    if remaining:
        raise ValueError("Solved output does not cover source volume cells")
    data = grid.find("PCellData")
    if data is None:
        data = ET.SubElement(grid, "PCellData")
    for entry in list(data):
        if entry.attrib.get("Name") == "GeometryIds":
            data.remove(entry)
    ET.SubElement(data, "PDataArray", type="Int32", Name="GeometryIds", NumberOfComponents="1")
    tree.write(result, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    attribute(Path(sys.argv[1]), Path(sys.argv[2]))
