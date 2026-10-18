"""Run with ParaView's pvpython to preserve Elmer arrays while adding body IDs."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from vtkmodules.vtkCommonCore import vtkIntArray, vtkIdList
from vtkmodules.vtkFiltersCore import vtkExtractCells
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader, vtkXMLUnstructuredGridWriter


def attribute(output: Path, result: Path) -> None:
    with (output / "three_body.msh").open() as stream:
        while next(stream).strip() != "$Nodes":
            pass
        nodes = {}
        for _ in range(int(next(stream))):
            row = next(stream).split()
            nodes[int(row[0])] = tuple(round(float(v), 7) for v in row[1:])
        while next(stream).strip() != "$Elements":
            pass
        bodies = {}
        for _ in range(int(next(stream))):
            row = [int(v) for v in next(stream).split()]
            if row[1] not in (4, 5, 6, 7):
                continue
            key = tuple(sorted(nodes[n] for n in row[3+row[2]:]))
            if key in bodies:
                raise ValueError("Duplicate volume cell")
            bodies[key] = row[3]
    tree = ET.parse(result)
    grid = tree.getroot().find("PUnstructuredGrid")
    if grid is None:
        raise ValueError("Invalid PVTU")
    seen = set()
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
        for i in range(mesh.GetNumberOfCells()):
            ids = mesh.GetCell(i).GetPointIds()
            key = tuple(sorted(
                tuple(round(v - u, 7) for v, u in zip(
                    mesh.GetPoint(ids.GetId(j)),
                    displacement.GetTuple(ids.GetId(j)), strict=True,
                ))
                for j in range(ids.GetNumberOfIds())
            ))
            labels.InsertNextValue(bodies[key])
            seen.add(key)
        mesh.GetCellData().AddArray(labels)
        writer = vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(mesh)
        if writer.Write() != 1:
            raise RuntimeError(f"Failed writing {path}")
    if seen != bodies.keys():
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
