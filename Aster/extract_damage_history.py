"""Export all native benchmark steps to ParaView using the bundled MED runtime."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from extract_result_fields import ml, np, stress_scalars


def export(result_path: Path, output: Path,
           labels: tuple[str, ...] = ("INTACT", "SEEDED", "LOCAL", "FREE", "NEAR", "NEARLOC")) -> dict:
    meshes = ml.GetMeshNames(str(result_path))
    if len(meshes) != 1:
        raise ValueError("Damage benchmark must contain exactly one result mesh")
    mesh_name = meshes[0]
    names = ml.GetAllFieldNames(str(result_path))
    report = {}
    for label in labels:
        fields = {}
        for suffix in ("DEPL", "SIGM_NOEU", "VARI_NOEU"):
            matches = [name for name in names if name.startswith(label + "_") and name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous {label} {suffix} in MED")
            fields[suffix] = matches[0]
        iterations = ml.GetAllFieldIterations(str(result_path), fields["VARI_NOEU"])
        pvd = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
        collection = ET.SubElement(pvd, "Collection")
        snapshots = []
        for iteration, order, expected_time in iterations:
            loaded = {
                suffix: ml.ReadFieldNode(str(result_path), mesh_name, 0, name, iteration, order)
                for suffix, name in fields.items()
            }
            reference = loaded["VARI_NOEU"]
            for field in loaded.values():
                if not np.array_equal(field.getMesh().getCoords().toNumPyArray(),
                                      reference.getMesh().getCoords().toNumPyArray()):
                    raise ValueError("Native damage, stress and displacement node identities differ")
                if not np.array_equal(field.getMesh().getNodalConnectivity().toNumPyArray(),
                                      reference.getMesh().getNodalConnectivity().toNumPyArray()):
                    raise ValueError("Native damage, stress and displacement cell identities differ")
            def components(field, requested):
                array = field.getArray()
                labels = [item.split()[0] for item in array.getInfoOnComponents()]
                values = np.asarray(array.toNumPyArray())
                if not set(requested).issubset(labels):
                    raise ValueError(f"Missing result components {requested}: {labels}")
                result = values[:, [labels.index(name) for name in requested]]
                if not np.isfinite(result).all():
                    raise ValueError("Native damage benchmark contains nonfinite output fields")
                return result
            damage = components(reference, ["V1"])
            displacement = components(loaded["DEPL"], ["DX", "DY", "DZ"])
            stress = components(loaded["SIGM_NOEU"], ["SIXX", "SIYY", "SIZZ", "SIXY", "SIXZ", "SIYZ"])
            values = {
                "damage_recovered_nodal": damage, "displacement_m": displacement,
                **{name: data.reshape(-1, 1) for name, data in stress_scalars(stress).items()},
            }
            vtk_fields = []
            for name, array in values.items():
                field = ml.MEDCouplingFieldDouble(ml.ON_NODES, ml.ONE_TIME)
                field.setMesh(reference.getMesh())
                field.setName(name)
                field.setArray(ml.DataArrayDouble(array.copy()))
                field.setTime(*reference.getTime())
                field.checkConsistencyLight()
                vtk_fields.append(field)
            stem = output / f"{label.lower()}_{iteration:04d}_{order:04d}"
            written = Path(ml.MEDCouplingFieldDouble.WriteVTK(str(stem), vtk_fields))
            if not written.is_file() or written.stat().st_size == 0:
                raise RuntimeError("Native damage VTU export did not produce a nonempty file")
            instant = float(reference.getTime()[0])
            if not np.isclose(instant, expected_time, rtol=0., atol=1e-12):
                raise ValueError("MED iteration time does not match the loaded damage field")
            ET.SubElement(collection, "DataSet", timestep=str(instant), part="0", file=written.name)
            snapshots.append({"time": instant, "vtu": written.name,
                              "damage_min": float(damage.min()), "damage_max": float(damage.max())})
        if not snapshots:
            raise ValueError(f"No converged output steps in {label}")
        pvd_path = output / f"{label.lower()}.pvd"
        ET.ElementTree(pvd).write(pvd_path, encoding="utf-8", xml_declaration=True)
        report[label] = {
            "pvd": pvd_path.name, "snapshots": snapshots,
            "damage_field": "damage_recovered_nodal",
            "damage_association": "nodes",
            "damage_recovery": "native_VARI_NOEU_unclipped",
        }
    (output / "damage_series.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_path", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--labels", nargs="+",
                        default=("INTACT", "SEEDED", "LOCAL", "FREE", "NEAR", "NEARLOC"))
    args = parser.parse_args()
    export(args.result_path, args.output, tuple(args.labels))
