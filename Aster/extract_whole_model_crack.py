"""Export the whole-model crack MED result to a ParaView series (bundled MED runtime).

Damage exists only in the quadratic wall band, and stress only on volume nodes;
both are placed on the full mesh so one series shows the wall, foundation and
bedrock response together. Damage is zero outside the band by construction.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_result_fields import ml, np, stress_scalars  # noqa: E402


def _field_name(names, suffix):
    matches = [name for name in names if name.startswith("CRACK") and name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous CRACK {suffix} field in MED: {names}")
    return matches[0]


def _node_values(path, name, mesh_file, node_count, components):
    """Every time step of a node field, scattered to the full node numbering."""
    series = ml.MEDFileFieldMultiTS.New(str(path), name, True)
    steps = []
    for iteration, order, instant in series.getTimeSteps():
        step = series.getTimeStep(iteration, order)
        values, profile = step.getFieldWithProfile(ml.ON_NODES, 0, mesh_file)
        labels = [item.split()[0] for item in values.getInfoOnComponents()]
        if not set(components).issubset(labels):
            raise ValueError(f"{name} lacks components {components}: {labels}")
        data = np.asarray(values.toNumPyArray()).reshape(len(values), -1)
        data = data[:, [labels.index(component) for component in components]]
        ids = (np.asarray(profile.toNumPyArray(), dtype=np.int64)
               if profile is not None and len(profile) else np.arange(len(data)))
        if len(ids) != len(data) or (len(ids) and ids.max() >= node_count):
            raise ValueError(f"{name} profile does not match the result mesh")
        full = np.full((node_count, len(components)), np.nan)
        full[ids] = data
        steps.append((float(instant), full, ids))
    return steps


def export(result_path: Path, output: Path, label: str = "CRACK") -> dict:
    meshes = ml.GetMeshNames(str(result_path))
    if len(meshes) != 1:
        raise ValueError("The crack result must contain exactly one mesh")
    mesh_file = ml.MEDFileUMesh.New(str(result_path), meshes[0])
    mesh = mesh_file.getMeshAtLevel(0)
    node_count = mesh.getNumberOfNodes()
    names = ml.GetAllFieldNames(str(result_path))
    displacement = _node_values(result_path, _field_name(names, "DEPL"), mesh_file, node_count,
                                ["DX", "DY", "DZ"])
    stress = _node_values(result_path, _field_name(names, "SIGM_NOEU"), mesh_file, node_count,
                          ["SIXX", "SIYY", "SIZZ", "SIXY", "SIXZ", "SIYZ"])
    damage = _node_values(result_path, _field_name(names, "VARI_NOEU"), mesh_file, node_count, ["V1"])
    times = [step[0] for step in displacement]
    if [step[0] for step in stress] != times or [step[0] for step in damage] != times:
        raise ValueError("Displacement, stress and damage have different time steps")
    pvd = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(pvd, "Collection")
    snapshots = []
    for index, instant in enumerate(times):
        moved = displacement[index][1]
        if not np.isfinite(moved).all():
            raise ValueError("Displacement is missing or nonfinite on some nodes")
        tensor = stress[index][1]
        known = np.isfinite(tensor).all(1)
        scalars = {name: np.full(node_count, np.nan) for name in ("principal_max_pa", "von_mises_pa")}
        computed = stress_scalars(tensor[known])
        for name in scalars:
            scalars[name][known] = computed[name]
        band_damage = damage[index][1][:, 0]
        in_band = np.isfinite(band_damage)
        if not np.isfinite(band_damage[in_band]).all():
            raise ValueError("Nonfinite recovered damage")
        band_damage = np.where(in_band, band_damage, 0.0)
        values = {"displacement_m": moved, "damage_recovered_nodal": band_damage.reshape(-1, 1),
                  **{name: data.reshape(-1, 1) for name, data in scalars.items()}}
        fields = []
        for name, array in values.items():
            field = ml.MEDCouplingFieldDouble(ml.ON_NODES, ml.ONE_TIME)
            field.setMesh(mesh)
            field.setName(name)
            field.setArray(ml.DataArrayDouble(np.ascontiguousarray(array, dtype=np.float64)))
            field.setTime(instant, index, 0)
            field.checkConsistencyLight()
            fields.append(field)
        stem = output / f"{label.lower()}_{index:04d}"
        written = Path(ml.MEDCouplingFieldDouble.WriteVTK(str(stem), fields))
        if not written.is_file() or written.stat().st_size == 0:
            raise RuntimeError("Whole-model crack VTU export produced no file")
        ET.SubElement(collection, "DataSet", timestep=repr(instant), part="0", file=written.name)
        snapshots.append({"time": instant, "vtu": written.name,
                          "damage_min": float(band_damage[in_band].min()),
                          "damage_max": float(band_damage[in_band].max()),
                          "maximum_principal_stress_pa": float(np.nanmax(scalars["principal_max_pa"]))})
    pvd_path = output / f"{label.lower()}.pvd"
    ET.ElementTree(pvd).write(pvd_path, encoding="utf-8", xml_declaration=True)
    report = {label: {"pvd": pvd_path.name, "snapshots": snapshots, "damage_field": "damage_recovered_nodal",
                      "damage_association": "nodes", "damage_recovery": "native_VARI_NOEU_unclipped",
                      "damage_outside_band": "zero (linear elastic cells carry no damage)"}}
    (output / "damage_series.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract_whole_model_crack.py <result.med> <output directory>")
    print(json.dumps(export(Path(sys.argv[1]), Path(sys.argv[2]))))
