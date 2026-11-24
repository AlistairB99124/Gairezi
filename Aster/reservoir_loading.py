"""Exact nodal hydrostatic loads for moving reservoir surfaces."""
from __future__ import annotations

import numpy as np
from Aster.hydraulic_loads import boundary_faces, triangle_pressure_forces


class ReservoirPressure:
    def __init__(self, arrays: dict, gradient: float):
        if not np.isfinite(gradient) or gradient <= 0:
            raise ValueError("Reservoir pressure gradient must be finite and positive")
        self.nodes = arrays["nodes"]
        self.gradient = gradient
        source = arrays["source_node_ids"]
        volumes = np.vstack((arrays["tetra4_concrete"], arrays["tetra4_bedrock"]))
        boundary, normals = boundary_faces(volumes, self.nodes, source)
        lookup = {tuple(sorted(face)): normal for face, normal in zip(source[boundary], normals)}
        self.surfaces = {}
        for name in ("tri3_upstream", "tri3_downstream", "tri3_crest"):
            faces = arrays[name]
            try:
                outward = np.asarray([lookup[tuple(sorted(face))] for face in source[faces]]).reshape(-1, 3)
            except KeyError as error:
                raise ValueError(f"{name} is not an exterior volume surface") from error
            self.surfaces[name] = faces, outward

    def forces(self, surface_z: float, groups: tuple[str, ...]) -> np.ndarray:
        if not np.isfinite(surface_z):
            raise ValueError("Water surface elevation must be finite")
        forces = np.zeros_like(self.nodes)
        for group in groups:
            faces, normals = self.surfaces[group]
            for face, normal in zip(faces, normals):
                weights = triangle_pressure_forces(
                    self.nodes[face], surface_z - self.nodes[face, 2]) * self.gradient
                np.add.at(forces, face, -weights[:, None] * normal)
        return forces
