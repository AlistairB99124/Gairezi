"""Fixed-head drained crack equilibrium with exact free-surface clipping.

Reservoir access is proved on released, open lip pairs, not on all potential
wall facets. Reference geometry and prescribed head give conservative pressure
work. This is instantaneous drained equilibrium, not transient filling.
"""
import numpy as np

if __package__:
    from .fracture_faces import PairedCrackFaces, bonded_node_representatives
    from .fracture_steady_flow import HydrostaticEquilibriumHydraulics, SteadyCrackLoad
    from .hydrostatic_load import triangle_pressure_weights
else:
    from fracture_faces import PairedCrackFaces, bonded_node_representatives
    from fracture_steady_flow import HydrostaticEquilibriumHydraulics, SteadyCrackLoad
    from hydrostatic_load import triangle_pressure_weights


class ReservoirAccessError(ValueError):
    """A mechanically equilibrated pressure trial has no admissible water path."""


class ReservoirCrackHydraulics(HydrostaticEquilibriumHydraulics):
    """One reservoir head per released cut, including partially wet triangles."""

    def trial_load(self, faces, displacement_m, *, bonded_face_mask=None):
        ids = self.face_ids(faces)
        if bonded_face_mask is not None:
            bonded_node_representatives(faces, bonded_face_mask)
            if np.asarray(bonded_face_mask)[ids].any():
                raise ValueError("Bonded wall faces cannot receive internal water pressure")
        if (self.parent_node_ids.shape != (len(faces.coordinates),)
                or not np.array_equal(self.parent_node_ids[faces.minus[ids]],
                                      self.parent_node_ids[faces.plus[ids]])):
            raise ValueError("Reservoir crack lips require matching source identities")
        head = self.boundaries[0].elevation_m
        points = faces.coordinates[faces.minus[ids]]
        gradient = self.water_density_kg_m3 * self.gravity_m_s2
        pressure = gradient * np.maximum(head - points[:, :, 2], 0.)
        forces = np.zeros_like(faces.coordinates)
        weights = np.array([triangle_pressure_weights(p, head, gradient) for p in points])
        vectors = weights[:, :, None] * faces.normals[ids, None, :]
        np.add.at(forces, faces.plus[ids], vectors)
        np.add.at(forces, faces.minus[ids], -vectors)
        return SteadyCrackLoad(np.full_like(pressure, head), pressure, forces, {
            "status": "unaccepted_reservoir_connected_clipped_hydrostatic_load",
            "solver_ready": False, "engineering_validated": False,
            "final_aperture_access_audit_pending": True,
            "mechanical_inventory_face_ids": ids.tolist(),
            "boundary_elevations_m": {b.name: b.elevation_m for b in self.boundaries},
            "water_model": "instantaneous drained equilibrium; exact submerged P1 traction, no filling kinetics",
        })

    def load(self, faces, displacement_m, *, bonded_face_mask=None):
        load = self.trial_load(faces, displacement_m, bonded_face_mask=bonded_face_mask)
        ids = self.face_ids(faces)
        representatives = (np.arange(len(faces.coordinates)) if bonded_face_mask is None
                           else bonded_node_representatives(faces, bonded_face_mask))
        displacement = np.asarray(displacement_m)
        if (displacement.shape != faces.coordinates.shape or not np.isfinite(displacement).all()
                or np.max(np.abs(displacement - displacement[representatives])) > self.closure_tolerance_m):
            raise ValueError("Water access requires converged finite displacement and full vector bonds")
        subset = PairedCrackFaces(
            faces.coordinates, representatives[faces.minus[ids]],
            representatives[faces.plus[ids]], faces.normals[ids])
        gaps = np.asarray(subset.measure(displacement)["signed_vertex_gaps_m"])
        if np.min(gaps) < -self.closure_tolerance_m:
            raise ValueError("Water access requires nonpenetrating crack contact")
        head = self.boundaries[0].elevation_m
        heights = subset.coordinates[subset.minus, 2]
        wet = heights.min(axis=1) < head
        owners = {}
        for face, (minus, plus) in enumerate(zip(subset.minus, subset.plus)):
            for i, j in ((0, 1), (1, 2), (2, 0)):
                key = tuple(sorted(tuple(sorted((int(minus[k]), int(plus[k])))) for k in (i, j)))
                source = tuple(sorted(map(int, self.parent_node_ids[minus[[i, j]]])))
                z = heights[face, [i, j]]
                aperture = gaps[face, [i, j]]
                # On a partially submerged edge evaluate aperture at the
                # waterline intersection, not at its dry endpoint.
                if z.min() >= head:
                    passing = False
                else:
                    values = list(aperture[z < head])
                    if z.min() < head < z.max():
                        t = (head - z[0]) / (z[1] - z[0])
                        values.append(float(aperture[0] + t * (aperture[1] - aperture[0])))
                    passing = max(values) > self.closure_tolerance_m
                owners.setdefault(key, []).append((face, passing, source))
        adjacency = [set() for _ in ids]
        seeds = set()
        inlets = {tuple(sorted(map(int, edge)))
                  for b in self.boundaries for edge in b.inlet_source_edges}
        matched = set()
        for entries in owners.values():
            passing = {face for face, open_edge, _ in entries if open_edge}
            for face in passing:
                adjacency[face].update(passing - {face})
            if len(entries) == 1 and entries[0][2] in inlets:
                if entries[0][2] in matched:
                    raise ValueError("Reservoir inlet matches multiple unrelated split crack edges")
                matched.add(entries[0][2])
                if entries[0][1]:
                    seeds.add(entries[0][0])
        if matched != inlets:
            raise ValueError("Reservoir inlet must be an unambiguous exterior edge of the released crack")
        reached = set(seeds)
        pending = list(seeds)
        while pending:
            face = pending.pop()
            for other in adjacency[face] - reached:
                reached.add(other)
                pending.append(other)
        missing = set(map(int, np.flatnonzero(wet))) - reached
        if missing:
            raise ReservoirAccessError(
                "Pressurised faces have no open submerged reservoir path; filling/trapped contact is unresolved")
        load.report.update({
            "status": "accepted_reservoir_connected_clipped_hydrostatic_load",
            "final_aperture_access_audit_pending": False,
            "pressurised_face_ids": np.flatnonzero(wet).tolist(),
            "above_water_face_ids": np.flatnonzero(~wet).tolist(),
            "reachable_open_face_ids": sorted(reached),
            "tied_front_vertex_ids": np.unique(subset.minus[subset.minus == subset.plus]).tolist(),
            "partially_submerged_face_ids": np.flatnonzero(
                wet & (heights.max(axis=1) > head)).tolist(),
        })
        return load
