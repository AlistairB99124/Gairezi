"""Exact consistent P1 loads on the submerged part of an affine triangle."""
import numpy as np


def triangle_pressure_weights(points, elevation, pressure_gradient):
    """Return integral N_i rho*g*max(head-z, 0) dA in triangle vertex order."""
    xyz = np.asarray(points)
    if xyz.dtype.kind not in "fiu" or xyz.shape != (3, 3) or not np.isfinite(xyz).all():
        raise ValueError("Pressure triangle must contain three finite real 3-D vertices")
    for name, value in (("elevation", elevation), ("pressure_gradient", pressure_gradient)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
    if pressure_gradient < 0:
        raise ValueError("pressure_gradient must be nonnegative")
    xyz = xyz.astype(float)
    area_scale = np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0]))
    if not np.isfinite(area_scale) or area_scale <= 0:
        raise ValueError("Pressure triangle must have finite positive area")
    # Clip in original barycentric coordinates so no parent shape functions
    # are lost when the submerged polygon is triangulated.
    polygon = []
    for first, second in zip(np.eye(3), np.roll(np.eye(3), -1, axis=0)):
        a = float(elevation - first @ xyz[:, 2])
        b = float(elevation - second @ xyz[:, 2])
        if a >= 0:
            polygon.append(first)
        if (a < 0 < b) or (b < 0 < a):
            polygon.append(first + a / (a - b) * (second - first))
    result = np.zeros(3)
    quadrature = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))
    for index in range(1, len(polygon) - 1):
        barycentric = np.asarray((polygon[0], polygon[index], polygon[index + 1]))
        triangle = barycentric @ xyz
        area = np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) / 2
        samples = quadrature @ barycentric
        pressure = pressure_gradient * np.maximum(elevation - samples @ xyz[:, 2], 0.)
        result += area / 3 * (samples.T @ pressure)
    if not np.isfinite(result).all():
        raise ValueError("Integrated pressure forces must be finite")
    return result


def assemble_water_forces(imported, space, gradient, elevations, constant_pressures):
    """Serial native blocked-P1 nodal forces; identity comes from connectivity."""
    from dolfinx import mesh

    domain = imported.domain
    element = space.ufl_element()
    if (space.mesh is not domain or domain.comm.size != 1
            or domain.topology.cell_name() != "tetrahedron" or domain.geometry.cmaps[0].degree != 1
            or element.degree != 1 or element.discontinuous or element.family_name != "P"
            or space.dofmap.index_map_bs != 3 or tuple(space.value_shape) != (3,)):
        raise ValueError("Exact water forces require serial continuous blocked vector P1 affine tetrahedra")
    if set(elevations) & set(constant_pressures):
        raise ValueError("A boundary cannot have both hydrostatic and constant pressure")
    for name, value, nonnegative in (
        ("pressure_gradient", gradient, True),
        *[(f"elevation[{tag}]", head, False) for tag, head in elevations.items()],
        *[(f"pressure[{tag}]", pressure, True) for tag, pressure in constant_pressures.items()],
    ):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or (nonnegative and value < 0)):
            raise ValueError(f"{name} must be finite real; pressure and gradient must be nonnegative")
    force = np.zeros((space.dofmap.index_map.size_local, 3))
    domain.topology.create_connectivity(2, 3)
    owners = domain.topology.connectivity(2, 3)
    for tag in sorted(set(elevations) | set(constant_pressures)):
        facets = imported.facet_tags.find(tag)
        geometry = mesh.entities_to_geometry(domain, 2, facets)
        for facet, nodes in zip(facets, geometry):
            cells = owners.links(int(facet))
            if len(cells) != 1:
                raise ValueError("Water pressure tags must contain exterior facets only")
            cell = int(cells[0])
            cell_nodes = domain.geometry.dofmaps[0][cell]
            local = np.array([np.flatnonzero(cell_nodes == node).item() for node in nodes])
            dofs = space.dofmap.list[cell][local]
            points = domain.geometry.x[nodes]
            cross = np.cross(points[1] - points[0], points[2] - points[0])
            magnitude = np.linalg.norm(cross)
            if not np.isfinite(magnitude) or magnitude <= 0:
                raise ValueError("Pressure facet must have finite positive area")
            normal = cross / magnitude
            if np.dot(normal, points.mean(axis=0) - domain.geometry.x[cell_nodes].mean(axis=0)) < 0:
                normal = -normal
            if tag in elevations:
                weights = triangle_pressure_weights(points, elevations[tag], gradient)
            else:
                pressure = constant_pressures[tag]
                weights = np.full(3, pressure * magnitude / 6)
            np.add.at(force, dofs, -weights[:, None] * normal)
    if not np.isfinite(force).all():
        raise ValueError("Water forces must be finite")
    return force.ravel()
