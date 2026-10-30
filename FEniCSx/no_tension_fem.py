"""Cellwise isotropic/no-tension coefficients for affine-P1 verification.

The selection is in LOCAL cell order (owned and ghosts), not catalog/source
element IDs. Material inputs are scalars, local cell arrays or same-mesh scalar
DG0 Functions. Production source/refinement mapping is deliberately not inferred.
"""
import numbers

from dolfinx import fem
from mpi4py import MPI
import numpy as np
import ufl

if __package__:
    from .no_tension import elastic_constants, tensile_strength_response
else:
    from no_tension import elastic_constants, tensile_strength_response


class NoTensionMaterial:
    """Isotropic cells; selected cells become damaged concrete when ``selected`` is set.

    ``young`` is the intact modulus of every cell. While a cell is selected it uses
    ``selected_young_pa`` (when given) and the reduced ``tensile_strength_pa`` cap; the
    moduli follow the current ``selected`` mask, so an intact solve with the mask
    cleared keeps the intact stiffness everywhere.
    """

    def __init__(self, displacement, young, poisson, selected_cells, tensile_strength_pa=0.,
                 selected_young_pa=None):
        domain = displacement.function_space.mesh
        if (domain.topology.cell_name() != "tetrahedron" or domain.geometry.cmaps[0].degree != 1
                or displacement.function_space.ufl_element().degree != 1
                or displacement.function_space.ufl_element().family_name != "P"
                or displacement.function_space.ufl_element().discontinuous
                or tuple(displacement.ufl_shape) != (3,)
                or displacement.function_space.dofmap.index_map_bs != 3):
            raise ValueError("No-tension verification requires a continuous blocked vector P1 on affine tetrahedra")
        self.displacement = displacement
        if (isinstance(tensile_strength_pa, bool) or not isinstance(tensile_strength_pa, numbers.Real)
                or not np.isfinite(tensile_strength_pa) or tensile_strength_pa < 0):
            raise ValueError("Selected tensile strength must be finite and nonnegative")
        self.tensile_strength_pa = float(tensile_strength_pa)
        if selected_young_pa is not None and (
                isinstance(selected_young_pa, bool) or not isinstance(selected_young_pa, numbers.Real)
                or not np.isfinite(selected_young_pa) or selected_young_pa <= 0):
            raise ValueError("Selected Young's modulus must be finite and positive")
        self.selected_young_pa = None if selected_young_pa is None else float(selected_young_pa)
        tensor_space = fem.functionspace(domain, ("DG", 0, (3, 3)))
        scalar_space = fem.functionspace(domain, ("DG", 0))
        self.strain_field = fem.Function(tensor_space)
        self.stress = fem.Function(tensor_space, name="stress_pa")
        self.relaxed_strain = fem.Function(tensor_space, name="relaxed_strain_not_opening")
        self.basis = fem.Function(tensor_space)
        self.normal = fem.Function(tensor_space)
        self.shear = fem.Function(tensor_space)
        self.energy = fem.Function(scalar_space, name="elastic_energy_j_m3")
        self.tensor_dofs = tensor_space.dofmap.list[:, 0]
        self.scalar_dofs = scalar_space.dofmap.list[:, 0]
        self.young = self._cell_values(young, "Young's modulus", domain)
        self.poisson = self._cell_values(poisson, "Poisson ratio", domain)
        self.mu = fem.Function(scalar_space, name="shear_modulus_pa")
        self.lam = fem.Function(scalar_space, name="lame_lambda_pa")
        self.selected = np.asarray(selected_cells)
        if self.selected.dtype != np.dtype(bool) or self.selected.shape != self.tensor_dofs.shape:
            raise ValueError("Selection must be a boolean mask in local cell order including ghosts")
        self.selected = self.selected.copy()
        self._moduli_selection = None
        self._refresh_moduli()
        self.expression = fem.Expression(ufl.sym(ufl.grad(displacement)),
                                         tensor_space.element.interpolation_points)
        self.update()

    def _cell_values(self, value, name, domain):
        if isinstance(value, fem.Function):
            space = value.function_space
            element = space.ufl_element()
            if (space.mesh is not domain or element.degree != 0 or not element.discontinuous
                    or tuple(value.ufl_shape) != ()):
                raise ValueError(f"{name} Function must be scalar DG0 on the displacement mesh")
            raw = value.x.array[space.dofmap.list[:, 0]]
        else:
            raw = np.asarray(value)
        if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
            raise ValueError(f"{name} must be finite real scalar or local-cell values")
        if raw.ndim == 0:
            return np.full(len(self.tensor_dofs), float(raw))
        if raw.shape != self.tensor_dofs.shape:
            raise ValueError(f"{name} must match local cell order including ghosts")
        return raw.astype(float).copy()

    def effective_young(self):
        """Young's modulus of each cell under the current selection."""
        if self.selected_young_pa is None:
            return self.young
        return np.where(self.selected, self.selected_young_pa, self.young)

    def _refresh_moduli(self):
        if self._moduli_selection is not None and np.array_equal(self._moduli_selection, self.selected):
            return
        young = self.effective_young()
        self.material_groups = []
        mu = np.empty(len(self.tensor_dofs))
        lam = np.empty_like(mu)
        for pair in np.unique(np.column_stack((young, self.poisson)), axis=0):
            young_value, poisson_value = map(float, pair)
            shear, lame = elastic_constants(young_value, poisson_value)
            members = (young == young_value) & (self.poisson == poisson_value)
            mu[members], lam[members] = shear, lame
            self.material_groups.append((members, young_value, poisson_value))
        self.mu_values, self.lam_values = mu, lam
        for field, values in ((self.mu, mu), (self.lam, lam)):
            field.x.array[self.scalar_dofs] = values
            field.x.scatter_forward()
        self._moduli_selection = self.selected.copy()

    def update(self):
        self._refresh_moduli()
        self.strain_field.interpolate(self.expression)
        strain = self.strain_field.x.array.reshape(-1, 3, 3)[self.tensor_dofs]
        domain = self.displacement.function_space.mesh
        if domain.comm.allreduce(bool(np.any(~np.isfinite(strain))), op=MPI.LOR):
            raise RuntimeError("No-tension material encountered non-finite strain")
        stress = 2 * self.mu_values[:, None, None] * strain + self.lam_values[:, None, None] * np.trace(
            strain, axis1=-2, axis2=-1,
        )[:, None, None] * np.eye(3)
        relaxed = np.zeros_like(stress)
        basis = np.broadcast_to(np.eye(3), strain.shape).copy()
        normal = 2 * self.mu_values[:, None, None] * np.eye(3) + self.lam_values[:, None, None]
        shear = np.broadcast_to(2 * self.mu_values[:, None, None], strain.shape).copy()
        energy = .5 * np.einsum("nij,nij->n", stress, strain)
        for members, young, poisson in self.material_groups:
            active = members & self.selected
            if not np.any(active):
                continue
            result = tensile_strength_response(
                strain[active], young, poisson, self.tensile_strength_pa,
            )
            stress[active] = result.stress_pa
            relaxed[active] = result.relaxed_strain
            normal[active] = result.normal_tangent_pa
            shear[active] = result.shear_tangent_pa
            basis[active] = result.basis
            energy[active] = result.energy_j_m3
        for field, values in ((self.stress, stress), (self.relaxed_strain, relaxed),
                              (self.basis, basis), (self.normal, normal), (self.shear, shear)):
            field.x.array.reshape(-1, 3, 3)[self.tensor_dofs] = values
            field.x.scatter_forward()
        self.energy.x.array[self.scalar_dofs] = energy
        self.energy.x.scatter_forward()

    def tangent_stress(self, direction):
        local = self.basis.T * ufl.sym(ufl.grad(direction)) * self.basis
        derivative = ufl.as_matrix([
            [sum(self.normal[i, k] * local[k, k] for k in range(3)) if i == j
             else self.shear[i, j] * local[i, j] for j in range(3)]
            for i in range(3)
        ])
        return self.basis * derivative * self.basis.T
