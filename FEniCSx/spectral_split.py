"""Experimental principal-strain split for affine P1 tetrahedral mechanics.

Negative principal strains and negative volumetric strain remain undegraded.
This energy split is not crack-face contact, friction or concrete crushing.
"""
from __future__ import annotations

from dolfinx import fem
from mpi4py import MPI
import numpy as np
import ufl

from phase_field_solver import degradation, strain


def positive_spectrum(tensors: np.ndarray):
    """Return eigenvectors, positive eigenvalues and divided differences.

    The divided differences are the derivative of the positive matrix part
    in its eigenbasis. At repeated zero eigenvalues use the tensile-side
    generalized derivative, consistent with the volumetric tangent.
    """
    values, vectors = np.linalg.eigh(tensors)
    positive = np.maximum(values, 0.0)
    difference = values[..., :, None] - values[..., None, :]
    scale = np.max(np.abs(values), axis=-1)[..., None, None]
    repeated = np.abs(difference) <= 32 * np.finfo(float).eps * np.maximum(
        scale, np.finfo(float).tiny,
    )
    midpoint = 0.5 * (values[..., :, None] + values[..., None, :])
    derivative = np.empty_like(difference)
    np.divide(
        positive[..., :, None] - positive[..., None, :],
        difference, out=derivative, where=~repeated,
    )
    derivative[repeated] = (midpoint[repeated] >= 0).astype(float)
    return vectors, positive, derivative


class SpectralSplit:
    """DG0 spectral coefficients; exact cellwise strain for affine P1 tets."""

    def __init__(self, displacement, damage, bulk: float, mu: float, kappa: float):
        domain = displacement.function_space.mesh
        if domain.topology.cell_name() != "tetrahedron" or domain.geometry.cmaps[0].degree != 1:
            raise ValueError("Experimental spectral mechanics requires affine tetrahedral geometry")
        element = displacement.function_space.ufl_element()
        if element.degree != 1:
            raise ValueError("Experimental spectral mechanics requires P1 displacement")
        self.displacement = displacement
        self.lam, self.mu = bulk - 2 * mu / 3, mu
        if self.lam < 0:
            raise ValueError("Experimental spectral energy requires a nonnegative first Lame parameter")
        self.factor = degradation(damage, kappa)
        tensor_space = fem.functionspace(domain, ("DG", 0, (3, 3)))
        vector_space = fem.functionspace(domain, ("DG", 0, (3,)))
        self.strain = fem.Function(tensor_space)
        self.basis = fem.Function(tensor_space)
        self.divided_difference = fem.Function(tensor_space)
        self.positive_values = fem.Function(vector_space)
        self.strain_expression = fem.Expression(
            strain(displacement), tensor_space.element.interpolation_points,
        )
        self.tensor_dofs = tensor_space.dofmap.list[:, 0]
        self.vector_dofs = vector_space.dofmap.list[:, 0]
        self.update()

    def update(self):
        self.strain.interpolate(self.strain_expression)
        tensors = self.strain.x.array.reshape(-1, 3, 3)[self.tensor_dofs]
        if self.displacement.function_space.mesh.comm.allreduce(
            bool(np.any(~np.isfinite(tensors))), op=MPI.LOR,
        ):
            raise RuntimeError("Experimental spectral mechanics encountered non-finite strain")
        basis, positive, derivative = positive_spectrum(tensors)
        self.basis.x.array.reshape(-1, 3, 3)[self.tensor_dofs] = basis
        self.divided_difference.x.array.reshape(-1, 3, 3)[self.tensor_dofs] = derivative
        self.positive_values.x.array.reshape(-1, 3)[self.vector_dofs] = positive

    @property
    def positive_strain(self):
        return self.basis * ufl.diag(self.positive_values) * self.basis.T

    @property
    def tensile_energy(self):
        eps = strain(self.displacement)
        positive = self.positive_strain
        return (
            0.5 * self.lam * ufl.max_value(ufl.tr(eps), 0)**2
            + self.mu * ufl.inner(positive, positive)
        )

    @property
    def energy(self):
        eps = strain(self.displacement)
        negative = eps - self.positive_strain
        return (
            self.factor * self.tensile_energy
            + 0.5 * self.lam * ufl.min_value(ufl.tr(eps), 0)**2
            + self.mu * ufl.inner(negative, negative)
        )

    @property
    def stress(self):
        eps = strain(self.displacement)
        trace = ufl.tr(eps)
        return (
            2 * self.mu * (eps + (self.factor - 1) * self.positive_strain)
            + self.lam * (
                self.factor * ufl.max_value(trace, 0) + ufl.min_value(trace, 0)
            ) * ufl.Identity(3)
        )

    def tangent_stress(self, trial):
        increment = strain(trial)
        local = self.basis.T * increment * self.basis
        positive_derivative = self.basis * ufl.as_matrix([
            [self.divided_difference[i, j] * local[i, j] for j in range(3)]
            for i in range(3)
        ]) * self.basis.T
        return (
            2 * self.mu * (increment + (self.factor - 1) * positive_derivative)
            + self.lam * ufl.conditional(
                ufl.ge(ufl.tr(strain(self.displacement)), 0), self.factor, 1.0,
            ) * ufl.tr(increment) * ufl.Identity(3)
        )
