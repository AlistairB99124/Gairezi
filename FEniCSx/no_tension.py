"""Experimental relaxed elastic no-tension material; not a fracture/contact law.

Minimise 1/2 (eps - eta):C:(eps - eta) over positive-semidefinite eta.
The relaxed strain eta accommodates tension without stress. Compression keeps
the original isotropic compliance. eta is a strain, NOT a crack opening.
This reversible local law contains no friction, crushing, fracture energy,
irreversible propagation or hydraulics and must not be used as their substitute.
"""
from dataclasses import dataclass
import numbers

import numpy as np


def elastic_constants(young: float, poisson: float) -> tuple[float, float]:
    for name, value in (("Young's modulus", young), ("Poisson ratio", poisson)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
    if young <= 0 or not -1 < poisson < .5:
        raise ValueError("Require Young's modulus > 0 and -1 < Poisson ratio < 0.5")
    return young / (2 * (1 + poisson)), young * poisson / ((1 + poisson) * (1 - 2 * poisson))


def symmetric_tensor(value, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must contain real 3 by 3 tensors")
    tensor = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(tensor)):
        raise ValueError(f"{name} must contain finite values")
    scale = np.maximum(np.max(np.abs(tensor), axis=(-2, -1), keepdims=True), np.finfo(float).tiny)
    if np.any(np.abs(tensor - tensor.swapaxes(-1, -2)) > 64 * np.finfo(float).eps * scale):
        raise ValueError(f"{name} must be symmetric")
    return .5 * (tensor + tensor.swapaxes(-1, -2))


@dataclass(frozen=True)
class NoTensionResponse:
    stress_pa: np.ndarray
    relaxed_strain: np.ndarray
    energy_j_m3: np.ndarray
    basis: np.ndarray
    normal_tangent_pa: np.ndarray
    shear_tangent_pa: np.ndarray
    compressed_directions: np.ndarray

    def tangent(self, increment) -> np.ndarray:
        """A consistent branch derivative; a compressive-side choice at zero."""
        direction = symmetric_tensor(increment, "Strain increment")
        local = self.basis.swapaxes(-1, -2) @ direction @ self.basis
        diagonal = np.einsum("...ij,...j->...i", self.normal_tangent_pa,
                             np.diagonal(local, axis1=-2, axis2=-1))
        result = self.shear_tangent_pa * local
        for i in range(3):
            result[..., i, i] = diagonal[..., i]
        return self.basis @ result @ self.basis.swapaxes(-1, -2)


def response(strain, young: float, poisson: float) -> NoTensionResponse:
    """Spectral active-set solution of the convex, local relaxed-energy problem."""
    mu, _ = elastic_constants(young, poisson)
    tensor = symmetric_tensor(strain, "Strain")
    eigenvalues, basis = np.linalg.eigh(tensor)
    shape = eigenvalues.shape
    principal = np.zeros(shape)
    normal = np.zeros((*shape[:-1], 3, 3))
    count = np.full(shape[:-1], -1, dtype=int)
    tolerance = 128 * np.finfo(float).eps * np.maximum(
        np.max(np.abs(eigenvalues), axis=-1), np.finfo(float).tiny,
    )
    # Sorted eigenstrains imply the compressed directions form a leading subset.
    # Try the largest set first to select elastic stiffness at the origin.
    for k in (3, 2, 1, 0):
        candidate = np.zeros(shape)
        coefficient = poisson / (1 - (k - 1) * poisson)
        if k:
            candidate[..., :k] = 2 * mu * (
                eigenvalues[..., :k] + coefficient * np.sum(eigenvalues[..., :k], axis=-1)[..., None]
            )
        relaxed = eigenvalues + poisson / young * np.sum(candidate, axis=-1)[..., None]
        valid = (count < 0)
        if k:
            valid &= np.all(candidate[..., :k] <= 2 * mu * tolerance[..., None], axis=-1)
        if k < 3:
            valid &= np.all(relaxed[..., k:] >= -tolerance[..., None], axis=-1)
        principal[valid] = np.minimum(candidate[valid], 0)
        branch = np.zeros((3, 3))
        branch[:k, :k] = 2 * mu * (np.eye(k) + coefficient)
        normal[valid] = branch
        count[valid] = k
    if np.any(count < 0):
        raise RuntimeError("No-tension constitutive active set failed")
    stress = np.einsum("...ik,...k,...jk->...ij", basis, principal, basis)
    elastic_strain = ((1 + poisson) * stress - poisson * np.trace(
        stress, axis1=-2, axis2=-1,
    )[..., None, None] * np.eye(3)) / young
    gap = tensor - elastic_strain
    difference = eigenvalues[..., :, None] - eigenvalues[..., None, :]
    stress_difference = principal[..., :, None] - principal[..., None, :]
    shear = np.zeros_like(difference)
    distinct = np.abs(difference) > tolerance[..., None, None]
    np.divide(stress_difference, difference, out=shear, where=distinct)
    # At repeated eigenvalues, an isotropic branch has shear derivative
    # d sigma_i / d eps_i - d sigma_i / d eps_j.
    repeated = np.diagonal(normal, axis1=-2, axis2=-1)[..., :, None] - normal
    shear = np.where(distinct, shear, repeated)
    energy = .5 * np.einsum("...ij,...ij->...", stress, tensor)
    return NoTensionResponse(stress, gap, energy, basis, normal, shear, count)


def tensile_strength_response(strain, young: float, poisson: float,
                              tensile_strength_pa: float) -> NoTensionResponse:
    """Reversible tensile cap: min 1/2(eps-eta):C:(eps-eta) + ft*tr(eta), eta >= 0.

    Below ft this is the original elastic law; above ft extension is relaxed.
    This is not irreversible softening or a calibrated crack-nucleation law.
    """
    if (isinstance(tensile_strength_pa, bool)
            or not isinstance(tensile_strength_pa, numbers.Real)
            or not np.isfinite(tensile_strength_pa) or tensile_strength_pa < 0):
        raise ValueError("Selected tensile strength must be finite and nonnegative")
    elastic_constants(young, poisson)
    tensor = symmetric_tensor(strain, "Strain")
    shift = tensile_strength_pa * (1 - 2 * poisson) / young * np.eye(3)
    relaxed = response(tensor - shift, young, poisson)
    stress = relaxed.stress_pa + tensile_strength_pa * np.eye(3)
    energy = (
        .5 * np.einsum("...ij,...ij->...", stress, tensor - relaxed.relaxed_strain)
        + tensile_strength_pa * np.trace(relaxed.relaxed_strain, axis1=-2, axis2=-1)
    )
    return NoTensionResponse(
        stress, relaxed.relaxed_strain, energy, relaxed.basis,
        relaxed.normal_tangent_pa, relaxed.shear_tangent_pa, relaxed.compressed_directions,
    )
