"""Experimental extrinsic Mode-I interface law, not an equilibrium solver.

The caller supplies an already initiated interface with nonnegative opening.
Intact bonding, initiation traction transfer, compression contact, tangential
response and hydraulic access are deliberately not inferred by this local law.
Linear softening dissipates G_f J/m2 at complete separation; no additional
Griffith area charge belongs to that same fracture process.
"""
from dataclasses import dataclass
import numbers

import numpy as np


def _finite_nonnegative(value, name):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not np.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return float(value)


@dataclass(frozen=True)
class ModeIHistory:
    maximum_opening_m: float = 0.

    def __post_init__(self):
        object.__setattr__(self, "maximum_opening_m",
                           _finite_nonnegative(self.maximum_opening_m, "Maximum opening"))


@dataclass(frozen=True)
class ModeIResponse:
    tensile_traction_pa: float
    tangent_pa_per_m: float
    stored_energy_j_m2: float
    dissipated_energy_j_m2: float
    trial_history: ModeIHistory


@dataclass(frozen=True)
class LinearModeICohesiveLaw:
    tensile_strength_pa: float
    fracture_energy_j_m2: float

    def __post_init__(self):
        strength = _finite_nonnegative(self.tensile_strength_pa, "Tensile strength")
        energy = _finite_nonnegative(self.fracture_energy_j_m2, "Fracture energy")
        if strength == 0 or energy == 0:
            raise ValueError("Tensile strength and fracture energy must be positive")
        opening = energy / strength * 2
        if not np.isfinite(opening) or opening <= 0:
            raise ValueError("Fracture parameters must define a finite positive critical opening")
        slope = strength / opening
        if not np.isfinite(slope) or slope <= 0:
            raise ValueError("Fracture parameters must define a finite positive softening slope")
        object.__setattr__(self, "tensile_strength_pa", strength)
        object.__setattr__(self, "fracture_energy_j_m2", energy)

    @property
    def critical_opening_m(self) -> float:
        return self.fracture_energy_j_m2 / self.tensile_strength_pa * 2

    def response(self, opening_m: float, committed_history: ModeIHistory) -> ModeIResponse:
        """Return a trial without mutating committed history.

        Equality with the historical maximum selects the loading-side tangent.
        At first activation, zero opening carries f_t with zero stored energy.
        Subsequent unloading is secant to the origin. Negative gaps require
        a separate contact solve, not compressive use of this tensile law.
        """
        opening = _finite_nonnegative(opening_m, "Opening (compression requires contact)")
        if not isinstance(committed_history, ModeIHistory):
            raise ValueError("An explicit committed ModeIHistory is required")
        previous = committed_history.maximum_opening_m
        maximum = max(previous, opening)
        critical = self.critical_opening_m
        history = ModeIHistory(maximum)
        if maximum >= critical:
            return ModeIResponse(0., 0., 0., self.fracture_energy_j_m2, history)
        strength = self.tensile_strength_pa
        envelope = strength * (1 - maximum / critical)
        if opening >= previous:
            traction = envelope
            tangent = -strength / critical
        else:
            traction = envelope * (opening / maximum)
            tangent = envelope / maximum
            if not np.isfinite(tangent):
                raise ValueError("Opening history gives a non-finite unloading stiffness")
        stored = .5 * traction * opening
        dissipated = .5 * strength * maximum
        return ModeIResponse(traction, tangent, stored, dissipated, history)
