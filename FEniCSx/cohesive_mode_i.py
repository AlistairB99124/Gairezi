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


class FacetCohesiveLaws:
    """Per-facet assignment of linear Mode-I laws (e.g. intact concrete vs weak lift joints).

    ``laws`` is a tuple of LinearModeICohesiveLaw and ``law_index`` one law index per
    facet. The strength/critical-opening/fracture-energy properties are arrays over the
    facets so vectorised callers broadcast naturally; ``law_for`` serves the per-facet
    response loop. ``wrap`` accepts a plain law and spreads it over ``count`` facets so
    single-material callers are unchanged.
    """

    def __init__(self, laws, law_index):
        laws = tuple(laws)
        if not laws or any(not isinstance(law, LinearModeICohesiveLaw) for law in laws):
            raise ValueError("FacetCohesiveLaws needs at least one LinearModeICohesiveLaw")
        index = np.asarray(law_index)
        if index.ndim != 1 or index.dtype.kind not in "iu" or (len(index) and (index.min() < 0 or index.max() >= len(laws))):
            raise ValueError("law_index must be a 1-D integer array of valid law indices")
        self.laws = laws
        self.law_index = index.astype(np.int64)
        self.law_index.setflags(write=False)
        self.tensile_strength_pa = np.array([law.tensile_strength_pa for law in laws])[self.law_index]
        self.fracture_energy_j_m2 = np.array([law.fracture_energy_j_m2 for law in laws])[self.law_index]
        self.critical_opening_m = np.array([law.critical_opening_m for law in laws])[self.law_index]
        for array in (self.tensile_strength_pa, self.fracture_energy_j_m2, self.critical_opening_m):
            array.setflags(write=False)

    @classmethod
    def wrap(cls, law, count: int) -> "FacetCohesiveLaws":
        if isinstance(law, cls):
            if len(law.law_index) != count:
                raise ValueError("Per-facet laws do not match the interface pair count")
            return law
        return cls((law,), np.zeros(int(count), dtype=np.int64))

    @property
    def count(self) -> int:
        return len(self.law_index)

    @property
    def uniform(self) -> bool:
        return len(self.laws) == 1

    @property
    def reference(self) -> LinearModeICohesiveLaw:
        """The law of facet 0's material when uniform; the first law otherwise."""
        return self.laws[0]

    def law_for(self, index: int) -> LinearModeICohesiveLaw:
        return self.laws[self.law_index[index]]

    def subset(self, ids) -> "FacetCohesiveLaws":
        return FacetCohesiveLaws(self.laws, self.law_index[np.asarray(ids)])

    def summary(self) -> dict:
        counts = np.bincount(self.law_index, minlength=len(self.laws))
        return {
            "materials": [{"tensile_strength_pa": law.tensile_strength_pa,
                           "fracture_energy_j_m2": law.fracture_energy_j_m2,
                           "critical_opening_m": law.critical_opening_m,
                           "facets": int(counts[i])} for i, law in enumerate(self.laws)],
            "uniform": self.uniform,
        }
