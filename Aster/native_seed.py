"""Exact physical UI-seed placement for native ENDO_FISS_EXP initial states."""
from __future__ import annotations

import math

import numpy as np

from FEniCSx.automatic_crack import sector_signed_distance
from FEniCSx.crack_contract import validate_selected_seed


def seed_damage(points: np.ndarray, contract: dict, initial_damage: float) -> np.ndarray:
    """Assign the native damage level inside, without expanding the shared seed."""
    geometry = validate_selected_seed(contract)["geometry"]
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Seed evaluation needs finite (n, 3) physical coordinates")
    if (isinstance(initial_damage, bool) or not np.isfinite(initial_damage)
            or not 0 < initial_damage < 1):
        raise ValueError("Native seed damage must lie strictly between zero and one")
    distance = sector_signed_distance(
        points, tuple(geometry["annular_sector"]),
        geometry["elevation_m"], geometry["seed_thickness_m"],
    )
    return np.where(distance <= 0., initial_damage, 0.)


def stiffness_ratio(damage: np.ndarray, m: float, p: float) -> np.ndarray:
    """Native ENDO_FISS_EXP ENDORIGI stiffness (not an AT2 degradation law)."""
    damage = np.asarray(damage, dtype=float)
    if (not np.isfinite(damage).all() or np.any((damage < 0) | (damage >= 1))
            or not np.isfinite(m) or m <= 0 or not np.isfinite(p) or p < 0):
        raise ValueError("Invalid native stiffness parameters or damage")
    intact = (1 - damage) ** 2
    return intact / (intact + m * damage * (1 + p * damage))


def initial_field(model, material, contract: dict, initial_damage: float,
                  wall_group: str = "WALL", volume_group: str = "VOLUMES"):
    """Build and audit the nine native internal variables at actual Gauss points."""
    from code_aster.Commands import CREA_CHAMP, FORMULE
    from code_aster.Cata.Syntax import _F

    geometry = validate_selected_seed(contract)["geometry"]
    # Validate the native level even when the mesh misses the entire seed.
    seed_damage(np.empty((0, 3)), contract, initial_damage)
    mesh = model.getMesh()
    wall_cells, volume_cells = set(mesh.getCells(wall_group)), set(mesh.getCells(volume_group))
    if not wall_cells or not volume_cells or not wall_cells.issubset(volume_cells):
        raise ValueError("Native seed requires a nonempty wall group within the volume group")
    m = material.getValueReal("ENDO_FISS_EXP", "M")
    p = material.getValueReal("ENDO_FISS_EXP", "P")
    stiffness_ratio(np.array([initial_damage]), m, p)
    inner, outer, start, end = geometry["annular_sector"]
    elevation, half_thickness = geometry["elevation_m"], geometry["seed_thickness_m"] / 2

    def damage_at(x, y, z):
        radius, angle = math.hypot(x, y), math.atan2(x, y)
        return initial_damage if (
            inner <= radius <= outer and start <= angle <= end and
            abs(z - elevation) <= half_thickness
        ) else 0.

    damage = FORMULE(NOM_PARA=("X", "Y", "Z"), VALE="seed(X,Y,Z)", seed=damage_at)
    recovery = FORMULE(
        NOM_PARA=("X", "Y", "Z"), VALE="ratio(seed(X,Y,Z))",
        seed=damage_at, ratio=lambda value: float(stiffness_ratio(value, m, p)),
    )
    zero = FORMULE(NOM_PARA=("X", "Y", "Z"), VALE="0.")
    one = FORMULE(NOM_PARA=("X", "Y", "Z"), VALE="1.")
    components = tuple(f"X{i}" for i in range(1, 10))
    functions = CREA_CHAMP(
        TYPE_CHAM="ELGA_NEUT_F", OPERATION="AFFE", MODELE=model, PROL_ZERO="OUI",
        AFFE=(
            _F(GROUP_MA=volume_group, NOM_CMP=components,
               VALE_F=(zero, zero, one, zero, zero, zero, zero, zero, zero)),
            _F(GROUP_MA=wall_group, NOM_CMP=components,
               VALE_F=(damage, zero, recovery, zero, zero, zero, zero, zero, zero)),
        ),
    )
    nodes = CREA_CHAMP(
        TYPE_CHAM="NOEU_GEOM_R", OPERATION="EXTR", MAILLAGE=mesh, NOM_CHAM="GEOMETRIE",
    )
    coordinates = CREA_CHAMP(
        TYPE_CHAM="ELGA_GEOM_R", OPERATION="DISC", MODELE=model, CHAM_GD=nodes,
    )
    values = CREA_CHAMP(
        TYPE_CHAM="ELGA_NEUT_R", OPERATION="EVAL", CHAM_F=functions, CHAM_PARA=coordinates,
    )
    initial = CREA_CHAMP(
        TYPE_CHAM="ELGA_VARI_R", OPERATION="ASSE", MODELE=model,
        ASSE=_F(GROUP_MA=volume_group, CHAM_GD=values, NOM_CMP=components,
                NOM_CMP_RESU=tuple(f"V{i}" for i in range(1, 10))),
    )
    damage_values, support = initial.getValuesWithDescription(["V1"])
    cells, _, points, subpoints = support
    identities = list(zip(cells, points, subpoints))
    if len(set(identities)) != len(identities) or not identities:
        raise ValueError("Native seed has missing or ambiguous Gauss-point identities")
    xyz = []
    for component in ("X", "Y", "Z"):
        data, description = coordinates.getValuesWithDescription([component])
        ids = list(zip(description[0], description[2], description[3]))
        mapping = dict(zip(ids, data))
        if len(mapping) != len(ids) or not set(identities).issubset(mapping):
            raise ValueError("Native seed coordinates do not match internal-variable identities")
        xyz.append([mapping[key] for key in identities])
    xyz = np.asarray(xyz).T
    expected = seed_damage(xyz, contract, initial_damage)
    expected[~np.isin(cells, list(wall_cells))] = 0.
    actual = np.asarray(damage_values)
    if not np.allclose(actual, expected, rtol=0., atol=1e-12):
        raise ValueError("Native seed initial field differs from the frozen physical geometry")
    if not np.any(expected > 0):
        raise ValueError("Native mesh misses the physical UI seed; refine it without relocating or expanding the seed")
    for number in range(2, 10):
        data, description = initial.getValuesWithDescription([f"V{number}"])
        if list(zip(description[0], description[2], description[3])) != identities:
            raise ValueError("Native internal-variable Gauss-point identities differ")
        expected_component = stiffness_ratio(expected, m, p) if number == 3 else np.zeros_like(expected)
        if not np.allclose(data, expected_component, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Native seed has inconsistent internal variable V{number}")
    audit = {
        "shared_seed_sha256": contract["sha256"], "initial_damage": initial_damage,
        "gauss_points": len(actual), "seeded_gauss_points": int(np.count_nonzero(expected)),
        "wall_group": wall_group, "volume_group": volume_group,
        "seeded_point_ids": [list(key) for key, value in zip(identities, expected) if value > 0],
        "coordinates_m": xyz.tolist(), "damage_point_ids": [list(key) for key in identities],
        "damage_values": actual.tolist(),
        "placement": "exact_physical_seed_no_expansion_or_relocation",
        "limitations": [
            "A sampled initial state is not evidence of mesh or band convergence.",
            "Native damage level is not an AT2-equivalent damage profile or physical opening.",
            "Internal crack pressure is not applied by this initial-state adapter.",
        ],
    }
    return initial, audit
