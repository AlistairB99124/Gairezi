"""Fixed-mesh cohesive activation and native history-preserving round continuation."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from Aster.automatic_cohesive import wall_cell_stress


def dormant_ties(joints: ArrayLike, active: ArrayLike, fixed_pairs: ArrayLike) -> np.ndarray:
    """Independent constraints closing inactive facets, without redundant node-star cycles."""
    joints = np.asarray(joints, dtype=np.int64).reshape(-1, 6)
    active = np.asarray(active, dtype=bool)
    if active.shape != (len(joints),):
        raise ValueError("The active mask must match the cohesive facet bank")
    parent = {}

    def root(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def join(a, b):
        ra, rb = root(int(a)), root(int(b))
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    for a, b in np.asarray(fixed_pairs, dtype=np.int64).reshape(-1, 2):
        join(a, b)
    pairs = []
    for face in joints[~active]:
        for a, b in zip(face[:3], face[3:]):
            if join(a, b):
                pairs.append((int(a), int(b)))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def assess_round(points: ArrayLike, cells: ArrayLike, wall_mask: ArrayLike,
                 displacement: ArrayLike, owners: ArrayLike, active: ArrayLike,
                 target: float, young: float, poisson: float) -> tuple[dict, np.ndarray]:
    """Screen updated P1 wall stresses; only two-wall-owner facets can become active."""
    points = np.asarray(points, dtype=float)
    cells = np.asarray(cells, dtype=np.int64)
    wall_mask = np.asarray(wall_mask, dtype=bool)
    displacement = np.asarray(displacement, dtype=float)
    owners = np.asarray(owners, dtype=np.int64)
    if (owners.shape != (len(active), 2) or np.any(owners < 0)
            or np.any(owners >= len(cells)) or not wall_mask[owners].all()):
        raise ValueError("Continuation facets must have two wall-cell owners")
    wall = cells[wall_mask]
    x = points[wall]
    stress = wall_cell_stress(points, wall, displacement, young, poisson)
    principal = np.linalg.eigvalsh(stress)[:, -1]
    over = np.zeros(len(cells), dtype=bool)
    over[wall_mask] = principal > target
    eligible = over[owners].all(axis=1) & ~np.asarray(active, dtype=bool)
    peak = int(np.argmax(principal))
    return {
        "threshold_pa": float(target),
        "maximum_principal_tensile_stress_pa": float(principal[peak]),
        "peak_centroid_xyz_m": x[peak].mean(axis=0).tolist(),
        "wall_cells_above_threshold": int(np.count_nonzero(principal > target)),
        "maximum_wall_displacement_m": float(np.linalg.norm(displacement[np.unique(wall)], axis=1).max()),
    }, eligible


def render(comm: str, settings: dict) -> str:
    """Add activation rounds after the design-load solve, on the same native mesh/result."""
    from inspect import getsource
    for anchor in ("history = {", "RESU = None\nreached = 0.0", "    target_index += 1\n",
                   "VALE=(0.0, 0.0, 100.0, 100.0), PROL_DROITE='LINEAIRE'"):
        if comm.count(anchor) != 1:
            raise ValueError(f"Cannot safely render fixed-mesh continuation: ambiguous/missing {anchor!r}")
    helpers = ("import numpy as np\nfrom numpy.typing import ArrayLike\n\n"
               + getsource(wall_cell_stress) + "\n" + getsource(dormant_ties) + "\n" + getsource(assess_round))
    comm = comm.replace("history = {", (
        helpers + "\n"
        f"continuation_settings = {settings!r}\n"
        f"with np.load({settings['arrays_path']!r}, allow_pickle=False) as stored:\n"
        "    continuation_arrays = {key: stored[key] for key in ("
        "'nodes', 'tetra4_concrete', 'concrete_body_ids', 'penta6_joints', 'tie_node_pairs', "
        "'continuation_active_faces', 'continuation_face_owners')}\n"
        "continuation_active = continuation_arrays['continuation_active_faces'].astype(bool)\n"
        "continuation_round = 1\n"
        "continuation_round_started = time.perf_counter()\n"
        "continuation_loads = []\n"
        "history = {"), 1)
    comm = comm.replace(
        "VALE=(0.0, 0.0, 100.0, 100.0), PROL_DROITE='LINEAIRE'",
        "VALE=(0.0, 0.0, 1.0, 1.0), PROL_DROITE='CONSTANT'")
    comm = comm.replace("RESU = STAT_NON_LINE(reuse=RESU, ETAT_INIT=_F(EVOL_NOLI=RESU), **options)",
                        "previous = RESU\n"
                        "            trial = STAT_NON_LINE(ETAT_INIT=_F(EVOL_NOLI=previous, INST=reached), **options)\n"
                        "            RESU = trial\n"
                        "            DETRUIRE(CONCEPT=_F(NOM=previous))")
    anchor = "RESU = None\nreached = 0.0"
    comm = comm.replace(anchor, '''
BASE_EXCIT = EXCIT

def activation_load():
    pairs = dormant_ties(continuation_arrays['penta6_joints'], continuation_active,
                         continuation_arrays['tie_node_pairs'])
    if not len(pairs):
        return BASE_EXCIT
    load = AFFE_CHAR_MECA(MODELE=MODELE, LIAISON_DDL=tuple(
        _F(NOEUD=(f'N{a + 1}', f'N{b + 1}'), DDL=(dof, dof),
           COEF_MULT=(1., -1.), COEF_IMPO=0.)
        for a, b in pairs for dof in ('DX', 'DY', 'DZ')))
    continuation_loads.append(load)
    return BASE_EXCIT + (_F(CHARGE=load),)

EXCIT = activation_load()
RESU = None
reached = 0.0''')
    comm = comm.replace('history["load_factor"].append(target)',
                        'history["load_factor"].append(min(target, 1.0))\n'
                        '    history["solver_instant"].append(target)')
    comm = comm.replace('history = {"load_factor": []',
                        'history = {"solver_instant": [], "load_factor": []')
    comm = comm.replace('status["converged_load_factors"].append(target)',
                        'status["converged_load_factors"].append(min(target, 1.0))')
    comm = comm.replace('"load_factor": target, "mechanical_s":',
                        '"load_factor": min(target, 1.0), "solver_instant": target, "mechanical_s":')
    comm = comm.replace('load_factor={target:.6g}', 'load_factor={min(target, 1.0):.6g}')
    comm = comm.replace('\n        save({"status": "stopped", "stopped_at_load_factor": target,', (
        '\n        if target > 1.0:\n'
        '            status["continuation_stop_reason"] = "solver_failure_or_incomplete_load"\n'
        '            status["failed_continuation_instant"] = target\n'
        '        save({"status": "stopped", "stopped_at_load_factor": min(target, 1.0),'))
    # Hydraulic resultants must use physical load, not continuation pseudo-time.
    comm = comm.replace("'load_factor': target, 'iterations':", "'load_factor': min(target, 1.0), 'iterations':")
    comm = comm.replace("(target * water_forces", "(min(target, 1.0) * water_forces")
    comm = comm.replace("(target * np.cross", "(min(target, 1.0) * np.cross")
    comm = comm.replace("float(target * np.linalg.norm", "float(min(target, 1.0) * np.linalg.norm")
    comm = comm.replace("(target * hydraulic_uplift_resultant)", "(min(target, 1.0) * hydraulic_uplift_resultant)")
    # Bisecting pseudo-time does not reduce the constraint release or the physical load.
    comm = comm.replace(
        "\n        retry = bisect_load_interval(reached, target, target_level, hydraulic_subdivision_levels)\n",
        "\n        retry = (bisect_load_interval(reached, target, target_level, hydraulic_subdivision_levels)\n"
        "                 if target <= 1.0 else None)\n")
    comm = comm.replace("    target_index += 1\n", RUNTIME_ROUND + "\n    target_index += 1\n")
    return comm


def render_preload(comm: str, start: float, strength_pa: float) -> str:
    """Create an intact equilibrium on the fixed mesh before the first facet release."""
    def replace_once(old, new):
        nonlocal comm
        if comm.count(old) != 1:
            raise ValueError(f"Cannot safely render intact preload: missing/ambiguous {old!r}")
        comm = comm.replace(old, new, 1)

    metadata = {"status": "pending" if start > 0 else "skipped", "load_factor": start}
    replace_once("save()\nEXCIT = (", f"status['preload'] = {metadata!r}\nsave()\nEXCIT = (")
    if start == 0:
        return comm
    plateau = (0., start, 1., 1.) if start == 1 else (0., start, start, start, 1., 1.)
    replace_once("VALE=(0.0, 0.0, 1.0, 1.0), PROL_DROITE='CONSTANT'",
                 f"VALE={plateau!r}, PROL_DROITE='CONSTANT'")
    anchor = ("\n    try:\n        water_load = None" if "\n    try:\n        water_load = None" in comm
              else "\n    try:\n        mechanical_started = time.perf_counter()")
    block = PRELOAD_RUNTIME.replace("PRELOAD_FACTOR", repr(start)).replace("PRELOAD_STRENGTH", repr(strength_pa))
    replace_once(anchor, block + anchor)
    if "retry = (bisect_load_interval" in comm:
        replace_once("if target <= 1.0 else None)", "if target <= 1.0 and target_index > 0 else None)")
    return comm


PRELOAD_RUNTIME = '''
    if target_index == 0:
        from pathlib import Path
        status['phase'] = 'preload'
        status['preload'].update(status='running', started_at=time.time())
        save()
        try:
            preload_active = continuation_active.copy()
            try:
                continuation_active[:] = False
                preload_excit = activation_load()
            finally:
                continuation_active[:] = preload_active
            preload_ramp = DEFI_FONCTION(
                NOM_PARA='INST', VALE=(-1., 0., 0., PRELOAD_FACTOR), PROL_DROITE='CONSTANT')
            preload_options = dict(options)
            preload_options['EXCIT'] = tuple(
                _F(**dict(entry, FONC_MULT=preload_ramp)) if 'FONC_MULT' in entry else entry
                for entry in preload_excit)
            preload_options['INCREMENT'] = _F(LIST_INST=DEFI_LIST_REEL(
                DEBUT=-1., INTERVALLE=_F(JUSQU_A=0., NOMBRE=1)))
            RESU = STAT_NON_LINE(**preload_options)
            preload_last = RESU.getLastIndex()
            preload_vari = RESU.getField('VARI_ELGA', preload_last)
            preload_damage = joint_gauss(preload_vari, 'V3')[0]
            preload_dissipation = joint_gauss(preload_vari, 'V5')[0]
            preload_jump = joint_gauss(preload_vari, 'V7')[0]
            if np.any(preload_damage > 1.e-9) or np.any(np.abs(preload_jump) > 1.e-12):
                raise RuntimeError('Closed-bank preload unexpectedly damaged or opened a facet')
            preload_displacement = nodal_displacement(RESU.getField('DEPL', preload_last))
            preload_assessment, _ = assess_round(
                continuation_arrays['nodes'], continuation_arrays['tetra4_concrete'],
                continuation_arrays['concrete_body_ids'] == 1, preload_displacement,
                continuation_arrays['continuation_face_owners'], np.zeros_like(continuation_active),
                PRELOAD_STRENGTH, continuation_settings['youngs_modulus_pa'],
                continuation_settings['poissons_ratio'])
            warnings = ['Intact preload assumes no earlier cracking; omitted fracture history is not simulated.']
            if preload_assessment['maximum_principal_tensile_stress_pa'] >= PRELOAD_STRENGTH:
                warnings.append('Preload wall tension reaches material strength; earlier cracking may have occurred.')
            folder = Path(STEPS_PATH).parent / 'preload'
            with root_archive_write() as writer:
                if writer:
                    folder.mkdir(exist_ok=False)
                    np.savez(folder / 'accepted_state.npz', displacement_m=preload_displacement,
                             damage_state_gauss=preload_damage, dissipated_energy_gauss=preload_dissipation)
            DEFI_FICHIER(ACTION='ASSOCIER', UNITE=91, FICHIER=str(folder / 'accepted_result.med'))
            IMPR_RESU(FORMAT='MED', UNITE=91, RESU=_F(RESULTAT=RESU, INST=0.,
                      NOM_CHAM=('DEPL', 'SIEF_ELGA', 'VARI_ELGA')))
            DEFI_FICHIER(ACTION='LIBERER', UNITE=91)
            status['preload'].update(status='completed', solver_instant=0., finished_at=time.time(),
                                     residual_tension_assessment=preload_assessment, warnings=warnings,
                                     maximum_damage=float(preload_damage.max()), result_directory=str(folder))
            with root_archive_write() as writer:
                if writer:
                    with open(folder / 'manifest.json', 'w', encoding='utf-8') as stream:
                        json.dump(status['preload'], stream, indent=2)
            status['phase'] = 'cracking'
            save()
            options['INCREMENT'] = _F(LIST_INST=DEFI_LIST_REEL(
                DEBUT=0., INTERVALLE=_F(JUSQU_A=target, NOMBRE=1)))
        except (CA.AsterError, RuntimeError, OSError, MemoryError) as error:
            detail = str(error).strip() or type(error).__name__
            status['preload'].update(status='failed', message=detail, finished_at=time.time())
            if not stop_for_resource_failure(error):
                save({'status': 'failed', 'message': 'Intact preload failed: ' + detail})
            break
'''


RUNTIME_ROUND = '''
    if target >= 1.0 and target == TARGETS[-1]:
        from pathlib import Path
        displacement = nodal_displacement(RESU.getField('DEPL', last))
        damage_gauss = joint_gauss(RESU.getField('VARI_ELGA', last), 'V3')[0]
        dissipation_gauss = joint_gauss(RESU.getField('VARI_ELGA', last), 'V5')[0]
        if continuation_round > 1:
            if (np.any(damage_gauss < accepted_damage_gauss - 1.e-9)
                    or np.any(dissipation_gauss < accepted_dissipation_gauss - 1.e-9)):
                raise RuntimeError('Cohesive history decreased at a retained integration point')
        accepted_damage_gauss = damage_gauss.copy()
        accepted_dissipation_gauss = dissipation_gauss.copy()
        assessment, eligible = assess_round(
            continuation_arrays['nodes'], continuation_arrays['tetra4_concrete'],
            continuation_arrays['concrete_body_ids'] == 1, displacement,
            continuation_arrays['continuation_face_owners'], continuation_active,
            continuation_settings['target_pa'], continuation_settings['youngs_modulus_pa'],
            continuation_settings['poissons_ratio'])
        folder = Path(STEPS_PATH).parent / f'round_{continuation_round:04d}'
        with root_archive_write() as writer:
            if writer:
                folder.mkdir(exist_ok=False)
                np.savez(folder / 'accepted_state.npz', displacement_m=displacement,
                         active_faces=continuation_active, damage_state=history['damage_state'][-1],
                         normal_jump_m=history['normal_jump_m'][-1], damage_state_gauss=damage_gauss,
                         dissipated_energy_gauss=dissipation_gauss)
        DEFI_FICHIER(ACTION='ASSOCIER', UNITE=91, FICHIER=str(folder / 'accepted_result.med'))
        IMPR_RESU(FORMAT='MED', UNITE=91, RESU=_F(RESULTAT=RESU, INST=target,
                  NOM_CHAM=('DEPL', 'SIEF_ELGA', 'VARI_ELGA')))
        DEFI_FICHIER(ACTION='LIBERER', UNITE=91)
        row = {
            'iteration': continuation_round, 'solver_status': 'completed', 'accepted': True,
            'reached_load_factor': 1.0, 'solver_instant': target,
            'active_faces': int(continuation_active.sum()), 'new_eligible_faces': int(eligible.sum()),
            'candidate_faces': int(continuation_active.sum()),
            'residual_tension_assessment': assessment,
            'maximum_principal_tensile_stress_pa': assessment['maximum_principal_tensile_stress_pa'],
            'maximum_displacement_m': assessment['maximum_wall_displacement_m'],
            'result_directory': str(folder),
            'duration_s': time.perf_counter() - continuation_round_started,
            'opening_assessment': {
                'peak_opening_m': float(max(history['normal_jump_m'][-1].max(), 0.)),
                'initiated_joint_cells': int((history['damage_state'][-1] >= 1.).sum()),
                'separated_joint_cells': int((history['damage_state'][-1] >= 2. - 1.e-9).sum()),
            },
            'state_preservation': 'native EVOL_NOLI on unchanged mesh and integration points',
        }
        facet_damage = history['damage_state'][-1].reshape(len(continuation_active), 3).max(axis=1)
        facet_jump = history['normal_jump_m'][-1].reshape(len(continuation_active), 3).max(axis=1)
        initiated = facet_damage >= 1. - 1.e-9
        opened = initiated & (facet_jump > continuation_settings['opening_limit_m'] * (1. + 1.e-6))
        corners = continuation_arrays['nodes'][continuation_arrays['penta6_joints'][:, :3]]
        areas = .5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                             corners[:, 2] - corners[:, 0]), axis=1)
        row['opening_assessment'].update(
            initiated_faces=int(initiated.sum()), opened_faces=int(opened.sum()),
            separated_faces=int((facet_damage >= 2. - 1.e-9).sum()))
        row['crack_surface_measure_m2'] = float(areas[opened].sum())
        print(f"ASTER_COHESIVE_ROUND accepted round={continuation_round} load_factor=1 "
              f"active_faces={row['active_faces']} new_eligible_faces={row['new_eligible_faces']} "
              f"maximum_principal_tensile_stress_pa={row['maximum_principal_tensile_stress_pa']:.6g}",
              flush=True)
        status.setdefault('continuation_rounds', []).append(row)
        with root_archive_write() as writer:
            if writer:
                with open(folder / 'manifest.json', 'w', encoding='utf-8') as stream:
                    json.dump(row, stream, indent=2)
        if assessment['maximum_principal_tensile_stress_pa'] <= continuation_settings['target_pa']:
            status['continuation_stop_reason'] = 'residual_target_reached'
        elif not eligible.any():
            status['continuation_stop_reason'] = 'no_additional_eligible_facets'
        elif continuation_round >= continuation_settings['maximum_rounds']:
            status['continuation_stop_reason'] = 'iteration_limit'
        else:
            continuation_active |= eligible
            continuation_round += 1
            continuation_round_started = time.perf_counter()
            EXCIT = activation_load()
            TARGETS.append(target + 0.01)
            TARGET_LEVELS.append(0)
            status['targets'] = TARGETS
        save()
'''
