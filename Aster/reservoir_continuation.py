"""Render actual reservoir filling on the retained cohesive mesh."""
from __future__ import annotations

from inspect import getsource
from pathlib import Path
import re

from Aster import hydraulic_loads
from Aster.cohesive_continuation import RUNTIME_ROUND
from Aster.reservoir_loading import ReservoirPressure


def render(comm: str, loading: dict, strength_pa: float, selection_enabled: bool = True) -> str:
    start = comm.index("EXCIT = activation_load()\n")
    end = comm.index('if history["load_factor"]:\n', start)
    collection_start = comm.index("    reached = target\n", start)
    collection_end = comm.index(RUNTIME_ROUND, collection_start)
    collection = comm[collection_start:collection_end]
    collection = collection.replace("min(target, 1.0)", "accepted_depth / final_depth")
    collection = collection.replace("    reached = target\n", (
        "    reached = target\n"
        "    accepted_depth = trial_depth\n"
        "    history['water_depth_m'].append(accepted_depth)\n"
        "    history['water_surface_z_m'].append(reference_z + accepted_depth)\n"
        "    status['reached_water_depth_m'] = accepted_depth\n"
        "    status['water_surface_z_m'] = reference_z + accepted_depth\n"))
    comm = comm[:start] + (
        f"exec(compile({Path(hydraulic_loads.__file__).read_text(encoding='utf-8')!r}, "
        "'hydraulic_loads.py', 'exec'))\n"
        + getsource(ReservoirPressure) + "\n"
        f"reservoir_loading = {loading!r}\n"
        f"reservoir_strength = {strength_pa!r}\n"
        f"reservoir_selection_enabled = {selection_enabled!r}\n"
        + INITIALISE + PRELOAD + SOLVE + collection + activation_runtime()
        + "\n    target_index += 1\n\n"
    ) + comm[end:]
    # Only the new mode removes proportional multipliers. Supports/ties stay unchanged.
    comm = re.sub(r"_F\(CHARGE=(GRAVITE|CONDENSED_GRAVITY), FONC_MULT=RAMP\)",
                  r"_F(CHARGE=\1)", comm)
    for entry in ("    _F(CHARGE=PRESSIONS, FONC_MULT=RAMP),\n",
                  "    _F(CHARGE=CREST_SURCHARGE, FONC_MULT=RAMP),\n"):
        if comm.count(entry) != 1:
            raise ValueError("Missing or ambiguous proportional pressure entry")
        comm = comm.replace(entry, "", 1)
    comm = comm.replace(
        "stopped_at_load_factor=min(target, 1.0), reached_load_factor=min(reached, 1.0),",
        "stopped_at_water_depth_m=globals().get('trial_depth', accepted_depth), "
        "reached_water_depth_m=accepted_depth,")
    return comm


def activation_runtime() -> str:
    block = RUNTIME_ROUND
    block = block.replace("    if target >= 1.0 and target == TARGETS[-1]:", "    if True:")
    block = block.replace("'reached_load_factor': 1.0, 'solver_instant': target,",
                          "'reached_load_factor': accepted_depth / final_depth, 'solver_instant': target,\n"
                          "            'water_depth_m': accepted_depth, 'water_surface_z_m': reference_z + accepted_depth,\n"
                          "            'crest_surcharge_pa': reservoir_loading['gradient_pa_per_m'] * max(\n"
                          "                reference_z + accepted_depth - reservoir_loading['crest_z_m'], 0.),\n"
                          "            'activation_round_at_level': level_round,")
    block = block.replace("load_factor=1 ", "water_depth_m={accepted_depth:.6g} ")
    block = block.replace(
        "        row = {",
        "        if not reservoir_selection_enabled:\n"
        "            eligible[:] = False\n"
        "        row = {")
    begin = block.index("        if assessment['maximum_principal_tensile_stress_pa'] <=")
    block = block[:begin] + '''
        final_level = accepted_depth == final_depth
        if assessment['maximum_principal_tensile_stress_pa'] <= continuation_settings['target_pa']:
            if final_level:
                status['continuation_stop_reason'] = 'residual_target_reached'
        elif not eligible.any():
            if final_level:
                status['continuation_stop_reason'] = 'no_additional_eligible_facets'
        elif level_round >= continuation_settings['maximum_rounds']:
            warning = (f'Activation limit reached at water depth {accepted_depth:g} m with '
                       f'{int(eligible.sum())} eligible dormant facets unresolved.')
            status['activation_history_complete'] = False
            status.setdefault('activation_warnings', []).append(warning)
            row['warning'] = warning
            if final_level:
                status['continuation_stop_reason'] = 'iteration_limit'
        else:
            continuation_active |= eligible
            DEPTHS.insert(target_index + 1, accepted_depth)
            TARGET_LEVELS.insert(target_index + 1, 0)
        status['water_depth_targets_m'] = DEPTHS
        with root_archive_write() as writer:
            if writer:
                with open(folder / 'manifest.json', 'w', encoding='utf-8') as stream:
                    json.dump(row, stream, indent=2)
        continuation_round += 1
        save()
'''
    return block


INITIALISE = '''
from pathlib import Path
with np.load(continuation_settings['arrays_path'], allow_pickle=False) as stored:
    reservoir_arrays = {key: stored[key] for key in stored.files}
reference_z = reservoir_loading['depth_reference_z_m']
final_depth = reservoir_loading['final_water_depth_m']
accepted_depth = reservoir_loading['water_depth_start_m']
DEPTHS = list(reservoir_loading['water_depth_targets_m'])
TARGET_LEVELS = [0] * len(DEPTHS)
pressure = ReservoirPressure(reservoir_arrays, reservoir_loading['gradient_pa_per_m'])
tail_entries = force_entries(pressure.forces(
    reservoir_loading['tailwater_surface_z_m'], ('tri3_downstream',)))
tail_load = (AFFE_CHAR_MECA(MODELE=MODELE, FORCE_NODALE=tuple(_F(**e) for e in tail_entries))
             if tail_entries else None)
if tail_load is not None:
    BASE_EXCIT += (_F(CHARGE=tail_load),)
continuation_active[:] = False
history['water_depth_m'] = []
history['water_surface_z_m'] = []
status.update(loading_mode='reservoir_filling', reservoir_loading=reservoir_loading,
              water_depth_targets_m=DEPTHS, activation_history_complete=True)
status['preload'] = dict(status='pending', loading_mode='reservoir_filling',
                        water_depth_m=accepted_depth, water_surface_z_m=reference_z + accepted_depth,
                        crest_surcharge_pa=reservoir_loading['gradient_pa_per_m'] * max(
                            reference_z + accepted_depth - reservoir_loading['crest_z_m'], 0.))
RESU = None
reached = 0.
target = 1.
mesh_coordinates = np.asarray(MAIL.getCoordinates().getValues(), dtype=float).reshape(-1, 3)
accepted_depl = None
hydraulic_history = []
hydraulic_masks = []
hydraulic_settings = reservoir_loading.get('hydraulic_settings')
crack_water = (CrackWater(reservoir_arrays, reference_z + accepted_depth,
                         reservoir_loading['gradient_pa_per_m'],
                         hydraulic_settings['opening_tolerance_m']) if hydraulic_settings else None)
HYDRAULIC_OPENING_LIMIT = continuation_settings['opening_limit_m']

def hydraulic_open_faces(result):
    if result is None:
        return np.zeros(len(reservoir_arrays['penta6_joints']), dtype=bool)
    field = result.getField('VARI_ELGA', result.getLastIndex())
    damage = joint_component(field, 'V3')
    jump = joint_component(field, 'V7')
    return ((damage >= 1. - 1e-9) & (jump > HYDRAULIC_OPENING_LIMIT)).reshape(-1, 3).any(axis=1)

def reservoir_excit(initial_forces, final_forces, initial_instant, final_instant):
    concepts = []
    entries = activation_load()
    for forces, ends in ((initial_forces, (1., 0.)), (final_forces, (0., 1.))):
        nodal = force_entries(forces)
        if not nodal:
            continue
        load = AFFE_CHAR_MECA(MODELE=MODELE, FORCE_NODALE=tuple(_F(**e) for e in nodal))
        ramp = DEFI_FONCTION(NOM_PARA='INST', VALE=(
            initial_instant, ends[0], final_instant, ends[1]), PROL_DROITE='CONSTANT')
        concepts.extend((load, ramp))
        entries += (_F(CHARGE=load, FONC_MULT=ramp),)
    return entries, concepts

def reservoir_options(excit, initial_instant, final_instant, previous=None):
    options = dict(MODELE=MODELE, CHAM_MATER=CHMAT, CARA_ELEM=CARA_ROCK,
                   EXCIT=excit, COMPORTEMENT=COMPORT,
                   INCREMENT=_F(LIST_INST=DEFI_LIST_REEL(DEBUT=initial_instant,
                       INTERVALLE=_F(JUSQU_A=final_instant, NOMBRE=1))),
                   NEWTON=_F(MATRICE='TANGENTE', REAC_ITER=1),
                   CONVERGENCE=_F(RESI_GLOB_RELA=reservoir_loading['relative_residual_tolerance'],
                                  ITER_GLOB_MAXI=reservoir_loading['maximum_iterations']),
                   SOLVEUR=_F(METHODE='MULT_FRONT'))
    if reservoir_loading['line_search_iterations']:
        options['RECH_LINEAIRE'] = _F(METHODE='CORDE',
            ITER_LINE_MAXI=reservoir_loading['line_search_iterations'])
    if previous is not None:
        options['ETAT_INIT'] = _F(EVOL_NOLI=previous, INST=initial_instant)
    return options

def release_loads(concepts):
    for concept in concepts + continuation_loads:
        DETRUIRE(CONCEPT=_F(NOM=concept))
    continuation_loads.clear()
'''


PRELOAD = '''
status['phase'] = 'preload'
status['preload'].update(status='running', started_at=time.time())
save()
try:
    weight_ramp = DEFI_FONCTION(NOM_PARA='INST', VALE=(-2., 0., -1., 1.),
                               PROL_DROITE='CONSTANT')
    closed = activation_load()
    weight_charges = [globals()[name] for name in ('GRAVITE', 'CONDENSED_GRAVITY')
                      if name in globals()]
    if tail_load is not None:
        weight_charges.append(tail_load)
    weight_excit = tuple(_F(**dict(entry, FONC_MULT=weight_ramp))
                         if entry['CHARGE'] in weight_charges else entry
                         for entry in closed)
    weight = STAT_NON_LINE(**reservoir_options(weight_excit, -2., -1.))
    release_loads([weight_ramp])
    starting_forces = pressure.forces(reference_z + accepted_depth,
                                     ('tri3_upstream', 'tri3_crest'))
    preload_excit, preload_loads = reservoir_excit(
        np.zeros_like(starting_forces), starting_forces, -1., 0.)
    RESU = STAT_NON_LINE(**reservoir_options(preload_excit, -1., 0., weight))
    DETRUIRE(CONCEPT=_F(NOM=weight))
    release_loads(preload_loads)
    preload_last = RESU.getLastIndex()
    preload_vari = RESU.getField('VARI_ELGA', preload_last)
    preload_damage = joint_gauss(preload_vari, 'V3')[0]
    preload_dissipation = joint_gauss(preload_vari, 'V5')[0]
    preload_jump = joint_gauss(preload_vari, 'V7')[0]
    if np.any(preload_damage > 1.e-9) or np.any(np.abs(preload_jump) > 1.e-12):
        raise RuntimeError('Closed-bank preload unexpectedly damaged or opened a facet')
    preload_displacement = nodal_displacement(RESU.getField('DEPL', preload_last))
    preload_assessment, eligible = assess_round(
        continuation_arrays['nodes'], continuation_arrays['tetra4_concrete'],
        continuation_arrays['concrete_body_ids'] == 1, preload_displacement,
        continuation_arrays['continuation_face_owners'], continuation_active,
        continuation_settings['target_pa'], continuation_settings['youngs_modulus_pa'],
        continuation_settings['poissons_ratio'])
    warnings = ['Intact preload assumes no earlier cracking; omitted fracture history is not simulated.']
    if preload_assessment['maximum_principal_tensile_stress_pa'] >= reservoir_strength:
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
    if reservoir_selection_enabled:
        continuation_active |= eligible
    status['phase'] = 'cracking'
    save()
except (CA.AsterError, RuntimeError, OSError, MemoryError) as error:
    detail = str(error).strip() or type(error).__name__
    status['preload'].update(status='failed', message=detail, finished_at=time.time())
    stop_for_resource_failure(error)
    save({'status': 'failed', 'message': 'Intact preload failed: ' + detail})

target_index = 0
level_round = 0
previous_depth = None
while status['status'] == 'running' and target_index < len(DEPTHS):
'''


SOLVE = '''
    trial_depth = DEPTHS[target_index]
    target_level = TARGET_LEVELS[target_index]
    target = reached + 1.
    level_round = level_round + 1 if trial_depth == previous_depth else 1
    previous_depth = trial_depth
    continuation_round_started = time.perf_counter()
    step_started = continuation_round_started
    mechanical_s = 0.
    status.update(target_water_depth_m=trial_depth, target_solver_instant=target,
                  activation_round_at_level=level_round,
                  target_crest_surcharge_pa=reservoir_loading['gradient_pa_per_m'] * max(
                      reference_z + trial_depth - reservoir_loading['crest_z_m'], 0.))
    save()
    print(f"ASTER_COHESIVE_STEP start water_depth_m={trial_depth:.6g} solver_instant={target:g}", flush=True)
    initial_forces = pressure.forces(reference_z + accepted_depth, ('tri3_upstream', 'tri3_crest'))
    final_forces = pressure.forces(reference_z + trial_depth, ('tri3_upstream', 'tri3_crest'))
    previous = RESU
    if crack_water is not None:
        crack_water.set_surface(reference_z + accepted_depth)
        _, accepted_water = crack_water.evaluate(
            nodal_displacement(previous.getField('DEPL', previous.getLastIndex())), hydraulic_open_faces(previous))
        initial_forces += accepted_water
        crack_water.set_surface(reference_z + trial_depth)
        wet, water_forces = crack_water.evaluate(
            nodal_displacement(previous.getField('DEPL', previous.getLastIndex())), hydraulic_open_faces(previous))
    else:
        wet = np.zeros(len(reservoir_arrays['penta6_joints']), dtype=bool)
        water_forces = np.zeros_like(final_forces)
    concepts = []
    trial = None
    hydraulic_count = len(hydraulic_history)
    try:
        maximum = hydraulic_settings['maximum_iterations'] if hydraulic_settings else 1
        for hydraulic_iteration in range(1, maximum + 1):
            excit, concepts = reservoir_excit(initial_forces, final_forces + water_forces, reached, target)
            mechanical_started = time.perf_counter()
            trial = STAT_NON_LINE(**reservoir_options(excit, reached, target, previous))
            mechanical_s += time.perf_counter() - mechanical_started
            if crack_water is not None:
                updated_wet, updated_forces = crack_water.evaluate(
                    nodal_displacement(trial.getField('DEPL', trial.getLastIndex())), hydraulic_open_faces(trial))
            else:
                updated_wet, updated_forces = wet, water_forces
            if np.array_equal(wet, updated_wet):
                if crack_water is not None:
                    hydraulic_history.append({
                        'load_factor': trial_depth / final_depth, 'water_depth_m': trial_depth,
                        'water_surface_z_m': reference_z + trial_depth, 'solver_instant': target,
                        'iterations': hydraulic_iteration, 'active_set_stable': True,
                        'wet_faces': int(wet.sum()),
                        'paired_resultant_n': water_forces.sum(axis=0).tolist(),
                        'paired_moment_n_m': np.cross(reservoir_arrays['nodes'], water_forces).sum(axis=0).tolist(),
                        'sum_nodal_force_magnitudes_n': float(np.linalg.norm(water_forces, axis=1).sum()),
                        'uplift_resultant_n': [0., 0., 0.], 'pore_pressure_faces': 0,
                    })
                    hydraulic_masks.append(wet.copy())
                    with root_archive_write() as writer:
                        if writer:
                            path = STEPS_PATH.replace('.npz', '_hydraulic.npz')
                            with open(path + '.tmp', 'wb') as stream:
                                np.savez(stream, load_factor=np.array([r['load_factor'] for r in hydraulic_history]),
                                         water_depth_m=np.array([r['water_depth_m'] for r in hydraulic_history]),
                                         wet_faces=np.asarray(hydraulic_masks))
                            os.replace(path + '.tmp', path)
                    status['hydraulic_steps'] = hydraulic_history
                RESU = trial
                DETRUIRE(CONCEPT=_F(NOM=previous))
                release_loads(concepts)
                concepts = []
                break
            DETRUIRE(CONCEPT=_F(NOM=trial))
            trial = None
            release_loads(concepts)
            concepts = []
            wet, water_forces = updated_wet, updated_forces
        else:
            raise RuntimeError('Crack-water active set did not converge; no hydraulic step accepted')
    except (CA.AsterError, RuntimeError, OSError, MemoryError) as error:
        del hydraulic_history[hydraulic_count:]
        del hydraulic_masks[hydraulic_count:]
        if stop_for_resource_failure(error):
            break
        if trial is not None:
            DETRUIRE(CONCEPT=_F(NOM=trial))
        release_loads(concepts)
        detail = str(error).strip() or type(error).__name__
        retry = (bisect_load_interval(accepted_depth, trial_depth, target_level, MAX_LOAD_INTERVAL_BISECTIONS)
                 if trial_depth > accepted_depth else None)
        if retry is not None:
            midpoint, next_level = retry
            DEPTHS[target_index:target_index + 1] = [midpoint, trial_depth]
            TARGET_LEVELS[target_index:target_index + 1] = [next_level, next_level]
            status.setdefault('hydraulic_subdivisions', []).append({
                'from_water_depth_m': accepted_depth, 'failed_water_depth_m': trial_depth,
                'retry_water_depth_m': midpoint, 'level': next_level, 'reason': detail})
            status['water_depth_targets_m'] = DEPTHS
            save()
            continue
        save({'status': 'stopped', 'stopped_at_water_depth_m': trial_depth,
              'message': f'Code_Aster failed at water depth {trial_depth:g} m: {detail}'})
        break
'''
