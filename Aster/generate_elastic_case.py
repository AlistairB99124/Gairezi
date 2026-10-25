"""Work Package 4-5 of ``Aster_Porting.md``: generate the (unverified) linear-
elastic Code_Aster command file for the single-WALL baseline.

This renders ``Aster/cases/00_elastic_baseline.comm`` from live
``config.json`` values using the *same* loader functions Elmer uses
(``Elmer.regions.combined_structure.load_material``, ``load_environment``,
``load_foundation_support``), so the two solvers are guaranteed to start from
identical numbers rather than a hand-copied second source of truth.

IMPORTANT - unverified syntax: there is no Code_Aster installation in this
environment, so none of the generated ``.comm`` syntax below has been run or
checked against a live command catalogue (``CATA.py``). Every command is
cited against public Code_Aster documentation references (document codes,
e.g. ``u4.42.01``) in the comments, but a user with a working Code_Aster
install MUST review/run it before trusting it as a cross-check. This matches
``Aster_Porting.md`` Work Package 4 guidance.

The foundation support (Work Package 5) is the single biggest open risk: the
Elmer model applies a distributed, direction-dependent spring stiffness
(N/m^3) directly on the FOUNDATION face of a 3-D solid. The best documented
Code_Aster candidate for this semantics is ``AFFE_CARA_ELEM(RIGI_PARASOL=...)``
(soil/raft spring stiffness, doc u4.42.03), but that keyword's reference
usage is for a raft modelled with 2-D/shell elements, not an arbitrary 3-D
solid face - whether it accepts the FOUNDATION face of this all-hexahedral
solid mesh directly is NOT confirmed here. Per ``Aster_Porting.md`` ("stop ...
if [the catalogue] does not [support it] ... do not substitute DDL_IMPO fixed
constraints"), this script emits RIGI_PARASOL as the primary candidate with a
prominent comment block, and documents the fallback (explicit DIS_TR point
springs on the FOUNDATION face nodes) as the next thing to try if
RIGI_PARASOL rejects a 3-D face group - it does NOT fall back to a fixed
support automatically.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Elmer.regions.combined_structure import load_environment, load_foundation_support, load_material

DEFAULT_CASE_PATH = ROOT / "Aster" / "cases" / "00_elastic_baseline.comm"
DEFAULT_INPUTS_PATH = ROOT / "Aster" / "cases" / "00_elastic_baseline_inputs.json"
CONFIG_PATH = ROOT / "config.json"


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def render_comm(root: Path = ROOT) -> tuple[str, dict]:
    config_path = root / "config.json"
    material = load_material(config_path)
    loads = load_environment(config_path)
    support = load_foundation_support(config_path)

    gravity_magnitude = abs(loads.gravity_z_m_s2)
    upstream_surface_z = loads.upstream_surface_z_m
    water_gradient = loads.water_pressure_gradient_pa_per_m
    # Mirrors Elmer.structure_assembly.write_combined_solver_input: tailwater
    # elevation/datum is measured from the lowest mesh node (the deepest
    # bedrock seat), not from z=0.
    mesh_manifest_path = root / "Aster" / "results" / "00_elastic_baseline" / "mesh_manifest.json"
    if not mesh_manifest_path.is_file():
        raise FileNotFoundError(
            f"Run Aster.build_med_mesh first; missing {mesh_manifest_path}",
        )
    mesh_manifest = json.loads(mesh_manifest_path.read_text(encoding="utf-8"))
    foundation_elevation_m = mesh_manifest["coordinate_bounds"]["min_xyz_m"][2]
    tailwater_elevation_m = foundation_elevation_m + loads.tailwater_head_m
    overflow_surcharge_pa = loads.crest_surcharge_pa

    # RIGI_PARASOL's COOR_CENTRE is only used for the rotational-stiffness
    # (K_TR_*) distribution formulas; we use translation-only K_T_D_N, but
    # the keyword is still mandatory (UN_PARMI with GROUP_NO_CENTRE in the
    # AFFE_CARA_ELEM catalogue), so the midpoint of the FOUNDATION group's
    # bounding box is a safe, inert placeholder.
    foundation_bounds = mesh_manifest["groups"]["FOUNDATION"]
    foundation_centre = tuple(
        (lo + hi) / 2.0
        for lo, hi in zip(foundation_bounds["min_xyz_m"], foundation_bounds["max_xyz_m"])
    )
    # RIGI_PARASOL's VALE is the raft's GLOBAL total stiffness (N/m), which
    # Code_Aster then distributes across the GROUP_MA_POI1 nodes weighted by
    # each node's tributary area (doc u4.42.01 section 17.4.4: "it is
    # necessary that they be proportional to the area represented by the
    # node"; K_x = sum_p k_x(p)). config.json's spring_*_n_per_m3 values are
    # a per-unit-area Winkler modulus (matching Elmer's per-area "Spring"
    # BC), so they must be scaled by the FOUNDATION group's total area to
    # get the global stiffness RIGI_PARASOL expects -- passing the raw
    # modulus directly (as originally written) under-stiffens the
    # foundation by a factor of ~864 (the real FOUNDATION area in m^2) and
    # was confirmed, by comparing a real solve's DEPL field against
    # elmer_baseline.json, to produce displacements ~30-50x larger than the
    # Elmer baseline's ~8mm maximum.
    foundation_area_m2 = foundation_bounds["area_m2"]

    inputs = {
        "schema_version": 1,
        "source_config_sha256": file_hash(config_path),
        "source_mesh_manifest_sha256": file_hash(mesh_manifest_path),
        "material": {
            "density_kg_m3": material.density_kg_m3,
            "youngs_modulus_pa": material.youngs_modulus_pa,
            "poissons_ratio": material.poissons_ratio,
        },
        "gravity_m_s2": gravity_magnitude,
        "upstream_water_surface_z_m": upstream_surface_z,
        "water_pressure_gradient_pa_per_m": water_gradient,
        "tailwater_elevation_m": tailwater_elevation_m,
        "overflow_surcharge_pa": overflow_surcharge_pa,
        "foundation_elevation_m": foundation_elevation_m,
        "foundation_spring_n_per_m3": {
            "x": support.spring_x_n_per_m3,
            "y": support.spring_y_n_per_m3,
            "z": support.spring_z_n_per_m3,
        },
        "foundation_area_m2": foundation_area_m2,
        "foundation_global_stiffness_n_per_m": {
            "x": support.spring_x_n_per_m3 * foundation_area_m2,
            "y": support.spring_y_n_per_m3 * foundation_area_m2,
            "z": support.spring_z_n_per_m3 * foundation_area_m2,
        },
    }

    comm = f'''# 00_elastic_baseline.comm - Code_Aster linear-elastic cross-check of the
# Elmer single-WALL baseline (Aster_Porting.md). GENERATED by
# Aster/generate_elastic_case.py - do not hand-edit the numeric values below;
# regenerate instead so config.json stays the single source of truth.
#
# VERIFIED: run end-to-end against the real Code_Aster 17.4 install (see
# Aster_Porting.md "Implementation Status Amendment") -- MECA_STATIQUE
# completes with "ARRET NORMAL" and produces a real result.med with no
# uncaught Python exception anywhere in the full solver log
# (Aster/run_case.py writes the full stdout/stderr to stdout.log/stderr.log
# next to run_manifest.json for this exact check; its truncated 4000-char
# tail alone is not sufficient evidence of a clean run).
#
# Mesh: LIRE_MAILLAGE reads the MED mesh written by
# Aster/convert_arrays_to_med.py (mesh.med, UNITE=20) from the raw node/cell
# arrays Aster/build_med_mesh.py exports (mesh_arrays.npz). Code_Aster's
# native LIRE_MAILLAGE(FORMAT='GMSH') reader was found to not finish within
# 9+ minutes of continuous ~100% CPU on this ~101k-cell mesh (see
# Aster_Porting.md amendment); FORMAT='MED' reads the identical mesh (same
# nodes/cells/groups) in well under a second because the file is already in
# Code_Aster's native binary layout, built directly via medcoupling instead
# of being re-parsed from Gmsh ASCII text. The MED file carries named groups
# WALL (volume), FOUNDATION/UPSTREAM/DOWNSTREAM/CREST/LEFT_ABUTMENT/
# RIGHT_ABUTMENT (faces), identical to the Gmsh physical-name groups.

DEBUT(LANG='EN')

MAIL = LIRE_MAILLAGE(FORMAT='MED', UNITE=20, NOM_MED='WALL', VERI_MAIL=_F(VERIF='NON'))
# Doc u4.21.01 LIRE_MAILLAGE / FORMAT='MED'. GROUP_MA names WALL,
# FOUNDATION, UPSTREAM, DOWNSTREAM, CREST, LEFT_ABUTMENT, RIGHT_ABUTMENT.
# VERI_MAIL is explicitly disabled: Code_Aster defaults it to VERIF='OUI'
# even when LIRE_MAILLAGE doesn't set it, and that geometric re-check was
# observed to take >9 minutes on this ~101k-cell mesh when reading the
# (unrelated, slower) Gmsh path without finishing.
# Aster/validate_mesh.py already independently re-validates the exported
# mesh before this case is generated, so the extra in-solver check is
# redundant here.

MAIL = MODI_MAILLAGE(
    reuse=MAIL,
    MAILLAGE=MAIL,
    ORIE_PEAU_3D=_F(GROUP_MA=('UPSTREAM', 'DOWNSTREAM', 'CREST')),
)
# Doc u4.23.04 MODI_MAILLAGE / ORIE_PEAU_3D: (re)orients the 2-D skin face
# elements that receive PRES_REP loads so their normal consistently points
# outward from the adjacent WALL volume. PRES_REP's sign convention
# (positive = compressive, pushing into the solid) only holds if the face
# normal is outward; the raw exported mesh does not guarantee a consistent
# face winding, so this step is required, not optional.

MODELE = AFFE_MODELE(
    MAILLAGE=MAIL,
    AFFE=(
        _F(GROUP_MA='WALL', PHENOMENE='MECANIQUE', MODELISATION='3D'),
        _F(
            GROUP_MA=('UPSTREAM', 'DOWNSTREAM', 'CREST', 'FOUNDATION'),
            PHENOMENE='MECANIQUE',
            MODELISATION='3D',
        ),
        _F(GROUP_MA='FOUNDATION_POI1', PHENOMENE='MECANIQUE', MODELISATION='DIS_T'),
    ),
)
# Doc u4.41.01 AFFE_MODELE: 3-D continuum mechanics on the WALL volume group,
# plus the 2-D skin face groups that later receive a surface load (PRES_REP
# on UPSTREAM/DOWNSTREAM/CREST, RIGI_PARASOL on FOUNDATION). Code_Aster
# requires every GROUP_MA referenced by a load/elemental characteristic to
# be present in the model (same MODELISATION='3D'; it detects the lower
# geometric dimension itself) -- without this, MECA_STATIQUE/AFFE_CARA_ELEM
# raise MODELISA6_96 ("ces mailles n'appartiennent pas au modele"). The
# FOUNDATION_POI1 point elements (one per FOUNDATION node, built by
# Aster/convert_arrays_to_med.py) need their own MODELISATION='DIS_T'
# (discrete translational) entry, matching RIGI_PARASOL's GROUP_MA_POI1
# requirement below. The current model has no PLINTH/HAUNCH split (see
# Aster_Porting.md "Implementation Status Amendment").

CONCRETE = DEFI_MATERIAU(
    ELAS=_F(
        E={material.youngs_modulus_pa:.12g},
        NU={material.poissons_ratio:.12g},
        RHO={material.density_kg_m3:.12g},
    ),
)
# Doc u4.43.01 DEFI_MATERIAU / ELAS: isotropic linear elasticity, matching
# config.json material.concrete (youngs_modulus, poisson_ratio, density) -
# the same values Elmer.regions.combined_structure.load_material reads.

CHMAT = AFFE_MATERIAU(
    MAILLAGE=MAIL,
    AFFE=_F(GROUP_MA='WALL', MATER=CONCRETE),
)
# Doc u4.43.03 AFFE_MATERIAU.

GRAVITE = AFFE_CHAR_MECA(
    MODELE=MODELE,
    PESANTEUR=_F(GRAVITE={gravity_magnitude:.12g}, DIRECTION=(0.0, 0.0, -1.0)),
)
# Doc u4.44.01 AFFE_CHAR_MECA / PESANTEUR: magnitude from abs(config.json
# loads.gravity), direction matches Elmer's Gravity(4) = 0 0 -1 |g|.

# Elevation-dependent hydrostatic pressure functions (Z is the vertical
# coordinate; crest is Z=0; FORMULE doc u4.31.03). PRES is a pressure acting
# normal to the surface, positive = compressive (pushing into the solid),
# matching a positive hydrostatic pressure. Clamped to zero above the free
# surface, matching Elmer's "(tx < surface)" MATC guard.
UPSTREAM_PRES = FORMULE(
    NOM_PARA='Z',
    VALE='{water_gradient:.12g} * max({upstream_surface_z:.12g} - Z, 0.0)',
)
DOWNSTREAM_PRES = FORMULE(
    NOM_PARA='Z',
    VALE='{water_gradient:.12g} * max({tailwater_elevation_m:.12g} - Z, 0.0)',
)

PRESSIONS = AFFE_CHAR_MECA_F(
    MODELE=MODELE,
    PRES_REP=(
        _F(GROUP_MA='UPSTREAM', PRES=UPSTREAM_PRES),
        _F(GROUP_MA='DOWNSTREAM', PRES=DOWNSTREAM_PRES),
    ),
)
# Doc u4.44.01 AFFE_CHAR_MECA_F / PRES_REP with a FORMULE argument: the
# standard way to apply an elevation-varying pressure without exporting a
# per-node pressure field. Mirrors Elmer's UpstreamHydrostaticPressure /
# DownstreamTailwater "Normal Force" boundary conditions in
# Elmer/structure_assembly.write_combined_solver_input.

CREST_SURCHARGE = AFFE_CHAR_MECA(
    MODELE=MODELE,
    PRES_REP=_F(GROUP_MA='CREST', PRES={overflow_surcharge_pa:.12g}),
)
# Constant overflow surcharge pressure on CREST, matching Elmer's
# CrestOverflowSurcharge boundary condition (constant Normal Force).

# --- Foundation support --------------------------------------------------
# RIGI_PARASOL (doc u4.42.03, AFFE_CARA_ELEM): Code_Aster's mechanism for a
# distributed foundation spring "carpet", closest in intent to Elmer's
# "Spring 1/2/3" boundary condition on the FOUNDATION face
# (geometry.bedrock.support.spring_n_per_m3 in config.json). Verified
# against the installed 17.4 catalogue (code_aster/Cata/Commands/
# affe_cara_elem.py): it requires an explicit POI1 (single-node) element at
# every FOUNDATION node (GROUP_MA_POI1, built by Aster/convert_arrays_to_med.py
# into the FOUNDATION_POI1 group); COEF_GROUP (one constant coefficient per
# GROUP_MA, here 1.0); and K_T_D_N (translation-only, no rotational
# coupling, matching Elmer's purely translational springs) with exactly 3
# VALE entries (Kx, Ky, Kz). Per doc u4.42.01 section 17.4.4, VALE is the
# raft's GLOBAL total stiffness (N/m), which Code_Aster distributes across
# the GROUP_MA_POI1 nodes weighted by each node's tributary area -- it is
# NOT a per-unit-area modulus, so config.json's spring_*_n_per_m3 Winkler
# modulus is scaled by the FOUNDATION group's total area here
# (foundation_global_stiffness_n_per_m) to get the value RIGI_PARASOL
# expects. (Passing the raw per-area modulus directly, as an earlier
# version of this script did, under-stiffened the foundation by the
# FOUNDATION area factor (~864) and was confirmed -- by comparing a real
# solve's DEPL field against elmer_baseline.json -- to produce displacements
# ~30-50x larger than the Elmer baseline's ~8mm maximum.) COOR_CENTRE is
# mandatory syntax even for K_T_D_N (it only matters for the rotational
# K_TR_* variants), so the FOUNDATION group's bounding-box midpoint is used
# as an inert placeholder.
CARA_FOUNDATION = AFFE_CARA_ELEM(
    MODELE=MODELE,
    RIGI_PARASOL=_F(
        GROUP_MA='FOUNDATION',
        GROUP_MA_POI1='FOUNDATION_POI1',
        COEF_GROUP=(1.0,),
        REPERE='GLOBAL',
        CARA='K_T_D_N',
        VALE=(
            {support.spring_x_n_per_m3 * foundation_area_m2:.12g},
            {support.spring_y_n_per_m3 * foundation_area_m2:.12g},
            {support.spring_z_n_per_m3 * foundation_area_m2:.12g},
        ),
        COOR_CENTRE=({foundation_centre[0]:.12g}, {foundation_centre[1]:.12g}, {foundation_centre[2]:.12g}),
    ),
)


RESU = MECA_STATIQUE(
    MODELE=MODELE,
    CHAM_MATER=CHMAT,
    CARA_ELEM=CARA_FOUNDATION,
    EXCIT=(
        _F(CHARGE=GRAVITE),
        _F(CHARGE=PRESSIONS),
        _F(CHARGE=CREST_SURCHARGE),
    ),
)
# Doc u4.51.01 MECA_STATIQUE: single linear static load case combining
# gravity, the two elevation-dependent pressure loads, and the crest
# surcharge - matching the Elmer SIF's single steady-state solve.

RESU = CALC_CHAMP(
    reuse=RESU,
    RESULTAT=RESU,
    CONTRAINTE=('SIEF_ELGA', 'SIGM_NOEU'),
    CRITERES=('SIEQ_NOEU',),
)
# Doc u4.81.02 CALC_CHAMP: recover nodal stresses (SIGM_NOEU) and nodal
# principal/equivalent stress invariants (SIEQ_NOEU) for comparison against
# the Elmer nodal "stress_xx..stress_xz" / principal-stress fields.

IMPR_RESU(
    FORMAT='MED',
    UNITE=80,
    RESU=_F(RESULTAT=RESU, NOM_CHAM=('DEPL', 'SIGM_NOEU', 'SIEQ_NOEU')),
)
# Doc u4.91.01 IMPR_RESU / MED: write the result in MED format (the
# authoritative Code_Aster result per Aster_Porting.md "Target Layout");
# Aster/run_case.py converts this to VTU afterwards for ParaView, never the
# other way around.

FIN()
'''
    return comm, inputs


def generate(root: Path = ROOT, case_path: Path = DEFAULT_CASE_PATH, inputs_path: Path = DEFAULT_INPUTS_PATH) -> dict:
    comm, inputs = render_comm(root)
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text(comm, encoding="utf-8", newline="\n")
    inputs["case_sha256"] = hashlib.sha256(comm.encode("utf-8")).hexdigest()
    inputs_path.write_text(json.dumps(inputs, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return inputs


def main() -> None:
    inputs = generate()
    print(json.dumps(inputs, indent=2))


if __name__ == "__main__":
    main()
