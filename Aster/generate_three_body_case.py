"""Generate a source-pinned elastic case with all joints and directional supports.

Material/body-force assignments are checked against the saved SIF. Pressure
datums and spring coefficients come from that SIF, not reconstructed geometry.
Surface-spring integration remains lumped; see Aster_Porting.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Elmer.regions.combined_structure import load_environment, load_material

DEFAULT_CASE_PATH = ROOT / "Aster" / "cases" / "01_three_body_baseline.comm"
DEFAULT_INPUTS_PATH = ROOT / "Aster" / "cases" / "01_three_body_baseline_inputs.json"
DEFAULT_MESH_MANIFEST_PATH = ROOT / "Aster" / "results" / "01_three_body_baseline" / "mesh_manifest.json"
CONFIG_PATH = ROOT / "config.json"


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_bedrock_material(path: Path) -> dict:
    values = json.loads(path.read_text(encoding="utf-8"))["material"]["bedrock"]
    return {
        "density_kg_m3": float(values["density"]),
        "youngs_modulus_pa": float(values["youngs_modulus"]),
        "poissons_ratio": float(values["poisson_ratio"]),
    }


def load_rock_outer_spring(path: Path) -> dict:
    # material.bedrock.support.spring_n_per_m3 (not geometry.bedrock.support,
    # which is the foundation/bedrock interface stiffness consumed only by
    # the three-body Elmer model's "bonded" check) -- this is the rock-box
    # OUTER boundary roller stiffness (Elmer.three_body_diagnostic.rock_outer_support).
    values = json.loads(path.read_text(encoding="utf-8"))["material"]["bedrock"]["support"]["spring_n_per_m3"]
    return {axis: float(values[axis]) for axis in ("x", "y", "z")}


def render_comm(root: Path = ROOT, mesh_manifest_path: Path = DEFAULT_MESH_MANIFEST_PATH) -> tuple[str, dict]:
    mesh_manifest = json.loads(mesh_manifest_path.read_text(encoding="utf-8"))
    config_path = Path(mesh_manifest["source_config"])
    if file_hash(config_path) != mesh_manifest["source_hashes"]["config.json"]:
        raise ValueError("Saved source configuration changed since mesh export")
    from Aster.source_contract import audit_source_materials
    audit_source_materials(Path(mesh_manifest["source_mesh"]))
    concrete = load_material(config_path)
    bedrock = load_bedrock_material(config_path)
    rock_outer_spring = load_rock_outer_spring(config_path)
    loads = load_environment(config_path)

    gravity_magnitude = abs(loads.gravity_z_m_s2)
    source_pressures = mesh_manifest["source_pressures"]
    upstream_surface_z = source_pressures["UpstreamHydrostaticPressure"]["surface_z_m"]
    water_gradient = source_pressures["UpstreamHydrostaticPressure"]["gradient_pa_per_m"]
    if water_gradient != source_pressures["DownstreamTailwater"]["gradient_pa_per_m"]:
        raise ValueError("Different upstream/downstream pressure gradients are not supported")
    foundation_elevation_m = mesh_manifest["groups"]["CONCRETE"]["min_xyz_m"][2]
    tailwater_elevation_m = source_pressures["DownstreamTailwater"]["surface_z_m"]
    overflow_surcharge_pa = mesh_manifest["source_crest_pressure_pa"]

    rock_bottom_area_m2 = mesh_manifest["groups"]["ROCK_BOTTOM"]["area_m2"]
    rock_bottom_global_stiffness_n_per_m = rock_outer_spring["z"] * rock_bottom_area_m2

    wall_foundation_springs = mesh_manifest["wall_foundation_springs"]
    wall_foundation_spring_group_all = mesh_manifest["wall_foundation_spring_group_all"]
    wall_foundation_spring_entries = [
        {
            "group": spring["group"],
            "vale": tuple(spring["stiffness_n_per_m"]),
        }
        for spring in wall_foundation_springs
    ]

    inputs = {
        "schema_version": 1,
        "source_config_sha256": file_hash(config_path),
        "source_hashes": mesh_manifest["source_hashes"],
        "source_mesh_manifest_sha256": file_hash(mesh_manifest_path),
        "material": {
            "concrete": {
                "density_kg_m3": concrete.density_kg_m3,
                "youngs_modulus_pa": concrete.youngs_modulus_pa,
                "poissons_ratio": concrete.poissons_ratio,
            },
            "bedrock": bedrock,
        },
        "gravity_m_s2": gravity_magnitude,
        "upstream_water_surface_z_m": upstream_surface_z,
        "water_pressure_gradient_pa_per_m": water_gradient,
        "tailwater_elevation_m": tailwater_elevation_m,
        "overflow_surcharge_pa": overflow_surcharge_pa,
        "foundation_elevation_m": foundation_elevation_m,
        "rock_outer_spring_n_per_m3": rock_outer_spring,
        "rock_bottom_area_m2": rock_bottom_area_m2,
        "rock_bottom_global_stiffness_n_per_m": rock_bottom_global_stiffness_n_per_m,
        "source_interface_coefficients_n_per_m3": {
            spring["interface"]: [
                value / spring["tributary_area_m2"] for value in spring["stiffness_n_per_m"]
            ] for spring in wall_foundation_springs
        },
        "wall_foundation_spring_pairs": len(wall_foundation_spring_entries),
        "wall_foundation_spring_total_area_m2": mesh_manifest["wall_foundation_spring_total_area_m2"],
        "simplifications_vs_elmer": mesh_manifest["simplifications_vs_elmer"],
    }

    wall_foundation_discret_entries = ",\n        ".join(
        f"_F(GROUP_MA='{entry['group']}', REPERE='GLOBAL', CARA='K_T_D_L', "
        f"VALE=({entry['vale'][0]:.12g}, {entry['vale'][1]:.12g}, {entry['vale'][2]:.12g}))"
        for entry in wall_foundation_spring_entries
    )
    outer_discret_entries = ",\n        ".join(
        f"_F(GROUP_MA='{spring['group']}', REPERE='GLOBAL', CARA='K_T_N', SYME='NON', "
        f"VALE={tuple(spring['matrix_n_per_m'][i][j] for j in range(3) for i in range(3))!r})"
        for spring in mesh_manifest["support_springs"]
    )

    comm = f'''# 01_three_body_baseline.comm - Code_Aster linear-elastic cross-check of the
# CANONICAL Elmer wall+foundation+bedrock model (Elmer.three_body_solver;
# Gairezi/README.md "Integrated wall/foundation/bedrock solver"), as opposed
# to 00_elastic_baseline which only ever ported the legacy wall-only model.
# GENERATED by Aster/generate_three_body_case.py - do not hand-edit the
# numeric values below. Inputs are pinned to the saved source run.
# All source volume cells and split joint nodes are retained. Surface
# springs use lumped tributary areas, not a consistent Galerkin assembly.

DEBUT(LANG='EN')

MAIL = LIRE_MAILLAGE(FORMAT='MED', UNITE=20, NOM_MED='THREE_BODY', VERI_MAIL=_F(VERIF='NON'))
# Doc u4.21.01 LIRE_MAILLAGE / FORMAT='MED'. GROUP_MA names CONCRETE,
# BEDROCK (volumes); UPSTREAM, DOWNSTREAM, CREST, ROCK_BOTTOM, ROCK_SIDES
# (faces); OUTER_SPRINGS (points). VERI_MAIL disabled for the same reason
# as 00_elastic_baseline (avoids a slow geometric re-check on a mesh this
# size; Aster/validate_mesh.py already independently re-validates it).

MAIL = MODI_MAILLAGE(
    reuse=MAIL,
    MAILLAGE=MAIL,
    ORIE_PEAU_3D=_F(GROUP_MA=('UPSTREAM', 'DOWNSTREAM', 'CREST')),
)
# Doc u4.23.04 MODI_MAILLAGE / ORIE_PEAU_3D: same outward-normal fix-up as
# 00_elastic_baseline, required before PRES_REP's sign convention is valid.

MODELE = AFFE_MODELE(
    MAILLAGE=MAIL,
    AFFE=(
        _F(
            GROUP_MA=(
                'CONCRETE', 'BEDROCK', 'UPSTREAM', 'DOWNSTREAM', 'CREST',
                'ROCK_BOTTOM', 'ROCK_SIDES',
            ),
            PHENOMENE='MECANIQUE',
            MODELISATION='3D',
        ),
        _F(GROUP_MA='OUTER_SPRINGS', PHENOMENE='MECANIQUE', MODELISATION='DIS_T'),
        _F(GROUP_MA='{wall_foundation_spring_group_all}', PHENOMENE='MECANIQUE', MODELISATION='DIS_T'),
    ),
)
# DIS_T covers both wall joints, the wall/foundation interface and all
# exterior normal-spring matrices.

CONCRETE_MAT = DEFI_MATERIAU(
    ELAS=_F(
        E={concrete.youngs_modulus_pa:.12g},
        NU={concrete.poissons_ratio:.12g},
        RHO={concrete.density_kg_m3:.12g},
    ),
)
BEDROCK_MAT = DEFI_MATERIAU(
    ELAS=_F(
        E={bedrock["youngs_modulus_pa"]:.12g},
        NU={bedrock["poissons_ratio"]:.12g},
        RHO={bedrock["density_kg_m3"]:.12g},
    ),
)
# Doc u4.43.01 DEFI_MATERIAU / ELAS: config.json material.concrete (wall +
# foundation, both concrete) and material.bedrock (rock), matching the same
# values Elmer.regions.combined_structure / Elmer.graded_bedrock read.

CHMAT = AFFE_MATERIAU(
    MAILLAGE=MAIL,
    AFFE=(
        _F(GROUP_MA='CONCRETE', MATER=CONCRETE_MAT),
        _F(GROUP_MA='BEDROCK', MATER=BEDROCK_MAT),
    ),
)
# Doc u4.43.03 AFFE_MATERIAU.

GRAVITE = AFFE_CHAR_MECA(
    MODELE=MODELE,
    PESANTEUR=_F(GRAVITE={gravity_magnitude:.12g}, DIRECTION=(0.0, 0.0, -1.0)),
)
# Doc u4.44.01 AFFE_CHAR_MECA / PESANTEUR: applies to the whole model (both
# materials/densities), matching Elmer's "rock self-weight and dam loads
# applied together" (diagnostic.json assumptions).

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
# Doc u4.44.01 AFFE_CHAR_MECA_F / PRES_REP with a FORMULE argument, identical
# in form to 00_elastic_baseline's hydrostatic loads.

CREST_SURCHARGE = AFFE_CHAR_MECA(
    MODELE=MODELE,
    PRES_REP=_F(GROUP_MA='CREST', PRES={overflow_surcharge_pa:.12g}),
)

CARA_ROCK = AFFE_CARA_ELEM(
    MODELE=MODELE,
    DISCRET=(
        {wall_foundation_discret_entries},
        {outer_discret_entries}
    ),
)
# K_T_D_L: saved per-axis interface coefficient times tributary area.
# K_T_N: global k*n*n.T matrices, preserving free tangential directions.
# Nine-entry SYME='NON' encoding is column-major (installed zzzz274f test).

RESU = MECA_STATIQUE(
    MODELE=MODELE,
    CHAM_MATER=CHMAT,
    CARA_ELEM=CARA_ROCK,
    EXCIT=(
        _F(CHARGE=GRAVITE),
        _F(CHARGE=PRESSIONS),
        _F(CHARGE=CREST_SURCHARGE),
    ),
    SOLVEUR=_F(METHODE='MULT_FRONT'),
)
# Doc u4.51.01 MECA_STATIQUE: single linear static load case, same structure
# as 00_elastic_baseline. SOLVEUR/METHODE='MULT_FRONT' (Code_Aster's own
# built-in multifrontal solver) instead of the MUMPS default: on this mesh
# MUMPS 5.6.2 (this Code_Aster 17.4 Windows distribution's bundled version)
# reports a clean factorization (INFOG(1)=0, no null pivots detected) but
# the solve/iterative-refinement step silently returns an all-NaN solution
# vector (RINFOG(5) "NORM OF Computed SOLUT" = NaN) with no error or alarm
# anywhere in the log -- confirmed via a systematic isolation (see
# Aster_Porting.md Amendment 3): ruled out mesh validity, element Jacobian
# conditioning, duplicate/degenerate elements, node connectivity, and the
# two-material (CONCRETE/BEDROCK) assembly as root causes (a CONCRETE-only
# sub-case, fixed at its real interface nodes with BEDROCK, still produced
# 100% NaN under MUMPS); switching only the solver to MULT_FRONT on that
# exact same sub-case produced a clean, physically sane result (max
# displacement ~0.5mm). This isolates the bug to MUMPS's solve phase on
# this specific problem, not the model.

RESU = CALC_CHAMP(
    reuse=RESU,
    RESULTAT=RESU,
    GROUP_MA=('CONCRETE', 'BEDROCK'),
    CONTRAINTE=('SIEF_ELGA', 'SIGM_NOEU'),
)
# GROUP_MA restricts CALC_CHAMP to the two 3-D volume groups, excluding the
# TRIA3 surface-load groups (UPSTREAM/DOWNSTREAM/CREST/ROCK_BOTTOM/
# ROCK_SIDES) and the ROCK_BOTTOM_POI1 discrete springs, which have no real
# stress tensor. CRITERES=('SIEQ_NOEU',) (equivalent/principal stresses) is
# intentionally OMITTED here: the real solved three-body mesh contains a
# small number of severely sliver bedrock tetrahedra (max edge / min edge
# ratio up to ~190, vs <3 in the concrete) inherited from Elmer's own
# graded_bedrock.py mesh generation, and Code_Aster's SIEQ_NOEU eigenvalue
# maths (fgequi_/te0335_) raises a Windows floating-point-exception crash on
# at least one of them. SIGM_NOEU (plain nodal stress-tensor components, no
# eigen-decomposition) is numerically robust to this and is what the
# Elmer-vs-Aster comparison already uses; see Aster_Porting.md Amendment 3.

IMPR_RESU(
    FORMAT='MED',
    UNITE=80,
    RESU=_F(RESULTAT=RESU, NOM_CHAM=('DEPL', 'SIGM_NOEU')),
)

FIN()
'''
    return comm, inputs


def generate(root: Path = ROOT, case_path: Path = DEFAULT_CASE_PATH, inputs_path: Path = DEFAULT_INPUTS_PATH,
             mesh_manifest_path: Path = DEFAULT_MESH_MANIFEST_PATH) -> dict:
    comm, inputs = render_comm(root, mesh_manifest_path)
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
