from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "calibration_case.json"
OUTPUT_PATH = ROOT / "dam_model_calibration.sif"
MESH_PATH = "mesh"


def _bool_scale(flag: bool) -> float:
    return 1.0 if flag else 0.0


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing calibration config: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text())


def _support_block(config: dict) -> str:
    support_type = config["support_type"].strip().lower()
    base_spring = config["support"]["base_spring"]
    abutment_spring = config["support"]["abutment_spring"]

    if support_type == "fixed":
        bc1 = """Boundary Condition 1
  Name = \"BedrockBaseFixed\"
  Target Boundaries(1) = 1
  Displacement 1 = Real 0.0
  Displacement 2 = Real 0.0
  Displacement 3 = Real 0.0
End
"""
    elif support_type == "spring":
        bc1 = f"""Boundary Condition 1
  Name = \"BedrockBaseSpring\"
  Target Boundaries(1) = 1
  Spring 1 = Real {base_spring[0]}
  Spring 2 = Real {base_spring[1]}
  Spring 3 = Real {base_spring[2]}
End
"""
    else:
        raise ValueError("support_type must be 'spring' or 'fixed'")

    bc5 = f"""Boundary Condition 5
  Name = \"LeftAbutmentSpring\"
  Target Boundaries(1) = 5
  Spring 1 = Real {abutment_spring[0]}
  Spring 2 = Real {abutment_spring[1]}
  Spring 3 = Real {abutment_spring[2]}
End
"""

    bc6 = f"""Boundary Condition 6
  Name = \"RightAbutmentSpring\"
  Target Boundaries(1) = 6
  Spring 1 = Real {abutment_spring[0]}
  Spring 2 = Real {abutment_spring[1]}
  Spring 3 = Real {abutment_spring[2]}
End
"""

    return f"{bc1}\n{bc5}\n{bc6}"


def build_sif(config: dict) -> str:
    material = config["material"]
    water = config["water"]
    uplift = config.get("uplift", {})

    rho_concrete = float(material["density_kg_m3"])
    e_modulus = float(material["youngs_modulus_pa"])
    nu = float(material["poisson_ratio"])

    rho_water = float(water["density_kg_m3"])
    g = float(water["gravity_m_s2"])
    h_up = float(water["upstream_head_m"])
    h_down = float(water["downstream_head_m"])
    h_crest = float(water["crest_head_m"])
    crest_overflow_pressure_pa = water.get("crest_overflow_pressure_pa")
    pressure_datum_z = float(water.get("pressure_datum_z_m", 0.0))
    target_peak_upstream_pa = water.get("target_peak_upstream_pressure_pa")

    use_gravity = _bool_scale(bool(config["include_gravity"]))
    use_hydro = _bool_scale(bool(config["include_hydrostatic"]))
    use_crest = _bool_scale(bool(config["include_crest_surcharge"]))
    use_uplift = _bool_scale(bool(config.get("include_uplift", False)))
    uplift_head = float(uplift.get("head_m", 0.0))
    uplift_pressure_factor = float(uplift.get("pressure_factor", 1.0))

    gravity_bodyforce = -rho_concrete * g * use_gravity

    implied_peak_upstream_pa = rho_water * g * h_up
    if target_peak_upstream_pa is not None:
      if abs(float(target_peak_upstream_pa) - implied_peak_upstream_pa) > 1.0:
        raise ValueError(
          "target_peak_upstream_pressure_pa must equal rho*g*upstream_head_m at the pressure datum. "
          f"target={float(target_peak_upstream_pa):.3f} Pa, implied={implied_peak_upstream_pa:.3f} Pa"
        )

    # Apply pressure as normal traction to keep hydrostatic effects explicit and active.
    upstream_surface_z = pressure_datum_z + h_up
    downstream_surface_z = pressure_datum_z + h_down
    upstream_expr = (
      f"-{rho_water} * {g} * ({upstream_surface_z} - tx) * (tx < {upstream_surface_z}) * {use_hydro}"
    )
    downstream_expr = (
      f"-{rho_water} * {g} * ({downstream_surface_z} - tx) * (tx < {downstream_surface_z}) * {use_hydro}"
    )
    if crest_overflow_pressure_pa is not None:
      crest_force = -float(crest_overflow_pressure_pa) * use_crest
    else:
      crest_force = -rho_water * g * h_crest * use_crest
    uplift_force = rho_water * g * uplift_head * uplift_pressure_factor * use_uplift

    support_block = _support_block(config)
    results_dir = config.get("results_directory", "results_calibration")

    uplift_block = ""
    if use_uplift > 0.0:
      uplift_block = f'''Boundary Condition 7
    Name = "BaseUplift"
    Target Boundaries(1) = 1
    Normal Force = Real {uplift_force}
  End
  '''

    return f"""Header
  CHECK KEYWORDS Warn
  Mesh DB \"{MESH_PATH}\" \"/\"
  Include Path \"\"
  Results Directory \"{results_dir}\"
End

Simulation
  Max Output Level = 5
  Coordinate System = Cartesian 3D
  Simulation Type = Steady State
  Steady State Max Iterations = 1
  Output Intervals = 1
End

Constants
  Gravity(4) = Real 0 0 -1 {g}
  Stefan Boltzmann = 5.670374419e-8
End

Body 1
  Name = \"DamBody\"
  Equation = 1
  Material = 1
  Body Force = 1
End

Body Force 1
  Stress Bodyforce 3 = Real {gravity_bodyforce}
End

Equation 1
  Name = \"Elasticity\"
  Active Solvers(2) = 1 2
End

Solver 1
  Equation = \"StressSolve\"
  Procedure = \"StressSolve\" \"StressSolver\"
  Variable = \"Displacement\"
  Variable DOFs = 3
  Calculate Stresses = Logical True
  Calculate Principal = Logical True
  Nonlinear System Max Iterations = 20
  Linear System Solver = Direct
  Linear System Direct Method = UMFPACK
End

Material 1
  Name = \"Concrete\"
  Youngs Modulus = {e_modulus}
  Poisson Ratio = {nu}
  Density = {rho_concrete}
End

{support_block}

Boundary Condition 2
  Name = \"UpstreamHydrostaticPressure\"
  Target Boundaries(1) = 2
  ! Hydrostatic datum-aware expression: pressure_datum_z_m + upstream_head_m defines free-surface elevation.
  ! At tx = pressure_datum_z_m, applied pressure is rho*g*upstream_head_m = {implied_peak_upstream_pa:.3f} Pa.
  Normal Force = Variable Coordinate 3
    Real MATC \"{upstream_expr}\"
End

Boundary Condition 3
  Name = \"DownstreamTailwater\"
  Target Boundaries(1) = 3
  ! Hydrostatic datum-aware expression: pressure_datum_z_m + downstream_head_m defines free-surface elevation.
  Normal Force = Variable Coordinate 3
    Real MATC \"{downstream_expr}\"
End

Boundary Condition 4
  Name = \"CrestOverflowSurcharge\"
  Target Boundaries(1) = 4
  Normal Force = Real {crest_force}
End

{uplift_block}

Solver 2
  Equation = \"ResultOutput\"
  Procedure = \"ResultOutputSolve\" \"ResultOutputSolver\"
  Output File Name = \"dam_results\"
  Vtu Format = Logical True
End
"""


def main() -> None:
    config = _load_config()
    sif_text = build_sif(config)
    OUTPUT_PATH.write_text(sif_text)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
