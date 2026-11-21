# Wall-only Code_Aster cracking: first-pass result

## Withdrawn no-tension shortcut

The initial wall-damage acceptance rule was too weak because it inspected only
points already classified as cracked. A subsequent whole-wall run was performed
after changing the constitutive tensile activation strength to 1 Pa. That run
must **not** be accepted as a crack result.

The withdrawn run completed through 100% load in 112.9 seconds with:

- numerical tensile activation strength: 1 Pa
  (`no_tension_strength_fraction = 0.00001` of the configured 0.10 MPa `FT`);
- whole-wall tensile tolerance: 1 Pa
  (`residual_tension_fraction = 0.00001`);
- maximum principal tension anywhere in the wall:
  `1.05e-9 Pa`, effectively numerical zero;
- zero points above the 1 Pa whole-wall tension tolerance;
- maximum principal compression: -5.714 MPa;
- 19,883 of 19,890 wall cells at `V1 >= 0.95`;
- 19,883 of 19,890 cells damaged to `V1 = 1.0`;
- maximum displacement component: 10.018 mm;
- `zero_tension_acceptance.passed = true`;
- native nonlinear convergence gate passed at every requested load state.

This result is a numerical no-tension stress redistribution, not spontaneous
cracking. The 1 Pa value removed the configured 0.10 MPa concrete tensile
strength; 99.96% of the wall then damaged because any numerical positive
principal strain activated the law. It does not identify a localized crack path
or a crack opening.

The full-damage envelope at final load was:

| Direction | Minimum | Maximum | Span |
|---|---:|---:|---:|
| Chainage | 5.279 m | 157.467 m | 152.187 m |
| Elevation | -29.689 m | -0.418 m | 29.271 m |
| Radius | 76.327 m | 79.663 m | 3.336 m |

This almost-whole-wall damage extent is expected from the explicit no-tension
idealisation: continuum points are allowed to lose all tensile resistance once
even a numerical tensile demand appears. It is not evidence that 99.96% of the
dam physically contains one visible, open crack. The `V1` field remains a local
damage measure, not a crack-opening or crack-width field.

The result is invalid for the user's crack-analysis objective. A valid follow-on
must retain the physical concrete tensile strength and use discrete/cohesive
interfaces, with a native normal-jump field for opening. The Code_Aster cohesive
interface path is being evaluated for this purpose.

## Earlier provisional numerical run (superseded for acceptance)

The first complete wall-only cracking run finished successfully through 100% of the
configured gravity and water loads. It used `ENDO_ISOT_BETON` with a Newton
solution over a linear TETRA4 mesh of the concrete wall only. The foundation and
bedrock volumes were absent from the mesh and were represented by distributed
elastic springs at the wall base.

At full load:

- 1,598 of 19,890 wall cells were at or above the 95% damage threshold (8.03%).
- Maximum damage was 1.0.
- The maximum principal tensile stress among those damaged points was
  3.015 kPa.
- The residual-tension acceptance tolerance was 5.0 kPa, so the requested
  zero-tension criterion passed at the damaged points.
- Maximum wall displacement was 8.651 mm.
- The run completed in 450.5 seconds (7 minutes 30.5 seconds).

This is a successful **numerical first pass**, not an engineering validation.
The result demonstrates that the wall-only damage workflow can converge to full
load and relieve tensile stress in the damaged regions. It does not yet establish
the physical crack width, the mesh-independent crack extent, or a test-calibrated
concrete fracture response.

## Analysis definition

| Item | First-pass value |
|---|---:|
| Solver | Code_Aster 17.4.0 |
| Constitutive law | `ENDO_ISOT_BETON` |
| Solution method | Newton |
| Element model | 3D linear TETRA4 wall |
| Wall cells | 19,890 |
| Wall nodes | 4,804 |
| Foundation cells | 0 |
| Bedrock cells | 0 |
| Wall-base faces carrying spring support | 564 |
| Load output states | 0.0 to 1.0 in increments of 0.1 |
| Internal nonlinear load increment | 0.01 |

The wall began undamaged. The final report therefore describes newly developed
damage rather than the opening of a pre-seeded joint.

## Boundary replacement for the removed bodies

`bedrock_analysis.wall_only_body` was `true`, so the foundation and bedrock were
not geometric bodies in the damage model. The wall-base boundary used the
distributed spring stiffness from `bedrock_analysis.spring_n_per_m3`:

| Direction | Stiffness |
|---|---:|
| X | 20.0 GPa/m (`20.0e9 N/m³`) |
| Y | 20.0 GPa/m (`20.0e9 N/m³`) |
| Z | 53.333 GPa/m (`53.333e9 N/m³`) |

The represented base area was 708.075 m². There are no foundation or bedrock
stress results in this reduced model.

## Material and softening assumptions

| Property | Value |
|---|---:|
| Young's modulus | 35.0 GPa |
| Poisson ratio | 0.2 |
| Tensile strength `FT` | 0.10 MPa |
| Compressive strength `FC` | 30.0 MPa |
| Configured cohesive fracture energy | 150 J/m² |
| Post-peak slope `D_SIGM_EPSI` | -3.5 GPa (`-0.1E`) |

The original `D_SIGM_EPSI = -E` trial was not numerically usable: Newton
stagnated at the first damaging state. The completed run used a softer, more
regularized post-peak slope of `-0.1E`. This remains a provisional numerical
softening assumption and must be included in sensitivity testing.

## Full-load stress result

The full-load wall result contained:

| Measure | Result |
|---|---:|
| Maximum principal tensile stress anywhere in the wall | 1.615 MPa |
| Minimum principal stress (peak compression) | -4.791 MPa |
| Maximum principal stress at damaged points (`V1 >= 0.95`) | 0.003015 MPa |
| Maximum component of displacement | 8.651 mm |

The 1.615 MPa value must not be compared directly with the 0.003015 MPa cracked
residual. The former is the global peak, including undamaged wall points. The
latter is the relevant zero-tension result at points classified as damaged.

The residual cracked tension was 3.015 kPa, below the 5.0 kPa acceptance limit.
The limit represents 5% of the configured 0.10 MPa tensile strength.

The stress result at the early load states was:

| Load factor | Global maximum principal tension | Peak compression | Maximum tension at damaged points |
|---:|---:|---:|---:|
| 0.1 | 0.341 MPa | -0.473 MPa | 0.004369 MPa |
| 0.2 | 0.464 MPa | -0.941 MPa | 0.003723 MPa |
| 0.3 | 0.663 MPa | -1.422 MPa | 0.002698 MPa |
| 0.4 | 0.925 MPa | -1.903 MPa | 0.004471 MPa |
| 0.5 | 1.094 MPa | -2.372 MPa | 0.003282 MPa |
| 1.0 | 1.615 MPa | -4.791 MPa | 0.003015 MPa |

## Damage initiation and development

Damage was present from the 10% output state onward:

| Load factor | Cells above `V1 = 0.95` |
|---:|---:|
| 0.0 | 0 |
| 0.1 | 355 |
| 0.2 | 674 |
| 0.3 | 1,099 |
| 0.4 | 1,215 |
| 0.5 | 1,352 |
| 0.6 | 1,441 |
| 0.7 | 1,523 |
| 0.8 | 1,554 |
| 0.9 | 1,582 |
| 1.0 | 1,598 |

The early damage count indicates that local wall tensile demand reaches the
configured 0.10 MPa strength during the first 10% of the load ramp. The damaged
extent then grows as the load increases, with most of the final damaged count
already present by 70% load.

## First-pass crack/damage extent

At full load, the set of wall points with `V1 >= 0.95` occupied:

| Direction | Minimum | Maximum | Span |
|---|---:|---:|---:|
| Chainage | 36.566 m | 143.408 m | 106.842 m |
| Elevation | -29.689 m | -12.127 m | 17.563 m |
| Radius | 76.329 m | 79.663 m | 3.334 m |

These extents show that the first-pass damage is concentrated near the lower
portion of the wall but is not a single geometric crack surface. The radial
extent nearly spans the 4 m wall thickness at the reported damaged coordinates.
The chainage range is a bounding envelope and should not be interpreted as one
continuous 106.8 m crack.

`V1` is a local continuum damage measure. It is not a crack width, crack
opening displacement, crack area, or a direct equivalent of phase-field damage.

## Convergence and acceptance

Code_Aster converged every requested load state through full load. The
native `STAT_NON_LINE` residual criterion was `1e-4`, and no requested state was
missing from the final result. This is the relevant nonlinear convergence
evidence for this run.

The zero-tension acceptance gate passed:

- Damage threshold: `V1 >= 0.95`
- Residual cracked tensile stress: 3.015 kPa
- Tolerance: 5.0 kPa

### Important audit note

The stored report from this run contains an obsolete post-processing field named
`equilibrium` that reports `passed: false` with a 100% relative residual. That
check compared nodal force fields in a model containing discrete spring elements
and did not represent a valid global equilibrium measure. It should not be read
as a nonlinear solver failure.

The implementation has since been corrected to use successful native
`STAT_NON_LINE` convergence at every archived state as the convergence gate. A
fresh app/API run generated with that corrected version should replace this
first-pass audit field.

## Findings

1. **The wall-only replacement works.** The damage solve completed with no
   foundation or bedrock body and 564 base faces carrying the equivalent spring
   support.
2. **The previous divergence was not caused only by model size.** The decisive
   improvement was a stable local constitutive/continuation setup, specifically
   `ENDO_ISOT_BETON`, fine internal increments, and a regularized post-peak
   slope.
3. **Tension is relieved in the damaged region.** At full load, the maximum
   principal tension at points above 95% damage was approximately 3.0 kPa,
   compared with a 5.0 kPa acceptance tolerance.
4. **Damage develops early and remains localized to a minority of cells.**
   The final count is 1,598 of 19,890 cells, or 8.03%.
5. **The complete wall is still under both tension and compression.** Peak
   tension remains in undamaged parts of the wall; the criterion is therefore a
   no-tension condition in the damaged zone, not zero tension everywhere in the
   solid.
6. **Compression is far below the configured 30 MPa strength.** The peak
   compressive principal stress was about 4.79 MPa.

## Limitations

- The result is marked `engineering_validated: false`.
- The concrete fracture response is not calibrated to dam-specific tests.
- The 150 J/m² fracture-energy input is retained in the material contract, but
  its relation to the selected local `ENDO_ISOT_BETON` softening requires
  engineering confirmation.
- The `V1` damage field depends on mesh size and does not directly calculate
  crack opening.
- The 0.01E post-peak slope is a numerical regularization choice, not a
  calibrated material parameter.
- Foundation and bedrock stresses are intentionally unavailable.
- Water pressure is not applied to newly damaged crack surfaces.
- No mesh, spring-stiffness, or softening-slope sensitivity study has yet been
  completed.

## Recommended next validation

1. Rerun once with the latest native-convergence audit so the app/API receives an
   internally consistent final report.
2. Repeat the full-load solve at the existing target mesh and at a substantially
   finer wall mesh, then compare:
   - damage onset load;
   - damaged-cell count and normalized damaged volume;
   - chainage/elevation envelopes;
   - cracked-point residual tension;
   - displacement.
3. Vary the post-peak slope (`D_SIGM_EPSI`) around `-0.1E` while checking
   convergence and damage extent.
4. Vary X/Y/Z base-spring stiffnesses to establish foundation-compliance
   sensitivity.
5. Only after those checks, identify a physical crack-width/opening method if a
   discrete opening measurement is required for engineering interpretation.

## Result artifacts

The first-pass run contains:

- `crack_report.json`
- `native_crack.json`
- `damage_history.npz`
- `result.med`
- `damage_series.json`
- `crack.pvd` and the accompanying ParaView snapshots
- `solver.log` and `solver.live.log`

The run status in the manifest was
`completed_not_engineering_validated`, which accurately reflects both the
successful numerical solve and the remaining engineering validation work.