# Dam cracking baseline with Code_Aster's own dam-joint method

Written 11 November 2026. This records how Code_Aster's standard dam-joint method was
transferred to the Gairezi dam, what it gives, and what was changed from the standard
to make it run. It replaces the earlier cracking attempts as the starting point.

**Status: proof of concept.** The method runs on this dam to full reservoir in about four
minutes, and the app runs it after the baselines when `CRACKING_ENABLED` is set (section 8).
A temperature load can be added from the command line (section 9): the wall cooled below a
closure temperature of 16 C. The joint values are generic and the expansion coefficient is
provisional; none is measured for this site, and nothing here is an engineering assessment
of the dam.

Evidence labels: **measured** = computed from a run's output files; **read** = read in the
installed Code_Aster 17.4 or its manual; **reported** = from web research, not opened by me.

## 1. Why the earlier attempts did not work

Code_Aster was not the problem. It has joint laws written for dams ("concrete/rock joints
or joints between the blocks of a dam", manual R7.01.25, **read**) and ships dam test cases
that use them. The earlier runs did not follow that method.

| | Code_Aster's dam test (SSNP142) | Earlier runs |
|---|---|---|
| Where the crack can form | A joint layer on the weak plane: the dam/ground contact | Element faces inside a 1.1 m band of intact concrete at z = -25 m |
| The dam/ground contact | A joint that carries almost no tension, so it can open | Linear springs that hold tension, so it could never open |
| Loading | Self-weight in full, then the reservoir raised in steps | Weight and water scaled up together |

In the last elastic baseline the wall/foundation springs held up to 0.98 MPa of tension
(34.8 MN in total) at the upstream heel, and the foundation/bedrock springs up to 0.52 MPa
(68.5 MN) (**measured**). That is where a dam is expected to open first, and it could not.

## 2. The standard that was adopted

All **read** from the installed tests in
`~/opt/code_aster_17.4.0/share/aster/tests` and the manual R7.01.25.

- **SSNP142, "filling of a dam".** A dam block on its ground with a joint layer at the
  contact. Self-weight is applied, the reservoir is raised in steps, and the case is solved
  dry and with water pressure in the joint.
  - `ssnp142b`: 3D, rupture law `JOINT_MECA_RUPT` with zero tensile strength.
  - `ssnp142d`: 3D, friction law `JOINT_MECA_FROT`.
- **Element types of `ssnp142b`:** linear tetrahedra for the bodies and prism (PENTA6)
  joint cells. These are the element types of our dam mesh, so no re-meshing was needed.
- **Solver settings of the test:** tangent matrix updated every iteration, line search of
  5 iterations, relative residual 1e-5, at most 12 Newton iterations, automatic step
  cutting, MUMPS.
- **Values the manual recommends for a concrete dam:** normal and tangential stiffness
  3e12 Pa/m, tensile threshold 50 kPa (rupture law), friction coefficient 1 and no adhesion
  (friction law).

## 3. What was checked before touching the dam

| Check | Result |
|---|---|
| 13 of Code_Aster's bundled joint tests, run unchanged (8 of SSNP142, 2 of SSNP143, 3 of SSNP162) | All pass: 191 checks, none failed. The three SSNP162 runs raise the expected alarm about zero-thickness joint cells. |
| Two-block case against a hand calculation (a wall on a block, water on one face, joint with no tension), both laws, dry and with water in the joint | Opening profile within 1.3 to 1.7% of the hand calculation. On a mesh half as fine the error is 4 to 5%, so the gap is mesh error. |
| Control on the dam: joint cells in place of the springs, same stiffness, joint that cannot break | Peak wall displacement 10.346 mm against 10.293 mm for the elastic baseline (0.5%). Nothing broke. |

All **measured**. The two-block case also confirmed the node order of the joint cells and
the sign of the opening: Code_Aster's own opening equals the opening rebuilt from nodal
displacements to rounding error.

## 4. Result on the dam

Joint layer at the contact between the wall and the concrete foundation (2,020 joint
cells, 751 m2), friction law, last default-geometry job (`3a64965e...`). Self-weight first,
then the reservoir raised in four equal steps of head to the configured level.

| At full reservoir | Dry joint | Water in the joint | Elastic baseline |
|---|---|---|---|
| Largest opening | 0.88 mm | 1.13 mm | cannot open |
| Where | upstream heel, valley bottom (z = -29.2 m) | same | |
| Contact area opened | 100 m2 (13%) | 142 m2 (19%) | |
| Contact area sliding | 484 m2 | 505 m2 | |
| Largest relative sliding at the contact | 1.67 mm | 2.46 mm | |
| Peak wall displacement | 11.33 mm | 11.82 mm | 10.29 mm |
| Peak wall tension | 1.75 MPa | 1.78 MPa | 1.64 MPa |
| Peak wall compression | 3.34 MPa | 3.39 MPa | 3.23 MPa |
| Solve time for the five levels | 176 s | 202 s | |

All **measured**. How the dry case develops as the reservoir rises:

| Reservoir (share of head) | Largest opening | Opened area | Peak wall displacement |
|---|---|---|---|
| 0% (weight only) | 0.07 mm | 19 m2 | 1.10 mm |
| 25% | 0.06 mm | 17 m2 | 1.02 mm |
| 50% | 0.02 mm | 2 m2 | 1.22 mm |
| 75% | 0.07 mm | 12 m2 | 4.01 mm |
| 100% | 0.88 mm | 100 m2 | 11.33 mm |

Reading the result:

- The contact opens at the upstream heel in the valley bottom, where the elastic baseline
  held the most tension.
- The opening is under 1 mm dry and just over 1 mm with water in the joint. Almost all of it
  appears in the last quarter of the head.
- Peak wall tension rises by about 0.1 MPa. The wall itself is still modelled as intact
  concrete; only the contact can open.

### Other wall geometries (the `Reporting/FEA_Matrix.xlsx` sweep)

Same method, run from the command line on 12 November 2026 on the saved job of each row of
the sweep. Everything but the wall geometry is the default. All **measured**, at full
reservoir; every run completed in four to six minutes.

| Run | Wall | Opening, dry / water in joint | Contact opened | Peak wall tension | Peak wall displacement | Elastic baseline displacement |
|---|---|---|---|---|---|---|
| 1 | 4 m | 0.88 / 1.13 mm | 13% / 19% | 1.75 / 1.78 MPa | 11.33 / 11.82 mm | 10.29 mm |
| 2 | 4.5 m | 0.71 / 0.96 mm | 12% / 17% | 1.41 / 1.45 MPa | 9.71 / 10.16 mm | 9.04 mm |
| 3 | 5 m | 0.54 / 0.80 mm | 10% / 17% | 1.09 / 1.13 MPa | 8.43 / 8.82 mm | 7.99 mm |
| 4 | 5.5 m | 0.40 / 0.67 mm | 9% / 15% | 0.83 / 0.88 MPa | 7.42 / 7.76 mm | 7.12 mm |
| 5 | 4 m, 4.5 m below z = -20 m, 5 m below z = -25 m | 1.58 / 2.23 mm | 20% / 27% | 1.00 / 1.04 MPa | 10.80 / 11.54 mm | 9.49 mm |
| 6 | 4 m, 4.5 m below z = -20 m | 1.34 / 1.75 mm | 19% / 24% | 1.13 / 1.18 MPa | 10.92 / 11.51 mm | 9.68 mm |
| 7 | 4 m, 5 m below z = -25 m | 1.35 / 1.92 mm | 18% / 25% | 1.33 / 1.39 MPa | 11.07 / 11.83 mm | 9.82 mm |

- In every run the largest opening is at the upstream heel in the valley bottom.
- A thicker uniform wall opens less.
- **The stepped walls open more than the uniform 4 m wall**, although their peak wall tension
  is lower. Run 5 with water in the joint is the only run above 2 mm.
- A stepped wall is meshed in parts, with a horizontal joint between the parts at each step.
  Those joints are not part of the wall/foundation contact: they keep the linear springs of
  the elastic baseline and cannot open. In run 5 the joint at z = -20 m holds up to 1.1 MPa
  of tension (dry) and 1.2 MPa (water in the joint) at the downstream face, 37 and 46 MN in
  total, and stretches by up to 0.4 mm. The "peak wall tension" column is the stress in the
  concrete and does not include this.
- Why the stepped walls open more was tested on run 5 (dry joint) by one extra solve with the
  springs between the wall parts made 100 times stiffer, so that the wall acts as one piece
  (**measured**):

  | Run 5, dry joint | Springs as configured | Springs 100 times stiffer |
  |---|---|---|
  | Largest opening | 1.58 mm | 1.14 mm |
  | Peak wall tension | 1.00 MPa | 1.46 MPa |
  | Peak wall displacement | 10.80 mm | 10.84 mm |
  | Tilt of the wall section just below / just above z = -20 m | 0.59 / 0.29 mm per m | 0.46 / 0.39 mm per m |

  With the configured springs the wall hinges at the joint at z = -20 m: the lowest 9 m tilts
  about twice as much as the wall above and lifts the upstream heel. That accounts for about
  two thirds of the extra opening over the uniform 4 m wall. It also relieves vertical bending
  in the wall, which is why the peak wall tension of the stepped walls is low. With the parts
  tied, run 5's peak tension is about that of the uniform 4.5 m wall (1.41 MPa).
- The rest (1.14 mm against 0.88 mm for the uniform 4 m wall) remains with the parts tied. At
  the valley-bottom contact the tied stepped wall carries a bending moment close to the 5 m
  wall's (3.0 against 3.4 MN m per metre) with a vertical force close to the 4 m wall's (1.8
  against 1.6 and 2.4 MN per metre) (**measured**). That the thicker base draws the moment of a
  thick wall without its weight is my reading of those figures, not a separate test.
- The factor of 100 is a test value. How the parts of a stepped wall are connected in the
  model is `geometry.wall.joint_support` in `config.json`; whether a real stepped wall would
  have a construction joint there is for the engineer.

Checks made for the stepped walls (**measured**, run 5):

| Check | Result |
|---|---|
| Control: joint cells that cannot break, the springs between the parts kept | 9.517 mm peak wall displacement against 9.492 mm for the elastic baseline (0.3%); peak tension 0.942 against 0.947 MPa. |
| Joint water pressure Code_Aster applied, against the same formula evaluated outside Code_Aster, at every joint cell and water level | Largest difference under 1e-6 Pa. |
| Case files of a uniform wall, before and after the change that added stepped walls | Identical, byte for byte. |

Joint values used in all of these runs:

| Parameter | Value | Source |
|---|---|---|
| Normal and tangential stiffness | 1.25e9 Pa/m | `config.json` (the existing wall/foundation spring stiffness) |
| Friction coefficient | 1 | Manual R7.01.25, recommended for a concrete dam |
| Adhesion | 1000 Pa (a tensile strength of 1 kPa) | Test `ssnp142d` |
| Sliding hardening | one tenth of the tangential stiffness | Test `ssnp142d` |
| Water in the joint | Reservoir head at the upstream edge, falling linearly to tailwater head at the downstream edge | As `ssnp142b`'s imposed joint pressure |

## 5. What did not work

Each run was stopped by an eight-minute solver limit. Runs that need more than 20 minutes
are not acceptable for this project, so none of these was pursued further.

| Case | What happened |
|---|---|
| Rupture law, manual's values (3e12 Pa/m, 50 kPa) | Did not converge even for weight alone. |
| Friction law, manual's stiffness (3e12 Pa/m) | Did not converge for weight alone beyond 0.1% of the weight. |
| Rupture law, zero strength, configured stiffness (the values first selected) | Converged for weight and for 25% and 50% of the head, then cycled without converging on the way to 75%. |

All **measured**. My reading, not verified by a further test:

- The manual's 3e12 Pa/m is a penalty stiffness. With real concrete (35 GPa) the joint is
  about 70 times stiffer than the elements beside it. In Code_Aster's own tests the joint
  is softer than the elements, because the tests use an unrealistically stiff "concrete".
  The configured 1.25e9 Pa/m is in the tests' regime.
- This contact carries a lot of shear, because the arch pushes into the stepped abutments.
  The rupture law drops the shear abruptly to zero when a point opens, which makes the
  solve cycle. The friction law loses shear gradually.

## 6. Departures from the Code_Aster test

Each is recorded in every run's `joint_report.json`.

1. Joint cells have zero thickness, because the contact is meshed with coincident nodes;
   the test's are 1 mm thick. Their node order is set explicitly, so `ORIE_FISSURE` is not
   used.
2. Weight alone is solved first, then the reservoir is raised in four equal steps; the test
   applies weight and the first water level together. The weight is ramped up during that
   first step so that step cutting can reduce the increment.
3. Each water level is its own solve continuing the previous one, so a level that does not
   converge still leaves the earlier levels as results.
4. Water pressure in the joint varies across the base at each elevation, from reservoir
   head upstream to tailwater head downstream; the test's varies along one axis to zero.
5. The concrete foundation still rests on the existing bedrock springs, which hold tension.
   The test's ground is a meshed block.
6. The friction runs use the test's adhesion of 1000 Pa. The manual's default of 0 leaves the
   unloaded joint with no stiffness and the first solve is singular (**measured** on the
   two-block case).
7. The baseline's spring cells stay in the mesh file but are given no model.
8. A wall with thickness steps is meshed in parts with a horizontal joint between them at
   each step. Those joints keep the linear springs of the elastic baseline, which hold
   tension. Only the wall/foundation contact can open.

## 7. How to run it

PowerShell, from the `Gairezi` directory. Each command needs WSL with Code_Aster 17.4.

```powershell
# Code_Aster's own dam tests, unchanged (about 2.5 minutes)
.venv\Scripts\python.exe -m Aster.bundled_dam_tests

# Two-block check against the hand calculation (under 1 minute)
.venv\Scripts\python.exe -m Aster.dam_joint_baseline fixture

# One case on the dam (about 4 minutes); --job is a finished job folder of the app
.venv\Scripts\python.exe -m Aster.dam_joint_baseline run friction_configured_dry --job <job folder> --progress-log <file>

# Two-block check of the temperature load (under 1 minute)
.venv\Scripts\python.exe -m Aster.dam_joint_baseline fixture --cooling

# The dam with the temperature load: closure at 16 C, wall at 12, 10 and 8 C (about 8 minutes)
.venv\Scripts\python.exe -m Aster.dam_joint_baseline run friction_configured_uplift_cooling --job <job folder> --progress-log <file>

# The same with other wall temperatures or another closure temperature, in degrees Celsius
# (each wall temperature is about a minute; four fit ten minutes only with a dry joint)
.venv\Scripts\python.exe -m Aster.dam_joint_baseline run friction_configured_dry_cooling --job <job folder> --wall-temperatures-c 14 12 10 8
.venv\Scripts\python.exe -m Aster.dam_joint_baseline run friction_configured_dry_cooling --job <job folder> --closure-temperature-c 18

# Unit tests (no solver)
.venv\Scripts\python.exe -m unittest Aster.test_dam_joint_baseline
```

- Cases: `control`, `friction_configured_dry`, `friction_configured_uplift`, the two with
  the temperature load (`friction_configured_dry_cooling`,
  `friction_configured_uplift_cooling`), and the ones that did not converge
  (`reference_dry`, `reference_uplift`, `selected_dry`, `selected_uplift`,
  `friction_reference_dry`, `friction_reference_uplift`).
- `--progress-log` appends a timestamped line for each stage, water level and Newton
  iteration, so a run can be watched.
- The solver time limit defaults to 10 minutes and cannot be set above 20.
- Results go to `Aster/results/dam_joint_baseline/<case>/` (not in git): `joint_report.json`
  is the summary, `run/solver.live.log` is Code_Aster's own output.

Code: [bundled_dam_tests.py](./Aster/bundled_dam_tests.py),
[dam_joint_baseline.py](./Aster/dam_joint_baseline.py),
[dam_joint_fixture.py](./Aster/dam_joint_fixture.py),
[test_dam_joint_baseline.py](./Aster/test_dam_joint_baseline.py).

## 8. In the app

With `CRACKING_ENABLED` set on the analysis host, a job runs this method as the stage
**Contact joint opening (Code_Aster)** once both elastic baselines have ended. The flag and
the app's behaviour are described in the parent repository's README.

- The API starts one worker, which runs `friction_configured_dry` and then
  `friction_configured_uplift` on the job's own Code_Aster baseline and writes
  `joint_opening/joint_opening_report.json` in the job folder:

  ```powershell
  .venv\Scripts\python.exe -m Aster.dam_joint_baseline app --baseline-dir <job>\cracking\aster_elastic_baseline --output-dir <job>\joint_opening --time-limit-s 600
  ```

- The worker exits with 0 only when both cases reach full reservoir. Otherwise the stage
  fails and the job is marked failed; the baseline results stay readable.
- The stage is blocked, and the job ends on its baselines, for a configuration without a
  concrete foundation, with the bedrock meshed as a volume, or with the no-bedrock
  comparison.
- The desktop shows the two cases beside the elastic baseline on the Comparison tab.

Checked through the API on 12 November 2026, on a copy of job `3a64965e...` in a separate
job store (**measured**):

| Check | Result |
|---|---|
| Whole stage through the API | Succeeded in 494 s. Dry: 0.884 mm opening, 99.9 of 751.3 m2 opened, 11.327 mm peak wall displacement, 1.749 MPa peak tension. Water in the joint: 1.130 mm, 142.3 m2, 11.820 mm, 1.782 MPa. These equal the stand-alone runs of section 4. |
| Cancel while Code_Aster was solving | Job and stage cancelled; 7 Linux processes of the job before, 0 after; the stage can be restarted. |

## 9. Temperature load

Not part of SSNP142 and not run by the app. Added from the command line to see what a
temperature drop does, because self-weight and water alone opened the contact by about 1 mm.

Inputs:

| Input | Value | Basis |
|---|---|---|
| Closure temperature of the wall (the temperature at which it is free of thermal stress) | 16 C | Set by Alistair on 12 November 2026. The project holds no construction record of it. |
| Coldest wall temperature solved | 8 C | The minimum of the climate statistics for Nyanga as quoted by Alistair (8 C minimum, 26 C maximum). An air temperature; I did not look the statistics up. |
| Thermal expansion of concrete | 1e-5 per kelvin | **Provisional.** Generic value for concrete, not measured for this dam. |

What is applied: the reservoir is raised to full as in section 4 with the wall at 16 C. Then
the wall alone is cooled, uniformly through its thickness, in steps of 2 K down to 8 C. The
foundation keeps its temperature.

Checks (**measured**):

| Check | Result |
|---|---|
| Two-block case, closure at 20 C, wall cooled to 0 C: change of the wall's size away from the contact against expansion x temperature drop x length | Across, along and over the upper half of the height: 0.200 mm each, within 0.001% of the hand value. The uncooled block changes by 0.03% of that. |
| Same case, the five water levels before cooling | Identical to the run without cooling. |
| Dam, full reservoir with the wall at 16 C | 0.884 mm dry and 1.130 mm with water in the joint, as in section 4. |

Result on the dam at full reservoir (**measured**; job `3a64965e...`). The run with water in
the joint has no 14 C step: three steps are what fits a ten-minute run.

| Wall at | Below closure | Largest opening, dry joint | Largest opening, water in the joint | Contact opened (dry / water) | Peak wall displacement (dry / water) | Peak wall tension (dry / water) | Wall nodes above 2 MPa (dry / water) |
|---|---|---|---|---|---|---|---|
| 16 C | 0 K | 0.88 mm | 1.13 mm | 13% / 19% | 11.3 / 11.8 mm | 1.75 / 1.78 MPa | none |
| 14 C | 2 K | 1.21 mm | not run | 18% / - | 13.9 mm / - | 1.99 MPa / - | none / - |
| 12 C | 4 K | 1.58 mm | 1.92 mm | 26% / 31% | 16.6 / 17.2 mm | 2.50 / 2.55 MPa | 1.1% / 1.3% |
| 10 C | 6 K | 1.96 mm | 2.35 mm | 32% / 38% | 19.3 / 19.9 mm | 3.34 / 3.37 MPa | 3.6% / 3.4% |
| 8 C | 8 K | 2.36 mm | 2.84 mm | 37% / 43% | 22.0 / 22.7 mm | 4.16 / 4.17 MPa | 7.1% / 6.2% |

Reading the result:

- **A 2 mm opening at the wall/foundation contact** needs the wall at about 10 C with a dry
  joint (1.96 mm solved at 10 C) and at about 11.5 C with water in the joint (a straight
  line between the solved 12 C and 10 C values).
- **With these inputs the opening cannot pass 2.4 mm dry or 2.8 mm with water in the
  joint**, because the wall is not taken colder than 8 C.
- **The wall reaches its tensile strength first.** The configured tensile strength of the
  concrete is 2 MPa. Peak wall tension is 1.99 MPa at 14 C, on the downstream face at about
  z = -19 m, where the peak already is without a temperature load. From 12 C the peak is on
  the downstream face just above the contact, near the valley bottom.
- So below about 14 C the figures describe a wall that stays intact when the model says it
  would start to crack. The wall is modelled as elastic; only the contact can open. A
  cracked wall would carry the load differently, and these runs do not show how.
- The largest opening stays at the upstream heel in the valley bottom (z = -29.2 m).
- The foundation/bedrock springs hold more tension as the wall cools: 30 MN at 16 C and
  52 MN at 8 C in the dry run. That contact still cannot open.

Whether the wall gets as cold as 10 to 12 C is not established:

- 8 C is a minimum air temperature. A 4 m wall does not follow daily swings: by a slab
  heat-conduction estimate with a typical diffusivity of 1e-6 m2/s (my calculation, not
  measured), they reach about 0.2 m into the concrete, while the wall's average follows the
  seasonal swing of its faces almost fully. What sets the wall's temperature is therefore
  the monthly mean of the coldest months, which the project does not have.
- The upstream face is against reservoir water, which varies less than air.

Run times: 418 s for the dry run (nine levels) and 448 s with water in the joint (eight).

An earlier sweep, before the closure temperature was set, cooled the wall by 5, 10 and
15 K. With closure at 16 C those are wall temperatures of 11, 6 and 1 C; the last two are
below the coldest quoted air temperature. Its 5 K values (1.77 mm dry, 2.13 mm with water
in the joint) lie between the 12 C and 10 C values above. At 10 K it gave 2.78 / 3.33 mm and
4.90 MPa, at 15 K 3.93 / 4.68 mm and about 6.8 MPa. A fourth step of 20 K did not fit a
nine-minute limit: Code_Aster stopped it for lack of time, not for lack of convergence.

What this does not cover:

- The warm side. With closure at 16 C the quoted maximum of 26 C is 10 K of warming. The code
  handles cooling only; warming was not run.
- No measured concrete temperatures and no measured expansion coefficient exist in the
  project. The result scales with the expansion coefficient and with the closure
  temperature: closure one kelvin higher is one kelvin more cooling at the same wall
  temperature.
- The cooling is uniform through the wall. A real wall is colder at its faces than inside,
  and its two faces differ. A gradient was not applied.
- The foundation is not cooled, so the wall shrinks against it at the contact. Part of the
  tension next to the contact comes from that assumption.
- The two-block check verifies the load. The dam's response to it has no independent check.
- One mesh. The peak tension next to the contact is a local value and depends on element size.

## 10. What can be adjusted next

In rough order of value. Each should stay inside the 20-minute limit.

1. **Joint values from the engineer.** Stiffness, friction and tensile strength of the
   wall/foundation contact are the main unknowns.
2. **The foundation/bedrock contact.** It still holds tension (30 MN in the dry run,
   **measured**). Giving it a joint needs the bedrock meshed as a volume.
3. **Real temperatures.** Section 9 shows the contact and the wall are both sensitive to a
   few kelvin of cooling. It needs the monthly mean temperatures at the site, a closure
   temperature from construction records, the expansion coefficient, a gradient through
   the wall and the warm season.
   Guidelines for arch dams treat seasonal temperature as the usual driver of cracking
   (**reported**: FERC Engineering Guidelines chapter 11, ICOLD benchmark work).
4. **The same joint on lift planes or vertical joints** inside the wall, where the
   engineer expects weakness. Section 9 shows the wall reaching its tensile strength before
   the contact opens by 2 mm, so this is where cracking of the wall itself would be
   modelled. Code_Aster's SSNP143 covers grouted and sawn vertical joints.
5. **Water flow solved in the joint** instead of an imposed pressure (`3D_JOINT_HYME`). The
   test converts the mesh to quadratic elements for this.
6. **A mesh check.** One refinement near the heel, to see how much the opening depends on
   element size. Joint cells have one integration point each.

## 11. Not verified

- The joint values are generic. No test data for this site's contact exists in the project.
- The explanation in section 5 for the cases that did not converge is my reading of the
  solver output. I did not isolate it with a dedicated test.
- The opening has not been checked on a finer dam mesh.
- The guideline statements in section 10 come from web research; I did not open the documents.
- The SSNP142 validation document (V6.03.142) and the cohesive-zone user guide (U2.05.07)
  were reported by the same research and not opened by me. The test files themselves and
  the manual's parameter table were read directly.
- Other geometries: only the seven of the sweep in section 4 and one 2 m wall through the
  app have been run. Water in the joint is implemented for a wall on a circular arc, of
  uniform thickness or with thickness steps at fixed elevations.
- A stepped wall has not been run through the app since stepped walls were added, only
  from the command line.
- The temperature load: see the list at the end of section 9. The expansion coefficient is
  the generic value for concrete as I recall it from design codes; I did not open a source.
