# Interim finite-element analysis report

## Concrete dam: elastic corroboration and seeded phase-field cracking

**Date:** 17 October 2026  
**Status:** Draft technical progress report; not issued for design or construction  
**Assessment status:** Numerical investigation

**Confirmed tensile acceptance rule (client clarification, 17 October 2026):**
maximum principal tensile stress above **1.2 MPa (1,200,000 Pa)** is a
**FAILURE of the tensile acceptance criterion**. The comparison is strictly
`peak > 1,200,000 Pa`; equality does not exceed this limit but does not establish
overall design acceptance. This rule applies to the assessment of all reported
cases, including historical cases, without altering their saved numerical data.
Solver completion/convergence is a separate status, not an acceptance result.
Where recovered and raw peaks are both reported, assess and label each
separately; do not conceal an exceedance by choosing the lower measure.

> This report does not establish dam safety, an allowable operating level, a
> compression safety factor or a validated crack-propagation prediction.
> Several important physical mechanisms remain absent or uncalibrated.
> Numerical convergence must not be interpreted as evidence that those omissions
> are acceptable.

### Latest clarification: sizing baseline discrepancy, 17 October 2026

**The ascending width sweep is a sensitivity study of the provisional bonded
model, not a justified required wall thickness. The 9 m result is withdrawn as
a client sizing recommendation.** The historical sub-2 MPa starting result must
be reconciled with the changed mechanical model before issuing sizing advice.

Read-only checks of the saved app reports establish:

| Case | Wall width | Recovered tensile peak | Raw element tensile peak | Mechanical support model |
|---|---|---|---|---|
| App job `d84722f4cce34ef7920f0547241b698b`, intact baseline | 4.0 m | 1.64094 MPa | 1.82838 MPa | Split wall/plinth; wall grounded independently |
| Corrected intact reference | 4.0 m | 3.60402 MPa | 5.15408 MPa | Provisional full bond; foundation support only |
| First ascending sizing trial | 4.5 m | 3.07166 MPa | 4.39556 MPa | Same provisional full bond |
| Earlier successful app job `f25c4a50d25647eeb57535cf963a5927` | 4.5 m | 1.36869 MPa | 1.57445 MPa | Historical split model |

The old 4 m Elmer result independently records 1.64094 MPa recovered tension.
The client's recollection of a sub-2 MPa result is therefore supported by
saved outputs. The above 3 MPa sizing starting point was not obtained by merely
increasing that same historical wall by 0.5 m: it followed replacement of the
support/interface model. The old wall spring reacted wall load directly against
ground; the new model transmits load through the plinth and enforces continuous
displacement. This can change restraint, bending and junction concentrations.
Numerical agreement between solvers does not establish that the provisional bond
or foundation idealisation represents the real structure.

Other provenance differences must also remain explicit. The historical 4 m
snapshot records water-height/overflow-water-level metadata of 29/32 m,
versus 31/34 m in the corrected reference; both have 3 m overflow head, which
the present upstream pressure expression uses. These metadata differences
must not be misrepresented as a demonstrated 2 m rise in applied reservoir
pressure. The historical successful 4.5 m wall kept its upstream radius at
80 m (centreline 77.75 m), whereas the sizing trial kept the 78 m centreline
and moved the upstream radius to 80.25 m. That is another reason not to compare
the two 4.5 m peaks as a thickness-only experiment.

The subsequent user-directed sweep used widths 4.5, 5.0, 5.5, ... m, a 0.5 m
mesh target and fixed 1 m overhangs. On that discretisation, recovered tension
first fell below 1.2 MPa at 8 m, while both raw and recovered peaks first fell
below it at 9 m. Neither threshold establishes a mesh-independent minimum.
The separate 9 m / 0.35 m exploratory mesh check failed force-balance validation
and provides no accepted result. The 9 m Elmer solver completed its mechanics
stage, but its development worker exceeded the time limit during postprocessing
and was stopped; no completed cross-check report is claimed.

Next work must reconcile the physical joint/support model and compare the
governing stress fields at fixed locations on matched inputs, before any further
claim of a required thickness. Do not restore the disconnected bypass merely
to reproduce a favourable stress number, or discard raw peaks merely because
recovery gives a lower value. The 1.2 MPa acceptance rule remains unchanged.

### Corrected bonded intact reference, 17 October 2026

The disconnected model described in the audit below has now been replaced for
**new runs** by an explicitly provisional fully bonded intact reference in both
Elmer and FEniCSx. Conforming wall/plinth interface nodes share displacement
degrees of freedom. The separate wall-ground spring bypass is removed; only
the plinth foundation is supported by the configured bedrock springs.
`geometry.plinth.support` is retained in input for compatibility but is inactive.
Materials, applied loads and the authoritative configuration have not been tuned
to obtain this result. Existing jobs and all earlier tables remain historical.

This is an intact monolithic idealisation, **not** a verified construction-joint
law. It transfers tension and shear as well as compression and does not allow
opening or sliding. Construction details, uplift, joint behaviour and foundation
idealisation still require engineering review.

The app now completes the two elastic baselines and marks cracking **blocked
pending model review**, rather than starting the inherited fracture scenario.
The serial and MPI production fracture entry points also enforce the hold.
There is no new cracked result and no claim that the previous nonlinear failure
has been resolved by this change.

Fresh paired solves on the unchanged 4 m wall / 6 m plinth case gave:

| Verification | Measured result |
|---|---|
| Face-connected bodies | 1 |
| Shared wall/plinth interface | 3,520 quads / 7,040 tetrahedral faces, internal |
| Solver mesh | 117,128 nodes; 606,984 tetrahedra |
| FEniCSx true relative residual | 6.4983e-10 |
| FEniCSx relative force imbalance | 7.1415e-12 |
| Displacement relative L2 difference from Elmer | 6.3932e-11 |
| Reconstructed foundation reaction relative difference | 3.0766e-11 |
| Foundation reaction `(X,Y,Z)` | `(393.8110, 115.5963, 328.6838)` MN |
| Plinth volume-mean raw `(xx,yy,zz,xy,yz,xz)` stress | `(-0.16016,-0.27719,-0.26642,0.01458,-0.09359,-0.32101)` MPa |
| Maximum displacement | 7.9398 mm |
| Recovered maximum principal tension / compression | +3.6040 / -4.6723 MPa |
| Allowable tensile peak, supplied by client | 1.2 MPa |
| Tensile demand / allowable | 3.003; FAILURE of tensile acceptance criterion |

The Elmer reaction is reconstructed from displacement and spring coefficients,
not native reaction output. Raw versus recovered plinth mean tensor difference
is 0.4401%, a diagnostic with different recovery methods. A separate 3-D vector
elasticity benchmark applies **only a 10 kN wall crown load**: that force is
reacted at the plinth foundation and the plinth carries compressive stress,
with no plinth self-weight, direct plinth load or wall-ground support. This
isolates the corrected load path rather than inferring it from a nonzero colour.

**The applicable allowable tensile peak is 1.2 MPa, as clarified by the client,
not the configured 2 MPa reporting value.** The +3.6040 MPa recovered peak is
3.003 times that allowable (approximately 200.3% above it). On this peak-stress
comparison, the result **does not meet the supplied tensile acceptance
criterion**. Successful load-path correction and solver agreement do not make
the stress result acceptable.

The earlier 2 MPa suggestion is superseded as an acceptance threshold by the
client-confirmed 1.2 MPa allowable. It remains a recorded configuration input,
not the design allowable and not a constitutive stress cap. This clarification
does not change saved results, solver material properties or fracture
calibration. Supporting design references should still be recorded for
traceability; their absence does not suspend the confirmed acceptance rule.

The intact elastic model cannot relax stress by cracking; its peak must not be
accepted as a physically sustainable material response beyond its assumptions.
Possible stress singularity, mesh sensitivity and physical joint assumptions
remain unresolved and must be investigated, not used to dismiss the exceedance.
This is a failed tensile criterion in the current model, not by itself a verdict
on the actual dam's safety or physical failure.

Evidence is retained in the session's `bonded-intact-reference-20261017` directory:
`elmer-report.json`, `comparison.json`, the fresh solver fields and the completed
`Baseline_NoPhase` report. Configuration SHA-256:
`afd99de14023eedebdf8d2481695f1d588701b6738a24a32b6a8f767a0c831e4`.
The first verification attempt exposed ElmerGrid's dotted-ancestor output-path
truncation; conversion now uses a local output name and verifies the expected
mesh header. The subsequent fresh Elmer solve completed; failed-attempt output
is not used as a result.

### Historical addendum: wall/plinth load-path audit, 17 October 2026

**A material modelling limitation is now directly confirmed:** wall and plinth
were mechanically separate bodies in the audited model. Wall-side springs acted
against ground, not against plinth displacement. Consequently wall loading
cannot redistribute directly into the modelled plinth through that interface.
The plinth can still carry its own weight and directly applied loads; it is not
necessarily stress free.

The saved mesh for failed job `d84722f4cce34ef7920f0547241b698b` was audited
without rerunning or continuing the calculation:

| Audit measure | Finding |
|---|---|
| Wall/plinth shared vertex IDs | 0 |
| Coincident coordinates, rounded to 1e-9 m | 3,969, with separate node identities |
| Wall-side / plinth-side interface areas | Both approximately 659.99548 m2 |
| Wall-side / plinth-side interface triangles | 36,196 / 5,280; not matching triangle lists |
| Shared wall/plinth damage DOFs | 0 |
| Maximum wall-base damage in saved iterate | 0.9997813 |
| Plinth-side damage | Zero |
| Saved hotspot strain norm before iteration 21 | 0.105908 |
| Saved hotspot principal compressive strain | -0.105416 |
| Saved hotspot nodal damage | 0.9989466 to 1.0 |
| Hybrid stiffness fraction at those vertices | Approximately 1e-8 to 1.12e-6 of intact |

The saved state is the **input to iteration 21**, not the rejected subsequent
state with strain norm 51.14, and not an accepted physical solution. The hotspot
centroid is at elevation -27.90625 m, just above the local wall-base support at
-28 m. The inspection supports investigating a nearly stiffness-free
compressive load path close to a grounded support; it does not prove the
disconnection alone caused the nonlinear breakdown.

**Correction to earlier interface language:** zero damage in the plinth does
not impose zero damage on a separate wall-side node. Earlier descriptions of
a shared zero-damage wall/plinth trace do not apply to this split mesh. The
wall-side damage is demonstrably free to evolve almost to 1 at the base.

A small scalar split-versus-bonded block test now verifies the load-path
distinction. A unit wall-only load is reacted entirely by the wall's ground
support in the split model; plinth response is zero. In the bonded variant,
without that bypass support, the load reaches the plinth foundation.
Eleven focused tests passed across the new audit checks and existing reduced
contact tests. This is not a validated three-dimensional wall/plinth joint.

This finding limits the interpretation of every earlier case below: agreement
between Elmer and FEniCSx includes agreement on the same disconnected support
idealisation. No earlier stress table demonstrates actual redistribution of
wall load into the plinth.

The audit identified the next modelling milestone as a separately identified coupled intact
reference, with the intended wall/plinth bond or joint justified from construction
details. A bonded diagnostic would remove the separate wall-ground bypass and
retain foundation support; an opening/sliding joint instead requires an
appropriate interface law. The provisional bonded reference above now implements
the former explicitly; the latter remains unimplemented.
The nonmatching refined interfaces require a deliberate mesh/coupling treatment.
Simply enabling fracture in plinth cells will not connect the bodies.

Audit implementation: [load_path_audit.py](../FEniCSx/load_path_audit.py);
checks: [test_load_path_audit.py](../FEniCSx/test_load_path_audit.py).
The detailed local audit is
`failed-wall-plinth-load-path-audit-v2.json` in the investigation session's
`files` directory identified in the evidence inventory.

## 1. Executive summary

The work has established a local application workflow linking configuration,
Elmer elasticity, an uncracked FEniCSx comparison and a seeded FEniCSx
phase-field cracking calculation. The work has also exposed significant
limitations in interpreting cracking results from global stress maxima alone.

The principal findings to date are:

1. For the successful case documented below, Elmer and original-mesh FEniCSx
   recovered elastic principal-stress peaks agree extremely closely once mesh
   treatment and stress recovery are made compatible. This is useful solver
   corroboration, but both models can share the same incorrect assumptions.
2. Some automatically generated near-plinth crack prescriptions failed.
   One developed strains grossly outside small-strain kinematics and was
   correctly rejected. Those failures have not been converted into valid
   physical solutions by the later success of a different case.
3. Controlled replays succeeded with the historical crack prescription,
   including on the newer tetrahedral mesh. This argues against the proposition
   that tetrahedra alone broke fracture solving. The automatically selected
   crack location and extent changed and are material to the investigation.
4. A later application case completed, but mesh refinement accounted for a
   substantial portion of the apparent rise in tensile stress. On the same
   refined mesh, the cracked recovered global tensile peak was still higher
   than the refined intact peak. That outcome must be reported, not hidden.
5. At the original coarse-baseline tensile hotspot, recovered tension fell
   substantially after cracking. This can coexist with a higher peak elsewhere.
   The available coarse-to-cracked local comparison still includes mesh effects.
6. A permanent refined-mesh, zero-damage baseline and matched-location reporting
   have now been implemented and exercised. They improve attribution of
   numerical changes; they do not validate the fracture model.
7. The agreed next reporting deliverable is a post-analysis PDF. It is currently
   a strategy, not an implemented automatic PDF reporting capability.

The immediate engineering priority is to establish the physical adequacy and
resolution of the crack model, then explain redistribution with directional
stress plots, actual sections and matched-location profiles. Presentation
quality cannot substitute for that work.

## 2. Purpose, scope and reporting precedent

This draft records completed implementation and investigations, representative
results, unresolved failures and work still required. It is not a comprehensive
design report and does not cover hydrology, site investigation, geology,
spillway design, abutment stability or construction specifications.

The reporting precedent is
[Final Design Rpt Text May 06.docx](Final%20Design%20Rpt%20Text%20May%2006.docx),
particularly section 11.3 and the stress/deflection table. Its useful structure
is retained: model, properties, loads, analysis procedure, results and engineering
interpretation. It calls for crown/cantilever sections, upstream/downstream
directional stress plates and crest-deflection evidence.

The reference describes figures 52-71 but the supplied DOCX contains no embedded
images. Their actual graphical presentation has therefore not been reviewed.
Its brick-element foundation representation, thermal cases, crack assumptions,
capacities and safety conclusions are not adopted by this investigation.
Its vertical axis is Y; the current model uses Z.

This Markdown draft contains verified numerical tables and a figure schedule.
It does not pretend that the proposed contour plates or PDF have been produced.

## 3. Evidence and case identification

Results from different configurations and prescriptions must not be combined
into one apparent load case.

| Evidence set | Identification | Purpose |
|---|---|---|
| Successful application case | Job `f25c4a50d25647eeb57535cf963a5927` | Main numerical results in sections 4-8 |
| Original FEniCSx baseline | Run `20261017T124255133632Z` in that job | Original-mesh elastic reference |
| Cracked FEniCSx result | Run `20261017T124334437423Z` in that job | Completed default-hybrid fracture result |
| Independent mesh-only audit | `client-mesh-only-elastic/audit.json` | Intact solve on that case's refined crack mesh |
| Fixed-location audit | `client-stress-redistribution-audit.json` | Read-only sampling of saved fields |
| Historical replay | Original job `8a188fd0670544d5b657df2fe9242d4d` and fresh replay | Diagnose whether the formerly successful prescription still solves |
| New-mesh/historical-seed diagnostic | Controlled replay with equal configuration | Separate mesh change from seed change |
| New refined-baseline integration check | Fresh eight-rank historical-case replay | Verify the permanent baseline/report feature, not the main application geometry |

The main case's saved configuration SHA-256 is:

`0dd77e14c04d7b437765f7f35d32939823e2a5a36165f6f55c47e9aa00992288`

Its saved geometry differs from the currently documented 4 m wall default.
The main case is a **4.5 m wall / 6.5 m plinth** configuration. Current checkout
defaults must not be substituted when reproducing it.

Primary application artifacts are held locally under:

`%LOCALAPPDATA%\Lamb\DamAnalysisWorkbench\jobs\f25c4a50d25647eeb57535cf963a5927\`

The audits and development replay outputs are local investigation evidence,
not yet a packaged, independently reviewable client evidence bundle. Their
source scripts, manifests and outputs must be archived with a formal issue.
This draft is not sufficient by itself for independent numerical reproduction.

## 4. Model definition for the main application case

### 4.1 Geometry and discretisation

| Item | Saved case value or description |
|---|---|
| Wall radial width | 4.5 m |
| Wall downstream / upstream radius | 75.5 / 80.0 m |
| Wall centreline radius | 77.75 m |
| Plinth radial width | 6.5 m |
| Plinth downstream / upstream radius | 75.0 / 81.5 m |
| Plinth centreline radius | 78.25 m |
| Nominal height metadata | 29 m above plinth; actual geometry follows contours |
| Lowest listed plinth elevation | -28 m relative to crest datum |
| Global mesh-size input | 0.5 m |
| Local mesh-size input | 0.1 m; not proof of achieved crack-band resolution |
| Original reported Elmer element count | 680,142 |
| Solid discretisation | Three-dimensional tetrahedral mesh |
| Foundation treatment | Distributed grounded face springs, not a meshed rock mass |

The wall and plinth centrelines differ in this case; they must not be described
as a symmetric centred arrangement. Actual topology, boundary tags and interface
constraints govern the analysis, not the geometry sketch alone.

### 4.2 Material and support data

| Parameter | Main case input | Qualification |
|---|---|---|
| Concrete density | 2,400 kg/m3 | Input, not independently verified test data |
| Young's modulus | 35 GPa | Linear elastic intact stiffness |
| Poisson's ratio | 0.20 | Isotropic material |
| Configured tensile strength | 2.0 MPa | Historical reporting input, not the design allowable or a constitutive stress cap |
| Allowable tensile peak | 1.2 MPa | Client clarification; acceptance criterion, distinct from configured strength |
| Foundation springs X/Y/Z | 1.0 / 1.0 / 5.0 GN/m3 | Grounded directional face supports |
| Wall-side/plinth springs X/Y/Z | 1.25 / 1.25 / 3.0 GN/m3 | Grounded directional face supports |

Bedrock properties are present in the saved configuration but do not establish
a deformable rock continuum in this model. The spring stiffnesses are explicit
inputs; they are not automatically derived from the bedrock modulus.
No independently established calibration of those support stiffnesses is
demonstrated by the evidence reviewed here.

### 4.3 Applied loading

The saved inputs include gravity -9.81 m/s2, water density 1,000 kg/m3,
overflow head 3 m and tailwater head 3 m. Reference `water_height` is 29 m and
`overflow_water_level` is 32 m.

The upstream pressure uses actual elevation and overflow head:

`p(z) = rho_water * abs(g) * max(overflow_head - z, 0)`.

Consequently the reference water-depth fields must not be read as an independent
reservoir surface elevation. Tailwater is defined relative to the model's minimum
elevation. Gravity, external water loading and the implemented face loads must
be reproduced from the saved input and solver definitions.

Internal crack pressure is **0 Pa** in this case. Temperature loading is not
assessed. No comprehensive uplift, seepage or seismic load-case assessment is
claimed. The historical report's full-supply, flood and thermal combinations
have not been reproduced merely by running baseline and cracked stages.

## 5. Analysis method

### 5.1 Elastic corroboration and stress recovery

Elmer supplies the reference elastic result. FEniCSx/DOLFINx imports the compatible
tetrahedral geometry and boundary/region information. Import is not evidence of
an unnecessary wholesale remesh; local fracture refinement is a distinct step.

Reported recovered principal peaks follow a compatible sequence:

1. Recover stress tensors into a continuous P1 nodal tensor field.
2. Average the nodal tensors over each tetrahedron.
3. Calculate principal stresses of that averaged tensor.
4. Identify the global extrema and their element-centroid coordinates.

Raw element peaks are reported separately. Tensor averaging followed by
eigenvalues is not equivalent to averaging principal stress scalars. Neither
recovered nor raw peaks may be selected merely because they fit an expectation.

Stress sign is tension positive, compression negative. Stresses are presented
in MPa, displacement in mm and coordinates in m.

### 5.2 Seeded phase-field cracking

The default calculation uses a staggered hybrid AT2 phase-field formulation:
small-strain elasticity with irreversible damage and residual stiffness.
Damage 0 denotes intact material; damage 1 denotes fully damaged material.
The degradation factor is `(1 - damage)^2 + residual_stiffness`.

| Main case fracture input | Value |
|---|---|
| Seed | Automatically generated annular sector near Elmer tensile hotspot |
| Seed plane elevation | -27.875 m |
| Seed radial depth | 3.375 m, or 75% of wall width |
| Seed arc extent | Approximately 16.0484 m |
| Geometric seed thickness | 0.002 m |
| Phase-field length scale | 0.1 m |
| Critical energy release rate | 2,700 J/m2 |
| Residual stiffness | 1e-8 |
| Applied load increments | 10 |
| Crack-face contact / friction | Not implemented |

The geometric seed thickness is not a predicted physical crack opening.
The diffuse damaged band is controlled by the length scale and mesh.
Non-wall damage constraints are retained; the plinth is not allowed to develop
the same evolving damage field as the wall. In this split mesh there are no
shared wall/plinth damage DOFs: the wall-side interface is not fixed to zero
by that constraint.

The hybrid formulation degrades full elastic stiffness, including compression.
Its tensile-driving history does not provide unilateral crack closure/contact.
Experimental volumetric compression-preserving and spectral variants have been
investigated, but they do not supply a validated 3D crack-contact model.
The main application results below use **hybrid**, not an experimental law.

### 5.3 Important strength inconsistency

The homogeneous AT2 implied tensile-strength diagnostic is approximately
**9.983 MPa**, versus the configured reporting strength of **2.0 MPa**.
The ratio is approximately five.
Relative to the client-supplied **1.2 MPa allowable**, that diagnostic is
approximately 8.32 times larger. An allowable stress is not automatically a
material tensile strength or a fracture-calibration target; these quantities
must remain distinct.

This is a material-model consistency issue, not a cosmetic reporting issue.
The 2 MPa value does not cause the current constitutive law to cap stress or
initiate damage at that stress. The parameters need independent justification
or calibration before quantitative fracture predictions can be relied upon.
The homogeneous diagnostic is not itself a validated strength prediction for
this seeded, nonuniform dam calculation.

## 6. Main case numerical results

### 6.1 Recovered elastic corroboration

| Global recovered peak | Elmer | Original-mesh FEniCSx |
|---|---:|---:|
| Maximum principal tension, MPa | 1.368691238 | 1.368691219 |
| Minimum principal compression, MPa | -2.911718613 | -2.911718728 |

The peak values agree to much better than 0.001%. Peak coordinates are not
claimed identical, and these two peak comparisons alone do not constitute a
full-field validation. The close agreement supports the corrected elastic
comparison/recovery path, not the physical validity of support or crack assumptions.

### 6.2 Isolating refinement from cracking

The refined intact column below was obtained by a separate fresh elastic audit
of the completed crack mesh. It was not an output of the original app run.
The permanent baseline feature now performs this step before cracking for new
runs.

| Stress measure, MPa | Original intact FEniCSx | Refined intact audit | Refined cracked |
|---|---:|---:|---:|
| Recovered maximum principal tension | 1.368691 | 1.881716 | 2.144378 |
| Recovered minimum principal compression | -2.911719 | -2.912648 | -2.906205 |
| Raw element maximum principal tension | 1.574447 | 1.982093 | 3.659801 |
| Raw element minimum principal compression | -2.981471 | -2.982454 | -3.016847 |

Interpretation:

- The original-to-cracked recovered tensile increase is approximately **56.7%**.
- Refinement alone raises the recovered tensile peak by approximately **37.5%**.
- Cracked versus refined intact raises that recovered peak by approximately
  **14.0%**. Refinement does not explain the entire increase.
- Same-mesh raw compression magnitude rises by approximately **1.15%**.
- Same-mesh recovered compression magnitude falls by approximately **0.22%**.
- These percentages have different reference denominators and are not additive.

The raw/recovered differences demonstrate sensitivity to postprocessing and
local gradients. They do not authorise choosing the raw compression increase
and ignoring the recovered decrease. No mesh-independent peak has been
established.

### 6.3 Matched-location evidence and migrating peaks

At the original coarse-baseline tensile hotspot,
approximately `(78.000, 17.204, -27.875)` m, recovered principal tension changes
from **1.368691 MPa** to **0.209435 MPa**, an approximate **84.7% reduction**.
These are evaluations at the same physical coordinate, not the difference
between two unrelated global maxima.

**Caveat:** the intact field in this particular audit is the coarse baseline.
The comparison therefore still contains refinement effects. It is evidence of
local relief in the saved fields, not an isolated same-mesh crack-only reduction.
The new permanent feature samples cracked stresses at the refined intact
peak coordinates to improve this comparison.

The cracked recovered tensile peak lies at approximately
`(78.270, 15.851, -27.953)` m, below the seed plane. The refined intact tensile
peak is at approximately `(76.831, 21.725, -27.484)` m. The recovered compressive
peak remains near the crest at elevation -0.25 m; the raw compression audit
instead identifies a controlling location near the base.

These distinctions explain why the original summary table could not tell the
whole story. A locally relieved point does not require every global tensile
peak to decrease, nor does cracking require every compressive measure to rise.

### 6.4 Displacement, convergence and global equilibrium

| Main case diagnostic | Reported result |
|---|---:|
| Original intact maximum displacement magnitude | 8.149663 mm |
| Cracked maximum displacement magnitude | 8.225053 mm |
| Final load factor | 1.0 |
| Final-step staggered iterations | 13 |
| Final maximum damage change | 6.2361e-6 |
| Final relative displacement change | 4.1152e-9 |
| Final mechanics true relative residual | 7.3696e-9 |
| Relative global force-balance error | 2.9845e-11 |
| Final diffuse crack-surface measure | 48.9951 m2 |

The displacement values are **whole-model maxima, not verified crest
deflections**. They occur in different discretisations and must not be reported
as an isolated crack-only stiffness change. The diffuse surface measure includes
the regularised seed/evolved field and is not a measured new crack area.
Small residuals and good global force balance do not establish local stress
accuracy, adequate fracture resolution or correct material behaviour.

## 7. Failure investigation and sensitivity

An earlier automatically seeded calculation was rejected at the final load step
after a maximum strain norm of approximately **51.14**, with a principal
compressive strain of approximately **-50.66**. Such strains are far outside a
small-strain model. This is not an admissible result and must not be used as a
physical prediction or a dam-failure threshold.

The historical-to-newer automatic prescription moved from elevation -27.75 m to
-27.875 m. At the local plinth elevation of -28 m, the clearance reduced from
0.25 m to 0.125 m, with an accompanying change in arc extent.
That is significant relative to the 0.1 m length scale and grounded wall base.
Proximity is a warning, not proof that plinth refinement or damage caused failure.

| Controlled fresh calculation | Outcome | Runtime |
|---|---|---:|
| Historical mesh and historical seed, current hybrid solver | Ten steps converged | 233.71 s |
| Newer tetrahedral mesh and historical seed, equal configuration | Ten steps converged | 198.13 s |

The historical replay reproduced maximum displacement of approximately
9.623353 mm and maximum strain norm approximately 0.002193. This supports
continued solvability of that prescription. It does not resolve the changed
near-interface automatic case or establish that the historical seed is a
physically correct substitute. New app jobs were not silently given that seed.

GMRES/GAMG conditioning, distorted cells and expensive nonlinear iterations were
investigated. Frozen-state CG and stronger preconditioner-reuse trials did not
provide the hoped-for improvement and were not adopted on that basis.
Slivers are a possible numerical concern, not an established sole cause.
The evidence does not support a claim that inaccurate unconverged mechanics was
accepted as a valid final result.

## 8. Implementation and verification completed

The application now preserves the sequence:

`Elmer -> original intact FEniCSx -> refinement -> refined intact FEniCSx -> cracking`.

New refined baselines use identical mesh/material/external loads/springs and
zero damage, without internal crack pressure. Their independent report and
fields are persisted before cracking. Completed crack reports embed that
baseline and matched-location stresses.

The desktop retains all four comparisons and defines crack difference against
the refined intact baseline. Older jobs without that reference remain
unavailable for that difference rather than silently using the coarse baseline.
The API checks required baseline values and mesh counts. Count checks alone
are not a general proof of independent mesh identity.

Recent focused verification recorded:

- 37 FEniCSx tests passed, including refined baseline, recovery, reporting and MPI.
- 43 API tests passed.
- Desktop compilation and regression checks passed.
- A fresh eight-worker historical-case baseline-plus-fracture run completed ten
  load increments in approximately 274 s, on 347,131 nodes and 1,871,878 cells.
  Independent and embedded baselines, counts and fixed-coordinate probes agreed.
  The intact true relative residual was approximately 7.97e-10.

The last item is a separate controlled historical case, not a rerun of the
4.5 m wall application case above. A post-run verification script initially used
an incorrect report key and exited after the solver had saved its completed
results; independent inspection subsequently verified the persisted artifacts.
That harness error is not counted as an initial clean end-to-end pass.

These are targeted development checks, not an independent certification suite.
No claim is made that every working-tree change has a complete release-level
regression or that the running installed app already contains the updated code.
The desktop/API need restart/rebuild to use changes. Assistant-run development
checks use a ten-minute abandon-and-re-evaluate limit; user app/API jobs do not.

## 9. Required figures and missing client-facing evidence

The following schedule adapts the client's historical reporting pattern.
These are **required future figures, not figures attached to this draft**.

| Figure | Required content | Current reporting gap |
|---|---|---|
| 1 | Actual geometry, datum, water surfaces, support zones and seed | Automated annotated report plate not implemented |
| 2 | Crown section, vertical normal stress and scaled deformation | Crown cut and directional reporting need verified processing |
| 3 | Upstream vertical stress, refined intact versus cracked | Reproducible face-specific plate not implemented |
| 4 | Upstream explicitly defined horizontal stress comparison | Global/local direction must be defined and checked |
| 5 | Downstream vertical stress comparison | Same-view/common-range plate required |
| 6 | Downstream horizontal stress comparison | Same component and range as upstream comparison |
| 7 | Crest plan deformation and station profile | Whole-model maximum is not a substitute |
| 8 | Principal tensile/compressive paired plates with old/new peaks | Data exists; automated PDF figures not implemented |
| 9 | Seed, final damage and crack/plinth section | Need actual tetrahedral cut, not exterior clipping |
| 10 | Common-location stress profiles and differences | Additional verified volume sampling required |

Face/section maxima must be distinguished from whole-volume maxima. Vertical
normal stress is sigma_ZZ, not the historical sigma_YY. Global horizontal
stress is not automatically arch stress; a local tangent/radial/vertical basis
must be verified before using arch/cantilever labels.

Every plate requires common comparison limits, units, load factor, provenance,
coordinates and recovery definition. Any deformation exaggeration or damage
threshold must be visible. Data must not be interpolated into fictitious
accepted load states when only final spatial fields are available.

## 10. Limitations and exclusions

| Limitation | Consequence |
|---|---|
| No validated 3D unilateral crack contact or friction | Compression transfer and crack closure may be physically wrong |
| Full-stiffness degradation in default hybrid model | Compression may soften across damaged material |
| AT2 parameters not reconciled with configured tensile strength | Initiation/propagation cannot be treated as calibrated concrete behaviour |
| Automatic seed depends on recovered Elmer peak | Mesh/reporting changes can change the physical prescription |
| Near-plinth seed and independently grounded wall base | Interaction needs dedicated sensitivity/verification |
| Wall and plinth mechanically disconnected | Wall-to-plinth redistribution is absent; bonding or an interface law needs explicit modelling |
| No demonstrated mesh/length-scale convergence | Local peaks, diffuse area and crack evolution remain resolution dependent |
| Spring-supported foundation idealisation | Foundation/abutment deformation and force paths may be misrepresented |
| Small-strain kinematics | Gross displacement gradients invalidate the formulation |
| Limited analysed loading | No comprehensive thermal, seismic, uplift or reservoir-combination assessment |
| No validated crushing/plasticity/capacity assessment | No compression safety factor or failure-load claim |
| Seeded diffuse damage | No direct prediction of crack aperture, leakage or sharp crack-face position |
| Missing standardised directional plates/crest profiles | Current presentation falls short of the client's historical FEA report |
| Local unbundled audit evidence and evolving source | Formal reproducibility/release record remains incomplete |

Successful calculation of a later case does not erase the failed cases.
Agreement of elastic solvers does not validate fracture. Good equilibrium does
not establish safety. The absence of a mechanism from the model is not evidence
that it is unimportant.

## 11. Conclusions and next actions

### Conclusions supported by the evidence

The elastic comparison and stress-recovery alignment are functioning for the
documented case. A seeded hybrid cracking solve can complete, and fixed-location
evidence shows local relief alongside migration/increase of global tension.
Mesh refinement materially affects the reported tensile maximum.
The new refined intact baseline addresses a real attribution defect in the
comparison workflow.

### Conclusions not supported

The work does not establish that cracking improves the dam, that the predicted
damage path is correct, that compression transfer is realistic, or that a
particular reservoir condition is safe. It does not reproduce the historical
design report's full case suite or justify its safety factors.

### Prioritised next work

1. Archive immutable inputs, source revisions, manifests, fields and audit scripts
   for the cases reported here; make the evidence independently reproducible.
2. Agree the physical crack prescription and contact/compression assumptions
   with the responsible engineer. Do not adjust the seed merely to obtain convergence.
3. Justify/calibrate fracture and support parameters, including the discrepancy
   between AT2 implied strength and the configured reporting strength.
4. Perform mesh, length-scale, seed-position and support-sensitivity studies,
   including the unresolved near-interface cases and proper strain diagnostics.
5. Produce the crown/face directional figures, crest profiles and matched-location
   sections required to explain load redistribution.
6. Implement the separate post-cracking PDF stage with explicit report failures,
   reproducible figure settings and report-only retry.
7. Obtain independent engineering review before interpreting these results as
   design acceptance or an operational recommendation.

## References and evidence locations

- [Client historical design-report text](Final%20Design%20Rpt%20Text%20May%2006.docx)
- [Application/workflow plan](app_plan.md)
- [Current FEniCSx methods, investigations and limitations](../FEniCSx/README.md)
- [Scientific stress-recovery implementation](../FEniCSx/stress_recovery.py)
- [Dam phase-field adapter](../FEniCSx/dam_phase_field.py)
- [Existing engineering Markdown reporting](../FEniCSx/engineering_report.py)
- [Post-analysis PDF strategy](../../report-stategy.md)

Local evidence inventory:

- Main job: original `baseline_report.json`, cracked `phase_field_report.json`,
  manifests, configuration snapshots and Elmer `result_report.json`.
- Investigation session `b0d144b6-3753-4bfd-b095-ed5e0260d88b`, under the local
  Copilot session-state `files` directory: `client-mesh-only-elastic/audit.json`,
  `client-stress-redistribution-audit.json`, historical replay logs,
  `full-refined-baseline-verification.log` and
  `parallel-fracture-verification-8-historical-hybrid-refined-baseline/`.

These local locations identify the drafting evidence; they are not a substitute
for a controlled client deliverable archive. The reference to the parent PDF
strategy also crosses the nested repository boundary and requires the parent
workspace to resolve.