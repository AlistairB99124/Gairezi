# Saved-job Aster diagnostics

Open the parent Lamb workspace, then use **Terminal > Run Task**. The
`Aster case study:` tasks read the copied application job in
`case_study_00`, not the live AppData job or today's root configuration.

Run one task at a time, inspecting each result before proceeding:

| Task | Purpose |
|---|---|
| 0 - Inventory archived results | Record saved baseline/cracked reports and SHA-256 input hashes; no solver |
| 1 - Elastic on cracking mesh | Use corrected consistent interface integration on the saved mixed-order volume mesh, with unchanged ties and external loads; use ELAS everywhere, with no initial damage |
| 2 - Damage law without seed | Keep the same mesh and band material/formulation, but omit the initial seeded internal-variable field; damage may develop naturally |
| 3 - Seeded crack replay | Replay the saved seed and damage model with corrected interfaces using current command code |
| 4 - Compare diagnostic results | Summarise the latest attempt of each stage, including failed/prepared/missing stages; no solver |

These are independent tasks, not an automatically chained multi-hour run.
Only one diagnostic solver may run per study; a shared lock rejects overlapping
stages. Do not run these memory-intensive tasks beside an active app solver.
Each manually launched solver task has an explicitly approved **four-hour**
wall-clock limit. The command-line default remains **570 seconds**, unless
`--allow-extended-budget` is supplied. Agent verification must retain its
ten-minute cap.

New diagnostics default to `consistent_two_sided_TRI3_TRI6`. If the archive
predates this correction, preparation rebuilds the interface mesh/operator from
the archived baseline's source mesh, SIF and diagnostics, checking their saved
hashes. It uses the archived seed geometry, band width, materials and loads,
not today's configuration. Missing/changed source artifacts are an explicit
failure; there is no fallback to vertex-only springs. A newer corrected archive
reuses its hash-checked `quadratic_discrete.npz` and mesh.

Use `--legacy-lumped` **only** for an intentional historical replay of an old
archive. It cannot downgrade a corrected archive. Run all three new stages
before comparing them: comparison rejects mixtures of old and corrected
operators. To preserve an independent series, add
`--output-dir .\case_study_00\diagnostics\corrected_interfaces` to each command.
All output subdirectories share the study-level solver lock.

## Outputs and interpretation

Each solve creates a timestamped directory under
`case_study_00/diagnostics/<stage>/`, with:

- `case.comm`, `inputs.json`, `diagnostic.export`: exact replay inputs;
- `manifest.json`: provenance, runtime limits, state and explicit errors;
- `solver.live.log` and, on normal process exit, `solver.log`;
- `summary.json`: every archived load factor, regional Gauss-point tensile
  and compressive extrema, peak coordinates/cell identities, displacement
  component maximum and band damage maximum for damage stages;
- `result.med`: displacement, recovered nodal stress and, for damage stages,
  recovered damage, viewable in ParaView.
- Corrected legacy-archive preparations also retain `baseline_mesh/`,
  `whole_model_mesh.npz`, `mesh.med`, `conversion.log`,
  `quadratic_discrete.npz` and its encoding audit JSON.

Task 4 writes `diagnostics/comparison.json` and a terminal table. Archived
baseline peaks are nodal, whereas the diagnostic summaries use Gauss points;
do not interpret their subtraction as a pure crack effect. Compare the
diagnostic stages with each other using the same stress measure.

If stage 1 already develops the large wall-end peaks, investigate slicing,
mixed-order elements, ties and load transfer before damage. Stage 1 combines
those changes; it does not separate them individually. If the increase starts
at stage 2, investigate the band material/formulation and natural damage.
If it starts only at stage 3, investigate seed initialisation and redistribution.
These are diagnostic distinctions, not engineering acceptance criteria.

## Historical case-study findings (vertex-only springs)

The completed elastic, unseeded and seeded attempts for `case_study_00`
reached full load with matching archived input hashes. Their wall Gauss-point
tension/compression extrema were respectively 26.232/-40.348,
15.443/-51.733 and 15.450/-51.793 MPa. The seed changes the unseeded
headline extrema by less than 0.12%; it does not explain the original large
stress discrepancy. This does not exclude local seed effects.

Read-only audits of the saved MED fields produced `spatial_audit.json`,
`constraint_audit.json` and `peak_strain_audit.json` under the Git-ignored
`case_study_00/diagnostics` directory. They establish:

- Using recovered nodal stress in both elastic models still gives wall
  tension peaks of 0.996 MPa for the original baseline and 25.303 MPa for
  the elastic cracking mesh. Stress recovery alone does not explain the gap.
  The nodal tensile peak is at the other wall-end neighbourhood
  `(71.0684, 36.7326, -27.4960)` m; it is not the Gauss-point tensile peak.
- Independent TETRA10 displacement-gradient recovery at the elastic
  Gauss-point peaks reproduces 26.2316195 and -40.3480204 MPa. The recovery
  also passes an affine constant-strain check on those cells. These peaks
  exist in the computed displacement gradients, rather than only in summary
  extraction. At the same physical points, the original baseline's parent
  linear tetrahedra give principal stresses of approximately
  `(-1.583, -0.222, 0.418)` and `(-1.653, -0.084, 0.444)` MPa.
- All 9,678 hanging-node displacement constraints agree with their intended
  edge-endpoint averages to within 9.8e-17 m. Quadratic mid-edge geometry/order
  is consistent. This strongly reduces suspicion of incorrect tie projection,
  but is not a complete constraint-force or topology audit.
- Outer-support stiffnesses match the baseline arrays. Reactions balance the
  baseline applied resultant force and moment to relative errors of about
  4.1e-16 and 1.3e-14. This is a global balance check, not an independent
  verification of local pressure distribution.
- At common original wall nodes, the elastic cracking mesh differs from
  the baseline by about 0.98% in displacement L2 norm, with a largest
  nodal vector difference of 0.703 mm. Similar headline displacement maxima
  do not imply identical displacement fields or local strains.

The cut planes at -27.492088 and -27.490088 m create a 2 mm layer across
the whole wall, not just the seeded sector. Numerical quality checks reveal
features that may not be visible in an overall ParaView view:

- Original wall tetrahedra have mean-ratio quality at least 0.679
  (1 denotes an equilateral tetrahedron).
- Of 76,780 quadratic band tetrahedra, 24,683 have mean ratio below 0.01.
  The elastic tensile-peak cell has mean ratio 0.00119 and volume
  3.11e-7 cubic metres, versus 0.01953 cubic metres for its baseline parent.
- The compressive-peak cell has good mean ratio, 0.752, but is tiny:
  7.74e-8 cubic metres. Thus bad shape alone cannot explain both peaks.

These observations make cut-layer scale, local refinement, mixed-order
approximation and wall-end load transfer the next controlled investigation,
not a proven mesh defect. Positive volume and visual appearance do not prove
adequate conditioning; conversely, thin elements alone do not prove an
incorrect solution. A real geometric/boundary stress concentration resolved
by the new discretisation has not been ruled out.

### Controlled elastic experiments

Five subsequent isolated full-model elastic experiments converged within
their enforced 540-second Linux watchdogs. Solver/export elapsed times were
272-360 seconds. They use a single full-load increment; that control
reproduces the archived ten-increment elastic peaks to numerical round-off,
so changing the elastic load stepping is not responsible for the findings.
Inputs, command/mesh hashes, convergence, runtime bounds and unchanged
physical inputs were checked in the local
[investigation report](../case_study_00/diagnostics/elastic_controls/investigation_report.json).
The experiment driver and audit scripts are snapshotted with hashes in its
`audit_sources` sibling directory. All generated files remain Git-ignored.

| Elastic experiment | Wall Gauss tension MPa | Wall Gauss compression MPa | Maximum displacement component mm |
|---|---:|---:|---:|
| Sliced mesh, all linear elements | 4.395 | -8.000 | 9.7360 |
| Sliced mixed-order mesh, unchanged supports | 26.232 | -40.348 | 9.7392 |
| Same mixed mesh, remove only four springs at two hotspot nodes | 19.407 | -28.342 | 9.7407 |
| Same mixed mesh, remove springs on the cut-adjacent z = -27.5 m tread level | 4.220 | -6.251 | 9.7786 |
| Mixed-order mesh with a 0.1 m numerical slice, unchanged supports | 15.616 | -24.442 | 9.7396 |

The two spring-removal experiments are deliberate causal ablations, not
candidate engineering models or recommended fixes. The broader ablation
zeros 160 of 8,056 spring entries at 108 original wall nodes, representing
approximately 1.85% of interface X/Y stiffness and 1.19% of Z stiffness.
It retains the identical thin, poorly shaped mesh, materials and external
loads. The slice-width experiment is homogeneous elasticity with no damage
or initial seed: its cut spacing is purely a numerical partition, not a
change to a physical crack thickness. It is not approval to thicken a
seed in a fracture calculation.

At the same physical point as the original compressive Gauss peak,
removing just the two hotspot nodes' springs reduces compression from
40.348 to 2.978 MPa; the new extrema move to neighbouring loaded regions.
Removing the entire cut-adjacent tread spring set reduces the original
tensile-point maximum principal stress from 26.232 to 0.202 MPa, and the
original compressive-point minimum from -40.348 to -0.142 MPa.
These fixed-location values are independently recovered from displacement
gradients, not selected from a changed extrema location.

Before ablation, the compression-hotspot wall node carries spring-resultant
components of approximately `(59.2, 87.4, 224.2)` kN. The original baseline
gives `(58.1, 91.5, 231.9)` kN at that same node. Thus a broadly similar
concentrated force is being resolved by much smaller, enriched cells.
Enlarging the purely numerical cut spacing changes the global peaks
substantially while leaving material properties, physical geometry,
spring coefficients and external loading unchanged. This demonstrates
discretisation sensitivity; it does not establish a converged continuum peak.

An additional corner-topology audit found no nonmanifold faces and no
new boundary-face centroids off the original wall boundary: all 72,280
cut-wall boundary faces match original boundary triangles to the tested
tolerance. This rules against an obvious internal hole caused by slicing,
but is not a complete quadratic-interface audit.

**Causal conclusion:** the large elastic hotspots are driven by concentrated
original-node interface spring transfer, amplified by slicing and quadratic
enrichment near those nodes. They are not primarily a seeded-crack effect.
Bad element shape alone is not sufficient to explain them: the same
poorly shaped mesh loses the original hotspots when the local spring
transfer is removed. This identifies the computational mechanism behind
the reported amplification, not the physically correct stress distribution.

The previous implementation explains the mismatch:
[archive_arrays](whole_model_crack_mesh.py) carries the original
`seg2_wall_foundation_springs` into the cut/enriched mesh unchanged, and
[whole_model_crack.comm](cases/whole_model_crack.comm) continues applying
the original lumped `DIS_T` springs. Those are not an updated surface
operator for the refined quadratic trace. The earlier quadratic baseline
work already identifies the vertex-only coupling problem in
[Aster_Porting.md](../Aster_Porting.md); its
[quadratic_surface.py](quadratic_surface.py) provides consistent TRI3/TRI6
surface-integration building blocks.

**Correction requirements established by those experiments:** reuse that surface-operator work to
integrate the configured joint law over the actual refined wall trace,
with verified projection onto the unchanged foundation trace and preservation
of distinct coincident node identities. Existing `surface_blocks` assumes
matching triangle corners, so it cannot be applied blindly after wall-only
slicing; the nonmatching subdivision requires verified mapping/integration.
Retain the original physical stiffness coefficients and validate native
matrix assembly, patch tests, reactions and mesh-convergent spatial stress
profiles before rerunning damage comparisons. Do not remove supports,
suppress extrema, relax tolerances or select a convenient cut width as a
production fix. No production model or source job was changed by these
experiments. An approved physical corner treatment and a continuum
convergence study remain outstanding.

One initial linearization attempt failed before solving because resetting a
MED level discarded its family field. That experimental converter was
corrected to preserve family arrays and verify body/pressure groups on
reread. The failed attempt is retained and excluded from the result table;
it is not evidence about the dam model.

## Implemented consistent-interface correction

The whole-model production and diagnostic paths now integrate the saved
two-sided surface laws on the actual TRI3/TRI6 traces. This includes
wall-to-wall joints as well as wall-to-foundation interfaces:

- [source_contract.py](source_contract.py) exports paired source triangles
  and their unchanged directional coefficients in N/m3. The original nodal
  spring arrays remain available for the unchanged linear baseline.
- [whole_model_crack_mesh.py](whole_model_crack_mesh.py) tracks both sides'
  source-face identities through the two seed cuts and wall-band enrichment.
- [refined_interface.py](refined_interface.py) forms the common refinement
  of independently split traces, checks coverage and each source triangle's
  area conservation, and integrates the displacement-jump energy using exact
  TRI6 polynomial mass products. Coincident nodes remain distinct.
- [convert_whole_model_to_med.py](convert_whole_model_to_med.py) writes only
  the corrected joint operator plus the unchanged linear rock exterior
  supports. The general POI1/SEG2 encoding and grouped native importer are
  reused; old lumped joint springs are not added a second time.
  `quadratic_discrete.npz` is mandatory input unit 92 in production and
  corrected diagnostics. Missing operator output is a failure.

### Verification on the saved dam

The corrected elastic diagnostic is saved separately at
`diagnostics/corrected_interfaces/elastic/20261027T154055260027Z`.
Its [verification report](../case_study_00/diagnostics/corrected_interfaces/elastic/20261027T154055260027Z/verification_report.json)
and `audit_sources/` retain result hashes, code/script snapshots, a native
matrix benchmark and the independent saved-field audit. The original
archives and earlier diagnostics were not modified.

| Same full-dam elastic volume mesh and loads | Wall Gauss tension MPa | Wall Gauss compression MPa |
|---|---:|---:|
| Historical vertex-only joint springs | 26.232 | -40.348 |
| Corrected two-sided consistent interfaces | 1.517 | -2.937 |

This is not just relocation of the extrema. At the original tensile peak's
physical location, independently recovered maximum principal stress falls
from 26.232 to 0.092 MPa; at the original compressive peak's location, minimum
principal stress changes from -40.348 to -2.460 MPa. The corrected global
maximum displacement component is 9.73942 mm.

Measured checks:

- The native solver/export completed in **475.69 seconds**, inside an
  enforced 540-second Linux watchdog, at load factors 0 and 1. Earlier
  elastic controls established single-increment equivalence for this
  proportional homogeneous elastic problem.
- All coordinates, volume connectivity, body/band/seed/hanging-master groups
  and pressure/crack face connectivity match the archived mixed-order mesh
  exactly. Only the discrete interface operator changes.
- 10,796 source joint triangles become 11,016 slave and 10,796 master faces.
  Maximum relative overlap-coverage error is 8.28e-12.
  Integrated X/Y/Z stiffness stays
  `(1.768084257235e12, 1.768084257235e12, 2.762321143685e12)` N/m.
- A separate 54-DOF native matrix benchmark with independently partitioned
  quadratic traces and a directional cross-coupled support matches every
  assembled entry to **1.23e-11 relative matrix error**, below the existing
  1e-10 coefficient-encoding bound.
- Global reaction force/moment relative balance errors are **1.52e-11** and
  **2.80e-11**. All 9,678 hanging-node ties agree with endpoint averages to
  **8.94e-17 m**.
- Common original-wall-node displacement L2 difference from the old linear
  baseline is 0.303%; the largest nodal vector difference is 0.0402 mm.
- Real production `prepare` reproduces all 21 saved mesh/operator arrays
  exactly, including after the per-source-area validation was added.
- Focused source/mapping/patch/energy/encoding/workflow tests pass, as do
  three native MED roundtrip tests, including the corrected whole-model path.

The native run retains `DISCRETS_26` alarms for massless discrete springs,
their repeat-message limit, and the existing face-orientation deprecation
warning. The springs intentionally have no mass in this static model; no
warning was suppressed or convergence tolerance relaxed.

**Remaining acceptance work:** corrected unseeded and seeded nonlinear
damage comparisons have not been run. Neither the old results nor this
elastic check establish fracture predictions or mesh-convergent corner
stresses. Keep engineering validation false pending those comparisons,
spatial/mesh convergence and independent review.

## Reproducibility and safety

The saved `mesh.med` is either passed as read-only input for an explicit legacy
replay, or left untouched while a corrected copy is rebuilt. Existing source
files and archived results are not overwritten. Current code is snapshotted
into each `case.comm`; its hash and the current native-seed helper hash are
recorded. This is not a promise of bitwise reproduction of a historical runtime.
The inventory warns if the current production command differs from its
archived hash. A nonzero archived crack-face pressure is rejected for the
uncracked stages rather than silently changing the loading.

Linux solves run in isolated process groups. Cancellation/timeout cleanup
targets that diagnostic group, not app jobs. The `setsid --fork --wait`
launcher waits for the solver and propagates its exit code before cleanup
or output validation. A missing `summary.json` is a failed attempt, even if
the launcher returns zero; inspect the solver logs and retained Linux scratch.
Successful native scratch
directories are removed; failed native scratch is retained under the unique
`/var/tmp/lamb_aster_diagnostics/` location recorded in the manifest.
It can be large. After a hard host shutdown, inspect `diagnostics/solver.lock`
and its PID before manually removing a stale lock.

The entire `case_study_00` directory is Git-ignored: copied jobs, logs,
meshes and solver results must not enter source control.

## Bounded preparation check (PowerShell, from Lamb)

```powershell
Set-Location .\Gairezi
.\.venv\Scripts\python.exe -B -m Aster.case_study_diagnostics inventory --study-dir .\case_study_00
.\.venv\Scripts\python.exe -B -m Aster.case_study_diagnostics elastic --study-dir .\case_study_00 --prepare-only
.\.venv\Scripts\python.exe -B -m unittest Aster.test_case_study_diagnostics
```

`--prepare-only` writes a new prepared attempt without launching Code_Aster.
Launch the VS Code solver task to create and execute a fresh attempt.
Convergence and matching peaks are not dam-safety validation.
