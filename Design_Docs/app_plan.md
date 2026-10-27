# Dam Analysis Workbench - implementation plan

## 1. Goal and scope

Saved-job Code_Aster discrepancy investigation: see
[stage-by-stage diagnostic tasks](../Aster/CASE_STUDY_DIAGNOSTICS.md).
These operate outside the app pipeline on a Git-ignored job copy, preserving
the production configuration and archived outputs.

Build a Windows WPF application in Visual Studio that edits one dam configuration,
submits it to a lightweight local Python API, runs the existing Elmer workflow,
and displays the resulting analysis and reports.

The existing [config.json](../config.json) is the dam-data contract.
Preserve its `geometry`, `material`, `loads`, and `outputs` structure.
Do not create another independently maintained set of dam properties.

Use [Dam Analysis Workbench UI.png](./Dam%20Analysis%20Workbench%20UI.png) as the
visual reference: dark navy/red styling, left navigation, configuration tabs,
contour grids, configuration summary, JSON preview, and Submit button.
This is a native WPF frontend, not a Python-rendered web form.

**First release:** local Windows use, one active Elmer analysis, complete
configuration editing, validation, job monitoring, cancellation, and result
viewing. Code_Aster integration and interactive 3-D results are later milestones.
Do not build authentication accounts, a cloud service, a distributed queue,
or a web frontend for this first release.

## 2. Proposed architecture

```text
WPF application (C#, MVVM)
    |
    | HTTP over loopback: configuration JSON and job requests
    v
Python API (FastAPI + Uvicorn, one server process)
    |
    | starts an isolated Python worker process for each accepted job
    v
Existing mesh builder -> ElmerGrid -> MPI ElmerSolver -> post-processing
    |
    v
Per-job configuration snapshot, logs, mesh, solver result, summary and plots
    |
    v
API job status / result endpoints -> WPF results screen
```

Elmer does not consume JSON directly. The Python worker consumes the submitted
configuration and generates the Gmsh mesh and Elmer SIF before invoking Elmer.
Reuse the existing numerical model rather than reimplementing it in the API.

### Suggested project locations

| Location | Responsibility |
|---|---|
| Parent repository `Lamb.sln` | Existing Visual Studio solution; open this for Workbench development |
| Parent repository `Lamb.Windows.App/` | Existing WPF project, views, view models, models and HTTP client |
| Parent repository `Lamb.API/` | Existing Python API project; inspect and reuse rather than creating a duplicate |
| This repository's `Elmer/` | Shared configuration validation, mesh generation, solver and post-processing |
| This repository's `Aster/` | Future alternative backend using the same dam configuration |
| This repository's `config.json` | Version-controlled example/default dam configuration |
| `%LOCALAPPDATA%\Lamb\DamAnalysisWorkbench\jobs\<job-id>\` | Generated job inputs, outputs and logs, outside Git |

#### Repository and Visual Studio solution boundary

The WPF application and its solution already live in the parent `Lamb`
repository. Continue using its existing `Lamb.sln` and
`Lamb.Windows.App/Lamb.Windows.App.csproj`; do not create a second solution,
move the WPF project, or add this entire repository as a C# project. The parent
solution also already contains `Lamb.API`; inspect that project before adding
or replacing API infrastructure.

This `Gairezi` checkout is a separate, nested Git repository containing the
existing dam model. Copilot can work across it when Visual Studio is opened at
the parent repository/solution context, but the nested repository is not a
project reference. Changes under `Gairezi/` are versioned in this repository;
changes under `Lamb.Windows.App/` or `Lamb.API/` are versioned in the parent
repository. Keep those Git boundaries intact unless the owner explicitly
chooses to merge the repositories or configure a submodule.

The Visual Studio solution is an IDE/build container, not a requirement that
every source directory be a project. Use Python for the dam model and backend
tests, and keep `Gairezi/config.json` as the single dam-data contract.

Keep API address, executable locations, job storage and other application settings
separate from dam properties. These are deployment settings, not a second dam JSON.
Keep a single configuration snapshot per job as its authoritative input.

## 3. Existing workspace: reuse and gaps

The baseline workflow is already working:

- [model_config.py](../Elmer/model_config.py) loads and validates root configuration.
- [structure_assembly.py](../Elmer/structure_assembly.py) builds the combined mesh,
  converts/partitions it with ElmerGrid and runs four MPI solver processes.
- [analyze_stress.py](../Elmer/analyze_stress.py) generates CSV, summary JSON and plots.
- The combined solve and client-report generation are currently separate steps.
- The CLI writes into fixed workspace directories and some compatibility radius
  constants are populated from root configuration when modules are imported.

Before HTTP submission can be reliable:

1. Accept explicit configuration and output-directory inputs throughout the
   worker path. Never overwrite the repository configuration to submit a job.
2. Remove reliance on import-time root-configuration values from the submitted-job
   path, including audits and post-processing. Two different jobs must not share
   stale radius/material values.
3. Run mesh generation, solving and post-processing as one tracked pipeline.
4. Detect solver and report failures explicitly; existing files must not turn a
   failed job into apparent success.
5. Validate generated results before exposing them to the frontend.

Preserve the existing CLI defaults so VS Code tasks and command-line runs continue
to work.

The explicit `Elmer.three_body_solver` entry point and current VS Code solver
tasks now include the approved rock volume and configured interface springs.
They snapshot configuration and use unique external run directories. The
existing `structure_assembly.py combined-solve` CLI remains the legacy two-body
workflow. The app/API now explicitly launch the three-body entry point for
new jobs using `--config-root`, `--output-dir` and `--report-path`; historical
jobs without a `model_kind` retain legacy paths. The worker preserves the job's
snapshot and rejects nonempty output/report paths. API success requires the
full PVTU partitions, concrete-only reporting VTU and matching finite per-body summary.
The WPF schema/model preserve rock domain/mesh settings through
load/save/submission, while bedrock remains read-only in the editor. Default
ParaView opening uses the reporting VTU, with a separate full-domain action.
Headline stress extrema/utilization cover the wall; the table reports wall and
optional concrete foundation only. Reference-material stresses are excluded
from reporting entirely. The desktop rejects older API versions lacking
`canonical_model: three_body` before submission.

FEniCSx now imports the exact saved canonical source mesh (legacy name
"three-body"; it now holds two bodies: wall = 1, bedrock = 2) and uses distinct
concrete/rock material fields. There is no plinth: the wall sits directly on
the bedrock and its underside follows the bedrock contours. The approved
bedrock support model is `distributed_elastic_granite`: without an optional
concrete foundation, the wall/rock seat is a directional elastic spring joint
using `geometry.bedrock.support.spring_n_per_m3`, discretised as
relative-displacement mortar springs in both solvers. The alternative `bonded`
model shares wall/rock seat displacement nodes, with zero relative movement
but nonzero common deformation allowed; its spring coefficients are inactive.
In the foundation-trench workflow, when `geometry.foundation` is configured,
`geometry.foundation.support` selects the wall/foundation connection and
`geometry.bedrock.support.model` selects the foundation/bedrock connection
independently; either may be bonded or use its configured springs. The Python
and desktop validators require explicit `geometry.foundation.support` whenever
a foundation is present; it never inherits `geometry.wall.joint_support`.
The checked-in configuration uses explicit wall/foundation springs
(X/Y = 1.25e9, Z = 3e9 N/m3) with a bonded foundation/bedrock interface.
New foundations enabled in the desktop explicitly select bonded support.
The Elmer and FEniCSx paths require a bonded interface to have conforming shared-node
faces, and do not assemble spring terms for it. Do not ground a bonded
interface. Elmer spring BCs explicitly use the normal Galerkin projector
(`Normal Projector = True` with `Galerkin Projector = True`), not the
deprecated weighted fallback; spring axes remain global X/Y/Z and connected
near-planar groups remain separate. New jobs run an
elastic comparison first. Fracture requires matching
source/configuration hashes, passed displacement and recovered concrete nodal
tensor parity (each relative error <= 1e-4), explicit nodal peak metadata,
force equilibrium (<= 1e-6), crack-band resolution and a converged refined
zero-damage baseline. Canonical headline peaks use recovered nodes, not
element-mean tensors; the desktop warns about historical unlike measures.
Canonical elastic solves use GMRES/right GAMG instead of CG. Fracture requires
2.0 elements per length scale (0.25 m length scale, up to five refinement levels
and 4,000,000 cells); lower explicit canonical targets are rejected.
If the limit is exhausted, stop rather than accept an under-resolved solve.
Wall damage evolves; bedrock and seat springs remain elastic.
The crack location is never chosen automatically. After the verified elastic
baseline the job pauses at stage `awaiting_crack_location` (cracking status
`awaiting_location`); `GET /jobs/{id}/crack-selection` returns the binned
Z-Y wall maximum principal stress map with peaks, the desktop plots it,
and the user's click is submitted with `POST /jobs/{id}/crack-location`
(`y_m`, `z_m`; rejected unless the point lies on a concrete bin). The API then
runs `run-fracture --crack-location Y Z`. Cancelling while paused skips cracking
and keeps the completed baselines.
Failures preserve the completed Elmer result and expose the failed FEniCSx
stage. Hybrid and experimental Amor are supported; Amor is the app default
(the API `POST /jobs` default and the pre-ticked checkbox; unticking sends an
explicit `fracture_mechanics=hybrid`), and the Python `JobManager.submit` default
and stored historical jobs stay hybrid. Hybrid degrades compressive stiffness:
a full-dam job's seeded, fully damaged band collapsed under self-weight (strain
norm about 13.7 at load step 2). Amor cracking runs on 4 MPI processes; hybrid
stays serial because its parallel parity is unverified. The historical
bonded-reference hold remains for legacy jobs. Full-size elastic parity and
small serial/MPI fracture benchmarks are verified, not full-dam fracture
convergence or engineering safety. Historical results remain readable.
The wall underside follows piecewise-linear interpolation of the unchanged
bedrock survey contours across the full wall thickness, without elevation
snapping. The wall spans the chainages where the rock lies below the crest
(z = 0) and tapers to zero height at both ends. The locked rock seat conforms
exactly.
There is no automatic time limit on user app jobs; cancellation stops only the
selected job's worker tree. API dependency staging includes Gmsh. See the
FEniCSx README for verification evidence and remaining limitations.

### Multi-solver cracking comparison objective

The intended submitted-job sequence is Elmer baseline, FEniCSx baseline, Aster
baseline, then FEniCSx and Aster cracking analyses, followed by baseline and
cracking comparisons. The cracking comparison is the primary deliverable.
The user accepted the observed elastic wall/foundation stress discrepancies
for progressing this comparison work; that is not independent engineering
validation or acceptance of unmeasured future cases.

Both cracking analyses must use the location explicitly selected in the UI.
They must share a frozen physical seed geometry and loading history, not
independently choose a peak-stress location or seed face. Each analysis retains
its own working directory, convergence evidence and outputs. Reports compare
wall and optional concrete foundation separately.

The seed rule comes from the optional `cracking` array in `config.json`
(zero or one crack object is currently accepted; an empty array means no crack,
so the job skips crack selection and both cracking analyses, reports
`cracking_status` `skipped`, and runs only the FEniCSx and Code_Aster
baselines; the array leaves room for
multiple cracks, which are not yet implemented; the app has no Outputs tab and
always writes displacement and tensile-stress outputs, with `outputs.cracking_risk`
true exactly when a crack is defined, which the validator requires)
(edited on the Configuration > Cracking tab; validated by
`Elmer/model_config.py` `validate_cracking`), falling back to
`FEniCSx/fracture_defaults.json` `automatic_crack` when absent: `face`
upstream or downstream (downstream seeds start at the local downstream face and
carry tailwater pressure), `depth_m` (default 3 m, less than the wall radial
width), `opening_m` (0.001-0.01 m, default 2 mm), and either `full_width` (end
`buffer_m`, 0-2 m, default 1 m, true distance at the seed's inner radius, from
the wall/foundation contour so only the wall is cracked) or a `length_m` arc
centred on the selection and clipped to that buffer. A zero buffer still keeps
0.01 m so the seed stays in wall concrete. The API snaps a selection inside that
clearance, or above the crest, to the nearest valid point, stores both the
snapped `crack_location` and the `requested_crack_location`, logs the snap and
the UI reports it.

An optional `fracture_debug` block (Configuration > Cracking > Debug, which
also holds the Amor/hybrid mechanics choice) overrides the FEniCSx and Aster
`fracture_defaults.json` values per job for process-of-elimination debugging.
The block is nullable: null or absent means the solvers use those files
unchanged. `config.json` ships it with Friday's working values (returned as-is
by `/config/default`); `/config/fracture-defaults` builds the shipped-default
block from the two files so they stay the single source. `Elmer/model_config.py`
`validate_fracture_debug` checks it. See the README cracking section for the
fields.

The current policy is **sequential cracking** (schema-v3 metadata: mode
`sequential`, concurrency one), superseding the earlier parallel-only policy.
Running both solvers exhausted the 24 GB WSL VM: FEniCSx cracking held about
10 GB, and the whole-model Aster crack failed at its 18 GB limit. FEniCSx
cracking runs first. Aster cracking is `queued` and starts when FEniCSx
cracking and the Aster elastic baseline have both ended, whatever their
outcome. The Windows `GetPhysicallyInstalledSystemMemory` probe is retained
for advisory diagnostics only: below 16 GiB produces a visible memory warning,
not a block. Every detected capacity explicitly carries no memory-fit
guarantee.
Historical metadata is preserved as recorded, including schema-v1 policies;
jobs without a policy do not acquire a fabricated one on reload.
Future remote/VM solver hosting (for example Azure) is deferred and is not
implemented or configured by this change.

**Implementation boundary:** for three-body jobs, the API now launches native
Aster cracking (`Aster.whole_model_crack`: the whole three-body model with a
damageable wall band at the selected elevation, run by Linux Code_Aster in WSL)
after FEniCSx cracking ends, at the same frozen UI seed. Each run has its own status (`aster_cracking_status`,
`aster_cracking_error`), log (`aster_cracking.log`, `GET
/jobs/{id}/cracking/aster/logs`) and output root (`cracking/aster_crack_submodel`).
The job finishes only when both runs end; cancellation stops the running
process tree and cancels a queued Aster run. An Aster failure, including a missing installation, is reported without
failing FEniCSx. The API checks the Aster manifest's configuration and frozen
seed hashes before exposing scenario `Aster_Crack_Submodel`. The desktop
shows Aster status, a summary beside the FEniCSx crack peak and a ParaView
button.

**Per-job Aster elastic baseline and grouped comparison:** after Elmer
validation, three-body jobs also launch the full-dam Code_Aster linear elastic
pipeline (`Aster.run_three_body_pipeline --source-mesh
<job>/output/model/three_body.msh --output-dir
<job>/cracking/aster_elastic_baseline`) beside the FEniCSx baseline. It has its
own status (`aster_baseline_status`, `aster_baseline_error`), log
(`aster_baseline.log`, `GET /jobs/{id}/aster/baseline/logs`) and report (`GET
/jobs/{id}/aster/baseline`, concrete wall/foundation peaks only). The pipeline
exits 2 while parity is unverified, so the API judges success from
`run_manifest.json` status `completed`, the job's own source mesh and
configuration hash, and finite wall peaks. Its failure never blocks FEniCSx.
The job stays active (stage `aster_baseline`) until it ends; cancelling crack
selection does not stop it, cancelling the job does. The Comparison table
groups columns as Baseline (Elmer, FEniCSx, FEniCSx refined, Aster), Cracked
Scenario (FEniCSx, Aster) and Crack Difference (FEniCSx, Aster). The Aster
cracked value is the wall-submodel peak tension, so it is not comparable with
the whole-wall FEniCSx peak, and the Aster crack difference stays blank until
a same-scope uncracked submodel reference exists. Full-dam (not submodel)
Aster cracking and resource enforcement remain future work. See
`Aster_Porting.md` for limitations.

Native Aster concrete damage versus FEniCSx AT2 was approved as a comparison
of different formulations. The installed Aster 17.4 catalogue provides
`ENDO_FISS_EXP` with `3D_GRAD_VARI`. Its material interface requires tensile
strength, compressive strength, fracture energy, full damage-band width and
either softening parameter P or initial fracture energy G_INIT. The user has
since supplied 20 MPa compressive strength (stored in the authoritative
`material.concrete.compressive_strength_pa`) and authorised provisional
regularisation and softening choices. Initial Aster defaults are a 0.50 m full
band and P = 5, explicitly not test-calibrated. Shared fracture energy is now a
provisional 200 J/m2 (fib MC2010 estimate from 20 MPa with a large-aggregate
allowance; previously 2700 J/m2). It is read by both solvers from the FEniCSx
defaults, with no second material copy.
See `Aster_Porting.md` for the documentation basis, resolution/sensitivity
requirements and native nine-case material-admissibility check. Do not assume
Aster's band width is equivalent to the FEniCSx phase-field length scale or G_INIT equals GF.

User-located FEniCSx cracks now use AT1 with the length scale calibrated to the
configured tensile strength, `l = 3 E Gc / (8 ft^2)` (Tanné et al. 2018), i.e.
0.656 m for the current 35 GPa / 200 J/m2 / 2 MPa values. The previous AT2
`l = 0.25 m` implied about 1.72 MPa with no elastic phase, which let diffuse
damage spread below the configured strength; a job then failed when GAMG could
not solve the resulting stiffness contrast. After a linear-solver failure,
mechanics falls back once to MUMPS LU for the rest of the run, as the GMRES
preconditioner (hybrid; and Amor, which reuses the factor and refactorises
only when GMRES needs more than 30 iterations). This is recorded in the
diagnostics and report. The fallback factorises out-of-core on the WSL disk
after an in-core MUMPS factor of a refined hybrid dam (about 2.9 million cells,
serial) failed to allocate 7.9 GB (INFOG(1)=-13); in a 181,875-DOF benchmark
this cut MUMPS memory from 2,717 MB to 340 MB at the same solve time. When Amor Newton reaches its 30-iteration limit while
its residual is still falling, it continues from the last iterate, up to 3
more rounds, with the same convergence test. It does not switch to MUMPS,
because the linear solves had converged. Each Newton iteration is logged as a
`newton_iteration` mechanics event, which the app shows as the live stage text
(residual and target). Older AT2
reports remain readable. This calibration is a documented
choice, not a fit to the Aster result or test data.
Native 3-D nonlinear material benchmarks now cover intact, uniformly
pre-damaged and locally pre-damaged confined tension, plus free-contraction
tension; unloading/irreversibility and all-step ParaView output are checked.
These are small synthetic benchmarks, not resolved dam crack propagation.
The verified three-body FEniCSx baseline now saves the original Elmer selection
fields with a hash. On the UI click the API freezes the physical seed and
solver inputs once into the job's `cracking/prepared_seed.json` before spawning
the cracking worker. It verifies the baseline, configuration and source archive;
missing or changed source evidence explicitly fails without launching cracking.
The FEniCSx worker consumes `--prepared-seed`, verifies the configuration and
independently imported source fields, and never chooses another face or reads
mutable placement defaults. Results must report the exact prepared hash and
selection metadata. Older baselines without source evidence require a new run.
The runner also records the hashed `shared_seed` alongside its UI-selected crack
report, including exact geometry, face and load schedule. Placement and packet
preparation do not require importing DOLFINx. The same frozen seed drives the
parallel native Aster crack submodel described above.
See `Aster_Porting.md` for evidence and limitations. Aster GRAD_VARI needs a
quadratic displacement mesh, so a new dam zero-damage baseline and its
interfaces/supports must be checked before app jobs can launch native cracking.
Source-pinned quadratic degree elevation and exact TRI6 joint/support
integration are implemented and checked on a small native matrix benchmark.
The full candidate retains every original volume cell and split interface,
but increases the displacement system to at least 4.30 million unknowns.
The native characteristic setup bottleneck is resolved by coefficient grouping
with an audited 1e-10 per-block relative encoding bound and group allocation
during native import. The full candidate uses 17,924 coefficient groups in
place of 421,911 individual entries; measured encoding errors are below 5e-11.
It now reaches `MECA_STATIQUE`, but the approximately 456-second attempt fails
native memory allocation under an 8 GiB budget: an additional 9,986 MiB was
requested with 3,198 MiB already in use. A MUMPS out-of-core attempt still
estimated at least 13,561 MiB for factorisation against 3,758 MiB remaining
and did not finish within the bounded developer attempt. A distinct native
GCPC/zero-fill incomplete-factor trial likewise produced no converged output
within that budget; it is not an accepted solver replacement. Neither
attempt establishes convergence or a fit on a 16 GiB machine. Resolve the
assembly/discretisation memory or verification-host requirement and recheck the baseline before
enabling native cracking app jobs. A zero launcher exit is not success when
native exceptions occurred. Do not impose developer timeouts on user jobs.
The paired scheduler and final cracking comparison remain unimplemented.
For the planar native zero-load seed test, the official continuum damage
equation now supplies an independent obstacle-profile reference. Targeted
normal mesh grading preserves the exact physical seed and obtains a
0.0678% damage-integral error and 0.235% profile L2 error at N256. A 2x2
lateral repeat reproduces the 1x1 normal profile and all state integrals.
This verifies that particular planar equilibrium against its analytic
reference; the separate two-successive-refinement test remains unmet.
Finite crack edges, loaded dam propagation and paired execution remain
unverified. See `Aster_Porting.md` for exact scope and failed finer attempts.
The finite-sector specimen detects damage at its sampled lateral boundary.
Its developer runner now supports increasing the lateral domain at unchanged
physical mesh spacing, with consistent MED geometry, volume weights and a
fixed 0.05 m boundary diagnostic. Native mesh verification passes; the
extent-sensitivity study is not yet accepted.
At 0.125 m lateral spacing, increasing the finite-sector specimen width
from 1 to 2 m changes its equilibrated scalar damage integral by 0.01016%;
the larger specimen has zero sampled outer-shell damage. This checks only
that integral's extent sensitivity at that spacing, not finite-edge mesh
convergence or dam propagation.
The finer enlarged-domain specimen currently fails reproducibly with a
native `FULL_MECA` / `MVCA_HEXA20` floating-point exception at the same
element under both 2048 and 4096 MiB allocations. The larger allocation
removes the memory warning but not the exception. No accepted finer Windows
history is available; that native exception remains unresolved, not a timeout.
An independent Linux 17.4.0 build subsequently solved the exact failed
Windows case in 33.3 minutes using identical saved mesh bytes and physical
inputs (only the repository path changed). All five states pass the strict
history gates and actual ParaView field checks. This provides a viable
verification runtime, not a diagnosed Windows defect or mesh-converged
dam-cracking implementation.
Separate native TETRA10 seed checks now pass on both 1 m and 2 m lateral
domains at the same spacing. Their equilibrium damage integrals differ by
0.002503%, with zero sampled outer-shell damage on the larger domain.
The coarser tetrahedral integral differs from the hexahedral result by
12.81%; this is unresolved discretisation sensitivity, not accepted
cross-element agreement or dam cracking verification.
An opt-in mixed-order candidate now retains the original linear elastic
domains and elevates only the split wall, preserving every source volume
cell. Exact mixed TRI3/TRI6 interface assembly is verified. With explicit
approval for longer developer runs, native GCPC completed its elastic solve
in approximately 35.6 minutes within an 8 GiB process budget; wall/foundation
stress VTUs were read in ParaView. This resolves execution for that candidate,
not baseline acceptance: the wall recovered tensile peak is 1.731 MPa versus
Elmer's 1.122 MPa, despite a 0.406% displacement L2 difference. Matching
cell-centroid constitutive stresses have a 7.03% tensor L2 difference and
peaks 1.090/1.135 MPa. Investigation must separate recovery, displacement
order and joint-law effects before enabling dam cracking. Full evidence and
separate concrete-foundation figures are in `Aster_Porting.md`.
Six small native histories now include uniform and single-cell initial damage
0.999, verifying retained seeded integration-point identities without healing.
Recovered nodal damage may leave [0, 1] during extrapolation; these contours are
explicitly named `damage_recovered_nodal`, remain unclipped and produce
persisted warnings. Physical damage checks use integration-point histories.
The native initial-state adapter now consumes the frozen UI seed without
relocating or widening it, audits all nine internal variables against native
Gauss-point identities, restricts initial damage to the wall and rejects a
mesh that misses the physical seed. A synthetic native check confirms exact
placement, no healing and explicit missed-seed rejection. Its very small
domain develops strong nonlocal damage outside the seed, reported as a warning:
this is not accepted propagation accuracy, seed-equilibration evidence or a
resolved dam calculation. Four-thread OpenMP did not resolve the bounded
full-dam baseline runtime blocker either; no experimental iterative solver
was retained as an accepted replacement.
A larger one-metre planar native specimen now holds the same 0.002 m seed
in an explicit mesh layer. Zero-load equilibration produces a symmetric,
localized damage profile, with seeded points retained and no damage beyond
0.3 m from the seed plane. Normal-direction refinements N=8/16/32/64 pass
placement, bounded state, no-healing and localization checks; all five states
on N=16/32/64 were read in actual ParaView. The Gauss-integrated damage measure
still changes by about 2.41% from N=32 to N=64, nonmonotonically over the
sequence. N=128 reached its native CPU limit before accepting equilibration.
This does not establish mesh/band convergence, finite crack-edge behavior or
full-dam propagation accuracy. See `Aster_Porting.md` for measured evidence.

Three-body stress review uses a separate reporting export: the whole wall and
rock within 5 m of their nearest surface in reference coordinates. Keep the
full-domain solution for audit; never replace or truncate the solver domain to
implement this display filter. The filtered VTU and per-body stress summary are
recorded separately in the run manifest. Distance and fields are interpolated
at clipped-cell boundaries, and remote-boundary influence still requires
independent sensitivity verification. See the README for existing-run export.

## 4. Configuration contract and validation

### Preserve current physical meanings

| Input | UI/API treatment |
|---|---|
| Wall radii and radial width | Validate width and midpoint relationships; display units in metres |
| Contour `height` | Label as signed elevation relative to crest z = 0 |
| `geometry.wall.height_m` | Clearly identify current nominal metadata; contours control the actual height |
| Global element size | Controls the shared combined mesh |
| Local element size | Identify which generator uses it; do not imply it refines the current combined mesh |
| `water_height` | Reservoir depth above the deepest wall seat; with `overflow_head` sets the water surface |
| `overflow_head` | Depth above the crest when overflowing; adds the crest surcharge |
| `overflow_water_level` | Derived reference depth: water height plus overflow head |
| Explicit directional springs | Without `geometry.foundation`, the wall/bedrock seat uses `geometry.bedrock.support.spring_n_per_m3`; with it, that setting controls foundation/bedrock and `geometry.foundation.support` controls wall/foundation. Springs act on relative displacement only (no grounded spring bypass); bonded interfaces share nodes and leave spring coefficients inactive. The legacy wall-only solve retains its grounded bedrock support. Material modulus does not recalculate springs. The temporary top-level `interfaces` override is no longer accepted. |
| Bedrock material | Active in the integrated three-body solver; the legacy wall-only solve represents the foundation with grounded springs instead |
| `geometry.bedrock.domain` | Optional explicit stream/bank margins and bottom elevation shared by the preview, integrated three-body solver and diagnostic; inactive in the legacy wall-only solve |
| `geometry.bedrock.mesh` | Optional conforming 3-D rock grading: 0.5 m target within 2 m of the locked wall seat, grading to 15 m by 50 m; shared by preview and three-body preparation. Missing settings retain legacy layered meshing; lengths are not cell volumes or maximum-edge guarantees. |
| Output flags | Currently not fully wired; implement defined behavior before presenting them as working controls |

No silent physical reinterpretation as part of the API work. If actual-height
editing or reservoir-depth behavior is wanted later, agree the transformation
and add numerical regression tests first.

### Contract tasks

- Refactor validation to accept a JSON object as well as a file path.
- Use one Python validation/model definition to generate the API schema and to
  validate worker inputs; avoid competing Pydantic and legacy validators.
- Reject missing/unknown fields, incorrect types, booleans used as numbers,
  nonfinite values, invalid radii, invalid mesh sizes and invalid material values.
- Validate unique, increasing bedrock chainages with finite signed elevations
  that dip below the crest (z = 0) so the wall has a seat.
- Validate exact X/Y/Z spring entries, supported support models/material mappings,
  boolean output flags, and the reference water-depth relationship.
- Return field paths and messages that WPF can associate with individual controls.
- Return warnings for nominal or inactive properties instead of implying they
  affect the analysis.
- Keep derived frontend fields read-only where appropriate; serialize them into
  the existing contract without creating a second persisted source.
- Record an API/contract version in service/job metadata without changing the
  dam JSON unless a schema change is deliberately agreed.

## 5. Lightweight Python API

Use FastAPI for request validation and documented endpoints, and Uvicorn as the
local HTTP host. Add dependencies only when implementing this phase and verify
compatibility with the project's selected Python interpreter.

### Proposed endpoints

| Method and route | Purpose |
|---|---|
| `GET /health` | API version, backend availability and prerequisite checks |
| `GET /config/schema` | Current editable configuration schema and field capabilities |
| `GET /config/default` | Read the repository's example/default configuration |
| `POST /config/validate` | Validate a configuration; return errors, warnings and derived preview values |
| `POST /jobs` | Submit the dam configuration as the JSON body; return HTTP 202 with job ID |
| `GET /jobs` | List persisted local job history |
| `GET /jobs/{id}` | State, stage, timestamps, warnings and failure information |
| `GET /jobs/{id}/logs?offset=...` | Incremental bounded log retrieval |
| `GET /jobs/{id}/progress?since=...` | Normalised solver convergence events (log10 residual/tolerance per iteration) for the activity monitor; reads the job's own logs only (`job.log`, `cracking.log`, Aster `solver.live.log`/`solver.log`) |
| `POST /jobs/{id}/cancel` | Request cancellation of that job's process tree |
| `GET /jobs/{id}/crack-selection` | Baseline Z-Y stress map for user crack-location selection (409 until the verified baseline map exists) |
| `POST /jobs/{id}/crack-location` | Submit the selected `y_m`, `z_m` and start cracking (only while awaiting) |
| `POST /jobs/{id}/stages/{stage}/rerun` | Rerun a failed, cancelled or interrupted cracking stage (`fenicsx_cracking` or `aster_cracking`) with `{"fracture_debug": {...} \| null}`; 409 frozen/busy, 422 invalid (Phase 6) |
| `GET /jobs/{id}/attempts` | Every attempt of the job, oldest first, with its `fracture_debug`, stage results and archive folders (Phase 6) |
| `GET /jobs/{id}/results` | Structured metrics and available artifact manifest |
| `GET /jobs/{id}/artifacts/{artifact-id}` | Serve an allowlisted file belonging to that job |

For version one, `POST /jobs` always selects Elmer; backend choice is service/job
metadata, not a duplicate dam configuration. Add Code_Aster selection later.

### Job lifecycle

`queued -> running -> succeeded | failed | cancelled | interrupted`

Running stages: `validating`, `meshing`, `converting`, `solving`, `postprocessing`.
Show stage and elapsed time, not fabricated percentage progress from solver logs.

- Permit one active job initially; reject a second submission with a clear 409
  busy response rather than adding queue infrastructure.
- Persist the accepted configuration, its hash, executable/version metadata,
  timestamps, status, complete logs and result manifest under the job directory.
- Use atomic writes for job metadata and configuration snapshots.
- Execute the heavy work in a separate process; do not block API request handling.
- Launch subprocesses with explicit argument lists and working directories,
  never interpolate request values into shell commands.
- Capture stdout/stderr from ElmerGrid, MPI Elmer and report generation.
- Mark success only after required fresh artifacts exist and parse successfully.
- Validate the PVTU and all referenced partition VTUs, not just the wrapper file.
- Keep partial artifacts for diagnostics, marked incomplete on failed jobs.
- Cancellation must stop only the job's worker/MPI process tree and wait for
  termination before accepting another job. Never kill processes by executable name.
- On restart, reconcile persisted active jobs and owned process identifiers;
  mark abandoned jobs interrupted and prevent orphaned workers from overlapping.
- Poll status/logs from WPF for the MVP; streaming/WebSockets are unnecessary.
- A finished three-body job can rerun a failed, cancelled or interrupted
  cracking stage (Phase 6). The job returns to `running`; earlier stages are
  reused, succeeded stages are frozen, and superseded outputs and logs are
  archived under `attempts/<n>/` rather than deleted.

### Local-service boundary

Bind to `127.0.0.1`, not all interfaces. Use a per-session local token or equivalent
local-client protection; loopback alone does not authorize callers.
Do not enable permissive browser CORS. Limit payload size, job storage and
expensive mesh requests, and reject unsafe artifact paths.
Do not accept arbitrary executable paths or shell commands from HTTP requests.
Keep user job data local.

## 6. WPF frontend

Use C# WPF with MVVM, asynchronous `HttpClient`, `System.Text.Json`, and
`CommunityToolkit.Mvvm`. Select a supported .NET target available in the user's
Visual Studio installation at implementation time.

### Screens based on the mock-up

| Screen | First-release content |
|---|---|
| Home | Backend readiness, recent jobs and new/open configuration actions |
| Configuration / Geometry | Mesh spacing, wall radii, bedrock contour DataGrid and support editors |
| Configuration / Material | Concrete/bedrock fields with units and applicability notes |
| Configuration / Loads | Gravity, water density, tailwater and overflow controls; reference-depth labels |
| Configuration / Outputs | Implemented result selections, with unsupported features visibly disabled |
| JSON Preview | Read-only JSON synchronized with the form; import/export full configuration |
| Run / Results | Stage, elapsed time, logs, cancellation, metrics, images and artifact links |
| Settings | API connectivity and local environment status; no duplicate dam properties |
| About | App/backend versions and engineering-validation limitations |

### UI behavior

- Match the mock-up using reusable WPF resources/styles, rather than hard-coded
  styling in every control. Use approved local branding assets.
- Keep one configuration model bound to forms and JSON preview.
- Provide add/remove/edit contour rows and load/save the complete configuration.
- Support invariant-culture JSON serialization independently of Windows number
  formatting. Convert display units explicitly, for example GPa to Pa.
- Show inline validation and section-level summary states; Submit is enabled only
  when API validation and backend readiness pass.
- Preserve the editable draft while a job runs. Results show the submitted
  configuration snapshot, not the current edited draft.
- Prevent duplicate submissions and handle network timeout/disconnection without
  presenting a running job as failed or submitting it twice.
- Show real result extrema and units, links to locations and configuration hash.
- Display existing report images first; offer CSV downloads and opening the PVTU
  in an installed ParaView application. Embedded 3-D viewing is not an MVP dependency.
- Use 2-D contour charts for geometry preview. Any schematic 3-D preview must be
  labelled schematic, not the audited solver mesh.
- Maintain keyboard navigation, readable contrast, resize behavior and DPI scaling.

## 7. Result contract

The normal comparison flow is Elmer uncracked baseline, FEniCSx uncracked
linear-elastic baseline, then FEniCSx fracture, all isolated under the same job
and submitted configuration snapshot. The uncracked FEniCSx stage is required,
not selected by environment/debug flags. Its failure blocks fracture while
preserving Elmer results; fracture failure preserves both completed baselines.
Track each FEniCSx stage's status and error, validate completed reports and
configuration hashes, and never substitute stale results.

The comparison table contains Baseline (Elmer), Baseline Not Phased (FEniCSx),
Crack Scenario (FEniCSx), and Difference. Difference is cracked FEniCSx minus
uncracked FEniCSx, not cracked FEniCSx minus Elmer. Historical jobs without the
FEniCSx baseline remain viewable with an explicit missing-baseline message and
no difference. Never borrow another job's baseline.

Create a small backend-independent results DTO containing:

- Job ID, backend, configuration hash and completion status.
- Maximum displacement magnitude in metres and its physical location.
- Maximum tensile and compressive principal stress in Pa and their locations.
- Tensile-strength utilization where requested; label it an indicator, not proof
  of cracking or dam safety.
- Mesh statistics, convergence status and warnings.
- Artifact IDs, types, labels and download URLs.

Reuse existing stress calculations. Add a verified displacement-summary
calculation because the current stress report is not a complete API result contract.
Define output selections precisely: required native solver fields remain available,
while optional reporting metrics/plots can be omitted when not requested.
Do not advertise nonlinear damage/crack propagation that the elastic model does
not implement.

## 8. Phased TODO list and acceptance gates

### Phase 1 - submission-ready solver boundary

- [ ] Document the current configuration fields and active/nominal capabilities.
- [ ] Refactor shared configuration validation for object and file input.
- [ ] Thread explicit configuration and job-output paths through mesh, audits,
  solver input and post-processing; remove submission-path global state.
- [ ] Keep the existing CLI working; add explicit config/output options.
- [ ] Join solve and client-report generation into one callable worker pipeline.
- [ ] Add fresh-artifact checks, result manifest and structured summary.
- [ ] Test two distinct configurations in one Python process and in separate
  workers without modifying the repository configuration.
- [ ] Confirm a clean job directory regenerates all required artifacts.

**Gate:** CLI runs from an arbitrary job configuration, produces complete results
in its own directory, and preserves the existing numerical baseline.

### Phase 2 - API and job management

- [ ] Add FastAPI/Uvicorn dependencies and the API entry point.
- [ ] Implement health/default/schema/validation endpoints.
- [ ] Implement submission, single-job busy handling, polling and logs.
- [ ] Persist snapshots, hashes, job metadata and history.
- [ ] Implement cancellation and restart/interrupted-job handling.
- [ ] Implement result/artifact endpoints and local-service protections.
- [ ] Test with a fake backend: success, nonconvergence, missing artifacts,
  post-processing failure, invalid input, busy response and cancellation.
- [ ] Run one real Elmer job through HTTP and compare against the same CLI input.

**Gate:** JSON submission returns promptly, a real job completes asynchronously,
and results are attributable to the exact submitted snapshot.

### Phase 3 - WPF editor and mock-up

- [ ] Create the Visual Studio solution and MVVM application shell.
- [ ] Add mock-up styling and responsive left/content/summary layout.
- [ ] Implement configuration models and explicit JSON property mappings.
- [ ] Implement geometry/material/load/output editors and contour grids.
- [ ] Implement import/export, synchronized JSON preview and dirty-draft handling.
- [ ] Connect validation/readiness and map API errors to controls.
- [ ] Implement charts and applicability/help text.
- [ ] Verify JSON round trips preserve every field and number.

**Gate:** importing the existing configuration, editing it and exporting it produces
a valid object accepted by the shared validator.

### Phase 4 - end-to-end desktop analysis

- [ ] Implement Submit, job polling, incremental logs and Cancel.
- [ ] Implement result cards, report-image viewing and artifact downloads.
- [ ] Implement local job history and reopening a submitted snapshot.
- [ ] Add offline/reconnect/busy/failure UX and prevent duplicate submissions.
- [ ] Run the supplied dam configuration from WPF through the API to Elmer.
- [ ] Change a known active parameter and verify the expected mesh/load/output change.
- [ ] Verify changes to nominal/reference fields show the documented warnings.
- [ ] Compare WPF/API and CLI results using identical config and solver versions.

**Gate:** the user can configure, submit, monitor and inspect a complete analysis
without opening VS Code or editing Python.

### Phase 5 - installation and handover

**Deployment decision:** provide one Windows installer that bundles WPF, the
Python backend and a tested Elmer distribution, and handles Microsoft MPI as a
prerequisite. The client experience is **Install -> Launch -> Configure -> Submit**.
Users must not need to install Python, run pip, configure PATH or set environment
variables manually.

#### Application-local dependencies

Suggested installed layout:

```text
DamAnalysisWorkbench\
  Workbench.exe
  backend\
    ...packaged Python interpreter, API and dependencies...
  dependencies\
    elmer\
      bin\
        ElmerGrid.exe
        ElmerSolver_mpi.exe
        ...matching runtime DLLs...
      lib\
        ...solver modules...
      ...other required distribution files...
```

Bundle a pinned, tested Windows Elmer distribution, not just its executables:
solver modules and compiler/numerical runtime libraries must remain compatible.
ElmerGrid and ElmerSolver are currently found through PATH; the packaged backend
will instead use absolute executable paths relative to its installation.
Set `ELMER_HOME` and any required PATH additions only in child-process
environments. Do not change the user's global environment.
Installed application files are read-only inputs; generated configurations,
meshes, logs and results remain in the writable per-job application-data storage.

Package the Python runtime and dependencies so no system Python is required.
Publish WPF with its required .NET runtime or have the installer handle that
prerequisite. Validate packaging compatibility with the selected Python/.NET
versions during implementation.

#### Setup wizard and MPI prerequisite

Retain the current four-process MPI workflow. The installer should detect a
compatible Microsoft MPI runtime and install the official prerequisite when
missing, subject to its distribution terms and user approval/elevation.
Do not assume copying MPI DLLs into the application directory replaces supported
runtime installation. Resolve the installed MPI launcher explicitly at runtime.

The wizard should show prerequisite checks, installation progress and actionable
failures, then run a small mesh/solver smoke test. Handle declined administrator
permission and failed prerequisite installation without claiming setup succeeded.
An offline installer should include the approved prerequisite payload; otherwise
clearly disclose any required download.

A serial Elmer variant could reduce prerequisites, but is not the initial
deployment choice. It would require separate performance and numerical testing,
and must not be used as a silent fallback when MPI is missing.

#### Redistribution and maintenance

Elmer can be redistributed subject to the applicable licences. Review the exact
selected distribution and its third-party components before release.
Include licence/copyright notices and provide corresponding source in a manner
that satisfies the applicable GPL obligations.
Invoking Elmer as a separate process is the intended integration boundary, not
a blanket legal assurance about licensing the complete product.
Verify Microsoft MPI installer redistribution terms separately.

Reference sources:
- [Elmer repository licence](https://github.com/ElmerCSC/elmerfem/blob/devel/LICENSE.md)
- [Microsoft MPI documentation and runtime downloads](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)

Record bundled versions and checksums. Upgrade dependencies deliberately and
rerun numerical regressions before shipping an update. Installer repair/uninstall
must not remove an independently installed shared MPI runtime or user job history
without explicit consent.

#### Installation TODOs

- [ ] Start with manually launched API during development.
- [ ] Add managed API startup/shutdown from WPF for the desktop release.
- [ ] Select and pin the Windows Elmer distribution and its required runtime files.
- [ ] Package the Python interpreter/API dependencies and WPF runtime.
- [ ] Invoke bundled Elmer executables by absolute path and set only child-process
  environment variables.
- [ ] Implement setup-wizard checks and installation of an approved Microsoft MPI
  runtime; handle elevation, declined consent and installation failure.
- [ ] Include approved offline prerequisites or document required downloads.
- [ ] Review redistribution obligations and include notices, source provision,
  versions and checksums.
- [ ] Handle executable paths with spaces and working directories independent
  of the source checkout.
- [ ] Run a small post-install mesh/solver smoke test and expose diagnostics.
- [ ] Test repair, upgrades and uninstall without deleting user jobs or shared MPI.
- [ ] Add retention/manual cleanup for completed jobs with confirmation.
- [ ] Add build/test/run instructions and verify on a clean Windows machine with
  no preinstalled Python, Elmer, developer PATH entries or MPI.

**Gate:** installation and launch work outside the developer workspace without
manual dependency or environment-variable configuration, and the bundled backend
successfully runs an analysis.

### Phase 6 - stage pipeline view and rerunning failed stages

**Goal:** the History tab shows each job as a pipeline, like an Azure DevOps
stage view, and lets the user rerun a failed or cancelled cracking stage without
repeating the Elmer and elastic-baseline work. Motivation: in job 3e9220fe,
FEniCSx cracking succeeded but the Aster crack submodel stalled, so the whole
job had to be cancelled.

**Agreed behaviour**
- Selecting a job in History shows the pipeline below the job table:
  Elmer baseline -> FEniCSx baseline and Aster baseline (in parallel) ->
  crack location -> FEniCSx cracking -> Aster cracking. Each box shows its state
  (succeeded, failed, cancelled, interrupted, blocked, running, queued, skipped
  or waiting), duration, attempt number and error. Running boxes update live.
- **Rerunnable stages:** FEniCSx cracking and Aster cracking.
- **A stage that succeeded is frozen** and can't be rerun. Any other finished
  state can be rerun: failed, cancelled, interrupted or blocked. A job's overall
  `succeeded` status does not freeze its stages, because a job counts as
  succeeded even when Aster cracking failed.
- **Downstream stages chain automatically:** rerunning FEniCSx cracking then runs
  Aster cracking, unless Aster cracking already succeeded (frozen).
- A rerun may **edit `fracture_debug`**. The Debug panel opens prefilled from the
  latest attempt, and null means the solvers' defaults. The edit applies to that
  attempt only.
- **Every attempt is kept.** Results views show the latest attempt; earlier
  attempts stay on disk with their logs, config and errors, and are listed per
  job.
- A rerun starts only when no other job is active (the existing one-active-job
  rule) and can be cancelled like a normal run. If the API restarts during a
  rerun, the stage becomes `interrupted` and can be rerun again.

**Config and hash rules**
- The job's `config.json` stays the submitted snapshot and keeps binding the
  Elmer, FEniCSx and Aster elastic baselines. Attempt N uses
  `attempts/N/config.json`.
- An attempt config may differ from the submitted snapshot **only in
  `fracture_debug`**, and the API enforces this. The baseline and refined-baseline
  checks (`_prepare_cracking_seed`, `_validate_fenicsx_output`, and the
  `verified_elastic_baseline` check in `FEniCSx/run.py`) compare against the
  submitted config's hash. The cracking-stage checks (prepared seed, Aster
  manifest) compare against the attempt config's hash.
- The prepared seed is rebuilt for each attempt from the job's crack location
  and the baseline fields.
- FEniCSx and Aster must always analyse the same shared seed
  (`shared_seed.sha256`). An edit that changes the seed (model, Gc, length scale,
  residual stiffness, water factor, load steps) moves an Aster-only rerun up to
  start at FEniCSx cracking, then Aster runs after it. This is the only case
  where a stage that already succeeded is run again; its earlier result is
  archived as an attempt, not lost. Edits that leave the seed unchanged (for
  example `aster.*`) can rerun Aster on its own.
- Each attempt records the Gairezi `git HEAD` and dirty flag, so a rerun that
  picks up a code fix (for example the Aster `RESI_REFE_RELA` tolerance) is
  traceable.

**Design by layer**
1. *API, job model* (`Lamb.API/app/jobs.py`):
   - per-stage `started_at` and `finished_at`;
   - `attempts`: number, stages, `started_at`, `finished_at`, per-stage
     status and error, `fracture_debug`, config SHA-256, prepared-seed and
     shared-seed SHA-256, Gairezi commit, and archive folder;
   - persisted in `job.json`; older jobs load with no attempts and unknown
     durations.
2. *API, pipeline*: `to_dict` adds `pipeline`, a list of
   `{id, label, column, status, error, started_at, finished_at, attempt,
   rerunnable, rerun_blocker}`. The server decides whether a stage can be
   rerun; the client only displays it.
3. *API, rerun*: `POST /jobs/{id}/stages/{stage}/rerun` with
   `{"fracture_debug": {...} | null}`.
   - The request is validated with `validate_fracture_debug` and the hash rules.
   - The current outputs of the stage and its downstream stages are moved to
     `attempts/<n>/`: `dam_phase_field/`, `aster_crack_submodel/`,
     `prepared_seed.json`, `cracking.log`, `aster_cracking.log`.
   - The job returns to `running` and the existing `_start_cracking` or
     `_start_aster_cracking` path runs, so the "exactly one manifest" checks
     still hold.
   - The response is 409 when the stage is frozen or another job is active, and
     422 for an invalid `fracture_debug`. A shared-seed change on an Aster-only
     rerun moves it up to start at FEniCSx cracking; the response reports this.
   - `GET /jobs/{id}/attempts` lists the attempts.
4. *Solvers* (as implemented): no new argument. FEniCSx compares the verified
   baseline's own `config.json` (its SHA-256 must match the baseline manifest
   and report) with the attempt config using `physical_configuration`, which
   drops only `fracture_debug`; any other difference is rejected. The API
   checks the baseline evidence against the submitted snapshot's hash and
   passes the attempt config (`attempts/<n>/config.json`) to the cracking
   commands.
5. *WPF*:
   - pipeline columns in `HistoryView` (stage cards with status-coloured edges,
     duration, attempt and a Rerun button; blockers in the tooltip);
   - `HistoryViewModel`: selected-job pipeline, polling while the job runs, a
     rerun command and an attempts list;
   - rerun dialog reusing the Debug panel fields, extracted into a shared
     `FractureDebugEditor` UserControl so Configuration and History stay
     identical;
   - `LambApiClient.RerunStageAsync` and `GetAttemptsAsync`;
   - after a rerun starts, the Run/Results view follows the job as it does now.

**Risks**
- Hash rebinding is the critical part: a mistake would let a cracking run use
  a baseline from a different physical config. Tests must show that any change
  outside `fracture_debug` is rejected.
- Archiving moves large result folders; use a same-volume rename, never copy and
  delete.
- Stage history of jobs created before this phase is incomplete; display
  "unknown" rather than inventing durations.

#### Pipeline and rerun TODOs

- [x] Job model: per-stage times, `attempts`, and persistence and loading of
  older jobs.
- [x] Attempt config snapshot; enforce "differs only in fracture_debug"; split
  the baseline hash binding from the attempt hash binding.
- [x] Baseline binding through `physical_configuration` in
  `FEniCSx/three_body_model.py` (replaces the planned `--baseline-config`);
  first attempts are unchanged.
- [x] Rerun endpoint: archive, restart, auto-chain, check the shared-seed hash
  on Aster-only reruns, plus the attempts endpoint.
- [x] `pipeline` in the job DTO, with server-side rerunnable flags and reasons.
- [x] WPF: History pipeline columns, `FractureDebugEditor` extraction,
  `StageRerunWindow`, client methods, attempts list and polling.
- [x] Tests:
  - API, using fake executables: Aster-only rerun after a failure; FEniCSx rerun
    auto-chains Aster; rejection of succeeded or frozen stages; 409 when another
    job is active; changes outside `fracture_debug` rejected; Aster-only rerun
    that changes the shared seed moved up to FEniCSx cracking; archived attempt contents; restart
    leaves the stage `interrupted`, which can be rerun;
  - FEniCSx baseline binding (`test_rerun_configuration.py`);
  - WPF: pipeline states, rerun enabled only when allowed, request payload,
    unchanged Debug panel behaviour.
- [x] Update the README job-lifecycle section and the app_plan endpoints table.
- [ ] Gate below: one real Aster-only rerun from History (not yet run).

**Gate:** on a job whose Aster cracking failed or was cancelled, Aster cracking
alone reruns from History with edited `aster.*` values. FEniCSx results are
untouched and the first attempt's logs and config stay available. A FEniCSx
cracking rerun with a new Gc rebuilds the seed and chains Aster.
### Phase 7 - multi-point crack path (on hold)

**Goal:** on the baseline crack-location plot, the user clicks several points that
are joined by straight segments to define the crack trace. Both FEniCSx and
Code_Aster honour that path. One point keeps today's horizontal crack, so existing
behaviour and numerics are unchanged.

**Agreed behaviour**
- A click **adds a point**; there is no freehand drawing or smoothing. Right-click
  or Backspace undoes the last point. **Clear** and **Submit crack** buttons replace
  today's immediate submit-on-click. Submit is enabled with at least one point.
- One point gives the existing `horizontal_annular_sector` seed. Two or more points
  give a path crack.
- Segments are straight in the physical unrolled face (chainage
  s = R acos(y/R), z). The plot is drawn in y, so segments are sampled and may
  look slightly curved.
- Through-thickness, the trace is **extruded radially by `depth_m`** from
  `cracking[0].face`. The radial band follows the wall face, using
  `wall_radii_at(z)`, so it follows thickness steps.
- `opening_m` is measured normal to the path, and `buffer_m` and the minimum
  clearance still apply. For path cracks the endpoints define the extent, so
  `full_width` and `length_m` are ignored and the report says so.
- Crack water pressure is **hydrostatic and varies with z** along the path,
  scaled by `fracture_debug.crack_water_pressure_factor`.
- Server-side validation is authoritative:
  - at most 20 points, each snapped like today's single point;
  - each segment, sampled densely, stays on wall concrete within the clearance;
  - each segment is at least `opening_m` long;
  - the path does not intersect itself;
  - `depth_m` is less than the wall thickness at every z along the path.
- The seed is an initial notch, not a prescribed crack path; phase-field damage
  can still propagate elsewhere. The UI wording must say so.

**Design by layer**
1. *Shared geometry* (`FEniCSx/automatic_crack.py`): a path geometry in (s, z) and
   `path_signed_distance`. This is the distance in the (s, z) plane to the
   polyline minus opening/2, intersected with the radial band and end caps.
   `locate_crack_path` reuses snapping, buffer, concrete-crossing and
   peak-stress reporting.
2. *Frozen seed contract* (`FEniCSx/crack_contract.py`): `schema_version` 2.
   - Geometry: `shape: "path_band"`, `path_m`, face, `depth_m`, seed thickness.
   - Loading: `pressure_model: "hydrostatic"`, water level, density and factor.
   - Version 1 stays valid and unchanged; the hash covers the whole payload.
3. *FEniCSx* (`crack_field.py`, `dam_phase_field.py`): `signed_distance`
   dispatches to the path distance, so refinement and the initial damage reuse it.
   Audit `_long_band_edges` and `resolution_report` for sector assumptions. Path
   cracks use the pressure term `-p(z) grad(alpha)` with a UFL expression of
   `SpatialCoordinate`; horizontal cracks keep the constant p.
4. *Code_Aster* (`whole_model_crack_mesh.py`, `whole_model_crack.py`,
   `native_seed.py`):
   - Generalise `slice_plane` from `z - Z` to a nodal level set phi, the signed
     offset from the path, and cut at phi = +/- t/2 by linear interpolation along
     edges.
   - Order cap polygons in the local tangent plane rather than in xy.
   - Keep the volume-conservation and orientation checks.
   - Band selection, `crack_faces` and the native seed damage use the path
     distance.
   - `PRES_REP` on `CRACK_FACES` becomes a FORMULE in Z.
5. *API* (`Lamb.API/app/jobs.py`, `main.py`): `POST /jobs/{id}/crack-location`
   accepts `{y_m, z_m}` (unchanged) or `{points: [{y_m, z_m}, ...]}`. Each point
   is checked against the selection map, then the whole path is validated.
   `job.json` stores `crack_path` (requested and snapped); `crack_location`
   remains for single points. `selected_seed.prepare` handles both.
6. *WPF* (`CrackLocationPlot.cs`, `RunResultsViewModel.cs`,
   `RunResultsView.xaml`, `LambApiClient.cs`):
   - pending points on the plot and view model;
   - add, undo, clear and submit commands;
   - the polyline and its vertices drawn on the plot;
   - the read-only results plot shows the submitted, snapped path;
   - the client sends `points`.

**Risks**
- In the cylindrical wall, a sloped crack is a curved (helicoidal) surface. The
  Aster cut is a linear level-set approximation within each tetrahedron and must
  be reported as such.
- A path that crosses a thickness step gets a radial band jump at the step. This
  is validated and reported, not smoothed.
- Horizontal (one-point) runs must reproduce today's seed and contract exactly.

#### Crack path TODOs

- [ ] Path geometry, signed distance, path validation and `locate_crack_path`.
- [ ] Seed contract schema v2 (`path_band` and hydrostatic loading), keeping v1.
- [ ] FEniCSx: path distance in refinement and initial damage, plus the
  hydrostatic `p(z)` pressure term.
- [ ] Aster: level-set wall cut, path band and crack faces, path native seed, and
  the hydrostatic PRES_REP FORMULE.
- [ ] API: multi-point request, validation, `crack_path` persistence and seed
  preparation.
- [ ] WPF: add, undo, clear and submit; polyline rendering; read-only results
  path; client payload.
- [ ] Tests:
  - path distance and validation (off-wall, self-intersection, step crossing,
    clearance);
  - contract v1/v2 round-trip and tamper rejection;
  - FEniCSx refinement and initial damage on a small mesh (WSL), and the
    hydrostatic pressure against the analytic value;
  - Aster level-set cut (volume conserved, no degenerate cells, faces on
    phi = +/- t/2) and the PRES_REP formula;
  - API single-point and multi-point acceptance and rejection;
  - WPF view-model and payload checks;
  - a regression test pinning the one-point seed.
- [ ] Update the Gairezi README cracking section.

**Gate:** a two-point sloped crack runs end to end through FEniCSx and Aster on
the default dam, and both report the same frozen path and hydrostatic loading. A
one-point selection reproduces the existing seed hash and results.

### Later milestones

- [ ] Code_Aster backend using the same configuration snapshot and result DTO.
- [ ] Scenario-specific crack inputs only after their data contract is agreed.
- [ ] Interactive 3-D mesh/results viewer.
- [ ] Side-by-side job comparison and parameter sweeps.
- [ ] Remote/multi-user service only after a separate security/deployment design.

## 9. Testing and immediate next step

Use existing mesh/pressure/solver regression tests as the numerical foundation.
Add API tests with temporary job directories and fake executables so most tests
do not need MPI or an expensive dam solve. Add WPF model/serialization/view-model
tests and a small manual UI checklist for the supplied mock-up.

The first implementation task is **Phase 1**, not the styled frontend:
make one isolated configuration-to-results worker run reliably.
Then expose that worker through the small API, and connect the WPF application.
This sequence avoids building a polished Submit button on top of shared files,
stale results or unsupported configuration controls.

Successful execution and convergence do not constitute engineering validation
or establish that the dam is safe.