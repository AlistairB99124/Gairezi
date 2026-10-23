# Dam Analysis Workbench - implementation plan

## 1. Goal and scope

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
full PVTU partitions, 5 m reporting VTU and matching finite per-body summary.
The WPF schema/model preserve rock domain/mesh settings through
load/save/submission, while bedrock remains read-only in the editor. Default
ParaView opening uses the reporting VTU, with a separate full-domain action.
Headline stress extrema/utilization cover concrete only; the table also shows
near-field bedrock separately. The desktop rejects older API versions lacking
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
stage. Hybrid and explicit experimental Amor are supported; the historical
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
| `POST /jobs/{id}/cancel` | Request cancellation of that job's process tree |
| `GET /jobs/{id}/crack-selection` | Baseline Z-Y stress map for user crack-location selection (409 until the verified baseline map exists) |
| `POST /jobs/{id}/crack-location` | Submit the selected `y_m`, `z_m` and start cracking (only while awaiting) |
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