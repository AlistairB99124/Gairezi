# Configurable FEniCSx dam study

## Current milestone

This directory is the WSL-based adapter. **The canonical model is now the
spring-jointed wall/plinth/bedrock model**, including the approved plinth
cutbacks and wall-footprint-only infill. The bonded and earlier split-interface
results below are historical and must not be combined with this reference.
The verified scope includes environment/configuration checking, a migration
inventory, an exact elastic patch test, conforming dam import and a full-dam
elastic comparison against the current Elmer run, and an analytically checked
two-bar frictionless contact benchmark. It now also includes a staggered hybrid
AT2 phase-field fracture solver with irreversible bound-constrained damage,
local crack-band refinement and ParaView output. **3-D crack-face contact is
not implemented.** Acceptance limits are provisional porting criteria, not
independent engineering validation.

### Canonical three-body model (19 October 2026)

The app runs the original Elmer solve, a FEniCSx elastic comparison, local
crack-band refinement, a zero-damage solve on the refined mesh, then fracture.
Failed verification stops that sequence and preserves the completed Elmer
results. Successful execution is not engineering validation.

- Import the saved `three_body.msh`, `config.json` and `diagnostic.json` from
  the isolated Elmer job, rather than regenerating geometry in WSL. Record
  their SHA-256 hashes and reverse the saved boundary-ID remapping.
- Preserve separate coincident interface nodes and body IDs: 1 plinth,
  2 wall, 3 bedrock. Remove only unused volume nodes.
- Use the configured concrete/rock density, Young's modulus and Poisson ratio.
  Gravity includes rock self-weight. Apply the outer rock support recorded in
  the Elmer diagnostic (`rock_outer_support`): for `rollers`, the same
  normal-only penalty springs `k n n` as Elmer on the bottom and four vertical
  rock faces (tags 10, 13-16), with no Dirichlet condition; otherwise fix the
  rock bottom in all directions. The exposed rock top stays free. The tailwater datum remains
  the concrete base, not the deep rock bottom.
- Use the configured global x/y/z springs for both interfaces, with no
  ground-spring bypass. The approved default uses
  `geometry.bedrock.support.model: distributed_elastic_granite`, so the
  plinth/rock seat is a mortar spring joint with the configured
  `geometry.bedrock.support.spring_n_per_m3` stiffness, like the wall/plinth
  joint. The alternative `model: bonded` shares plinth/rock
  seat nodes in the source mesh and therefore displacement DOFs in both solvers.
  A bonded seat can move with deformable rock but cannot slip or separate. No zero
  absolute displacement is imposed there, and no penalty stiffness approximates
  the bond. Bedrock spring coefficients are inactive for a bonded seat;
  wall/plinth springs and the outer rock support are unchanged. Elmer checks
  solved seat continuity, and the importer checks connection metadata and shared
  body-node identities.
  Elmer's Galerkin mortar formulation uses a lumped
  multiplier compliance: eliminating that multiplier gives
  `K = B.T diag(k / w) B`, where `B` integrates the slave-weighted relative
  displacement and `w` is the slave nodal area. This is not the same discrete
  operator as directly integrating `k * jump(u) * jump(v)`.
- Intersect opposite surface triangles after refinement. The mortar operator
  remains symmetric, conservative and invariant under rigid translation,
  including nonmatching interfaces and partitions on different MPI ranks.
- Damage and its driving history are restricted to the wall. Plinth/rock
  remain exactly elastic, and interface springs are not degraded. Hybrid and
  explicitly selected experimental Amor mechanics are supported; three-body
  experimental spectral mechanics is rejected.

Fracture requires a completed `Baseline_NoPhase` with matching submitted
configuration and mesh/config/diagnostic hashes, relative displacement L2
error at most `1e-4`, recovered concrete nodal tensor relative error at most
`1e-4`, nodal peak metadata, and relative force imbalance at most `1e-6`. Crack-band
resolution must meet at least **2.0 elements per length scale**. Canonical jobs
reject explicit lower targets as well as under-resolved automatic runs. The refined
zero-damage solve and final fracture equilibrium must also pass. Missing,
failed, nonfinite or stale evidence is an error, not permission to continue.
The older bonded-model fracture hold still applies to that legacy path.

Full-size elastic verification: **164,457 used nodes / 821,178 tetrahedra**,
latest GMRES Elmer/FEniCSx displacement L2 difference **6.578e-8**, recovered
concrete nodal tensor difference **3.511e-7**, relative force imbalance
**6.855e-11**, and true residual **9.636e-10** (135 iterations).
The initial direct-jump discretisation differed by
`8.081e-4`; it was corrected to match the mortar operator, not accepted by
loosening the tolerance. Analytical gravity, directional load transfer,
refined nonmatching interfaces, wall-only damage and serial/two-rank hybrid
and Amor benchmarks are covered by `test_three_body.py`. Full-size fracture
convergence, fracture calibration and dam safety remain unverified.

Canonical elastic solves now use restarted GMRES (restart 100), right GAMG and
an unpreconditioned residual check; legacy elastic solves retain CG/left GAMG.
The failed 9,429,660-cell job had stopped in refined uncracked elasticity with
CG's indefinite-preconditioner error, before damage evolution. A bounded
verification with GMRES passed mechanics but exhausted a 12 GiB process limit
while assembling reactions. The mechanics matrix/AMG hierarchy is now released
before postprocessing and element averaging is chunked. A follow-up attempt
produced partial fields but exited without a completed report; full-size refined
postprocessing and cracking remain unverified. No partial run is accepted as success.
`linear_solver_report.json` now persists successful mechanics diagnostics before
postprocessing; stage messages distinguish reactions, raw-field export and nodal
stress recovery. This diagnostic alone is not a completed baseline report.

For a direct run, in **WSL Ubuntu**, with `MODEL` pointing to the existing
isolated Elmer model directory and `OUTPUT` to a new output root:

```bash
python3 FEniCSx/run.py run-baseline --dam-config "$MODEL/config.json" \
  --elmer-model "$MODEL" --elmer-result "$MODEL/results/three_body_results_t0001.pvtu" \
  --output "$OUTPUT"
python3 FEniCSx/run.py run-fracture --dam-config "$MODEL/config.json" \
  --elmer-model "$MODEL" --elmer-result "$MODEL/results/three_body_results_t0001.pvtu" \
  --verified-baseline "$OUTPUT/dam_elastic_baseline/<completed-run>" --output "$OUTPUT"
```

Raw/degraded stress outputs include a cell `body_id` for ParaView Threshold filtering.
Headline recovered and raw stress extrema cover wall/plinth only. Per-body
`body_stress.bedrock` describes the **full** rock domain, not the separate
Elmer 5 m reporting clip; do not compare it to the clipped Elmer rock peak.
The full rock domain is retained in mechanics. No infinite-earth boundary,
geostatic initialisation, uplift, unilateral contact or friction is added.

### Bonded plinth/rock seat verification (19 October 2026)

A fresh four-rank Elmer solve and serial FEniCSx elastic comparison on the
approved bonded-seat configuration both completed. The wall still contains
501,024 tetrahedra; the complete domain has 821,178. Shared seat nodes reduce
the source node count to 161,036 without changing the wall geometry.
At 3,641 seat locations the maximum measured relative displacement is
`1.212e-19 m`, while the plinth itself moves: this is a bond, not a grounded seat.

Elmer/FEniCSx relative displacement L2 difference is `4.249e-8`, concrete nodal
stress difference `3.278e-7`, and relative force imbalance `8.037e-11`.
Wall peak nodal tension is `2.118841 MPa` at `(76.228153, 24.274858, -28.0) m`,
at the wall base; plinth peak tension is `2.698981 MPa` at
`(75.599479, 24.596115, -27.5) m`. The earlier stress table describes the
two-spring configuration, not this changed model. The mesher still warns about
minimum rock scaled Jacobian `0.00678428`; engineering use requires mesh-quality
review. No full-size fracture/refinement was run for this check, and the
automatic crack-length/placement issue is not resolved by this bond change.

### Historical bonded intact reference and fracture hold (17 October 2026)

The conforming source mesh now shares wall/plinth nodes. The interface is
internal, not two free exterior surfaces. The audit requires exactly one
face-connected body and shared region-1/region-2 faces, with no duplicate
coordinates or exterior tags 8/9. The deterministic tetrahedral subdivision
retains that conformity.

Only foundation tag 1 has directional ground springs. The wall-ground bypass
has been removed from the Elmer SIF and the new FEniCSx baseline.
`geometry.plinth.support` remains in the configuration for compatibility but
is explicitly inactive in this reference; `geometry.bedrock.support` remains
active. No material values or loads have been retuned. Fully bonded means
continuous displacement, including tensile/shear transfer: **no joint opening,
sliding, friction or uplift at an opening interface is represented**. Construction
details and independent engineering review are still needed.

At that milestone, app jobs ran Elmer and `Baseline_NoPhase`, then persisted
`cracking_status = blocked` with a review reason. The desktop displays the
baseline-only result and hold, not a fictitious successful crack run.
`run-fracture`, `--baseline-phase` and the MPI entry point reject newly generated
bonded fracture models. Historical jobs are unchanged. Low-level benchmark
solvers remain available for research; their availability does not release
the production hold.

Verification on the unchanged 4 m wall / 6 m plinth configuration:

| Check | Result |
|---|---|
| Source connected components / shared interface quads | 1 / 3,520 |
| Tetrahedral nodes / cells | 117,128 / 606,984 |
| Shared interface triangles after subdivision | 7,040; none exterior |
| FEniCSx true relative equilibrium residual | 6.4983e-10 |
| FEniCSx relative global force imbalance | 7.1415e-12 |
| Elmer/FEniCSx displacement relative L2 difference | 6.3932e-11 |
| Foundation reaction relative difference | 3.0766e-11 |
| Plinth mean raw stress `(xx, yy, zz, xy, yz, xz)` MPa | `(-0.16016, -0.27719, -0.26642, 0.01458, -0.09359, -0.32101)` |
| Maximum displacement | 7.9398 mm |
| Recovered principal tension / compression peaks | +3.6040 / -4.6723 MPa |

The foundation alone balances the applied resultant: approximately
`(+393.8110, +115.5963, +328.6838)` MN. Elmer reactions are reconstructed from
nodal displacement and spring laws, not native solver reaction output.
The raw plinth mean tensor differs from Elmer's recovered nodal mean by 0.4401%;
the measures are not identical stress-recovery operations. A separate **3-D
vector elasticity wall-only-load benchmark** verifies that a 10 kN wall crown
load reaches the plinth foundation, with no plinth body load or wall support.
This distinguishes actual interbody transfer from plinth self-weight response.

The client-confirmed **allowable tensile peak is 1.2 MPa (1,200,000 Pa)**,
superseding the early 2 MPa suggestion as an acceptance threshold.
**Any maximum principal tensile stress strictly above 1.2 MPa is a FAILURE
of the tensile acceptance criterion**, independently of solver convergence.
Equality does not exceed this limit but does not establish overall acceptance.
Assess recovered and raw peaks separately when both are reported; do not
select a lower measure to conceal an exceedance.
The +3.6040 MPa recovered peak is 3.003 times the
allowable and **FAILS that tensile criterion**. The configured 2 MPa reporting input
remains unchanged; it is neither the allowable nor a constitutive stress cap.
Possible singularity, mesh sensitivity and joint assumptions require investigation,
not dismissal of the exceedance. This is not a calibrated physical-failure
prediction or a dam-safety verdict. Solver parity establishes matching
implementations of the chosen assumptions, not that those assumptions are correct.
The source configuration hash for this verification is
`afd99de14023eedebdf8d2481695f1d588701b6738a24a32b6a8f767a0c831e4`.

The phase-field path supports evolution from a configured diffuse crack. The
older `predefined_crack` template remains an explicit-surface/contact concept
and is still blocked; these are different models and are not silently mixed.

The shared source geometry reads [root config.json](../config.json): a 4 m wall
(radii 76-80 m) centred on a 6 m plinth (radii 75-81 m), with 1 m overhang on
each side and centreline radius 78 m. Regenerate the Elmer reference and reimport the dam;
previous dam meshes and parity results are historical, not validation of
these updated dimensions.

### Historical automatic-crack sensitivity investigation

Before the bonded-reference hold, cracking jobs solved an additional **zero-damage elastic baseline after
refinement and before seeding/evolving damage**, using the exact same mesh,
material, external water/gravity loads and ground springs as the crack solve.
Internal crack pressure is absent from this uncracked reference. This does not
change either the original Elmer-parity baseline or the fracture law.

The completed report embeds `refined_mesh_baseline` and saves its independent
report and ParaView fields under `refined_baseline/`. The baseline report is
retained if subsequent cracking fails; no completed crack report is fabricated.
Both serial and MPI paths perform this solve, checking the true linear
equilibrium residual. The job stage identifies `fenicsx_refined_baseline`
before returning to `cracking`.

The desktop comparison retains the original mesh baseline, adds **Refined
Uncracked**, and defines **Crack difference = cracked - refined uncracked**.
Historical jobs without the refined baseline show an unavailable difference,
never a silent fallback to the coarse mesh. Raw element peaks, mesh counts,
global-peak coordinates, and cracked stresses evaluated at the refined
baseline's two fixed peak locations provide additional context. Fixed-point
stresses use the same recovered nodal tensor field as the global peak report.
Global extrema can move; negative compression means larger compressive
magnitude. These measures do not replace field inspection, a mesh-convergence
study, calibration or engineering review.

The automatic crack is regenerated from the **Elmer peak-stress element
centroid**, not a fixed physical crack prescription. Changes to element type
or stress recovery can therefore change the fracture scenario even when the
authoritative configuration is identical. In the October 16/17 comparison,
its elevation changed from `-27.75` to `-27.875` m, halving clearance above the
local `-28` m plinth from `0.25` to `0.125` m. Its arc extent also changed from
approximately `16.9743` to `16.0352` m; depth remained `3` m.

Two fresh eight-rank full-dam diagnostic runs isolated this change without
continuing interrupted checkpoints or changing materials/loads:

| Case, using the current production hybrid solver | Outcome | Runtime |
| --- | --- | --- |
| Historical successful mesh and historical crack | All ten steps converged | 233.71 s |
| New tetrahedral mesh and historical crack | All ten steps converged | 198.13 s |

The historical replay reproduced peak displacement `0.009623352645` m
(relative difference `6.51e-8` from the historical output) and maximum strain
norm `0.002192971`. The newer mesh with the same historical seed gave
`0.009623147674` m, maximum strain norm `0.002192547`, final staggered
residual `3.2265e-6` (limit `1e-5`), and mechanics relative true residual
`6.1226e-9` (limit `1e-8`). Dam configurations were checked equal before
the controlled comparison. These tests demonstrate that the current solver
still solves the formerly successful prescription, including on the newer
tetrahedral mesh. The current automatic seed instead produced the reported
gross hybrid strain breakdown; the comparisons do not establish an admissible
solution for that changed scenario.

Neither diagnostic substitutes the historical seed into new app jobs.
Automatic generation and physical configuration remain unchanged. Do not
move the crack, weaken guards or change fracture properties solely to force
success. Selecting a different physical prescription requires an explicit
modelling decision; the unresolved near-interface propagation is not a
dam-safety verdict.

`automatic_crack.json` and the run manifest now include `placement_provenance`
with peak chainage, interpolated plinth elevation at that location, and
clearance in metres and length scales. If the seed is within two length
scales above the interface at the peak location, an explicit warning is
also written to the report and job log. Two length scales is the existing
diffuse-band reporting extent, **not an engineering acceptance threshold**.
Plinth-side damage remains fixed at zero. The actual split wall/plinth mesh has
no shared interface DOFs; this does not constrain wall-side damage to zero.
This records proximity rather
than claiming that plinth damage or mesh refinement caused the failure.

## Environment

Run with Ubuntu on WSL2, not Windows Python. The current installation is DOLFINx
0.11.0.post0, PETSc 3.24.4, Open MPI 5.0.10, Python 3.14.4, NumPy 2.3.5, UFL
2026.1.0, meshio 5.3.5, Gmsh 4.14.0 and SciPy 1.16.3. The adapter checks for DOLFINx 0.11.
Nothing is installed or upgraded by these tasks. The first stage runs on one MPI
process; distributed result management is deliberately not claimed.

## VS Code tasks

Use **Terminal > Run Task**:

- **FEniCSx: Check WSL environment and scenarios**
- **FEniCSx: Inventory Elmer dam for migration**
- **FEniCSx: Import and audit dam mesh**
- **FEniCSx: Verify dam elastic parity**
- **FEniCSx: Verify frictionless contact benchmark**
- **FEniCSx: Run phase-field fracture**
- **FEniCSx: Update configured results**

The tasks invoke the Ubuntu `python3` directly through `wsl.exe`, with the current
workspace as working directory. Existing Elmer and Code_Aster tasks are preserved.

Equivalent commands, from PowerShell in the repository:

```powershell
wsl -d Ubuntu --cd "$PWD" --exec python3 -B FEniCSx/run.py check
wsl -d Ubuntu --cd "$PWD" --exec python3 -B FEniCSx/run.py inventory
wsl -d Ubuntu --cd "$PWD" --exec python3 -B FEniCSx/run.py import-dam
wsl -d Ubuntu --cd "$PWD" --exec python3 -B FEniCSx/run.py run --scenario dam_elastic_parity
wsl -d Ubuntu --cd "$PWD" --exec python3 -B FEniCSx/run.py run --scenario frictionless_contact_benchmark
wsl -d Ubuntu --cd "$PWD" --exec python3 -u -B FEniCSx/run.py run-fracture --dam-config config.json --elmer-result Elmer/structure_output/combined_solver/results/combined_results_mpi_t0001.pvtu --output FEniCSx/results
wsl -d Ubuntu --cd "$PWD" --exec python3 -B FEniCSx/run.py run
```

Or, from an Ubuntu shell in the repository:

```bash
python3 -B FEniCSx/run.py check
python3 -B FEniCSx/run.py inventory
python3 -B FEniCSx/run.py import-dam
python3 -u -B FEniCSx/run.py run-fracture --dam-config config.json --elmer-result Elmer/structure_output/combined_solver/results/combined_results_mpi_t0001.pvtu --output FEniCSx/results
python3 -B FEniCSx/run.py run
python3 -B FEniCSx/run.py run --scenario elastic_benchmark
python3 -B -m unittest discover -s FEniCSx -p 'test_*.py' -v
```

## Scenario configuration and outputs

### Stress comparison convention

Three-body principal-stress peaks in new elastic baseline and fracture reports use
[stress_recovery.py](stress_recovery.py): consistent L2 projection of the
Cauchy stress tensor into a continuous P1 tensor space, followed by eigenvalues
at the recovered **nodes**, matching the canonical Elmer headline summary.
The previous FEniCSx headline instead averaged the four nodal tensors per
tetrahedron before taking eigenvalues. Those unlike measures explained the
apparent baseline discrepancy (2.395 MPa Elmer versus 1.741 MPa FEniCSx tension).
The corrected full-size concrete peaks, in MPa, are:

| Body | Elmer tension | FEniCSx tension | Elmer compression | FEniCSx compression |
|---|---:|---:|---:|---:|
| Wall | 1.791210 | 1.791210 | -3.563739 | -3.563738 |
| Plinth | 2.395260 | 2.395259 | -3.533062 | -3.533062 |

The original element-mean extrema remain under `element_mean_stress`;
legacy two-body reports retain that headline convention.
The Elmer stress solver uses a mass-matrix nodal projection; see upstream
[StressSolve.F90](https://github.com/ElmerCSC/elmerfem/blob/devel/fem/src/modules/StressSolve.F90).
The projection integrates the actual degraded stress expression for phase-field
runs, rather than first sampling it into a DG0 field.

The same operator applies to NoPhase, initially intact Phase, and seeded-crack
runs. No material, loading, support, mesh, or damage parameters are calibrated to
force stress agreement. Separate coincident interface nodes remain separate.
Source-order Elmer tensor parity requires one MPI process; refined fracture
and stress recovery also have serial/two-rank benchmark coverage.

`recovered_stress.pvd` contains recovered nodal stress in Pa.
Existing `stress.pvd`/`degraded_stress.pvd` retain the raw DG0 stress.
Reports retain its principal extrema under `raw_element_stress`, and record the
recovery method, solver tolerance, peak measure, body IDs and reference-coordinate
locations under `stress_recovery`. For compatibility the location keys retain
`*_peak_centroid_xyz_m`, but canonical locations are **nodes**, explicitly labelled
by `peak_coordinate_location`. Recovered peaks can smooth local crack-tip concentrations:
use the recovered values for like-for-like Elmer comparison, but inspect the raw
fields and mesh sensitivity when interpreting local cracking.

Historical reports/CSV exports are not rewritten. The desktop warns when an old
FEniCSx element-mean report is compared with canonical Elmer nodal peaks.
Rerun the baseline to satisfy the new full-field stress parity gate; historical
element-mean evidence cannot enable canonical fracture. Numerical parity is
not engineering validation; real crack-induced stress differences must not
be forced below the intact-baseline tolerance.

### Isolated uncracked baselines

The normal API flow is Elmer -> uncracked FEniCSx baseline -> FEniCSx fracture.
The baseline is mandatory, not an environment-selected debug mode.
Its `Baseline_NoPhase` report supplies the Comparison table's
**Baseline Not Phased (FEniCSx)** column. Difference is cracked FEniCSx minus
uncracked FEniCSx; Elmer remains a separate corroboration column.
Both FEniCSx stages use the same job's immutable submitted configuration.
The elastic baseline uses the Elmer tetrahedral solver mesh; fracture retains
its existing crack-band refinement, recorded in the refinement report.
Failure to produce a completed, matching, finite baseline report blocks fracture.
Both stages append to the job's `cracking.log`, with distinct stage markers.
The API records baseline status/errors independently and preserves completed
baselines when fracture fails. Historical single-mode jobs are not rewritten.

Standalone baseline commands remain available directly from Ubuntu:

```bash
python3 FEniCSx/run.py run-baseline --dam-config config.json --output FEniCSx/results
python3 FEniCSx/run.py run-fracture --baseline-phase --dam-config config.json --output FEniCSx/results
```

`run-baseline` produces `Baseline_NoPhase` in
`dam_elastic_baseline/<timestamp>/baseline_report.json`.
The standalone diagnostic `run-fracture --baseline-phase` produces `Baseline_Phase` in
`dam_phase_field/<timestamp>/phase_field_report.json`.
Both write a configuration snapshot, manifest, import report, ParaView fields
and `results.md`; failures retain a failed manifest and traceback.
The API can still read both historical report types. The API no longer selects
either mode through `Baseline_Phase`/`Baseline_NoPhase` environment variables.

Neither mode generates or seeds a crack or consumes an Elmer result.
The phase baseline starts with damage zero everywhere and permits AT2 damage
evolution in wall region 2 (plinth and wall, regions 1-2, for the three-body
model); other damage remains fixed at zero. Where an
import has shared interface nodes those DOFs are also fixed, but the current
split wall/plinth model has no such shared interface nodes.
It uses the existing [fracture defaults](fracture_defaults.json), recorded in
its report. It retains the shared Elmer solver mesh without local crack-band
refinement and emits a warning and resolution diagnostic when that mesh does
not resolve the configured length scale. Convergence is not mesh independence
or engineering validation.

### Job-specific Elmer baseline handover

The API passes the freshly validated baseline to `run-fracture` using
`--elmer-result <job-result.pvtu>`. The API coordinates the processes; the
consumer adapter in [elmer_baseline.py](elmer_baseline.py) owns parsing.
The submitted configuration is rebuilt using Elmer's shared
`tetrahedralize_structure` conversion before matching result cells. The
fracture mesh and CLI elastic-parity reference check use that same solver
topology. Old mixed-cell reference files must be regenerated for the current
tetrahedral Elmer workflow; the mapper still supports explicitly supplied
matching mixed-cell structures, but never bypasses a topology mismatch.
Standalone runs with explicit legacy scenarios may omit the argument.
Scenario-free runs require the baseline to generate the crack.

The adapter uses the existing elastic-parity mapper to recover undeformed
coordinates from Elmer's deformed output, match cells to the submitted
configuration's rebuilt mesh, and preserve separate wall/plinth interface nodes.
It rejects missing pieces/fields, paths escaping the result directory,
nonfinite fields and mismatched geometry. The run records:

- `elmer_baseline_fields.npz`: displacement and six stress components mapped
  to source nodes, undeformed coordinates, and node region IDs.
- `elmer_baseline.json`: input hashes, configuration hash, units and the
  location of the largest principal stress: for three-body models the largest
  nodal value over wall and plinth nodes; otherwise the element centroid of the
  largest element-mean nodal tensor over all source volume elements.
- `manifest.json` includes the baseline report; a handover error fails the
  fracture run explicitly, while the API retains the successful Elmer result.

**This imports a baseline reference, not a mechanical initial condition.**
The phase-field solver still solves its own configured loading. An explicit stress/displacement/history transfer
law and verification are required before using these arrays as initial state.
The recorded peak can be compressive if the entire model is in compression;
`has_tension` distinguishes this. These outputs are not engineering validation.

### Automatic horizontal crack

After each successful Elmer baseline, the API starts FEniCSx even without
`crack_simulations`. For the three-body model the consumer uses the largest
**nodal** maximum principal stress over **wall and plinth** nodes (bedrock nodes
are excluded) and that node's undeformed position. A nodal value is used because
element means smear the wall/plinth corner peak: for the reference job the
concrete nodal peak is 1.99 MPa in the plinth at r 79.5 m, chainage 98.0 m,
Z -27.5 m, while the wall element-mean peak lies at Z -17.1 m. The corner peak
is mesh sensitive. Legacy two-body runs keep the element-mean centroid peak.

The initial crack is a **local** horizontal annular-sector prism at the peak Z,
using the authoritative wall centreline radius to convert chainage to angle.
It is seeded from the concrete face nearer the peak (upstream if the peak radius
is at or beyond the centreline radius, otherwise downstream). Radially it runs
from that face of whichever concrete body is wider (wall or plinth) to **75% of
wall thickness** into the wall, e.g. r 77-81 m for an upstream seed. It has a
**0.002 m vertical opening** and an arc of **4 m**
(`automatic_crack.arc_length_m` in [fracture_defaults.json](fracture_defaults.json)).
The arc is positioned to contain both the peak and the nearest wall/plinth
junction chainage at that elevation (the wall-end intersection of the plinth top
contours), so the crack passes through **both wall and plinth**. It is centred on
the peak where possible, kept at least `min(2 l, arc/4)` from the junction on each
side and kept inside the concrete extent given by the plinth underside contours.
If one 4 m arc cannot contain both peak and junction, or the wall does not reach
the peak elevation, generation fails rather than moving the crack. For the
reference job the arc is chainage 95.99-99.99 m about the junction at 98.44 m.
This is not a straight rectangular chord.

The named result is `automatic_peak_tension`; existing result endpoints and
ParaView controls discover it from the run manifest. `automatic_crack.json`
records the source peak, seed face, concrete and wall chainage extents, junction
chainage, crack chainage range, exact sector, generated specification,
numerical settings and limitations. The submitted `config.json` is not rewritten.

In the three-body model fracture is permitted in **plinth (region 1) and wall
(region 2)**; bedrock (region 3) damage DOFs, including any shared bonded-seat
interface DOFs, are fixed at zero. The damage mask degrades plinth and wall
stiffness and lets their tensile energy drive damage; bedrock stays elastic and
undegraded. This is a physical modelling change from the earlier wall-only
assumption and uses the same concrete Gc and length scale for the plinth. The
wall and plinth are separate meshes with duplicate coincident nodes joined by
interface springs: damage evolves in each body, but the phase-field gradient does
not couple across the joint and the joint springs do not soften, so a crack
crossing the junction is represented as two separately damaged bands.
For legacy two-body runs damage remains restricted to wall region 2.

Damage-region cells (plinth and wall for three-body) drive crack-resolution
checks and refinement marking; only under-resolved damage-region cells near the
crack are marked. Neighbouring cells may still be subdivided for conformity.
Automatic three-body placement must put diffuse-band cells in both plinth and
wall. Refinement reports and logs record wall and plinth cell counts and marked
cells separately. For the reference job refinement resolved in 4 levels to
2.54 M cells (1.26 M wall, 1.08 M plinth) at 2.31 elements per length scale.
Automatic runs allow up to **five** refinement levels, stopping early when the
required resolution target is met and failing if that limit is exhausted.
Refinement also fails cleanly, with a `maximum_refined_cells` message, when a
level is predicted (lower bound: each marked tetrahedron becomes 8) or found to
exceed **4,000,000** global cells. Previously such runs were killed by the
operating system for lack of memory (exit code 9).

[fracture_defaults.json](fracture_defaults.json) holds the automatic workflow's
AT2 settings: length scale **0.25 m**, Gc 2700 J/m2, zero internal crack pressure,
ten load steps and the solver/refinement controls. These are explicit research
defaults, not a calibration or engineering validation. The target remains
**2.0 elements per length scale** (`h_max <= 0.125 m` at `l=0.25 m`). The length
scale was raised from 0.1 m because the 0.5 m source cells have diameters up to
0.866 m: `l=0.1 m` needs five halvings and about 19 million cells even for the
local 4 m crack, while `l=0.25 m` resolves in three levels (about 2.0 million
cells for the October 2026 peak). In AT2 the length scale also sets the effective
strength; the 1-D estimate `(3/16) sqrt(3 E Gc / l)` gives about 6.3 MPa at
0.25 m versus 10 MPa at 0.1 m (E = 35 GPa), against a configured 2 MPa tensile
strength. Treat it as a regularisation choice that needs a sensitivity study.
The 2 mm value defines the initial damaged prism thickness; the existing
phase-field model does not enforce a physical 2 mm displacement discontinuity
or unilateral crack-face contact.

No tensile peak, an out-of-range Z, a horizontal coincident plinth segment or
anything other than two unambiguous intersections fails the cracking stage
explicitly; it does not clamp Z or substitute a default crack. The successful
Elmer baseline remains available. API-launched Python uses unbuffered output.

Example (Ubuntu shell):

```bash
python3 -B FEniCSx/run.py run-fracture --dam-config /path/job/config.json --elmer-result /path/job/output/combined_solver/results/combined_results_mpi_t0001.pvtu --output /path/job/cracking
```

[cracking.json](cracking.json) is an array. Each entry has a unique safe `id`,
boolean `enabled` and explicit `kind`. Unknown fields, invalid units/ranges,
duplicate names and non-finite numbers are rejected, including in disabled
templates. Geometry bounds and mesh alignment must be validated by the future
dam adapter; passing configuration validation does not prove geometric feasibility.

The enabled `elastic_benchmark` scenario is a one-metre cube with affine
uniaxial strain. It uses concrete E and Poisson ratio from the root
[dam configuration](../config.json), **not the dam geometry or dam loads**.
It verifies relative L2 displacement error and strain-energy error against an exact
solution using the configured threshold. The default limit is 1e-8.
Its linear solver uses CG/Jacobi and the configured tolerance/iteration limit.
Displacements are in metres and stresses in Pa.

Export regression tests use the installed VTK 9.5.2 reader. DOLFINx writes VTU
version 2.2, which the installed meshio 5.3.5 reader does not accept; use ParaView
or VTK for these outputs rather than feeding them directly to meshio.

Each run writes a new timestamped directory under `results/<id>/`, containing:

- `manifest.json`: configuration, environment, input hashes, convergence status;
- `summary.json`: errors, strain energy, solver iterations and residual;
- `displacement.pvd` and `stress.pvd`, with associated VTU files for ParaView.

`latest.json` points to the most recent **successful benchmark or provisional parity run**. Failed runs
record their error and traceback, return a failing process exit status, and do
not replace that pointer. Inspect manifests rather than assuming older results
represent the current configuration. Results are ignored by Git.

The disabled crack entry is a **configuration template, not a validated dam
scenario**. Horizontal cracks use `chainage_m`, `z_m`, `width_m`, `depth_m` and
`initial_opening_mm`. Vertical cracks replace `z_m`/`width_m` with `z_lo_m` and
`z_hi_m`. Contact may be `frictionless`, or `coulomb` with an explicit
`friction_coefficient`; `water_pressure_on_faces` is an explicit modelling choice.
All enabled crack scenarios currently fail before any solve/output is started.
No penalty stiffness, friction law implementation, opening field or crack result
is fabricated.

## Phase-field fracture workflow

`run-fracture` reads the root dam contract through an explicit `--dam-config`
path and writes only beneath the explicit `--output` path. Every job receives a
configuration snapshot, input hash, environment record, import and placement
reports, and either a completed or failed manifest. A successful job also
writes `results.md`, a civil-engineering-oriented summary generated directly
from the configuration snapshot and numerical JSON reports. Repository
`config.json` is never overwritten.

The automatic job requires `2.0` elements per length scale and allows five
local refinement levels within a 4,000,000-cell budget. If unmet, the job fails
before the refined baseline or cracking; it does not lower the target or change
the length scale at run time. The automatic default length scale is 0.25 m (see
above); explicit `crack_simulations` keep their own declared values.
Historical preliminary runs used `0.9`, achieving `h_max` about `0.109 m`
for `l=0.1 m`; these are under-resolved. A historical target of
`2.0` with `l=0.1 m` generated roughly 12.5 million crack-band tetrahedra and a
1.8 GB mesh, then exhausted serial WSL memory. Meeting the minimum is necessary
but does not replace a mesh and length-scale convergence study.

The solver uses `alpha=0` for intact and `alpha=1` for fully damaged material.
Mechanics uses `g(alpha)=(1-alpha)^2+kappa`. A
volumetric/deviatoric tensile history drives AT2 damage, while mechanics uses
the hybrid full-stiffness degradation. Volumetric compression is excluded from
the driving energy, but deviatoric energy remains active in compressive/shear
states. This does not provide unilateral contact. Each load step alternates:

1. degraded linear mechanics with gravity, water and spring supports;
2. history update by pointwise maximum tensile energy;
3. PETSc TAO trust-region damage minimisation with
   `alpha_previous <= alpha <= 1`;
4. a convergence check on displacement and damage changes.

Fracture mechanics uses restarted GMRES (restart 100), right-preconditioned
GAMG and the six rigid-body near-nullspace modes also used by the elastic
baseline. Unlike CG, GMRES does not require a positive-definite AMG
preconditioner. A refined, seeded-crack dam previously triggered PETSc's
`Diverged due to indefinite preconditioner` with CG and no elasticity
near-nullspace. This solver change does not alter the mesh, spring supports,
loads, fracture properties or configured residual tolerance.
The configured mechanics iteration limit remains enforced. Krylov exhaustion,
residual-check failure and the outer runtime deadline remain failures, not
successful comparison results.
Each mechanics solution is independently checked using the unpreconditioned
residual `norm(Au-b)/norm(b)` against `mechanics_relative_tolerance`.

`mechanics_diagnostics.jsonl` and the console log record the load step,
staggered iteration, start/completion/failure, Krylov iterations and true
relative residual. The final report retains these residuals and identifies
the solver configuration. A failure includes its step and iteration and
never produces a successful report as a fallback.
Mechanics warm-starts from the previous displacement. Diagnostics include
timestamps, mechanics construction/solve timings and damage setup/solve timings.
The mechanics solve timing includes matrix assembly and AMG setup as well as
Krylov iterations; it must not be interpreted as Krylov-only time.
The diagnostics also record actual TAO gradient norms, staggered residuals,
displacement maxima and counts of nearly fully damaged degrees of freedom.
Damage convergence uses absolute projected-gradient tolerance or reduction
relative to the initial projected gradient, with an explicit post-solve check.
Objective-relative gradient convergence is disabled: the quadratic objective
omits a history-dependent constant, so its magnitude is not a valid convergence
scale. In particular, a large negative objective must not relax the configured
gradient criterion.
Mechanics reuses its assembled-system storage and GAMG hierarchy between
staggered iterations instead of rebuilding a hierarchy on every solve. The
matrix and right-hand side are still reassembled for current damage and loading,
and every solution must pass the same explicit true-residual tolerance. A
Krylov iteration count exceeding twice the problem's first solve requests a
fresh preconditioner on the next iteration rather than indefinitely reusing a
hierarchy that no longer represents the current stiffness. When
damage changes nodal degradation by a factor greater than two, the next
mechanics solve likewise rebuilds GAMG, even if its preceding solve was cheap.
When load-dependent Dirichlet conditions are provided, a new problem is constructed
at each load step to apply those conditions correctly.
For serial runs, `iteration_checkpoint.npz` atomically retains displacement,
damage, history and previous-load-step damage bounds at the start of the
latest mechanics iteration, along with mesh coordinate/order identifiers.
MPI runs retain one `iteration_checkpoint_rank<N>.npz` per rank, including
rank/process-count metadata. A separate `checkpoint_layout[...].npz` records
input cell/vertex IDs, geometry connectivity and each function's cell-to-DOF
maps. MPI partitioning can change between launches even at the same process
count; local field arrays must be remapped using these identifiers, not copied
by rank or coordinate order. A coherent rank set must also have matching load
step and iteration metadata. A replay/remapping tool is not provided yet.
It is diagnostic state, not a completed result or an automatic restart file.
`load_step_progress.json` preserves converged steps without claiming final
completion. These files allow late-load failures to be investigated without
discarding every preceding iteration's evidence.

API-launched stages run under a ten-minute process deadline (TERM at 595 seconds
and up to five seconds termination grace). Timeouts preserve completed baseline
results but mark the interrupted FEniCSx stage failed. Standalone CLI commands
should likewise be run with an explicit timeout during investigation.

`run-fracture --fracture-processes N` keeps configuration reconstruction,
Elmer handover and mesh refinement in the serial coordinator, then delegates
the saved mesh to `N` MPI workers for the coupled solve and stress recovery.
The standalone CLI default remains one process; `N=0` selects up to eight
local CPUs. Displacement/damage convergence checks, crack placement, load
elevations and recovered stress extrema use collective reductions. Worker
errors are preserved in rank-specific JSON files and propagated to the
coordinator's failed manifest. The worker group inherits the outer execution
deadline; it does not receive a fresh time budget.

Nonconvergence, invalid placement, exhausted refinement, PETSc failure and
nonfinite energy all fail the job explicitly. The final report includes an AT2
homogeneous tensile-strength estimate because `E`, `Gc` and `l` jointly imply a
strength that can differ from `material.concrete.tensile_strength_pa`; this is
a calibration warning, not an automatic material-property rewrite.

After every converged mechanics solve, the solver checks the strain Frobenius
norm before updating tensile history or damage. Nonfinite strain or a norm
of at least `1.0` is rejected as gross breakdown of small-strain kinematics,
with the load step/iteration, offending cell centroid and principal strains in
the error. The diagnostic also records the cell's vertex damage values, so
compressive and tensile breakdown can be distinguished without guessing from
a scalar norm. Nonfinite strain on any MPI rank is a collective failure.
This deliberately loose guard is
not an engineering allowable strain and does not validate responses below it.
It prevents metre-scale/local order-unity-strain runaway from being processed
as a valid small-strain fracture comparison. Such a response requires review
of fracture, unilateral contact and kinematic assumptions; it does not prove
that the physical dam is safe or unsafe.

The reduced 3-D compression patch regression
`ConstitutiveTests.test_hybrid_compression_patch_loses_stiffness_without_creating_tensile_history`
isolates a limitation of the current law. A uniformly damaged unit cube under
uniform pressure has the exact affine contraction
`u = -p*x/(3*K*((1-alpha)^2+kappa))`. Tests on two tetrahedral resolutions
verify this displacement, the mechanics residual, zero tensile driving history
and no additional damage. Nevertheless, at `alpha=1`, compressive compliance
is amplified by `1/kappa` (100 million for `kappa=1e-8`). Excluding volumetric
compression from damage growth does not restore compressive load transfer
through an existing damaged zone. This benchmark demonstrates the constitutive
limitation; it is not a contact benchmark or proof that this alone causes the
dam's local runaway. Neither a contact law nor a different degradation model
is silently substituted for the configured hybrid model.

### Experimental compression-preserving mechanics

The Python `solve_staggered` and dam-adapter `solve` functions accept the
explicit keyword `mechanics_formulation="experimental_amor"`. This is an
investigation gate, not a new app, API or CLI default. The default remains
`"hybrid"`; there is no environment-variable override.
Standalone `run-fracture --fracture-mechanics experimental_amor` opts in
explicitly and records the choice in the run manifest and worker request.
The flag is rejected for baselines and unrelated actions. The app/API keeps
hybrid as its default; its explicitly labelled experimental checkbox passes
this flag for the cracking stage only, with up to eight MPI workers. The
selection is stored in parent job metadata and is not a dam material input.
`--accelerate-damage` additionally selects bounded scalar Aitken relaxation
for that experimental formulation only. The factor is computed collectively
from owned damage DOFs, bounded to `[0.5, 2]`, and reset at each load step.
Growing same-direction increments would give a negative Aitken factor;
these use the original minimisation update (factor `1`) rather than
clamping a negative extrapolation to artificial under-relaxation.
Intermediate iterates are projected onto the previous-load damage bounds and
`alpha <= 1`; fixed non-wall DOFs remain zero. Convergence uses the **unrelaxed**
fixed-point residual. No extrapolation is applied to an accepted iteration,
which must still pass the original mechanics, obstacle and staggered checks.
Obstacle diagnostics describe the minimised candidate before intermediate
relaxation, not a second optimisation of the extrapolated field.
Reduced tests compare stepwise elastic/fracture energies and final fields,
and verify that plinth damage remains zero. These are not full-dam validation:
the eight-rank saved-mesh accelerated replay reached the final load step but
was terminated at 590 seconds. Its last completed staggered iteration (13)
had unrelaxed damage change `0.048585`, versus the required `1e-5`;
the residual was increasing despite relaxation reaching its lower bound.
Mechanics equilibrium passed at that iteration, so this trial does not
support further linear-solver tuning alone as a cure. Do not resume that
checkpoint or extend the timeout to claim a successful bounded analysis.

`--fracture-final-substeps N` is an explicit load-increment investigation
option (default `1`, preserving the configured uniform ramp). It divides
only the final configured load interval into `N` equal increments; e.g. ten
configured steps with `N=4` retain `0.1,...,0.9`, then apply
`0.925,0.95,0.975,1.0`. Reports retain both the configured count and actual
load factors. All accepted increments use the same equilibrium, damage and
staggered tolerances, advancing the irreversible lower bound only after
convergence. Final VTK fields are written at full load even if the resulting
step count is not a multiple of the output interval. This is not an automatic
retry, checkpoint restart, change to material data, or validated app default.
The bounded eight-rank subdivision replay also failed the runtime requirement:
at 92.5% load its last completed iteration (17) had damage change `1.18465e-5`.
It had not reached full load when the 590-second deadline stopped it.

The experiment uses the Amor energy
`g(alpha)*(K*<tr(eps)>_+^2/2 + mu*dev(eps):dev(eps))`
`+ K*<tr(eps)>_-^2/2`.
Only negative volumetric energy is undegraded. Deviatoric stiffness still
degrades, so this is **not unilateral crack-face contact**, friction or
compression/crushing modelling. Material properties, crack seed and loading
are unchanged. History-driven irreversible AT2 damage and the gross
small-strain guard remain active.
For the experimental formulation, the tensile history at each staggered
iteration is `max(H_previous_accepted_load, psi_plus(u_current))`.
Intermediate trial peaks are not committed as irreversible load history;
the accepted history advances only when the load step converges. This
matters when relaxation or extrapolation causes a trial overshoot.
The hybrid default retains its existing running-maximum history convention.
Reports identify the convention. A fresh bounded full-dam test of the
accepted-step convention still failed at final load: the last completed
iteration (13) had damage change `0.048603` before the 590-second cutoff.
This correction is not evidence that history overshoot caused the observed
dam failure, nor a successful full-dam repair.
Resetting negative Aitken extrapolations to the original minimisation
update also failed the bounded full-dam test. At final-load iteration 11,
damage change remained `0.048571`; mechanics passed its independent
equilibrium test but needed six Newton corrections and 191 Krylov
iterations (41.76 seconds for that mechanics update). The run stopped
at 590 seconds. Inspection of its saved, unconverged state found principal
strains approximately `[-0.001077, -0.000010, 0.012340]` near
`(78.2161, 15.7248, -27.8906)` m, with vertex damage
`[1, 1, 1, 0.97925]`. Unlike the original hybrid runaway, the largest
principal strain here is tensile. This does not prove contact deficiency,
an unstable equilibrium, physical failure, or validity of a completed
fracture solution. No complete result was produced by these trials.

A fresh one-increment, full-load diagnostic (same mesh, seed, materials and
final loads, without Aitken) also hit the 590-second development deadline.
At iteration 19 damage change was `0.048649`; the preceding mechanics
update passed equilibrium but took 60.38 seconds. This diagnostic changes
the loading path and cannot replace or validate the configured ten-step run.

`ForceControlledPatchTests` in `test_phase_field_solver` verifies a separate
analytical load-control limitation. In homogeneous positive dilation
`eps=e*I`, the AT2 stationary damage is
`alpha=c*e^2/(1+c*e^2)`, where `c=9*K*ell/Gc`. With zero residual stiffness,
the applied tensile pressure is `p=3*K*e/(1+c*e^2)^2`, peaking at
`e=sqrt(Gc/(27*K*ell))`, `alpha=1/4`. The implemented small residual
stiffness does not make the high-strain branch a valid small-strain result.
A 3-D traction-driven patch below the peak matches affine displacement,
damage and traction analytically; one above the peak develops runaway
tensile strain and is rejected, rather than reported as successful.
These synthetic tests show why more runtime, a smaller linear residual,
or undegraded compression need not cure tensile load-controlled softening.
They do **not** establish that the dam has exceeded its physical capacity,
or prove that no admissible dam equilibrium exists. A validated continuation
or fracture/contact treatment requires separate verification; do not infer
one from solver iteration changes or silently substitute a different law.

Fixed-damage mechanics uses PETSc SNES Newton with backtracking and an explicit
semismooth tangent (the tensile-side bulk derivative at zero trace).
Automatic differentiation of both positive/negative parts at zero can
otherwise omit the bulk tangent. GMRES/GAMG solves the tangent systems.
The nonlinear residual is reassembled independently before acceptance;
its absolute and load-relative tolerances are retained in each iteration's
`nonlinear_mechanics` record. The reference is the assembled applied load
vector on free DOFs, not the sometimes tiny residual of a nearly converged
warm start. Zero-applied-load, displacement-driven tests instead use the
initial residual and absolute tolerance. Energy, raw stress and recovered stress all use
the selected law, and reports explicitly identify the experimental model.
The coupled solver retains its nonlinear system between staggered iterations;
matrix, residual and applied load are still reassembled. GAMG may be reused
for the first Newton correction of a nearby state. Further Newton corrections,
nodal degradation changing by more than a factor of two, or Krylov iteration
growth request a fresh hierarchy. Displacement boundary-condition changes
recreate the mechanics problem at each load step. Reuse never bypasses the
independent equilibrium check.
Each Newton linear correction uses an absolute residual tolerance of one
tenth of the current load-scaled equilibrium tolerance. This avoids solving
every correction to the much smaller residual of a nearly converged warm
start. The outer, independently reassembled equilibrium requirement is
unchanged; the selected inner tolerance is included in the diagnostic report.
Reuse is controlled at SNES level with persistent preconditioner lagging:
setting only the KSP preconditioner's reuse flag is insufficient because
SNES otherwise rebuilds it on the first Jacobian of each solve. A requested
refresh uses PETSc's `-2` lag (build next, then retain); nonlinear tolerances
are preserved when changing these options.

`test_compression_preserving` checks pressure-driven 3-D opening, closure
and reopening on two tetrahedral meshes, exact affine displacement and
energy, energy/stress consistency, unchanged damage during fixed-damage
mechanics and explicit failure propagation. Small direct-LU checks include
`alpha=1`; iterative checks cover `alpha=0` and `0.99`. Direct factorisation
is selected explicitly only by the benchmark, never used as a fallback.
Fully damaged material combines undegraded bulk stiffness with residual
shear stiffness, which makes the tangent nearly incompressible and can
defeat GAMG. Passing the small direct tests is not evidence that the
full-dam iterative solve or runtime requirement is satisfied.
Current full-dam verification is incomplete: bounded saved-mesh trials were
stopped by the runtime deadline before all ten load steps completed.
Partial progress, including a converged fixed-damage full-load probe, must
not be presented as a successful complete fracture analysis. Do not promote
this experiment to the app/API default on the strength of these tests.
The optional Python `solve_parallel` coordinator also accepts
`mechanics_formulation="experimental_amor"` and persists that choice in the
isolated worker request. Serial/two-rank tests check matching displacement,
damage, energy and recovered stress for both formulations.

### Experimental principal-strain spectral investigation

Standalone `run-fracture --fracture-mechanics experimental_spectral` selects
an additional investigation law; this is **not an app/API option or default**.
The optional `--accelerate-damage` flag also accepts this experimental law.
It uses
`psi_plus=lambda*<tr(eps)>_+^2/2 + mu*eps_plus:eps_plus` and
`psi_minus=lambda*<tr(eps)>_-^2/2 + mu*eps_minus:eps_minus`,
where `eps_plus`/`eps_minus` are positive/negative principal-strain tensor
parts. The mechanics energy is `g(alpha)*psi_plus + psi_minus`, with the
same `psi_plus` driving damage history. Negative principal strains, not
merely negative volume change, retain their stiffness. This still does not
implement unilateral crack-face contact, friction or crushing.

`spectral_split.py` evaluates strain eigensystems in DG0, which is exact
cellwise for affine P1 tetrahedra. Other displacement orders or curved/non-tet
geometry are rejected; the experimental energy also requires nonnegative
lambda. Nodal damage and its degradation remain P1/quadratic expressions,
not cell-averaged substitutes. A divided-difference spectral derivative
provides the consistent semismooth tangent, including repeated eigenvalues;
zero eigenvalues use the tensile-side generalized derivative. Public DOLFINx
residual/Jacobian callbacks synchronize displacement and update coefficients
before each assembly, including line-search trial residuals and the independent
final equilibrium check. Non-finite strain is rejected collectively.
Mechanics, history, internal energy, exported stress and recovered stress use
the same selected law. Both manifest and report retain its experimental name.

`test_spectral_split` verifies rotated/repeated principal-strain branches,
finite-difference energy/stress/tangent consistency, fully damaged
nonhydrostatic compression, explicit failure acceptance, and accelerated versus
unaccelerated coupled fields/energies. The parallel tests also check spectral
serial/two-rank agreement in energy, displacement, damage and recovered stress.
These checks do not establish full-dam convergence or engineering validity.

All three fresh eight-rank full-dam spectral diagnostics were stopped by the
590-second development deadline, with no complete report:

| Loading/iteration strategy | Last completed state | Damage change |
|---|---|---|
| Original ten steps, no acceleration | 90% load, iteration 5 | 0.001041 |
| Original ten steps, bounded Aitken | Full load, iteration 3 | 0.006581 |
| One full-load increment, bounded Aitken | Full load, iteration 22 | 0.042066 |

The last trial changed the loading path and cannot replace the configured
analysis. Its final recorded strain norm was approximately 0.01184 and
mechanics still passed equilibrium, but damage had not settled. Do not call
these trials a working fix or claim that undegraded compression alone cures
the coupled response. The ten-minute cutoff belongs to local agent development
only; the app/API does not impose it on user jobs. No checkpoint continuation
was used to bypass that development rule.

Saved-state audit of the single-increment spectral trial confirmed plinth
damage was exactly zero and the seed lower bound was retained to roundoff
(maximum removal approximately `1.1e-13`). Damage increased by more than
`0.1` at 17,973 wall DOFs, with maximum increase `0.5874` near
`(78.1990, 15.9654, -28.0)` m. No additional DOFs had reached `alpha>=0.99`.
The maximum-strain cell was predominantly tensile (principal strains
approximately `[-6.11e-5, -9.17e-6, 0.01184]`). This is evidence of evolving
wall damage in an unconverged state, not proof of crack propagation stability,
loss of global equilibrium, or physical dam failure.

Each staggered diagnostic/report now includes `damage_change_location`:
the coordinate and signed largest **unrelaxed** damage update, previous
iterate, minimised candidate and previous-load lower bound. Selection uses
owned DOFs collectively, excluding duplicate ghosts. It is captured before
any Aitken extrapolation. Consecutive locations/updates can distinguish an
oscillating region from a moving region of damage growth; neither pattern
alone establishes an admissible completed fracture solution.

`solver.field_output_interval` controls VTK snapshots independently of the JSON
diagnostics, which are retained for every load step. The default value `10`
writes final fields for the ten-step example and avoids the roughly 3.5 GB
produced when displacement, damage and history were written at every step.

The checked-in horizontal crack uses the existing scenario location at
chainage `108.5 m`, elevation `-25 m`, converted on the 78 m wall centreline to
global Cartesian centre `(76.743008, 13.946709, -25) m`. The placement guard
confirms it lies in wall region 2. Do not treat a converged run as a dam-safety
assessment without independent engineering verification, mesh/length-scale
sensitivity studies and fracture-property calibration.

## Reduced frictionless contact benchmark

`frictionless_contact_benchmark` is a **synthetic 1-D two-bar verification case**,
not a dam crack or fracture model. It exercises independent crack-lip degrees of
freedom, gap closure, compressive contact pressure, release and load reversal.
The bars have lengths 1 m and 1.5 m, area 0.01 m2, and the current load case's
Young's modulus (35 GPa). These lengths, area and initial gap are benchmark inputs,
not inferred dam fracture properties. There are no body forces or tangential DOFs.

The left outer end is fixed. The right outer end has displacement `-c`, where
positive `c` is prescribed approach and negative `c` is separation.
Let `u_left` and `u_right` be the interface displacements and `g0` the initial
geometrical gap:

```text
g = g0 + u_right - u_left
g >= 0, P >= 0, P*g = 0
```

`P` is compressive contact force in N; pressure `p=P/A` is positive in compression.
Bulk axial stress is `-p`. The two disconnected interval meshes retain separate
lip identities even at zero initial gap.

DOLFINx assembles P1 axial elasticity. A single Lagrange multiplier enforces
contact exactly (no penalty stiffness). An active-set iteration solves the sparse
linear KKT system while closed, or unconstrained elasticity while open. Negative
trial contact force releases the constraint; penetration activates it. Matrix
scaling uses a representative elastic stiffness. SciPy's installed sparse direct
solver handles this small system; singular/non-finite solves and exhausted active
set iterations raise errors. This is a bounded benchmark implementation, not a
general-purpose 3-D contact library.

The independent analytical solution for the equal-modulus bars is:

```text
p = max(c - g0, 0) * E / (L_left + L_right)
g = max(g0 - c, 0)
u_left(x)  = -p*x/E
u_right(x) = -c + p*(right_outer_coordinate - x)/E
```

Only verification uses those expressions; the numerical contact solution is
obtained from the assembled matrix and inequality constraints.
The configured eight-step sequence opens, touches, compresses, unloads and opens
again on two meshes (2 and 8 elements per bar). At maximum closure of 0.2 mm
with a 0.1 mm initial gap, the expected pressure is 1.4 MPa and force is 14 kN.
Unloading below the initial gap must return zero pressure without tensile contact
or retained adhesion.

Configuration exposes lengths, area, gap, meshes, ordered closures, gap and
displacement tolerances, relative error tolerance and maximum active-set iterations.
Validation requires both opening and closure and reopening after maximum closure.
Every increment checks gap, pressure, nodal displacements, cell stresses,
equilibrium, outer reactions, nonpenetration and complementarity.
The relative errors use `max(E*A*max(g0,abs(c))/(L_left+L_right),1 N)` as force scale,
not a division by the contact force (which vanishes while open).
Complementarity `abs(P*g)/force_scale` has units of metres.

Each mesh's output folder contains displacement and stress PVD/VTU time series;
time values are **step indices**, not physical seconds. `summary.json` includes
every step's pressure, aperture, exact values, active-set iterations and errors.
Benchmark success is recorded as `passed_reduced_contact_benchmark`; it does not
unlock `predefined_crack` scenarios.

The maintained [dolfinx-contact](https://github.com/Wells-Group/asimov-contact)
project targets DOLFINx 0.11, but is experimental and requires a native build;
it is not currently installed and no packages were added for this benchmark.
Before dam contact runs, evaluate a version-pinned 3-D contact implementation,
verify matching/nonmatching crack surfaces and partial opening on small meshes,
then test friction and water loading explicitly when requested.
This reduced case cannot verify surface search, mixed active/open faces, friction,
water pressure, a crack front, crack propagation or dam safety.

## Dam import and elastic parity

### Corrected reservoir load

The current load case has 3 m of overflow head. The upstream pressure is
`9810 * max(3 - z, 0)` Pa: 29430 Pa at the crest and 313920 Pa at z = -29 m.
The contours in root `config.json` retain crest z = 0 without an implicit shift.
The 31 m reference water depth gives a reported reference peak of 333540 Pa,
not the maximum pressure on the current mesh.
This is a head offset, not an increased gradient. Rerun the combined Elmer solver
before the dam parity task: the earlier reference and reported parity numbers
below predate this correction and are historical, not corrected-load results.
The reference-input check intentionally rejects those older inputs. The reduced
two-bar contact benchmark is unaffected; it does not model water.

The enabled `dam_elastic_parity` scenario consumes the current procedural model
and [root configuration](../config.json). The corresponding generated
Gmsh/SIF inputs must match the reference inputs exactly, ignoring newline style;
reference outputs must not predate those inputs. The PVTU and each piece are hashed
into the manifest. This detects input mismatch and obvious stale files but cannot
prove the provenance or convergence of an externally produced Elmer run.

DOLFINx 0.11's experimental mixed-topology form API does not yet handle the
coefficients and tagged boundary integrals required here. The approved alternative
is deterministic pulling triangulation with shared face-centre nodes on warped
quadrilaterals and cell-centre nodes in the affected cells. A simple corner-only
split changed individual warped-prism volumes by up to 0.4444%, despite preserving
total volume, and was therefore replaced before accepting the port.
All 121,097 original nodes are retained, including separate coincident interface
nodes; 2,240 nodes are added and 101,724 source cells become 614,824 tetrahedra.
Adjacent faces share the same subdivision, with no hanging faces.
Every exterior triangle must match a source tag exactly. Volume differences are
checked both globally and for every parent cell against the configured limit.
The verified import has zero measured total-volume change and a worst
parent-cell relative difference of 3.19e-14.
The element formulation changes from hex/prism to linear tetrahedra; geometric
equivalence does not imply discretization equivalence.

`import-dam` writes `results/dam_import/mesh.xdmf`, its HDF5 companion and
`import_report.json`. Open the XDMF in ParaView to inspect the mesh, `regions`
and `boundaries`. DOLFINx performs this compiled HDF5 I/O in Ubuntu; Windows
`h5py` is not used.

The legacy dam solve uses small-strain isotropic elasticity, CG/GAMG with rigid-body
near-nullspace modes, gravity, upstream pressure, tailwater and crest surcharge.
Foundation springs act on tag 1. **The historical split-interface Elmer SIF grounded wall-side tag 8
with directional springs; it does not couple wall displacement to plinth
displacement.** Saved split meshes remain identifiable in the low-level adapter
as `legacy_split_grounded`, retaining their two support laws. Newly generated
meshes instead use the separately identified bonded reference above.

### Read-only wall/plinth load-path audit

`load_path_audit.py` checks a saved serial crack run without solving, resuming
or changing its data. It verifies region connectivity, tagged surface areas,
checkpoint mapping and the last pre-iteration strain/damage state. It requires
the new refined-baseline artifacts and refuses an existing output filename.
The checkpoint is **not an accepted final state** and precedes the failed
mechanics update.

From the `FEniCSx` directory in Ubuntu:

```bash
python3 load_path_audit.py --run /path/to/isolated/crack/run --output /path/to/new-audit.json
python3 -m unittest test_load_path_audit test_contact_benchmark
```

The October 17 failed job `d84722f4cce34ef7920f0547241b698b` has zero shared
wall/plinth vertices, 3,969 coincident coordinates and separate exterior faces
on tags 8/9. Each face set has area approximately 659.99548 m2, but refinement
leaves different facet counts (36,196 wall-side versus 5,280 plinth-side).
Those surfaces cannot simply be paired by triangle index or welded without
checking conformity.

Its checkpoint before step 10 iteration 21 has strain norm 0.10591, principal
compression -0.10542, and hotspot vertex damage 0.9989466-1.0. Hybrid vertex
stiffness fractions are approximately 1e-8 to 1.12e-6. Wall-base damage reaches
0.9997813; plinth damage is zero. This contradicts an explanation based on a
shared, zero-damage wall-side trace in this particular mesh.

The scalar split-versus-bonded regression applies a unit wall-only load:
independent ground supports produce zero plinth response; a bonded block
transfers that load to the plinth foundation. This verifies connectivity
semantics, **not 3D elasticity, friction, separation or dam behaviour**.
The existing two-bar unilateral-contact benchmark remains separate.

The physical connection must ultimately be justified from construction evidence:
bonded connection for an intact monolithic idealisation, or an interface law
for an opening/sliding joint. A controlled bonded reference is a useful
diagnostic, but not a substitute for a joint model when separation is intended.
Do not silently retain the wall-ground bypass spring in a bonded reference:
that would provide a second, different support path. Both Elmer and FEniCSx
now use the newly checked bonded reference above. Permitting plinth damage alone
cannot create the missing mechanical connection.

Elmer exports deformed coordinates and includes surface cells. The comparator
subtracts its displacement field to recover source positions, filters volume
cells, resolves coincident interface nodes by region, checks all source cells
and nodes are covered, and verifies ghost-node displacement consistency.
Comparison measures:

- exact tetrahedral L2 norm of the difference between **P1 interpolants of both
  nodal solutions**, divided by the norm of the Elmer nodal interpolant (5% limit);
- vector displacement differences at named probes in the same source region
  (10% limit), with actual coordinates and nearest-node offsets reported;
- spring-reaction differences integrated from each solution's nodal values on
  the same tagged triangles (5% limit); these are reconstructed Elmer reactions,
  not a native reaction-output comparison;
- force balance between applied loads and FEniCSx ground-spring reactions
  (0.1% limit);
- volume-averaged stress components by region, diagnostic only because Elmer's
  recovered nodal stresses differ from tetrahedral DG0 stresses.

Limits, probe positions/search distances and solver tolerances are configurable
in `cracking.json`. A failure persists outputs and diagnostics but does not
advance `latest.json`, and returns a failing exit status. Dam runs also write
`import_report.json` and `nodal_displacement.npz`.

The accepted refined-mesh comparison measured 0.697% global displacement difference,
0.227-1.058% at the four probes, 0.00264-0.00429% reconstructed spring-reaction
differences and a 1.33e-11 force-balance error. The solver
converged in 41 iterations. These are model-port checks, not evidence of dam safety.
Visual mesh inspection, mesh-sensitivity checks and physical review of the
grounded joint remain necessary before relying on fracture scenarios.

## Porting and validation gates

1. **Inventory**: audit the procedural model and preserve region/boundary tags,
   mixed element types, units, current material/load data and spring definitions.
   `results/migration_inventory.json` captures the actual model counts.
2. **Import (implemented)**: resolve mixed-cell compatibility with checked subdivision;
   export/inspect geometry and tags in ParaView. New meshes preserve shared bonded
   wall/plinth nodes; historical split nodes remain distinct when reading old fields.
3. **Elastic dam parity (bonded reference checked)**: reproduce the current Elmer foundation
   directional springs, gravity, hydrostatic pressure, tailwater and overflow.
   The provisional bond is an explicit model change, not a contact model. Compare matching
   displacement probes, reactions, equilibrium and stress recovery.
4. **Contact benchmark (reduced 1-D checks implemented)**: verify an explicitly selected unilateral contact
   formulation, gap convention, closure/opening, friction when requested and
   convergence under load stepping. Avoid inheriting mesh-plane snapping from
   the Code_Aster crack inserter without measuring actual geometry.
5. **Dam crack scenarios**: activate the task-driven crack adapter only after
   those gates pass. Establish mesh sensitivity and report limitations.

The existing Elmer material JSON includes a nonlinear-damage flag, but the current
combined SIF uses linear elasticity; this inventory does not claim to port a
damage law. Benchmarks and solver convergence do not establish dam safety.
Engineering validation requires independent review.
