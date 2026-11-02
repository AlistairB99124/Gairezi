# Fixed-load Aster campaign: what produces 100 mm wall displacement?

## Scope

> **Amendment (2026-11-02):** the original reference job was deleted from app
> storage. The campaign is re-anchored to job `8bef80bd1c324e808a7358271a5b578f`
> (intact best-case baseline, concrete ft 2.0 MPa, completed native elastic
> staircase to 2.0x design load). Selection is now rank-based: every tensioned
> interior wall facet is eligible and each cap keeps the top-stress connected
> patch (reference cap 2000; extent ladder 1000/2000/4000/8000). A-controls use
> intact strength ft 2.0 MPa; the historical-material cases keep ft 0.8 MPa.
> The paragraphs below describe the original campaign against the deleted job
> and are retained as history.

Reference job: `d7052bc6d4a34315a6daacdb77a72b42`, native run
`20261101T151043Z-30b55786`. Preserve its frozen geometry, mesh, intact
concrete, foundation, springs, density, gravity, reservoir/tailwater and crest
loads, except for the hydraulic scope amendment below. Change only selected-wall
properties and automatic connected selection extent once the new full-load
reference is established. Keep CZM_LIN_REG; no MAZARS, crushing, load
amplification or relaxed convergence tolerances.

### Required hydraulic scope amendment

The user requires full exterior hydrostatic pressure, undrained base uplift
varying from upstream reservoir head at the heel to tailwater head at the toe
without drainage reduction, and full reservoir pressure on hydraulically
connected opened crack faces. Do not pressurise dry/disconnected cracks.

The saved native baseline has **zero uplift**. The isolated campaign now enables
`hydraulic_loads` in its native mesh manifest. Historical/default app workflows
are unchanged; old results are not retrospectively re-labelled as full-load.

`Aster.hydraulic_loads` applies undrained pressure into the wall-side foundation
boundary only. Measured heel/toe radii along the source arch define the transverse
head fraction; crest thickness is not used. Source node identities preserve
the physical boundary through cohesive/relief copies. Linear triangular pressure
uses consistent P1 nodal forces with exact clipped-waterline integration.

Open crack triangles receive paired equal/opposite hydrostatic forces only when
reachable from a submerged upstream exterior edge through open shared edges.
Opening is the signed normal jump, with an explicit 1e-9 m numerical tolerance.
Face/edge-average activation is a discrete facet approximation: partial wetting,
permeability, transient flow and trapped-water pressure are not modelled.
Pressure acts on reference-geometry normals (small strain, not follower loads).
Closure/disconnection deactivates pressure; mechanical separation remains
irreversible. A dry isolated crack is never automatically filled.

Each load instant iterates its hydraulic active set, at most20 trials. Every
trial starts from the last accepted mechanical state; rejected trial damage is
discarded. Only a stable wet mask is accepted. Mechanical/hydraulic failures
stop explicitly, with previous accepted evidence retained. Hydraulic runs use
explicit outer load instants without automatic internal subdivisions: internally
converged mechanical instants with an unverified wet mask are not valid hydraulic
evidence. This intentionally differs from historical dry adaptive subdivision;
solver tolerances remain unchanged.

Execution first solves a new isolated intact uplift baseline, checks force and
moment equilibrium (1% diagnostic gate), and re-screens its stresses. This solve
counts against the64-launch budget. A1/A2 use original concrete and compare
against that new baseline. A4 reproduces historical selected material under the
new hydraulic loads, without comparing it to old no-uplift maxima. Historical
preflight caps cannot be reused in a hydraulic execution directory.

Target: first **converged physical wall-node displacement magnitude >=0.100 m**,
not display deformation, auxiliary nodes or displacement difference.
This diagnostic threshold does not establish real-world dam failure/safety.
The app's existing 25-100 mm inclusive gauge remains unchanged.

User authorised synthetic ft0.8-0.02 MPa, Gf150-1 J/m2, E35-0.1 GPa,
increasing connected selection caps, and no agent wall-clock cap.
These are sensitivity values, not calibrated concrete. Stop after a bounded
matrix or a verified threshold plus causal checks; never retry indefinitely.

## Reference

Intact: E35 GPa, nu0.2, rho2400, ft0.8 MPa, Gf100 J/m2.
Saved selected: E10 GPa, nu0.15, rho2400, ft0.8 MPa, Gf150 J/m2.
Cap3000 selected2998 cells (whole node-star groups), approximately2.12% wall
volume,5313 candidate facets. Original/final maxima9.342/9.248 mm at LF1;
tension1.082/1.599 MPa. 397 facets initiated, none fully separated; peak opening
0.037 micrometres versus375 micrometres for separation. A smaller maximum
displacement did not mean increased stiffness.

## Required controls

- One native solver at a time; maximum64 full-dam launches including refinement
  and controls, reserving8 for causal verification.
- Isolated outputs outside source control. Never modify/restart the saved job.
- Authenticate original baseline/config/mesh hashes. Screen/prepare with the
  original configuration, then record only authorised selected-property
  differences. No rewritten original hashes or fictitious modified baseline.
- Freeze newly screened selected rows/candidate topology per cap; original ft0.8 MPa
  still controls automatic screening.
- LF<=1 only. Disable the campaign stop-at-first-opening overload search, not
  production defaults. Start low-strength cases at LF0.01; retain damage
  irreversibility and accept only hydraulically converged explicit instants.
- Save actual accepted native instants, physical wall maxima/components/locations,
  matched gauges and fields before advancing; preserve last valid state on failure.
- No outer wall-clock timeout. Keep native iteration/CPU safeguards and explicit
  process-tree cancellation. The installed launcher's native CPU budget must be
  disclosed separately: unlimited wall-clock is not an unlimited native CPU budget.
  No unattended execution until this distinction is agreed.
- Resource preflight, live heartbeat and progress reports at least every15 minutes
  when deliverable; persistent on-disk status while the user is offline.

## Ordered matrix

### A: constitutive gate and controls

Small native specimen: elastic response, initiation, softening, full separation,
irreversible unloading/reloading, reactions and traction/opening/state1/state2
checks. Stop if constitutive response or extraction disagrees.
At cap3000:
- A1/A2 original material on the new full-load candidate topology, compared
  against the new uplift baseline.
- A3 E10/nu0.2/ft0.8/Gf100: separate stiffness from Poisson/toughness.
- A4 saved E10/nu0.15/ft0.8/Gf150 material, now under full hydraulic loads.
- Same-topology bonded/elastic-joint control for decisive cases, labelled
  diagnostic rather than an accepted fracture result.
Investigate unexplained reproduction/topology discrepancies before continuing.
The runner stops if either native A1/A2 maximum differs from its reference by
more than1% (a diagnostic gate, not a safety acceptance threshold).

### B: strength-only initiation

Cap3000, E35 GPa, nu0.2, rho2400, Gf100 J/m2.
ft=[0.8,0.4,0.2,0.1,0.05,0.02] MPa.
Compressive reference30 MPa is not an active crushing law and is not swept.
At fixed Gf, lower ft increases2Gf/ft: this is not automatically a
complete-separation weakening sweep.

### C: independent toughness

E35/nu0.2/rho2400/cap3000; Gf=[100,25,5,1] at ft0.2 and0.05 MPa.
If necessary add ft0.02 with Gf=[25,5,1]. Moderate-to-extreme ordering.
Do not couple Gf to ft automatically. Gf150 is only a saved toughness control.
Track initiation/softening/separation separately.

### D: extent/connectivity

Preflight caps=[1500,3000,6000,9000]. Record actual count/volume fraction,
source-row/topology hashes, connected components, area and remaining ties.
Smaller patches are not assumed nested.
Apply up to2 informative B/C material pairs to each feasible cap at E35.
Keep existing20000 candidate-facet limit; record preparation-limited cases.
No truncation, tie removal or raised limits. Uncapped selection is neither a
whole-wall nor necessarily the same connected patch and is outside this campaign.

### E: elastic-softening contrast

Largest feasible cap; E=[10,5,1,0.1] GPa, nu0.2/rho2400.
First ft0.8/Gf100; repeat most informative low-ft/low-Gf pair at up to2 E levels.
Compare E35 at identical rows/topology. If only lower E yields100 mm, identify
imposed pre-cracking elastic softening, not fracture alone.
No zero/negative materials or expansion below approved bounds.

### F: threshold and causal confirmation

Repeat a credible>=100 mm case with finer loading. Restore ft0.8, Gf100 and
E35 separately where different; add same-topology bonded control.
Refine up to3 material points between nearest converged below/above cases
without assuming monotonicity. Refine accepted LF crossing width<=0.025 where
solvable. Inspect physical peak nodes, crest gauges, contact, connected separated
facets, load paths and block motion. Separate isolated/sliver motion from global
wall deformation. No hidden penalty tuning or mesh changes.

## Evidence and outcomes

Statuses: converged_below_100mm; converged_at_or_above_100mm;
nonconverged_unresolved (last accepted LF/displacement); preparation_limited;
resource_failure; invalid_evidence; cancelled/interrupted.
Non-convergence is not proof of physical failure and never supplies a fabricated
displacement or a supporting-solver fallback.

Record cap/count, E/nu/rho/ft/Gf, actual LF history, runtime/residuals,
all input hashes, maxima/components/physical peak coordinates, matched gauges,
local vector change, raw P1 stresses, initiated/opened/separated counts/areas,
normal/tangential jumps/critical-opening ratios and termination reason.
Verify native reaction/energy evidence or mark unavailable.
Reconstructed external-load work is not a native nonlinear energy balance;
peak-over-patch dissipation cannot be integrated as exact fracture energy.
Mesh convergence remains unverified unless separately demonstrated.

Checkpoint every case and accepted step. Stop on exhausted memory/disk without
disrupting other users; interrupted solves are not silently re-run unchanged.
Reserve confirmation launches. Deliver CSV/JSON, parameter/LF plots and a causal
report stating threshold conditions or not reached within approved bounds,
unresolved cases and constitutive limitations.

## Execution

First implement and verify the isolated runner with focused unit/native smoke
tests. Full overnight execution requires explicit approval and has not started.
Use `python -m Aster.displacement_campaign --help` from the Gairezi environment.
`--preflight-only` authenticates/prepares historical caps without native solves;
this remains a no-uplift preview, not the new hydraulic screening.
Never use an application-job directory as campaign output.
Full campaign preparation/execution uses Ubuntu's SciPy-capable Python with the
native Aster environment activated, not the application Windows virtualenv.
Omitting `--case-time-limit` removes outer wall-clock enforcement; execute then
requires `--accept-native-cpu-guard` to acknowledge the installed CPU safeguard.
Execution also requires `--constitutive-gate` pointing to hash-authenticated
native specimen step evidence demonstrating initiation, state2 separation and
retained separation on unloading. The runner rejects missing/stale gate evidence.
No full-matrix execution is authorised merely by passing unit/smoke tests.

### Current verification state

Four full-dam cap preflights are feasible: cap1500 ->1499 cells/2621 facets;
3000 ->2998/5313; 6000 ->6000/10737; 9000 ->8999/16184.
Small native fine-step displacement/joint capture and same-topology bonding
controls passed. No full-dam sweep has started.

The separation anomaly was traced to an inherited elastic-only specimen with
ft=1e12 Pa and Gf=1e10 J/m2: its material/penalty ratio was clipped to
PENA_ADHERENCE=1, collapsing the elastic/softening branch distinction. Production
now rejects incompatible ratios rather than silently clipping them.

A corrected native specimen with ft0.8 MPa, Gf100 J/m2 and
PENA_ADHERENCE=9.308565293e-5 passes elastic/initiation/softening/separation,
complete closure and reopening. Initial state0 has peak opening2.865e-14 m.
Pointwise separated open joints carry zero tensile traction; separation remains
irreversible on closure/reload. Facet maxima may combine different integration
points, so a facet touched by separation can still carry traction elsewhere.
This corrects the specimen, not a demonstration that the full dam separates.
Hydraulic implementation and the isolated native reference verification now pass.
Full-matrix execution has not been started.

Hydraulic implementation checks now pass for small native uplift and wet/dry
crack-water specimens: uplift has correct sign and force/moment balance;
wet pressure activates with2 active-set iterations, while dry cracks stay
unpressurised. Full mesh uplift preparation identifies2472 base triangles,
845.51 m2, upward resultant85.82 MN (plus inclined-base horizontal components).
This is load preparation, not a full-dam failure result.

The new intact uplift reference solved in154.4 seconds. Maximum physical wall
displacement is9.370 mm; relative force/moment residuals are6.02e-6/3.09e-6.
Re-screening cap3000 now selects2999 cells and5311 candidate faces, rather than
reusing old2998/5313 topology. Full native hydraulic crack preparation/campaign
handoff compiles and retains authenticated original source hashes.
The bounded100 mm fracture matrix itself remains unexecuted.

### VS Code launch task

Run **Aster: Run full hydraulic displacement sweep** from the root workspace's
`.vscode/tasks.json`. It uses the existing Ubuntu Aster17.4 installation and
saved reference job, creates a timestamped directory under
`Aster/results/hydraulic_sweeps`, and uses the locally preserved, authenticated
constitutive gate at `Aster/results/hydraulic_gate/gate.json`. Gate/result data
are ignored by Git. Missing/stale gate data cause an explicit refusal.
The task discloses and acknowledges the installed native CPU safeguard; there
is no outer wall-clock cap. It is not run automatically after editing.

## Visible progress log

[sweep.log](./sweep.log) records the investigation and current execution blockers.
It is generated and ignored by Git. The campaign runner appends UTC-timestamped
start/finish events, cap preparation, case parameters, displacement/threshold
outcomes, errors and cancellation. During a run it appends a heartbeat every
30 seconds with active case, launch count, available memory and native solver
log timestamp/path. Native iteration details remain in each case's
`native/run/solver.live.log`; detailed accepted-step fields remain in the case
output. A heartbeat indicates runner activity, not solver convergence.
Existing log entries are preserved across campaign invocations.

### Additional Google model suggestion

Installed Code_Aster17.4 contains neither `ENDO_FISS_URG_DIAG` nor `RT_DIAG`,
`GF_DIAG`, `ALPH_DIAG`, `BET_DIAG` in its material catalog. Do not add those
unverified keywords. The installed `ENDO_FISS_EXP` is a different gradient-damage
formulation with different parameters and modelisation; it is not a drop-in
replacement for CZM_LIN_REG. A calibrated continuum/gradient-damage comparison
would be a separate study, outside this fixed-model campaign.