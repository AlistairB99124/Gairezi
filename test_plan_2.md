# Fixed-load Aster campaign: what produces 100 mm wall displacement?

## Scope

Reference job: `d7052bc6d4a34315a6daacdb77a72b42`, native run
`20261101T151043Z-30b55786`. Preserve its frozen geometry, mesh, intact
concrete, foundation, springs, density, gravity, reservoir/tailwater and crest
loads. Change only selected-wall properties and automatic connected selection
extent. Keep CZM_LIN_REG; no MAZARS, crushing, new crack-water pressure,
load amplification or relaxed convergence tolerances.

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
- Freeze exact selected rows/candidate topology per cap; original ft0.8 MPa
  still controls automatic screening.
- LF<=1 only. Disable the campaign stop-at-first-opening overload search, not
  production defaults. Start low-strength cases at LF0.01; retain adaptive
  subdivisions and damage irreversibility.
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
- A1 isolated saved-run reproduction E10/nu0.15/ft0.8/Gf150.
- A2 selected properties equal original concrete, identical candidate topology.
- A3 E10/nu0.2/ft0.8/Gf100: separate stiffness from Poisson/toughness.
- Same-topology bonded/elastic-joint control for decisive cases, labelled
  diagnostic rather than an accepted fracture result.
Investigate unexplained reproduction/topology discrepancies before continuing.

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
`--preflight-only` authenticates/prepares caps without native solves.
Never use an application-job directory as campaign output.