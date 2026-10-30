# FEniCSx dam analysis adapter

## Painted tensile-loss building blocks

These are experimental verification components, **not a runnable painted dam
scenario**, and none is selected as a substitute for the retired legacy seed.

The target is eventual quasi-static equilibrium, not second-by-second dynamics
or filling time. Numerical continuation is not physical time. Existing
transient-flow benchmarks remain verification components, not mandatory
physical-time inputs for drained hydrostatic cases.

`EquilibriumOpeningObservation` / `assess_equilibrium_opening` in
`horizontal_fracture_contract.py` report accepted equilibrium state IDs,
peak and final opening separately, without fabricated timestamps. A peak
at/above the opening limit remains a warning after reclosure; incomplete
histories cannot satisfy the criterion. No-fracture states remain reportable.
The existing physical-time assessment is retained unchanged for transient
callers, sharing threshold/warning logic. Native hydrostatic opening checks
feed actual measured displacement jumps into the equilibrium assessment.
This is a result contract, not production scenario/result UI integration.

- `fracture_hydrostatic.py` assigns equilibrium pressure to explicitly
  saturated, fully submerged, open crack faces connected to named hydraulic
  inlet edges. Pressure is `rho*g*(head-z)` on the reference small-displacement
  geometry, integrated by the existing consistent paired-face force operator.
  Saturation and previously wet face masks are explicit; geometric access does
  not invent filling or drainage. Wet isolated/reclosed pockets fail pending
  a trapped-fluid/contact state. Connected different-head boundaries require
  steady flow and fail rather than using an arbitrary head; disconnected
  components may have independent heads. Partially submerged crack faces fail
  pending free-surface hydraulic connectivity, even though exterior wall loads
  already support clipped pressure integration.
  Tests check analytical pressure gradients, resultant/moment balance and
  virtual work, same-head nonduplication, independent heads, saturation and
  trapped-pocket exclusions. A native 3-D explicit-cut two-block check converts
  a specified reservoir head into analytical zero-initial-gap opening and
  verifies hydraulic access/load consistency again on the solved displacement.
  This is a reduced prescribed-saturation equilibrium loading component, not
  irreversible growth, a coupled contact active-set solver or a stability
  assessment. No app submission is enabled.

- `fracture_steady_flow.py` extends reduced equilibrium loading to different
  boundary heads on an explicitly saturated, prescribed fully open surface.
  Surface P1 head solves `div(w^3 grad_surface H)=0`; pressure is
  `rho*g*(H-z)` and the existing paired-face operator returns consistent loads.
  Affine aperture cubed is integrated exactly on each reference triangle.
  A spatially uniform viscosity factor cancels from the source-free head solve;
  no viscosity default or physical discharge is invented. Boundary reactions
  are explicitly viscosity-scaled (`m4`), not leakage rates in `m3/s`.
  Independent residual/boundary-balance audits reject failed solves. Head
  offsets avoid cancellation at equal heads. Both-lip identities preserve
  coincident disconnected sheets and owner reversals; point-only junctions,
  nonmanifold edges, conflicting boundary values, closed/tied vertices,
  unsaturated/isolated wet faces and negative pressure fail explicitly.
  Nine tests check analytical rectangular head/flow, cubic aperture scaling,
  variable-aperture analytical convergence, hydrostatic equivalence,
  disconnected components, identity and input/unsupported-state guards.
  This is not a contact/flow iteration or dam adapter; hydrostatic loading
  still rejects unequal heads unless the caller explicitly chooses this
  steady-flow component. Trapped pockets, partial saturation/free surfaces and
  irreversible growth remain unimplemented.

- `compliant_steady_crack` in the same module additionally couples pressure
  and aperture in both directions for a reduced already-open planar specimen:
  `w = w_ref + C*(p-p_ref)`, followed by the aperture-dependent surface head
  solve. `SteadyCouplingOptions` requires explicit aperture/flow tolerances,
  relaxation and an iteration bound. Acceptance checks the local mechanical
  residual on the same aperture whose head solve passed independent flow/
  boundary balance. Nonconvergence and predicted closure fail, with no
  residual-aperture regularisation or success-shaped fallback.
  Five tests verify hydrostatic response, the zero-compliance limit, reference
  pressure and owner-reversal invariance, explicit failure, and mesh convergence
  against the independent integrated nonlinear steady solution on a horizontal
  surface. The pressure/aperture feedback produces a non-linear head distinct
  from the fixed-aperture solution. Reference aperture/compliance values in
  these tests are synthetic specimen data, **not an initial gap or replacement
  constitutive law for the user's dam**. This is not bulk 3-D elasticity,
  contact/friction, irreversible propagation or a production coupling adapter.

- `open_crack_equilibrium.py` now couples actual serial blocked-P1 **3-D bulk
  elasticity** to the steady surface-flow component on a prescribed split mesh.
  The dry external-load equilibrium must already open every hydraulic vertex;
  no gap, seed displacement or minimum aperture is imposed. Unchanged UFL bulk
  elasticity/load forms and BCs are reused. Pressure forces map through source
  connectivity, never coordinate welding. Each candidate recomputes pressure
  from its own aperture; acceptance requires independent bulk free residual,
  global force balance, flow balance and displacement correction checks on
  that same state. Explicit coupling options bound iteration/relaxation.
  Output must be a new directory. Dry state/input evidence and failure trace
  persist; accepted-only NPZ/VTK exports contain the coupled solution. VTK
  states 0/1 mean dry/coupled equilibrium, not physical seconds.
  Optional existing mortar spring operators are retained with serial Dirichlet
  lifting/elimination. The independent audit uses the original unconstrained
  coefficients, archived with cell DOFs; dry/wet spring energies are reported.
  Fixed nodal external loads in displacement-DOF order are retained throughout
  iteration and archived, allowing exact clipped exterior water-pressure loads.
  Six native tests check aperture-to-head feedback under actual bulk
  deformation, the hydrostatic-head limit, archived fields and explicit
  closed-dry-cut/nonconvergence failures. Inventory lips must be actual exterior
  bulk triangles with normals pointing out of minus solid into plus; mismatched
  coordinates/orientation fail before output creation. A four-block mortar-seat
  check verifies analytical dry opening/energy under nonzero support motion,
  unchanged coefficients and wet-state balance; another verifies equivalent
  nodal/form loads produce identical coupled fields. Synthetic block inputs are not
  dam calibration. This fixed-cut elastic adapter does **not** solve the
  painted tensile-loss event, contact/friction, irreversible growth, trapped
  fluid or the full dam. It does not expose scenario submission.

- `paired_contact.py` adds serial **3-D frictionless contact** on explicitly
  supplied paired affine-P1 faces. Unique vertex-normal constraints enforce
  nonpenetration across each face; shared front vertices remain tied. An exact
  multiplier active set uses the original bulk stiffness and prescribed
  support motion, with no penalty stiffness, residual aperture or initial gap.
  Acceptance independently checks free residual, multiplier sign, gap and
  complementarity. Both tangential directions remain free: this is explicitly
  frictionless, not a calibrated Coulomb law. Multipliers are nodal forces in N,
  not pointwise contact pressures.
  The native entry point validates actual exterior lips/source identity,
  accepts existing mortar springs and fixed external nodal loads, and applies
  only explicitly prescribed water tractions. Inputs, assembled stiffness,
  prescribed DOFs, active-set trace and failure diagnostics persist in a new
  output directory; accepted NPZ/VTK exports contain contact forces, reactions
  and measured opening. Eight tests cover analytic pressure release/reclosure,
  support-motion compression, oblique normals, both free tangential components,
  partial contact versus independent constrained energy minimisation, identity/
  virtual work, and explicit failure without accepted output.
  This does not infer hydraulic access, saturation or trapped-fluid pressure,
  couple contact to flow/tensile-loss, initiate/grow a crack, or enable dam
  submission. Its sparse active-set implementation has not been demonstrated
  at dam scale; singular/redundant constraint systems fail explicitly.

- `tensile_loss_event.solve_contact_event` now combines these mechanics in one
  solve sequence: an intact **fully bonded** loaded equilibrium, followed by
  selected-cell tensile loss, release of an explicitly supplied paired cut,
  and nonlinear material/contact re-equilibration. Bonded baseline constraints
  tie all three jump components; post-event constraints are unilateral and
  frictionless. The exact no-tension tangent is used without residual tensile
  stiffness. Exterior loads/support motion remain unchanged. New-face water
  can be prescribed separately or calculated using explicit
  `SteadyCrackHydraulics` boundary data. The latter first requires a converged
  dry post-event state that opens the cut, then recomputes surface head/pressure
  on every candidate's own aperture. Final acceptance checks the actual
  nonlinear material residual, contact complementarity, global force balance,
  and hydraulic residual on that same candidate, not a lagged pressure load.
  Closed cuts cannot get an artificial hydraulic gap. Hydraulic input snapshots
  retain parent identities, saturation, boundary heads/edges and fluid parameters.
  Inputs, intact/dry states and iteration/failure evidence persist; post-event
  material/displacement/contact and hydraulic results are accepted-only.
  The existing equilibrium opening assessment is applied to accepted bonded/
  dry/final states, preserving peak opening without inventing physical seconds.
  The numeric limit (default 0.002 m) produces warnings, not a capped gap or
  failed equilibrium.
  Additional tests verify actual selected tensile stress release and load
  redistribution, analytical bonded baseline fields, unchanged compression,
  joint contact/steady-pressure equilibrium, boundary-data immutability and
  failure persistence. This is **still not the painted workflow**: the caller
  must supply the cut. Selected edges do not yet create/propagate a surface,
  and no irreversible growth law, friction calibration, trapped-fluid/free-
  surface treatment, full-dam resource verification or API submission is claimed.

- `dry_fracture_growth.py` supplies an experimental **dry finite-candidate
  Griffith energy search**, not an unrestricted growth method. For each
  edge-front triangle trial it re-equilibrates the entire accumulated cut,
  compares mechanical potential release against `Gc * new area`, and accepts
  only a strict total-energy decrease beyond an explicit joule tolerance.
  Existing broken surfaces never heal; marker edges create no free gap or
  energy-free surface. Shared edges, not point contacts or coordinate proximity,
  define eligible extensions. Unresolved/failed trials abort rather than
  counting as evidence of arrest, and an extension limit with a favourable
  remaining trial is failure, not convergence. Reports say only
  `arrested_in_tested_single_face_candidate_set`: multi-face energy barriers
  and unrepresented orientations are not ruled out.
  `interior_wall_candidates` extracts all actual wall/wall interior facets
  with canonical source IDs, excludes rock/exterior faces, and verifies marker
  incidence without a horizontal/axis filter. Facet restriction remains a
  mesh-direction bias; this has **not** replaced the required unrestricted
  physical path model or been enabled in the app.
  `solve_contact_event(release_cut=False)` permits post-loss bonded comparison.
  Dry stored material/support/mortar energy and original exterior dead-load
  work give mechanical potential; a SHA256 identity binds geometry, material,
  selection, support operator, prescribed motion and loads before comparison.
  `contact_event_growth_state` accepts only audited dry event results.
  Prescribed crack pressure or transported water makes this dry potential
  unavailable: hydraulic reservoir work must not be silently omitted.
  Seven search tests cover irreversibility, exact energy ties, changed/failed
  trials, stopping limits, source-edge topology and a native displacement-
  controlled 3-D fracture threshold with independently known energy/area.
  Additional native checks compare the same selected-material problem before/
  after cut release and independently verify stored support energy/load work.
  `solve_dry_contact_growth` now performs native selected-material/contact
  trial orchestration on a prepared paired candidate mesh. Unbroken faces
  retain full vector bonds, released faces receive unilateral normal contact,
  and all trials retain the same source mesh, selection, loads and support
  operator. Shared bonded junctions use independent equality constraints
  rather than redundant multiplier rows; tied front nodes cannot open.
  Canonical parent/source-marker identities are archived, and the result
  points to the exact accepted trial's material/contact/VTK exports rather
  than the last attempted candidate. Native eight-candidate tests verify
  energy-driven growth at low resistance and no release at high resistance.
  This lower-level driver takes a prepared candidate mesh; topology resolution,
  tensile-strength nucleation, wet growth, full-dam feasibility and production
  integration are unfinished.
  `solve_source_dry_growth` now prepares that candidate mesh from a canonical
  multi-body source structure and actual element-row/marker-node IDs. Every
  wall/wall internal facet is represented, while rock/foundation cells and
  inherited exterior tags retain identity. Cell rows survive subdivision and
  native reordering; selected original rows affect all corresponding children.
  Existing materials, gravity, roller/fixed supports and exact clipped exterior
  reservoir/tailwater/top-pressure loads enter the trial solves unchanged.
  "Dry" means **new crack surfaces only**; exterior water is not removed.
  An explicit original-parent mortar multiplier map preserves the original
  projector space across duplicated wall nodes. Simply rebuilding multipliers
  independently on split triangles changes foundation stiffness even with
  all candidates bonded; analytical projector/energy and native baseline
  parity tests now guard this. Default standalone mortar assembly is unchanged.
  Source preparation/mapping and downstream failure reports persist; successful
  source reports point to the exact accepted trial. Native source tests check
  original mixed-element row/material/boundary identity, intact displacement
  parity (`rtol=1e-8`, `atol=1e-11 m`), unchanged mortar coefficients/projector
  space, gravity/reactions and partially submerged exterior hydraulic forces.
  Source structures are **not** authenticated saved-request/baseline-bound by
  this experimental entry point (`source_binding_verified=false`). All-facet
  node splitting/trial cost is not demonstrated at dam scale; no full-dam run,
  internal wet propagation, unrestricted path capability or API route is claimed.
  `solve_bound_dry_growth` adds the same baseline/selection verification used by
  `solve_bound_event` before source preparation. The shared verifier rebuilds the
  canonical structure and catalog, verifies baseline field/configuration/mesh
  identity, and rechecks every recorded source diagnostic hash. Catalog marker
  endpoints are explicitly translated from original Gmsh node IDs to the
  compacted source identities used by preparation; edge IDs are not node IDs.
  Selection/baseline evidence, resolved face choices and the user opening limit
  are retained in source reports, including downstream failures. Successful
  bound growth marks source binding verified, but still reports
  `solver_ready=false`: this is dry mesh-constrained experimental growth only.
  Tests combine actual Gmsh catalog/compaction checks, pre-preparation rejection
  of foreign evidence and a native growth run with substituted verified-baseline
  loading. This is not a full saved-dam run or an authenticated submission route.
  Saturated steady flow now has an explicit `allow_tied_fronts=False` option
  on `steady_crack_load`, `SteadyCrackHydraulics` and the elastic bulk driver.
  Opting in permits exact zero aperture only where the two lips are the
  **same mesh node**. Each face must still have positive measured opening and
  boundary access; untied closed/contact vertices, closed inlets and closed
  internal edges fail rather than acquiring a residual gap or hydraulic
  short circuit. Exact integration of the measured affine aperture cubed is
  unchanged. A wedge benchmark checks `H=4-y`, integrated conductance
  `w0^3/4`, pressure virtual work, cubic scaling down to `w0=1e-7 m` and lip
  reversal. A native three-tetrahedron cut retains its two shared tip nodes
  at exactly zero gap while solving material/contact/hydrostatic equilibrium
  and independently rechecking accepted pressure from accepted displacement.
  The option and zero regularisation are recorded in reports. This does not
  model closed saturated faces, trapped water, moving wetting fronts or wet
  propagation; the default strict-open behaviour is preserved.
  `SteadyCrackHydraulics.inventory_face_ids` now optionally identifies the wet
  rows within a larger mechanical candidate inventory. Its immutable ID order
  defines pressure/head archive rows; omitted faces are explicitly declared
  dry, not automatically drained. The material/contact event accepts a partial
  release only when every hydraulic face is released. Unbroken candidates
  retain all-component bonds and receive no crack-pressure load.
  Hydraulic topology can use transitive node equivalence proved by those same
  mechanical bonds; displacement ties and original parent identities are
  checked before flow. This supports a connected wet subset with a bonded
  crack front without coordinate welding or an artificial gap. Pressure is
  integrated back onto the **original physical lip nodes**, not concentrated
  onto hydraulic representatives. Reports retain hydraulic-to-mechanical face
  IDs and bond representatives; input/state archives retain the face mapping.
  Tests cover reversed subset ordering, immutable/invalid IDs, dry/unbroken
  sheet exclusion, transitive/reversed bond identities, rejection of broken
  displacement ties, native connected/disconnected wet-subset equilibria and
  pressure recomputation from accepted displacement. Wet results still cannot
  supply the conservative dry-growth potential. This is a fixed explicitly
  saturated inventory, not fluid conservation through growth/reclosure.
  `hydrostatic_fracture_growth.solve_hydrostatic_contact_growth` now supplies
  an experimental **common-head** wet extension specialization of the shared
  conservative finite-candidate search. Every released face is explicitly
  saturated and must open and connect to declared reservoir edges at accepted
  equilibrium (optional exact tied front). For fixed reference geometry
  and a common head, `p=rho*g*(H-z)` is displacement independent. The event
  independently checks that pressure and reports
  `Pi=stored_material_support_mortar_energy-exterior_dead_load_work-integral(p*w)dA`.
  Reservoir work has no factor of one half. Different-head steady through-flow
  reports **no** conservative hydrostatic potential and is rejected by this
  growth entry point. The dry-only growth evidence type still rejects wet states.
  Mechanics and immutable reservoir data are jointly SHA256-bound across trials;
  reservoir provenance persists even on failure. Accepted surfaces cannot heal,
  costs use one geometric area, and a failed/closed/disconnected wet trial fails
  the search rather than becoming an arrest verdict or a silent dry retry.
  Native tests check low/high resistance release/arrest, an energy threshold
  whose decision changes if reservoir work is omitted, analytical P1 pressure
  virtual work, and extension of an existing seven-face saturated crack to eight
  faces with the previous surfaces retained. The final state is the exact
  accepted trial, not the last attempted candidate.
  Its `HydrostaticEquilibriumHydraulics` specialization applies conservative
  common-head trial pressure without a dry-opening prerequisite, permitting
  pressure-driven opening from closed contact. It then independently verifies
  final opening, source/bond identity, reservoir access and steady pressure
  before exporting an accepted state. Trial-load reports are explicitly
  unaccepted. A native compression-contact test is dry-closed but opens under
  reservoir pressure, with no initial gap or dry-open intermediate; final
  reactions/material residual/pressure work are checked. Final closed or
  disconnected states still fail, and the original different-head steady-flow
  driver retains its dry-opening requirement. This explicitly saturated
  eventual-equilibrium branch is not a prediction of fluid filling history.
  This bounded common-head model is **not** unrestricted wet propagation,
  different-head growth,
  wetting history, trapped-fluid conservation, tensile-strength nucleation,
  full-dam validation or API/WPF submission.
  Both native growth specializations now preserve an accepted-equilibrium
  evidence history and aggregate the opening warning across accepted states.
  A later smaller opening does not erase the earlier accepted maximum;
  rejected trial surfaces do not contribute. Failed/incomplete searches retain
  observed accepted peaks but leave the opening criterion unassessed.
  Continuation state IDs never imply physical seconds. Regression tests cover
  accepted reclosure, a rejected trial with a much larger opening, extension-cap
  failure, and JSON/native-adapter persistence.
  `solve_source_hydrostatic_growth` now shares canonical source preparation,
  materials, gravity, exact exterior water loads, supports and original-parent
  mortar assembly with the dry adapter. `source_reservoir_boundaries` derives
  inlet edges by intersecting candidate source edges with **inherited exterior
  reservoir tags**, using the unchanged canonical head helper. Painted markers
  are not hydraulic inlets. Whole submerged edges are required; waterline
  crossings, different connected reservoir heads, candidate edges meeting
  positive constant-pressure crest loading, and absent reservoir access fail
  explicitly before any growth trial. No reservoir head, fluid property or
  material parameter is silently reinterpreted.
  `solve_bound_hydrostatic_growth` reuses verified painting/baseline binding,
  compacted marker identities and opening limits. Direct source calls still
  report `source_binding_verified=false`; successful bound calls retain the
  verified baseline/catalog evidence. Failure retains source mapping and errors
  without a top-level accepted growth result. Native tests cover actual source
  preparation/contact/water/mortar trials and strict intact displacement parity
  (`rtol=1e-8`, `atol=1e-11 m`) with the unchanged elastic operator. Bound
  unit tests substitute baseline loading; the additional Gmsh integration
  exercises the actual baseline gate with synthetic evidence. A full saved-dam
  run is not claimed. Rejection tests cover stale fields/mesh/configuration/diagnostics and
  unsupported hydraulic connections.
  The source fixture exposed fixed rock nodes on mortar interfaces: the serial
  elastic baseline now uses the existing checked spring/Dirichlet elimination,
  with original spring forces retained in support reactions. This preserves
  prescribed fixed nodes rather than modifying their rows with raw springs.
  Tests verify exact fixed displacements, baseline parity and global force
  balance; canonical nonoverlapping-support regressions remain unchanged.
  None of these adapters enables app submission, unrestricted crack paths,
  different-head growth, partial saturation, trapped fluid or dam-scale use.
  The reduced source test now also accepts a fracture at low resistance and
  arrests at high resistance. A further native end-to-end component test writes
  an actual Gmsh source, builds the real painting catalog, resolves its actual
  element/edge IDs and face inference, verifies hashes/compaction, then runs
  canonical wet growth through accepted fracture and reopens input/hydraulic/
  source archives to check their identities. Only structure rebuilding is
  substituted in that test; catalog parsing, selection binding, preparation,
  materials/loads/supports, equilibria and growth decisions execute normally.
  This still does not establish full-dam elastic parity, mesh-independent crack
  trajectories, physical tensile nucleation or production readiness.
  The Gmsh/catalog integration now exercises the real saved-baseline gate too:
  explicitly synthetic reduced-test evidence supplies its documents; only
  structure rebuilding is substituted. Mutated completion status, force balance
  or source identities fail before numerical growth. These fixture documents
  are not independent Elmer parity evidence.

  Both dry and common-head drivers accept
  `GrowthOptions(..., maximum_patch_faces=2)` to test connected two-face jumps
  as well as single faces. The default remains one face. Every connected patch
  up to this bound that touches the current broken/marker edge front is tested
  once per growth step, including patches whose first face alone is
  energetically unfavourable. The complete accumulated cut is re-equilibrated;
  resistance charges `Gc` times all newly released area exactly once. A native
  wet specimen verifies six-to-eight-face release in one accepted jump,
  high-resistance arrest and full common-head reservoir work. An independent
  exhaustive subset test checks patch enumeration, and a synthetic energy
  barrier demonstrates one-face arrest versus favourable two-face release.
  Reports archive `candidate_surface_ids` and `maximum_patch_faces`; multi-face
  arrest is `arrested_in_tested_connected_patch_candidate_set`. The extension
  limit counts accepted patches, not faces. Failed patch solves or a favourable
  remaining patch at the extension limit cannot report arrest. Patch counts
  grow combinatorially; this bounded search is not unrestricted tracking or a
  dam-scale performance claim, and it does not exclude larger fracture jumps.
  Simultaneous patches are eventual-equilibrium energetic alternatives, not a
  prediction of dynamically reachable crack histories.
  A separate read-only actual saved-dam check completed real baseline/catalog/
  source binding on 194,316 nodes and 990,840 tetrahedra in 129.47 s, without
  substitutions. Existing baseline displacement/stress differences
  (`1.767e-7` / `5.991e-7`) and force-balance error (`7.989e-11`) passed their
  gates. That check revalidated saved evidence; it did not recompute the
  baseline, alter saved selections or run full-dam fracture propagation.

### Isolated saved-painting experiments

Exact bond condensation is now the contact solver default
(`ContactOptions.condense_bonds=True`). The intact/unbroken all-component
equality forest defines an identity quotient `u=P*q`; both stiffness and loads
are projected with `P.T*K*P` / `P.T*f`, including original support lifting.
Normal contact remains unilateral. Original bond multipliers are recovered by
subtree force balance, then the original split-space residual, bond jumps,
nonpenetration, multiplier signs and complementarity are checked. Coincident
coordinates are not welded, no penalty or residual stiffness is added, and
conflicting prescribed values on one bonded identity fail rather than being
averaged. Redundant fixed supports can admit nonunique internal force/reaction
distributions; a fixed-node-rooted forest supplies one admissible distribution.

`condense_bonds=False` retains the multiplier implementation for comparisons.
Tests compare native bonded/partial-contact displacement, forces, reactions,
work and energy against that implementation, including nonzero supports and
reordered source/DOF identities. Transitive and fully supported forests are
checked separately. The source fixture verifies that the intact quotient has
exactly the original unsplit displacement DOF count while retaining strict
baseline displacement parity. Reports retain original and condensed system
dimensions. Full split-mesh material/operator assembly is still required: this
is not a full-dam memory or performance certification.

The linear solver remains `direct` by default. An explicit
`ContactOptions(linear_solver="minres", linear_relative_tolerance=1e-13,
linear_maximum_iterations=2000, ...)` uses SciPy's symmetric MINRES on the
equivalent sign-symmetric contact system with positive diagonal mechanical/
approximate-Schur preconditioning. It avoids LU factorisation but does not
guarantee scalable conditioning. Nonpositive preconditioner diagonals,
iteration exhaustion or failed original-space audits fail explicitly, without
regularization or a direct fallback. Per-active-set system size, method and
iteration counts are reported. `linear_solver="mumps"` factorises the same
sign-symmetric system with PETSc/MUMPS symmetric LDLᵀ (petsc4py required, no
fallback). SciPy SuperLU (`direct`) fill on the full-dam painted event
(~0.6–1.1 M DOFs) exhausted the 28 GB WSL memory; on the intact 1.07 M-DOF
dam stiffness MUMPS needed ~10.6 GB and ~8 minutes (unordered by METIS).
App-submitted painted events therefore use `mumps`. The same fields can be supplied in the
standalone runner's `contact` options. Native partial-contact and wet two-face
jump tests cover this path.

**Equilibrium is not uniqueness or stability.** Contact reports explicitly
retain `displacement_uniqueness_verified=false` and `stability_verified=false`.
In the reduced frictionless released-source specimen, direct and iterative
solves can choose different zero-energy displacement modes while matching
stress, reactions, energy and normal opening. The intact state still matches
strictly. A separate semidefinite counterexample verifies that shifted
displacements can satisfy identical force/contact/energy checks. Neither
method's arbitrary tangential/rigid-mode choice is validated as a unique
physical displacement. No artificial restraint is added to hide this issue.

`python -B -m FEniCSx.painted_event_runner --help` exposes an experimental
runner, not an app solve route. Run it from Ubuntu/WSL with the Gairezi package
root on `PYTHONPATH`. It accepts an existing app-format saved-selection JSON,
explicit source job ID, configuration, source model, verified baseline and
baseline field paths. Its shared snapshot reader is also used by the API:
snapshot digest, job/configuration identities, selection ID/filename and
preview-only schema/status must match before the real catalog/source/baseline
gate is evaluated. Raw painting requests are not accepted in place of saved
snapshots; the SHA256 check detects modification, not cryptographic authorship.

The output directory must be new. It retains immutable configuration, selection
and numerical-option snapshots, environment/provenance, failure diagnostics and
an accepted-results index only after successful native growth. Failed or
unsupported hydraulics never fall back to dry growth. Existing saved selections
and source jobs are not modified. The source-cell ceiling is explicitly supplied
by the experimenter and checked before candidate splitting; it is not evidence
that a mesh below that ceiling is computationally feasible.

Add `--prepare-only` to check identities and the unsplit native candidate
inventory without duplicating wall nodes or running any equilibrium/growth
solve. Its successful status is `prepared_experimental_not_solved`, never a
completed analysis. Native facet-to-cell connectivity determines the same
ordered wall candidates as the independent cell-owner inventory, without
building the latter's Python dictionary for every cell face. Candidate owner
orientation, concrete-only membership and source identities remain checked.
The same vectorised exterior-edge inlet proof runs on the unsplit and split
mesh. Inapplicable hydraulics fail before candidate splitting; `preflight.json`
retains candidate counts and all detected reservoir incompatibilities even on
failure. Manifests include elapsed time and explicit failure timestamps.

The actual saved-dam diagnostic passed dry preparation on 194,316 nodes /
990,840 tetrahedra, finding 1,177,580 candidate faces and a conservative
7,859,916 upper bound on prepared displacement DOFs. This is a count, not a
memory estimate or direct-solver feasibility result. The wet diagnostic
completed its expected rejection in 73.70 s (10.90 s for native inventory and
hydraulic screening after source binding): the real candidate set meets
positive constant-pressure crest loading **and differing reservoir heads**.
Both reasons are recorded; no split mesh, growth directory or accepted-result
index is created. The diagnostic used an isolated test selection, did not
alter user snapshots and did not rerun the baseline. This real dam cannot
currently be submitted to the common-head-only growth model.

Example numerical options (save as an experiment's `solver_options.json`):

```json
{
  "maximum_source_cells": 1000,
  "maximum_volume_change": 1e-12,
  "event": {
    "relative_tolerance": 1e-8,
    "absolute_tolerance_n": 1e-7,
    "maximum_iterations": 100
  },
  "contact": {
    "gap_tolerance_m": 1e-12,
    "force_tolerance_n": 1e-7,
    "relative_residual_tolerance": 1e-8,
    "maximum_iterations": 100
  },
  "growth": {
    "fracture_energy_j_m2": 2700.0,
    "energy_tolerance_j": 1e-7,
    "maximum_extensions": 10,
    "maximum_patch_faces": 2
  },
  "hydraulics": {
    "closure_tolerance_m": 1e-12,
    "flow_relative_tolerance": 1e-10
  }
}
```

These are explicit experimental numerical controls, not production calibration.
The runner records both requested fracture energy and any configured
`fracture_debug.critical_energy_release_rate_j_m2` without changing either.
It does not import the old AT2 residual stiffness, length scale or seed law.
For explicitly dry new surfaces use `--water-mode dry` and omit `hydraulics`
from the numerical options. Original exterior water/gravity loads are retained.

Ubuntu example, with each variable set to the corresponding existing input:

```bash
python3 -B -m FEniCSx.painted_event_runner --experimental \
  --configuration "$CONFIGURATION" --model-directory "$MODEL" \
  --baseline-directory "$BASELINE" --fields "$FIELDS" \
  --saved-selection "$SAVED_SELECTION" --job-id "$JOB_ID" \
  --options "$OPTIONS" --water-mode common_head_hydrostatic --output "$NEW_OUTPUT"
```

Successful output has status `completed_experimental_not_engineering_validated`
and relative paths to the accepted growth/event reports, displacement and
material PVD files, and native state archive. The root manifest retains measured
opening warnings and exact accepted surface IDs; continuation states are not
physical seconds. Failure exits nonzero and records
`failed_experimental_painted_event` without an accepted-results index.
The reduced real-Gmsh integration runs both dry and wet paths through native
solves and checks those files; only structure rebuilding and synthetic baseline
evidence remain substituted. General different-head/trapped-water behaviour,
strength-consistent unrestricted paths and dam-scale solve validation remain
unsupported.

The WPF painter's **Submit** saves the selection and then calls the
experimental `POST /jobs/{id}/painted-event` API stage. That stage launches this
runner through WSL in `painted_events/<run>/` in the job directory. The runner
and the shared growth drivers print flushed `Progress: <phase>` lines to
`runner.log` (selection check, binding, preflight, candidate preparation,
each growth-step candidate, arrest, surface output); the app's activity monitor
shows the latest one. It writes
the numerical options there, with fracture energy taken from
`fracture_debug.critical_energy_release_rate_j_m2`, and accepts results only
from a manifest that matches the job, selection and water mode. New app calls
require `common_head_hydrostatic`; dry mode remains a standalone verification
option only. Failure never falls back to dry or legacy cracking.

The sequence is an intact gravity/exterior-water equilibrium, selected-cell
tensile-strength reduction at unchanged exterior loads, then irreversible
finite-candidate cut trials with pressure on reservoir-connected faces.
The inventory can include different reservoir heads without treating all
potential faces as already cracked. Only released-cut inlet connections choose
the hydraulic head. Horizontal crest pressure is converted to head using its
actual elevation and pressure gradient, not a second arbitrary load.
`reservoir_crack_load.py` integrates pressure and its work exactly on submerged
triangle portions, including waterline crossings, and checks final open
submerged access. Isolated cuts are explicitly unpressurised; no pressure or
initial aperture is invented. An actual cut meeting different heads still
fails: different-head dissipative flow needs a separate growth law.
The model assumes instantaneous drained equilibrium, not transient filling,
trapped fluid or evolving free-surface flow.

Trials reuse the initial accepted equilibrium only with an identical hashed
mechanical problem, the same native function space and monotone cut release.
They still solve and independently audit each candidate's equilibrium, contact
and reservoir work. A full saved Gairezi selection has completed this path
with exterior water loads retained and selected strength 0.505 MPa, but accepted
no extension in its two tested single-face candidates. Subsequent kinematic
checks found that single-face releases can remain completely tied by
neighbouring bonds. Historical completion/arrest in such a search is **not**
evidence of physical arrest.

Native growth now excludes patches that do not split any bonded displacement
identity. If all proposed extensions are locked this way, the run fails with
`failure_kind=no_kinematically_admissible_extension`, records the excluded
patches, and does not issue an arrest verdict. The initial candidate set is
screened before any intact/selected-material equilibrium solve. A wholly
locked initial front fails during preparation, with
`initial_equilibrium_started=false`, no accepted equilibrium and no initial
solve output. Native dry/wet tests retain this failure check when resolution
is disabled.

New painted app runs enable `GrowthOptions.resolve_locked_patches` (standalone
default: false). `NodeStarPatchResolver` in `fracture_topology.py` constructs
opening-capable candidate disks by minimum-cardinality cuts of the remaining
bond graph at each seed vertex. This generates at most three completions per
seed without an exponential connected-subset search, then removes any
completion that strictly contains another opening-capable completion of that
seed. These are irreducible first-opening increments, not an unrestricted
finite-jump search. Larger redundant supersets must be requested through the
explicit connected-patch bound rather than introduced by the resolution step.
It preserves the
seed/front, parent identities and every unreleased bond; it changes no mesh
geometry, material, cell count or initial gap. Disk connectivity, manifold
edges and the increase in independent bonded identities are checked before
solving. Cardinality only proposes a resolved discretisation: **all** added
faces pay their actual geometric area times `Gc`, and only a lower total
energy at a verified contact/hydraulic equilibrium can accept them.
`maximum_patch_faces` still bounds the explicit exhaustive search, not these
recorded node-star completions. Reports retain the generated source facet IDs
and use `arrested_in_tested_node_star_patch_candidate_set`; this is limited to
that finite candidate set, not proof of unrestricted physical arrest.

Released-sheet hydraulic adjacency uses both lip identities, invariant under
owner reversal. Only topologically inlet-connected released faces are loaded;
new disconnected sheets remain explicitly dry. The fixed-head clipped loader
handles fully and partially submerged surfaces consistently, including contact
at individual vertices when the final nonpenetrating state still has an open
submerged reservoir path. Closed access, ambiguous inlets, different active
heads and unsupported junctions still fail explicitly.

For the current convex tensile-cap material and frictionless contact, loading
every topologically connected candidate face with the nonnegative reservoir
pressure gives an **optimistic lower bound** on the drained potential:
`min(U-Wext-Wfully_pressurised) <= min(U-Wext-Wphysically_accessible)`,
because normal gaps are nonnegative. A mechanically converged, force/contact-
audited pressure trial whose final water-access audit fails is never accepted
as a wet equilibrium. Its explicitly recorded bound may reject a patch only
when even that optimistic energy release cannot pay `Gc * added_area`.
Such rows are labelled `rejected_by_optimistic_pressure_energy_bound` and
contain upper/lower bounds, not an accepted energy release or state.
The failed trial retains `unaccepted_pressure_bound_state.npz` and its audit,
but no accepted post-event or hydraulic state. If the bound could favour
growth, the search still fails as unresolved; neither failed numerical solves
nor unsupported heads acquire this rejection certificate. This argument
depends on the present convex material/contact problem and cannot be carried
over to a future nonconvex softening law without a new bound.

Native tests start with a locked single facet, test real positive-opening wet
patch equilibria, accept two irreversible extensions at low synthetic `Gc`,
reject high-cost cuts and preserve prior cuts when `Gc` increases. They audit
all geometric surface costs, reservoir work, force balance and nonpenetration.
The synthetic distributed supports keep the benchmark in the small-displacement
range; they are not added to the dam. This fixes single-face kinematic locking,
not the reversible tensile-cap law, strength-calibrated nucleation,
mesh-independent propagation, crushing or engineering safety validation.

A saved three-body dam verification now completes in an isolated output
directory (351.9 s): its previously locked seed resolves into two irreducible
two-face trials. At the snapshot's selected strength of 2 MPa, the optimistic
energy-release bounds are 0.0485 J and 4.6311 J, against surface costs of
2172.78 J and 2572.35 J respectively. Both are rejected by the audited bound;
no new crack or positive opening is claimed. This verifies completion of the
finite-patch workflow, not unrestricted crack arrest or material calibration.
Earlier saved job records are unchanged.

`test_selected_tensile_capacity.py` checks the native material/contact
equilibrium under prescribed tractions: a fully selected specimen with
0.5 MPa tensile capacity stays elastic at 0.4 MPa tension, cannot return an
accepted equilibrium at 1.2 MPa tension, and retains compressive compliance
at 1.2 MPa compression. Failure of an overloaded capped specimen is not a
simulation of crushing or proof of irreversible damage. The existing
reversible tensile cap still does not implement that damage law.

The Configuration / Material slider (zero to concrete tensile strength) sets
`material.concrete.selected_tensile_strength_pa` (MPa in the UI, Pa in JSON).
When omitted it inherits `material.concrete.tensile_strength_pa`. Only the
painted wall cells use the chosen value after the intact solve. The local law
minimises `0.5*(eps-eta):C:(eps-eta) + ft*tr(eta)` over positive-semidefinite
`eta`: original elasticity below the tensile cap `ft`, relaxed extension at
the cap, and unchanged compressive compliance. Zero recovers the original
no-tension law. This reversible experimental model is **not** irreversible
softening or strength-calibrated nucleation; discrete surface growth still
uses the finite-candidate energy test and configured fracture energy.
Native event archives and the run manifest record the actual selected
strength. Existing job snapshots/results are not rewritten; changing the
slider requires a new job/baseline. Hydrostatic pressure is applied only to
reservoir-connected released crack faces, not invented on closed bonded
interior faces.

The accepted-results index also includes `crack_surface.vtu` for ParaView.
It preserves both physical lips and their split-node/original-parent IDs, exact
mechanical face IDs, released/wet flags, measured signed normal gap and
nonnegative opening, tangential jump and accepted vertex water pressure.
Reference coordinates are undeformed: apply **Warp By Vector** using
`displacement_m`, then threshold cell `released=1` to view only actual cuts.
Bonded candidates are included for identity checking, not counted as fracture.
Zero pressure on declared dry faces is distinguished by `wet=0`. Neither
relaxed material strain nor a seeded displacement is used as crack opening.
VTU round-trip tests check units/data order, both lips, pressure, shear and
negative signed penetration separately; native dry/wet runs reopen the actual
accepted surface and verify source parents, wet/released IDs and peak opening.

- `no_tension.py`: minimises the original isotropic elastic energy over an
  added positive-semidefinite relaxed strain. Stress is negative-semidefinite;
  elastic compressive stress states retain the original compliance. There is
  no artificial residual tensile stiffness or prescribed initial gap. The
  analytical active-set tangent is checked against energy/stress derivatives,
  rotations, repeated eigenvalues and an independent constrained minimisation
  oracle. All numerical material values in tests are synthetic verification
  inputs, not new dam defaults.
- `no_tension_fem.py`: local-cell DG0 stress/tangent coefficients for affine P1
  tetrahedra on DOLFINx 0.11.0.post0. Only selected cells use the no-tension law;
  other cells remain isotropic elastic. Its boolean mask is **local cell order,
  including ghosts**, not saved catalog IDs. Native tests verify selection
  isolation, retained compression and independently checked equilibrium after
  imposed loss in mixed-loading patches on two mesh sizes.
  Young's modulus and Poisson ratio accept scalars, local-cell arrays or
  same-mesh scalar DG0 functions. Each material pair reuses the same verified
  constitutive law. Values are snapshotted; editing a supplied field later
  cannot silently change the event. Mixed concrete/rock tests use the existing
  `three_body_model.materials` helper and check analytical layered compliance.
- `tensile_loss_event.py` implements the reduced **intact loaded baseline ->
  imposed selected-cell tensile loss -> post-event equilibrium** sequence.
  The external UFL load functional and boundary conditions are assembled once
  and reused unchanged. The event starts from the actual intact displacement;
  no gap, density change, load ramp or residual tensile stiffness is introduced.
  Native force-controlled tests on two meshes show tensile-stress release in
  interior selected cells and redistribution into intact material, retained
  compression compliance and independently checked reactions/free residual.
  Removing every tensile load path must fail; it cannot report a converged
  equilibrium or opening verdict.
  This driver supports serial affine P1 tetrahedra, cellwise isotropic materials
  and direct MUMPS. It accepts the existing `InterfaceSprings` mortar operator
  on the same displacement space and an optional unchanged UFL linear-support
  form, including `three_body_model.support_form` roller penalties. Both are
  included in the intact solve, post-event residual/tangent and independent
  reaction audit. Tests check analytical bulk-plus-interface compliance,
  unchanged mortar coefficients during tensile loss and distributed support
  force/energy. Serial Dirichlet-constrained mortar DOFs now use eliminated
  rows/columns and the original spring operator's imposed-displacement lifting.
  Raw spring forces are retained for independent reactions/energy; coefficients
  are never modified. Analytical zero/nonzero imposed-displacement tests and a
  nonlinear tensile-release test check constrained values and equilibrium.
  This elimination is opt-in in `SpringLinearProblem`; the event enables it
  without changing existing standalone solver defaults. Load/support boundary
  forms must share the same facet tags. A synthetic pressure-loaded patch
  also checks unchanged gravity, reservoir and tailwater force resultants
  after actual tensile-stress release; nonuniform compressive pressure need
  not leave every intact principal stress compressive.
  **These are reduced checks, not a production dam adapter or coupled crack
  growth.** No full-dam resource or engineering verification is implied.
  `EventSolverOptions` supplies explicit convergence tolerances/iteration bound.
  A requested output directory must be new: the driver retains a status report,
  Newton trace, native mesh/selection/load archive and accepted intact state even
  when the event fails. Success adds post-event arrays and ParaView VTK material/
  displacement series. Output states 0/1 identify intact/post-event states;
  they are not physical hydraulic times. Archive selections are local native
  cell order, not original catalog IDs. Archives include cellwise elastic
  properties and, when supplied, the actual mortar CSR operator and native
  displacement cell DOFs. No failed run emits accepted post-event
  fields. `solver_ready` and `engineering_validated` remain false.
- `solve_source_event` in the same module now imports a supplied canonical
  multi-body structure itself and maps selected **source volume rows** through
  subdivision/native reordering to wall-only local cells. It reuses canonical
  material, interface and outer-support helpers, and the same `external_load`
  helper as `dam_elasticity.solve` for gravity, reservoir, tailwater and top
  pressure. A native baseline comparison checks source-order displacement
  agreement to `rtol=1e-8, atol=1e-11 m` and independent analytical force
  integrals. Separate two/three-body checks preserve bedrock/foundation
  properties and both foundation interfaces. Native input archives retain
  source rows, including on post-event failure.
  This experimental Python entry point requires a new persistent output
  directory, one MPI rank and explicit import/solver tolerances. It does not
  accept or authenticate a saved app request, construct cracks from selected
  edges or enable submission. Source rows must refer to the supplied structure;
  saved catalog/configuration/baseline hash binding is still required.
  Exterior water forces now use `hydrostatic_load.py` to clip each triangle
  at the actual water elevation, integrate pressure times the **original**
  P1 shape functions exactly on the submerged polygon, and assemble inward
  forces by native cell/facet connectivity. Partly submerged triangles are
  supported without changing the mesh or initial gap. Analytical checks verify
  force, moment, affine virtual work, one/two submerged vertices, orientation,
  translation, scaling and subdivision invariance. Opposing reservoir/tailwater
  faces and constant top pressure remain independent.
  The resulting fixed nodal loads enter both linear/nonlinear states,
  constrained-node reactions, the independent equilibrium audit and archives.
  This avoids ordinary positive-part quadrature errors observed on coarse
  waterline-crossing facets. It is exterior loading, **not pressure in an
  evolving crack**. Existing standalone elastic baseline numerical behavior
  remains unchanged; equality to that solver is tested with aligned waterlines.
- `solve_bound_event` is an additional experimental Python mechanics entry
  point. It rebuilds the structure from the submitted configuration and saved
  model, requires the existing passed baseline parity/equilibrium gate, rebuilds
  the catalog and validates the request against current source/configuration/
  baseline-field hashes. `bind_source_selection` checks source tetrahedron
  row order, regions and exact compacted node connectivity/coordinates without
  merging coincident identities. Source/configuration/diagnostic hashes are
  rechecked before native preparation. Foreign/stale inputs fail before solve.
  Binding and verified-baseline provenance, resolved faces, selected edges and
  the opening limit are retained in the event report on success/failure.
  **Edges remain markers only**: no initiation/growth law or physical opening
  assessment is claimed, and this entry point is not exposed through the API.
  Pure tests include an actual Gmsh-file binding and parser-change/stale-ID
  rejection. Native orchestration tests substitute baseline/structure loading
  and then run actual reduced mechanics to check provenance persistence;
  they do not constitute a full saved-dam end-to-end verification.
- `dam_mesh.py` supplies serial source-row provenance tags and a checked
  source-selection-to-local-wall mask. Initial labels compose subdivision
  parents with native `original_cell_index`; refinement must transfer the tags,
  never reuse a refined local cell index as a saved source ID. Tests retain
  selected volume through two native refinement levels, reject foreign/non-wall
  selections and demonstrate that coincident but disconnected concrete retains
  separate IDs and material response. Catalog/source hash binding is still the
  production caller's responsibility; this is not yet a dam-submission path.
- This local law is reversible relaxation, not irreversible crack growth.
  Relaxed strain is not a displacement jump or crack aperture. It does not
  supply friction, compression crushing, a tracked fracture surface, fracture
  energy, propagation or hydraulic tractions. No full dam safety conclusion
  follows from these checks.
- `fracture_flow.py`: reduced 1-D saturated flow along an explicitly prescribed,
  uniformly open fracture, with cubic-law conductivity, gravity, linearised
  fluid storage, aperture-rate source and backward-Euler stepping. Viscosity,
  compressibility, density, physical times, aperture and boundary/initial
  pressures must be supplied explicitly. There are no physical defaults.
  Independent residual and storage/flow balance checks run at every step.
  Tests cover analytical Poiseuille flux, cubic aperture dependence, no-flow
  hydrostatic equilibrium, transient analytical decay with mesh/time refinement,
  and pressure/flow caused by a prescribed aperture change.
- `solve_compliant_fracture` in the same module additionally eliminates a local
  linear normal-mechanics relation, `w = w_ref(t) + C * (p - p_ref)`. The explicitly
  supplied compliance `C` has units m/Pa. Pressure calculates opening; opening
  changes fluid storage and cubic conductivity and therefore feeds back on
  pressure. A native monolithic Newton solve includes both derivatives. This is
  a reduced, **already-wet open path**, not a 3-D dam contact/propagation model.
  Tests check nonlinear steady pressure/flux against the integrated cubic-law
  solution, pressure interpolation convergence inside elements, small-signal
  transient decay including mechanical storage, and the stiff-wall limit.
  Accepted steps report solver iterations/reason and independently assembled
  residual/continuity errors; exhausted iterations fail rather than returning
  a success-shaped history. `maximum_iterations` is an explicit numerical input.
- `contact_benchmark.solve(..., face_pressures_pa=...)` optionally applies
  externally prescribed equal/opposite opening tractions to the native two-bar
  split interface. Tests start with zero geometric gap under compression, release
  contact with increasing pressure, calculate face displacement jumps from bar
  compliance and reclose on unloading. Contact pressure, bulk stress, outer
  reactions and complementarity are checked independently. The default dry
  benchmark and CLI inputs are unchanged. This optional traction history does
  not calculate water access or transport through closed contact; it is not
  coupled to the compliant-flow solver or a production dam stage.
- The same native contact benchmark optionally accepts tangential displacement
  histories, an explicit friction coefficient and an explicit shear modulus.
  Scalar Coulomb return mapping resolves stick, sliding and reversal, using
  **solid contact force only** for friction strength. Water-induced contact
  release therefore reduces the friction limit; an open interface transmits
  no shear. Tests verify analytical forces/slip/dissipation, unloading/reversal,
  explicit zero-friction behavior and exported tangential/shear-stress fields.
  Nonnegative dissipation, bulk shear equilibrium, reactions and yielding are
  checked independently. This is a reduced scalar slip model, not a 3-D
  frictional contact search or unrestricted mixed-mode fracture formulation.
- `fracture_topology.py` splits **explicitly supplied** interior wall facets.
  Connectivity within each affected node's cell star, across uncut faces,
  determines which corners remain tied. It retains shared crack-front nodes,
  original cell rows/regions and parent-node provenance; no coordinate merging,
  material deletion or initial gap occurs. Transferring an accepted nodal state
  duplicates the state with zero initial displacement jump. Non-wall/boundary/
  nonmanifold cuts, overlapping owners and isolated fully tied P1 cuts fail
  explicitly rather than creating a fictitious opening.
  Exterior boundary triangles/tags transfer by owning-cell connectivity,
  preserving support/load identities and orientation; new crack lips never
  inherit a reservoir label automatically. Native integration starts from a
  bonded 3-D two-block mesh, cuts the specified interface, then verifies pressure
  opening, compressive stress and independent equilibrium on the created faces.
  This is a serial, unbenchmarked-at-dam-scale topology component, **not an
  initiation criterion, selected-edge-to-surface rule or propagation solver**.
- `fracture_faces.py` measures actual small-displacement opening on explicitly
  paired affine P1 triangular crack lips. Corresponding source-node identities
  must be initially coincident and distinct away from shared crack-front nodes;
  normals are explicit and oriented
  from minus to plus. It reports the positive maximum normal jump and its source
  location, **also retaining signed penetration and tangential jumps**. P1
  normal-gap extrema occur at triangle vertices; no damage-to-opening conversion
  is used. Source-vector extraction uses native connectivity/input IDs rather
  than coordinate matching and rejects discontinuous, higher-order or nonserial
  fields.
- The same face component integrates prescribed P1 water pressure into exact
  consistent, equal/opposite nodal loads. Tests verify force/moment balance and
  pressure/displacement virtual work. A native 3-D two-block elastic solve with
  independent coincident lips, zero initial gap and applied face loads checks
  the analytical opening `p*(L_minus+L_plus)/E`, compressive bulk stress, free
  equilibrium and balanced reactions. This verifies extraction/loading only:
  paired surfaces must already exist, and pressure access/wetting and surface
  creation/propagation are not supplied.
- `assess_opening_history` accepts measured opening observations without the
  superseded horizontal-request geometry. Painted-history tests retain the
  peak after reclosure, produce a warning at/above the supplied limit and never
  treat an incomplete solve as satisfying the criterion. The existing
  horizontal contract is retained as a compatibility wrapper.
- `fracture_water_access.py` checks geometric reachability from explicit
  exterior hydraulic inlet edges, using paired lip identities and measured
  P1 normal gaps. It distinguishes reachable open faces, isolated open faces,
  closed faces and blocked inlets. A point contact, a closed connecting edge
  or a coincident unrelated sheet cannot carry an access path. Either owner's
  face orientation is supported. Input penetration beyond the explicit numerical
  closure tolerance fails instead of being clipped. The tolerance only gates
  connectivity; no minimum physical aperture or permeability is inserted.
  Integration checks use a topology-created surface with tied crack-front
  vertices: its accepted loaded state has no access before opening.
  **Reachability is not saturation or hydrostatic pressure.** The function
  assigns no pressure, removes no trapped water and solves no wetting front.
  Inlet edges must come from verified hydraulic boundary geometry; a user's
  crack-start edge is not automatically an inlet. Multiple reservoir heads,
  trapped fluid, filling time and new wetting require a conserved-fluid model.
- No moving wetting front, closed-contact flow, cavitation, branching or
  fracture propagation is implemented. Closed apertures and negative-pressure
  states needing those models fail explicitly, rather than receiving guessed
  minimum apertures or pressure clipping. Linearised fluid storage is checked
  against its stated equation, not against a nonlinear density/large-volume
  mass-conservation model.

Run reduced tests **in WSL Ubuntu**, from this directory:

```bash
PYTHONPATH=.. timeout --signal=TERM --kill-after=10s 240s python3 -B -m unittest test_dam_mesh test_no_tension_fem test_tensile_loss_event test_hydrostatic_load test_dam_elasticity test_three_body test_fracture_flow test_compliant_fracture_flow test_contact_benchmark test_fracture_faces test_fracture_topology test_fracture_water_access test_fracture_hydrostatic test_fracture_steady_flow test_compliant_steady_crack test_open_crack_equilibrium test_paired_contact test_dry_fracture_growth
cd ..
timeout --signal=TERM --kill-after=10s 120s python3 -B -m unittest FEniCSx.test_no_tension
```

The independent minimisation oracle needs SciPy, available in the verified WSL
runtime; the Windows application interpreter need not contain it.
The production API remains `painted_damage_solver_ready: false`. Selection
snapshots can be saved/restored, but production provenance wiring, dam mechanics,
edge-to-surface construction, dam opening extraction, friction and evolving
wetting/contact/pressure coupling require further implementation and physical
inputs. In particular, existing `crack_water_pressure_factor` is a legacy load
multiplier, not fluid viscosity/compressibility or a physical time history;
none is inferred from it. Code_Aster parity is not established by these checks.

For the remaining 3-D solver work, prefer maintained native implementations
over treating the reduced components as a complete fracture solver.
[dolfinx-contact](https://github.com/Wells-Group/asimov-contact) currently
declares DOLFINx 0.11.0 support but explicitly describes itself as highly
experimental; its availability alone does not establish hydraulic fracture or
concrete calibration. The [NewFrac examples](https://newfrac.github.io/fenicsx-fracture/)
also declare DOLFINx 0.11.0 and are computational fracture examples, not evidence
of a complete coupled wetting/frictional dam adapter. Neither is installed or
introduced as a project dependency by these changes.

This directory contains the DOLFINx 0.11 mechanics and phase-field adapter. The authoritative geometry and material/load input is Gairezi/config.json; the adapter does not maintain a second dam configuration. Elmer?s three_body.msh and diagnostic.json are the canonical solver-mesh contract when using the saved source mesh.

## Canonical regions and connection

The canonical mesh has wall = 1 and bedrock = 2, plus concrete foundation = 3 when `geometry.foundation` is configured. Without a foundation the wall sits directly on surveyed bedrock. With a foundation, the exact saved Elmer wall/foundation/rock geometry is preserved, including the excavated staircase seat. Each interface is either spring-connected through duplicated nodes or bonded through conforming shared-node faces, as selected independently in configuration. The wall is the only damageable region; bedrock and foundation remain intact, even though foundation uses the wall concrete material. The legacy module names three_body_model.py, three_body.msh, and three_body_geometry.pvtu are retained for compatibility.

The configured bedrock support model controls the wall/bedrock seat: bonded meshes use shared-node faces, while spring-supported seats use the opposing boundary tags 11 and 12 and the configured x/y/z stiffnesses. Rock-bottom support uses tag 10; outer rock boundaries use tags 13?16. Elastic reports use wall_connection and wall_bedrock_connection. Wall headline stress peaks remain wall-only; bedrock and optional foundation peaks are reported separately in `body_stress`.

For a foundation source, the mesh importer verifies dimensions, body IDs, interface connectivity, spring tags and stiffness against the submitted configuration and diagnostic. `geometry.foundation.support` is required and never inherits wall-joint settings. Spring-connected `wall_foundation` uses `geometry.foundation.support.spring_n_per_m3`; `foundation_bedrock` uses `geometry.bedrock.support.spring_n_per_m3`. Spring-connected interfaces require their opposing tagged faces and duplicated nodes. Bonded interfaces require matching exposed faces with shared nodes, and their spring coefficients are inactive. Legacy spring sources use tags 17/18 and 11/12 respectively. Exact-plane spring sources declare nested `spring_pairs[logical].faces[plane]` semantic tag pairs, resolved through `boundary_id_map`; the first plane retains the legacy tags and further planes use tags >=1000. The importer checks unique tags, configured logical coefficients, body membership, triangle counts, plane metadata and coplanarity. Each plane multiplier is eliminated independently before summing operators, including at shared riser/tread nodes; pooling plane projectors before elimination is a different discretisation. Reports retain logical interface names, with per-plane details under `faces` and `grouping: exact_plane`. Wall thickness-block joint pairs are retained. Reports preserve foundation metadata and set `wall_bedrock_connection` to `not_in_contact`; the two configured foundation connections are reported independently. Exposed untagged foundation faces are traction-free, as for bedrock. Without a foundation, the existing wall/bedrock support and damage behavior is unchanged.

## Near-planar chamfer interfaces

Foundation sources may declare `grouping: near_planar`, positive `tolerance_m`
and `max_angle_deg`, and `continuity: edge_connected; single orientation class`.
Each group's `planes` entry declares its fitted normal/offset, orientation class,
maximum off-plane distance and pairwise normal spread. The importer recomputes
these from both source mesh sides, checks edge connectivity, one orientation class,
coincident triangles, configured coefficients, body memberships and unique tags.
Fits use Elmer's unique 8-decimal coordinate keys; geometry checks allow only
serialization roundoff (1e-8 m), not extra physical grouping tolerance.

The group fit plane is bookkeeping, not a replacement integration surface.
FEniCSx integrates each actual slave triangle against its coincident master
triangle, preserving curved surface area and the exported Elmer `[M,-M]` mass.
It eliminates the multiplier independently for each declared group, then sums.
Exact-plane and legacy interface numerics remain unchanged. Reports retain logical
names and carry grouping limits/continuity plus `source_plane` and semantic `tags`
per group. The API checks these against the job diagnostic, including refined
baseline reports; WPF does not parse spring-pair inventories.

`audit_interface_projectors.py --model MODEL --projectors EXPORTS --output REPORT`
reuses production block assembly and elimination to compare each group with the
Elmer projector exports (run in WSL with the same 520 s agent-run cap). This is an
operator audit only, not an elastic equilibrium solve or a verified fracture
baseline. The 2026-10-22 audit of all 783 `foundation_chamfered_grouped` foundation
groups (280 wall/foundation, 503 foundation/bedrock) passed: maximum relative
Frobenius projector difference `2.8501584e-14`, eliminated operator difference
`3.9532666e-14`. Evidence is in
`results/near_planar_integration/projector_comparison.json`.
`foundation_chamfered_grouped` has no converged Elmer reference, so full
elastic parity is unavailable. App jobs retain no implicit runtime cap; a nonzero
Elmer worker exit fails the job before FEniCSx starts, even if leftover result
files exist. Existing displacement/stress/equilibrium gates are unchanged.

## Staircase-reference parity caveat

The wall stress gate is unchanged at `1e-4`. Its metric is the unweighted Frobenius
norm of the difference of the full symmetric 3x3 recovered nodal tensors, divided
by the corresponding Elmer tensor norm, over each wall P1 node once. Off-diagonal
components therefore occur twice; foundation and bedrock nodes are excluded.

The saved `Thickness_tests/foundation_stepped_benched` Elmer 26.2 reference has a
weighted-projector point-search artifact at angled staircase riser/tread corners.
A Gaussian point on a tread can pass an adjacent riser's axis-aligned bounding-box
and projected-barycentric tests despite being off its plane. The exported Elmer
wall/foundation projector has the same slave mass as the exact matching-triangle
operator, but a different master coupling. FEniCSx deliberately requires coplanar
triangle intersections; it must not reproduce this off-plane mapping to pass a gate.

Diagnosis on 2026-10-22 found wall stress error `1.0913933e-4`, versus `7.7378759e-7`
for the pre-foundation `stepped_blocks_springs` case. Recovering stress from the
*same Elmer displacement* agrees to `1.8776440e-9`, excluding stress averaging as
the cause. A diagnostic-only FEniCSx replay of the exported reference projectors
reduces stress error to `5.3500442e-7`. This replay is not the production operator
and is not an approved fracture baseline. That historical reference still fails;
its off-plane search artifact must not be copied into the production operator.

The corrected `Thickness_tests/foundation_stepped_split` reference groups both
foundation interfaces by exact plane (174 wall/foundation and 849 foundation/rock
subpairs). Its exported projectors agree with exact integration to relative
Frobenius error <= `1.954084e-14`; the volume mesh is unchanged. Production FEniCSx
parity passes with displacement error `1.1998608e-7`, wall nodal stress error
`4.6972060e-7`, and relative force imbalance `8.0311953e-11`. Results are saved under
`results/foundation_split_parity/dam_elastic_baseline/20261022T104209944357Z`.
No gate or physical input was changed.

## Running

Run commands from WSL Ubuntu, where DOLFINx 0.11 and the solver dependencies are installed. The saved Elmer model directory must contain the matching config.json, diagnostic.json and three_body.msh. A phase-field fracture run requires a matching completed elastic baseline, the Elmer result used for the user-selected crack location, and the canonical model directory. Do not treat convergence or successful execution as engineering validation or evidence of dam safety.

The test suite is run with:

    python3 -B -m unittest discover -p 'test_*.py'

Small mechanics, mesh-import, crack-placement and report tests are in the test_*.py modules. Tests requiring an external canonical Elmer result are skipped when that result is not available.
