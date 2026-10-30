# Dam Analysis Workbench - implementation plan

## Automatic whole-wall cracking: formulation gate

The planned replacement for manual painting is a read-only whole-wall assessment
followed by explicit submission. It is not a drop of whole-element Young's
modulus to 1,000 Pa. The proposed cohesive/discrete approach must preserve
compressive contact and include verified water pressure on evolving,
reservoir-connected cracks before automatic app submission can be enabled.
Existing painted submissions and historical results are not changed by this
formulation work.

`FEniCSx/cohesive_mode_i.py` supplies the experimental, already-initiated
Mode-I interface law for specimen verification: linear traction-opening
softening with critical opening `2*G_f/f_t`, secant unloading/reloading,
irreversible maximum-opening history and explicit stored/dissipated energy.
Trials return new history; only an audited equilibrium may commit it.
Do not charge the legacy `Gc*area` fracture cost again for this cohesive process.
This module alone does not implement intact bonds, stress-driven initiation,
topology insertion, compression contact, tangential transfer, equilibrium or
hydraulic access (see `cohesive_contact.py` below). Its unit tests do not verify coupled crack propagation.

The proposed tensile fracture energy is user-supplied 400 J/m2; no traceable
test/material source has yet been supplied, so it remains independently
unverified. The existing `fracture_debug.critical_energy_release_rate_j_m2`
must not silently be treated as this separate cohesive material property.
The 400 J/m2 and 2 MPa unit-test values give a 0.0004 m terminal softening
opening, not an accepted dam calibration or the existing 0.002 m opening
warning threshold. It is now the optional configuration field
`material.concrete.cohesive_fracture_energy_j_m2` (validated positive in
Python and WPF, editable in the Configuration view). Historical configurations
without it remain valid and are never given a guessed value; automatic
screening reports it as absent and cannot be confirmed for them.
Softening invalidates assumptions behind the current convex hydraulic-bound
certificate.

`FEniCSx/cohesive_contact.py` adds the experimental coupled specimen solver:
uninitiated paired vertices are exactly tied, an audited initiation releases a
pair whose recovered normal tie traction reaches `f_t` to the cohesive law,
initiated pairs carry cohesive traction in opening and frictionless unilateral
contact in closure (closure never heals history), and reservoir water is
applied to cracks connected to the reservoir. Its specimen tests pass; they do
not verify the real dam. Paired nodes must belong to exactly one pair;
`FEniCSx/cohesive_network.py` adds the node-junction network used for
whole-wall facet growth.

`FEniCSx/automatic_screening.py` is the read-only first step of the automatic
workflow. It solves the job's intact native equilibrium
(`tensile_loss_event.solve_source_event(intact_only=True)`), computes each wall
cell's maximum principal stress and lists cells above
`f_t*(1+relative_tolerance)` with their volume, peak, location and direction,
plus the near-critical count. It is an initial-state screening, not a crack
prediction: redistribution after cracking can extend or arrest damage. The API
exposes it at `/jobs/{id}/automatic-cracking/screening` (start/read/cancel) and
`/confirmation` (immutable record of the exact run, its configuration digest,
baseline and Gf). The desktop crack-definition window defaults to this
automatic mode, highlighting overstressed cells in magenta; manual painting
remains available as the experimental alternative. For the saved reference job
the intact wall peak is 1.051 MPa (0.53 `f_t`), so screening reports no
overstressed cells.

`FEniCSx/automatic_cracking.py` is the experimental real-dam runner. It takes a
completed screening, rebuilds the native wall candidates and runs the cohesive
network under proportional load control (reservoir water only on cracks
connected to tag 2). Outcomes: `no_damage_initiated` (exact when the screening
peak is below `f_t*(1-tol)` for proportional linear loading) and
`accepted_equilibrium` are the only accepted results; `unsupported_geometry`,
`unsupported_hydraulics`, `nonconvergence`, `unresolved_initiation`,
`resource_limit` and `cancelled` are reported honestly as not accepted. When
the cell principal peak exceeds `f_t`, the runner first evaluates the
network's own initiation measure, the one-sided facet-normal stress
`max(n.sigma_L.n, n.sigma_R.n)` (larger owner cell; the two-cell average
diluted one-element-thick surface tension below `f_t` and never initiated), on
every interior wall facet from the verified screening archives
(`intact_facet_screening` in the summary; the screening reports the same
`facet_check` so review shows whether cracking will initiate). If every facet
is below `f_t*(1-tol)` the result is `no_damage_initiated` without mesh work;
otherwise the cells beside critical facets seed the candidate region, which
sheds neighbour rings (down to zero) to fit the face preflight. The final
state is accepted only if no non-candidate wall facet reaches `f_t`. Screening
cell counts therefore overstate initiation: isolated P1 cell peaks (for example
a 1.24 MPa surface-layer peak with `f_t` = 1 MPa) need not act normal to any
shared facet. It
writes `summary.json` (load factor, steps, dissipation, wet faces, bonded peak)
and, when cracks exist, `cohesive_surface.vtu`.

Submission is gated: `POST /jobs/{id}/automatic-cracking/submit` returns 202
only when the latest screening succeeded, has a verified confirmation
(matching run id, configuration digest, screening digest and Gf) and nothing
else runs for the job; otherwise 409. A newer unconfirmed screening supersedes
an older confirmation. `GET /jobs/{id}/automatic-cracking/run` returns the
latest run (`summary` without per-step data, `accepted`,
`engineering_validated: false`) and `POST .../run/cancel` stops it. The API
verifies the runner manifest, summary digest and provenance before reporting
success. The desktop Submit button follows `submission_allowed`, shows the run
description and limitations, and opens the crack surface in ParaView.

A submitted run is the job's fracture stage, exactly as a painted event is:
the job becomes `running` with stage `automatic_cracking`, `fracture_source`
becomes `automatic` (a later painted event sets it back to `painted`), Crack
definition shows succeeded and the three fracture stages (group "Fracture
analysis (automatic cohesive)") follow the runner manifest phase: screening
checks, mesh import and candidate preparation are Preparing; assembling the
native system is Initial equilibrium; the cohesive network solve is Growth.
When the runner ends the job returns to its previous status and
`automatic_cracking_status` records the outcome: `succeeded` only for an
accepted outcome; a completed but unaccepted outcome is `failed` with its
`outcome_basis` (at Preparing when refused before any load step, otherwise at
Growth). Cancel Job stops only this run (`cancelled`); an API restart marks a
running run `interrupted`; restarting a fracture stage resubmits the latest
confirmed screening. The desktop window closes on submit and the Run/Results
pipeline, activity monitor and job log (the runner log, via
`/painted-event/logs`) take over. The network solve reports no per-step
progress, so Growth has no ETA.

Known limits: on structured hex-derived tetrahedral meshes, growth beyond
initial softening stops with `unsupported_geometry` (a detaching wedge needs
two normals at one vertex pair); load control cannot trace snap-back (coarse
meshes report `nonconvergence`); real-scale solve cost, Gf (400 J/m2) and an
overstressed real-dam run are unverified. Jobs submitted before the Gf field
existed cannot run: the provenance chain forbids adding Gf afterwards, so a new
job must be submitted.

## Current execution restriction

Legacy seeded-crack app submission and cracking reruns are retired. New API jobs
run the elastic baselines only; configured legacy cracking stages are blocked
with explicit feedback rather than launching a fallback solver. The old
single-point POST and stage-rerun POST return authenticated HTTP 410 responses.
The desktop removes those launch controls and labels submission Run baselines.
Painting saves a selection and can launch the separate experimental FEniCSx
painted-event solver; it does not run Code_Aster fracture. New app submissions
require common-head hydrostatic crack loading, with no dry fallback.
Historical results and standalone numerical CLI defaults are preserved.
The implementation sequence below must not be interpreted as permission to wire
the preview into the legacy seed workflow.

### Current painted-scenario progress

- Configuration's Material tab exposes selected-element tensile strength from
  zero to the configured concrete tensile strength in MPa, stored as optional
  `material.concrete.selected_tensile_strength_pa`.
  Absent values inherit the configured concrete tensile strength. Only painted
  wall cells use this reversible tensile-cap law after the intact equilibrium;
  zero reproduces the no-tension case. This is not irreversible softening or
  strength-calibrated nucleation. The cap, energy and tangent are used together
  in the growth equilibria and recorded in numerical inputs/results.
- Optional `material.concrete.selected_youngs_modulus` sets the damaged-concrete
  Young's modulus of painted cells (positive, not above the concrete modulus;
  absent keeps the original stiffness). It applies with the tensile cap after
  the intact equilibrium and is recorded in the manifest, event report and
  native inputs. The shipped values follow `element_matrix.csv` (35/15 GPa,
  2/0.5 MPa, fc 30 MPa, Gf 100 J/m2). Automatic cohesive cracking does not use
  the damaged properties.
- The painter and API require hydrostatic mode. Pressure on released faces is
  derived from proven reservoir connectivity/head and retained exterior loads;
  arbitrary pressure on every internal face is not assigned. Reservoir
  discovery now retains upstream, tailwater, waterline-crossing and horizontal
  constant-pressure crest edges; head/access checks apply to released cuts,
  not the whole potential candidate inventory. Crest head is derived from
  its actual elevation and configured pressure, with a horizontal-face check.
  Partially submerged crack-face forces and pressure work use exact clipped
  P1 integrals. Reservoir-disconnected cuts remain explicitly unpressurised.
  Final wet acceptance checks open submerged access and nonpenetrating contact.
  An actual released cut meeting different heads still fails explicitly; a
  general different-head wet-growth model and trapped/filling history remain pending.
  Historical dry events remain viewable but cannot be restarted as dry; reopen
  the painter for explicit hydrostatic resubmission. Existing job configuration
  snapshots stay frozen, so changing tensile strength requires a new baseline job.
- The growth trial reuses its initial accepted intact/reduced equilibrium only
  after verifying the full mechanical-problem identity and monotone cut release.
  Every trial still re-equilibrates and audits residual/contact/force balance.
  This avoids repeating the intact/strength-drop solve for each possible face.
- Single-face releases can be kinematically ineffective when neighbouring
  full-vector bonds still identify every crack-lip vertex. Native growth now
  excludes patches that split no bonded identities and fails explicitly if
  none of the proposed extensions can open; this must not be reported as
  arrested growth. Initial candidate screening happens before the intact/
  selected-material equilibrium solve, so a wholly locked initial front fails
  during preparation without spending time on an unusable equilibrium.
  The report records `initial_equilibrium_started=false` and the excluded
  patches; previously saved job outputs are unchanged.
  New painted app runs explicitly enable `resolve_locked_patches`. For each
  locked front seed, a minimum-cardinality cut of the remaining bonds around
  each seed vertex proposes a finite, edge-connected manifold surface disk.
  This resolves the P1 node-star locking without changing the bulk mesh,
  inserting a gap, removing cells or releasing faces for free. Up to three
  completions per seed are generated, rather than enumerating all connected
  subsets. Redundant supersets of another opening-capable completion of that
  seed are excluded: resolution tests irreducible first-opening increments,
  while larger finite jumps remain the explicit patch-bound search's scope.
  Unlisted faces remain bonded. The full geometric area of each
  proposed completion is charged at the configured fracture energy, and
  acceptance still requires lower total potential and an audited equilibrium.
  Native dry/wet callers retain the old default unless explicitly enabled;
  reports distinguish node-star-patch-set arrest from single-face arrest.
  Hydraulic adjacency uses both lip identities after remaining bonds are
  condensed. Only inlet-connected released sheets receive pressure; other
  sheets are explicitly dry. The clipped fixed-head loader is used for fully
  and partially submerged cuts alike, allowing contact at individual vertices
  only if a nonpenetrating, open submerged reservoir path is independently
  established on the solved state.
  A force/contact-audited pressure trial with closed hydraulic access is not
  an accepted wet state. In the current convex material/contact problem,
  fully applying nonnegative reservoir pressure bounds the potential from
  below for every physically accessible drained subset. If even that bound
  cannot pay the added geometric fracture area, the candidate can be rejected
  with an explicit energy-bound certificate instead of aborting the search.
  Favourable bounds with unresolved access and numerical failures still fail;
  no dry fallback or invented aperture is used. This convex bound must not
  be assumed valid for future nonconvex softening.
  Historical completed single-face searches require that
  caveat. This correction does not turn the reversible tensile cap into
  irreversible softening or strength-calibrated nucleation. Native node-star
  tests demonstrate positive opening, full-area energy costs, two accepted
  wet extensions, irreversibility under increased surface resistance and
  rejection of high-cost extensions. These are synthetic verification cases,
  not dam-property calibration.
- Native specimen tests at 0.5 MPa tensile capacity check elastic equilibrium
  at 0.4 MPa tension, explicit nonacceptance at 1.2 MPa tension and unchanged
  compressive compliance at 1.2 MPa compression. Surface pressure must not be
  compared directly to tensile strength as a crushing criterion.

- The painter can explicitly save append-only, job-bound selection snapshots
  and restore the latest on reopening. Snapshots include source element/edge
  IDs, face choices and numeric opening limit (default 0.002 m). Authenticated
  GET/POST `/jobs/{id}/damage-selection/saved` routes revalidate configuration,
  catalog identity and integrity. Corrupt/stale latest snapshots fail explicitly;
  unsaved changes are still discarded. Saving starts no solver and changes no
  baseline/job configuration.
- Experimental FEniCSx checks now cover a relaxed no-tension local material
  retaining original compressive compliance and native 3D post-event equilibrium
  patches. This reversible local relaxation is not an irreversible propagating
  crack or a calculated face opening.
- A reduced native load-controlled event driver now solves the intact state,
  imposes the selected-cell material change and re-equilibrates with the same
  external load functional/BCs. Two-mesh tests check load redistribution and
  reactions; complete loss of tensile load paths fails explicitly. New isolated
  outputs preserve input mesh/selection/load and accepted intact state, stage
  status and Newton trace even on event failure; accepted post-event material
  and displacement exports appear only after equilibrium checks. Cellwise
  properties now accept the existing concrete/rock DG0 material fields.
  The existing mortar spring operator and canonical roller support form can
  be retained in both solves and the independent reaction audit; reduced
  analytical checks cover their compliance/forces/energy. Cellwise properties
  and optional mortar CSR coefficients are archived. Serial constrained mortar
  DOFs now use verified row/column elimination and imposed-displacement lifting,
  retaining original coefficients for reactions/energy. Tests cover nonzero
  prescribed displacements during nonlinear tensile release. A synthetic
  patch verifies unchanged gravity/reservoir/tailwater forces during tensile
  release. This original reduced driver is serial/direct; the later contact
  driver also provides opt-in iterative solving. Production provenance/load/support
  wiring, resource feasibility, hydraulic times and unrestricted propagation
  remain unimplemented.
- An experimental source-import event entry point now imports a canonical
  structure itself, maps source rows through native reordering and rejects
  bedrock/foundation selections. It reuses the baseline's shared gravity/
  reservoir/tailwater/top-pressure helper, canonical materials, mortar springs
  and outer supports. Native two/three-body checks verify force integrals,
  both foundation interfaces and source-order intact displacement agreement
  with the existing elastic solver (`rtol=1e-8`, `atol=1e-11 m`). Source rows
  are preserved in native archives. This is not saved-request hash binding or
  an API submission route. Coarse triangles crossing a waterline exposed
  positive-part quadrature errors. Exact clipped-triangle P1 water loads now
  support these cases, with independent analytical force/moment/virtual-work
  and subdivision checks. Fixed nodal pressure forces enter both event states,
  reaction checks and archives; source-event tests include partially submerged
  reservoir and tailwater facets. This is not evolving crack-water pressure.
  Baseline material parameters and loads remain unchanged. The later
  source-growth fixture also corrects overlapping serial mortar/fixed-support
  constraints and includes original mortar forces in fixed-support reactions
  (see the source-growth validation below).
- A baseline-bound experimental Python mechanics entry point now rebuilds the
  canonical structure, requires existing elastic parity/equilibrium evidence,
  rebuilds/validates the catalog and checks exact compacted source topology/
  regions/coordinates. Mesh/configuration/baseline-field identities and source
  diagnostic hashes must match before solving. The report retains binding
  evidence, resolved faces, marker edges and numeric opening limit. Tests cover
  actual Gmsh binding, stale/reordered/changed inputs and reduced native
  provenance persistence with substituted baseline-loading orchestration.
  This is not a full saved-dam integration test, API route or edge-growth law;
  no submission capability has been enabled.
- Serial native source-row tags and checked wall-selection masks preserve
  source identity across subdivision, native reordering and two refinement
  levels in reduced tests. Coincident split nodes are not merged; non-wall
  selections fail. Production must still bind the source to the validated
  catalog and preserve these tags through its own preparation/refinement.
- Reduced hydraulic checks cover prescribed-open saturated 1-D flow, cubic-law
  conductivity, gravity, transient linearised storage and aperture-rate effects.
  Fluid properties and physical pressure/time histories are explicit test
  inputs, not new production calibration. A further native reduced variant
  couples local normal compliance to pressure-dependent opening, storage and
  conductivity with a monolithic Newton solve. Independent contact tests apply
  prescribed water tractions, release compression, calculate zero-initial-gap
  face opening and reclose on unloading. These separate reduced models do not
  implement the dam's evolving wetting/contact/crack system.
- Optional scalar tangential loading in the native contact benchmark now checks
  Coulomb stick/slip/reversal and nonnegative dissipation with explicit friction
  coefficient/shear modulus. Friction uses solid contact force only, so pressure
  reduces friction capacity and open faces carry no shear. 3-D contact search,
  surface growth and production calibration are not supplied by this benchmark.
- Explicit paired 3-D P1 face utilities measure normal opening from source-node
  displacement jumps, retaining signed penetration/tangential slip separately,
  and integrate consistent equal/opposite P1 water-pressure forces. A native
  zero-gap two-block test verifies analytical opening, stress and equilibrium.
  Scenario-neutral measured-opening assessment preserves the peak after
  reclosure and warns at/above the limit. Neither the extractor nor that
  assessment constructs a surface from a selected edge or implements growth.
- An explicit-facet topology utility now cuts interior wall faces with node-star
  connectivity, preserving tied crack-front nodes, cell rows/regions, parent
  node IDs and exterior boundary tags. Accepted-state transfer imposes no jump.
  Native integration verifies a bonded mesh through explicit cut and pressure
  loading to measured 3-D opening/equilibrium. Boundary/non-wall/nonmanifold and
  fully tied under-resolved cuts fail explicitly. This does not choose a
  fracture surface from a selected edge or a constitutive propagation criterion.
- Paired-surface water-access checks now use explicit hydraulic inlet source
  edges and measured opening, not coordinate proximity or painted markers.
  Closed connecting edges, tied fronts, point contacts and unrelated coincident
  sheets do not transmit access. Reachable/isolated open faces and blocked
  inlets remain explicit. Connectivity assigns no pressure or saturation and
  must not silently erase trapped fluid after reclosure; transient filling
  and pressure require a conserved-fluid state.
- See [verification scope and commands](../FEniCSx/README.md#painted-tensile-loss-building-blocks).
  Full-dam provenance wiring, edge-to-surface construction, growth,
  friction, wetting/contact and pressure feedback, actual opening extraction,
  Code_Aster parity and resource/engineering validation remain prerequisites
  before exposing scenario submission.

The working crack-growth scope is unrestricted turning/branching through the
concrete, matching the painted-event intent; no horizontal-only constraint or
initial opening is imposed. The physical growth/surface formulation is not yet
implemented by the local no-tension law.

### Target: eventual post-event equilibrium

The requested result is eventual quasi-static equilibrium, not the response
second by second. Preserve the loaded intact baseline, impose permanent tensile
capacity loss in the selected wall elements, and seek an admissible post-event
equilibrium under unchanged gravity and exterior hydraulic loads.

Structural inertia, impact, filling duration and transient pressure histories
are outside this first scenario. Numerical continuation steps are not physical
seconds. Fluid viscosity/compressibility and an event duration are not mandatory
inputs for an explicitly drained hydrostatic equilibrium model; they must not
be demanded merely because transient benchmark components exist.

The opening assessment now has an equilibrium-specific accepted-state contract:
state IDs replace physical timestamps, peak and final opening are distinct,
and reclosure cannot hide an accepted peak at/above the warning threshold.
Incomplete runs remain unassessed. Shared threshold/warning logic preserves
legacy transient reports. Three pure tests and the native measured hydrostatic
opening check cover this contract; production result UI wiring remains gated.

Hydraulic equilibrium assumptions remain explicit: an open, saturated path in
communication with a reservoir uses its hydrostatic head, with pressure acting
back on crack faces. Geometric connectivity alone does not establish saturation
or drainage of closed/isolated pockets. Connected reservoirs at different heads
can require steady flow rather than a single hydrostatic pressure field.
Trapped fluid/air and disconnected or closed pockets require a stated treatment;
do not silently discard their contents or assign reservoir pressure.

Irreversible fracture/contact history is still relevant to the final state.
Static nonconvergence is not proof of collapse, and convergence alone does not
establish stability, uniqueness or dam safety. Fracture calibration, friction,
appropriate hydraulic boundary assumptions and independent validation remain
required; the scope reduction does not make the production solver ready.

The eventual-equilibrium scope and stated hydraulic exclusions have been
accepted. The first drained hydrostatic crack-loading component now assigns
`rho*g*(head-z)` only to explicitly saturated, fully submerged, open faces
reachable from named verified hydraulic inlet edges. It reuses paired-face
consistent tractions. Explicit previously-wet masks prevent silent drainage on
isolation/reclosure. Connected unequal heads fail pending steady flow, and
partial saturation/free surfaces fail pending appropriate hydraulic topology.
Independent components may use different heads. Seven pure checks and a native
3-D pressure/opening/access-consistency check verify this limited component.
It does not choose fracture resistance/friction values or implement
irreversible propagation/contact; `solver_ready` remains false.

A separate reduced steady surface-flow component now handles unequal heads
on prescribed explicitly saturated, fully open paired surfaces. It solves
`div(w^3 grad H)=0` and converts `rho*g*(H-z)` to consistent paired-face
loads. Uniform viscosity cancels from source-free head; only scaled boundary
reactions are reported, not physical leakage. Nine analytical/convergence/
identity tests check head, cubic conductance, variable aperture, hydrostatic
equivalence and conservation. Closed/tied vertices, isolated wet surfaces,
point-only or nonmanifold junctions, conflicting heads and negative pressure
remain explicit failures. Hydrostatic callers do not silently switch models.
This is not coupled evolving contact/flow, an irreversible fracture model
or a full-dam scenario; production submission stays disabled.

The steady surface component now also has a reduced local-compliance feedback
solve: pressure changes aperture via `w=w_ref+C*(p-p_ref)`, and that aperture
changes the steady head through cubic conductance. Explicit options bound
iterations and set aperture/flow tolerances and relaxation. Five checks verify
analytical hydrostatic/stiff-wall limits, reference/owner invariance, explicit
nonconvergence/closure and mesh convergence to an independently integrated
nonlinear steady solution. Accepted states satisfy both compliance and audited
flow on the same aperture. Synthetic already-open specimen data are not dam
defaults; this does not implement bulk mechanics/contact/friction or growth.

A reduced fixed-cut adapter now couples actual 3-D bulk elastic deformation
to aperture-dependent steady surface head. Unchanged external-load dry
equilibrium must already open all hydraulic vertices; no seed gap or artificial
aperture is imposed. Accepted candidates recompute pressure on their own opening
and pass independent bulk free residual/global force, surface flow balance and
displacement-correction checks. Six native tests verify actual aperture/head
feedback, hydrostatic limit, exterior-face/orientation provenance and
closed-dry-cut/nonconvergence failure, retained mortar seats under nonzero
prescribed support motion, and equivalent fixed nodal/form external loads.
Mortar Dirichlet lifting uses the existing helper; independent audits retain
the original spring operator. Original CSR coefficients, dry/wet spring energy,
and fixed nodal external loads are retained in diagnostics/archives.
Isolated outputs retain dry/input/failure evidence, and accepted-only NPZ/VTK
fields use state IDs rather than physical time. This is not yet a painted
tensile-loss/contact/propagation adapter or full-dam validation.

A native serial 3-D fixed-cut frictionless-contact entry point now enforces
vertex-normal nonpenetration with nodal-force multipliers, without a penalty
stiffness or artificial aperture. Both tangential directions remain free.
Existing mortar matrices, fixed nodal loads and imposed support motion enter
the original bulk stiffness/load system. Exterior-lip/source identity checks
are shared with the open-cut flow adapter. Eight tests cover analytical closure,
pressure reopening/reclosure, oblique normals, free tangential motion,
partial contact against independent constrained-energy minimisation, and
accepted-only output/failure trace. Water is an explicitly prescribed traction
in these checks, not inferred saturation or transported pressure. This does not
yet couple contact to the selected no-tension event/flow/growth, supply Coulomb
calibration, or establish dam-scale feasibility. Production submission remains
disabled.

The experimental `solve_contact_event` now integrates the selected-cell material
change with native 3-D contact instead of merely testing them separately.
It establishes a fully bonded loaded state (all jump components tied), then
releases an explicitly supplied cut while selected cells lose tensile capacity.
Compression and external loads/support motion are retained. The nonlinear
material tangent/contact solve is independently audited using the actual
material stress, reactions and global force balance.
Explicit steady hydraulic data can now couple surface head/pressure to the
same accepted material/contact displacement, after a dry post-event equilibrium
opens the hydraulic inventory. Closed dry cuts and wet closure remain explicit
failures, not residual-aperture regularisation or silent drainage. Inputs,
bonded/dry states, iterations and failures persist; only accepted post-event
states receive result exports. Native tests include actual tensile redistribution,
analytical bonded fields, retained compression, same-state coupled pressure and
failure persistence. **Edge-to-surface initiation, irreversible unrestricted
growth, calibrated friction, trapped-fluid/free-surface states, canonical saved-
dam integration/resource validation and production submission remain missing.**

An experimental dry finite-candidate Griffith search now supplies an energy
decision for explicit source triangle extensions: re-equilibrate the accumulated
cut, compare mechanical potential release with `Gc * area`, and never heal an
accepted surface. Actual interior wall/wall facets in all orientations can be
extracted from source identities and marker incidence; point-only/coincident
contacts are not edge adjacency. This remains a mesh-facet restriction, not an
implementation of unrestricted physical paths. Arrest is labelled only within
the tested single-triangle candidate set; multi-facet jumps are not excluded.
Failed trials and a limit with favourable remaining extension fail explicitly.
Native tests verify the analytical displacement-controlled energy threshold and
selected-material bonded/released comparison. The event driver now reports
dry stored material/support/mortar energy, dead-load work and a bound problem
SHA256. Pressure-loaded/flow states cannot use that dry mechanical potential:
reservoir work is not supplied. Wet growth, strength-controlled nucleation,
candidate preparation/resolution, full-dam feasibility and
production integration remain incomplete. This does not change submission gates.
Prepared-candidate native orchestration is now implemented: each energy trial
solves the selected material/contact problem with full bonds on unbroken faces
and unilateral contact on released faces. Source-parent/marker identities and
the exact final accepted trial directory persist. A native eight-candidate
example verifies growth versus no growth under low/high surface resistance.
This is explicitly dry and mesh-constrained, not an unrestricted wet path model.
Canonical source preparation/orchestration is now experimental code rather
than a caller-only responsibility: actual wall/wall source facets are split,
exterior tags/materials/source rows retained, and the original gravity, exact
exterior water loads, mortar seats and outer supports are reused. New surfaces
remain explicitly dry. Original-parent multiplier identities are retained
when wall nodes split; rebuilding independent per-triangle multiplier rows
would otherwise change foundation stiffness even in the bonded state.
Native displacement parity and projector/energy tests guard that invariant.
Source mapping and trial failures persist without a success-shaped fallback.
The source entry point is not bound to authenticated saved request/baseline
evidence; all-facet preparation/trial performance is not validated on the dam.
It remains dry and mesh-constrained, not unrestricted water-coupled production.
An experimental bound dry-growth adapter now shares the existing event
baseline/selection verifier: it rebuilds structure/catalog, verifies field,
mesh and configuration identity, rechecks diagnostic hashes, and translates
catalog marker endpoints through source-node compaction. Selection/baseline
provenance and the opening warning limit persist with source/trial results.
The direct source adapter still makes no binding claim. Tests cover actual
Gmsh catalog mapping, rejection before preparation and native bound-adapter
orchestration with substituted baseline loading, not a full saved-dam solve.
No API/WPF solve route is enabled, and unrestricted wet propagation is unchanged.
An opt-in saturated-flow extension now accepts exact tied crack-front nodes
without a minimum hydraulic gap. Positive face conductance is integrated from
the actual P1 aperture cubed; untied contact vertices, closed inlets and closed
internal barriers remain explicit failures. Analytical wedge head/conductance/
virtual-work tests and a native three-tet material/contact/water equilibrium
check verify this bounded case. Default strict-open hydraulics is unchanged.
This is not a closed/trapped-fluid state, wetting model or wet growth law.
Explicit hydraulic face IDs can now select a released wet subset from the
larger bonded/contact candidate inventory. Unbroken faces cannot receive water
loads. The same transitive all-component mechanical bond identities establish
hydraulic front connectivity, with checked source parents/displacement ties;
coordinates alone never merge sheets. Hydraulic pressures integrate onto the
original solid lip nodes. Native connected/subset tests verify unchanged bonds,
same-state mechanics/flow and archived hydraulic row mappings. Omitted faces
are explicitly dry; this cannot silently discard a previously wet fluid state
and does not implement wet propagation, wetting or trapped-water conservation.
An experimental common-head hydrostatic finite-candidate growth driver now
includes reservoir work explicitly in `Pi=U-W_exterior-integral(p*w)dA`.
This is conservative only for fixed reference geometry and common reservoir
head; different-head dissipative flow is rejected as a growth potential.
Every released surface is explicitly saturated and must open and connect to
declared inlets at accepted equilibrium (optional tied tip). Mechanics/reservoir
identities bind all trial comparisons. Native tests verify pressure virtual
work, low/high resistance release/arrest, a reservoir-work-dependent decision,
and irreversible extension of a seven-face wet surface to eight faces.
Failed or hydraulically inadmissible trials fail explicitly, never claim arrest.
The common-head specialization now permits pressure to open a closed trial
without an initial gap or dry-open intermediate. Final aperture/access/flow are
independently audited before acceptance; a native dry-closed compression-contact
specimen opens under reservoir pressure and passes material/reaction/work checks.
This explicitly saturated eventual-equilibrium branch does not predict filling
history, and general steady through-flow retains the dry-opening prerequisite.
This is still mesh-constrained, not unrestricted propagation; wetting/trapped-fluid conservation, general through-flow
growth, full-dam validation and production integration remain unfinished.
Dry/common-head native growth reports now retain accepted equilibrium evidence
and aggregate opening-limit warnings over accepted states only. Reclosure does
not erase an earlier peak; rejected trial openings do not enter the warning.
Incomplete searches retain observed peaks without claiming the criterion
passed. State IDs remain continuation labels, not physical time.
Canonical source and verified-baseline adapters now reach the experimental
common-head wet-growth driver through shared source/load/support preparation.
Hydraulic inlets derive from inherited exterior reservoir tags and canonical
heads, independently of painted markers. Different-head connections,
waterline-crossing edges, ambiguous constant-pressure crest connections and
absent reservoir access fail explicitly; no head or fluid data is invented.
Native reduced source trials verify unchanged loads/materials/mortar and intact
displacement parity. Bound-orchestration tests substitute baseline loading;
the actual saved-dam workflow remains unverified and disabled.
This fixture also verifies serial baseline spring/Dirichlet elimination where
fixed rock nodes meet mortar seats, including original spring forces in support
reactions; fixed displacements and global force balance are regression checked.
Reduced source tests now cover accepted low-resistance wet fracture as well as
high-resistance arrest. An actual Gmsh/catalog/selection-binding/native wet-growth
test reopens accepted source/input/hydraulic archives and checks real mapped
element/edge/face identities. Only structure rebuilding is substituted,
not the catalog, binding, preparation or numerical solves. This is a reduced
component test with synthetic baseline evidence, not a full saved-dam workflow
or engineering validation.
That integration now runs the actual saved-baseline gate against explicitly
synthetic reduced-test evidence; only structure rebuilding is substituted.
Changing completion status, force balance or source identities prevents growth
before output creation. This does not constitute independent Elmer parity.

Dry/common-head energy search now optionally tests every connected patch up to
`GrowthOptions.maximum_patch_faces` (default 1). This addresses bounded
multi-face jumps over single-face energy barriers without imposing free crack
area or altering materials. All candidate patches touch the current edge front,
retain earlier cuts and re-equilibrate the full problem. Reports record the
patch bound and exact candidate surface IDs, and distinguish patch-set arrest
from single-face arrest. Native six-to-eight-face wet release checks full
reservoir work and irreversible cuts; exhaustive subset tests independently
check candidate completeness within the bound. Larger jumps, mesh-independent
paths, nucleation and full-dam cost remain unresolved. The extension cap counts
accepted patch steps, not newly released faces; a favourable remaining trial or
failed patch equilibrium is failure rather than arrest.
The current native regression passes 266 distinct tests. A read-only binding
check against the actual saved dam (194,316 nodes, 990,840 tetrahedra) also
completed in 129.47 s: real structure rebuilding, source/configuration/diagnostic
hashes, saved-baseline gate, catalog and exact element/edge mapping all passed,
with no substituted loading. The existing independent elastic evidence reports
relative displacement difference `1.767e-7`, nodal stress difference `5.991e-7`
and force-balance error `7.989e-11`. These values were verified from the saved
baseline, not recomputed in that check. The diagnostic selected one eligible
element and actual edge only to exercise binding; it neither changed saved
user selections nor launched full-dam fracture growth.
An isolated experimental module runner now consumes the app's saved painting
snapshot format and the same integrity reader as API restore. Explicit job,
configuration, source model, baseline and field identities are revalidated;
numerical controls and a source-cell resource ceiling are supplied separately.
It preserves input snapshots, environment, verified binding, stage failure and
accepted PVD/native-state/report paths in a new output directory only. Dry and
common-head wet reduced native integration tests run through completion;
unsupported water never silently switches to dry. This is a standalone
experimental tool, not production API/WPF submission or dam-scale validation.
Accepted output also includes a paired-lip ParaView surface with exact split/
original-parent/face identities, released/wet masks, displacement, measured
normal opening/signed gap/tangential jump and accepted water pressure. Both
unit VTU round trips and native dry/wet runs reopen and verify these fields.
The runner's `--prepare-only` mode now inventories actual unsplit native wall
facets and checks canonical reservoir connections before any candidate-node
splitting or solve. Native inventory is compared against the independent
cell-owner implementation on reduced concrete/rock and jointed sources,
including exact ordering, marker identity and orientation. Unsplit/split inlet
sets agree. Multiple water-model incompatibilities are reported together and
preserved with inventory counts even on failure.

Actual saved-dam preparation found 1,177,580 candidate faces and an upper bound
of 7,859,916 split displacement DOFs. Dry preparation succeeded without any
fracture solve. The actual wet preflight explicitly rejects both positive
constant-pressure crest connections and differing reservoir heads; 73.70 s
total, of which 10.90 s was native inventory/hydraulic screening after binding.
It created no split/growth mesh or accepted-result index and modified no user
selection. Therefore the currently verified common-head growth component is
not applicable to this saved dam's complete candidate inventory. General
nonconservative hydraulic growth and scalable dam-level nonlinear mechanics
remain concrete blockers, not solved by the standalone runner.
Native contact now eliminates intact/unbroken identity bonds exactly through
a displacement quotient, rather than enlarging every solve with bond
multipliers. It recovers original forest forces and audits the original split
space, preserving nonzero supports, forces, energy, irreversible releases and
contact inequalities. The intact quotient recovers the original unsplit DOF
count in the source regression. The previous multiplier path remains available
for direct parity checks. This reduces solve size, not split-space assembly
cost, and does not establish full-dam resource feasibility.

An opt-in symmetric MINRES contact solve now avoids LU factorisation, with
explicit tolerances/iteration cap and positive mechanical/approximate-Schur
diagonal preconditioning. Direct remains the default. Tests compare direct/
iterative native partial contact and wet multi-face growth, and verify
iteration-limit/nonpositive-diagonal failures with no direct fallback.
Actual full-dam nonlinear scalability remains unverified.

The released frictionless source fixture also exposes a physical/numerical
qualification: equilibrium and equal stress/energy/opening do not prove unique
displacement. Direct and iterative methods can select different zero-energy
displacement modes. Reports now explicitly mark uniqueness/stability unverified;
a separate semidefinite counterexample checks this. No friction, extra support
or residual stiffness is invented to remove the ambiguity.

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

Painted-event results are a separate comparison from crack scenarios. The
Comparison tab shows the initial intact and final accepted equilibria in their
own group, including raw cellwise whole-model stress extrema and matched-node
displacement-field change. It must not place a weakened-but-bonded state in
the cracked-scenario column; that column remains empty when growth accepts no
crack extension. The raw stress extrema are not directly comparable with the
baseline's recovered wall peaks.

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
| Global element size | Starting suggestion; the wall analysis mesh scales it to stay at or below `geometry.target_wall_tetrahedral_elements` (default 150,000) final tetrahedra, while the conforming foundation and near-seat bedrock adapt to the effective target size |
| Target wall tetrahedral elements | Optional integer ≥ 1,000 (default 150,000) capping wall-body tetrahedra; a number input in the Geometry tab |
| Local element size | Identify which generator uses it; do not imply it refines the current combined mesh |
| `water_height` | Reservoir depth above the deepest wall seat; with `overflow_head` sets the water surface |
| `overflow_head` | Depth above the crest when overflowing; adds the crest surcharge |
| `overflow_water_level` | Derived reference depth: water height plus overflow head |
| Explicit directional springs | Without `geometry.foundation`, the wall/bedrock seat uses `geometry.bedrock.support.spring_n_per_m3`; with it, that setting controls foundation/bedrock and `geometry.foundation.support` controls wall/foundation. Springs act on relative displacement only (no grounded spring bypass); bonded interfaces share nodes and leave spring coefficients inactive. The legacy wall-only solve retains its grounded bedrock support. Material modulus does not recalculate springs. The temporary top-level `interfaces` override is no longer accepted. |
| Bedrock material | Active in the integrated three-body solver; the legacy wall-only solve represents the foundation with grounded springs instead |
| `geometry.bedrock.domain` | Optional explicit stream/bank margins and bottom elevation shared by the preview, integrated three-body solver and diagnostic; inactive in the legacy wall-only solve |
| `geometry.bedrock.mesh` | Optional conforming 3-D rock grading: near-seat size follows the effective wall size, grading toward the configured far size by the configured distance; shared by preview and three-body preparation. Only wall tetrahedra count toward the 150,000 cap. Missing settings retain legacy layered meshing; lengths are not cell volumes or maximum-edge guarantees. |
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