# Configurable FEniCSx dam study

## Current milestone

This directory is the new WSL-based adapter; the Elmer model is unchanged.
The verified scope includes environment/configuration checking, a migration
inventory, an exact elastic patch test, conforming dam import and a full-dam
elastic comparison against the current Elmer run, and an analytically checked
two-bar frictionless contact benchmark. It now also includes a staggered hybrid
AT2 phase-field fracture solver with irreversible bound-constrained damage,
local crack-band refinement and ParaView output. **3-D crack-face contact is
not implemented.** Acceptance limits are provisional porting criteria, not
independent engineering validation.

The phase-field path supports evolution from a configured diffuse crack. The
older `predefined_crack` template remains an explicit-surface/contact concept
and is still blocked; these are different models and are not silently mixed.

The shared source geometry reads [root config.json](../config.json): a 4 m wall
(radii 76-80 m) centred on a 6 m plinth (radii 75-81 m), with 1 m overhang on
each side and centreline radius 78 m. Regenerate the Elmer reference and reimport the dam;
previous dam meshes and parity results are historical, not validation of
these updated dimensions.

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

### Job-specific Elmer baseline handover

The API passes the freshly validated baseline to `run-fracture` using
`--elmer-result <job-result.pvtu>`. The API coordinates the processes; the
consumer adapter in [elmer_baseline.py](elmer_baseline.py) owns parsing.
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
  element-centroid location of the largest principal stress, calculated from
  element-mean nodal tensors over all source volume elements.
- `manifest.json` includes the baseline report; a handover error fails the
  fracture run explicitly, while the API retains the successful Elmer result.

**This imports a baseline reference, not a mechanical initial condition.**
The phase-field solver still solves its own configured loading. An explicit stress/displacement/history transfer
law and verification are required before using these arrays as initial state.
The recorded peak can be compressive if the entire model is in compression;
`has_tension` distinguishes this. These outputs are not engineering validation.

### Automatic horizontal crack

After each successful Elmer baseline, the API starts FEniCSx even without
`crack_simulations`. The consumer uses the **global** maximum principal stress
of element-mean nodal tensors and its **undeformed element-centroid Z**. It
interpolates exactly two plinth chainage intersections at that signed elevation,
using the authoritative wall centreline radius to convert chainage to angle.
The initial crack is a horizontal annular-sector prism following the upstream
wall between those angles, extending radially inward by **75% of wall thickness**
with a **0.002 m vertical opening**. This is not a straight rectangular chord.

The named result is `automatic_peak_tension`; existing result endpoints and
ParaView controls discover it from the run manifest. `automatic_crack.json`
records the source peak, upstream intersection coordinates, exact sector,
generated specification and numerical settings. The submitted `config.json`
is not rewritten.

Fracture is restricted to **Wall, region 2**. Initial and evolving damage are
fixed at zero on all non-wall cells' damage DOFs, including shared interface
DOFs. This is an explicit no-fracture-in-plinth modelling assumption and imposes
a zero-damage trace where wall and plinth share nodes; it is not a contact law.
The plinth remains in the elastic mechanical solve.

Only wall cells drive crack-resolution checks and refinement marking; only
under-resolved wall cells near the crack are marked. Neighbouring cells may
still be subdivided for conformity. Refinement reports and logs record wall and
plinth cell counts separately, with marked plinth cells always zero. Automatic
runs allow up to **five** refinement levels, stopping early when the unchanged
resolution target is met.

[fracture_defaults.json](fracture_defaults.json) preserves the previous workflow's
AT2 settings: length scale 0.1 m, Gc 2700 J/m2, zero internal crack pressure,
ten load steps and existing solver/refinement controls. These are explicit
research defaults, not a new calibration or engineering validation. In
particular, the retained 0.9 elements-per-length-scale target is preliminary.
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

The default single-process WSL job uses three local refinement levels and a
preliminary target of `0.9` length scales per maximum tetrahedron diameter. The
resulting `h_max` is about `0.109 m` for `l=0.1 m`, so it is explicitly
under-resolved and unsuitable for mesh-independent conclusions. A target of
`2.0` generated roughly 12.5 million crack-band tetrahedra and a 1.8 GB mesh,
then exhausted serial WSL memory. That stricter target requires a parallel
workflow and remains mandatory for a later convergence study.

The solver uses `alpha=0` for intact and `alpha=1` for fully damaged material.
Mechanics uses `g(alpha)=(1-alpha)^2+kappa`. A
volumetric/deviatoric tensile history drives AT2 damage, while mechanics uses
the hybrid full-stiffness degradation. This suppresses compression-driven
damage but does not provide unilateral contact. Each load step alternates:

1. degraded linear mechanics with gravity, water and spring supports;
2. history update by pointwise maximum tensile energy;
3. PETSc TAO trust-region damage minimisation with
   `alpha_previous <= alpha <= 1`;
4. a convergence check on displacement and damage changes.

Nonconvergence, invalid placement, exhausted refinement, PETSc failure and
nonfinite energy all fail the job explicitly. The final report includes an AT2
homogeneous tensile-strength estimate because `E`, `Gc` and `l` jointly imply a
strength that can differ from `material.concrete.tensile_strength_pa`; this is
a calibration warning, not an automatic material-property rewrite.

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

The dam solve uses small-strain isotropic elasticity, CG/GAMG with rigid-body
near-nullspace modes, gravity, upstream pressure, tailwater and crest surcharge.
Foundation springs act on tag 1. **The current Elmer SIF grounds wall-side tag 8
with directional springs; it does not couple wall displacement to plinth
displacement.** The adapter reproduces that behavior, leaves tag 9 free and does
not silently bond the regions. Revising this physical assumption would be a
separate model change requiring a new reference.

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
   export/inspect geometry and tags in ParaView. Preserve separate coincident
   wall/plinth nodes and tags `WALL_PLINTH_WALL`/`WALL_PLINTH_PLINTH`.
3. **Elastic dam parity (provisional checks passed)**: reproduce the current Elmer foundation and wall/plinth
   directional springs, gravity, hydrostatic pressure, tailwater and overflow.
   Do not silently turn the joint into a bonded or fixed support. Compare matching
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
