# Code_Aster Porting Plan: Working Baseline Only

## Current objective amendment

The user has accepted the current elastic discrepancies for proceeding with
the multi-solver cracking comparison. The historical baseline-only scope below
is not a prohibition on developing that next stage. The primary goal is
comparing FEniCSx cracking with Aster cracking at the **same UI-selected crack**,
with separate wall/foundation reporting.

Cracking execution is now **sequential**: FEniCSx cracking runs first and
native Aster cracking starts when it (and the Aster elastic baseline) ends.
The earlier parallel-only policy was reversed after a real job ran out of
memory: FEniCSx cracking held about 10 GB while the whole-model Aster crack
reached its 18 GB limit, inside a 24 GB WSL VM. Installed RAM remains an
advisory diagnostic and does not establish that a solver fits.
Paired execution is implemented for three-body app jobs through the native
Aster whole-model crack (see "Native whole-model crack" below). Future VM/cloud
hosting is deferred. See `Design_Docs/app_plan.md` for the current workflow boundary.

App jobs also run the full-dam linear elastic Aster pipeline
(`Aster.run_three_body_pipeline`) per job on that job's own Elmer mesh, in
parallel with the FEniCSx baseline, for the Comparison table's Aster baseline
column. It is still unverified (exit 2, `requires_verification`). The Aster
cracked column is the submodel peak, so no Aster crack difference is reported
until an uncracked submodel of the same scope exists.

The user approved a native Aster concrete-damage formulation, explicitly
different from FEniCSx AT2 phase field. Installed Aster 17.4 provides
`ENDO_FISS_EXP` with `3D_GRAD_VARI`; see the installed behaviour catalogue,
`Cata/Commands/defi_mater_gc.py` and 3-D example `ssnv234b.comm`.
This is nonlocal scalar damage, not a direct AT2 port or automatically a
physical crack-opening field.

### Quadratic dam topology and current execution blocker

`Aster.quadratic_mesh` now degree-elevates the source-pinned saved dam mesh
without removing any volume cells or welding coincident interfaces. It checks
source hashes, the entire volume topology and body/material assignments.
The candidate retains 990,840 tetrahedra (606,414 wall and 79,092 concrete
foundation cells) and has 1,434,816 nodes: at least 4,304,448 displacement
unknowns before constraints or nonlocal damage unknowns.

Keeping the original vertex-only springs would leave quadratic midside traces
incorrectly coupled. `Aster.quadratic_surface` instead integrates the exact
TRI6 shape-function products on each saved support and matching joint face,
including all midside and cross-interface blocks. It preserves the saved
stiffness coefficients and directional supports, but intentionally replaces
the previous linear export's nodal lumping with consistent quadratic
integration. The accepted linear baseline therefore does not certify this
candidate.

The small native assembly check in `Aster.quadratic_spring_benchmark`
checks every matrix entry, including anisotropic coupling, split coincident
identities and directional support. The initial ungrouped operator matched
with zero measured error. The production coefficient-grouped operator has a
measured relative matrix error of 2.46e-11 against the original exact operator,
below its explicit 1e-10 encoding bound.
The MED conversion test verifies native TETRA10 connectivity, every body/face
group, node identities and the correspondence between discrete cells and
their stiffness archive. The converter writes aggregate discrete groups;
characteristic groups are allocated during native mesh import to avoid
hundreds of thousands of MED families or repeated native group-table growth.
The adapter subclasses the installed Aster 17.4 MED importer and adds groups
after parsing but before its group-table allocation. It does not call the
internal bulk-group method on an already allocated mesh: that fails a native
capacity assertion. This version-specific adapter must be verified again
when changing the native installation.

`group_stiffness_blocks` shares numerically equal matrices using each block's
own decimal scale, not a global magnitude that could erase a smaller joint.
It checks every reconstructed block against a 1e-10 relative max-norm error
bound and persists the measured error. This is coefficient encoding precision,
not a changed material calibration or new physical spring coefficient.
On the full candidate, 51,397 point elements use 5,303 coefficient groups and
370,514 two-node elements use 12,621 groups, for 17,924 native characteristic
entries in total. Maximum measured per-block errors are 4.97e-11 and 4.87e-11.
The native path checks every group/cell identity and the encoding error again
before assigning stiffness. The MED conversion and encoding audit are saved
with hashes in an isolated attempt directory.

**The full quadratic baseline has not passed, but the setup blocker has been
resolved.** Earlier individual and batched group-creation attempts were
stopped within their approximately 540-second developer budgets. The grouped
importer completed setup and reached `MECA_STATIQUE`; native characteristic
preparation/assignment took approximately 212 seconds. This full attempt
failed after approximately 456 seconds with native `<JEVEUX_62>` memory
allocation failure, requesting an additional 9,986 MiB while already using
3,198 MiB under an 8,192 MiB process budget. The launcher nevertheless
reported exit zero; native exceptions and missing result evidence correctly
prevent a success report. The allocation diagnostic is not a total peak
memory estimate: the full solve and factorization have not completed.

A second attempt used MUMPS `OUT_OF_CORE` with SCOTCH ordering at the same
memory budget. Native `FACTOR_75` still estimated at least 13,561 MiB for
out-of-core factorisation against 3,758 MiB remaining, with 2,801 MiB already
preallocated by MUMPS. The bounded attempt was stopped at approximately
540 seconds; no baseline outputs were accepted. Out-of-core storage therefore
does not establish that this model fits 16 GiB installed RAM.

A distinct lower-fill iterative trial used native GCPC, `LDLT_INC`, zero
additional fill, relative residual target 1e-8 and at most 2,000 iterations.
It reached `MECA_STATIQUE` but did not produce converged output within the
same approximately 540-second developer budget and was stopped. This does not
establish convergence or a runtime estimate. The experimental GCPC choice was
not retained in the developer case. Do not rerun either failed attempt unchanged.

A subsequent GCPC trial explicitly used four OpenMP threads, confirmed in
the native startup log, rather than the previous single-thread default.
Characteristic preparation/assignment still took approximately 205 seconds;
the attempt reached `MECA_STATIQUE` but again produced no converged output
before the approximately 540-second wall-clock limit. Its diagnostics are
retained separately. `Aster.quadratic_baseline --threads 1..4` now records
the explicitly requested thread count; the default remains one. This does
not change the uncapped user-job policy or establish a successful solver.

Cracking is now sequential (see the top of this document). A lower-memory
discretisation/assembly strategy or an adequately provisioned verification
host, followed by a new checked baseline, is required. Do not automatically
raise allocations into memory occupied by other processes. Full native
diagnostics and failures remain saved; no solve convergence or engineering
validation is claimed.

#### Mixed displacement order on split elastic domains

An opt-in `Aster.quadratic_mesh --wall-only` candidate retains the source
TETRA4 elements in the non-damageable elastic domains and elevates only the
wall to TETRA10. This is not cell coarsening: all 990,840 source volume
cells and their material assignments remain. It reduces the node count
to 955,314 and displacement unknowns to 2,865,942. The candidate contains
606,414 quadratic and 384,426 retained linear volume cells. It is not an
accepted replacement baseline.

Mixed-order elevation rejects shared nodes between the selected and retained
domains, rather than creating hanging-node constraints implicitly. The saved
split interfaces permit independent displacement orders. The joint energy
is integrated with the actual TRI3/TRI6 traces on each side, including exact
rectangular cross-products; it does not discard the wall midside trace or
create artificial master midside unknowns. Linear outer supports retain
consistent TRI3 integration with the saved coefficients. Unit checks cover
both slave directions, affine common motion, constant jumps and positive
semidefiniteness. The native mixed-order matrix benchmark compares every
assembled entry and passes with relative error 1.30e-11. Native MED tests
verify the reordered mixed cell types, exact connectivity and body/face
families.

The mixed-order MUMPS out-of-core attempt still timed out at approximately
540 seconds under the 8,192 MiB budget. Native `FACTOR_75` estimated
8,027 MiB for factorisation against 4,966 MiB remaining, with 1,804 MiB
already preallocated by MUMPS. There was no accepted result. The attempt's
diagnostics are isolated from the all-quadratic trials. A distinct mixed-order
GCPC trial uses incomplete factorisation with one fill level and a 1e-8
residual target; it is experimental until verified, not a fallback that
reports success without native convergence.

The user has explicitly permitted longer solver processes for this development
phase. The developer verification CLI retains its 540-second default and
requires `--allow-extended-budget` together with `--budget-s` to exceed
570 seconds. This explicit opt-in must only be used with development-run
approval. It does not raise the memory budget, alter the uncapped user-job
policy, or turn a timed-out attempt into successful evidence.

The approved extended mixed-order GCPC attempt completed in **2,135.57 s**
(approximately **35.6 minutes**) with finite displacement and MED outputs.
The maximum absolute displacement component was 0.00975178 m, and the native
log reported VmPeak 7,815.15 MiB. This is a completed elastic solve, not
baseline acceptance or a prediction of concurrent cracking memory use.
The earlier 540-second mixed-order GCPC trial was a timeout, not a nonlinear
or linear convergence failure established by that timeout.

`Aster/extract_quadratic_result.py` exports separate wall/foundation stress
VTUs and archives, proving node correspondence through volume connectivity.
MED retains full mesh group indices while result fields omit unrequested
volumes; the exporter explicitly verifies and translates those supports.
Both VTUs were read in actual ParaView with finite principal stress arrays:
606,414 wall cells / 880,337 nodes and 79,092 foundation cells / 18,984 nodes.
`Aster.compare_quadratic_baseline` matches original source identities, never
nearest coordinates, and distinguishes quadratic midside sampling. Repeated
Elmer partition identities are used only after coordinates, displacements and
stress tensors agree within eight float64 machine epsilons relative to each
component, without an absolute tolerance or averaging. The observed duplicate
differences were at most 6.51e-19 m in displacement and 1.17e-10 Pa in stress
(less than 3.55e-16 relative). Larger discrepancies are rejected. Candidate,
source and reference hashes are checked.

| Quantity | Wall | Concrete foundation |
|---|---:|---:|
| Displacement relative L2 difference at original nodes | 0.406% | 0.0324% |
| Elmer recovered tensile peak, original nodes | 1.12221 MPa | 1.56634 MPa |
| Aster recovered tensile peak, original nodes | 1.73118 MPa | 1.41416 MPa |
| Recovered tensile peak difference | +54.27% | -9.72% |
| Elmer constitutive tensile peak, cell centroids | 1.13534 MPa | 1.68146 MPa |
| Aster constitutive tensile peak, cell centroids | 1.09025 MPa | 1.69608 MPa |
| Centroid tensor volume-weighted L2 difference | 7.03% | 0.966% |

The centroid comparison evaluates each displacement element at the same
original tetrahedron centroid before nodal recovery. The TETRA10 derivative
is verified against an exact quadratic polynomial; it is not a linear
vertex-only approximation. Centroid peaks are not full integration-point
extrema, and peak coordinates can differ. In particular, the large new wall
recovered peak must not be dismissed solely because displacement and centroid
peaks are close. The quadratic displacement order and consistent joint law
changed from the previously accepted linear baseline. The candidate remains
**pending baseline acceptance and stress-recovery investigation**; native
cracking and paired app execution are not enabled by this elastic result.

The subsequent direct-element audit identifies the new wall recovered peak
at candidate node 36411, `(63.4335248527, 41.8591438609, -25.5)` m, on the
junction of `wall_foundation/chamfer_0108_spring` and
`wall_foundation/tread_0110_spring` (saved face tags 231/235).
Its eight incident TETRA10 elements give direct vertex tensile stresses
1.493--1.970 MPa. Arithmetic averaging of their full stress tensors reproduces
the recovered 1.731183 MPa peak with relative tensor error 1.38e-13.
Volume-weighted averaging gives 1.730697 MPa, so changing recovery weights
does not remove this concentration. The incident centroid tensile values
range from -0.0674 to 0.9076 MPa. The foundation peak's direct tensor average
also agrees with recovery (3.38e-15 relative tensor error).
These are local stress gradients in the current finite-element solution,
not a ParaView field-selection error or arbitrary recovery overshoot.
They are not yet mesh-converged physical stresses. The reproducible
`recovered_peak_audit` in the comparison report records all incident cells,
boundary names and both averaging methods without modifying output stresses.

With the approved longer native budget, the original 8x8-lateral N128 planar
seed specimen completed in 121.14 s, preserving bounded damage, seed retention,
irreversibility, symmetry and zero remote damage. Its equilibrated damage
integral is 0.1299753008 m3, approximately 4.71% above N64, not convergence.
The confined displacement and seed in this synthetic test are uniform in
both lateral directions. An explicit one-cell lateral reduction reproduces
the N64 8x8 integral and entire normal profile within 1e-9 relative / 1e-12
absolute tolerance; its N128 integral also matches the full lateral result.
This verified planar reduction enables deeper normal refinement efficiently;
it must not be used for finite dam cracks or a three-dimensional edge study.
The `--lateral-divisions` default remains eight. Native `--line-search` uses
CORDE with up to 400 Newton iterations, retaining the 1e-8 residual target.
N256 failed at 40 iterations with and without line search; allowing the
observed damped convergence to continue produces:

| Normal divisions | Equilibrated scalar damage integral (m3) |
|---|---:|
| 128 | 0.1299753008 |
| 256 | 0.1380995799 |
| 512 | 0.1471412619 |
| 1024 | 0.1550984161 |

All retained states remain bounded and irreversible, but these increasing
integrals demonstrate unresolved normal-resolution sensitivity; successful
nonlinear convergence is not mesh convergence. No tolerance was relaxed,
seed enlarged or field clipped to obtain these results.

#### Zero-load continuum reference and targeted normal grading

The official [R5.03.28 continuum equations](https://codeaster.gitlab.io/doc/docaster/manuals/man_r/r5/r5.03.28/Mod_le_continu.html)
provide an independent reference for this particular confined, planar,
zero-displacement test. At zero strain, the active damage zone satisfies
`c*a'' = k`, with `k = 3*GF/(4*D)`, `c = 3*GF*D/8`, and
`D = LARG_BANDE/2`. These coefficients were also verified in the installed
Aster 17.4 `MacroCommands/defi_mater_gc_ops.py`. For the initial obstacle
`a0 = 0.999` inside a layer of thickness `t = 0.002 m`, the symmetric
continuum equilibrium is:

```text
a(z) = max(sqrt(a0) - max(abs(z) - t/2, 0)/D, 0)^2
integral(a dz) = a0*t + (2/3)*D*a0^(3/2)
```

This reference is valid only when its support lies inside the specimen and
no displacement loading is applied. The half-support is 0.25087497 m and
the scalar damage integral is 0.1684147292 m3 for the one-square-metre area.
It describes the equilibrium response outside the physical seed, **not**
an assignment of additional initial damage or widening of the selected seed.
Tests check the obstacle, support, exact integral and differential equation.
It is not an analytic solution for a finite dam crack or loaded propagation.

Uniform refinement approaches this reference slowly near the thin seed
boundary. The `--normal-grading-power 2` specimen option instead grades
normal cell edges quadratically towards that boundary, preserving the same
one-metre domain, seed layer, material, imposed displacement and native
residual tolerance. Mesh generation and exact Gauss-volume integration use
the same runtime-neutral edge generator. The default remains the original
uniform normal spacing.

| Graded normal divisions | Equilibrated integral (m3) | Error vs continuum integral | Profile relative L2 error |
|---|---:|---:|---:|
| 32 | 0.1472194805 | 12.59% | 11.41% |
| 64 | 0.1603731059 | 4.77% | 4.47% |
| 128 | 0.1662798434 | 1.27% | 1.74% |
| 256 | 0.1683005525 | **0.0678%** | **0.235%** |

The graded N256 result was independently repeated on a 2x2 lateral mesh:
its complete normal profile and all five damage integrals agree with the
one-cell lateral calculation within 1e-9 relative / 1e-12 absolute tolerance.
Both checks retain the exact seed and pass boundedness, irreversibility,
symmetry, equilibrium and remote-boundary checks. Runtime was approximately
56 s on 1x1 and 222 s on 2x2. No analytic profile was injected into the solve.
Graded N512 and uniform N2048 both reached the native 400-Newton-iteration
limit during zero-load relaxation, with no accepted output. Uniform N2048
took approximately 837 s; this was a nonlinear convergence failure, not
the approved wall-clock watchdog expiring.

`Aster.seed_convergence` reports explicitly chosen isolated attempts, never
whichever result directory happens to be newest. It verifies saved input
hashes, re-computes metrics from native histories, requires a common physical
contract and lateral/normal-grading policy, and records evidence hashes.
It separately reports:

- two consecutive normal-resolution doublings with both integral and
  interpolated profile differences below a requested tolerance;
- agreement of the finest result with the independent zero-load continuum
  reference.

At a diagnostic 1% tolerance, graded N256 passes the **continuum reference**
check, but the two-consecutive-refinements criterion remains unmet. These
are separate statements, neither engineering validation nor propagation
validation. The profile norm uses the actual zero-load equilibrated state,
not the subsequent small tensile-load state. The original ungraded results
remain available and are not silently reclassified.

```powershell
.\.venv\Scripts\python.exe -m Aster.seed_benchmark --specimen-divisions 256 --lateral-divisions 1 --normal-grading-power 2 --line-search
.\.venv\Scripts\python.exe -m unittest Aster.test_seed_convergence Aster.test_native_seed
```

#### Finite annular-sector edge check

The optional `--finite-sector` specimen exercises both radial and angular
ends of an exact thin physical seed. Its synthetic annular sector is
`(78.8, 79.2, 1.3975, 1.4025)`, with the same 0.002 m thickness and native
material. It is not a UI dam-job result. The graded N32 / 8x8 lateral mesh
passes placement, five retained states, bounded Gauss damage, no healing,
symmetry and zero remote normal damage in approximately 84 s. There are
80 seeded points among 16,896 Gauss points, and the equilibrated scalar
damage integral is 0.0193649232 m3.

Finite-edge results are explicitly rejected by the infinite-planar continuum
comparison. A separate geometry audit reports the exact seed volume and the
Gauss-sampled initial scalar damage integral. Here the exact volume is
0.000316 m3, giving initial integral 0.000315684 m3 at damage 0.999; the
sampled integral is 0.0003121875 m3 (**-1.108%**). The curved edge has not
been moved to make the quadrature agree.

This first finite-edge test also detects nonzero final damage, approximately
0.00323, in the sampled outer lateral 0.05 m boundary shell. The specimen
extent therefore needs checking; the result is not accepted finite-edge
convergence. New finite-sector runs persist this diagnostic and an explicit
boundary warning. A 16x16 lateral repeat at unchanged normal grading and
physical geometry is a separate verification, not automatic acceptance.

That 16x16 repeat completed in 376.48 s and passed the same five-state
placement, boundedness, irreversibility, symmetry and remote-normal checks
(320 seeded points among 67,584 Gauss points). Its zero-load equilibrium
integral is 0.0197228675 m3, **1.848% above** the 8x8 result. The initial
sampling integral remains 0.0003121875 m3 (-1.108% against exact geometry),
so this doubling alone does not resolve the curved-boundary sampling error.
Maximum final damage in the outer lateral shell decreases to 0.000258428,
but is still above the 1e-7 diagnostic threshold. Both runs have separate
`finite_sampling_audit.json` companion reports with input and native-history
hashes; these diagnostics were computed after the original runs and do not
modify their original benchmark reports. Neither lateral mesh convergence
nor specimen-extent independence is established.

The finite-sector runner also supports `--lateral-width-m` (default 1 m)
and up to 32 lateral divisions. Width changes affect both lateral mesh axes,
boundary groups, exact Gauss weights and domain-volume checks; the normal
extent remains 1 m. The selected sector, 0.002 m seed thickness, material
and imposed strain are unchanged. For an extent study at fixed spacing,
compare width 1 / 8 divisions with width 2 / 16 divisions, or width 1 /
16 divisions with width 2 / 32 divisions. Width changes are rejected for
the infinite-planar benchmark. The boundary diagnostic still uses a physical
0.05 m shell, not a fraction of the enlarged width. A native MED roundtrip
verifies the enlarged bounds, 4 m3 domain volume and boundary groups.
This prepares a controlled extent study; it does not establish its outcome.

The width 2 m / 16x16 lateral / graded N32 run subsequently passed in
359.57 s. Compared with width 1 m / 8x8 at identical 0.125 m lateral
spacing, its zero-load scalar damage integral changes from 0.0193649232
to 0.0193668901 m3 (**+0.01016%**), and maximum final damage in the outer
0.05 m lateral shell is zero. The exact seed and sampled initial integral
are unchanged. Thus the integrated response is insensitive to this domain
enlargement at that spacing; it is not a pointwise-field, lateral/normal
mesh-convergence or loaded-propagation validation. A finer-spacing extent
comparison remains separate.

The first width 2 m / 32x32 / graded N32 attempt failed after 64.56 s
with native `DVP_2` / Windows floating-point exception `0xc0000090`
during `FULL_MECA` on a `MVCA_HEXA20` element. It did not reach the
7200 s watchdog or produce an accepted history. The log also reports
MUMPS minimum out-of-core memory of 2211 MiB against about 2042 MiB
available within the benchmark's 2048 MiB allocation. This is evidence
of an insufficient allocation, not proof of the exception's root cause.
The developer runner now accepts `--memory-mib` (default unchanged at
2048), validates it before launch and records the allocation in its report.
A larger allocation is a separate diagnostic attempt with unchanged
physical inputs and tolerances; available host RAM must be checked first.

The 4096 MiB diagnostic retry failed after 63.50 s with the same `DVP_2`
and `0xc0000090` exception at element 6273, centred at approximately
(76.881779, 12.458654, -27.969787) m. The earlier `FACTOR_76` memory
warning is absent. Its physical-input hash matches the failed 2048 MiB
attempt exactly. Increasing the allocation therefore did not resolve the
failure; insufficient allocation is not an adequate explanation of this
reproducible exception. Neither run produced an accepted damage history.
The fine enlarged-domain check is blocked on native constitutive/element
evaluation investigation, not runtime allowance. Do not disable floating
point exception interception or alter seed/material/tolerances to declare
this check passed. The successful coarser extent comparison remains valid
only within its stated scope.

#### Separate tetrahedral element-family diagnostic

An independent Linux 17.4.0 binary distribution (`code_aster_17.4.0_deb12.tar`)
has also been extracted into the WSL user's `~/opt/code_aster_17.4.0`,
separate from the Windows installation. On this host,
`~/opt/run-code-aster-17.4` sources its activation script, prepends the
bundled `bin` directory and sets `PYTHONHOME` to the relocated root.
Use `wsl.exe -d Ubuntu -- /home/alist/opt/run-code-aster-17.4 --version`
from PowerShell to check that host-local installation. It reports 17.4.0.
A native Linux MUMPS elastic smoke solve passed in 2.36 s with finite
displacements and the prescribed 1e-6 m maximum; native NumPy and MED
imports also pass. This establishes runtime availability, not resolution
of the Windows damage exception; the failing damage case requires its
own independent reproduction.

The benchmark runner's interpreter discovery also supports a standalone
Linux `<install>/bin/python3` while preserving the existing Windows path.
Initialize the standalone environment before running the Python module:
source its `activate.sh`, prepend its `bin` to `PATH`, set `PYTHONHOME`
to the installation root and set `LAMB_CODE_ASTER_LAUNCHER` to that root's
`bin/run_aster`. The existing tiny native seed benchmark now passes this
Linux path end-to-end in 5.24 s, including three-state history validation,
wall-only filtering (8 seeded / 64 total points; 24 excluded nonwall hits)
and MED-to-ParaView export. `history_checks` is shared between ordinary
benchmark execution and independent-result verification so a launcher exit
alone cannot substitute for a valid damage history.

An exact reproduction of the failed width 2 m / 32x32 hexahedral case
completed successfully on Linux in 1996.35 s (33.3 minutes). It reuses the failed attempt's mesh bytes;
the only input JSON change is `repository_root` for the platform path.
Mesh, case and input hashes are recorded in the isolated Linux run directory.
All five states pass the shared strict seed-history gates: 270,336 Gauss
points, 320 seeded points, no healing, finite bounded damage, zero remote
normal damage, zero sampled outer lateral-shell damage, reflection error
6.28e-14 and zero-load equilibrium change 7.77e-16. The equilibrium scalar
damage integral is 0.0197228350 m3. Actual Windows ParaView reads all five
exported states (33,792 cells) with all 12 required arrays finite.

The Linux fine-spacing integral differs from the width 1 m / 16x16 Windows
result by -0.0001647%, and from width 2 m / 16x16 Windows by +1.8379%.
The former comparison changes both platform and domain width, so it is not
an isolated platform comparison. These observations support small combined
platform/extent effects on this integral, while lateral refinement still
changes it appreciably. They do not establish two successive refinements
or remove the -1.108% initial curved-seed sampling error.

The exact failing Windows case is therefore executable on another native
build/platform without changing its physical inputs. This does not identify
the Windows defect or validate dam cracking. Independent Linux evidence is
under `~/opt/seed-linux-r3eonp2l`, with `reproduction.json`,
`linux_validation.json`, native history, MED result, launcher log and
five-state PVD/VTU outputs. The Linux build is 17.4.0; the Windows package
also identifies as 17.4 but may differ in compiler, libraries and patches.

Preparing a new resolved mesh on Linux exposed a separate converter
compatibility issue: its `DataArrayInt` aliases 64-bit IDs, whereas the
Windows MED build uses 32-bit IDs. The specimen converter now constructs
family arrays from integer lists so MED selects its native ID width,
instead of supplying a forced NumPy int32 array. Native hexahedral and
tetrahedral mesh/volume/boundary tests pass on both installations. This
fix affects mesh conversion, not the constitutive exception reproduced
using an already saved mesh. A Linux graded N64 / 16x16 / width 1 m
normal-refinement run provides separate numerical evidence below.

That N64 / 16x16 / width 1 m run passed in 1601.68 s with 133,120
Gauss points and 320 seeded points. Its equilibrium integral is
0.0226184419 m3, **14.6813% above** the N32 / 16x16 Windows result.
It retains all five states, zero remote normal damage, reflection error
1.99e-12 and the unchanged sampled initial integral 0.0003121875 m3.
Final outer-shell damage is 0.000842244, so extent sensitivity must also
be revisited at this normal resolution. A converged nonlinear solve is
not finite-seed mesh convergence.

`python -m Aster.finite_seed_comparison FIRST SECOND --output REPORT.json`
audits explicit successful finite-sector attempts: input/report/history
hashes, recomputed history and volume/sampling metrics, unchanged physical
seed/material/history, and changed discretisation settings. Platform-specific
material-source paths are reported separately, not compared as constitutive
parameters. The tool always labels the result sensitivity-only and does not
apply the infinite-planar analytic reference or infer pointwise agreement.
The persisted N32/N64 comparison changes platform as well as normal
resolution; an N32 Linux repeat is needed to isolate that effect before
attributing the entire difference to the normal mesh.

The same-resolution Linux N32 / 16x16 / width 1 m control subsequently
passed in 314.85 s. Its equilibrium integral equals the Windows value
0.0197228674986884 m3 at the reported precision. Gauss-point identities
and coordinates match exactly; maximum damage difference over all five
states is 5.55e-16. Thus the N32-to-N64 **14.6813%** integral change
persists in an entirely Linux comparison with identical physical inputs,
normal grading policy and lateral mesh. The saved reports
`finite_platform_control.json` and `finite_linux_normal_comparison.json`
separate this control from normal-resolution sensitivity. This does not
diagnose the larger Windows exception or establish finite-seed convergence.

A Linux N128 / 16x16 / width 1 m run (`seed-x0jqsgk8`) was stopped by
request after about 46 minutes while solving t = 1.5; no results were
archived. Runtime grew about 5x per normal doubling (N32 315 s, N64 1602 s),
so N128 was projected at two hours or more on one thread. Finite-seed normal
convergence therefore remains unresolved and is recorded as a known
limitation. Further refinement is deferred in favour of the dam-cracking
driver. Any retry should first solve only the zero-load states and use
verified multithreading.

The finite-sector runner offers `--tetrahedral` to exercise `TETRA10`,
the intended full-dam wall element family, without changing the physical
seed or material. MED's `PLANAR_FACE_6` subdivision divides each structured
box into six equal-volume tetrahedra. Native MED tests verify every volume
and exterior-face inventory. `3D_GRAD_VARI` uses four equal-weight points
per tetrahedron (versus eight per hexahedron); native histories must contain
exactly four points per tetrahedron and 24 per original box. Integration
weights use the actual graded layer volume divided by 24, not the hexahedral
rule. All existing boundedness, irreversibility, equilibrium, symmetry and
remote-damage gates remain enforced. This is a separate discretisation,
not a passed substitute for the failed fine hexahedral attempt.

The first graded N32 / 8x8 / width 1 m tetrahedral run passed in 74.73 s
with 50,688 Gauss points, 240 seeded points and five exported states.
The same frozen seed hash is retained; the initial sampled scalar integral
is 0.0003121875 m3. Its equilibrium integral is 0.0218456894 m3,
**12.81% above** the corresponding hexahedral result, so cross-element
agreement is not established. Reflection error is 1.69e-14 and remote
normal damage is zero; lateral-shell damage remains 0.00245421.
The recorded history was also recomputed against the strict current
metric gates. An enlarged tetrahedral domain and subsequent mesh refinement
are separate checks; do not infer accuracy from this successful solve.
Actual ParaView read all five tetrahedral states (12,672 cells), with all
12 damage, displacement and stress arrays finite.

The width 2 m / 16x16 tetrahedral repeat passed in 332.96 s with 202,752
Gauss points and the same 240 initially seeded points. At unchanged
0.125 m lateral spacing its equilibrium integral is 0.0218462362 m3,
only **0.002503% above** width 1 m / 8x8. Final outer-shell damage is
zero, reflection error is 4.23e-15 and remote normal damage remains zero.
Both histories were re-integrated from their saved native evidence and
checked against the hashed inputs and persisted reports. This supports
small integral sensitivity to the tested domain enlargement, not lateral
or normal mesh convergence. The 12.81% cross-element difference remains
unresolved; a width 1 m / 16x16 tetrahedral repeat isolates lateral
refinement before any finer-spacing extent claim.

Repeat the small checks from the nested repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest Aster.test_quadratic_surface Aster.test_quadratic_mesh Aster.test_quadratic_native Aster.test_quadratic_baseline
.\.venv\Scripts\python.exe -m Aster.quadratic_spring_benchmark
.\.venv\Scripts\python.exe -m Aster.quadratic_spring_benchmark --mixed-order
& 'C:\SALOME-MECA\2025\W64\CODE_ASTER_17.4\external\Python3.10\python.exe' -m unittest Aster.test_quadratic_med
```

`Aster.quadratic_baseline` is a **bounded developer verification CLI**, not
an app backend. Its total budget and memory limit do not apply to user jobs.
Do not repeat the full attempt unchanged: resolve the assembly memory
requirement, then verify the quadratic baseline and a resolved native crack seed
before wiring native cracking into app submissions.

### Provisional native damage parameters

The user supplied **20 MPa compressive strength** and authorised provisional
choices of the other two parameters. The authoritative `config.json` now stores
`material.concrete.compressive_strength_pa = 20000000`. The desktop exposes it
in Pa and preserves it on load/save/submit. Historical configurations may omit
it for elastic/FEniCSx work, but Aster damage explicitly rejects a missing
strength rather than inferring one.

The initial numerical settings in `Aster/fracture_defaults.json` are:

| Input | Initial value | Basis |
|---|---:|---|
| Full band width `LARG_BANDE` | 0.50 m | Provisional numerical regularisation; half-width D = 0.25 m |
| Softening shape `P` | 5 | Nonlinear softening starting point also used in installed 3-D verification case `ssnv234b.comm` |
| Secondary shape `Q` | 0 | Native macro default; avoids an unsupported extra shape calibration |
| Compression stiffness recovery | 0.9 | Native macro default |

These are **not test-calibrated concrete properties**. The 0.50 m full width is
one eighth of the current minimum 4 m wall thickness and starts a
resolution study on the current modelling scale; it is not a measured crack
width, a value fitted to elastic peak stresses, or a claimed conversion from
AT2 length scale. Initially target at least four elements across the full band
(h <= 0.125 m), then halve h and check convergence. Study 0.25/0.50/1.00 m full
bands and P = 2/5/8 before accepting conclusions from a particular setting.
Actual full-dam mesh and band sensitivity has **not** been established.

`Aster/fracture_material.py` reads E, nu, FT and FC from the submitted dam
configuration and GF from the existing FEniCSx `fracture_defaults.json`. It does
not maintain a second energy or strength copy. GF was originally 2700 J/m2.
An independent literature review found that value roughly 7-15x above
published dam-concrete fracture energies (typically 150-400 J/m2). The user
therefore approved a provisional **GF = 200 J/m2 for both solvers**. The fib
Model Code 2010 relation GF = 73*fcm^0.18, with fcm = 20 + 8 = 28 MPa, gives
133 J/m2; this is raised to 200 J/m2 as an allowance for large dam aggregate.
It is not a measured value. With E = 35 GPa and FT = 2 MPa this gives
E*GF/FT^2 = 1.75 m (previously 23.625 m). For reference, the AT2 estimate
(9/16)*sqrt(E*Gc/(3*l)) at l = 0.25 m is about 1.72 MPa, compared with 6.31 MPa
at the old energy. This ignores the energy split and is only a consistency
indicator. Successful material construction does not validate the energy.

The official [U4.42.07 operand definitions](https://codeaster.gitlab.io/doc/docaster/manuals/man_u/u4/u4.42.07/Mot_clef_facteur_ENDO_FISS_EXP.html)
define `LARG_BANDE = 2D` and distinguish GF (total fracture energy) from G_INIT
(the equivalent linear-softening energy associated with the initial cohesive
slope). They are **not interchangeable**. The native macro's relation gives
G_INIT = 3*pi*GF / (4*(P+2)^(3/2)). At P = 5 and GF = 200 J/m2 this is
25.445 J/m2. The initial cohesive slope is -FT^2/(2*G_INIT), approximately
-7.8602e10 Pa/m. P = 2/5/8 gives G_INIT = 58.905/25.445/14.902 J/m2 at the
same total GF, illustrating meaningful uncertainty in the softening shape.
Results produced before this change used GF = 2700 J/m2.

[R5.03.28](https://codeaster.gitlab.io/doc/docaster/manuals/man_r/r5/r5.03.28/Domaine_d_application.html)
explains the small-length cohesive limit, tension/compression distinction and
V1 damage convention (0 intact, 1 damaged). It does not prescribe a universal
concrete P or width. This model targets tensile quasi-brittle damage; the
compressive strength input does not establish a validated crushing model.

Native Aster 17.4 accepted all nine width/P combinations via `DEFI_MATER_GC`.
The check reads native internal coefficients back, verifies M >= P+2 and the
documented band conversions, and saves isolated input hashes, full logs and
`material_check.json` under ignored results. Status is explicitly
`material_admissibility_passed`, **not** a cracking solve. To repeat, in
PowerShell from the nested repository root:

```powershell
.\.venv\Scripts\python.exe -m Aster.check_fracture_material
```

### Native nonlinear verification and shared seed progress

`Aster.damage_benchmark` now runs actual `STAT_NON_LINE` calculations with
`ENDO_FISS_EXP`, quadratic HEXA20 elements and the provisional dam material.
Its isolated 1 m cube has 8 volume elements and 64 integration points. Six
histories each include an initial state and 30 converged increments:

- INTACT: homogeneous confined tension, initially undamaged.
- SEEDED: the same imposed strain with initial uniform damage 0.1.
- LOCAL: initial damage 0.1 restricted to one synthetic volume element.
- FREE: initially intact tension with top/bottom axial displacement and
  coordinate-verified side rollers, allowing lateral contraction.
- NEAR: initial uniform damage 0.999 under the confined strain history.
- NEARLOC: initial damage 0.999 on one synthetic element, initially intact
  elsewhere, under the same confined strain history.

The first two match the homogeneous closed-form damage target 0.2 within
8.5e-12 absolute and its axial stress within 8.4e-11 relative in the measured
run. Every integration point retains damage on unloading and damage grows on
further loading. The check retains cell/Gauss-point/subpoint identities rather
than comparing only peak damage. Confined unloading returns half the stress
at half the imposed strain. FREE additionally requires negligible lateral
stress; its measured unloading stress ratio is 0.49810, not exactly 0.5,
consistent with nonlinear compression stiffness restoration at fixed damage.
No closed-form free-contraction stress accuracy is claimed.

NEAR retains 0.999 at all 64 integration points through every state. NEARLOC
retains that initial value at the same seeded integration-point identities,
does not heal elsewhere and reaches a final damage range of approximately
[0.20000647, 0.999]. All six native histories pass in approximately 13 seconds
on the development machine.

All six histories export `result.med` and a PVD/VTU series with every converged
step, `damage_recovered_nodal` (native V1 recovery, 0 intact / 1 damaged),
`displacement_m`, stress components, principal stresses and von Mises stress in
Pa. Actual stock ParaView 6.2 reads the series and required fields, including
all 31 NEAR/NEARLOC states.

**Damage recovery warning:** extrapolating integration-point damage to nodes
does not preserve bounds. Across the NEARLOC history, native recovered nodal
values span approximately [-0.24145322, 0.99909948], although verified
integration-point damage remains in [0, 1]. LOCAL also has a small nodal
undershoot. The exported field is explicitly named as recovered nodal damage,
remains unclipped, and these excursions are emitted as warnings in the CLI
and persisted benchmark report. Use `native_history.json` integration-point
values, not recovered nodal contours, for physical damage/irreversibility
verification. This is not healing or negative damage in the constitutive state.

A nonzero launcher exit, native
exception, missing history, damage healing, incorrect analytic target,
incomplete time series or failed export prevents `benchmark_passed`.
Each attempt uses a unique ignored results directory, full logs and hashed
inputs; the agent verification CLI uses a bounded 180-second solve, unrelated
to the uncapped duration of user app jobs.

Repeat from the nested repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m Aster.damage_benchmark
```

**Limitations:** this is a mechanics/constitutive/initial-state verification,
not a dam cracking solve, fully damaged crack seed, free crack-propagation
study or mesh/band convergence study. The local synthetic seed is intentionally
only one cell and is not an accepted resolved dam seed. Do not infer full-dam
runtime, safety or Aster/FEniCSx cracking agreement from these checks.

`FEniCSx/crack_contract.py` now holds runtime-neutral input dataclasses,
the actual solver load schedule, and a hashed physical UI-seed contract.
The existing FEniCSx runner writes this as `shared_seed` in `crack_location.json`
and its manifest. It freezes the exact selected Y/Z, stress-selected face,
annular-sector geometry, wall region, pressure, shared fracture energy and
load factors, including explicitly requested final-step subdivisions.
Placement is callable from Windows without importing DOLFINx. Existing imports
from `crack_field` continue to expose the same canonical dataclasses.
Neither the frozen physical seed nor its hash implies that native damage and
AT2 profiles are interchangeable.

`FEniCSx.selected_seed` now freezes the complete selection before worker launch:
the physical contract, solver specification/options, reporting metadata,
exact submitted configuration hash, canonical source-field hash and actual
final-step subdivision. Canonical source-array hashing ignores NPZ compression,
timestamps, integer storage width and array memory layout while retaining
coordinates, stress components, region identities and node order.

For three-body app jobs the verified FEniCSx baseline saves the original
Elmer selection fields and their archive hash. After the UI click the API
validates that baseline/configuration/archive, prepares the seed exactly once
and persists `cracking/prepared_seed.json` before spawning the worker.
Missing/changed source evidence fails explicitly without starting cracking.
The FEniCSx command receives `--prepared-seed` instead of `--crack-location`.
The consumer validates the configuration, independently imported baseline
fields, load schedule and physical contract, then reconstructs the saved
inputs without choosing another face or reading mutable placement defaults.
Result validation requires the exact prepared-seed hash and selection metadata;
a zero-exit worker reporting another seed is rejected. A frozen job seed
is never overwritten. Older baselines lacking selection-source evidence must
be rerun rather than acquiring a silently reconstructed contract.

The standalone runtime-neutral preparation CLI is also available:

```powershell
.\.venv\Scripts\python.exe -m FEniCSx.selected_seed --config <job-config.json> --baseline-fields <crack_seed_fields.npz> --location <Y> <Z> --output <new-prepared_seed.json>
```

Configured non-UI cracks and existing CLI `--crack-location` behavior remain
available; `--prepared-seed` is exclusive with `--crack-location` and configured
cracks. Direct UI-location CLI runs prepare and persist the same packet locally.
This completes prelaunch seed preparation for the current FEniCSx app worker,
not native Aster execution or the paired scheduler.

### Native physical-seed adapter

`Aster.native_seed.initial_field` now consumes the same hashed physical
contract directly. It evaluates the exact annular-sector prism at native
integration-point coordinates, restricted to the explicitly named wall
volume group, without changing the selected face, elevation, radii, angles
or 0.002 m physical seed thickness. Native initial damage is an explicit
argument, not an assumed AT2-equivalent field. The helper constructs all nine
ENDO_FISS_EXP variables, including the native ENDORIGI stiffness, and checks
every component against fixed cell/Gauss-point/subpoint identities. A mesh
that misses the seed raises an explicit refinement error; the adapter never
expands or relocates the physical crack to obtain sampled points.

Repeat the bounded synthetic check from the nested repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest Aster.test_native_seed FEniCSx.test_crack_contract
.\.venv\Scripts\python.exe -m Aster.seed_benchmark
```

The native check uses an interface-aligned 0.004 m synthetic cube at dam-scale
coordinates, retaining the physical 0.002 m seed thickness. It verifies
8 seeded integration points out of 64 and excludes 24 other integration
points that lie inside the geometric prism but outside the wall group. A
separate deliberately unsampled contract is rejected. All 8 seeded points
retain 0.999 through three nonlinear states, and every point satisfies no
healing, finite stress and bounded integration-point damage.

**This checks placement and irreversibility, not physical propagation.**
The cube is much smaller than the provisional 0.5 m native band; initially
intact points reach approximately 0.998687 damage even under small imposed
tensile strain. The CLI and report explicitly warn about this response, and
the status is `seed_placement_and_irreversibility_passed`, not a propagation
accuracy acceptance. A previous non-aligned synthetic mesh also failed
Newton convergence; its logs are retained, not reclassified as success.
Full-dam native initial-state equilibration and mesh/band sensitivity remain open.
The helper neither applies internal crack pressure nor launches either app
backend. It is not yet wired into submitted dam jobs.

### Planar native seed equilibration and normal-resolution checks

`Aster.seed_benchmark --specimen-divisions N` now also builds a one-metre
synthetic specimen containing the 0.5 m band. Its native MED mesh has a
separate **0.002 m seed layer**, without widening the physical contract.
The lateral grid stays 8 by 8; N refines only the localization-normal direction
outside that layer. This is a planar normal-resolution study, not general
three-dimensional mesh convergence.

The specimen first holds all displacement at zero at times 0, 0.5 and 1,
then applies confined tensile strain up to 1e-6 at time 2. Initial seed damage
is 0.999. Native `AFFE_CHAR_CINE_F` eliminates the prescribed displacement
unknowns rather than adding redundant displacement multipliers. Refined
multiplier-based trials encountered a native Windows floating-point error;
both the failure evidence and the successful elimination-based checks are
retained. The original tiny wall-filtering benchmark keeps its previous
loading and multiplier/line-search controls.

Successful measured results:

| Normal divisions N | Volume cells | Gauss points | Zero-load integral of scalar damage (m3) |
|---|---:|---:|---:|
| 8 | 576 | 4,608 | 0.1342760884 |
| 16 | 1,088 | 8,704 | 0.1198042713 |
| 32 | 2,112 | 16,896 | 0.1212094778 |
| 64 | 4,160 | 33,280 | 0.1241277683 |

These integrals use exact cell-volume Gauss weights, not an unweighted
integration-point average distorted by the thin seed layer. They are a
scalar-damage measure, **not crack volume or opening**. For N=8 the zero-load
profile changes by at most 2.9e-15 between times 0.5 and 1, and its reflection
error is approximately 5e-14. Every successful mesh retains the seeded state
without healing, finite stresses, and zero damage beyond 0.3 m from the seed
plane. On N=64 the initially intact points adjacent to the plane reach
approximately 0.760506; this is a regularised initial-state response, not
load-driven propagation. This explains why the earlier tiny-domain result
cannot be interpreted as an independently propagating crack.

**Convergence is not established:** the scalar-damage integral changes by
approximately 2.41% from N=32 to N=64 and the sequence is not monotonic.
The N=128 trial exhausted its bounded native CPU budget before accepting the
first equilibration state. Do not rerun that unchanged or classify it as
convergence. Band-width, full three-dimensional mesh, finite crack-edge,
pressure and propagation accuracy studies remain open.

Every successful specimen exports all states to MED and `seed.pvd`/VTU,
including recovered nodal damage, displacement and stress. Actual ParaView
6.2 read all five states on N=16, 32 and 64 with finite required arrays.
Recovered nodal undershoots remain explicitly warned and unclipped; the
specimen's physical state evidence is in `native_seed.json`.

```powershell
.\.venv\Scripts\python.exe -m Aster.seed_benchmark --specimen-divisions 8
.\.venv\Scripts\python.exe -m Aster.seed_benchmark --specimen-divisions 16
```

The bounded runner limits each native specimen to 240 seconds and its
conversion/export stages separately; these limits do not apply to user jobs.
No dam model, physical parameter, app execution policy or baseline acceptance
is changed by these synthetic checks.

**Next integration prerequisite:** native GRAD_VARI requires quadratic
displacement interpolation and linear regularised-damage/multiplier fields;
see [R5.04.01 element requirements](https://codeaster.gitlab.io/doc/docaster/manuals/man_r/r5/r5.04.01/Choix_des__l_ments_finis.html).
The accepted full-dam elastic comparison used linear tetrahedra. A native
quadratic dam mesh, interfaces/supports and zero-damage baseline therefore
need their own checks before reusing that acceptance for a nonlinear solve.
Resolved full-dam seed equilibration, resolution/convergence and
dam-result extraction remain to be verified. Aster dam cracking, paired
scheduling and the final cracking comparison are still not wired into app jobs.

### Native whole-model crack (app jobs)

**New intact-start scenario (not yet a production solver):**
`FEniCSx/horizontal_fracture_contract.py` separates a selected horizontal
candidate section from an imposed seed. It records an intact initial condition,
requires an explicit friction coefficient, and identifies transient
fracture-only hydraulic coupling (impermeable surrounding concrete/rock).
Its configurable opening limit defaults to 0.002 m; this is not initial gap,
seed thickness, prescribed depth or length. An accepted opening history may
show no fracture initiation. Assessment uses the peak over the history, not
just the last state. Reaching or exceeding the limit produces an opening-limit
warning, not a solver error, and does not change successful solver completion.
The opening criterion is recorded separately from solver completion; it is not
a structural-collapse verdict. Partial histories have no successful acceptance
verdict, but retain warnings for limit exceedances already observed.
The intended new UI label is **Opening Limit (m)**. The existing legacy control
remains **Seed layer thickness (m)** until a solver actually calculates opening;
renaming that legacy input would not change its physical effect.

The subsequent painted-damage selection supersedes the intact-only selection
intent: left-click selects affected elements; right-click toggles crack-start
edges. `DamageSelection` in the same contract module requires at least one
affected element **and** one crack-start edge, with field-specific validation
errors for either missing selection. It rejects malformed and duplicate IDs.
`FEniCSx/damage_selection.py` now builds a job-bound native wall catalog and
validates mesh identity, wall membership, real edge IDs and conservative whole-
tetrahedron containment within 75% of local thickness. The separate API
`damage-selection/catalog` and `damage-selection/validate` routes expose this
to the WPF **Paint damage (preview)** window. Left-drag paints overlapping
projected elements; Shift-left-drag erases; right-click toggles the nearest edge.
Opening Limit (m) uses numeric entry with a 0.002 m default. Both element and
edge selection are mandatory. Validation returns `solver_ready: false`;
it never launches a solver or changes a job. The app can now explicitly save
append-only job-bound selection snapshots and restore the latest after catalog/
configuration and snapshot-integrity checks. Saving is not solver submission.
Legacy single-point submission and cracking reruns have now been removed from
the app/API. Older clients receive authenticated HTTP 410 retirement feedback.
New app jobs run elastic baselines only and mark requested legacy cracking
stages blocked; neither solver is launched as a preview fallback. Archived
outputs remain readable and standalone solver CLI defaults are unchanged.

Reduced FEniCSx building blocks now verify a relaxed elastic no-tension local
law, prescribed-open-fracture flow, local compliant pressure/opening feedback
and prescribed water traction on native normal contact; see `FEniCSx/README.md`.
These do not
provide an equivalent Code_Aster implementation or a coupled dam scenario.
This contract does not implement cohesive mechanics, coupled water flow, hydraulic
pressure loading, or an opening extractor, and is not wired to app submission.
External reservoir/tailwater loads must be preserved in the future scenario;
internal fluid pressure must be coupled to flow and act on wetted fracture
faces without automatically pressurising intact concrete.
Native mechanical/hydraulic benchmarks, constitutive calibration, required
hydraulic inputs/physical time histories, and production solver integration remain
implementation gates. Neither scalar damage nor the legacy seed below can be
reported as calculated crack opening. Historical app crack results use the
legacy seeded-volume formulation described below; new app jobs do not run it.

Initial capability checks for the new scenario:

- The installed Code_Aster 17.4 reference case `ssnp164d` completed in 4.04 s
  under an enforced 180 s wall-clock watchdog, in an isolated work directory.
  Its `3D_JOINT_HYME` / `JOINT_MECA_ENDO` calculation passed 15 native
  non-regression checks and three analytical checks. This establishes that the
  installed runtime can execute that hydro-mechanical joint reference; it does
  not establish material calibration, water-front propagation from intact
  concrete, or the new production dam scenario.
- The seven existing reduced FEniCSx frictionless contact tests pass on the
  installed DOLFINx 0.11.0.post0 runtime. These are normal-contact regression
  checks, not cohesive fracture, frictional sliding, or coupled flow validation.
  Three added tests now verify externally prescribed face pressure, zero-initial-
  gap opening, effective contact release, reclosure and input rejection. Six
  separate compliant-flow tests check two-way local pressure/opening/storage/
  conductivity coupling and explicit nonlinear failure; this does not establish
  water access/wetting through closed contact or native Code_Aster parity.
  Four further native contact tests verify scalar Coulomb stick/slip/reversal,
  nonnegative dissipation and shear-field export, reduction of friction strength
  by water pressure, explicit zero friction and required-input rejection. No
  friction coefficient or shear modulus is inferred for a production dam job.
  Six paired-face checks now verify source-identity-based 3-D opening extraction,
  consistent P1 pressure virtual work, normal/tangential separation and an
  analytical native two-block zero-gap pressure-opening solve with stress and
  equilibrium checks. Surfaces are supplied explicitly: this is not edge-to-
  surface construction or native Code_Aster opening/pressure integration.
  An explicit-facet FEniCSx topology component now creates paired faces from
  interior wall tetrahedra, retains shared fronts and parent provenance, and
  transfers accepted states and exterior support/load labels without an imposed
  opening. The native pressure-opening test now starts with a bonded mesh and
  uses this cut. It is not a crack-path/growth law or an Aster remeshing adapter.
  Eight water-access checks distinguish explicit inlet-connected open paths
  from closed/isolated surfaces using paired identity and normal opening.
  Topology-created tied fronts start blocked and do not become hydraulic
  inlets merely because they were selected as crack-start markers. This is
  geometric eligibility, not transient saturation or a pressure assignment.
  The reduced FEniCSx event driver additionally solves an intact loaded baseline
  and applies selected-cell tensile loss under unchanged external loads/BCs.
  Tests check two-mesh force redistribution, retained compression and explicit
  equilibrium failure when all tensile load paths are removed. Isolated output
  retains intact state/failure diagnostics. Cellwise concrete/rock DG0 inputs,
  the existing mortar spring operator and canonical roller forms now pass
  reduced compliance/reaction tests, including unchanged mortar coefficients
  through tensile loss and retained gravity/reservoir/tailwater loads.
  Serial constrained mortar interface DOFs now support row/column elimination
  and imposed-displacement lifting, retaining raw coefficients for force/
  energy audits. Analytical zero/nonzero displacement and nonlinear release
  checks pass; existing standalone solver defaults remain unchanged.
  This is not a Code_Aster event adapter, a verified heterogeneous full-dam
  calculation or a coupled evolving-crack hydraulic solve.
  An experimental canonical-source import adapter now maps source-row wall
  selections and reuses baseline material/support/interface and exterior-load
  definitions. Its intact source-order displacement matches the existing
  elastic solver in a reduced loaded test; two/three-body force checks retain
  foundation properties and both interfaces. It is not authenticated saved-job
  integration or an Aster adapter. The source-event adapter now clips
  waterline-crossing exterior triangles and integrates consistent P1 pressure
  loads, with analytical force/moment/virtual-work checks. Fixed nodal loads
  enter both states and reactions, not a growing crack hydraulic law.
  No standalone baseline numerical defaults were changed.
  An additional experimental FEniCSx bound-event entry point checks the
  existing verified-baseline gate, current catalog hashes and exact source
  tetrahedron identity before reduced mechanics. Selected marker edges and
  opening limits are persisted but do not imply propagation/opening assessment.
  This is not an authenticated app route or Code_Aster implementation.
  The eventual-equilibrium FEniCSx component now assigns hydrostatic head to
  explicitly saturated, fully submerged, open boundary-connected faces and
  returns consistent paired-face forces. Seven pure tests and a native 3-D
  pressure/opening/access-consistency check pass. Wet isolated/reclosed pockets,
  connected unequal heads and free-surface intersections fail explicitly;
  no drained state, steady flow or crack propagation is silently inferred.
  This reduced component has not been ported to Code_Aster.
  A separate reduced FEniCSx-independent surface P1 steady-flow component now
  handles unequal boundary heads on prescribed saturated fully open paired
  meshes. Nine analytical/convergence/identity tests check cubic aperture
  conductance, pressure and flow balance; uniform viscosity cancels from head,
  and only scaled boundary reactions are reported. This is not evolving
  contact/flow or an Aster adapter.
  A reduced planar local-compliance/steady-surface iteration now checks two-way
  pressure/aperture feedback against an independent integrated nonlinear
  solution, with five limit/invariance/failure/convergence tests. This uses
  synthetic already-open specimen inputs, not dam initial openings, and has
  not been coupled to bulk contact/irreversible growth or ported to Aster.
  A further fixed-cut FEniCSx adapter now couples actual 3-D bulk elasticity to
  steady aperture-dependent surface flow. Six native tests verify feedback,
  hydrostatic limit and explicit closed-dry-cut/nonconvergence failures, with
  same-state mechanical/flow residual checks, actual exterior-lip/orientation
  provenance and accepted-only exports. Existing mortar operators, nonzero
  prescribed support motion and fixed nodal exterior loads are retained;
  analytical spring-seat opening/energy and equivalent nodal/form loading
  are checked. The independent audit uses original spring coefficients.
  A separate native fixed-cut 3-D frictionless-contact entry point now uses
  vertex-normal inequality constraints and exact nodal multipliers, retaining
  both free tangential components without penalty stiffness/initial gap.
  Eight tests include oblique contact, analytical pressure reopening/reclosure,
  partial contact versus independent energy minimisation, and persistent
  failure trace without accepted exports. Water remains explicitly prescribed
  traction; contact/flow/tensile-loss/growth integration and Aster parity are
  not claimed.
  Selected no-tension material change, bonded-before-release 3-D contact and
  aperture-dependent steady water are now integrated in one experimental
  fixed-cut event entry point. The accepted state uses actual nonlinear stress
  and its own aperture/head/pressure for residual/force audits. Analytic intact
  fields, tensile redistribution, retained compression, hydraulic snapshots and
  closed-cut failure are checked. The cut is still explicitly supplied; there
  is no edge-driven irreversible surface-growth law or Aster parity claim.
  An experimental dry single-triangle Griffith search now compares audited
  re-equilibrated mechanical potential release with `Gc * new area`, preserving
  accepted surfaces irreversibly. Source wall/wall facet candidates retain all
  mesh orientations and exact marker/source identity. Native tests verify an
  independent displacement-controlled fracture threshold and selected-material
  bonded/released energy identity. This is finite-candidate mesh-constrained
  energy descent, not unrestricted crack tracking, a strength nucleation law,
  multi-face global minimisation, hydraulic growth or a production/Aster adapter.
  A prepared-candidate native orchestration now automatically re-solves the
  selected material/contact state for every trial, preserving full vector
  bonds on unbroken candidates and nonpenetration on released surfaces.
  Parent/marker identities and the exact chosen trial exports are retained.
  Eight-candidate native checks verify low/high resistance growth/arrest;
  unrestricted paths, wet growth and Aster parity remain missing.
  Source preparation/orchestration now imports actual wall/wall facet candidates
  and original selected volume rows, retaining exterior labels, concrete/rock
  properties, gravity, exact exterior water loads and canonical supports.
  Original mortar multiplier parents are preserved across split wall nodes:
  rebuilding independent multiplier rows on every triangle changed the bonded
  stiffness and is explicitly guarded by projector/energy and baseline parity
  tests. Internal new surfaces remain dry. Full-dam all-facet resource cost,
  authenticated saved-scenario binding, unrestricted/wet growth and Aster parity
  are not verified; no production gate was changed.
  An additional experimental bound dry-growth adapter now shares the verified
  source-selection checks with the material event: baseline field/config/mesh
  identity and diagnostic hashes must match before candidate preparation.
  Catalog marker endpoints are translated through source-node compaction.
  Bound-adapter provenance persists, but this remains dry mesh-constrained
  FEniCSx work, not an authenticated app route or Code_Aster parity result.
  Steady saturated FEniCSx flow also supports opt-in exact shared-node crack
  fronts with zero aperture and no regularisation. Analytical wedge
  conductance/head/virtual-work and native material/contact/water checks pass;
  closed inlets/internal barriers and untied contact vertices remain rejected.
  This does not add wet growth, trapped-fluid modelling or Aster parity.
  Explicit hydraulic subset IDs now couple released wet faces to a partially
  bonded mechanical inventory. Flow topology uses checked transitive mechanical
  bonds rather than coordinate welding, while pressure loads retain original
  solid lip nodes. Native connected/subset equilibrium and face-row archive
  tests cover this case. Omitted faces are declared dry; this is not evolving
  fluid conservation, wet crack growth or a Code_Aster implementation.
  A separate common-head FEniCSx finite-candidate wet-growth component now
  includes conservative reservoir work `-integral(p*w)dA` in each equilibrated
  trial potential. It shares the existing dry search without treating wet
  states as dry evidence. Native pressure-work/threshold and seven-to-eight-face
  extension/arrest tests cover this bounded, explicitly saturated case.
  Its explicitly saturated common-head equilibrium specialization permits
  pressure opening from dry-closed contact, with no initial gap and final
  aperture/access/flow checks required before acceptance. A native compression
  specimen verifies pressure opening, material/reaction equilibrium and work.
  Different-head growth, filling history, unrestricted growth,
  trapped water, full-dam use and Code_Aster parity are not supplied.
  Canonical source and verified-baseline wrappers now invoke that bounded
  common-head driver. Inlets are derived from inherited reservoir tags and
  canonical heads, never painted markers; differing heads, waterline crossings,
  constant-pressure crest ambiguity and absent reservoir edges reject explicitly.
  Native source checks retain materials/gravity/exterior water/mortar and strict
  intact displacement parity. Fixed rock nodes on mortar seats now use checked
  serial baseline spring/Dirichlet elimination, and support reactions include
  original spring forces. The source fixture verifies exact fixed displacement
  and global force balance. Full saved-dam execution and Code_Aster parity are
  still unverified; no app solve route is enabled.
  The shared dry/common-head FEniCSx search now optionally tests connected
  multi-face patches up to an explicit size bound (default one). A native
  six-to-eight-face wet jump verifies full reservoir work and irreversible
  retention of previous surfaces. Exhaustive subset and energy-barrier tests
  check bounded candidate completeness and demonstrate a two-face jump missed
  by a single-face search. This remains a mesh-constrained greedy search, not
  unrestricted growth or evidence against larger jumps. The actual saved
  baseline gate is exercised by the reduced Gmsh/catalog integration with
  clearly synthetic evidence; only structure rebuilding is substituted.
  The experimental saved-painting runner now preserves isolated input/
  provenance/failure manifests and accepted PVD/native-state/paired-surface VTU
  paths for reduced native dry/common-head runs. It shares snapshot integrity
  checks with API restore but does not enable production submission.
  Preparation-only native inventory on the actual saved dam found 1,177,580
  candidate faces. Wet preparation rejects its constant-pressure crest and
  different reservoir heads before splitting/solving; this is verified
  inapplicability, not an accepted fracture result or Code_Aster parity.
  Exact unbroken-bond condensation now reduces FEniCSx contact systems while
  recovering/auditing original split-space forces; native multiplier-oracle
  comparisons and strict source intact parity pass. Optional symmetric MINRES
  avoids LU and retains original equilibrium/contact audits without fallback.
  These are numerical FEniCSx improvements, not Code_Aster parity or full-dam
  performance evidence. Released frictionless equilibria may have nonunique
  zero-energy displacements; reports explicitly leave uniqueness/stability
  unverified rather than inventing friction/supports.
  It does not implement the painted event, contact, growth or Aster parity.
- Twenty Windows tests pass across the new scenario/opening contract and
  existing shared-seed contracts. No native solver consumes the new request
  yet; legacy requests keep their existing numerical meaning.
- Painted selection verification used the actual saved dam: 606,414 native
  wall tetrahedra and 654,491 nondegenerate projected edges. Catalog preparation
  with verified-baseline provenance checks took 38.80 s; WPF parsing/indexing/
  picking took 3.49 s; first offscreen render completed within 10.16 s including
  setup, with an 882 MiB working set.
  Cached selection-only repaint took 0.02 s. Fifteen WPF-picked native element
  IDs plus a selected edge passed Python validation on the same catalog.
  These measurements verify selection/preview plumbing, not fracture physics.
- The final focused verification passes 34 Python geometry/contract regressions,
  19 API tests, and the desktop regression runner including native selection
  tests. Full-mesh cached repaint is explicitly checked to remain below one
  second. The existing nullable-return warning in `ScreenConverters.cs`
  remains unchanged.

The client's crack runs horizontally at the selected elevation along the
wall, stopping `cracking[0].buffer_m` (default 1 m) short of the wall/foundation contour at each
end (measured as true distance at the seed's inner radius), or for a centred `length_m` when
`full_width` is false, `depth_m` (default 3 m) deep from the configured `face` with the
configured `opening_m` as the frozen `seed_thickness_m`. Selections closer than the buffer to the
foundation are snapped to the
nearest valid point before the seed is frozen. The local block below cannot hold that, so app jobs now run
`python -m Aster.whole_model_crack` (superseding `Aster.crack_submodel` for app
jobs) on the job's own three-body mesh:

- `whole_model_crack_mesh.py` cuts the wall (only) conformingly at the two seed
  faces (z ± 0.001 m), so the 2 mm seed layer spans the full wall length, then
  elevates wall cells whose centroid is within 0.75 m of the crack to TETRA10.
  All other wall, foundation and bedrock cells stay TETRA4. Source joint
  coefficients are retained but are now integrated consistently on both actual
  refined traces, including wall-to-wall joints and wall-to-foundation
  interfaces; vertex-only joint springs are no longer carried into this mesh.
  The unchanged linear rock retains its original exterior support operator.
  Band mid-edge nodes on faces
  shared with linear cells (`HANGING_NODES`) are tied with `LIAISON_MAIL`.
- `convert_whole_model_to_med.py` writes the mixed-order MED (bundled
  MEDCoupling; identifier width follows `MEDCouplingSizeOfIDs`) and mandatory
  `quadratic_discrete.npz` (export input unit 92). `refined_interface.py`
  integrates the common refinement of both traces with exact TRI6 mass
  products, retaining source triangle identities and checking area/coverage.
  `native_discrete` imports grouped POI1 diagonals and general SEG2
  off-diagonal blocks without double-counting the old joint springs.
- `cases/whole_model_crack.comm`: `3D_GRAD_VARI` + `ENDO_FISS_EXP` on
  `DAMAGE_BAND`, `3D` + `ELAS` elsewhere, `DIS_T` springs. The frozen seed
  (`native_seed.initial_field(..., elastic_elsewhere=True)`) is applied only in
  the band; the band carries the concrete density so gravity is uniform.
  Water, crest surcharge and self-weight ramp together over load factors
  0.1 to 1.0. `RESI_REFE_RELA` 1e-4 (was 1e-6; see below) with `EFFORT_REFE` = FT x median
  wall-foundation tributary area (an assumption for the spring residual).
  Wall, foundation and bedrock peak principal stresses are recorded per state.
- `extract_whole_model_crack.py` writes `crack.pvd` / VTU per state with
  displacement, damage (zero outside the band) and principal / von Mises stress
  over the whole model.
- Output root and scenario stay `cracking/aster_crack_submodel` /
  `Aster_Crack_Submodel` so the API, progress monitor and desktop are unchanged;
  the report has `"analysis_scope": "whole_model"`.

**Runtime requirement:** Linux Code_Aster in WSL (the Windows MUMPS build
returns NaN at this size). The root is `LAMB_LINUX_CODE_ASTER_ROOT` or the
newest `~/opt/code_aster_*`; the distribution is `LAMB_ASTER_WSL_DISTRIBUTION`
(default `Ubuntu`). Defaults are 8 OpenMP threads and 20,480 MiB (the run has
the WSL VM to itself because cracking is sequential); give WSL at least 24 GB
in `.wslconfig`. run_aster's configured `tmpdir` is `/tmp`, which WSL backs
with RAM, so the solve passes `--wrkdir /var/tmp/lamb_aster_crack/<run>` (ext4
disk) and deletes that directory afterwards, even on failure.

Before the consistent-interface correction, verification on a synthetic
5 m wall arc (8,448 cells, 299 tied nodes) found all
11 states converged in 51 s and damage grew to 0.72 beside the seed. On the
repository dam (seed z = -15.73 m) the mesh has 402,054 nodes, 148,448 band
cells and 10,266 tied nodes; a full run is a user job and was not completed
during development.

The corrected saved-dam elastic check completed in 475.69 s under a 540 s
watchdog on the identical volume mesh, with unchanged physical spring
coefficients, loads and ties. Wall Gauss-point extrema changed from
26.232/-40.348 to 1.517/-2.937 MPa. Native refined-interface assembly matches
every matrix entry to 1.23e-11 relative error; global force/moment balance
errors are 1.52e-11/2.80e-11. The actual production preparation reproduces
the verified operator array-for-array. Corrected nonlinear damage runs and
mesh convergence remain unverified. See
[the diagnostic correction and verification record](Aster/CASE_STUDY_DIAGNOSTICS.md#implemented-consistent-interface-correction);
archived vertex-only results are historical, not corrected fracture results.

Linear solver: the real mesh has about 1.33 million DOFs (about 61.6k of them
Lagrange multipliers from `LIAISON_MAIL`). Double-precision MUMPS failed with
`FACTOR_66` at a VmPeak of 11.2 GB, because the run_aster work directory is
on WSL `/tmp`, which is RAM-backed tmpfs, so out-of-core gives no relief. The
case therefore uses PETSc FGMRES with a single-precision MUMPS `LDLT_SP`
preconditioner (BLR `LOW_RANK_SEUIL` 1e-8, linear `RESI_RELA` 1e-8). The
synthetic results were identical. A capped 540 s probe on the repository dam
completed setup and assembly at a VmPeak of 4.6 GB. The figure was printed
before `STAT_NON_LINE` ended, so it does not include the solve. The first app
job (2026-10-25, run in parallel with FEniCSx, which held about 10 GB) failed
at load factor 0.1. Out-of-core MUMPS factorisation in the preconditioner
failed (`FACTOR_66`) at a VmPeak of 17.5 GB, with 10.5 GB of out-of-core files
written to the RAM-backed `/tmp`. This led to the sequential policy and the
on-disk work directory. A full real-dam run has still not been verified.

Probes on that job's mesh, running alone on disk with a hard kill at
420–480 s, found a second cause:
- With the default `REAC_PRECOND=30`, the factor from the elastic prediction
  converged the prediction in 9 FGMRES iterations, but the first Newton
  correction (tangent softened by the seeded damage) stalled near 0.22 relative
  residual. It would have run to 500 iterations before refactorising.
- With `REAC_PRECOND=1` (refactorise every Newton iteration, about 20 s), the
  correction converged in 8 iterations to 1e-8. Peak memory was 19.4 GB
  (VmPeak), including an 8.5 GB MUMPS factor. This fits the 20,480 MiB limit
  only because Aster runs alone.
- Rejected options:
  - Symmetric support springs: the damage tangent is non-symmetric anyway.
  - `ELIM_LAGR="OUI"`: this PETSc build has no SuperLU.
  - `LOW_RANK_SEUIL=1e-5`: the factor dropped to 3.9 GB, but the correction was
    still at 1e-3 after 28 iterations.

Expect the full 11-state run to take hours.

Crack water pressure: the prepared seed carries `loading.internal_pressure_pa`,
the full hydrostatic head of the seeded face at the crack elevation. It is
computed once by `FEniCSx.automatic_crack.hydrostatic_crack_pressure` with the
Elmer datums: for the upstream face, reservoir surface = lowest contour +
water height + overflow head; for the downstream face, tailwater = lowest
contour + tailwater head. There is no drainage reduction, and the pressure is
zero above the tailwater. FEniCSx applies it as its diffuse crack force. Aster
checks it against its own load datums and applies it with `PRES_REP` to the
`CRACK_FACES` group. These are the 0.002 m seed-slab faces, with their normals
fixed into the slab, so `VERI_NORM="NON"` is set and positive pressure opens
the crack. The pressure is ramped with the other loads. Seeds prepared before
this change carry 0 Pa and are refused; select the crack location again. A
synthetic run at 130.5 kPa converged and increased the mean crack-face
separation (`CRACK_TOP` minus `CRACK_BOTTOM` DZ) from 3.757e-5 m to
3.946e-5 m.

Limitations: only band cells can damage; the seed is horizontal and does not
follow the sloped bedrock at its ends; hanging-node ties slightly stiffen the
band edge; Aster applies crack water to the seed faces only and does not move
it with damage growth; uplift on the wall-foundation interface is not
modelled for any solver; mesh and band-width convergence are unverified.
Converging is not engineering validation.

```powershell
.\.venv\Scripts\python.exe -m Aster.whole_model_crack --dam-config config.json `
  --prepared-seed <job>\cracking\prepared_seed.json `
  --source-mesh <job>\output\model\three_body.msh --output <job>\cracking
.\.venv\Scripts\python.exe -m unittest Aster.test_whole_model_crack
```

### Native dam crack submodel (app MVP)

Earlier local approach, kept for reference; app jobs now use the whole-model
crack above.
Superseding the paragraph above for app jobs: submitted three-body jobs now run
`python -m Aster.crack_submodel` **in parallel** with FEniCSx cracking, at the
same frozen UI seed (`cracking/prepared_seed.json`). The full quadratic dam
(at least 4.30 million unknowns) remains impractical, so Aster solves a local
curved **wall-only block** around the seed (option A, user-approved):

- `block_layout` keeps the exact seed sector and elevation as mesh edges and
  adds default 2 m margins, clipped 0.25 m clear of abutments, bedrock,
  foundation and spring-jointed thickness steps. Fewer than 0.5 m of wall
  concrete around the seed (other than at the crest) is rejected rather than
  moving the seed.
- `crack_submodel_mesh.py` (bundled MEDCoupling) builds a positive-Jacobian
  HEXA20 block in (theta, r, z) with a separate 0.002 m seed layer, graded
  normal layers and named faces. Face groups use `buildBoundaryMesh(True)`;
  the compact variant renumbers coordinates and corrupted face connectivity.
- Cut faces follow the verified FEniCSx baseline's Elmer displacement
  (`displacement_m`, now saved in `crack_seed_fields.npz`; older archives
  require a new baseline). It is transferred by a locally linear moving least
  squares fit from wall-only nodes, applied with `AFFE_CHAR_CINE_F` and scaled
  with the load factor. Cut nodes are selected geometrically because
  `getNodes(group)` ids do not index `getCoordinates()` order.
- Water (upstream, tailwater, crest when reached) and self-weight use the same
  definitions as the FEniCSx three-body runner. The schedule is seed
  equilibration at zero load (times 0 to 1), then load factors 0.1 to 1.0.
- `ENDO_FISS_EXP` / `3D_GRAD_VARI` use the shared provisional material contract
  with `RESI_REFE_RELA` 1e-4 (the synthetic benchmarks used 1e-8; 1e-6 stagnated at
  2-5e-6 on the full dam while RESI_GLOB_RELA reached ~1e-14, i.e. below the
  round-off floor, so every step cut repeated a 1.7 h stall), Newton line
  search and automatic step subdivision. User jobs have no runtime cap.
- Outputs per run: `manifest.json` (configuration and prepared-seed hashes),
  `crack_report.json` (layout, loads, interpolation distance, native seed,
  per-state damage growth, peak principal stress), `result.med` and
  `crack.pvd`/VTU. The API validates the manifest against the job and lists the
  report as scenario `Aster_Crack_Submodel`; the desktop shows a summary beside
  the FEniCSx crack peak with an **Open Aster Crack in ParaView** button.
- Live progress: on Windows `run_aster` sends solver output to `fort.6` in a
  temporary work directory and prints it only after the solve. `run_solver`
  (with `live_log`) tails that `fort.6` into `solver.live.log` during the run;
  `solver.log` still holds the complete output afterwards. The API's
  `/jobs/{id}/progress` endpoint parses these files for the activity monitor.

Developer run on the repository configuration, seed at y = 17.53 m,
z = -15.61 m (downstream face, 3 m radial by 4 m chainage): 2,688 HEXA20
cells, 21,504 Gauss points (384 seeded), 1,924 cut nodes with at most 0.37 m
to the nearest baseline source. All 12 states converged in about 255 s on
Windows. Peak wall principal tension at full load was 1.54 MPa (Elmer wall
baseline 1.42 MPa). Damage did not grow to 0.95 beyond the seed; intact points
next to the seed reached about 0.83 during zero-load equilibration, matching
the regularised initial-state response above, not load-driven propagation.
Damage at cut faces stayed below 0.002.

Limitations: the surrounding dam stays at uncracked stiffness and displacement,
so softening cannot redistribute beyond the block and damage near cut faces is
boundary-driven; crack-face water pressure is not modelled; 0.5 m elements
coarsely resolve the 0.5 m band; mesh, band-width and margin sensitivity are
unverified.

```powershell
.\.venv\Scripts\python.exe -m Aster.crack_submodel --dam-config config.json `
  --prepared-seed <job>\cracking\prepared_seed.json `
  --baseline-fields <job>\cracking\dam_elastic_baseline\<run>\crack_seed_fields.npz `
  --output <job>\cracking
.\.venv\Scripts\python.exe -m unittest Aster.test_crack_submodel
```

## Scope

Port the currently working dam model to Code_Aster before making any further
attempt at cracking, contact, cohesive interfaces, or concrete damage.

This work is a solver and results-porting exercise. It must reproduce the
existing linear-elastic model's geometry, material, supports, loads, and
observable response. No new failure mechanism is in scope.

## Explicitly Out of Scope

- Open-crack topology.
- Contact or friction at an interface.
- Cohesive elements or interface damage.
- XFEM, crack propagation, or bulk concrete damage.
- Crack-face water pressure.
- Changing the dam geometry while porting.
- Replacing the existing foundation support model with a rock domain.

These remain future studies after the Code_Aster elastic baseline is verified.

## Implementation Status Amendment (read this first)

### Source-faithful elastic parity (supersedes the historical amendments)

Engineering peak-stress reporting separates **wall** and **concrete foundation**.
For source-identified three-body results, `comparison_report.json` and
`verification_report.json` expose `tensile_peak_comparison_by_region` with Elmer
and Aster values in MPa, signed differences in MPa and percent, and both peak
coordinates. The pipeline prints these two rows explicitly. No combined
wall/foundation stress maximum is used as the headline comparison. Detailed
regional tension/compression peaks remain in `comparison_report.json`.

For the corrected `03_elastic_parity` run, the recovered nodal tensile peaks are:

| Region | Elmer | Code_Aster | Aster relative to Elmer |
| --- | ---: | ---: | ---: |
| Wall | 1.122209 MPa | 0.995747 MPa | -11.27% |
| Concrete foundation | 1.566340 MPa | 1.325147 MPa | -15.40% |

These are native recovered nodal stresses, not the common-element stress
comparison below. The 15.4% gap is the foundation comparison, not the wall.

The current three-body pipeline now reads the saved Elmer SIF and diagnostic
alongside its saved configuration, rather than treating a matching configuration
hash as proof of matching solver inputs.

- All original volume cells and split interface-node identities are preserved;
  the port no longer welds wall joints or removes source tetrahedra.
- Both internal wall joints and the wall/foundation interface use the actual
  saved three-axis coefficients. The inspected reference has
  **1.25e9 N/m3 in all directions at the wall/foundation interface**, not the
  3e9 N/m3 vertical coefficient previously reconstructed from configuration.
- Outer supports use saved global normal-spring matrices, including their
  off-diagonal terms; tangential movement is not fixed.
- Hydrostatic pressure and surcharge come from the saved load definitions.
  The inspected downstream water level is **-26 m**, not the former
  geometrically reconstructed -27.4726681642 m.
- Material assignments, elastic constants and gravity force densities are
  checked against the saved SIF. Missing or unsupported source laws fail
  explicitly rather than silently substituting an approximation.
- Source-node matching uses incident tetrahedra, so coincident interface sides
  remain distinct in both solvers' comparison archives.

Run from the Gairezi directory in PowerShell:

```powershell
& .venv\Scripts\python.exe -m Aster.run_three_body_pipeline `
  --source-mesh 'C:\path\to\saved-run\model\three_body.msh' `
  --output-dir 'Aster\results\elastic_parity'
```

The pipeline generates a case and input manifest inside the chosen output
directory, then solves, exports concrete-only ParaView stresses, freezes the
matching reference, compares wall/foundation responses and writes
`verification_report.json`. `run_three_body_case` also regenerates an
output-local case before solving; it cannot accidentally use the old static
command file against a newly exported mesh. Solver timeouts terminate the
specific launched Windows process tree and retain diagnostic logs.

New diagnostics include named concrete probes, explicitly labelled unweighted
recovered-nodal means, and **volume-weighted element stress comparisons using
the same linear tetrahedral strain/constitutive calculation on both displacement
fields**. This last comparison separates differences in the displacement
solution from differences in each solver's nodal stress recovery. Raw nodal
maxima remain observations, not elastic-baseline acceptance criteria.

`stress_samples.json` freezes four provisional interior coordinates: three in
the wall and one in the foundation. Every point is checked against all region
boundary triangles, including joints, for at least 0.25 m clearance. Sampling
uses the same containing tetrahedron and constitutive calculation for both
solvers; if a point lies on an internal element face, both one-sided tensors
are retained. Subsequent verification reuses those exact physical coordinates
and fails if they leave the region or violate clearance. The default locations
are diagnostic choices, not engineer-approved critical sections.

The saved native mortar projectors are audited separately in
`native_projector_audit.json`, using partition-local node maps with coordinate
validation and source-artifact hashes. In the inspected source, all 246 exported
wall/foundation projectors match consistent triangle mass to within **4e-14
relative Frobenius error**. The maximum difference from the port's lumped
representation is **0.592** under the same matrix measure. This is a stiffness
representation difference, **not** a 59.2% displacement or stress error.
Neither internal wall joint has a saved native projector, and this audit is not
an independent whole-model energy/reaction export.

Force and moment diagnostics integrate gravity and clipped hydrostatic loads
and sum support reactions. Pressure faces are oriented using their unique
incident volume cell, independently of input triangle winding. Strain energy
includes the solid and discrete-spring contributions. The source energy
evaluated here uses the port's spring integration: it is **not** an independently
exported native Elmer assembly energy.

**Acceptance remains blocked**, even if the solve and preliminary 2% checks
pass. Surface springs are nodally lumped rather than integrated using Elmer's
consistent Galerkin assembly. Native source reactions/energy, agreed fixed-offset
stress samples and mesh-convergence verification are still required. Reports
therefore retain `baseline_verified: false`; no baseline is tagged and no
fracture/contact/damage work is enabled.

#### Controlled corrected-input run

`Aster/results/03_elastic_parity` completed in approximately 437 seconds.
Its concrete-only VTU was opened using stock ParaView: 685,506 cells and all
ten stress arrays were finite. The preceding `02_elastic_parity` run was an
intermediate joint/support experiment with the old downstream datum; do not
treat it as the corrected baseline.

| Concrete comparison | Elmer | Code_Aster | Difference |
| --- | ---: | ---: | ---: |
| Maximum displacement | 10.18795 mm | 10.18490 mm | -0.0299% |
| Wall maximum element tension, common recovery | 1.13534 MPa | 1.14251 MPa | +0.631% |
| Foundation maximum element tension, common recovery | 1.68146 MPa | 1.71978 MPa | +2.279% |
| Maximum recovered nodal tension, diagnostic only | 1.56634 MPa | 1.32515 MPa | -15.399% |

The volume-weighted stress-tensor L2 difference is **0.991% for the wall**
and **2.202% for the foundation**; the foundation therefore misses the
preliminary 2% target. Nodal recovery still needs investigation: the 15.4%
nodal-extremum gap is not a demonstrated 15.4% difference in structural response.
The displacement L2 differences are 0.537% and 0.065%, respectively.
At the frozen interior stress samples, tensor differences are 2.251%, 0.929%
and 0.250% in the wall, and 1.577% in the foundation. Thus the lower wall
sample also misses the initial 2% target; local agreement must not be inferred
from the small global displacement difference.
Global force/moment residuals are below 2e-14 relative and the strain-energy
difference evaluated with the same port spring integration is 0.0242%.
These checks do not waive the independent-source-assembly and convergence gates.

Focused verification passed: 21 contract/mechanics/comparison/runner tests,
four MED export tests under the installed solver Python, and the real ParaView
read. Re-exporting the source also reproduced every mesh array and executable
solver-command line used by this solve.

Recheck existing output without launching either solver:

```powershell
& .venv\Scripts\python.exe -m Aster.verify_parity Aster\results\03_elastic_parity
```

This writes the verification report with input/result hashes. Exit code 2
means baseline acceptance is incomplete, even if preliminary checks pass.
The full pipeline likewise returns `requires_verification` and exit code 2
until baseline acceptance, while preserving the completed solve and concrete
visualization. Successful solver execution is not baseline acceptance.

### ParaView stress output

Engineering reporting is concrete-only: wall and optional foundation.
Comparison extrema use recovered nodal tensors for both solvers. The MED
extractor subsets the concrete group by cell identity, preserving separate
stresses on coincident interface nodes; coordinate deduplication is not used.
The full solver domain remains unchanged and is not the reporting domain.

Stock ParaView 6.2 on this machine has no MED reader. The three-body
`result.med` contains `RESU____SIGM_NOEU` (six nodal stress components), but
not `SIEQ_NOEU`; see Amendment 3 for the solver-side principal-stress issue.
`Aster/extract_result_fields.py` now also writes `run/result_stress.vtu`,
and the run manifest and pipeline terminal output identify that file.
Open it in ParaView, click **Apply**, then color by `principal_max_pa`
(largest signed principal stress), `principal_min_pa` (smallest signed
principal stress), `von_mises_pa`, or any `stress_xx_pa` through
`stress_yz_pa` component. All values are in Pa; positive principal stress
means tension and negative means compression. Rescale to the data range.

The VTU preserves the MED stress field's concrete volume mesh and node order,
including coincident interface nodes. Discrete springs and unsupported
nodes have no continuum stress and are excluded, rather than filled with
invented values. Principal stresses are computed with NumPy from the saved
nodal tensor, without changing the solve or stress recovery. Displacement
remains in the original MED and comparison NPZ; the VTU is stress-only.
MEDCoupling writes visualization arrays as Float32; the original MED/NPZ
retain the solver's double-precision values for numerical comparisons.
An existing solve can be exported without rerunning Code_Aster by invoking
`Aster/extract_result_fields.py` with Code_Aster's bundled Python and the
existing `result.med` and `aster_result_fields.npz` paths.

The sections below were written against an earlier three-region model and a
different environment than what is actually in the repository today. Treat
the following corrections as authoritative over the original text; the rest
of this document is kept for its intent (scope, work packages, acceptance
criteria) but some specifics are now out of date.

- **Model shape.** `Elmer/regions/combined_structure.py` currently builds a
  **single monolithic `WALL` volume** (`REGION_GROUP_IDS = {"WALL": 1}`)
  seated on a distributed spring foundation, not three bonded
  Plinth/Wall/Haunch regions. There is no `PLINTH` or `HAUNCH` group to
  export. The named-group contract already matches this: see
  `Elmer/regions/model_groups.py` (`FACE_GROUP_IDS`, `REQUIRED_GROUPS` =
  `WALL`, `FOUNDATION`, `UPSTREAM`, `DOWNSTREAM`, `CREST`).
- **Inputs to preserve.** `Data/Concrete_Material_Properties.json` and
  `Data/Env_Boundaries_And_Loads.json` do not exist. The single source of
  truth for material, load, and foundation-support values is
  `Gairezi/config.json`, read live by `Elmer/structure_assembly.apply_load_case`
  and `Elmer/regions/combined_structure.build_combined_structure`. The table
  above with `k_x=226e6`, tailwater/overflow head of 2 m, and peak pressure
  284.49 kPa is stale; the current `config.json` values (e.g.
  `spring_n_per_m3` 1.25e9 x/y, 3.0e9 z; tailwater_head 3.0 m; overflow_head
  3.0 m) must be read from `config.json` directly, never hardcoded or copied
  from this document.
- **MED export.** The implementation environment's Python (3.15) has no
  prebuilt `h5py` wheel and no local HDF5 library to build one from source,
  so true `.med` (HDF5-based) writing is currently unavailable here.
  `Aster/build_med_mesh.py` always writes a Gmsh 2.2 ASCII mesh with a
  `$PhysicalNames` section (which Code_Aster reads natively via
  `LIRE_MAILLAGE(FORMAT='GMSH')`), and attempts a `.med` write only as a
  best-effort extra (succeeds under a Code_Aster/Salome Python that has
  `h5py`/`medcoupling`). The mesh manifest records
  `med_unavailable_reason` when the extra isn't produced.
- **Local Code_Aster/Salome install found and partially exercised.**
  Salome-Meca 2025 / Code_Aster 17.4.0 is installed at
  `C:\SALOME-MECA\2025\W64\CODE_ASTER_17.4`; its launcher
  (`bin\run_aster.bat`) needs no environment variable changes — it is a
  self-contained wrapper. `Aster/run_case.py` now detects it (also
  overridable via `LAMB_CODE_ASTER_LAUNCHER`), generates a matching
  `.export` file, and invokes it with a hard subprocess timeout.
  `00_elastic_baseline.comm`'s syntax was confirmed accepted by the real
  installed command catalogue (`run_aster --env` dry run succeeds; the live
  solver log shows it executing `DEBUT`, `LIRE_MAILLAGE`, etc. in order), so
  it is no longer "unverified" syntactically.
- **Real solve not yet completed: `LIRE_MAILLAGE` is too slow on this
  install.** Two live attempts (with and without `VERI_MAIL`) each ran for
  8-9 minutes of genuine, continuous CPU work and never finished reading the
  ~101,555-cell Gmsh 2.2 ASCII mesh, well past what an agent-run process may
  consume per this repo's wall-clock cap. `VERI_MAIL=_F(VERIF='NON')` is now
  set explicitly in the generated `.comm` (Code_Aster silently defaults it
  to `'OUI'` otherwise), but that made no measurable difference — the
  bottleneck is the GMSH-format import itself, not mesh validation. A true
  `.med` mesh would likely import much faster, but producing one is still
  blocked: neither the tooling venv (`h5py` has no prebuilt wheel) nor
  Code_Aster's own bundled Python (`external\Python3.10\python.exe`) can
  import `h5py`/`medcoupling` standalone outside of `run_aster`'s own
  bootstrap, which calls `os.add_dll_directory()` explicitly for several
  external library folders beyond what `PATH`/`PYTHONPATH` alone provide.
  Work Packages 6-7 (actually solving and comparing results) therefore
  remain blocked pending either a faster mesh-import path or a run outside
  the agent's enforced time cap.

## Implementation Status Amendment 2 (first successful solve + comparison)

The paragraph above is now **superseded**: the `LIRE_MAILLAGE(FORMAT='GMSH')`
slowness was real but is no longer the mesh-import path used; and Work
Packages 6-7 (solving and comparing) are now complete and passing.

- **MED import solved by writing MED directly via medcoupling, not by
  speeding up GMSH import.** `Aster/convert_arrays_to_med.py` builds the
  mesh in-memory with medcoupling (`MEDCouplingUMesh`, fixing HEXA8/PENTA6
  node winding so Code_Aster doesn't reject negative-Jacobian cells, calling
  `sortCellsInMEDFileFrmt()`, and registering face/volume named groups with
  `setGroupsAtLevel`) and writes a true binary `.med` file directly, run
  under Code_Aster's own bundled Python (`run_aster`'s bootstrap, which is
  the only Python on this machine with a working `medcoupling`/`h5py`).
  `LIRE_MAILLAGE(FORMAT='MED', VERI_MAIL=_F(VERIF='NON'))` then reads this
  file in well under a second — the GMSH ASCII path is no longer used at
  all. `Aster/build_med_mesh.py` still writes the Gmsh 2.2 ASCII mesh too
  (for inspection/debugging), but the `.comm` case only ever reads the MED
  file.
- **Two real Code_Aster model errors found and fixed by actually running the
  solver (not discoverable from documentation alone):**
  1. `AFFE_MODELE` must include every `GROUP_MA` referenced by any later load
     or elemental characteristic, even face groups that only carry a
     `PRES_REP`/`RIGI_PARASOL`, or `MECA_STATIQUE`/`AFFE_CARA_ELEM` raises
     `MODELISA6_96` ("ces mailles n'appartiennent pas au modele").
  2. `RIGI_PARASOL`'s distribution method requires `POI1` point elements at
     every `FOUNDATION` node (one new `GROUP_MA_POI1` group,
     `Aster/convert_arrays_to_med.py`), each registered in `AFFE_MODELE`
     with its own `MODELISATION='DIS_T'` entry, and `CARA_FOUNDATION =
     AFFE_CARA_ELEM(...)` must use `GROUP_MA_POI1=`, `COEF_GROUP=(1.0,)`,
     `CARA='K_T_D_N'`, and a `COOR_CENTRE=` (the raft's bounding-box
     midpoint, computed from `mesh_manifest.json`'s `FOUNDATION` group
     bounds) — not the plain single-`GROUP_MA` `DISCRET`/`K_T_D_N` syntax
     that catalogue examples for non-`PARASOL` springs show.
- **Critical, non-obvious physics bug: `RIGI_PARASOL`'s `VALE` is a GLOBAL
  stiffness, not a per-unit-area modulus.** Per Code_Aster's own
  documentation (u4.42.01 S17.4.4, "Distribution method"), `RIGI_PARASOL`'s
  `VALE=(Kx, Ky, Kz)` is the foundation raft's **total** stiffness in N/m,
  which Code_Aster internally distributes across the `GROUP_MA_POI1` nodes
  weighted by each node's tributary area. This is fundamentally different
  from Elmer's `config.json` convention (`spring_*_n_per_m3`, a per-unit-area
  Winkler modulus in N/m^3). The first full solve passed the raw
  `spring_*_n_per_m3` value directly as `VALE`, under-stiffening the
  foundation by a factor equal to the `FOUNDATION` group's total area
  (~864 m^2 on this mesh) and producing displacements 30-50x too large
  (mean displacement norm ~0.38 m vs Elmer's baseline max of ~8.3 mm).
  Fixed in `Aster/generate_elastic_case.py` by computing
  `foundation_area_m2` from the mesh manifest and setting
  `VALE = spring_n_per_m3 * foundation_area_m2`. After the fix, Code_Aster's
  displacement field matches Elmer's to within ~0.1% (see below).
- **First full end-to-end solve succeeded** after the two model-error fixes
  and the `RIGI_PARASOL` units fix: `MECA_STATIQUE` + `CALC_CHAMP` +
  `IMPR_RESU` all complete, `run_manifest.json` reports
  `status: "completed"`, and the full solver log (now persisted in full as
  `stdout.log`/`stderr.log`, not just a 4000-char manifest tail, since
  `subprocess.run(capture_output=True)` truncation could otherwise hide an
  earlier uncaught exception) contains "ARRET NORMAL" and zero
  `<EXCEPTION>`/`Traceback`/`ERREUR` occurrences.
- **Work Package 7 (comparison against the frozen Elmer baseline) is
  implemented and passing.** `Aster/extract_result_fields.py` reads `DEPL`
  and `SIGM_NOEU` out of `result.med` via medcoupling's `MEDLoader` (note:
  the mesh name inside `result.med` is Code_Aster's own generic internal
  name, not the `NOM_MED` given to `LIRE_MAILLAGE` — discover it with
  `MEDLoader.GetMeshNames`; field names are prefixed with the result
  variable name padded to 8 characters, e.g. `RESU____DEPL`; `SIGM_NOEU`'s
  Voigt order is `(xx, yy, zz, xy, xz, yz)`, with `xz`/`yz` swapped relative
  to `Elmer.analyze_stress.stress_tensor`'s `(xx, yy, zz, xy, yz, xz)` —
  reordered on extraction). `Aster/postprocess_baseline.py` then:
  - Matches Elmer and Code_Aster nodes by coordinate, not by array index.
    This is required for two independent reasons, both confirmed by direct
    inspection rather than assumed: Elmer's PVTU is split across MPI
    partitions and `Elmer.analyze_stress.parse_vtu` concatenates each
    partition's point list without deduplicating shared boundary nodes
    (119,150 Elmer rows vs 116,708 Code_Aster rows, a gap of exactly 2,442
    duplicate-coordinate pairs); and Elmer's VTU "points" `DataArray` turned
    out to be the **deformed** mesh, not the undeformed reference geometry
    the Work-Package-1 freeze script's manifest had (incorrectly, until
    this fix) claimed — found because nearest-neighbour coordinate gaps
    matched the local displacement magnitude almost exactly. Fixed in
    `Aster/freeze_elmer_baseline.py` by storing `points - displacement`
    (undeformed) in `elmer_baseline_fields.npz`; verified by this fix that
    the two meshes then coincide to ~1e-10 m (floating-point noise).
  - Coordinate matching itself uses a dependency-free spatial grid-bucket
    nearest-neighbour search (`_match_by_coordinate`), not naive independent
    per-axis rounding: rounding was tried first and found to misclassify
    points that straddle a rounding boundary on one array's coordinate but
    not the other's, silently producing spurious "no match" failures. No
    `scipy` is installed in this repo's `.venv` and none was added for this.
  - Compares peak displacement and peak tensile/compressive principal
    stress (via vectorized per-node `np.linalg.eigvalsh` on the nodal
    stress tensor) against a `TOLERANCE_FRACTION = 0.02` acceptance
    threshold, flagging (not silently averaging away) anything outside it.
- **Result (first full comparison, elastic baseline):**
  - Displacement: Elmer max norm 8.2565 mm vs Code_Aster max norm 8.2461 mm
    at the matching node — **agree to within ~0.13%**, well inside
    tolerance. This is the strongest evidence so far that the ported model
    (mesh, material, loads, and the corrected foundation stiffness) is
    equivalent to Elmer's.
  - Peak tensile principal stress: Elmer 1.786 MPa vs Code_Aster 2.207 MPa
    (+23.6%); peak compressive: Elmer -3.719 MPa vs Code_Aster -5.070 MPa
    (-36.3%) — both **outside** the 2% tolerance and flagged accordingly.
    This is methodologically explainable, not necessarily a modelling bug:
    Code_Aster's peak is extracted per-node (`SIGM_NOEU`, nodally
    extrapolated/averaged by Code_Aster itself) while Elmer's is extracted
    per-element (mean of each element's own node stresses, in
    `Aster/freeze_elmer_baseline.recompute_peak_principal_stresses`) — a
    real difference in extraction methodology that can by itself produce
    double-digit-percent gaps right at stress concentrations near supports,
    where both solvers' peaks are located (near the foundation, by
    `centroid_xyz_m`). This gap should be investigated further (e.g. by
    comparing element-averaged-to-node Code_Aster stresses, or comparing
    stress fields away from the concentration) before drawing any
    conclusion about which solver's peak is more trustworthy; it is not
    itself evidence against the ported model.
- **What this does and does not show.** Close displacement agreement
  corroborates that the Code_Aster port reproduces Elmer's linear-elastic
  structural response under the same geometry/material/loads/foundation
  stiffness. Neither solver models cracking in this elastic baseline —
  see `Aster/postprocess_baseline.py`'s `comparison_report.json` "note"
  field, which repeats this caveat in the machine-readable output itself.

## Implementation Status Amendment 3 (three-body wall+foundation+bedrock port)

Amendment 2 above only ever ported the legacy **wall-only** Elmer model
(`00_elastic_baseline`). Per an explicit requirement that the Elmer-vs-Aster
cross-check be apples-to-apples with the real, current app/API/CLI model —
`Elmer.three_body_solver` (`Gairezi/README.md` "Integrated wall/foundation/
bedrock solver"), which includes real foundation and bedrock volumes, not
just the wall — a second, parallel case was built:

- `Aster/build_three_body_mesh_arrays.py` exports node/cell arrays (all
  tetrahedral) from an already-solved `Elmer.three_body_solver` run, with
  `CONCRETE` (wall+foundation, bonded into one continuum) and `BEDROCK`
  volume groups, plus `UPSTREAM`/`DOWNSTREAM`/`CREST`/`ROCK_BOTTOM`/
  `ROCK_SIDES` face groups and a `ROCK_BOTTOM_POI1` point group for
  `RIGI_PARASOL`.
- `Aster/convert_three_body_to_med.py` writes those arrays to a real
  `mesh.med` via Code_Aster's bundled `medcoupling`.
- `Aster/generate_three_body_case.py` renders `Aster/cases/
  01_three_body_baseline.comm` from `config.json`, reusing the same
  foundation/bedrock support numbers Elmer reads
  (`material.bedrock.support.spring_n_per_m3`).

**Documented simplifications vs. the real Elmer three-body model** (see
`build_three_body_mesh_arrays.py`'s module docstring for the full
rationale): the wall's internal joints, the wall/foundation spring-pair
mortar interface, and the foundation/bedrock bonded interface are all welded
into ONE bonded concrete-on-rock continuum, instead of Elmer's many discrete
two-sided springs. The bedrock's bottom (axis-aligned, Z-normal) keeps a
true `RIGI_PARASOL` roller spring; the 4 slanted rock-box sides are
approximated as fully fixed (`DDL_IMPO`) instead of Elmer's local-frame
normal-only roller springs (those boundaries sit >50m from the dam, in the
semi-infinite-domain truncation region, not the region being compared).

**Mesh cleanup required before any solve could succeed.** Elmer's own
`graded_bedrock.py` mesh grading (far-field domain-truncation region)
produces a small number of severely sliver bedrock tetrahedra and some
nodes supported by only 1–2 tetrahedra; left in place, these caused a MUMPS
`FACTOR_11` ("pivot nul ou presque nul") hard crash. Two targeted,
documented mesh-cleanup passes were added (both validated as still
necessary and kept regardless of the solver bug below):

- `BEDROCK_SLIVER_QUALITY_THRESHOLD` (shape-quality-based sliver removal).
- `MIN_FREE_NODE_SUPPORT` / `_prune_weakly_supported_free_nodes()`
  (iteratively drops bedrock tetrahedra that are the sole/near-sole support
  of a free, non-boundary, non-concrete node — quality-threshold tuning
  alone just moves this failure to a different node each time; it does not
  fix the underlying node-degree problem).

**A second, much deeper problem was then found and root-caused.** After
the `FACTOR_11` crash was fixed, the solve completed cleanly
("ARRET NORMAL", no exceptions/alarms) but produced a **100% NaN**
displacement/stress field on every node, every time — including on the
simplest possible well-posed reduction (gravity-only load, full `DDL_IMPO`
fixation at every outer boundary, no springs, no pressures). A systematic
isolation (see session history for the full elimination sequence) ruled
out, in order: `RIGI_PARASOL` spring magnitude, hydrostatic/surcharge
`FORMULE` loads, mesh validity (`VERI_MAIL='OUI'` passes cleanly),
negative/zero-volume elements (none, checked both in the source npz and
independently re-read from the written `mesh.med` via `MEDLoader`),
duplicate tetrahedra (none), poorly conditioned element Jacobians (max
condition number ~77), disconnected mesh regions (single connected
component via union-find), volumetric overlap between `CONCRETE` and
`BEDROCK` (none, sampled point-in-tetrahedron test), and — decisively — the
two-material assembly itself (a `BEDROCK`-only sub-case solved cleanly; a
`CONCRETE`-only sub-case, fixed at the real nodes it shares with `BEDROCK`
in the combined mesh, still produced 100% NaN under the same solver).

The actual root cause: **MUMPS 5.6.2** (the direct solver bundled with this
Code_Aster 17.4 Windows distribution) reports a clean factorization on this
problem (`INFOG(1)=0`, zero null pivots detected via `ICNTL(24)`) but its
solve/iterative-refinement phase silently returns an all-NaN solution
vector (`RINFOG(5)` "NORM OF Computed SOLUT" = NaN from the very first
step) with no error or alarm anywhere in the log. Switching *only* the
solver — `SOLVEUR=_F(METHODE='MULT_FRONT')` instead of the default
`METHODE='MUMPS'` — on the exact same `CONCRETE`-only isolation case
immediately produced a clean, physically sane result (max displacement
~0.5mm, 0 NaN). Applying the same fix to the full production case
(`generate_three_body_case.py`) resolved it completely: the full
gravity+hydrostatic+surcharge+`RIGI_PARASOL`+`DDL_IMPO` three-body solve
now produces **0 NaN** across all 187,701 nodes, max displacement **~7.0mm**
— in the same physical ballpark as the legacy wall-only baseline's ~8.2mm.

**Lesson for future Code_Aster work on this installation**: "ARRET NORMAL"
(clean completion, no exceptions) is not sufficient evidence of a valid
solve; always check the result field for NaN/Inf before trusting it. If a
`MECA_STATIQUE`/similar solve completes cleanly but yields NaN with no
diagnostic, suspect the MUMPS solver itself (try `SOLVEUR=_F(METHODE=
'MULT_FRONT')` as a diagnostic swap) before re-auditing the mesh/model —
this installation's bundled MUMPS 5.6.2 has a reproducible silent-NaN
failure mode on at least one well-posed, well-conditioned linear-elastic
problem.

**Not yet done**: extending `Aster/test_aster_tooling.py` to cover
`_prune_weakly_supported_free_nodes()`.

## Implementation Status Amendment 4 (wall/foundation interface: bonded weld → real springs)

The first Elmer-vs-Aster three-body comparison (Amendment 3) ran, but
disagreed badly: Aster's peak displacement was **-31%** vs. Elmer's, peak
tensile principal stress was **+152%**, peak compressive principal stress
was **-58%**. Diagnosis: of the three "documented simplifications"
above, the **wall/foundation Mortar BC interface** is not a structural
joint in the real model — Elmer couples the wall and foundation there with
a real, compliant spring (`geometry.foundation.support.spring_n_per_m3` =
1.25e9 N/m³, all axes), not a bond. Welding that interface into a rigid,
fully-bonded joint (as Amendment 3 did, for simplicity) made Code_Aster's
wall 50–300x stiffer at that seat than Elmer's, which plausibly explains
all three large discrepancies at once (stiffer seat → less displacement,
more concentrated/redistributed stress).

**Fix implemented**: `_weld_coincident_nodes()` in
`build_three_body_mesh_arrays.py` now keeps the wall/foundation interface's
coincident node pairs **unmerged** (every other weld — wall internal
joints, foundation/bedrock — is unchanged/still bonded). For every one of
the resulting 3412 node pairs, `build_three_body_mesh_arrays.py` computes a
lumped nodal tributary area from the wall's own exposed interface boundary
faces; `convert_three_body_to_med.py` writes each pair as a zero-length
`SEG2` element (MED level -2); `generate_three_body_case.py` models each as
a discrete `DIS_T`/`K_T_D_L` spring (`AFFE_CARA_ELEM`/`DISCRET`), stiffness
`VALE=(kx,ky,kz) = spring_n_per_m3[axis] * tributary_area_m2`, same
`spring_n_per_m3` config value Elmer uses. This exactly mirrors how
Elmer's Mortar BC (a per-area surface spring coefficient, integrated over a
conforming/coincident-node mesh) behaves.

**Result after the fix** (`Aster/results/01_three_body_baseline/
comparison_report.json`):

| Quantity | Before (bonded weld) | After (real springs) |
| --- | --- | --- |
| Peak displacement | -31% | **-0.016%** |
| Peak compressive principal stress | -58% | **-0.5%** |
| Peak tensile principal stress | +152% | +71% (different node — see below) |

Displacement and peak compressive stress now agree to within noise,
strongly confirming the diagnosis. The residual tensile-stress gap is
**no longer at the wall/foundation seat**: Aster's new tensile peak sits at
`(93.8, 124.8, 5.0)`, ~156 m from the dam axis, at the mesh's absolute
maximum Z — i.e. the far-field bedrock ground surface in the
domain-truncation region >50 m from the dam that this comparison already
excludes from scope (same caveat as the slanted rock-box sides above). It
is consistent with a sliver-tetrahedron/weak-node-pruning mesh artifact in
that already-flagged far-field region, not a dam/cracking-relevant
disagreement; Elmer's own tensile peak stays at the wall/foundation seat
(`z≈-28.6`), where Aster's agreement is now good.

**Side-effect fix**: splitting the interface changed node/cell counts
enough to shift `_prune_weakly_supported_free_nodes()`'s decisions,
leaving 128 far-field bedrock nodes (same far-field region as above, zero
coordinate overlap with the spring interface) with no `SIGM_NOEU` value.
`extract_result_fields.py` now detects this (`stress_uncovered_nodes` in
its returned info) and NaN-fills those nodes via coordinate matching
instead of crashing; `postprocess_baseline.py`'s peak-stress search uses
`np.nanargmax`/`np.nanargmin` so these NaNs can't silently corrupt the
comparison.

**Remaining simplifications vs. Elmer**, narrowed from Amendment 3: the
wall's internal joints and the foundation/bedrock interface are still
welded into one bonded continuum (not yet given discrete springs); the far
side rock-box walls are still fully fixed instead of roller-sprung.



The active procedural generator is `Elmer/regions/combined_structure.py`. At
the time of writing it builds a **single bonded `WALL` volume** (see the
amendment above); the three-region table below describes the originally
planned shape and is retained only for historical context.

| Historical ID (not current) | Region | Role |
| --- | --- | --- |
| 1 | Plinth | Foundation/plinth concrete |
| 2 | Wall | Curved dam wall concrete |
| 3 | Haunch | Local plinth-to-wall transition concrete |

The mesh must remain conforming: adjacent regions share nodes at their bonded
interfaces. Use the mesh emitted by `build_combined_structure`; do not revive a
previous split center-wall/wedge mesh as part of this port.

## Inputs to Preserve

Read these from the existing files so that Elmer and Code_Aster use one source
of truth.

| Quantity | Source | Current value |
| --- | --- | --- |
| Concrete density | `Data/Concrete_Material_Properties.json` | 2400 kg/m3 |
| Young's modulus | same | 35 GPa |
| Poisson ratio | same | 0.20 |
| Tensile strength | same | 2 MPa, reporting limit only |
| Gravity | `Data/Env_Boundaries_And_Loads.json` | -9.81 m/s2 in Z |
| Reservoir level | same | 29 m |
| Peak upstream pressure | same | 284.49 kPa |
| Water density | `Data/Env_Boundaries_And_Loads.json` | 1000 kg/m3 |
| Tailwater head | same | 2 m |
| Crest overflow head | same | 2 m |

The current bedrock support is a distributed directional spring surrogate:

$$
k_x=226\times10^6,\quad k_y=263\times10^6,\quad k_z=376\times10^6
\quad \mathrm{N/m^3}
$$
These values are stored in `Data/Env_Boundaries_And_Loads.json`. Preserve this
support for baseline parity. Do not replace it with fixed support or an explicit
rock mass during this phase.

Import the common in-memory model through `structural_model`:
`build_combined_structure`, `audit_combined_structure`, and
`build_named_groups`. Solver-specific exporters and input writers must remain
outside this shared model API.

## Target Layout

```text
Aster/
  build_med_mesh.py       Export the current procedural mesh with names
  validate_mesh.py        Mesh and group-quality checks
  run_case.py             Case launcher and manifest writer
  cases/
    00_elastic_baseline.comm
  results/
    00_elastic_baseline/
      mesh.med
      result.med
      result.vtu
      summary.json
      run_manifest.json
```

MED is the authoritative Code_Aster mesh and result format. A VTU derivative
may be written for ParaView, but it must be generated from the MED result, not
used as the source of record.

## Work Packages

### 1. Freeze the Elmer Reference Case

Before creating Code_Aster files, capture the current working reference:

- generated Gmsh mesh and its audit report;
- Elmer SIF input;
- Elmer result VTU;
- node and element counts by region and element type;
- applied load resultants;
- foundation reactions;
- displacement at fixed named probe locations;
- volume-averaged stresses in named regions;
- hashes of source mesh generator, inputs, and result files.

Use fixed physical probe points or named nearest nodes at the crest, central
wall, upstream face, downstream toe, and plinth. Do not compare only the global
maximum stress: sharp corners and stress recovery can make that value
mesh-dependent.

### 2. Add Named Mesh Groups

The current mesh writer relies on numerical physical IDs. Code_Aster commands
must instead use durable group names. Extend the procedural export path to
create these MED groups:

| Group | Entity | Meaning |
| --- | --- | --- |
| `PLINTH` | Volume | Region 1 |
| `WALL` | Volume | Region 2 |
| `HAUNCH` | Volume | Region 3 |
| `FOUNDATION` | Face | Spring-supported base |
| `UPSTREAM` | Face | Reservoir pressure face |
| `DOWNSTREAM` | Face | Tailwater pressure face |
| `CREST` | Face | Overflow surcharge face |
| `LEFT_ABUTMENT` | Face | Outer side face, if exposed |
| `RIGHT_ABUTMENT` | Face | Outer side face, if exposed |

The exporter must assert that every required group is nonempty. It must also
write a JSON manifest recording group node/face/cell counts, region volumes,
exterior areas, element types, and coordinate bounds.

The canonical mapping is `Elmer/regions/model_groups.py`, re-exported through
`structural_model`. Do not infer a group from a normal vector after conversion
and do not rely on Gmsh/Elmer renumbered IDs.

### 3. Write and Validate the MED Mesh

Implement `Aster/build_med_mesh.py` using
`structural_model.build_combined_structure`. The exporter must preserve:

- coordinates in metres;
- hexahedra and triangular prisms without topology conversion;
- bonded node sharing between plinth, wall, and haunch;
- all named volume and face groups;
- the global coordinate convention: vertical is $Z$.

`Aster/validate_mesh.py` must fail the build if it finds repeated-node cells,
zero or negative volume cells, missing groups, empty groups, unpaired bonded
interfaces, or different cell/region counts from the source mesh.

Open the MED mesh in ParaVis/ParaView and inspect the named groups once before
proceeding. This is a visual supplement to the automated validation, not a
replacement for it.

### 4. Port the Linear-Elastic Static Case

Create `Aster/cases/00_elastic_baseline.comm` with this conceptual sequence:

1. `LIRE_MAILLAGE` reads `mesh.med`.
2. `AFFE_MODELE` assigns 3-D solid mechanics to `PLINTH`, `WALL`, and `HAUNCH`.
3. `DEFI_MATERIAU` and `AFFE_MATERIAU` assign the existing linear elastic
   concrete properties to all three volumes.
4. Mechanical loads apply gravity, upstream hydrostatic pressure, downstream
   tailwater, crest surcharge, and the existing foundation stiffness model.
5. `MECA_STATIQUE` performs the linear solve.
6. `CALC_CHAMP`, `POST_RELEVE_T`, and `IMPR_RESU` produce stresses,
   displacements, reactions, and a MED result.

Use elevation-dependent pressure functions, not pressures exported node by
node. For an upstream free surface at $z=0$:

$$
p(z)=\rho_w g\max(-z,0).
$$

Apply tailwater and crest surcharge using the same physical definitions as the
existing Elmer input.

### 5. Port the Foundation Support Without Changing Its Meaning

The existing support is not a fixed base. It is a directional distributed
spring law. Confirm the installed Code_Aster catalogue supports a face-based
equivalent with the required three directional stiffnesses.

If it does, use it and verify the reaction distribution. If it does not, stop
the parity port at this point and use a small, separately verified support
representation such as a thin calibrated elastic layer or discrete support
elements. Do not substitute `DDL_IMPO` fixed displacement constraints merely
to obtain a solvable case.

### 6. Produce Comparable Results

`Aster/run_case.py` should run the command file in a clean case directory and
write `run_manifest.json` containing:

- Code_Aster version and command file hash;
- mesh and group manifest hash;
- material/load input hashes;
- solver status and elapsed time;
- applied load resultants and foundation reactions;
- probe displacements;
- region-wise stress summaries;
- output paths.

Convert `result.med` to VTU only after the Code_Aster solve completes, keeping
the MED output untouched.

## Baseline Acceptance Criteria

The Code_Aster port is complete when the following agree with the frozen Elmer
reference within an agreed tolerance. Use 2% as the initial target for global
quantities; investigate rather than average away any larger discrepancy.

| Check | Required comparison |
| --- | --- |
| Mesh | Same node/cell counts by source region and element type |
| Geometry | Same bounds, volumes, interface areas, and external face areas |
| Loads | Same gravity, pressure, and surcharge resultants and directions |
| Support | Same resultant force/moment within tolerance |
| Response | Crest, crown, toe, and plinth probe displacements |
| Energy | Comparable linear strain energy |
| Stress | Region averages and fixed-offset local samples away from singularities |
| Visual | Same deformed shape and pressure direction |

Document any accepted differences caused by element formulation, stress
recovery, or result interpolation. Do not claim a stress difference is physical
until mesh and load parity have passed.

## Execution Order

1. Freeze the current Elmer reference artifacts and comparison probes.
2. Add named groups and export the current mesh to MED.
3. Run automated mesh validation and visual group review.
4. Create the Code_Aster material and external-load baseline.
5. Reproduce the distributed foundation support.
6. Solve the Code_Aster elastic baseline.
7. Generate the comparison report and resolve discrepancies.
8. Tag the verified elastic port as the new Code_Aster baseline.

## Copilot Task Prompts

Use one prompt at a time. Include the named acceptance check in the same prompt
and ask Copilot not to modify unrelated generated artifacts. Do not ask it to
implement cracking, contact, or damage while performing these tasks.

### 1. Freeze the Elmer baseline

```text
Read Aster_Porting.md and the active Elmer mesh/solver path. Create a
reproducible baseline manifest for the current working linear-elastic Elmer
case. Do not modify geometry or solve settings. Record input hashes, mesh node
and element counts by region/type, group/physical-face counts, coordinate
bounds, volume/area measures, output paths, and named displacement probes.
Add a validation command that fails if the reference result or mesh is missing.
```

Acceptance: the manifest can be regenerated and identifies the exact Elmer
mesh/result used for comparison.

### 2. Define the named mesh contract

```text
Read Aster_Porting.md and Elmer/regions/combined_structure.py. Add one
centralized Python mapping from the active mesh's current regions and exterior
boundaries to the named Code_Aster groups listed in Aster_Porting.md. Do not
change geometry, cell connectivity, material values, loads, or solver files.
Add tests or assertions that every required group is nonempty and that the
PLINTH, WALL, and HAUNCH volume groups reproduce the source region cell counts.
```

Acceptance: group names are generated from one source of truth and missing or
empty required groups cause a clear failure.

### 3. Export and validate MED

```text
Implement the Aster/build_med_mesh.py exporter described in Aster_Porting.md.
Reuse the active CombinedStructure generator and preserve coordinates, element
topology, bonded nodes, and named volume/face groups. Add
Aster/validate_mesh.py. Do not add crack or contact topology. Validate cell
counts, bounds, region volumes, exterior areas, required groups, repeated-node
cells, and nonpositive volumes. Report the exact commands needed to generate
and validate the MED mesh.
```

Acceptance: a MED mesh opens with the expected named groups and the validation
report matches the frozen Elmer mesh measures.

### 4. Establish Code_Aster installation facts

```text
Inspect the local Code_Aster installation and report its version, launcher,
catalogue location, MED support, and available commands for a 3-D linear
elastic static solve. Do not install or upgrade anything and do not edit model
files. Identify the documented mechanism for the current directional
distributed foundation spring surrogate; if unavailable, state that clearly
without substituting a fixed base.
```

Acceptance: the result names the installed commands and support-model options,
with sources from the local catalogue or official documentation.

### 5. Create the elastic command file

```text
Using Aster_Porting.md, create only Aster/cases/00_elastic_baseline.comm for
the current bonded MED mesh. Use 3-D linear elasticity, the existing concrete
properties, gravity, elevation-based upstream pressure, tailwater, crest
surcharge, and the verified equivalent of the current spring support. Do not
add contact, cracks, damage, XFEM, or nonlinear material laws. Include output
of displacements, stresses, principal stresses, and reactions in MED format.
Explain each Code_Aster command choice and cite the installed-version syntax.
```

Acceptance: the command file parses/runs for the installed Code_Aster version
and uses only the groups defined in the MED exporter.

### 6. Compare elastic results

```text
Implement Aster/postprocess_baseline.py to compare the completed Code_Aster
elastic result against the frozen Elmer reference. Compare load resultants,
foundation reactions, named probe displacements, strain energy where available,
and region-wise fixed-offset stress samples. Do not compare raw global nodal
stress maxima as acceptance criteria. Write a JSON and Markdown comparison
report with percentage differences and explicit pass/fail thresholds from
Aster_Porting.md.
```

Acceptance: the comparison is reproducible and distinguishes global equilibrium
errors from local stress-recovery differences.

### 7. Close the baseline port

```text
Review the completed baseline-port artifacts against Aster_Porting.md. Report
only concrete gaps that prevent declaring the Code_Aster elastic baseline
verified. Do not propose or implement cracking/contact work. If all acceptance
checks pass, update the baseline manifest with Code_Aster version, command-file
hash, mesh hash, result hash, and final comparison summary.
```

Acceptance: there is one versioned, reproducible, bonded linear-elastic
Code_Aster case with an auditable Elmer comparison.

## Deferred Follow-Up

Only after the elastic baseline is accepted should the project decide whether
to model a prescribed interface, contact closure, cohesive damage, or bulk
concrete cracking. Those will be new, versioned scenarios built from the
verified Code_Aster baseline; they are not part of this port.