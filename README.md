# Gairedzi Dam Structural Model

This repository is the clean v2 working copy for the Gairedzi dam study.

The purpose of this repo is to separate the newer surveyed-foundation geometry work from the earlier baseline model and keep the design assumptions explicit.

On this branch the wall sits **directly on the bedrock**: there is no plinth. The model has two bodies, wall (1) and bedrock (2). Module and file names containing `three_body` are kept for compatibility.

## Integrated wall/bedrock solver

This is now the canonical **WPF app/API** model as well as the CLI model.
New app submissions use the existing isolated job directories, preserve all
configured domain/mesh/spring data, and run `Elmer.three_body_solver`
through its `--config-root`, `--output-dir`, `--report-path` job adapter.
The desktop opens the 5 m reporting VTU by default, retains a full-domain
ParaView action, and shows separate wall/near-field rock principal
stresses. Bedrock is read-only in the editor. Historical jobs retain their
old result paths; legacy FEniCSx baseline/fracture is explicitly blocked for
new three-body jobs, not silently run with incompatible joints. Restart the
updated API and rebuild/relaunch the desktop. See the parent app README.

**Elmer: Run solver** (the default build task) and **Elmer: Run combined solver**
now run the wall and bedrock, including the approved graded 2H bedrock domain.
Equivalent PowerShell command:

```powershell
.\.venv\Scripts\python.exe -m Elmer.three_body_solver
```

Each invocation snapshots `config.json` and creates a fresh run under
`%LOCALAPPDATA%\Lamb\three-body-solver\<timestamp-and-unique-id>\`.
The command prints the run directory before preparation and the solved result
path after verification. `run.json` records preparing/prepared/solving/succeeded
or failed status, and errors propagate as a failed command. `latest-run.json`
in the output root points only to the last successfully verified run; check the
current run's status, not that pointer, to establish whether a new run succeeded.

Open the printed `model\results\three_body_results_t0001.pvtu` in ParaView.
Keep its adjacent VTU pieces. Colour by cell `GeometryIds`: **1 = wall,
2 = bedrock**, or select displacement and stress fields. These are
solved fields, unlike the geometry-only preview below.

### Stress reporting region: concrete and nearby bedrock

Three-body runs also generate `model\reporting\structure_and_near_rock.vtu`
and `model\reporting\stress_summary.json`. **Use this VTU for stress review**:
it contains all wall cells and only bedrock within **5 m of the
nearest wall surface**, in undeformed coordinates. All solved fields
are retained. Rescale ParaView's colour map to this filtered dataset; do not
reuse the full-domain colour range.

The full-domain PVTU is preserved and remains `result` in the run manifest.
`reporting_result` and `stress_summary` identify the filtered deliverables.
This is post-processing only: it does not shorten the rock domain, alter
restraints or rerun the solver. Remote bottom stresses are excluded from the
filtered bedrock summary, **not erased from the full solution**. Fixed-boundary
stresses can include physical overburden as well as restraint/domain effects;
hiding them does not establish that the 5 m region is free of boundary effects.
Foundation-domain sensitivity and geostatic initialization remain separate
verification tasks.

Surface distance is computed at rock vertices; the 5 m boundary is clipped
using linearly interpolated distances within cells. Boundary fields are
interpolated too, and principal stresses are recomputed from the interpolated
stress tensor. This is a mesh-dependent approximation to the exact distance
offset, not a bounding-box or centroid selection. Mesh refinement is needed
if a more exact offset is required. The summary records these limitations and
reports each body's tensile/compressive principal stress peaks and locations.

For an existing attributed result, use ParaView's Python (PowerShell):

```powershell
& "C:\Program Files\ParaView 6.2.0\bin\pvpython.exe" .\Elmer\stress_reporting_region.py --result "C:\path\to\model\results\three_body_results_t0001.pvtu" --output "C:\path\to\new-reporting-folder"
```

The output directory must be new or empty; source results are never rewritten.
`--distance-m` defaults to 5. Run the focused VTK tests using
`pvpython.exe -m unittest Elmer.test_stress_reporting_region` from the repository.

The analysis uses the same 80 m survey reference, seat and graded rock mesh as
the approved preview. Concrete cells are subdivided into conforming linear
tetrahedra without adding or moving nodes, following the legacy solver's concrete
element formulation; rock uses conforming distance-graded linear tetrahedra.
The shared wall builder is used by both this solver and the geometry preview.
It uses each body's configured density, Young's modulus
and Poisson ratio and applies self-weight to both bodies. With a spring seat, the wall/rock
interface uses the directional support values with separate coincident nodes:

| Joint | Configuration source | Current x / y / z stiffness (N/m³) |
|---|---|---|
| Wall / bedrock | `geometry.bedrock.support.spring_n_per_m3` | 1.25e9 / 1.25e9 / 3e9 |

With `geometry.bedrock.support.model: bonded` the wall and rock share seat
nodes instead and these coefficients are inactive.

Directions are **global Cartesian x/y/z**, not local normal/tangential axes.
For each component, traction = configured stiffness times the relative
displacement between the two bodies. Zero stiffness releases that component;
an inadequately restrained model can fail to solve. No stiffness is inferred
from Young's modulus. These coefficients are applied only between bodies:
the old grounded foundation springs are **not also applied**.
The temporary top-level `interfaces` section has been removed and is rejected
by validation rather than silently overriding the support values. Remove that
obsolete section from older input configurations before starting new runs.
Each new run snapshots its directional values and records their source paths
in `diagnostic.json`; already-running jobs keep their original snapshots/SIFs.
The gap plausibility check uses the corresponding directional stiffness
and rejects any component implying traction above 100 MPa. This is not a design
acceptance criterion. Interface springs carry tension; they are not a
compression-only, frictional or opening/sliding contact law.

Initial boundary/load assumptions retained from the three-body diagnostic:

- Outer rock support from `material.bedrock.support`. With `model: rollers`
  (the approved default), the rock bottom (-87 m) and the four vertical rock
  sides carry normal-only penalty springs `k n n` (`spring_n_per_m3` x = bank
  faces, y = upstream/downstream faces, z = bottom). Tangential motion is free,
  so the rock can expand laterally under its own weight instead of being
  artificially clamped. If the block is absent, the historical fixed bottom
  (all components zero) with free sides is used.
- Exposed rock top traction-free.
- Existing upstream/downstream concrete water loads, preserving the concrete
  tailwater datum rather than lowering it to the new rock bottom.
- Gravity and water loads applied simultaneously; no geostatic initialization.
- Zero uplift and no direct hydrostatic pressure on exposed rock.

`model\diagnostic.json` records these assumptions, mesh dimensions, cell/body
counts, joint stiffnesses and checks. Its historical filename is retained because
the integrated solver reuses that preparation/verification implementation.
`model\solver.log` contains the MPI solver output. The task relays new log output
and prints an elapsed-time heartbeat every 30 seconds while solving; a heartbeat
means the process is still running, not that assembly or convergence is advancing.
Ctrl+C in the task terminal stops that run's MPI process tree and records
`cancelled` in `run.json`. Successful runs require solver
success, complete source-cell coverage, finite displacement/stress fields, and
both spring joints passing the existing gap plausibility checks. These checks
are not equilibrium, mesh-convergence or dam-safety certification.

Install Python meshing dependencies with `pip install -r client_requirements.txt`
(including Gmsh 4.15.x). ElmerGrid, MPI-enabled ElmerSolver, Microsoft MPI and ParaView's `pvpython` must
be installed. Spring partitioning keeps both sides of the joints on the same
rank; this may limit parallel scaling. The integrated solve retains the legacy
BiCGStabl method and 1e-10 convergence tolerance, but uses **ILU1** with
`Linear System ILU Factor = 0.01`. This relaxes the preconditioner diagonal only,
not the physical stiffness matrix or spring values. Fixed-level ILU fill replaces
the previous 1e-8 ILUT threshold: that threshold caused severe memory pressure
on the approved rock mesh (about 83 GiB committed on a 32 GiB machine).
ILU1 limits graph-level fill, not absolute RAM usage, and full-resolution
convergence must still be checked. Residuals are printed every 20 iterations.
The selected settings are recorded in `model\diagnostic.json`.
`--partitions 1` uses an unpartitioned
mesh and wraps serial output in a labelled PVTU.
The configured graded mesh contains approximately
215,000 rock tetrahedra (rather than the former 6.56 million layered cells).
It can still require substantial RAM and runtime. User runs
have **no automatic ten-minute cap**. `--timeout-s` explicitly requests a solver
timeout; agent development checks remain separately limited to ten minutes.
Other options are `--root`, `--output-root`, `--partitions` and `--prepare-only`.
Preparation-only runs never advertise a solved result.

If an older run is already paging heavily, cancel it before restarting with the
new task; source changes do not update an already-loaded SIF. Do not run two
full-resolution jobs concurrently. Check RAM, committed memory and the latest
residuals rather than assuming that an active CPU means useful progress.

The prior two-body CLI `Elmer\structure_assembly.py combined-solve` and its
defaults remain unchanged, available as **Elmer: Run legacy wall-only
foundation-spring solver**. Legacy client reports and the existing FEniCSx
tasks still consume that two-body result; they are not three-body post-processing.
Use the labelled ParaView fields for the new analysis, not the legacy
concrete-only report generator.

### Wall underside on the bedrock

There is no plinth. The wall **underside** follows straight sloping segments
between the unchanged `geometry.bedrock.contours` elevations, across the full
wall thickness. It is not rounded to horizontal courses and is not spline
smoothed. Survey breakpoints and the flat valley-bottom segment remain. The wall
spans the chainages where the rock lies below the crest (z = 0; about 6 to 174 m
for the supplied contours) and tapers to zero height at both ends.
Standalone preview and solver meshes use the same linear seat profile. The rock
mesher locks its seat to those exact wall-underside triangles; a bonded seat
shares displacement nodes, while a spring seat keeps coincident but separate
wall and rock nodes (boundary tags 11 = wall side, 12 = rock side).
`geometry.wall.height_m` is nominal metadata; the contours set the real height.

## Three-body bedrock geometry review (Elmer scope)

**The legacy combined solver result contains only the wall.** Its
bedrock support is a spring boundary condition, not a visible rock volume.
Changing the bedrock contours does not add a rock body to that solve or its
stress CSV. Do not use the geometry preview as a bedrock stress result.

Run **Elmer: Preview wall and bedrock (geometry only)** from VS Code,
or generate the standalone ParaView geometry preview in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m Elmer.bedrock_preview --output "$env:LOCALAPPDATA\Lamb\bedrock-preview"
```

Open `three_body_geometry.pvtu` and colour by cell `RegionId`: 1 = wall,
2 = bedrock. Both adjacent VTU files are required. Concrete
uses `material.concrete`; rock uses `material.bedrock` from `config.json`.
The rock starts from the approved 80 m-survey chord section, extruded through
the concrete's upstream/downstream bounds. Its top
matches the wall underside exactly at the seat, with a triangulated
transition to the projected canyon profile at the outer perimeter. By default
the rock uses `geometry.bedrock.domain` from `config.json`:

| Domain setting | Current 2H candidate |
|---|---:|
| `stream_margin_m` | 58 m upstream and downstream of the concrete's projected depth bounds |
| `bank_margin_m` | 58 m beyond each end of the survey/concrete cross-valley span |
| `bottom_z_m` | -87 m: 58 m below the deepest contact at -29 m |

These are explicit dimensions based on H = 29 m (crest to deepest dam/rock
contact), not a claim of compliance with a universal design rule. They do not
automatically change if the contours are edited. A domain-size and boundary-condition
sensitivity study is still required. Beyond the survey endpoints, the bank terrain
is assumed flat at the end survey heights; no additional site geology is implied.
The canyon section is held constant upstream/downstream outside the seat transition.
The current bedrock profile reaches z = +5 m
at chainages 0 and 180, above the z = +3 m surcharge water surface, while the
wall ends where the rock reaches the crest (chainages 6 and 174). The diagnostic below accepts the same
options. `--stream-margin-m`, `--bank-margin-m` and `--bottom-z-m` override the
configured dimensions for either command without rewriting the configuration.
Configurations without the optional `domain` section retain the legacy 20 m
stream margins, 5 m bank margins and -31 m bottom.

**The preview is geometry only, not an Elmer solution.**
Both preview and three-body solver now use the same Gmsh **conforming 3-D
tetrahedral rock mesh**, controlled by `geometry.bedrock.mesh`:

| Setting | Default | Meaning |
|---|---:|---|
| `near_size_m` | 0.5 | Nominal near-interface element length |
| `far_size_m` | 15 | Nominal remote-rock element length |
| `near_distance_m` | 2 | Fine buffer from the wall underside |
| `far_distance_m` | 50 | Distance at which the far-field target is reached |

A Gmsh `Distance` field samples the actual triangulated wall-seat **surfaces**
(`SurfacesList`, not `VolumesList`); a `Threshold` field grades linearly between
these distances in all three dimensions. Existing seat nodes and all seat
triangles are locked, including their diagonals. The rock exterior retains the
underlying faceted terrain/control profile instead of resampling it onto a
tensor-product grid. No terrain-smoothing filter is applied.
Fine surface nodes no longer force fine columns all the way down to the bottom.
The volume is generated from a closed discrete shell to avoid artificial CAD
seams trapping flat tetrahedra. Mesh generation verifies exact seat faces, positive
cell volumes, nondegenerate scaled Jacobians, total volume, paired interior faces
and a closed exterior matching the generated surface mesh.

These are **nominal lengths, not cell volumes or maximum-edge guarantees**:
tetrahedron diagonals and locked seat edges can be longer. The surface/internal
mesh is unstructured, so the old parallel-row appearance is intentionally not
retained. Configurations without `geometry.bedrock.mesh` retain the previous
layered-mesh policy for compatibility.
For the current configuration, a full preview and integrated preparation produced
**215,042 rock tetrahedra**, versus 6,561,612 old layered cells (96.7% fewer).
All **6,600 seat triangles** matched; sampled near/far median edge lengths were
approximately **0.70 m / 15.0 m**. The minimum scaled Jacobian was about **0.0108**;
positive quality is not a mesh-convergence study or engineering acceptance.
`geometry_review.json` records the counts, quality and deterministic spacing
sample (centroid distance to nearest seat vertex, an approximation to surface
distance). Full four-rank ElmerGrid conversion and spring partition checks passed;
this does not establish convergence of the full dam solve.
A small flat-terrain, reduced-load fixture also passed the complete four-rank
solve, source-cell attribution, finite stress/displacement and spring-gap checks.
The preview uses the confirmed 80 m
survey reference to convert chainage to angle for concrete, retaining configured
physical radii, thicknesses, elevations and topology. This differs from the legacy
production builder's 78 m chainage reference. Review `geometry_review.json` for
the exact seat-match error, body counts, volume and extent. A matching geometric
surface alone does not assign a contact/bond law. The integrated solver assigns
the configured spring laws and the bottom restraint described above.
The optional `bedrock_top_cross_section` draft is
validated and retained, but not used as a second source of survey data.
No API, FEniCSx, existing support or solver changes are made by this
preview command.

### Isolated bonded three-body static diagnostic

```powershell
.\.venv\Scripts\python.exe -m Elmer.three_body_diagnostic --output "$env:LOCALAPPDATA\Lamb\three-body-diagnostic-new"
```

Use a new empty directory. This runs four MPI ranks of Elmer with concrete
and rock self-weight and the existing concrete water loads. The wall/rock
seat is bonded for this diagnostic only; the rock bottom
(currently z = -87 m) and rock sides use the `material.bedrock.support` model
(normal-only roller springs by default; fixed bottom and free sides if absent),
and uplift is zero. Existing
grounded springs are not active. The concrete tailwater datum is preserved.
The rock uses the same configured domain and graded mesh as the preview, rather
than four coarse vertical layers. Enlarging the domain does not validate the
outer rock support assumptions. This is not a mesh-converged engineering result.
It does not implement the selected opening/sliding law.

The solved PVTU is `results/three_body_diagnostic_t0001.pvtu`.
ParaView's `pvpython` must be available on PATH or in its standard Windows
installation directory for body attribution. That step uses VTK to preserve
Elmer field arrays, removes auxiliary surface cells, and matches volume cells
to the source mesh after subtracting displacement from Elmer's deformed output
coordinates. Displacement and stress values are not recomputed.
Its cell `GeometryIds` identifies 1 = wall, 2 = bedrock.
Unlike the preview's region 6, solver body IDs are contiguous to prevent
ElmerGrid renumbering. `diagnostic.json`, the configuration snapshot, SIF,
mesh and `solver.log` record the run. The development solve has a 480-second
limit and terminates its own process tree if exceeded; this does not set an
app/API timeout or change the production workflow.

## Python environment setup

The local environment needs NumPy, Matplotlib, and meshio. It does not require
`h5py`: Code_Aster MED files are read and written using MEDCoupling inside the
existing Code_Aster Docker container, independently of the local Python version.

### Windows (PowerShell)

Create the repository virtual environment and install the Python dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r client_requirements.txt
```

If a previous install failed while building `h5py`, activate your existing
environment and rerun the install with the updated requirements:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r client_requirements.txt
python -c "import numpy, matplotlib, meshio; print('Dependencies OK')"
```

If PowerShell blocks activation scripts, allow them for the current user, then activate the environment:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

Activate the environment in each new PowerShell session before running Python workflows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r client_requirements.txt
```

Activate the environment in each new shell before running the Python workflows:

```bash
source .venv/bin/activate
```

### Code_Aster MED result conversion

Start Docker Desktop before running Code_Aster or converting its MED output.
The existing `CODE_ASTER_IMAGE` setting selects the solver image (default:
`jesusbill/code_aster:latest`); it must provide Python, NumPy, and MEDCoupling.
If Docker is not on PATH, set `DOCKER_BIN` to the full path of its executable.

```powershell
python Aster\convert_med_to_vtu.py Aster\results --overwrite
```

Conversion leaves the MED source unchanged and exports the top-dimensional mesh
and the last stored iteration of every nodal field. Original field names, including
`result__DEPL` and `result__SIGM_NOEU`, are preserved, with six stress components
and principal stresses added for ParaView and the crack-sweep analysis.
Non-nodal fields are reported and omitted; MED remains the authoritative source
for groups, lower-dimensional mesh levels, and other result data.
Temporary export files are cleaned up, and an existing VTU is replaced only after
conversion succeeds.

### Elmer combined MPI solve

The combined solve requires an MPI-enabled Elmer installation (`ElmerSolver_mpi`
and `ElmerGrid` on PATH) and an MPI launcher. On Windows, install Microsoft MPI:
the launcher is `mpiexec.exe`, not `mpirun`. The script searches PATH, `MSMPI_BIN`,
and the standard `C:\Program Files\Microsoft MPI\Bin` location. On Unix, it prefers
`mpirun` and also supports `mpiexec`.

```powershell
python Elmer\structure_assembly.py combined-solve
```

The solve uses four MPI processes. `ELMER_HOME` is preserved if set; otherwise
it defaults to the installation directory above the selected solver's `bin`
directory. Missing solver or launcher errors are reported before rebuilding the
mesh.

## FEniCSx migration and configurable scenarios

The new [FEniCSx workflow](FEniCSx/README.md) runs in Ubuntu/WSL2 and reads
[cracking.json](FEniCSx/cracking.json) for task-driven scenario updates.
It provides environment checks, a dam migration inventory, an exact elastic
benchmark, a checked tetrahedral dam import and a provisional elastic comparison
against Elmer, an exact two-bar normal-contact benchmark, and a staggered hybrid
AT2 phase-field fracture solver. Explicit 3-D crack-face contact remains
unimplemented; the phase-field solver therefore does not prevent crack-face
interpenetration under compression.

## Root dam configuration

[config.json](config.json) supplies the dam properties for the combined Elmer
solver and shared Code_Aster/FEniCSx mesh. Neither the deleted `Data` inputs nor
the deleted `Elmer/load_cases.json` is read by these workflows.

| Section | Meaning |
|---|---|
| `geometry.global_element_size_m` | Shared mesh spacing in metres |
| `geometry.local_element_size_m` | Corner refinement for standalone geometry generators |
| `geometry.wall` | Wall face radii, radial width and centreline |
| `geometry.bedrock` | Signed bedrock contour elevations (the wall seat) and wall/rock seat support |
| `material.concrete` | Density, Young's modulus, Poisson ratio and tensile-strength reporting threshold |
| `material.bedrock` | Rock material data; no rock volume is currently meshed |
| `loads` | Gravity, water density, reference water depth, overflow and tailwater heads |
| `outputs` | Requested result metadata; the existing solver/post-processor still writes its standard result fields |
| `crack_simulations` | Optional; declared phase-field cracks consumed by [FEniCSx/crack_field.py](FEniCSx/crack_field.py) |

Contour `height` is a signed elevation relative to crest z = 0, not a positive
wall height. Bedrock stations must be strictly increasing and must dip below
the crest so the wall has a seat. A `geometry.plinth` section is rejected as an
unknown field. The mesh follows these contours; `geometry.wall.height_m` is nominal metadata,
not a second instruction to shift their elevations. Changing dam height means
updating the contour elevations. There is no implicit crest raise or chainage
rescaling. Radial width must equal upstream minus downstream radius, and each
centreline must lie midway between its own faces.

The shared validator in [Elmer/model_config.py](Elmer/model_config.py) accepts
either a configuration file or a parsed JSON object. It rejects missing or
unknown fields, non-numeric/nonfinite values, invalid geometry and material
values, mismatched contour stations, incorrect support mappings, and invalid
output-flag types. Validation errors include JSON-style field paths.

### Crack simulations

`crack_simulations` is an optional array describing pre-existing cracks. Each
entry declares `type` (`phase_field`), `shape` (`rectangular`),
`central_crack_pos_m`, `size_m`, an `orientation` normal,
`phase_field_parameters`, loading increments and explicit solver/refinement
controls. Only `model: AT2` and `energy_split: volumetric_deviatoric` are
currently supported; unsupported alternatives fail validation.

`size_m.x` and `size_m.y` span the crack plane while `size_m.z` is the opening
measured **along the orientation normal**, so a crack is a thin rectangular box
rather than a zero-thickness surface. The crack axes come from the shortest-arc
rotation carrying global +z onto the configured normal, so the default normal
`(0, 0, 1)` leaves `size_m.x`/`size_m.y` aligned with the global axes.

[FEniCSx/crack_field.py](FEniCSx/crack_field.py) turns these entries into a
`dolfinx.fem.Function` holding the regularised damage of the crack, built from
the exact signed distance to that box. **Damage is 1 for fully cracked material
and 0 for intact material**, the usual AT1/AT2 convention and the opposite of a
stiffness-retention factor. The field is an *initial condition* representing a
declared crack: it is neither a computed crack path nor a physical crack opening
displacement, and producing it asserts nothing about dam safety.

The initial field saturates at 1 inside the crack box and uses the AT2 optimal
profile `exp(-d / l)` outside it. Overlapping cracks combine by maximum damage.
The solver locally refines the diffuse band until the configured minimum
elements per `length_scale_m` is reached or fails after the configured maximum
levels. It preserves cell and exterior-facet tags through refinement.

The checked-in serial WSL example targets `0.9` length scales per maximum
tetrahedron diameter and at most three refinements. This produces a runnable
preliminary mesh (`h_max` about `0.109 m` near this crack) but does **not**
resolve `l=0.1 m` to the recommended 2-4 elements. Raising the target to `2.0`
requires five refinements and generated approximately 12.5 million crack-band
tetrahedra plus a 1.8 GB mesh in testing; that exceeded the available serial WSL
memory. Use parallel/distributed meshing and solve resources for that target,
then perform a mesh/length-scale convergence study. The app workflow instead
pauses after the elastic baseline and asks the user to click the crack location
on a Z-Y plot of baseline wall stress (no automatic placement), then
seeds a local 4 m crack there so that the wall (but not
bedrock) can damage, with a 0.25 m length scale and a 4,000,000-cell refinement
budget; see [FEniCSx/README.md](FEniCSx/README.md).

The staggered solve alternates degraded linear mechanics and a PETSc TAO
bound-constrained AT2 damage minimisation. The lower bound is the damage from
the previous converged load step, enforcing irreversibility. Gravity,
hydrostatic boundary pressure and the existing directional support springs are
included. Nonzero internal crack pressure uses the regularised force
`-p*grad(alpha)`; this is not an explicit face traction. Outputs include
displacement, damage, tensile-energy history, degraded stress, strain,
crack-surface density, energies, convergence histories, reactions and force
balance. Successful convergence is explicitly marked as not engineering
validated.

`solver.field_output_interval` controls expensive ParaView field snapshots;
JSON convergence and energy diagnostics are still recorded at every load step.
The default `10` writes the final fields for the current ten-step run and avoids
multi-gigabyte per-step output.

The configured central horizontal crack uses the existing scenario location at
chainage `108.5 m`, elevation `-25 m`, mapped to the wall centreline by
`(x, y) = (r sin(s/r), r cos(s/r))`. For `r=78 m` this is Cartesian centre
`(76.743008, 13.946709, -25) m`. Placement validation confirms that centre and
the diffuse band lie in concrete-wall region 2. Crack coordinates in this
section are always global Cartesian coordinates, not chainage/radius values.

| Field/capability | Combined Elmer workflow |
|---|---|
| `geometry.global_element_size_m` | Active; controls the shared mesh spacing |
| `geometry.local_element_size_m` | Validated, but does not refine the combined mesh |
| Wall radii and bedrock contour elevations | Active; define the mesh geometry |
| `geometry.wall.height_m` | Nominal metadata; contour elevations define actual geometry |
| Directional support springs | Active; explicit X/Y/Z stiffnesses are used as configured |
| Bedrock material properties | Retained and validated; no bedrock volume is currently meshed |
| `loads.water_height`, `loads.overflow_head`, gravity and water density | Active in hydrostatic loading |
| `loads.overflow_water_level` | Reservoir depth above the deepest seat; must equal `water_height + overflow_head` |
| `outputs.*` flags | Validated metadata only; the current solver still writes its standard fields and reports |
| `outputs.cracking_risk` | Not a crack/damage analysis; the elastic model does not implement crack propagation |
| `crack_simulations[*]` | Active in the FEniCSx `run-fracture` action; placement, refinement, load stepping and solver controls are validated |

The supplied model has a **4 m wall** at radii **76-80 m** (centreline 78 m)
seated directly on the bedrock. The deepest wall/rock seat is z = -29 m.
Support stiffness uses the explicit directional `spring_n_per_m3` values;
rock/concrete elastic moduli do not silently recalculate these springs.

### Reservoir overflow head and pressure

The reservoir depth is measured from the deepest wall seat (the lowest
bedrock contour, z = -29 m; crest z = 0). The upstream water surface is at
`z_s = z_seat + water_height + overflow_head` and
`p(z) = water_density * abs(gravity) * max(z_s - z, 0)`.
The crest surcharge is `water_density * abs(gravity) * max(z_s, 0)`, so it is
zero when the reservoir is below the crest. `overflow_water_level` must equal
`water_height + overflow_head`.

With the supplied 29 m water height and 3 m overflow head, z_s = +3 m: the
pressure at the deepest seat is **313920 Pa** and at the crest **29430 Pa**.
Setting `water_height` and `overflow_head` to 0 empties the reservoir
(gravity-only upstream face); tailwater is controlled separately by
`tailwater_head`.

Run **Elmer: Run combined solver**, or in PowerShell:

```powershell
.\.venv\Scripts\python.exe -B Elmer\structure_assembly.py combined-solve
```

Then run **Elmer: Regenerate combined client results** to refresh plots/reports.
Directly running an existing SIF does not reload JSON. Code_Aster's
`build_med_mesh.py` derives mesh and generated `case_inputs.json` from the same
root configuration. Rebuild existing Code_Aster cases and FEniCSx imports before
comparing new results; archived results are not automatically migrated.
The separate calibration and crack/scenario JSONs remain scenario-specific
inputs, not additional sources for the shared baseline dam properties.

## Background: v1 baseline

> Historical notes. The v2 plinth-and-wedge interpretation described below was
> later superseded; on this branch the wall sits directly on the bedrock.

The original v1 model was a straightforward curved dam wall built directly from the general base contour data in Data/Dam_Base_Contours.json.

Key characteristics of v1:

- dam wall represented as a curved wall section generated from the base contour profile
- wall thickness treated as a fixed nominal value of 4.0 m
- crest elevation effectively taken as the dam height level above the local ground profile
- no explicit plinth profile in the geometry generator
- no separate wedge transition from ground to dam body
- the geometry was developed as a general curved wall approximation rather than a plinth-and-wedge foundation detail

So v1 is best thought of as a baseline structural model for the dam body itself.

## v2 design intent

The v2 model is intended to represent the updated client geometry more faithfully, using the information in:

- Data/plinth.json
- Data/plinth.png
- Data/Dam_Base_Contours_clean.json

The design change is not just a cosmetic adjustment. It represents a change in geometry logic:

- the dam body should sit above a plinth line
- the plinth is interpreted as the top-of-plinth elevation above datum, not as a thickness value
- the rise above ground is calculated as plinth minus groundLevel
- a wedge transition is introduced between the plinth and the wall body
- the wall is built above the plinth instead of directly from the original generic contour profile

This matches the visual cross-section in Data/plinth.png: the wall height, the wedge, and the plinth are separate geometric features that should be captured in the model.

## Current v2 assumptions and parameters

The v2 workflow is controlled via the config file at v2/config.json.

Current values:

- geometry_mode: v2
- wall_thickness_m: 5.0
- wall_height_above_plinth_m: 29.0
- wedge_angle_deg: 60.0

These values should be treated as editable design inputs for the v2 geometry.

## What is changing from v1 to v2

### 1. Wall thickness

In v1, thickness is effectively an embedded constant in the mesh generator.

In v2, thickness is parameterized so it can be changed without editing the geometry script itself.

This is the purpose of the config file in v2/config.json.

### 2. Plinth

In v1, the model does not explicitly include a plinth profile. In v2, the plinth is treated as a real top-of-plinth elevation profile along the dam chainage.

Interpretation:

- chainage = distance along the dam
- groundLevel = existing ground level at that station
- plinth = top of plinth elevation at that station
- riseAboveGround = plinth - groundLevel

### 3. Wedge

The wedge is the sloped transition between the plinth and the dam wall profile.

The design sketch implies a wedge angle of 60 degrees, and the horizontal offset of the wedge is governed by:

- wedge horizontal offset = (plinth rise) / tan(60°)

This is intended to create the stepped transition seen in the section drawing.

### 4. Wall height above plinth

The wall is not simply taken as a fixed 30 m crest above the original baseline profile.

In v2 the wall sits above the plinth:

- crest elevation = plinth elevation + wall_height_above_plinth_m

This is the geometric change that is currently intended for the updated design model.

## Repository layout

```text
Data/
  Computational_Grid_Controls.json
  Dam_Base_Contours.json
  Dam_Base_Contours_clean.json
  plinth.json
  plinth.png
  ...

Elmer/
  build_curved_dam_geometry.py
  dam_model.sif
  analyze_stress.py
  curved_dam_mesh.msh
  results/

v2/
  build_plinth_geometry.py
  config.json
  README.md
  validate_client_profile.py
```

## Current working intent

This repo is intended to be the clean v2 branch, with the original v1 model left in the earlier project and not mixed into the active workflow.

The immediate goal is:

1. validate the v2 plinth interpretation against the client data
2. keep wall thickness and geometric assumptions configurable
3. generate the v2 mesh from the plinth-based profile
4. run the Elmer solve on the new geometry and compare against v1

## Notes

- v1 should be considered the baseline engineering approximation
- v2 should be considered the revised client design geometry
- the current state is intentionally a transition from one geometry basis to another
- any final design comparison should be made between the v1 and v2 outputs rather than mixing them into a single model

## Practical next step

Run the geometry validation script:

```bash
source .venv/bin/activate
python3 v2/build_plinth_geometry.py
```

Then, once the geometry assumptions are accepted, regenerate the Elmer mesh and solve with the v2 config applied.

- linear elastic concrete
- surrogate crest detailing
- simplified foundation/abutment interaction
- no explicit galleries, contraction joints, fillets, uplift modeling, thermal loading, or nonlinear cracking/damage

These simplifications are expected to affect compressive stress distribution and peak values.

## Recommended Next Steps

1. Align a formal load-combination matrix with the client for the target compression benchmark.
2. Upgrade support/foundation representation beyond spring surrogates.
3. Add additional load mechanisms (uplift, thermal, silt/operational variants as required).
4. Perform mesh-convergence checks specifically on principal stress peaks.
5. Progress to nonlinear material/damage modeling for cracking risk studies.

## Additional Documentation

- `Elmer/ENGINEERING_SUMMARY.md` for current conclusions and recorded next moves.
