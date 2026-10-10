# Gairedzi Dam Structural Model

This repository is the clean v2 working copy for the Gairedzi dam study.

The purpose of this repo is to separate the newer plinth-based geometry work from the earlier baseline model and keep the design assumptions explicit.

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
against Elmer, plus an exact two-bar normal-contact benchmark. A 3-D crack-surface
adapter remains the next validation gate; enabled dam crack templates are
currently blocked.

### Reservoir overflow head and pressure

The active reservoir load includes 3 m of water above the z = 0 crest:
`p(z) = rho * abs(g) * max(OverflowHead - z, 0)`.
With fresh water at 1000 kg/m3 and gravity 9.81 m/s2, pressure is 29430 Pa
at the crest and **333540 Pa at z = -31 m** (`1000 * 9.81 * (31 + 3)`).
The slope remains 9810 Pa/m; do not use the corrected peak divided by 31
as a new pressure gradient. The crest also receives 29430 Pa normal pressure
on its own face, which is distinct from upstream-face loading.

`Data/Env_Boundaries_And_Loads.json` stores this reference-depth peak.
The active combined Elmer/FEniCSx run uses `Elmer/load_cases.json`, whose
overflow head is also 3 m. Different tailwater/support settings in these files
remain intentional per-workflow inputs; this pressure correction does not
change them. The independently configured Elmer calibration case is unchanged.
Code_Aster upstream loads use the same head-offset law; crack-face pressure
inherits the case's overflow head unless an explicit scenario override is supplied.
The existing high-pressure crack control retains its deliberate 3 MPa override.

**Previously generated results are stale for this corrected load.** Rerun
**Elmer: Run combined solver**, then **FEniCSx: Verify dam elastic parity**.
The parity runner rejects the old reference SIF rather than comparing different
load cases. For Code_Aster, rebuild case inputs/meshes before rerunning the
elastic or crack cases; archived `case_inputs.json` and scenario overrides are
not automatically rewritten. Dam crack contact in FEniCSx remains blocked;
the synthetic two-bar benchmark does not use reservoir pressure.

## Current radial dimensions

### Raised-crest scenario

The nominal height is now 31 m, raised 2 m from the 29 m reference.
`config.json` records `crest_raise_m = 2`: imported plinth and bedrock
elevations are reduced by 2 m to retain crest z = 0. The surveyed
`Data/plinth.json` remains unchanged; the physical foundation is not lowered.
The shared Elmer/FEniCSx/Code_Aster mesh and standalone curved generator use
these rebased elevations. Local wall heights still follow the plinth contour:
the deepest plinth top is now z = -30 m, with bedrock at z = -31 m.
Thus upstream pressure is 323730 Pa at that wall/plinth interface and
333540 Pa at the reference foundation depth. The 3 m overflow head gives a
34 m reference water depth. Tailwater heads and support properties are unchanged.
All existing meshes/results must be regenerated before reporting this scenario.

The active shared Elmer/FEniCSx model has a **5 m wall** on a **7 m plinth**,
leaving **1 m overhang on both sides**. The upstream wall face remains at
radius 80 m; the downstream face is at 75 m. The wall centreline is 77.5 m,
midway between the plinth faces at radii 74 m and 81 m. Chainage-to-angle
mapping and the source Code_Aster crack locations use this updated centreline.
The standalone curved-dam generator also uses 1 m plinth offsets on both sides.

Existing meshes, dam results and elastic parity references predate this
geometry change. Regenerate the Elmer combined mesh/results before rerunning
FEniCSx dam parity; rebuild Code_Aster meshes and case inputs before rerunning
crack cases. Saved results are not rewritten automatically.

## Background: v1 baseline

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
