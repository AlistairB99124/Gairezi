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

## Root dam configuration

[config.json](config.json) supplies the dam properties for the combined Elmer
solver and shared Code_Aster/FEniCSx mesh. Neither the deleted `Data` inputs nor
the deleted `Elmer/load_cases.json` is read by these workflows.

| Section | Meaning |
|---|---|
| `geometry.global_element_size_m` | Shared mesh spacing in metres |
| `geometry.local_element_size_m` | Corner refinement for standalone geometry generators |
| `geometry.wall` | Wall face radii, radial width and centreline |
| `geometry.plinth` | Plinth radii, signed contour elevations and wall/plinth spring support |
| `geometry.bedrock` | Signed bedrock contour elevations and foundation spring support |
| `material.concrete` | Density, Young's modulus, Poisson ratio and tensile-strength reporting threshold |
| `material.bedrock` | Rock material data; no rock volume is currently meshed |
| `loads` | Gravity, water density, reference water depth, overflow and tailwater heads |
| `outputs` | Requested result metadata; the existing solver/post-processor still writes its standard result fields |

Contour `height` is a signed elevation relative to crest z = 0, not a positive
wall height. Plinth and bedrock stations must match and be strictly increasing.
The mesh follows these contours; `height_above_plinth_m` is nominal metadata,
not a second instruction to shift their elevations. Changing dam height means
updating the contour elevations. There is no implicit crest raise or chainage
rescaling. Radial width must equal upstream minus downstream radius, and each
centreline must lie midway between its own faces.

The shared validator in [Elmer/model_config.py](Elmer/model_config.py) accepts
either a configuration file or a parsed JSON object. It rejects missing or
unknown fields, non-numeric/nonfinite values, invalid geometry and material
values, mismatched contour stations, incorrect support mappings, and invalid
output-flag types. Validation errors include JSON-style field paths.

| Field/capability | Combined Elmer workflow |
|---|---|
| `geometry.global_element_size_m` | Active; controls the shared mesh spacing |
| `geometry.local_element_size_m` | Validated, but does not refine the combined mesh |
| Wall/plinth radii and contour elevations | Active; define the mesh geometry |
| `geometry.wall.height_above_plinth_m` | Nominal metadata; contour elevations define actual geometry |
| Directional support springs | Active; explicit X/Y/Z stiffnesses are used as configured |
| Bedrock material properties | Retained and validated; no bedrock volume is currently meshed |
| `loads.overflow_head`, gravity and water density | Active in hydrostatic loading |
| `loads.water_height` and `loads.overflow_water_level` | Reference-depth metadata; the latter must equal their documented sum |
| `outputs.*` flags | Validated metadata only; the current solver still writes its standard fields and reports |
| `outputs.cracking_risk` | Not a crack/damage analysis; the elastic model does not implement crack propagation |

The supplied model has a **4 m wall** at radii **76-80 m**, on a **6 m plinth**
at radii **75-81 m**. Both centrelines are 78 m, with 1 m overhang on each side.
The deepest wall/plinth interface is z = -28 m and bedrock is z = -29 m.
Support stiffness uses the explicit directional `spring_n_per_m3` values;
rock/concrete elastic moduli do not silently recalculate these springs.

### Reservoir overflow head and pressure

As intended, the water surface is at `loads.overflow_head` above crest z = 0:
`p(z) = water_density * abs(gravity) * max(overflow_head - z, 0)`.
`water_height` is a reference depth, not a reservoir-level control.
`overflow_water_level` must equal `water_height + overflow_head` and is also
a reference depth. The reported reference peak is calculated from that sum,
not independently configured.

With the supplied 31 m reference depth and 3 m overflow head, the reference
peak is **333540 Pa**, but the actual pressure at the model's deepest bedrock
z = -29 m is **313920 Pa**, and at the wall/plinth interface z = -28 m it is
**304110 Pa**. Crest pressure is **29430 Pa**. This distinction is intentional.
Changing `overflow_head`, gravity or water density changes the applied pressure;
changing only the reference depth does not.

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
