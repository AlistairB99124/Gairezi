# Gairedzi Dam Structural Model

This repository is the clean v2 working copy for the Gairedzi dam study.

The purpose of this repo is to separate the newer plinth-based geometry work from the earlier baseline model and keep the design assumptions explicit.

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
- wall_thickness_m: 4.0
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
