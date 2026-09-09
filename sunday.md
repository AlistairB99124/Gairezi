# Gairezi Dam Model - 2026-09-06

## Purpose

Today's work focused on rebuilding the downstream wedge from the supplied plinth profile, making the mesh topology valid at the wall-plinth-wedge junction, and establishing what mesh refinement can and cannot resolve.

The governing design inputs are in `Data/plinth.json`, `Data/Computational_Grid_Controls.json`, and `config.json`.

## Geometry Rules Implemented

The downstream wedge is generated from the local plinth depth at each chainage. The
depth values are positive distances below the crest (`z = 0`); the corresponding
mesh elevations are negative (`p = 29` means `z = -29`). The integer depth threshold
and slope angle are the current design inputs:

$$
wedgeZLength = plinth - 25\,m
$$

$$
wedgeXLength = wedgeZLength\tan(26.57^\circ)
$$

The wedge is only active where `plinth >= 25 m` (`wedgeZLength >= 0`). It starts and
ends at the plinth-derived threshold crossings:

- Start chainage: `89.719870 m`
- End chainage: `123.032520 m`

The wall centreline radius is `78 m`; the downstream wall face and wedge attachment are on the exact `76 m` radius. The generator now uses the analytic radial normal rather than an estimated chord normal, so the downstream face radius is numerically constant at 76 m.

## Mesh Changes Completed

### Curved wall and wedge

- Restored the wedge after confirming that a 1 m chainage trim created hard end caps and distorted fan-shaped cells.
- Removed artificial 0.05 m station clustering that created visible vertical seam bands at wedge boundaries.
- Replaced the two wall station slabs that cross the wedge-convergence boundaries with tetrahedral transition cells.
- This removed the previously detected collapsed wall columns. The current topology audit reports no collapsed wall edges, no zero-volume elements, and no internal 2D boundary faces.

### Plinth

- Converted the finite-thickness plinth from a single tall extrusion to layered hexahedral cells.
- Maximum plinth thickness from the profile is `4.380 m`.
- The current plinth has `9` vertical layers, giving a maximum nominal vertical increment of `0.486667 m`.
- Normal wall thickness spacing is `0.5 m` across `8` layers.

### Current production mesh state

- `403` chainage stations
- `59` wall vertical layers
- `9` plinth vertical layers
- `251,920` volume elements in the latest validated hybrid baseline
- Hexahedral wall/plinth bulk with local tetrahedral wall transition slabs at the two wedge convergence boundaries
- Wedge base remains bonded to the plinth through shared faces

The current manual writer is `Elmer/build_curved_dam_geometry.py`. The current production mesh is `Elmer/curved_dam_mesh.msh`, converted for Elmer in `Elmer/mesh`.

## Mesh-Size Findings

`0.5 m` is the mesh target, not a literal requirement for every curved or transitional element. The normal structured bulk is close to this target:

- Normal bulk hexes: median edge length about `0.500 m`
- Normal bulk hexes: 95th-percentile edge length about `0.5385 m`
- Normal bulk hexes: maximum edge length about `0.6405 m`

The remaining local outliers are in the wedge-to-plinth transition:

- `144` wedge-adjacent hexahedral edges exceed `0.75 m`
- Maximum remaining hexahedral edge: `0.991982 m`

Directly refining the wedge radial/plinth grid from `0.5 m` to `0.25 m` increased model size without reducing this maximum edge. Direct tetrahedral replacement of the wedge-adjacent plinth strip also failed because it broke the quadrilateral wedge-base interface (`unmatched-wedge-bases=280`), so that experiment was reverted.

The strict mesh-size checker is `Elmer/audit_mesh_size.py`. The topology checker is `Elmer/audit_curved_dam_mesh.py`.

## Gmsh Transition-Mesh Investigation

Installed and verified the current Gmsh Python API:

- Gmsh version: `4.15.2`

Created a staged Gmsh mesh generator, `Elmer/build_transition_mesh.py`, to prove a non-distorting monolithic transition geometry:

- Builds wall, finite-thickness plinth, and curved wedge as separate ruled volumes.
- Fuses them into one concrete body before meshing.
- Applies global `0.5 m` sizing and local `0.1 m` convergence sizing.
- Creates all six physical surface groups required by `Elmer/dam_model.sif`.
- Enforces a minimum tetrahedral quality of `0.005`.
- Current staged mesh minimum tetrahedral quality: `0.038095`.
- Gmsh reports no ill-shaped tetrahedra.
- ElmerGrid converts it successfully to `Elmer/mesh_transition`.

This staged mesh solves the geometric distortion and bonding problem, but it is all tetrahedral. It is not promoted to the production mesh because it does not meet the desired hexahedral wall/plinth bulk approach.

An experiment reducing its global size from `0.5 m` to `0.25 m` was rejected:

- Produced about `4.79 million` tetrahedra.
- Required about `203 s` for 3D meshing.
- Left ill-shaped tetrahedra and failed the quality threshold.

## Solver Investigation

The elasticity solve uses iterative `BiCGStabl` with `ILU1`, tolerance `1e-10`, and a maximum of `6000` iterations.

The earlier refined mesh solve stagnated around residual values of order `1`, rather than dropping toward `1e-10`. A direct UMFPACK trial was also attempted; it reached factorization and failed with `umf4num: -1`, indicating insufficient memory for the direct solve.

The practical conclusion is to keep the iterative path for now and avoid unbounded whole-model refinement. Future solver tuning should follow mesh-topology stabilization.

## Current Results

Reports were regenerated from the latest solver result file, `Elmer/results/dam_results_t0001.vtu`, after the current mesh was generated.

Current values in `Elmer/results/stress_summary.json`:

- Analysed elements: `243,880`
- Maximum principal tension: `10.717 MPa`
- Maximum principal compression: `-17.794 MPa`
- Concrete tensile threshold: `2.000 MPa`
- Tension utilization: `535.9%`
- Maximum tension centroid: `(79.75, -0.51, -25.66) m`

The large peak is at the wedge/plinth/wall convergence elevation. Local refinement can establish convergence of stresses away from that sharp feature, but it may increase the reported point peak if the geometry creates a linear-elastic corner singularity. A mesh refinement alone is not a guaranteed stress-reduction measure.

## Reports Regenerated

The following current client-facing outputs are in `Elmer/results`:

- `client_stress_report.png` and `client_stress_report.svg`
- `principal_stress_distribution.png`
- `critical_stress_locations.png`
- `arch_dam_orientations.png` and `arch_dam_orientations.svg`
- `local_section_wedge.png`
- `cross_section_wedge.png`
- `principal_stress_by_element.csv`
- `stress_summary.json`

## TODOs

### 1. Complete the wedge-to-plinth transition patch

Replace the remaining wedge-adjacent plinth-top hexahedral strip with a local pyramid-and-tetrahedral transition patch.

- Pyramid bases must remain quadrilateral and shared with the wedge base.
- Pyramid side faces can connect to local tetrahedral cells.
- This should remove the remaining `0.991982 m` local hexahedral edge without breaking the wedge/plinth bond.
- Keep all normal wall/plinth bulk blocks hexahedral and near the `0.5 m` target.

### 2. Extend and pass the strict mesh-size audit

Use `Elmer/audit_mesh_size.py` to report edge distributions by region, not just global maxima. Set agreed acceptance limits for:

- normal wall/plinth bulk edge lengths;
- local wedge transition edge lengths;
- minimum volume-element quality;
- number and location of non-hexahedral transition cells.

### 3. Run a stress convergence study

After the transition patch is complete, use at least three local mesh levels around each convergence zone, for example `0.5 m`, `0.25 m`, and `0.1 m`.

Compare averaged stresses and stresses at fixed offsets such as `0.25 m` and `0.5 m` away from the sharp corner. Do not use only the maximum element/node stress as the design result.

### 4. Review physical junction detail

If stress away from the corner remains excessive, refine the physical geometry/model rather than only the mesh:

- add a finite fillet or chamfer;
- replace the abrupt wedge termination with a finite transition radius;
- model the actual construction joint, reinforcement, shear key, or contact condition if the parts are not monolithic concrete;
- reassess plinth and wedge dimensions.

### 5. Tune the iterative Elmer solver only after mesh stabilization

Review the iterative solver/preconditioner and convergence behavior after the transition mesh is finalized. UMFPACK is not viable on the current machine because its numeric factorization exhausted available memory.

### 6. Regenerate deliverables after every mesh/solver change

After any geometry or mesh modification:

1. Run `python3 Elmer/build_curved_dam_geometry.py`.
2. Run `python3 Elmer/audit_curved_dam_mesh.py Elmer/curved_dam_mesh.msh`.
3. Run `python3 Elmer/audit_mesh_size.py Elmer/curved_dam_mesh.msh <accepted-limit>`.
4. Run the Elmer solver.
5. Run `python3 Elmer/analyze_stress.py --vtu Elmer/results/dam_results_t0001.vtu --mesh Elmer/curved_dam_mesh.msh --out-dir Elmer/results`.

