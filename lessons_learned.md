# Lessons Learned — Gairezi Dam Elmer FEA Solver & Mesh Diagnostics

## Summary of Critical Issue

Whenever `config.json` parameter values were updated (e.g., setting `wedge_start_below_crest_m = 25.0` or changing wedge angles), `ElmerSolver` failed to converge and hit the `6000` linear iteration limit.

This document records the exact root causes, technical mechanisms, and rules to prevent recurring solver failures.

---

## 1. Root Cause #1: Non-Uniform Wall Z-Grading (Layer Squeezing)

### Technical Mechanism
- In `Elmer/build_curved_dam_geometry.py`, `ordinary_wall_z()` was re-grading the lower wall vertical levels proportionally to the local wedge height:
  $$h(S) = \text{crest\_z} - \text{base\_z} - \text{wedge\_start\_below\_crest\_m}$$
- **Near Wedge Convergence Tips** (chainages $\approx 88\text{ m}$ and $\approx 124\text{ m}$): The wedge height $h(S) \to 0$ (e.g. $h \approx 0.05 \to 0.20\text{ m}$).
- **Element Compression**: All `wedge_interface_layer` layers (7 or 8 layers) were forced into a near-zero height block. Vertical layer height dropped to $dz \approx 6.25 \to 25\text{ mm}$, while horizontal node spacing remained $dx = 0.50\text{ m}$ ($500\text{ mm}$).
- **Aspect Ratio Distortion**: Produced hexahedral elements with extreme aspect ratios exceeding **40:1 to 100:1**.
- **Cross-Thickness Coupling**: Wall $Z$-coordinates were shared across all thickness columns (from downstream face to upstream face). Squeezing the downstream lower block forced the upstream face to deform as well, ill-conditioning the global stiffness matrix across the entire dam body.

### Permanent Fix
- Replaced `ordinary_wall_z()` with **uniform vertical wall layer spacing**:
  $$z = \text{base\_z} + \text{level\_index} \times \frac{\text{crest\_z} - \text{base\_z}}{\text{vertical\_layers}}$$
- Wall elements now retain near 1:1:1 aspect ratios ($dz \approx 0.50\text{ m}$) from base to crest across all stations.

---

## 2. Root Cause #2: Floating-Point Overshoot in Discretization (`math.ceil`)

### Technical Mechanism
- Wedge radial segments were computed using `wedge_base_segments = math.ceil(wedge_thickness / target_block_size)`.
- Using rounded angle conversions like `26.57°` produced $t = 4.0 \times \tan(26.57^\circ) = 2.000432\text{ m}$.
- Because $2.000432 > 2.0$, `math.ceil(2.000432 / 0.5)` jumped from **4 segments to 5 segments**.
- This $0.4\text{ mm}$ floating-point overshoot altered the entire wedge cross-section topology and added an extra layer of elements, degrading the stiffness matrix.

### Permanent Fix
- Use **exact Pythagoras geometric ratios** instead of rounded trigonometric angle conversions:
  - `wedge_ratio_opposite_m = 2.0`
  - `wedge_ratio_adjacent_m = 4.0`
  - `wedge_tangent = 2.0 / 4.0 = 0.5`
- This guarantees $t = 2.0\text{ m}$ exactly and results in precisely 4 radial segments ($\lceil 2.0 / 0.5 \rceil = 4$).

---

## 3. Root Cause #3: Linear Solver Preconditioning & Tolerance Settings

### Technical Mechanism
- In `dam_model.sif`, the elasticity solver was configured with:
  - `Linear System Preconditioning = ILU1`
  - `Linear System Convergence Tolerance = 1.0e-10`
  - `Linear System Max Iterations = 6000`
- `ILU1` preconditioning is too weak for ~280,000-node 3D elasticity models containing mixed element types (hex/tetra/prism) and re-entrant corner stress concentrations.
- The BiCGStabl residual plateaued around $10^{-7}$ to $10^{-8}$ and hit the 6000 iteration cap without reaching $10^{-10}$.

### Permanent Fix
- Configured solver settings in `dam_model.sif`:
  - `Linear System Preconditioning = ILU2`
  - `Linear System Convergence Tolerance = 1.0e-8`
  - `Linear System Max Iterations = 12000`
- With uniform wall element aspect ratios, `BiCGStabl` now converges smoothly in **1,376 iterations** to $0.65 \times 10^{-10}$.

---

## Rules to Follow Going Forward

1. **Rule 1: Keep Wall Vertical Levels Uniform**:
   Never tie the wall's $Z$-level node heights to local variable-height geometry features like wedge tip convergence points.
2. **Rule 2: Use Exact Geometric Ratios**:
   Define wedge slopes using exact aspect ratios (e.g. 2 m : 4 m = 0.5) rather than rounded degree angles to avoid `math.ceil()` discretization jumps.
3. **Rule 3: Environment Variable for Elmer**:
   Always run ElmerSolver with `ELMER_HOME=/usr/local` so it can find `elements.def`.
4. **Rule 4: Verify Mesh Quality Before Solving**:
   Always run `python3 Elmer/audit_curved_dam_mesh.py` and `python3 Elmer/audit_mesh_size.py` before launching a long FEA solve.
