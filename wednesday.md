# End Of Day Notes - 2026-09-16

## Objective

Investigate the persistent stress concentration at the low-side plinth-to-flat-bedrock transition while retaining the monolithic wall, plinth, and trapezoid wedge geometry.

## What Was Confirmed

- The original mesh had 22 downward-facing, untagged exterior facets beneath the side-wedge/plinth transition. They lay in the same low-side region as the stress peak, including two facets at station approximately 100.25 m.
- Those facets were not valid bedrock-contact faces. They exposed a nonconforming wedge-to-plinth interface, so assigning the foundation spring directly to them would have masked an unbonded concrete interface rather than fixed it.
- The parent plinth uses coarse top faces while the trapezoid wedge uses a 0.5 m fine lattice. The original direct connection did not provide a conforming face split between them.

## Implemented Repair

- Updated `Elmer/build_downstream_toe_rib.py` to insert a local coarse-plinth-to-fine-wedge transition layer.
- The transition replaces the bottom 0.5 m of noncollapsed side-wedge strips with tetrahedral fan cells; the remaining 1.5 m of each trapezoid remains structured hexahedra.
- Each parent coarse plinth interface quad is covered by the matching pair of transition-tetrahedral faces. The previously exposed parent boundary record is removed when the transition owns that interface.
- The zero-width outer taper sections are excluded from the transition fan to avoid degenerate tetrahedra.

## Validation

The rebuilt mesh passed:

- `python3 -m py_compile build_downstream_toe_rib.py`
- `python3 build_curved_dam_geometry.py`
- `python3 build_downstream_toe_rib.py`
- `ElmerGrid 14 2 curved_dam_mesh.msh -out mesh`
- `python3 audit_curved_dam_mesh.py`
- `python3 audit_mesh_size.py curved_dam_mesh.msh 1.5`

Transition-build results:

- 4,798 new nodes
- 2,876 new hexahedra
- 4,728 new tetrahedra
- Zero degenerate cells

Focused interface evidence after rebuilding:

- Low-side stations 98.75-100.75 m: 12 parent coarse interface quads are covered by transition-tetrahedral face pairs.
- High-side stations 116.25-119.25 m: 20 parent coarse interface quads are covered by transition-tetrahedral face pairs.

## Fresh Solver Result

ElmerSolver completed successfully on the repaired mesh on 2026-09-16 at 22:48. The VTU output was confirmed newer than the mesh before post-processing.

| Quantity | Before transition | After transition |
| --- | ---: | ---: |
| Maximum tensile principal stress | +9.342 MPa | +8.067448 MPa |
| Maximum compressive principal stress | -9.296 MPa | -12.734387 MPa |
| Tensile-peak station | about 100.25 m | 100.249983 m |
| Compressive-peak station | about 100.25 m | 100.249995 m |

The transition repair reduced peak tensile stress by 1.274552 MPa, but the hotspot stayed at station 100.25 m and peak compression increased in magnitude by 3.438387 MPa.

## Conclusion / Next Investigation

The nonconforming wedge-to-plinth interface was real and is now repaired, but it was not the sole cause of the stress concentration. The remaining governing feature is likely the abrupt change in the actual foundation geometry and reaction path as the plinth underside transitions into the flat z=-29 m founded reach near station 101 m.

Do not alter the foundation spring stiffness simply to lower the reported peak. The next useful model change is a longer, physically continuous plinth/foundation-underside transition into the flat bedrock plane, then a new solve comparing stress at fixed offsets from station 100.25 m.

## Rock-Equivalent Plinth Proof Case

For an additional causality check, the plinth and wall-plinth transition cells were assigned to Body 2, `RockEquivalentPlinth`, while retaining shared nodes with the concrete wall and wedge. Material 2 uses the granite elastic assumption documented for the foundation spring:

- Young's modulus: 35 GPa
- Poisson ratio: 0.25
- Density: 2400 kg/m3, deliberately unchanged from concrete to avoid changing gravity load

This is a material-substitution proof case, not a finite-bedrock-volume model. The actual bedrock remains represented by the Boundary ID 1 distributed springs.

The two-body mesh passed compilation, ElmerGrid conversion, and the topology audit. ElmerSolver loaded two bodies and two materials and completed successfully. Mesh volume-element counts were Body 1: 93,108 and Body 2: 48,804.

| Quantity | Concrete plinth | Rock-equivalent plinth | Change |
| --- | ---: | ---: | ---: |
| Maximum tensile principal stress | +8.067448 MPa | +8.250576 MPa | +0.183128 MPa (+2.27%) |
| Maximum compressive principal stress | -12.734387 MPa | -12.900589 MPa | -0.166202 MPa (1.30% greater magnitude) |
| Peak station | 100.25 m | 100.25 m | unchanged |

Conclusion: assigning the plinth the available rock-equivalent elastic properties does not relieve or relocate the stress concentration. It modestly increases both peak magnitudes, so plinth material mismatch is not the primary cause. The physical foundation-geometry/support transition remains the leading hypothesis.
