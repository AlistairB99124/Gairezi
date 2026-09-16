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
