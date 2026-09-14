# Wall/Plinth Thickness Experiment

## Setup

Three geometries were solved with the same single-body, uniform-concrete
material (Young's Modulus = 3.5e10 Pa, Poisson Ratio = 0.2, Density = 2400
kg/m^3) and the same load case. Each +1 m increment was projected entirely
upstream: the downstream wall face (76 m radius) and downstream plinth face
(74 m radius) stayed fixed in every case.

| Case | Wall thickness | Plinth width | Results folder |
|---|---|---|---|
| Baseline | 4 m | 7 m | `Elmer/results` |
| Experiment 1 | 5 m | 8 m | `Elmer/experiment_wall5m_plinth8m_upstream` |
| Experiment 2 | 6 m | 9 m | `Elmer/experiment_wall6m_plinth9m_upstream` |

## Results

| Case | Max tension (MPa) | Max compression (MPa) | Tension utilization (vs 2.0 MPa strength) |
|---|---|---|---|
| Baseline (4 m / 7 m) | 10.578 | -14.681 | 5.29x |
| Experiment 1 (5 m / 8 m) | 8.849 | -11.978 | 4.42x |
| Experiment 2 (6 m / 9 m) | 7.550 | -10.130 | 3.77x |

Change relative to baseline:

- 5 m / 8 m: tension -16.3%, compression magnitude -18.4%
- 6 m / 9 m: tension -28.6%, compression magnitude -31.0%

In all three cases the peak tension and peak compression occur at the same
physical location (the wedge/wall/plinth junction, centroid approximately
x=69-76 m, z=-25 to -28 m) — the known corner singularity, not a different
failure mode.

## Finding

The data does not support "little to no bearing." Each 1 m added to the
wall/plinth thickness produced a consistent, non-trivial double-digit
percentage reduction in both peak tension and peak compression (roughly
16-19% at +1 m, 29-31% at +2 m relative to baseline). Thickness has a real
and measurable effect on the reported peak stress magnitude at this
corner.

What thickness has *not* done, over the range tested, is resolve the
underlying issue: even at 6 m wall / 9 m plinth, the peak tensile stress
(7.55 MPa) is still 3.77x the concrete's 2.0 MPa tensile strength, and the
critical location and character (the wedge/wall/plinth corner singularity)
are unchanged across all three geometries. Thickening the section reduces
the demand but has not, within this range, brought the corner stress under
the tensile strength.
