# Mesh comparison baseline

## Reference run: 100,000-target legacy tetrahedral job

Job: `C:\Users\alist\AppData\Local\Lamb\DamAnalysisWorkbench\jobs\852976359e054d50bd1e629bbf2afbe0`

This is the preserved baseline for later like-for-like comparisons. The
submitted configuration targeted 100,000 wall tetrahedra. It is **not** a
native hexahedral/LEGO job: the wall, foundation, and bedrock sent to the
solvers were all TETRA4 cells. The foundation was still constructed from
source hexahedra before subdivision for the solver.

The job's Elmer baseline and Code_Aster elastic baseline succeeded. The intact
preload and cracking stages were subsequently cancelled, so the values below
are mesh, baseline, screening, and preparation statistics—not accepted
cracking results.

### Submitted and effective mesh sizing

| Setting / result                  |                   Value |
| --------------------------------- | ----------------------: |
| Wall height                       |                    30 m |
| Radial width                      |                     4 m |
| Submitted wall target             | 100,000 TETRA4 elements |
| Configured global element size    |                   0.5 m |
| Configured local element size     |                   0.1 m |
| Effective wall element size       |           0.907254779 m |
| Actual wall elements              |          118,560 TETRA4 |
| Wall target overshoot             |   18,560 cells (18.56%) |
| Effective foundation element size |           0.907254779 m |
| Effective bedrock near size       |           0.907254779 m |
| Effective bedrock far size        |                  15.0 m |
| Total source/solver nodes         |                  46,206 |
| Total volume cells                |          213,556 TETRA4 |

The wall target applies only to the wall body. Foundation and bedrock are
additional to the target.

### Source and elastic-baseline volumes and topology

Values below were recalculated directly from
`cracking\aster_elastic_baseline\three_body_mesh_arrays.npz`. Per-body node
counts overlap at shared interfaces and therefore should not be summed.

| Body            | Solver type |             Cells |       Unique body nodes |               Volume (m³) | Share of total volume |
| --------------- | ----------- | ----------------: | ----------------------: | -------------------------: | --------------------: |
| Wall            | TETRA4      |           118,560 |                  24,852 |              11,654.634026 |              0.36294% |
| Foundation      | TETRA4      |             7,980 |                   3,056 |                 591.395397 |              0.01842% |
| Bedrock         | TETRA4      |            87,016 |                  20,220 |           3,198,935.007073 |             99.61864% |
| **Total** | TETRA4      | **213,556** | **46,206 global** | **3,211,181.036496** |        **100%** |

Concrete totals:

- 126,540 solver cells (wall plus foundation).
- 12,246.029424 m³.
- The 7,980 solver foundation tetrahedra originated from 1,330
  construction hexahedra.

### Body bounds and element scale

| Body        | Minimum XYZ (m)                    | Maximum XYZ (m)                   | Edge length min / median / max (m) | Cell volume min / median / max (m³) |
| ----------- | ---------------------------------- | --------------------------------- | ---------------------------------- | ------------------------------------ |
| Wall        | (4.746908, -35.099749, -30.846663) | (79.999820, 79.843801, 0.0)       | 0.791663 / 1.128418 / 1.566237     | 0.095764 / 0.098788 / 0.120964       |
| Foundation  | (3.809876, -36.185812, -31.610015) | (80.499819, 80.397442, -0.907255) | 0.500000 / 0.934134 / 2.029390     | 0.049634 / 0.082271 / 0.101778       |
| Bedrock     | (-79.873025, -120.767997, -87.0)   | (185.074761, 181.990798, 5.0)     | 0.038098 / 1.741549 / 36.330958    | 0.000589 / 0.492094 / 2240.590076    |
| Full domain | (-79.873025, -120.767997, -87.0)   | (185.074761, 181.990798, 5.0)     | 0.038098 / 1.209591 / 36.330958    | 0.000589 / 0.099796 / 2240.590076    |

### Boundary/interface inventory

| Surface or interface             | TRI3 faces |    Area (m²) |
| -------------------------------- | ---------: | ------------: |
| Wall upstream                    |      7,904 |  2,988.408122 |
| Wall downstream                  |      7,904 |  2,838.987716 |
| Wall crest                       |      1,880 |    612.288865 |
| Wall/foundation interface        |      1,880 |    719.129464 |
| Bedrock bottom restraint surface |        608 | 40,361.786535 |
| Bedrock side restraint surface   |      3,004 | 69,794.245696 |

The wall/foundation interface was represented by 1,952 coupled spring
segments in the saved Aster arrays, with 719.129464 m² total tributary area.

### Elastic and cracking model scope

The accepted Code_Aster elastic baseline used the complete source volume:

- `bedrock_analysis.mode = full_volume`
- `wall_only_body = false`
- Wall: 118,560 TETRA4
- Foundation: 7,980 TETRA4
- Bedrock: 87,016 TETRA4
- Aster baseline run time recorded by `run_manifest.json`: 62.852991 s

The nonlinear cracking preparation was scoped differently:

| Cracking preparation measure                |                                            Value |
| ------------------------------------------- | -----------------------------------------------: |
| Source bedrock cells                        |                                           87,016 |
| Retained nonlinear bedrock cells            |                                                0 |
| Bedrock representation in nonlinear model   | Distributed foundation/bedrock interface springs |
| Interior wall facets available to screening |                                          227,326 |
| Screened overstressed wall cells            |                                           13,988 |
| Screened critical facets                    |                                            8,032 |
| Initially active cohesive faces             |                                           18,845 |
| Prepared candidate faces                    |                                           75,460 |
| Native cohesive joint cells                 |                                          226,380 |
| Split nodes                                 |                                            9,878 |
| Nodes added by splitting                    |                                          138,780 |
| Interpolation nodes                         |                                          396,449 |

The 75,460 candidate faces and 226,380 joint cells are the fixed bank prepared
after applying the one-ring, initially-overstressed-cell continuation scope.
They are not counts of activated or accepted cracks.

### Run status and comparison cautions

- Elmer baseline: succeeded.
- Code_Aster elastic baseline: succeeded.
- Intact preload: cancelled.
- Discrete cohesive cracking: cancelled.
- No accepted nonlinear water-depth result exists for this job.
- The saved reports explicitly mark the model `engineering_validated: false`.
- The 100,000 figure is a requested wall target, not the achieved wall count.
- Bedrock accounts for 99.61864% of the full-domain volume; full-domain
  element/cell totals should therefore always be interpreted together with
  body-specific wall and concrete counts.
