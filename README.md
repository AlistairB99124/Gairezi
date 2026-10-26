# Gairedzi Dam Structural Model

This repository is the clean v2 working copy for the Gairedzi dam study.

The purpose of this repo is to separate the newer surveyed-foundation geometry work from the earlier baseline model and keep the design assumptions explicit.

There is no plinth. The Elmer solver optionally places the wall on a concrete-filled
trench: wall (1), bedrock (2), foundation (3). The checked-in configuration enables
this foundation; omitting `geometry.foundation` retains the historical two-body
wall-on-bedrock model exactly. Historical `three_body` filenames remain compatible.
Foundation-aware API/WPF and FEniCSx integration is not included in this Elmer change.

## Integrated wall/foundation/bedrock solver

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

The graded rock mesher represents the flat domain bottom as one planar surface,
without a centre-point triangle fan. This avoids artificial bottom seams that
can produce zero-volume tetrahedra and a native Netgen optimizer access violation
with some thickness-step configurations (observed with Gmsh 4.15.2 on Windows).
Bottom mesh connectivity may change; physical geometry, configured mesh sizes,
locked interface triangles, and the volume/conformity/quality checks are retained.
Low scaled-Jacobian warnings still require mesh-quality inspection; completing.

Open the printed `model\results\three_body_results_t0001.pvtu` in ParaView.
Keep its adjacent VTU pieces. Colour by cell `GeometryIds`: **1 = wall,
2 = bedrock, 3 = foundation (when configured)**, or select displacement and stress fields. These are
solved fields, unlike the geometry-only preview below.

### Spring-interface projector

Generated spring boundary conditions explicitly select both
`Galerkin Projector = Logical True` and `Normal Projector = Logical True`.
This uses Elmer's normal-based Galerkin integration instead of the legacy
weighted-projector fallback that warns `Use "Normal Projector" or "Level
Projector" instead!`. "Normal" describes the interface projection search;
it does not change the directional spring coefficients to normal-only
restraints or rotate the global X/Y/Z displacement components.

The connected near-planar interface groups, independent group elimination,
spring coefficients and bonded interfaces remain unchanged. Existing saved
SIFs/results are not rewritten; start a fresh run to use the new projector,
and repeat full-model projector/parity checks before relying on new results.
The tilted two-block benchmark on Elmer 26.2 reproduces the warning with the
old projector and removes it with the normal projector. Its analytical gap
error is below 1.3e-7 relative, with unchanged serial displacement fields to
floating-point precision. MPI verification uses the canonical connected
mortar partition options.

### Optional concrete foundation trench

```json
"foundation": {"depth_m": 0.5, "side_margin_m": 0.5, "support": {"model": "bonded"}}
```

`geometry.foundation.support` controls the **wall/foundation** connection,
independently of the wall's internal thickness-block joints:

```json
"support": {"model": "bonded"}
```

Bonded means **zero relative displacement**, `u_wall = u_foundation`, implemented
with shared interface nodes. Neither body is fixed to ground at this interface;
both can deform together and transfer tension, compression and shear. The bodies
retain separate IDs (wall 1, foundation 3). At a wall-joint/seat intersection,
both wall-block node copies follow the same foundation node; internal wall-joint
springs remain active away from that seat line. This is a perfect bond, not
opening/sliding contact.

Alternatively use `"support": {"model": "spring", "spring_n_per_m3":
{"x": 1.25e9, "y": 1.25e9, "z": 3e9}}` for finite directional compliance in
global axes (N/m3). `support` is required whenever a foundation is configured;
missing support is a validation error, never an inheritance from
`geometry.wall.joint_support`. Older foundation configurations must explicitly
copy their intended spring coefficients into `geometry.foundation.support`
or select bonded support.
With a foundation present, `geometry.bedrock.support.model` independently
selects the **foundation/bedrock** connection. `"bonded"` shares interface
nodes and enforces zero relative displacement without grounding either body;
`"distributed_elastic_granite"` retains the configured directional mortar
springs. The checked-in configuration uses wall/foundation springs
(X/Y = 1.25e9, Z = 3e9 N/m3) and bonds foundation/bedrock.
Its `geometry.bedrock.support.spring_n_per_m3` values are
inactive for that bonded interface. Rock outer-boundary restraints are separate
and remain controlled by `material.bedrock.support`. Without
`geometry.foundation`, the same bedrock setting applies directly to the
wall/bedrock seat.
The wall's internal joints remain controlled by `geometry.wall.joint_support`.
`"model": "displacement"` is not supported here: an absolute zero-displacement
condition would ground an internal interface instead of bonding two deformable
bodies. The historical spring-case results below are not results for the
checked-in bonded-interface configuration.

Place this optional block under `geometry`. Dimensions must be finite, positive
and at most 5 m. The trench follows the complete valley-floor and abutment seat,
with **constant width** derived from the largest `geometry.wall.radial_width_m`
or `thickness_steps[*].radial_width_m`, plus twice `side_margin_m`.
The current width is 5.0 + 2*0.5 = **6.0 m**, radially **74.5 to 80.5 m**
along the whole seat, with no thickness-band corners in the trench outline.
With the foundation block present, the wall seat is an **excavation-only
staircase**, not linear interpolation between stations: each chainage column
uses the **lower of its two adjacent excavated station courses**. Treads are
horizontal on the existing 0.5 m courses (including thickness-step courses);
vertical risers lie on the radial station planes. The wall construction mesh
contains only hexahedra. Horizontal courses, stepped downstream thickness and
spring-separated blocks at -10/-20 m are retained. Removing the foundation
block restores the original linear seat and mixed hex/prism wall exactly.

The foundation is built from structured hexahedra on the same station/radial/
course lattice, with nominal 0.5 m radial cells and the exact wall radial nodes.
Its floor is at least `depth_m` below each tread, rounded down to whole courses.
The excavation is also dilated along chainage by at least `depth_m` at the
innermost radius: neighboring lower treads extend beneath higher columns,
providing a full station/course of backing behind wall risers, not a diagonal
normal offset. This intentionally makes the lining locally thicker than 0.5 m.
One extra station extends beyond each wall end by at least `depth_m` (and one
element size), at crest level, eliminating the former zero-area crest wall/rock
coincidences. Outside the constant-width trench the surveyed terrain is kept;
exposed vertical rock excavation walls connect it to the foundation treads.
The exposed 0.5--1.5 m radial shoulders follow the same staircase.

**Rock-only lip excavation shaping (benched iteration):** rather than introduce
a new fixed-width terrace and move the same fractional-height wedges to its
outer edge, snap the rock terrain nodes on each trench lip **down to the higher
of the two incident tread courses**. This is never above the surveyed terrain,
retains the whole foundation/rock contact surface, and replaces fractional-course
lip wedges by course-sized triangles. Exterior terrain triangles interpolate
from this lowered lip back to the original surveyed terrain anchors. There is
no new configurable band/parameter; the interpolation footprint is defined by
the existing exterior triangulation, **not** guaranteed to be only 0.5 m wide.
Wall, foundation, loads, materials, far-field sizes and spring coefficients are
unchanged. Exact cell-coordinate audits compare all 633450 wall and 40896
foundation tets with the prior staircase mesh. This is an explicit excavation-
shaping assumption, not a change to the survey data in config.json. The maximum
lip lowering is **0.483871 m**. The full Elmer suite after this change passed
**119 tests and 61 subtests** in 426.99 s, under the enforced 540 s cap.

To preserve the existing all-tetrahedral solver/mortar path, construction hexes
are split deterministically without adding nodes. Thus the source/solver Gmsh
mesh still contains tetrahedra, not native hex elements. This avoids expensive
Gmsh foundation volume meshing and retains one-to-one, identical triangles on
both mortar sides. It does **not** coarsen the configured rock far field or
change partition count, materials, loads, or spring coefficients. The foundation
uses `material.concrete`, but is separate from wall stress reporting/damage.

The wall/foundation mortar springs use `geometry.wall.joint_support` (model must
be `spring`); foundation/rock uses `geometry.bedrock.support`. The foundation
requires the configured graded rock mesh. Bonded foundation/rock is explicitly
rejected; removing the block preserves the legacy bonded path. All three bodies
have separate displacement node IDs. Exact wall coordinates and matching quad
diagonals are preserved before tetrahedral splitting; all cut floor/side
triangles also match the rock one-to-one.
The geometry-only legacy preview still shows the two-body seat; use the solved
three-body source mesh/PVTU to inspect the trench. The FEniCSx adapter accepts the
same saved foundation mesh and interface contract, with concrete material on body
3 but damage restricted to wall body 1. The local API checks foundation metadata
and spring-pair consistency before accepting a baseline/fracture report. The WPF
configuration editor preserves optional foundation dimensions and exposes its
per-body stress peaks separately from the wall-only headline extrema. An elastic
comparison must still pass the unchanged numerical parity/equilibrium gates
before the API allows fracture; successful convergence alone is insufficient.

**Diagnostic/report contract:** body IDs, boundary IDs, spring-pair names and
stress fields are unchanged from the smooth 6 m foundation. The `foundation`
object retains `present`, `depth_m`, `side_margin_m`, `material`, `region_id`,
`volume_m3`, `width_m`, `radial_limits_m`, `width_definition`,
`outline_radial_step_edges` (zero: no radial width changes), `element_size_m`,
`crest_end_definition`, and `wall_seat_node_snap` (`wall_nodes`, `matched_nodes`,
`unmatched_nodes`, `maximum_coincidence_distance_m`, `tolerance_m`, `matched_faces`).
It adds `seat_definition`, `construction_cell_type` (`hexahedron`),
`construction_hexahedra`, `solver_cell_type` (`tetrahedron`),
`minimum_vertical_thickness_m`, `minimum_riser_backing_m`, and
`open_excavation_volume_m3` (reference terrain cap to staircase top, excluded
from rock; the snapped-lip variant uses its lowered cap rather than the original
surveyed cap).
The rock-lip iteration additionally exposes `rock_bench_definition` and
`rock_lip_maximum_lowering_m`; that iteration did not change IDs/pair fields.
The subsequent exact-plane mortar split changes foundation pair entries as
specified below; historical runs retain their original contracts.
`depth_definition` now describes course/station backing, not smooth-seat normals.
The obsolete `maximum_normal_depth_error_m`,
`maximum_profile_normal_constraint_error_m`, and `faceted_normal_depth_range_m`
are removed; consumers must not interpret the staircase as a normal-offset
smooth surface. `wall_underside.profile` (diagnostic only) becomes
`excavated_course_staircase`, with `elevation_quantization: true`. Without a
foundation these fields retain their legacy values. Definition/assumption text
now states the stepped seat and protected crest ends.
`solver_body_ids` adds `foundation: 3`; `cells_by_body` (diagnostic) / `regions`
(report) adds key `3`. `stress_bodies.foundation` has the same recovered nodal
peak/coordinate/cell-count fields as wall. `wall_bedrock_connection` becomes
`not_in_contact`; `wall_foundation_connection` and `foundation_bedrock_connection`
are `spring`. `wall_connection: spring_jointed_blocks` and `wall_joints` stay
unchanged. Historical unsplit `spring_pairs` used semantic/source tags:

| Pair | Slave/master tags | Coefficients |
|---|---|---|
| `wall_foundation` | 17 / 18 | wall joint support |
| `foundation_bedrock` | 11 / 12 | bedrock seat support |
| `wall_joint_1` | 21 / 22 | wall joint support |
| `wall_joint_2` | 23 / 24 | wall joint support |

`boundary_id_map` maps these source tags to actual mesh/SIF tags. For the historical unsplit run,
11/12 -> 5/6, 17/18 -> 11/12, 21/22 -> 13/14, 23/24 -> 15/16.
Both reports now expose `spring_pairs`, `solver_body_ids`, `boundary_id_map`,
and `joint_gaps` (per-body maximum displacement and per-pair gap summary).
Gap samples include both wall-block copies at wall/seat joint intersections.
There is no `wall_bedrock` spring pair with a foundation. Spring coefficients
remain global Cartesian N/m3, not normal/tangent or compression-only contact.

Prior trials are preserved in `Thickness_tests\foundation_trench` (local-width),
`Thickness_tests\foundation_trench_6m` (smooth constant-width), and
`Thickness_tests\stepped_blocks_springs` (no foundation). Their near-wall rock
quality (centroid to nearest wall vertex <2 m) is respectively 1.643 degrees /
27 tets below 5 degrees, 1.279 / 29, and 1.684 / 26. The smooth 6 m run took
398.14 s Elmer time / 492.85 s pipeline time; its worst rock tet was at the
right-bank crest termination (62.350797,-43.047677,-0.162632) m.
The staircase trial is isolated in `Thickness_tests\foundation_stepped`.
It completed on four ranks in **306.97 s Elmer time / 385.75 s total pipeline**
(420 s solver timeout inside the 540 s process cap), leaving 154 s pipeline
headroom. Its separate `report.json`, `mesh_quality.json`, and
`interface_audit.json` preserve numerical results and reproducible source audits.

| Body | Peak tension MPa @ reference XYZ m | Peak compression magnitude MPa @ reference XYZ m | Max displacement m |
|---|---|---|---:|
| Wall | 2.998241 @ (74.742848,28.522038,-29.000000) | 4.872866 @ (74.699505,6.707003,-29.000000) | 0.009676128 |
| Foundation | 3.598852 @ (74.275705,28.343775,-28.500000) | 5.177789 @ (70.538563,26.917673,-28.500000) | 0.001674118 |
| Near rock (5 m reporting region) | 2.653205 @ (61.639040,-43.164699,0.223714) | 4.957956 @ (61.564967,-43.126951,-0.300205) | 0.001913291 |

Maximum vector spring gaps: wall/foundation **0.001111429 m**,
foundation/rock **0.000548652 m**, wall joint 1 **0.000369143 m**, and wall joint 2
**0.000588691 m**. These are linear spring relative displacements, not unilateral
contact openings. Independent source-mesh auditing confirms **4642/4642 wall
seat nodes matched**, zero unmatched, zero maximum coincidence distance, and
zero wall/rock coordinate coincidences. Wall/foundation has **8282 matching
triangles per side**, area **1012.700568 m2** each; foundation/rock has **13216**,
area **1612.038554 m2** each. Spring node IDs remain duplicated, not shared.
The repeatable quality tool is `Elmer\mesh_quality.py` (meshio/NumPy/SciPy,
available in WSL), with a mesh path and required `--output` JSON path.
All runs/tests have an enforced 540 s wall-clock cap. The staircase Elmer suite
passed **118 tests and 61 subtests** in 428.22 s. The construction foundation has
6816 hexes, converted to 40896 tets (versus 58022 smooth-foundation tets), without
Gmsh foundation remeshing. The wall has 633450 tets and rock 263344 tets.

| Seat/trench | Wall minimum angle | Foundation minimum angle | Rock minimum near wall | Rock count <5 degrees near wall |
|---|---:|---:|---:|---:|
| No foundation | 34.328 | -- | 1.684 | 26 |
| Local-width smooth seat | 34.328 | 12.800 | 1.643 | 27 |
| 6 m smooth seat | 34.328 | 12.068 | 1.279 | 29 |
| 6 m staircase seat | 34.245 | 33.242 | 0.842 | 197 |
| 6 m staircase, snapped rock lip | 34.245 | 33.242 | 0.945 | 53 |
| Same mesh, exact-plane mortar split | 34.245 | 33.242 | 0.945 | 53 |

Angles are internal tetrahedral dihedrals in degrees. All wall/foundation rows
have zero tets below 5 degrees. The staircase foundation is substantially better
shaped, and no wall/rock geometric coincidences remain. **The rock quality goal
is not achieved:** exposed excavation/unchanged surveyed-terrain transitions
create thin rock features (smallest near-wall edge 0.008065 m), despite the
right-angle concrete interlock. The worst near-wall rock tet is at
(61.330321,-42.949652,0.219138) m, 1.254 m from the nearest wall vertex. The prior
worst locations were (50.343099,55.562645,-20.107540),
(50.192535,54.932116,-19.916577), and
(62.350797,-43.047677,-0.162632) m, respectively. Far-field slivers are still out
of scope. The rock-only snapped-lip variant cuts the sub-5-degree count by
**73.1%**, but does not meet a 5-degree minimum: its worst near-wall rock tet is
(61.176443,-42.841979,-0.185502) m. The smallest near-wall edge improves from
0.008065 to **0.014044 m**. Remaining crest/terrain transition issues still need
investigation. The design deliberately does not broaden or coarsen the terrain
or change far-field sizes; only the documented lip lowering/interpolation is
introduced.

#### Benched/snapped-lip solve

The fresh `Thickness_tests\foundation_stepped_benched` run completed on four
ranks in **234.53 s Elmer time / 317.94 s total pipeline**, within a 400 s solver
timeout and enforced 540 s pipeline cap. `report.json`, `mesh_quality.json`,
`interface_audit.json`, and `unchanged_concrete_audit.json` preserve evidence.
The final mesh is byte-identical to the geometry/quality probe used before the
solve. Wall and foundation tetrahedra match the unbenched staircase geometry
exactly. No loads, materials, spring coefficients, partitions, configuration
values or far-field sizes changed.

| Body | Peak tension MPa @ XYZ m | Peak compression magnitude MPa @ XYZ m | Max displacement m |
|---|---|---|---:|
| Wall | 2.998774 @ (74.742848,28.522038,-29) | 4.868157 @ (74.699505,6.707003,-29) | 0.009674339 |
| Foundation | 3.595406 @ (74.275705,28.343775,-28.5) | 5.178524 @ (70.538563,26.917673,-28.5) | 0.001670422 |
| Near rock (5 m reporting region) | 2.280506 @ (79.916123,3.662426,-28.5) | 4.477040 @ (61.222120,-42.877650,-0.242007) | 0.001911936 |

Maximum vector gaps: wall/foundation **0.001111929 m**, foundation/rock
**0.000548667 m**, wall joint 1 **0.000369145 m**, wall joint 2
**0.000589097 m**. The source audit retains **4642/4642 matched wall seat nodes**,
zero unmatched, zero maximum coincidence distance, and identical spring faces:
8282 triangles/1012.700568 m2 per wall/foundation side, and
13216 triangles/1612.038554 m2 per foundation/rock side. Relative spring
displacements are not unilateral contact openings.

#### Exact-plane staircase mortar split

Elmer's Galerkin master-face search can project a tread Gaussian point onto a
riser face across the nonplanar staircase. The unsplit exported projectors differ
from the exact coincident triangle `[M,-M]` operator by relative Frobenius errors
**0.0174046354 wall/foundation** and **0.0146434544 foundation/rock** (largest entry
errors 0.009230264 and 0.009288134 m2 respectively). These are coupling errors,
not material changes. Both interfaces therefore now have **one mortar sub-pair
per exact plane** (normal AND offset): all horizontal treads at one elevation
may share a group; distinct riser station planes and the faceted radial side
planes cannot. Merely collecting every vertical face together is insufficient.
Wall joints at -10/-20 m are already planar and unchanged. Geometry, stiffness
coefficients, loads, materials, node numbering and mesh quality are unchanged by
retagging. Galerkin elimination is now independent for each plane: assemble and
sum `Bp.T @ diag(k/wp) @ Bp`, where `wp` is that plane's slave row-sum. Do **not**
pool all plane projectors before elimination: shared edge nodes have separate
plane dual fields. This is part of the adapter contract and can change local
responses even after fixing off-plane search.

**New precise contract** in diagnostic.json and report.json:

```json
"spring_pairs": {
  "wall_foundation": {
    "grouping": "exact_plane",
    "legacy_tags": [17, 18],
    "faces": {"vertical_0000": [17, 18], "tread_NNNN": [1000, 1001]},
    "planes": {"vertical_0000": {
      "unit_normal": [0, 1, 0], "offset_m": 0,
      "triangles_per_side": 16
    }}
  },
  "foundation_bedrock": {"grouping": "exact_plane", "legacy_tags": [11, 12],
    "faces": {}, "planes": {}},
  "wall_joint_1": [21, 22], "wall_joint_2": [23, 24]
}
```

The example is schematic, not a fixed geometry/tag inventory. Consume the saved
`faces` dictionaries: each value is a semantic slave/master tag pair, and `planes`
has the same keys. `tread_` denotes a horizontal plane; `vertical_` includes
risers, radial side facets and end caps. Four-decimal-digit suffixes are sorted
plane indices, not wall station numbers. The first plane retains 11/12 or 17/18;
**those tags no longer denote the complete interface**. All additional paired
tags start at 1000, allocated first for foundation/rock then wall/foundation,
well above wall-joint IDs 21..60; allocation depends on actual plane geometry.
`boundary_id_map` remains authoritative for actual dense mesh/SIF boundary IDs.
`interface_stiffness_n_per_m3` and `interface_stiffness_sources` remain keyed by
logical pair name, with unchanged values/sources. `Elmer.mortar_interfaces.
expand_spring_pairs` flattens the contract to `logical/part -> (slave,master)`.
Both sides of all sub-pairs are still kept on one MPI rank for Elmer mortar.
For large inventories, a **partition-only copy** of the Gmsh mesh coalesces all
mortar faces to one temporary boundary tag. After ElmerGrid partitioning, all
original boundary labels (including non-mortar labels) are restored by exact
source-node face keys, **before** ownership checks or solver launch. Unknown or
altered face keys fail explicitly. The authoritative `three_body.msh` and SIF
retain the complete plane inventory throughout. This avoids ElmerGrid's fixed
explicit-connect list limit without changing its interface-node constraints,
physical surfaces, four-rank count or final boundary tags. Diagnostic-only
`partition_boundary_restoration` records restored record/file counts and method.
The installed ElmerGrid also failed a tested `-connectall` sentinel path; that
workaround is **not used**. Failed conversion evidence is preserved under
`conversion_failed_output` and `sentinel_failed_output`; neither launched a
solver. A bounded full-mesh partition probe verified all 1025 spring ownerships
with the coalesced-copy/restoration method before the successful solve attempt.

`joint_gaps.joints.wall_foundation` and `.foundation_bedrock` retain their
original aggregate summaries (unique coincident node/displacement copies).
Each adds `faces: {part: <same gap-summary fields>}`; a node on two planes occurs
in both detail summaries, so sub-pair sample counts must not be summed to infer
the aggregate. Wall-joint summaries are unchanged. Body IDs remain wall=1,
rock=2, foundation=3; config.json and all materials/loads remain unchanged.

The split implementation passes **122 Elmer tests and 1082 subtests** in
322.95 s under the enforced 540 s cap. The configured geometry creates **174
wall/foundation planes and 849 foundation/rock planes** (1023 foundation
sub-pairs, plus the two unchanged wall joints). Focused topology checks verify
identical opposing triangles/areas and that every sub-pair fits one actual
plane to better than 1e-7 m. This count depends on geometry, not a fixed schema.

All new sub-pair BCs export local-numbered `p<slaveBC>_part<rank>.dat` projectors.
`Elmer\mortar_projector_audit.py` compares them to exact consistent triangle mass
and checks all exported stencil nodes lie on the group's actual plane. Run with
meshio/NumPy/SciPy (WSL), `--model`, optional `--projectors`, and `--output`.
The fresh `Thickness_tests\foundation_stepped_split\projector_audit.json`
passes all **1023** foundation sub-pairs: maximum relative Frobenius error
**1.9541e-14**, maximum exported stencil-node distance to its actual plane
**1.0566e-9 m**, and **zero nodes** above 1e-7 m. Exact `[M,-M]` agreement also
checks the integrated coupling, not just nodal coplanarity. All **192372 source
nodes and 931083 volume tets** are unchanged from the benched mesh; only boundary
labels/operators change. Local indices map through partition node-file **row
order**, not source/global DOF numbering. Plane metadata uses normals rounded to seven decimals; validation
fits actual face coordinates to avoid interpreting that rounding as a gap.

The fresh split-interface four-rank run completed in **303.59 s Elmer time /
385.45 s total pipeline**, with 400 s solver timeout inside the enforced 540 s
pipeline cap. All displacement/stress fields are finite. The two earlier
conversion failures launched no solver; their diagnostics are retained. No
solver/test run exceeded its cap. The split source has unchanged wall/foundation
node snapping: **4642/4642 matched**, zero unmatched, zero coincidence distance.
The same mesh-quality limitation remains: rock minimum 0.945 degrees / 53 tets
below 5 degrees within 2 m of wall, worst at (61.176443,-42.841979,-0.185502) m;
wall/foundation minima 34.245/33.242 degrees, neither has any tets below 5 degrees.

| Split-run body | Peak tension MPa @ XYZ m | Peak compression magnitude MPa @ XYZ m | Max displacement m |
|---|---|---|---:|
| Wall | 3.031125 @ (74.742848,28.522038,-29) | 4.911093 @ (74.699505,6.707003,-29) | 0.009673885 |
| Foundation | 3.591785 @ (74.275705,28.343775,-28.5) | 5.205300 @ (70.538563,26.917673,-28.5) | 0.001670412 |
| Near rock (5 m reporting region) | 2.285141 @ (79.916123,3.662426,-28.5) | 4.455439 @ (61.222120,-42.877650,-0.242007) | 0.001911935 |

Maximum logical vector spring gaps: wall/foundation **0.001111026 m**,
foundation/rock **0.000548423 m**, wall joint 1 **0.000369104 m**, wall joint 2
**0.000587948 m**. Per-plane gap details are saved in report.json and diagnostic.json.
The paired source areas remain 1012.700568 m2 wall/foundation (8282 triangles per
side) and 1612.038554 m2 foundation/rock (13216 per side). All plane-side source
coordinates and areas match; spring node IDs remain duplicated. Downstream
FEniCSx/API consumers must adapt to the nested contract and independent plane
elimination; this Elmer-only change does not claim a new downstream parity run.

#### Localisation and staircase wall corner sensitivity

Read-only diagnostics for the unbenched staircase run are saved as
`Thickness_tests\foundation_stepped\sliver_localisation.json` and
`wall_corner_diagnostic.json`. Their diagnostic-only script is in the Copilot
session files (`bench_diagnostics.py`), not application source. Of 197 near-wall
rock tets below 5 degrees, **80** are at the upstream lip, **54** downstream,
**63** at crest extensions, **0** on floor/risers and **0** elsewhere. These are
exclusive proximity classes: first within 1.5 m chainage of a wall crest end;
then within 0.75 m radial distance of a lip; then within 0.75 m of a contact-face
centroid; otherwise other. Thus the classification localises features rather
than claiming exact surface incidence of every tetrahedron.

Chainage histogram (10 m bins; unlisted bins are zero):
0--10:17, 10--20:2, 20--30:17, 40--50:14, 50--60:14, 60--70:8,
70--80:14, 80--90:9, 90--100:6, 100--110:2, 130--140:16,
140--150:1, 150--160:7, 160--170:9, 170--180:61.

The staircase wall tension peak is at chainage **96.5 m**, upstream radius
**80 m**, z=-29 m; compression is at chainage **118.5 m**, downstream radius
**75 m**, z=-29 m. Both lie on the **convex 90-degree foot of a riser**, at its
intersection with a radial exterior face; **neither is the re-entrant
270-degree riser head** (which is at z=-28.5 m). Their distances from the radial
riser-foot edges are below 4e-11 m in recovered source coordinates.

Excluding any wall tet with at least one vertex within 0.5 m of a tread/riser
intersection edge removes **25578 / 633450 cells** (4.04%). Using the existing
recovered nodal tensors without re-solving or re-recovering stress, the retained
peak tension is **1.277270 MPa** at (77.346125,20.434701,-29) m; compression is
**2.922728 MPa** at (78.057869,17.520535,-10) m. This supports a local corner/
restraint concentration explanation for the change from the smooth-seat
1.62 MPa tension to 3.00 MPa, and compression to 4.87 MPa. Added vertical spring
faces and the thicker interlocking backing change local restraint and load
transfer, even though materials/loads/spring coefficients are unchanged.
It does **not** prove an elastic singularity, establish mesh convergence, or
justify excluding corners from an engineering assessment.

### Stress reporting region: concrete only

Three-body runs also generate `model\reporting\structure.vtu`
and `model\reporting\stress_summary.json`. **Use this VTU for stress review**:
it contains only wall and optional concrete foundation cells. Wall and foundation peaks
are separate; headline stress/utilization remains wall-only. All solved fields
are retained. Rescale ParaView's colour map to this filtered dataset; do not
reuse the full-domain colour range.

The full-domain PVTU is preserved and remains `result` in the run manifest.
`reporting_result` and `stress_summary` identify the filtered deliverables.
This is post-processing only: it does not change the analysis domain,
restraints or solver. Engineering reports, stress summaries, comparison peaks
and default ParaView exports are concrete-only.

The VTU also carries `DistanceToSharpEdge_m`: the undeformed distance to the
nearest sharp wall/foundation surface edge (faces meeting at 60 degrees or
more: thickness-step corners, joint and interface perimeters, rims; 45-degree
seat chamfers are not sharp). Each body in `stress_summary.json` keeps its raw
peaks and adds `away_from_sharp_edges` peaks, ignoring points closer than
`--edge-exclusion-m` (default 0.5 m, one wall element). Raw peaks on such edges
are mesh-dependent in linear elasticity. The headline/API stresses still use
the raw peaks; the edge-excluded view is extra, not a design stress.

On run `20261023-171602` (1.25e9 springs), the wall peak tension stays near
the -25 m joint/seat crossings (chainage 79--81 and 131--133): 1.95, 1.57 and
1.37 MPa at 0.5, 1.0 and 1.5 m exclusion. This slow decay points to a real
local concentration where the 7 m band starts in plan, not only an edge
singularity. Reassigning the joint-level seat wedges to the band above (a
square stop end) was tried and rejected: it raised the peaks to 2.29 MPa
tension and 6.28 MPa compression. `geometry.wall.step_end_taper_ratio` (see
Stepped wall thickness) removes this concentration instead.

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

Without `geometry.foundation`, `geometry.bedrock.support` controls the
wall/bedrock seat shown above. With a foundation, it controls the
foundation/bedrock interface instead; the wall/foundation interface is
configured independently by `geometry.foundation.support`. Either interface
can be bonded with shared nodes or spring-connected as configured. Spring
coefficients are inactive for a bonded interface.

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

### Stepped wall thickness

The wall radii describe the crest section. Optional
`geometry.wall.thickness_steps` thicken the wall **downstream** below given
elevations; the upstream face stays at `upstream_radius_m`:

```json
"thickness_steps": [
  { "top_z_m": -10.0, "radial_width_m": 4.5 },
  { "top_z_m": -20.0, "radial_width_m": 5.0 }
]
```

With the 80 m upstream radius this gives downstream faces at 76 m (crest to
z = -10), 75.5 m (z = -10 to -20) and 75 m (below z = -20). Steps must go
down and must not thin the wall. Each step elevation is a mesh course, so the
horizontal treads are flat; they are tagged as downstream faces and receive the
tailwater load where submerged. Omit the list for a constant-thickness wall.
The `centerline_radius_m` (78 m) remains the chainage reference radius.

Optional `"step_end_taper_ratio"` (positive, needs `thickness_steps` and
`geometry.foundation`) tapers each step where it ends against the abutment.
In each seat column, a step adds at most `ratio x (step top_z_m - seat z)` of
extra width, so the thicker band grows in plan with seat depth instead of
starting at full width on a one-course shelf. Ratio 1.0 gives about a 45-degree
plan taper, built in whole 0.5 m columns. The foundation roof left uncovered
becomes an exposed margin. Omitting the key keeps the full step widths.

On the 1.25e9 spring configuration of run `20261023-191733`, ratio 1.0 moved
the wall peak tension away from the step ends:

| Wall peak | Without taper (MPa) | Ratio 1.0 (MPa) |
| --- | --- | --- |
| Tension | 1.95 | 0.90 |
| Compression | 3.46 | 2.69 |

The new tension peak is at the valley bottom (chainage about 106, z -29).
Foundation and bedrock peaks were unchanged. This is a single linear-elastic
comparison, not a mesh-convergence study or design verification.

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
| `geometry.wall` | Wall face radii, radial width and centreline (crest section); optional downstream `thickness_steps`, `joint_support`, `step_end_taper_ratio` |
| `geometry.bedrock` | Signed bedrock contour elevations (the wall seat) and wall/rock seat support |
| `material.concrete` | Density, Young's modulus, Poisson ratio and tensile-strength reporting threshold |
| `material.bedrock` | Rock material data; no rock volume is currently meshed |
| `loads` | Gravity, water density, reference water depth, overflow and tailwater heads |
| `outputs` | Requested result metadata; the existing solver/post-processor still writes its standard result fields |
| `crack_simulations` | Optional; declared phase-field cracks consumed by [FEniCSx/crack_field.py](FEniCSx/crack_field.py) |
| `fracture_debug` | Optional; per-job overrides of the fracture defaults (see the cracking section below) |

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
controls. `model` may be `AT2` or `AT1`, and only `energy_split:
volumetric_deviatoric` is currently supported; unsupported alternatives fail
validation. AT1 (`w = alpha`) is linear elastic until its homogeneous strength
`sqrt(3 E Gc / (8 l))`; AT2 (`w = alpha^2`) starts damaging at any tensile energy.

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

The initial field saturates at 1 inside the crack box and uses the optimal
profile of the selected model outside it: `exp(-d / l)` for AT2 and
`(1 - d / (2 l))^2` (support `2 l`) for AT1. Overlapping cracks combine by maximum damage.
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
seeds a horizontal crack at that elevation running along the wall to within
the configured `cracking[0].buffer_m` (default 1 m) of the foundation contour at each end, or for a
centred `cracking[0].length_m` when `full_width` is false (opened `depth_m`, default 3 m, from the
configured `face`, default upstream, with a Z thickness of `opening_m`, default 2 mm; a config
without `cracking` uses these defaults; clicks within the buffer of the foundation are
snapped to the nearest valid point and the snap is logged and shown) so that the wall (but not bedrock) can damage, with an AT1 model whose length scale is calibrated to the
configured concrete tensile strength (`l = 3 E Gc / (8 ft^2)`, Tanné et al.
2018; 0.656 m for E = 35 GPa, Gc = 200 J/m2, ft = 2 MPa) and a 4,000,000-cell
refinement budget; see [FEniCSx/README.md](FEniCSx/README.md). Refinement
bisects only the wall edges longer than `l / target` near the crack, so a
101 m crack at z = -15.7 m reaches 2.15 elements per length scale with
3.25 million cells (whole-cell refinement predicted over 7 million). After
FEniCSx cracking ends, app jobs run Code_Aster cracking on the whole three-body
model with the same
wall-length seed (`Aster.whole_model_crack`, Linux Code_Aster in WSL;
only the wall can damage, foundation and bedrock respond elastically); see
[Aster_Porting.md](Aster_Porting.md). Both solvers load the crack with the
full hydrostatic head of the seeded face at the crack elevation (reservoir or
tailwater, the Elmer datums, no drainage), stored in the shared prepared seed;
seeds prepared earlier carry 0 Pa and must be selected again. Uplift on the
wall-foundation interface is still not modelled. The earlier AT2
default `l = 0.25 m` implied only about 1.72 MPa and had no elastic phase.

For process-of-elimination debugging, an optional top-level `fracture_debug`
block (Configuration > Cracking > Debug in the app) overrides, per job, the
values in [FEniCSx/fracture_defaults.json](FEniCSx/fracture_defaults.json) and
[Aster/fracture_defaults.json](Aster/fracture_defaults.json) without editing
them: `model` (AT1/AT2), `critical_energy_release_rate_j_m2` (both solvers),
`length_scale_m` (null = calibrate to the tensile strength, otherwise fixed; the
report records the implied strength), `residual_stiffness`,
`crack_water_pressure_factor` (0 = dry crack, 1 = full head; both solvers),
FEniCSx `load_steps`, the ten `solver` fields of `PhaseFieldSolverOptions`, and
`aster.band_width_m`/`aster.softening_p`. Any subset may be given; validation is
in `Elmer/model_config.py` `validate_fracture_debug`. The block is nullable:
`"fracture_debug": null` (or omitting it) means the solvers use the shipped
defaults files unchanged. [config.json](config.json) ships the block with
Friday's working values, so the API's `/config/default` (config.json returned
unchanged) and the app's Load Default show them: AT2, Gc 2700 J/m2, fixed
`l = 0.25 m`, residual stiffness 1e-8, factor 0 (dry crack) and 10 load steps
(implied strength about 6.3 MPa). Friday also used hybrid mechanics (the Amor
checkbox) and a local 4 m seed, which are set separately. In the app, unticking
Override solver defaults sends null; ticking it loads the shipped defaults from
`/config/fracture-defaults` for editing. The block is part of the job's frozen
config.

The staggered solve alternates degraded linear mechanics and a PETSc TAO
bound-constrained AT1/AT2 damage minimisation. The lower bound is the damage from
the previous converged load step, enforcing irreversibility. Mechanics uses
GMRES with GAMG; if that fails (for example the stiffness contrast of fully
damaged zones), the same solve is retried once with a MUMPS LU preconditioner,
which is then kept for the rest of the run. The switch is recorded as a
`direct_solver_fallback` event in `mechanics_diagnostics.jsonl` and as
`mechanics_solver.direct_fallback_used` in the report; a MUMPS failure still
fails the run. MUMPS needs considerably more memory than GAMG, so the fallback
factorises out-of-core (ICNTL(22)=1): factors are written to a per-process
directory under `$MUMPS_OOC_TMPDIR`, defaulting to `~/.cache/lamb/mumps_ooc`
(WSL `/tmp` is RAM-backed tmpfs). Directories left by killed runs are removed
the next time the fallback starts. Gravity,
hydrostatic boundary pressure and the existing directional support springs are
included. Nonzero internal crack pressure uses the regularised force
`-p*grad(alpha)`; this is not an explicit face traction. Outputs include
displacement, damage, tensile-energy history, degraded stress, strain,
crack-surface density, energies, convergence histories, reactions and force
balance.

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
| `outputs.*` flags | Validated metadata only; the current solver still writes its standard fields and reports. The app fixes `max_displacement`/`max_tensile_stress` to true and sets `cracking_risk` to whether `cracking` defines a crack (no Outputs tab) |
| `outputs.cracking_risk` | Must be true when `cracking` defines a crack and false for an empty `cracking` array (no crack: app jobs run baselines only); the elastic Elmer model itself does not implement crack propagation |
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


### Chamfered foundation seat: geometry verified, solve blocked

When `geometry.foundation` is present, every station top is now the minimum of
its two incident former staircase column treads (one incident tread at a wall
end). The wall/foundation split interpolates straight between these station
tops. The 114 former 0.5 m risers become chamfers: actual wall-interface triangle
angles are 45.000187--46.847796 degrees across the arch thickness. There are no
vertical wall/foundation faces. Any nonzero riser other than exactly one course
is rejected explicitly rather than silently chamfering across a joint or several
stations. Because the seat can also sit on a thickness-step tread, the
configuration validator (and the app's schema) rejects any
`thickness_steps[].top_z_m` that is not a multiple of `global_element_size_m`
when a foundation is present, naming the nearest valid levels; for example
0.75 m courses with steps at -20 m and -25 m produced 0.25, 0.5 and 1.0 m
risers. No wedge crosses a wall joint: each belongs to the block below its
higher endpoint. Without foundation, the previous wall behavior is unchanged.

The cut wedges transfer **62.188918 m3** from foundation to wall, independently
checked against the wall prism-cell volumes. The exposed margins follow the
same profile. Trench radii remain 74.5--80.5 m; **3490/3490** wall seat nodes snap
exactly to duplicated foundation nodes. The lining uses two structured hex
layers, deterministically split into tets. Its floor follows the same chamfered
profile with a conservative common vertical offset **0.733675 m**, derived as
`depth_m*sqrt(1+maximum_inner_radius_slope**2)`. Thus normal thickness is at least
0.5 m, not exactly 0.5 m everywhere: flat treads have extra thickness. The floor
is deeper than the old stepped trench; exterior terrain is only lowered, never
filled. The lip follows the chamfered top and exterior triangles reconnect to
unchanged survey anchors (maximum lowering 0.947427 m). Crest-end extensions
retain the adjacent lowered wall-end top, not z=0.

HXT generated coplanar, numerically zero-volume rock tets on the new cut side
walls (minimum scaled Jacobian 1.98e-14), correctly rejected by the existing
quality gate. Foundation-present rock therefore uses Gmsh Delaunay (algorithm 1)
instead; no near/far size or distance configuration was changed. Legacy meshes
retain HXT. The final full suite passed **124 tests / 5285 subtests in 229.11 s**,
with the expected rock-quality warning (minimum scaled Jacobian 0.00517965).

| Geometry | Wall min deg | Foundation min deg | Rock near wall min deg | Rock near wall <5 deg |
|---|---:|---:|---:|---:|
| No foundation | 34.328 | -- | 1.684 | 26 |
| Local-width trench | 34.328 | 12.800 | 1.643 | 27 |
| Smooth 6 m trench | 34.328 | 12.068 | 1.279 | 29 |
| Staircase | 34.245 | 33.242 | 0.842 | 197 |
| Benched staircase / exact-plane split | 34.245 | 33.242 | 0.945 | 53 |
| Chamfered | 34.328 | 14.558 | 0.478 | 47 |

Near-wall rock uses centroid-to-nearest-wall-node distance <2 m. Its worst cell
is at (61.663303,-43.185814,-0.742138) m. Smallest near-wall rock edge is 28.620 mm;
wall/foundation have no cells below 5 degrees. The sliver count improves slightly
but the minimum angle worsens: this is not a resolved rock-quality issue.

Exact-plane spring contract shape, region IDs, legacy tags and per-plane
independent elimination are unchanged. Actual inventory becomes **1963
wall_foundation / 3263 foundation_bedrock planes**, because curved-arch inclined
quads resolve to separate true triangular planes; consumers must discover them
from `faces`, not hardcode counts/tags. New `foundation` metadata includes
`seat_definition="stepped_45deg_chamfer"`, `chamfer_definition`, `chamfer_count`,
`chamfer_angle_range_deg`, `transferred_foundation_to_wall_volume_m3`,
`cut_wedge_wall_gain_m3`, `cut_wedge_foundation_loss_m3`,
`minimum_normal_thickness_m`, `riser_course_height_m`,
`risers_more_than_one_course`, and `wall_joint_chamfer_definition`.
`wall_underside.profile` is also `stepped_45deg_chamfer`. Per-plane audit uses a
pair-local sparse index space (including unexpected exported nodes) to avoid
O(global_nodes * plane_count) allocations; the previous global-size audit hit its
400 s limit. All **5226** exported projectors then passed: maximum relative error
**4.824558e-14**, maximum off-plane distance **4.260632e-10 m**, zero nodes above
1e-7 m. Audit source is `eight_rank_timeout_output/model`; its Gmsh source hash is
identical to the final attempt. This verifies the operator, not equilibrium.

**No converged chamfered solve exists.** Fresh complete pipeline attempts, each
with enforced <=540 s wall clock, failed at their solver limit:

| Changed attempt | Solver cap s | Pipeline s | Last logged iteration / relative residual |
|---|---:|---:|---|
| 4 ranks BiCGStabl/ILU1 | 400 | 470.39 | 540 / 2.544e-4 |
| 4 ranks BiCGStabl/ILU2 | 400 | 468.57 | 180 / 1.583e-5 |
| 4 ranks GMRES(200)/ILU1 | 400 | 465.97 | No iteration output before timeout |
| 8 ranks BiCGStabl/ILU1 | 440 | 504.19 | 260 / 2.451e-3 |
| 4 ranks BiCGStabl/ILUT(1e-3) | 440 | 517.42 | No iteration output before timeout |
| 1 rank BiCGStabl/ILU1 | 440 | 500.05 | 360 / 9.796e-4 |

Each retry changed the numerical strategy/decomposition; no unchanged retry,
time-limit extension above 540 s, run continuation, or relaxed convergence gate
was used. Failed preconditioner experiments are not production defaults: source
retains the prior BiCGStabl/ILU1/tolerance 1e-10. Failure archives retain mesh,
SIF, config and residual histories. `Thickness_tests/foundation_chamfered` holds
`geometry_verified.json`, `mesh_quality.json`, `projector_audit.json` and
`verification_summary.json`, but deliberately **no report.json**. Stress peaks,
displacements, gap maxima, Elmer completion time and chamfer-edge-excluded wall
peaks are unavailable; prior split-case values (wall tension/compression
3.031/4.911 MPa, earlier corner-excluded 1.28/2.92 MPa) must not be presented as
chamfer results. Downstream integration has been notified not to consume this
unconverged reference. Next work must address solver conditioning/cost without
weakening the accuracy gate or silently changing mesh/material/load semantics.


### Near-planar chamfer mortar groups: audit passed, equilibrium still blocked

This supersedes **exact-plane grouping for foundation-present source generation**,
not the historical split-case results above. `geometry.foundation` and all
physical dimensions, meshes, loads, material constants and spring stiffnesses
are unchanged. Full source mesh comparison confirmed identical **196557 nodes /
961696 tets / volume-body labels** against the exact-plane chamfer mesh.

Cheap diagnosis (`Thickness_tests/near_planar_diagnosis.json`) found substantial
fragmentation, but does **not** prove it is the sole cause of slow convergence:

| Reference/interface | Groups | One-triangle groups | Shared slave nodes | Maximum groups per slave node |
|---|---:|---:|---:|---:|
| Converged staircase split, wall/foundation | 174 | 0 | 2252 | 2 |
| Converged staircase split, foundation/rock | 849 | 0 | 4324 | 4 |
| Exact-plane chamfer, wall/foundation | 1963 | 1762 | 2090 | 5 |
| Exact-plane chamfer, foundation/rock | 3263 | 2318 | 4355 | 5 |
| Near-planar chamfer, wall/foundation | 280 | 0 | 1979 | 4 |
| Near-planar chamfer, foundation/rock | 503 | 0 | 3292 | 4 |

Full triangle-count and node-membership histograms are saved in the diagnosis
and grouped `verification_summary.json`. Shared edge nodes are legitimate FE
nodes; independent elimination changes when groups merge. Arbitrarily assigning
each node to one face group would discard portions of its spring-support area
and was not implemented. The converged staircase itself also has shared rows,
so their presence alone cannot establish a failure mechanism.

Grouping is a deterministic edge-connected flood, using coincident geometric
edges, not vertex proximity. Each group has one orientation class: `tread`
(horizontal), `chamfer` (inclined), `side_wall` (vertical radial side), or `riser`
(other vertical faces, including trench end caps). A candidate is accepted only
if all face-normal pairs differ by <=**3 degrees** (ignoring normal sign), and
all group vertices are within **0.005 m** of an SVD-fitted plane. Both constraints
are rechecked for the proposed whole group, so incremental angular drift is
not allowed. No tread/chamfer/side/end orientation changes are pooled.
Actual maximum normal spread is **1.518795 degrees**. Class inventories:
wall/foundation 95 tread +185 chamfer; foundation/rock 96 tread +226 chamfer
+179 side_wall +2 vertical end-cap groups. The previous exact-pair helper name
`split_planar_interfaces` is retained, but now produces the bounded near-planar
contract below.

Logical names, nested `faces`/`planes`, independent group elimination, region IDs
and gap aggregation remain structurally unchanged. Each foundation entry now
has `grouping="near_planar"`, `tolerance_m=0.005`, `max_angle_deg=3.0`,
`continuity="edge_connected; single orientation class"`. Each `planes[part]`
retains fitted `unit_normal`, `offset_m`, `triangles_per_side` and adds
`orientation_class`, `maximum_off_plane_distance_m`, `normal_spread_deg`.
Part names use the four class prefixes above, plus deterministic traversal
indices. First groups retain 11/12 and 17/18; extra semantic tags are currently
**1000--2561**. Consume inventory and `boundary_id_map`, not fixed counts/tags.
The no-foundation legacy contract is unchanged. Existing downstream strict
`exact_plane`/coplanarity gates require a separately authorized adaptation;
structural similarity is **not** verified downstream compatibility.

`mortar_projector_audit` now independently checks fitted-plane distance against
the declared tolerance, actual normal spread, single orientation class and
absence of exported nodes outside that group's slave/master inventory. It also
**retains** the strict exact matching-triangle mass comparison (relative 1e-8
limit), and returns a failing exit status for any failed gate. All **783**
exported groups pass: maximum exact-mass relative error **2.845610e-14**;
maximum exported-stencil distance to fitted plane **0.004992395 m**; zero nodes
outside 0.005 m; zero orientation-class violations. This fitted-plane distance
is the permitted curvature, **not** evidence of a wrong orientation projection;
the independently assembled exact triangle operator matches to roundoff.
Run `python -m Elmer.mortar_projector_audit --model <model> --output <json>` in the
SciPy environment (WSL here), inside the enforced wall-clock cap.

Full suite: **126 passed / 842 subtests passed in 238.65 s**, one unchanged rock
mesh-quality warning. Tests cover connectedness, no cross-orientation merging,
whole-group tolerance/normal bounds, matching pairs, snapping, wedge volumes,
and absent-foundation behavior. Fewer subtests reflect fewer groups, not lost
logical coverage.

The user-authorized fresh run `Thickness_tests/foundation_chamfered_grouped`
remains **NOT_CONVERGED**. Group reduction alone improved iteration throughput
but did not meet the unchanged 1e-10 convergence gate:

| Attempt | Solver cap s | Pipeline s | Last logged iteration | Last relative residual |
|---|---:|---:|---:|---:|
| Grouped, four-rank BiCGStabl/ILU1 | 440 | 509.023 | 840 | 1.954e-5 |
| Sole permitted follow-up: same model, ILU2 | 440 | 511.639 | 360 | 6.222e-9 |

Both attempts failed at their limits; neither was continued or split to evade
the 540 s rule. No further solver attempt was made. Source defaults retain
BiCGStabl/ILU1 rather than persisting an unsuccessful solver experiment; the
final run's saved SIF/diagnostic records ILU2 accurately. No accuracy gate was
relaxed. Raw logs/SIFs/exports and timings are retained; `ilu1_timeout_output`
contains the first attempt. There is deliberately **no report.json**, solved
stress/displacement/gap table, completed Elmer time, or chamfer-edge-excluded
wall peak. The prior split-case 3.031/4.911 MPa and 1.28/2.92 MPa excluded peaks
are not results for this model. No solved downstream reference was announced.

Geometry/quality is unchanged by grouping: wall/foundation minimum dihedrals
34.328/14.558 degrees, zero below 5 degrees; near-wall rock minimum **0.477558
degrees**, **47** below 5 degrees, minimum edge 28.620 mm, worst centroid
(61.663303,-43.185814,-0.742138) m. Earlier no-foundation/local-width/smooth6m/
staircase/benched-split minima and counts remain 1.684/26, 1.643/27, 1.279/29,
0.842/197, 0.945/53, respectively. `mesh_quality.json` records the exact array
comparison supporting reuse of the measured chamfer quality, rather than
claiming a new mesh measurement. Diagnostic, audit, validation and source
snapshots are retained in the fresh run.


### Installed solver capability check and final bounded AMG attempt

The installed `C:\Program Files\Elmer 26.2-Release` binaries/libraries were
inspected, then tested on a tiny clamped 3D elastic cube (125 nodes / 384 tets).
No MUMPS, Pardiso or HYPRE dependency libraries were found. Their actual fixture
runs each exited 1 with explicit `CheckLinearSolverOptions` errors:
`MUMPS solver has not been installed`, `Pardiso solver has not been installed`,
and `Hypre requested but not compiled with`. Thus neither a MUMPS full solve nor
BoomerAMG was attempted speculatively. Internal algebraic multigrid did solve
the serial fixture (exit 0, 0.15 s solver time); its MPI fixture failed explicitly
with `AMGSolve: This algebraic multigrid is not parallel`.

The one authorized full attempt therefore used serial Elmer internal AMG with
BiCGStabl, the unchanged relative tolerance 1e-10, `MG Method=algebraic`,
`MG Levels=10`, block symmetric Gauss-Seidel (`bsgs`), one pre/post smoothing
sweep, determining component 1, strong connection limit .06, projection limit
.6 and non-direct interpolation. Settings follow the upstream Elmer elasticity
fixture `fem/tests/amultg2/stress_amg.sif`; production source defaults were not
changed. The fresh output is `foundation_chamfered_grouped/internal_amg_output`.
The serial experiment changes only solver/decomposition settings, not physical
or geometric inputs, and is reproducible via the saved `solver_capabilities/
run_internal_amg.py` within an enforced 540 s wrapper.

**The full AMG attempt failed, not by timeout or memory pressure:** Elmer exited
1 while building coarse interpolation, with
`InterpolateFineToCoarse: There are more neighbours than expected` (reported
neighbour counter 1001). Hierarchy coarse-node counts were 77244, 25496, 12294,
then 9604 before failure. There was no iterative residual history or completed
solution. Total pipeline time was **155.42 s**; the memory-sampled solver interval
was approximately **48.13 s** (not a completed Elmer solve time). No further full
solver attempts were made.

Because this is a shared 32 GB machine, global committed memory was already
**44.11 GB** before launch (pagefile-backed commit is not resident physical RAM).
The user-mentioned 24 GB total-commit threshold was therefore already exceeded;
MUMPS was unavailable regardless. The AMG attempt used more conservative
process-specific safety limits: solver private commit <=8 GB and physical RAM
available >=2.5 GB, checked every second. Observed solver private peak was
**3.872 GB**, global commit peak **47.858 GB**, minimum physical available
**7.975 GB**. Neither guard triggered. No unrelated process was stopped. Solver
limit 440 s, whole-pipeline internal guard 515 s and outer enforced 540 s remained
in place; useful logs/telemetry were retained.

`solver_capabilities/availability.json`, fixture logs, `binary_evidence.txt`,
`internal_amg_attempt.json` and `internal_amg_output/model/solver.log` record the
findings. No source behavior changed, so the previous **126 tests / 842
subtests** result remains the applicable full-suite baseline; this iteration
added capability fixtures, not another full suite. No solved report.json,
new stress/displacement/gap values, corner-excluded peaks or downstream parity
exists. Previous successful projector audit and unchanged mesh-quality evidence
remain construction checks only. Production BiCGStabl/ILU1 is retained; the
prior grouped ILU2 residual 6.222e-9 versus required 1e-10 remains the best logged
bounded iterative progress. A user-submitted app job may run without the
agent-development cap, but the verified internal-AMG implementation failure
will not be fixed merely by allowing more time. No FEniCSx/API/WPF/config edits,
no contract change, no commits.
