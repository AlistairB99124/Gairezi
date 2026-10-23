# FEniCSx dam analysis adapter

This directory contains the DOLFINx 0.11 mechanics and phase-field adapter. The authoritative geometry and material/load input is Gairezi/config.json; the adapter does not maintain a second dam configuration. Elmer?s three_body.msh and diagnostic.json are the canonical solver-mesh contract when using the saved source mesh.

## Canonical regions and connection

The canonical mesh has wall = 1 and bedrock = 2, plus concrete foundation = 3 when `geometry.foundation` is configured. Without a foundation the wall sits directly on surveyed bedrock. With a foundation, the exact saved Elmer wall/foundation/rock geometry is preserved, including the excavated staircase seat. Each interface is either spring-connected through duplicated nodes or bonded through conforming shared-node faces, as selected independently in configuration. The wall is the only damageable region; bedrock and foundation remain intact, even though foundation uses the wall concrete material. The legacy module names three_body_model.py, three_body.msh, and three_body_geometry.pvtu are retained for compatibility.

The configured bedrock support model controls the wall/bedrock seat: bonded meshes use shared-node faces, while spring-supported seats use the opposing boundary tags 11 and 12 and the configured x/y/z stiffnesses. Rock-bottom support uses tag 10; outer rock boundaries use tags 13?16. Elastic reports use wall_connection and wall_bedrock_connection. Wall headline stress peaks remain wall-only; bedrock and optional foundation peaks are reported separately in `body_stress`.

For a foundation source, the mesh importer verifies dimensions, body IDs, interface connectivity, spring tags and stiffness against the submitted configuration and diagnostic. `geometry.foundation.support` is required and never inherits wall-joint settings. Spring-connected `wall_foundation` uses `geometry.foundation.support.spring_n_per_m3`; `foundation_bedrock` uses `geometry.bedrock.support.spring_n_per_m3`. Spring-connected interfaces require their opposing tagged faces and duplicated nodes. Bonded interfaces require matching exposed faces with shared nodes, and their spring coefficients are inactive. Legacy spring sources use tags 17/18 and 11/12 respectively. Exact-plane spring sources declare nested `spring_pairs[logical].faces[plane]` semantic tag pairs, resolved through `boundary_id_map`; the first plane retains the legacy tags and further planes use tags >=1000. The importer checks unique tags, configured logical coefficients, body membership, triangle counts, plane metadata and coplanarity. Each plane multiplier is eliminated independently before summing operators, including at shared riser/tread nodes; pooling plane projectors before elimination is a different discretisation. Reports retain logical interface names, with per-plane details under `faces` and `grouping: exact_plane`. Wall thickness-block joint pairs are retained. Reports preserve foundation metadata and set `wall_bedrock_connection` to `not_in_contact`; the two configured foundation connections are reported independently. Exposed untagged foundation faces are traction-free, as for bedrock. Without a foundation, the existing wall/bedrock support and damage behavior is unchanged.

## Near-planar chamfer interfaces

Foundation sources may declare `grouping: near_planar`, positive `tolerance_m`
and `max_angle_deg`, and `continuity: edge_connected; single orientation class`.
Each group's `planes` entry declares its fitted normal/offset, orientation class,
maximum off-plane distance and pairwise normal spread. The importer recomputes
these from both source mesh sides, checks edge connectivity, one orientation class,
coincident triangles, configured coefficients, body memberships and unique tags.
Fits use Elmer's unique 8-decimal coordinate keys; geometry checks allow only
serialization roundoff (1e-8 m), not extra physical grouping tolerance.

The group fit plane is bookkeeping, not a replacement integration surface.
FEniCSx integrates each actual slave triangle against its coincident master
triangle, preserving curved surface area and the exported Elmer `[M,-M]` mass.
It eliminates the multiplier independently for each declared group, then sums.
Exact-plane and legacy interface numerics remain unchanged. Reports retain logical
names and carry grouping limits/continuity plus `source_plane` and semantic `tags`
per group. The API checks these against the job diagnostic, including refined
baseline reports; WPF does not parse spring-pair inventories.

`audit_interface_projectors.py --model MODEL --projectors EXPORTS --output REPORT`
reuses production block assembly and elimination to compare each group with the
Elmer projector exports (run in WSL with the same 520 s agent-run cap). This is an
operator audit only, not an elastic equilibrium solve or a verified fracture
baseline. The 2026-10-22 audit of all 783 `foundation_chamfered_grouped` foundation
groups (280 wall/foundation, 503 foundation/bedrock) passed: maximum relative
Frobenius projector difference `2.8501584e-14`, eliminated operator difference
`3.9532666e-14`. Evidence is in
`results/near_planar_integration/projector_comparison.json`.
`foundation_chamfered_grouped` has no converged Elmer reference, so full
elastic parity is unavailable. App jobs retain no implicit runtime cap; a nonzero
Elmer worker exit fails the job before FEniCSx starts, even if leftover result
files exist. Existing displacement/stress/equilibrium gates are unchanged.

## Staircase-reference parity caveat

The wall stress gate is unchanged at `1e-4`. Its metric is the unweighted Frobenius
norm of the difference of the full symmetric 3x3 recovered nodal tensors, divided
by the corresponding Elmer tensor norm, over each wall P1 node once. Off-diagonal
components therefore occur twice; foundation and bedrock nodes are excluded.

The saved `Thickness_tests/foundation_stepped_benched` Elmer 26.2 reference has a
weighted-projector point-search artifact at angled staircase riser/tread corners.
A Gaussian point on a tread can pass an adjacent riser's axis-aligned bounding-box
and projected-barycentric tests despite being off its plane. The exported Elmer
wall/foundation projector has the same slave mass as the exact matching-triangle
operator, but a different master coupling. FEniCSx deliberately requires coplanar
triangle intersections; it must not reproduce this off-plane mapping to pass a gate.

Diagnosis on 2026-10-22 found wall stress error `1.0913933e-4`, versus `7.7378759e-7`
for the pre-foundation `stepped_blocks_springs` case. Recovering stress from the
*same Elmer displacement* agrees to `1.8776440e-9`, excluding stress averaging as
the cause. A diagnostic-only FEniCSx replay of the exported reference projectors
reduces stress error to `5.3500442e-7`. This replay is not the production operator
and is not an approved fracture baseline. That historical reference still fails;
its off-plane search artifact must not be copied into the production operator.

The corrected `Thickness_tests/foundation_stepped_split` reference groups both
foundation interfaces by exact plane (174 wall/foundation and 849 foundation/rock
subpairs). Its exported projectors agree with exact integration to relative
Frobenius error <= `1.954084e-14`; the volume mesh is unchanged. Production FEniCSx
parity passes with displacement error `1.1998608e-7`, wall nodal stress error
`4.6972060e-7`, and relative force imbalance `8.0311953e-11`. Results are saved under
`results/foundation_split_parity/dam_elastic_baseline/20261022T104209944357Z`.
No gate or physical input was changed. Passing parity is not engineering validation.

## Running

Run commands from WSL Ubuntu, where DOLFINx 0.11 and the solver dependencies are installed. The saved Elmer model directory must contain the matching config.json, diagnostic.json and three_body.msh. A phase-field fracture run requires a matching completed elastic baseline, the Elmer result used for the user-selected crack location, and the canonical model directory. Do not treat convergence or successful execution as engineering validation or evidence of dam safety.

The test suite is run with:

    python3 -B -m unittest discover -p 'test_*.py'

Small mechanics, mesh-import, crack-placement and report tests are in the test_*.py modules. Tests requiring an external canonical Elmer result are skipped when that result is not available.
