# FEniCSx dam analysis adapter

This directory contains the DOLFINx 0.11 mechanics and phase-field adapter. The authoritative geometry and material/load input is Gairezi/config.json; the adapter does not maintain a second dam configuration. Elmer?s three_body.msh and diagnostic.json are the canonical solver-mesh contract when using the saved source mesh.

## Canonical regions and connection

The canonical mesh has two volume regions: wall = 1 and bedrock = 2. The wall sits directly on surveyed bedrock and its underside follows the configured piecewise-linear bedrock contour. The wall is the only damageable material; bedrock remains intact. The legacy module names three_body_model.py, three_body.msh, and three_body_geometry.pvtu are retained for compatibility.

The configured bedrock support model controls the wall/bedrock seat: bonded meshes use shared-node faces, while spring-supported seats use the opposing boundary tags 11 and 12 and the configured x/y/z stiffnesses. Rock-bottom support uses tag 10; outer rock boundaries use tags 13?16. Elastic reports use wall_connection and wall_bedrock_connection. Wall stress peaks are reported separately from bedrock results.

## Running

Run commands from WSL Ubuntu, where DOLFINx 0.11 and the solver dependencies are installed. The saved Elmer model directory must contain the matching config.json, diagnostic.json and three_body.msh. A phase-field fracture run requires a matching completed elastic baseline, the Elmer result used for the user-selected crack location, and the canonical model directory. Do not treat convergence or successful execution as engineering validation or evidence of dam safety.

The test suite is run with:

    python3 -B -m unittest discover -p 'test_*.py'

Small mechanics, mesh-import, crack-placement and report tests are in the test_*.py modules. Tests requiring an external canonical Elmer result are skipped when that result is not available.
