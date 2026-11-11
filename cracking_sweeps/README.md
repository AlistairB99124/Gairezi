# Selected-material cracking sweep

This is a hypothetical sensitivity experiment, not calibrated concrete or a dam
safety assessment. It searches for **maximum local normal opening >= 2 mm at
full design load (factor 1)**. It does not claim a connected 2 mm-wide crack,
engineering validation, or guarantee that any scenario reaches the target.

## Fixed baseline and scenario design

The runner snapshots `../config.json` at launch. It never edits that file.
Every case retains its geometry, mesh, intact materials, loads, supports, selected
cell identities and candidate facets. Only the selected material's E, tensile
strength and fracture energy can change. Density, Poisson's ratio and compressive
strength remain fixed. Selected foundation cells, if present, receive the same
assigned properties as the selected wall cells.

**The current config selects the base band and retains interface_scale = 0.01.**
The user explicitly selected this baseline, including its 99% wall-foundation
spring reduction. It is not the successful historical run with interface_scale=1.
Changes subsequently made to the baseline are captured at the next launch.
The runner requires cohesive selection enabled, no manual cracks or friction,
and maximum_overload_factor=1.

With the current uncapped base-band selection, cells are selected geometrically,
even if no intact facet exceeds the concrete tensile strength. The intact concrete
strength still drives the overstress diagnostics, and capped selection uses the
critical facets. These are distinct from the selected material's strength, which
controls its cohesive resistance after assignment. This sweep does not lower the
intact concrete strength to force a different selection.

`scenarios.json` defines 23 ordered cases as factors relative to the baseline's
selected material, not the intact concrete:

| Family | Cases | Changes (with current selected E=35 GPa, ft=2 MPa, Gf=100 J/m2) |
|---|---:|---|
| A | 1 | Unchanged baseline control |
| B | 6 | ft = 1.5, 1, 0.5, 0.2, 0.1, 0.02 MPa; E and Gf fixed |
| C | 9 | ft = 1, 0.2, 0.02 MPa crossed with Gf = 50, 25, 10 J/m2 |
| D | 3 | E = 17.5, 3.5, 0.35 GPa; ft and Gf fixed |
| E | 4 | E = 3.5, 0.35 GPa crossed with ft = 0.2, 0.02 MPa; Gf=10 J/m2 |

B isolates initiation strength, C explores propagation resistance, D isolates
bulk stiffness, and E explores combined severe degradation. The extremes are
deliberately hypothetical bounds, not suggested measured material data.
`2 Gf / ft` is the characteristic full-separation opening, **not an imposed or
maximum equilibrium opening**. Making this quantity 2 mm does not force the dam
to open 2 mm.

## Launch

In the Lamb or Gairezi workspace, run the VS Code task:

**Aster: Run 2 mm selected-material cracking sweep**

Alternatively, in PowerShell from Gairezi:

```powershell
# Write a fresh plan only; no solver processes:
.\.venv\Scripts\python.exe -B -m cracking_sweeps.run_sweep

# Launch a fresh, complete sequential campaign:
.\.venv\Scripts\python.exe -u -B -m cracking_sweeps.run_sweep --execute
```

Use the existing project environment and installed Elmer/Ubuntu WSL Code_Aster.
Host preparation dependencies are listed in `../client_requirements.txt`,
including SciPy for the shared preparation imports.
Do not run app solver jobs or other campaigns concurrently: memory usage can be
large. Keep the PC awake and powered. There is no 10-minute user-job cap; native
Code_Aster stages retain the existing 30-day safeguard. Conversion helpers retain
their normal limits. This campaign can take many hours and substantial disk space.

## Isolation, execution and outputs

Every invocation creates a unique ignored `runs/<UTC timestamp>-<id>/` directory.
There is no implicit resume or overwriting of past evidence.

1. Save baseline/configuration and all planned case configurations in `plan.json`.
2. Execute one fresh isolated Elmer reference and Aster elastic baseline.
3. Validate source hashes, finite displacement and physical node/cell identities
   using existing Aster preparation. Screen and prepare candidates **once**.
4. Copy the candidate archive for each material case; preserve physical selection,
   baseline hashes and source config. Record derivative config and its independent
   hash in `provenance.json`; do not present it as a new solved elastic baseline.
5. Execute existing native cohesive solving and postprocessing sequentially.
   The production load steps, convergence criteria and material laws are unchanged.
6. Persist `sweep.json`, `results.csv` and each case's `result.json` after every case.
   Per-case `case.log` and `aster/run/solver.live.log` retain detailed diagnostics;
   native reports and ParaView artifacts remain with their cases.
7. Stop after the first accepted full-design-load result reaches at least 2 mm
   normal opening. Save that result before stopping; `sweep.json` records
   `stop_reason` and the remaining `skipped_cases`.

The intact baseline is reusable because selected-material overrides apply only
after screening. No app jobs or historical files are read or modified. A completed
baseline with a cross-solver comparison requiring verification is explicitly
recorded as such; it is not engineering acceptance.

Only a completed, accepted, whole-model authoritative Aster report with native
return code 0, finite nonnegative opening and final load factor 1 can reach the
target. The comparison uses the unrounded opening in metres, not plotted
exaggeration or wall displacement. `results.csv` reports opening in millimetres,
initiation/opening/separation counts, touched facet area and wall displacement.
Touched area is not necessarily fully separated area.

Numerical failures are retained and the next material case is attempted. A failed
A control is conspicuous in the summary: later cases are exploratory scenarios,
not a validated comparison to a converged cracked control. Shared-reference
corruption or reference/preparation failure aborts the sweep. Cancellation stops
the campaign and preserves completed evidence. There is no automatic weakening,
tolerance relaxation, support change or overload. A large opening in a failed,
partial or unaccepted solve does not trigger the early stop.
Exit code 2 means at least one case failed; no hits with all cases converged is
a valid completed sweep, not a claim that the 2 mm target was achieved.

If no accepted scenario reaches 2 mm, these bounds have not demonstrated that
target. Non-convergence is not evidence of physical failure or a 2 mm crack.
Post-peak load-path control, mesh sensitivity and physical calibration would be
separate follow-up studies.

## Hydraulic loading scope

This config-based sweep retains external upstream/downstream hydrostatic pressure
and crest surcharge. The baseline now explicitly enables
`damage_selection.crack_water_pressure`: native Aster applies paired hydrostatic
forces inside actually initiated, opened facets connected to the reservoir
through open, submerged edges. Elastic penalty compliance alone is not a crack.
Only stable hydraulic/mechanical active sets are accepted; missing wet-solve
evidence cannot count as a target hit. Each new sweep snapshots this setting.

Foundation uplift remains **off**. Reducing interface spring stiffness is not
uplift pressure. Missing/false `crack_water_pressure` preserves historical dry
behavior; `fracture_debug.crack_water_pressure_factor` is not this Aster switch.
The intact baseline remains dry because no internal crack is open.

Wet runs start at load factor 0.01, then 0.025, 0.05, ... to full design load,
with no internal mechanical subdivisions that bypass pressure updates. This
uses instantaneous reservoir head and small-strain reference geometry, not
transient flow, permeability, drainage, entrapped water or follower pressure.
There is no silent fallback to a dry solve if pressure coupling fails.
