# Wall-geometry cracking sweep

Runs the baselines and the contact-joint cracking stage for a list of wall geometries, then
writes the results into `FEA_Matrix.xlsx`. It is a comparison of model variants, not an
engineering assessment; the method and its limits are in
[../aster_dam_baseline.md](../aster_dam_baseline.md).

## Files

| File | What it is |
|---|---|
| `config.json` | The configuration every scenario starts from. |
| `scenarios.json` | An array of overwrites of `config.json`, one per run. Runs are numbered from 1 in file order. |
| `FEA_Matrix.xlsx` | The results table. Run *n* is row *n* + 2 of the `Sweep` sheet. |
| `run_sweep.py` | The runner. |
| `runs/` | One folder per sweep (not in git). |

`../config.json` is not used for the scenarios, but the solver modules load it when they
start, so it has to be a valid configuration. No file outside this folder is changed.

## Scenarios

A scenario is the part of `config.json` it replaces. Objects are merged key by key; a list
or a value is replaced as a whole. Anything a scenario does not name keeps the value in
`config.json`.

```json
{
    "geometry": {
        "wall": {
            "radial_width_m": 4.0,
            "centerline_radius_m": 78.0,
            "downstream_radius_m": 76.0,
            "thickness_steps": [
                { "top_z_m": -20.0, "radial_width_m": 4.5 }
            ]
        }
    }
}
```

Every scenario is checked with the project's configuration validator before anything
starts, so a misplaced or misspelt key stops the sweep with the scenario's number.
`thickness_steps` belongs inside `geometry.wall`. The sweep also refuses a scenario the
contact-joint stage cannot run (no foundation, or a bedrock mode other than
`elastic_layer`) and a wall with more than the two thickness steps the sheet has columns for.

Runs 8 to 10 are runs 5 to 7 with a mortar fill on the steps: the same `geometry.wall` and a
`material.mortar` block (see "Mortar fill on the steps" in [../README.md](../README.md)). The
fill slopes the ledges of the stepped walls so they look smooth; it is a soft body of its
own, not concrete. Its material values are placeholders.

## What a run does

For each scenario, one at a time, in its own folder `runs/<sweep>/scenario_NN/`:

1. **Elmer baseline** (`Elmer.three_body_solver`).
2. **Code_Aster elastic baseline** (`Aster.run_three_body_pipeline`).
3. **Code_Aster contact joint** at the wall/foundation contact
   (`Aster.dam_joint_baseline app`): dry joint, then water in the joint. Each of the two
   solves is stopped after 10 minutes.

These are the commands the app runs for a job with `CRACKING_ENABLED` set. The app's
FEniCSx baseline is not run; the sheet does not use it.

A stage that fails ends its scenario and the sweep goes on to the next one. Figures from the
stages that did complete are kept.

## The sheet

When the scenarios are done, each run's row is replaced: run number, geometry, the Elmer and
Code_Aster baseline figures (peak wall tension and compression in MPa, peak wall
displacement in mm), the concrete volume of the wall and foundation (m3), and the cracking
columns (largest opening of the contact in mm, peak wall tension, compression and
displacement; dry and with water in the joint). The averages and the volume difference are
formulas. The last column (AC, "Mortar fill / Volume (m^3)") is the volume of the mortar fill,
or "-" for a wall without one; the fill is not part of the concrete volume. The column is
added to a workbook that does not have it yet.

- A figure a run did not produce is left empty. Nothing from an earlier sweep stays in a row
  this sweep wrote.
- Rows of runs that were not part of this sweep are left as they are.
- The workbook as it was is copied to `runs/<sweep>/FEA_Matrix.before.xlsx` first.
- A note below the table names the sweep and any runs that did not complete.
- The workbook must be closed in Excel. This is checked before the first solver starts. If it
  is open when the sweep ends, the results are kept and can be written later (see below).

## Launch

VS Code task, in the Lamb or the Gairezi workspace:

**Cracking sweep: Run scenarios and update FEA_Matrix.xlsx**

Or in PowerShell from `Gairezi`:

```powershell
# Check the scenarios and write the plan only; no solver is started
.\.venv\Scripts\python.exe -B -m cracking_sweeps.run_sweep

# Run every scenario, then update the sheet
.\.venv\Scripts\python.exe -u -B -m cracking_sweeps.run_sweep --execute

# Run only scenarios 5 and 6; only their rows are written
.\.venv\Scripts\python.exe -u -B -m cracking_sweeps.run_sweep --execute --only 5 6

# Run only the three stepped walls with a mortar fill (rows 10 to 12 of the sheet)
.\.venv\Scripts\python.exe -u -B -m cracking_sweeps.run_sweep --execute --only 8 9 10

# Write the results of a finished sweep into the sheet, without solving
.\.venv\Scripts\python.exe -B -m cracking_sweeps.run_sweep --update-sheet cracking_sweeps\runs\<sweep>
```

`--config`, `--scenarios`, `--sheet` and `--output-root` point the sweep at other files.

- **Time.** On the development machine one scenario of the default mesh took about 15
  minutes through the app (Elmer 4, Code_Aster baseline 3, contact joint 8). Seven scenarios
  are then somewhat under two hours. This is an estimate: a full sweep has not been timed.
  The sweep of 12 November 2026 ran scenarios 1 to 7 in 110 minutes. Scenarios 8 to 10 have
  not been run at the default mesh. With the fill, Elmer took 3.4 times as long on a coarse
  mesh (168 s against 50 s), so their Elmer stage will be longer than 4 minutes; how much
  longer at the default mesh is not known.
- **Do not run app jobs or another sweep at the same time.** Each solver uses all cores.
- **Disk.** A scenario on a coarse mesh (40,000 wall elements) left 61 MB. The default mesh
  will leave several times that per scenario.
- **Stopping.** Ctrl+C stops the running stage and its solver processes. Finished scenarios
  stay in the sweep folder; `--update-sheet` writes them.
- **Needs** the project's `.venv` (`../client_requirements.txt`, which now includes
  `openpyxl`), Elmer, and Code_Aster in WSL.

Exit code 0: every scenario completed and the sheet was written. 2: a scenario failed or
the sheet could not be written. 1: the sweep was not started. 130: interrupted.

## Output of a sweep

`runs/<UTC time>-<id>/`:

| File | Content |
|---|---|
| `config.json`, `scenarios.json` | Copies of the inputs as they were at launch. |
| `plan.json` | Every scenario with its full configuration. |
| `progress.log` | One line per stage, and the contact-joint solver's progress. |
| `results.json`, `results.csv` | The figures of every scenario, saved after each one. |
| `scenario_NN/` | That scenario's `config.json`, the stage logs (`elmer.log`, `aster_baseline.log`, `joint_opening.log`), solver output and `result.json`. |
| `FEA_Matrix.before.xlsx` | The workbook before this sweep wrote to it. |

## Tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest cracking_sweeps.test_run_sweep
```

They use stand-ins for the solvers and take a few seconds.

## Earlier content of this folder

Until 12 November 2026 this folder held a sweep of the retired selected-material cracking
method (23 material cases). That runner and its scenarios are in the git history.
