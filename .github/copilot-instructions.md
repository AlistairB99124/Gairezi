# Repository guidance for Copilot

## Project context

This repository contains the existing dam-analysis Python workflows and the
design plan for a future Windows WPF application. Read
`Design_Docs/app_plan.md` before implementing Workbench or API features; it
defines the configuration contract, implementation order, job lifecycle,
security boundary, and acceptance gates.

The root `config.json` is the authoritative dam-data contract. Do not create a
second independently maintained configuration or silently reinterpret physical
inputs. Preserve the documented distinction between active inputs, nominal
metadata, and currently unsupported output flags.

## Repository and Visual Studio layout

- The WPF application and Visual Studio solution live in the parent `Lamb`
  repository: `..\Lamb.sln` and `..\Lamb.Windows.App\`.
- The parent solution already includes `Lamb.API\`; inspect and reuse that
  project before creating another API project.
- This `Gairezi` checkout is a separate nested Git repository. Do not move,
  merge, or change its Git boundary as part of ordinary code work.
- Do not add the entire Gairezi repository as a C# project or create another
  WPF solution here. Open the parent `Lamb.sln` for desktop application work.
- The Python solver remains Python code; it does not need a C# wrapper or
  project reference. Keep `config.json` here as the dam-data contract.
- Keep generated meshes, solver outputs, logs, and job data out of source
  control. User jobs belong in application data, not beside installed binaries.
- Do not invent or change the WPF project's target framework or replace its
  project settings without checking the existing project and installed Visual
  Studio/.NET support.

## Implementation order and engineering safeguards

- Follow the phases and gates in `Design_Docs/app_plan.md`. The first
  submission milestone is an isolated configuration-to-results Python worker;
  do not connect a Submit button to the current shared-output CLI until that
  worker and the API job boundary are ready.
- Preserve existing command-line defaults and numerical behavior when
  refactoring the Elmer workflow. Require explicit config/output paths for
  isolated jobs, and never overwrite the repository's `config.json`.
- Keep solver execution and post-processing failures explicit. Do not report
  success based on stale artifacts; validate generated results before exposing
  them.
- Bind any local API to loopback, use the local-client protection described in
  the plan, and never accept arbitrary executable paths or shell commands from
  HTTP input.
- Do not claim engineering validation or dam safety from successful execution
  or convergence.

## Working practices

- Inspect the existing implementation and tests before changing behavior.
- Make small changes that preserve existing patterns; do not modify unrelated
  FEniCSx, Code_Aster, or client-result workflows.
- Add or update focused regression tests for behavior changes. Run the smallest
  relevant test set and report exactly what was and was not verified.
- Keep documentation in sync with changed contracts and behavior.
- Preserve existing user changes and never commit or discard files as part of
  ordinary implementation work.
