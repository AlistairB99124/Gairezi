
PROMPT 1

You are acting as the lead computational mechanics and software engineer on this project.

I need you to implement a reliable automated concrete wall cracking workflow in the existing dam FEA application.

IMPORTANT PROJECT CONTEXT

- The project already has a valid static FEA analysis without cracking.
- The mesh contains both the concrete dam wall and the foundation.
- We ONLY want to introduce cracks into the concrete dam wall.
- The foundation must remain intact and must not be modified by the cracking algorithm.
- The existing static analysis is our trusted baseline. Preserve it.
- The application uses Code_Aster and Python orchestration.
- The project has previously attempted cohesive facet cracking and ENDO_FISS_EXP, without obtaining a reliable complete workflow.
- The existing cohesive implementation, stress screening, mesh processing, solver orchestration and reporting must be inspected before deciding what to reuse.
- Do not assume the existing implementation is correct merely because it exists.
- Do not assume a completely new fracture formulation is necessary.

THE REQUIRED END RESULT

I want the application to accept a configurable maximum permitted tensile stress in the concrete wall.

The application should then:

1. Run or load the valid static baseline analysis.
2. Evaluate the wall's stress state.
3. Identify regions where maximum principal tensile stress exceeds the configured limit.
4. Introduce cracks into the wall in those regions using a mechanically defensible fracture approach.
5. Recalculate equilibrium with the cracks present.
6. Allow stresses to redistribute.
7. Repeat the process until the wall satisfies the configured tensile-stress criterion, or until a clearly identified stopping condition is reached.
8. Produce a report showing the cracking process, remaining tensile stresses, crack locations, convergence and whether the target was achieved.

The foundation must remain unchanged throughout.

ENGINEERING RULES

- Use maximum principal stress, with tension positive, for tensile-stress assessment.
- Do not confuse the tensile initiation strength of concrete with the user's permitted residual tensile-stress threshold. They are separate concepts.
- Do not arbitrarily reduce material strength to force cracking.
- Do not delete overloaded elements simply to make the stress report pass.
- Preserve appropriate fracture-energy regularisation and cohesive/contact behaviour if using a cohesive approach.
- Do not claim that a stress reduction is physically meaningful unless the mechanical solution has converged.
- Never report success merely because the solver completed a partial load step.
- Do not silently discard or reset fracture history between cracking iterations.
- Do not introduce cracks in foundation elements or at the wall-foundation interface.
- Do not modify the trusted static baseline workflow unnecessarily.

YOUR FIRST TASK

Inspect the entire relevant codebase and establish:

1. How the wall and foundation are identified in the mesh.
2. How the static baseline is generated and stored.
3. How stresses are extracted and mapped to mesh cells.
4. How maximum principal stress is calculated.
5. How the existing cohesive cracking workflow operates.
6. How ENDO_FISS_EXP was previously attempted, if its implementation or configuration remains in the project.
7. How the nonlinear solver is called and how convergence is assessed.
8. How adaptive cracking iterations currently work.
9. Whether fracture state is preserved between iterations.
10. Whether foundation condensation or matrix reduction is responsible for any memory bottlenecks.
11. How the API receives and reports results.

Then decide on the most reliable implementation strategy.

Prioritise the simplest approach that can reliably achieve the required behaviour with the existing mesh and Code_Aster installation.

Do not start by replacing the entire FEA architecture.

Do not modify files yet. Produce a concise implementation plan identifying the files that need to change, the proposed algorithm, the known risks and the tests required.

Do not ask me to choose between fracture formulations unless there is a genuine decision that cannot be made from the codebase and available evidence.

PROMPT 2


Proceed with implementing the automated wall cracking workflow described in your investigation.

You have permission to modify the necessary project files.

Do not stop after producing recommendations. Implement the solution, run the available tests, diagnose failures and correct them.

CORE ALGORITHM

Implement an iterative stress-relief cracking process:

ITERATION 0 — BASELINE

- Use the existing valid static solution.
- Extract the stress tensor for concrete wall elements.
- Calculate maximum principal tensile stress.
- Record the initial maximum tensile stress and its location.
- Do not include foundation elements in the wall cracking assessment.

ITERATION 1 AND ONWARDS — CRACKING

For each iteration:

1. Identify wall regions exceeding the configured residual tensile-stress threshold.
2. Determine suitable crack locations and orientations using the available stress information.
3. Introduce cracks only into eligible wall regions.
4. Ensure that crack insertion does not damage or alter the foundation.
5. Re-establish mechanical equilibrium with the cracks present.
6. Extract the updated wall stress field.
7. Calculate the maximum principal tensile stress in the remaining concrete.
8. Record the change in stress, crack count, crack area, displacement and solver status.
9. Repeat while the stopping criteria permit further cracking.

STOPPING CRITERIA

The process must stop when one of the following occurs:

A. The maximum principal tensile stress in the assessed intact wall concrete is at or below the configured threshold.

B. No additional eligible crack locations can be generated.

C. The solver cannot obtain an acceptable converged equilibrium.

D. A configured maximum number of cracking iterations is reached.

E. A numerical or resource limit prevents further progress.

Do not treat B, C, D or E as successful achievement of the target.

CRACK IMPLEMENTATION

Investigate whether the existing cohesive facet implementation can be made reliable before replacing it.

If using cohesive interfaces:

- Preserve the appropriate traction-separation law.
- Use fracture energy consistently.
- Avoid arbitrary element deletion.
- Ensure crack faces can close under compression without interpenetration.
- Preserve appropriate shear/contact behaviour.
- Ensure that crack insertion and mesh subdivision do not unintentionally alter the fracture energy per unit crack area.
- Avoid introducing duplicate or overlapping cohesive interfaces.
- Ensure cracks are restricted to the wall volume.
- Handle boundaries and existing cracks safely.

The stress direction should inform crack orientation. Do not claim that the crack follows the principal direction if the actual implementation merely selects existing mesh facets.

If the existing mesh cannot support a mechanically defensible crack orientation, document that limitation and implement the best justified alternative supported by the current mesh.

STATE MANAGEMENT

This is particularly important.

Investigate and correct the existing problem where adaptive cracking rounds rebuild the model and restart the nonlinear solution without transferring the previous fracture state.

The workflow must not silently forget previously established cracks or their damage history.

If full state transfer is not supported by the selected Code_Aster formulation, use an explicit and mechanically justified alternative. Do not pretend that a newly generated model has inherited the old damage state.

WALL AND FOUNDATION SEPARATION

- Crack selection must use wall elements only.
- Crack insertion must use wall-wall internal facets only, unless a separately justified wall boundary treatment is required.
- Foundation elements must remain intact.
- Do not create cohesive elements inside the foundation.
- Do not alter the wall-foundation support definition as a side effect of wall cracking.

IMPLEMENTATION

- Integrate the process into the existing application architecture.
- Add a configurable residual tensile-stress threshold.
- Use consistent SI units internally.
- Keep material tensile strength and residual stress threshold as separate configuration parameters.
- Make the cracking iteration limit configurable.
- Make solver and iteration settings explicit rather than hidden constants.
- Preserve backward compatibility with existing static analysis configurations wherever practical.

Implement the workflow now. Run tests and fix issues rather than merely describing them.

PROMPT 3


Now perform a dedicated robustness and reliability pass on the automated cracking implementation.

The goal is to make the workflow capable of running unattended for a realistic 3D dam model.

Do not add new features unless they are needed for reliability.

1. MEMORY MANAGEMENT

Investigate all memory-intensive operations, particularly:

- Foundation condensation.
- Dense matrix construction.
- Matrix inversion and factorisation.
- Mesh duplication.
- Cohesive element generation.
- Stress-field extraction.
- Repeated Code_Aster model construction.

The previous project encountered a NumPy allocation of approximately 41.4 GiB while constructing a dense reduced matrix.

The machine used for development has:

- Intel Core i7-13620H.
- 32 GB RAM.
- Windows with WSL.

Do not allow the workflow to attempt impractical dense allocations without checking their estimated memory requirements first.

Prefer sparse matrix operations where appropriate.

If foundation condensation is optional, make sure the cracking workflow can run without it, while retaining the intended foundation support behaviour.

Do not silently replace the foundation with an arbitrary fixed boundary condition just to make the model run.

2. SOLVER CONVERGENCE

Investigate and improve:

- Load stepping.
- Nonlinear iteration limits.
- Convergence tolerances.
- Line search or other suitable nonlinear stabilisation.
- Cohesive penalty stiffness.
- Contact and crack closure behaviour.
- Solver diagnostics.

Do not solve convergence problems by arbitrarily reducing concrete strength, fracture energy or the configured residual stress threshold.

If a solve fails, capture the actual failure reason.

3. CRACK ITERATION MANAGEMENT

Ensure that each iteration:

- Has a unique identifier.
- Records the input model and crack state.
- Records the solver result and convergence status.
- Can be inspected independently.
- Does not corrupt the previous accepted iteration.
- Does not lose previously inserted cracks.
- Can recover cleanly after a failed iteration.

Avoid repeatedly inserting cracks into the same already-cracked location.

Avoid generating enormous numbers of cohesive patches from an uncontrolled selection of overloaded elements.

Introduce sensible limits and diagnostics for candidate facets and generated cohesive elements.

4. STRESS REDISTRIBUTION

Verify that stress screening is performed on the updated equilibrium solution after every accepted cracking iteration.

The reported residual maximum tensile stress must represent the remaining intact concrete wall, not an average that conceals local overstressed regions.

Report at least:

- Maximum principal tensile stress.
- Location of maximum stress.
- Number of wall elements above threshold.
- Number of new cracks introduced.
- Total crack area or another defensible crack-size measure.
- Maximum displacement.
- Solver convergence status.
- Iteration duration.
- Peak memory estimate where available.

5. FAILURE HANDLING

A failed iteration must not be reported as a successful cracking result.

If the solver fails, preserve the last accepted converged state and report:

- The last successful iteration.
- The stress threshold achieved.
- The remaining overstressed regions.
- The reason the next iteration failed.

Implement these improvements and run the available tests.

PROMPT 4

Perform the final validation, reporting and application integration for the automated wall cracking workflow.

Do not merely produce a validation plan. Implement the required tests and correct any defects found.

A. AUTOMATED TESTS

Create or extend tests covering:

1. Principal stress calculation:

   - Uniaxial tension.
   - Uniaxial compression.
   - Pure shear.
   - A rotated stress tensor.
2. Wall/foundation separation:

   - Confirm that only wall elements can be selected for cracking.
   - Confirm that no foundation elements are modified.
   - Confirm that the wall-foundation support definition remains unchanged.
3. Crack insertion:

   - Confirm that eligible wall facets are identified correctly.
   - Confirm that already-cracked facets are not inserted again.
   - Confirm that no duplicate cohesive elements are created.
   - Confirm that crack geometry remains valid.
4. Stress redistribution:

   - Use a small representative model where the expected mechanical response is understood.
   - Verify that the updated equilibrium solution is used for the next screening iteration.
   - Verify that the algorithm stops when the threshold is satisfied.
5. Failure handling:

   - Verify that a failed nonlinear solve does not overwrite the last accepted state.
   - Verify that a partial result cannot be labelled as a successful completed analysis.

B. FRACTURE VALIDATION

Create small benchmark models before attempting the full dam.

At minimum:

- A simple concrete tensile specimen.
- A specimen with a cohesive crack under tension.
- A crack closure case under compression.
- A case involving stress redistribution around a crack.

Check that the selected fracture formulation behaves consistently with its intended constitutive law.

Do not claim mesh independence unless a mesh refinement study has actually been performed.

C. FULL DAM VALIDATION

Use the existing valid static dam analysis as the reference.

Compare the baseline and cracked results under identical loading and support conditions.

Report:

- Initial maximum wall tensile stress.
- Final maximum wall tensile stress.
- Configured residual tensile-stress threshold.
- Number of cracking iterations.
- Total crack area or equivalent crack measure.
- Maximum displacement before and after cracking.
- Solver convergence for every iteration.
- Any remaining regions exceeding the threshold.
- Whether the target was actually achieved.

D. WPF AND API INTEGRATION

Expose the residual tensile-stress threshold as a user-configurable input in the WPF application.

Make the distinction clear between:

- Concrete tensile strength: the material parameter used by the fracture model.
- Maximum permitted residual tensile stress: the user's requested stopping criterion.

Do not use one value for both purposes.

Ensure that the API configuration, validation, execution and result reporting all support the new parameter.

E. REPORTING

Produce a clear machine-readable report and a human-readable summary.

The report must distinguish:

- Completed successfully: target achieved.
- Completed with remaining overstressed regions.
- Stopped because no further eligible cracks could be generated.
- Solver failure.
- Resource limit.
- User or iteration limit.

Do not report a successful engineering result merely because the program exited without an exception.

Complete the implementation, run the tests and provide a concise summary of changed files, test results, known limitations and the command needed to run the full dam analysis.


PROMPT 5

AUTONOMOUS DEVELOPMENT INSTRUCTIONS

I want you to operate as an autonomous engineering development agent for this task.

Do not treat this as a question-and-answer exercise. Your responsibility is to investigate, implement, test, debug and complete the requested functionality.

You are authorised to:

- Inspect the entire project.
- Modify relevant source files.
- Create and modify tests.
- Run Python scripts and Code_Aster jobs.
- Inspect solver logs and diagnose failures.
- Refactor existing code where necessary.
- Correct defects discovered during testing.
- Update configuration schemas and documentation.

WORKING RULES

1. Preserve the existing working static FEA workflow.
2. Make incremental changes rather than rewriting unrelated functionality.
3. Before significant changes, ensure the existing work is recoverable using version control.
4. Do not overwrite useful baseline results.
5. Do not introduce arbitrary engineering parameters to make a test pass.
6. Do not conceal failed tests or solver failures.
7. Do not claim that a feature works unless you have tested it.
8. If a test fails, investigate and fix the underlying problem before moving on.
9. Keep a record of important implementation decisions.
10. Continue working through the requested stages without asking for approval after every minor change.

If a genuine technical blocker requires a decision that cannot be resolved from the codebase, document the exact issue and the alternatives. Otherwise, make the most defensible engineering and software decision and continue.

At the end, provide:

- A summary of implemented functionality.
- Files modified.
- Tests executed and their results.
- Any remaining limitations.
- Instructions for running the full dam analysis.
- A clear statement of whether the configured residual tensile-stress target was achieved.

The priority is a working, repeatable and technically defensible automated cracking workflow, not a demonstration that a particular fracture formulation can be made to run.
