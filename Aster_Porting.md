# Code_Aster Porting Plan: Working Baseline Only

## Scope

Port the currently working dam model to Code_Aster before making any further
attempt at cracking, contact, cohesive interfaces, or concrete damage.

This work is a solver and results-porting exercise. It must reproduce the
existing linear-elastic model's geometry, material, supports, loads, and
observable response. No new failure mechanism is in scope.

## Explicitly Out of Scope

- Open-crack topology.
- Contact or friction at an interface.
- Cohesive elements or interface damage.
- XFEM, crack propagation, or bulk concrete damage.
- Crack-face water pressure.
- Changing the dam geometry while porting.
- Replacing the existing foundation support model with a rock domain.

These remain future studies after the Code_Aster elastic baseline is verified.

## Existing Model to Port

The active procedural generator is `Elmer/regions/combined_structure.py`.
At the time of writing it builds three bonded concrete regions:

| Current ID | Region | Role |
| --- | --- | --- |
| 1 | Plinth | Foundation/plinth concrete |
| 2 | Wall | Curved dam wall concrete |
| 3 | Haunch | Local plinth-to-wall transition concrete |

The mesh must remain conforming: adjacent regions share nodes at their bonded
interfaces. Use the mesh emitted by `build_combined_structure`; do not revive a
previous split center-wall/wedge mesh as part of this port.

## Inputs to Preserve

Read these from the existing files so that Elmer and Code_Aster use one source
of truth.

| Quantity | Source | Current value |
| --- | --- | --- |
| Concrete density | `Data/Concrete_Material_Properties.json` | 2400 kg/m3 |
| Young's modulus | same | 35 GPa |
| Poisson ratio | same | 0.20 |
| Tensile strength | same | 2 MPa, reporting limit only |
| Gravity | `Data/Env_Boundaries_And_Loads.json` | -9.81 m/s2 in Z |
| Reservoir level | same | 29 m |
| Peak upstream pressure | same | 284.49 kPa |
| Water density | `Data/Env_Boundaries_And_Loads.json` | 1000 kg/m3 |
| Tailwater head | same | 2 m |
| Crest overflow head | same | 2 m |

The current bedrock support is a distributed directional spring surrogate:

$$
k_x=226\times10^6,\quad k_y=263\times10^6,\quad k_z=376\times10^6
\quad \mathrm{N/m^3}
$$
These values are stored in `Data/Env_Boundaries_And_Loads.json`. Preserve this
support for baseline parity. Do not replace it with fixed support or an explicit
rock mass during this phase.

Import the common in-memory model through `structural_model`:
`build_combined_structure`, `audit_combined_structure`, and
`build_named_groups`. Solver-specific exporters and input writers must remain
outside this shared model API.

## Target Layout

```text
Aster/
  build_med_mesh.py       Export the current procedural mesh with names
  validate_mesh.py        Mesh and group-quality checks
  run_case.py             Case launcher and manifest writer
  cases/
    00_elastic_baseline.comm
  results/
    00_elastic_baseline/
      mesh.med
      result.med
      result.vtu
      summary.json
      run_manifest.json
```

MED is the authoritative Code_Aster mesh and result format. A VTU derivative
may be written for ParaView, but it must be generated from the MED result, not
used as the source of record.

## Work Packages

### 1. Freeze the Elmer Reference Case

Before creating Code_Aster files, capture the current working reference:

- generated Gmsh mesh and its audit report;
- Elmer SIF input;
- Elmer result VTU;
- node and element counts by region and element type;
- applied load resultants;
- foundation reactions;
- displacement at fixed named probe locations;
- volume-averaged stresses in named regions;
- hashes of source mesh generator, inputs, and result files.

Use fixed physical probe points or named nearest nodes at the crest, central
wall, upstream face, downstream toe, and plinth. Do not compare only the global
maximum stress: sharp corners and stress recovery can make that value
mesh-dependent.

### 2. Add Named Mesh Groups

The current mesh writer relies on numerical physical IDs. Code_Aster commands
must instead use durable group names. Extend the procedural export path to
create these MED groups:

| Group | Entity | Meaning |
| --- | --- | --- |
| `PLINTH` | Volume | Region 1 |
| `WALL` | Volume | Region 2 |
| `HAUNCH` | Volume | Region 3 |
| `FOUNDATION` | Face | Spring-supported base |
| `UPSTREAM` | Face | Reservoir pressure face |
| `DOWNSTREAM` | Face | Tailwater pressure face |
| `CREST` | Face | Overflow surcharge face |
| `LEFT_ABUTMENT` | Face | Outer side face, if exposed |
| `RIGHT_ABUTMENT` | Face | Outer side face, if exposed |

The exporter must assert that every required group is nonempty. It must also
write a JSON manifest recording group node/face/cell counts, region volumes,
exterior areas, element types, and coordinate bounds.

The canonical mapping is `Elmer/regions/model_groups.py`, re-exported through
`structural_model`. Do not infer a group from a normal vector after conversion
and do not rely on Gmsh/Elmer renumbered IDs.

### 3. Write and Validate the MED Mesh

Implement `Aster/build_med_mesh.py` using
`structural_model.build_combined_structure`. The exporter must preserve:

- coordinates in metres;
- hexahedra and triangular prisms without topology conversion;
- bonded node sharing between plinth, wall, and haunch;
- all named volume and face groups;
- the global coordinate convention: vertical is $Z$.

`Aster/validate_mesh.py` must fail the build if it finds repeated-node cells,
zero or negative volume cells, missing groups, empty groups, unpaired bonded
interfaces, or different cell/region counts from the source mesh.

Open the MED mesh in ParaVis/ParaView and inspect the named groups once before
proceeding. This is a visual supplement to the automated validation, not a
replacement for it.

### 4. Port the Linear-Elastic Static Case

Create `Aster/cases/00_elastic_baseline.comm` with this conceptual sequence:

1. `LIRE_MAILLAGE` reads `mesh.med`.
2. `AFFE_MODELE` assigns 3-D solid mechanics to `PLINTH`, `WALL`, and `HAUNCH`.
3. `DEFI_MATERIAU` and `AFFE_MATERIAU` assign the existing linear elastic
   concrete properties to all three volumes.
4. Mechanical loads apply gravity, upstream hydrostatic pressure, downstream
   tailwater, crest surcharge, and the existing foundation stiffness model.
5. `MECA_STATIQUE` performs the linear solve.
6. `CALC_CHAMP`, `POST_RELEVE_T`, and `IMPR_RESU` produce stresses,
   displacements, reactions, and a MED result.

Use elevation-dependent pressure functions, not pressures exported node by
node. For an upstream free surface at $z=0$:

$$
p(z)=\rho_w g\max(-z,0).
$$

Apply tailwater and crest surcharge using the same physical definitions as the
existing Elmer input.

### 5. Port the Foundation Support Without Changing Its Meaning

The existing support is not a fixed base. It is a directional distributed
spring law. Confirm the installed Code_Aster catalogue supports a face-based
equivalent with the required three directional stiffnesses.

If it does, use it and verify the reaction distribution. If it does not, stop
the parity port at this point and use a small, separately verified support
representation such as a thin calibrated elastic layer or discrete support
elements. Do not substitute `DDL_IMPO` fixed displacement constraints merely
to obtain a solvable case.

### 6. Produce Comparable Results

`Aster/run_case.py` should run the command file in a clean case directory and
write `run_manifest.json` containing:

- Code_Aster version and command file hash;
- mesh and group manifest hash;
- material/load input hashes;
- solver status and elapsed time;
- applied load resultants and foundation reactions;
- probe displacements;
- region-wise stress summaries;
- output paths.

Convert `result.med` to VTU only after the Code_Aster solve completes, keeping
the MED output untouched.

## Baseline Acceptance Criteria

The Code_Aster port is complete when the following agree with the frozen Elmer
reference within an agreed tolerance. Use 2% as the initial target for global
quantities; investigate rather than average away any larger discrepancy.

| Check | Required comparison |
| --- | --- |
| Mesh | Same node/cell counts by source region and element type |
| Geometry | Same bounds, volumes, interface areas, and external face areas |
| Loads | Same gravity, pressure, and surcharge resultants and directions |
| Support | Same resultant force/moment within tolerance |
| Response | Crest, crown, toe, and plinth probe displacements |
| Energy | Comparable linear strain energy |
| Stress | Region averages and fixed-offset local samples away from singularities |
| Visual | Same deformed shape and pressure direction |

Document any accepted differences caused by element formulation, stress
recovery, or result interpolation. Do not claim a stress difference is physical
until mesh and load parity have passed.

## Execution Order

1. Freeze the current Elmer reference artifacts and comparison probes.
2. Add named groups and export the current mesh to MED.
3. Run automated mesh validation and visual group review.
4. Create the Code_Aster material and external-load baseline.
5. Reproduce the distributed foundation support.
6. Solve the Code_Aster elastic baseline.
7. Generate the comparison report and resolve discrepancies.
8. Tag the verified elastic port as the new Code_Aster baseline.

## Copilot Task Prompts

Use one prompt at a time. Include the named acceptance check in the same prompt
and ask Copilot not to modify unrelated generated artifacts. Do not ask it to
implement cracking, contact, or damage while performing these tasks.

### 1. Freeze the Elmer baseline

```text
Read Aster_Porting.md and the active Elmer mesh/solver path. Create a
reproducible baseline manifest for the current working linear-elastic Elmer
case. Do not modify geometry or solve settings. Record input hashes, mesh node
and element counts by region/type, group/physical-face counts, coordinate
bounds, volume/area measures, output paths, and named displacement probes.
Add a validation command that fails if the reference result or mesh is missing.
```

Acceptance: the manifest can be regenerated and identifies the exact Elmer
mesh/result used for comparison.

### 2. Define the named mesh contract

```text
Read Aster_Porting.md and Elmer/regions/combined_structure.py. Add one
centralized Python mapping from the active mesh's current regions and exterior
boundaries to the named Code_Aster groups listed in Aster_Porting.md. Do not
change geometry, cell connectivity, material values, loads, or solver files.
Add tests or assertions that every required group is nonempty and that the
PLINTH, WALL, and HAUNCH volume groups reproduce the source region cell counts.
```

Acceptance: group names are generated from one source of truth and missing or
empty required groups cause a clear failure.

### 3. Export and validate MED

```text
Implement the Aster/build_med_mesh.py exporter described in Aster_Porting.md.
Reuse the active CombinedStructure generator and preserve coordinates, element
topology, bonded nodes, and named volume/face groups. Add
Aster/validate_mesh.py. Do not add crack or contact topology. Validate cell
counts, bounds, region volumes, exterior areas, required groups, repeated-node
cells, and nonpositive volumes. Report the exact commands needed to generate
and validate the MED mesh.
```

Acceptance: a MED mesh opens with the expected named groups and the validation
report matches the frozen Elmer mesh measures.

### 4. Establish Code_Aster installation facts

```text
Inspect the local Code_Aster installation and report its version, launcher,
catalogue location, MED support, and available commands for a 3-D linear
elastic static solve. Do not install or upgrade anything and do not edit model
files. Identify the documented mechanism for the current directional
distributed foundation spring surrogate; if unavailable, state that clearly
without substituting a fixed base.
```

Acceptance: the result names the installed commands and support-model options,
with sources from the local catalogue or official documentation.

### 5. Create the elastic command file

```text
Using Aster_Porting.md, create only Aster/cases/00_elastic_baseline.comm for
the current bonded MED mesh. Use 3-D linear elasticity, the existing concrete
properties, gravity, elevation-based upstream pressure, tailwater, crest
surcharge, and the verified equivalent of the current spring support. Do not
add contact, cracks, damage, XFEM, or nonlinear material laws. Include output
of displacements, stresses, principal stresses, and reactions in MED format.
Explain each Code_Aster command choice and cite the installed-version syntax.
```

Acceptance: the command file parses/runs for the installed Code_Aster version
and uses only the groups defined in the MED exporter.

### 6. Compare elastic results

```text
Implement Aster/postprocess_baseline.py to compare the completed Code_Aster
elastic result against the frozen Elmer reference. Compare load resultants,
foundation reactions, named probe displacements, strain energy where available,
and region-wise fixed-offset stress samples. Do not compare raw global nodal
stress maxima as acceptance criteria. Write a JSON and Markdown comparison
report with percentage differences and explicit pass/fail thresholds from
Aster_Porting.md.
```

Acceptance: the comparison is reproducible and distinguishes global equilibrium
errors from local stress-recovery differences.

### 7. Close the baseline port

```text
Review the completed baseline-port artifacts against Aster_Porting.md. Report
only concrete gaps that prevent declaring the Code_Aster elastic baseline
verified. Do not propose or implement cracking/contact work. If all acceptance
checks pass, update the baseline manifest with Code_Aster version, command-file
hash, mesh hash, result hash, and final comparison summary.
```

Acceptance: there is one versioned, reproducible, bonded linear-elastic
Code_Aster case with an auditable Elmer comparison.

## Deferred Follow-Up

Only after the elastic baseline is accepted should the project decide whether
to model a prescribed interface, contact closure, cohesive damage, or bulk
concrete cracking. Those will be new, versioned scenarios built from the
verified Code_Aster baseline; they are not part of this port.