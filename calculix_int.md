# CalculiX: considered and set aside

Written 11 November 2026. CalculiX was considered as one more option for crack simulation
after the FEniCSx and Code_Aster attempts. It was **not installed and not used**. This note
records why, so the question does not have to be researched again.

## Why it was set aside

| Capability needed for dam cracking | CalculiX | Code_Aster (already installed) |
|---|---|---|
| Joint or cohesive elements with a strength and friction law | No such element in the element list | Yes: dam joint laws `JOINT_MECA_RUPT`, `JOINT_MECA_FROT` |
| Built-in crack propagation | Fatigue only: growth per load cycle from the stress field of the uncracked structure. It gives no opening width. | Not needed; joints open directly |
| Contact that can open and slide | Yes (penalty contact with friction) | Yes |
| Concrete damage material | None built in | Several |

Sources: the CalculiX capabilities and new-features pages for version 2.23 on dhondt.de,
and the "Crack propagation" section of the CalculiX manual (read in its 2.18 edition).

CalculiX could only have repeated what Code_Aster already does with contact, and with
fewer tools. The cracking work therefore went back to Code_Aster and its own dam-joint
method, which is recorded in [aster_dam_baseline.md](./aster_dam_baseline.md).

## If it is ever revisited

- WSL (Ubuntu 26.04) offers `calculix-ccx` 2.21 through `apt`; the official site is at 2.23.
  Installing through `apt` needs `sudo`.
- `gmsh` and `meshio` are already present in WSL and can write CalculiX input meshes.
- The dam's mesh, interface springs, bedrock support and loads all map onto standard
  CalculiX keywords. This mapping was planned but not built or tested.

## Not verified

- Whether CalculiX has any arc-length or opening control was not confirmed in the manual.
- The manual for the packaged version 2.21 was not read; version 2.18 and the 2.23 feature
  pages were.