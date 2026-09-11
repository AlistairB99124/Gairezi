# Corner Element-Size Rule

## Intent

Use the global mesh control as the normal element size, while refining only
wall and wedge corner regions where the available vertical height is too small
to fit a normal global element.

Let:

- `hz` = the available height between the local upper and lower boundaries,
	in metres;
- `es` = the target edge size for the local element, in metres;
- `global_es` = `Global Element Size` from
	`Data/Computational_Grid_Controls.json`.

The rule is:

```text
if hz <= 0.5 m:
		es = 0.1 m
else:
		es = 0.5 m
```

The implementation must use the configured global value for the normal case;
the literal `0.5 m` above describes the current control value and should not
replace the configuration lookup. The selected `es` applies in all local
element directions: X, Y/chainage, and Z. Therefore a refined corner target is
approximately `0.1 m x 0.1 m x 0.1 m`, subject to curved geometry and the
available boundary shape.

## Scope

Apply this rule only to the wall and wedge corner regions. The rest of the
wall, plinth, and wedge use `global_es`. `Local Corner Refinement` is not a
separate override for this rule.

## Topology Requirements

- Preserve the exact wedge start/interface elevation at `z = -25 m`.
- Preserve the wedge geometry defined by the configured horizontal and vertical
	ratio dimensions.
- Fill wall boundary gaps with valid hexahedra where the local geometry allows
	them.
- Use a geometry-compatible transition element when a triangular wedge tip
	cannot be represented by a non-degenerate hexahedron.
- Reject zero-volume elements and internal duplicate boundary faces during
	validation.
