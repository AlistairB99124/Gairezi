# Wedge / Plinth / Wall Junction — Stress Root-Cause Analysis

**Date:** 2026-09-07
**Status:** Root cause identified. Design decision required from the civil/structural
engineer (the client). No geometry change has been committed — the workspace is on the
validated baseline mesh.

---

## 1. The observed problem

- A sharp, localised stress spike at the downstream **wedge tip**, where the wall
  face, the wedge slope, and the plinth top converge to a single point.
  - Raw peak principal tension ≈ **10.7 MPa** at the corner element (centroid
    ≈ (79.75, −0.51, −25.66)).
  - The surrounding body field is far lower (base-bending gradient, ≈ 3.5–4 MPa at
    the base, decaying smoothly to ~0 at the crest).
- A **second, separate distortion + stress spike on the upstream face**, where there
  is no wedge at all. This was the key diagnostic clue (see §4).

The downstream spike does **not** converge under mesh refinement — refining only makes
the peak larger. That is the signature of a **true geometric stress singularity**
(sharp re-entrant corner), not a mesh-quality bug.

---

## 2. Root cause (downstream): a genuine 3-body sharp-corner singularity

At each wedge tip the wedge cross-section tapers to a **zero-width point** and three
materials meet at one vertex:

```
        wall face (downstream, r = 76 m)
           │\
           │ \  30° wedge slope
   (wall)  │  \
           │   \  (wedge)
           │____\
           │ plinth top
           ▼
     single shared vertex  ← acute re-entrant corner
```

In linear elasticity the stress at such a corner behaves as σ ∝ r^(−α) as r → 0.
The value *at* the corner is therefore **non-physical and mesh-dependent** — it is not
a real stress in the structure and cannot be "meshed away."

This is a property of the **geometry**, not of the solver or the element size. Any
conforming linear-element mesh of this exact sharp geometry will exhibit it.

---

## 3. Why every "mesh trick" failed

We attempted four local geometry/mesh patches to remove the corner. All were reverted
because each either made things worse or violated another constraint:

| Approach | Result |
|---|---|
| **Chamfer the wedge end** | Distorted the wall grid (wall split logic and wedge shared the tapering value). |
| **Trim the last ~1 m of wedge** | Created an abrupt **step/cliff** between the last full station and the wedge-free wall — visible deformed-shape distortion, *worse* than the original point. |
| **Single "key" tetrahedron at the crossing** | Degenerate (near-zero volume, ~1e-14) — the crossing cross-section is zero by definition. |
| **Clamp the wedge cross-section near the ends** | Mismatch between the wedge's held interior shape and the wall's true split elevation → inverted/overlapping cells. |
| **Decouple the wedge tips (unbonded, "not cemented")** | Introduced a near-rigid-body mode (floating, overlapping body) → iterative solver **stalled** (~5500 iterations, no convergence). |

**Lesson:** the singularity cannot be removed by local meshing. It requires a genuine
geometry change (a finite fillet/round, or a redesign of the junction) — which is a
design decision, not a meshing decision.

---

## 4. Root cause (upstream): the wall's vertical grading is coupled to the wedge

This is the deeper, non-obvious finding, and it explains the upstream distortion.

**The wall is meshed as a single column of nodes per station.** For each
`(station, level_index)` there is *one* `z_value`, applied across the **entire wall
thickness** (all `thickness_index`, from the upstream face to the downstream face).

The wedge needs the wall's downstream lower block to be re-graded to make room for the
wedge slope. Because that `z` is shared across the thickness, **raising the downstream
lower block also raises the upstream face** at the same station — even though there is
no wedge upstream.

Verified on the baseline mesh:

- At **all 403 stations**, the upstream and downstream faces share **> 95 % of their
  z-levels** — proving the grading is locked together through the thickness.
- The downstream face carries **145 collapsed (duplicate-z) level pairs** where the
  lower block is squeezed against the wedge — themselves a local stress-spike source.

So the upstream spike is **not** independent: it is the downstream wedge coupling
propagating through the shared wall column.

**Experimental confirmation:** when the wall was switched to a **uniform** base→crest
grading (independent of the wedge), the upstream face became **perfectly uniform**
(constant 0.4915 m vertical steps) and the upstream distortion disappeared. This
isolates the coupling as the cause.

---

## 5. The fundamental constraint (why we can't have all three)

With a structured, conforming, shared-node prism/hex mesh, you can have **any two** of
the following — but not all three at once:

1. **Uniform wall vertical grading** (clean upstream face, no distortion).
2. **Smooth, exact 30° wedge geometry** (no stepping/quantisation).
3. **Conforming shared-node bond** between wedge and wall (no contact/tie elements).

The wedge top lands on a wall level that varies continuously along the arc — measured
from **wall layer ≈ 0.17 at the tips** up to **≈ 6.92 at the crown** (with 0.5 m layers).
A prism ring needs a **constant node count**; a conforming bond to a uniform wall needs
the slope subdivisions to **equal the local wall-level count**. These are mathematically
irreconcilable for a smoothly-varying wedge — which is exactly why each patch produced
degenerate or non-conforming cells.

---

## 6. Options put to the client (design decision)

The meshing is sound; the open question is the **junction geometry**, which belongs to
the civil/structural engineer.

**Option A — Relax the strict 30°/60° rule to align with wall Z-levels** *(the user's
hypothesis, and our recommendation).*
Quantise the wedge so its top and slope breakpoints land exactly on the wall's uniform
Z-levels. The wedge becomes very slightly "stepped" (sub-element) instead of a
mathematically perfect 30° line, but the bond is fully conforming, the wall stays
uniform (upstream clean), and the singularity is removed by giving the corner a finite,
mesh-convergent form. **This is the smallest change that resolves both the upstream
distortion and the downstream singularity.** It requires the client to accept a
sub-element deviation from the nominal 30°/60° angles.

**Option B — Add a real geometric fillet/round at the junction.**
Give the corner a finite radius. Physically correct and convergent, but requires either
(a) a genuinely curved, high-order mesh (all-tetrahedral via Gmsh — breaks the
hexahedral wall/plinth bulk requirement), or (b) a faceted fillet, which we showed
cannot be applied cleanly because the wedge toe is a *shared plinth-top node* in the
bonded monolithic mesh.

**Option C — Keep the exact geometry, use a non-conforming (tied/contact) interface.**
Model the wedge as a separate body tied to a uniform wall with contact/constraint
equations. Geometrically exact and keeps the wall uniform, but adds real solver
complexity — and our decoupled-body experiment already showed the iterative solver
stalls on the resulting near-null modes.

**Option D — Accept the corner as a singularity and report around it.**
No geometry change. Report stress at a fixed physical distance from the corner (or an
averaged measure), treating the corner value as non-physical — standard practice for
singular FEM corners. Already implemented in `Elmer/analyze_stress.py`
(`--exclude-corner-radius-m`, default 2-element = 1.0 m radius; see
`stress_decisions.md`). Useful for characterisation, but does not *fix* the model.

---

## 7. Supporting artifacts

- `stress_decisions.md` — decision log for the fixed-distance exclusion reporting.
- `Elmer/analyze_stress.py` — radial corner-exclusion analysis (`--exclude-corner-radius-m`).
- `Elmer/build_curved_dam_geometry.py` — manual MSH writer; wall grading in
  `ordinary_wall_z` / wedge bond in `generate_curved_wall_mesh`.
- Baseline mesh (committed): 403 stations, 251,920 volume elements, 68,384 boundary
  faces; topology audit passes (zero-volume = 0, internal 2D boundaries = 0).
- Troubleshooting screenshots under `troubleshooting/`.

## 8. Bottom line for the client

- The downstream wedge-tip spike is a **real geometric singularity** — not a mesh bug.
  It will not converge with refinement.
- The upstream distortion is the **same root cause** (wall grading coupled to the wedge),
  confirmed by the fact that a uniform wall eliminates it.
- The clean, minimal fix (**Option A**) is to let the wedge align to the wall's Z-levels,
  relaxing the strict 30°/60° angle by a sub-element amount. This is a **design
  decision for the civil/structural engineer**, since it changes the nominal junction
  geometry.
