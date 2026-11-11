# Taking a breath: why the damaged dam does not crack the way we expected

Written 10 November 2026, after job `e50889ee…` (Aster run `20261110T065402Z-cca883c7`),
the first run with degraded elements **and** weak lift joints together.

> **Update, later the same day.** The failings below that could be fixed in the model and
> the reporting have been. The last section, "What was changed", lists them, says what has
> been verified and what has not, and gives the first numbers from the new plane survey.
> No dam-scale run has been made with the corrected model yet.

## The short answer

The run did not fail to find a crack because the damage was too mild. It failed because of
how we built the damage:

1. **The weak lift joints never took load.** Not one of the 1116 lift-joint cells initiated.
   The highest tension any of them saw was 17 kPa, which is 8.5 % of their 200 kPa strength.
2. **The displacement change is not cracking.** The damaged model moves 15.9 mm against
   6.6 mm intact, but it moves that much from the very first load step, before anything
   cracks. Cracking added 0.02 mm to the peak (and at most 0.37 mm anywhere on the wall).
   The extra movement comes from the stiffness we removed by hand: softer seat springs,
   softer elements, and a soft lift joint.
3. **The cracks that did form are the wrong kind, in the wrong place.** They are short
   cracks on vertical and inclined facets, mostly on the dry downstream side, so no water
   entered any of them. Meanwhile the highest stress moved to concrete just above the band,
   where there are no joints, so it could not crack there at all.
4. **The intact dam is not close to cracking under this load.** Its peak tension is
   0.87 MPa against a strength of 0.9 MPa, and that tension is a restraint stress that
   disappears when the base is allowed to move.

Underneath these sits one conceptual mistake: we have been asking "where is the highest
tensile stress, and what happens if we weaken it?" The dam-safety literature says that is
the wrong question for an arch dam. The right question is "what mechanism could let the
reservoir push part of the dam out of the way?" Our model cannot currently represent the
mechanism a weak lift joint would actually fail by, which is sliding under uplift.

The rest of this document gives the evidence, the mistakes ranked by how sure I am of them,
what I checked and ruled out, what I have not verified, and what to do next.

How to read the evidence labels:

- **Measured**: computed from this run's own output files.
- **Primary source**: I read the passage in the original document.
- **Secondary**: from a search summary; I could not open the original.
- **Inferred**: my reasoning from the above, not directly tested.

## What the last run actually did

Configuration of the run:

| Item | Value |
|---|---|
| Intact concrete | E 35 GPa, tensile strength 0.9 MPa, fracture energy 100 J/m² |
| Degraded elements | E 10 GPa, density 2100, tensile strength 0.5 MPa, fracture energy 150 J/m² |
| Lift joints | tensile strength 0.2 MPa, fracture energy 100 J/m², normal stiffness 2.2e9 Pa/m |
| Seat springs (wall on foundation) | scaled to 10 % (`support_degradation.interface_scale` 0.1; 3899 springs) |
| Selected band | 2520 wall cells and 2112 foundation cells, z −29 to −27 m, full wall thickness |
| Joints inserted | 4408 facets, all inside the band: 372 lift (160 m², at z = −28 m), 4036 "damaged", 0 intact |
| Loads | self-weight and reservoir scaled together from 0; base uplift off; crack water on |

Results (all **measured**):

| Quantity | Intact baseline | Damaged run |
|---|---|---|
| Peak principal tension in the wall | 0.87 MPa at design load, at the upstream heel | 3.44 MPa at load factor 0.85, on the downstream face at z = −26.5 m |
| Where that peak sits | inside the band | just above the band, in concrete with no joints |
| Peak displacement at load factor 0.85 | 6.62 mm | 15.90 mm |
| Same, predicted from the first step alone (no cracking yet) | n/a | 15.88 mm |
| Lift-joint cells initiated | n/a | 0 of 1116 |
| Highest lift-joint tension | n/a | 17 kPa (strength 200 kPa) |
| Largest lift-joint opening | n/a | 0.008 mm (elastic limit 0.091 mm) |
| "Damaged" joint cells initiated | n/a | 5315 of 12108, tension capped at 0.5 MPa |
| Fully separated joint cells | n/a | 2 |
| Largest opening anywhere | n/a | 0.146 mm (full separation needs 0.6 mm) |
| Initiated cells on the downstream half of the wall | n/a | 67 % |
| Crack faces wetted by the reservoir | n/a | 0 in all 35 load steps |
| Last converged load factor | n/a (linear solve) | 0.85 (Newton failed between 0.85 and 0.875) |

Two facts about the intact stress field matter for what follows (**measured**):

- The peak tension at the heel points about 42° off vertical (direction cosines 0.74
  vertical, 0.62 radial). The vertical stress in that cell is only 0.36 MPa.
- On the lift plane itself, the intact field puts 54 of the 372 facets above 200 kPa of
  vertical tension, up to 0.73 MPa, all in the upstream 2 m. So in the intact dam the lift
  plane *should* have opened at about 28 % of design load. In the damaged run it never got
  near. Something we changed removed that tension.

## The failings, ranked

### 1. The displacement change is our own softening, not cracking

**Evidence (measured).** At load factor 0.01, before any joint initiates, the damaged model
already deflects 2.4 times as much as the intact one. Scaling that first step linearly to
load factor 0.85 gives 15.88 mm. The actual final value, after 5315 joint cells initiated,
is 15.90 mm. At the crest node with the largest movement the three displacement components
match the elastic prediction to 0.02 mm. Over the whole wall the largest difference between
the two displacement fields is 0.37 mm, against 11.9 mm between the softened and the intact
fields (both from the replayed report; see "What was changed").

**What it means.** Before the solver takes a single nonlinear step we have already:

- cut the seat springs to a tenth,
- cut the band's Young's modulus from 35 to 10 GPa, and
- inserted a lift joint with a stiffness of 2.2e9 Pa/m, which is as compliant as 16 m of
  concrete.

Any of these increases deflection in a purely linear way. The report compares the damaged
run with the *intact* baseline, so it shows this softening as if it were the effect of
cracking. I have not measured how the 9.3 mm splits between the three; that needs the
control runs listed at the end. My expectation (**inferred**) is that the seat springs
dominate, because they act over the whole base.

**The mistake.** We changed several things at once and had no control run with the same
softening but no cracking. Without that, a displacement number cannot be attributed.

### 2. The lift joints were bypassed

**Evidence (measured).** Zero lift cells initiated. Their peak tension rose in proportion to
load and reached 17 kPa at load factor 0.85. The intact field predicted up to 730 kPa on the
same facets at full load.

There are two reasons, and both are things we chose.

**(a) The seat below the joint is softer than the joint, and it carries tension.** The
wall/foundation seat is a layer of ordinary linear springs (`K_T_D_L` in the command file).
A linear spring pulls back when stretched, so the seat resists tension without limit.
Scaled to 10 %, its vertical stiffness is about 3e8 N/m³, seven times softer than the lift
joint one metre above it. When the heel wants to lift, the seat stretches and the lift
joint does not need to open. Turning on `include_base_contact` together with
`interface_scale: 0.1` therefore protects the lift joint instead of loading it.

This contradicts the regulator's guidance directly (**primary source**, FERC Engineering
Guidelines ch. 11, §11-5.2.1):

> the dam/foundation interface shall be assumed to be incapable of resisting tension normal
> to it. Analyses that assume that the dam is "welded" to the foundation are not acceptable.
> This usually requires the use of a gap/friction contact at the dam/foundation contact.

FERC ch. 3 says the same for gravity dams: "The tensile strength of the rock-concrete
interface should be assumed to be 0."

**(b) The lift joint itself is far too soft.** `JOINT_MECA_RUPT` has no fracture-energy
input. Energy follows from strength, stiffness and the softening parameter `PENA_RUPTURE`.
To make the law dissipate the 100 J/m² we asked for, I set `PENA_RUPTURE` to its catalogue
maximum of 10 and derived the stiffness from the energy, which gave 2.2e9 Pa/m. Code_Aster's
own documents say this is the wrong way round:

- R7.01.25 §1.2.1 (**primary source**): the normal stiffness can be estimated from the
  joint filler and thickness as `K_n = E/e ≈ 10 GPa / 1 cm = 1e12 Pa/m`, and "in practice the
  value is taken close to `P_rupt ≈ 1`". We are 450 times softer and at ten times the
  softening parameter.
- U2.05.07 §4.1.2 (**primary source**): "In all cases, the initial stiffness of the joint
  elements must be greater than the stiffness of the neighbouring volume elements." Our
  neighbours are about 2.9e10 Pa/m (intact, E/h with h = 1.2 m) and 8e9 Pa/m (degraded).
  The lift joint is below both.

With the documented values (`K_n` 1e12, `P_rupt` 1, strength 0.2 MPa) the law dissipates
about 0.04 J/m². That is the honest reading of this law: it models a nearly brittle joint.
A fracture energy of 100 J/m² cannot be put into it without turning it into a soft elastic
layer, which is what we did.

**The mistake.** Both choices were mine, made while fixing earlier failures (the
`PENA_RUPTURE` catalogue cap, and the base-contact option). Each fix made the solver run
and each made the physics worse.

### 3. The tension that remains is horizontal, so the cracks are vertical and dry

**Evidence (measured).** All initiation is on the "damaged" class, which is every facet in
the band that is *not* near-horizontal. Two thirds of the initiated cells are in the
downstream half of the wall. In the intact field the largest hoop (along-the-arch) tension
is 0.58 MPa on the downstream face at the base. In the damaged run the peak traction on
these facets at load factor 0.225 is 396 kPa, which scales to about 1.8 MPa at full load,
twice the intact peak anywhere.

**What it means (inferred).** Releasing the base makes the bottom strip of the wall span
sideways between the valley sides instead of being held by the foundation. That bends it
horizontally and puts the downstream face in hoop tension. Hoop tension opens vertical,
radial cracks. Those are on the dry side, which is why the crack-water coupling never
activated. They are also crossed by arch compression higher up, so they stay short: the
largest opening is a quarter of what full separation needs.

**The mistake.** We weakened horizontal planes and then created a stress state whose
tension is not across horizontal planes.

### 4. The model cannot crack where the stress goes

**Evidence (measured).** The peak principal tension in the damaged run is 3.44 MPa, nearly
four times the intact concrete's 0.9 MPa strength, in the first element layer above the
band. There are no cohesive joints there, so that concrete is linear elastic and
unbreakable. All 4408 joints are inside the band; zero are "intact"-class joints outside it.

**What the source says (primary source, Code_Aster U2.05.07 §3.1 and §3.5).** Cohesive
models "represent a crack on a path given a priori. These models account for the evolution
but not the direction of propagation." The same note advises against placing cohesive
elements everywhere, because "the results for the crack path are strongly dependent on the
mesh."

**What it means.** A cohesive model answers "does a crack grow along the surface I gave
it?" It does not answer "where does the dam crack?" Our automatic selector picks the band
from the *intact* stress field, then we change the stiffness, the stress field moves, and
the joints are no longer where the tension is.

### 5. Peak tensile stress is the wrong target for an arch dam

This is the conceptual failing behind the others.

**Evidence (measured).** The intact dam peaks at 0.87 MPa, below its own strength. The
facet check finds zero critical facets. Only 2.9 % of the wall volume is above 0.5 MPa.

**What the source says (primary source, FERC ch. 11).**

- §11-3.7.2: tensile strength "has little significance in the performance of an arch dam
  … the tensile strength of an arch dam is usually limited by the ability of horizontal
  lift joints, vertical contraction joints, and pre-existing cracks to resist tension", and
  "If a failure mode is contingent upon high tensile strength, there is very likely a
  problem."
- §11-4.3, on restraint stresses: "Thermal cracking will almost certainly happen at the
  dam/foundation interface … they are not follower forces. As soon as cracking or shifting
  occurs, thermal stress is relieved … Because forces resulting from temperature loading are
  not follower forces, they cannot drive a failure mechanism."
- §11-1.4.1: a failure mode stated as "Overstressing of dam causes failure" is "not
  acceptable". A failure mode must be a chain of physical events ending in loss of the
  reservoir.
- §11-4.3 again: "Often fictitious high tensile stresses were indicated, especially in the
  case of linear models. In the absence of a failure mode, these stresses were
  meaningless."

**What it means.** The heel tension in our intact model is a restraint stress: it exists
because the base is held. Weaken or release the base and it goes away, and the arch
carries the load in compression. That is what an arch dam is for, and it is what our run
showed. Base cracking in an arch dam is expected and, by itself, benign. Your brother's
point about poorly prepared lift surfaces is well supported (FERC says lift joints, not
parent concrete, set the tensile capacity, and that old dams can have "drastically lower"
joint strength). What does not follow is that a weak lift joint shows up as a tensile
crack with a large deflection under normal static load.

A weak horizontal joint matters when it lets something *slide*. FERC ch. 11 §11-5.2.1
requires that "Uplift forces acting on potential failure surfaces cannot be ignored" and
that horizontal cracks under tension "must be included in the model" as gap/friction
elements. FERC ch. 3 assesses horizontal planes by shear-friction sliding, says uplift
"should also be applied within any cracks within the dam", takes the tensile strength of
the rock-concrete interface as zero, and warns that lift-joint strength may be well below
the parent concrete's.

### 6. The physics that would let a weak lift joint fail is switched off

Each item is a property of the current model (**measured** from the command file and
report unless noted).

| Missing or wrong | Why it matters |
|---|---|
| Base uplift is `none` | Uplift is the follower force that drives sliding. FERC: it "cannot be ignored". |
| Lift joints use `JOINT_MECA_RUPT` | R7.01.25 §1.2.1 says this law describes real joints "at small displacements, as long as the sliding regime is not reached". It has no Coulomb slip limit, so a broken lift joint cannot slide, and sliding is the failure mode. `JOINT_MECA_FROT` or `JOINT_MECA_ENDO` exist for it. |
| Seat is linear springs | Carries tension; see failing 2. |
| No vertical contraction joints | FERC §11-1.4.2: "Almost all arch dams consist of cantilever monoliths separated by vertical joints … Analyses that fail to account for jointing and known pre-existing cracks are often misleading". Our wall is one welded shell, which is the most redundant structure possible. |
| Lift plane is a 160 m² patch at the valley bottom | It sits 1 m above the seat in the deepest, most confined part of the valley, under the most self-weight. A pour plane runs bank to bank at one elevation. |
| Self-weight applied to the finished shell | Real arch dams are built as free-standing cantilevers and grouted afterwards, so self-weight is not carried by arch action. USACE EM 1110-2-2201 is widely cited for this (**secondary**: I could not open the manual). |
| No temperature load | FERC calls thermal loading "often the largest contributor to stress" and says thermal cracking "will almost certainly happen at the dam/foundation interface". Our only loads are weight and water, under which this dam is almost entirely in compression. |

### 7. "No static equilibrium above 85 %" is not established

**Evidence (measured).** Newton stopped converging between load factors 0.85 and 0.875.
The report then says "the structure cannot carry that load: the configured damage brings
the dam to failure before full reservoir load."

**What the source says (primary source, Code_Aster U2.05.07 §4.3).** For cohesive
calculations "the use of load control (pilotage) is essential" for robustness. It is "often
necessary, even when the global response has no snap-back", because of "local snap-backs
linked to abrupt opening at element level". Line search "is not the best suited to pass the
instabilities". We use line search, no pilotage, and a relaxed residual tolerance of 36 kN.

FERC ch. 11 does say non-convergence indicates failure, but in an example where an
independent hand calculation of a sliding wedge reached the same conclusion.

**What it means.** The stop at 0.875 may be a real limit or may be the solver failing to
step past a local softening event. We have no independent check, and at the last converged
state the largest opening is 0.146 mm with two separated cells, which does not look like a
structure about to collapse. The message overclaims. I wrote that message.

### 8. The "damaged" material is tougher than the intact one in the way that matters

**Evidence (measured).** Full separation needs an opening of `2 Gf / f_t`: 0.22 mm for
intact concrete, 0.6 mm for the degraded material, 1.0 mm for the lift joints. Lower
strength with equal or higher fracture energy gives a joint that starts yielding earlier
but holds on for longer. The app counts a crack as confirmed at full separation, which the
weakened joints are the least able to reach.

**What it means.** A poorly bonded surface should have low strength *and* low fracture
energy. The app's own guidance text says not to raise fracture energy when lowering
strength; the current configuration does.

### 9. We verified the software, not the model

The test suites I ran after each change (small column fixtures, including real Code_Aster
runs) show that the code builds the case it claims to build and that the solver accepts it.
They do not show that the case answers the engineering question. When I reported the
combined mode as "implemented and verified end to end", that was true of the software only.
None of the checks in this document existed before today, and the first two take seconds
to run on any finished job.

## What I checked and ruled out

- **A stress singularity at the heel.** I expected the 0.87 MPa peak to be a mesh artefact
  at the wall/foundation corner. It is not a sharp spike: the layer peaks I sampled over
  the bottom 4 m are all between 0.81 and 0.87 MPa, and the Elmer report's "away from sharp
  edges" measure gives 0.82 MPa. The stress is real for this model; it is just low.
- **Cohesive zone too coarse for the mesh.** U2.05.07 asks for four or five cohesive
  elements in the cohesive zone. Estimated zone lengths are about 4 m (intact), 6 m
  (degraded) and far longer for the lift joint, against 1.2 m facets. Adequate.
- **Lift facets wrongly identified.** The 372 facets are horizontal, on one plane, and the
  intact field does put tension on them. The selection is doing what it says.

## What I have not verified

- How the 9.3 mm of extra deflection splits between seat springs, soft elements and the
  lift joint.
- Whether the stop at load factor 0.875 is physical.
- Whether linear tetrahedra with six elements through the wall thickness give an accurate
  bending response. There is no mesh-convergence or element-order study for this model.
- The staged-construction claim against the USACE manual itself.
- Whether the configured seat stiffness (1.25e9 N/m³ horizontal, 3e9 vertical, before
  scaling) reflects the real foundation. Even unscaled, the vertical value is as compliant
  as 12 m of concrete.

## What to do next

In order. The first step is a decision, the second is cheap and removes most of the
guesswork, and the rest depend on both.

1. **Choose the failure mode to test.** For a poorly prepared lift surface the credible
   chain is: the joint has little or no tensile strength; the upstream edge opens under
   reservoir load; water pressure enters the opening and acts as uplift on the plane; the
   normal force on the plane drops; the block above slides or the arch loses support along
   that elevation. The question for the model becomes "what is the sliding margin on this
   plane, and how far does it move?", not "does it crack?"

2. **Run controls so results can be attributed.** One change per run, all compared at the
   same load:
   - intact (exists);
   - seat springs at 10 % only, linear;
   - degraded element modulus only, linear;
   - joints inserted with intact properties and nothing softened (should reproduce intact);
   - lift joints only, seat untouched.

   The app should then report cracking against the matching softened-but-uncracked control,
   never against the intact baseline.

3. **Stop softening the seat when testing a lift joint.** Either leave the seat alone or
   model it as FERC requires: no tension, with friction.

4. **Give the lift joint a physical stiffness.** At least as stiff as the neighbouring
   elements, of the order R7.01.25 suggests, with `PENA_RUPTURE` near 1. Accept that this law
   is then nearly brittle and drop the fracture-energy input for it, or use a law that
   takes fracture energy directly.

5. **Make sliding and uplift possible.** A friction law on the lift plane
   (`JOINT_MECA_FROT` or `JOINT_MECA_ENDO`), water pressure in the opened part, and base
   uplift on.

6. **Put the weak plane where a pour plane would be.** One elevation, bank to bank, chosen
   by vertical stress on horizontal planes, not by peak principal stress. For water to
   enter, the tension has to be at the upstream face. The intact field's largest vertical
   tension (0.82 MPa, near z = −15.5 m) is on the downstream face, so that elevation would
   open dry; the z = −28 m plane does have its tension upstream. Which elevation to test
   should come out of a map of vertical stress by elevation and face, which I have not
   made.

7. **Do not report non-convergence as failure** until load control is in place or the
   mechanism is confirmed by an independent force balance on the plane.

8. **Then consider what is still missing**: vertical contraction joints, a temperature
   load case, and an element-order check.

Steps 3 to 6 change the model's physics and its results. They are a new plan, not a patch,
and should be agreed before any of it is built.

## What was changed

You agreed to the plan above, with the failure mode from step 1: a weak pour plane that
opens, takes water and slides. This section records what was built, in the order of the
steps, then what is verified and what is not.

### Changes

| Step | Change |
|---|---|
| 2. Attribution | The report now splits the displacement change into elastic softening and cracking (`displacement_attribution`), using the first load step as the softened-but-uncracked control, and lists the softening sources. It reports each joint class separately (`joint_classes`), so a bypassed plane is visible, and it counts wall cells that exceed their strength but have no joint to crack along (`unjointed_overstress`). The app's results view shows these as sentences. |
| 3. Seat | `include_base_contact` no longer requires a softened seat. Validation warns when lift joints and `support_degradation.interface_scale` below 1 are combined. `config.json` no longer softens the seat. |
| 4. Joint stiffness | The friction law uses the mesh penalty. The rupture law is refused, at element selection, when its fracture energy would make the joint softer than the elements beside it; the message gives the largest energy it can carry. Your last run's settings (0.2 MPa, 100 J/m²) are refused: the limit there is about 6.6 J/m². |
| 5. Sliding and uplift | New lift law `JOINT_MECA_FROT` (Mohr-Coulomb; needs `friction_coefficient`). New `uplift: linear_upstream_to_downstream`: pore pressure inside the plane on closed faces, full head on wet faces. The report gives the plane's shear demand against its friction capacity per load step (`lift_plane_sliding`). Friction runs apply self-weight before the water. |
| 6. Plane placement | New `plane_elevation_m` puts the plane at a pour elevation, bank to bank. With lift joints the band always sits on one mesh plane, so it is exactly two element layers. Element selection writes `lift_plane_survey`: area, vertical stress, upstream tension and the shear-to-compression ratio of every horizontal plane. |
| 7. Non-convergence | The capacity message no longer says the dam fails. It says the solver stopped, what state it had reached, and that a capacity claim needs supporting evidence. `capacity_assessment.established` is false. |
| Presets | `cracked_config.json`: the same dam as `config.json` with a frictional plane at z = −25 m (friction 0.7, tensile strength 20 kPa), uplift and crack water on, elements intact, seat untouched. The app has a **Load Cracked** button beside **Load Default**. A test keeps the two files identical outside `damage_selection`. |

Three related things changed with them:

- `config.json`: the default lift-joint fracture energy went from 10 to 5 J/m² (the old
  value fails the stiffness rule on this mesh), and `support_degradation` was removed.
- The sweep scenarios now pair each strength factor with an energy factor of that factor
  squared or less. A strength cut alone would soften a rupture joint until it is refused.
- Both joint settings and degraded elements remain available together, as you asked. In
  the app the degraded elements can now be switched off when lift joints are on, which is
  how the cracked preset keeps the joint's effect separate from elastic softening.

### What the plane survey shows

This is the map of vertical stress by elevation that step 6 said was missing. It is from
the intact baseline of job `e50889ee…` (**measured**; that job's wall has a thickness step,
so the numbers are indicative for `config.json`, whose wall is uniform).

| Plane z (m) | Area (m²) | Peak upstream vertical tension (MPa) | Compression N (MN) | Shear V (MN) | V / N |
|---|---|---|---|---|---|
| −28 | 160 | 0.58 | 68 | 62 | 0.90 |
| −26 | 259 | 0.40 | 106 | 84 | 0.79 |
| −25 | 290 | 0.31 | 116 | 86 | 0.74 |
| −24 | 326 | 0.20 | 127 | 89 | 0.70 |
| −22 | 388 | 0.01 | 140 | 90 | 0.64 |
| −19 | 384 | 0.01 | 148 | 73 | 0.49 |
| −15 | 456 | none | 144 | 53 | 0.37 |
| −10 | 537 | none | 121 | 26 | 0.21 |
| −5 | 635 | 0.03 | 84 | 12 | 0.14 |

V / N is the friction coefficient a plane would need to carry its shear by friction alone,
with no uplift. A poorly bonded lift surface offers about 0.5 to 0.7. So in the intact dam
the planes below about z = −24 m already carry more shear than friction alone could hold,
and they are also the ones with vertical tension at the upstream edge, where water gets
in. Higher planes have margin. That is why the cracked preset uses z = −25 m, and it is a
different answer from "the most stressed element".

For that plane, a hand estimate (**inferred**, not a solver result): friction 0.7 on
116 MN of compression gives 81 MN, plus 4 MN of adhesion, against 86 MN of shear. Uplift
from a 28 m reservoir head upstream to none downstream removes roughly 40 MN of
compression, leaving a capacity near 57 MN. The plane cannot hold its intact shear, so it
should slide and pass about 30 MN to arch action in the wall above. Whether the arch
carries that, and what it does to displacement, is what the run will show.

### Vertical crack path (default E and loads)

- **Why.** The run with only horizontal lift joints opened 418 faces to a peak of
  0.32 mm, against the engineer's 2 mm through-crack opening. The wall stayed a stiff
  cantilever (crest displacement 0.0094 m against 0.1 m). A horizontal joint cannot
  release the bending of a vertical crack.
- **What changed.** A vertical joint set (`damage_selection.vertical_joint`) is now
  placed on the radial vertical plane through the intact tensile peak. The plane contains
  the z-axis and the peak's radial direction, so it uses both Y and Z. The band defaults
  to one mean facet edge, and facets within 15° of that plane are selected. Lift and
  damaged faces are excluded.
- **Calibration, stated plainly.** Fracture energy is 150 J/m², taken from the realistic
  concrete range of about 50–500 N/m. The tensile strength is then set to
  `f_t = 2·G_F / w_c = 150 kPa` so that a 2 mm opening is reached. Real concrete is about
  1–3 MPa, so this strength is roughly 1 % of it. It is a tuning value, not a material
  property. The number to report is the crack opening, not the strength.
- **Not verified.** No full dam-scale Code_Aster run has been done with the vertical set.
  Whether 150 kPa gives 2 mm and 0.1 m crest displacement is unknown until it runs.
- **Bug found in the first vertical run, now fixed.** The run aborted after 16 s with
  `VERTICAL_JOINTS not in mesh`. `prepare()` saved the split arrays before the vertical
  mask was added, so the MED converter never saw the group. It skips empty groups, but
  `case.comm` still referenced it. The fix saves the arrays again after the mask is set
  (`save_split_arrays`). Checked on that job's inputs: the saved arrays hold 60 vertical
  faces, and the converted MED lists `VERTICAL_JOINTS`. A unit test covers the save.

### Verified

- **The frictional law and the uplift, through Code_Aster.** A fixture with a full
  horizontal plane under self-weight and pore pressure only: the plane's compression
  equals weight minus load factor times uplift to within 2 %, at each load step.
- **The stiff rupture law, through Code_Aster.** The two existing lift fixtures still
  crack and converge with a joint stiffer than its neighbours.
- **The new report on your last run.** Replaying the post-processing on a copy of the run
  gives: lift joints at 8.9 % of their strength with none initiated; the softened inputs
  changing the displacement field by up to 11.9 mm before any cracking, and cracking by
  up to 0.37 mm; 2736 wall cells (397 m³) above the 0.9 MPa strength with no joint, up to
  3.44 MPa. These match what I found by hand.
- **Plane selection on your mesh.** With the cracked preset's settings on that job's
  baseline: plane at z = −25 m, 672 facets, 290 m², through the full wall thickness.
- **Tests.** Gairezi (Aster, Elmer, sweep): 127 pass. FEniCSx under WSL: 65 pass. API:
  58 pass. The desktop app builds, and its lift-joint, cracked-preset and
  support-degradation checks pass.

### Not verified

- **No dam-scale run exists with the corrected model.** I have not shown that the cracked
  preset converges, slides, or changes the displacement. That run takes hours and is
  yours to start. The estimate above says what to expect, not what happens.
  - The first attempt (job `976a0300…`, 10 Nov 2026) says nothing either way. Windows
    Update installed WSL 3.0.5.0 during the run and shut the Linux session down at
    14:16:41, which killed Code_Aster and FEniCSx within seconds of each other. Code_Aster
    had converged every load step up to 17.5 % (about 50 s each, one hydraulic pass, 780
    plane faces under uplift, nothing open or wet, shear mobilisation rising from 0.09 to
    0.24) and was part-way through 20 %. FEniCSx was still bisecting its first openings
    at 3.7 %. Neither log contains a solver error. A run killed this way is now reported
    as terminated from outside instead of "stopped unexpectedly".
- **Friction-limit behaviour.** The Code_Aster fixture checks a plane that sticks. I did
  not build one that slides to its friction limit; the capacity arithmetic is unit-tested
  on synthetic numbers.
- **Convergence of a sliding plane at dam scale.** Friction runs have stalled before in
  this project. If it stops, the report now says so without calling it failure, and the
  sliding table will show how far the plane had mobilised.
- Unrelated test failures remain, from causes outside this work: `Aster.test_aster_tooling`
  needs a generated mesh folder that is not present; the desktop test runner stops at an
  API connection-settings check (environment variables overriding the settings file); and
  the older `tests.test_three_body_jobs` suite in the API fails because the committed
  `config.json` uses `bedrock_analysis.mode: elastic_layer`, which its fixtures do not
  expect. I did not fix these.

### Still open from the list

- **The seat is still a linear spring layer that carries tension.** Replacing it with a
  no-tension, frictional contact changes the baseline of all three solvers and needs its
  own verification. Base uplift stays off for the same reason: a seat that cannot open
  cannot respond to it. The report does not yet measure how much tension the seat carries.
- **Load control** for cohesive runs (step 7's proper fix) is not built.
- **Step 8** is untouched: vertical contraction joints, a temperature load case and an
  element-order check.
- **JOINT_MECA_FROT keeps its tensile capacity after opening.** `JOINT_MECA_ENDO` loses it
  and would be the better law; it is parameterised through hardening moduli with coupled
  bounds and was not attempted.

### To run it

Restart the API and rebuild the desktop app, then **Load Cracked** and submit. The API
process running under the Visual Studio debugger was not answering when I checked, so I
left it alone. In the results, read the findings in this order: did the lift joints take
load (`joint_classes`), how far did the plane mobilise (`lift_plane_sliding`), and how
much of the displacement change is the joints (`displacement_attribution`; for a friction
run with nothing else softened, the whole change from the intact dam is theirs).

## Sources

Primary, read in the original:

- FERC, *Engineering Guidelines for the Evaluation of Hydropower Projects*, Chapter 11,
  Arch Dams. <https://www.ferc.gov/sites/default/files/2020-04/chap11.pdf>
  Sections 11-1.4, 11-3.7.2, 11-4.3, 11-5.2.1, 11-2 (sliding and the non-convergence example).
- FERC, same guidelines, Chapter 3, Gravity Dams.
  <https://www.ferc.gov/sites/default/files/2020-04/chap3.pdf>
  Tensile strength of the rock-concrete interface and of lift joints; cracked-base analysis.
- Code_Aster U2.05.07, *Notice d'utilisation des modèles de zones cohésives* (rev. 2021).
  <https://code-aster.org/doc/default/fr/man_u/u2/u2.05.07.pdf>
  Sections 3.1, 3.3, 3.5, 4.1.2, 4.3.
- Code_Aster R7.01.25, *Lois de comportement des joints des barrages* (rev. 2024).
  <https://code-aster.org/doc/default/fr/man_r/r7/r7.01.25.pdf>
  Section 1.2.1.

Secondary, from search summaries only (originals not opened):

- USACE EM 1110-2-2201, *Arch Dam Design* (1994), §4-3c: dead load and construction
  sequence. The publisher's site refused the download.
- Kölnbrein dam (Austria, 1978): heel cracking in a wide valley, with uplift entering the
  crack and load shifting to arch action. Relevant as the best-known real case of the
  mechanism in step 1; I have not read the papers.
- Wieland, "Investigating stress concentration" and "Tensile stresses in concrete dams":
  on mesh-dependent heel stresses. Not opened; and the singularity turned out not to be
  our problem.

The quotations from the two Code_Aster documents are my translations from the French.

Run data used: `aster_cohesive_report.json`, `run/aster_cohesive_steps.npz`,
`run/aster_cohesive_steps_displacement.npz`, `three_body_mesh_arrays.npz`, `case.comm`
and the job's Aster elastic baseline, all under the job directory in
`%LOCALAPPDATA%\Lamb\DamAnalysisWorkbench\jobs\e50889ee2df24434ac78a04742100284`.
