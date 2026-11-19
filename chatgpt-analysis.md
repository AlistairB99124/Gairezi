
I've gone through both files, and I think I've found a much more fundamental issue than the foundation simply dominating the stress calculation.

**Your current Code_Aster model is modelling opening and sliding at the dam–foundation joint. It is not actually modelling cracking or damage within the concrete dam wall.**

That distinction explains quite a lot of what you're seeing.

There is also a problem with how the final wall stress appears to be extracted: the code calculates stress for the `CONCRETE` group, but the subsequent extraction does not explicitly restrict the result to dam-wall nodes. If your foundation is also in that group, its stresses can influence the reported maximum.

## **1. What your results actually show**

At full reservoir level, your results are:

| Measure                    | Dry joint   | Water in joint |
| -------------------------- | ----------- | -------------- |
| Maximum tensile stress     | 2.422 MPa   | 2.588 MPa      |
| Maximum compressive stress | −5.731 MPa | −5.691 MPa    |
| Maximum wall displacement  | 7.63 mm     | 7.86 mm        |
| Opened joint area          | 374.05 m²  | 408.78 m²     |
| Opened share of contact    | 46.8%       | 51.1%          |
| Largest joint opening      | 1.124 mm    | 1.308 mm       |
| Damaged concrete area      | 0 m²       | 0 m²          |
| Broken concrete area       | 0 m²       | 0 m²          |

The elastic baseline has a maximum tensile stress of approximately **1.379 MPa** and a maximum compressive stress of **−3.015 MPa**.&#x20;

joint_opening_report.json

So your tensile stress has increased substantially:

- Dry joint: approximately 76% higher than baseline.
- Water in joint: approximately 88% higher than baseline.

But the most revealing figures are `damaged_area_m2 = 0` and `broken_area_m2 = 0`.

Your concrete is still elastic. You're opening and sliding the interface, not propagating cracks through the dam.

## **2. Your suspicion about the foundation is justified**

I found something important in `case.comm`.

Your model assigns:

```
AFFE=(    _F(GROUP_MA='CONCRETE', MATER=CONCRETE_MAT),    _F(GROUP_MA='JOINTS', MATER=JOINT_MAT),)
```

And your stress calculation is:

```
RESU = CALC_CHAMP(    reuse=RESU,    RESULTAT=RESU,    GROUP_MA='CONCRETE',    CONTRAINTE=('SIGM_NOEU',),    INST=wanted)
```

The subsequent extraction uses:

```
extra["stress_pa"] = nodal(    RESU.getField('SIGM_NOEU', indexes[-1]),    STRESS_COMPONENTS)
```

The `nodal()` function extracts values across the entire mesh node list. It does not filter for dam-wall nodes.

This means:

- If the wall and foundation are both in `CONCRETE`, the reported stress can include both.
- If they share nodes, recovered nodal stresses can also be influenced by contributions from adjacent elements.
- Your report's description of these values as stresses over wall nodes is not actually demonstrated by the extraction code.

**My recommendation:** create a dedicated `DAM_WALL` element group in the mesh and calculate the wall stress independently of the foundation. Ideally, extract stresses at integration points (`SIEF_ELGA`) on the wall elements and calculate the principal stresses there, rather than relying exclusively on recovered nodal stresses.

That would let us distinguish genuine wall tension from a foundation hotspot.

## **3. Your bedrock is not actually being modelled as a solid**

There is another interesting discrepancy.

You define:

```
BEDROCK_MAT = DEFI_MATERIAU(    ELAS=_F(        E=50000000000,        NU=0.25,        RHO=2700,    ),)
```

But `BEDROCK_MAT` is never assigned in `AFFE_MATERIAU`.

Furthermore, `BEDROCK` does not appear in your `AFFE_MODELE` assignment.

Instead, the model uses `OUTER_SPRINGS` with `DIS_T` elements and discrete stiffness matrices.

This is not necessarily wrong. A distributed spring representation can be a legitimate simplification of the surrounding rock. However, it is important to understand what you have:

- Concrete wall/foundation continuum.
- Joint elements.
- Distributed bedrock support springs.
- No bedrock continuum in the actual mechanical model.

Your report also shows that the support system is carrying substantial tension at full reservoir:

- Dry: approximately 115.8 MN.
- Water in joint: approximately 97.4 MN.

That is worth investigating. If your intended foundation/bedrock interface is compression-only, springs that can carry tension may be providing restraint that a real opening interface would not provide. If the rock is bonded to the foundation, tensile support may be physically appropriate.

I would not change this without establishing which physical condition you intend to represent.

## **4. Your concrete is explicitly configured as elastic**

This is probably the biggest issue.

Your nonlinear behaviour is:

```
COMPORTEMENT=(    _F(RELATION='ELAS', TOUT='OUI'),    _F(RELATION='JOINT_MECA_FROT', GROUP_MA='JOINTS'))
```

That tells Code_Aster:

- Concrete: linear elastic.
- Joints: frictional contact with opening and sliding.

There is no concrete damage model, no tensile softening and no crack propagation in this calculation.

Consequently, the following report values are entirely consistent with your implementation:

```
"damaged_area_m2": 0.0,
"broken_area_m2": 0.0
```

The model cannot produce the concrete cracking result you're ultimately looking for in its current configuration.

## **5. A few other things I would correct or investigate**

**A. Friction coefficient discrepancy**

Your JSON case description reports a friction coefficient of `1.0`, but the actual material definition in `case.comm` uses:

```
MU=0.8
```

The joint configuration in the JSON also reports `0.8`.

This looks like a reporting/configuration inconsistency. Make sure the intended coefficient is used everywhere.

**B. The joint is already opening before reservoir loading**

At instant 1, before the reservoir is raised:

- Opened area: 34.13 m².
- Opened share: 4.27%.
- Largest opening: 0.0388 mm.
- Sliding area: 259.16 m².

This is the gravity-only state. I would investigate why the joint opens and slides under that condition before proceeding to the full reservoir case.

Your adhesion is only 1,000 Pa (1 kPa). That is a very small tensile capacity for an interface, so the choice deserves particular attention.

**C. The reported baseline stress differs from your earlier figure**

You previously gave me an accepted baseline of approximately 1.12 MPa tension. The attached report records 1.379 MPa.

That is a difference worth reconciling. It could be a different model version, a different stress extraction method, or a different baseline result. We should not assume they are interchangeable.

**D. Water in the joint increases opening**

At full reservoir, adding joint water increases the opened area from 374.05 to 408.78 m² and the largest opening from 1.124 to 1.308 mm.

That is not necessarily a problem: uplift pressure can reduce effective compression across the interface. But it is another reason to separate the foundation-contact behaviour from cracking inside the wall.

## **6. What I would do next**

I would not change the loads or randomly weaken the concrete yet. I would make three diagnostic changes first.

Priority 1

**Separate wall and foundation stresses**

Create distinct element groups for `DAM_WALL` and `FOUNDATION`. Report their maximum principal stresses independently, preferably at integration points.

Priority 2

**Investigate the gravity-only joint opening**

Check the joint normals, contact law, adhesion, foundation restraint and spring reactions at instant 1. Establish whether the opening is physically expected.

Priority 3

**Introduce a concrete damage model**

Once the elastic and contact behaviour is verified, introduce an appropriate tensile damage/fracture formulation for the concrete wall. This is where actual crack propagation should happen.

## **One thing I want to establish before we change the model**

Your mesh appears to contain a concrete foundation as well as the dam wall. I want to understand how you intended the two to behave.

**Is the concrete foundation intended to be a solid, continuous extension of the dam wall, or is it a separate concrete block sitting on bedrock, with the dam wall connected to it?**

That distinction will influence how I would model the wall/foundation interface, where cracks should be allowed to propagate, and which stresses should be included in the final dam-wall assessment.

My overall assessment: the current solver is successfully completing its frictional joint calculation, but it is not yet answering your original concrete crack-propagation question. I think we can fix the modelling approach without throwing away the work you've already done.
