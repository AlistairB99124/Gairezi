# How the applied loads create the computed stresses

This model is a linear elastic, 3D solid analysis of the dam. The solver is not computing arbitrary stress values; it is solving the equilibrium equation for a deformable body under the loads defined in the Elmer input file.

The governing concept is:

- The dam is the solid body.
- Its displacement field is solved from the elastic equations.
- The strain field is derived from displacement.
- The stress field is derived from strain via Hooke's law.
- The loads are the body force caused by gravity and the boundary tractions caused by water pressure and surcharge.

The relevant setup is in `Elmer/dam_model.sif` and the key pieces are:

- `Body 1` has `Material = 1` and `Body Force = 1`
- `Body Force 1` applies a gravity body force
- `Material 1` defines the concrete stiffness and density
- boundary conditions 1-6 apply the support and pressure loads
- `StressSolve` computes displacement and then computes stress and principal stress from that displacement field

## 1. The structure is modeled as an elastic solid

The model uses:

- `Coordinate System = Cartesian 3D`
- `Simulation Type = Steady State`
- `Variable = "Displacement"` with 3 DOFs
- `StressSolve` and `Calculate Stresses = Logical True`

So the unknown field is the displacement vector:

- u(x, y, z) = [u1, u2, u3]

The engineering strain is then:

- epsilon = 1/2 (grad u + grad u^T)

and the stress tensor is:

- sigma = C : epsilon

where C is the elastic constitutive matrix for isotropic concrete defined by:

- Young's Modulus = 3.5e10 Pa
- Poisson Ratio = 0.2

This means every element in the dam has a stiffness. When loads are applied, the solver computes the displacement needed to satisfy equilibrium, and then converts that displacement into stress.

In plain terms:

- more load = more displacement
- more stiffness = less displacement
- stress is the internal force required to resist those displacements

## 2. Gravity is a distributed body force

The body force is defined as:

```text
Body Force 1
  Stress Bodyforce 3 = Real -23544.0
End
```

This is the vertical gravity load, already converted to force per unit volume:

- density = 2400 kg/m^3
- g = 9.81 m/s^2
- rho * g = 23544 N/m^3
- the negative sign is because gravity acts downward in the global z direction

So every element of the dam body is loaded by a downward force density of about 23.5 kN/m^3.

This produces:

- vertical compression through the dam body
- a downward reaction force at the foundation
- bending and shearing if the dam geometry and support conditions are not perfectly symmetric
- a stress field that is largest in the lower portion of the dam where the self-weight accumulates

The effect is not just a uniform compression: because the dam is curved and is restrained differently at the base and abutments, the weight leads to a non-uniform stress distribution, including local stress concentrations around the base and near the abutments.

## 3. The base is fixed, so the dam cannot move there

The bedrock base condition is:

```text
Boundary Condition 1
  Name = "BedrockBase"
  Target Boundaries(1) = 1
  Displacement 1 = Real 0.0
  Displacement 2 = Real 0.0
  Displacement 3 = Real 0.0
End
```

This is a fully fixed foundation boundary. The base nodes cannot translate in x, y, or z.

This is crucial because it creates the reaction forces required to balance:

- self-weight
- hydrostatic pressure on the upstream face
- downstream pressure
- crest surcharge
- any arching restraint from the abutments

Without a fixed base, the dam would simply translate or deform freely under load. The fixed foundation is what makes the stress field physically meaningful.

## 4. Upstream water pressure acts as a normal pressure load on the upstream face

The upstream face boundary condition is:

```text
Boundary Condition 2
  Name = "UpstreamHydrostaticPressure"
  Target Boundaries(1) = 2
  Normal Force = Variable Coordinate 3
    Real MATC "-1000.0 * 9.81 * (33.0 - tx) * (tx < 33.0) * 1.0"
End
```

This is a hydrostatic pressure distribution:

- water density = 1000 kg/m^3
- pressure = rho * g * h
- the free surface is at z = 33 m
- pressure grows linearly with depth below the water surface
- pressure is applied only where `tx < 33.0`

So the boundary load is:

- p(z) = 1000 * 9.81 * (33 - z)

This means:

- at the surface: pressure = 0
- at the base: pressure = 1000 * 9.81 * (33 - 4) = 284,490 Pa

This load is applied as a normal traction on the upstream boundary, which pushes against the dam from the water side. Because the sign is negative, it is directed so that the dam feels a compressive force from the water. This is the principal external load that tends to push the dam downstream and create high compressive stresses on the upstream face.

The stress pattern from this loading is usually:

- high compressive stress near the upstream face at depth
- bending from the hydrostatic load across the section
- reaction forces at the base and abutments to balance the hydrostatic thrust

Because the dam is arch-shaped and restraint is provided by the foundation and abutments, the structure does not simply translate; instead, the load is redistributed into a complex 3D stress state with tension/compression combinations depending on location.

## 5. Downstream tailwater applies a smaller opposing pressure

The downstream load is:

```text
Boundary Condition 3
  Name = "DownstreamTailwater"
  Target Boundaries(1) = 3
  Normal Force = Variable Coordinate 3
    Real MATC "-1000.0 * 9.81 * (6.0 - tx) * (tx < 6.0) * 1.0"
End
```

This represents a lower downstream water level, with a free surface at z = 6 m.

Thus:

- the downstream water acts against the dam, but less strongly than the upstream water
- it opposes the upstream hydrostatic thrust
- the net horizontal load is the difference between upstream and downstream pressures, not the sum

This is important because the headed difference across the dam produces the main horizontal bending/arching demand. The downstream pressure partially cancels the upstream pressure, but not completely; the remaining unbalanced pressure creates the net load that must be carried by the dam and by the foundation.

## 6. Crest surcharge adds a local traction on the top edge

The crest condition is:

```text
Boundary Condition 4
  Name = "CrestOverflowSurcharge"
  Target Boundaries(1) = 4
  Normal Force = Real -20000.0
End
```

This applies a uniform traction of 20 kPa on the crest boundary.

Its effect is more localized than the hydrostatic loads:

- it adds a compressive boundary force near the crest
- it contributes to local stress near the top of the dam
- it can increase compressive stress around the crest region and may alter the local principal stress directions near the top surface

The crest load is usually smaller than the hydrostatic thrust, but it still contributes to the stress field and is important for a realistic dam loading combination.

## 7. Abutment springs resist horizontal and vertical motion at the ends

The abutment boundaries are:

```text
Boundary Condition 5
  Name = "LeftAbutmentSpring"
  Target Boundaries(1) = 5
  Spring 1 = Real 1.0e11
  Spring 2 = Real 1.0e11
  Spring 3 = Real 5.0e10
End

Boundary Condition 6
  Name = "RightAbutmentSpring"
  Target Boundaries(1) = 6
  Spring 1 = Real 1.0e11
  Spring 2 = Real 1.0e11
  Spring 3 = Real 5.0e10
End
```

These are not rigid supports like the base; they are spring restraints with very large stiffness.

They model the lateral restraint offered by the abutments or valley wall restraint around the arch dam, but with finite flexibility instead of perfect fixity.

Their job is to:

- prevent free movement at the ends of the arch
- redistribute the horizontal load into arch action
- provide the stiffness needed for the dam to carry water load through the arch structure rather than as a simple cantilever

This is why the stress field in an arch dam is not the same as that in a gravity dam. The abutments and foundation are essential to the arching effect, and the spring stiffness directly affects the computed stress state.

## 8. Why the solver produces stress rather than just displacement

`StressSolve` solves the displacement field first. Once the nodal displacements are known, the element strains and stresses are computed.

At the element level:

- displacement fields are interpolated across each element
- strains are obtained from displacement gradients
- stresses are obtained from the constitutive matrix

Therefore the final stress field is a consequence of all load combinations and support conditions working together.

The actual finite element equilibrium is effectively:

- K u = F

where:

- K is the global stiffness matrix from the material and geometry
- u is the vector of all nodal displacements
- F is the total applied force vector from gravity, water pressure, surcharge, and reactions

The computed stress tensor at each element is then the internal force state needed to balance those applied loads while satisfying compatibility and support constraints.

## 9. Combined effect on the dam

The dam is loaded by the following main actions:

1. self-weight downward through the solid body
2. upstream hydrostatic pressure on the upstream face
3. downstream tailwater pressure on the downstream face
4. crest surcharge on the crest boundary
5. foundation restraint at the base
6. end restraint from abutment springs

These combine to produce a highly three-dimensional stress state.

The dominant effects are usually:

- compressive stress from self-weight in the lower dam
- strong horizontal compressive stress from upstream water load
- bending stress from the water load acting against the foundation and abutment restraints
- local stress concentration at the base and abutments where reactions are concentrated
- principal stresses that rotate through the structure due to the arch geometry and boundary restraints

This is why the output files contain stress and principal stress results: the code is not just solving for a single scalar load effect, but for the full internal force state throughout the dam body.

## 10. Summary

The computed stresses are the result of the dam being forced to satisfy equilibrium under the following loads:

- gravity body force from concrete self-weight
- varying hydrostatic pressure on the upstream face
- lower tailwater pressure on the downstream face
- crest surcharge
- fixed base support
- spring-supported abutments

Those applied forces generate a displacement field. The displacement field creates strain. The strain creates stress through the elastic constitutive law. The resulting stress tensor is the internal force state that balances the external loads while respecting the foundation and abutment constraints.

So when you look at the stress plots or principal stress output, you are seeing the elastic response of the structure to the exact load path defined by the boundary conditions and body forces in the Elmer model.
