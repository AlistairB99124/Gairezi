## Geometry Rules

### Required inputs and coordinates

The downstream wedge is generated from the local plinth **depth** at each
chainage. The plinth profile is supplied in `Data/plinth.json` using positive
depths measured downward from the crest. The crest is the signed mesh datum:

```text
crest depth = 0 m  -> mesh z = 0 m
depth 25 m         -> mesh z = -25 m
depth 29 m         -> mesh z = -29 m
```

The wedge dimensions are:

```text
h = local wedge height, measured vertically
t = local wedge thickness, measured radially/perpendicular to the wall
w = wedge width along the chainage direction
```

The nominal design inputs are:

```text
c = 25 m
a = 26.57 degrees from vertical
```

### Local dimensions

At each chainage, let `p` be the local positive plinth depth. The wedge is
active only where `p > c`:

$$
h = p - c
$$

$$
t = h\tan(a) = (p - 25)\tan(26.57^\circ)
$$

The design limits are:

```text
0 < h <= 4 m
0 < t <= approximately 2 m
```

At the maximum section, `p = 29 m`, so `h = 4 m` and
`t = 2.000432 m` using the rounded angle `26.57 degrees`.

$$
wedgeThickness = wedgeHeight\tan(26.57^\circ)
$$

### Chainage width

The wedge starts and ends where the plinth depth reaches the threshold `p = 25
m`, which is the signed elevation `z = -25 m`. Let `s` and `e` be those two
chainages. The wedge is active for `s < chainage < e`, with zero height at the
two limiting crossings, and:

```text
s = 87.960912 m
e = 124.130081 m
w = e - s = 36.169169 m
```

These values are obtained by linear interpolation of the supplied plinth-depth
profile. The start crossing is between `(chainage, plinth) = (62, 17.03)` and
`(92, 26.24)`; the end crossing is between `(116, 29.00)` and `(131, 21.62)`.

### Cross-section and radii

The wall centreline radius is `78 m`. The wedge-facing downstream wall face and
wedge attachment are on the exact `76 m` radius. At the maximum section, the
cross-section is:

```text
wall-facing wedge vertex: r = 76 m, z = -25 m
toe vertex:               r = approximately 74 m, z = -29 m
```

The toe radius is calculated from the local thickness:

$$
r_{toe} = 76 - t
$$

At maximum depth, this gives `r_toe = 73.999568 m` with the rounded angle, or
nominally `74 m`. The `r = 74 m` value applies to the chainage-running toe
vertex line; it does **not** mean that the entire wedge bottom face has a
constant radius of `74 m`. The wedge slope is `26.57 degrees` from vertical,
equivalently `63.43 degrees` to horizontal.

The generator should use the analytic radial normal rather than an estimated
chord normal, so the downstream wall face remains numerically constant at
`r = 76 m`.

### Plinth, bedrock, and bonding

Where `p = 29 m` and the ground depth is also `29 m`, there is no plinth. At
that section the wall base, wedge base, and bedrock meet at `z = -29 m`.

The wedge is part of the same bonded monolithic body as the wall and plinth:

- wedge-to-wall interfaces must use shared nodes and faces;
- wedge-to-plinth interfaces must use shared nodes and faces where a plinth exists;
- wedge-to-bedrock interfaces must use shared nodes and faces where no plinth exists;
- wall-to-bedrock interfaces must remain shared;
- no gaps, duplicate coincident interface nodes, contact elements, or floating
	wedge body are permitted.

The geometry must remain conforming and must not introduce zero-volume elements
or unpaired internal two-dimensional boundary faces.
