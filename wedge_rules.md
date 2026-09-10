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
h = local wedge height, measured vertically (adjacent = 4 m at max depth)
t = local wedge thickness, measured radially/perpendicular to wall (opposite = 2 m at max depth)
hypotenuse = sloped wedge face length = sqrt(h^2 + t^2) (4.472136 m at max depth)
w = wedge width along the chainage direction
```

The nominal design ratio inputs are based on Pythagoras theorem:

```text
c = 25 m (depth threshold)
adjacent_max = 4 m (height at max depth 29 m)
opposite_max = 2 m (thickness at max depth 29 m)
hypotenuse_max = sqrt(4^2 + 2^2) = 4.472136 m
ratio (opposite / adjacent) = 2 / 4 = 0.5
```

### Local dimensions

At each chainage, let `p` be the local positive plinth depth. The wedge is
active only where `p > c`:

$$
h = p - c = p - 25
$$

$$
t = h \times \left(\frac{\text{opposite}}{\text{adjacent}}\right) = 0.5 \times h
$$

$$
\text{hypotenuse} = \sqrt{h^2 + t^2} = h \times \sqrt{1 + 0.5^2} \approx 1.118034 \times h
$$

The design limits are:

```text
0 < h <= 4 m
0 < t <= 2.0 m (exact)
0 < hypotenuse <= 4.472136 m
```

At the maximum section (`p = 29 m`):

```text
h = 29 - 25 = 4.0 m
t = 0.5 * 4.0 = 2.0 m (exact)
hypotenuse = sqrt(4^2 + 2^2) = 4.472136 m (exact)
```

$$
wedgeThickness = wedgeHeight \times 0.5
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
wedge attachment are on the exact `76 m` radius. At the maximum section (`p = 29 m`), the
cross-section dimensions are:

```text
wall-facing wedge vertex: r = 76 m, z = -25 m
toe vertex:               r = 74.0 m (exact), z = -29 m
adjacent (height h):      4.0 m
opposite (thickness t):   2.0 m
hypotenuse (slope):       4.472136 m = sqrt(4^2 + 2^2)
```

The toe radius is calculated from the local thickness:

$$
r_{toe} = 76 - t = 76 - (0.5 \times h)
$$

At maximum depth, this gives `r_toe = 74.0 m` (exact). The `r = 74.0 m` value
applies to the chainage-running toe vertex line; it does **not** mean that the
entire wedge bottom face has a constant radius of `74 m`. The wedge slope has an
exact 1:2 gradient (opposite:adjacent), corresponding to `atan(0.5) = 26.565051°`
from vertical, or `63.434949°` to horizontal.

The generator should use the exact ratio `0.5` rather than rounded trigonometric
functions to guarantee exact mesh segment counts and grid alignment. The analytic
radial normal is used so the downstream wall face remains numerically constant at
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
