let t = thickness of wedge
let h = height of wedge
let w = width of wedge (variable)
let p = plinth depth below crest
let a = 26.57°
let c = 25

The values `p` and `c` are absolute positive depths measured downward from the
crest. In the signed mesh coordinate system, the crest is `z = 0`, so a depth of
`25` corresponds to `z = -25`, and a plinth depth of `p` corresponds to
`z = -p`. The wall base at the maximum plinth depth is `z = -29`.

h = p - c; where p >= c
t = h * tan(a)

The wedge width `w` is the available chainage interval at or above the threshold
depth `c`. If `s_start` and `s_end` are the two chainages where `p = c`, then:

w = s_end - s_start