"""
The paths the Julia constant c can follow, one per mode.

Each takes a continuous position `s` (advanced by the render loop, faster
when the music is louder) and returns c as a complex number, so c glides
instead of jumping from point to point.

1. sequence_drift   - one golden-angle spiral whose radius profile slowly
                      crossfades between the halving / doubled / Fibonacci
                      sequences.
2. fibonacci_flower - c on the Mandelbrot main cardioid at Fibonacci-ratio
                      rotation numbers 1/2, 2/3, 3/5 ... 34/55. Each is a
                      Julia "flower" with that many arms (2, 3, 5 ... 55),
                      closing in on the golden-mean Siegel disk.
3. fibonacci_power  - for z^d + c with d a Fibonacci number (3, 5, 8), so the
                      set has d-fold symmetry. c rides around the main
                      component with a radius that wobbles at phi times the
                      orbit rate; phi being the "most irrational" number, the
                      path never closes on itself.
4. metallic_sample  - mode 1's spiral generalized off phi: each metallic mean
                      (bronze, silver, golden, plastic) brings its own
                      divergence angle 360/mu^2 and its own integer
                      sequences, and the mode walks from the tightest angle
                      to the widest.
5. FareyWalk        - mode 2 generalized: instead of only the Fibonacci
                      ratios, c descends the Stern-Brocot tree, so the arm
                      count at each step is the mediant of the last two and
                      the branch choices spell out a continued fraction the
                      music is writing.
6. phoenix_path     - z -> z^d + c + p*z_prev, the Phoenix map: one step of
                      memory, which breaks the Julia set's rotational
                      symmetry into wings. p glides between -1/phi and
                      -1/phi^2.
7. mandelbar_path   - antiholomorphic maps: the tricorn z-bar^d + c and the
                      burning ship's absolute-value fold. Neither is complex
                      differentiable, so the smooth escape-time bands grow
                      creases and flames instead of circles.

Modes 6 and 7 need the shader to iterate something other than z^d + c; the
FORMULA_* ids below are what gets passed to it in u_formula.
"""


import cmath
import math

import numpy as np

from sequences import ALL_SEQUENCES, METALLIC_FAMILIES, SEQUENCES

PHI = (1 + 5 ** 0.5) / 2

# What the shader should iterate (u_formula).
FORMULA_POWER = 0    # z^d + c
FORMULA_PHOENIX = 1  # z^d + c + p*z_prev
FORMULA_TRICORN = 2  # conj(z)^d + c
FORMULA_SHIP = 3     # (|Re z| + i|Im z|)^d + c


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def ping_pong(s, n):
    """Map s onto 0..n-1 and back again, so walking a list never jumps
    from its last entry straight to its first."""
    if n < 2:
        return 0.0
    period = 2 * (n - 1)
    u = s % period
    return u if u <= n - 1 else period - u


def sequence_radii(n_terms, radius_scale=1.4):
    """Log-scaled, normalized radius profile for each sequence (same scaling
    as golden_angle_walk), keyed by sequence name."""
    radii = {}
    for name, gen in SEQUENCES.items():
        logs = np.log1p(np.array(gen(n_terms), dtype=float))
        radii[name] = logs / (logs.max() or 1.0) * radius_scale
    return radii


def sequence_drift(radii, s, angle_deg, blend_phase):
    """Mode 1. The angle advances continuously by angle_deg per term; the
    radius is a blend of the three sequences' profiles, with weights that
    rotate through them as blend_phase goes round."""
    names = list(radii)
    n = len(radii[names[0]])
    pos = ping_pong(s, n)
    i0 = int(pos)
    i1 = min(i0 + 1, n - 1)
    f = smoothstep(pos - i0)

    weights = [(0.5 + 0.5 * math.cos(blend_phase - 2 * math.pi * k / len(names))) ** 2
               for k in range(len(names))]
    total = sum(weights)
    r = sum(w / total * ((1 - f) * radii[name][i0] + f * radii[name][i1])
            for w, name in zip(weights, names))
    theta = math.radians(angle_deg) * (pos + 1)
    return cmath.rect(r, theta)


def _fib_ratios(count):
    a, b = 1, 2
    out = []
    for _ in range(count):
        out.append((a, b))
        a, b = b, a + b
    return out


FIB_RATIOS = _fib_ratios(8)  # 1/2, 2/3, 3/5, 5/8, 8/13, 13/21, 21/34, 34/55


FLOWER_ARMS = tuple(q for _, q in FIB_RATIOS)  # 2, 3, 5, 8, 13, 21, 34, 55


def nearest_flower(cycle_len):
    """Index into FIB_RATIOS of the flower whose arm count is closest (in
    ratio) to a repeat length of `cycle_len` pulses."""
    return min(range(len(FLOWER_ARMS)), key=lambda i: abs(math.log(FLOWER_ARMS[i] / cycle_len)))


def cardioid_bulb(rotation, inv_q2, push):
    """c just outside the Mandelbrot main cardioid at rotation number
    `rotation` = p/q, i.e. inside the p/q bulb, where the Julia set is q arms
    meeting at the fixed point. The bulb's size falls off like 1/q^2, so the
    offset off the cardioid is scaled by `inv_q2`; `push` (0 up to ~2, from
    the music and kicks) thins the arms, and past ~1 shoves c out of the bulb
    entirely, where the set falls to dust until the push fades.

    Returns (c, fixed_point) so the view can center on the pattern.
    """
    rho = 1 + (0.4 + 1.2 * push) * inv_q2
    w = cmath.rect(rho, 2 * math.pi * rotation)
    return w / 2 - w * w / 4, w / 2


def fibonacci_flower(pos, push, mirror=False):
    """Mode 2. `pos` is a position along FIB_RATIOS: whole numbers sit on
    that flower, fractions glide toward the next. c sits just past the
    cardioid, inside the p/q bulb, where the Julia set is q arms meeting at
    the fixed point. The bulb shrinks like 1/q^2, so the offset does too.
    `push` (0 up to ~2, from the music and kicks) thins the arms, and past
    ~1 blows the flower out of its bulb into dust until it settles back.
    `mirror` runs the ratios as 1 - p/q, the flower's mirror image.

    Returns (c, fixed_point) so the view can center on the flower."""
    n = len(FIB_RATIOS)
    pos = min(max(pos, 0.0), n - 1.0)
    i0 = min(int(pos), n - 1)
    i1 = min(i0 + 1, n - 1)
    f = smoothstep((pos - i0 - 0.35) / 0.65)  # short dwell, long glide
    p0, q0 = FIB_RATIOS[i0]
    p1, q1 = FIB_RATIOS[i1]
    rotation = (1 - f) * p0 / q0 + f * p1 / q1
    if mirror:
        rotation = 1 - rotation
    inv_q2 = (1 - f) / q0 ** 2 + f / q1 ** 2
    return cardioid_bulb(rotation, inv_q2, push)


FIB_POWERS = (3, 5, 8)


def fibonacci_power(s, push, degree):
    """Mode 3. For z -> z^d + c: pick the fixed point's multiplier
    lam = rho * e^(i*alpha), then z* = (lam/d)^(1/(d-1)) and
    c = z* (1 - lam/d). alpha orbits at rate 1, rho wobbles at rate phi, so
    the path is quasi-periodic and never repeats. rho ranges from inside
    the main component (filled, glowing) out past its edge (spiral arms,
    then dust); `push` (0 up to ~2, from the music and kicks) shoves it out."""
    alpha = s
    rho = 0.8 + 0.35 * (0.5 + 0.5 * math.sin(PHI * s)) + 0.15 * push
    lam = cmath.rect(rho, alpha)
    z = cmath.rect((rho / degree) ** (1 / (degree - 1)), alpha / (degree - 1))  # continuous root
    return z * (1 - lam / degree)


def metallic_profiles(n_terms, radius_scale=1.1):
    """Mode 4's table: one entry (label, angle_deg, radii) per sequence that
    belongs to a metallic family, ordered tightest divergence angle first
    (bronze 33deg, silver 61.8deg, golden 137.5deg, plastic 205.1deg).

    `radii` is the same log-scaled, normalized profile golden_angle_walk and
    sequence_radii use, so every entry sweeps the same 0..radius_scale range —
    what changes between them is the shape of the sweep and the angle.
    """
    out = []
    for family, (_mean, angle, members) in METALLIC_FAMILIES.items():
        for name in members:
            logs = np.log1p(np.array(ALL_SEQUENCES[name](n_terms), dtype=float))
            out.append((f"{family} / {name}", angle,
                        logs / (logs.max() or 1.0) * radius_scale))
    return out


def metallic_sample(profiles, s, family_pos, push=0.0):
    """Mode 4. Returns (radius, angle_deg, label).

    `s` walks along one spiral (ping-ponged over the terms, so c sweeps out
    to the rim and back); `family_pos` walks along `profiles`, crossfading
    both the radius profile and the divergence angle between neighbours.

    The angle comes back instead of a finished c because the caller
    accumulates it into a phase: multiplying a changing angle by the term
    index, the way mode 1 does, would whip c around the plane every time the
    family slid on.
    """
    n = len(profiles)
    family_pos = min(max(family_pos, 0.0), n - 1.0)
    j0 = int(family_pos)
    j1 = min(j0 + 1, n - 1)
    g = smoothstep(family_pos - j0)

    terms = len(profiles[j0][2])
    pos = ping_pong(s, terms)
    i0 = int(pos)
    i1 = min(i0 + 1, terms - 1)
    f = smoothstep(pos - i0)

    def radius_of(entry):
        r = entry[2]
        return (1 - f) * r[i0] + f * r[i1]

    r = (1 - g) * radius_of(profiles[j0]) + g * radius_of(profiles[j1])
    angle = (1 - g) * profiles[j0][1] + g * profiles[j1][1]
    return r * (1 + 0.25 * push), angle, profiles[j1 if g >= 0.5 else j0][0]


class FareyWalk:
    """Mode 5. A descent of the Stern-Brocot tree down the main cardioid.

    Two Farey parents p0/q0 < p1/q1 bracket an interval; their mediant
    (p0+p1)/(q0+q1) lands inside it, and keeping it as the new left or right
    parent descends a level. Every rational on the way is a bulb on the
    cardioid whose Julia set has q arms, so the arm counts come out as
    mediants: always-left gives 1/2, 1/3, 1/4, 1/5 — one petal added at a
    time; always-right gives 1/2, 2/3, 3/4, 4/5; strict alternation gives the
    Fibonacci ratios mode 2 is built on, and any other pattern of branches
    spells out the continued fraction of whatever irrational the walk is
    converging to.

    So the branch is worth giving to the music: the caller counts the kicks
    in each bar and sends an odd count right, an even one left. Leaning hard
    either way is the dull case — all-left is 1/2, 1/3, 1/4 and all-right is
    1/2, 2/3, 3/4, both adding one arm per step — so what makes the tour move
    is the branches *changing*, which is what a parity rule reacts to.

    Once q passes q_max the arms are finer than a pixel, so the walk restarts
    from the root, a level deeper to the left each time so successive tours
    start somewhere new.
    """

    def __init__(self, q_max=89):
        self.q_max = q_max
        self.restarts = 0
        self.history = []
        self._restart()

    def _restart(self):
        self.left, self.right = (0, 1), (1, 1)
        self.prev = self.cur = (1, 2)
        self.history = [self.cur]
        for _ in range(self.restarts % 4):
            self.step(False)

    def step(self, go_right):
        """Descend one level, taking the mediant as the new right parent
        (go_right) or the new left one. Restarts instead if that mediant's
        arms would be finer than q_max. Returns the new (p, q)."""
        left, right = (self.cur, self.right) if go_right else (self.left, self.cur)
        mediant = (left[0] + right[0], left[1] + right[1])
        if mediant[1] > self.q_max:
            was = self.cur
            self.restarts += 1
            self._restart()
            # Keep the bulb we came from as the glide's starting point, so a
            # new tour slides across the cardioid instead of cutting to it.
            self.prev = was
            return self.cur
        self.left, self.right = left, right
        self.prev, self.cur = self.cur, mediant
        self.history.append(self.cur)
        del self.history[:-8]
        return self.cur

    def c(self, f, push):
        """c for a position `f` (0..1) gliding from the previous mediant's
        bulb to the current one's. Returns (c, fixed_point), like mode 2.

        No mirror option, unlike mode 2: the Fibonacci ratios all cluster
        around 0.618 of the way round the cardioid and need mirroring to get
        off that spot, while the mediants spread over the whole of it.
        """
        f = smoothstep(f)
        (p0, q0), (p1, q1) = self.prev, self.cur
        rotation = (1 - f) * p0 / q0 + f * p1 / q1
        inv_q2 = (1 - f) / q0 ** 2 + f / q1 ** 2
        return cardioid_bulb(rotation, inv_q2, push)

    @property
    def arms(self):
        return self.cur[1]

    def label(self):
        p, q = self.cur
        return f"{p}/{q}, {q} arms"


# p for the Phoenix map glides between these two, the reciprocals of phi.
PHOENIX_P = (-1 / PHI, -1 / PHI ** 2)   # -0.6180, -0.3820
PHOENIX_C = 0.5667                      # the classic Phoenix constant


def phoenix_path(s, push):
    """Mode 6. z -> z^2 + c + p*z_prev: one step of memory, which is enough
    to break the Julia set's rotational symmetry into a pair of wings (the
    map isn't conjugate to anything of the form z^d + c).

    c rides a small circle around the classic 0.5667, and p swings between
    -1/phi and -1/phi^2 at 1/phi times the orbit rate; a `push` (0 up to ~2,
    from the music and kicks) widens c's circle until the wings tear, and
    tilts p off the real axis, which shears them.

    Returns (c, p).
    """
    lo, hi = PHOENIX_P
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    p = complex(mid + half * math.sin(s / PHI), 0.08 * push)
    c = complex(PHOENIX_C, 0.0) + cmath.rect(0.03 + 0.06 * push, s)
    return c, p


# Mode 7 cuts between these four: (formula, degree, center, radius band).
#
# Each family's connectedness locus sits somewhere different, and only part of
# it gives a Julia set with both a filled core and structured bands around it —
# too far in and it's one flat blob, too far out and the escape time is a wash.
# These bands were picked by measuring both offline (see README, "Where the
# math actually shows up"): the tricorn's quadratic body is tight around the
# origin, the burning ship's sits well to the left, and raising the degree
# pushes both outward.
MANDELBAR_VARIANTS = (
    (FORMULA_TRICORN, 2, 0.0, (0.26, 0.44)),
    (FORMULA_SHIP, 2, -0.6, (0.10, 0.26)),
    (FORMULA_TRICORN, 3, 0.0, (0.44, 0.58)),
    (FORMULA_SHIP, 3, 0.0, (0.52, 0.66)),
)


def mandelbar_path(s, push, variant):
    """Mode 7. c for the antiholomorphic maps: the tricorn conj(z)^d + c and
    the burning ship's (|Re z| + i|Im z|)^d + c. Conjugation and the absolute
    values are not complex differentiable, so the escape-time bands come out
    creased and flame-edged instead of smoothly nested.

    c orbits its variant's body once per unit of `s` with a radius that
    crosses the band at phi times the orbit rate, so, as in mode 3, the path
    is quasi-periodic and never closes. `push` (0 up to ~2, from the music and
    kicks) drives it out past the band, where the flames break into embers.
    """
    _formula, _degree, center, (r_lo, r_hi) = variant
    width = r_hi - r_lo
    r = r_lo + width * (0.5 + 0.5 * math.sin(PHI * s)) + 0.25 * width * push
    return cmath.rect(r, s) + center
