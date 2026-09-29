"""
The three paths the Julia constant c can follow, one per mode.

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
"""

import cmath
import math

import numpy as np

from sequences import SEQUENCES

PHI = (1 + 5 ** 0.5) / 2


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
    rho = 1 + (0.4 + 1.2 * push) * inv_q2
    w = cmath.rect(rho, 2 * math.pi * rotation)
    return w / 2 - w * w / 4, w / 2


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
