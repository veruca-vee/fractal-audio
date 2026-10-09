"""
Veruca's sequence math, extracted for reuse.

The generators all return a list of Python ints (arbitrary precision, since
plain Fibonacci and friends blow up fast).

The two hand-rolled ones:

- halving_variant(n):  a1=1, a2=2; for k>=3, s = a[k-1] + a[k-2];
                        a[k] = s // 2 if s is even else s.
                        Bounded growth — the fold keeps it from exploding.
- doubled_variant(n):  b1=1, b2=1; b[k] = b[k-1] + b[k-2] + 1.
                        Equals 2*fibonacci(k) - 1.

The classical ones, grouped by what irrational their ratio converges to:

- fibonacci / lucas            -> phi = 1.6180 (golden mean)
- pell / pell_lucas            -> 1 + sqrt(2) = 2.4142 (silver mean)
- bronze                       -> (3 + sqrt(13))/2 = 3.3028 (bronze mean)
- padovan / perrin             -> rho = 1.3247 (plastic number)
- tribonacci                   -> 1.8393 (tribonacci constant)
- jacobsthal                   -> 2 exactly, so its spiral angle is rational
                                  and the walk closes on itself — the odd one
                                  out, and useful as a contrast.

METALLIC_FAMILIES names the four that drive mode 4: each pairs its mean with
its own divergence angle 360/mu^2 (137.5 deg for phi, 61.8 for silver, 33.0
for bronze, 205.1 for plastic) and the sequences that grow at that rate.

Also: golden_angle_walk(seq, angle_deg=None) turns any of the above into a
list of (x, y) points on the phyllotaxis spiral — radius from n, angle
n * golden angle (or a supplied override). This is the walk the fractal's
Julia constant follows.
"""

import math

GOLDEN_ANGLE_DEG = 137.5077640500378


def halving_variant(n: int) -> list[int]:
    if n < 1:
        return []
    a = [1, 2][:n]
    while len(a) < n:
        s = a[-1] + a[-2]
        a.append(s // 2 if s % 2 == 0 else s)
    return a


def doubled_variant(n: int) -> list[int]:
    if n < 1:
        return []
    b = [1, 1][:n]
    while len(b) < n:
        b.append(b[-1] + b[-2] + 1)
    return b


def fibonacci(n: int) -> list[int]:
    if n < 1:
        return []
    f = [1, 1][:n]
    while len(f) < n:
        f.append(f[-1] + f[-2])
    return f


def _linear(n: int, seed: list[int], coeffs: list[int]) -> list[int]:
    """Generic linear recurrence: a[k] = sum(coeffs[i] * a[k-1-i]).

    `seed` supplies as many leading terms as there are coefficients.
    """
    if n < 1:
        return []
    out = seed[:n]
    while len(out) < n:
        out.append(sum(c * out[-1 - i] for i, c in enumerate(coeffs)))
    return out


def lucas(n: int) -> list[int]:
    """2, 1, 3, 4, 7, 11 ... same recurrence as Fibonacci, other seed, so the
    ratio still goes to phi but the terms never coincide."""
    return _linear(n, [2, 1], [1, 1])


def pell(n: int) -> list[int]:
    """1, 2, 5, 12, 29 ... p[k] = 2*p[k-1] + p[k-2]; ratio -> 1 + sqrt(2)."""
    return _linear(n, [1, 2], [2, 1])


def pell_lucas(n: int) -> list[int]:
    """2, 2, 6, 14, 34 ... the companion Pell sequence, same silver ratio."""
    return _linear(n, [2, 2], [2, 1])


def bronze(n: int) -> list[int]:
    """1, 3, 10, 33, 109 ... b[k] = 3*b[k-1] + b[k-2]; ratio -> (3+sqrt(13))/2."""
    return _linear(n, [1, 3], [3, 1])


def tribonacci(n: int) -> list[int]:
    """1, 1, 2, 4, 7, 13, 24 ... three terms back; ratio -> 1.8393."""
    return _linear(n, [1, 1, 2], [1, 1, 1])


def padovan(n: int) -> list[int]:
    """1, 1, 1, 2, 2, 3, 4, 5, 7, 9 ... p[k] = p[k-2] + p[k-3]; ratio -> the
    plastic number 1.3247. The slowest grower here, so its spiral walks
    outward in the finest steps."""
    return _linear(n, [1, 1, 1], [0, 1, 1])


def perrin(n: int) -> list[int]:
    """3, 0, 2, 3, 2, 5, 5, 7, 10 ... Padovan's recurrence from another seed.
    Starts with a 0, which log1p maps to radius 0 — the walk opens from the
    middle of the plane."""
    return _linear(n, [3, 0, 2], [0, 1, 1])


def jacobsthal(n: int) -> list[int]:
    """1, 1, 3, 5, 11, 21, 43 ... j[k] = j[k-1] + 2*j[k-2]; ratio -> exactly 2.
    Rational, so unlike all the others its angle closes into a finite
    rosette instead of filling the disk."""
    return _linear(n, [1, 1], [1, 2])


SEQUENCES = {
    "halving": halving_variant,
    "doubled": doubled_variant,
    "fibonacci": fibonacci,
}

# Everything, for mode 4 and for anyone poking at this from a REPL.
ALL_SEQUENCES = {
    "halving": halving_variant,
    "doubled": doubled_variant,
    "fibonacci": fibonacci,
    "lucas": lucas,
    "pell": pell,
    "pell_lucas": pell_lucas,
    "bronze": bronze,
    "tribonacci": tribonacci,
    "padovan": padovan,
    "perrin": perrin,
    "jacobsthal": jacobsthal,
}


def metallic_mean(p: int, q: int = 1) -> float:
    """Positive root of x^2 = p*x + q: the mean that x[k] = p*x[k-1] + q*x[k-2]
    converges to. p=1 gives phi, p=2 the silver mean, p=3 the bronze mean."""
    return (p + math.sqrt(p * p + 4 * q)) / 2


def divergence_angle(mean: float) -> float:
    """The phyllotaxis angle belonging to a mean: 360 / mean^2 degrees.

    For phi this is the golden angle 137.5077deg; the more irrational the
    mean, the less the spiral's arms line up.
    """
    return 360.0 / (mean * mean)


PLASTIC = 1.324717957244746  # real root of x^3 = x + 1, the Padovan ratio
TRIBONACCI_CONSTANT = 1.839286755214161

# name -> (mean, angle in degrees, sequences growing at that rate).
# Ordered from the tightest angle outward, so mode 4 crossfading through them
# in order opens the spiral up step by step.
METALLIC_FAMILIES = {
    "bronze": (metallic_mean(3), divergence_angle(metallic_mean(3)), ("bronze",)),
    "silver": (metallic_mean(2), divergence_angle(metallic_mean(2)), ("pell", "pell_lucas")),
    "golden": (metallic_mean(1), GOLDEN_ANGLE_DEG, ("fibonacci", "lucas")),
    "plastic": (PLASTIC, divergence_angle(PLASTIC), ("padovan", "perrin")),
}


def golden_angle_walk(
    seq: list[int], angle_deg: float = GOLDEN_ANGLE_DEG, radius_scale: float | None = None
) -> list[tuple[float, float]]:
    """
    Map a sequence onto a phyllotaxis-style spiral.

    Radius comes from the sequence value itself (log-scaled, so the doubled
    variant and plain Fibonacci don't fly off to infinity by term 30), angle
    comes from n * angle_deg (golden angle by default — pass a different
    value to explore the straight-line / S-curve regimes).

    Returns points normalized so the whole walk fits inside a unit circle
    (radius <= 1.0), ready to be scaled into wherever the fractal's c-plane
    window sits.
    """
    if not seq:
        return []

    rad = math.radians(angle_deg)
    log_vals = [math.log1p(v) for v in seq]
    max_log = max(log_vals) or 1.0

    points = []
    for i, lv in enumerate(log_vals):
        n = i + 1
        r = (lv / max_log)  # 0..1
        theta = n * rad
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append((x, y))

    if radius_scale is not None:
        points = [(x * radius_scale, y * radius_scale) for x, y in points]

    return points


if __name__ == "__main__":
    # Quick sanity check when run directly.
    for name, gen in ALL_SEQUENCES.items():
        print(f"{name:11}:", gen(12))
    print()
    for name, (mean, angle, members) in METALLIC_FAMILIES.items():
        print(f"{name:8} mean {mean:.6f}  angle {angle:8.3f} deg  <- {', '.join(members)}")
    print()
    print("walk[:5]:", golden_angle_walk(halving_variant(15))[:5])
