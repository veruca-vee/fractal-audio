"""
Veruca's sequence math, extracted for reuse.

Three generators, all returning a list of Python ints (arbitrary precision,
since plain Fibonacci and the doubled variant blow up fast):

- halving_variant(n):  a1=1, a2=2; for k>=3, s = a[k-1] + a[k-2];
                        a[k] = s // 2 if s is even else s.
                        Bounded growth — the fold keeps it from exploding.
- doubled_variant(n):  b1=1, b2=1; b[k] = b[k-1] + b[k-2] + 1.
                        Equals 2*fibonacci(k) - 1.
- fibonacci(n):         plain Fibonacci, f1=f2=1.

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


SEQUENCES = {
    "halving": halving_variant,
    "doubled": doubled_variant,
    "fibonacci": fibonacci,
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
    h = halving_variant(15)
    d = doubled_variant(15)
    f = fibonacci(15)
    print("halving :", h)
    print("doubled :", d)
    print("fib     :", f)
    print("walk[:5]:", golden_angle_walk(h)[:5])
