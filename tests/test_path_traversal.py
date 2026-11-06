"""Traversal: a clocked path must actually travel, not just lock.

The beat-lock tests (test_beat_clock.py) drive the path with a phase
that advances steadily and check the path stays on that grid. These
tests cover the other half of the contract: the path is the *pace*
of c (README, "What follows what" -- one term or one loop per bar),
and a pace that can be cancelled by the thing it follows is not a
pace. The phase the tracker hands over is an estimate, re-anchored
every half second, and it can stall outright -- the 2026-11-06
session log has consecutive rows a second apart with every tracker
value, cycle_phase included, identical. However the phase behaves,
the path must keep travelling at a good fraction of the bar rate,
the way it did before the phase was handed over for k > 1 at all:
then nothing the tracker said could slow the path down.

Built without __init__, like the other window tests.
"""

import math
import unittest

import main
import phi_paths


class FakeTracker:
    """Just the rhythm quantities main.py reads. The test sets
    cycle_phase by hand -- or pointedly does not set it, which is
    what a stalled estimate looks like from the path's side."""

    def __init__(self, cycle_len=4, cycle_seconds=1.3, conf=0.9):
        self.cycle_len = cycle_len
        self.cycle_conf = conf
        self.cycle_seconds = cycle_seconds
        self.cycle_phase = 0.0


def make_window(tr):
    win = main.FractalWindow.__new__(main.FractalWindow)
    win.tr = tr
    win.paused = False
    win.blend_phase = 0.0
    win.path_s = {m: 0.0 for m in main.MODE_NAMES}
    return win


def travel_fraction(cycle_seconds, min_loop_s, seconds=120.0, dt=0.02,
                    unit=1.0, mode=1, phase_fn=None):
    """Drive _advance with the given phase signal and return how far
    the path travelled, as a fraction of the nominal distance (one
    loop per bar). phase_fn(t) -> cycle phase; None means the phase
    never moves from where it started."""
    tr = FakeTracker(cycle_seconds=cycle_seconds)
    tr.cycle_phase = 0.83
    win = make_window(tr)
    clock = win._music_clock(min_loop_s)
    k = clock[2]
    t = 0.0
    while t < seconds:
        if phase_fn is not None:
            tr.cycle_phase = phase_fn(t)
        win._advance(mode, unit, win._music_clock(min_loop_s), 0.05, dt)
        t += dt
    nominal = seconds / (k * cycle_seconds)
    return (win.path_s[mode] / unit) / nominal, k


class TravelWhenThePhaseStalls(unittest.TestCase):
    def test_long_cycles_keep_travelling(self):
        # Repeat lengths past ~1.3 s are where an uncapped pull can
        # match the whole cycle rate and hold the path where it is.
        for cycle_seconds in (2.0, 2.4, 3.0):
            frac, k = travel_fraction(cycle_seconds, main.MODE3_MIN_LOOP_S,
                                      unit=2 * math.pi, mode=3)
            self.assertGreaterEqual(
                frac, 0.45,
                f"phase stalled on a {cycle_seconds}s repeat (bar = {k} "
                f"repeats): path travelled {frac:.2%} of its bar pace")

    def test_the_logged_sessions_cycle_keeps_travelling(self):
        # The 2026-11-06 session ran on a ~1.23 s repeat; its log's
        # stalled rows must not cost the path its pace either.
        frac, _ = travel_fraction(1.23, main.MODE3_MIN_LOOP_S,
                                  unit=2 * math.pi, mode=3)
        self.assertGreaterEqual(frac, 0.45,
                                f"path travelled {frac:.2%} of its bar pace")

    def test_travel_matches_the_clock_when_the_phase_is_healthy(self):
        # Control: with the phase advancing the way rhythm.py's does,
        # travel is the bar pace itself, neither held back nor hurried.
        for cycle_seconds, min_loop in ((1.3, main.MODE1_MIN_LOOP_S),
                                        (2.4, main.MODE3_MIN_LOOP_S)):
            frac, _ = travel_fraction(
                cycle_seconds, min_loop, seconds=60.0,
                unit=2 * math.pi, mode=3,
                phase_fn=lambda t, cs=cycle_seconds: (t / cs) % 1.0)
            self.assertTrue(0.9 <= frac <= 1.1,
                            f"healthy {cycle_seconds}s repeat: path "
                            f"travelled {frac:.2%} of its bar pace")


class Mode5WhenThePhaseStalls(unittest.TestCase):
    def test_branches_keep_coming(self):
        tr = FakeTracker(cycle_len=4, cycle_seconds=2.4)
        tr.cycle_phase = 0.83    # stalled estimate: never advanced
        win = make_window(tr)
        win.mode = 5
        win.rabbit = False
        win.kick = 0.0
        win.flower_arms = 0
        win.farey = phi_paths.FareyWalk(q_max=main.FAREY_Q_MAX)
        win.farey_f = 0.0
        win.farey_kicks = 0
        steps, prev = 0, win.farey.cur
        t, dt = 0.0, 0.02
        while t < 60.0:
            win._update_path(dt, 0.5, False)
            t += dt
            if win.farey.cur != prev:
                prev = win.farey.cur
                steps += 1
        # Bar = 3 repeats x 2.4 s = 7.2 s, so ~8 branches at full pace;
        # at the guaranteed minimum pace, half that. A held path takes
        # none at all.
        self.assertGreaterEqual(steps, 3,
                                f"only {steps} branches in 60 s on a "
                                f"stalled phase")


if __name__ == "__main__":
    unittest.main()
