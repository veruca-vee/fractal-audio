"""Beat-locking: the path must advance on the music's cycle grid.

The repeat length sets the pace of c's path (README, "What follows
what"): modes 1 and 4 step one sequence term per bar, modes 3, 6 and 7
make one loop per bar, mode 5 takes one Stern-Brocot branch per bar,
where a bar is the repeat stretched to a whole number of repeats at
least a mode-specific minimum long.

A pace is a rate AND a phase. A term that steps at the right average
rate but lands in the middle of a bar is not following the beat, it is
drifting past it -- which is what these tests catch. They drive the
window with a fake tracker whose cycle phase advances steadily, the
way rhythm.py's does, and check the path stays on the grid the phase
defines: the path's position within its current cycle must match the
tracker's cycle phase, so a loop completes on a cycle boundary.

Built without __init__, like the other window tests -- main.py forces
fullscreen and opens an audio stream -- so only the attributes the
clock, the advance and the mode-5 branch touch are set here.
"""

import math
import types
import unittest

import main
import phi_paths
from rhythm import wrap_half


class FakeTracker:
    """Just the rhythm quantities main.py reads. The test advances
    cycle_phase by hand, so the beat is exactly where the test says."""

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


class MusicClockContract(unittest.TestCase):
    def test_clock_carries_the_phase_for_a_multi_repeat_bar(self):
        # A 1.3 s repeat under mode 1's 6 s minimum loop is a bar of
        # k = 5 repeats. The grid lives in the phase, so the clock has
        # to hand it over, not drop it whenever k > 1.
        tr = FakeTracker(cycle_seconds=1.3)
        tr.cycle_phase = 0.42
        win = make_window(tr)
        clock = win._music_clock(main.MODE1_MIN_LOOP_S)
        self.assertIsNotNone(clock)
        self.assertEqual(len(clock), 3,
                         "clock should be (loops/s, cycle phase, repeats per loop)")
        rate, phase, k = clock
        self.assertEqual(k, 5)
        self.assertAlmostEqual(rate, 1.0 / (5 * 1.3))
        self.assertEqual(phase, 0.42)

    def test_no_repeat_no_clock(self):
        win = make_window(FakeTracker(cycle_len=0))
        self.assertIsNone(win._music_clock(main.MODE1_MIN_LOOP_S))


class AdvanceLocksToTheGrid(unittest.TestCase):
    def run_lock(self, cycle_seconds, min_loop_s, unit=1.0, mode=1,
                 seconds=40.0, dt=0.02):
        """Start the path half a cycle off the grid and measure the
        worst phase error over the last two bars, once any lock has
        had time to settle."""
        tr = FakeTracker(cycle_seconds=cycle_seconds)
        win = make_window(tr)
        k = max(1, math.ceil(min_loop_s / cycle_seconds))
        win.path_s[mode] = unit * 0.5 / k   # cycle-scale phase 0.5: dead off-grid
        t, worst = 0.0, 0.0
        while t < seconds:
            tr.cycle_phase = (t / cycle_seconds) % 1.0
            win._advance(mode, unit, win._music_clock(min_loop_s), 0.05, dt)
            t += dt
            if t > seconds - 2 * k * cycle_seconds:
                cyc = (win.path_s[mode] / unit * k) % 1.0
                worst = max(worst, abs(wrap_half(tr.cycle_phase - cyc)))
        return worst

    def test_term_steps_lock_when_the_bar_is_several_repeats(self):
        worst = self.run_lock(cycle_seconds=1.3, min_loop_s=main.MODE1_MIN_LOOP_S)
        self.assertLess(worst, 0.05,
                        f"path drifted {worst:.2f} of a cycle off the beat grid")

    def test_loops_lock_when_the_bar_is_several_repeats(self):
        # The loop modes (3, 6, 7) advance in radians: one loop per bar.
        worst = self.run_lock(cycle_seconds=1.05, min_loop_s=main.MODE3_MIN_LOOP_S,
                              unit=2 * math.pi, mode=3)
        self.assertLess(worst, 0.05,
                        f"loop drifted {worst:.2f} of a cycle off the beat grid")

    def test_loop_locks_when_the_bar_is_one_repeat(self):
        # An 8 s repeat against a 6 s minimum is a bar of one repeat:
        # the case that already locked, kept as a control.
        worst = self.run_lock(cycle_seconds=8.0, min_loop_s=main.MODE1_MIN_LOOP_S)
        self.assertLess(worst, 0.05,
                        f"path drifted {worst:.2f} of a cycle off the beat grid")

    def test_no_clock_still_runs_at_the_fallback_pace(self):
        win = make_window(FakeTracker(cycle_len=0))
        for _ in range(100):
            win._advance(1, 1.0, None, 0.2, 0.01)
        self.assertAlmostEqual(win.path_s[1], 0.2)


class FareyBranchesOnTheBar(unittest.TestCase):
    def make_mode5(self, tr):
        win = make_window(tr)
        win.mode = 5
        win.rabbit = False
        win.kick = 0.0
        win.flower_arms = 0
        win.farey = phi_paths.FareyWalk(q_max=main.FAREY_Q_MAX)
        win.farey_f = 0.0
        win.farey_kicks = 0
        return win

    def run_mode5(self, tr, seconds, dt=0.02, phase0=0.0):
        """Returns the cycle phases at which branches were taken."""
        win = self.make_mode5(tr)
        t, steps, prev = 0.0, [], win.farey.cur
        while t < seconds:
            tr.cycle_phase = (phase0 + t / tr.cycle_seconds) % 1.0
            win._update_path(dt, 0.5, False)
            t += dt
            if win.farey.cur != prev:
                prev = win.farey.cur
                steps.append(tr.cycle_phase)
        return steps

    def test_branches_land_on_bar_boundaries(self):
        # Repeat 1.0 s, mode 5's minimum step 3 s: a bar is 3 cycles,
        # so its boundary is a cycle boundary (phase ~0). The music
        # starts mid-cycle, which is exactly when a free-running glide
        # and the beat grid disagree.
        tr = FakeTracker(cycle_len=4, cycle_seconds=1.0)
        steps = self.run_mode5(tr, seconds=30.0, phase0=0.6)
        bars = 30.0 / 3.0
        self.assertGreaterEqual(len(steps), bars - 2,
                                "about one branch per bar should be taken")
        self.assertLessEqual(len(steps), bars + 2,
                             "about one branch per bar should be taken")
        for ph in steps[2:]:   # the first bars are the lock settling
            self.assertLess(min(ph, 1.0 - ph), 0.1,
                            f"branch landed at cycle phase {ph:.2f}, off the bar grid")

    def test_mode_5_without_a_clock_still_steps_at_the_fallback_pace(self):
        tr = FakeTracker(cycle_len=0, cycle_seconds=1.0)
        steps = self.run_mode5(tr, seconds=10.0)
        # amp 0.5 -> loud 0.25 -> 0.12 + 0.4 * 0.25 = 0.22 bars/s -> ~2 steps.
        self.assertTrue(1 <= len(steps) <= 3,
                        f"expected ~2 fallback steps in 10 s, got {len(steps)}")


if __name__ == "__main__":
    unittest.main()
