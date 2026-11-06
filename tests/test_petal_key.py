"""Auditioning the vocal shape's petal count by hand.

The petal count normally follows the mode's own symmetry, which is right
for a performance and useless for choosing: you cannot compare 5 petals
with 8 if the only way to reach 8 is to wait for the music to take you
there. A key that steps through the counts, and hands them back to the
music when it runs out, is what auditioning them needs.

Built without __init__ -- main.py forces fullscreen and opens an audio
stream -- so only the attributes the keyboard and _update_voice touch are
set here. Key codes are the real pyglet ones, not stand-ins.
"""

import types
import unittest

from moderngl_window.context.pyglet.keys import Keys

import main


def settle(win, seconds=60.0, dt=0.1):
    """Run _update_voice long enough for the eased count to arrive."""
    for _ in range(int(seconds / dt)):
        win._update_voice(dt)
    return win.voice_petals


class PetalKey(unittest.TestCase):
    def setUp(self):
        self.win = main.FractalWindow.__new__(main.FractalWindow)
        w = self.win
        w.wnd = types.SimpleNamespace(keys=Keys)
        w.player = None
        w.toasts = []
        w.show_toast = w.toasts.append
        w.tr = types.SimpleNamespace(vocal=0.0)
        w.mode = 3           # a power mode: the petal count follows u_power
        w.power = 8
        w.flower_arms = 0
        w.voice_out = 0.0
        w.voice_petals = 5.0
        w.petals_held = None
        w.track_menu = None   # merged key handler checks the modal track menu first
        w.now = 0.0

    def press(self, key):
        self.win.on_key_event(key, Keys.ACTION_PRESS, 0)

    def test_k_holds_the_petal_count_away_from_what_the_mode_would_pick(self):
        self.assertAlmostEqual(settle(self.win), 8.0, places=2,
                               msg="mode 3 with power 8 should ask for 8 petals")

        self.press(Keys.K)
        held = settle(self.win)
        self.assertNotAlmostEqual(held, 8.0, places=1,
                                  msg=f"k left the count at the mode's {held}")
        self.assertGreaterEqual(held, 2.0, f"{held} petals is not a shape")

        # The music moving on must not take the held count with it.
        self.win.power = 3
        self.assertAlmostEqual(settle(self.win), held, places=2,
                               msg="the held count drifted when the mode changed")

    def test_k_again_steps_on_to_a_different_count(self):
        self.press(Keys.K)
        first = settle(self.win)
        self.press(Keys.K)
        second = settle(self.win)
        self.assertNotAlmostEqual(first, second, places=1,
                                  msg=f"both presses gave {first} petals")

    def follows_the_music(self):
        """Is the count the music's again?

        Asked by moving what the mode would ask for and seeing whether the
        count goes with it. Comparing against one particular number would
        not do: every count the mode can ask for is also one of the counts
        `k` offers, so only *responding to a change* tells the two apart.
        """
        w = self.win
        w.power = 8
        with_eight = settle(w)
        w.power = 5
        with_five = settle(w)
        w.power = 8
        settle(w)
        return abs(with_eight - with_five) > 0.5

    def test_cycling_right_round_hands_the_count_back_to_the_music(self):
        """Auditioning has to be leaveable: keep pressing and the mode's own
        symmetry comes back, without having to restart."""
        self.assertTrue(self.follows_the_music(), "did not start on automatic")

        seen = set()
        for _ in range(40):
            self.press(Keys.K)
            seen.add(round(settle(self.win), 3))
            if self.follows_the_music():
                break
        else:
            self.fail(f"never handed the count back; saw {sorted(seen)}")
        self.assertGreater(len(seen), 3,
                           f"too few counts to audition: {sorted(seen)}")

    def test_each_press_says_which_count_it_is_showing(self):
        """Auditioning without being told which one you are looking at is
        guesswork, and the morph takes seconds, so the number has to be on
        screen."""
        self.press(Keys.K)
        self.assertTrue(self.win.toasts, "pressing k said nothing")
        first = self.win.toasts[-1]
        self.assertIn(str(int(self.win.petals_held)), first,
                      f"toast {first!r} does not name the count")

        self.press(Keys.K)
        self.assertNotEqual(self.win.toasts[-1], first,
                            "the second count got the first one's toast")

        while self.win.petals_held is not None:
            self.press(Keys.K)
        self.assertIn("auto", self.win.toasts[-1].lower(),
                      f"toast {self.win.toasts[-1]!r} does not say it is back "
                      "on automatic")
