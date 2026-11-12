"""Fold response: the fold must follow the music, not pin at the top.

The fold count (and the twist, which is fold_amount() scaled) is the
picture's scaffold: when it stops moving, everything else reads as a
static spiral that only bumps with the kick and spins slowly -- which
is what Vee reported on 2026-11-06 and again on 2026-11-11.

Measured cause (offline, with this repo's own MusicTracker, on real
tracks from her library): the arrangement density the fold was moved
to in f1de2b2 is a near-constant per track. On Stone Temple Pilots'
"Interstate Love Song" its per-10 s median sits at 0.70-0.74 for the
whole body of the song, so the old absolute window (0.35 -> 0.70)
saturated within seconds and the fold spent 88% of the track at 6,
with one unbroken 69 s stretch at 6. The vocal level, by contrast,
swings 0.2 -> 0.8 phrase by phrase on every track measured.

The contract these tests pin down:

- density on its own can never pin the fold at the top -- a dense
  instrumental passage gets a floor, not the full spiral;
- nothing inflates instantly: from rest, even a full arrangement
  with the voice up takes seconds to reach the top fold;
- vocal phrases move the fold up and back down at phrase scale;
- silence and sparse material stay whole / low;
- the top fold is still reachable when everything is genuinely up.

Built without __init__, like the other window tests.
"""

import unittest

import main

DT = 1.0 / 60.0


def make_window():
    win = main.FractalWindow.__new__(main.FractalWindow)
    win.now = 0.0
    win.density_slow = 0.0
    win.fold_vocal_slow = 0.0
    win.fold_from = 1
    win.fold_to = 1
    win.fold_mix = 1.0
    win.fold_changed_at = 0.0
    return win


def drive(win, seconds, density, vocal):
    """Step _update_fold at 60 fps. density/vocal may be callables of t."""
    t = 0.0
    while t < seconds:
        d = density(t) if callable(density) else density
        v = vocal(t) if callable(vocal) else vocal
        win.now = t
        win._update_fold(DT, d, v)
        t += DT
    return win


class DensityAloneCannotPinTheFold(unittest.TestCase):
    def test_dense_instrumental_gets_a_floor_not_the_top(self):
        # 0.72 is Interstate Love Song's body density, held for two
        # minutes with no voice at all.
        win = drive(make_window(), 120.0, density=0.72, vocal=0.0)
        self.assertLessEqual(win.fold_to, 4,
                             f"density alone held fold at {win.fold_to}")
        self.assertLess(win.fold_amount(), 0.62,
                        f"density alone held fold_amount at "
                        f"{win.fold_amount():.2f}")


class NoInstantInflation(unittest.TestCase):
    def test_full_signal_from_rest_climbs_over_seconds(self):
        win = make_window()
        drive(win, 1.0, density=0.85, vocal=0.9)
        self.assertLess(win.fold_amount(), 0.5,
                        f"1 s from rest fold_amount already "
                        f"{win.fold_amount():.2f} -- that is the instant "
                        f"inflation Vee sees")
        t_to_top = None
        t = 1.0
        while t < 30.0:
            win.now = t
            win._update_fold(DT, 0.85, 0.9)
            if win.fold_amount() >= 0.9:
                t_to_top = t
                break
            t += DT
        self.assertIsNotNone(t_to_top, "top fold never reached")
        self.assertGreater(t_to_top, 3.0,
                           f"fold_amount hit 0.9 after only {t_to_top:.1f} s")


class VocalPhrasesMoveTheFold(unittest.TestCase):
    def test_phrases_raise_and_lower_the_fold(self):
        # Dense backing throughout; the voice comes and goes in
        # 8-second phrases, the way verses and choruses actually do.
        win = make_window()
        drive(win, 32.0, density=0.72, vocal=0.15)  # settle into the backing
        folds_high, folds_low = [], []
        t = 0.0
        while t < 64.0:
            vocal = 0.85 if (t // 8) % 2 == 0 else 0.15
            win.now = 32.0 + t
            win._update_fold(DT, 0.72, vocal)
            if (t // 8) % 2 == 0:
                folds_high.append(win.fold_to)
            else:
                folds_low.append(win.fold_to)
            t += DT
        self.assertGreaterEqual(max(folds_high), 5,
                                f"voice up, fold only reached "
                                f"{max(folds_high)}")
        self.assertLessEqual(min(folds_low), 3,
                             f"voice gone, fold only fell to "
                             f"{min(folds_low)}")


class QuietMaterialStaysWhole(unittest.TestCase):
    def test_silence_stays_whole(self):
        win = drive(make_window(), 30.0, density=0.0, vocal=0.0)
        self.assertEqual(win.fold_to, 1)
        self.assertEqual(win.fold_amount(), 0.0)

    def test_sparse_material_stays_low(self):
        # Around Nirvana's "Dumb": density ~0.5, voice low in the mix.
        win = drive(make_window(), 60.0, density=0.5, vocal=0.25)
        self.assertLessEqual(win.fold_to, 3,
                             f"sparse material inflated to {win.fold_to}")


class TheTopIsStillReachable(unittest.TestCase):
    def test_everything_up_reaches_six(self):
        win = drive(make_window(), 30.0, density=0.85, vocal=0.85)
        self.assertEqual(win.fold_to, 6,
                         f"full arrangement + voice peaked at {win.fold_to}")


if __name__ == "__main__":
    unittest.main()
