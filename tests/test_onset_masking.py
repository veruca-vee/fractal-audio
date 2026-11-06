"""Onsets have to survive the rest of the mix.

Every "the song changes the shape" event in main.py hangs off the kick:
mode 3 cuts the power on it, mode 5 takes its Stern-Brocot branch on its
parity, mode 7 cuts the map on it, mode 2 blows the flower out of its bulb
with it. If the kick is not heard, all of that variety is dead and the
picture holds whatever shape it reached.

It was dead. The 2026-11-06 session logs have `kick_per_s 0.0` for all
594 s of one session, and a synthesised 4-on-the-floor mix with a bass
line under it scored 0 kicks and 0 snares against 461 hats -- while the
hats, which have the 7-14 kHz band to themselves, came through fine.

The detector measures a band's *level* against its own slow average. A
sustained bass note lives in the same 40-150 Hz band as the kick and
dominates the mean, so the kick's transient barely moves it. The band is
masked, and the fix is to measure the thing a kick has and a held note
does not: a sharp positive change.

Synthesised, not real music, so the expected count is arguable from first
principles: we put N kicks in, we should get about N out.
"""

import math
import unittest

import numpy as np

import rhythm

SR, BLOCK = 44100, 1024


def pink(n, rng, amp):
    """A noise floor with a 1/sqrt(f) tilt, which every recording has."""
    w = rng.standard_normal(n)
    s = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    y = np.fft.irfft(s / np.sqrt(np.maximum(f, 20.0)), n)
    return amp * y / (np.abs(y).max() + 1e-9)


def harmonic_stack(t, f0, partials, amp):
    return sum(amp / (k ** 0.8) * np.sin(2 * np.pi * f0 * k * t)
               for k in range(1, partials + 1))


def kick_at(t, period):
    """A kick drum: a pitch sweep from ~165 Hz down to 45 Hz under a fast
    decay, which is what a kick is."""
    ph = t % period
    return 0.9 * np.exp(-ph * 28.0) * np.sin(
        2 * np.pi * (120.0 * np.exp(-ph * 12.0) + 45.0) * ph)


def snare_at(t, period, offset, rng):
    ph = (t - offset) % period
    return 0.5 * np.exp(-ph * 40.0) * rng.standard_normal(t.size)


def run(sig, seconds):
    """Feed `sig(t, rng)` to a tracker block by block; return the tracker."""
    rng = np.random.default_rng(3)
    tr = rhythm.MusicTracker(SR, BLOCK)
    for i in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + i * BLOCK) / SR
        tr.process(sig(t, rng))
    return tr


class KickUnderBass(unittest.TestCase):
    SECONDS = 30.0
    PERIOD = 1.0          # a kick on every beat: 30 of them in 30 s

    def test_kick_is_found_with_nothing_in_its_way(self):
        """The control. A bare kick track must of course be heard."""
        tr = run(lambda t, rng: kick_at(t, self.PERIOD) + pink(t.size, rng, 0.02),
                 self.SECONDS)
        expected = self.SECONDS / self.PERIOD
        self.assertGreater(tr.n_kick, 0.75 * expected,
                           f"a bare kick track: heard {tr.n_kick} of {expected:.0f}")

    def test_kick_is_found_under_a_sustained_bassline(self):
        """The real case. A bass line sits in the kick's own band; the kick
        is still a transient and still has to be heard."""
        def sig(t, rng):
            bass = 0.5 * np.sin(2 * np.pi * 55.0 * t) + 0.25 * np.sin(2 * np.pi * 110.0 * t)
            return kick_at(t, self.PERIOD) + bass + pink(t.size, rng, 0.04)
        tr = run(sig, self.SECONDS)
        expected = self.SECONDS / self.PERIOD
        self.assertGreater(tr.n_kick, 0.6 * expected,
                           f"kick under a bassline: heard {tr.n_kick} of {expected:.0f}")

    def test_kick_is_found_in_a_full_mix(self):
        """And under everything else as well: bass, voice, guitar, pad,
        hats and a backbeat snare."""
        def sig(t, rng):
            bass = 0.5 * np.sin(2 * np.pi * 55.0 * t)
            hat = 0.18 * np.exp(-(t % 0.25) * 120.0) * rng.standard_normal(t.size)
            return (kick_at(t, self.PERIOD) + bass + hat
                    + snare_at(t, self.PERIOD, 0.5, rng)
                    + harmonic_stack(t, 220.0, 6, 0.20)
                    + harmonic_stack(t, 330.0, 8, 0.16)
                    + harmonic_stack(t, 165.0, 10, 0.12)
                    + pink(t.size, rng, 0.06))
        tr = run(sig, self.SECONDS)
        expected = self.SECONDS / self.PERIOD
        self.assertGreater(tr.n_kick, 0.5 * expected,
                           f"kick in a full mix: heard {tr.n_kick} of {expected:.0f}")

    def test_a_held_bass_note_is_not_a_kick(self):
        """The other half of the contract: the detector must not simply
        fire on bass energy. A bassline with no drums at all has no kicks
        in it, and a flux detector that counts every note onset as a kick
        would pass the test above while being just as wrong."""
        def sig(t, rng):
            # Long held notes, one every 4 s, no percussion anywhere.
            step = (t // 4.0).astype(int)
            f0 = np.where(step % 2 == 0, 55.0, 73.0)
            return 0.6 * np.sin(2 * np.pi * f0 * t) + pink(t.size, rng, 0.03)
        tr = run(sig, self.SECONDS)
        # Eight note changes in 30 s; a handful of those may read as
        # transients, but nothing like a beat.
        self.assertLess(tr.n_kick, 12,
                        f"a drumless bassline scored {tr.n_kick} kicks")


class SnareUnderMix(unittest.TestCase):
    """The snare drives the colour flip and the hue step, and it was just as
    masked as the kick: 0 heard out of 30 in the full mix above. It needs a
    low-mid crack *and* a high-band crack together -- that part of the
    design is right, it is what makes it mean "broadband hit" rather than
    "a guitar note" -- but both halves were watching levels that a guitar
    and a cymbal wash keep high.
    """

    SECONDS = 30.0
    PERIOD = 1.0

    def test_snare_is_found_in_a_full_mix(self):
        def sig(t, rng):
            bass = 0.5 * np.sin(2 * np.pi * 55.0 * t)
            hat = 0.18 * np.exp(-(t % 0.25) * 120.0) * rng.standard_normal(t.size)
            return (kick_at(t, self.PERIOD) + bass + hat
                    + snare_at(t, self.PERIOD, 0.5, rng)
                    + harmonic_stack(t, 220.0, 6, 0.20)
                    + harmonic_stack(t, 330.0, 8, 0.16)
                    + pink(t.size, rng, 0.06))
        tr = run(sig, self.SECONDS)
        expected = self.SECONDS / self.PERIOD
        self.assertGreater(tr.n_snare, 0.5 * expected,
                           f"snare in a full mix: heard {tr.n_snare} of {expected:.0f}")

    def test_a_sustained_guitar_chord_is_not_a_kick_either(self):
        """A held chord has noisy low bins, and a flux detector reading them
        can invent a beat out of the noise floor. False kicks are not a
        harmless error here: a kick cuts the power in mode 3, takes the
        branch in mode 5 and cuts the map in mode 7, so inventing them
        makes the picture twitch to music that is not playing."""
        def sig(t, rng):
            return (harmonic_stack(t, 165.0, 14, 0.25)
                    + harmonic_stack(t, 220.0, 14, 0.22)
                    + harmonic_stack(t, 330.0, 14, 0.18)
                    + pink(t.size, rng, 0.03))
        tr = run(sig, self.SECONDS)
        self.assertLess(tr.n_kick, 15,
                        f"a held chord scored {tr.n_kick} kicks in {self.SECONDS:.0f}s")

    def test_a_sustained_guitar_chord_is_not_a_snare(self):
        """The guard. A snare is a broadband *hit*; held strings are not,
        however much low-mid and treble they carry."""
        def sig(t, rng):
            return (harmonic_stack(t, 165.0, 14, 0.25)
                    + harmonic_stack(t, 220.0, 14, 0.22)
                    + harmonic_stack(t, 330.0, 14, 0.18)
                    + pink(t.size, rng, 0.03))
        tr = run(sig, self.SECONDS)
        self.assertLess(tr.n_snare, 10,
                        f"a held chord scored {tr.n_snare} snares")


if __name__ == "__main__":
    unittest.main()
