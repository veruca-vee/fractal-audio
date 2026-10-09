"""The vocal contour against synthesised audio with a known answer.

A held note does not change pitch, so whatever the contour does during one
is measurable without arguing about what the "right" value is: it has to
stay put.
"""

import math
import unittest

import numpy as np

from rhythm import MusicTracker

SR = 44100
BLOCK = 1024


def harmonic_stack(t, f0, amps):
    """A vowel-ish voice: a fundamental plus harmonics at set amplitudes."""
    sig = np.zeros_like(t)
    for k, a in enumerate(amps, start=1):
        sig += a * np.sin(2 * math.pi * f0 * k * t)
    return sig


def drum_bed(t, rng, period=0.5):
    """A low sine plus a short noise burst every `period` seconds."""
    sig = 0.4 * np.sin(2 * math.pi * 55.0 * t)
    for start in np.arange(0.0, t[-1], period):
        i = int(start * SR)
        n = int(0.06 * SR)
        env = np.exp(-np.arange(n) / (0.015 * SR))
        chunk = 0.25 * rng.standard_normal(n) * env
        sig[i:i + len(chunk)] += chunk[:len(sig) - i]
    return sig


def band_noise(n, lo, hi, rng):
    """Noise confined to lo..hi Hz, normalised to +-1.

    Stands in for low-mid accompaniment -- pad, strings, electric piano,
    room tone -- which fills a region of the spectrum rather than adding
    one more clean partial.
    """
    x = np.fft.rfft(rng.standard_normal(n))
    fr = np.fft.rfftfreq(n, 1.0 / SR)
    x[(fr < lo) | (fr > hi)] = 0.0
    y = np.fft.irfft(x, n)
    return y / (np.abs(y).max() or 1.0)


def feed(tracker, sig):
    """Run `sig` through the tracker a block at a time, sampling the contour.

    Returns a list of (seconds, vocal_pitch).
    """
    out = []
    for i in range(0, len(sig) - BLOCK, BLOCK):
        tracker.process(sig[i:i + BLOCK].astype(np.float32))
        out.append(((i + BLOCK) / SR, tracker.vocal_pitch))
    return out


def held_note_with_pad_entry(seed=7):
    """220 Hz held 1..6 s; low-mid accompaniment enters at 3 s.

    The pad covers 150-340 Hz, so it sits around the sung fundamental and
    nowhere near its harmonics. Nothing about the voice changes: same
    pitch, same amplitudes, right through.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(7.0 * SR)) / SR
    sig = drum_bed(t, rng)

    voice = (t >= 1.0) & (t < 6.0)
    sig[voice] += harmonic_stack(t[voice], 220.0, [0.30, 1.00, 0.45, 0.20])

    pad = t >= 3.0
    sig[pad] += 0.5 * band_noise(int(pad.sum()), 150.0, 340.0, rng)
    return sig


class HeldNote(unittest.TestCase):
    def test_contour_holds_when_accompaniment_enters_mid_note(self):
        """A note held at one pitch keeps one contour value.

        Halfway through the note a low-mid pad enters, crowding the
        fundamental but not its harmonics. The sung pitch never changes,
        so the contour must not move -- least of all by the ~0.41 of the
        range that one octave spans here.
        """
        trace = feed(MusicTracker(SR, BLOCK), held_note_with_pad_entry())
        held = [v for (sec, v) in trace if 1.8 <= sec <= 5.8]
        self.assertTrue(held, "no contour samples during the held note")
        spread = max(held) - min(held)
        self.assertLess(
            spread, 0.10,
            "contour moved {:.3f} of its range during a held note "
            "(min {:.3f}, max {:.3f}); one octave is 0.41".format(
                spread, min(held), max(held)))

    def test_an_octave_leap_in_the_voice_lifts_the_contour(self):
        """220 then 440, with the pad playing under both.

        A steadiness test cannot catch the opposite error -- a detector
        that always takes the octave below reads a perfectly steady 0.5,
        because the register centre initialises to whatever it first
        accepted. Only a voice that actually moves shows whether the
        contour follows the right partial, so this one leaps an octave and
        the contour has to leap with it.
        """
        rng = np.random.default_rng(11)
        t = np.arange(int(8.0 * SR)) / SR
        sig = drum_bed(t, rng)

        low = (t >= 1.0) & (t < 4.0)
        high = (t >= 4.0) & (t < 7.5)
        sig[low] += harmonic_stack(t[low], 220.0, [0.30, 1.00, 0.45, 0.20])
        sig[high] += harmonic_stack(t[high], 440.0, [0.30, 1.00, 0.45, 0.20])

        pad = t >= 0.5
        sig[pad] += 0.5 * band_noise(int(pad.sum()), 150.0, 340.0, rng)

        trace = feed(MusicTracker(SR, BLOCK), sig)
        before = [v for (sec, v) in trace if 3.0 <= sec <= 3.9]
        after = [v for (sec, v) in trace if 6.0 <= sec <= 7.3]
        self.assertTrue(before and after, "missing contour samples")
        rise = sum(after) / len(after) - sum(before) / len(before)
        self.assertGreater(
            rise, 0.25,
            "contour rose only {:.3f} over an octave leap "
            "({:.3f} -> {:.3f}); an octave is 0.41 of the range".format(
                rise, sum(before) / len(before), sum(after) / len(after)))


if __name__ == "__main__":
    unittest.main()
