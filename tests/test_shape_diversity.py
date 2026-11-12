"""The picture has to keep changing, and the math pattern has to matter.

The complaint this covers, in the words it arrived in: it "instantly
inflates into a spiral and doesn't do much from there". Measured over
three minutes of a full-production mix, that is what the numbers said
too -- the fold count, which cuts the plane into that many mirrored and
twisted copies, sat at its ceiling of 6 for 77% of the track and reached
only 3 of its 6 values in the whole three minutes. The arrangement of the
picture is the largest thing on screen, and it was arriving in the first
second and then holding.

The cause is that the fold count was a continuous readout of one slow
quantity -- the share of the spectrum above the bass -- and real music
keeps that quantity pinned near the top of the window it was mapped
through. Within a single track it moved over 0.58..0.72 out of 0..1.

So the intent changes here, deliberately and not by weakening anything:
the arrangement is no longer a dial soldered to one measurement. It is
chosen, at the moments the music gives a boundary, out of a set that the
*mode* decides, by a key that several of the song's qualities contribute
to. That is what "more diverse shapes from a combination of the song and
the math pattern" has to mean mechanically.
"""

import collections
import unittest

from tests import synth


class ArrangementMoves(unittest.TestCase):
    """Driven over the same recorded music for every mode, so differences
    between modes are the modes and not the audio."""

    SECONDS = 180.0

    @classmethod
    def setUpClass(cls):
        # The analysis is the slow part, so it runs once for the whole class.
        cls.rows = synth.record(cls.SECONDS)
        cls.runs = {m: synth.drive(cls.rows, m) for m in sorted(__import__("main").MODE_NAMES)}

    def share_of_time_at_the_commonest(self, folds):
        counts = collections.Counter(folds)
        return counts.most_common(1)[0][1] / len(folds)

    def test_the_arrangement_does_not_park_at_one_value(self):
        """No single fold count may hold most of a track. 55% is already
        generous -- a verse/chorus/bridge track spends about a third of
        itself in each -- and the measured failure was 77%."""
        for mode, run in self.runs.items():
            folds = run["folds"]
            share = self.share_of_time_at_the_commonest(folds)
            with self.subTest(mode=mode):
                self.assertLess(share, 0.55,
                                f"mode {mode}: one fold count held {100*share:.0f}% "
                                f"of the track ({collections.Counter(folds)})")

    def test_the_arrangement_uses_more_than_a_corner_of_its_range(self):
        """Three values out of six over three minutes is the measured
        failure. A track with three distinct sections should reach at
        least four."""
        for mode, run in self.runs.items():
            distinct = sorted(set(run["folds"]))
            with self.subTest(mode=mode):
                self.assertGreaterEqual(len(distinct), 4,
                                        f"mode {mode}: only reached folds {distinct}")


if __name__ == "__main__":
    unittest.main()
