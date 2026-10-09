"""Asking for a track straight after building the player must work.

`loadlist` is asynchronous like every other mpv command, so a
`playlist-play-index` sent immediately after it can arrive while the
playlist is still empty, where mpv silently drops it and plays the first
entry instead. Resuming into the middle of a library is exactly that call,
and whether it works was decided by which message won a race.

main.py calls `wait_ready()` before `play_index` to avoid it, but a fix in
one caller is not a fix: the next caller races again, which is how this came
back. The guarantee belongs to the player.
"""

import shutil
import time
import unittest
from pathlib import Path

import player

LIBRARY = Path(__file__).resolve().parent / "fixtures" / "library"


def wait_until(predicate, timeout=6.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@unittest.skipIf(shutil.which("mpv") is None, "mpv is not on PATH")
@unittest.skipIf(not LIBRARY.is_dir(), f"no test library at {LIBRARY}")
class PlaylistStart(unittest.TestCase):
    def setUp(self):
        self.tracks = player.build_playlist([LIBRARY])
        self.assertGreaterEqual(len(self.tracks), 3, "fixture library too small")
        self.player = None

    def tearDown(self):
        if self.player is not None:
            self.player.close()

    def test_a_track_asked_for_immediately_is_the_one_that_plays(self):
        self.player = player.MpvPlayer(self.tracks, volume=0, start_paused=True)
        self.assertIsNone(self.player.failed, self.player.failed)
        # No wait_ready() on purpose: this is the call as a caller would
        # naturally write it, and it is the one that used to race.
        self.player.play_index(1)
        # Wait for the track we asked for, not merely for *a* track: mpv
        # starts entry 0 by itself and the player then moves off it, so the
        # first thing to load is the wrong one by design.
        wait_until(lambda: self.player.current_path == self.tracks[1])
        self.assertEqual(self.player.current_path, self.tracks[1],
                         f"asked for track 2, got {self.player.current_path}")

    def test_a_position_asked_for_immediately_is_seeked_to(self):
        self.player = player.MpvPlayer(self.tracks, volume=0, start_paused=True)
        self.assertIsNone(self.player.failed, self.player.failed)
        self.player.play_index(2, seek=4.0)
        self.assertTrue(wait_until(lambda: self.player.position > 3.5),
                        f"never reached the position (got {self.player.position:.1f})")
        self.assertEqual(self.player.current_path, self.tracks[2])


if __name__ == "__main__":
    unittest.main()
