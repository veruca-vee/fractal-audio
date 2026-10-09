"""The edge light reacts to the hi-hats.

The hats are the fastest thing the tracker hears, so they are the one layer
that can flick something without it reading as a wobble. Everything here is
rendered headlessly into an FBO and asserted on as pixel statistics: how
bright the border strip is, and that the middle of the screen is untouched.
"""

import unittest

import numpy as np

try:
    import moderngl
    CTX = moderngl.create_standalone_context(backend="egl", require=330)
except Exception as exc:                                  # pragma: no cover
    CTX = None
    REASON = f"no standalone GL context: {exc}"

import main
import overlay

WIDTH, HEIGHT = 320, 200


class FakePlayer:
    """Only what the overlay reads off it."""

    alive = True
    paused = False
    duration = 200.0
    position = 100.0

    def title(self):
        return "a track"


@unittest.skipIf(CTX is None, globals().get("REASON", "no GL"))
class EdgeLightHats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = CTX
        cls.tex = cls.ctx.texture((WIDTH, HEIGHT), 3)
        cls.fbo = cls.ctx.framebuffer(color_attachments=[cls.tex])
        cls.vbo = cls.ctx.buffer(
            np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        cls.overlay = overlay.NowPlayingOverlay(cls.ctx, cls.vbo)

    def frame(self, **kw):
        """One drawn frame as a (HEIGHT, WIDTH) float array of brightness."""
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0)
        self.overlay.draw(FakePlayer(), 0.0, (WIDTH, HEIGHT), (WIDTH, HEIGHT), **kw)
        buf = np.frombuffer(self.fbo.read(components=3), dtype=np.uint8)
        return buf.reshape(HEIGHT, WIDTH, 3).astype(float).mean(axis=2) / 255.0

    def border(self, img):
        """Mean brightness of the strip the light lives in."""
        band = int(overlay.EDGE_WIDTH_PX * 2)
        mask = np.zeros(img.shape, dtype=bool)
        mask[:band, :] = mask[-band:, :] = True
        mask[:, :band] = mask[:, -band:] = True
        return img[mask].mean()

    def test_a_hat_hit_brightens_the_edge_light(self):
        quiet = self.border(self.frame(hat=0.0))
        flick = self.border(self.frame(hat=1.0))
        self.assertGreater(flick, quiet * 1.15,
                           f"border {quiet:.4f} quiet vs {flick:.4f} on a hat")

    def test_the_hat_flick_leaves_the_middle_of_the_screen_alone(self):
        """It is an edge light, not a flash: the fractal keeps the frame."""
        img = self.frame(hat=1.0)
        y0, y1 = int(HEIGHT * 0.25), int(HEIGHT * 0.75)
        x0, x1 = int(WIDTH * 0.25), int(WIDTH * 0.75)
        middle = img[y0:y1, x0:x1].max()
        self.assertLess(middle, 0.01, f"middle lit to {middle:.4f} on a hat")

    def test_the_flick_fades_with_the_hat_as_it_decays(self):
        """A hit decays over a few frames; the border should follow it down
        rather than switch off, or it reads as a strobe."""
        quiet, half, full = (self.border(self.frame(hat=h)) for h in (0.0, 0.5, 1.0))
        self.assertGreater(half, quiet, f"half {half:.4f} vs quiet {quiet:.4f}")
        self.assertGreater(full, half, f"full {full:.4f} vs half {half:.4f}")


if __name__ == "__main__":
    unittest.main()


class RecordingOverlay:
    """Stands in for the real overlay and remembers how it was drawn."""

    def __init__(self):
        self.calls = []

    def draw(self, player, now, size, buffer_size, hat=0.0):
        self.calls.append(hat)


class FakeWindow:
    size = (320, 200)
    buffer_size = (320, 200)


class HatReachesTheEdgeLight(unittest.TestCase):
    """The complaint being fixed: the hats moved the fractal and nothing else."""

    def window(self):
        win = main.FractalWindow.__new__(main.FractalWindow)
        win.overlay = RecordingOverlay()
        win.player = FakePlayer()
        win.wnd = FakeWindow()
        win.hat = 0.0
        return win

    def test_the_overlay_is_drawn_with_the_current_hat_strength(self):
        win = self.window()
        win.hat = 0.8
        win._draw_overlay(1.0)
        self.assertEqual(win.overlay.calls, [0.8])

    def test_no_hats_leave_the_edge_light_as_it_was(self):
        win = self.window()
        win._draw_overlay(1.0)
        self.assertEqual(win.overlay.calls, [0.0])
