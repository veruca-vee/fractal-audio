"""The voice's shape must not cost the fractal its smoothness.

Two things here were learned the hard way and are worth keeping written
down, because both produced confident numbers that were simply false.

**Timing a draw by the wall clock does not work.** With nothing reading the
framebuffer, the driver is free to notice that the result is never consumed
and skip the work: after a few iterations the loop reports six-figure frame
rates. That is what made the first version of this test flaky -- it passed
whenever the elision kicked in and failed when it didn't, and neither
outcome had anything to do with the shader. Reading the framebuffer back
each frame does force the work, but the transfer then costs several times
the render and swamps what is being measured. A GPU timer query measures the
GPU's own execution of the draw and is immune to both.

**Absolute frame rates are not reproducible here.** The same shader measured
16 ms on an idle machine and 55 ms with other work on the GPU, so any fixed
fps threshold is a coin toss. The ratio between the silent and singing
frames survives that: when the machine is loaded both get slower together.
Under contention the ratio understates the cost, which fails safe -- a
loaded machine cannot produce a spurious failure, only a lenient pass.
"""

import statistics
import unittest

import numpy as np

try:
    import moderngl
    CTX = moderngl.create_standalone_context(backend="egl", require=330)
except Exception as exc:                                  # pragma: no cover
    CTX = None
    REASON = f"no standalone GL context: {exc}"

WIDTH, HEIGHT = 1024, 576      # 1366x768 at --scale 0.75
ITERATIONS = 128               # the --iters default
FRAMES = 18

# The two bounds bracket the cost from both sides, and they need each other.
#
# Most music is not wall-to-wall singing, so a silent frame must not pay for
# the shape: the trap is branched out of the escape loop entirely to make
# that true. On its own, though, "singing costs little" is satisfied just as
# well by putting the trap back in the loop unconditionally and making every
# frame equally slow -- which is the regression this is here to catch. So
# the silent frame also has to be measurably cheaper than the singing one.
# Both numbers come from measuring an idle machine, not from taste. The
# shape costs 21%: a silent frame is 16.1 ms, a singing one 19.4 ms, so
# silent sits at 0.83 of singing. Put the trap back inside the escape loop
# and that becomes 1.0, since every frame then pays alike; 0.90 sits between
# the two with room either side.
SILENT_SAVING_MIN = 0.90      # silent must be at most this fraction of singing
# 1.21x measured. The version before the trap was branched out and given a
# polynomial atan ran 1.93x, so a bound of 2.10 would have watched that whole
# regression go by; 1.50 catches it and still leaves the real figure room.
SINGING_OVERHEAD_MAX = 1.50   # ...and singing at most this multiple of silent


def uniforms(**over):
    base = {
        "u_resolution": (WIDTH, HEIGHT), "u_c": (-0.4, 0.6), "u_power": 2,
        "u_formula": 0, "u_p": (0.0, 0.0), "u_center": (0.0, 0.0), "u_zoom": 1.1,
        "u_amp": 0.4, "u_bass": 0.3, "u_mid": 0.3, "u_treble": 0.3, "u_kick": 0.2,
        "u_snare": 0.0, "u_hat": 0.2, "u_hue": 0.1, "u_max_iter": ITERATIONS,
        "u_time": 3.0, "u_spin": 0.3, "u_fold_a": 1.0, "u_fold_b": 3.0,
        "u_fold_mix": 0.5, "u_twist": 0.5, "u_pal_a": (0.5, 0.5, 0.5),
        "u_pal_b": (0.5, 0.5, 0.5), "u_pal_c": (1.0, 1.0, 1.0),
        "u_pal_d": (0.0, 0.33, 0.67), "u_trap_weight": 0.0, "u_trap_k": 5.0,
        "u_trap_rot": 0.0, "u_trap_radius": 0.4, "u_trap_sharp": 20.0,
        "u_trap_hue": 0.5,
    }
    base.update(over)
    return base


@unittest.skipIf(CTX is None, globals().get("REASON", "no GL"))
class VoiceShapeCost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = CTX
        cls.tex = cls.ctx.texture((WIDTH, HEIGHT), 3)
        cls.fbo = cls.ctx.framebuffer(color_attachments=[cls.tex])
        cls.quad = cls.ctx.buffer(
            np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        with open("julia.vert") as f:
            vert = f.read()
        with open("julia.frag") as f:
            frag = f.read()
        cls.prog = cls.ctx.program(vertex_shader=vert, fragment_shader=frag)
        cls.vao = cls.ctx.simple_vertex_array(cls.prog, cls.quad, "in_position")
        cls.fbo.use()
        for name, value in uniforms().items():
            cls.prog[name].value = value

    def gpu_ms(self, trap_off=False, **over):
        """Median GPU milliseconds for one frame.

        u_time is nudged every frame so no two draws are identical, which
        keeps the driver from serving a cached result.

        The minimum, not the median: a frame can be stalled by anything else
        on the machine, and a shader that has just been compiled stalls for
        its first few regardless. Nothing can make a frame finish faster
        than the GPU can do the work, so the floor is the measurement and
        everything above it is interference.
        """
        settings = uniforms(**over)
        if trap_off:
            settings["u_trap_weight"] = 0.0
        for name, value in settings.items():
            self.prog[name].value = value
        samples = []
        for i in range(FRAMES):
            self.prog["u_time"].value = 3.0 + i * 0.01
            query = self.ctx.query(time=True)
            with query:
                self.vao.render(moderngl.TRIANGLE_STRIP)
            self.ctx.finish()
            samples.append(query.elapsed / 1e6)
        # Six discarded, not two: a freshly compiled shader stalls for
        # several frames, which produced one spurious failure in six runs.
        return min(samples[6:])

    def both(self):
        """Silent and singing, measured back to back so they see the same
        machine. Measured per test rather than once, because the two tests
        want the pair compared, not a stored number from minutes ago."""
        silent = self.gpu_ms(trap_off=True)
        singing = self.gpu_ms(u_trap_weight=0.85, u_trap_rot=0.3)
        return silent, singing

    def test_a_silent_frame_skips_the_voice_shape_entirely(self):
        silent, singing = self.both()
        self.assertLess(silent, singing * SILENT_SAVING_MIN,
                        f"a silent frame costs {silent:.1f} ms against "
                        f"{singing:.1f} ms singing -- the trap is being "
                        f"paid for whether or not anyone is singing")

    def test_the_voice_shape_does_not_cost_the_frame_rate(self):
        silent, singing = self.both()
        self.assertLess(singing, silent * SINGING_OVERHEAD_MAX,
                        f"singing costs {singing / silent:.2f}x a silent frame "
                        f"({singing:.1f} ms vs {silent:.1f} ms)")


if __name__ == "__main__":
    unittest.main()
