"""
Audio-reactive Julia set fractal, driven by Veruca's sequence math and phi.

Three modes, each a different path for the Julia constant c (see
phi_paths.py):
    1  Sequence drift     - golden-angle spiral whose radius crossfades between
                            the halving / doubled / Fibonacci sequences; one
                            sequence term per musical "bar"
    2  Fibonacci flowers  - Julia flowers with 2, 3, 5, 8, 13, 21, 34, 55 arms;
                            the music's repeat length picks the flower
    3  Fibonacci powers   - z^3, z^5, z^8 + c (3/5/8-fold symmetry), cutting
                            between them on kicks; c makes one loop per repeat

Every layer of the music is measured (rhythm.py) and does one thing:
    loudness        how fast c moves (when there's no steady repeat to follow)
    kick            zoom punch, brightness flash, spin jolt, shoves c (modes 2/3)
    snare           colour flips toward its negative, and steps the hue round
    hats            fine contour lines flick through the escape-time bands
    bass notes      swirl: twist the picture into spiral arms
    bass level      breathes the zoom
    vocals          fold the fractal into a ring of copies twisted into spirals
    pitch           the dominant note shifts the palette's hue
    repeat length   the tempo of c's path: mode 1 steps a sequence term, mode 3
                    makes a loop, per bar; mode 2 picks the matching flower
    section change  next palette

A CSV of everything it heard goes to logs/, with a summary when you quit.

Run:
    python main.py [--mode 1|2|3] [--terms 60] [--angle 137.5]
                   [--device NAME|INDEX] [--scale 0.75] [--iters 128]
                   [--fps] [--no-log]

Keys while running:
    1 / 2 / 3   switch mode
    r           the Douady rabbit (a Julia set), on/off
    [ / ]       decrease / increase mode 1's spiral angle by 0.5 degrees
    space       pause/resume the path
    z           freeze/resume the zoom cycle
    p           next color palette
    a           toggle automatic palette changes (on section changes)
    c           pause/resume the spin
    esc         quit
"""

import cmath
import csv
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import moderngl
from PIL import Image, ImageDraw, ImageFont
import moderngl_window as mglw
import sounddevice as sd

import phi_paths
import pw_monitor
from rhythm import FIBONACCI, MusicTracker, wrap_half
from sequences import GOLDEN_ANGLE_DEG

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
HERE = Path(__file__).resolve().parent

# name, then cosine-palette a, b, c, d: color(t) = a + b * cos(2pi * (c * t + d))
PALETTES = [
    ("Prism", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (0.0, 0.33, 0.67)),
    ("Ember", (0.65, 0.35, 0.15), (0.35, 0.3, 0.15), (1.0, 1.0, 1.0), (0.0, 0.05, 0.1)),
    ("Ocean", (0.15, 0.45, 0.6), (0.1, 0.3, 0.35), (1.0, 1.0, 1.0), (0.45, 0.35, 0.25)),
    ("Vapor", (0.65, 0.45, 0.75), (0.35, 0.35, 0.25), (1.0, 1.0, 1.0), (0.0, 0.55, 0.3)),
    ("Acid", (0.45, 0.6, 0.25), (0.45, 0.45, 0.3), (1.5, 1.0, 0.5), (0.2, 0.0, 0.6)),
    ("Ice", (0.7, 0.8, 0.9), (0.3, 0.25, 0.15), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5)),
]
PALETTE_AUTO_SECONDS = 60.0  # fallback change time if no section change comes
PALETTE_FADE_AUTO = 4.0
PALETTE_FADE_KEY = 0.8

FOLD_MAX = 6        # most copies the vocals can split the fractal into
FOLD_HOLD = 0.3     # min seconds between fold-count changes, so it doesn't flicker
FOLD_FADE = 0.3     # crossfade time for each change

MODE_NAMES = {1: "Sequence drift", 2: "Fibonacci flowers", 3: "Fibonacci powers"}
POWER_LABELS = {3: "z³ + c", 5: "z⁵ + c", 8: "z⁸ + c"}
POWER_MIN_SECONDS = 12.0  # mode 3 switches power on the first kick after this...
POWER_MAX_SECONDS = 24.0  # ...or after this, kick or not
BLEND_SECONDS = 40.0      # mode 1: time to cycle through all three sequences

# The music's repeat length sets the pace of c's path. A repeat shorter than
# the minimum loop time is stretched to a whole number of repeats ("a bar").
CLOCK_MIN_CONF = 0.35
MODE1_MIN_LOOP_S = 6.0    # one sequence term per bar
MODE3_MIN_LOOP_S = 8.0    # one loop around the main component per bar
FLOWER_GLIDE = 0.35       # flowers per second toward the one the music picks
PLL_GAIN = 1.5            # how hard c's phase is pulled onto the music's

RABBIT_C = complex(-0.123, 0.745)  # the Douady rabbit

KICK_DECAY = 5.0
SNARE_DECAY = 6.0
HAT_DECAY = 14.0
BASS_DECAY = 3.0
HUE_PER_SNARE = 0.04       # palette phase steps per snare hit
PITCH_HUE_RANGE = 0.3      # palette phase swing from the dominant note

# Zoom cycle: from each mode's base framing (the closest it gets) out to
# ZOOM_OUT_MAX times wider and back again.
ZOOM_OUT_MAX = 1.8
ZOOM_CYCLE_SECONDS = 24.0

LOG_EVERY_S = 0.5

TOAST_SECONDS = 1.5
TOAST_FADE = 0.4
TOAST_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

TOAST_VERT = """#version 330
uniform vec4 u_rect;  // x0, y0, x1, y1 in NDC
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    vec2 p = mix(u_rect.xy, u_rect.zw, v_uv);
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

TOAST_FRAG = """#version 330
uniform sampler2D u_tex;
uniform float u_alpha;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec4 c = texture(u_tex, v_uv);
    fragColor = vec4(c.rgb, c.a * u_alpha);
}
"""


def resolve_device(spec):
    """Turn a --device value into something sounddevice can open.

    "monitor" aims the stream at the PipeWire monitor of system output, i.e.
    whatever you are listening to, via pw_monitor; "monitor=<fragment>" picks a
    sink other than the default one by part of its name. Anything else is passed
    through untouched, except an all-digit string, which becomes an index
    because sounddevice would otherwise read it as a device name.
    """
    if spec is None:
        return None
    if spec.isdigit():
        return int(spec)
    head, _, hint = spec.partition("=")
    if head.lower() != "monitor":
        return spec

    found = pw_monitor.sink_node_id(hint or None)
    if found is None:
        sinks = pw_monitor.describe_sinks()
        detail = (
            "available sinks:\n  " + "\n  ".join(sinks) if sinks
            else "no sinks found at all -- is PipeWire reachable? try `pw-dump`."
        )
        # Refuse rather than fall through: the fallback is the microphone, and a
        # mic next to playing speakers looks enough like the monitor to fool you.
        raise SystemExit(f"--device {spec}: no matching PipeWire sink.\n{detail}")

    node_id, node_name = found
    # Read by the pipewire ALSA plugin when the stream is opened below.
    os.environ["PIPEWIRE_NODE"] = str(node_id)
    print(f"capturing the monitor of {node_name} (node {node_id})", file=sys.stderr)
    return "pipewire"


class AudioAnalyzer:
    """Feeds the sound card's blocks to a MusicTracker (see rhythm.py).
    Point it at your PipeWire monitor source — see README."""

    def __init__(self, device=None):
        self.tracker = MusicTracker(SAMPLE_RATE, BLOCK_SIZE)
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            device=device,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        self.tracker.process(indata[:, 0])

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()


LOG_COLUMNS = [
    "t", "mode", "fps", "pulse_s", "cycle_pulses", "cycle_s", "cycle_conf", "cycle_phase",
    "note", "sections", "kick_per_s", "snare_per_s", "hat_per_s", "bass_note_per_s",
    "amp", "bass", "mid", "treble", "vocal", "folds", "power", "flower_arms", "palette",
]


class SessionLog:
    """One CSV row every half second of what the tracker heard and what the
    visuals did about it, plus a summary when the session ends."""

    def __init__(self, path):
        path.parent.mkdir(exist_ok=True)
        self.path = path
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(LOG_COLUMNS)
        self.rows = 0
        self.seconds = {}                 # mode -> seconds
        self.cycles = Counter()           # repeat length -> rows, confident rows only
        self.notes = Counter()
        self.rate_sums = np.zeros(4)      # kick, snare, hat, bass-note per second
        self.last_sections = 0

    def add(self, row, mode):
        self.writer.writerow([row[c] for c in LOG_COLUMNS])
        self.file.flush()  # so a force-quit doesn't lose the session
        self.rows += 1
        self.seconds[mode] = self.seconds.get(mode, 0.0) + LOG_EVERY_S
        self.rate_sums += [row["kick_per_s"], row["snare_per_s"], row["hat_per_s"], row["bass_note_per_s"]]
        self.notes[row["note"]] += 1
        if row["cycle_pulses"] and row["cycle_conf"] >= CLOCK_MIN_CONF:
            self.cycles[row["cycle_pulses"]] += 1
        self.last_sections = row["sections"]

    def close(self):
        self.file.close()
        if not self.rows:
            return
        lines = [f"session {self.path.name}: {self.rows * LOG_EVERY_S:.0f} s"]
        lines.append("time in each mode: " + ", ".join(
            f"{m} {s:.0f} s" for m, s in sorted(self.seconds.items(), key=lambda kv: str(kv[0]))))
        r = self.rate_sums / self.rows
        lines.append(f"average hits per second: kick {r[0]:.1f}, snare {r[1]:.1f}, hats {r[2]:.1f}, "
                     f"bass notes {r[3]:.1f}")
        lines.append(f"section changes: {self.last_sections}")
        lines.append("most heard notes: " + ", ".join(
            f"{n} {100 * c / self.rows:.0f}%" for n, c in self.notes.most_common(4)))
        total = sum(self.cycles.values())
        if total:
            lines.append(f"repeat lengths (pulses) while the tracker was confident, {total * LOG_EVERY_S:.0f} s:")
            for k, c in self.cycles.most_common(6):
                lines.append(f"  {k:2d}  {100 * c / total:4.1f}%" + ("  <- Fibonacci" if k in FIBONACCI else ""))
            fib = sum(c for k, c in self.cycles.items() if k in FIBONACCI)
            lines.append(f"  Fibonacci share {100 * fib / total:.0f}% (chance level for lengths 3-40 is 16%)")
        text = "\n".join(lines)
        self.path.with_suffix(".summary.txt").write_text(text + "\n")
        print(text, file=sys.stderr)


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class FractalWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Sequence-driven fractal"
    window_size = (1366, 768)
    aspect_ratio = None
    resizable = True

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--mode", type=int, choices=sorted(MODE_NAMES), default=1)
        parser.add_argument("--terms", type=int, default=60, help="mode 1: sequence length; more terms = slower sweep outward")
        parser.add_argument("--angle", type=float, default=GOLDEN_ANGLE_DEG)
        parser.add_argument("--device", type=str, default=None, help="input device: \"monitor\" for the PipeWire monitor of system output (or monitor=<sink name fragment>), else a sounddevice name or index")
        parser.add_argument("--scale", type=float, default=0.75, help="render resolution relative to the window; lower is faster")
        parser.add_argument("--iters", type=int, default=128, help="max Julia iterations; lower is faster, higher is more detail")
        parser.add_argument("--fps", action="store_true", help="print frames per second to stderr every 5 seconds")
        parser.add_argument("--no-log", action="store_true", help="don't write a session log to logs/")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set after creation: moderngl-window 3.1's own fullscreen path calls
        # pyglet.canvas, which no longer exists in pyglet 2.1.
        self.wnd.fullscreen = True

        self.args = self.argv  # populated by moderngl_window from add_arguments

        self.mode = self.args.mode
        self.angle = self.args.angle
        self.radii = phi_paths.sequence_radii(self.args.terms, radius_scale=1.1)
        self.path_s = {1: 0.0, 2: 0.0, 3: 0.0}  # each mode resumes where it left off
        self.blend_phase = 0.0
        self.paused = False
        self.rabbit = False
        self.flower_pos = 2.0     # along phi_paths.FIB_RATIOS
        self.flower_dir = 1
        self.flower_mirror = False
        self.power_idx = 1  # into phi_paths.FIB_POWERS; starts at z^5
        self.power_time = 0.0
        self.spin = 0.0
        self.spin_paused = False
        self.zoom_phase = 0.0
        self.zoom_cycling = True

        vert = (HERE / "julia.vert").read_text()
        frag = (HERE / "julia.frag").read_text()

        self.prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)

        quad = np.array(
            [-1, -1, 1, -1, -1, 1, 1, 1],
            dtype="f4",
        )
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_position")

        self.audio = AudioAnalyzer(device=resolve_device(self.args.device))
        self.tr = self.audio.tracker
        self.audio.start()

        # The layers' decaying hit strengths, and the counters they watch.
        self.kick = self.snare = self.hat = self.bass_pulse = 0.0
        self.hue_steps = self.hue_smooth = 0.0
        self.seen = dict(kick=0, snare=0, hat=0, bass=0, section=0)

        self.toast_prog = self.ctx.program(vertex_shader=TOAST_VERT, fragment_shader=TOAST_FRAG)
        self.toast_vao = self.ctx.simple_vertex_array(self.toast_prog, self.vbo, "in_position")
        self.toast_tex = None
        self.toast_size = (1, 1)
        self.toast_until = 0.0
        self.now = 0.0
        self.fps_frames = 0
        self.fps_t0 = 0.0

        self.log = None
        if not self.args.no_log:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log = SessionLog(HERE / "logs" / f"session-{stamp}.csv")
        self.log_next = LOG_EVERY_S
        self.log_frames = 0
        self.log_counts = (0, 0, 0, 0)
        self.flower_arms = 0
        self.power = 2

        # The fractal renders at --scale into this, then gets stretched to
        # the window. Built lazily so it follows fullscreen/resizes.
        self.scene_tex = None
        self.scene_fbo = None

        self.vocal_slow = 0.0
        self.fold_from = 1
        self.fold_to = 1
        self.fold_mix = 1.0
        self.fold_changed_at = 0.0

        self.pal_idx = 0
        self.pal_from = [np.array(v) for v in PALETTES[0][1:]]
        self.pal_to = self.pal_from
        self.pal_t0 = 0.0
        self.pal_fade = 1.0
        self.pal_auto = True
        self.pal_next_auto = PALETTE_AUTO_SECONDS

    def show_toast(self, text):
        """Render `text` into a small rounded pill and show it briefly."""
        font = ImageFont.truetype(TOAST_FONT, 22)
        l, t, r, b = font.getbbox(text)
        pad_x, pad_y = 18, 10
        w, h = (r - l) + 2 * pad_x, (b - t) + 2 * pad_y
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=(0, 0, 0, 170))
        d.text((pad_x - l, pad_y - t), text, font=font, fill=(255, 255, 255, 255))
        img = img.transpose(Image.FLIP_TOP_BOTTOM)  # GL origin is bottom-left
        if self.toast_tex is not None:
            self.toast_tex.release()
        self.toast_tex = self.ctx.texture((w, h), 4, img.tobytes())
        self.toast_size = (w, h)
        self.toast_until = self.now + TOAST_SECONDS

    # ---- the music, layer by layer -----------------------------------------

    def _update_layers(self, dt):
        """Turn the tracker's counters into decaying hit strengths. Returns
        (kick_hit, section_change) for this frame."""
        tr, seen = self.tr, self.seen
        hits = dict(kick=tr.n_kick != seen["kick"], snare=tr.n_snare != seen["snare"],
                    hat=tr.n_hat != seen["hat"], bass=tr.n_bass != seen["bass"],
                    section=tr.sections != seen["section"])
        seen.update(kick=tr.n_kick, snare=tr.n_snare, hat=tr.n_hat, bass=tr.n_bass, section=tr.sections)

        self.kick = 1.0 if hits["kick"] else self.kick * math.exp(-KICK_DECAY * dt)
        self.snare = 1.0 if hits["snare"] else self.snare * math.exp(-SNARE_DECAY * dt)
        self.hat = 1.0 if hits["hat"] else self.hat * math.exp(-HAT_DECAY * dt)
        self.bass_pulse = min(1.0, self.bass_pulse + 0.7) if hits["bass"] else self.bass_pulse * math.exp(-BASS_DECAY * dt)
        if hits["snare"]:
            self.hue_steps += HUE_PER_SNARE
        self.hue_smooth += (self.hue_steps - self.hue_smooth) * min(dt * 10.0, 1.0)
        return hits["kick"], hits["section"]

    def _music_clock(self, min_loop_s):
        """(loops per second, phase to lock to or None) when the music has a
        steady repeat to follow, else None. A repeat shorter than min_loop_s
        is stretched to a whole number of repeats."""
        tr = self.tr
        if not tr.cycle_len or tr.cycle_conf < CLOCK_MIN_CONF or tr.cycle_seconds < 0.3:
            return None
        k = max(1, math.ceil(min_loop_s / tr.cycle_seconds))
        return 1.0 / (k * tr.cycle_seconds), (tr.cycle_phase if k == 1 else None)

    def _advance(self, mode, unit, clock, fallback_loops_per_s, dt):
        """Move a mode's path position. It runs at the music's pace when
        there's a clock (and, if the repeat is a single loop, is pulled onto
        the music's phase), else at the loudness-driven fallback pace."""
        if self.paused:
            return
        rate = clock[0] if clock else fallback_loops_per_s
        step = rate * dt
        if clock and clock[1] is not None:
            frac = (self.path_s[mode] / unit) % 1.0
            step += PLL_GAIN * wrap_half(clock[1] - frac) * dt
        self.path_s[mode] += unit * step

    def _update_path(self, dt, amp, kick_hit):
        """Advance the current mode's path; return (c, power, center, zoom)."""
        loud = amp * amp
        self.power = 2
        if self.rabbit:
            # The rabbit sits in a small bulb, so a kick only shivers c.
            return RABBIT_C + 0.004 * self.kick * cmath.rect(1.0, 3 * self.spin), 2, 0j, 1.1

        if not self.paused:
            self.blend_phase += 2 * math.pi * dt / BLEND_SECONDS

        if self.mode == 1:
            self._advance(1, 1.0, self._music_clock(MODE1_MIN_LOOP_S), 0.04 + 0.5 * loud, dt)
            c = phi_paths.sequence_drift(self.radii, self.path_s[1], self.angle, self.blend_phase)
            return c, 2, 0j, 1.1

        if self.mode == 2:
            n = len(phi_paths.FIB_RATIOS)
            clock = self._music_clock(0.0)
            if not self.paused:
                if clock:
                    # The repeat length picks the flower with the nearest arm count.
                    target = phi_paths.nearest_flower(self.tr.cycle_len)
                    step = FLOWER_GLIDE * dt
                    self.flower_pos += min(max(target - self.flower_pos, -step), step)
                else:
                    self.flower_pos += self.flower_dir * (0.1 + 0.45 * loud) * dt
                    if self.flower_pos >= n - 1 or self.flower_pos <= 0:
                        self.flower_dir = -self.flower_dir
                        self.flower_mirror = not self.flower_mirror
            self.flower_pos = min(max(self.flower_pos, 0.0), n - 1.0)
            self.flower_arms = phi_paths.FLOWER_ARMS[round(self.flower_pos)]
            # Kicks blow the flower out of its bulb; it re-forms as they fade.
            c, fixed_point = phi_paths.fibonacci_flower(
                self.flower_pos, 0.6 * amp + 1.4 * self.kick, self.flower_mirror)
            return c, 2, fixed_point, 0.7

        self._advance(3, 2 * math.pi, self._music_clock(MODE3_MIN_LOOP_S), (0.1 + 0.6 * loud) / (2 * math.pi), dt)
        self.power_time += dt
        if ((kick_hit and self.power_time > POWER_MIN_SECONDS)
                or self.power_time > POWER_MAX_SECONDS):
            self.power_idx = (self.power_idx + 1) % len(phi_paths.FIB_POWERS)
            self.power_time = 0.0
            self.show_toast(f"3  {POWER_LABELS[phi_paths.FIB_POWERS[self.power_idx]]}")
        power = phi_paths.FIB_POWERS[self.power_idx]
        self.power = power
        c = phi_paths.fibonacci_power(self.path_s[3], 0.8 * amp + 1.2 * self.kick, power)
        return c, power, 0j, 1.0

    def _current_palette(self):
        k = smoothstep((self.now - self.pal_t0) / self.pal_fade)
        return [(1 - k) * a + k * b for a, b in zip(self.pal_from, self.pal_to)]

    def _go_to_palette(self, idx, fade):
        self.pal_from = self._current_palette()
        self.pal_idx = idx % len(PALETTES)
        self.pal_to = [np.array(v) for v in PALETTES[self.pal_idx][1:]]
        self.pal_t0 = self.now
        self.pal_fade = fade
        self.pal_next_auto = self.now + PALETTE_AUTO_SECONDS

    def _update_fold(self, dt, vocal):
        # Vocals set the fold count; each change crossfades from the old one.
        # Below ~0.45 (voice not standing out) it stays whole.
        self.vocal_slow += (vocal - self.vocal_slow) * min(dt * 3.0, 1.0)
        target = 1 + int(round(self.vocal_split() * (FOLD_MAX - 1)))
        if self.fold_mix < 1.0:
            self.fold_mix = min(1.0, self.fold_mix + dt / FOLD_FADE)
        elif target != self.fold_to and self.now - self.fold_changed_at > FOLD_HOLD:
            self.fold_from = self.fold_to
            self.fold_to = target
            self.fold_mix = 0.0
            self.fold_changed_at = self.now

    def vocal_split(self):
        return min(max((self.vocal_slow - 0.45) / 0.4, 0.0), 1.0)

    def _write_log(self):
        tr = self.tr
        counts = (tr.n_kick, tr.n_snare, tr.n_hat, tr.n_bass)
        d = [(c - p) / LOG_EVERY_S for c, p in zip(counts, self.log_counts)]
        self.log_counts = counts
        mode = "rabbit" if self.rabbit else self.mode
        self.log.add({
            "t": f"{self.now:.1f}", "mode": mode, "fps": round(self.log_frames / LOG_EVERY_S),
            "pulse_s": f"{tr.pulse_s:.3f}", "cycle_pulses": tr.cycle_len,
            "cycle_s": f"{tr.cycle_seconds:.2f}", "cycle_conf": round(tr.cycle_conf, 2),
            "cycle_phase": f"{tr.cycle_phase:.2f}", "note": tr.pitch_name, "sections": tr.sections,
            "kick_per_s": round(d[0], 1), "snare_per_s": round(d[1], 1),
            "hat_per_s": round(d[2], 1), "bass_note_per_s": round(d[3], 1),
            "amp": f"{tr.amp:.2f}", "bass": f"{tr.bass:.2f}", "mid": f"{tr.mid:.2f}",
            "treble": f"{tr.treble:.2f}", "vocal": f"{tr.vocal:.2f}",
            "folds": self.fold_to, "power": self.power,
            "flower_arms": self.flower_arms if self.mode == 2 and not self.rabbit else "",
            "palette": PALETTES[self.pal_idx][0],
        }, mode)
        self.log_frames = 0

    def _ensure_scene_fbo(self):
        bw, bh = self.wnd.buffer_size
        size = (max(1, int(bw * self.args.scale)), max(1, int(bh * self.args.scale)))
        if self.scene_tex is not None and self.scene_tex.size == size:
            return
        if self.scene_fbo is not None:
            self.scene_fbo.release()
            self.scene_tex.release()
        self.scene_tex = self.ctx.texture(size, 3)
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_fbo = self.ctx.framebuffer(color_attachments=[self.scene_tex])

    def _draw_quad(self, tex, rect, alpha):
        self.toast_prog["u_rect"].value = rect
        self.toast_prog["u_alpha"].value = alpha
        tex.use(0)
        self.toast_prog["u_tex"].value = 0
        self.toast_vao.render(moderngl.TRIANGLE_STRIP)

    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return
        mode_keys = {keys.NUMBER_1: 1, keys.NUMBER_2: 2, keys.NUMBER_3: 3}
        if key in mode_keys:
            self.mode = mode_keys[key]
            self.rabbit = False
            self.show_toast(f"{self.mode}  {MODE_NAMES[self.mode]}")
        elif key == keys.R:
            self.rabbit = not self.rabbit
            self.show_toast(f"Douady rabbit  c = {RABBIT_C.real:.3f} + {RABBIT_C.imag:.3f}i"
                            if self.rabbit else f"{self.mode}  {MODE_NAMES[self.mode]}")
        elif key in (keys.LEFT_BRACKET, keys.RIGHT_BRACKET):
            self.angle += 0.5 if key == keys.RIGHT_BRACKET else -0.5
            self.show_toast(f"Angle  {self.angle:.1f}°" + ("" if self.mode == 1 else "  (mode 1)"))
        elif key == keys.SPACE:
            self.paused = not self.paused
        elif key == keys.Z:
            self.zoom_cycling = not self.zoom_cycling
            self.show_toast(f"Zoom  {'cycling' if self.zoom_cycling else 'frozen'}")
        elif key == keys.P:
            self._go_to_palette(self.pal_idx + 1, PALETTE_FADE_KEY)
            self.show_toast(f"Palette  {PALETTES[self.pal_idx][0]}")
        elif key == keys.A:
            self.pal_auto = not self.pal_auto
            self.pal_next_auto = self.now + PALETTE_AUTO_SECONDS
            self.show_toast(f"Auto palettes  {'on' if self.pal_auto else 'off'}")
        elif key == keys.C:
            self.spin_paused = not self.spin_paused
            self.show_toast(f"Spin  {'paused' if self.spin_paused else 'on'}")
        elif key == keys.ESCAPE:
            self.wnd.close()

    def on_render(self, time, frame_time):
        self.now = time
        self.fps_frames += 1
        self.log_frames += 1
        if self.args.fps and time - self.fps_t0 >= 5.0:
            print(f"{self.fps_frames / (time - self.fps_t0):.0f} fps", file=sys.stderr)
            self.fps_frames, self.fps_t0 = 0, time

        tr = self.tr
        amp, bass, mid, treble = tr.amp, tr.bass, tr.mid, tr.treble
        kick_hit, section = self._update_layers(frame_time)

        c, power, center, zoom = self._update_path(frame_time, amp, kick_hit)
        if self.zoom_cycling:
            self.zoom_phase += 2 * math.pi * frame_time / ZOOM_CYCLE_SECONDS
        # Geometric, so zooming out and back in feel equally paced. Bass
        # breathes it in, kicks punch it in.
        zoom *= ZOOM_OUT_MAX ** (0.5 - 0.5 * math.cos(self.zoom_phase))
        zoom *= (1 - 0.12 * bass) * (1 - 0.08 * self.kick)

        if not self.spin_paused:
            self.spin += (0.03 + 0.1 * amp + 0.6 * self.kick) * frame_time
        self._update_fold(frame_time, tr.vocal)
        if self.pal_auto and (section or time >= self.pal_next_auto):
            self._go_to_palette(self.pal_idx + 1, PALETTE_FADE_AUTO)
        if self.log and time >= self.log_next:
            self.log_next += LOG_EVERY_S
            self._write_log()

        self._ensure_scene_fbo()
        self.scene_fbo.use()
        pal_a, pal_b, pal_c, pal_d = self._current_palette()
        hue = (PITCH_HUE_RANGE * math.sin(2 * math.pi * tr.hue) + self.hue_smooth) % 1.0
        uniforms = {
            "u_resolution": self.scene_tex.size,
            "u_c": (c.real, c.imag),
            "u_power": power,
            "u_center": (center.real, center.imag),
            "u_zoom": zoom,
            "u_amp": amp,
            "u_bass": bass,
            "u_mid": mid,
            "u_treble": treble,
            "u_kick": self.kick,
            "u_snare": self.snare,
            "u_hat": self.hat,
            "u_hue": hue,
            "u_max_iter": self.args.iters,
            "u_time": time,
            "u_spin": self.spin,
            "u_fold_a": float(self.fold_from),
            "u_fold_b": float(self.fold_to),
            "u_fold_mix": self.fold_mix,
            "u_twist": self.vocal_split() * 1.2 + 0.5 * self.bass_pulse,
            "u_pal_a": tuple(pal_a),
            "u_pal_b": tuple(pal_b),
            "u_pal_c": tuple(pal_c),
            "u_pal_d": tuple(pal_d),
        }
        for name, value in uniforms.items():
            self.prog[name].value = value
        self.vao.render(moderngl.TRIANGLE_STRIP)

        self.wnd.use()
        self.ctx.clear(0.0, 0.0, 0.0)
        self._draw_quad(self.scene_tex, (-1.0, -1.0, 1.0, 1.0), 1.0)

        remaining = self.toast_until - time
        if self.toast_tex is not None and remaining > 0:
            bw, bh = self.wnd.buffer_size
            tw, th = self.toast_size
            margin = 24
            x0 = -1 + 2 * margin / bw
            y0 = -1 + 2 * margin / bh
            self.ctx.enable(moderngl.BLEND)
            self._draw_quad(self.toast_tex, (x0, y0, x0 + 2 * tw / bw, y0 + 2 * th / bh),
                            min(1.0, remaining / TOAST_FADE))
            self.ctx.disable(moderngl.BLEND)

    def on_close(self):
        self.audio.stop()
        if self.log:
            self.log.close()


def main():
    mglw.run_window_config(FractalWindow)


if __name__ == "__main__":
    main()
