"""
Audio-reactive Julia set fractal, driven by Veruca's sequence math and phi.

Seven modes, each a different path for the Julia constant c — and, for the
last two, a different map to iterate (see phi_paths.py):
    1  Sequence drift     - golden-angle spiral whose radius crossfades between
                            the halving / doubled / Fibonacci sequences; one
                            sequence term per musical "bar"
    2  Fibonacci flowers  - Julia flowers with 2, 3, 5, 8, 13, 21, 34, 55 arms;
                            the music's repeat length picks the flower
    3  Fibonacci powers   - z^3, z^5, z^8 + c (3/5/8-fold symmetry), cutting
                            between them on kicks; c makes one loop per repeat
    4  Metallic spirals   - mode 1 off phi: the bronze, silver, golden and
                            plastic means, each with its own divergence angle
                            (33, 61.8, 137.5, 205.1 deg) and its own integer
                            sequences, walked from the tightest angle outward
    5  Farey tour         - mode 2 off the Fibonacci rails: c descends the
                            Stern-Brocot tree, so the arm count is the mediant
                            of the last two, and kicks pick the branch — the
                            music writes a continued fraction
    6  Phoenix wings      - z^2 + c + p*z_prev, one step of memory, which
                            breaks the symmetry into wings; p swings between
                            -1/phi and -1/phi^2
    7  Mandelbar flames   - the antiholomorphic maps: tricorn (z-bar^d + c)
                            and burning ship (|Re z| + i|Im z|)^d + c, d = 2
                            or 3, cutting between the four on kicks

Every layer of the music is measured (rhythm.py) and does one thing:
    loudness        how fast c moves (when there's no steady repeat to follow)
    kick            zoom punch, brightness flash, spin jolt, shoves c (modes 2/3)
    snare           colour flips toward its negative, and steps the hue round
    hats            fine contour lines flick through the escape-time bands
    bass notes      swirl: twist the picture into spiral arms
    bass level      breathes the zoom
    vocals          fold the fractal into a ring of copies twisted into spirals
    pitch           the dominant note shifts the palette's hue
    repeat length   the tempo of c's path: modes 1 and 4 step a sequence term,
                    modes 3, 6 and 7 make a loop, mode 5 takes a branch, per
                    bar; mode 2 picks the flower with the matching arm count
    section change  next palette

With --play it also runs the music: mpv plays the files or folders you name
(player.py) and the visuals listen to the monitor of system output, so the
same capture path hears it. Knowing the player means the log says *where in
which track* something was heard, and a new track moves the palette on.

A CSV of everything it heard goes to logs/, with a summary when you quit.

Run:
    python main.py [--mode 1..7] [--terms 60] [--angle 137.5]
                   [--device NAME|INDEX] [--scale 0.75] [--iters 128]
                   [--fps] [--no-log]
                   [--play PATH... [--shuffle] [--seed N] [--volume 85]
                                   [--no-resume]]

Keys while running:
    1 ... 7     switch mode
    r           the Douady rabbit (a Julia set), on/off
    [ / ]       decrease / increase mode 1's spiral angle by 0.5 degrees
    space       pause/resume the path
    z           freeze/resume the zoom cycle
    p           next color palette
    a           toggle automatic palette changes (on section changes)
    c           pause/resume the spin
    esc         quit

  with --play:
    click       bring the transport up; click away to put it back; click a
                button to use it. Position in the track is the light running
                round the edge of the screen, which is always on.
    m           play/pause the music (space pauses the path, not the sound)
    n / b       next / previous track
    left/right  seek 10 s back / forward
    up / down   volume
    i           show/hide the transport, for when the mouse isn't to hand
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

import overlay
import phi_paths
import player
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

MODE_NAMES = {
    1: "Sequence drift", 2: "Fibonacci flowers", 3: "Fibonacci powers",
    4: "Metallic spirals", 5: "Farey tour", 6: "Phoenix wings",
    7: "Mandelbar flames",
}
POWER_LABELS = {3: "z³ + c", 5: "z⁵ + c", 8: "z⁸ + c"}
POWER_MIN_SECONDS = 12.0  # mode 3 switches power on the first kick after this...
POWER_MAX_SECONDS = 24.0  # ...or after this, kick or not
BLEND_SECONDS = 40.0      # mode 1: time to cycle through all three sequences

FAMILY_SECONDS = 100.0    # mode 4: time to walk the metallic families end to end
FAREY_Q_MAX = 89          # mode 5: restart once the arms are finer than this
# Mode 7 cuts between its four maps the way mode 3 cuts between powers.
MANDELBAR_MIN_SECONDS = 14.0
MANDELBAR_MAX_SECONDS = 28.0
FORMULA_NAMES = {
    phi_paths.FORMULA_POWER: "power",
    phi_paths.FORMULA_PHOENIX: "phoenix",
    phi_paths.FORMULA_TRICORN: "tricorn",
    phi_paths.FORMULA_SHIP: "ship",
}
MANDELBAR_LABELS = {
    (phi_paths.FORMULA_TRICORN, 2): "tricorn  z̄² + c",
    (phi_paths.FORMULA_TRICORN, 3): "tricorn  z̄³ + c",
    (phi_paths.FORMULA_SHIP, 2): "burning ship, d = 2",
    (phi_paths.FORMULA_SHIP, 3): "burning ship, d = 3",
}

# The music's repeat length sets the pace of c's path. A repeat shorter than
# the minimum loop time is stretched to a whole number of repeats ("a bar").
CLOCK_MIN_CONF = 0.35
MODE1_MIN_LOOP_S = 6.0    # one sequence term per bar
MODE3_MIN_LOOP_S = 8.0    # one loop around the main component per bar
MODE4_MIN_LOOP_S = 5.0    # one metallic-spiral term per bar
MODE5_MIN_STEP_S = 3.0    # one Stern-Brocot branch per bar
MODE6_MIN_LOOP_S = 8.0    # one loop of the Phoenix map's c per bar
MODE7_MIN_LOOP_S = 7.0    # one orbit of the mandelbar body per bar
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

# --play only: mpv plays, and the visuals hear it the same way they hear
# anything else, through the monitor of the default sink (see player.py).
PLAYER_STATE_FILE = HERE / ".player_state.json"
SEEK_STEP = 10.0          # seconds per left/right press
VOLUME_STEP = 5.0         # percent per up/down press

TOAST_SECONDS = 1.5
TOAST_FADE = 0.4
TOAST_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

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
    "amp", "bass", "mid", "treble", "vocal", "folds", "power", "formula", "arms", "palette",
    "track", "track_s",
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
        self.tracks = Counter()           # track title -> rows, when --play is on

    def add(self, row, mode):
        self.writer.writerow([row[c] for c in LOG_COLUMNS])
        self.file.flush()  # so a force-quit doesn't lose the session
        self.rows += 1
        self.seconds[mode] = self.seconds.get(mode, 0.0) + LOG_EVERY_S
        self.rate_sums += [row["kick_per_s"], row["snare_per_s"], row["hat_per_s"], row["bass_note_per_s"]]
        self.notes[row["note"]] += 1
        if row["track"]:
            self.tracks[row["track"]] += 1
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
        if self.tracks:
            lines.append(f"{len(self.tracks)} tracks, longest on screen:")
            for name, c in self.tracks.most_common(5):
                lines.append(f"  {c * LOG_EVERY_S:5.0f} s  {name}")
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
        parser.add_argument("--play", type=str, nargs="+", metavar="PATH", help="play these files/folders with mpv, and listen to the monitor of system output unless --device says otherwise")
        parser.add_argument("--shuffle", action="store_true", help="--play: shuffle the playlist")
        parser.add_argument("--seed", type=int, default=None, help="--play: seed for --shuffle, so a shuffle can be repeated")
        parser.add_argument("--volume", type=float, default=85.0, help="--play: mpv volume, 0-130")
        parser.add_argument("--no-resume", action="store_true", help="--play: start at the top instead of where the last session stopped")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set after creation: moderngl-window 3.1's own fullscreen path calls
        # pyglet.canvas, which no longer exists in pyglet 2.1.
        self.wnd.fullscreen = True

        self.args = self.argv  # populated by moderngl_window from add_arguments

        self.mode = self.args.mode
        self.angle = self.args.angle
        self.radii = phi_paths.sequence_radii(self.args.terms, radius_scale=1.1)
        # Each mode resumes where it left off.
        self.path_s = {m: 0.0 for m in MODE_NAMES}
        self.blend_phase = 0.0
        self.paused = False
        self.rabbit = False
        self.flower_pos = 2.0     # along phi_paths.FIB_RATIOS
        self.flower_dir = 1
        self.flower_mirror = False
        self.power_idx = 1  # into phi_paths.FIB_POWERS; starts at z^5
        self.power_time = 0.0

        # Mode 4: the metallic families, and the phase c's angle accumulates
        # into (see phi_paths.metallic_sample for why it's accumulated).
        self.metallic = phi_paths.metallic_profiles(self.args.terms, radius_scale=1.1)
        self.metal_theta = 0.0
        self.family_pos = 0.0
        self.family_dir = 1
        self.metal_label = self.metallic[0][0]

        # Mode 5: the Stern-Brocot descent, the glide between its last two
        # mediants, and the kick count that picks the next branch.
        self.farey = phi_paths.FareyWalk(q_max=FAREY_Q_MAX)
        self.farey_f = 0.0
        self.farey_kicks = 0

        # Modes 6 and 7: which map the shader iterates, and its parameter.
        self.formula = phi_paths.FORMULA_POWER
        self.p_param = 0j
        self.mandel_idx = 0
        self.mandel_time = 0.0
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

        # With --play, what mpv plays comes back in through the monitor of
        # the default sink, so that's the capture to default to.
        device = self.args.device
        if device is None and self.args.play:
            device = "monitor"
            print("--play: capturing the monitor of system output", file=sys.stderr)
        self.audio = AudioAnalyzer(device=resolve_device(device))
        self.tr = self.audio.tracker
        self.audio.start()

        self.player = None
        self.player_state = None
        self.track_seen = 0
        self.overlay = None
        if self.args.play:
            self._start_player()

        # The layers' decaying hit strengths, and the counters they watch.
        self.kick = self.snare = self.hat = self.bass_pulse = 0.0
        self.hue_steps = self.hue_smooth = 0.0
        self.seen = dict(kick=0, snare=0, hat=0, bass=0, section=0)

        self.toast_prog = self.ctx.program(vertex_shader=overlay.QUAD_VERT,
                                           fragment_shader=overlay.QUAD_FRAG)
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

    def _start_player(self):
        """Build the playlist, start mpv, and pick up where we left off.

        A player that won't start is reported and then dropped: the visuals
        work on whatever else is making sound, so there's no reason to take
        the session down with it.
        """
        tracks = player.build_playlist(self.args.play, self.args.shuffle, self.args.seed)
        if not tracks:
            print(f"--play: no audio files under {', '.join(self.args.play)}", file=sys.stderr)
            return
        self.player_state = player.PlayerState(PLAYER_STATE_FILE)
        volume = self.args.volume
        if self.args.volume == 85.0:  # not given explicitly; use the saved one
            volume = float(self.player_state.data.get("volume", volume))

        # Starts paused so that resuming can seek before a note is played,
        # rather than the first track blipping out before we move off it.
        p = player.MpvPlayer(tracks, volume=volume, start_paused=True)
        if p.failed:
            print(f"--play: {p.failed}", file=sys.stderr)
            return
        self.player = p

        # loadlist is asynchronous; without this the play_index below can
        # land before the playlist exists and be dropped.
        if not p.wait_ready():
            print("--play: mpv never loaded the playlist", file=sys.stderr)

        start_at, seek = 0, None
        last = self.player_state.last if not self.args.no_resume else None
        if last:
            path, pos = last
            try:
                start_at = tracks.index(path)
                seek = pos if pos > 5.0 else None  # near the top: just restart it
            except ValueError:
                pass  # that track isn't in this playlist any more
        p.play_index(start_at, seek=seek)
        self.overlay = overlay.NowPlayingOverlay(self.ctx, self.vbo)
        print(f"--play: {len(tracks)} tracks, starting at {start_at + 1}", file=sys.stderr)

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
        self.formula = phi_paths.FORMULA_POWER
        self.p_param = 0j
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

        if self.mode == 4:
            # Same step as mode 1 — one sequence term per bar — but the angle
            # is the current metallic family's, accumulated into a phase so
            # sliding between families doesn't whip c around the plane.
            before = self.path_s[4]
            self._advance(4, 1.0, self._music_clock(MODE4_MIN_LOOP_S), 0.05 + 0.5 * loud, dt)
            last = len(self.metallic) - 1
            if not self.paused:
                self.family_pos += self.family_dir * len(self.metallic) * dt / FAMILY_SECONDS
                if not 0.0 <= self.family_pos <= last:
                    self.family_dir = -self.family_dir
                    self.family_pos = min(max(self.family_pos, 0.0), float(last))
            r, angle, label = phi_paths.metallic_sample(
                self.metallic, self.path_s[4], self.family_pos, 0.3 * amp + 0.9 * self.kick)
            self.metal_theta += math.radians(angle) * (self.path_s[4] - before)
            if label != self.metal_label:
                self.metal_label = label
                self.show_toast(f"4  {label}")
            return cmath.rect(r, self.metal_theta), 2, 0j, 1.1

        if self.mode == 5:
            # One branch of the Stern-Brocot tree per bar, and the music picks
            # it: an odd number of kicks in the bar goes right, an even number
            # (none included) goes left. Parity rather than "any kick at all"
            # because a rule that saturates one way just adds an arm per step;
            # it's the branch changing that makes the arm counts interesting.
            clock = self._music_clock(MODE5_MIN_STEP_S)
            if not self.paused:
                self.farey_kicks += kick_hit
                self.farey_f += (clock[0] if clock else 0.12 + 0.4 * loud) * dt
                while self.farey_f >= 1.0:
                    self.farey_f -= 1.0
                    self.farey.step(self.farey_kicks % 2 == 1)
                    self.farey_kicks = 0
            self.flower_arms = self.farey.arms
            c, fixed_point = self.farey.c(self.farey_f, 0.5 * amp + 1.3 * self.kick)
            return c, 2, fixed_point, 0.7

        if self.mode == 6:
            self._advance(6, 2 * math.pi, self._music_clock(MODE6_MIN_LOOP_S),
                          (0.08 + 0.5 * loud) / (2 * math.pi), dt)
            c, self.p_param = phi_paths.phoenix_path(
                self.path_s[6], 0.7 * amp + 1.3 * self.kick)
            self.formula = phi_paths.FORMULA_PHOENIX
            return c, 2, 0j, 1.0

        if self.mode == 7:
            self._advance(7, 2 * math.pi, self._music_clock(MODE7_MIN_LOOP_S),
                          (0.1 + 0.6 * loud) / (2 * math.pi), dt)
            self.mandel_time += dt
            if ((kick_hit and self.mandel_time > MANDELBAR_MIN_SECONDS)
                    or self.mandel_time > MANDELBAR_MAX_SECONDS):
                self.mandel_idx = (self.mandel_idx + 1) % len(phi_paths.MANDELBAR_VARIANTS)
                self.mandel_time = 0.0
                variant = phi_paths.MANDELBAR_VARIANTS[self.mandel_idx]
                self.show_toast(f"7  {MANDELBAR_LABELS[variant[0], variant[1]]}")
            variant = phi_paths.MANDELBAR_VARIANTS[self.mandel_idx]
            self.formula = variant[0]
            self.power = variant[1]
            c = phi_paths.mandelbar_path(self.path_s[7], 0.6 * amp + 1.1 * self.kick, variant)
            return c, variant[1], 0j, 1.0

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

    def _poll_player(self):
        """Announce the track when it changes. Returns True on the frame it
        does, so the render loop can treat it as the scene change it is.

        Serial 0 is "mpv hasn't loaded anything yet", which is not a change
        and whose title would be "nothing playing"; every load after that is
        announced, the first one included, since arriving mid-album and being
        told what's on is the point.
        """
        p = self.player
        if p is None or not p.track_serial or p.track_serial == self.track_seen:
            return False
        self.track_seen = p.track_serial
        self.show_toast(p.title())
        return True

    def _now_playing(self):
        """(title, position in seconds) for the log, or ("", "")."""
        if self.player is None or not self.player.alive:
            return "", ""
        return self.player.title(), f"{self.player.position:.1f}"

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
            "formula": FORMULA_NAMES[self.formula],
            # Arm count at the bulb c is sitting in, for the two modes that
            # ride the main cardioid.
            "arms": self.flower_arms if self.mode in (2, 5) and not self.rabbit else "",
            "palette": PALETTES[self.pal_idx][0],
            # So a row says not just what was heard but where in which track.
            **dict(zip(("track", "track_s"), self._now_playing())),
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
        mode_keys = {keys.NUMBER_1: 1, keys.NUMBER_2: 2, keys.NUMBER_3: 3,
                     keys.NUMBER_4: 4, keys.NUMBER_5: 5, keys.NUMBER_6: 6,
                     keys.NUMBER_7: 7}
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
        elif self.player is not None:
            self._transport_key(key, keys)

    def _transport_key(self, key, keys):
        """The player's keys, which only exist when --play gave it one.

        Play/pause is `m`, not space: space already pauses the path, and a
        session where the music stops but the fractal keeps moving (or the
        reverse) is worth being able to ask for.
        """
        p = self.player
        if key == keys.M:
            p.toggle_pause()
            # The cached flag is the pre-press one; `cycle` makes it the other.
            self.show_toast("Play" if p.paused else "Pause")
        elif key in (keys.N, keys.B):
            p.next_track() if key == keys.N else p.prev_track()
        elif key in (keys.LEFT, keys.RIGHT):
            p.seek(SEEK_STEP if key == keys.RIGHT else -SEEK_STEP)
        elif key in (keys.UP, keys.DOWN):
            p.add_volume(VOLUME_STEP if key == keys.UP else -VOLUME_STEP)
            self.show_toast(f"Volume  {p.state.get('volume', 0):.0f}%")
        elif key == keys.I:
            # The same thing a click does, for when the mouse isn't to hand.
            self.overlay.toggle(self.now)

    def on_mouse_press_event(self, x, y, button):
        """A click brings the transport up, another puts it away, and one on
        a button does what it says."""
        if self.overlay is None:
            return
        hit = self.overlay.click(x, y, self.wnd.size, self.now)
        if hit == "play_pause":
            self.player.toggle_pause()
        elif hit == "next":
            self.player.next_track()
        elif hit == "prev":
            self.player.prev_track()

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
        new_track = self._poll_player()
        # A new track is a bigger change than a section within one, so it
        # moves the palette on for the same reason a section change does.
        if self.pal_auto and (section or new_track or time >= self.pal_next_auto):
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
            "u_formula": self.formula,
            "u_p": (self.p_param.real, self.p_param.imag),
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

        if self.overlay is not None:
            self.overlay.draw(self.player, time, self.wnd.size, self.wnd.buffer_size)

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
        if self.overlay is not None:
            self.overlay.release()
        if self.player is not None:
            # Saved before the quit, while the position can still be read.
            if self.player_state is not None and self.player.alive:
                self.player_state.save(self.player.current_path, self.player.position,
                                       self.player.state.get("volume"))
            self.player.close()
        if self.log:
            self.log.close()


def main():
    mglw.run_window_config(FractalWindow)


if __name__ == "__main__":
    main()
