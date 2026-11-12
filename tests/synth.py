"""Synthesised music with a known answer, and a headless FractalWindow.

Synthesised rather than real music for the reason the TDD skill gives: the
expected result has to be arguable from first principles. We put N kicks in,
so about N should come out; we build a verse and a chorus, so the picture
should treat them differently.

The window half exists because main.py forces fullscreen and opens an audio
stream, so it cannot be driven directly. Everything that decides the shape of
the picture is computed on the CPU from the tracker's readings, and all of it
is reachable on a FractalWindow built without __init__.
"""

import types

import numpy as np

import main
import phi_paths
import rhythm

SR = 44100
BLOCK = 1024


# ---- the instruments ---------------------------------------------------

def pink(n, rng, amp):
    """A noise floor with a 1/sqrt(f) tilt, which every recording has and
    no bare sum of sines does. It is not decoration: a band holding only
    this is what made the first flux detector invent 87 kicks."""
    w = rng.standard_normal(n)
    s = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    y = np.fft.irfft(s / np.sqrt(np.maximum(f, 20.0)), n)
    return amp * y / (np.abs(y).max() + 1e-9)


def harmonic_stack(t, f0, partials, amp):
    """An instrument, rather than a sine: a 1/k^0.8 harmonic series."""
    return sum(amp / (k ** 0.8) * np.sin(2 * np.pi * f0 * k * t)
               for k in range(1, partials + 1))


def kick_at(t, period):
    """A pitch sweep from ~165 Hz down to 45 Hz under a fast decay."""
    ph = t % period
    return 0.9 * np.exp(-ph * 28.0) * np.sin(
        2 * np.pi * (120.0 * np.exp(-ph * 12.0) + 45.0) * ph)


def snare_at(t, period, offset, rng):
    ph = (t - offset) % period
    return 0.5 * np.exp(-ph * 40.0) * rng.standard_normal(t.size)


def hats_at(t, period, rng):
    return 0.18 * np.exp(-(t % period) * 120.0) * rng.standard_normal(t.size)


def song(t, rng, beat=0.5):
    """A full-production track with a structure: a sparse verse, a dense
    chorus, and a bridge, each 30 s, repeating. The rhythm section runs
    throughout at `beat`; what changes between sections is how much is
    playing over it and in what key, which is what a real arrangement does.
    """
    section = int((t[0] // 30.0)) % 3
    bassline = 0.5 * np.sin(2 * np.pi * (55.0 if section != 2 else 73.0) * t)
    x = (kick_at(t, 2 * beat) + snare_at(t, 2 * beat, beat, rng)
         + hats_at(t, beat / 2, rng) + bassline + pink(t.size, rng, 0.06))
    if section == 0:                       # verse: a voice over the rhythm
        x = x + harmonic_stack(t, 220.0 + 30.0 * np.sin(2 * np.pi * 0.12 * t), 6, 0.22)
    elif section == 1:                     # chorus: everything at once
        x = (x + harmonic_stack(t, 330.0, 8, 0.20) + harmonic_stack(t, 165.0, 10, 0.16)
             + harmonic_stack(t, 440.0, 12, 0.14) + harmonic_stack(t, 660.0, 14, 0.10)
             + pink(t.size, rng, 0.10))
    else:                                  # bridge: a pad and a high line
        x = x + harmonic_stack(t, 293.0, 10, 0.18) + harmonic_stack(t, 880.0, 6, 0.08)
    return x


# ---- the tracker, recorded once ---------------------------------------

TRACKED = ("amp", "bass", "mid", "treble", "vocal", "density", "vocal_pitch",
           "n_kick", "n_snare", "n_hat", "n_bass", "sections", "cycle_len",
           "cycle_conf", "cycle_seconds", "cycle_phase", "pulse_s",
           "pitch_class", "hue")


class Replay:
    """One recorded block of tracker readings, quacking like a tracker."""

    def __init__(self, row):
        self.__dict__.update(row)


def record(seconds, sig=song, seed=3):
    """Run the real tracker over the song once and keep what it read, so
    several modes can be driven over identical music without paying for
    the analysis again (it is by far the slowest part)."""
    rng = np.random.default_rng(seed)
    tr = rhythm.MusicTracker(SR, BLOCK)
    rows = []
    for i in range(int(seconds * SR / BLOCK)):
        t = (np.arange(BLOCK) + i * BLOCK) / SR
        tr.process(sig(t, rng))
        rows.append({k: getattr(tr, k) for k in TRACKED})
    return rows


# ---- the window, headless ---------------------------------------------

def make_window(mode=1, terms=60):
    """A FractalWindow with every attribute the path, fold and voice
    updates touch, and nothing that needs a GL context or a sound card."""
    w = main.FractalWindow.__new__(main.FractalWindow)
    w.args = types.SimpleNamespace(terms=terms, iters=128, scale=0.75)
    w.mode = mode
    w.angle = main.GOLDEN_ANGLE_DEG
    w.radii = phi_paths.sequence_radii(terms, radius_scale=1.1)
    w.path_s = {m: 0.0 for m in main.MODE_NAMES}
    w.blend_phase = 0.0
    w.paused = False
    w.rabbit = False
    w.flower_pos, w.flower_dir, w.flower_mirror = 2.0, 1, False
    w.power_idx, w.power_time = 1, 0.0
    w.metallic = phi_paths.metallic_profiles(terms, radius_scale=1.1)
    w.metal_theta, w.family_pos, w.family_dir = 0.0, 0.0, 1
    w.metal_label = w.metallic[0][0]
    w.farey = phi_paths.FareyWalk(q_max=main.FAREY_Q_MAX)
    w.farey_f, w.farey_kicks = 0.0, 0
    w.formula, w.p_param = phi_paths.FORMULA_POWER, 0j
    w.mandel_idx, w.mandel_time = 0, 0.0
    w.spin, w.spin_paused = 0.0, False
    w.zoom_phase, w.zoom_cycling = 0.0, True
    w.kick = w.snare = w.hat = w.bass_pulse = 0.0
    w.hue_steps = w.hue_smooth = 0.0
    w.seen = dict(kick=0, snare=0, hat=0, bass=0, section=0)
    w.density_slow = 0.0
    w.voice_out, w.voice_petals, w.petals_held = 0.0, 5.0, None
    w.fold_from = w.fold_to = 1
    w.fold_mix, w.fold_changed_at = 1.0, 0.0
    w.now = 0.0
    w.flower_arms, w.power = 0, 2
    w.toasts = []
    w.show_toast = w.toasts.append
    return w


def drive(rows, mode, dt=1.0 / 50.0, terms=60):
    """Replay recorded tracker readings through one mode and return, for
    every frame, the values that decide what the picture looks like."""
    w = make_window(mode=mode, terms=terms)
    seconds = len(rows) * BLOCK / SR
    out = {k: [] for k in ("t", "c", "folds", "twist", "arms", "power",
                           "formula", "zoom", "center", "voice_out")}
    t = 0.0
    while t < seconds:
        w.tr = Replay(rows[min(int(t * SR / BLOCK), len(rows) - 1)])
        w.now = t
        kick_hit, _section = w._update_layers(dt)
        c, power, center, zoom = w._update_path(dt, w.tr.amp, kick_hit)
        w._update_fold(dt, w.tr.density)
        w._update_voice(dt)
        out["t"].append(t)
        out["c"].append(c)
        out["folds"].append(w.fold_to)
        # the same expression on_render hands the shader as u_twist
        out["twist"].append(w.fold_amount() * 1.2 + 0.5 * w.bass_pulse)
        out["arms"].append(w.flower_arms)
        out["power"].append(power)
        out["formula"].append(w.formula)
        out["zoom"].append(zoom)
        out["center"].append(center)
        out["voice_out"].append(w.voice_out)
        t += dt
    return out
