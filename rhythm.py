"""
Music analysis, one block of samples at a time. No audio I/O in here, so the
same code runs live (main.py) and offline against a decoded file
(tools/check_rhythm.py).

Every layer the shapes can follow, and how it's measured:

  levels    amp / bass / mid / treble / vocal, each auto-gained to 0..1
            against its own recent range, so they swing fully on any track
  onsets    kick, snare, hats, bass notes: a band's energy jumping above its
            own recent average (the snare needs a low-mid AND a high jump
            together, so it means "broadband crack", not "any guitar note")
  pulse     the shortest steady beat unit (~0.25-0.5 s), refined so its
            multiples line up with the autocorrelation peaks
  cycle     how many pulses the pattern takes to repeat (the "bar"), found as
            the smallest autocorrelation peak that stands out, then
            stabilized. This is measured, not assumed, so a 9/8 - 8/8 - 7/8
            riff shows up as whatever it actually repeats at
  phase     where we are inside the pulse and inside the cycle, from folding
            the recent onset envelope at the cycle length
  pitch     the dominant pitch class (C, C#, ...) of the 80-1000 Hz range,
            as a hue that moves the shortest way round the circle
  sections  a big, sustained change in the overall mix
"""

import math
from collections import deque

import numpy as np

PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FIBONACCI = (1, 2, 3, 5, 8, 13, 21, 34, 55)


class AutoGain:
    """Rescales a level (in log units) to 0..1 against its own recent range,
    which relaxes inward slowly so it adapts to quiet and loud passages
    within ~10 seconds."""

    def __init__(self, drift=0.002, min_span=1.0):
        self.lo = None
        self.hi = None
        self.drift = drift        # log units per block
        self.min_span = min_span  # never stretch a narrower range, so silence stays dark

    def __call__(self, x):
        if self.lo is None:
            self.lo, self.hi = x, x
        self.lo = min(x, self.lo + self.drift)
        self.hi = max(x, self.hi - self.drift)
        span = max(self.hi - self.lo, self.min_span)
        return min(max((x - self.lo) / span, 0.0), 1.0)


class OnsetDetector:
    """Fires when a fast average of some log-energy jumps `jump` above its
    slow average, then stays quiet for `refractory` blocks."""

    def __init__(self, jump, refractory):
        self.jump = jump
        self.refractory = refractory
        self.fast = None
        self.slow = None
        self.since = 10 ** 6

    def excess(self, x):
        if self.fast is None:
            self.fast = self.slow = x
        self.fast = 0.5 * self.fast + 0.5 * x
        self.slow = 0.97 * self.slow + 0.03 * x
        return self.fast - self.slow

    def update(self, x, extra_ok=True):
        e = self.excess(x)
        self.since += 1
        if e > self.jump and extra_ok and self.since >= self.refractory:
            self.since = 0
            return True
        return False


def wrap_half(x):
    """Wrap to [-0.5, 0.5)."""
    return (x + 0.5) % 1.0 - 0.5


class MusicTracker:
    HISTORY_S = 30.0
    ANALYZE_EVERY_S = 0.5
    MIN_HISTORY_S = 8.0
    PULSE_RANGE_S = (0.25, 0.5)
    CYCLE_RANGE = (3, 40)

    def __init__(self, sample_rate, block_size):
        self.sr = sample_rate
        self.block = block_size
        self.rate = sample_rate / block_size   # blocks per second
        self.window = np.hanning(block_size)
        self.freqs = np.fft.rfftfreq(block_size, 1.0 / sample_rate)
        self.frame = 0

        # levels, 0..1
        self.amp = self.bass = self.mid = self.treble = self.vocal = 0.0
        self._gain = {k: AutoGain() for k in ("amp", "bass", "mid", "treble", "vocal")}
        self._smooth = 0.5

        # onset counters; the render loop watches them change
        self.n_kick = self.n_snare = self.n_hat = self.n_bass = 0
        self._kick = OnsetDetector(0.7, 7)
        self._snare_lo = OnsetDetector(0.5, 6)
        self._snare_hi = OnsetDetector(0.5, 6)
        self._hat = OnsetDetector(0.6, 3)
        self._bassnote = OnsetDetector(0.6, 8)

        # rhythm
        n = int(self.HISTORY_S * self.rate)
        self._env = deque(maxlen=n)
        self._prev_log = None
        self._flux_mask = (self.freqs >= 40) & (self.freqs < 10000)
        self.pulse_s = 0.0
        self.cycle_len = 0        # in pulses; 0 until it has enough music to say
        self.cycle_conf = 0.0     # 0..1
        self._cycle_votes = deque(maxlen=5)
        self._ref_frame = 0.0     # frame at which a cycle starts
        self._have_ref = False

        # pitch
        self._pcm = deque(maxlen=4096)
        self._chroma = np.zeros(12)
        self.pitch_class = 0
        self.pitch_strength = 0.0
        self._hue_vec = complex(1, 0)
        self.hue = 0.0

        # sections
        self.sections = 0
        self._feat_hist = deque(maxlen=int(6.0 * self.rate))
        self._section_frame = -10 ** 6

    # ---- per-block ----------------------------------------------------

    def process(self, samples):
        x = np.asarray(samples, dtype=np.float64)
        spec = np.abs(np.fft.rfft(x * self.window)) / (0.5 * self.window.sum())
        f = self.freqs

        def band(lo, hi):
            return math.log(float(np.mean(spec[(f >= lo) & (f < hi)])) + 1e-6)

        bass = band(20, 250)
        treble = band(2000, 8000)
        levels = {
            "amp": math.log(float(np.sqrt(np.mean(x ** 2))) + 1e-6),
            "bass": bass,
            "mid": band(250, 2000),
            "treble": treble,
            "vocal": band(300, 3400) - 0.5 * (bass + treble),
        }
        s = self._smooth
        for name, v in levels.items():
            setattr(self, name, s * getattr(self, name) + (1 - s) * self._gain[name](v))

        # onsets
        if self._kick.update(band(40, 150)):
            self.n_kick += 1
        lo_ok = self._snare_lo.excess(band(150, 400)) > self._snare_lo.jump
        if self._snare_hi.update(band(2000, 6000), extra_ok=lo_ok):
            self.n_snare += 1
        if self._hat.update(band(7000, 14000)):
            self.n_hat += 1
        if self._bassnote.update(band(60, 250)):
            self.n_bass += 1

        # rhythm envelope: half-wave-rectified spectral flux, 40-10000 Hz
        cur = np.log1p(spec[self._flux_mask] * 50)
        flux = 0.0 if self._prev_log is None else float(np.maximum(cur - self._prev_log, 0).sum())
        self._prev_log = cur
        self._env.append(flux)

        self.frame += 1
        if self.frame % max(1, int(self.ANALYZE_EVERY_S * self.rate)) == 0:
            self._analyze_rhythm()
        self._update_pitch(x)
        self._update_sections()

    # ---- rhythm -------------------------------------------------------

    def _acf(self, e):
        n = len(e)
        e = e - np.convolve(e, np.ones(21) / 21, mode="same")  # drop the slow trend
        e = np.maximum(e, 0)
        e = e - e.mean()
        size = 1 << (2 * n - 1).bit_length()
        ac = np.fft.irfft(np.abs(np.fft.rfft(e, size)) ** 2)[:n]
        ac = ac / (n - np.arange(n))          # unbiased: long lags aren't penalized
        return ac / (ac[0] if ac[0] else 1.0), e

    def _analyze_rhythm(self):
        if len(self._env) < self.MIN_HISTORY_S * self.rate:
            return
        env = np.array(self._env)
        ac, e = self._acf(env)
        lags = np.arange(len(ac))

        def acv(sec):
            i = sec * self.rate
            return float(np.max(np.interp(i + np.array([-1, -0.5, 0, 0.5, 1]), lags, ac)))

        cands = np.arange(self.PULSE_RANGE_S[0], self.PULSE_RANGE_S[1], 0.002)
        p = float(max(cands, key=lambda c: sum(acv(k * c) for k in range(1, 9))))
        self.pulse_s = p if self.pulse_s == 0 else 0.7 * self.pulse_s + 0.3 * p
        p = self.pulse_s

        k_lo, k_hi = self.CYCLE_RANGE
        k_hi = min(k_hi, int(self.HISTORY_S / (2.5 * p)))
        ks = list(range(k_lo, k_hi + 1))
        pk = {k: acv(k * p) for k in range(k_lo - 3, k_hi + 4)}
        prom = {}
        for k in ks:
            around = [pk[j] for j in range(k - 3, k + 4) if j != k]
            prom[k] = pk[k] - float(np.median(around))
        best = max(prom.values())
        if best <= 0.015:
            self._cycle_votes.append(0)
            self.cycle_conf *= 0.7
        else:
            # smallest peak that's nearly as strong as the strongest
            k = min(k for k in ks if prom[k] >= 0.7 * best)
            self._cycle_votes.append(k)
            self.cycle_conf = min(1.0, prom[k] / 0.08)

        votes = [v for v in self._cycle_votes if v]
        if votes:
            top = max(set(votes), key=votes.count)
            if votes.count(top) >= 3 or not self.cycle_len:
                self.cycle_len = top

        if self.cycle_len:
            self._update_reference(e)

    def _update_reference(self, e):
        """Fold the envelope at the cycle length; the strongest point of the
        fold is where the cycle starts (usually the downbeat)."""
        period = self.cycle_len * self.pulse_s * self.rate   # frames
        n = len(e)
        first = self.frame - n                                # absolute frame of e[0]
        nb = max(8, self.cycle_len * 4)
        pos = ((np.arange(n) + first) % period) / period
        prof = np.bincount((pos * nb).astype(int) % nb, weights=e, minlength=nb)
        prof = prof + 0.5 * (np.roll(prof, 1) + np.roll(prof, -1))
        start = (int(np.argmax(prof)) + 0.5) / nb * period    # frames into a cycle
        # the cycle starts at start + m * period; take the one nearest the last reference
        m = round((self._ref_frame - start) / period) if self._have_ref else 0
        cand = start + m * period
        if self._have_ref and abs(wrap_half((cand - self._ref_frame) / period)) < 0.25:
            self._ref_frame = 0.7 * self._ref_frame + 0.3 * cand   # move gently
        else:
            self._ref_frame = cand
        self._have_ref = True

    @property
    def cycle_phase(self):
        """0..1 through the current cycle (0 = its start), or 0 without a cycle."""
        if not (self.cycle_len and self._have_ref and self.pulse_s):
            return 0.0
        period = self.cycle_len * self.pulse_s * self.rate
        return ((self.frame - self._ref_frame) / period) % 1.0

    @property
    def pulse_phase(self):
        if not (self._have_ref and self.pulse_s):
            return 0.0
        return ((self.frame - self._ref_frame) / (self.pulse_s * self.rate)) % 1.0

    @property
    def cycle_seconds(self):
        return self.cycle_len * self.pulse_s

    # ---- pitch --------------------------------------------------------

    def _update_pitch(self, x):
        self._pcm.extend(x)
        if self.frame % 4 or len(self._pcm) < 4096:
            return
        seg = np.array(self._pcm)
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        f = np.fft.rfftfreq(len(seg), 1.0 / self.sr)
        ok = (f >= 80) & (f < 1000)
        pc = np.round(12 * np.log2(f[ok] / 440.0) + 9).astype(int) % 12
        chroma = np.bincount(pc, weights=spec[ok], minlength=12)
        chroma /= chroma.sum() or 1.0
        self._chroma = 0.85 * self._chroma + 0.15 * chroma
        self.pitch_class = int(np.argmax(self._chroma))
        self.pitch_strength = float(self._chroma.max() * 12 - 1)  # 0 = flat, larger = one clear note
        # hue follows the pitch class round the circle, taking the short way
        target = complex(math.cos(2 * math.pi * self.pitch_class / 12),
                         math.sin(2 * math.pi * self.pitch_class / 12))
        self._hue_vec = 0.96 * self._hue_vec + 0.04 * target
        self.hue = (math.atan2(self._hue_vec.imag, self._hue_vec.real) / (2 * math.pi)) % 1.0

    @property
    def pitch_name(self):
        return PITCH_NAMES[self.pitch_class]

    # ---- sections -----------------------------------------------------

    def _update_sections(self):
        self._feat_hist.append((self.amp, self.bass, self.mid, self.treble, self.vocal))
        n = int(3.0 * self.rate)
        if len(self._feat_hist) < 2 * n or self.frame - self._section_frame < 8 * self.rate:
            return
        h = np.array(self._feat_hist)
        change = float(np.abs(h[-n:].mean(0) - h[-2 * n:-n].mean(0)).sum())
        if change > 0.9:
            self.sections += 1
            self._section_frame = self.frame
