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
  voice     vocal_pitch, the melodic contour of the voice: where the energy
            sits inside the vocal formant range, 0 (low) to 1 (high). This is
            a contour, not pitch detection -- it follows a line rising and
            falling without ever naming a note, which is what a shape that
            moves with the singing needs. It holds its last value when the
            voice drops out, so the shape it drives settles instead of
            snapping to the floor
  density   how much of the mix is above the bass: 0 for a bass-and-drums
            verse, 1 for a full arrangement
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
    VOICE_LOG_LO = math.log(150.0)   # the span the vocal contour is placed in
    VOICE_LOG_HI = math.log(800.0)
    VOICE_SWING = 1.1                # how much of 0..1 a phrase's range fills
    VOICE_PROMINENCE = 2.5           # partial over its neighbours to count as a voice
    VOICE_LOWEST = 0.6               # how strong a lower partial must be to win
    VOICE_HARMONIC = 1.8             # prominence at 2f or 3f confirming a voice
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

        # the voice's melodic contour, and how full the arrangement is
        self.vocal_pitch = 0.5
        self.density = 0.0
        self._vp_centre = None    # the singer's own register, tracked slowly
        self._vp_last = None      # last accepted partial, for the agreement test
        self._voice_mask = (self.freqs >= 150) & (self.freqs < 1000)
        self._voice_freqs = self.freqs[self._voice_mask]
        self._low_mask = self.freqs < 250
        self._high_mask = (self.freqs >= 250) & (self.freqs < 8000)

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

        self._update_voice(spec)

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

    def _update_voice(self, spec):
        """How full the arrangement is.

        Density is the share of the spectrum above the bass, taken from raw
        magnitudes rather than the auto-gained levels -- those are rescaled
        to each track's own recent range, so by that measure a drums-only
        loop reads as "full" exactly as much as an orchestra does.
        """
        low = float(spec[self._low_mask].sum())
        high = float(spec[self._high_mask].sum())
        self.density += (high / (low + high + 1e-12) - self.density) * 0.05

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
        self._update_contour(spec, f)
        self.pitch_strength = float(self._chroma.max() * 12 - 1)  # 0 = flat, larger = one clear note
        # hue follows the pitch class round the circle, taking the short way
        target = complex(math.cos(2 * math.pi * self.pitch_class / 12),
                         math.sin(2 * math.pi * self.pitch_class / 12))
        self._hue_vec = 0.96 * self._hue_vec + 0.04 * target
        self.hue = (math.atan2(self._hue_vec.imag, self._hue_vec.real) / (2 * math.pi)) % 1.0

    def _update_contour(self, spec, f):
        """Where the voice is sitting in its range, 0 (low) to 1 (high).

        Shares the 4096-sample spectrum above because that is the only one
        with the resolution for it: the per-block FFT has 43 Hz bins, which
        across a sung fundamental is a span of several semitones and useless
        as a contour. This one has 10.8 Hz bins.

        What it looks for is a partial standing above its *neighbours*, not
        above the band's median. A median test sounds equivalent and isn't:
        a loud kick leaks a smooth decaying slope across the whole range, on
        which the lowest bin towers over the median, and the contour then
        happily tracks the drums -- measured, it scored a cleaner "voice"
        with the singer muted than with the singer in.

        Three things it is deliberately not:

          - not a pitch tracker. It never names a note, it just rises and
            falls with the line, which is the gesture worth following and is
            far more robust against a full mix than a tracker that
            octave-jumps -- and an octave jump is a jolt in whatever it
            drives.
          - not a spectral centroid. Weighted over linearly spaced bins, a
            centroid sits wherever the broadband content is and barely moves
            when the voice does; a cymbal pins it as firmly as a singer.
          - not run through AutoGain like the levels. That tracks a running
            minimum, so a line descending for several seconds is its own new
            low at every step and reads as a flat zero the whole way down --
            the exact gesture this exists to catch. Only the centre of the
            span moves, and slowly enough (tens of seconds) that no phrase
            is flattened.

        The prominence-and-agreement test *is* the detector; there is no
        separate gate on the vocal level, which was the first design and a
        bad one. Those levels are auto-gained, which measures deviation from
        the recent range, so a voice that simply keeps singing settles to
        zero within seconds and switches its own gate off -- measured, the
        contour froze three seconds into a twelve-second phrase. A
        structural test ("is there a steady periodic partial in the sung
        range") does not rot that way, and an instrumental passage fails it
        anyway, which leaves the last contour standing as intended.

        The level still decides how *present* the shape driven by this is;
        it just has no business deciding whether the line exists.
        """
        # Out to 4 kHz, because the harmonic test below needs to look at
        # twice and three times a candidate down at 1 kHz.
        m = (f >= 150) & (f < 4000)
        band, freqs = spec[m], f[m]
        if band.size < 32:
            return
        local = np.convolve(band, np.ones(21) / 21, mode="same")
        score = band / (local + 1e-12)

        sung = freqs < 1000
        best = float(score[sung].max())
        if best < self.VOICE_PROMINENCE:
            self._vp_last = None
            return

        # The lowest partial that is nearly as strong as the strongest, not
        # the strongest outright: a harmonic often out-peaks the fundamental
        # it belongs to, and following it jumps an octave mid-note.
        strong = np.flatnonzero(sung & (score >= max(self.VOICE_PROMINENCE,
                                                     best * self.VOICE_LOWEST)))
        # ...and it has to have harmonics. Without this the "lowest partial"
        # rule is actively harmful: a noise burst's prominent bins get picked
        # at the bottom of the range every time, so successive bursts agree
        # with each other and walk the contour down to the floor. A voice has
        # energy at 2f and 3f; a cymbal does not.
        i = None
        for k in strong:
            f0 = float(freqs[k])
            support = sum(score[int(np.argmin(np.abs(freqs - mult * f0)))] > self.VOICE_HARMONIC
                          for mult in (2, 3))
            if support:
                i = int(k)
                break
        if i is None:
            self._vp_last = None
            return

        # A partial sitting an octave above another real partial is that
        # one's second harmonic, not a fundamental. The "nearly as strong
        # as the strongest" rule above is *relative*, and that is how the
        # fundamental gets lost mid-note: anything crowding it -- a pad, a
        # second instrument, reverb filling the low-mid -- raises the local
        # average around it, so its prominence drops below the loudest
        # harmonic's share of `best` while the harmonic, sitting in a
        # cleaner part of the spectrum, keeps all of its own. Measured on a
        # held 220 Hz note, the fundamental fell from 8.8 to 4.0 when a
        # 150-340 Hz pad came in -- still plainly a partial, but under
        # best * VOICE_LOWEST, and the contour lurched an octave.
        #
        # Prominence against its own neighbours does not rot that way, so
        # the absolute bar is what decides whether the octave below is
        # really there. Twice, to catch a latch two octaves up.
        for _ in range(2):
            half = float(freqs[i]) / 2.0
            if half < freqs[0]:
                break
            j = int(np.argmin(np.abs(freqs - half)))
            if score[j] <= self.VOICE_PROMINENCE:
                break
            i = j

        raw = ((math.log(float(freqs[i])) - self.VOICE_LOG_LO)
               / (self.VOICE_LOG_HI - self.VOICE_LOG_LO))

        # Two consecutive looks have to agree on roughly the same partial.
        # A held or gliding note moves a fraction of this between analyses
        # 90 ms apart; a prominent bin thrown up by a cymbal or a noise
        # burst lands somewhere different next time and never gets in.
        agreed = self._vp_last is not None and abs(raw - self._vp_last) < 0.12
        self._vp_last = raw
        if not agreed:
            return

        if self._vp_centre is None:
            self._vp_centre = raw
        self._vp_centre += (raw - self._vp_centre) * 0.0004
        target = min(1.0, max(0.0, 0.5 + (raw - self._vp_centre) * self.VOICE_SWING))
        # Eased, so a shape following this glides along the line rather than
        # stepping from one analysis to the next.
        self.vocal_pitch += (target - self.vocal_pitch) * 0.12

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
