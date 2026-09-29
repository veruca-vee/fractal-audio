# Sequence-driven audio-reactive fractal

A Julia set whose `c` constant follows one of three phi-flavored paths:
a golden-angle spiral that drifts between your halving-variant Fibonacci,
the doubled variant and plain Fibonacci; Fibonacci-ratio "flowers"; or
z^3 / z^5 / z^8 + c sets on a golden quasi-periodic path. Every layer of
the music is measured and does one thing (table under "What follows what"):
kick, snare, hats, bass notes, vocals, pitch, the beat, how long the
pattern takes to repeat, and section changes. The view is flat and slowly
spinning.

## Files

- `sequences.py`  — the three sequence generators + golden-angle spiral walk
- `phi_paths.py` — the path `c` follows in each of the three modes
- `rhythm.py` — the music analysis: levels, onsets, pulse, repeat length, pitch, sections
- `julia.frag` / `julia.vert` — the GLSL Julia set shader
- `main.py` — ties audio capture to the shader via moderngl, and writes the session log
- `tools/check_rhythm.py` — runs an audio file through the same analysis and prints what it hears
- `logs/` — a CSV per session, with a `.summary.txt` when you quit

## Setup — Debian distrobox on Bluefin

GPU passthrough into a distrobox needs the host's GL drivers visible inside
the container. Create it with `--nvidia` (if you're on Nvidia) or plain
`--init-hooks` for Mesa/AMD/Intel — distrobox handles this automatically for
most setups since it binds the host's `/usr/lib` GL stack in by default.
If `glxinfo` inside the box doesn't show your real GPU, that's the thing to
fix first — moderngl needs a real GL 3.3 context, not llvmpipe software
rendering (it'll run, just at a few FPS).

```bash
# from the host, if you don't already have a general-purpose box:
distrobox create --name fractal-box --image debian:trixie
distrobox enter fractal-box
```

Inside the box:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv \
    libgl1 libglvnd0 mesa-utils \
    libportaudio2 portaudio19-dev \
    pipewire-audio-client-libraries

glxinfo | grep "OpenGL renderer"   # should NOT say "llvmpipe" if you want real speed

python3 -m venv venv
source venv/bin/activate
pip install moderngl moderngl-window numpy sounddevice
```

### This machine (Ubuntu 26.04 box, Python 3.14)

There's no prebuilt `glcontext` wheel for Python 3.14, so pip builds it from
source and needs the X11/GL headers first:

```bash
sudo apt install -y python3-venv python3-pip python3-dev \
    libportaudio2 portaudio19-dev mesa-utils pulseaudio-utils \
    libx11-dev libgl-dev libegl-dev
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Routing PipeWire audio in

You want the *monitor* of whatever's playing your music (e.g. the sink
monitor), not a mic. From inside the box (PipeWire client libs are shared
with the host session):

```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

Look for an entry with "Monitor" in the name — that's your loopback of
system output. Note its index or exact name, then either export it or pass
it directly:

```bash
python3 main.py --device "Monitor of Built-in Audio Analog Stereo"
# or by index:
python3 main.py --device 4
```

If nothing shows a monitor device, PipeWire's pulse-compat layer may not be
exposing it to the container — run `pactl list sources short` on the host
first to confirm the monitor source name exists, then check that
`pipewire-pulse` is reachable from inside the box (it usually is, since
distrobox shares the host's runtime dir by default).

## Running

```bash
python3 main.py --mode 1 --terms 60 --angle 137.5
```

On this machine the monitor source is
`alsa_output.pci-0000_00_0e.0.analog-stereo.monitor`, so:

```fish
source venv/bin/activate.fish
python main.py --device analog-stereo.monitor
```

It starts fullscreen.

| Flag | Default | Effect |
|---|---|---|
| `--mode` | `1` | starting mode (see keys below) |
| `--scale` | `0.75` | render resolution relative to the window; lower is faster, softer |
| `--iters` | `128` | max Julia iterations; lower is faster, higher is more detail |
| `--fps` | off | print frames per second to stderr every 5 s |
| `--no-log` | off | don't write a session log to `logs/` |

Keys while it's running:

| Key | Effect |
|---|---|
| `1` | Sequence drift: golden-angle spiral drifting between halving / doubled / Fibonacci |
| `2` | Fibonacci flowers: 2, 3, 5 ... 55-armed Julia flowers; the music's repeat length picks one |
| `3` | Fibonacci powers: z³ / z⁵ / z⁸ + c (3/5/8-fold), cutting between them on kicks |
| `r` | the Douady rabbit (c = −0.123 + 0.745i), on/off. Fibonacci's original sequence came from a rabbit-breeding problem |
| `[` `]` | nudge mode 1's spiral angle down/up by 0.5° (try 179.5° for the S-curve) |
| `space` | pause/resume the path |
| `z` | freeze/resume the zoom cycle |
| `p` | next color palette (Prism, Ember, Ocean, Vapor, Acid, Ice) |
| `a` | toggle automatic palette changes (on by default: on each section change, or after 60 s) |
| `c` | pause/resume the spin |
| `f11` | toggle fullscreen |
| `esc` | quit |

## What follows what

Every level is auto-gained (measured in log units, rescaled against its own
recent low and high) so it swings across 0..1 on any track at any volume.

| Layer of the music | How it's measured | What it does |
|---|---|---|
| Kick | 40–150 Hz energy jumping above its recent average | zoom punch, brightness flash, spin jolt; shoves c in modes 2 and 3 |
| Snare | 150–400 Hz *and* 2–6 kHz jumping together | the colours flip toward their negative, and the hue steps round |
| Hats | 7–14 kHz onsets | fine contour lines flick through the escape-time bands (and ripple the stalks inside the set) |
| Bass notes | 60–250 Hz onsets | swirl: twist the picture into spiral arms |
| Bass level | 20–250 Hz level | breathes the zoom |
| Vocals | 300–3400 Hz relative to the lows and highs | past a threshold, fold the fractal into up to 6 copies twisted into spirals |
| Pitch | dominant pitch class of 80–1000 Hz | shifts the palette's hue, moving the short way round the circle |
| Loudness | RMS | speed of c's path when there's no steady repeat to follow |
| Repeat length | autocorrelation of the onset envelope, in pulses | the pace of c's path: mode 1 steps one sequence term per bar, mode 3 makes one loop per bar, mode 2 picks the flower whose arm count is nearest |
| Section change | the mix shifting for 3 s against the 3 s before | next palette |

The repeat length is *measured*, not assumed from a time signature. The
pulse is the shortest steady beat (0.25–0.5 s); the repeat length is the
smallest number of pulses whose autocorrelation peak stands out, stabilized
over five readings. A "bar" here is the repeat stretched to at least 6 s
(mode 1) or 8 s (mode 3), so the path isn't frantic on a short pattern.

Check the tracker against a track whose structure you know:

```fish
./venv/bin/python tools/check_rhythm.py "~/Music/some song.mp3" 15
```

It prints the pulse, repeat length, onset rates, dominant note and section
count every 15 s, then how often each repeat length came up and whether it
was a Fibonacci number. On Mos Def's *Mathematics* it locks to a 0.322 s
pulse and a 4-pulse repeat (the kick-snare pair), with no Fibonacci lengths.
On Tool's *Lateralus* the pulse wobbles between 0.26 and 0.35 s, it finds a
24-pulse repeat (9+8+7) around 1:44, and 44% of repeat lengths are
Fibonacci numbers (3, 5, 8) against 16% by chance. That's suggestive, not
proof: 3 and 5 are small, common numbers, and the 9/8, 8/8, 7/8 pattern
shows up in only a few sections.

## Where the math actually shows up

- `c` (the Julia constant) moves continuously along the current mode's
  path, and the path *is* the journey the fractal takes through parameter
  space. With a steady repeat to follow it runs at the music's pace;
  otherwise louder music moves it faster (speed goes with loudness squared,
  so quiet and loud passages read clearly apart).
- Mode 1, Sequence drift: the angle advances by the golden angle per term;
  the radius is the log-scaled sequence value, crossfading between halving,
  doubled and Fibonacci every ~40 s. (Doubled and Fibonacci are both
  exponential, so their log radii are nearly the same line; halving's
  folds are what make it different.) It's the calm mode: kicks only flash
  and punch the zoom, they don't move c.
- Mode 2, Fibonacci flowers: c sits on the Mandelbrot main cardioid at
  rotation numbers 1/2, 2/3, 3/5 ... 34/55, the ratios of consecutive
  Fibonacci numbers, which converge to 1/phi. Just past the cardioid,
  inside each p/q bulb, the Julia set is q arms meeting at the fixed point,
  so you get 2, 3, 5 ... 55-armed flowers approaching the golden-mean
  Siegel disk. The view centers on that fixed point. The music's repeat
  length picks the flower with the nearest arm count (a 24-pulse repeat
  settles on 21 arms, a 12 or 16 on 13). With no steady repeat it wanders up
  and down the list on its own, running mirrored on alternate passes. Kicks
  blow the flower out of its bulb into a spiky dendrite that re-forms as
  the kick fades.
- Mode 3, Fibonacci powers: z^d + c has d-fold symmetry, and d steps
  through the Fibonacci numbers 3, 5, 8, cutting on the first kick after
  12 s (or at 24 s regardless). c rides around the main component,
  orbiting at rate 1 while its radius wobbles at rate phi between filled
  shapes, spiral arms and dust; phi is the "most irrational" number, so
  the path never closes on itself. Kicks shove it outward.
- Zoom cycles every 24 s between each mode's base framing and 1.8× wider;
  bass breathes it in and kicks punch it in.
- Vocals: once the voice stands out it sets a fold count from 1 to 6: the
  plane is mirrored into that many wedges, each holding a smaller copy
  pushed out into a ring, then twisted by log(r) into spiral arms. Changes
  crossfade.
- Color: cosine palettes that crossfade. Hue tracks escape-time + mid-band
  energy, saturation tracks treble, value tracks escape-time, bass, and
  overall amplitude. The inside of the set is colored by how close each
  orbit passes to the axes, which draws the glowing stalks.

## Tuning notes

If it stutters, drop `--scale` to `0.6` or `--iters` to `96` first.

The onset detectors and their thresholds are in `rhythm.py` (`MusicTracker.__init__`):
the first number in each `OnsetDetector(jump, refractory)` is how far above
its recent average a band must jump, so higher means fewer hits. If vocals
split the fractal too eagerly, raise the 0.45 threshold in `vocal_split()`
in `main.py`.

The session log (`logs/session-*.csv`) has one row every half second: mode,
fps, pulse, repeat length and confidence, position in the repeat, dominant
note, section count, hits per second for each layer, the levels, the fold
count, the power, the flower's arm count and the palette. The
`.summary.txt` next to it is printed when you quit. It's the place to check
whether the tracker heard what you heard.
