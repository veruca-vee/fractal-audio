# Sequence-driven audio-reactive fractal

A Julia set whose `c` constant follows one of seven phi-flavored paths:
a golden-angle spiral that drifts between your halving-variant Fibonacci,
the doubled variant and plain Fibonacci; Fibonacci-ratio "flowers"; or
z^3 / z^5 / z^8 + c sets on a golden quasi-periodic path — plus four more
that take each of those off its rails: the same spiral run on the other
metallic means (bronze, silver, plastic, each with its own divergence
angle), the flowers generalized from the Fibonacci ratios to a walk down
the Stern-Brocot tree, and two maps that aren't z^d + c at all (the
Phoenix map's one step of memory, and the antiholomorphic tricorn and
burning ship). Every layer of the music is measured and does one thing
(table under "What follows what"): kick, snare, hats, bass notes, vocals,
pitch, the beat, how long the pattern takes to repeat, and section
changes. The view is flat and slowly spinning.

## Files

- `sequences.py`  — the integer sequences (Fibonacci, Lucas, Pell, bronze, tribonacci, Padovan, Perrin, Jacobsthal, and the two hand-rolled variants), the metallic means and their divergence angles, + the golden-angle spiral walk
- `phi_paths.py` — the path `c` follows in each of the seven modes, and which map the shader iterates for each
- `pw_monitor.py` — finds the PipeWire sink whose monitor carries system output
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
    libgl1 libglvnd0 libgl-dev mesa-utils \
    libportaudio2 portaudio19-dev \
    pipewire-audio-client-libraries

glxinfo | grep "OpenGL renderer"   # should NOT say "llvmpipe" if you want real speed

python3 -m venv venv
source venv/bin/activate
pip install moderngl moderngl-window numpy sounddevice
```

### This machine (Debian trixie distrobox, Python 3.13)

The box is `fractal-box` (`debian:trixie`) on a Bluefin host, with the host's
Mesa stack bound in — `glxinfo` reports `Mesa Intel(R) UHD Graphics 600`
and GL 4.6, so rendering is on the real GPU, around 55 fps at the defaults.

`libgl-dev` is not optional, even though nothing here is compiled: moderngl
loads GL with `ctypes.CDLL("libGL.so")`, the unversioned symlink that only the
dev package ships. With just `libgl1` you get `libGL.so.1`, `glxinfo` works
fine, and moderngl still dies with
`OSError: libGL.so: cannot open shared object file`.

```bash
sudo apt install -y python3-venv python3-pip python3-dev \
    libportaudio2 portaudio19-dev mesa-utils \
    libx11-dev libgl-dev libegl-dev
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

If you build the venv on a Python with no prebuilt `glcontext` wheel (3.14 at
the time of writing), pip compiles it from source, which is what the X11/GL
headers above are for.

## Routing PipeWire audio in

You want the *monitor* of whatever's playing the music, not a mic. Pass
`--device monitor` and it finds it for you:

```fish
python main.py --device monitor
```

It prints which sink it attached to, e.g. `capturing the monitor of
alsa_output.pci-0000_00_0e.0.analog-stereo (node 47)`. With more than one
output, `--device monitor=hdmi` picks the first sink whose name contains
that fragment; plain `monitor` follows the current default sink.

It has to find it for you, because you can't just name it. Listing devices:

```fish
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

...shows no monitor at all — only `default`, `pipewire`, `sysdefault` and the
raw `hw:` cards. PortAudio (under sounddevice) builds its list from ALSA and
has no PulseAudio backend, so PulseAudio-style source names
(`alsa_output.<card>.analog-stereo.monitor`, the kind `pactl list sources`
prints) are invisible to it. The ALSA `pipewire` device *is* there, but it
follows the default **source** — the microphone.

What works is the pipewire ALSA plugin's own target, read from
`$PIPEWIRE_NODE`: capture against a **sink** and PipeWire links you to that
sink's monitor ports. `pw_monitor.py` resolves the sink to a node id and sets
it. Two sharp edges it exists to handle:

- That variable takes only a **numeric node id** here. A node *name* is
  silently ignored and you get the microphone — and a mic next to playing
  speakers shows plenty of signal, so it looks like it worked. Check the
  routing, not the level: `pw-link -l` should show your stream fed by
  `...analog-stereo:monitor_FL`, not `alsa_input...:capture_FL`. For the same
  reason `--device monitor` fails loudly rather than falling back.
- Node ids are handed out at runtime and change across reboots and device
  changes, so the id is resolved at launch instead of written down.

An `~/.asoundrc` with a `type pipewire` PCM and `capture_node` is the tidier
answer on paper, and it does get the PCM enumerated (via its `hint` block),
but this client build (1.4.2 against a 1.6.8 server) ignores `capture_node`
and hands back the mic. Hence the env var.

If `--device monitor` reports no sinks, PipeWire isn't reachable from the box:
check that `pw-dump` runs and that `/run/user/1000/pipewire-0` is present
(distrobox shares the host's runtime dir by default).

## Running

```bash
python3 main.py --mode 1 --terms 60 --angle 137.5
```

So on this machine:

```fish
source venv/bin/activate.fish
python main.py --device monitor
```

It starts fullscreen.

| Flag | Default | Effect |
|---|---|---|
| `--device` | default input | `monitor` for the monitor of system output, `monitor=<fragment>` to pick a sink, else a sounddevice name or index |
| `--mode` | `1` | starting mode, `1`–`7` (see keys below) |
| `--terms` | `60` | modes 1 and 4: how many sequence terms the spiral walks; more means a slower sweep outward |
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
| `4` | Metallic spirals: mode 1 off phi — bronze (33°), silver (61.8°), golden (137.5°) and plastic (205.1°), walked end to end over ~100 s |
| `5` | Farey tour: mode 2 off the Fibonacci rails — c descends the Stern-Brocot tree, kicks pick the branch, arm counts come out as mediants |
| `6` | Phoenix wings: z² + c + p·z_prev, one step of memory; p swings between −1/φ and −1/φ² |
| `7` | Mandelbar flames: tricorn (z̄^d + c) and burning ship, d = 2 or 3, cutting between the four on kicks |
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
| Kick | 40–150 Hz energy jumping above its recent average | zoom punch, brightness flash, spin jolt; shoves c in modes 2, 4, 5, 6 and 7; cuts the power in mode 3 and the map in mode 7; picks the branch in mode 5 |
| Snare | 150–400 Hz *and* 2–6 kHz jumping together | the colours flip toward their negative, and the hue steps round |
| Hats | 7–14 kHz onsets | fine contour lines flick through the escape-time bands (and ripple the stalks inside the set) |
| Bass notes | 60–250 Hz onsets | swirl: twist the picture into spiral arms |
| Bass level | 20–250 Hz level | breathes the zoom |
| Vocals | 300–3400 Hz relative to the lows and highs | past a threshold, fold the fractal into up to 6 copies twisted into spirals |
| Pitch | dominant pitch class of 80–1000 Hz | shifts the palette's hue, moving the short way round the circle |
| Loudness | RMS | speed of c's path when there's no steady repeat to follow |
| Repeat length | autocorrelation of the onset envelope, in pulses | the pace of c's path: modes 1 and 4 step one sequence term per bar, modes 3, 6 and 7 make one loop per bar, mode 5 takes one Stern-Brocot branch per bar, and mode 2 picks the flower whose arm count is nearest |
| Section change | the mix shifting for 3 s against the 3 s before | next palette |

The repeat length is *measured*, not assumed from a time signature. The
pulse is the shortest steady beat (0.25–0.5 s); the repeat length is the
smallest number of pulses whose autocorrelation peak stands out, stabilized
over five readings. A "bar" here is the repeat stretched to a whole number
of repeats at least 3 s (mode 5), 5 s (mode 4), 6 s (mode 1), 7 s (mode 7)
or 8 s (modes 3 and 6) long, so the path isn't frantic on a short pattern.

Check the tracker against a track whose structure you know:

```fish
./venv/bin/python tools/check_rhythm.py "~/Music/some song.mp3" 15
```

It shells out to `ffmpeg` to decode, so `sudo apt install ffmpeg` first (the
live app doesn't need it — only this tool does).

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
- Mode 4, Metallic spirals: the golden angle is 360/phi^2 = 137.5deg, and
  phi is only the first of the metallic means — the positive roots of
  x^2 = p*x + 1. Each brings its own angle and its own integer sequences:
  bronze 33.0deg (1, 3, 10, 33, 109), silver 61.8deg (Pell: 1, 2, 5, 12,
  29), golden 137.5deg (Fibonacci and Lucas), and the plastic number
  205.1deg (Padovan and Perrin, the slowest growers here, so their radius
  climbs in the finest steps). The mode walks the seven sequences from the
  tightest angle to the widest and back, ~100 s end to end; at the tight
  angles the spiral's arms line up into visible rays, at the golden angle
  they refuse to, which is the whole point of phi. The angle is
  accumulated into a running phase rather than multiplied by the term
  index, so sliding from one family to the next doesn't whip c across the
  plane.
- Mode 5, Farey tour: mode 2 rides the Fibonacci ratios, but those are one
  path of many down the Stern-Brocot tree. Two Farey parents p0/q0 < p1/q1
  bracket an interval, their mediant (p0+p1)/(q0+q1) lands inside it, and
  keeping it as the new left or right parent descends a level. Every
  rational on the way is a bulb on the cardioid, so the arm count of each
  flower is the mediant of the last two: always-left gives 1/2, 1/3, 1/4
  (one petal at a time), strict alternation gives exactly mode 2's
  Fibonacci ratios, and anything else writes the continued fraction of
  whatever irrational the branches converge on. The music picks the
  branch: an odd number of kicks in the bar goes right, an even number
  goes left. Parity, not "any kick at all" — a rule that saturates one way
  just adds an arm per step, and it's the branches changing that makes the
  arm counts jump (2, 3, 5, 7, 9, 16, 27, 41, 69 in one tour here). Past
  89 arms they're finer than a pixel, so it restarts from the root, one
  level deeper to the left each time, gliding across the cardioid rather
  than cutting.
- Mode 6, Phoenix wings: z -> z^2 + c + p*z_prev. That one step of memory
  makes it a map of the *pair* (z, z_prev), not of z, so it isn't
  conjugate to anything of the form z^d + c and the Julia set loses its
  rotational symmetry — it comes apart into wings and feathers instead of
  arms. c rides a small circle around the classic 0.5667 and p swings
  between -1/phi and -1/phi^2 at 1/phi times the orbit rate. Kicks widen
  c's circle until the wings tear, and tilt p off the real axis, which
  shears them.
- Mode 7, Mandelbar flames: two maps that conjugate or fold z before
  squaring it — the tricorn, conj(z)^d + c, and the burning ship,
  (|Re z| + i*|Im z|)^d + c. Neither is complex differentiable — both
  reverse orientation — so the escape-time bands come out creased and
  flame-edged rather than smoothly nested, and the arms meet at corners
  instead of winding into smooth spirals. (The three-fold symmetry the
  tricorn is known for belongs to its parameter set and to the Julia set
  at c = 0; off zero it goes, like the Mandelbrot set's own.)
  It cuts between tricorn and ship at d = 2 and 3 on the first kick
  after 14 s (or at 28 s regardless). Each of the four has its own patch
  of parameter space — the tricorn's quadratic body is tight around the
  origin, the ship's sits well to the left, and raising the degree pushes
  both outward — so each gets its own center and radius band, picked by
  rendering the four over a grid of c off-screen and keeping the bands
  where the sets have both a filled core and structure around it. c orbits
  its band once per bar with the radius crossing it at phi times the orbit
  rate; kicks drive it past the band, where the flames break into embers.
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
count, the power, which map was being iterated, the arm count of the bulb
c was sitting in (modes 2 and 5) and the palette. The
`.summary.txt` next to it is printed when you quit. It's the place to check
whether the tracker heard what you heard.
