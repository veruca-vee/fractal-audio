---
name: tdd
description: Write code test-first with red-green-refactor. Use before adding or changing any behaviour in this project — a feature, a bug fix, an audio detector, a mapping from sound to visuals, a shader change. Covers the cycle, how to keep a test honest, and how to verify this project headlessly (EGL shader renders, stubbed window config, real mpv, synthesised audio). Not for pure refactors whose tests are already green, or for throwaway scratchpad scripts.
---

# TDD

One test at a time. No implementation before a failing test exists.

## The cycle

1. **Red.** Write one small test naming the behaviour you want. Run it.
   Confirm it fails *for the right reason* — an `ImportError` or `NameError`
   means the test is broken, not that you are red. Report the failure before
   writing any implementation.
2. **Green.** The smallest change that passes. Ugly is fine here; that is
   what step 3 is for.
3. **Refactor.** Clean up with the tests staying green. Re-run before moving
   on.

Then the next test. Do not write several up front — each test should be able
to change what the next one looks like, which is the entire point.

## Keeping tests honest

- **Test behaviour, not implementation.** `contour tracks a melodic arch,
  r > 0.8` — not `calls argmax on the band`. A test that knows how the code
  works has to be rewritten every time the code is, and stops being evidence.
- **Name tests as sentences.** `test_contour_holds_when_the_voice_stops`.
- **Never weaken a test to make red go away.** This is the failure mode the
  skill exists to prevent. A threshold nudged down to get green is a bug
  shipped with a passing test over it.
- **But a wrong test is allowed to be replaced** — out loud, with the reason,
  on its merits. "No singer glides 1.5 octaves monotonically over ten seconds
  while the register tracker is still learning their range; replacing it with
  a correlation against an arch, which is what a phrase does" is legitimate.
  "0.8 was a bit strict, using 0.7" is not. If you cannot say what is
  unrealistic about the scenario, the test is right and the code is wrong.
- **A test that has never failed has proved nothing.** If it passes the first
  time you run it, break the implementation deliberately and check it goes
  red.

## Running them

```fish
./venv/bin/python -m unittest discover -s tests -t . -v
```

Stdlib `unittest` — there is no pytest in the venv. `tests/__init__.py` must
exist or discovery fails with "Start directory is not importable".

## Verifying this project without running it

`main.py` forces fullscreen and opens an audio stream, so it cannot be used
to check anything unattended. Everything is reachable headlessly:

- **Shaders.** `moderngl.create_standalone_context(backend="egl",
  require=330)` works here (Mesa Intel UHD 600). Render to an FBO, read the
  pixels back, assert on statistics — brightness in a region, where the
  bright part sits, that the centre is untouched.
- **Window logic.** Build the config without `__init__`:
  `FractalWindow.__new__(FractalWindow)`, set attributes by hand, stub
  `show_toast` with a list's `append`. `_start_player` needs `ctx` and `vbo`,
  so hand it the standalone context.
- **The player.** Real mpv against generated wav files. Make them ~25 s —
  4 s tracks auto-advance in the middle of a test and produce failures that
  look exactly like bugs.
- **The tracker.** Synthesised audio with a known answer (a harmonic stack
  over a drum bed), never real music, so the expected result is arguable
  from first principles.

**Anything aesthetic cannot be asserted.** Render a PNG and look at it with
the Read tool. "Is the edge light too bright" is not a test, and three
rounds of pixel statistics will not answer it.

## The trap that has caused three bugs here

The auto-gained levels — `amp`, `bass`, `mid`, `treble`, `vocal` — measure
**deviation from a recent range**, not loudness. Consequences, all of which
have bitten:

- A steady signal decays toward 0. A voice that simply keeps singing switches
  off any gate built on `tr.vocal` within a few seconds.
- Two tracks are not comparable. A drums-only loop reads as "full" exactly as
  much as an orchestra.
- They recover during silence as the gain adapts, so a threshold crossed
  downward gets crossed again from below a few seconds later.

Anything needing an absolute reading — a gate, a presence test, a density —
must come from raw spectrum magnitudes. Anything needing "how much is this
standing out right now" is what the auto-gained value is actually for.
