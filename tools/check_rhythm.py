"""
Run an audio file through the same MusicTracker the live app uses and print
what it hears, section by section. Use it to check the tracker against a track
whose structure you know.

    ./venv/bin/python tools/check_rhythm.py "~/Music/some song.mp3" [seconds-per-row]
"""

import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rhythm import FIBONACCI, MusicTracker  # noqa: E402

SR, BLOCK = 44100, 1024

if len(sys.argv) < 2:
    sys.exit(f"usage: {Path(sys.argv[0]).name} <audio file> [seconds-per-row]")

path = str(Path(sys.argv[1]).expanduser())
row_s = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
# ffmpeg does the decoding, so this reads whatever it reads.
try:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
except FileNotFoundError:
    sys.exit("needs ffmpeg on PATH to decode audio: sudo apt install ffmpeg")
except subprocess.CalledProcessError as e:
    sys.exit(f"ffmpeg could not read {path}:\n{e.stderr.decode(errors='replace').strip()}")
x = np.frombuffer(raw, dtype="<f4")
print(f"{Path(path).name}: {len(x) / SR:.0f} s")

t = MusicTracker(SR, BLOCK)
per_row = int(row_s * SR / BLOCK)
prev = (0, 0, 0, 0, 0)
cycles = Counter()
print(" time   pulse  cycle(conf)  cycle_s  kick snare hats bass /s   note  sections")
for i in range(len(x) // BLOCK):
    t.process(x[i * BLOCK:(i + 1) * BLOCK])
    if t.cycle_len:
        cycles[t.cycle_len] += 1
    if (i + 1) % per_row == 0:
        cur = (t.n_kick, t.n_snare, t.n_hat, t.n_bass, t.sections)
        d = [c - p for c, p in zip(cur, prev)]
        prev = cur
        m, s = divmod((i + 1) * BLOCK / SR, 60)
        cyc = f"{t.cycle_len:2d} ({t.cycle_conf:.1f})" if t.cycle_len else "   -    "
        print(f" {int(m)}:{s:04.1f}  {t.pulse_s:5.3f}  {cyc}   {t.cycle_seconds:6.2f}   "
              f"{d[0]/row_s:4.1f} {d[1]/row_s:4.1f} {d[2]/row_s:4.1f} {d[3]/row_s:4.1f}      "
              f"{t.pitch_name:2s}    {t.sections}")

total = sum(cycles.values())
print("\ncycle lengths (pulses) over the whole track:")
for k, c in cycles.most_common(8):
    print(f"  {k:2d}  {100 * c / total:4.1f}%" + ("   <- Fibonacci" if k in FIBONACCI else ""))
fib = sum(c for k, c in cycles.items() if k in FIBONACCI)
print(f"  Fibonacci share: {100 * fib / total:.0f}%  (numbers 3-40 that are Fibonacci: 3, 5, 8, 13, 21, 34 = 6 of 38 = 16%)")
