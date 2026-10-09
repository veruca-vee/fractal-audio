"""mpv as the player engine, driven over its JSON IPC socket.

mpv runs as a headless subprocess rather than audio being decoded in here,
because the visuals already listen to the PipeWire monitor of the default sink
(pw_monitor.py): whatever mpv plays reaches rhythm.py through the capture path
that is already tuned, unchanged. What mpv adds is the half of the picture the
monitor can't see -- which file is playing, how long it is, where we are in it
-- which is what a player UI is made of, and what makes a log row say *where in
which track* something was heard.

The protocol is one JSON object per line over a unix socket:

    {"command": ["loadfile", path]}              -> {"request_id": 1, "error": "success"}
    {"command": ["observe_property", 1, "pause"]} -> {"event": "property-change", ...}

Replies and events are interleaved on that one socket, so a reader thread sorts
them: property changes land in `self.state`, command replies are matched to
their caller by request id. Three sharp edges, all of which bite quietly:

  - The socket is created by mpv asynchronously, some way into its startup, so
    connecting has to retry -- while also watching for the process having died,
    or a bad argument turns into a hang instead of an error.
  - Commands from the render thread must never wait for a reply. A round trip
    is sub-millisecond until mpv is busy opening a file over a slow path, and
    then it is a visible stall in the fractal. So `send()` is fire-and-forget
    and the UI reads only the cached `state`, which the reader thread keeps
    current.
  - A seek is only meaningful once the file is open, so a seek asked for at
    load time is held until mpv reports `file-loaded`.
"""

import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

AUDIO_EXTS = {
    ".mp3", ".flac", ".m4a", ".opus", ".ogg", ".oga", ".wav",
    ".aac", ".wma", ".aiff", ".alac", ".mka",
}

# Asked for once at startup and then watched; everything the UI draws comes
# from these rather than from a query per frame.
OBSERVED = [
    "pause", "time-pos", "duration", "media-title", "path",
    "playlist-pos", "playlist-count", "volume", "idle-active", "metadata",
]


def scan_library(roots):
    """Every audio file under `roots`, in a stable order, deduplicated.

    Dot-directories are skipped: ~/Music/.thumbnails is full of files that
    aren't music but do have audio extensions.
    """
    found, seen = [], set()
    for root in roots:
        root = Path(root).expanduser()
        if root.is_file():
            candidates = [root]
        else:
            candidates = sorted(
                p for p in root.rglob("*")
                if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
            )
        for p in candidates:
            if p.suffix.lower() not in AUDIO_EXTS:
                continue
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                found.append(rp)
    return found


class MpvPlayer:
    """A headless mpv holding a playlist, controlled over its IPC socket.

    Everything the caller reads is a cached property in `state`; everything it
    writes is a command queued at the socket. Nothing here blocks the caller.
    """

    def __init__(self, tracks, volume=85, start_paused=True):
        self.tracks = list(tracks)
        self.state = {}
        self.track_serial = 0        # bumped on every file-loaded, so the UI
        self.failed = None           # and the log can notice a track change
        self._lock = threading.Lock()
        self._req = 0
        self._pending_seek = None
        self._sock = None
        self._proc = None
        self._buf = b""

        if not shutil.which("mpv"):
            self.failed = "mpv is not on PATH"
            return
        try:
            self._start(volume, start_paused)
        except OSError as exc:
            self.failed = f"could not start mpv: {exc}"

    # ---- lifecycle ---------------------------------------------------------

    def _start(self, volume, start_paused):
        run = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
        self._sock_path = os.path.join(run, f"fractal-mpv-{os.getpid()}.sock")
        self._m3u = None
        cmd = [
            "mpv",
            "--idle=yes",
            "--no-video",
            "--force-window=no",
            # Without --no-terminal mpv takes over the tty it was launched from
            # and the shell is left in a strange state after quitting.
            "--no-terminal",
            "--gapless-audio=yes",
            f"--volume={volume}",
            f"--input-ipc-server={self._sock_path}",
        ]
        if start_paused:
            cmd.append("--pause")
        self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        deadline = time.monotonic() + 5.0
        while True:
            if self._proc.poll() is not None:
                self.failed = f"mpv exited at startup (code {self._proc.returncode})"
                return
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self._sock_path)
                self._sock = s
                break
            except OSError:
                s.close()
                if time.monotonic() > deadline:
                    self.failed = "mpv never created its IPC socket"
                    self._proc.terminate()
                    return
                time.sleep(0.05)

        threading.Thread(target=self._read_loop, daemon=True).start()
        for i, name in enumerate(OBSERVED, start=1):
            self.send("observe_property", i, name)
        if self.tracks:
            self._load_playlist()

    def _load_playlist(self):
        """Hand mpv the whole list at once, as an m3u.

        One `loadlist` beats 457 `loadfile` appends, and it leaves mpv owning
        the playlist, so auto-advance at the end of a track, gapless decoding
        and playlist-next all come for free.
        """
        fd, self._m3u = tempfile.mkstemp(prefix="fractal-playlist-", suffix=".m3u")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(str(p) for p in self.tracks) + "\n")
        self.send("loadlist", self._m3u, "replace")

    def close(self):
        if self._sock is not None:
            try:
                self.send("quit")
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        for path in (getattr(self, "_m3u", None), getattr(self, "_sock_path", None)):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def wait_ready(self, timeout=2.0):
        """Block until mpv has built the playlist. Startup only — never call
        this from the render thread.

        `loadlist` is asynchronous like everything else here, so a
        `playlist-play-index` sent straight after it can arrive while the
        playlist is still empty, where it is silently dropped and mpv just
        plays the first entry instead. Resuming into the middle of a library
        is exactly the case that hits it.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.state.get("playlist-count"):
                return True
            if not self.alive:
                return False
            time.sleep(0.02)
        return False

    @property
    def alive(self):
        return self._sock is not None and self._proc is not None and self._proc.poll() is None

    # ---- the socket --------------------------------------------------------

    def send(self, *command):
        """Queue a command. Fire and forget: the reply is never waited for."""
        if self._sock is None:
            return
        with self._lock:
            self._req += 1
            line = json.dumps({"command": list(command), "request_id": self._req}) + "\n"
            try:
                self._sock.sendall(line.encode("utf-8"))
            except OSError:
                self._sock = None  # mpv went away; the UI reads `alive`

    def _read_loop(self):
        while True:
            try:
                chunk = self._sock.recv(65536)
            except (OSError, AttributeError):
                return
            if not chunk:
                return
            self._buf += chunk
            while b"\n" in self._buf:
                raw, self._buf = self._buf.split(b"\n", 1)
                if not raw.strip():
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                self._handle(msg)

    def _handle(self, msg):
        event = msg.get("event")
        if event == "property-change":
            self.state[msg.get("name")] = msg.get("data")
        elif event == "file-loaded":
            self.track_serial += 1
            if self._pending_seek is not None:
                pos, self._pending_seek = self._pending_seek, None
                self.send("seek", pos, "absolute")

    # ---- what the UI reads -------------------------------------------------

    @property
    def paused(self):
        return bool(self.state.get("pause", True))

    @property
    def position(self):
        return self.state.get("time-pos") or 0.0

    @property
    def duration(self):
        return self.state.get("duration") or 0.0

    @property
    def index(self):
        pos = self.state.get("playlist-pos")
        return pos if isinstance(pos, int) and pos >= 0 else None

    @property
    def current_path(self):
        p = self.state.get("path")
        return Path(p) if p else None

    def title(self):
        """What to call the current track.

        Tags first when they carry both fields, since "Tool - Lateralus" from
        an artist/title pair beats a filename that may be a YouTube id; the
        filename stem is the fallback, and these files are mostly named
        "Artist - Title" anyway.
        """
        meta = self.state.get("metadata") or {}
        lower = {str(k).lower(): v for k, v in meta.items()} if isinstance(meta, dict) else {}
        artist, title = lower.get("artist"), lower.get("title")
        if artist and title:
            return f"{artist} - {title}"
        path = self.current_path
        media = self.state.get("media-title")
        # For an untagged file mpv sets media-title to the basename, extension
        # and all, which is exactly what the stem is here to avoid; it's worth
        # having only when it says something the filename doesn't.
        if media and not (path and str(media) == path.name):
            return str(media)
        return path.stem if path else "nothing playing"

    # ---- transport ---------------------------------------------------------

    def toggle_pause(self):
        # `cycle` rather than reading the cached value and negating it, so two
        # quick presses can't both see the same stale state.
        self.send("cycle", "pause")

    def play_index(self, i, seek=None):
        if not 0 <= i < len(self.tracks):
            return
        self._pending_seek = seek
        self.send("playlist-play-index", i)
        self.send("set_property", "pause", False)

    def next_track(self):
        self.send("playlist-next", "force")

    def prev_track(self):
        self.send("playlist-prev", "force")

    def seek(self, delta):
        self.send("seek", delta, "relative")

    def add_volume(self, delta):
        vol = self.state.get("volume")
        vol = 85.0 if vol is None else float(vol)
        self.send("set_property", "volume", max(0.0, min(130.0, vol + delta)))


# ---- what to play, and where we left off ----------------------------------


def build_playlist(roots, shuffle=False, seed=None):
    tracks = scan_library(roots)
    if shuffle:
        random.Random(seed).shuffle(tracks)
    return tracks


class PlayerState:
    """The last track and position, remembered between runs.

    Small enough to be worth no more than a JSON file; a missing or corrupt one
    simply means "no idea where we were", never a crash on startup.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.data = {}
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            pass

    @property
    def last(self):
        """(path, position) of the track playing when we last quit, or None."""
        p, pos = self.data.get("path"), self.data.get("position")
        if not p:
            return None
        return Path(p), float(pos or 0.0)

    def save(self, path, position, volume=None):
        self.data = {"path": str(path) if path else None, "position": round(float(position or 0), 1)}
        if volume is not None:
            self.data["volume"] = round(float(volume), 1)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1) + "\n")
        except OSError as exc:
            print(f"could not save player state: {exc}", file=sys.stderr)
