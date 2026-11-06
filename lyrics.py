"""Synced lyrics for the player: LRC in, the current line out.

Three sources, in order: a sidecar .lrc sitting next to the track, the
local cache (~/.cache/fractal-lyrics), and LRCLIB (lrclib.net, free, no
key), whose syncedLyrics are LRC already. A fetch result is written to
the cache, so a track costs one network trip ever.

Sync comes from the player owning the clock: player.py knows the position
to a fraction of a second, so a line is just "the last timestamp at or
before now". In monitor-only mode there is no position worth the name and
this module is never asked.

Only *synced* lyrics are used. Plenty of tracks have plain text on
LRCLIB; with no timestamps there is nothing to sync to, so they read as
"no lyrics" rather than scrolling a guess.
"""

import bisect
import json
import re
import threading
import urllib.parse
import urllib.request
from pathlib import Path

CACHE_DIR = Path("~/.cache/fractal-lyrics").expanduser()
LRCLIB = "https://lrclib.net/api"
UA = {"User-Agent": "fractal-audio (lyrics for a personal player)"}

_TAG = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
_OFFSET = re.compile(r"\[offset:\s*(-?\d+(?:\.\d+)?)\]", re.I)


def parse_lrc(text):
    """LRC text -> [(seconds, line), ...] sorted by time.

    A line may carry several timestamps (a repeated chorus written once);
    each gets its own entry. Blank lines are kept: they are how an
    instrumental gap clears the words off the screen.
    """
    entries = []
    offset = 0.0
    for raw in text.splitlines():
        m = _OFFSET.search(raw)
        if m:
            offset = float(m.group(1)) / 1000.0
            continue
        stamps = _TAG.findall(raw)
        if not stamps:
            continue
        line = _TAG.sub("", raw).strip()
        for mm, ss in stamps:
            entries.append((int(mm) * 60 + float(ss) + offset, line))
    entries.sort(key=lambda e: e[0])
    return entries


class LyricTrack:
    """One track's timed lines, with the lookup the render loop needs."""

    def __init__(self, entries):
        self.times = [t for t, _ in entries]
        self.lines = [s for _, s in entries]

    def __len__(self):
        return len(self.lines)

    def line_at(self, pos):
        """The line sung at `pos` seconds, or "" before the first/blank."""
        i = bisect.bisect_right(self.times, pos) - 1
        return self.lines[i] if i >= 0 else ""


def _slug(artist, title):
    s = re.sub(r"[^A-Za-z0-9]+", "-", f"{artist} - {title}").strip("-").lower()
    return s or "track"


def _fetch_lrclib(artist, title, duration):
    """Synced LRC text from LRCLIB, or None. Blocking: call from a thread."""
    params = {"artist_name": artist, "track_name": title}
    if duration and duration > 0:
        params["duration"] = str(int(round(duration)))
    url = f"{LRCLIB}/get?" + urllib.parse.urlencode(params)
    for attempt in range(2):   # one retry: a dropped fetch shouldn't cost
        try:                   # the track its lyrics for the whole session
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.load(r)
            synced = data.get("syncedLyrics")
            return synced if synced else None
        except Exception:
            if attempt:
                return None
    return None


def load_lyrics(path, artist, title, duration):
    """The LyricTrack for a file, or None. Blocking: call from a thread.

    `path` is the track file (for the sidecar); artist/title are its best
    identity. A LRCLIB hit lands in the cache under that identity.
    """
    if path is not None:
        sidecar = Path(path).with_suffix(".lrc")
        try:
            if sidecar.is_file():
                track = LyricTrack(parse_lrc(
                    sidecar.read_text(encoding="utf-8", errors="replace")))
                if len(track):
                    return track
        except OSError:
            pass
    if not (artist and title):
        return None
    cache = CACHE_DIR / (_slug(artist, title) + ".lrc")
    try:
        if cache.is_file():
            track = LyricTrack(parse_lrc(
                cache.read_text(encoding="utf-8", errors="replace")))
            if len(track):
                return track
    except OSError:
        pass
    text = _fetch_lrclib(artist, title, duration)
    if not text:
        return None
    track = LyricTrack(parse_lrc(text))
    if not len(track):
        return None
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    except OSError:
        pass
    return track


_DECORATION = re.compile(
    r"\s*[\(\[][^\(\)\[\]]*\b(official|lyric|lyrics|video|audio|visualizer|"
    r"hd|hq|4k|mv)\b[^\(\)\[\]]*[\)\]]\s*$", re.I)


def _clean_title(artist, title):
    """Undo YouTube-rip naming, which is most of this library: the title
    tag carrying "Artist - Title (Official Audio)" whole, decorations and
    all. LRCLIB matches on the song's actual name, so the artist repeated
    at the front comes off and one trailing decoration with it. Kept
    conservative -- a real parenthetical in the middle of a title stays.
    """
    title = str(title).strip()
    if artist and title.lower().startswith(str(artist).lower() + " - "):
        title = title[len(str(artist)) + 3:].strip()
    return _DECORATION.sub("", title).strip() or title


def identity(player):
    """(artist, title) for the player's current track.

    Tags first, then the "Artist - Title" convention the library mostly
    follows in both its tags and its filenames (player.title() already
    prefers the tags and falls back to the stem), split the same way.
    Either way the title goes through _clean_title before LRCLIB sees it.
    """
    meta = player.state.get("metadata") or {}
    lower = {str(k).lower(): v for k, v in meta.items()} \
        if isinstance(meta, dict) else {}
    artist, title = lower.get("artist"), lower.get("title")
    if artist and title:
        return str(artist), _clean_title(artist, title)
    name = player.title()
    if " - " in name:
        a, t = name.split(" - ", 1)
        return a.strip(), _clean_title(a, t)
    return "", _clean_title("", name)


class LyricFetcher:
    """Background loads for the render loop, which must never block.

    A load is keyed by the track it was started for, so a result that
    arrives after the listener skipped ahead is dropped by the caller,
    not displayed against the wrong song.
    """

    def __init__(self):
        self._result = None      # (key, LyricTrack | None)
        self._busy_key = None

    def want(self, key, path, artist, title, duration):
        if self._busy_key == key:
            return
        self._busy_key = key

        def run():
            try:
                track = load_lyrics(path, artist, title, duration)
            except Exception:
                track = None
            self._result = (key, track)
            if self._busy_key == key:
                self._busy_key = None

        threading.Thread(target=run, daemon=True).start()

    def poll(self):
        """The finished (key, track) once, then None until the next."""
        result, self._result = self._result, None
        return result
