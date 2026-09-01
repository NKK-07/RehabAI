"""
player.py
Plays the live correction cue without stalling the camera.

WHY NOT TEXT-TO-SPEECH AT RUNTIME
=================================
pyttsx3's runAndWait() blocks the calling thread until speech finishes. Called
from the frame loop, a one-second cue freezes capture for a second -- during
the rise, which is the precise moment the meter must keep updating and the
detector must keep watching. The most important second of the demo would be
the one where the video stops.

So the phrases are rendered to WAV ahead of time (see scripts/render_cues.py)
and this module only plays files. Playback starts in single-digit milliseconds,
leaving essentially the whole 200ms budget from TRD.md 5a to detection.

pyttsx3 remains a dependency, but as a build-time tool. It must never be
imported here.

    build time                          run time
    ──────────                          ────────
    render_cues.py                      CuePlayer.play()
      pyttsx3 -> WAV        ─────▶        sounddevice, non-blocking
      (slow, blocking,                    (fast, returns immediately)
       runs once)
"""

from __future__ import annotations

import threading
from pathlib import Path

from rehab_ai.rules.loader import CueRules

DEFAULT_CUE_DIR = Path(__file__).resolve().parents[2] / "assets" / "cues"


class CueAudioError(Exception):
    """Raised at startup when a clip named in the rules file is missing.

    Loud on purpose. A cue that silently fails to play looks exactly like a
    cue that correctly decided not to fire, and you would not discover the
    difference until the demo.
    """


class CuePlayer:
    """Plays pre-rendered cue clips. Never blocks the caller.

    Verify eagerly at construction rather than lazily at first play: the first
    play happens mid-rep in front of an audience.
    """

    def __init__(
        self,
        rules: CueRules,
        cue_dir: Path | str | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self._rules = rules
        self._dir = Path(cue_dir) if cue_dir is not None else DEFAULT_CUE_DIR
        self._enabled = enabled
        self._clips: dict[str, tuple] = {}
        self._lock = threading.Lock()
        self._last_error: str | None = None

        if enabled:
            self._preload()

    # -- startup ------------------------------------------------------------

    def _preload(self) -> None:
        """Load every clip into memory once, at startup.

        Reading a WAV off disk mid-rep would add file I/O to the latency
        budget for no reason -- the whole clip set is a few hundred kilobytes.
        """
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise CueAudioError(
                "soundfile is required to play cue clips. Install requirements.txt."
            ) from exc

        missing: list[str] = []
        for key, phrase in self._rules.phrases.items():
            path = self._dir / phrase.clip
            if not path.is_file():
                missing.append(f"{key} -> {path.name}")
                continue
            data, samplerate = sf.read(str(path), dtype="float32")
            self._clips[key] = (data, samplerate)

        if missing:
            raise CueAudioError(
                "cue clips are missing from "
                f"{self._dir}:\n  " + "\n  ".join(missing) + "\n\n"
                "Generate them with:  python scripts/render_cues.py"
            )

    # -- runtime ------------------------------------------------------------

    @property
    def available_cues(self) -> list[str]:
        return sorted(self._clips)

    @property
    def last_error(self) -> str | None:
        """Set when a background playback failed. Surfaced by the UI so a
        silent cue is visible rather than mistaken for a clean rep."""
        return self._last_error

    def play(self, key: str) -> bool:
        """Start playing a cue and return immediately.

        Returns True if playback was started. Never raises: a failure to make
        a sound must not take down a session that is otherwise working, but it
        is recorded in `last_error` rather than swallowed.
        """
        if not self._enabled:
            return False

        clip = self._clips.get(key)
        if clip is None:
            self._last_error = f"no clip loaded for cue {key!r}"
            return False

        thread = threading.Thread(target=self._play_blocking, args=clip, daemon=True)
        thread.start()
        return True

    def _play_blocking(self, data, samplerate: int) -> None:
        """Runs on a throwaway thread. Nothing in the frame loop waits on it."""
        try:
            import sounddevice as sd

            with self._lock:  # one cue at a time; a second would overlap the first
                sd.play(data, samplerate)
                sd.wait()
        except Exception as exc:  # noqa: BLE001 - audio devices fail in many ways
            self._last_error = f"cue playback failed: {exc}"


class NullCuePlayer(CuePlayer):
    """A player that loads nothing and makes no sound.

    For tests and for headless runs. Deliberately a real class rather than
    `player = None` scattered behind if-statements at every call site.
    """

    def __init__(self) -> None:  # noqa: D107 - see class docstring
        self._rules = None  # type: ignore[assignment]
        self._dir = DEFAULT_CUE_DIR
        self._enabled = False
        self._clips = {}
        self._lock = threading.Lock()
        self._last_error = None
        self.played: list[str] = []

    def play(self, key: str) -> bool:
        self.played.append(key)
        return False
