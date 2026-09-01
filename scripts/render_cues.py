"""
render_cues.py
Build-time tool. Renders the fixed cue phrases from the rules file to WAV.

Run once, commit the output. The demo machine then needs no working
text-to-speech engine, and playback at runtime is a file read rather than
speech synthesis -- which is what keeps the cue inside the 200ms budget from
TRD.md 5a.

    python scripts/render_cues.py

This is the ONLY place pyttsx3 is allowed to be imported. Its runAndWait()
blocks the calling thread; anywhere near the frame loop that would freeze the
camera mid-rise.
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rehab_ai.audio.player import DEFAULT_CUE_DIR  # noqa: E402
from rehab_ai.rules.loader import load_rules  # noqa: E402


def render(text: str, out_path: Path) -> None:
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 165)  # a touch slower than default; this is a cue
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()
    engine.stop()


def verify(path: Path) -> str:
    """Confirm the file is a readable WAV with actual audio in it.

    pyttsx3's save_to_file fails quietly on some drivers, leaving a zero-length
    or headerless file. Discovering that during a demo is not acceptable, so
    check now.
    """
    if not path.is_file():
        return "file was not created"
    if path.stat().st_size == 0:
        return "file is empty"
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
    except wave.Error as exc:
        return f"not a readable WAV: {exc}"
    if frames == 0:
        return "WAV contains no audio frames"
    return f"ok  {frames / rate:.2f}s"


def main() -> int:
    rules = load_rules()
    DEFAULT_CUE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"rendering {len(rules.cue.phrases)} cue(s) to {DEFAULT_CUE_DIR}\n")

    failures = 0
    for key, phrase in rules.cue.phrases.items():
        out = DEFAULT_CUE_DIR / phrase.clip
        print(f"  {key:24} {phrase.text!r}")
        try:
            render(phrase.text, out)
            # Some TTS backends return from runAndWait before the file is
            # flushed. A short settle avoids a spurious failure report.
            time.sleep(0.3)
        except Exception as exc:  # noqa: BLE001
            print(f"  {'':24} RENDER FAILED: {exc}")
            failures += 1
            continue

        status = verify(out)
        print(f"  {'':24} -> {out.name}  [{status}]")
        if not status.startswith("ok"):
            failures += 1

    print()
    if failures:
        print(f"{failures} cue(s) failed. The app will refuse to start until they exist.")
        print("If pyttsx3 will not cooperate, record the phrases by hand and save them")
        print(f"under the exact filenames above in {DEFAULT_CUE_DIR}.")
        return 1

    print("all cues rendered. Commit them so the demo machine needs no TTS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
