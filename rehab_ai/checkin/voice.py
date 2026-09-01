"""
voice.py
Spoken check-in. An input method, not a separate data model.

A spoken check-in produces exactly the fields a tapped one does -- the same
PainReport and the same SwellingReport, differing only in `source`. There is no
"voice pain score" that behaves differently downstream.

    speech ──▶ transcript ──▶ parse ──▶ PainReport + SwellingReport
                              (pure)     (identical to the tap path)

WHY THE PARSER REFUSES TO GUESS
===============================
A misheard pain score is a safety-relevant error: it feeds policy/ and can
unlock loaded exercise. So parsing is strict and conservative. If the
transcript does not clearly contain a value, the field comes back None and the
UI asks -- it never picks the most likely number.

"Absent" is a state this system already models properly. Reaching for a best
guess here would smuggle a fabricated observation into the one module that is
supposed to be the patient's own words.

MODEL DOWNLOAD
==============
faster-whisper fetches its weights from the network on first use, then runs
entirely offline. That first fetch is the one moment this application touches
the internet. Do it before demo day; ensure_model() exists to make that
explicit rather than incidental.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rehab_ai.models.session import InputSource, PainReport, SwellingReport

# Spoken numbers a patient might use for a 0-10 scale.
_NUMBER_WORDS = {
    "zero": 0, "nought": 0, "none": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Ordered longest-first so "about the same" is tested before "same".
_SWELLING_PHRASES: tuple[tuple[str, SwellingReport], ...] = (
    ("about the same", SwellingReport.SAME),
    ("more swollen", SwellingReport.PUFFIER),
    ("less swollen", SwellingReport.LESS),
    ("less puffy", SwellingReport.LESS),
    ("more puffy", SwellingReport.PUFFIER),
    ("gone down", SwellingReport.LESS),
    ("went down", SwellingReport.LESS),
    ("puffier", SwellingReport.PUFFIER),
    ("swollen", SwellingReport.PUFFIER),
    ("bigger", SwellingReport.PUFFIER),
    ("the same", SwellingReport.SAME),
    ("unchanged", SwellingReport.SAME),
    ("smaller", SwellingReport.LESS),
    ("better", SwellingReport.LESS),
    ("worse", SwellingReport.PUFFIER),
    ("same", SwellingReport.SAME),
    ("less", SwellingReport.LESS),
)


@dataclass(frozen=True)
class VoiceCheckin:
    """What was heard, and what could be read from it with confidence.

    `transcript` is always present so the UI can show the patient what the app
    thought it heard. A field being None means "ask", never "assume".
    """

    transcript: str
    pain: PainReport | None
    swelling: SwellingReport | None

    @property
    def is_complete(self) -> bool:
        return self.pain is not None and self.swelling is not None

    @property
    def needs(self) -> list[str]:
        """Fields the UI still has to collect by tap."""
        missing = []
        if self.pain is None:
            missing.append("pain")
        if self.swelling is None:
            missing.append("swelling")
        return missing


# Ways of naming the scale itself. These carry no pain value, but they contain
# numbers, so they are removed before extraction. Without this, the commonest
# phrasing of all -- "three out of ten" -- reads as the two values 3 and 10 and
# is rejected as ambiguous.
_SCALE_MARKERS = (
    r"\bout of (?:ten|10)\b",
    r"/\s*10\b",
    r"\bon a scale (?:of|from)?\s*(?:zero|nought|0|one|1)?\s*(?:to|-)?\s*(?:ten|10)\b",
    r"\b(?:zero|0|one|1)\s*(?:to|-)\s*(?:ten|10)\b",
)


def _strip_scale_markers(text: str) -> str:
    for pattern in _SCALE_MARKERS:
        text = re.sub(pattern, " ", text)
    return text


def parse_pain(transcript: str) -> PainReport | None:
    """Extract a 0-10 pain score, or None.

    Accepts digits and number words. Refuses anything ambiguous: two different
    numbers in one sentence returns None rather than picking the first, because
    "it was four yesterday, today it's seven" and "seven, it was four
    yesterday" mean the same thing and parse to opposite answers.

    Scale markers are stripped first, so "three out of ten" is a score of three
    rather than a contradiction. That is recognising an idiom, not relaxing the
    ambiguity rule -- "out of ten" names the scale, it does not report a value.
    """
    text = _strip_scale_markers(transcript.lower())

    found: list[int] = []

    for match in re.finditer(r"\b(\d{1,2})\b", text):
        value = int(match.group(1))
        if 0 <= value <= 10:
            found.append(value)

    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            found.append(value)

    unique = set(found)
    if len(unique) != 1:
        return None  # nothing found, or genuinely ambiguous

    return PainReport(value=unique.pop(), source=InputSource.VOICE)


def parse_swelling(transcript: str) -> SwellingReport | None:
    """Extract a swelling comparison, or None.

    Longest phrase wins, so "about the same" is not read as "same" plus noise.
    Two contradictory phrases return None.
    """
    text = transcript.lower()

    matched: list[SwellingReport] = []
    consumed = text
    for phrase, report in _SWELLING_PHRASES:
        if phrase in consumed:
            matched.append(report)
            consumed = consumed.replace(phrase, " ")

    unique = set(matched)
    if len(unique) != 1:
        return None

    return unique.pop()


def parse(transcript: str) -> VoiceCheckin:
    """Turn a transcript into the same fields the tap path produces."""
    return VoiceCheckin(
        transcript=transcript.strip(),
        pain=parse_pain(transcript),
        swelling=parse_swelling(transcript),
    )


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


class Transcriber:
    """Wraps faster-whisper. Loads the model once, lazily.

    Kept deliberately thin: everything interesting is in parse(), which is a
    pure function and can be tested exhaustively without audio.
    """

    def __init__(self, model_size: str = "tiny", device: str = "cpu") -> None:
        self._model_size = model_size
        self._device = device
        self._model = None

    def ensure_model(self) -> None:
        """Load (downloading on first use) so the fetch happens at a moment of
        our choosing rather than mid-check-in in front of an audience."""
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self._model_size, device=self._device, compute_type="int8"
        )

    def transcribe(self, audio_path: str) -> str:
        self.ensure_model()
        assert self._model is not None
        segments, _ = self._model.transcribe(audio_path, language="en", beam_size=1)
        return " ".join(segment.text for segment in segments).strip()

    def check_in(self, audio_path: str) -> VoiceCheckin:
        return parse(self.transcribe(audio_path))
