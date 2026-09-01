"""
Tests for the spoken check-in parser (CP 8).

Gate: "speak a check-in; get the same structured fields a tap gives."

All of these run without audio and without a model, because the interesting
part -- turning words into a pain score and a swelling comparison -- is a pure
function. The Whisper wrapper around it is deliberately thin.

The theme throughout: a misheard pain score feeds policy/ and can unlock loaded
exercise, so the parser refuses to guess. Ambiguity returns None and the UI
asks, rather than the app picking the most likely number.
"""

import pytest

from rehab_ai.checkin.voice import parse, parse_pain, parse_swelling
from rehab_ai.models.session import InputSource, PainReport, SwellingReport


# --------------------------------------------------------------------------
# Pain
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transcript,expected",
    [
        ("about a four today", 4),
        ("4", 4),
        ("my pain is seven", 7),
        ("it's a ten", 10),
        ("zero pain today", 0),
        ("no pain, none at all", 0),
        ("I'd say three out of ten", 3),
    ],
    ids=["word-four", "digit", "seven", "ten", "zero", "none", "three-of-ten"],
)
def test_a_clear_pain_score_is_read(transcript, expected):
    report = parse_pain(transcript)
    assert report is not None
    assert report.value == expected


def test_a_spoken_score_is_marked_as_voice():
    """The only thing that differs from the tap path. Everything downstream
    treats the two identically."""
    assert parse_pain("about a four").source is InputSource.VOICE


def test_a_spoken_score_is_the_same_type_as_a_tapped_one():
    spoken = parse_pain("about a four")
    tapped = PainReport(4, InputSource.TAP)
    assert spoken.value == tapped.value
    assert type(spoken) is type(tapped)


def test_no_number_means_no_reading():
    assert parse_pain("it hurts quite a lot today") is None


def test_two_different_numbers_are_refused():
    """'it was four yesterday, today it's seven' and 'seven, it was four
    yesterday' mean the same thing and would parse to opposite answers. Better
    to ask than to pick."""
    assert parse_pain("it was four yesterday, today it's seven") is None
    assert parse_pain("seven, it was four yesterday") is None


def test_the_same_number_twice_is_still_a_reading():
    """Repetition is not ambiguity."""
    report = parse_pain("four, yeah four")
    assert report is not None and report.value == 4


def test_a_number_outside_the_scale_is_ignored():
    """'2026' is a year, not a pain score. Out-of-range digits are discarded
    rather than clamped -- clamping would invent a value."""
    assert parse_pain("it's 2026 and my knee hurts") is None


def test_out_of_range_alongside_a_valid_score_still_reads():
    report = parse_pain("since 2026 it's been about a five")
    assert report is not None and report.value == 5


# --------------------------------------------------------------------------
# Swelling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transcript,expected",
    [
        ("it's puffier than yesterday", SwellingReport.PUFFIER),
        ("a bit more swollen", SwellingReport.PUFFIER),
        ("looks bigger today", SwellingReport.PUFFIER),
        ("about the same", SwellingReport.SAME),
        ("the same as yesterday", SwellingReport.SAME),
        ("unchanged", SwellingReport.SAME),
        ("less puffy", SwellingReport.LESS),
        ("the swelling has gone down", SwellingReport.LESS),
        ("it's smaller today", SwellingReport.LESS),
    ],
)
def test_a_clear_swelling_report_is_read(transcript, expected):
    assert parse_swelling(transcript) is expected


def test_longest_phrase_wins():
    """'about the same' must not be read as 'same' plus noise, and must not
    also match 'less' from a later clause."""
    assert parse_swelling("about the same as yesterday") is SwellingReport.SAME


def test_contradictory_reports_are_refused():
    assert parse_swelling("it's puffier, well, maybe less puffy") is None


def test_no_swelling_words_means_no_reading():
    assert parse_swelling("my pain is about a four") is None


# --------------------------------------------------------------------------
# The whole check-in
# --------------------------------------------------------------------------


def test_a_complete_spoken_checkin_yields_both_fields():
    result = parse("pain's about a four today, and it's a bit more swollen")

    assert result.pain is not None and result.pain.value == 4
    assert result.swelling is SwellingReport.PUFFIER
    assert result.is_complete
    assert result.needs == []


def test_a_partial_checkin_names_what_is_still_needed():
    """The UI collects the rest by tap. It does not fill the gap itself."""
    result = parse("it's a bit more swollen today")

    assert result.swelling is SwellingReport.PUFFIER
    assert result.pain is None
    assert not result.is_complete
    assert result.needs == ["pain"]


def test_an_unusable_transcript_asks_for_both():
    result = parse("um, I'm not really sure")
    assert result.needs == ["pain", "swelling"]


def test_the_transcript_is_always_kept():
    """So the UI can show the patient what the app thought it heard, rather
    than silently discarding a misheard sentence."""
    result = parse("  pain is about a four  ")
    assert result.transcript == "pain is about a four"


def test_an_empty_transcript_produces_nothing_rather_than_failing():
    result = parse("")
    assert result.transcript == ""
    assert result.pain is None
    assert result.swelling is None


# --------------------------------------------------------------------------
# Scale markers name the scale; they do not report a value
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transcript,expected",
    [
        ("I'd say three out of ten", 3),
        ("about a 6 out of 10", 6),
        ("on a scale of one to ten it's a four", 4),
        ("7/10 today", 7),
        ("zero out of ten, no pain at all", 0),
    ],
)
def test_scale_phrasing_is_not_read_as_a_second_score(transcript, expected):
    """'three out of ten' contains two numbers and one score. Without stripping
    the scale marker, the commonest phrasing of all would be rejected as
    ambiguous."""
    report = parse_pain(transcript)
    assert report is not None, f"failed to read {transcript!r}"
    assert report.value == expected


def test_stripping_the_scale_marker_does_not_hide_real_ambiguity():
    """The rule is still strict. Two genuine values remain a refusal."""
    assert parse_pain("it was four out of ten yesterday, seven today") is None
