"""
Tests for cue playback (CP 6, second half).

The checkpoint gate has two parts. The latch logic is covered in
test_detector.py; this file covers the other half:

    "play a cue while the camera loop is running and watch the frame rate.
     If it dips, the audio is on the wrong thread and the demo will freeze at
     its most important moment."

Which is the failure this whole module exists to prevent: pyttsx3's
runAndWait() blocks the calling thread, so a naive implementation would stall
capture for the length of the phrase -- roughly two seconds, during the rise.
"""

import time
from pathlib import Path

import pytest

from rehab_ai.audio.player import CueAudioError, CuePlayer, NullCuePlayer
from rehab_ai.rules.loader import load_rules


@pytest.fixture(scope="module")
def rules():
    return load_rules()


@pytest.fixture(scope="module")
def player(rules):
    return CuePlayer(rules.cue)


# --------------------------------------------------------------------------
# The rules file and the asset directory must agree
# --------------------------------------------------------------------------


def test_every_phrase_in_the_rules_file_has_a_clip_on_disk(rules):
    """The contract between rules/thresholds.v1.json and assets/cues/.
    Adding a phrase without rendering it would produce a cue that decides to
    fire and then makes no sound."""
    cue_dir = Path(__file__).resolve().parents[1] / "assets" / "cues"
    for key, phrase in rules.cue.phrases.items():
        assert (cue_dir / phrase.clip).is_file(), f"missing clip for {key}: {phrase.clip}"


def test_clips_are_preloaded_at_construction(player, rules):
    """Loaded once at startup, not on first use. First use happens mid-rep in
    front of an audience."""
    assert set(player.available_cues) == set(rules.cue.phrases)


def test_a_missing_clip_fails_loudly_at_startup(rules, tmp_path):
    """Not at first play. A cue that silently fails to sound is
    indistinguishable from a cue that correctly decided not to fire."""
    with pytest.raises(CueAudioError, match="missing"):
        CuePlayer(rules.cue, cue_dir=tmp_path)


def test_the_missing_clip_error_says_how_to_fix_it(rules, tmp_path):
    with pytest.raises(CueAudioError, match="render_cues.py"):
        CuePlayer(rules.cue, cue_dir=tmp_path)


# --------------------------------------------------------------------------
# THE GATE: playback must not block the caller
# --------------------------------------------------------------------------


def test_play_returns_immediately(player, rules):
    """The clips are roughly two seconds long. If play() takes anything like
    that, it is synchronous and the camera is frozen for the duration."""
    clip_seconds = 1.8  # shortest rendered phrase, conservatively

    started = time.perf_counter()
    player.play("hip_dominant")
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05, f"play() blocked for {elapsed:.3f}s"
    assert elapsed < clip_seconds / 10


def test_the_frame_loop_keeps_its_cadence_while_a_cue_plays(player):
    """Simulates the real failure: a cue fires mid-rise and the capture loop
    has to keep running at frame rate.

    Twenty frames at 25fps is 0.8s of wall clock. If playback were synchronous
    this would take 0.8s plus the length of the phrase.
    """
    frame_budget = 1.0 / 25

    player.play("hip_dominant")

    started = time.perf_counter()
    for _ in range(20):
        frame_start = time.perf_counter()
        # stand in for capture + pose + detector work
        sum(i * i for i in range(500))
        remaining = frame_budget - (time.perf_counter() - frame_start)
        if remaining > 0:
            time.sleep(remaining)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.4, f"frame loop took {elapsed:.2f}s -- audio stalled capture"


def test_two_cues_in_a_row_do_not_block_each_other(player):
    started = time.perf_counter()
    player.play("hip_dominant")
    player.play("descent_uncontrolled")
    elapsed = time.perf_counter() - started
    assert elapsed < 0.05


# --------------------------------------------------------------------------
# Failure is recorded, never swallowed
# --------------------------------------------------------------------------


def test_unknown_cue_key_is_reported_not_raised(player):
    """A failure to make a sound must not take down a session that is
    otherwise working -- but it must be visible."""
    assert player.play("no_such_cue") is False
    assert "no_such_cue" in (player.last_error or "")


# --------------------------------------------------------------------------
# The null player
# --------------------------------------------------------------------------


def test_null_player_records_without_sounding():
    """A real class rather than `player = None` behind if-statements at every
    call site."""
    null = NullCuePlayer()
    assert null.play("hip_dominant") is False
    assert null.played == ["hip_dominant"]


def test_null_player_needs_no_clips_on_disk():
    null = NullCuePlayer()
    assert null.available_cues == []
