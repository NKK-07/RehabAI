"""
Tests for the rules loader (CP 2).

The gate for this checkpoint is explicitly "including the malformed-file case".
A loader that silently accepts junk and falls back to defaults is worse than no
loader at all: you would ship built-in placeholders while believing you shipped
the values someone spent an afternoon tuning against real footage.

So most of what follows is about refusing bad input loudly.
"""

import json
from pathlib import Path

import pytest

from rehab_ai.rules.loader import DEFAULT_RULES_PATH, RulesError, load_rules


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def base_rules() -> dict:
    return json.loads(DEFAULT_RULES_PATH.read_text(encoding="utf-8"))


def write(tmp_path: Path, data) -> Path:
    path = tmp_path / "thresholds.v1.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def load_mutated(tmp_path: Path, mutate) -> None:
    """Apply `mutate` to a copy of the real file, then load it."""
    data = base_rules()
    mutate(data)
    load_rules(write(tmp_path, data))


# --------------------------------------------------------------------------
# The shipped file must actually be valid
# --------------------------------------------------------------------------


def test_the_real_rules_file_loads():
    """If this fails, nothing downstream can start."""
    rules = load_rules()
    assert rules.schema_version == 1
    assert rules.strategy.trigger_threshold > rules.strategy.reset_threshold
    assert rules.capture.mirror_before_inference is False


def test_the_shipped_file_is_marked_untuned():
    """Placeholder values must announce themselves. An untuned build that looks
    tuned is a fabricated result wearing a real result's clothes."""
    rules = load_rules()
    assert rules.is_tuned is False
    assert "UNTUNED" in rules.tuned_against


def test_cue_phrases_are_present_and_have_clips():
    rules = load_rules()
    assert "hip_dominant" in rules.cue.phrases
    assert rules.cue.phrases["hip_dominant"].clip.endswith(".wav")
    assert rules.cue.phrases["hip_dominant"].text


def test_latency_budget_matches_the_trd():
    assert load_rules().cue.max_latency_ms <= 200


# --------------------------------------------------------------------------
# Malformed input -- the CP 2 gate
# --------------------------------------------------------------------------


def test_missing_file_raises(tmp_path):
    with pytest.raises(RulesError, match="not found"):
        load_rules(tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "thresholds.v1.json"
    path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(RulesError, match="not valid JSON"):
        load_rules(path)


def test_top_level_array_raises(tmp_path):
    with pytest.raises(RulesError, match="JSON object at the top level"):
        load_rules(write(tmp_path, [1, 2, 3]))


def test_unsupported_schema_version_raises(tmp_path):
    """Refuse to guess at a file written for a different schema. The Kotlin
    build reads this same file; a silent version mismatch would let the two
    codebases diverge without either noticing."""
    with pytest.raises(RulesError, match="not supported"):
        load_mutated(tmp_path, lambda d: d.update(schema_version=99))


def test_missing_section_raises(tmp_path):
    with pytest.raises(RulesError, match="missing required section: 'strategy'"):
        load_mutated(tmp_path, lambda d: d.pop("strategy"))


def test_missing_key_raises(tmp_path):
    with pytest.raises(RulesError, match="strategy.trigger_threshold"):
        load_mutated(tmp_path, lambda d: d["strategy"].pop("trigger_threshold"))


def test_wrong_type_raises(tmp_path):
    with pytest.raises(RulesError, match="must be a number"):
        load_mutated(tmp_path, lambda d: d["strategy"].update(trigger_threshold="0.7"))


def test_boolean_is_not_accepted_as_a_number(tmp_path):
    """True == 1 in Python. Without an explicit bool check this would sail
    through and produce a threshold of 1.0."""
    with pytest.raises(RulesError, match="must be a number"):
        load_mutated(tmp_path, lambda d: d["strategy"].update(trigger_threshold=True))


def test_empty_cue_phrases_raises(tmp_path):
    with pytest.raises(RulesError, match="non-empty object"):
        load_mutated(tmp_path, lambda d: d["cue"].update(phrases={}))


def test_cue_phrase_missing_clip_raises(tmp_path):
    with pytest.raises(RulesError, match="cue.phrases.hip_dominant.clip"):
        load_mutated(tmp_path, lambda d: d["cue"]["phrases"]["hip_dominant"].pop("clip"))


# --------------------------------------------------------------------------
# Coherence -- files that parse cleanly but would behave wrongly
# --------------------------------------------------------------------------


def test_mirroring_before_inference_is_refused(tmp_path):
    """The single most dangerous value in this file. Mirroring swaps the pose
    model's LEFT/RIGHT labels, so the operated-side lock would measure the
    healthy leg -- silently, with no test downstream able to notice."""
    with pytest.raises(RulesError, match="swaps the pose model"):
        load_mutated(tmp_path, lambda d: d["capture"].update(mirror_before_inference=True))


def test_reset_at_or_above_trigger_is_refused(tmp_path):
    """Without a gap there is no hysteresis, and a signal hovering at the line
    fires the cue on every other frame."""
    with pytest.raises(RulesError, match="must be BELOW"):
        load_mutated(tmp_path, lambda d: d["strategy"].update(reset_threshold=0.70))


def test_reset_above_trigger_is_refused(tmp_path):
    with pytest.raises(RulesError, match="must be BELOW"):
        load_mutated(tmp_path, lambda d: d["strategy"].update(reset_threshold=0.85))


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_signal_thresholds_outside_zero_to_one_are_refused(tmp_path, value):
    with pytest.raises(RulesError, match="within 0.0..1.0"):
        load_mutated(
            tmp_path,
            lambda d: d["strategy"].update(trigger_threshold=value, reset_threshold=value - 1),
        )


def test_visibility_bands_that_cannot_produce_degraded_are_refused(tmp_path):
    """If the degraded floor is at or above the good floor, no frame can ever
    be DEGRADED -- collapsing three quality states back into two."""
    with pytest.raises(RulesError, match="no frame can ever be classified DEGRADED"):
        load_mutated(tmp_path, lambda d: d["observation"].update(degraded_min_visibility=0.65))


def test_trigger_smoothed_harder_than_meter_is_refused(tmp_path):
    """Inverts the whole reason the two signal paths are separate: the trigger
    is latency-critical, the meter is not."""
    with pytest.raises(RulesError, match="inverts the reason"):
        load_mutated(tmp_path, lambda d: d["strategy"].update(trigger_smoothing_frames=9))


def test_zero_frame_smoothing_window_is_refused(tmp_path):
    with pytest.raises(RulesError, match="at least 1 frame"):
        load_mutated(
            tmp_path,
            lambda d: d["strategy"].update(trigger_smoothing_frames=0, meter_smoothing_frames=0),
        )


def test_rise_onset_at_or_below_seated_angle_is_refused(tmp_path):
    """This is the old exercises.py bug expressed as configuration: a rise
    detected before the patient has moved."""
    with pytest.raises(RulesError, match="before the patient has moved"):
        load_mutated(tmp_path, lambda d: d["phase"].update(rise_onset_knee_angle=110.0))


def test_standing_below_rise_onset_is_refused(tmp_path):
    with pytest.raises(RulesError, match="standing_min_knee_angle must be above"):
        load_mutated(tmp_path, lambda d: d["phase"].update(standing_min_knee_angle=115.0))


def test_rest_only_below_lock_threshold_is_refused(tmp_path):
    """Rest-only is the stricter response; it cannot trigger at a lower pain
    score than the lesser one."""
    with pytest.raises(RulesError, match="cannot trigger at a lower pain score"):
        load_mutated(tmp_path, lambda d: d["policy"].update(pain_rest_only_threshold=3))


def test_flag_rate_outside_zero_to_one_is_refused(tmp_path):
    with pytest.raises(RulesError, match="within 0.0..1.0"):
        load_mutated(tmp_path, lambda d: d["policy"].update(compensation_flag_rate_lock=1.4))
