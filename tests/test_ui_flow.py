"""
Headless UI and flow tests.

Qt runs under QT_QPA_PLATFORM=offscreen, so views are constructed and driven
without a display. This is the desktop-app equivalent of the browser pass /qa
does for a web app: walk the screens, fire the interactions, check the states.

TWO THINGS THESE COVER THAT NOTHING ELSE DID
============================================
1. build_swelling_comparison -- a safety-relevant rule that was living in
   main_window.py with no tests at all. It decides BASELINE_ONLY vs
   NO_COMPARISON vs AVAILABLE, which is the skipped-day case from CP 4. It has
   since moved into policy/ where it belongs; these are its tests.

2. Intake's absence handling. The pain slider resting at 0 must not be
   indistinguishable from a reported 0, and Continue must not be reachable
   until all three answers exist.
"""

import os
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from rehab_ai.models.session import (  # noqa: E402
    CompensationStatus,
    Decision,
    InputQuality,
    InputSource,
    LockDecision,
    PainReport,
    ReasonCode,
    RehabSession,
    RepResult,
    RepValidity,
    SessionStatus,
    Side,
    SwellingComparison,
    SwellingComparisonStatus,
    SwellingReport,
)
from rehab_ai.policy.engine import build_swelling_comparison  # noqa: E402
from rehab_ai.rules.loader import load_rules  # noqa: E402
from rehab_ai.ui.intake_view import IntakeView  # noqa: E402
from rehab_ai.ui.summary_view import SummaryView  # noqa: E402


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def rules():
    return load_rules()


# ==========================================================================
# build_swelling_comparison -- previously untested, and safety-relevant
# ==========================================================================

NOW = datetime(2026, 9, 10, 9, 0)


def test_no_prior_session_is_baseline_not_missing(rules):
    """Day one. Expected structure, not absent evidence -- and CP 4 proves the
    difference changes the lock decision."""
    result = build_swelling_comparison(SwellingReport.PUFFIER, None, NOW, rules.policy)

    assert result.status is SwellingComparisonStatus.BASELINE_ONLY
    assert result.report is None  # nothing to compare, so no value is carried


def test_yesterday_gives_an_available_comparison(rules):
    yesterday = NOW - timedelta(days=1)
    result = build_swelling_comparison(SwellingReport.PUFFIER, yesterday, NOW, rules.policy)

    assert result.status is SwellingComparisonStatus.AVAILABLE
    assert result.report is SwellingReport.PUFFIER


def test_a_gap_produces_no_comparison_and_drops_the_value(rules):
    """Monday 4, Tuesday skipped, Wednesday 5.

    Comparing Wednesday against Monday would invent a trend across a day nobody
    observed. The value is dropped, not carried with a caveat -- the type
    refuses to hold one, which is what makes this checkable.
    """
    long_ago = NOW - timedelta(days=5)
    result = build_swelling_comparison(SwellingReport.PUFFIER, long_ago, NOW, rules.policy)

    assert result.status is SwellingComparisonStatus.NO_COMPARISON
    assert result.report is None


def test_the_gap_boundary_comes_from_the_rules_file(rules):
    """Not a magic number in the UI. It travels to the Kotlin build with every
    other threshold."""
    limit = rules.policy.max_comparison_gap_days

    inside = build_swelling_comparison(
        SwellingReport.SAME, NOW - timedelta(days=limit), NOW, rules.policy
    )
    outside = build_swelling_comparison(
        SwellingReport.SAME, NOW - timedelta(days=limit, seconds=1), NOW, rules.policy
    )

    assert inside.status is SwellingComparisonStatus.AVAILABLE
    assert outside.status is SwellingComparisonStatus.NO_COMPARISON


def test_the_rule_is_pure_and_takes_no_repository(rules):
    """It receives a timestamp, not a database handle. That is what lets it be
    tested exhaustively and ported to Kotlin as logic rather than as plumbing."""
    import inspect

    params = list(inspect.signature(build_swelling_comparison).parameters)
    assert params == ["report", "previous_session_at", "now", "rules"]


# ==========================================================================
# Intake -- absence must stay absent
# ==========================================================================


def test_continue_is_disabled_until_every_answer_exists(qt_app):
    view = IntakeView(Side.LEFT)
    assert not view._continue.isEnabled()

    view._on_pain(4)
    assert not view._continue.isEnabled(), "enabled with no swelling or orientation"

    view._on_swelling(SwellingReport.SAME)
    assert not view._continue.isEnabled(), "enabled with no orientation answer"

    view._on_facing(Side.LEFT)
    assert view._continue.isEnabled()


def test_an_untouched_slider_is_not_a_reported_zero(qt_app):
    """0 is a real pain score meaning none at all. A patient who never touched
    the slider has not reported that, and the two must not be conflated."""
    view = IntakeView(Side.LEFT)

    assert view._pain is None
    assert "Not answered" in view._pain_value.text()


def test_a_deliberate_zero_is_a_real_answer(qt_app):
    view = IntakeView(Side.LEFT)
    view._on_pain(0)

    assert view._pain == 0
    assert "0 out of 10" in view._pain_value.text()


def test_submitting_emits_the_same_types_the_tap_path_produces(qt_app):
    view = IntakeView(Side.LEFT)
    captured = []
    view.submitted.connect(lambda *args: captured.append(args))

    view._on_pain(6)
    view._on_swelling(SwellingReport.PUFFIER)
    view._on_facing(Side.LEFT)
    view._submit()

    pain, swelling, facing = captured[0]
    assert isinstance(pain, PainReport)
    assert pain.value == 6 and pain.source is InputSource.TAP
    assert swelling is SwellingReport.PUFFIER
    assert facing is Side.LEFT


def test_facing_the_wrong_way_warns_but_does_not_block(qt_app):
    """The patient may genuinely be set up this way and the session records it.
    Blocking would be worse than saying plainly what it costs."""
    view = IntakeView(Side.LEFT)
    view._on_pain(3)
    view._on_swelling(SwellingReport.SAME)
    view._on_facing(Side.RIGHT)

    assert view._continue.isEnabled()
    assert "left knee" in view._facing_warning.text()
    assert "won't be scored" in view._facing_warning.text()


def test_facing_the_right_way_clears_the_warning(qt_app):
    view = IntakeView(Side.LEFT)
    view._on_facing(Side.RIGHT)
    assert view._facing_warning.text()

    view._on_facing(Side.LEFT)
    assert view._facing_warning.text() == ""


def test_voice_fills_only_what_it_could_read(qt_app):
    """The parser refuses to guess; the screen must not guess either. What voice
    could not read stays unanswered and Continue stays disabled."""
    view = IntakeView(Side.LEFT)
    view.apply_voice(PainReport(4, InputSource.VOICE), None)

    assert view._pain == 4
    assert view._swelling is None
    assert not view._continue.isEnabled()


# ==========================================================================
# Summary -- the model's failure must be visible, never papered over
# ==========================================================================


def _decision(*codes: ReasonCode, decision=Decision.LOCK_LOADED) -> LockDecision:
    return LockDecision(
        decision=decision,
        reason_codes=list(codes),
        input_quality=InputQuality(
            pain_present=True,
            swelling_status=SwellingComparisonStatus.AVAILABLE,
            compensation_status=CompensationStatus.AVAILABLE,
            session_status=SessionStatus.COMPLETED,
        ),
    )


def _session(reps=None) -> RehabSession:
    return RehabSession(
        session_id="s1",
        operated_side=Side.LEFT,
        protocol_day=20,
        status=SessionStatus.COMPLETED,
        pain=PainReport(4),
        swelling=SwellingComparison(SwellingComparisonStatus.AVAILABLE, SwellingReport.PUFFIER),
        reps=reps or [],
        started_at=datetime(2026, 9, 10, 9, 0),
    )


def rep(index=0, validity=RepValidity.VALID, compensating=False) -> RepResult:
    return RepResult(
        rep_index=index, side=Side.LEFT, validity=validity, compensating=compensating,
        peak_hip_drive=None if compensating is None else 0.4,
        descent_control=None if compensating is None else 0.8,
        frames_observed=40, frames_total=40, cue_fired=False,
        started_at=0.0, duration_s=2.2,
    )


def test_a_generated_sentence_is_labelled_as_model_output(qt_app):
    view = SummaryView()
    view.present(_session([rep()]), _decision(ReasonCode.PAIN_ELEVATED), "Squats are off today.")

    assert view._sentence.text() == "Squats are off today."
    assert "language model" in view._sentence_label.text()


def test_a_failed_model_says_so_instead_of_substituting_text(qt_app):
    """No template stands in for model output. The decision and reasons are
    unaffected and still shown -- only the prose is missing."""
    view = SummaryView()
    decision = _decision(ReasonCode.PAIN_ELEVATED, ReasonCode.SWELLING_INCREASED)
    view.present(_session([rep()]), decision, None, "connection refused")

    assert "isn't available" in view._sentence.text()
    assert "connection refused" in view._sentence.text()
    assert "unaffected" in view._sentence_label.text()
    # The reasons survive the model's failure entirely.
    assert "Pain was higher" in view._reasons.text()
    assert "puffier" in view._reasons.text()


def test_unobservable_reps_are_reported_not_folded_into_the_total(qt_app):
    """A session summary that hid them would present a number the camera did
    not earn."""
    view = SummaryView()
    reps = [rep(0), rep(1, RepValidity.INVALID, None), rep(2, RepValidity.VALID, True)]
    view.present(_session(reps), _decision(ReasonCode.COMPENSATION_FREQUENT), "x")

    text = view._stands.text()
    assert "2 of 3 stands scored" in text
    assert "1 not clear enough to score" in text


def test_a_session_nobody_could_score_says_exactly_that(qt_app):
    view = SummaryView()
    reps = [rep(i, RepValidity.INVALID, None) for i in range(4)]
    view.present(_session(reps), _decision(ReasonCode.MOVEMENT_OBSERVATION_INCOMPLETE,
                                           decision=Decision.HOLD), "x")

    assert "none clear enough to score" in view._stands.text()


def test_a_session_with_no_reps_does_not_crash_the_summary(qt_app):
    view = SummaryView()
    view.present(_session([]), _decision(ReasonCode.SESSION_INCOMPLETE, decision=Decision.HOLD), "x")
    assert "0 stands" in view._stands.text() or "none clear" in view._stands.text()


@pytest.mark.parametrize(
    "decision,expected",
    [
        (Decision.ALLOW_FULL, "on the plan"),
        (Decision.LOCK_LOADED, "Squats are off"),
        (Decision.REST_ONLY, "rest day"),
        (Decision.HOLD, "keeping tomorrow the same"),
    ],
)
def test_every_decision_has_a_calm_headline(qt_app, decision, expected):
    """PRD 8: never alarming. Every branch needs copy, including HOLD, which is
    the one a hurried implementation would leave blank."""
    view = SummaryView()
    view.present(_session([rep()]), _decision(ReasonCode.PAIN_WITHIN_RANGE, decision=decision), "x")
    assert expected in view._headline.text()


def test_every_reason_code_renders_as_plain_english(qt_app):
    """A code with no mapping would show the raw enum value to a patient."""
    from rehab_ai.ui.summary_view import _REASON_TEXT

    for code in ReasonCode:
        assert code in _REASON_TEXT, f"no patient-facing text for {code.value}"
        assert "_" not in _REASON_TEXT[code], f"{code.value} maps to a raw-looking string"
