"""
Tests for the shared vocabulary (CP 1).

These are not "does the dataclass hold values" tests. Every case here pins one
of the design decisions from the engineering review -- the ones that are easy
to undo accidentally six commits from now, because undoing them makes the code
look simpler while making it quietly wrong.

The load-bearing case is `test_all_invalid_reps_are_unobservable_not_clean`.
"""

from datetime import date, datetime

import pytest

from rehab_ai.models.session import (
    CompensationMetrics,
    CompensationStatus,
    CompensationSummary,
    Decision,
    InputQuality,
    InputSource,
    JointAngles,
    LockDecision,
    Observation,
    ObservationQuality,
    PainReport,
    PolicyInput,
    Procedure,
    Profile,
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


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_rep(
    index: int = 0,
    validity: RepValidity = RepValidity.VALID,
    compensating: bool | None = False,
    frames_observed: int = 40,
    frames_total: int = 40,
) -> RepResult:
    return RepResult(
        rep_index=index,
        side=Side.LEFT,
        validity=validity,
        compensating=compensating,
        peak_hip_drive=0.3 if compensating is not None else None,
        descent_control=0.8 if compensating is not None else None,
        frames_observed=frames_observed,
        frames_total=frames_total,
        cue_fired=bool(compensating),
        started_at=0.0,
        duration_s=2.4,
    )


# --------------------------------------------------------------------------
# Side and Profile -- the operated-side binding chain
# --------------------------------------------------------------------------


def test_side_other_flips():
    assert Side.LEFT.other is Side.RIGHT
    assert Side.RIGHT.other is Side.LEFT


def test_profile_rejects_a_non_side_operated_side():
    """Every clinically meaningful measurement depends on this field. A string
    that looks right ("left") would silently work everywhere until something
    compared it with `is`."""
    with pytest.raises(TypeError, match="operated_side must be a Side"):
        Profile(
            name="Test",
            procedure=Procedure.TKA,
            operated_side="left",  # type: ignore[arg-type]
            surgery_date=date(2026, 8, 1),
        )


def test_protocol_day_counts_from_surgery():
    profile = Profile("Test", Procedure.TKA, Side.LEFT, date(2026, 8, 1))
    assert profile.protocol_day(date(2026, 8, 1)) == 0
    assert profile.protocol_day(date(2026, 8, 15)) == 14


def test_session_inherits_operated_side_from_profile():
    """Bound once, at session creation -- never rediscovered per frame."""
    profile = Profile("Test", Procedure.ACL, Side.RIGHT, date(2026, 8, 1))
    session = RehabSession.for_profile("s1", profile, on=date(2026, 8, 11))
    assert session.operated_side is Side.RIGHT
    assert session.protocol_day == 10


# --------------------------------------------------------------------------
# The three concepts stay three concepts
# --------------------------------------------------------------------------


def test_unobservable_is_not_trustworthy_but_degraded_is():
    assert ObservationQuality.GOOD.is_trustworthy
    assert ObservationQuality.DEGRADED.is_trustworthy
    assert not ObservationQuality.UNOBSERVABLE.is_trustworthy


def test_unobservable_observation_carries_no_angles():
    """An unobservable frame is a recorded fact, not an exception and not a
    gap. But it must not carry measurements it never had."""
    obs = Observation.unobservable(timestamp=1.0, visibility=0.31)
    assert obs.quality is ObservationQuality.UNOBSERVABLE
    assert obs.angles is None
    assert obs.visibility == 0.31  # the raw measurement survives


def test_good_observation_keeps_visibility_and_angles_separate():
    obs = Observation(
        timestamp=1.0,
        quality=ObservationQuality.GOOD,
        visibility=0.94,
        angles=JointAngles(hip=95.0, knee=88.0),
    )
    assert obs.visibility == 0.94
    assert obs.angles.knee == 88.0


def test_invalid_reps_do_not_count_toward_the_compensation_rate():
    assert RepValidity.VALID.counts_toward_compensation_rate
    assert RepValidity.DEGRADED.counts_toward_compensation_rate
    assert not RepValidity.INVALID.counts_toward_compensation_rate


# --------------------------------------------------------------------------
# A rep nobody saw has no verdict
# --------------------------------------------------------------------------


def test_invalid_rep_cannot_claim_a_compensation_verdict():
    """False would be a lie shaped like data."""
    with pytest.raises(ValueError, match="INVALID rep cannot carry"):
        make_rep(validity=RepValidity.INVALID, compensating=False)


def test_invalid_rep_with_no_verdict_is_allowed():
    rep = make_rep(validity=RepValidity.INVALID, compensating=None, frames_observed=6)
    assert rep.compensating is None
    assert rep.observation_coverage == pytest.approx(6 / 40)


def test_observation_coverage_handles_zero_frames():
    rep = make_rep(compensating=None, validity=RepValidity.INVALID, frames_observed=0, frames_total=0)
    assert rep.observation_coverage == 0.0


# --------------------------------------------------------------------------
# Check-in
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-1, 11, 100])
def test_pain_outside_range_is_rejected(value):
    with pytest.raises(ValueError, match="pain must be 0..10"):
        PainReport(value=value)


def test_voice_and_tap_produce_the_same_shape():
    """Voice is an input method, not a separate data model."""
    tapped = PainReport(4, InputSource.TAP)
    spoken = PainReport(4, InputSource.VOICE)
    assert tapped.value == spoken.value


# --------------------------------------------------------------------------
# Typed absence: swelling
# --------------------------------------------------------------------------


def test_available_swelling_requires_a_value():
    with pytest.raises(ValueError, match="requires a report value"):
        SwellingComparison(status=SwellingComparisonStatus.AVAILABLE)


@pytest.mark.parametrize(
    "status",
    [
        SwellingComparisonStatus.BASELINE_ONLY,
        SwellingComparisonStatus.NO_COMPARISON,
        SwellingComparisonStatus.UNAVAILABLE,
    ],
)
def test_absent_swelling_cannot_smuggle_a_value(status):
    """Missing information must stay missing. A NO_COMPARISON that quietly
    carries yesterday's value is how a skipped day becomes a false trend."""
    with pytest.raises(ValueError, match="cannot carry a report value"):
        SwellingComparison(status=status, report=SwellingReport.SAME)


def test_first_session_is_baseline_not_missing():
    """Day 1 has no yesterday. That is expected structure, not absent evidence,
    and the two must be distinguishable."""
    first = SwellingComparison(status=SwellingComparisonStatus.BASELINE_ONLY)
    skipped = SwellingComparison(status=SwellingComparisonStatus.NO_COMPARISON)
    assert first.status is not skipped.status
    assert first.report is None and skipped.report is None


# --------------------------------------------------------------------------
# Typed absence: compensation  ***the load-bearing case***
# --------------------------------------------------------------------------


def test_all_invalid_reps_are_unobservable_not_clean():
    """THE test. If every rep was unobservable, the summary must say so --
    not report zero flags, which reads downstream as a perfect session.

    "We could not watch" and "we watched and it was clean" are different
    facts. Collapsing them would let policy/ unlock loaded exercise on the
    strength of evidence that does not exist.
    """
    reps = [make_rep(i, RepValidity.INVALID, compensating=None) for i in range(5)]
    summary = CompensationSummary.from_reps(reps)

    assert summary.status is CompensationStatus.UNOBSERVABLE
    assert summary.metrics is None


def test_a_clean_session_reports_zero_flags_with_metrics_present():
    """Contrast with the case above: same zero, entirely different meaning."""
    reps = [make_rep(i, RepValidity.VALID, compensating=False) for i in range(5)]
    summary = CompensationSummary.from_reps(reps)

    assert summary.status is CompensationStatus.AVAILABLE
    assert summary.metrics.reps_flagged == 0
    assert summary.metrics.flag_rate == 0.0


def test_unobservable_summary_cannot_carry_metrics():
    with pytest.raises(ValueError, match="Do not manufacture"):
        CompensationSummary(
            status=CompensationStatus.UNOBSERVABLE,
            metrics=CompensationMetrics(reps_total=5, reps_scored=5, reps_flagged=0),
        )


def test_available_summary_requires_metrics():
    with pytest.raises(ValueError, match="requires metrics"):
        CompensationSummary(status=CompensationStatus.AVAILABLE)


def test_flag_rate_divides_by_scored_reps_not_attempted():
    """Unobserved reps must not dilute a real signal. Two flagged out of two
    scored is a rate of 1.0, even though eight reps were attempted."""
    reps = [make_rep(i, RepValidity.VALID, compensating=True) for i in range(2)]
    reps += [make_rep(i, RepValidity.INVALID, compensating=None) for i in range(2, 8)]

    summary = CompensationSummary.from_reps(reps)

    assert summary.metrics.reps_total == 8
    assert summary.metrics.reps_scored == 2
    assert summary.metrics.flag_rate == 1.0


def test_degraded_reps_still_count():
    """Brief occlusion degrades a rep; it does not discard it."""
    reps = [make_rep(0, RepValidity.DEGRADED, compensating=True)]
    summary = CompensationSummary.from_reps(reps)
    assert summary.status is CompensationStatus.AVAILABLE
    assert summary.metrics.reps_scored == 1


def test_empty_session_is_unobservable():
    assert CompensationSummary.from_reps([]).status is CompensationStatus.UNOBSERVABLE


# --------------------------------------------------------------------------
# The decision must state why
# --------------------------------------------------------------------------


def _quality() -> InputQuality:
    return InputQuality(
        pain_present=True,
        swelling_status=SwellingComparisonStatus.AVAILABLE,
        compensation_status=CompensationStatus.AVAILABLE,
        session_status=SessionStatus.COMPLETED,
    )


def test_a_decision_without_reasons_is_rejected():
    """A lock the patient cannot be told the reason for, and a clinician
    cannot audit, is not a decision this system is allowed to make."""
    with pytest.raises(ValueError, match="must state why"):
        LockDecision(decision=Decision.LOCK_LOADED, reason_codes=[], input_quality=_quality())


def test_decision_carries_its_reasons_and_provenance():
    decision = LockDecision(
        decision=Decision.LOCK_LOADED,
        reason_codes=[ReasonCode.PAIN_ELEVATED, ReasonCode.SWELLING_INCREASED],
        input_quality=_quality(),
    )
    assert not decision.loaded_work_allowed
    assert ReasonCode.PAIN_ELEVATED in decision.reason_codes
    assert decision.input_quality.is_complete


def test_input_quality_knows_when_it_is_incomplete():
    incomplete = InputQuality(
        pain_present=True,
        swelling_status=SwellingComparisonStatus.NO_COMPARISON,
        compensation_status=CompensationStatus.UNOBSERVABLE,
        session_status=SessionStatus.ABANDONED,
    )
    assert not incomplete.is_complete


# --------------------------------------------------------------------------
# Orientation provenance (CP 3 depends on this being recorded)
# --------------------------------------------------------------------------


def test_orientation_is_unconfirmed_until_set():
    session = RehabSession(session_id="s1", operated_side=Side.LEFT, protocol_day=10)
    assert not session.orientation_confirmed


def test_orientation_confirmed_only_when_operated_side_faces_camera():
    session = RehabSession(session_id="s1", operated_side=Side.LEFT, protocol_day=10)

    session.camera_facing_side = Side.RIGHT
    assert not session.orientation_confirmed

    session.camera_facing_side = Side.LEFT
    assert session.orientation_confirmed


# --------------------------------------------------------------------------
# PolicyInput has no defaults -- absence must be passed explicitly
# --------------------------------------------------------------------------


def test_policy_input_requires_every_field_to_be_stated():
    """There is deliberately no constructor that fills a gap. A caller with no
    pain reading must say so, rather than omitting the argument."""
    with pytest.raises(TypeError):
        PolicyInput(pain=None, swelling=SwellingComparison(SwellingComparisonStatus.BASELINE_ONLY))  # type: ignore[call-arg]


def test_policy_input_accepts_explicit_absence():
    given = PolicyInput(
        pain=None,
        swelling=SwellingComparison(SwellingComparisonStatus.BASELINE_ONLY),
        compensation=CompensationSummary(CompensationStatus.UNOBSERVABLE),
        protocol_day=1,
        session_status=SessionStatus.ABANDONED,
    )
    assert given.pain is None
    assert given.compensation.metrics is None
