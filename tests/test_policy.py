"""
Tests for the deterministic lock policy (CP 4).

Gate: five named cases green, and every decision returns non-empty reason
codes. A decision with no stated reason cannot be explained to a patient or
audited by a clinician, which is the whole point of this module having no AI
in it.
"""

import pytest

from rehab_ai.models.session import (
    CompensationMetrics,
    CompensationStatus,
    CompensationSummary,
    Decision,
    InputSource,
    PainReport,
    PolicyInput,
    ReasonCode,
    SessionStatus,
    SwellingComparison,
    SwellingComparisonStatus,
    SwellingReport,
)
from rehab_ai.policy.engine import decide
from rehab_ai.rules.loader import load_rules


@pytest.fixture(scope="module")
def rules():
    return load_rules().policy


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def observed(total=8, scored=8, flagged=0) -> CompensationSummary:
    return CompensationSummary(
        status=CompensationStatus.AVAILABLE,
        metrics=CompensationMetrics(reps_total=total, reps_scored=scored, reps_flagged=flagged),
    )


UNOBSERVED = CompensationSummary(status=CompensationStatus.UNOBSERVABLE)


def given(
    pain=2,
    swelling=SwellingReport.SAME,
    swelling_status=SwellingComparisonStatus.AVAILABLE,
    compensation=None,
    protocol_day=30,
    status=SessionStatus.COMPLETED,
) -> PolicyInput:
    return PolicyInput(
        pain=None if pain is None else PainReport(pain, InputSource.TAP),
        swelling=SwellingComparison(
            status=swelling_status,
            report=swelling if swelling_status is SwellingComparisonStatus.AVAILABLE else None,
        ),
        compensation=observed() if compensation is None else compensation,
        protocol_day=protocol_day,
        session_status=status,
    )


# --------------------------------------------------------------------------
# The universal invariant
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        given(),
        given(pain=9),
        given(pain=None),
        given(swelling_status=SwellingComparisonStatus.BASELINE_ONLY, swelling=None),
        given(swelling_status=SwellingComparisonStatus.NO_COMPARISON, swelling=None),
        given(compensation=UNOBSERVED),
        given(status=SessionStatus.ABANDONED),
        given(protocol_day=3),
    ],
    ids=[
        "normal",
        "severe-pain",
        "no-pain-reading",
        "first-session",
        "skipped-day",
        "unobservable",
        "abandoned",
        "early-protocol",
    ],
)
def test_every_decision_states_why(case, rules):
    """The gate's second half. No path through decide() may return a bare
    verdict -- a lock a patient cannot be given a reason for is not a decision
    this system is allowed to make."""
    decision = decide(case, rules)
    assert decision.reason_codes, "decision returned no reason codes"
    assert all(isinstance(c, ReasonCode) for c in decision.reason_codes)


# --------------------------------------------------------------------------
# Case 1 -- session 1, no yesterday to compare against
# --------------------------------------------------------------------------


def test_first_session_is_not_blocked_by_having_no_baseline(rules):
    """BASELINE_ONLY is expected structure, not absent evidence. Day one must
    not be punished for the calendar."""
    decision = decide(
        given(
            swelling_status=SwellingComparisonStatus.BASELINE_ONLY,
            swelling=None,
            protocol_day=30,
        ),
        rules,
    )
    assert decision.decision is Decision.ALLOW_FULL
    assert ReasonCode.SWELLING_NO_BASELINE in decision.reason_codes


def test_first_session_reason_differs_from_a_skipped_day(rules):
    """Both mean 'no comparison was made'. They are different facts and get
    different codes, so the stored decision says which one happened."""
    first = decide(
        given(swelling_status=SwellingComparisonStatus.BASELINE_ONLY, swelling=None), rules
    )
    skipped = decide(
        given(swelling_status=SwellingComparisonStatus.NO_COMPARISON, swelling=None), rules
    )
    assert ReasonCode.SWELLING_NO_BASELINE in first.reason_codes
    assert ReasonCode.SWELLING_COMPARISON_UNAVAILABLE in skipped.reason_codes


# --------------------------------------------------------------------------
# Case 2 -- the normal session
# --------------------------------------------------------------------------


def test_settled_session_allows_full_plan(rules):
    decision = decide(given(pain=2, swelling=SwellingReport.LESS), rules)
    assert decision.decision is Decision.ALLOW_FULL
    assert decision.loaded_work_allowed
    assert ReasonCode.PAIN_WITHIN_RANGE in decision.reason_codes
    assert ReasonCode.SWELLING_IMPROVED in decision.reason_codes
    assert ReasonCode.COMPENSATION_INFREQUENT in decision.reason_codes


def test_elevated_pain_locks_loaded_work(rules):
    decision = decide(given(pain=6), rules)
    assert decision.decision is Decision.LOCK_LOADED
    assert ReasonCode.PAIN_ELEVATED in decision.reason_codes


def test_severe_pain_is_a_rest_day(rules):
    decision = decide(given(pain=9), rules)
    assert decision.decision is Decision.REST_ONLY


def test_swelling_up_locks_loaded_work(rules):
    decision = decide(given(pain=2, swelling=SwellingReport.PUFFIER), rules)
    assert decision.decision is Decision.LOCK_LOADED
    assert ReasonCode.SWELLING_INCREASED in decision.reason_codes


def test_frequent_compensation_locks_loaded_work(rules):
    """The camera's contribution. Half the observed reps were hip-dominant, so
    adding load would reinforce the compensation rather than the quad."""
    decision = decide(given(compensation=observed(scored=8, flagged=4)), rules)
    assert decision.decision is Decision.LOCK_LOADED
    assert ReasonCode.COMPENSATION_FREQUENT in decision.reason_codes


def test_pain_and_camera_agree_produces_both_reasons(rules):
    """explain/ needs every contributing reason, not just the first one that
    tripped -- 'squats are off because pain was up AND you were compensating'
    is a different sentence from either half."""
    decision = decide(given(pain=7, compensation=observed(scored=8, flagged=5)), rules)
    assert ReasonCode.PAIN_ELEVATED in decision.reason_codes
    assert ReasonCode.COMPENSATION_FREQUENT in decision.reason_codes


# --------------------------------------------------------------------------
# Case 3 -- a skipped day
# --------------------------------------------------------------------------


def test_skipped_day_does_not_block_on_its_own(rules):
    """A gap in the calendar is not evidence of a problem. Swelling simply
    contributes nothing today."""
    decision = decide(
        given(swelling_status=SwellingComparisonStatus.NO_COMPARISON, swelling=None), rules
    )
    assert decision.decision is Decision.ALLOW_FULL
    assert ReasonCode.SWELLING_COMPARISON_UNAVAILABLE in decision.reason_codes


def test_skipped_day_never_produces_a_swelling_verdict(rules):
    """The bug this prevents: comparing Wednesday against Monday and reporting
    'swelling increased' as though Monday were yesterday."""
    decision = decide(
        given(swelling_status=SwellingComparisonStatus.NO_COMPARISON, swelling=None), rules
    )
    for code in (
        ReasonCode.SWELLING_INCREASED,
        ReasonCode.SWELLING_STABLE,
        ReasonCode.SWELLING_IMPROVED,
    ):
        assert code not in decision.reason_codes


# --------------------------------------------------------------------------
# Case 4 -- abandoned session
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status", [SessionStatus.ABANDONED, SessionStatus.ACTIVE, SessionStatus.NOT_STARTED]
)
def test_incomplete_session_holds(status, rules):
    """Not a bad day and not a good one -- an absence of information. The
    honest response is to withhold a progression, not invent one."""
    decision = decide(given(status=status), rules)
    assert decision.decision is Decision.HOLD
    assert ReasonCode.SESSION_INCOMPLETE in decision.reason_codes


def test_abandoned_session_still_records_the_pain_reading(rules):
    """The patient did report pain before quitting. That fact is real and
    survives into the decision even though the session did not."""
    decision = decide(given(pain=7, status=SessionStatus.ABANDONED), rules)
    assert ReasonCode.PAIN_ELEVATED in decision.reason_codes


def test_abandoned_session_is_not_an_allow(rules):
    decision = decide(given(pain=1, swelling=SwellingReport.LESS, status=SessionStatus.ABANDONED), rules)
    assert not decision.loaded_work_allowed


# --------------------------------------------------------------------------
# Case 5 -- nothing observable  ***the one that matters***
# --------------------------------------------------------------------------


def test_unobservable_session_holds_rather_than_allowing(rules):
    """THE case. Every input the patient controls looks perfect: low pain,
    swelling down, session completed. But the camera assessed nothing.

    Absence of evidence must not read as evidence of absence. Loaded work
    stays off until there is something to justify it.
    """
    decision = decide(
        given(pain=1, swelling=SwellingReport.LESS, compensation=UNOBSERVED), rules
    )
    assert decision.decision is Decision.HOLD
    assert not decision.loaded_work_allowed
    assert ReasonCode.MOVEMENT_OBSERVATION_INCOMPLETE in decision.reason_codes


def test_unobservable_never_reports_a_compensation_verdict(rules):
    """It must not claim the patient compensated, nor that they did not."""
    decision = decide(given(compensation=UNOBSERVED), rules)
    assert ReasonCode.COMPENSATION_FREQUENT not in decision.reason_codes
    assert ReasonCode.COMPENSATION_INFREQUENT not in decision.reason_codes


def test_unobservable_is_recorded_in_provenance(rules):
    decision = decide(given(compensation=UNOBSERVED), rules)
    assert decision.input_quality.compensation_status is CompensationStatus.UNOBSERVABLE
    assert not decision.input_quality.is_complete


def test_severe_pain_outranks_an_unobservable_camera(rules):
    """A patient reporting pain of 9 gets a rest day whether or not the camera
    was working. Self-report is not gated behind machine vision."""
    decision = decide(given(pain=9, compensation=UNOBSERVED), rules)
    assert decision.decision is Decision.REST_ONLY


# --------------------------------------------------------------------------
# Missing pain reading
# --------------------------------------------------------------------------


def test_absent_pain_produces_no_pain_reason(rules):
    """Not 'assume it is fine'. The absence shows up as a missing code and in
    the provenance, so a later reader can see it was never asked."""
    decision = decide(given(pain=None), rules)
    assert ReasonCode.PAIN_ELEVATED not in decision.reason_codes
    assert ReasonCode.PAIN_WITHIN_RANGE not in decision.reason_codes
    assert not decision.input_quality.pain_present


# --------------------------------------------------------------------------
# Early protocol days
# --------------------------------------------------------------------------


def test_early_protocol_day_is_conservative(rules):
    decision = decide(given(pain=1, swelling=SwellingReport.LESS, protocol_day=3), rules)
    assert decision.decision is Decision.LOCK_LOADED
    assert ReasonCode.EARLY_PROTOCOL_DAY in decision.reason_codes


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_the_same_input_always_gives_the_same_output(rules):
    """No clock, no randomness, no model. This is what makes the decision
    auditable months later from a stored row."""
    case = given(pain=6, swelling=SwellingReport.PUFFIER, compensation=observed(scored=6, flagged=3))
    first = decide(case, rules)
    for _ in range(50):
        again = decide(case, rules)
        assert again.decision is first.decision
        assert again.reason_codes == first.reason_codes
