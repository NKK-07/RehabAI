"""
engine.py
The lock decision. A pure function, no AI, fully auditable.

    PolicyInput  ──▶  decide()  ──▶  LockDecision
    (typed absence)   (this file)    (decision + reason codes + provenance)

Nothing in this module calls a model, reads a clock, touches the filesystem, or
looks at a camera. Given the same input it returns the same output, forever.
That is the entire reason it exists: explain/ can phrase this decision, and a
clinician can audit it, precisely because no model produced it.

THE INVARIANT
=============
Policy never substitutes defaults for missing clinical or observation data.
Every absence state that can affect a LockDecision is represented explicitly
and has a documented deterministic outcome.

There is no `swelling or SwellingReport.SAME` anywhere below, and there must
never be. A default is a fabricated observation.

THE FIVE NAMED CASES
====================

    situation              swelling status    compensation      outcome
    ─────────────────────────────────────────────────────────────────────────
    session 1              BASELINE_ONLY      available         normal rules
                           (expected structure, not absent evidence)

    normal session         AVAILABLE          available         normal rules

    skipped a day          NO_COMPARISON      available         normal rules,
                           (never compare across the gap)       no swelling input

    abandoned session      any                any               HOLD

    nothing observable     any                UNOBSERVABLE      HOLD

The distinction that matters most is the first row against the last. "There is
no yesterday to compare with" is expected structure on day one and must not
block anything. "The camera could not assess a single rep" is absent evidence,
and loaded work stays off until there is evidence to permit it.
"""

from __future__ import annotations

from rehab_ai.models.session import (
    CompensationStatus,
    Decision,
    InputQuality,
    LockDecision,
    PolicyInput,
    ReasonCode,
    SessionStatus,
    SwellingComparisonStatus,
    SwellingReport,
)
from rehab_ai.rules.loader import PolicyRules


def decide(given: PolicyInput, rules: PolicyRules) -> LockDecision:
    """Return the lock decision for one completed session.

    `rules` comes from rules/thresholds.v1.json so the thresholds travel to the
    Kotlin build unchanged. The *logic* here is what gets ported; the numbers
    are loaded, not retyped.
    """
    quality = InputQuality(
        pain_present=given.pain is not None,
        swelling_status=given.swelling.status,
        compensation_status=given.compensation.status,
        session_status=given.session_status,
    )

    reasons: list[ReasonCode] = []

    # -- Gate 1: was there a session at all? --------------------------------
    # An abandoned session says nothing about how the patient is doing. It is
    # not a bad day and it is not a good one; it is an absence of information,
    # and the honest response is to withhold a progression rather than invent
    # one from a partial record.
    if given.session_status is not SessionStatus.COMPLETED:
        reasons.append(ReasonCode.SESSION_INCOMPLETE)
        reasons.extend(_pain_reasons(given, rules))
        return LockDecision(Decision.HOLD, reasons, quality)

    # -- Gate 2: pain is self-reported and always outranks the camera -------
    # Pain is the one signal the patient is the authority on. If it is high
    # enough, nothing the camera saw makes loaded work a good idea.
    pain_reasons = _pain_reasons(given, rules)
    reasons.extend(pain_reasons)

    if given.pain is not None and given.pain.value >= rules.pain_rest_only_threshold:
        reasons.extend(_swelling_reasons(given))
        return LockDecision(Decision.REST_ONLY, reasons, quality)

    # -- Gate 3: swelling, only where a comparison is actually possible -----
    swelling_reasons = _swelling_reasons(given)
    reasons.extend(swelling_reasons)
    swelling_worse = (
        given.swelling.status is SwellingComparisonStatus.AVAILABLE
        and given.swelling.report is SwellingReport.PUFFIER
    )

    # -- Gate 4: did the camera actually see anything? ----------------------
    # Reached only after the self-report gates, because a patient reporting
    # severe pain gets a rest day whether or not the camera was working.
    if given.compensation.status is CompensationStatus.UNOBSERVABLE:
        reasons.append(ReasonCode.MOVEMENT_OBSERVATION_INCOMPLETE)
        return LockDecision(Decision.HOLD, reasons, quality)

    metrics = given.compensation.metrics
    assert metrics is not None  # guaranteed by CompensationSummary's invariant

    compensating_often = metrics.flag_rate >= rules.compensation_flag_rate_lock
    reasons.append(
        ReasonCode.COMPENSATION_FREQUENT
        if compensating_often
        else ReasonCode.COMPENSATION_INFREQUENT
    )

    # -- Gate 5: early protocol days are conservative by default ------------
    early = given.protocol_day <= rules.early_protocol_days
    if early:
        reasons.append(ReasonCode.EARLY_PROTOCOL_DAY)

    pain_elevated = ReasonCode.PAIN_ELEVATED in pain_reasons

    if pain_elevated or swelling_worse or compensating_often or early:
        return LockDecision(Decision.LOCK_LOADED, reasons, quality)

    return LockDecision(Decision.ALLOW_FULL, reasons, quality)


# ---------------------------------------------------------------------------
# Reason helpers -- each returns the codes for one input, and nothing else
# ---------------------------------------------------------------------------


def _pain_reasons(given: PolicyInput, rules: PolicyRules) -> list[ReasonCode]:
    """No pain reading produces no pain reason.

    Deliberately not "assume it is fine". A missing pain score is reflected by
    the absence of a pain code and by input_quality.pain_present being False,
    so a reader of the stored decision can see it was never asked.
    """
    if given.pain is None:
        return []
    if given.pain.value >= rules.pain_lock_threshold:
        return [ReasonCode.PAIN_ELEVATED]
    return [ReasonCode.PAIN_WITHIN_RANGE]


def _swelling_reasons(given: PolicyInput) -> list[ReasonCode]:
    """One code per swelling status. Each of the four is a different fact.

    BASELINE_ONLY and NO_COMPARISON both mean "no comparison was made", but
    for different reasons and with different implications, so they get
    different codes rather than being folded together.
    """
    status = given.swelling.status

    if status is SwellingComparisonStatus.BASELINE_ONLY:
        return [ReasonCode.SWELLING_NO_BASELINE]

    if status in (
        SwellingComparisonStatus.NO_COMPARISON,
        SwellingComparisonStatus.UNAVAILABLE,
    ):
        return [ReasonCode.SWELLING_COMPARISON_UNAVAILABLE]

    return {
        SwellingReport.PUFFIER: [ReasonCode.SWELLING_INCREASED],
        SwellingReport.SAME: [ReasonCode.SWELLING_STABLE],
        SwellingReport.LESS: [ReasonCode.SWELLING_IMPROVED],
    }[given.swelling.report]
