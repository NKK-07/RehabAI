"""
Tests for explain/ (CP 8) and the fact-containment eval (CP 10 / T12).

Two separate concerns:

  1. The boundary holds -- the model is downstream of the decision, sees only
     the verdict, and never gets a chance to decide anything.
  2. The generated sentence restates the decision and nothing more.

The live-model eval is marked `llm` and skipped when Ollama is unreachable, so
the suite stays fast and green on a machine without it. Run it explicitly:

    pytest -m llm -v

Everything else here runs with no model at all, because the containment check
is a pure function over a string and a decision.
"""

import pytest

from rehab_ai.explain.phrasing import (
    FORBIDDEN_TERMS,
    REASON_FACTS,
    REASON_KEYWORDS,
    ExplainUnavailable,
    build_prompt,
    check_containment,
    health_check,
    phrase,
)
from rehab_ai.models.session import (
    CompensationStatus,
    Decision,
    InputQuality,
    LockDecision,
    ReasonCode,
    SessionStatus,
    SwellingComparisonStatus,
)
from rehab_ai.rules.loader import ExplainRules, PolicyRules, load_rules


@pytest.fixture(scope="module")
def rules():
    return load_rules()


def quality() -> InputQuality:
    return InputQuality(
        pain_present=True,
        swelling_status=SwellingComparisonStatus.AVAILABLE,
        compensation_status=CompensationStatus.AVAILABLE,
        session_status=SessionStatus.COMPLETED,
    )


def lock(*codes: ReasonCode, decision: Decision = Decision.LOCK_LOADED) -> LockDecision:
    return LockDecision(decision=decision, reason_codes=list(codes), input_quality=quality())


# --------------------------------------------------------------------------
# Completeness of the mapping tables
# --------------------------------------------------------------------------


def test_every_reason_code_has_a_plain_fact():
    """A code with no fact would reach the prompt as nothing, and the model
    would be asked to explain a decision it was not told the basis for."""
    for code in ReasonCode:
        assert code in REASON_FACTS, f"no plain fact for {code.value}"


def test_every_reason_code_has_containment_keywords():
    """Without keywords the eval would silently pass any sentence for that
    code -- a hole in the guard rather than a failure of it."""
    for code in ReasonCode:
        assert REASON_KEYWORDS.get(code), f"no containment keywords for {code.value}"


# --------------------------------------------------------------------------
# The boundary: the model sees the verdict, never the inputs
# --------------------------------------------------------------------------


def test_the_prompt_contains_the_decision_and_its_facts(rules):
    prompt = build_prompt(lock(ReasonCode.PAIN_ELEVATED, ReasonCode.SWELLING_INCREASED), rules.policy)
    assert REASON_FACTS[ReasonCode.PAIN_ELEVATED] in prompt
    assert REASON_FACTS[ReasonCode.SWELLING_INCREASED] in prompt
    assert "already been made" in prompt


def test_the_prompt_never_contains_raw_inputs_or_thresholds():
    """The model is handed the verdict, not the evidence. It cannot second-guess
    a decision whose inputs it was never shown.

    Uses sentinel threshold values rather than the real ones. A bare-digit
    substring check against real thresholds is unsound -- "5" appears in the
    word-limit instruction, so the test would fail on prose that has nothing to
    do with the guarantee.
    """
    sentinel = PolicyRules(
        pain_lock_threshold=7777,
        pain_rest_only_threshold=8888,
        compensation_flag_rate_lock=0.9999,
        early_protocol_days=6666,
        max_comparison_gap_days=5555,
        copy={"lock_loaded": "Squats are off today."},
    )
    prompt = build_prompt(lock(ReasonCode.PAIN_ELEVATED), sentinel)

    for secret in ("7777", "8888", "6666", "5555", "0.9999"):
        assert secret not in prompt, f"threshold {secret} leaked into the prompt"
    assert "flag_rate" not in prompt
    assert "threshold" not in prompt.lower()


def test_the_prompt_forbids_adding_reasons(rules):
    prompt = build_prompt(lock(ReasonCode.PAIN_ELEVATED), rules.policy)
    assert "Add NOTHING else" in prompt
    assert "No causes" in prompt
    assert "no diagnoses" in prompt


def test_the_prompt_states_how_many_facts_must_be_covered(rules):
    """Small models drop facts for brevity when handed a bare list. An explicit
    count gives them something to check themselves against -- and it is exactly
    what check_containment() measures."""
    prompt = build_prompt(
        lock(ReasonCode.PAIN_ELEVATED, ReasonCode.SWELLING_INCREASED), rules.policy
    )
    assert "2 facts" in prompt
    assert "cover all 2" in prompt

    single = build_prompt(lock(ReasonCode.PAIN_ELEVATED), rules.policy)
    assert "1 fact behind it" in single


def test_the_prompt_carries_only_the_codes_in_this_decision(rules):
    """A fact for a code that is not part of this decision must not leak in --
    that would be handing the model a reason to mention that policy did not
    actually give."""
    prompt = build_prompt(lock(ReasonCode.PAIN_ELEVATED), rules.policy)
    assert REASON_FACTS[ReasonCode.SWELLING_INCREASED] not in prompt


# --------------------------------------------------------------------------
# Containment -- coverage
# --------------------------------------------------------------------------


def test_a_sentence_covering_every_reason_passes():
    decision = lock(ReasonCode.PAIN_ELEVATED, ReasonCode.SWELLING_INCREASED)
    result = check_containment(
        "Squats are off today. Your pain was higher and the knee is puffier than yesterday.",
        decision,
    )
    assert result.ok, result.describe()


def test_a_sentence_that_drops_a_reason_fails():
    """The patient is entitled to the whole reason, not the half that was
    easiest to write."""
    decision = lock(ReasonCode.PAIN_ELEVATED, ReasonCode.SWELLING_INCREASED)
    result = check_containment("Squats are off today because your pain was higher.", decision)

    assert not result.ok
    assert ReasonCode.SWELLING_INCREASED in result.missing_codes


def test_missing_codes_are_named_in_the_description():
    decision = lock(ReasonCode.COMPENSATION_FREQUENT)
    result = check_containment("Squats are off today.", decision)
    assert "compensation_frequent" in result.describe()


# --------------------------------------------------------------------------
# Containment -- forbidden vocabulary
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("Your pain is higher; this may indicate an infection.", "infection"),
        ("Warning: pain elevated.", "warning"),
        ("Asymmetry detected in your pain response.", "asymmetr"),
        ("Your pain is up because you overdid it yesterday.", "overdid"),
        ("Pain is up. There is a risk of further damage.", "damage"),
    ],
)
def test_diagnostic_and_alarming_language_is_caught(sentence, expected):
    """This product flags a movement pattern. It does not diagnose, and PRD 8
    rules out alarming framing outright."""
    result = check_containment(sentence, lock(ReasonCode.PAIN_ELEVATED))
    assert not result.ok
    assert expected in result.forbidden_found


def test_the_forbidden_list_covers_the_phrasing_the_prd_names():
    """PRD 8 explicitly rejects 'asymmetry detected' and 'you are unsafe'."""
    assert "asymmetr" in FORBIDDEN_TERMS
    assert "unsafe" in FORBIDDEN_TERMS


def test_a_calm_specific_sentence_passes():
    """The tone PRD 8 asks for, and the copy bank salvaged from FEATURES.md."""
    result = check_containment(
        "Squats are off today. You marked the knee puffier than yesterday.",
        lock(ReasonCode.SWELLING_INCREASED),
    )
    assert result.ok


# --------------------------------------------------------------------------
# The guard's known limit, pinned so nobody mistakes it for a proof
# --------------------------------------------------------------------------


def test_containment_does_not_catch_a_plausible_invented_nuance():
    """Documented limitation, asserted so it stays visible.

    This sentence covers the reason code and uses no forbidden word, but adds
    a clinical implication nobody supplied. The eval passes it.

    That is why the model stays downstream of the decision: the guard reduces
    the risk, it does not remove it. TRD 9 records this.
    """
    result = check_containment(
        "Squats are off today because your pain is higher, which usually settles "
        "once the swelling from the operation has fully drained.",
        lock(ReasonCode.PAIN_ELEVATED),
    )
    assert result.ok  # <- the limitation, not an endorsement


# --------------------------------------------------------------------------
# No silent fallback
# --------------------------------------------------------------------------


def test_an_unreachable_model_raises_rather_than_returning_a_template(rules):
    """A canned sentence presented as model output would be a fabricated result
    arriving through a side door."""
    unreachable = ExplainRules(
        provider="ollama",
        endpoint="http://127.0.0.1:1",  # nothing listens here
        model="whatever",
        timeout_s=1,
        max_tokens=40,
    )
    with pytest.raises(ExplainUnavailable, match="cannot reach"):
        health_check(unreachable)

    with pytest.raises(ExplainUnavailable):
        phrase(lock(ReasonCode.PAIN_ELEVATED), unreachable, rules.policy)


def test_the_unreachable_error_tells_you_how_to_fix_it(rules):
    unreachable = ExplainRules("ollama", "http://127.0.0.1:1", "m", 1, 40)
    with pytest.raises(ExplainUnavailable, match="ollama serve"):
        health_check(unreachable)


def test_the_error_states_that_it_will_not_substitute_text(rules):
    unreachable = ExplainRules("ollama", "http://127.0.0.1:1", "m", 1, 40)
    with pytest.raises(ExplainUnavailable, match="does not substitute canned text"):
        health_check(unreachable)


# --------------------------------------------------------------------------
# Live model -- the actual eval
# --------------------------------------------------------------------------


def _model_available(rules) -> bool:
    try:
        health_check(rules.explain)
        return True
    except ExplainUnavailable:
        return False


@pytest.mark.llm
@pytest.mark.parametrize(
    "codes,decision",
    [
        ([ReasonCode.PAIN_ELEVATED, ReasonCode.SWELLING_INCREASED], Decision.LOCK_LOADED),
        ([ReasonCode.SWELLING_STABLE, ReasonCode.COMPENSATION_FREQUENT], Decision.LOCK_LOADED),
        ([ReasonCode.MOVEMENT_OBSERVATION_INCOMPLETE], Decision.HOLD),
        ([ReasonCode.SESSION_INCOMPLETE], Decision.HOLD),
        ([ReasonCode.PAIN_WITHIN_RANGE, ReasonCode.SWELLING_IMPROVED,
          ReasonCode.COMPENSATION_INFREQUENT], Decision.ALLOW_FULL),
        ([ReasonCode.SWELLING_NO_BASELINE, ReasonCode.PAIN_WITHIN_RANGE,
          ReasonCode.COMPENSATION_INFREQUENT], Decision.ALLOW_FULL),
    ],
    ids=["pain+swelling", "compensation", "unobservable", "abandoned", "clean", "first-session"],
)
def test_the_live_model_restates_the_decision_and_nothing_more(rules, codes, decision):
    """The CP 8 eval. Every reason code reflected, no forbidden language.

    Catches drift when the model changes -- TRD 3 leaves gemma2 vs phi3 open,
    and phrasing behaviour differs between them.
    """
    if not _model_available(rules):
        pytest.skip("no local model available")

    verdict = lock(*codes, decision=decision)
    sentence = phrase(verdict, rules.explain, rules.policy)

    result = check_containment(sentence, verdict)
    assert result.ok, f"{result.describe()}\n  generated: {sentence!r}"


@pytest.mark.llm
def test_the_live_model_stays_short(rules):
    """A patient reads one screen. PRD 8 wants calm and specific, not a essay."""
    if not _model_available(rules):
        pytest.skip("no local model available")

    sentence = phrase(lock(ReasonCode.PAIN_ELEVATED), rules.explain, rules.policy)
    assert len(sentence.split()) <= 60, f"too long: {sentence!r}"
