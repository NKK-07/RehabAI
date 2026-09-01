"""
phrasing.py
Turns a finished LockDecision into a sentence a patient can read.

THE BOUNDARY
============
This module receives a decision that has ALREADY been made, deterministically,
by policy/. It phrases it. It cannot:

    * change the decision
    * add a reason that is not in reason_codes
    * decide anything at all

    policy/  ──▶  LockDecision  ──▶  phrasing  ──▶  a sentence
    (decides)     (facts + codes)    (this file)     (words only)
         │                                 │
         └── deterministic ────────────────┴── generative
                                               and downstream of the decision

That ordering is the product's central safety claim, and it is worth being
precise about why it holds: the model never sees the inputs, only the verdict.
It has nothing to decide with.

NO SILENT FALLBACK
==================
If the model is unavailable, this raises. It does NOT quietly substitute a
template. A canned sentence presented as model output would be a fabricated
result -- exactly what the project's no-faking rule forbids, arriving through
a side door. The app refuses to start instead, loudly, with instructions.

THE GUARD, AND ITS LIMIT
========================
check_containment() asserts every reason code is reflected in the sentence and
that no diagnostic or alarming language appears. It verifies coverage and
vocabulary. It does NOT prove the absence of a misleading clinical
implication -- a well-phrased wrong nuance can pass. That limit is known,
stated in TRD 9, and is the reason the model is kept downstream of the
decision rather than inside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from rehab_ai.models.session import Decision, LockDecision, ReasonCode
from rehab_ai.rules.loader import ExplainRules, PolicyRules


class ExplainUnavailable(Exception):
    """The local model cannot be reached, or is not the one we asked for.

    Raised at startup by health_check(). Deliberately fatal: see the module
    docstring on why there is no template fallback.
    """


# ---------------------------------------------------------------------------
# Reason codes -> the plain facts the model is allowed to restate
# ---------------------------------------------------------------------------

REASON_FACTS: dict[ReasonCode, str] = {
    ReasonCode.PAIN_ELEVATED: "the pain score today is higher than the level we allow loaded exercise at",
    ReasonCode.PAIN_WITHIN_RANGE: "the pain score today is within the normal range",
    ReasonCode.SWELLING_INCREASED: "the knee was reported as puffier than yesterday",
    ReasonCode.SWELLING_STABLE: "the knee was reported as about the same as yesterday",
    ReasonCode.SWELLING_IMPROVED: "the knee was reported as less puffy than yesterday",
    ReasonCode.SWELLING_NO_BASELINE: "this is the first session, so there is no previous day to compare swelling against",
    ReasonCode.SWELLING_COMPARISON_UNAVAILABLE: "there is no recent session to compare swelling against",
    ReasonCode.COMPENSATION_FREQUENT: "on most of the stands the camera watched, the hips were doing the work instead of the knee",
    ReasonCode.COMPENSATION_INFREQUENT: "on most of the stands the camera watched, the knee was doing the work",
    ReasonCode.MOVEMENT_OBSERVATION_INCOMPLETE: "the camera could not see the movement clearly enough to assess it",
    ReasonCode.SESSION_INCOMPLETE: "the session was not finished",
    ReasonCode.EARLY_PROTOCOL_DAY: "it is still early after the operation",
}

# Words that must appear for a reason code to count as reflected. Any one is
# enough -- this checks the fact was mentioned, not how it was worded.
REASON_KEYWORDS: dict[ReasonCode, tuple[str, ...]] = {
    ReasonCode.PAIN_ELEVATED: ("pain", "sore"),
    ReasonCode.PAIN_WITHIN_RANGE: ("pain", "comfortable", "sore"),
    ReasonCode.SWELLING_INCREASED: ("swell", "puffier", "puffy"),
    ReasonCode.SWELLING_STABLE: ("swell", "puffy", "same"),
    ReasonCode.SWELLING_IMPROVED: ("swell", "puffy", "less"),
    ReasonCode.SWELLING_NO_BASELINE: ("first", "compare", "yesterday", "swell"),
    ReasonCode.SWELLING_COMPARISON_UNAVAILABLE: ("compare", "yesterday", "recent", "swell"),
    ReasonCode.COMPENSATION_FREQUENT: ("hip", "knee", "stand"),
    ReasonCode.COMPENSATION_INFREQUENT: ("knee", "stand", "well"),
    ReasonCode.MOVEMENT_OBSERVATION_INCOMPLETE: ("see", "saw", "camera", "view", "clear"),
    ReasonCode.SESSION_INCOMPLETE: ("finish", "complete", "session", "stop"),
    ReasonCode.EARLY_PROTOCOL_DAY: ("early", "operation", "surgery", "soon"),
}

# Vocabulary a patient-facing sentence must never contain. Two groups:
# clinical claims this product explicitly does not make, and alarming framing
# PRD 8 rules out.
FORBIDDEN_TERMS: tuple[str, ...] = (
    # diagnostic claims -- this is not a diagnostic device
    "diagnos", "infection", "infected", "tear", "torn", "rupture", "damage",
    "injury", "injured", "complication", "clot", "thrombo", "abnormal",
    # alarming framing
    "warning", "danger", "alarm", "severe", "serious", "urgent", "emergency",
    "risk of", "unsafe", "worrying", "concerning",
    # the specific phrasing PRD 8 names
    "asymmetr",
    # invented causal attribution about the patient's behaviour
    "overdid", "too much yesterday", "you did not do", "you failed",
)


@dataclass(frozen=True)
class ContainmentResult:
    """Outcome of checking a generated sentence against its decision."""

    ok: bool
    missing_codes: list[ReasonCode]
    forbidden_found: list[str]

    def describe(self) -> str:
        parts = []
        if self.missing_codes:
            parts.append(
                "reason codes not reflected: "
                + ", ".join(c.value for c in self.missing_codes)
            )
        if self.forbidden_found:
            parts.append("forbidden terms present: " + ", ".join(self.forbidden_found))
        return "; ".join(parts) or "ok"


def check_containment(sentence: str, decision: LockDecision) -> ContainmentResult:
    """Assert the sentence restates the decision and nothing more.

    Coverage plus vocabulary. Not a proof of clinical accuracy -- see the
    module docstring.
    """
    text = sentence.lower()

    missing = [
        code
        for code in decision.reason_codes
        if not any(word in text for word in REASON_KEYWORDS.get(code, ()))
    ]
    forbidden = [term for term in FORBIDDEN_TERMS if term in text]

    return ContainmentResult(
        ok=not missing and not forbidden,
        missing_codes=missing,
        forbidden_found=forbidden,
    )


# ---------------------------------------------------------------------------
# The local model
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You rewrite a rehabilitation app's decision as one or two short sentences "
    "for a patient recovering from knee surgery.\n"
    "RULES:\n"
    "- State only the facts you are given. Do not add reasons, causes, advice "
    "or diagnoses that are not listed.\n"
    "- Be calm and specific. Never alarming. Do not use words like warning, "
    "risk, danger, abnormal or asymmetry.\n"
    "- Do not tell the patient what caused their symptoms.\n"
    "- Plain English, second person, under 40 words. No preamble, no bullet "
    "points, no quotation marks. Output the sentences only."
)


def build_prompt(decision: LockDecision, policy_rules: PolicyRules) -> str:
    """The model sees the verdict and the facts behind it -- never the raw
    inputs, and never the thresholds. It has nothing to decide with."""
    headline = policy_rules.copy.get(decision.decision.value, "")
    facts = "\n".join(f"- {REASON_FACTS[code]}" for code in decision.reason_codes)

    return (
        f"{_SYSTEM}\n\n"
        f"The decision has already been made. It is: {headline}\n\n"
        f"The facts behind it:\n{facts}\n\n"
        "Rewrite this for the patient."
    )


def health_check(rules: ExplainRules) -> str:
    """Verify the model is reachable and present. Raises otherwise.

    Called at startup, before the first screen. The app must not reach the
    summary screen and only then discover it cannot explain the decision it
    just made.
    """
    try:
        response = requests.get(f"{rules.endpoint}/api/tags", timeout=rules.timeout_s)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ExplainUnavailable(
            f"cannot reach the local model at {rules.endpoint}.\n"
            f"  {exc}\n\n"
            "Start it with:  ollama serve\n"
            "The app will not run without it -- it does not substitute canned text "
            "for model output."
        ) from exc

    available = [m.get("name", "") for m in response.json().get("models", [])]
    if rules.model not in available:
        raise ExplainUnavailable(
            f"model {rules.model!r} is not installed.\n"
            f"  available: {', '.join(available) or '(none)'}\n\n"
            f"Install it with:  ollama pull {rules.model}\n"
            "Or set explain.model in rules/thresholds.v1.json to one you have."
        )

    return rules.model


def phrase(decision: LockDecision, rules: ExplainRules, policy_rules: PolicyRules) -> str:
    """Generate the patient-facing sentence.

    Raises ExplainUnavailable rather than returning a fallback. The caller
    decides what to show; this module will not pretend.
    """
    prompt = build_prompt(decision, policy_rules)

    try:
        response = requests.post(
            f"{rules.endpoint}/api/generate",
            json={
                "model": rules.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": rules.max_tokens, "temperature": 0.2},
            },
            timeout=rules.timeout_s,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ExplainUnavailable(f"the local model did not respond: {exc}") from exc

    text = (response.json().get("response") or "").strip()
    if not text:
        raise ExplainUnavailable("the local model returned an empty response")

    return _tidy(text)


def _tidy(text: str) -> str:
    """Strip the wrappers small models habitually add.

    Cosmetic only -- this must never alter meaning, so it removes quoting and
    leading labels and nothing else.
    """
    text = text.strip().strip('"').strip()
    for label in ("Patient:", "Answer:", "Response:", "Output:"):
        if text.startswith(label):
            text = text[len(label) :].strip()
    return " ".join(text.split())
