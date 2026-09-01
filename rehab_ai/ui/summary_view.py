"""
summary_view.py
The decision, and what the model made of it.

The decision shown here came from policy/ -- a pure function. The sentence
underneath it came from a language model that was handed that finished decision
and asked to phrase it.

Both are shown, and they are labelled differently, because they have different
standing: the verdict and its reason codes are auditable, the sentence is
phrasing. Presenting the sentence alone would hide which half a clinician can
actually check.

If the model failed, this screen says so plainly rather than substituting a
template. A canned sentence displayed where model output belongs would be a
fabricated result on screen.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from rehab_ai.models.session import (
    CompensationSummary,
    Decision,
    LockDecision,
    RehabSession,
    ReasonCode,
)

_REASON_TEXT = {
    ReasonCode.PAIN_ELEVATED: "Pain was higher than usual",
    ReasonCode.PAIN_WITHIN_RANGE: "Pain was in the usual range",
    ReasonCode.SWELLING_INCREASED: "You marked the knee puffier",
    ReasonCode.SWELLING_STABLE: "Swelling was about the same",
    ReasonCode.SWELLING_IMPROVED: "Swelling was down",
    ReasonCode.SWELLING_NO_BASELINE: "First session — nothing to compare with yet",
    ReasonCode.SWELLING_COMPARISON_UNAVAILABLE: "No recent session to compare with",
    ReasonCode.COMPENSATION_FREQUENT: "Most stands went through your hips",
    ReasonCode.COMPENSATION_INFREQUENT: "Most stands went through your knee",
    ReasonCode.MOVEMENT_OBSERVATION_INCOMPLETE: "The camera couldn't see enough to score",
    ReasonCode.SESSION_INCOMPLETE: "The session wasn't finished",
    ReasonCode.EARLY_PROTOCOL_DAY: "It's still early after the operation",
}

_HEADLINE = {
    Decision.ALLOW_FULL: "Everything is on the plan tomorrow.",
    Decision.LOCK_LOADED: "Squats are off tomorrow.",
    Decision.REST_ONLY: "Tomorrow is a rest day.",
    Decision.HOLD: "We're keeping tomorrow the same.",
}


class SummaryView(QWidget):
    """End of session: what happened, what was decided, and why."""

    done = Signal()
    show_sheet = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 32, 28, 28)
        layout.setSpacing(16)

        self._headline = QLabel("")
        self._headline.setObjectName("title")
        self._headline.setWordWrap(True)
        layout.addWidget(self._headline)

        self._stands = QLabel("")
        self._stands.setObjectName("subtitle")
        self._stands.setWordWrap(True)
        layout.addWidget(self._stands)

        layout.addSpacing(6)

        self._sentence = QLabel("")
        self._sentence.setObjectName("card")
        self._sentence.setWordWrap(True)
        layout.addWidget(self._sentence)

        self._sentence_label = QLabel("")
        self._sentence_label.setObjectName("subtitle")
        self._sentence_label.setWordWrap(True)
        layout.addWidget(self._sentence_label)

        layout.addSpacing(6)

        self._reasons = QLabel("")
        self._reasons.setObjectName("card")
        self._reasons.setWordWrap(True)
        layout.addWidget(self._reasons)

        layout.addStretch(1)

        sheet = QPushButton("Recovery sheet")
        sheet.clicked.connect(self.show_sheet.emit)
        layout.addWidget(sheet)

        done = QPushButton("Done")
        done.setObjectName("primary")
        done.clicked.connect(self.done.emit)
        layout.addWidget(done)

    def present(
        self,
        session: RehabSession,
        decision: LockDecision,
        sentence: str | None,
        sentence_error: str | None = None,
    ) -> None:
        self._headline.setText(_HEADLINE.get(decision.decision, ""))
        self._stands.setText(self._describe_stands(session))

        if sentence:
            self._sentence.setText(sentence)
            self._sentence_label.setText("Written by the on-device language model.")
        else:
            # No template substitution. Say what went wrong.
            self._sentence.setText(
                "The plain-language summary isn't available — the on-device model "
                f"didn't respond.\n\n{sentence_error or ''}".strip()
            )
            self._sentence_label.setText(
                "The decision above and the reasons below are unaffected — they are "
                "produced without the model."
            )

        reasons = "\n".join(
            f"·  {_REASON_TEXT.get(code, code.value)}" for code in decision.reason_codes
        )
        self._reasons.setText(f"Why:\n{reasons}")

    def _describe_stands(self, session: RehabSession) -> str:
        """Report unobserved reps explicitly.

        A session summary that folds them into the total would present a number
        the camera did not earn.
        """
        summary = CompensationSummary.from_reps(session.reps)
        total = len(session.reps)

        if summary.metrics is None:
            return f"{total} stand{'s' if total != 1 else ''} — none clear enough to score."

        metrics = summary.metrics
        unseen = metrics.reps_total - metrics.reps_scored
        text = f"{metrics.reps_scored} of {metrics.reps_total} stands scored"
        text += f" · {metrics.reps_flagged} went through the hips"
        if unseen:
            text += f" · {unseen} not clear enough to score"
        return text
