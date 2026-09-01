"""
main_window.py
The session flow, and the wiring between the layers.

    Intake  ──▶  LiveSession  ──▶  Summary  ──▶  Recovery sheet
       │                              ▲
       │         (abandoned) ─────────┘

One QStackedWidget, no extra OS windows. The window is deliberately narrow --
this is a product used seated, at arm's length, and it should read as a phone
even on a laptop.

WHERE THE SAFETY BOUNDARY SITS IN THIS FILE
===========================================
_finish_session() is the one place the whole architecture becomes visible:

    reps + check-in  ──▶  policy.decide()   deterministic, auditable
                              │
                          LockDecision
                              │
                              ├──▶ storage    persisted with reason codes
                              └──▶ explain()  phrasing only, cannot alter it

If the model fails, the decision and the reasons are still shown. Only the
prose is missing, and the screen says so rather than substituting a template.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from PySide6.QtWidgets import QMainWindow, QStackedWidget

from rehab_ai.audio.player import CuePlayer
from rehab_ai.explain.phrasing import ExplainUnavailable, phrase
from rehab_ai.models.session import (
    CompensationSummary,
    PainReport,
    PolicyInput,
    Profile,
    RehabSession,
    SessionStatus,
    Side,
    SwellingComparison,
    SwellingReport,
)
from rehab_ai.policy.engine import build_swelling_comparison, decide
from rehab_ai.rules.loader import Rules
from rehab_ai.storage.repository import SessionRepository
from rehab_ai.ui.intake_view import IntakeView
from rehab_ai.ui.live_view import LiveView
from rehab_ai.ui.sheet_view import SheetView
from rehab_ai.ui.summary_view import SummaryView
from rehab_ai.ui.theme import WINDOW_HEIGHT, WINDOW_WIDTH
from rehab_ai.ui.vision_worker import VisionController


class MainWindow(QMainWindow):
    def __init__(
        self,
        profile: Profile,
        rules: Rules,
        repository: SessionRepository,
        cue_player: CuePlayer,
        device_index: int = 0,
    ) -> None:
        super().__init__()
        self._profile = profile
        self._rules = rules
        self._repo = repository
        self._player = cue_player
        self._device_index = device_index

        self.setWindowTitle("RehabAI")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._session = RehabSession.for_profile(str(uuid.uuid4()), profile)
        self._controller: VisionController | None = None
        self._live: LiveView | None = None

        self._intake = IntakeView(profile.operated_side)
        self._intake.submitted.connect(self._on_intake)
        self._stack.addWidget(self._intake)

        self._summary = SummaryView()
        self._summary.done.connect(self.close)
        self._summary.show_sheet.connect(self._show_sheet)
        self._stack.addWidget(self._summary)

        self._sheet = SheetView()
        self._sheet.back.connect(lambda: self._stack.setCurrentWidget(self._summary))
        self._stack.addWidget(self._sheet)

    # -- intake -------------------------------------------------------------

    def _on_intake(
        self, pain: PainReport, swelling: SwellingReport, facing: Side
    ) -> None:
        self._session.pain = pain
        self._session.swelling = self._build_swelling_comparison(swelling)
        self._session.camera_facing_side = facing
        self._session.status = SessionStatus.ACTIVE
        self._session.started_at = datetime.now()

        # Persist before the camera starts. If the app dies mid-session the
        # check-in is not lost, and the row is there to be marked ABANDONED.
        self._repo.save(self._session)

        self._start_live()

    def _build_swelling_comparison(self, report: SwellingReport) -> SwellingComparison:
        """Fetch the prior session, then let policy/ decide what it means.

        This method does lookup only. The rule about what a calendar gap means
        lives in policy.build_swelling_comparison -- a caller-side guard is a
        secondary defence, not where business rules belong, or two screens
        could each decide it differently and the deterministic policy would
        depend on which one called it.
        """
        now = datetime.now()
        previous = self._repo.previous_session(now, self._profile.operated_side)
        previous_at = previous.started_at if previous else None

        return build_swelling_comparison(report, previous_at, now, self._rules.policy)

    # -- live session -------------------------------------------------------

    def _start_live(self) -> None:
        self._controller = VisionController(
            self._profile.operated_side, self._rules, self._player, self._device_index
        )
        self._live = LiveView(self._controller)
        self._live.finished.connect(lambda: self._finish_session(SessionStatus.COMPLETED))
        self._live.abandoned.connect(lambda: self._finish_session(SessionStatus.ABANDONED))
        self._live.failed.connect(self._on_vision_failed)

        self._stack.addWidget(self._live)
        self._stack.setCurrentWidget(self._live)
        self._live.start()

    def _on_vision_failed(self, message: str) -> None:
        """The camera stopped. Do not pretend the session happened."""
        self._finish_session(SessionStatus.ABANDONED, note=message)

    # -- the decision -------------------------------------------------------

    def _finish_session(self, status: SessionStatus, note: str | None = None) -> None:
        if self._controller is not None:
            self._session.reps = list(self._controller.worker.reps)
            self._controller.stop()

        self._session.status = status
        self._session.ended_at = datetime.now()

        assert self._session.swelling is not None
        decision = decide(
            PolicyInput(
                pain=self._session.pain,
                swelling=self._session.swelling,
                compensation=CompensationSummary.from_reps(self._session.reps),
                protocol_day=self._session.protocol_day,
                session_status=status,
            ),
            self._rules.policy,
        )

        self._repo.save(self._session, decision)

        sentence: str | None = None
        error: str | None = note
        try:
            sentence = phrase(decision, self._rules.explain, self._rules.policy)
        except ExplainUnavailable as exc:
            error = f"{note + '  ' if note else ''}{exc}"

        self._summary.present(self._session, decision, sentence, error)
        self._stack.setCurrentWidget(self._summary)

    # -- sheet --------------------------------------------------------------

    def _show_sheet(self) -> None:
        self._sheet.present(self._repo.recent(limit=60))
        self._stack.setCurrentWidget(self._sheet)

    # -- shutdown -----------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop the worker before the window goes.

        Also records an in-progress session as abandoned rather than leaving it
        ACTIVE forever -- policy/ has a defined outcome for an abandoned
        session, and none at all for one that never ended.
        """
        if self._controller is not None:
            self._controller.stop()
        if self._session.status is SessionStatus.ACTIVE:
            self._session.status = SessionStatus.ABANDONED
            self._session.ended_at = datetime.now()
            self._repo.save(self._session)
        super().closeEvent(event)
