"""
check_orientation.py
The CP 3 gate: prove the app is watching the leg you think it is.

    python scripts/check_orientation.py --side left

TWO PHASES
==========
    1. How-to   an animated loop showing the four steps, before the camera
    2. Live     full-bleed camera, steps ticking themselves off

WHAT CHANGED, AND WHY IT MATTERS
================================
The first version of this asked you to raise an arm and then eyeball whether
the right marker moved. But the app tracked only shoulder, hip, knee and ankle
-- no wrist -- so it had no idea whether your arm went up. You were the
instrument being used to test the instrument.

With the wrist tracked, the app checks its own binding:

    operated-side wrist rises   -> PASSED
    other-side wrist rises      -> SIDES SWAPPED, stop
    neither                     -> still waiting

Press  D  during the live phase for per-joint visibility numbers. Off by
default -- those are for tuning thresholds, and a patient reading 0.52 beside
their ankle learns nothing useful.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from rehab_ai.models.session import Side  # noqa: E402
from rehab_ai.rules.loader import load_rules  # noqa: E402
from rehab_ai.ui.setup_view import HowToPanel, LiveCheckView  # noqa: E402
from rehab_ai.ui.setup_worker import SetupController  # noqa: E402
from rehab_ai.ui.theme import STYLESHEET  # noqa: E402


class ResultPanel(QWidget):
    """The verdict, stated plainly, with what to do next."""

    def __init__(self, passed: bool, operated: Side, on_retry) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 40, 28, 28)
        layout.setSpacing(16)

        title = QLabel("CP 3 passed" if passed else "Stop — sides are swapped")
        title.setObjectName("title")
        title.setWordWrap(True)
        layout.addWidget(title)

        if passed:
            body = (
                f"The app tracked your <b>{operated.value}</b> side when you raised "
                f"that arm. The operated-side binding is correct, so every number "
                f"the session records describes the {operated.value} knee.<br><br>"
                "You can move on to the live session."
            )
        else:
            body = (
                f"You raised your <b>{operated.value}</b> arm, but the app saw the "
                "other side move. Left and right are attached to the wrong body.<br><br>"
                "<b>Do not proceed.</b> Every hip-drive number, every lock decision "
                "and every row in the recovery sheet would describe the healthy "
                "knee. Nothing in the test suite can catch this, because from "
                "inside the code the labels are self-consistent.<br><br>"
                "Check that the raw frame is not being mirrored before inference "
                "(<code>capture.mirror_before_inference</code> must be false)."
            )

        text = QLabel(body)
        text.setObjectName("card")
        text.setWordWrap(True)
        layout.addWidget(text)
        layout.addStretch(1)

        again = QPushButton("Run the check again")
        again.setObjectName("primary")
        again.clicked.connect(on_retry)
        layout.addWidget(again)


class CheckWindow(QMainWindow):
    def __init__(self, operated: Side, rules, device_index: int) -> None:
        super().__init__()
        self._operated = operated
        self._rules = rules
        self._device = device_index
        self._controller: SetupController | None = None

        self.setWindowTitle(f"RehabAI — camera check ({operated.value} knee)")
        self.resize(520, 880)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._howto = HowToPanel(operated)
        self._howto.ready.connect(self._start_live)
        self._stack.addWidget(self._howto)

    def _start_live(self) -> None:
        self._live = LiveCheckView(self._operated)
        self._live.passed.connect(lambda: self._finish(True))
        self._live.swapped.connect(lambda: self._finish(False))
        self._stack.addWidget(self._live)
        self._stack.setCurrentWidget(self._live)
        self._live.setFocus()

        self._controller = SetupController(self._operated, self._rules, self._device)
        self._controller.worker.frame_ready.connect(self._live.update_frame)
        self._controller.worker.failed.connect(self._on_failed)
        self._controller.start()

    def _on_failed(self, message: str) -> None:
        print(f"\n  camera error: {message}\n")
        self.close()

    def _finish(self, passed: bool) -> None:
        if self._controller is not None:
            self._controller.stop()
            self._controller = None

        result = ResultPanel(passed, self._operated, self._retry)
        self._stack.addWidget(result)
        self._stack.setCurrentWidget(result)

        print("\n  CP 3: PASSED\n" if passed else "\n  CP 3: FAILED - sides swapped\n")

    def _retry(self) -> None:
        self._stack.setCurrentWidget(self._howto)

    def closeEvent(self, event):  # noqa: N802
        if self._controller is not None:
            self._controller.stop()
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify operated-side tracking (CP 3)")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    operated = Side(args.side)
    rules = load_rules()

    print(f"\n  CP 3 — operated-side check   ({operated.value} knee)")
    print("  Follow the steps on screen. Press D during the live view for")
    print("  per-joint visibility numbers.\n")

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    window = CheckWindow(operated, rules, args.camera)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
