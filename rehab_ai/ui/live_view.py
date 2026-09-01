"""
live_view.py
The camera screen. Renders what the vision worker produced; computes nothing.

This is Lane B's side of the seam. Lane A composites the meter in OpenCV on the
worker thread (camera/renderer.py); this view converts the finished frame to a
QImage and paints it. Nothing here reaches back across that line.

    worker thread                    UI thread (here)
    ─────────────                    ────────────────
    capture                          read latest_frame
    pose                    ──▶      convert to QImage
    detector                signal   paint
    composite meter                  update rep counter

If this class ever grows a call into the detector, the threading model has been
broken and the frame loop will start waiting on repaints.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from rehab_ai.camera.renderer import to_qimage_bytes
from rehab_ai.models.session import RepResult
from rehab_ai.ui.vision_worker import VisionController


class LiveView(QWidget):
    """Camera preview, rep counter, and the finish/abandon controls."""

    finished = Signal()  # session completed normally
    abandoned = Signal()  # patient quit part-way
    failed = Signal(str)

    def __init__(self, controller: VisionController) -> None:
        super().__init__()
        self._controller = controller
        self._worker = controller.worker

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 24, 20, 24)
        layout.setSpacing(14)

        self._headline = QLabel("Stand up when you're ready")
        self._headline.setObjectName("title")
        self._headline.setWordWrap(True)
        layout.addWidget(self._headline)

        self._preview = QLabel("Starting the camera...")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumHeight(460)
        self._preview.setObjectName("card")
        layout.addWidget(self._preview, 1)

        self._cue_echo = QLabel("")
        self._cue_echo.setObjectName("subtitle")
        self._cue_echo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._cue_echo)

        self._tally = QLabel("No stands yet")
        self._tally.setObjectName("subtitle")
        layout.addWidget(self._tally)

        self._finish = QPushButton("I'm done")
        self._finish.setObjectName("primary")
        self._finish.clicked.connect(self._on_finish)
        layout.addWidget(self._finish)

        quit_button = QPushButton("Stop for now")
        quit_button.clicked.connect(self._on_abandon)
        layout.addWidget(quit_button)

        self._worker.rep_completed.connect(self._on_rep)
        self._worker.cue_fired.connect(self._on_cue)
        self._worker.failed.connect(self.failed.emit)

        # Repaint on a timer rather than on every frame_ready signal. The
        # worker may run faster than the screen refreshes, and repainting more
        # often than the display can show buys nothing while competing with
        # the frame loop for CPU.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._repaint)
        self._timer.setInterval(33)  # ~30 Hz

        self._cue_clear = QTimer(self)
        self._cue_clear.setSingleShot(True)
        self._cue_clear.timeout.connect(lambda: self._cue_echo.setText(""))

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self._controller.start()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._controller.stop()

    # -- rendering ----------------------------------------------------------

    def _repaint(self) -> None:
        frame = self._worker.latest_frame
        if frame is None:
            return

        data, width, height, stride = to_qimage_bytes(frame)
        image = QImage(data, width, height, stride, QImage.Format.Format_RGB888)
        self._preview.setPixmap(
            QPixmap.fromImage(image).scaled(
                self._preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # -- worker signals -----------------------------------------------------

    def _on_rep(self, rep: RepResult) -> None:
        total = len(self._worker.reps)
        scored = [r for r in self._worker.reps if r.validity.counts_toward_compensation_rate]
        unseen = total - len(scored)

        text = f"{total} stand{'s' if total != 1 else ''}"
        if unseen:
            # Say it plainly. An unobserved rep that quietly counts as clean is
            # the failure the whole quality model exists to prevent, so it must
            # not be invisible on screen either.
            text += f" — {unseen} we couldn't see well enough to score"
        self._tally.setText(text)

    def _on_cue(self, phrase: str) -> None:
        """Echo the spoken cue on screen.

        The correction is audio first -- that is the product. This is a caption
        for a noisy room, not the delivery mechanism.
        """
        self._cue_echo.setText(phrase)
        self._cue_clear.start(2500)

    # -- controls -----------------------------------------------------------

    def _on_finish(self) -> None:
        self.stop()
        self.finished.emit()

    def _on_abandon(self) -> None:
        self.stop()
        self.abandoned.emit()
