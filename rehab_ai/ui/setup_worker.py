"""
setup_worker.py
Feeds the setup screen. Same threading rule as the session worker.

    Qt UI thread                     setup worker thread
    ────────────                     ───────────────────
    paint the frame          ◀───    capture
    paint the skeleton       signal  pose
    handle keys                      evaluate the four steps

The worker emits a finished QImage plus the SetupState for that frame, so the
view paints and computes nothing. Stale frames are dropped rather than queued,
for the same reason as the session loop: a queue would show the patient their
own past, and the whole question here is "did that marker move when I moved".
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QImage

from rehab_ai.camera.capture import CameraError, CameraSource
from rehab_ai.camera.renderer import to_qimage_bytes
from rehab_ai.models.session import Side
from rehab_ai.pose.setup_check import SetupChecker
from rehab_ai.pose_utils import PoseTracker
from rehab_ai.rules.loader import Rules


class SetupWorker(QObject):
    frame_ready = Signal(QImage, object)  # QImage, SetupState
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, operated: Side, rules: Rules, device_index: int = 0) -> None:
        super().__init__()
        self._operated = operated
        self._rules = rules
        self._device_index = device_index
        self._running = False
        self._checker = SetupChecker(operated)

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        pose = PoseTracker(model_complexity=self._rules.capture.pose_model_complexity)
        try:
            with CameraSource(self._rules.capture, self._device_index) as camera:
                while self._running:
                    frame = camera.read(time.perf_counter())
                    result = pose.process(frame.rgb_for_inference)

                    # Landmarks come from the unmirrored inference frame, so the
                    # frame handed to the view must be the same one -- otherwise
                    # every marker lands at a flipped x and the overlay looks
                    # broken even when the binding is correct.
                    state = self._checker.update(
                        result.pose_landmarks, frame.width, frame.height
                    )

                    data, w, h, stride = to_qimage_bytes(frame.for_inference)
                    image = QImage(data, w, h, stride, QImage.Format.Format_RGB888).copy()
                    self.frame_ready.emit(image, state)
        except CameraError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"setup check stopped: {exc}")
        finally:
            pose.close()
            self.stopped.emit()


class SetupController:
    """Owns the thread so views never touch QThread directly."""

    def __init__(self, operated: Side, rules: Rules, device_index: int = 0) -> None:
        self.thread = QThread()
        self.worker = SetupWorker(operated, rules, device_index)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.stopped.connect(self.thread.quit)

    def start(self) -> None:
        self.thread.start()

    def stop(self, wait_ms: int = 3000) -> None:
        self.worker.stop()
        self.thread.quit()
        self.thread.wait(wait_ms)
