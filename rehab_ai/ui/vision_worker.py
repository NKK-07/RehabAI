"""
vision_worker.py
Runs capture, pose estimation and detection off the Qt thread.

THE THREADING RULE
==================
    Qt UI thread                        vision worker thread
    ────────────                        ────────────────────
    render the latest frame      ◀───   capture
    update the meter             signals pose estimation
    handle interaction                  detector state
                                        composite the meter

The worker publishes the latest available state; the UI renders it. Nothing on
the UI thread waits for a frame, and nothing in the frame loop waits for a
repaint.

FRESHNESS OVER COMPLETENESS
===========================
If the UI falls behind, stale frames are DROPPED rather than queued. A growing
frame queue would show the patient their own past -- which for a product whose
entire claim is "corrected while you are still standing up" would be a
contradiction rather than a lag.

Qt's queued signal delivery does not itself drop, so the worker keeps only the
newest rendered frame in a slot the UI reads, and emits a lightweight
notification rather than the image.

AUDIO
=====
The cue is played from the worker thread, not the UI thread, and CuePlayer.play
returns immediately. Neither thread ever blocks on audio.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from rehab_ai.audio.player import CuePlayer
from rehab_ai.camera.capture import CameraError, CameraSource
from rehab_ai.camera.renderer import draw_meter
from rehab_ai.detection.sit_to_stand import SitToStandDetector
from rehab_ai.models.session import ObservationQuality, RepPhase, RepResult, Side
from rehab_ai.pose.tracker import OperatedSideTracker
from rehab_ai.rules.loader import Rules


class VisionWorker(QObject):
    """Owns the camera, the pose model and the detector for one session."""

    frame_ready = Signal()  # "a new frame exists" -- the image is read from latest_frame
    rep_completed = Signal(object)  # RepResult
    cue_fired = Signal(str)  # the phrase, for an on-screen echo
    failed = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        operated_side: Side,
        rules: Rules,
        cue_player: CuePlayer,
        device_index: int = 0,
    ) -> None:
        super().__init__()
        self._side = operated_side
        self._rules = rules
        self._player = cue_player
        self._device_index = device_index

        self._running = False
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None

        self.detector = SitToStandDetector(operated_side, rules)
        self.reps: list[RepResult] = []
        self.frames_seen = 0
        self.started_at = 0.0

    # -- the UI reads this, never a queue ----------------------------------

    @property
    def latest_frame(self) -> np.ndarray | None:
        """The newest composited frame, or None.

        Deliberately a single slot. Anything older has been overwritten, which
        is the drop-stale-frames policy expressed as a data structure rather
        than as a queue-length check somebody has to remember to write.
        """
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    @property
    def measured_fps(self) -> float:
        elapsed = time.perf_counter() - self.started_at
        return self.frames_seen / elapsed if elapsed > 0 else 0.0

    # -- lifecycle ----------------------------------------------------------

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        """Entry point on the worker thread."""
        self._running = True
        self.started_at = time.perf_counter()

        try:
            self._loop()
        except CameraError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - the UI must hear about anything
            self.failed.emit(f"vision worker stopped: {exc}")
        finally:
            self.stopped.emit()

    def _loop(self) -> None:
        tracker = OperatedSideTracker(
            self._side,
            self._rules.observation,
            model_complexity=self._rules.capture.pose_model_complexity,
        )
        try:
            with CameraSource(self._rules.capture, self._device_index) as camera:
                while self._running:
                    self._tick(camera, tracker)
        finally:
            tracker.close()

    def _tick(self, camera: CameraSource, tracker: OperatedSideTracker) -> None:
        now = time.perf_counter()
        frame = camera.read(now)
        self.frames_seen += 1

        observation = tracker.observe(
            frame.rgb_for_inference, frame.width, frame.height, now
        )
        update = self.detector.update(observation)

        if update.cue is not None:
            # Returns immediately; playback happens on its own thread. The
            # frame loop does not wait, which is the whole point of CP 6.
            self._player.play(update.cue.key)
            self.cue_fired.emit(update.cue.text)

        if update.completed_rep is not None:
            self.reps.append(update.completed_rep)
            self.rep_completed.emit(update.completed_rep)

        composited = draw_meter(
            frame.for_display.copy(),
            signal=update.meter_signal,
            phase=update.phase,
            quality=update.quality,
            reps_completed=self.detector.reps_completed,
            trigger_threshold=self._rules.strategy.trigger_threshold,
        )

        with self._lock:
            self._latest = composited

        self.frame_ready.emit()


class VisionController:
    """Starts and stops the worker on its own QThread.

    Exists so the views never touch QThread directly -- starting a thread from
    a button handler is how a stop() ends up racing a running loop.
    """

    def __init__(
        self,
        operated_side: Side,
        rules: Rules,
        cue_player: CuePlayer,
        device_index: int = 0,
    ) -> None:
        self.thread = QThread()
        self.worker = VisionWorker(operated_side, rules, cue_player, device_index)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.stopped.connect(self.thread.quit)

    def start(self) -> None:
        self.thread.start()

    def stop(self, wait_ms: int = 3000) -> None:
        self.worker.stop()
        self.thread.quit()
        self.thread.wait(wait_ms)
