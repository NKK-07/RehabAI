"""
capture.py
Frame source, and the one rule that keeps left and right honest.

THE CAPTURE CONTRACT
====================
The frame handed to the pose model is NEVER mirrored. Flip for display only.

Webcam previews are conventionally mirrored, because seeing an unmirrored
version of yourself feels wrong. But the pose model labels landmarks
*anatomically* -- LEFT_KNEE means the person's left knee -- and it infers that
from the image. Hand it a mirrored frame and every label flips:

    raw frame  ──▶ pose model ──▶ LEFT_KNEE  == patient's left knee   ✓
    mirrored   ──▶ pose model ──▶ LEFT_KNEE  == patient's RIGHT knee  ✗

The second case is silent. Nothing raises, nothing looks wrong on screen, and
the operated-side binding -- the safeguard that exists specifically to make
sure we measure the correct leg -- now guarantees the wrong one.

So the rule is structural: this module owns the only cv2.flip call in the
codebase, it applies it after inference, and the rules file refuses to load
with mirror_before_inference set to true.

    ┌──────────┐   raw    ┌────────────┐  keypoints  ┌──────────┐
    │  camera  │─────────▶│ pose model │────────────▶│ detector │
    └──────────┘     │    └────────────┘             └──────────┘
                     │
                     └──▶ flip ──▶ display   (cosmetic only, never fed back)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from rehab_ai.rules.loader import CaptureRules


class CameraError(Exception):
    """Raised when the camera cannot be opened or read."""


@dataclass(frozen=True)
class Frame:
    """One captured frame, in both the forms the app needs.

    `for_inference` is always the raw, unmirrored image. `for_display` may be
    mirrored. They are separate attributes rather than a flag on one array so
    that passing the wrong one to the pose model is a visible mistake in the
    calling code rather than a boolean somebody flipped.
    """

    for_inference: np.ndarray  # BGR, never mirrored
    for_display: np.ndarray  # BGR, possibly mirrored
    timestamp: float
    width: int
    height: int

    @property
    def rgb_for_inference(self) -> np.ndarray:
        """MediaPipe expects RGB. Conversion happens here so no caller is
        tempted to reach for the raw BGR array and convert it themselves."""
        return cv2.cvtColor(self.for_inference, cv2.COLOR_BGR2RGB)


class CameraSource:
    """Wraps a cv2.VideoCapture and enforces the capture contract.

    Usage:
        with CameraSource(rules.capture) as cam:
            for frame in cam.frames():
                ...
    """

    def __init__(self, rules: CaptureRules, device_index: int = 0) -> None:
        if rules.mirror_before_inference:
            # Belt and braces: the rules loader already refuses this, but a
            # caller constructing CaptureRules directly must not slip past.
            raise ValueError(
                "mirror_before_inference must be False. Mirroring the frame before "
                "inference swaps the pose model's LEFT/RIGHT labels."
            )
        self._rules = rules
        self._device_index = device_index
        self._capture: cv2.VideoCapture | None = None

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> "CameraSource":
        capture = cv2.VideoCapture(self._device_index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"could not open camera at index {self._device_index}. "
                "Check that no other application is holding it."
            )
        capture.set(cv2.CAP_PROP_FPS, self._rules.target_fps)
        self._capture = capture
        return self

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "CameraSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reading ------------------------------------------------------------

    def read(self, timestamp: float) -> Frame:
        """Read one frame. Raises CameraError if the device stops responding."""
        if self._capture is None:
            raise CameraError("camera is not open; call open() first")

        ok, raw = self._capture.read()
        if not ok or raw is None:
            raise CameraError("camera returned no frame")

        return self.build_frame(raw, timestamp)

    def build_frame(self, raw: np.ndarray, timestamp: float) -> Frame:
        """Turn a raw BGR array into a Frame.

        Separated from read() so tests can exercise the contract with a
        synthetic array and no hardware.
        """
        height, width = raw.shape[:2]
        display = cv2.flip(raw, 1) if self._rules.mirror_display_only else raw

        return Frame(
            for_inference=raw,  # never flipped -- this is the contract
            for_display=display,
            timestamp=timestamp,
            width=width,
            height=height,
        )
