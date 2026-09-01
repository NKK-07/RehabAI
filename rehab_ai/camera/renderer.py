"""
renderer.py
Draws the strategy meter onto the camera frame.

This is the seam between the two build lanes. Lane A owns compositing (here);
Lane B owns display (ui/live_view.py). The boundary is one finished frame
crossing on one signal -- if either side reaches across it, the merge conflict
lands in the file you can least afford one in.

Compositing happens in OpenCV rather than Qt for a reason: it runs on the
vision worker thread, at frame rate, alongside pose estimation. Painting the
meter with QPainter would drag per-frame work onto the UI thread, which is
exactly what the threading model exists to prevent.

    ┌──────────────────────────────────────────┐
    │                                          │
    │              camera frame                │
    │                                          │
    │   ┌──────────────────────────────────┐   │
    │   │ KNEE ███████████░░░░░░░░░░░  HIP │   │  <- strategy meter
    │   └──────────────────────────────────┘   │
    │              Rep 4    knee-driven        │
    └──────────────────────────────────────────┘

The meter shows the smoothed (longer-window) signal, not the trigger signal.
A human eye wants steadiness; the cue wants speed. They are different consumers
of the same measurement and they get different windows.
"""

from __future__ import annotations

import cv2
import numpy as np

from rehab_ai.models.session import ObservationQuality, RepPhase

# BGR. Green reads as "the quad is working", amber as "the hips have taken over".
_KNEE_COLOUR = (110, 180, 90)
_HIP_COLOUR = (60, 130, 235)
_TRACK_COLOUR = (60, 60, 60)
_TEXT_COLOUR = (245, 245, 245)
_DIM_COLOUR = (160, 160, 160)
_WARN_COLOUR = (60, 130, 235)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _blend(colour_a, colour_b, t: float):
    return tuple(int(a + (b - a) * t) for a, b in zip(colour_a, colour_b))


def draw_meter(
    frame: np.ndarray,
    *,
    signal: float,
    phase: RepPhase,
    quality: ObservationQuality,
    reps_completed: int,
    trigger_threshold: float,
) -> np.ndarray:
    """Composite the meter and status onto a frame, in place.

    Returns the same array for convenience. Called once per frame on the vision
    worker thread, so it stays cheap: a handful of rectangles and two strings.
    """
    height, width = frame.shape[:2]

    bar_h = max(18, height // 22)
    margin = max(16, width // 40)
    bar_w = width - 2 * margin
    top = height - bar_h - margin - 34

    # track
    cv2.rectangle(frame, (margin, top), (margin + bar_w, top + bar_h), _TRACK_COLOUR, -1)

    # fill, coloured by how hip-dominant the movement currently is
    fill_w = int(bar_w * max(0.0, min(1.0, signal)))
    if fill_w > 0:
        cv2.rectangle(
            frame,
            (margin, top),
            (margin + fill_w, top + bar_h),
            _blend(_KNEE_COLOUR, _HIP_COLOUR, signal),
            -1,
        )

    # the line the cue fires at -- visible so the patient can see the target
    trigger_x = margin + int(bar_w * trigger_threshold)
    cv2.line(frame, (trigger_x, top - 4), (trigger_x, top + bar_h + 4), _TEXT_COLOUR, 1)

    cv2.rectangle(frame, (margin, top), (margin + bar_w, top + bar_h), _DIM_COLOUR, 1)

    cv2.putText(frame, "KNEE", (margin, top - 8), _FONT, 0.45, _KNEE_COLOUR, 1, cv2.LINE_AA)
    hip_label_x = margin + bar_w - 30
    cv2.putText(frame, "HIP", (hip_label_x, top - 8), _FONT, 0.45, _HIP_COLOUR, 1, cv2.LINE_AA)

    status = _status_line(phase, quality, signal, trigger_threshold)
    colour = _WARN_COLOUR if signal >= trigger_threshold else _DIM_COLOUR
    cv2.putText(
        frame, status, (margin, top + bar_h + 26), _FONT, 0.6, colour, 1, cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Rep {reps_completed}",
        (width - margin - 70, top + bar_h + 26),
        _FONT,
        0.6,
        _TEXT_COLOUR,
        1,
        cv2.LINE_AA,
    )

    return frame


def _status_line(
    phase: RepPhase, quality: ObservationQuality, signal: float, trigger: float
) -> str:
    """Plain language, never alarming -- the same tone rule the copy bank uses.

    'Cannot see you' is a statement about the camera, not about the patient.
    """
    if quality is ObservationQuality.UNOBSERVABLE:
        return "Step back into view"
    if phase is RepPhase.LOW_VISIBILITY:
        return "Step back into view"
    if phase is RepPhase.ABANDONED:
        return "Sit down and try again"
    if phase is RepPhase.READY:
        return "Ready when you are"
    if phase is RepPhase.RISING:
        return "Driving through hips" if signal >= trigger else "Driving through knee"
    if phase is RepPhase.STANDING:
        return "Standing"
    if phase is RepPhase.DESCENDING:
        return "Lower slowly"
    return ""


def to_qimage_bytes(frame: np.ndarray) -> tuple[bytes, int, int, int]:
    """Convert a BGR frame to the RGB buffer QImage expects.

    Returns (data, width, height, bytes_per_line). Kept here rather than in the
    view so that the only place that knows about the frame's memory layout is
    the module that produced it.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width, channels = rgb.shape
    return rgb.tobytes(), width, height, width * channels
