"""
setup_view.py
The camera setup check, in two phases.

    PHASE 1  how-to          an animated loop showing the four steps
    PHASE 2  live check      full-bleed camera, steps ticking themselves off

WHY AN ANIMATION AND NOT A VIDEO
================================
A recorded video would need footage of a person, which under the no-faking rule
means a real person filmed for the purpose, shipped as a binary asset that
cannot be themed and looks wrong at half the window sizes it will meet.

The loop below is drawn with QPainter from the same palette as the rest of the
app. It illustrates a posture; it does not claim to be a patient. That
distinction is why it is allowed: it is instruction, not data.

WHY FULL-BLEED IN PHASE 2
=========================
The patient is at arm's length looking for whether a marker moved. Chrome
competing with the camera is chrome in the way. Instructions overlay the feed,
one at a time, and the checklist collapses to a row of dots.

    ┌───────────────────────────┐
    │   Raise your left arm     │   <- one instruction, large
    │        ● ● ● ○            │   <- steps, ticking
    │                           │
    │         [camera]          │
    │        with skeleton      │
    │                           │
    │   ┌─────────────────┐     │
    │   │ Watching LEFT   │     │   <- verdict, on screen
    │   └─────────────────┘     │
    └───────────────────────────┘

Press D for the telemetry panel -- per-joint visibility with thresholds, for
tuning. Off by default so it never reaches a patient.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from rehab_ai.models.session import Side
from rehab_ai.pose.setup_check import (
    STEP_ORDER,
    SetupState,
    SetupStep,
    SetupVerdict,
    bones_for,
)
from rehab_ai.ui.theme import ACCENT, GOOD, GROUND, INK, INK_SOFT, LOCKED, SURFACE

_MARKER = QColor(107, 227, 155)
_MARKER_DIM = QColor(123, 139, 144, 130)
_WARN = QColor(224, 164, 106)


# ===========================================================================
# PHASE 1 -- the animated how-to
# ===========================================================================


class HowToAnimation(QWidget):
    """A looping side-view figure demonstrating the four setup steps.

    One beat per step, four seconds each. The figure is drawn rather than
    filmed, so it inherits the app's palette and stays sharp at any size.
    """

    BEATS = (
        ("Sit side-on to the camera", "Turn so the camera sees your profile"),
        ("Operated leg nearest", "The camera can only watch the near leg"),
        ("Lift that heel, and hold", "Keep it up for about half a second"),
        ("We confirm it", "The ankle marker rises, and we know it's your leg"),
    )
    BEAT_MS = 3600

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(320)
        self._t = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(33)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    @property
    def beat(self) -> int:
        return int(self._t // self.BEAT_MS) % len(self.BEATS)

    @property
    def beat_progress(self) -> float:
        return (self._t % self.BEAT_MS) / self.BEAT_MS

    def _tick(self) -> None:
        self._t += 33
        self.update()

    # -- painting -----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(GROUND))

        w, h = self.width(), self.height()
        cx, cy = w * 0.5, h * 0.58
        scale = min(w / 340.0, h / 300.0)

        beat = self.beat
        progress = self.beat_progress

        self._draw_chair(painter, cx, cy, scale)

        # Beat 0 shows the figure rotating from face-on to profile; after that
        # it stays side-on.
        turn = 1.0 - min(1.0, progress * 2.2) if beat == 0 else 0.0
        # Beat 2 lifts the heel, beat 3 holds it up. The arm stays down --
        # the check no longer uses it, and showing an arm raise here would
        # teach the wrong movement.
        if beat == 2:
            lift = min(1.0, progress * 2.4)
        elif beat == 3:
            lift = 1.0
        else:
            lift = 0.0
        arm = 0.0
        highlight = beat >= 1
        pulse = beat == 3

        self._draw_figure(painter, cx, cy, scale, turn=turn, arm=arm, lift=lift,
                          highlight=highlight, pulse=pulse, progress=progress)
        self._draw_caption(painter, w, h, beat)
        self._draw_beat_dots(painter, w, h, beat)
        painter.end()

    def _draw_chair(self, p: QPainter, cx: float, cy: float, s: float) -> None:
        pen = QPen(QColor(INK_SOFT), 3 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        seat_y = cy + 18 * s
        p.drawLine(int(cx - 52 * s), int(seat_y), int(cx + 14 * s), int(seat_y))
        p.drawLine(int(cx - 46 * s), int(seat_y), int(cx - 46 * s), int(seat_y + 46 * s))
        p.drawLine(int(cx + 8 * s), int(seat_y), int(cx + 8 * s), int(seat_y + 46 * s))
        p.drawLine(int(cx - 52 * s), int(seat_y), int(cx - 52 * s), int(cy - 42 * s))
        # ground
        p.setPen(QPen(QColor(INK_SOFT), 2 * s))
        p.drawLine(int(cx - 90 * s), int(seat_y + 46 * s), int(cx + 90 * s), int(seat_y + 46 * s))

    def _draw_figure(
        self, p: QPainter, cx: float, cy: float, s: float, *,
        turn: float, arm: float, lift: float, highlight: bool, pulse: bool,
        progress: float,
    ) -> None:
        """Side-view seated figure. `turn` 1.0 = facing camera, 0.0 = profile."""
        # Facing the camera, the two sides separate horizontally; in profile
        # they collapse onto each other. That is the same cue the live check
        # uses to detect side-on, so the animation teaches the real signal.
        spread = 26 * s * turn

        hip = (cx - 18 * s, cy + 16 * s)
        knee = (cx + 22 * s, cy + 22 * s)
        # The heel lift: the ankle rises and the knee straightens slightly,
        # which is what the live check measures on the real ankle landmark.
        ankle = (cx + 26 * s, cy + 62 * s - 26 * s * lift)
        shoulder = (cx - 26 * s, cy - 34 * s)
        head = (cx - 30 * s, cy - 56 * s)

        elbow_rest = (cx - 14 * s, cy - 10 * s)
        wrist_rest = (cx + 2 * s, cy + 10 * s)
        elbow_up = (cx - 44 * s, cy - 58 * s)
        wrist_up = (cx - 34 * s, cy - 92 * s)
        elbow = _lerp(elbow_rest, elbow_up, arm)
        wrist = _lerp(wrist_rest, wrist_up, arm)

        # far side, dimmed
        far = QColor(_MARKER_DIM)
        p.setPen(QPen(far, 5 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for a, b in ((shoulder, hip), (hip, knee), (knee, ankle)):
            p.drawLine(int(a[0] + spread), int(a[1]), int(b[0] + spread), int(b[1]))
        for pt in (shoulder, hip, knee, ankle):
            p.setBrush(far)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(int(pt[0] + spread - 4 * s), int(pt[1] - 4 * s), int(8 * s), int(8 * s))

        # near side
        colour = QColor(_MARKER) if highlight else QColor(INK_SOFT)
        width = 6 * s
        if pulse:
            width += 1.6 * s * math.sin(progress * math.pi * 6)
        p.setPen(QPen(colour, max(2.0, width), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for a, b in ((head, shoulder), (shoulder, elbow), (elbow, wrist),
                     (shoulder, hip), (hip, knee), (knee, ankle)):
            p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))

        p.setPen(QPen(colour, 4 * s))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(int(head[0] - 13 * s), int(head[1] - 20 * s), int(26 * s), int(26 * s))

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(colour)
        for pt in (shoulder, elbow, wrist, hip, knee, ankle):
            r = 6 * s
            p.drawEllipse(int(pt[0] - r), int(pt[1] - r), int(r * 2), int(r * 2))

    def _draw_caption(self, p: QPainter, w: int, h: int, beat: int) -> None:
        title, sub = self.BEATS[beat]

        font = QFont(self.font())
        font.setPointSizeF(15)
        font.setWeight(QFont.Weight.DemiBold)
        p.setFont(font)
        p.setPen(QColor(INK))
        p.drawText(0, 18, w, 30, Qt.AlignmentFlag.AlignHCenter, title)

        font.setPointSizeF(10.5)
        font.setWeight(QFont.Weight.Normal)
        p.setFont(font)
        p.setPen(QColor(INK_SOFT))
        p.drawText(0, 46, w, 24, Qt.AlignmentFlag.AlignHCenter, sub)

    def _draw_beat_dots(self, p: QPainter, w: int, h: int, beat: int) -> None:
        n = len(self.BEATS)
        gap, r = 16, 4
        total = (n - 1) * gap
        x0 = w / 2 - total / 2
        y = h - 18
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            p.setBrush(QColor(ACCENT) if i == beat else QColor(INK_SOFT).lighter(150))
            size = r * 2 if i == beat else r * 1.4
            p.drawEllipse(int(x0 + i * gap - size / 2), int(y - size / 2), int(size), int(size))


def _lerp(a, b, t: float):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


class HowToPanel(QWidget):
    """Phase 1: the animation, plus the button that starts the live check."""

    ready = Signal()

    def __init__(self, operated: Side) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 28, 24, 24)
        layout.setSpacing(14)

        title = QLabel("Before we start")
        title.setObjectName("title")
        layout.addWidget(title)

        sub = QLabel(
            f"We'll check the camera can see your <b>{operated.value}</b> knee — the "
            "operated one. It takes about fifteen seconds."
        )
        sub.setObjectName("subtitle")
        sub.setWordWrap(True)
        layout.addWidget(sub)

        self.animation = HowToAnimation()
        layout.addWidget(self.animation, 1)

        start = QPushButton("I'm set up — start the check")
        start.setObjectName("primary")
        start.clicked.connect(self.ready.emit)
        layout.addWidget(start)

    def showEvent(self, event) -> None:  # noqa: N802
        self.animation.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self.animation.stop()
        super().hideEvent(event)


# ===========================================================================
# PHASE 2 -- the live check, full bleed
# ===========================================================================


class LiveCheckView(QWidget):
    """Camera fills the widget. Everything else overlays it."""

    passed = Signal()
    swapped = Signal()

    def __init__(self, operated: Side) -> None:
        super().__init__()
        self._operated = operated
        self._frame: QImage | None = None
        self._state: SetupState | None = None
        self._show_telemetry = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(420)

    # -- input from the worker ---------------------------------------------

    def update_frame(self, image: QImage, state: SetupState) -> None:
        self._frame = image
        previous_verdict = self._state.verdict if self._state else SetupVerdict.WAITING
        self._state = state
        self.update()

        if state.verdict is not previous_verdict:
            if state.verdict is SetupVerdict.PASSED:
                self.passed.emit()
            elif state.verdict is SetupVerdict.SIDES_SWAPPED:
                self.swapped.emit()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """D toggles the telemetry panel.

        Off by default: per-joint visibility numbers are for tuning, and a
        patient reading 0.52 next to their ankle learns nothing useful.
        """
        if event.key() == Qt.Key.Key_D:
            self._show_telemetry = not self._show_telemetry
            self.update()
        else:
            super().keyPressEvent(event)

    # -- painting -----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#12191C"))

        if self._frame is None:
            self._centred(p, "Starting the camera...", 13, QColor(INK_SOFT))
            p.end()
            return

        scaled = QPixmap.fromImage(self._frame).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        p.drawPixmap(
            int((self.width() - scaled.width()) / 2),
            int((self.height() - scaled.height()) / 2),
            scaled,
        )

        if self._state is not None:
            self._draw_skeleton(p, scaled)
            self._draw_instruction(p)
            self._draw_steps(p)
            self._draw_blocked_reason(p)
            self._draw_hold_ring(p)
            self._draw_verdict(p)
            if self._show_telemetry:
                self._draw_telemetry(p)
        p.end()

    def _map(self, point, scaled: QPixmap) -> tuple[float, float]:
        """Frame coordinates -> widget coordinates, matching the scaled pixmap."""
        assert self._frame is not None
        sx = scaled.width() / max(self._frame.width(), 1)
        sy = scaled.height() / max(self._frame.height(), 1)
        ox = (self.width() - scaled.width()) / 2
        oy = (self.height() - scaled.height()) / 2
        return point.x * sx + ox, point.y * sy + oy

    def _draw_skeleton(self, p: QPainter, scaled: QPixmap) -> None:
        state = self._state
        assert state is not None

        # far side first, so the near side draws over it
        p.setPen(QPen(_MARKER_DIM, 2))
        for a, b in bones_for(state.other_points):
            ax, ay = self._map(a, scaled)
            bx, by = self._map(b, scaled)
            p.drawLine(int(ax), int(ay), int(bx), int(by))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_MARKER_DIM)
        for pt in state.other_points:
            if pt.visibility >= 0.4:
                x, y = self._map(pt, scaled)
                p.drawEllipse(int(x - 3), int(y - 3), 6, 6)

        colour = _WARN if state.verdict is SetupVerdict.SIDES_SWAPPED else _MARKER
        p.setPen(QPen(colour, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for a, b in bones_for(state.operated_points):
            ax, ay = self._map(a, scaled)
            bx, by = self._map(b, scaled)
            p.drawLine(int(ax), int(ay), int(bx), int(by))

        for pt in state.operated_points:
            if pt.visibility < 0.4:
                continue
            x, y = self._map(pt, scaled)
            # A soft halo, then the dot. Reads clearly against a busy room.
            p.setPen(Qt.PenStyle.NoPen)
            halo = QColor(colour)
            halo.setAlpha(60)
            p.setBrush(halo)
            p.drawEllipse(int(x - 11), int(y - 11), 22, 22)
            p.setBrush(colour)
            p.drawEllipse(int(x - 6), int(y - 6), 12, 12)

    def _draw_instruction(self, p: QPainter) -> None:
        state = self._state
        assert state is not None

        font = QFont(self.font())
        font.setPointSizeF(20)
        font.setWeight(QFont.Weight.DemiBold)
        p.setFont(font)

        text = state.instruction(self._operated)
        # A drop shadow rather than a panel: the camera stays unobstructed.
        p.setPen(QColor(0, 0, 0, 170))
        p.drawText(2, 30, self.width(), 44, Qt.AlignmentFlag.AlignHCenter, text)
        p.setPen(QColor("#FFFFFF"))
        p.drawText(0, 28, self.width(), 44, Qt.AlignmentFlag.AlignHCenter, text)

    def _draw_blocked_reason(self, p: QPainter) -> None:
        """Why it has not passed yet, in plain words.

        The previous build could stall with nothing on screen explaining it --
        a check you cannot debug from the chair you are sitting in.
        """
        state = self._state
        assert state is not None
        if not state.blocked_reason or state.verdict is not SetupVerdict.WAITING:
            return

        font = QFont(self.font())
        font.setPointSizeF(11)
        p.setFont(font)
        p.setPen(QColor(0, 0, 0, 160))
        p.drawText(1, 105, self.width(), 26,
                   Qt.AlignmentFlag.AlignHCenter, state.blocked_reason)
        p.setPen(QColor(255, 255, 255, 210))
        p.drawText(0, 104, self.width(), 26,
                   Qt.AlignmentFlag.AlignHCenter, state.blocked_reason)

    def _draw_hold_ring(self, p: QPainter) -> None:
        """A ring that fills as the hold accumulates, so the sustain
        requirement is visible rather than a silent timer."""
        state = self._state
        assert state is not None
        if state.hold_progress <= 0.0 or state.verdict is not SetupVerdict.WAITING:
            return

        size, margin = 54, 26
        x = self.width() - size - margin
        y = self.height() - size - margin

        p.setPen(QPen(QColor(255, 255, 255, 70), 5))
        p.drawArc(x, y, size, size, 0, 360 * 16)
        p.setPen(QPen(_MARKER, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(x, y, size, size, 90 * 16, int(-360 * 16 * state.hold_progress))

    def _draw_steps(self, p: QPainter) -> None:
        state = self._state
        assert state is not None

        gap, r = 22, 5
        total = (len(STEP_ORDER) - 1) * gap
        x0 = self.width() / 2 - total / 2
        y = 86
        p.setPen(Qt.PenStyle.NoPen)
        for i, step in enumerate(STEP_ORDER):
            done = state.steps_done.get(step, False)
            p.setBrush(QColor(GOOD) if done else QColor(255, 255, 255, 90))
            size = r * 2 if done else r * 1.5
            p.drawEllipse(int(x0 + i * gap - size / 2), int(y - size / 2), int(size), int(size))

    def _draw_verdict(self, p: QPainter) -> None:
        state = self._state
        assert state is not None

        if state.verdict is SetupVerdict.PASSED:
            text, bg = f"Watching your {self._operated.value} knee", QColor(GOOD)
        elif state.verdict is SetupVerdict.SIDES_SWAPPED:
            text, bg = "The other side moved — stop", QColor(LOCKED)
        else:
            done, total = state.progress
            text, bg = f"{done} of {total}", QColor(0, 0, 0, 150)

        font = QFont(self.font())
        font.setPointSizeF(12)
        font.setWeight(QFont.Weight.DemiBold)
        p.setFont(font)

        w = max(160, p.fontMetrics().horizontalAdvance(text) + 40)
        h = 38
        x = (self.width() - w) / 2
        y = self.height() - h - 22

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(int(x), int(y), int(w), h, 19, 19)
        p.setPen(QColor("#FFFFFF"))
        p.drawText(int(x), int(y), int(w), h, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_telemetry(self, p: QPainter) -> None:
        """Per-joint visibility with the quality thresholds marked.

        The numbers that decide whether reps get scored. Toggled with D.
        """
        state = self._state
        assert state is not None

        pad, w, row = 14, 190, 22
        x = self.width() - w - pad
        y = pad
        h = pad + row * (len(state.clinical_visibility) + 1)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 185))
        p.drawRoundedRect(x, y, w, h, 5, 5)

        font = QFont("Consolas")
        font.setPointSizeF(8.5)
        p.setFont(font)

        p.setPen(QColor(INK_SOFT))
        p.drawText(x + 12, y + 18, "VISIBILITY")

        for i, (name, value) in enumerate(state.clinical_visibility.items()):
            top = y + 14 + row * (i + 1)
            p.setPen(QColor("#A9BBC0"))
            p.drawText(x + 12, top, name.upper()[:8])
            p.drawText(x + w - 46, top, f"{value:.2f}")

            bar_x, bar_w = x + 12, w - 24
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 40))
            p.drawRect(bar_x, top + 4, bar_w, 4)
            p.setBrush(_MARKER if value >= 0.65 else _WARN)
            p.drawRect(bar_x, top + 4, int(bar_w * min(1.0, value)), 4)
            # threshold tick at the good/degraded boundary
            p.setBrush(QColor("#FFFFFF"))
            p.drawRect(int(bar_x + bar_w * 0.65), top + 2, 1, 8)

        # measured values against their thresholds, so a stall is diagnosable
        if state.diagnostics:
            dy = y + h + 8
            dh = 16 * (len(state.diagnostics) + 1)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 185))
            p.drawRoundedRect(x, dy, w, dh, 5, 5)
            p.setPen(QColor(INK_SOFT))
            p.drawText(x + 12, dy + 15, "GATE")
            for i, (key, value) in enumerate(state.diagnostics.items()):
                p.setPen(QColor("#A9BBC0"))
                p.drawText(x + 12, dy + 15 + 16 * (i + 1), f"{key[:10]:<11}{value}")

    def _centred(self, p: QPainter, text: str, size: float, colour: QColor) -> None:
        font = QFont(self.font())
        font.setPointSizeF(size)
        p.setFont(font)
        p.setPen(colour)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
