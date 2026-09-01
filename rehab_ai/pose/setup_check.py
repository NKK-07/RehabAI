"""
setup_check.py
Verifies the camera setup, and proves the operated-side binding on the
landmarks the product actually measures.

WHY A HEEL LIFT AND NOT AN ARM RAISE
====================================
The first version asked for an arm raise. That proved something true but
indirect: it tested `wrist` and `shoulder` to infer a binding that matters for
`hip`, `knee` and `ankle`. The inference is sound -- the pose model assigns
left and right consistently across every landmark -- but it left the joints the
hip-drive ratio is built from untested.

A heel lift exercises `ankle` directly. That is a CLINICAL landmark: the same
point the knee angle is computed from, on the same leg the whole product is
about. If the app sees the operated ankle rise when you lift that heel, the
binding is proven on the chain that carries the measurement.

WHY SUSTAINED, NOT A SINGLE FRAME
=================================
The pose model estimates occluded joints rather than declining to answer, and
in profile the far leg is occluded by the near one. Those estimates move. A
check that latches PASSED on one frame is a check a single jitter can fake --
and it latches permanently, so the noise only has to win once.

So a lift must hold for `_SUSTAIN_FRAMES` consecutive frames, clearing the
floor by a margin larger than the jitter.

    baseline = median ankle position over a rolling window   (the floor)
    lift     = ankle above baseline by _LIFT_CLEARANCE
    pass     = lift held for _SUSTAIN_FRAMES in a row

The baseline is a MEDIAN, not a running maximum. A running maximum is dragged
below the real floor by a single downward spike, after which every resting
frame reads as a lift -- measured at 16 of 20 stationary trials passing. See
_SideTrack for the numbers.

    "raise your operated heel"
              │
              ▼
    operated ankle rises, held    -> binding correct
    other ankle rises instead     -> LEFT AND RIGHT ARE SWAPPED
    neither, or only a flicker    -> still waiting

WHAT THIS STILL DOES NOT PROVE
==============================
Steps 2 and 3 (side-on, operated leg nearest) rest on visibility values the
model infers for occluded joints. They are framing guidance, not measurements,
and they can pass for the wrong reason. Step 4 is the load-bearing one, and it
is the only step whose verdict is recorded.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median

from rehab_ai.models.session import Side
from rehab_ai.pose.tracker import _SKELETON_BONES, _SKELETON_LANDMARKS
from rehab_ai.pose_utils import PoseLandmark, get_visibility, get_xy

LM = PoseLandmark


class SetupStep(str, Enum):
    """Ordered. Each only becomes checkable once the previous holds."""

    IN_FRAME = "in_frame"
    SIDE_ON = "side_on"
    OPERATED_NEAR = "operated_near"
    HEEL_LIFT = "heel_lift"


STEP_ORDER = (
    SetupStep.IN_FRAME,
    SetupStep.SIDE_ON,
    SetupStep.OPERATED_NEAR,
    SetupStep.HEEL_LIFT,
)

STEP_TEXT = {
    SetupStep.IN_FRAME: "Step into view",
    SetupStep.SIDE_ON: "Turn side-on to the camera",
    SetupStep.OPERATED_NEAR: "Put your {side} leg nearest the camera",
    SetupStep.HEEL_LIFT: "Lift your {side} heel, and hold it",
}


class SetupVerdict(str, Enum):
    WAITING = "waiting"
    PASSED = "passed"
    SIDES_SWAPPED = "sides_swapped"


@dataclass(frozen=True)
class JointPoint:
    name: str
    x: float
    y: float
    visibility: float


@dataclass(frozen=True)
class SetupState:
    """Everything the setup screen needs for one frame."""

    person_detected: bool
    steps_done: dict[SetupStep, bool]
    current_step: SetupStep | None
    verdict: SetupVerdict
    operated_points: list[JointPoint]
    other_points: list[JointPoint]
    clinical_visibility: dict[str, float]
    hold_progress: float = 0.0  # 0..1 through the sustained hold

    @property
    def progress(self) -> tuple[int, int]:
        return sum(self.steps_done.values()), len(STEP_ORDER)

    def instruction(self, operated: Side) -> str:
        if self.verdict is SetupVerdict.PASSED:
            return "All set"
        if self.verdict is SetupVerdict.SIDES_SWAPPED:
            return "The other leg moved"
        if self.current_step is None:
            return "All set"
        return STEP_TEXT[self.current_step].format(side=operated.value)


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------

_MIN_VISIBILITY = 0.4

# Framing guidance, deliberately generous -- a strict gate here just annoys
# people, and neither of these steps is what the verdict rests on.
_SIDE_ON_MAX_SHOULDER_SPREAD = 0.14
_NEAR_SIDE_MARGIN = 0.08

# The load-bearing numbers. Clearance is a fraction of frame height; sustain is
# frames at roughly 25fps, so 10 frames is about 0.4s -- long enough that pose
# jitter cannot hold it, short enough not to tire a post-op patient.
_LIFT_CLEARANCE = 0.05
_SUSTAIN_FRAMES = 10
# The floor is the median of a rolling window, not an all-time extreme. Wide
# enough that a lift cannot dominate it (a 10-frame hold against 45 samples),
# narrow enough to follow a patient who shifts position.
_BASELINE_WINDOW = 45
_BASELINE_MIN_SAMPLES = 15


class _SideTrack:
    """Per-side lift tracking: where the floor is, and how long a lift held.

    WHY A ROLLING MEDIAN AND NOT A RUNNING MAXIMUM
    ==============================================
    The obvious way to find the floor is "the lowest the ankle has ever been",
    a running maximum of y. It is also wrong, and measurably so.

    A running maximum is maximally sensitive to exactly the outlier it should
    ignore. One downward jitter spike drags the baseline below the real floor;
    from then on every RESTING frame sits more than `clearance` above it, the
    consecutive counter climbs, and the gate latches without the patient
    moving at all.

    Measured with a stationary foot and jitter at 2% of frame height: 16 of 20
    trials passed, 6 of them falsely reporting SIDES_SWAPPED.

    A median over a rolling window is robust to that. A handful of spikes
    cannot move it, and it still self-corrects for a patient who begins with
    the foot already raised -- once the foot comes down, the window fills with
    floor samples and the median follows.
    """

    __slots__ = ("_window", "consecutive")

    def __init__(self) -> None:
        self._window: deque[float] = deque(maxlen=_BASELINE_WINDOW)
        self.consecutive = 0

    @property
    def baseline_y(self) -> float | None:
        if len(self._window) < _BASELINE_MIN_SAMPLES:
            return None
        return median(self._window)

    def observe(self, y: float, clearance: float) -> bool:
        """Feed one ankle position. Returns True once a lift has been sustained."""
        self._window.append(y)

        baseline = self.baseline_y
        if baseline is None:
            return False  # not enough evidence about where the floor is yet

        if y < baseline - clearance:
            self.consecutive += 1
        else:
            self.consecutive = 0

        return self.consecutive >= _SUSTAIN_FRAMES

    @property
    def hold_progress(self) -> float:
        return min(1.0, self.consecutive / _SUSTAIN_FRAMES)

    def reset(self) -> None:
        self._window.clear()
        self.consecutive = 0


def _points(landmarks, side: Side, width: int, height: int) -> list[JointPoint]:
    out = []
    for name, index in _SKELETON_LANDMARKS[side].items():
        x, y = get_xy(landmarks, index, width, height)
        out.append(JointPoint(name, float(x), float(y), get_visibility(landmarks, index)))
    return out


def _by_name(points: list[JointPoint]) -> dict[str, JointPoint]:
    return {p.name: p for p in points}


def bones_for(points: list[JointPoint]) -> list[tuple[JointPoint, JointPoint]]:
    """Connected segments, skipping any whose endpoints are not both visible.

    Drawing a bone to a landmark the model is guessing at produces a limb that
    swings around the screen, which reads as a broken tracker rather than as
    low confidence.
    """
    lookup = _by_name(points)
    segments = []
    for a, b in _SKELETON_BONES:
        pa, pb = lookup.get(a), lookup.get(b)
        if pa and pb and pa.visibility >= _MIN_VISIBILITY and pb.visibility >= _MIN_VISIBILITY:
            segments.append((pa, pb))
    return segments


class SetupChecker:
    """Stateful across frames: latched steps, per-side floor baselines, holds.

    A class rather than threading `previous` through a free function, because
    the lift check genuinely needs memory -- where the floor is, and how many
    consecutive frames the foot has been off it.
    """

    def __init__(self, operated: Side) -> None:
        self.operated = operated
        self._done = {step: False for step in STEP_ORDER}
        self._verdict = SetupVerdict.WAITING
        self._tracks = {operated: _SideTrack(), operated.other: _SideTrack()}

    def reset(self) -> None:
        self._done = {step: False for step in STEP_ORDER}
        self._verdict = SetupVerdict.WAITING
        for track in self._tracks.values():
            track.reset()

    def update(self, landmarks, width: int, height: int) -> SetupState:
        if landmarks is None:
            return self._state(False, [], [], {}, 0.0)

        operated_points = _points(landmarks, self.operated, width, height)
        other_points = _points(landmarks, self.operated.other, width, height)
        op = _by_name(operated_points)
        ot = _by_name(other_points)

        self._done[SetupStep.IN_FRAME] = True

        # -- side-on: the two shoulders nearly overlap horizontally ----------
        if op.get("shoulder") and ot.get("shoulder"):
            spread = abs(op["shoulder"].x - ot["shoulder"].x) / max(width, 1)
            if spread <= _SIDE_ON_MAX_SHOULDER_SPREAD:
                self._done[SetupStep.SIDE_ON] = True

        # -- operated side is the near one -----------------------------------
        if self._done[SetupStep.SIDE_ON]:
            joints = ("shoulder", "hip", "knee", "ankle")
            if _mean_visibility(op, joints) >= _mean_visibility(ot, joints) + _NEAR_SIDE_MARGIN:
                self._done[SetupStep.OPERATED_NEAR] = True

        # -- the heel lift, which proves the binding on a clinical landmark --
        hold = 0.0
        if self._done[SetupStep.OPERATED_NEAR] and self._verdict is SetupVerdict.WAITING:
            clearance = _LIFT_CLEARANCE * height

            operated_lifted = self._track(self.operated, op, clearance)
            other_lifted = self._track(self.operated.other, ot, clearance)
            hold = self._tracks[self.operated].hold_progress

            if operated_lifted:
                self._done[SetupStep.HEEL_LIFT] = True
                self._verdict = SetupVerdict.PASSED
            elif other_lifted:
                # The patient lifted the heel they were told to. If the OTHER
                # ankle is what the model saw rise, the labels are attached to
                # the wrong body -- and every hip and knee angle this session
                # records would describe the healthy leg.
                self._verdict = SetupVerdict.SIDES_SWAPPED

        return self._state(
            True,
            operated_points,
            other_points,
            {n: op[n].visibility for n in ("shoulder", "hip", "knee", "ankle") if n in op},
            hold,
        )

    def _track(self, side: Side, points: dict[str, JointPoint], clearance: float) -> bool:
        ankle = points.get("ankle")
        if ankle is None or ankle.visibility < _MIN_VISIBILITY:
            # Cannot see the ankle, so cannot claim it moved OR that it did not.
            self._tracks[side].consecutive = 0
            return False
        return self._tracks[side].observe(ankle.y, clearance)

    def _state(self, detected, operated_points, other_points, visibility, hold) -> SetupState:
        return SetupState(
            person_detected=detected,
            steps_done=dict(self._done),
            current_step=_next_step(self._done),
            verdict=self._verdict,
            operated_points=operated_points,
            other_points=other_points,
            clinical_visibility=visibility,
            hold_progress=hold,
        )


def _mean_visibility(points: dict[str, JointPoint], names: tuple[str, ...]) -> float:
    values = [points[n].visibility for n in names if n in points]
    return sum(values) / len(values) if values else 0.0


def _next_step(done: dict[SetupStep, bool]) -> SetupStep | None:
    for step in STEP_ORDER:
        if not done[step]:
            return step
    return None
