"""
setup_check.py
Verifies the camera setup before a session, and proves the operated-side
binding is correct.

WHY THE APP SHOULD CHECK THIS ITSELF
====================================
The old check asked the patient to raise an arm and then eyeball whether the
right marker moved. But the app tracked only shoulder, hip, knee and ankle --
no elbow, no wrist -- so it had no idea whether the arm went up. The person
being tested was also the instrument.

With the wrist tracked, the app can confirm its own binding:

    "raise your LEFT arm"
              │
              ▼
    left wrist rises above left shoulder     -> binding correct
    right wrist rises instead                -> LEFT AND RIGHT ARE SWAPPED
    neither                                  -> still waiting

That last distinction is the whole point. A swap is otherwise invisible: from
inside the code the labels are self-consistent, they are just attached to the
wrong body.

THE FOUR STEPS
==============
Each is a condition the app can observe, so the sequence advances on its own
rather than asking the patient to decide when they have complied.

    1 IN_FRAME       a person is detected at all
    2 SIDE_ON        shoulders are close together horizontally (profile view)
    3 OPERATED_NEAR  the operated side is the more visible one
    4 ARM_RAISED     the operated-side wrist goes above its shoulder
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rehab_ai.models.session import Side
from rehab_ai.pose.tracker import _SKELETON_BONES, _SKELETON_LANDMARKS
from rehab_ai.pose_utils import PoseLandmark, get_visibility, get_xy

LM = PoseLandmark


class SetupStep(str, Enum):
    """Ordered. Each one only becomes checkable once the previous holds."""

    IN_FRAME = "in_frame"
    SIDE_ON = "side_on"
    OPERATED_NEAR = "operated_near"
    ARM_RAISED = "arm_raised"


STEP_ORDER = (
    SetupStep.IN_FRAME,
    SetupStep.SIDE_ON,
    SetupStep.OPERATED_NEAR,
    SetupStep.ARM_RAISED,
)

STEP_TEXT = {
    SetupStep.IN_FRAME: "Step into view",
    SetupStep.SIDE_ON: "Turn side-on to the camera",
    SetupStep.OPERATED_NEAR: "Put your {side} leg nearest the camera",
    SetupStep.ARM_RAISED: "Raise your {side} arm",
}


class SetupVerdict(str, Enum):
    WAITING = "waiting"
    PASSED = "passed"
    SIDES_SWAPPED = "sides_swapped"


@dataclass(frozen=True)
class JointPoint:
    """One landmark, ready to draw."""

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

    @property
    def progress(self) -> tuple[int, int]:
        return sum(self.steps_done.values()), len(STEP_ORDER)

    def instruction(self, operated: Side) -> str:
        if self.verdict is SetupVerdict.PASSED:
            return "All set"
        if self.verdict is SetupVerdict.SIDES_SWAPPED:
            return "Wrong side moved"
        if self.current_step is None:
            return "All set"
        return STEP_TEXT[self.current_step].format(side=operated.value)


# ---------------------------------------------------------------------------
# thresholds -- deliberately generous; this is framing guidance, not a
# clinical measurement, and a strict gate here just annoys people
# ---------------------------------------------------------------------------

_MIN_VISIBILITY = 0.4
# Side-on: in a profile view the two shoulders sit almost on top of each other
# horizontally. Face-on they are far apart. Measured as a fraction of frame
# width so it holds at any resolution.
_SIDE_ON_MAX_SHOULDER_SPREAD = 0.14
# The near side reads meaningfully more visible than the far one.
_NEAR_SIDE_MARGIN = 0.08
# The wrist must clear the shoulder by this fraction of frame height, so a
# hand resting on a lap never counts.
_ARM_RAISE_CLEARANCE = 0.04


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
    swings around the screen, which reads as the tracker being broken rather
    than as low confidence.
    """
    lookup = _by_name(points)
    segments = []
    for a, b in _SKELETON_BONES:
        pa, pb = lookup.get(a), lookup.get(b)
        if pa and pb and pa.visibility >= _MIN_VISIBILITY and pb.visibility >= _MIN_VISIBILITY:
            segments.append((pa, pb))
    return segments


def evaluate(
    landmarks,
    operated: Side,
    width: int,
    height: int,
    *,
    previous: SetupState | None = None,
) -> SetupState:
    """Assess one frame against the four setup steps.

    Steps latch once satisfied (via `previous`) so a momentary wobble does not
    reset the sequence and send the patient back to step one.
    """
    done = dict(previous.steps_done) if previous else {s: False for s in STEP_ORDER}
    verdict = previous.verdict if previous else SetupVerdict.WAITING

    if landmarks is None:
        return SetupState(
            person_detected=False,
            steps_done=done,
            current_step=_next_step(done),
            verdict=verdict,
            operated_points=[],
            other_points=[],
            clinical_visibility={},
        )

    operated_points = _points(landmarks, operated, width, height)
    other_points = _points(landmarks, operated.other, width, height)
    op = _by_name(operated_points)
    ot = _by_name(other_points)

    done[SetupStep.IN_FRAME] = True

    # -- side-on: shoulders nearly overlap horizontally ---------------------
    if op.get("shoulder") and ot.get("shoulder"):
        spread = abs(op["shoulder"].x - ot["shoulder"].x) / max(width, 1)
        if spread <= _SIDE_ON_MAX_SHOULDER_SPREAD:
            done[SetupStep.SIDE_ON] = True

    # -- operated side is the near one --------------------------------------
    if done[SetupStep.SIDE_ON]:
        op_vis = _mean_visibility(op, ("shoulder", "hip", "knee", "ankle"))
        ot_vis = _mean_visibility(ot, ("shoulder", "hip", "knee", "ankle"))
        if op_vis >= ot_vis + _NEAR_SIDE_MARGIN:
            done[SetupStep.OPERATED_NEAR] = True

    # -- the arm raise, which is what actually proves the binding ------------
    if done[SetupStep.OPERATED_NEAR] and verdict is SetupVerdict.WAITING:
        clearance = _ARM_RAISE_CLEARANCE * height
        operated_raised = _wrist_above_shoulder(op, clearance)
        other_raised = _wrist_above_shoulder(ot, clearance)

        if operated_raised:
            done[SetupStep.ARM_RAISED] = True
            verdict = SetupVerdict.PASSED
        elif other_raised:
            # The patient did as asked. If the OTHER side's wrist is what the
            # model saw rise, the labels are attached to the wrong body.
            verdict = SetupVerdict.SIDES_SWAPPED

    return SetupState(
        person_detected=True,
        steps_done=done,
        current_step=_next_step(done),
        verdict=verdict,
        operated_points=operated_points,
        other_points=other_points,
        clinical_visibility={
            name: op[name].visibility
            for name in ("shoulder", "hip", "knee", "ankle")
            if name in op
        },
    )


def _wrist_above_shoulder(side_points: dict[str, JointPoint], clearance: float) -> bool:
    """Screen y grows downward, so 'above' means a smaller y."""
    wrist, shoulder = side_points.get("wrist"), side_points.get("shoulder")
    if not wrist or not shoulder:
        return False
    if wrist.visibility < _MIN_VISIBILITY or shoulder.visibility < _MIN_VISIBILITY:
        return False
    return wrist.y < shoulder.y - clearance


def _mean_visibility(points: dict[str, JointPoint], names: tuple[str, ...]) -> float:
    values = [points[n].visibility for n in names if n in points]
    return sum(values) / len(values) if values else 0.0


def _next_step(done: dict[SetupStep, bool]) -> SetupStep | None:
    for step in STEP_ORDER:
        if not done[step]:
            return step
    return None
