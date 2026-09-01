"""
Tests for the camera setup check (CP 3, automated).

The APP verifies its own operated-side binding, rather than asking the patient
to eyeball whether the right marker moved.

The gate is a HEEL LIFT, not an arm raise, and that choice is the point. A heel
lift moves `ankle` -- a CLINICAL landmark, the same point the knee angle is
computed from. An arm raise would prove the binding on `wrist`, a landmark the
product never measures, leaving the measured chain untested.

It must also be SUSTAINED. The pose model estimates occluded joints rather than
declining to answer, so a single-frame check is one a jitter can fake -- and it
latches permanently, so the noise only has to win once.

Two tests carry the weight:
  * test_a_single_lifted_frame_does_not_pass    (the jitter defence)
  * test_the_wrong_ankle_rising_reports_a_swap  (the mirroring defence)
"""

import pytest

from rehab_ai.models.session import Side
from rehab_ai.pose.setup_check import (
    _BASELINE_WINDOW,
    _SUSTAIN_FRAMES,
    STEP_ORDER,
    SetupChecker,
    SetupStep,
    SetupVerdict,
    bones_for,
)
from rehab_ai.pose.tracker import _CLINICAL_LANDMARKS, _SKELETON_LANDMARKS

W, H = 640, 480


# --------------------------------------------------------------------------
# synthetic landmarks
# --------------------------------------------------------------------------


class _Landmark:
    def __init__(self, x, y, visibility):
        self.x, self.y, self.visibility = x, y, visibility


class _Landmarks:
    """Mimics MediaPipe's result shape: indexable by landmark enum."""

    def __init__(self, mapping):
        size = max(int(i) for i in mapping) + 1
        self.landmark = [_Landmark(0.5, 0.5, 0.0) for _ in range(size)]
        for index, (x, y, vis) in mapping.items():
            self.landmark[int(index)] = _Landmark(x, y, vis)


def pose(
    *,
    operated: Side = Side.LEFT,
    side_on: bool = True,
    operated_near: bool = True,
    lift_side: Side | None = None,
) -> _Landmarks:
    """Build a landmark set with the properties each test needs.

    Coordinates are normalised (MediaPipe convention); get_xy multiplies by
    frame size.
    """
    near, far = operated, operated.other
    if not operated_near:
        near, far = far, near

    near_vis, far_vis = 0.95, 0.45
    # Face-on separates the shoulders horizontally; profile collapses them.
    offset = 0.0 if side_on else 0.30

    mapping = {}
    for side, vis, dx in ((near, near_vis, 0.0), (far, far_vis, offset)):
        marks = _SKELETON_LANDMARKS[side]
        lifted = lift_side is side
        mapping.update({
            marks["ear"]: (0.44 + dx, 0.16, vis),
            marks["shoulder"]: (0.46 + dx, 0.30, vis),
            marks["elbow"]: (0.44 + dx, 0.44, vis),
            # A raised wrist clears the shoulder; a resting one sits well below.
            marks["wrist"]: (0.42 + dx, 0.56, vis),
            marks["hip"]: (0.48 + dx, 0.58, vis),
            marks["knee"]: (0.58 + dx, 0.72, vis),
            # Screen y grows downward: a lifted ankle sits HIGHER, so smaller y.
            marks["ankle"]: (0.60 + dx, 0.78 if lifted else 0.90, vis),
            marks["heel"]: (0.58 + dx, 0.81 if lifted else 0.93, vis),
            marks["toe"]: (0.66 + dx, 0.81 if lifted else 0.93, vis),
        })
    return _Landmarks(mapping)


def run(sequence, operated: Side = Side.LEFT, checker: SetupChecker | None = None):
    """Feed frames in order through a stateful checker."""
    checker = checker or SetupChecker(operated)
    state = None
    for landmarks in sequence:
        state = checker.update(landmarks, W, H)
    return state


def settled(**kw):
    """Enough identical frames to fully flush the rolling baseline window.

    Sized from _BASELINE_WINDOW rather than a literal, so tuning the window
    cannot silently leave these fixtures feeding a half-filled buffer -- which
    is exactly what happened when the window grew from 8 to 45.
    """
    return [pose(**kw)] * (_BASELINE_WINDOW + 5)


def held(frames: int = _SUSTAIN_FRAMES + 2, **kw):
    """A lift held long enough to satisfy the sustain requirement."""
    return [pose(**kw)] * frames


# --------------------------------------------------------------------------
# The landmark split -- extra points must not gate the clinical measurement
# --------------------------------------------------------------------------


def test_the_skeleton_set_is_richer_than_the_clinical_set():
    """More points to track accurately, without changing what is measured."""
    assert len(_SKELETON_LANDMARKS[Side.LEFT]) > len(_CLINICAL_LANDMARKS[Side.LEFT])
    assert len(_SKELETON_LANDMARKS[Side.LEFT]) == 9


def test_the_clinical_set_is_still_exactly_the_four_measured_joints():
    """The hip and knee angles are built from these and nothing else. Growing
    this set would silently change what every threshold was tuned against."""
    assert set(_CLINICAL_LANDMARKS[Side.LEFT]) == {"shoulder", "hip", "knee", "ankle"}


def test_a_hidden_wrist_does_not_make_a_visible_knee_unobservable():
    """THE regression this split exists to prevent.

    observe_landmarks takes the minimum visibility across the clinical set. If
    the skeleton set were folded in, a hand dropping out of frame would mark
    the rep UNOBSERVABLE while the knee sat in plain view at 0.95 -- the camera
    refusing to score a movement it could see perfectly well.
    """
    from rehab_ai.models.session import ObservationQuality
    from rehab_ai.pose.tracker import OperatedSideTracker
    from rehab_ai.rules.loader import load_rules

    rules = load_rules().observation
    marks = _SKELETON_LANDMARKS[Side.LEFT]

    mapping = {
        marks["shoulder"]: (0.46, 0.30, 0.95),
        marks["hip"]: (0.48, 0.58, 0.95),
        marks["knee"]: (0.58, 0.72, 0.95),
        marks["ankle"]: (0.60, 0.90, 0.95),
        marks["wrist"]: (0.20, 0.99, 0.02),   # hand out of frame
        marks["elbow"]: (0.30, 0.80, 0.05),
    }

    tracker = OperatedSideTracker.__new__(OperatedSideTracker)
    tracker.operated_side = Side.LEFT
    tracker._rules = rules
    tracker._landmarks = _CLINICAL_LANDMARKS[Side.LEFT]
    tracker._skeleton = marks

    observation = tracker.observe_landmarks(_Landmarks(mapping), W, H, 1.0)

    assert observation.quality is ObservationQuality.GOOD
    assert observation.angles is not None


# --------------------------------------------------------------------------
# The four steps
# --------------------------------------------------------------------------


def test_no_person_detected_reports_nothing_done():
    state = SetupChecker(Side.LEFT).update(None, W, H)
    assert not state.person_detected
    assert state.current_step is SetupStep.IN_FRAME
    assert state.progress == (0, 4)


def test_a_person_in_frame_clears_the_first_step():
    state = run([pose(side_on=False, operated_near=False)])
    assert state.steps_done[SetupStep.IN_FRAME]
    assert not state.steps_done[SetupStep.SIDE_ON]


def test_facing_the_camera_does_not_pass_the_side_on_step():
    """Face-on, the two shoulders sit far apart horizontally. That is the same
    cue the how-to animation demonstrates, so the illustration teaches the real
    signal rather than a stylised one."""
    state = run([pose(side_on=False)])
    assert not state.steps_done[SetupStep.SIDE_ON]
    assert state.current_step is SetupStep.SIDE_ON


def test_turning_side_on_clears_the_second_step():
    state = run([pose(side_on=True)])
    assert state.steps_done[SetupStep.SIDE_ON]


def test_the_wrong_leg_nearest_holds_at_step_three():
    state = run([pose(side_on=True, operated_near=False)])
    assert state.steps_done[SetupStep.SIDE_ON]
    assert not state.steps_done[SetupStep.OPERATED_NEAR]
    assert state.current_step is SetupStep.OPERATED_NEAR


def test_the_operated_leg_nearest_clears_the_third_step():
    state = run([pose(side_on=True, operated_near=True)])
    assert state.steps_done[SetupStep.OPERATED_NEAR]
    assert state.current_step is SetupStep.HEEL_LIFT


# --------------------------------------------------------------------------
# The heel lift -- what proves the binding, on a landmark we actually measure
# --------------------------------------------------------------------------


def test_a_sustained_lift_of_the_operated_heel_passes():
    state = run(settled() + held(lift_side=Side.LEFT), operated=Side.LEFT)

    assert state.verdict is SetupVerdict.PASSED
    assert state.steps_done[SetupStep.HEEL_LIFT]
    assert state.progress == (4, 4)


def test_a_single_lifted_frame_does_not_pass():
    """THE jitter defence, and the answer to "does lifting once pass it?"

    One frame where the ankle estimate wobbles upward must not latch a
    permanent PASSED. The model guesses occluded joints, and in profile the far
    leg is occluded, so those guesses move on their own.
    """
    state = run(settled() + [pose(lift_side=Side.LEFT)], operated=Side.LEFT)

    assert state.verdict is SetupVerdict.WAITING
    assert not state.steps_done[SetupStep.HEEL_LIFT]


def test_a_lift_shorter_than_the_sustain_window_does_not_pass():
    state = run(settled() + held(_SUSTAIN_FRAMES - 2, lift_side=Side.LEFT))
    assert state.verdict is SetupVerdict.WAITING


def test_a_lift_that_flickers_restarts_the_count():
    """Up, down, up must not sum to a sustained hold, or noise accumulates
    into a pass given enough frames."""
    frames = settled()
    frames += held(_SUSTAIN_FRAMES - 2, lift_side=Side.LEFT)
    frames += [pose()]
    frames += held(_SUSTAIN_FRAMES - 2, lift_side=Side.LEFT)

    assert run(frames).verdict is SetupVerdict.WAITING


def test_the_hold_progress_is_reported_so_the_screen_can_show_it():
    state = run(settled() + held(_SUSTAIN_FRAMES // 2, lift_side=Side.LEFT))
    assert 0.0 < state.hold_progress < 1.0


def test_the_wrong_ankle_rising_reports_a_swap():
    """THE mirroring defence.

    The patient lifted the heel they were told to. If the OTHER ankle is what
    the model saw rise, the labels are attached to the wrong body -- and every
    hip and knee angle this session records would describe the healthy leg.
    """
    state = run(settled() + held(lift_side=Side.RIGHT), operated=Side.LEFT)

    assert state.verdict is SetupVerdict.SIDES_SWAPPED
    assert not state.steps_done[SetupStep.HEEL_LIFT]


def test_a_foot_left_on_the_floor_never_passes():
    state = run(settled() + held(lift_side=None))
    assert state.verdict is SetupVerdict.WAITING


def test_the_lift_is_not_evaluated_before_framing_is_right():
    """Lifting a heel while face-on proves nothing about which leg the camera
    can see, so the step is not evaluated yet."""
    state = run(settled(side_on=False) + held(side_on=False, lift_side=Side.LEFT))
    assert state.verdict is SetupVerdict.WAITING


def test_the_floor_baseline_self_corrects_from_a_raised_start():
    """A patient who begins with the foot already up sets a wrong baseline.
    Lowering it must correct that, or the check would be permanently stuck."""
    frames = settled(lift_side=Side.LEFT)
    frames += settled()
    frames += held(lift_side=Side.LEFT)

    assert run(frames).verdict is SetupVerdict.PASSED


def test_an_invisible_ankle_cannot_pass_or_fail_the_lift():
    """If the ankle cannot be seen, the app can claim neither that it moved nor
    that it did not."""
    marks = _SKELETON_LANDMARKS[Side.LEFT]
    frames = settled()
    for _ in range(_SUSTAIN_FRAMES + 4):
        landmarks = pose(lift_side=Side.LEFT)
        landmarks.landmark[int(marks["ankle"])].visibility = 0.05
        frames.append(landmarks)

    assert run(frames).verdict is SetupVerdict.WAITING


def test_a_verdict_is_not_revised_once_reached():
    """A passed check stays passed -- lowering the foot must not undo it."""
    assert run(settled() + held(lift_side=Side.LEFT) + settled()).verdict is SetupVerdict.PASSED


def test_a_swap_verdict_also_sticks():
    """Especially this one. A swap must not scroll past because the patient
    put their foot down."""
    frames = settled() + held(lift_side=Side.RIGHT) + settled()
    assert run(frames).verdict is SetupVerdict.SIDES_SWAPPED


def test_steps_latch_through_a_momentary_wobble():
    """Turning slightly for one frame must not send the patient back to step
    one. Steps latch; only the verdict is decisive."""
    state = run([pose(side_on=True), pose(side_on=False), pose(side_on=True)])
    assert state.steps_done[SetupStep.SIDE_ON]


def test_it_works_for_a_right_side_patient():
    frames = settled(operated=Side.RIGHT) + held(operated=Side.RIGHT, lift_side=Side.RIGHT)
    assert run(frames, operated=Side.RIGHT).verdict is SetupVerdict.PASSED


def test_resetting_clears_everything_for_a_retry():
    checker = SetupChecker(Side.LEFT)
    run(settled() + held(lift_side=Side.RIGHT), checker=checker)
    assert checker._verdict is SetupVerdict.SIDES_SWAPPED

    checker.reset()
    state = checker.update(pose(), W, H)
    assert state.verdict is SetupVerdict.WAITING
    assert not state.steps_done[SetupStep.HEEL_LIFT]


# --------------------------------------------------------------------------
# Drawing data
# --------------------------------------------------------------------------


def test_bones_skip_segments_with_an_invisible_endpoint():
    """A bone drawn to a landmark the model is guessing at swings around the
    screen, which reads as a broken tracker rather than as low confidence."""
    marks = _SKELETON_LANDMARKS[Side.LEFT]
    mapping = {marks[name]: (0.5, 0.5, 0.9) for name in marks}
    mapping[marks["toe"]] = (0.5, 0.5, 0.05)

    from rehab_ai.pose.setup_check import _points

    points = _points(_Landmarks(mapping), Side.LEFT, W, H)
    names = {(a.name, b.name) for a, b in bones_for(points)}

    assert ("heel", "toe") not in names
    assert ("hip", "knee") in names


def test_a_full_skeleton_produces_every_bone():
    from rehab_ai.pose.setup_check import _SKELETON_BONES

    state = run([pose()])
    assert len(bones_for(state.operated_points)) == len(_SKELETON_BONES)


def test_clinical_visibility_is_reported_for_the_telemetry_panel():
    state = run([pose()])
    assert set(state.clinical_visibility) == {"shoulder", "hip", "knee", "ankle"}
    assert all(0.0 <= v <= 1.0 for v in state.clinical_visibility.values())


# --------------------------------------------------------------------------
# Instructions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("step", STEP_ORDER)
def test_every_step_has_instruction_text(step):
    from rehab_ai.pose.setup_check import STEP_TEXT

    assert step in STEP_TEXT
    assert STEP_TEXT[step].strip()


def test_the_instruction_names_the_operated_side():
    state = run([pose(side_on=True, operated_near=True)], operated=Side.RIGHT)
    assert "right" in state.instruction(Side.RIGHT)


def test_a_passed_check_stops_instructing():
    state = run(settled() + held(lift_side=Side.LEFT))
    assert state.instruction(Side.LEFT) == "All set"


def test_the_instruction_asks_for_a_hold_not_just_a_lift():
    """The sustain requirement is part of the instruction, not a hidden rule
    the patient fails without knowing why."""
    from rehab_ai.pose.setup_check import STEP_TEXT

    text = STEP_TEXT[SetupStep.HEEL_LIFT].lower()
    assert "hold" in text
    assert "heel" in text
