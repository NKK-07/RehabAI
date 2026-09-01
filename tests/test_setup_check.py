"""
Tests for the camera setup check (CP 3, automated).

The point of this module is that the APP verifies its own operated-side
binding, instead of asking the patient to eyeball whether the right marker
moved. That only became possible once the wrist was tracked -- with four
landmarks the app had no idea whether an arm went up.

The case that matters most is test_the_wrong_wrist_rising_reports_a_swap:
the patient does exactly as asked, and if the OTHER side's wrist is what the
model saw rise, left and right are attached to the wrong body. Nothing else in
the test suite can catch that, because from inside the code the labels are
self-consistent.
"""

import pytest

from rehab_ai.models.session import Side
from rehab_ai.pose.setup_check import (
    STEP_ORDER,
    SetupStep,
    SetupVerdict,
    bones_for,
    evaluate,
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
    raise_side: Side | None = None,
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
        raised = raise_side is side
        mapping.update({
            marks["ear"]: (0.44 + dx, 0.16, vis),
            marks["shoulder"]: (0.46 + dx, 0.30, vis),
            marks["elbow"]: (0.44 + dx, 0.44, vis),
            # A raised wrist clears the shoulder; a resting one sits well below.
            marks["wrist"]: (0.42 + dx, 0.12 if raised else 0.56, vis),
            marks["hip"]: (0.48 + dx, 0.58, vis),
            marks["knee"]: (0.58 + dx, 0.72, vis),
            marks["ankle"]: (0.60 + dx, 0.90, vis),
            marks["heel"]: (0.58 + dx, 0.93, vis),
            marks["toe"]: (0.66 + dx, 0.93, vis),
        })
    return _Landmarks(mapping)


def run(sequence, operated: Side = Side.LEFT):
    """Feed frames in order, carrying state forward the way the view does."""
    state = None
    for landmarks in sequence:
        state = evaluate(landmarks, operated, W, H, previous=state)
    return state


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
    state = evaluate(None, Side.LEFT, W, H)
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
    assert state.current_step is SetupStep.ARM_RAISED


# --------------------------------------------------------------------------
# The arm raise -- what actually proves the binding
# --------------------------------------------------------------------------


def test_raising_the_operated_arm_passes_the_check():
    state = run([pose(), pose(raise_side=Side.LEFT)], operated=Side.LEFT)

    assert state.verdict is SetupVerdict.PASSED
    assert state.steps_done[SetupStep.ARM_RAISED]
    assert state.progress == (4, 4)


def test_the_wrong_wrist_rising_reports_a_swap():
    """THE case this whole module exists for.

    The patient raised the arm they were told to. If the model reports the
    OTHER side's wrist as the one that rose, the labels are attached to the
    wrong body -- and every downstream number would describe the healthy knee.
    """
    state = run([pose(), pose(raise_side=Side.RIGHT)], operated=Side.LEFT)

    assert state.verdict is SetupVerdict.SIDES_SWAPPED
    assert not state.steps_done[SetupStep.ARM_RAISED]


def test_a_resting_hand_never_counts_as_raised():
    """A wrist on a lap sits below the shoulder. Without the clearance margin,
    noise around a shoulder-height hand would pass the check."""
    state = run([pose(), pose(raise_side=None)])
    assert state.verdict is SetupVerdict.WAITING


def test_the_arm_raise_is_not_checked_before_framing_is_right():
    """Raising an arm while face-on proves nothing about which leg the camera
    can see, so the step is not even evaluated yet."""
    state = run([pose(side_on=False, raise_side=Side.LEFT)])
    assert state.verdict is SetupVerdict.WAITING
    assert not state.steps_done[SetupStep.ARM_RAISED]


def test_a_verdict_is_not_revised_once_reached():
    """A passed check stays passed. Otherwise lowering your arm would undo it
    and the patient would be stuck repeating a step they completed."""
    state = run([pose(), pose(raise_side=Side.LEFT), pose(raise_side=None)])
    assert state.verdict is SetupVerdict.PASSED


def test_a_swap_verdict_also_sticks():
    """Especially this one. A swap must not scroll past because the patient
    put their arm down."""
    state = run([pose(), pose(raise_side=Side.RIGHT), pose(raise_side=None)])
    assert state.verdict is SetupVerdict.SIDES_SWAPPED


def test_steps_latch_through_a_momentary_wobble():
    """Turning slightly for one frame must not send the patient back to step
    one. Steps latch; only the verdict is decisive."""
    state = run([pose(side_on=True), pose(side_on=False), pose(side_on=True)])
    assert state.steps_done[SetupStep.SIDE_ON]


def test_it_works_for_a_right_side_patient():
    state = run([pose(operated=Side.RIGHT), pose(operated=Side.RIGHT, raise_side=Side.RIGHT)],
                operated=Side.RIGHT)
    assert state.verdict is SetupVerdict.PASSED


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
    from rehab_ai.pose.setup_check import _SKELETON_BONES, _points

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
    state = run([pose(), pose(raise_side=Side.LEFT)])
    assert state.instruction(Side.LEFT) == "All set"
