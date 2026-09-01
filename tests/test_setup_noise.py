"""
Noise-robustness tests for the CP 3 gate.

These exist because the gate shipped broken once and the unit tests all passed.

The first implementation used a running MAXIMUM of ankle y as the floor
baseline. Every hand-written test fed it clean synthetic positions, so every
test agreed it worked. Against realistic pose jitter it passed 16 times out of
20 with a completely stationary foot, six of those falsely reporting
SIDES_SWAPPED.

A running maximum is maximally sensitive to the outlier it should ignore: one
downward spike drags the baseline below the real floor, and from then on every
resting frame reads as a lift.

Deterministic tests could not catch that. These are statistical: feed hundreds
of noisy frames and assert on the rate.
"""

import random

import pytest

from rehab_ai.models.session import Side
from rehab_ai.pose.setup_check import SetupChecker, SetupVerdict
from rehab_ai.pose.tracker import _SKELETON_LANDMARKS

W, H = 640, 480
FLOOR_Y = 0.90
LIFTED_Y = 0.78  # a real heel lift, ~12% of frame height


class _Landmark:
    def __init__(self, x, y, visibility):
        self.x, self.y, self.visibility = x, y, visibility


class _Landmarks:
    def __init__(self, mapping):
        size = max(int(i) for i in mapping) + 1
        self.landmark = [_Landmark(0.5, 0.5, 0.0) for _ in range(size)]
        for index, (x, y, vis) in mapping.items():
            self.landmark[int(index)] = _Landmark(x, y, vis)


def frame(left_ankle_y: float, right_ankle_y: float) -> _Landmarks:
    """Side-on, left leg nearest, only the ankles vary."""
    mapping = {}
    for side, vis, ankle_y in (
        (Side.LEFT, 0.95, left_ankle_y),
        (Side.RIGHT, 0.45, right_ankle_y),
    ):
        k = _SKELETON_LANDMARKS[side]
        mapping.update({
            k["ear"]: (0.44, 0.16, vis),
            k["shoulder"]: (0.46, 0.30, vis),
            k["elbow"]: (0.44, 0.44, vis),
            k["wrist"]: (0.42, 0.56, vis),
            k["hip"]: (0.48, 0.58, vis),
            k["knee"]: (0.58, 0.72, vis),
            k["ankle"]: (0.60, ankle_y, vis),
            k["heel"]: (0.58, ankle_y + 0.03, vis),
            k["toe"]: (0.66, ankle_y + 0.03, vis),
        })
    return _Landmarks(mapping)


def stationary_trial(noise_sd: float, seed: int, frames: int = 300) -> SetupVerdict:
    """Both feet planted for the whole run. Only jitter moves them."""
    random.seed(seed)
    checker = SetupChecker(Side.LEFT)
    for _ in range(frames):
        state = checker.update(
            frame(
                FLOOR_Y + random.gauss(0, noise_sd),
                FLOOR_Y + random.gauss(0, noise_sd),
            ),
            W,
            H,
        )
        if state.verdict is not SetupVerdict.WAITING:
            return state.verdict
    return SetupVerdict.WAITING


def lift_trial(noise_sd: float, seed: int, lift_side: Side = Side.LEFT) -> SetupVerdict:
    """Settle, then genuinely lift one heel and hold it."""
    random.seed(seed)
    checker = SetupChecker(Side.LEFT)

    def step(left_y, right_y):
        return checker.update(
            frame(left_y + random.gauss(0, noise_sd), right_y + random.gauss(0, noise_sd)),
            W,
            H,
        )

    for _ in range(60):  # settle, both feet down
        step(FLOOR_Y, FLOOR_Y)

    for _ in range(40):  # hold the lift
        left = LIFTED_Y if lift_side is Side.LEFT else FLOOR_Y
        right = LIFTED_Y if lift_side is Side.RIGHT else FLOOR_Y
        state = step(left, right)
        if state.verdict is not SetupVerdict.WAITING:
            return state.verdict
    return SetupVerdict.WAITING


# --------------------------------------------------------------------------
# A stationary foot must NEVER pass, at any realistic noise level
# --------------------------------------------------------------------------


@pytest.mark.parametrize("noise_sd", [0.002, 0.005, 0.010, 0.020, 0.030])
def test_jitter_alone_never_passes_the_gate(noise_sd):
    """THE regression test.

    The running-maximum baseline passed 16/20 of these at sd=0.020. Zero is
    the only acceptable number: a gate that can be satisfied by noise is a gate
    that tells you nothing, and it latches, so it only has to be fooled once.
    """
    verdicts = [stationary_trial(noise_sd, seed) for seed in range(25)]
    passed = [v for v in verdicts if v is not SetupVerdict.WAITING]

    assert not passed, (
        f"{len(passed)}/25 stationary trials returned a verdict at sd={noise_sd}: "
        f"{[v.value for v in passed]}"
    )


def test_jitter_never_produces_a_false_swap_alarm():
    """A false SIDES_SWAPPED is worse than a false pass. It would send someone
    debugging a mirroring problem that does not exist, and the honest response
    to it is to stop and not record any sessions."""
    for seed in range(25):
        assert stationary_trial(0.020, seed) is not SetupVerdict.SIDES_SWAPPED


# --------------------------------------------------------------------------
# A real lift must still pass through the same noise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("noise_sd", [0.002, 0.010, 0.020])
def test_a_genuine_lift_still_passes(noise_sd):
    """Robustness that rejects everything is not robustness. The gate has to
    stay usable by a patient actually doing what it asked."""
    verdicts = [lift_trial(noise_sd, seed) for seed in range(25)]
    passed = [v for v in verdicts if v is SetupVerdict.PASSED]

    assert len(passed) == 25, (
        f"only {len(passed)}/25 genuine lifts passed at sd={noise_sd}"
    )


def test_a_genuine_lift_of_the_wrong_leg_still_reports_a_swap():
    """The detection that matters must survive the noise hardening."""
    for seed in range(15):
        assert lift_trial(0.010, seed, lift_side=Side.RIGHT) is SetupVerdict.SIDES_SWAPPED


# --------------------------------------------------------------------------
# The baseline itself
# --------------------------------------------------------------------------


def test_the_baseline_ignores_a_single_extreme_outlier():
    """The exact failure mode. One spike must not move the floor."""
    from rehab_ai.pose.setup_check import _SideTrack

    track = _SideTrack()
    for _ in range(40):
        track.observe(400.0, clearance=20.0)

    clean = track.baseline_y
    track.observe(999.0, clearance=20.0)  # a wild downward spike

    assert abs(track.baseline_y - clean) < 1.0, "one outlier moved the floor"


def test_the_baseline_still_follows_a_real_position_change():
    """Robust, but not frozen. A patient who shifts their chair must not be
    measured against where they used to be."""
    from rehab_ai.pose.setup_check import _BASELINE_WINDOW, _SideTrack

    track = _SideTrack()
    for _ in range(_BASELINE_WINDOW):
        track.observe(400.0, clearance=20.0)
    for _ in range(_BASELINE_WINDOW):
        track.observe(430.0, clearance=20.0)

    assert track.baseline_y == pytest.approx(430.0, abs=1.0)


def test_no_baseline_is_reported_before_enough_samples():
    """Deciding where the floor is from three frames would be guessing."""
    from rehab_ai.pose.setup_check import _SideTrack

    track = _SideTrack()
    for _ in range(3):
        assert track.observe(400.0, clearance=20.0) is False
    assert track.baseline_y is None


# --------------------------------------------------------------------------
# Replay path -- a clip must go through the identical capture contract
# --------------------------------------------------------------------------


def test_video_replay_shares_the_camera_capture_contract():
    """A replay path that mirrored differently from the live path would
    'reproduce' bugs that do not exist, and hide ones that do."""
    from rehab_ai.camera.capture import CameraSource, VideoFileSource

    assert issubclass(VideoFileSource, CameraSource)
    assert "build_frame" not in VideoFileSource.__dict__, (
        "VideoFileSource overrides build_frame -- the mirroring contract would fork"
    )


def test_video_replay_refuses_a_mirroring_capture_config():
    """Inherited from CameraSource, and worth pinning: replaying a clip through
    a mirrored config would flip left and right just as the live path would."""
    from rehab_ai.camera.capture import VideoFileSource
    from rehab_ai.rules.loader import CaptureRules

    bad = CaptureRules(
        mirror_before_inference=True, mirror_display_only=True,
        target_fps=25, pose_model_complexity=1,
    )
    with pytest.raises(ValueError, match="mirror_before_inference"):
        VideoFileSource(bad, "clip.mp4")


def test_a_missing_clip_fails_loudly():
    from rehab_ai.camera.capture import CameraError, VideoFileSource
    from rehab_ai.rules.loader import load_rules

    with pytest.raises(CameraError, match="could not open video file"):
        VideoFileSource(load_rules().capture, "does-not-exist.mp4").open()
