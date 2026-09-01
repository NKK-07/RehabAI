"""
tracker.py
Pose estimation, bound to one leg.

Wraps the existing MediaPipe PoseTracker from rehab_ai/pose_utils.py, which is
kept because it works and is offline by design -- the model weights ship inside
the pip wheel, so there is no download at demo time.

WHAT IS DELIBERATELY NOT REUSED
===============================
`pose_utils.best_side()` picks whichever side the model is more confident
about. That is correct for a general fitness app and wrong here.

    best_side()  ──▶ LEFT   frame 40
                 ──▶ LEFT   frame 41
                 ──▶ RIGHT  frame 42   ← patient turned slightly
                 ──▶ RIGHT  frame 43

Mid-rep, the measurement silently moves to the other leg. The hip-drive ratio
for one rise would then be computed from two different knees.

The operated side is a patient attribute, set once in the profile and passed
down explicitly. This module takes it as a constructor argument and never
reconsiders it. If the operated leg cannot be seen, that is an UNOBSERVABLE
observation -- never a measurement of the healthy leg wearing the operated
leg's label.
"""

from __future__ import annotations

from rehab_ai.models.session import (
    JointAngles,
    Observation,
    ObservationQuality,
    Side,
)
from rehab_ai.pose_utils import PoseLandmark, calculate_angle, get_visibility, get_xy
from rehab_ai.pose_utils import PoseTracker as _MediaPipePoseTracker
from rehab_ai.rules.loader import ObservationRules

LM = PoseLandmark

# The four landmarks the detector reasons about, per side.
_LANDMARKS = {
    Side.LEFT: {
        "shoulder": LM.LEFT_SHOULDER,
        "hip": LM.LEFT_HIP,
        "knee": LM.LEFT_KNEE,
        "ankle": LM.LEFT_ANKLE,
    },
    Side.RIGHT: {
        "shoulder": LM.RIGHT_SHOULDER,
        "hip": LM.RIGHT_HIP,
        "knee": LM.RIGHT_KNEE,
        "ankle": LM.RIGHT_ANKLE,
    },
}


def classify_quality(visibility: float, rules: ObservationRules) -> ObservationQuality:
    """Turn a raw visibility measurement into a quality judgement.

    This is the boundary between the two concepts. Above it, code reasons about
    quality; below it, about a float. The thresholds live in the rules file so
    that tuning them does not require touching code.
    """
    if visibility >= rules.good_min_visibility:
        return ObservationQuality.GOOD
    if visibility >= rules.degraded_min_visibility:
        return ObservationQuality.DEGRADED
    return ObservationQuality.UNOBSERVABLE


class OperatedSideTracker:
    """Pose tracking bound to one leg for the life of the session.

    The binding chain ends here:
        Profile.operated_side -> RehabSession.operated_side -> this
    """

    def __init__(
        self,
        operated_side: Side,
        rules: ObservationRules,
        model_complexity: int = 1,
    ) -> None:
        if not isinstance(operated_side, Side):
            raise TypeError(
                f"operated_side must be a Side, got {type(operated_side).__name__}. "
                "This decides which knee every downstream number describes."
            )
        self.operated_side = operated_side
        self._rules = rules
        self._landmarks = _LANDMARKS[operated_side]
        self._pose = _MediaPipePoseTracker(model_complexity=model_complexity)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "OperatedSideTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- observation --------------------------------------------------------

    def observe(self, frame_rgb, width: int, height: int, timestamp: float) -> Observation:
        """Run pose estimation on one frame and interpret what came back.

        Always returns an Observation. A frame with no person in it, or with
        the operated leg out of shot, produces quality=UNOBSERVABLE -- which is
        a recorded fact about the session, not an error and not a gap.
        """
        result = self._pose.process(frame_rgb)

        if result.pose_landmarks is None:
            return Observation.unobservable(timestamp=timestamp, visibility=0.0)

        landmarks = result.pose_landmarks
        return self.observe_landmarks(landmarks, width, height, timestamp)

    def observe_landmarks(self, landmarks, width: int, height: int, timestamp: float) -> Observation:
        """Interpret an already-computed landmark set.

        Split out from observe() so the quality and angle logic can be tested
        against synthetic landmarks with no camera and no model.
        """
        visibilities = [
            get_visibility(landmarks, idx) for idx in self._landmarks.values()
        ]
        # The weakest joint governs. An excellent hip reading does not
        # compensate for an ankle nobody can see -- both angles need all
        # three of their points.
        visibility = min(visibilities)
        quality = classify_quality(visibility, self._rules)

        if quality is ObservationQuality.UNOBSERVABLE:
            return Observation.unobservable(timestamp=timestamp, visibility=visibility)

        shoulder = get_xy(landmarks, self._landmarks["shoulder"], width, height)
        hip = get_xy(landmarks, self._landmarks["hip"], width, height)
        knee = get_xy(landmarks, self._landmarks["knee"], width, height)
        ankle = get_xy(landmarks, self._landmarks["ankle"], width, height)

        return Observation(
            timestamp=timestamp,
            quality=quality,
            visibility=visibility,
            angles=JointAngles(
                hip=calculate_angle(shoulder, hip, knee),
                knee=calculate_angle(hip, knee, ankle),
            ),
            landmarks=landmarks,
        )
