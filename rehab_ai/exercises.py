"""
exercises.py
Rep-counting state machines for each supported exercise, plus real-time
form-feedback rules.

Each exercise watches one primary joint angle that moves between a
"rest" position and a "target" position. Two exercises move in opposite
angular directions -- a squat's knee angle *decreases* as the patient
goes deeper, while a shoulder raise's arm angle *increases* as the
patient lifts -- so each exercise declares a `direction` and the shared
state machine handles both without duplicating logic.

A rep only counts if the patient actually reached the target angle
before returning to rest, so a shallow / "cheated" rep doesn't get
credit -- matching how a physio would count reps by eye.

Scope is intentionally narrow: this flags simple, well-studied,
single-plane failure modes (depth/range, gross alignment, tempo). It
does not attempt to diagnose compensatory movement patterns or make any
clinical judgment -- see README for why that boundary is deliberate.
"""

import time
from dataclasses import dataclass, field

from rehab_ai.pose_utils import PoseLandmark, get_xy, get_visibility, calculate_angle, best_side

LM = PoseLandmark
FEEDBACK_COOLDOWN_S = 2.0


@dataclass
class FeedbackEvent:
    text: str
    kind: str = "info"     # "info" | "good" | "warn"
    speak: bool = False    # whether this should also be spoken aloud


@dataclass
class RepResult:
    exercise: str
    good_form: bool
    peak_angle: float
    duration_s: float
    timestamp: float = field(default_factory=time.time)


class ExerciseBase:
    """Generic rep-counting state machine.

    Subclasses set:
      direction        "flex"   -> angle decreases from rest to target (e.g. squat)
                        "extend" -> angle increases from rest to target (e.g. shoulder raise)
      rest_threshold    angle at the relaxed/starting position
      target_threshold  angle that counts as "full" range of motion
      margin            how far past rest_threshold counts as "movement started"
    """

    name = "base"
    direction = "flex"
    rest_threshold = 160.0
    target_threshold = 100.0
    margin = 10.0
    min_visibility = 0.5
    max_safe_velocity = 220.0  # deg/sec

    def __init__(self, target_reps: int = 10):
        self.target_reps = target_reps
        self.reps_completed = 0
        self.reps_shallow = 0
        self.state = "rest"        # "rest" | "toward_target" | "at_target" | "toward_rest"
        self.rep_started_at = None
        self.reached_target_depth = False
        self.peak_progress_angle = None  # angle closest to target seen this rep
        self._last_angle = None
        self._last_angle_time = None
        self.last_feedback_time = 0.0
        self.results: list[RepResult] = []

    # --- to override -----------------------------------------------------
    def primary_angle(self, landmarks, frame_w, frame_h, side):
        raise NotImplementedError

    def form_checks(self, landmarks, frame_w, frame_h, side, angle):
        """Return extra FeedbackEvents for form issues beyond range of
        motion (e.g. joint alignment). Subclasses override."""
        return []

    # --- direction-aware helpers ------------------------------------------
    def _past_rest(self, angle: float) -> bool:
        if self.direction == "flex":
            return angle < self.rest_threshold - self.margin
        return angle > self.rest_threshold + self.margin

    def _reached_target(self, angle: float) -> bool:
        if self.direction == "flex":
            return angle <= self.target_threshold
        return angle >= self.target_threshold

    def _left_target(self, angle: float) -> bool:
        if self.direction == "flex":
            return angle > self.target_threshold + self.margin / 2
        return angle < self.target_threshold - self.margin / 2

    def _back_at_rest(self, angle: float) -> bool:
        if self.direction == "flex":
            return angle >= self.rest_threshold
        return angle <= self.rest_threshold

    def _progress_better(self, angle: float) -> bool:
        """True if `angle` is closer to the target than the best seen so far."""
        if self.peak_progress_angle is None:
            return True
        if self.direction == "flex":
            return angle < self.peak_progress_angle
        return angle > self.peak_progress_angle

    def _throttled(self, now: float) -> bool:
        if now - self.last_feedback_time > FEEDBACK_COOLDOWN_S:
            self.last_feedback_time = now
            return True
        return False

    # --- shared state machine --------------------------------------------
    def update(self, landmarks, frame_w, frame_h):
        """Feed one frame's landmarks in. Returns (angle_or_None, [FeedbackEvent])."""
        events: list[FeedbackEvent] = []
        side = best_side(landmarks, LM.LEFT_HIP, LM.RIGHT_HIP)

        angle = self.primary_angle(landmarks, frame_w, frame_h, side)
        if angle is None:
            return None, events

        now = time.time()
        angular_velocity = None
        if self._last_angle is not None and self._last_angle_time is not None:
            dt = now - self._last_angle_time
            if dt > 0:
                angular_velocity = abs(angle - self._last_angle) / dt
        self._last_angle = angle
        self._last_angle_time = now

        if self.state == "rest":
            if self._past_rest(angle):
                self.state = "toward_target"
                self.rep_started_at = now
                self.reached_target_depth = False
                self.peak_progress_angle = angle

        elif self.state == "toward_target":
            if self._progress_better(angle):
                self.peak_progress_angle = angle
            if self._reached_target(angle):
                self.reached_target_depth = True
                self.state = "at_target"
                events.append(FeedbackEvent("Good range of motion!", kind="good"))

        elif self.state == "at_target":
            if self._left_target(angle):
                self.state = "toward_rest"

        elif self.state == "toward_rest":
            if self._progress_better(angle):
                self.peak_progress_angle = angle
            if self._back_at_rest(angle):
                duration = now - (self.rep_started_at or now)
                good = self.reached_target_depth
                if good:
                    self.reps_completed += 1
                    events.append(FeedbackEvent(
                        f"Rep {self.reps_completed}/{self.target_reps} complete!",
                        kind="good", speak=True))
                else:
                    self.reps_shallow += 1
                    events.append(FeedbackEvent(
                        "Shallow rep - try for more range next time",
                        kind="warn", speak=True))
                self.results.append(RepResult(
                    exercise=self.name, good_form=good,
                    peak_angle=self.peak_progress_angle, duration_s=duration))
                self.state = "rest"

        if angular_velocity is not None and angular_velocity > self.max_safe_velocity:
            if self._throttled(now):
                events.append(FeedbackEvent("Slow down - control the movement", kind="warn"))

        events.extend(self.form_checks(landmarks, frame_w, frame_h, side, angle))
        return angle, events

    @property
    def is_complete(self) -> bool:
        return self.reps_completed >= self.target_reps


class SquatExercise(ExerciseBase):
    """Bodyweight squat / sit-to-stand. Tracks knee flexion angle
    (hip-knee-ankle). Rest = standing (~165 deg), target = a solid rehab
    working depth (~110 deg) -- not a deep athletic squat."""

    name = "Squat"
    direction = "flex"
    rest_threshold = 165.0
    target_threshold = 110.0
    margin = 10.0
    max_safe_velocity = 220.0

    def primary_angle(self, landmarks, frame_w, frame_h, side):
        hip = LM.LEFT_HIP if side == "left" else LM.RIGHT_HIP
        knee = LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE
        ankle = LM.LEFT_ANKLE if side == "left" else LM.RIGHT_ANKLE
        if min(get_visibility(landmarks, hip), get_visibility(landmarks, knee),
               get_visibility(landmarks, ankle)) < self.min_visibility:
            return None
        a = get_xy(landmarks, hip, frame_w, frame_h)
        b = get_xy(landmarks, knee, frame_w, frame_h)
        c = get_xy(landmarks, ankle, frame_w, frame_h)
        return calculate_angle(a, b, c)

    def form_checks(self, landmarks, frame_w, frame_h, side, angle):
        events = []
        now = time.time()
        knee = LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE
        ankle = LM.LEFT_ANKLE if side == "left" else LM.RIGHT_ANKLE
        knee_xy = get_xy(landmarks, knee, frame_w, frame_h)
        ankle_xy = get_xy(landmarks, ankle, frame_w, frame_h)
        # Coarse knee-over-toe cue (front/side camera assumed) -- not a
        # biomechanics-grade measurement, just a directional nudge.
        if self.state in ("toward_target", "at_target") and abs(knee_xy[0] - ankle_xy[0]) > 0.35 * frame_w:
            if self._throttled(now):
                events.append(FeedbackEvent("Keep your knee over your foot", kind="warn"))

        shoulder = LM.LEFT_SHOULDER if side == "left" else LM.RIGHT_SHOULDER
        hip = LM.LEFT_HIP if side == "left" else LM.RIGHT_HIP
        if get_visibility(landmarks, shoulder) >= self.min_visibility:
            s_xy = get_xy(landmarks, shoulder, frame_w, frame_h)
            h_xy = get_xy(landmarks, hip, frame_w, frame_h)
            k_xy = get_xy(landmarks, knee, frame_w, frame_h)
            torso_angle = calculate_angle(s_xy, h_xy, k_xy)
            if self.state in ("toward_target", "at_target") and torso_angle < 130:
                if self._throttled(now):
                    events.append(FeedbackEvent("Keep your chest up, don't lean forward", kind="warn"))
        return events


class ShoulderRaiseExercise(ExerciseBase):
    """Shoulder abduction / forward raise. Tracks arm elevation angle
    (hip-shoulder-elbow). Rest = arm at side (~20 deg), target = raised
    to roughly shoulder height (~100 deg) -- a standard early-stage
    rotator-cuff / frozen-shoulder ROM milestone."""

    name = "Shoulder Raise"
    direction = "extend"
    rest_threshold = 20.0
    target_threshold = 100.0
    margin = 10.0
    max_safe_velocity = 200.0

    def primary_angle(self, landmarks, frame_w, frame_h, side):
        hip = LM.LEFT_HIP if side == "left" else LM.RIGHT_HIP
        shoulder = LM.LEFT_SHOULDER if side == "left" else LM.RIGHT_SHOULDER
        elbow = LM.LEFT_ELBOW if side == "left" else LM.RIGHT_ELBOW
        if min(get_visibility(landmarks, hip), get_visibility(landmarks, shoulder),
               get_visibility(landmarks, elbow)) < self.min_visibility:
            return None
        a = get_xy(landmarks, hip, frame_w, frame_h)
        b = get_xy(landmarks, shoulder, frame_w, frame_h)
        c = get_xy(landmarks, elbow, frame_w, frame_h)
        return calculate_angle(a, b, c)

    def form_checks(self, landmarks, frame_w, frame_h, side, angle):
        events = []
        now = time.time()
        # A bent elbow during a raise usually means the patient is
        # compensating with a shoulder shrug instead of true abduction.
        shoulder = LM.LEFT_SHOULDER if side == "left" else LM.RIGHT_SHOULDER
        elbow = LM.LEFT_ELBOW if side == "left" else LM.RIGHT_ELBOW
        wrist = LM.LEFT_WRIST if side == "left" else LM.RIGHT_WRIST
        if get_visibility(landmarks, wrist) >= self.min_visibility:
            s_xy = get_xy(landmarks, shoulder, frame_w, frame_h)
            e_xy = get_xy(landmarks, elbow, frame_w, frame_h)
            w_xy = get_xy(landmarks, wrist, frame_w, frame_h)
            elbow_angle = calculate_angle(s_xy, e_xy, w_xy)
            if self.state in ("toward_target", "at_target", "toward_rest") and elbow_angle < 140:
                if self._throttled(now):
                    events.append(FeedbackEvent("Try to keep your elbow straighter", kind="warn"))
        return events


EXERCISES = {
    "squat": SquatExercise,
    "shoulder_raise": ShoulderRaiseExercise,
}
