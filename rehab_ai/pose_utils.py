"""
pose_utils.py
Thin wrapper around MediaPipe's legacy Pose solution, plus the geometry
helpers every exercise needs (joint angles, visibility checks).

Uses mediapipe.python.solutions.pose (the "legacy" API). This is
intentional: the model weights ship inside the pip wheel, so pose
tracking works completely offline once `pip install -r requirements.txt`
has run. No separate model download, no network dependency at demo time.
"""

import math
import numpy as np
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

# Re-export so other modules don't need to know the mediapipe import path.
PoseLandmark = mp_pose.PoseLandmark
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS


class PoseTracker:
    """Wraps a single MediaPipe Pose instance for a live video stream."""

    def __init__(self, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_rgb: np.ndarray):
        """Run pose detection on one RGB frame. Returns MediaPipe result
        (result.pose_landmarks is None if no person was detected)."""
        frame_rgb.flags.writeable = False
        result = self._pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return result

    def draw(self, frame_bgr: np.ndarray, result) -> None:
        """Draw the skeleton overlay in place on a BGR frame."""
        if result.pose_landmarks is None:
            return
        mp_drawing.draw_landmarks(
            frame_bgr,
            result.pose_landmarks,
            POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    def close(self):
        self._pose.close()


def get_xy(landmarks, idx: "PoseLandmark", frame_w: int, frame_h: int):
    """Pixel-space (x, y) for one landmark index."""
    lm = landmarks.landmark[idx]
    return np.array([lm.x * frame_w, lm.y * frame_h])


def get_visibility(landmarks, idx: "PoseLandmark") -> float:
    return landmarks.landmark[idx].visibility


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC (at vertex b) in degrees, given three 2D points."""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def best_side(landmarks, left_idx: "PoseLandmark", right_idx: "PoseLandmark") -> str:
    """Pick whichever side (left/right) MediaPipe is more confident about,
    so the app keeps working if the patient turns slightly or one side
    is out of frame."""
    left_vis = get_visibility(landmarks, left_idx)
    right_vis = get_visibility(landmarks, right_idx)
    return "left" if left_vis >= right_vis else "right"
