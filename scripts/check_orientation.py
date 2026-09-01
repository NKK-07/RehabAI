"""
check_orientation.py
The CP 3 gate: prove the app is watching the leg you think it is.

    python scripts/check_orientation.py --side left

WHY THIS EXISTS SEPARATELY FROM THE APP
=======================================
The live session screen draws the strategy meter, not the skeleton. You cannot
verify which landmarks are being tracked by looking at it -- you would be
trusting the very code you are trying to check.

This tool draws the four operated-side landmarks large and labelled, and the
other side dimmed for contrast, so the question becomes answerable by moving
and watching.

WHAT YOU ARE CHECKING
=====================
Stand or sit side-on. Raise the arm on your OPERATED side.

    the bright shoulder marker moves   -> correct, the contract holds
    the dim shoulder marker moves      -> LEFT AND RIGHT ARE SWAPPED

If they are swapped, every downstream number describes the wrong knee. Nothing
in the test suite can catch it, because from inside the code the labels are
self-consistent -- they are just attached to the wrong body.

This runs through the real CameraSource, so it verifies the actual capture
path, including the never-mirror-before-inference rule.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rehab_ai.camera.capture import CameraError, CameraSource  # noqa: E402
from rehab_ai.models.session import Side  # noqa: E402
from rehab_ai.pose.tracker import _LANDMARKS  # noqa: E402
from rehab_ai.pose_utils import PoseTracker, get_visibility, get_xy  # noqa: E402
from rehab_ai.rules.loader import load_rules  # noqa: E402

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BRIGHT = (90, 230, 120)   # operated side  (BGR)
_DIM = (110, 110, 110)     # the other side
_INK = (245, 245, 245)
_WARN = (60, 130, 235)


def draw_side(frame, landmarks, side: Side, w: int, h: int, *, bright: bool) -> None:
    colour = _BRIGHT if bright else _DIM
    radius = 11 if bright else 6
    thickness = -1 if bright else 2

    points = {}
    for name, index in _LANDMARKS[side].items():
        x, y = get_xy(landmarks, index, w, h)
        points[name] = (int(x), int(y))
        cv2.circle(frame, points[name], radius, colour, thickness)

    for a, b in (("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle")):
        cv2.line(frame, points[a], points[b], colour, 3 if bright else 1)

    if bright:
        for name, point in points.items():
            visibility = get_visibility(landmarks, _LANDMARKS[side][name])
            cv2.putText(
                frame,
                f"{side.value} {name}  {visibility:.2f}",
                (point[0] + 16, point[1] + 5),
                _FONT,
                0.5,
                _BRIGHT,
                1,
                cv2.LINE_AA,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify operated-side tracking (CP 3)")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    operated = Side(args.side)
    other = operated.other
    rules = load_rules()

    print(f"\n  CP 3 -- operated-side check   ({operated.value} knee)\n")
    print("  1. Sit or stand side-on to the camera.")
    print(f"  2. Raise the arm on your {operated.value.upper()} side.")
    print("  3. Watch the markers:")
    print("       BRIGHT GREEN moves  -> correct")
    print("       DIM GREY moves      -> LEFT AND RIGHT ARE SWAPPED\n")
    print("  Press  q  to quit.\n")

    pose = PoseTracker(model_complexity=rules.capture.pose_model_complexity)

    try:
        with CameraSource(rules.capture, args.camera) as camera:
            while True:
                frame = camera.read(time.perf_counter())
                result = pose.process(frame.rgb_for_inference)

                # Draw on the UNMIRRORED frame -- the same one the pose model
                # saw.
                #
                # This matters. Landmark pixel coordinates come from the
                # inference frame; drawing them onto the mirrored display frame
                # would place every marker at a flipped x position. The overlay
                # would look wrong even when the contract is correct, and you
                # could not tell that apart from left and right actually being
                # swapped -- which is the single question this tool exists to
                # answer.
                #
                # So the preview here is deliberately NOT mirrored. It will
                # feel slightly odd; that is the cost of an unambiguous answer.
                canvas = frame.for_inference.copy()

                if result.pose_landmarks is None:
                    cv2.putText(
                        canvas, "no person detected", (24, 44), _FONT, 0.8, _WARN, 2, cv2.LINE_AA
                    )
                else:
                    draw_side(canvas, result.pose_landmarks, other, frame.width, frame.height, bright=False)
                    draw_side(canvas, result.pose_landmarks, operated, frame.width, frame.height, bright=True)

                cv2.putText(
                    canvas,
                    f"OPERATED: {operated.value.upper()}  (bright)",
                    (24, 40),
                    _FONT, 0.75, _BRIGHT, 2, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"raise your {operated.value} arm - the BRIGHT marker should move",
                    (24, 72),
                    _FONT, 0.6, _INK, 1, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    "raw camera view, NOT mirrored - same frame the pose model sees",
                    (24, frame.height - 20),
                    _FONT, 0.5, _DIM, 1, cv2.LINE_AA,
                )

                cv2.imshow("RehabAI - CP 3 operated-side check", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except CameraError as exc:
        print(f"  camera error: {exc}")
        return 1
    finally:
        pose.close()
        cv2.destroyAllWindows()

    print("\n  Did the BRIGHT markers track your operated side?")
    print("    yes -> CP 3 passes.")
    print("    no  -> left/right are swapped. Do not proceed; every downstream")
    print("           number would describe the wrong knee.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
