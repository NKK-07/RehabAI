"""
replay_setup.py
Run the CP 3 setup gate against a recorded clip and report what happened,
frame by frame.

    python scripts/replay_setup.py --video data/recordings/cp3.mp4 --side left

WHY
===
A live gate that stalls tells you nothing: no artefact, no history, and the
person who saw it is the same person who has to describe it. Replaying a clip
turns the same run into a table you can read, share and re-run after changing
a threshold.

WHAT IT PRINTS
==============
  * per-frame ankle position against the floor baseline, when it changes
  * where each advisory step first passed
  * the frame a verdict was reached, or the reason it never was
  * a summary of the closest the clip ever came to passing

That last line is the one that matters when a gate does not fire: it says
whether the lift was too small, too brief, or never seen at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rehab_ai.camera.capture import CameraError, VideoFileSource  # noqa: E402
from rehab_ai.models.session import Side  # noqa: E402
from rehab_ai.pose.setup_check import (  # noqa: E402
    _LIFT_CLEARANCE,
    _SUSTAIN_FRAMES,
    STEP_ORDER,
    SetupChecker,
    SetupVerdict,
)
from rehab_ai.pose_utils import PoseTracker  # noqa: E402
from rehab_ai.rules.loader import load_rules  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the CP 3 gate against a clip")
    parser.add_argument("--video", required=True)
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--verbose", action="store_true", help="print every frame")
    args = parser.parse_args()

    operated = Side(args.side)
    rules = load_rules()

    clip_path = Path(args.video)
    if not clip_path.is_file():
        print(f"\n  No clip at {clip_path}\n")
        print("  Record one first:\n")
        print("    python scripts/record_clip.py --seconds 12\n")
        return 1

    checker = SetupChecker(operated)
    pose = PoseTracker(model_complexity=rules.capture.pose_model_complexity)

    print(f"\n  Replaying {args.video}   operated side: {operated.value}")
    print(f"  clearance {_LIFT_CLEARANCE}  sustain {_SUSTAIN_FRAMES} frames\n")

    step_first_seen: dict = {}
    verdict_frame: int | None = None
    best_rise = 0.0
    best_hold = 0
    detected_frames = 0
    frames = 0
    final = None

    try:
        with VideoFileSource(rules.capture, args.video) as clip:
            while True:
                try:
                    frame = clip.read(frames / max(rules.capture.target_fps, 1))
                except CameraError:
                    break  # end of file

                frames += 1
                result = pose.process(frame.rgb_for_inference)
                state = checker.update(result.pose_landmarks, frame.width, frame.height)
                final = state

                if state.person_detected:
                    detected_frames += 1

                for step in STEP_ORDER:
                    if state.steps_done.get(step) and step not in step_first_seen:
                        step_first_seen[step] = frames
                        print(f"  frame {frames:>4}   step passed: {step.value}")

                rise_text = state.diagnostics.get("ankle rise", "")
                if rise_text:
                    try:
                        best_rise = max(best_rise, float(rise_text.split()[0]))
                    except ValueError:
                        pass
                held = state.diagnostics.get("held", "")
                if held:
                    try:
                        best_hold = max(best_hold, int(held.split("/")[0]))
                    except ValueError:
                        pass

                if args.verbose and state.diagnostics:
                    joined = "  ".join(f"{k}={v}" for k, v in state.diagnostics.items())
                    print(f"  frame {frames:>4}   {joined}")

                if state.verdict is not SetupVerdict.WAITING and verdict_frame is None:
                    verdict_frame = frames
                    print(f"\n  frame {frames:>4}   VERDICT: {state.verdict.value.upper()}\n")
    finally:
        pose.close()

    print(f"  ── summary ──────────────────────────────────────────")
    print(f"  frames read            {frames}")
    print(f"  person detected in     {detected_frames} ({_pct(detected_frames, frames)})")
    for step in STEP_ORDER:
        seen = step_first_seen.get(step)
        print(f"  {step.value:<22} {'frame ' + str(seen) if seen else 'NEVER'}")
    print(f"  largest ankle rise     {best_rise:+.3f}  (need >= {_LIFT_CLEARANCE})")
    print(f"  longest hold           {best_hold} frames  (need {_SUSTAIN_FRAMES})")

    if verdict_frame is not None:
        print(f"\n  RESULT: {final.verdict.value.upper()} at frame {verdict_frame}\n")
        return 0

    print(f"\n  RESULT: no verdict — {_why(best_rise, best_hold, detected_frames)}\n")
    return 1


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "0%"


def _why(best_rise: float, best_hold: int, detected: int) -> str:
    """Name the single most likely reason, so the next change is obvious."""
    if detected == 0:
        return "no person detected in any frame"
    if best_rise <= 0.0:
        return "the operated ankle never rose at all — wrong leg, or not visible"
    if best_rise < _LIFT_CLEARANCE:
        return (
            f"the heel came up {best_rise:.3f} but needs {_LIFT_CLEARANCE} — "
            "lift higher, or lower the clearance"
        )
    return (
        f"high enough ({best_rise:.3f}) but only held {best_hold}/{_SUSTAIN_FRAMES} "
        "frames — hold it longer, or shorten the sustain window"
    )


if __name__ == "__main__":
    raise SystemExit(main())
