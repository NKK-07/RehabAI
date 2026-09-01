"""
record_clip.py
Record a short clip from the camera, for replaying the setup gate and for
tuning thresholds against real footage.

    python scripts/record_clip.py --seconds 12 --out data/recordings/cp3.mp4

WHY RECORD
==========
A live camera makes the setup gate impossible to debug. It runs once, in front
of a person, and leaves no artefact. When it stalls you are guessing at which
threshold it stalled on.

A clip makes it reproducible: replay it as many times as needed, watch the
frame-by-frame numbers, change one threshold, replay again. TRD 8 step 3 calls
for tuning the hip-drive threshold against real recorded reps, so this is the
harness that work needs, not scaffolding for this bug alone.

PRIVACY
=======
Footage holds a person's face and their home. `data/recordings/` and `*.mp4`
are already in .gitignore, and this script writes nowhere else. It records
video only -- no audio, and nothing is transmitted.

Delete clips when you are done with them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rehab_ai.camera.capture import CameraError, CameraSource  # noqa: E402
from rehab_ai.rules.loader import load_rules  # noqa: E402

DEFAULT_OUT = Path("data/recordings")


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a clip for gate replay and tuning")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    rules = load_rules()
    out = Path(args.out) if args.out else DEFAULT_OUT / f"clip-{int(time.time())}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Recording {args.seconds:.0f}s to {out}")
    print("  Do the whole sequence: sit side-on, operated leg nearest,")
    print("  then lift that heel and hold it for a couple of seconds.")
    print("  Press q to stop early.\n")

    writer = None
    frames = 0
    started = time.perf_counter()

    try:
        with CameraSource(rules.capture, args.camera) as camera:
            while True:
                elapsed = time.perf_counter() - started
                if elapsed >= args.seconds:
                    break

                frame = camera.read(time.perf_counter())

                if writer is None:
                    writer = cv2.VideoWriter(
                        str(out),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        rules.capture.target_fps,
                        (frame.width, frame.height),
                    )
                    if not writer.isOpened():
                        print(f"  could not open a writer for {out}")
                        return 1

                # The UNMIRRORED frame is what gets written, so a replay feeds
                # the pose model exactly what the live run did.
                writer.write(frame.for_inference)
                frames += 1

                preview = frame.for_display.copy()
                remaining = max(0.0, args.seconds - elapsed)
                cv2.putText(preview, f"REC  {remaining:4.1f}s", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 235), 2, cv2.LINE_AA)
                cv2.imshow("RehabAI - recording", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except CameraError as exc:
        print(f"  camera error: {exc}")
        return 1
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    seconds = time.perf_counter() - started
    print(f"\n  {frames} frames, {seconds:.1f}s -> {out}")
    print(f"  Replay the gate against it:\n")
    print(f"    python scripts/check_orientation.py --video {out} --report\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
