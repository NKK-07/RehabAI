"""
app.py
Entry point.

    python -m rehab_ai.app

STARTUP ORDER IS DELIBERATE
===========================
Everything that can fail is checked before the first screen appears, and each
failure is reported with what to do about it:

    1. rules file        loads and validates, or refuses to guess
    2. cue clips         present, or tells you to render them
    3. local model       reachable, installed, AND warmed
    4. database          opens

Step 3 matters most. Loading a 2B model takes roughly thirty seconds on CPU;
left unwarmed, that cost lands on the summary screen at the end of a session
in front of an audience. Paying it here, with a progress line, is honest.

The app refuses to start rather than degrading. There is no mode where it runs
without the model and substitutes canned text for its output -- that would be a
fabricated result, which is the one thing this project does not do.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta

from rehab_ai.audio.player import CueAudioError, CuePlayer, NullCuePlayer
from rehab_ai.explain.phrasing import ExplainUnavailable, health_check
from rehab_ai.models.session import Procedure, Profile, Side
from rehab_ai.rules.loader import RulesError, load_rules
from rehab_ai.storage.repository import SessionRepository


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rehab_ai.app", description="RehabAI - on-device rehab companion"
    )
    parser.add_argument(
        "--side",
        choices=[s.value for s in Side],
        default=Side.LEFT.value,
        help="which knee was operated on (patient profile attribute)",
    )
    parser.add_argument(
        "--procedure", choices=[p.value for p in Procedure], default=Procedure.TKA.value
    )
    parser.add_argument(
        "--days-post-op", type=int, default=21, help="days since the operation"
    )
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    parser.add_argument(
        "--no-audio", action="store_true", help="run without cue playback"
    )
    parser.add_argument(
        "--check", action="store_true", help="run the startup checks and exit"
    )
    return parser.parse_args(argv)


def preflight(args: argparse.Namespace):
    """Run every startup check. Returns (rules, cue_player) or exits.

    Each failure prints the fix. A startup error that only says what went wrong
    costs the reader a search; one that says what to type does not.
    """
    print("RehabAI starting\n")

    # 1 -- rules
    try:
        rules = load_rules()
    except RulesError as exc:
        print(f"  rules file       FAILED\n\n{exc}\n")
        raise SystemExit(1)
    tuned = "tuned" if rules.is_tuned else "UNTUNED (placeholder thresholds)"
    print(f"  rules file       ok  ({tuned})")

    # 2 -- cue clips
    if args.no_audio:
        cue_player = NullCuePlayer()
        print("  cue clips        skipped (--no-audio)")
    else:
        try:
            cue_player = CuePlayer(rules.cue)
        except CueAudioError as exc:
            print(f"  cue clips        FAILED\n\n{exc}\n")
            raise SystemExit(1)
        print(f"  cue clips        ok  ({len(cue_player.available_cues)} loaded)")

    # 3 -- the local model, loaded not just present
    print(f"  local model      loading {rules.explain.model} ...", end="", flush=True)
    started = time.perf_counter()
    try:
        health_check(rules.explain)
    except ExplainUnavailable as exc:
        print(f" FAILED\n\n{exc}\n")
        raise SystemExit(1)
    print(f" ok  ({time.perf_counter() - started:.1f}s)")

    return rules, cue_player


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rules, cue_player = preflight(args)

    repository = SessionRepository().open()
    print(f"  database         ok  ({repository.count()} sessions)\n")

    if args.check:
        print("all checks passed")
        repository.close()
        return 0

    profile = Profile(
        name="Patient",
        procedure=Procedure(args.procedure),
        operated_side=Side(args.side),
        surgery_date=date.today() - timedelta(days=args.days_post_op),
    )

    from PySide6.QtWidgets import QApplication  # imported late: preflight first

    from rehab_ai.ui.main_window import MainWindow
    from rehab_ai.ui.theme import STYLESHEET

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    window = MainWindow(profile, rules, repository, cue_player, args.camera)
    window.show()

    try:
        return app.exec()
    finally:
        repository.close()


if __name__ == "__main__":
    raise SystemExit(main())
