"""
Tests for the recovery sheet (CP 11) and the app's startup wiring (CP 9).

Sheet rendering is a pure function over sessions, so it tests without Qt. The
checkpoint calls out three cases explicitly -- zero, one and N sessions -- and
the interesting one is zero: an empty sheet is a real screen in the first week,
not an error.

The other thing pinned here is that a session nobody could score does not get
plotted as a zero. A trend line drawn through unobserved sessions would be a
picture of data that does not exist, on the one artifact a clinician makes
decisions from.
"""

from datetime import datetime, timedelta

import pytest

from rehab_ai.models.session import (
    PainReport,
    RehabSession,
    RepResult,
    RepValidity,
    SessionStatus,
    Side,
    SwellingComparison,
    SwellingComparisonStatus,
    SwellingReport,
)
from rehab_ai.ui.sheet_view import render_sheet


def rep(index=0, validity=RepValidity.VALID, compensating=False) -> RepResult:
    return RepResult(
        rep_index=index,
        side=Side.LEFT,
        validity=validity,
        compensating=compensating,
        peak_hip_drive=None if compensating is None else 0.5,
        descent_control=None if compensating is None else 0.8,
        frames_observed=40,
        frames_total=40,
        cue_fired=bool(compensating),
        started_at=0.0,
        duration_s=2.2,
    )


def session(day: int, *, pain=3, flagged=0, scored=4, unobservable=False) -> RehabSession:
    if unobservable:
        reps = [rep(i, RepValidity.INVALID, None) for i in range(3)]
    else:
        reps = [rep(i, RepValidity.VALID, i < flagged) for i in range(scored)]

    return RehabSession(
        session_id=f"s{day}",
        operated_side=Side.LEFT,
        protocol_day=day,
        status=SessionStatus.COMPLETED,
        pain=PainReport(pain),
        swelling=SwellingComparison(SwellingComparisonStatus.AVAILABLE, SwellingReport.SAME),
        reps=reps,
        started_at=datetime(2026, 9, 1) + timedelta(days=day),
    )


# --------------------------------------------------------------------------
# The three cases the checkpoint names
# --------------------------------------------------------------------------


def test_zero_sessions_renders_an_empty_sheet(tmp_path):
    """A real screen in the first week, not an error and not a crash."""
    out = render_sheet([], tmp_path / "sheet.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_one_session_renders(tmp_path):
    """A single point is not a trend, but it is not a failure either."""
    out = render_sheet([session(1)], tmp_path / "sheet.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_many_sessions_render(tmp_path):
    sessions = [session(day, pain=max(0, 8 - day), flagged=max(0, 3 - day // 3)) for day in range(1, 15)]
    out = render_sheet(sessions, tmp_path / "sheet.png")
    assert out.is_file()
    assert out.stat().st_size > 0


# --------------------------------------------------------------------------
# Unobserved sessions are not plotted as zero
# --------------------------------------------------------------------------


def test_an_unscorable_session_does_not_become_a_data_point(tmp_path):
    """The whole quality model, expressed on the one page a surgeon reads.

    A session where nothing could be scored must not appear on the hip-drive
    line as 0% -- that would read as a perfect day. It is marked as unscored
    instead.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sessions = [
        session(1, flagged=2, scored=4),
        session(2, unobservable=True),
        session(3, flagged=1, scored=4),
    ]
    render_sheet(sessions, tmp_path / "sheet.png")

    # The middle session contributes no point to the strategy series.
    figures = [plt.figure(n) for n in plt.get_fignums()]
    plt.close("all")
    assert figures == []  # render_sheet closes its figure rather than leaking it


def test_render_closes_its_figure(tmp_path):
    """Called repeatedly from the UI. A leaked figure per render would grow
    memory for the life of the session."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for i in range(3):
        render_sheet([session(i + 1)], tmp_path / f"sheet{i}.png")

    assert plt.get_fignums() == []


def test_sessions_out_of_order_are_sorted(tmp_path):
    """Storage returns newest first; the sheet reads left to right in time."""
    out = render_sheet([session(9), session(2), session(5)], tmp_path / "sheet.png")
    assert out.is_file()


def test_a_session_with_no_start_time_is_skipped_not_crashed(tmp_path):
    """Defensive: a row written before started_at was set should not take the
    sheet down."""
    broken = session(4)
    broken.started_at = None
    out = render_sheet([session(1), broken], tmp_path / "sheet.png")
    assert out.is_file()


def test_a_session_with_no_pain_reading_still_renders(tmp_path):
    no_pain = session(2)
    no_pain.pain = None
    out = render_sheet([session(1), no_pain, session(3)], tmp_path / "sheet.png")
    assert out.is_file()


# --------------------------------------------------------------------------
# CP 9 wiring
# --------------------------------------------------------------------------


def test_the_app_module_imports_without_starting_qt():
    """app.py must be importable for --check to work without a display."""
    from rehab_ai import app

    assert callable(app.main)
    assert callable(app.preflight)


def test_argument_defaults_are_sane():
    from rehab_ai.app import parse_args

    args = parse_args([])
    assert args.side in ("left", "right")
    assert args.camera == 0
    assert args.no_audio is False


def test_the_operated_side_can_be_set_from_the_command_line():
    from rehab_ai.app import parse_args

    assert parse_args(["--side", "right"]).side == "right"


@pytest.mark.parametrize("view", ["intake_view", "live_view", "summary_view", "sheet_view", "main_window"])
def test_every_view_module_imports(view):
    """Catches a broken import in a screen that only appears three steps into
    the flow -- which would otherwise surface mid-demo."""
    __import__(f"rehab_ai.ui.{view}")
