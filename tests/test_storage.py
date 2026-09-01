"""
Tests for local persistence (CP 7).

Gate: "write a session, restart the app, read it back with every field intact
-- including observation quality and reason codes, not just the headline
numbers."

The restart is simulated by closing the repository and opening a fresh one
against the same file, which is the part that actually catches serialisation
mistakes. An in-memory round trip would pass while the on-disk format was
quietly wrong.
"""

from datetime import datetime, timedelta

import pytest

from rehab_ai.models.session import (
    CompensationStatus,
    Decision,
    InputQuality,
    InputSource,
    LockDecision,
    PainReport,
    ReasonCode,
    RehabSession,
    RepResult,
    RepValidity,
    SessionStatus,
    Side,
    SwellingComparison,
    SwellingComparisonStatus,
    SwellingReport,
)
from rehab_ai.storage.repository import SCHEMA, SessionRepository


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "sessions.db"


@pytest.fixture
def repo(db_path):
    with SessionRepository(db_path) as repository:
        yield repository


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def rep(index=0, validity=RepValidity.VALID, compensating=False, cue=False) -> RepResult:
    return RepResult(
        rep_index=index,
        side=Side.LEFT,
        validity=validity,
        compensating=compensating,
        peak_hip_drive=None if compensating is None else 0.42,
        descent_control=None if compensating is None else 0.87,
        frames_observed=38,
        frames_total=40,
        cue_fired=cue,
        started_at=1.5,
        duration_s=2.4,
    )


def session(session_id="s1", when=None) -> RehabSession:
    started = when or datetime(2026, 9, 1, 9, 30)
    return RehabSession(
        session_id=session_id,
        operated_side=Side.LEFT,
        protocol_day=21,
        status=SessionStatus.COMPLETED,
        camera_facing_side=Side.LEFT,
        pain=PainReport(4, InputSource.VOICE),
        swelling=SwellingComparison(SwellingComparisonStatus.AVAILABLE, SwellingReport.PUFFIER),
        reps=[
            rep(0, RepValidity.VALID, False),
            rep(1, RepValidity.DEGRADED, True, cue=True),
            rep(2, RepValidity.INVALID, None),
        ],
        started_at=started,
        ended_at=started + timedelta(minutes=2),
    )


def decision() -> LockDecision:
    return LockDecision(
        decision=Decision.LOCK_LOADED,
        reason_codes=[ReasonCode.PAIN_WITHIN_RANGE, ReasonCode.SWELLING_INCREASED],
        input_quality=InputQuality(
            pain_present=True,
            swelling_status=SwellingComparisonStatus.AVAILABLE,
            compensation_status=CompensationStatus.AVAILABLE,
            session_status=SessionStatus.COMPLETED,
        ),
    )


# --------------------------------------------------------------------------
# THE GATE: survive a restart
# --------------------------------------------------------------------------


def test_a_session_survives_a_restart_intact(db_path):
    """Close the repository, open a new one against the same file. This is the
    part that catches serialisation mistakes an in-memory test would miss."""
    with SessionRepository(db_path) as writer:
        writer.save(session(), decision())

    with SessionRepository(db_path) as reader:
        loaded = reader.load("s1")

    assert loaded is not None
    assert loaded.operated_side is Side.LEFT
    assert loaded.protocol_day == 21
    assert loaded.status is SessionStatus.COMPLETED
    assert loaded.camera_facing_side is Side.LEFT
    assert loaded.pain.value == 4
    assert loaded.pain.source is InputSource.VOICE
    assert loaded.swelling.status is SwellingComparisonStatus.AVAILABLE
    assert loaded.swelling.report is SwellingReport.PUFFIER
    assert loaded.started_at == datetime(2026, 9, 1, 9, 30)


def test_observation_quality_survives_a_restart(db_path):
    """Not just the headline numbers. Without these columns a session read back
    months later cannot tell 'we watched and it was clean' from 'we could not
    watch' -- collapsing the distinction the type system upstream exists to
    preserve."""
    with SessionRepository(db_path) as writer:
        writer.save(session())

    with SessionRepository(db_path) as reader:
        reps = reader.load("s1").reps

    assert [r.validity for r in reps] == [
        RepValidity.VALID,
        RepValidity.DEGRADED,
        RepValidity.INVALID,
    ]
    assert reps[0].compensating is False
    assert reps[1].compensating is True
    assert reps[2].compensating is None  # unobserved stays unobserved
    assert reps[2].peak_hip_drive is None
    assert reps[0].frames_observed == 38
    assert reps[1].cue_fired is True


def test_reason_codes_survive_a_restart(db_path):
    """A lock a clinician cannot audit later is not auditable at all."""
    with SessionRepository(db_path) as writer:
        writer.save(session(), decision())

    with SessionRepository(db_path) as reader:
        loaded = reader.load_decision("s1")

    assert loaded.decision is Decision.LOCK_LOADED
    assert loaded.reason_codes == [ReasonCode.PAIN_WITHIN_RANGE, ReasonCode.SWELLING_INCREASED]
    assert loaded.input_quality.compensation_status is CompensationStatus.AVAILABLE
    assert loaded.input_quality.is_complete


def test_enums_are_stored_as_strings_not_ordinals(db_path):
    """An ordinal silently changes meaning the moment someone reorders an enum.
    The string survives, and the Kotlin build reads the same values."""
    with SessionRepository(db_path) as writer:
        writer.save(session(), decision())

    with SessionRepository(db_path) as reader:
        row = reader._db.execute("SELECT * FROM sessions WHERE session_id='s1'").fetchone()

    assert row["operated_side"] == "left"
    assert row["status"] == "completed"
    assert row["swelling_status"] == "available"


# --------------------------------------------------------------------------
# The privacy promise is enforced by the schema
# --------------------------------------------------------------------------


def test_the_schema_declares_no_binary_column(db_path):
    """TRD.md 4 promises the database never stores video or audio. That is a
    claim made to patients, so it is enforced by there being nowhere to put
    such data rather than by remembering not to write it.

    Checked against the live schema rather than the SQL text: every column is
    TEXT, INTEGER or REAL, and none is BLOB. `frames_observed` is a count, not
    a frame -- which is exactly why a substring search for "frame" would be the
    wrong test.
    """
    allowed = {"TEXT", "INTEGER", "REAL"}

    with SessionRepository(db_path) as repo:
        tables = [
            row["name"]
            for row in repo._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        for table in tables:
            for column in repo._db.execute(f"PRAGMA table_info({table})").fetchall():
                assert column["type"].upper() in allowed, (
                    f"{table}.{column['name']} is {column['type']}, "
                    "which could hold frames or audio"
                )


def test_no_column_stores_a_path_to_media(db_path):
    """A TEXT column holding a filename would route around the BLOB check --
    the database would not contain the video, but it would point at it."""
    with SessionRepository(db_path) as repo:
        names = {
            column["name"].lower()
            for table in ("sessions", "reps", "decisions")
            for column in repo._db.execute(f"PRAGMA table_info({table})").fetchall()
        }

    for suspicious in ("path", "file", "clip", "recording", "url", "uri"):
        assert not any(suspicious in name for name in names), (
            f"a column name contains {suspicious!r}"
        )


# --------------------------------------------------------------------------
# Abandoned sessions leave a row
# --------------------------------------------------------------------------


def test_an_abandoned_session_is_still_recorded(db_path):
    """It has to be. policy/ needs to be told the session was incomplete rather
    than inferring it from fields that happen to be empty."""
    abandoned = session("s-abandoned")
    abandoned.status = SessionStatus.ABANDONED
    abandoned.ended_at = None

    with SessionRepository(db_path) as writer:
        writer.save(abandoned)

    with SessionRepository(db_path) as reader:
        loaded = reader.load("s-abandoned")

    assert loaded.status is SessionStatus.ABANDONED
    assert loaded.ended_at is None


def test_saving_the_same_session_twice_updates_rather_than_duplicates(repo):
    """The session screen writes as it goes, so an in-progress session is saved
    repeatedly. It must not accumulate rows."""
    in_progress = session()
    in_progress.status = SessionStatus.ACTIVE
    in_progress.reps = [rep(0)]
    repo.save(in_progress)

    in_progress.status = SessionStatus.COMPLETED
    in_progress.reps = [rep(0), rep(1)]
    repo.save(in_progress)

    assert repo.count() == 1
    assert len(repo.load("s1").reps) == 2
    assert repo.load("s1").status is SessionStatus.COMPLETED


# --------------------------------------------------------------------------
# Empty states
# --------------------------------------------------------------------------


def test_an_empty_database_reports_nothing_rather_than_failing(repo):
    """The recovery sheet with zero sessions is a real screen, not an error."""
    assert repo.count() == 0
    assert repo.recent() == []
    assert repo.load("nope") is None
    assert repo.load_decision("nope") is None


def test_a_session_with_no_reps_round_trips(db_path):
    """The patient checked in and then quit before standing up once."""
    empty = session("s-norep")
    empty.reps = []
    with SessionRepository(db_path) as writer:
        writer.save(empty)
    with SessionRepository(db_path) as reader:
        assert reader.load("s-norep").reps == []


def test_a_session_with_no_pain_reading_round_trips(db_path):
    """Absent stays absent across the round trip -- it must not come back as 0,
    which is a valid pain score meaning 'none at all'."""
    no_pain = session("s-nopain")
    no_pain.pain = None
    with SessionRepository(db_path) as writer:
        writer.save(no_pain)
    with SessionRepository(db_path) as reader:
        assert reader.load("s-nopain").pain is None


# --------------------------------------------------------------------------
# History, for the sheet and the swelling comparison
# --------------------------------------------------------------------------


def test_recent_returns_newest_first(repo):
    base = datetime(2026, 9, 1, 9, 0)
    for i in range(5):
        repo.save(session(f"s{i}", when=base + timedelta(days=i)))

    recent = repo.recent()
    assert [s.session_id for s in recent] == ["s4", "s3", "s2", "s1", "s0"]


def test_recent_respects_its_limit(repo):
    base = datetime(2026, 9, 1, 9, 0)
    for i in range(10):
        repo.save(session(f"s{i}", when=base + timedelta(days=i)))
    assert len(repo.recent(limit=3)) == 3


def test_previous_session_finds_the_one_immediately_before(repo):
    base = datetime(2026, 9, 1, 9, 0)
    repo.save(session("mon", when=base))
    repo.save(session("wed", when=base + timedelta(days=2)))

    previous = repo.previous_session(base + timedelta(days=2), Side.LEFT)
    assert previous.session_id == "mon"


def test_previous_session_returns_the_session_not_a_verdict(repo):
    """So the caller can see the actual date and decide for itself whether it
    was yesterday. A boolean here would hide the gap that makes a skipped day
    NO_COMPARISON rather than AVAILABLE."""
    base = datetime(2026, 9, 1, 9, 0)
    repo.save(session("monday", when=base))

    previous = repo.previous_session(base + timedelta(days=2), Side.LEFT)
    gap_days = (base + timedelta(days=2) - previous.started_at).days
    assert gap_days == 2  # caller can now choose NO_COMPARISON


def test_no_previous_session_on_the_first_day(repo):
    repo.save(session("first", when=datetime(2026, 9, 1, 9, 0)))
    assert repo.previous_session(datetime(2026, 9, 1, 8, 0), Side.LEFT) is None
