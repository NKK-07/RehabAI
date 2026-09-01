"""
repository.py
Local SQLite persistence. No frames, no audio, ever.

WHAT IS STORED AND WHAT IS NOT
==============================
Stored:  pain score, swelling tap, per-rep summaries, lock decisions with
         their reason codes, timestamps, observation quality.
Never:   video frames, audio buffers, landmark coordinates, photographs.

TRD.md 4 says the database holds "session history, never raw video". That is a
promise made to patients on a slide, so it is enforced here by the schema
having nowhere to put such data -- not by remembering not to write it.

SCHEMA PARITY WITH ROOM
=======================
The Kotlin build uses Room over the same shape. Column names and types are
chosen to port directly:

    sessions        one row per day
      └── reps      many rows per session   (FK session_id)
      └── decision  one row per session     (FK session_id)

Enums are stored as their string values, not ordinals. An ordinal would
silently change meaning the moment someone reorders an enum; the string
survives.

WHY QUALITY IS PERSISTED
========================
A stored rep carries validity and observation coverage, so a session read back
months later can still distinguish "we watched and it was clean" from "we could
not watch". Dropping those columns would collapse the distinction the whole
type system upstream exists to preserve.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

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

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    operated_side       TEXT NOT NULL,
    protocol_day        INTEGER NOT NULL,
    status              TEXT NOT NULL,
    camera_facing_side  TEXT,
    pain_value          INTEGER,
    pain_source         TEXT,
    swelling_status     TEXT NOT NULL,
    swelling_report     TEXT,
    started_at          TEXT,
    ended_at            TEXT
);

CREATE TABLE IF NOT EXISTS reps (
    session_id          TEXT NOT NULL,
    rep_index           INTEGER NOT NULL,
    side                TEXT NOT NULL,
    validity            TEXT NOT NULL,
    compensating        INTEGER,
    peak_hip_drive      REAL,
    descent_control     REAL,
    frames_observed     INTEGER NOT NULL,
    frames_total        INTEGER NOT NULL,
    cue_fired           INTEGER NOT NULL,
    started_at          REAL NOT NULL,
    duration_s          REAL NOT NULL,
    PRIMARY KEY (session_id, rep_index),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS decisions (
    session_id                  TEXT PRIMARY KEY,
    decision                    TEXT NOT NULL,
    reason_codes                TEXT NOT NULL,
    q_pain_present              INTEGER NOT NULL,
    q_swelling_status           TEXT NOT NULL,
    q_compensation_status       TEXT NOT NULL,
    q_session_status            TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
"""

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "sessions.db"


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class SessionRepository:
    """Reads and writes sessions. Owns the connection lifecycle.

    Usage:
        with SessionRepository() as repo:
            repo.save(session, decision)
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_DB_PATH
        self._connection: sqlite3.Connection | None = None

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> "SessionRepository":
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
        connection.commit()
        self._connection = connection
        return self

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "SessionRepository":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("repository is not open; call open() first")
        return self._connection

    # -- writes -------------------------------------------------------------

    def save(self, session: RehabSession, decision: LockDecision | None = None) -> None:
        """Persist a session, its reps and its decision, atomically.

        Upserts on session_id so re-saving an in-progress session is safe --
        the session screen writes as it goes, and an abandoned session must
        still leave a row behind.
        """
        swelling = session.swelling or SwellingComparison(
            SwellingComparisonStatus.UNAVAILABLE
        )

        with self._db:  # transaction
            self._db.execute(
                """
                INSERT INTO sessions (
                    session_id, operated_side, protocol_day, status,
                    camera_facing_side, pain_value, pain_source,
                    swelling_status, swelling_report, started_at, ended_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    status             = excluded.status,
                    camera_facing_side = excluded.camera_facing_side,
                    pain_value         = excluded.pain_value,
                    pain_source        = excluded.pain_source,
                    swelling_status    = excluded.swelling_status,
                    swelling_report    = excluded.swelling_report,
                    started_at         = excluded.started_at,
                    ended_at           = excluded.ended_at
                """,
                (
                    session.session_id,
                    session.operated_side.value,
                    session.protocol_day,
                    session.status.value,
                    session.camera_facing_side.value if session.camera_facing_side else None,
                    session.pain.value if session.pain else None,
                    session.pain.source.value if session.pain else None,
                    swelling.status.value,
                    swelling.report.value if swelling.report else None,
                    _iso(session.started_at),
                    _iso(session.ended_at),
                ),
            )

            self._db.execute("DELETE FROM reps WHERE session_id = ?", (session.session_id,))
            self._db.executemany(
                """
                INSERT INTO reps (
                    session_id, rep_index, side, validity, compensating,
                    peak_hip_drive, descent_control, frames_observed,
                    frames_total, cue_fired, started_at, duration_s
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session.session_id,
                        rep.rep_index,
                        rep.side.value,
                        rep.validity.value,
                        None if rep.compensating is None else int(rep.compensating),
                        rep.peak_hip_drive,
                        rep.descent_control,
                        rep.frames_observed,
                        rep.frames_total,
                        int(rep.cue_fired),
                        rep.started_at,
                        rep.duration_s,
                    )
                    for rep in session.reps
                ],
            )

            if decision is not None:
                self._db.execute(
                    """
                    INSERT INTO decisions (
                        session_id, decision, reason_codes, q_pain_present,
                        q_swelling_status, q_compensation_status, q_session_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        decision              = excluded.decision,
                        reason_codes          = excluded.reason_codes,
                        q_pain_present        = excluded.q_pain_present,
                        q_swelling_status     = excluded.q_swelling_status,
                        q_compensation_status = excluded.q_compensation_status,
                        q_session_status      = excluded.q_session_status
                    """,
                    (
                        session.session_id,
                        decision.decision.value,
                        ",".join(code.value for code in decision.reason_codes),
                        int(decision.input_quality.pain_present),
                        decision.input_quality.swelling_status.value,
                        decision.input_quality.compensation_status.value,
                        decision.input_quality.session_status.value,
                    ),
                )

    # -- reads --------------------------------------------------------------

    def load(self, session_id: str) -> RehabSession | None:
        row = self._db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return self._session_from_row(row)

    def load_decision(self, session_id: str) -> LockDecision | None:
        row = self._db.execute(
            "SELECT * FROM decisions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return LockDecision(
            decision=Decision(row["decision"]),
            reason_codes=[ReasonCode(c) for c in row["reason_codes"].split(",") if c],
            input_quality=InputQuality(
                pain_present=bool(row["q_pain_present"]),
                swelling_status=SwellingComparisonStatus(row["q_swelling_status"]),
                compensation_status=CompensationStatus(row["q_compensation_status"]),
                session_status=SessionStatus(row["q_session_status"]),
            ),
        )

    def recent(self, limit: int = 30) -> list[RehabSession]:
        """Most recent first. Feeds the recovery sheet."""
        rows = self._db.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC NULLS LAST LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._session_from_row(row) for row in rows]

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]

    def previous_session(self, before: datetime, operated_side: Side) -> RehabSession | None:
        """The session immediately before `before`.

        Used to decide whether a swelling comparison is even possible: the
        caller checks whether this session was yesterday, and passes
        NO_COMPARISON when it was not. Deliberately returns the session rather
        than a boolean, so the caller can see the actual date rather than
        trusting this method's idea of "adjacent".
        """
        row = self._db.execute(
            """
            SELECT * FROM sessions
            WHERE started_at IS NOT NULL AND started_at < ? AND operated_side = ?
            ORDER BY started_at DESC LIMIT 1
            """,
            (before.isoformat(), operated_side.value),
        ).fetchone()
        return self._session_from_row(row) if row else None

    # -- mapping ------------------------------------------------------------

    def _session_from_row(self, row: sqlite3.Row) -> RehabSession:
        session = RehabSession(
            session_id=row["session_id"],
            operated_side=Side(row["operated_side"]),
            protocol_day=row["protocol_day"],
            status=SessionStatus(row["status"]),
            camera_facing_side=Side(row["camera_facing_side"])
            if row["camera_facing_side"]
            else None,
            pain=PainReport(row["pain_value"], InputSource(row["pain_source"]))
            if row["pain_value"] is not None
            else None,
            swelling=SwellingComparison(
                status=SwellingComparisonStatus(row["swelling_status"]),
                report=SwellingReport(row["swelling_report"])
                if row["swelling_report"]
                else None,
            ),
            started_at=_dt(row["started_at"]),
            ended_at=_dt(row["ended_at"]),
        )
        session.reps = self._reps_for(row["session_id"])
        return session

    def _reps_for(self, session_id: str) -> list[RepResult]:
        rows = self._db.execute(
            "SELECT * FROM reps WHERE session_id = ? ORDER BY rep_index", (session_id,)
        ).fetchall()
        return [
            RepResult(
                rep_index=row["rep_index"],
                side=Side(row["side"]),
                validity=RepValidity(row["validity"]),
                compensating=None if row["compensating"] is None else bool(row["compensating"]),
                peak_hip_drive=row["peak_hip_drive"],
                descent_control=row["descent_control"],
                frames_observed=row["frames_observed"],
                frames_total=row["frames_total"],
                cue_fired=bool(row["cue_fired"]),
                started_at=row["started_at"],
                duration_s=row["duration_s"],
            )
            for row in rows
        ]
