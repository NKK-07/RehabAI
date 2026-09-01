"""
session.py
Shared vocabulary for the whole application. Every other module imports
from here, so this file is written first and changed carefully.

THE THREE-CONCEPTS RULE
=======================
These are three different things and must never be collapsed into one:

    landmark visibility  ->  raw measurement       float 0..1, from the pose model
    observation quality  ->  interpretation        can this frame be trusted?
    rep validity         ->  scoring eligibility   does this rep reach policy/?

    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  visibility     │    │  quality        │    │  validity       │
    │  0.31           │───▶│  UNOBSERVABLE   │───▶│  INVALID        │
    │  (a number)     │    │  (a judgement   │    │  (a consequence │
    │                 │    │   about a frame)│    │   for a rep)    │
    └─────────────────┘    └─────────────────┘    └─────────────────┘

The reason for the separation: "we could not see this rep" and "this rep was
clean" must never produce the same value downstream. A compensation count of
zero because nothing was observed is not the same fact as a compensation count
of zero because the patient did it perfectly, and policy/ has to be able to
tell them apart.

UNOBSERVABLE IS NOT "NOTHING HAPPENED".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Sequence


# ---------------------------------------------------------------------------
# Patient-level facts
# ---------------------------------------------------------------------------


class Side(str, Enum):
    """Anatomical side. Always the patient's own left/right, never the
    camera's. See camera/capture.py for why the raw frame is never mirrored."""

    LEFT = "left"
    RIGHT = "right"

    @property
    def other(self) -> "Side":
        return Side.RIGHT if self is Side.LEFT else Side.LEFT


class Procedure(str, Enum):
    TKA = "tka"  # total knee arthroplasty
    ACL = "acl"  # anterior cruciate ligament repair


@dataclass(frozen=True)
class Profile:
    """Set once, at first run. `operated_side` is the origin of the binding
    chain that runs Profile -> RehabSession -> detector -> RepResult.

    It is a patient attribute. It is never inferred from the image, and never
    recomputed per frame -- doing so is how you silently score the wrong leg.
    """

    name: str
    procedure: Procedure
    operated_side: Side
    surgery_date: date

    def __post_init__(self) -> None:
        if not isinstance(self.operated_side, Side):
            raise TypeError(
                f"operated_side must be a Side, got {type(self.operated_side).__name__}. "
                "Every clinically meaningful measurement depends on this being correct."
            )

    def protocol_day(self, on: date | None = None) -> int:
        """Days since surgery. Day 0 is the day of surgery."""
        return ((on or date.today()) - self.surgery_date).days


# ---------------------------------------------------------------------------
# Observation -- what the camera saw, and how much we trust it
# ---------------------------------------------------------------------------


class ObservationQuality(str, Enum):
    """Our interpretation of whether a frame can be trusted.

    Deliberately distinct from the raw visibility float that produced it: the
    float is a measurement, this is a judgement, and the thresholds separating
    them live in rules/thresholds.v1.json rather than in code.
    """

    GOOD = "good"
    DEGRADED = "degraded"
    UNOBSERVABLE = "unobservable"

    @property
    def is_trustworthy(self) -> bool:
        """GOOD and DEGRADED both carry signal; UNOBSERVABLE carries none."""
        return self is not ObservationQuality.UNOBSERVABLE


@dataclass(frozen=True)
class JointAngles:
    """The angles the detector actually reasons about, in degrees, for one
    frame, on one side.

        shoulder
           \\
            \\          hip_angle  = shoulder-hip-knee
            hip ────┐   knee_angle = hip-knee-ankle
                     \\
                     knee ────┐
                               \\
                              ankle
    """

    hip: float
    knee: float


@dataclass(frozen=True)
class Observation:
    """One frame, interpreted.

    `landmarks` is None when the frame could not be read at all. Note that
    quality is UNOBSERVABLE in that case -- not missing, not an exception.
    An unreadable frame is a fact about the session, and it is recorded.
    """

    timestamp: float
    quality: ObservationQuality
    visibility: float
    angles: JointAngles | None = None
    landmarks: object | None = None  # raw pose-model result; never persisted

    @classmethod
    def unobservable(cls, timestamp: float, visibility: float) -> "Observation":
        """The operated side could not be seen well enough to measure.

        Never substitute the contralateral leg here. A measurement of the
        healthy knee is a different measurement, not a weaker version of the
        requested one.
        """
        return cls(
            timestamp=timestamp,
            quality=ObservationQuality.UNOBSERVABLE,
            visibility=visibility,
            angles=None,
            landmarks=None,
        )


# ---------------------------------------------------------------------------
# Reps
# ---------------------------------------------------------------------------


class RepPhase(str, Enum):
    """Where the patient is within one sit-to-stand cycle.

        READY ──rise onset──▶ RISING ──────────▶ STANDING
          ▲                     │  │                 │
          │                     │  └─low visibility─┐│
          │                     │                   ▼▼
          │                     │            LOW_VISIBILITY
          │                     │             │          │
          │                     ◀──recovered──┘          │
          │                                    sustained │
          │                                              ▼
          │                                          ABANDONED
          │                                              │
          └────────────── DESCENDING ◀───────────────────┘

    RISING is the diagnostic phase -- the hip-drive ratio is only meaningful
    there. DESCENDING carries the eccentric-control signal.
    """

    READY = "ready"
    RISING = "rising"
    STANDING = "standing"
    DESCENDING = "descending"
    LOW_VISIBILITY = "low_visibility"
    ABANDONED = "abandoned"


class RepValidity(str, Enum):
    """Whether a completed rep is eligible for scoring.

    A consequence of the observation quality across the rep, not a restatement
    of it -- a rep made of mostly-GOOD frames with one DEGRADED patch is still
    VALID, and the tolerance for that lives in the rules file.
    """

    VALID = "valid"
    DEGRADED = "degraded"
    INVALID = "invalid"

    @property
    def counts_toward_compensation_rate(self) -> bool:
        """INVALID reps never reach the compensation rate. Including them
        would dilute a real signal with frames nobody actually saw."""
        return self is not RepValidity.INVALID


@dataclass(frozen=True)
class RepResult:
    """One completed sit-to-stand, as the detector saw it.

    `compensating` is None -- not False -- when validity is INVALID. There is
    no answer to "was this rep compensating" for a rep that was not observed,
    and False would be a lie shaped like data.
    """

    rep_index: int
    side: Side
    validity: RepValidity
    compensating: bool | None
    peak_hip_drive: float | None
    descent_control: float | None
    frames_observed: int
    frames_total: int
    cue_fired: bool
    started_at: float
    duration_s: float

    def __post_init__(self) -> None:
        if self.validity is RepValidity.INVALID and self.compensating is not None:
            raise ValueError(
                "An INVALID rep cannot carry a compensation verdict. "
                "Set compensating=None -- unobserved is not the same as clean."
            )

    @property
    def observation_coverage(self) -> float:
        if self.frames_total == 0:
            return 0.0
        return self.frames_observed / self.frames_total


# ---------------------------------------------------------------------------
# Check-in -- self-reported, never inferred from pixels
# ---------------------------------------------------------------------------


class InputSource(str, Enum):
    """Voice is an input method, not a separate data model. A spoken check-in
    produces exactly the fields a tapped one does."""

    TAP = "tap"
    VOICE = "voice"


@dataclass(frozen=True)
class PainReport:
    value: int  # 0..10
    source: InputSource = InputSource.TAP

    def __post_init__(self) -> None:
        if not 0 <= self.value <= 10:
            raise ValueError(f"pain must be 0..10, got {self.value}")


class SwellingReport(str, Enum):
    """Relative to yesterday. Self-reported by a three-way tap.

    Never derived from a photograph: lighting fakes it completely, so a dim
    room reads as "more swollen" regardless of the actual knee (PRD.md 3).
    """

    PUFFIER = "puffier"
    SAME = "same"
    LESS = "less"


class SwellingComparisonStatus(str, Enum):
    """Whether a swelling comparison is even possible today.

    These are four genuinely different situations, and flattening them into
    "no data" loses the distinction policy/ needs:

      BASELINE_ONLY  first session -- there is no yesterday. Expected, not missing.
      AVAILABLE      yesterday exists and is adjacent. Compare normally.
      NO_COMPARISON  a day was skipped, so the prior session is NOT yesterday.
                     Never silently compare across the gap.
      UNAVAILABLE    the patient declined or the check-in was incomplete.
    """

    BASELINE_ONLY = "baseline_only"
    AVAILABLE = "available"
    NO_COMPARISON = "no_comparison"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class SwellingComparison:
    """A status plus, only when the status permits it, a value.

    `report` must be None unless status is AVAILABLE. This is enforced rather
    than documented, because "0", "False" and "unknown" quietly becoming valid
    inputs to a deterministic rule is exactly the failure this type prevents.
    """

    status: SwellingComparisonStatus
    report: SwellingReport | None = None

    def __post_init__(self) -> None:
        if self.status is SwellingComparisonStatus.AVAILABLE and self.report is None:
            raise ValueError("status=AVAILABLE requires a report value")
        if self.status is not SwellingComparisonStatus.AVAILABLE and self.report is not None:
            raise ValueError(
                f"status={self.status.value} cannot carry a report value "
                f"(got {self.report}). Missing information must stay missing."
            )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class SessionStatus(str, Enum):
    """ABANDONED exists so that policy/ is told a session was incomplete,
    rather than having to infer it from fields that happen to be empty."""

    NOT_STARTED = "not_started"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class RehabSession:
    """One day's session. Owns everything the screens hand between each other.

    `operated_side` is copied from Profile at construction and never changes
    for the life of the session. `camera_facing_side` records what the patient
    confirmed at intake, so a session framed the wrong way round is
    identifiable afterwards instead of silently wrong.
    """

    session_id: str
    operated_side: Side
    protocol_day: int
    status: SessionStatus = SessionStatus.NOT_STARTED
    camera_facing_side: Side | None = None
    pain: PainReport | None = None
    swelling: SwellingComparison | None = None
    reps: list[RepResult] = field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None

    @classmethod
    def for_profile(cls, session_id: str, profile: Profile, on: date | None = None) -> "RehabSession":
        return cls(
            session_id=session_id,
            operated_side=profile.operated_side,
            protocol_day=profile.protocol_day(on),
        )

    @property
    def orientation_confirmed(self) -> bool:
        """True when the patient confirmed the operated side faces the camera."""
        return self.camera_facing_side is self.operated_side


# ---------------------------------------------------------------------------
# What the session hands to policy/
# ---------------------------------------------------------------------------


class CompensationStatus(str, Enum):
    AVAILABLE = "available"
    UNOBSERVABLE = "unobservable"


@dataclass(frozen=True)
class CompensationMetrics:
    reps_total: int
    reps_scored: int
    reps_flagged: int

    @property
    def flag_rate(self) -> float:
        """Over reps that were actually scored -- never over reps attempted.
        Dividing by attempts would let unobserved reps dilute a real signal."""
        if self.reps_scored == 0:
            return 0.0
        return self.reps_flagged / self.reps_scored


@dataclass(frozen=True)
class CompensationSummary:
    """What the camera contributes to the lock decision.

    status=UNOBSERVABLE with metrics=None is NOT the same as a summary showing
    zero flags. The first says "we could not watch"; the second says "we
    watched and it was clean". Manufacturing the second from the first is the
    single most dangerous thing this codebase could do.
    """

    status: CompensationStatus
    metrics: CompensationMetrics | None = None

    def __post_init__(self) -> None:
        if self.status is CompensationStatus.AVAILABLE and self.metrics is None:
            raise ValueError("status=AVAILABLE requires metrics")
        if self.status is CompensationStatus.UNOBSERVABLE and self.metrics is not None:
            raise ValueError(
                "status=UNOBSERVABLE cannot carry metrics. "
                "Do not manufacture 'no compensation' from 'not observed'."
            )

    @classmethod
    def from_reps(cls, reps: Sequence[RepResult]) -> "CompensationSummary":
        scored = [r for r in reps if r.validity.counts_toward_compensation_rate]
        if not scored:
            return cls(status=CompensationStatus.UNOBSERVABLE)
        return cls(
            status=CompensationStatus.AVAILABLE,
            metrics=CompensationMetrics(
                reps_total=len(reps),
                reps_scored=len(scored),
                reps_flagged=sum(1 for r in scored if r.compensating),
            ),
        )


@dataclass(frozen=True)
class PolicyInput:
    """Every input to the lock decision, with absence made explicit.

    There is no constructor that fills a gap with a default. If a fact is
    missing, the type that represents it says so, and policy/ has a documented
    outcome for that case.
    """

    pain: PainReport | None
    swelling: SwellingComparison
    compensation: CompensationSummary
    protocol_day: int
    session_status: SessionStatus


# ---------------------------------------------------------------------------
# What policy/ returns
# ---------------------------------------------------------------------------


class Decision(str, Enum):
    ALLOW_FULL = "allow_full"
    LOCK_LOADED = "lock_loaded"
    REST_ONLY = "rest_only"
    HOLD = "hold"  # insufficient evidence to permit loaded work


class ReasonCode(str, Enum):
    """Every reason the policy can give. An enum rather than free strings so
    that explain/ can be tested for coverage: the eval asserts each code in a
    decision is reflected in the sentence the model produced."""

    PAIN_ELEVATED = "pain_elevated"
    PAIN_WITHIN_RANGE = "pain_within_range"
    SWELLING_INCREASED = "swelling_increased"
    SWELLING_STABLE = "swelling_stable"
    SWELLING_IMPROVED = "swelling_improved"
    SWELLING_NO_BASELINE = "swelling_no_baseline"
    SWELLING_COMPARISON_UNAVAILABLE = "swelling_comparison_unavailable"
    COMPENSATION_FREQUENT = "compensation_frequent"
    COMPENSATION_INFREQUENT = "compensation_infrequent"
    MOVEMENT_OBSERVATION_INCOMPLETE = "movement_observation_incomplete"
    SESSION_INCOMPLETE = "session_incomplete"
    EARLY_PROTOCOL_DAY = "early_protocol_day"


@dataclass(frozen=True)
class InputQuality:
    """Provenance for the decision: what policy/ actually had to work with.
    Persisted alongside the decision so a past lock can be understood later
    without re-deriving it from raw session rows."""

    pain_present: bool
    swelling_status: SwellingComparisonStatus
    compensation_status: CompensationStatus
    session_status: SessionStatus

    @property
    def is_complete(self) -> bool:
        return (
            self.pain_present
            and self.swelling_status is SwellingComparisonStatus.AVAILABLE
            and self.compensation_status is CompensationStatus.AVAILABLE
            and self.session_status is SessionStatus.COMPLETED
        )


@dataclass(frozen=True)
class LockDecision:
    """The output of policy/. Deterministic, auditable, and never produced by
    a model. explain/ receives one of these and phrases it -- it cannot alter
    the decision, add a reason, or change which codes apply."""

    decision: Decision
    reason_codes: list[ReasonCode]
    input_quality: InputQuality

    def __post_init__(self) -> None:
        if not self.reason_codes:
            raise ValueError(
                "A LockDecision must state why. A decision with no reason codes "
                "cannot be explained to a patient or audited by a clinician."
            )

    @property
    def loaded_work_allowed(self) -> bool:
        return self.decision is Decision.ALLOW_FULL
