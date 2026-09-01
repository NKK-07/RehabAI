"""
sit_to_stand.py
The core. Turns a stream of Observations into a live StrategySignal, a spoken
cue, and a per-rep summary.

Written clean-room rather than adapted from rehab_ai/exercises.py. That module
is parameterised as a squat -- rest is standing (165 degrees), target is down
(110) -- so a patient who begins seated at roughly 90 degrees satisfies its
"movement started" test on frame one and immediately trips its "good range of
motion" event, before moving at all. Its form checks are also gated to the
descent, which is the wrong half of a sit-to-stand.

THE STATE MACHINE
=================

                    ┌─────────┐
                    │  READY  │◀──────────────────────────┐
                    └────┬────┘  (armed only after a       │
        knee angle rises │        genuinely seated frame)  │
        past rise onset  │                                 │
                         ▼                                 │
                    ┌─────────┐   unobservable   ┌──────────────────┐
                    │ RISING  │─────────────────▶│  LOW_VISIBILITY  │
                    └────┬────┘◀────recovered────└────────┬─────────┘
                         │                                │ sustained
                         │                                ▼
                         │                          ┌───────────┐
                         │                          │ ABANDONED │
                         │                          └─────┬─────┘
                         ▼                                │
                   ┌──────────┐                           │
                   │ STANDING │                           │
                   └────┬─────┘                           │
                        ▼                                 │
                   ┌────────────┐                         │
                   │ DESCENDING │                         │
                   └────┬───────┘                         │
                        └──────────▶ RepResult ◀──────────┘
                                     validity: VALID / DEGRADED / INVALID

THE SIGNAL
==========
Per TRD.md 5: change in hip angle versus change in knee angle, during the rise.

    ratio  = |delta hip| / |delta knee|      over a trailing window
    signal = clamp(ratio / saturation, 0, 1) 0 = knee-dominant, 1 = hip-dominant

A knee-dominant rise opens the knee rapidly while the torso stays up, so the
knee term dominates and the signal stays low. A hip-dominant rise pitches the
torso forward and the hip term dominates.

Two windows, not one. The trigger path is latency-critical -- the cue has to
reach the patient before they finish standing up -- so it uses a short window.
The meter is watched by a human eye and should look steady, so it uses a longer
one. Smoothing the trigger as hard as the meter would spend the entire latency
budget on averaging.

THE CUE LATCH
=============
A StrategySignal cue fires on the first upward crossing of TRIGGER during a
rep. Hysteresis prevents boundary chatter; the per-rep latch guarantees at most
one cue per rep. Wall-clock cooldown does not suppress subsequent reps.

The latch clears ONLY on crossing below RESET, or on rep transition -- never on
a mere dip below TRIGGER, which is exactly what would reintroduce chatter.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from rehab_ai.models.session import (
    Observation,
    ObservationQuality,
    RepPhase,
    RepResult,
    RepValidity,
    Side,
)
from rehab_ai.rules.loader import Rules

_EPSILON = 1e-6


@dataclass(frozen=True)
class CueEvent:
    """A correction to speak. Chosen by lookup, never generated.

    The clip is pre-rendered; playback starts in single-digit milliseconds,
    leaving the latency budget to detection rather than synthesis.
    """

    key: str
    text: str
    clip: str
    rep_index: int
    timestamp: float
    signal: float


@dataclass(frozen=True)
class DetectorUpdate:
    """What one frame produced. Returned every frame, so the UI thread can
    render without asking the detector questions."""

    phase: RepPhase
    meter_signal: float
    trigger_signal: float
    quality: ObservationQuality
    cue: CueEvent | None = None
    completed_rep: RepResult | None = None


class _RepAccumulator:
    """Bookkeeping for one rep in progress.

    Kept separate from the state machine so that "what happened during this
    rep" is not tangled with "where are we now".
    """

    def __init__(self, index: int, started_at: float) -> None:
        self.index = index
        self.started_at = started_at
        self.frames_total = 0
        self.frames_observed = 0
        self.frames_degraded = 0
        self.peak_signal = 0.0
        self.abandoned = False
        self.cue_fired = False
        self.descent_speeds: list[float] = []

    def record(self, observation: Observation, signal: float) -> None:
        self.frames_total += 1
        if observation.quality is ObservationQuality.UNOBSERVABLE:
            return
        self.frames_observed += 1
        if observation.quality is ObservationQuality.DEGRADED:
            self.frames_degraded += 1
        self.peak_signal = max(self.peak_signal, signal)

    @property
    def unobserved_fraction(self) -> float:
        if self.frames_total == 0:
            return 1.0
        return 1.0 - (self.frames_observed / self.frames_total)

    def validity(self, rules: Rules) -> RepValidity:
        if self.abandoned or self.frames_observed == 0:
            return RepValidity.INVALID
        if self.unobserved_fraction >= rules.observation.invalid_min_unobserved_fraction:
            return RepValidity.INVALID
        if self.frames_degraded or self.unobserved_fraction > 0.0:
            return RepValidity.DEGRADED
        return RepValidity.VALID

    def descent_control(self) -> float | None:
        """Mean knee speed divided by peak knee speed, across the lowering.

        A controlled eccentric is steady, so mean and peak are close and the
        score approaches 1. A leg that gives way produces one large spike, the
        mean stays low, and the score collapses toward 0.

        Interpretable on purpose: a jury can be told exactly what it measures.
        """
        speeds = [s for s in self.descent_speeds if s > _EPSILON]
        if len(speeds) < 2:
            return None
        peak = max(speeds)
        mean = sum(speeds) / len(speeds)
        return min(1.0, mean / peak) if peak > _EPSILON else None


class SitToStandDetector:
    """Consumes Observations, emits DetectorUpdates.

    Bound to one leg for its lifetime. The operated side arrives from the
    session and is never reconsidered.
    """

    def __init__(self, operated_side: Side, rules: Rules) -> None:
        if not isinstance(operated_side, Side):
            raise TypeError(
                f"operated_side must be a Side, got {type(operated_side).__name__}."
            )
        self.operated_side = operated_side
        self._rules = rules

        self._phase = RepPhase.READY
        self._phase_before_occlusion = RepPhase.READY
        self._seen_seated = False
        self._consecutive_unobserved = 0

        self._rep_index = 0
        self._rep: _RepAccumulator | None = None

        self._cue_latched = False

        window = max(
            rules.strategy.trigger_smoothing_frames,
            rules.strategy.meter_smoothing_frames,
            rules.descent.smoothing_frames,
        ) + 1
        self._history: deque[Observation] = deque(maxlen=window)

        self._meter_signal = 0.0
        self._trigger_signal = 0.0

    # -- public state -------------------------------------------------------

    @property
    def phase(self) -> RepPhase:
        return self._phase

    @property
    def reps_completed(self) -> int:
        return self._rep_index

    # -- main entry point ---------------------------------------------------

    def update(self, observation: Observation) -> DetectorUpdate:
        """Feed one frame. Always returns an update; never raises on bad input."""
        usable = observation.quality.is_trustworthy and observation.angles is not None

        if usable:
            self._history.append(observation)
            self._consecutive_unobserved = 0
        else:
            self._consecutive_unobserved += 1

        self._meter_signal = self._signal(self._rules.strategy.meter_smoothing_frames)
        self._trigger_signal = self._signal(self._rules.strategy.trigger_smoothing_frames)

        if self._rep is not None:
            self._rep.record(observation, self._trigger_signal)

        cue: CueEvent | None = None
        completed: RepResult | None = None

        if not usable:
            completed = self._handle_unobserved(observation)
        else:
            cue, completed = self._advance(observation)

        return DetectorUpdate(
            phase=self._phase,
            meter_signal=self._meter_signal,
            trigger_signal=self._trigger_signal,
            quality=observation.quality,
            cue=cue,
            completed_rep=completed,
        )

    # -- occlusion ----------------------------------------------------------

    def _handle_unobserved(self, observation: Observation) -> RepResult | None:
        """Brief occlusion degrades a rep. Sustained occlusion invalidates it.

        Losing sight of the patient for two frames while they stand up is a
        camera problem, not a clinical event -- but losing them for half the
        rise means we did not watch the movement, and saying otherwise would
        be manufacturing an observation.
        """
        if self._phase in (RepPhase.READY, RepPhase.ABANDONED):
            return None

        if self._phase is not RepPhase.LOW_VISIBILITY:
            self._phase_before_occlusion = self._phase
            self._phase = RepPhase.LOW_VISIBILITY

        tolerance = self._rules.observation.degraded_max_consecutive_frames
        if self._consecutive_unobserved > tolerance:
            return self._abandon_rep(observation.timestamp)
        return None

    def _abandon_rep(self, timestamp: float) -> RepResult | None:
        if self._rep is None:
            self._phase = RepPhase.READY
            return None
        self._rep.abandoned = True
        self._phase = RepPhase.ABANDONED
        return self._finish_rep(timestamp)

    # -- state machine ------------------------------------------------------

    def _advance(self, observation: Observation) -> tuple[CueEvent | None, RepResult | None]:
        angles = observation.angles
        assert angles is not None  # guarded by the caller
        knee = angles.knee
        phase_rules = self._rules.phase

        if self._phase is RepPhase.LOW_VISIBILITY:
            # Recovered inside tolerance -- resume where we left off. The rep
            # is not discarded, it is marked degraded by the accumulator.
            self._phase = self._phase_before_occlusion

        if self._phase is RepPhase.ABANDONED:
            if knee <= phase_rules.seated_max_knee_angle:
                self._phase = RepPhase.READY
                self._seen_seated = True
            return None, None

        if self._phase is RepPhase.READY:
            if knee <= phase_rules.seated_max_knee_angle:
                self._seen_seated = True
            elif self._seen_seated and knee >= phase_rules.rise_onset_knee_angle:
                self._start_rep(observation.timestamp)
            return None, None

        if self._phase is RepPhase.RISING:
            cue = self._maybe_cue(observation.timestamp)
            if knee >= phase_rules.standing_min_knee_angle:
                self._phase = RepPhase.STANDING
            return cue, None

        if self._phase is RepPhase.STANDING:
            if knee <= phase_rules.descent_onset_knee_angle:
                self._phase = RepPhase.DESCENDING
            return None, None

        if self._phase is RepPhase.DESCENDING:
            self._record_descent_speed()
            if knee <= phase_rules.seated_max_knee_angle:
                return None, self._finish_rep(observation.timestamp)
            return None, None

        return None, None

    def _start_rep(self, timestamp: float) -> None:
        self._rep = _RepAccumulator(self._rep_index, timestamp)
        self._phase = RepPhase.RISING
        self._cue_latched = False  # a new rep is always eligible for its own cue

    def _finish_rep(self, timestamp: float) -> RepResult:
        rep = self._rep
        assert rep is not None

        validity = rep.validity(self._rules)
        compensating = (
            None
            if validity is RepValidity.INVALID
            else rep.peak_signal >= self._rules.strategy.rep_flag_min_signal
        )

        result = RepResult(
            rep_index=rep.index,
            side=self.operated_side,
            validity=validity,
            compensating=compensating,
            peak_hip_drive=None if validity is RepValidity.INVALID else rep.peak_signal,
            descent_control=None if validity is RepValidity.INVALID else rep.descent_control(),
            frames_observed=rep.frames_observed,
            frames_total=rep.frames_total,
            cue_fired=rep.cue_fired,
            started_at=rep.started_at,
            duration_s=max(0.0, timestamp - rep.started_at),
        )

        self._rep_index += 1
        self._rep = None
        self._cue_latched = False

        if rep.abandoned:
            # Do NOT re-arm here. The patient is still mid-movement -- we lost
            # sight of them, we did not watch them sit down. Arming now would
            # start a second rep partway up the same rise, and the session
            # would report two reps where a person did one.
            self._phase = RepPhase.ABANDONED
            self._seen_seated = False
        else:
            self._phase = RepPhase.READY
            self._seen_seated = True

        return result

    # -- the cue latch ------------------------------------------------------

    def _maybe_cue(self, timestamp: float) -> CueEvent | None:
        """Fire on the first upward crossing of TRIGGER within this rep.

        The latch clears on crossing below RESET or on rep transition. It does
        NOT clear on a dip below TRIGGER -- that gap between the two thresholds
        is the hysteresis, and removing it is what makes a signal sitting near
        the line fire the cue on alternate frames.
        """
        strategy = self._rules.strategy
        signal = self._trigger_signal

        if self._cue_latched:
            if signal <= strategy.reset_threshold:
                self._cue_latched = False
            return None

        if signal < strategy.trigger_threshold:
            return None

        self._cue_latched = True
        if self._rep is not None:
            self._rep.cue_fired = True

        phrase = self._rules.cue.phrases["hip_dominant"]
        return CueEvent(
            key="hip_dominant",
            text=phrase.text,
            clip=phrase.clip,
            rep_index=self._rep.index if self._rep else -1,
            timestamp=timestamp,
            signal=signal,
        )

    # -- signal -------------------------------------------------------------

    def _signal(self, window: int) -> float:
        """Hip-drive ratio over a trailing window, normalised to 0..1.

        Only meaningful during the rise; the caller gates on phase. Returns the
        previous value's floor of 0.0 when there is not yet enough history,
        rather than a spurious reading from a single frame pair.
        """
        if len(self._history) < 2:
            return 0.0

        span = min(window, len(self._history) - 1)
        newest = self._history[-1]
        oldest = self._history[-1 - span]
        if newest.angles is None or oldest.angles is None:
            return 0.0

        delta_hip = abs(newest.angles.hip - oldest.angles.hip)
        delta_knee = abs(newest.angles.knee - oldest.angles.knee)

        if delta_hip + delta_knee < _EPSILON:
            return 0.0  # not moving; no strategy to report

        ratio = delta_hip / max(delta_knee, _EPSILON)
        return _clamp(ratio / self._rules.strategy.hip_drive_saturation_ratio)

    def _record_descent_speed(self) -> None:
        if self._rep is None or len(self._history) < 2:
            return
        newest, previous = self._history[-1], self._history[-2]
        if newest.angles is None or previous.angles is None:
            return
        dt = newest.timestamp - previous.timestamp
        if dt <= _EPSILON:
            return
        self._rep.descent_speeds.append(abs(newest.angles.knee - previous.angles.knee) / dt)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
