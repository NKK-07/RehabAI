"""
Tests for the sit-to-stand detector (CP 5) and the cue latch (CP 6).

Driven entirely by synthetic observation sequences -- lists of numbers, no
camera, no pose model. Every state transition, quality state and latch case is
reachable deterministically, which is the reason the detector takes
Observations rather than frames.

The two gate cases the checkpoint names explicitly:
  * a seated start must NOT fire a success event on frame one (the old
    exercises.py bug), and
  * ten threshold crossings inside one rep must produce exactly one cue.
"""

import pytest

from rehab_ai.detection.sit_to_stand import SitToStandDetector
from rehab_ai.models.session import (
    JointAngles,
    Observation,
    ObservationQuality,
    RepPhase,
    RepValidity,
    Side,
)
from rehab_ai.rules.loader import load_rules

DT = 0.04  # 25 fps


@pytest.fixture(scope="module")
def rules():
    return load_rules()


@pytest.fixture
def detector(rules):
    return SitToStandDetector(Side.LEFT, rules)


# --------------------------------------------------------------------------
# synthetic sequence builders
# --------------------------------------------------------------------------


def observation(t: float, knee: float, hip: float, *, visible: bool = True) -> Observation:
    if not visible:
        return Observation.unobservable(timestamp=t, visibility=0.1)
    return Observation(
        timestamp=t,
        quality=ObservationQuality.GOOD,
        visibility=0.92,
        angles=JointAngles(hip=hip, knee=knee),
    )


def ramp(start: float, end: float, n: int) -> list[float]:
    if n <= 1:
        return [end]
    step = (end - start) / (n - 1)
    return [start + step * i for i in range(n)]


def cycle(
    *,
    rise_hip_start: float = 95.0,
    rise_hip_end: float = 130.0,
    rise_frames: int = 20,
    seated_frames: int = 4,
) -> list[tuple[float, float]]:
    """One full seated -> rise -> stand -> descend -> seated cycle.

    Returns (knee, hip) pairs. The default profile is knee-dominant: the knee
    opens 85 degrees while the hip changes 35, so the ratio stays well under
    the flag threshold.
    """
    frames: list[tuple[float, float]] = []
    frames += [(90.0, rise_hip_start)] * seated_frames
    knees = ramp(90.0, 175.0, rise_frames)
    hips = ramp(rise_hip_start, rise_hip_end, rise_frames)
    frames += list(zip(knees, hips))
    frames += [(175.0, rise_hip_end)] * 3
    frames += list(zip(ramp(175.0, 90.0, 14), ramp(rise_hip_end, rise_hip_start, 14)))
    frames += [(90.0, rise_hip_start)] * 2
    return frames


def from_deltas(
    knee_deltas: list[float], hip_deltas: list[float], knee0: float, hip0: float
) -> list[tuple[float, float]]:
    """Build frames from explicit per-frame deltas, for precise signal control."""
    knee, hip = knee0, hip0
    out = [(knee, hip)]
    for dk, dh in zip(knee_deltas, hip_deltas):
        knee += dk
        hip += dh
        out.append((knee, hip))
    return out


def run(detector: SitToStandDetector, frames, *, invisible: set[int] | None = None):
    """Feed frames, collect every update."""
    invisible = invisible or set()
    updates = []
    for i, (knee, hip) in enumerate(frames):
        obs = observation(i * DT, knee, hip, visible=i not in invisible)
        updates.append(detector.update(obs))
    return updates


def rising_indices(frames, rules) -> list[int]:
    """Frame indices that fall inside the rise.

    Derived from the knee trace rather than hardcoded, so occlusion tests
    always land inside the rep. Hardcoded indices silently drift the moment the
    cycle shape or the phase thresholds change, and an occlusion injected
    before the rep starts tests nothing at all.
    """
    return [
        i
        for i, (knee, _) in enumerate(frames)
        if rules.phase.rise_onset_knee_angle <= knee < rules.phase.standing_min_knee_angle
    ]


def reps(updates):
    return [u.completed_rep for u in updates if u.completed_rep is not None]


def cues(updates):
    return [u.cue for u in updates if u.cue is not None]


# ==========================================================================
# CP 5 -- the phase machine
# ==========================================================================


def test_seated_start_fires_nothing_on_frame_one(detector):
    """THE regression case. exercises.py:71 starts in state 'rest' with
    rest_threshold=165 and direction='flex', so a patient seated at 90 degrees
    satisfies _past_rest on frame one, immediately trips _reached_target, and
    emits 'Good range of motion!' before moving at all.
    """
    first = detector.update(observation(0.0, knee=90.0, hip=95.0))

    assert first.completed_rep is None
    assert first.cue is None
    assert first.phase is RepPhase.READY
    assert detector.reps_completed == 0


def test_sitting_still_never_produces_a_rep(detector):
    updates = run(detector, [(90.0, 95.0)] * 60)
    assert reps(updates) == []
    assert detector.phase is RepPhase.READY


def test_a_full_cycle_produces_exactly_one_rep(detector):
    updates = run(detector, cycle())
    assert len(reps(updates)) == 1
    assert detector.reps_completed == 1


def test_three_cycles_produce_three_reps(detector):
    frames = cycle() + cycle() + cycle()
    updates = run(detector, frames)
    assert len(reps(updates)) == 3
    assert [r.rep_index for r in reps(updates)] == [0, 1, 2]


def test_the_rise_is_identified_as_the_rise(detector):
    """Not the return stroke. The hip-drive ratio is only meaningful during the
    rise, so a machine that labels the descent 'toward_target' would tune every
    threshold against the wrong window."""
    frames = cycle()
    phases = [u.phase for u in run(detector, frames)]

    knees = [k for k, _ in frames]
    rising_indices = [i for i, p in enumerate(phases) if p is RepPhase.RISING]
    assert rising_indices, "never entered RISING"

    # Every RISING frame must sit in the ascending half of the knee trace.
    peak = knees.index(max(knees))
    assert all(i <= peak for i in rising_indices)


def test_phases_progress_in_order(detector):
    seen = []
    for update in run(detector, cycle()):
        if not seen or seen[-1] is not update.phase:
            seen.append(update.phase)

    assert seen[0] is RepPhase.READY
    for expected in (RepPhase.RISING, RepPhase.STANDING, RepPhase.DESCENDING):
        assert expected in seen, f"never entered {expected}"
    assert seen.index(RepPhase.RISING) < seen.index(RepPhase.STANDING)
    assert seen.index(RepPhase.STANDING) < seen.index(RepPhase.DESCENDING)


def test_starting_mid_stand_does_not_invent_a_rep(detector):
    """Someone already standing when the camera opens has not done a rep. The
    detector arms only after seeing a genuinely seated frame."""
    updates = run(detector, [(175.0, 170.0)] * 10 + [(174.0, 169.0)] * 10)
    assert reps(updates) == []


# ==========================================================================
# CP 5 -- the strategy signal
# ==========================================================================


def test_knee_dominant_rise_is_not_flagged(detector):
    """Knee opens 85 degrees, hip changes 35. The quad is doing the work."""
    rep = reps(run(detector, cycle(rise_hip_start=95.0, rise_hip_end=130.0)))[0]
    assert rep.validity is RepValidity.VALID
    assert rep.compensating is False
    assert rep.peak_hip_drive < rules_trigger()


def test_hip_dominant_rise_is_flagged(detector):
    """Mid-rise the torso pitches forward: the hip term outruns the knee term."""
    frames = [(90.0, 95.0)] * 4
    frames += list(zip(ramp(90.0, 125.0, 6), ramp(95.0, 85.0, 6)))       # lift off
    frames += list(zip(ramp(125.0, 140.0, 8), ramp(85.0, 40.0, 8)))      # pitch forward
    frames += list(zip(ramp(140.0, 175.0, 8), ramp(40.0, 170.0, 8)))     # finish
    frames += [(175.0, 170.0)] * 3
    frames += list(zip(ramp(175.0, 90.0, 14), ramp(170.0, 95.0, 14)))
    frames += [(90.0, 95.0)] * 2

    rep = reps(run(detector, frames))[0]
    assert rep.compensating is True
    assert rep.peak_hip_drive >= rules_trigger()


def test_a_stationary_patient_reports_no_strategy(detector):
    """Zero movement is not zero hip-drive; it is no reading at all. Dividing
    two near-zero deltas would otherwise produce noise shaped like a verdict."""
    updates = run(detector, [(90.0, 95.0)] * 20)
    assert all(u.meter_signal == 0.0 for u in updates)


def test_meter_and_trigger_are_separate_signals(rules):
    """Different windows on purpose: the trigger must be fast, the meter must
    look steady. Equal windows would mean the split bought nothing."""
    assert rules.strategy.trigger_smoothing_frames < rules.strategy.meter_smoothing_frames


def test_descent_control_is_scored_on_a_valid_rep(detector):
    rep = reps(run(detector, cycle()))[0]
    assert rep.descent_control is not None
    assert 0.0 <= rep.descent_control <= 1.0


# ==========================================================================
# CP 5 -- observation quality flows into rep validity
# ==========================================================================


def test_brief_occlusion_degrades_but_does_not_discard(detector, rules):
    """Losing the patient for two frames is a camera problem, not a clinical
    event. The rep survives, marked degraded."""
    frames = cycle(rise_frames=30)
    rising = rising_indices(frames, rules)
    blink = set(rising[2:4])
    assert len(blink) <= rules.observation.degraded_max_consecutive_frames

    completed = reps(run(detector, frames, invisible=blink))
    assert len(completed) == 1
    assert completed[0].validity is RepValidity.DEGRADED
    assert completed[0].compensating is not None  # still assessable


def test_sustained_occlusion_invalidates_the_rep(detector, rules):
    """Losing them for a long stretch of the rise means we did not watch the
    movement. Reporting a verdict anyway would be manufacturing an observation."""
    frames = cycle(rise_frames=30)
    tolerance = rules.observation.degraded_max_consecutive_frames
    rising = rising_indices(frames, rules)
    blackout = set(rising[2 : 2 + tolerance + 4])

    completed = reps(run(detector, frames, invisible=blackout))
    assert len(completed) == 1
    assert completed[0].validity is RepValidity.INVALID


def test_an_invalid_rep_carries_no_verdict_and_no_metrics(detector, rules):
    frames = cycle(rise_frames=30)
    tolerance = rules.observation.degraded_max_consecutive_frames
    rising = rising_indices(frames, rules)
    rep = reps(run(detector, frames, invisible=set(rising[2 : 2 + tolerance + 4])))[0]

    assert rep.compensating is None
    assert rep.peak_hip_drive is None
    assert rep.descent_control is None


def test_occlusion_enters_and_leaves_the_low_visibility_phase(detector, rules):
    frames = cycle(rise_frames=30)
    updates = run(detector, frames, invisible=set(rising_indices(frames, rules)[2:4]))
    assert any(u.phase is RepPhase.LOW_VISIBILITY for u in updates)
    assert updates[-1].phase is RepPhase.READY


def test_coverage_is_reported_even_when_the_rep_survives(detector, rules):
    frames = cycle(rise_frames=30)
    rep = reps(run(detector, frames, invisible=set(rising_indices(frames, rules)[2:4])))[0]
    assert rep.frames_observed < rep.frames_total
    assert 0.0 < rep.observation_coverage < 1.0


def test_recovery_after_occlusion_resumes_the_same_rep(detector, rules):
    """Not a new one. The rep index must not advance because the camera
    blinked."""
    frames = cycle(rise_frames=30)
    updates = run(detector, frames, invisible=set(rising_indices(frames, rules)[2:4]))
    assert [r.rep_index for r in reps(updates)] == [0]


# ==========================================================================
# CP 5 -- operated-side binding
# ==========================================================================


def test_every_rep_is_labelled_with_the_operated_side(rules):
    for side in (Side.LEFT, Side.RIGHT):
        detector = SitToStandDetector(side, rules)
        rep = reps(run(detector, cycle()))[0]
        assert rep.side is side


def test_the_side_never_changes_across_a_session(rules):
    detector = SitToStandDetector(Side.RIGHT, rules)
    completed = reps(run(detector, cycle() + cycle() + cycle()))
    assert {r.side for r in completed} == {Side.RIGHT}


def test_a_non_side_operated_side_is_refused(rules):
    with pytest.raises(TypeError, match="must be a Side"):
        SitToStandDetector("left", rules)  # type: ignore[arg-type]


# ==========================================================================
# CP 6 -- the cue latch
# ==========================================================================


def rules_trigger() -> float:
    return load_rules().strategy.trigger_threshold


def chattering_rise(rules) -> list[tuple[float, float]]:
    """A rise whose trigger signal repeatedly crosses TRIGGER while staying
    above RESET -- textbook boundary chatter.

    With a 3-frame window and a saturation ratio of 2.5:
        hip delta 4.0 / knee delta 2.0  -> ratio 2.00 -> signal 0.80  (> TRIGGER)
        hip delta 3.4 / knee delta 2.0  -> ratio 1.70 -> signal 0.68  (< TRIGGER,
                                                                      > RESET)
    """
    window = rules.strategy.trigger_smoothing_frames
    knee_deltas: list[float] = []
    hip_deltas: list[float] = []
    for block in range(10):
        hot = block % 2 == 0
        knee_deltas += [2.0] * window
        hip_deltas += [-4.0 if hot else -3.4] * window

    frames = [(90.0, 95.0)] * 4
    frames += list(zip(ramp(90.0, 125.0, 5), ramp(95.0, 92.0, 5)))
    frames += from_deltas(knee_deltas, hip_deltas, knee0=125.0, hip0=92.0)[1:]
    return frames


def test_ten_crossings_inside_one_rep_produce_exactly_one_cue(detector, rules):
    """The CP 6 gate. A raw threshold would fire on every upward crossing;
    hysteresis plus the latch collapses them to one."""
    updates = run(detector, chattering_rise(rules))
    assert len(cues(updates)) == 1


def test_a_dip_below_trigger_does_not_clear_the_latch(detector, rules):
    """Explicitly not `signal < TRIGGER -> unlatch`. That is the change that
    would reintroduce the chatter this whole mechanism exists to remove."""
    updates = run(detector, chattering_rise(rules))
    fired = cues(updates)
    assert len(fired) == 1
    assert fired[0].signal >= rules.strategy.trigger_threshold


def test_the_next_rep_gets_its_own_cue(detector, rules):
    """The beat the demo is built on: corrected mid-rep, and the next rep shows
    whether it worked. A wall-clock cooldown of 2s against a 2-3s rep cycle
    would silence exactly this."""
    frames = chattering_rise(rules)
    frames += list(zip(ramp(frames[-1][0], 175.0, 6), ramp(frames[-1][1], 170.0, 6)))
    frames += list(zip(ramp(175.0, 90.0, 14), ramp(170.0, 95.0, 14)))
    frames += [(90.0, 95.0)] * 3
    frames += chattering_rise(rules)[4:]

    fired = cues(run(detector, frames))
    assert len(fired) == 2
    assert fired[0].rep_index == 0
    assert fired[1].rep_index == 1


def test_a_clean_rep_is_silent(detector):
    """Nothing to correct, nothing said. The silence after a corrected rep is
    the payoff, so it has to be real."""
    updates = run(detector, cycle(rise_hip_start=95.0, rise_hip_end=130.0))
    assert cues(updates) == []


def test_the_cue_is_a_fixed_phrase_from_the_rules_file(detector, rules):
    """Never generated at runtime. A model call could not reach the patient
    before they finished standing up."""
    fired = cues(run(detector, chattering_rise(rules)))[0]
    phrase = rules.cue.phrases["hip_dominant"]
    assert fired.text == phrase.text
    assert fired.clip == phrase.clip
    assert fired.key == "hip_dominant"


def test_the_rep_records_that_a_cue_fired(detector, rules):
    frames = chattering_rise(rules)
    frames += list(zip(ramp(frames[-1][0], 175.0, 6), ramp(frames[-1][1], 170.0, 6)))
    frames += list(zip(ramp(175.0, 90.0, 14), ramp(170.0, 95.0, 14)))
    frames += [(90.0, 95.0)] * 2

    rep = reps(run(detector, frames))[0]
    assert rep.cue_fired is True


def test_no_cue_fires_outside_the_rise(detector):
    """The signal is only meaningful during the rise. A cue during the descent
    would be correcting a movement that already finished."""
    for update in run(detector, cycle()):
        if update.cue is not None:
            assert update.phase is RepPhase.RISING


def test_updates_are_returned_every_frame(detector):
    """The UI thread renders from these; it must never have to ask the
    detector a question or wait for a rep to finish."""
    frames = cycle()
    updates = run(detector, frames)
    assert len(updates) == len(frames)
    assert all(0.0 <= u.meter_signal <= 1.0 for u in updates)
    assert all(0.0 <= u.trigger_signal <= 1.0 for u in updates)
