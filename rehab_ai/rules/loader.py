"""
loader.py
Loads and validates rules/thresholds.v1.json -- the handoff artifact between
this prototype and the Kotlin build.

Why a file rather than constants in code: the hip-drive threshold is the one
number in this system with no literature behind it (TRD.md 5). It gets tuned
against real footage, and then the mobile app has to use the *same* number. A
constant retyped into Kotlin at hour 20 of a 30-hour build is indistinguishable
from a tuning decision when it turns out to be wrong.

    rules/thresholds.v1.json
             │
             ├──▶ Python prototype  (this loader)
             └──▶ Kotlin app        (bundled asset, same schema)

Validation is strict and fails loudly. A loader that silently accepts a
malformed file is worse than no loader: you would ship the defaults and never
know the tuning was lost.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "rules" / "thresholds.v1.json"


class RulesError(Exception):
    """Raised for any problem loading or validating the rules file.

    Deliberately not a warning and not a silent fallback to defaults -- see the
    module docstring.
    """


# ---------------------------------------------------------------------------
# Typed views over the file
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseRules:
    seated_max_knee_angle: float
    rise_onset_knee_angle: float
    standing_min_knee_angle: float
    descent_onset_knee_angle: float
    min_rise_duration_s: float
    max_rise_duration_s: float


@dataclass(frozen=True)
class ObservationRules:
    good_min_visibility: float
    degraded_min_visibility: float
    degraded_max_consecutive_frames: int
    invalid_min_unobserved_fraction: float


@dataclass(frozen=True)
class StrategyRules:
    hip_drive_saturation_ratio: float
    trigger_threshold: float
    reset_threshold: float
    trigger_smoothing_frames: int
    meter_smoothing_frames: int
    rep_flag_min_signal: float


@dataclass(frozen=True)
class DescentRules:
    min_control_score: float
    smoothing_frames: int


@dataclass(frozen=True)
class CuePhrase:
    text: str
    clip: str


@dataclass(frozen=True)
class CueRules:
    max_latency_ms: int
    phrases: dict[str, CuePhrase]


@dataclass(frozen=True)
class PolicyRules:
    pain_lock_threshold: int
    pain_rest_only_threshold: int
    compensation_flag_rate_lock: float
    early_protocol_days: int
    copy: dict[str, str]


@dataclass(frozen=True)
class CaptureRules:
    mirror_before_inference: bool
    mirror_display_only: bool
    target_fps: int
    pose_model_complexity: int


@dataclass(frozen=True)
class ExplainRules:
    provider: str
    endpoint: str
    model: str
    timeout_s: int
    max_tokens: int


@dataclass(frozen=True)
class Rules:
    schema_version: int
    tuned_against: str
    tuned_on: str | None
    phase: PhaseRules
    observation: ObservationRules
    strategy: StrategyRules
    descent: DescentRules
    cue: CueRules
    policy: PolicyRules
    capture: CaptureRules
    explain: ExplainRules

    @property
    def is_tuned(self) -> bool:
        """False while the file still holds placeholder values.

        The UI surfaces this so an untuned build cannot be mistaken for a
        tuned one during a demo -- which would be a fabricated result wearing
        a real result's clothes.
        """
        return self.tuned_on is not None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in data:
        raise RulesError(f"missing required section: {name!r}")
    section = data[name]
    if not isinstance(section, dict):
        raise RulesError(f"section {name!r} must be an object, got {type(section).__name__}")
    return section


def _num(section: dict[str, Any], path: str, key: str, *, cast=float):
    if key not in section:
        raise RulesError(f"missing required key: {path}.{key}")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RulesError(f"{path}.{key} must be a number, got {value!r}")
    return cast(value)


def _flag(section: dict[str, Any], path: str, key: str) -> bool:
    if key not in section:
        raise RulesError(f"missing required key: {path}.{key}")
    value = section[key]
    if not isinstance(value, bool):
        raise RulesError(f"{path}.{key} must be true or false, got {value!r}")
    return value


def _text(section: dict[str, Any], path: str, key: str) -> str:
    if key not in section:
        raise RulesError(f"missing required key: {path}.{key}")
    value = section[key]
    if not isinstance(value, str) or not value.strip():
        raise RulesError(f"{path}.{key} must be a non-empty string, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_rules(path: Path | str | None = None) -> Rules:
    """Read, parse and validate the rules file.

    Raises RulesError on anything wrong. There is no fallback to built-in
    defaults, by design: silently running on defaults after someone broke the
    file is how tuned thresholds get lost without anyone noticing.
    """
    rules_path = Path(path) if path is not None else DEFAULT_RULES_PATH

    if not rules_path.is_file():
        raise RulesError(f"rules file not found: {rules_path}")

    try:
        raw = json.loads(rules_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RulesError(f"{rules_path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise RulesError(f"{rules_path} must contain a JSON object at the top level")

    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise RulesError(
            f"schema_version {version!r} is not supported (this build expects {SCHEMA_VERSION}). "
            "Refusing to guess at a file written for a different schema."
        )

    phase_raw = _section(raw, "phase")
    phase = PhaseRules(
        seated_max_knee_angle=_num(phase_raw, "phase", "seated_max_knee_angle"),
        rise_onset_knee_angle=_num(phase_raw, "phase", "rise_onset_knee_angle"),
        standing_min_knee_angle=_num(phase_raw, "phase", "standing_min_knee_angle"),
        descent_onset_knee_angle=_num(phase_raw, "phase", "descent_onset_knee_angle"),
        min_rise_duration_s=_num(phase_raw, "phase", "min_rise_duration_s"),
        max_rise_duration_s=_num(phase_raw, "phase", "max_rise_duration_s"),
    )

    obs_raw = _section(raw, "observation")
    observation = ObservationRules(
        good_min_visibility=_num(obs_raw, "observation", "good_min_visibility"),
        degraded_min_visibility=_num(obs_raw, "observation", "degraded_min_visibility"),
        degraded_max_consecutive_frames=_num(
            obs_raw, "observation", "degraded_max_consecutive_frames", cast=int
        ),
        invalid_min_unobserved_fraction=_num(
            obs_raw, "observation", "invalid_min_unobserved_fraction"
        ),
    )

    strat_raw = _section(raw, "strategy")
    strategy = StrategyRules(
        hip_drive_saturation_ratio=_num(strat_raw, "strategy", "hip_drive_saturation_ratio"),
        trigger_threshold=_num(strat_raw, "strategy", "trigger_threshold"),
        reset_threshold=_num(strat_raw, "strategy", "reset_threshold"),
        trigger_smoothing_frames=_num(strat_raw, "strategy", "trigger_smoothing_frames", cast=int),
        meter_smoothing_frames=_num(strat_raw, "strategy", "meter_smoothing_frames", cast=int),
        rep_flag_min_signal=_num(strat_raw, "strategy", "rep_flag_min_signal"),
    )

    desc_raw = _section(raw, "descent")
    descent = DescentRules(
        min_control_score=_num(desc_raw, "descent", "min_control_score"),
        smoothing_frames=_num(desc_raw, "descent", "smoothing_frames", cast=int),
    )

    cue_raw = _section(raw, "cue")
    phrases_raw = cue_raw.get("phrases")
    if not isinstance(phrases_raw, dict) or not phrases_raw:
        raise RulesError("cue.phrases must be a non-empty object")
    phrases: dict[str, CuePhrase] = {}
    for key, entry in phrases_raw.items():
        if not isinstance(entry, dict):
            raise RulesError(f"cue.phrases.{key} must be an object")
        phrases[key] = CuePhrase(
            text=_text(entry, f"cue.phrases.{key}", "text"),
            clip=_text(entry, f"cue.phrases.{key}", "clip"),
        )
    cue = CueRules(
        max_latency_ms=_num(cue_raw, "cue", "max_latency_ms", cast=int),
        phrases=phrases,
    )

    pol_raw = _section(raw, "policy")
    copy_raw = pol_raw.get("copy")
    if not isinstance(copy_raw, dict):
        raise RulesError("policy.copy must be an object")
    policy = PolicyRules(
        pain_lock_threshold=_num(pol_raw, "policy", "pain_lock_threshold", cast=int),
        pain_rest_only_threshold=_num(pol_raw, "policy", "pain_rest_only_threshold", cast=int),
        compensation_flag_rate_lock=_num(pol_raw, "policy", "compensation_flag_rate_lock"),
        early_protocol_days=_num(pol_raw, "policy", "early_protocol_days", cast=int),
        copy={str(k): str(v) for k, v in copy_raw.items()},
    )

    cap_raw = _section(raw, "capture")
    capture = CaptureRules(
        mirror_before_inference=_flag(cap_raw, "capture", "mirror_before_inference"),
        mirror_display_only=_flag(cap_raw, "capture", "mirror_display_only"),
        target_fps=_num(cap_raw, "capture", "target_fps", cast=int),
        pose_model_complexity=_num(cap_raw, "capture", "pose_model_complexity", cast=int),
    )

    exp_raw = _section(raw, "explain")
    explain = ExplainRules(
        provider=_text(exp_raw, "explain", "provider"),
        endpoint=_text(exp_raw, "explain", "endpoint"),
        model=_text(exp_raw, "explain", "model"),
        timeout_s=_num(exp_raw, "explain", "timeout_s", cast=int),
        max_tokens=_num(exp_raw, "explain", "max_tokens", cast=int),
    )

    rules = Rules(
        schema_version=version,
        tuned_against=str(raw.get("tuned_against", "unknown")),
        tuned_on=raw.get("tuned_on"),
        phase=phase,
        observation=observation,
        strategy=strategy,
        descent=descent,
        cue=cue,
        policy=policy,
        capture=capture,
        explain=explain,
    )

    _check_coherence(rules)
    return rules


def _check_coherence(r: Rules) -> None:
    """Relationships between values that individually look fine.

    These catch the edits that produce a file which parses cleanly and then
    behaves wrongly at runtime -- the class of mistake a schema check alone
    will not find.
    """
    if r.capture.mirror_before_inference:
        raise RulesError(
            "capture.mirror_before_inference must be false. Mirroring the frame before "
            "inference swaps the pose model's LEFT/RIGHT labels, so the operated-side "
            "lock would silently measure the healthy leg."
        )

    if r.strategy.reset_threshold >= r.strategy.trigger_threshold:
        raise RulesError(
            f"strategy.reset_threshold ({r.strategy.reset_threshold}) must be BELOW "
            f"trigger_threshold ({r.strategy.trigger_threshold}). Without a gap there is no "
            "hysteresis, and a signal sitting near the line fires the cue repeatedly."
        )

    for name, value in (
        ("trigger_threshold", r.strategy.trigger_threshold),
        ("reset_threshold", r.strategy.reset_threshold),
        ("rep_flag_min_signal", r.strategy.rep_flag_min_signal),
    ):
        if not 0.0 <= value <= 1.0:
            raise RulesError(f"strategy.{name} must be within 0.0..1.0, got {value}")

    if r.observation.degraded_min_visibility >= r.observation.good_min_visibility:
        raise RulesError(
            "observation.degraded_min_visibility must be below good_min_visibility, "
            "otherwise no frame can ever be classified DEGRADED."
        )

    if r.strategy.trigger_smoothing_frames > r.strategy.meter_smoothing_frames:
        raise RulesError(
            "strategy.trigger_smoothing_frames must not exceed meter_smoothing_frames. "
            "The trigger path is latency-critical and the meter path is not; smoothing the "
            "trigger harder than the meter inverts the reason the two paths are separate."
        )

    if r.strategy.trigger_smoothing_frames < 1 or r.strategy.meter_smoothing_frames < 1:
        raise RulesError("smoothing window sizes must be at least 1 frame")

    if r.phase.rise_onset_knee_angle <= r.phase.seated_max_knee_angle:
        raise RulesError(
            "phase.rise_onset_knee_angle must be above seated_max_knee_angle, or the rise "
            "is detected before the patient has moved."
        )

    if r.phase.standing_min_knee_angle <= r.phase.rise_onset_knee_angle:
        raise RulesError(
            "phase.standing_min_knee_angle must be above rise_onset_knee_angle."
        )

    if r.policy.pain_rest_only_threshold < r.policy.pain_lock_threshold:
        raise RulesError(
            "policy.pain_rest_only_threshold must be at or above pain_lock_threshold -- "
            "rest-only is the stricter response and cannot trigger at a lower pain score."
        )

    if not 0.0 <= r.policy.compensation_flag_rate_lock <= 1.0:
        raise RulesError("policy.compensation_flag_rate_lock must be within 0.0..1.0")
