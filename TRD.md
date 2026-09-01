# RehabAI — Technical Requirements Document

**Status:** v1.0 — phone-native architecture (supersedes the earlier PySide6/laptop-desktop TRD entirely)
**Last updated:** 2026-09-01
**Companion to:** PRD.md
**Platform:** Android (Kotlin), Snapdragon NPU inference, Office Kit bridge to laptop for dev workflow + export

## 1. Why native Android, not cross-platform

Flutter/React Native would need a plugin bridge to reach the Qualcomm NPU delegate, which is documented and supported natively for Android/Kotlin (Google's own LiteRT-Qualcomm docs are Kotlin/Java) but not first-class in cross-platform frameworks. In a 30-hour build with 55% of that time phone-only (no laptop for troubleshooting a flaky plugin bridge), native Android is the lower-risk choice. This is a build-speed decision, not a strong architectural opinion — revisit if the team already has deep Flutter+native-plugin experience.

## 2. System Overview

```
                    ┌───────────────────────────────┐
                    │           Phone camera           │
                    └───────────────┬───────────────┘
                                    │ frames
                                    ▼
                    ┌───────────────────────────────┐
                    │  vision/  (on-device, NPU)        │
                    │  LiteRT + Qualcomm QNN delegate     │
                    │  pose/motion model → keypoints        │
                    └───────────────┬───────────────┘
                                    │ keypoints per frame
                                    ▼
                    ┌───────────────────────────────┐
                    │  strategy/                        │
                    │  hip-vs-knee compensation detector    │
                    │  (rule-based on kinematic ratios,       │
                    │   see §5 — not a black-box classifier)   │
                    └───────────────┬───────────────┘
                                    │ per-session compensation summary
                                    ▼
     ┌──────────────────┐   ┌───────────────────────────────┐
     │  checkin/            │   │  policy/                          │
     │  Whisper Tiny (NPU)     │──▶│  combines check-in + compensation    │
     │  pain + swelling tap/voice│   │  summary → lock/unlock decision       │
     └──────────────────┘   └───────────────┬───────────────┘
                                    │ structured decision (facts only)
                                    ▼
                    ┌───────────────────────────────┐
                    │  explain/                          │
                    │  Gemma 2B or Phi-3 (NPU)              │
                    │  phrases the decision in plain language  │
                    │  — cannot alter the decision itself         │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  ui/ (Jetpack Compose)              │
                    │  check-in · session · plan · sheet       │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  local Room/SQLite database            │
                    │  session history, never raw video          │
                    └───────────────────────────────┘

     Office Kit (vivo's existing bridge tool — screen mirror,
     clipboard, file transfer, remote control): used during Green
     Light dev time for phone-to-laptop workflow, and to export the
     recovery sheet to the laptop for a printable view. Not on the
     runtime critical path — the app is fully functional phone-only,
     matching Red Light (55% of build time, no laptop).
```

## 3. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Platform | Native Android, Kotlin | Direct, documented NPU delegate support (§1) |
| Vision inference | LiteRT + Qualcomm QNN delegate | Confirmed workflow: obtain a pretrained LiteRT model → add delegate dependency → initialize with backend config → run through the standard interpreter. Google's own assessment: "reasonably achievable" for a small team in a hackathon timeframe with an existing pretrained model ([source](https://developers.google.com/edge/litert/android/npu/qualcomm)). 6–9x speedup over CPU on supported Snapdragon chips. |
| Pose/motion model | A lightweight pose-estimation model converted to LiteRT (e.g. MoveNet or a MediaPipe pose model exported to TFLite/LiteRT) | Needs a pretrained model, not a from-scratch one — conversion + delegate wiring is the actual hackathon work, not model training |
| Voice | Whisper Tiny, on-device via NPU | Hackathon-recommended model; used for the spoken check-in, not just STT-as-a-checkbox |
| On-device LLM | Gemma 2B or Phi-3, on-device via NPU | Hackathon-recommended; used only to phrase structured facts into plain language — never to make or alter the lock decision (§6) |
| Local storage | Room (SQLite) | Session history for the recovery sheet and plan-adaptation logic; never stores video/audio, only structured session summaries |
| UI | Jetpack Compose | Standard modern Android UI toolkit; phone-shaped by construction |
| Dev/demo bridge | Office Kit (vivo, pc.vivoglobal.com) | Screen mirror + clipboard + file transfer + remote control — used for real dev workflow and sheet export (§7), not invented as a fake runtime feature |

## 4. Module Breakdown

- **`vision/`** — `PoseTracker`: wraps the LiteRT interpreter + QNN delegate; exposes per-frame keypoints. This is the one module needing actual hardware to build/test against — do this first (§8).
- **`strategy/`** — `CompensationDetector`: evaluates the kinematic ratios in §5 continuously, frame by frame, during the rise — not once at rep-end. Two outputs from the same computation: (1) a live `StrategySignal` (0.0 = knee-dominant, 1.0 = hip-dominant) pushed to the UI every frame for the on-screen meter and the instant cue trigger (§5a), and (2) a per-rep summary pushed to `policy/` once the rep completes. Deliberately rule-based on a small number of interpretable kinematic ratios, not a trained classifier — defensible in a jury Q&A ("why did it flag this rep"), buildable without a labeled dataset in the time available, and free to run every frame since it reuses the keypoints §5 already computes, no extra model call.
- **`checkin/`** — pain slider + three-way swelling tap; optional Whisper Tiny transcription of a spoken answer, mapped to the same structured fields (voice is an input method, not a separate data model).
- **`policy/`** — pure function: `(pain, swelling_tap, compensation_summary, protocol_day) → LockDecision`. No AI in this module by design — the safety-relevant decision is deterministic and auditable (see PRD §8 principle).
- **`explain/`** — takes a `LockDecision` (facts) and produces one sentence via the on-device LLM. Prompted narrowly: "phrase this decision, do not add reasons not present in the input." This is the one place an LLM touches the safety path, and only as a phrasing layer.
- **`session.py`-equivalent (`SessionRepository`)** — Room DAO; stores per-session: pain, swelling tap, compensation-flag count, lock decisions, timestamps. No frames, no audio.
- **`ui/`** — Compose screens: Check-in → Session → Plan/Locked cards → Summary → Recovery sheet. The Session screen overlays a live strategy meter on the camera preview (`Canvas` redrawn each frame from `StrategySignal`) and plays the instant correction cue (§5a) — the one place in the UI that reacts directly to vision output, with no round trip through `policy/` or `explain/`. One Activity, Compose Navigation between screens — no need for multiple Activities.

## 5. The Compensation-Detection Mechanism

The core novel piece (PRD §3). Per rep, from the keypoint stream:

- **Hip-drive ratio**: change in hip-angle vs. change in knee-angle during the rise phase of sit-to-stand. A knee-dominant rise should show the knee angle changing rapidly relative to the hip; a hip-dominant (compensating) rise shows the hip leading, torso pitching forward more than expected.
- **Descent control**: rate of knee-angle change during the lowering phase — a controlled eccentric lowering is slow and steady; loss of quad control shows up as an uneven or accelerating descent.
- **Left-right weight cue** (opportunistic, not the headline metric — learned from the earlier LSI mistake, PRD §3): ankle/hip midpoint drift toward the unaffected side during the rise, as a secondary, low-confidence signal only, never gating a lock decision on its own.

A rep is flagged "compensating" if hip-drive ratio exceeds a threshold **tuned against real recorded footage during the build**, not guessed from a formula — this is explicitly listed as an open item (PRD §10) because it's the one piece of this system without literature-backed accuracy numbers behind it, unlike the underlying pose model.

### 5a. The live loop: feedback inside the rep, not after it

The first working version of this only produced a result once a rep was already finished — a summary, not a correction. A physio corrects you mid-movement, not five reps later. So `strategy/` streams `StrategySignal` to the UI every frame during the rise phase, and two things happen off that stream — both deliberately not involving the on-device LLM, to keep latency low enough to matter while the patient is still mid-rep:

- **Live meter:** the Compose overlay redraws a hip-drive/knee-drive gauge every frame — free, since it consumes the same keypoints §5 already computes.
- **Instant cue:** the moment `StrategySignal` crosses the hip-dominant threshold mid-rise, a short, pre-written phrase plays immediately (e.g. "drive through your knee") — a fixed lookup keyed by which threshold fired, not a generated sentence. Budget: keypoints → signal → cue trigger should land under roughly 150–200ms, which a template lookup and a canned audio clip trivially hit; a live LLM call would not.

Worth being explicit about in a jury Q&A, alongside the three models in §6: the live correction cue is not model output. It's deterministic, same as `policy/`. The on-device LLM's job stays exactly what §6 describes — phrasing the end-of-session decision — and it never sits in this latency-critical live loop.

## 6. On-Device AI Pipeline — Three Models, Three Different Jobs

Worth stating plainly, since a jury will ask: this is not "one AI does everything" — and the fastest-reacting part of the loop, the live correction cue, is not a model at all (§5a).

1. **Vision (LiteRT + QNN):** turns camera frames into keypoints. Perception only, no judgment.
2. **Voice (Whisper Tiny):** turns speech into the same structured check-in fields a tap would produce. Input method only.
3. **Language (Gemma 2B / Phi-3):** turns a decision that has *already been made deterministically* into a sentence a patient can read. This model is never given the authority to decide anything — it is handed a `LockDecision` object and asked to phrase it, the same principle used throughout this project since the earlier agentic-AI research: **deterministic safety, generative framing, and the generative layer cannot touch the decision.**

## 7. Office Kit Integration

Two real, non-contrived uses:

- **Dev workflow (Green Light, 45% of build time):** mirror the phone to the laptop while coding/debugging the vision pipeline — faster iteration than passing the phone back and forth.
- **Recovery sheet export:** the local Room database can generate a printable/PNG recovery sheet; Office Kit's file transfer moves it to the laptop for a nicer print/preview, useful at the actual clinic-visit use case too, not just for the demo.

Neither is required for the app to function — Red Light phone-only operation must always work standalone (PRD §7).

## 8. Build Sequence (30 hours, Red/Green Light aware)

1. **Get one pretrained pose model running through LiteRT + QNN delegate on actual hardware, on a single static test image, before anything else.** This is the highest-risk, most novel piece of toolchain — Google's own docs put dependency/delegate setup at ~2–3 hours; budget more given hackathon conditions and unfamiliar hardware. Do this first, phone-only, so it's derisked before Green Light hybrid time even matters.
2. **Live camera feed through the same pipeline**, keypoints drawn/logged, still phone-only.
3. **`CompensationDetector` against a few self-recorded sit-to-stand reps** (team members as test subjects) — tune the hip-drive threshold against real footage (§5), then wire the same signal to the live meter overlay and the instant cue trigger (§5a). This is the demo's single highest-impact moment — worth building before the polish items further down this list.
4. **`checkin/` with tap inputs first, Whisper Tiny voice input second** — tap path de-risks the check-in screen before adding voice complexity.
5. **`policy/` deterministic lock logic** — pure functions, unit-testable without the camera at all.
6. **`explain/` with the on-device LLM** — wire last, since it's the least safety-critical and most "polish" piece.
7. **`ui/` screens wired to the above**, `SessionRepository` persisting real sessions.
8. **Recovery sheet + Office Kit export**, plus using Office Kit for real during this stretch (naturally falls in Green Light).
9. **Demo rehearsal**: the actual 3–5 minute script — stand up, get corrected out loud mid-rep, visibly fix it on the next rep, check in, plan changes.

## 9. Known Limitations (stated deliberately)

- The hip-drive/descent-control thresholds are tuned on a handful of self-recorded reps during the hackathon, not validated against a clinical dataset — an honest limitation, not a hidden one.
- 2D monocular pose estimation from a single phone camera approximates what a physio sees from multiple angles and touch — it's a proxy, not equivalent to hands-on assessment.
- One exercise (sit-to-stand) in this build; broader compensation-pattern coverage is roadmap (PRD §9).
- No clinical validation of the on-device LLM's phrasing quality beyond "does it accurately restate the structured facts" — a manual check during the build, not an automated test.

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| NPU delegate setup eats the whole first day on unfamiliar hardware | Build sequence step 1 isolates this as the very first, standalone task — fail fast, fall back to CPU-only LiteRT (still on-device, still real, just slower) if the delegate genuinely won't cooperate |
| Compensation-detection thresholds are noisy/wrong on demo day | Tune against real footage during the build (§5, §8 step 3), not guessed numbers; rehearse the demo with the actual thresholds in place |
| Whisper Tiny/Gemma 2B/Phi-3 model files are large or slow to set up mid-event | Download and test these before the event starts if permitted, or immediately in the first hour — don't discover a multi-GB model download issue mid-Red-Light |
| Judges question whether "on-device AI" is real vs. a thin wrapper | §6's explicit three-models-three-jobs framing, plus showing the delegate/backend config in code — this is a jury-scored line (technical depth), be ready to explain it, not just claim it |

## 11. Checkpoint / Milestone Tracking

A second, coarser-grained deterministic layer on top of `policy/` (§4), not a replacement for it.

**`progress/` module.** `CheckpointEngine`: a pure function `(session_history_window) → CheckpointAdvancement?`, evaluated once per session after `SessionRepository` persists it — not per-frame like `strategy/` (§5a), not per-day-only like `policy/`. Reads the same fields `SessionRepository` already stores (pain, swelling tap, compensation-flag count, lock decisions, timestamps); no new schema needed beyond a `current_checkpoint` column and a `goals` table (below).

**Stages** (illustrative post-op progression — same caveat as §5's thresholds: needs real clinical input to finalize, not literature-derived as-is):

1. Protective — pain/swelling trending down, gentle motion tolerated
2. Motor Control — consistent knee-dominant sit-to-stand (low compensation-flag rate over a rolling window)
3. Progressive Strength — loaded exercises tolerated without new lock events
4. Return to Function — sustained clean sessions over a longer window

**Advancement rule, stated precisely.** Each stage has a small set of `GateRule`s — e.g. `MinConsecutiveCleanSessions(n)`, `PainTrendNonIncreasing(days)`, `NoSwellingIncrease(consecutiveCheckins)`, `MinSessionCount(n)` — evaluated against a rolling window, never a single session. This mirrors the same "no black-box classifier" preference as §5's `CompensationDetector`, for the same reason: it has to be explainable to a jury, and to a physio.

**One-way ratchet, by design.** `CheckpointEngine` only advances `current_checkpoint`, never regresses it. A bad day is already handled by `policy/`'s daily lock (§4, §6) regardless of stage — auto-demoting a patient's clinical stage would be a judgment call this deterministic layer has no business making, and the daily lock already provides the safety response a regression would be trying to achieve. If this needs revisiting, it belongs with the other open items (§9), not as a silent assumption.

**`goals/` — motivational only, no gate logic at all.** A minimal table: free-text goal, target date, done/not-done, set by patient or physio, read by `ui/` for the recovery-sheet timeline and nothing else. Deliberately has zero code path into `policy/` or `CheckpointEngine` — keeping the motivational layer inert is the same "deterministic safety, generative framing never touches the decision" principle from §6, extended to "motivational content never touches the decision" too.

**Recovery-sheet export (§7)** gains one more section — current checkpoint stage and any goals marked done — still just structured data, no new export mechanism.

**Build cost, honestly:** `CheckpointEngine` is pure aggregation over data `SessionRepository` already persists — no new inference, no new camera work. `goals/` is one Room table and two Compose screens. Both are Green Light stretch scope (§8) once the core loop and the live-correction path (§5a) are working, not Day-1 priorities — if cut, the daily loop in §5–§6 stands on its own.
