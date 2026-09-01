# RehabAI — Product Requirements Document

**Status:** v1.0 — converged direction (supersedes prior drafts and FEATURES.md's tech assumptions; keeps its safety principles)
**Last updated:** 2026-09-01
**Built for:** iQOO Hackathon 2026 (HealthTech track), phone-first / on-device AI

## 1. Problem

Globally, an estimated **2.4 billion people** live with a health condition that could benefit from rehabilitation ([WHO](https://www.who.int/news-room/fact-sheets/detail/rehabilitation)). Against that, **World Physiotherapy** counts roughly **600,000 physiotherapists** across its member organizations worldwide — WHO reports fewer than 10 skilled rehab practitioners per million people in many low- and middle-income settings. In India specifically, one existing competitor cites **0.6 physiotherapists per 10,000 people vs. WHO's recommended 1 per 10,000**.

Inside that gap sits a specific, well-documented clinical failure: after a knee replacement or ACL repair, the quad muscle often doesn't fire properly — a reflex response to joint pain/swelling called **arthrogenic muscle inhibition (AMI)** — and patients compensate without knowing it, standing up by leaning on their hips instead of pushing through the knee. Home exercise sheets don't catch this. Nobody's watching between appointments.

## 2. Product

RehabAI is an **on-device, phone-based rehab companion** for post-TKA/ACL patients. Each day:

1. **Check in** — pain (0–10) and a quick spoken or tapped note on swelling (puffier / same / less than yesterday).
2. **Do the exercise** — a sit-to-stand, phone camera propped up, watching.
3. The phone **detects compensation live, mid-rep** — hip-dominant strategy (leaning forward, driving through the hips instead of the knee), instability, and loss of control on the way down — the same visible signs a physio would look for, not a joint-angle number. The moment it crosses threshold, the phone says so out loud, in the same rep it was caught in, not in a summary afterward — a live on-screen meter shows the same signal visually.
4. **The app can say no** — if pain is up or swelling is worse, tomorrow's loaded exercises lock, with a plain-language reason.
5. Tomorrow's plan is generated from today's combination of self-report + what the camera actually saw — never from either signal alone.
6. A **printable recovery sheet** compiles the trend for the next clinic visit.

**Everything runs on-device.** No video, audio, or health data leaves the phone in this build — vision, voice, and language all run locally on the Snapdragon NPU.

## 3. The design decisions that actually matter

This product went through several wrong versions before this one, and both false starts are worth stating explicitly, because they're exactly the mistakes a judge (or a future contributor) will also be tempted to make:

**Don't score what a camera can't reliably see.** An earlier version tried to grade knee swelling from a static photo. Lighting fakes that — a dim room reads as "more swollen" regardless of the actual knee. So the camera is never used to score swelling, and swelling is self-reported (a fast three-way tap), not inferred from pixels.

**Don't score what a camera *can* see as if it were something deeper than it is.** A separate earlier version tried to measure quad muscle activation directly, or a bilateral range-of-motion symmetry index against population-average recovery curves. A webcam cannot see muscle activation — that's an electrical signal, EMG-only. And a symmetry ratio has a known clinical blind spot: both legs can weaken together and the ratio still looks "fine" ([JOSPT, 2017](https://pubmed.ncbi.nlm.nih.gov/28355978/)). What a camera *can* reliably see, and what's actually diagnostic, is movement **strategy** — hip-dominant vs. knee-dominant sit-to-stand is a real, visible, well-established compensation pattern physios already watch for without any equipment. That's the signal this product tracks.

**Feedback has to land inside the rep, not after it.** The first working version of this loop only produced a result once a rep was already finished — a summary card, after the fact. That's not how a physio actually corrects you in the room: they speak up mid-movement, before you finish doing it wrong, not five reps later. So the same per-frame keypoint stream already being computed for detection also drives a live on-screen strategy meter, and the instant hip-drive crosses the flagged threshold mid-rise, the phone says so out loud — a short, pre-written phrase ("drive through your knee"), not a generated one, so it's fast enough to matter while the patient is still standing up. The correction lands in the same rep it was caught in, and the very next rep shows whether it worked. Zoom out from a single rep to the whole recovery, and the same discipline repeats one level up: §11 adds a slower, deterministic checkpoint layer over multi-day progress — stages that only ever move forward, plus goals that never touch the decision.

## 4. Users

| Persona | Need | This build |
|---|---|---|
| **Patient** (primary) | A daily program that adapts to how they're actually doing, not a static printed sheet | Primary |
| **Surgeon / physio at follow-up** | Something better than "did you do your exercises?" | Recovery sheet (§6) |
| Supervising physio, many-patient caseload | Escalation queue across patients | Roadmap, not this build |

## 5. Competitive Differentiation

Kemtai already does real-time computer-vision form correction with FDA designation. Sword Health and Hinge Health dominate digital MSK but sell into US employer benefits with human coaches doing the actual check-ins — which is exactly why their model can't reach the 2.4B figure, it's priced for rich employer contracts. Resolve360, Bengaluru-based, already claims to triple patient treatment capacity with camera-based AI-AR in India specifically.

None of them, as far as this research found, distinguish movement **strategy** from movement **range** — they measure "was the rep deep enough," not "did the patient use the muscle they're supposed to be retraining." That's the wedge: not more accurate rep-counting, a different, more clinically honest question.

## 6. Hackathon Fit (iQOO Hackathon 2026)

This product happens to be a strong fit for the actual rubric, not by coincidence — the "camera + voice + on-device AI, all local" requirement is the same design constraint the product needed anyway (privacy-by-design for health data, zero dependency on connectivity for a rural/low-resource deployment down the line).

| Rubric line | Weight | How this product covers it |
|---|---|---|
| End product quality (jury) | 30% | A complete, working 90-second daily loop — check-in → exercise → plan update |
| Novelty & impact (jury) | 20% | Movement-strategy detection (hip- vs. knee-dominant), not rep-counting or ROM |
| Creative phone use (device data) | 15% | Camera (on-device pose/motion via NPU, driving a live meter + instant spoken correction), voice (Whisper Tiny check-in), on-device AI (Gemma 2B/Phi-3 for plan explanations) — all real, telemetry-verified usage |
| Technical depth (jury) | 15% | Real NPU inference via LiteRT + Qualcomm QNN delegate, not a cloud API wrapper |
| Office Kit usage (device data) | 10% | Genuine dev workflow (phone-to-laptop mirroring while building) + exporting the recovery sheet via Office Kit's file transfer |
| Demo & presentation (jury) | 10% | Live: stand up, get corrected out loud mid-rep, visibly fix it on the very next rep, then watch tomorrow's plan change too |

## 7. Hackathon MVP — Scope

**In scope:**
- Local single-patient profile; procedure type and operated side set once.
- Check-in: pain slider + puffier/same/less (voice via Whisper Tiny or tap).
- Live sit-to-stand session: on-device pose/motion detection running continuously through the rep — a live on-screen strategy meter, plus an instant, pre-written spoken cue the moment hip-drive crosses threshold. Feedback lands inside the rep, not in a summary after it.
- End-of-session review combining check-in + session compensation frequency → tomorrow's plan.
- At least one lockable exercise with a plain-language unlock condition, phrased via the on-device LLM from structured facts only.
- Recovery sheet (local history → printable/exportable view).
- Office Kit used for real during development and for exporting the sheet.

**Out of scope:**
- Any diagnostic claim; this flags a movement pattern, it does not evaluate the surgical site or grade recovery clinically.
- Multiple exercises beyond sit-to-stand (one exercise, done well, beats a shallow library).
- Cloud sync, accounts, multi-patient physio dashboard.
- EMG/IMU hardware dependency — explicitly rejected; this build is camera + self-report only.

## 8. Safety & Clinical Boundaries

- Locking a loaded exercise is driven by **self-report (pain, swelling) plus session-level compensation frequency** — never by a single frame, never by raw photo brightness/color.
- The on-device LLM only phrases explanations from structured logged facts (pain value, swelling tap, compensation count). It cannot invent a reason or override a lock decision — same deterministic-safety-first-generative-framing-second principle used throughout.
- Copy never uses alarming language ("asymmetry detected," "you are unsafe"). A lock reads as "squats are off today because pain was up," not a warning.
- This is a self-management aid, not a diagnostic device. A human clinician remains the decision-maker; the recovery sheet is a conversation aid.

## 9. Roadmap

**Now (hackathon):** the loop above, one exercise, phone-native, fully on-device.

**Next:** more exercises with the same strategy-detection lens; richer end-of-session LLM summaries; multi-day trend view.

**Later:** supervising-physio escalation queue (structured data only, still no raw video leaving the device by default); hospital discharge pilot.

## 10. Open Questions

- Exact compensation-detection thresholds (how much forward lean / hip-drive counts as "hip-dominant") — needs tuning against real footage during the build, not guessed in advance.
- Kotlin native Android vs. a cross-platform framework — TRD §2 recommends native for NPU-delegate support; confirm once actual hardware is in hand.
- How much of the recovery-sheet export should specifically demonstrate Office Kit, beyond dev-workflow usage.

## 11. Checkpoint-Based Goal Tracking

Two layers, deliberately kept separate.

**Clinical checkpoints (deterministic, gate the stage — not the day).** The daily loop (§2) already handles bad days: pain spikes, swelling, and a hip-dominant rep lock a loaded exercise *today*, regardless of what stage the patient is nominally in. Checkpoints sit above that, on a slower clock — four broad post-op stages: Protective (settle pain/swelling, gentle motion), Motor Control (clean, knee-dominant sit-to-stand becomes consistent), Progressive Strength (loaded work becomes the norm, not the exception), Return to Function (higher-demand daily tasks). Advancing a stage is computed from the same session history already being logged — a rolling window of pain trend, swelling-tap trend, and compensation-flag rate — never a single good day.

**Checkpoints only move forward.** A setback after advancing is handled by the daily lock, not by rolling a patient's stage back. Auto-demoting someone's clinical stage would be a judgment call this app isn't positioned to make, and the daily lock already provides the safety response a regression would be trying to achieve.

**Patient/physio goals (motivational, gate nothing).** Alongside the clinical stages, a patient — or a supervising physio at a clinic visit — can add plain-language goals ("walk to the mailbox unassisted," "climb stairs without the rail") with a target date. These are self-reported done/not-done, shown as a simple timeline on the recovery sheet (§2), and never feed `policy/` or the checkpoint logic (TRD §11). Keeping this layer inert by design is the same principle as §3: the motivational layer never touches a safety-relevant decision, the same way the generative layer never touches one.

The recovery sheet (§2) gains one more section from this: current clinical stage and any goals marked done, giving the surgeon a compact multi-week view alongside the daily swelling/compensation data it already shows.

**Where this sits in a 30-hour build, honestly:** cheap relative to everything else in this document, since it's aggregation over data already logged, not new sensing or a new model. A reasonable Green Light stretch item once the core loop (§2) and live correction (§3) are solid — not a Day-1 priority. If it doesn't make the cut, the daily loop still stands on its own; checkpoints are additive, not load-bearing.
