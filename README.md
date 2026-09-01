# RehabAI

**A phone that watches you recover, and tells tomorrow what to do differently.**

An on-device rehab companion for people recovering from knee replacement or ACL surgery.
It watches one exercise, catches the specific way people cheat at it, corrects them out
loud while they're still moving, and rewrites tomorrow's plan from what it actually saw.

Nothing leaves the device.

> **Status: pre-implementation.** This repository currently holds the product and technical
> specifications, the package structure, and two legacy modules from a superseded design.
> No feature code is written yet. See [Project status](#project-status).

---

## The problem

About **2.4 billion people** worldwide live with a condition that rehabilitation would
help ([WHO](https://www.who.int/news-room/fact-sheets/detail/rehabilitation)). There are
roughly **600,000 physiotherapists**. In India it's about 0.6 physios per 10,000 people
against a WHO recommendation of 1 per 10,000. That ratio doesn't close by training more
people.

Inside that gap sits a specific, well-documented failure. After knee surgery the quad
muscle often stops firing properly — a reflex response to pain and swelling called
**arthrogenic muscle inhibition (AMI)**. Patients compensate without knowing: they stand
up by throwing their weight forward and driving through the hips, because the quad won't
do the job.

They believe they're doing their exercises. They're doing them with the wrong muscle, so
the quad never rebuilds. A printed home-exercise sheet cannot catch this, and nobody is
watching between appointments.

---

## The one thing to understand

There are two ways to stand up from a chair, and they look more similar than you'd expect.

```
     KNEE-DOMINANT  (what we want)          HIP-DOMINANT  (the compensation)

            (head)                                        (head)
              |                                          /
              |   torso stays upright                   /   torso pitches forward
              |                                        /
        hip @-----------@ knee                  hip @--------@ knee
                        |                                     |
                        |  knee angle                         |  knee angle
                        |  opens fast                         |  barely changes
                        @ ankle                               @ ankle
     ─────────────────────────────           ─────────────────────────────

     Both people get out of the chair. Both complete the rep.
     A system that measures DEPTH scores these two identically.
```

So the signal is a ratio: **how fast the hip angle changes versus how fast the knee angle
changes, during the rise.** Knee leading is good. Hip leading means the quad isn't
participating — precisely the muscle the whole rehab programme exists to rebuild.

This isn't something we invented. It's a compensation pattern physiotherapists already
watch for by eye, with no equipment. We're making it available on the other twenty days a
month when no physio is in the room.

**That's the wedge.** Existing camera-based rehab products measure *was the rep deep
enough*. None of them measure *which muscle did the work*.

---

## The daily loop

1. **Check in** — pain 0–10, and one tap: puffier / same / less puffy than yesterday.
   Can be spoken instead of tapped; speech is transcribed on-device into the same fields.
2. **Stand up** — phone propped side-on. An on-screen meter tracks hip-versus-knee drive
   in real time.
3. **Get corrected, mid-rep** — the moment the signal crosses threshold, the phone says a
   short fixed phrase out loud, *while you're still rising*. The next rep shows whether it
   worked.
4. **The app can say no** — if pain is up or swelling is worse, loaded exercises come off
   tomorrow's plan with a plain-language reason. Not a warning: *"squats are off today
   because pain was up."*
5. **Tomorrow's plan** is generated from today's self-report combined with what the camera
   actually saw — never from either signal alone.
6. **A recovery sheet** compiles the trend into one printable page for the next clinic visit.

---

## How it works

```
  camera ──▶ pose model ──▶ keypoints (hip, knee, ankle, shoulder)
                                  │
                                  ▼
                      sit-to-stand detector
                      seated → RISE → standing → descent
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
       live meter          spoken cue           per-rep summary
       (every frame)   (fixed phrase, once)   (+ observation quality)
                                                      │
  check-in ───────────────────────────────────────────┤
  pain + swelling                                     ▼
                                            policy  ◀── pure function,
                                                        no AI at all
                                                      │
                                              LockDecision
                                          (+ explicit reason codes)
                                                      │
                                                      ▼
                                        language model phrases it
                                        — cannot change it
                                                      │
                                                      ▼
                                              local database
                                                      │
                                                      ▼
                                              recovery sheet
```

Three models, three narrow jobs. **Vision** turns frames into joint positions.
**Speech** turns spoken check-ins into the same fields a tap produces. **Language** turns
a finished decision into a readable sentence.

The most important detail: **the fastest thing in the product is not a model at all.**
The live spoken cue is a fixed phrase on a deterministic trigger, because a model call
couldn't reach the patient before they finished standing up.

---

## Four rules that govern everything

| Rule | What it means |
|---|---|
| **Deterministic safety, generative framing** | Every safety-relevant decision is a pure function you can read on one screen. The language model is handed the finished decision and asked to phrase it. It can never alter one. |
| **Missing data stays missing** | If the camera couldn't observe a rep, it's marked *unobservable* — not scored clean, not scored compensating. A zero meaning "we saw nothing" must never become a zero meaning "they did it perfectly." |
| **Nothing is faked** | No seeded history, no simulated modes, no placeholder data in demos or screenshots. If it can't be shown working for real, it isn't shown. |
| **Feedback lands inside the rep** | A physio corrects you mid-movement, not five reps later. Any design delivering correction after the rep has lost what makes this different from a report card. |

---

## What this deliberately isn't

Three approaches were tried and rejected. Each sounds reasonable, so each is worth stating:

- **Grading swelling from a photo.** Lighting destroys it — a dim room reads as "more
  swollen" regardless of the actual knee. Swelling is self-reported now; the camera never
  scores it.
- **Measuring muscle activation from the camera.** A camera physically cannot see it —
  activation is an electrical signal, which means EMG and electrodes. Requiring extra
  hardware also breaks the premise that this runs on a phone somebody already owns.
- **A left-versus-right symmetry score.** It has a documented clinical blind spot: both
  legs can weaken together and the ratio still reads fine
  ([JOSPT, 2017](https://pubmed.ncbi.nlm.nih.gov/28355978/)).

Each rejection is the same rule applied three times: **don't produce a number unless you
can actually measure the thing the number claims to be about.**

---

## Project status

| Area | State |
|---|---|
| Product & technical specs | Complete — `PRD.md`, `TRD.md` |
| Engineering review | Complete — 17 tasks, 15 locked decisions, cross-model challenge |
| Package structure | Scaffolded |
| Detection, policy, storage, UI | **Not started** |
| Tests | **Not started** |

`rehab_ai/pose_utils.py` and `rehab_ai/exercises.py` are from a superseded design.
`pose_utils.py` will be reused; `exercises.py` will not — it's parameterised as a squat
(standing-first), so a seated start makes it report a completed rep before the patient has
moved.

---

## Repo layout

```
rehab_ai/
├── camera/       frame capture; never mirrors before inference
├── pose/         pose model wrapper
├── detection/    sit-to-stand detector — the core
├── policy/       deterministic lock decisions, no AI
├── explain/      local LLM phrasing (cannot alter decisions)
├── checkin/      voice → structured fields
├── audio/        pre-rendered cue playback
├── models/       shared types: Observation, RepResult, RehabSession
├── storage/      local database
├── rules/        threshold file loader
└── ui/           application shell and screens

rules/            thresholds.v1.json — the handoff artifact to the mobile build
assets/cues/      pre-rendered cue clips
tests/            pytest suites
```

---

## Getting started

**Python 3.11 is required.** `mediapipe==0.10.21` publishes no wheels for 3.13 or 3.14,
so a newer interpreter will fail to install rather than fail at runtime.

```bash
# with uv (recommended -- resolves and fetches 3.11 for you)
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt

# or with a 3.11 interpreter you already have
py -V:3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
```

Run everything through the venv interpreter explicitly (`.venv/Scripts/python.exe` on
Windows, `.venv/bin/python` elsewhere) rather than a bare `python`, which may resolve to a
different interpreter on your PATH.

The language model runs locally through [Ollama](https://ollama.com), which must be
installed and running:

```bash
ollama pull gemma2:2b
```

Verify the environment:

```bash
.venv/Scripts/python.exe -c "import mediapipe, cv2, PySide6, faster_whisper, sounddevice; print('ok')"
curl http://localhost:11434/api/tags
```

Then:

```bash
.venv/Scripts/python.exe -m pytest -q      # works now
.venv/Scripts/python.exe -m rehab_ai.app   # once the app entry point exists
```

---

## Two deliverables

|  | **This repo — the prototype** | **The phone build** |
|---|---|---|
| Purpose | Validates the approach, tunes the thresholds | The actual product |
| Platform | Python desktop, PySide6 | Native Android, Kotlin, Compose |
| Vision | MediaPipe on CPU | LiteRT + Qualcomm NPU delegate |
| Language model | Local, via Ollama | On-device via the NPU |

**What crosses over is `rules/thresholds.v1.json`.** The tuned thresholds and lock rules
are this prototype's real output — the mobile app loads them verbatim rather than having
them retyped.

To be precise about the claim: this prototype is **genuinely local** — no network calls
leave the machine — but it is **not** running on a phone's NPU. Those are two different
statements and we keep them separate.

---

## Limitations

Stated deliberately, not buried:

- Thresholds are tuned against a handful of recorded reps from healthy volunteers.
  **They are not clinically validated.**
- 2D pose from a single camera approximates what a physiotherapist sees from several
  angles and with their hands. It's a proxy, not an equivalent.
- One exercise: sit-to-stand.
- **This is not a diagnostic device.** It flags a movement pattern. It does not evaluate
  the surgical site or grade recovery. A clinician remains the decision-maker, and the
  recovery sheet is a conversation aid.

---

## Documentation

| File | Contents |
|---|---|
| [`PRD.md`](PRD.md) | Product requirements — problem, users, scope, safety boundaries |
| [`TRD.md`](TRD.md) | Technical requirements — architecture, detection mechanism, build sequence |
| `FEATURES.md`, `UI_UX_PLAN.md` | **Superseded.** Retained as history; do not build from them. |

---

Built for the iQOO Hackathon 2026, HealthTech track.
