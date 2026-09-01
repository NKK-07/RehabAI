# RehabAI — Features

> ## ⚠️ SUPERSEDED — do not build from this file
>
> **Superseded by:** [PRD.md](PRD.md) v1.0 and [TRD.md](TRD.md) v1.0, 2026-09-01.
>
> This describes an earlier, materially different product: quad sets as the core
> exercise, muscle activation measured via EMG, and "% of the good side" — a bilateral
> symmetry ratio — as the hero metric. PRD.md §3 rejects all three, with reasoning.
>
> **Retained as history, and for one thing worth salvaging:** the copy bank in §4. PRD.md
> §8 asserts a copy-tone rule but supplies no phrases; this file has them.

**Status:** ~~Canonical (v1.1)~~ — superseded 2026-09-01  
**Last updated:** 2026-09-01  
**Companion to:** [PRD.md](PRD.md), [UI_UX_PLAN.md](UI_UX_PLAN.md), [TRD.md](TRD.md)

~~This file is the product. Other docs must not contradict it.~~

RehabAI is a **home treatment loop** for post-knee-replacement and post-ACL rehab. It **doses** tomorrow’s session, **refuses** unsafe progressions, and **retrains the quad**. It is not a live skeleton form-checker.

All inference stays **on-device**. Stills, scores, and history do not leave the machine in the hackathon build.

---

## Who it is for

| Person | What they get |
|---|---|
| **Patient** (primary) | A kitchen-length session that changes with their knee, not a 20-minute webcam squat class |
| **Surgeon / physio at week 6** | One page they can actually look at: swelling, activation vs the good side, what was blocked, what progressed |

---

## Feature map

```
  Check-in (pain + puffier/same/less + optional still)
            │
            ▼
  Today's plan (locks from pain/self-report/protocol — not lighting)
            │
            ├── Find the muscle (quad sets)
            └── Allowed movement (only if unlocked)
            │
            ▼
  One number: % of the good side
            │
            ▼
  Tomorrow rewrites itself
            │
            ▼
  Week-6 sheet
```

---

## F1 — Tomorrow’s workout is not today’s workout

**User-facing:** After you finish, the next session is already different: fewer reps, longer holds, or a harder move. You do not pick from a static menu.

**What they do:** Complete today’s plan. Check in (F2). Leave.

**What they see next time:** A new card list generated from yesterday’s state. Copy like “We dropped squats and added 2 extra quad sets.”

**Must not:** Let the patient freely choose squat vs shoulder raise as the core loop. That is a fitness app.

**Hackathon:** Two canned trajectories (pain/puffier vs settled) so judges see the plan **rewrite** between Session A and Session B. Do **not** demo the rewrite by dimming the lights on a photo.

---

## F2 — Check-in: pain, puffier/same/less, then a still if it is usable

**Why this is split:** A kitchen still is **easy to ruin with lighting**. Color, shadows, and white-balance will fake “more swollen” or “better” if the app compares pixels to yesterday. That must not drive F3.

**User-facing, in order:**

1. Pain 0–10.  
2. One tap: **puffier / about the same / less puffy** than yesterday (first session: skip or “as expected”). This is the swelling **signal**.  
3. Optional still of the operated knee (coin in frame) for the **week-6 album** (F8), not as the lock brain.

**What they do:** Slider + three-way tap. Shutter. If the still fails a quality gate, **retake** (flash / face a window / match the day-0 thumbnail). After two failures: **skip photo**, continue the session. Copy: “Lighting is off. We will not score this picture. Pain and how puffy it feels still count.”

**What they see:** Day-0 still as a **ghost overlay or side-by-side** so they can match distance and light. Not “Saved. This photo decides tomorrow.”

**Quality gate (must reject, not score):** too dark, too bright, heavy color cast vs day-0, blur, knee not in the guide box. Rejected stills are not stored as “swelling scores.”

**Must not:** Stream video. Must not run pose. Must not histogram- or brightness-compare two stills and call that effusion. Must not lock squats because the overhead light was off.

**Hackathon:** Show the gate failing on a dark still (retake/skip). Show F3 changing from the **three-way tap + pain**, with the same lighting. Seed week-6 with a few **accepted** stills only.

---

## F3 — The app can say no

**User-facing:** If **pain jumped** or they tapped **puffier**, **squats (and other loaded moves) are not on the plan**. You get quad sets, allowed range, ice/elevation reminder. The move is gone, not marked red while you do it.

**Lock inputs (allowed):** protocol day, pain vs yesterday, puffier/same/less.  
**Lock inputs (not allowed):** raw still brightness, auto “swelling score” from a photo.

**What they see:** A locked card. Not a form cue on a squat.

**Copy (required tone):** Calm, specific.  
- Good: “Squats are off today. You marked the knee puffier (and pain is up).”  
- Bad: “Photo analysis: swelling increased.” / “Asymmetry detected.” / “Warning: high risk.”

**Must not:** Allow starting a locked exercise from a hidden menu. Must not scare. Must not blame the camera.

**Hackathon:** Two demo patients, same lighting: one puffier+pain, one settled. Locks follow the taps, not the room lights.

---

## F4 — Locked card with an unlock sentence

**User-facing:** You can see the next progression **before** you earn it.

**What they see:** Card titled e.g. Mini-squats. State: **Locked**. One sentence: “Unlocks in 4 days if you keep marking same/less puffy and pain stays ≤3.”

**What they do:** Nothing. The sentence is the feature. It makes the protocol visible.

**Must not:** Mystery unlocks. Must not dump the full clinical protocol as a wall of text.

**Hackathon:** At least two locked cards with different unlock rules (puffier tap, pain, days-since-op).

---

## F5 — Find the muscle (not “go deeper”)

**User-facing:** Early sessions are **quad sets**: squeeze, hold, match the good leg. You win when the muscle wakes up.

**What they see:** A target (bar, ring, or number) for the operated side. A quieter “teacher” trace for the other side. Success = reaching a % of the teacher, not a joint angle.

**Coaching line:** “Find the muscle.” Never “go deeper” as the primary cue.

**Must not:** Full-body skeleton overlay as the hero. Must not treat peak squat flexion as the score.

**Hackathon (sensor, pick what you have):**

| Available | What to use |
|---|---|
| Nothing extra | Phone/laptop **hold-on-thigh** during a seated quad set, or a **manual “I felt it” + timed hold** with contralateral as a scripted teacher — plus a **simulated EMG** mode for the judge demo |
| Cheap sEMG | Dual channel, operated vs contralateral, real teacher |
| Microphone on VMO | Experimental MMG path (label as experimental in UI) |

Product truth: this feature is **activation**, not ROM.

---

## F6 — One number that is not ROM

**User-facing:** End of session / end of week, the headline is **“You’re using 62% of the good side.”** That number should climb.

**What they see:** Big percentage. Small print: “vs your other quad, this session.” Secondary: pain, puffier/same/less, what was locked.

**Must not:** Lead with good-form %, LSI from squat depth, or a literature TKA ROM corridor as the hero metric.

**Hackathon:** Persist the % across a fake 6-session history so the progress screen is a climb, not a one-off.

---

## F7 — Ninety seconds, phone in your hand

**User-facing:** Kitchen rehab. Sit. Check-in. Three (or more) quad sets. Done unless a loaded move is unlocked.

**Time budget:** Check-in + activation should fit in **about 90 seconds** on a good day. Unlocked sit-to-stand adds a short block, still not a class.

**Must not:** Require a laptop across the room, a full-body frame, or a 10-minute exercise picker.

**Hackathon:** Desktop is allowed as the **shell** if mobile is not ready, but the **interaction** must be still + seated activation, not live squat theater. Layout should read as a phone even in a window (narrow, large tap targets).

---

## F8 — Week-6 sheet you actually bring

**User-facing:** One page (screen + printable/PNG): puffier/same/less trend, activation % trend, list of blocked moves with dates, list of progressions. **Accepted** stills as a small contact sheet (album), skipped days labeled “no usable photo.”

**Who it is for:** The patient holds it; the surgeon/physio reads it in the appointment. The photos are **evidence of appearance**, not an AI effusion grade.

**Must not:** A dense clinician dashboard. Must not include video or skeleton screenshots. Must not caption a still “swelling +18%.”

**Hackathon:** Generate the sheet from local SQLite. Seed 6 sessions so it looks like a real recovery, not an empty chart.

---

## Explicitly not features

Do not build or pitch these as RehabAI:

- Live MediaPipe / skeleton overlay
- Shoulder-raise library filler
- Patient-chosen exercise catalog as the home screen
- Webcam goniometer / bilateral squat LSI gauge
- Published population ROM corridor as “on track”
- Cloud LLM cheerleading
- Auto swelling score from pixel/histogram compare
- Accounts, sync, or a multi-patient physio inbox (later product, not this feature set)

---

## Demo script (hackathon)

1. **Bad lighting still:** shutter → quality gate fail → retake/skip. Plan does **not** change because of the dark frame.  
2. **Patient A:** same room light, tap **puffier** + pain up → squats locked → find-the-muscle only → week-6 shows blocked dates.  
3. **Patient B:** tap **same/less** → sit-to-stand unlocked → % climbing.

The “smart moment” is **the plan changing from what they reported**, plus the camera **refusing to pretend** when light is bad.

---

## Traceability

| ID | PRD | UI | TRD (modules) |
|---|---|---|---|
| F1 | Core loop, MVP | Home, Today’s plan | `policy.py` |
| F2 | Check-in | Check-in screen | `checkin.py`, still capture |
| F3 | Safety | Locked cards | `constraints.py` + policy |
| F4 | Safety / education | Locked card copy | `constraints.py` |
| F5 | Activation session | Find the muscle | `activation.py` |
| F6 | Success metric | Summary, history | `session.py` |
| F7 | UX constraint | All patient screens | UI layout, no live pose loop |
| F8 | Week-6 artifact | Progress / print | `session.py` export |
