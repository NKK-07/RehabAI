# RehabAI — UI/UX Plan

> ## ⚠️ SUPERSEDED — do not build from this file
>
> **Superseded by:** [PRD.md](PRD.md) v1.0 and [TRD.md](TRD.md) v1.0, 2026-09-01.
>
> This plans the UI for the earlier product described in [FEATURES.md](FEATURES.md), which
> is itself superseded. Three things here are now actively wrong: it makes "% of the good
> side" the hero metric (a symmetry ratio PRD.md §3 rejects), it bans pose overlays and
> live video — which are the current product's central feature — and it plans for IMU/EMG
> hardware that PRD.md §7 explicitly rules out.
>
> The instinct it got right, and which survives: the layout should read as **phone-shaped**
> — a narrow column with large controls.

**Status:** ~~Draft v0.2~~ — superseded 2026-09-01  
**Companion to:** [FEATURES.md](FEATURES.md) (also superseded), [PRD.md](PRD.md), [TRD.md](TRD.md)  
**Framework:** PySide6 (desktop shell). Layout is **phone-shaped**: narrow column, large controls. Live skeleton video is **not** a screen.

## 1. Design principles

- **Features first.** Every screen maps to an ID in FEATURES.md. If a widget does not serve F1–F8, cut it.
- **Kitchen, not gym.** Seated. Big type. One primary action per screen.
- **Locked is calm.** Missing cards, not alarms. Unlock sentence is always visible (F4).
- **Hero metric is % of the good side** (F6). Not ROM, not form %.
- **No fake chrome.** No login, no sync spinner, no account menu.
- **Camera = still, then a gate.** Check-in uses a shutter. Bad light → reject/retake/skip. The plan does not follow brightness (F2, F3).

## 2. Screen flow

```
  Home (today's plan already set)          F1, F6 peek, F8 entry
       │
       ▼
  Check-in (pain, puffier/same/less, gated still)   F2
       │
       ▼
  Today's plan (cards, some locked)        F3, F4
       │
       ├── Find the muscle                 F5, F7
       └── Allowed move (if unlocked)      F3
       │
       ▼
  Session summary (% of good side)         F6
       │
       ▼
  Week-6 sheet / history                   F8
```

One `QMainWindow` + `QStackedWidget`. No extra OS windows.

## 3. Screen-by-screen

**Home** — Name, procedure, day post-op. One line: last **% of good side**. Primary: **Start today’s session** (not “pick an exercise”). Secondary: **Week-6 sheet**. If no history, say so and send them to check-in.

**Check-in (F2)** — Pain 0–10. Three large taps: puffier / same / less. Then still: day-0 thumbnail beside live preview, guide box, shutter. If quality fails: “Lighting is off. We will not score this picture.” Retake or skip. No live skeleton. Confirm uses **pain + tap** even if photo was skipped.

**Today’s plan (F1, F3, F4)** — Stack of cards:

- **Ready:** Find the muscle (always, unless safety says rest-only).  
- **Ready or locked:** Sit-to-stand / mini-squat per policy.  
- Locked cards show the **unlock sentence** on the card. Tapping a locked card does not start the exercise; it only repeats the sentence.

**Find the muscle (F5)** — No video panel as the hero. Target vs teacher (two bars or one ring). Hold timer. Cue: “Find the muscle.” Quit always available; partial session still logs.

**Allowed movement (only if unlocked)** — Short sit-to-stand or hold. If we lack a non-pose sensor, **count with a tap** or a timer — do **not** fall back to MediaPipe to “make it feel live.”

**Session summary (F6)** — Huge **% of the good side**. Three lines: pain today, swelling vs yesterday (plain language), what stayed locked. **Done** → Home. Optional: “Tomorrow may look different.”

**Week-6 sheet (F8)** — Printable/PNG layout: activation % over sessions, swelling notes, table of blocked dates, table of unlocks. Export button. Not a clinician dashboard.

## 4. Copy bank (required)

| Situation | Use | Do not use |
|---|---|---|
| F3 lock | “Squats are off today. Swelling is up vs yesterday.” | “Asymmetry detected.” |
| F4 | “Unlocks in 4 days if swelling stays down and pain stays ≤3.” | “Complete more content to unlock.” |
| F5 | “Find the muscle.” / “Match the quiet side.” | “Go deeper.” |
| F6 | “You’re using 62% of the good side.” | “Good-form 91%.” |
| F8 | “Bring this to your appointment.” | “Clinical diagnosis.” |

## 5. Component → module mapping

| UI | Feature | Module |
|---|---|---|
| Pain + still | F2 | `checkin.py`, capture widget |
| Plan cards | F1 F3 F4 | `policy.py`, `constraints.py` |
| Target vs teacher | F5 | `activation.py` |
| Headline % | F6 | `session.py` |
| Sheet + PNG | F8 | `session.py` export |
| Voice (optional) | F5 | `voice_coach.py` — only “find the muscle” / hold cues, not form nags |

No pose overlay widgets. Existing `pose_utils.py` is **not** wired into this UI.

## 6. Build sequence

1. Qt shell, stacked screens, phone-shaped window, nav placeholders.  
2. Check-in still capture in a `QLabel` (one frame on shutter), SQLite row.  
3. Today’s plan cards from a **stub policy** (lock/unlock driven by demo patient A/B).  
4. Find-the-muscle UI with **simulated** teacher + operated traces (prove F5/F6 without hardware).  
5. Session summary + week-6 sheet from seeded + live rows.  
6. Plug real IMU/EMG into `activation.py` if hardware is in the room.  
7. Polish: copy bank, lock states, 90-second happy path.

Do **not** start with OpenCV video-in-a-window for a live squat. That rebuilds the old product.

## 7. Non-goals

- Exercise select catalog (squat vs shoulder raise).  
- Live video + skeleton.  
- Symmetry gauge / ROM corridor charts.  
- Accounts, network, physio inbox.
