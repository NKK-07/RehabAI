"""
theme.py
One place for the look, so no view invents its own.

PHONE-SHAPED BY CONSTRUCTION
============================
A narrow column with large controls, even in a desktop window. This is the one
instinct worth keeping from the superseded UI_UX_PLAN.md: the product is used
seated, in a kitchen, at arm's length -- not at a desk.

TONE
====
Calm and specific, never alarming. A locked exercise reads as a missing card
with a plain reason, not a warning. PRD 8 states the rule; the phrases below
are salvaged from FEATURES.md 4, which is the one thing that document had that
the PRD does not.
"""

from __future__ import annotations

# Narrow enough to read as a phone on a laptop screen.
WINDOW_WIDTH = 430
WINDOW_HEIGHT = 860

INK = "#15242B"
INK_SOFT = "#5C7079"
GROUND = "#F2F5F5"
SURFACE = "#FFFFFF"
LINE = "#D8E2E3"
ACCENT = "#0D6169"
ACCENT_SOFT = "#DDEBEB"
LOCKED = "#8A6A3A"
LOCKED_SOFT = "#F6EBDA"
GOOD = "#2C6B45"

STYLESHEET = f"""
QWidget {{
    background: {GROUND};
    color: {INK};
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 15px;
}}
QLabel#title {{
    font-size: 26px;
    font-weight: 600;
    color: {INK};
}}
QLabel#subtitle {{
    font-size: 15px;
    color: {INK_SOFT};
}}
QLabel#huge {{
    font-size: 46px;
    font-weight: 600;
    color: {ACCENT};
}}
QLabel#card {{
    background: {SURFACE};
    border: 1px solid {LINE};
    border-radius: 6px;
    padding: 18px;
}}
QLabel#lockedCard {{
    background: {LOCKED_SOFT};
    border: 1px solid {LOCKED};
    border-radius: 6px;
    padding: 18px;
    color: {LOCKED};
}}
QPushButton {{
    background: {SURFACE};
    border: 1px solid {LINE};
    border-radius: 6px;
    padding: 16px;
    font-size: 16px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:checked {{
    background: {ACCENT_SOFT};
    border: 2px solid {ACCENT};
    color: {ACCENT};
    font-weight: 600;
}}
QPushButton#primary {{
    background: {ACCENT};
    color: #FFFFFF;
    border: none;
    font-size: 18px;
    font-weight: 600;
    padding: 20px;
}}
QPushButton#primary:disabled {{
    background: {LINE};
    color: {INK_SOFT};
}}
QSlider::groove:horizontal {{
    height: 8px;
    background: {LINE};
    border-radius: 4px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 34px;
    height: 34px;
    margin: -14px 0;
    border-radius: 17px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 4px;
}}
"""

# Salvaged from FEATURES.md 4. PRD 8 asserts the tone rule but supplies no
# phrases; these are the phrases.
COPY = {
    "lock_pain": "Squats are off today. Your pain was higher than usual.",
    "lock_swelling": "Squats are off today. You marked the knee puffier than yesterday.",
    "lock_both": "Squats are off today. Pain was up and the knee is puffier.",
    "lock_compensation": "Squats are off today. Most stands went through your hips.",
    "hold": "We could not see enough today to add loaded work.",
    "rest": "Today is a rest day. Gentle movement only.",
    "allow": "Everything is on the plan today.",
    "sheet_footer": "Bring this to your appointment.",
}
