"""
sheet_view.py
The recovery sheet: one page the patient carries to their appointment.

WHAT A SURGEON NEEDS TO SEE
===========================
Not a dashboard. Three trends and a table of what was blocked, on one page,
readable in the thirty seconds an appointment affords it.

The sheet plots reps that were actually scored, and states separately how many
could not be. A trend line drawn through unobserved sessions would be a picture
of data that does not exist -- and this is the artifact a clinician would make
decisions from, so it is the last place to be loose about it.

EMPTY STATES ARE REAL SCREENS
=============================
Zero sessions and one session are both normal, especially in the first week.
Neither is an error, and neither gets a fabricated trend to make the chart look
populated.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from rehab_ai.models.session import CompensationSummary, RehabSession
from rehab_ai.ui.theme import COPY

DEFAULT_SHEET_DIR = Path(__file__).resolve().parents[2] / "data" / "sheets"


def render_sheet(sessions: list[RehabSession], out_path: Path) -> Path:
    """Draw the sheet to a PNG. Returns the path written.

    Uses matplotlib's non-interactive backend explicitly: this may be called
    from a thread, and the default backend would try to talk to a GUI toolkit
    that is already running its own event loop.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(
        [s for s in sessions if s.started_at is not None], key=lambda s: s.started_at
    )

    figure, axes = plt.subplots(3, 1, figsize=(8.27, 11.69), height_ratios=[1, 1, 1])
    figure.suptitle("Recovery sheet", fontsize=18, fontweight="bold", x=0.11, ha="left")

    if not ordered:
        for axis in axes:
            axis.axis("off")
        axes[1].text(
            0.5,
            0.5,
            "No sessions recorded yet.",
            ha="center",
            va="center",
            fontsize=13,
            color="#5C7079",
        )
        figure.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(figure)
        return out_path

    days = [s.protocol_day for s in ordered]

    # -- pain ---------------------------------------------------------------
    pain_days = [s.protocol_day for s in ordered if s.pain is not None]
    pain_values = [s.pain.value for s in ordered if s.pain is not None]
    axes[0].plot(pain_days, pain_values, marker="o", color="#0D6169", linewidth=1.8)
    axes[0].set_ylim(0, 10)
    axes[0].set_ylabel("Pain (0-10)")
    axes[0].set_title("Pain reported each session", loc="left", fontsize=11)
    axes[0].grid(alpha=0.25)

    # -- movement strategy --------------------------------------------------
    rate_days: list[int] = []
    rates: list[float] = []
    unscored: list[int] = []
    for session in ordered:
        summary = CompensationSummary.from_reps(session.reps)
        if summary.metrics is None:
            unscored.append(session.protocol_day)
            continue  # nothing to plot -- do NOT draw a zero
        rate_days.append(session.protocol_day)
        rates.append(summary.metrics.flag_rate * 100)

    axes[1].plot(rate_days, rates, marker="o", color="#8A6A3A", linewidth=1.8)
    for day in unscored:
        axes[1].axvline(day, color="#C9D4D6", linestyle=":", linewidth=1)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("% hip-driven")
    axes[1].set_title(
        "Stands driven through the hips instead of the knee", loc="left", fontsize=11
    )
    axes[1].grid(alpha=0.25)
    if unscored:
        axes[1].text(
            0.99,
            0.94,
            "dotted = session not clear enough to score",
            transform=axes[1].transAxes,
            ha="right",
            fontsize=8,
            color="#5C7079",
        )

    # -- swelling and what was blocked --------------------------------------
    axes[2].axis("off")
    lines = ["Swelling reported, and what was kept off the plan", ""]
    for session in ordered:
        swelling = (
            session.swelling.report.value
            if session.swelling and session.swelling.report
            else "not compared"
        )
        summary = CompensationSummary.from_reps(session.reps)
        scored = summary.metrics.reps_scored if summary.metrics else 0
        lines.append(
            f"Day {session.protocol_day:>3}   swelling: {swelling:<13} "
            f"stands scored: {scored:<3} status: {session.status.value}"
        )
    lines += ["", COPY["sheet_footer"]]

    axes[2].text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9.5,
        family="monospace",
        color="#15242B",
    )

    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(out_path, dpi=110)
    plt.close(figure)
    return out_path


class SheetView(QWidget):
    """Shows the rendered sheet, and where it was written."""

    back = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 24, 20, 24)
        layout.setSpacing(12)

        title = QLabel("Recovery sheet")
        title.setObjectName("title")
        layout.addWidget(title)

        self._caption = QLabel("")
        self._caption.setObjectName("subtitle")
        self._caption.setWordWrap(True)
        layout.addWidget(self._caption)

        self._image = QLabel("")
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setObjectName("card")
        self._image.setMinimumHeight(520)
        layout.addWidget(self._image, 1)

        back = QPushButton("Back")
        back.clicked.connect(self.back.emit)
        layout.addWidget(back)

    def present(self, sessions: list[RehabSession], out_dir: Path | None = None) -> Path:
        target = (out_dir or DEFAULT_SHEET_DIR) / "recovery_sheet.png"
        render_sheet(sessions, target)

        pixmap = QPixmap(str(target))
        self._image.setPixmap(
            pixmap.scaled(
                self._image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        count = len(sessions)
        if count == 0:
            self._caption.setText("No sessions recorded yet. This fills in as you go.")
        else:
            self._caption.setText(
                f"{count} session{'s' if count != 1 else ''}. Saved to {target}"
            )
        return target
