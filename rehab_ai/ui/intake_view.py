"""
intake_view.py
Pain, swelling, and the camera-orientation confirmation.

THE ORIENTATION STEP
====================
This is the screen that carries CP 3's contract to the patient. The app cannot
verify from the image which leg faces the camera -- inferring it is exactly the
per-frame side-guessing that Issue 6 removed. So it asks, once, and stores the
answer on the session.

That makes a session framed the wrong way round identifiable afterwards rather
than silently wrong.

ABSENCE IS A STATE, NOT A DEFAULT
=================================
The pain slider starts unset. There is no "0 by default" -- 0 is a real score
meaning no pain at all, and a patient who skipped the question has not reported
that. Continue stays disabled until both are answered.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from rehab_ai.models.session import InputSource, PainReport, Side, SwellingReport


class IntakeView(QWidget):
    """Collects the check-in. Emits once, with everything or not at all."""

    submitted = Signal(object, object, object)  # PainReport, SwellingReport, Side

    def __init__(self, operated_side: Side) -> None:
        super().__init__()
        self._operated_side = operated_side
        self._pain: int | None = None
        self._swelling: SwellingReport | None = None
        self._facing: Side | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 32, 28, 28)
        layout.setSpacing(18)

        title = QLabel("Today's check-in")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Two questions, then we'll set up the camera.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(8)
        layout.addWidget(self._pain_section())
        layout.addSpacing(4)
        layout.addWidget(self._swelling_section())
        layout.addSpacing(4)
        layout.addWidget(self._orientation_section())
        layout.addStretch(1)

        self._continue = QPushButton("Start the session")
        self._continue.setObjectName("primary")
        self._continue.setEnabled(False)
        self._continue.clicked.connect(self._submit)
        layout.addWidget(self._continue)

    # -- sections -----------------------------------------------------------

    def _pain_section(self) -> QWidget:
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)

        column.addWidget(QLabel("How is the pain today?"))

        self._pain_value = QLabel("Not answered yet")
        self._pain_value.setObjectName("subtitle")
        column.addWidget(self._pain_value)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 10)
        slider.setValue(0)
        slider.setPageStep(1)
        # Only a deliberate interaction counts. Without this, the slider's
        # resting position would be indistinguishable from a reported 0.
        slider.valueChanged.connect(self._on_pain)
        column.addWidget(slider)

        scale = QHBoxLayout()
        scale.addWidget(QLabel("0 · none"))
        scale.addStretch(1)
        scale.addWidget(QLabel("10 · worst"))
        column.addLayout(scale)

        return box

    def _swelling_section(self) -> QWidget:
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)

        column.addWidget(QLabel("Compared with yesterday, the knee is:"))

        row = QHBoxLayout()
        row.setSpacing(8)
        self._swelling_group = QButtonGroup(self)
        for label, value in (
            ("Puffier", SwellingReport.PUFFIER),
            ("About the same", SwellingReport.SAME),
            ("Less puffy", SwellingReport.LESS),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _, v=value: self._on_swelling(v))
            self._swelling_group.addButton(button)
            row.addWidget(button)
        column.addLayout(row)

        return box

    def _orientation_section(self) -> QWidget:
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)

        side = self._operated_side.value
        column.addWidget(QLabel("Camera setup"))

        note = QLabel(
            f"Sit side-on to the camera with your <b>{side}</b> leg — the operated "
            "one — nearest to it. Which side is facing the camera?"
        )
        note.setObjectName("subtitle")
        note.setWordWrap(True)
        column.addWidget(note)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._facing_group = QButtonGroup(self)
        for value in (Side.LEFT, Side.RIGHT):
            button = QPushButton(value.value.capitalize())
            button.setCheckable(True)
            button.clicked.connect(lambda _, v=value: self._on_facing(v))
            self._facing_group.addButton(button)
            row.addWidget(button)
        column.addLayout(row)

        self._facing_warning = QLabel("")
        self._facing_warning.setObjectName("subtitle")
        self._facing_warning.setWordWrap(True)
        column.addWidget(self._facing_warning)

        return box

    # -- state --------------------------------------------------------------

    def _on_pain(self, value: int) -> None:
        self._pain = value
        self._pain_value.setText(f"{value} out of 10")
        self._refresh()

    def _on_swelling(self, value: SwellingReport) -> None:
        self._swelling = value
        self._refresh()

    def _on_facing(self, value: Side) -> None:
        self._facing = value
        if value is not self._operated_side:
            # Not blocked -- the patient may genuinely be set up this way, and
            # the session records it. But say plainly what it costs, because a
            # session framed this way cannot assess the operated knee.
            self._facing_warning.setText(
                f"We can only watch the {self._operated_side.value} knee. "
                "Turn around so it faces the camera, or the stands won't be scored."
            )
        else:
            self._facing_warning.setText("")
        self._refresh()

    def _refresh(self) -> None:
        self._continue.setEnabled(
            self._pain is not None and self._swelling is not None and self._facing is not None
        )

    def _submit(self) -> None:
        assert self._pain is not None and self._swelling is not None
        self.submitted.emit(
            PainReport(self._pain, InputSource.TAP), self._swelling, self._facing
        )

    # -- voice path ---------------------------------------------------------

    def apply_voice(self, pain: PainReport | None, swelling: SwellingReport | None) -> None:
        """Fill in whatever a spoken check-in produced.

        Whatever it could not read stays unset, and the patient answers it by
        tap. The parser refuses to guess and this screen does not guess either.
        """
        if pain is not None:
            self._pain = pain.value
            self._pain_value.setText(f"{pain.value} out of 10  (heard)")
        if swelling is not None:
            self._swelling = swelling
            for button in self._swelling_group.buttons():
                if button.text().lower().startswith(swelling.value[:4]):
                    button.setChecked(True)
        self._refresh()
