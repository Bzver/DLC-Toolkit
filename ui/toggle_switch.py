from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame
from PySide6.QtCore import Signal, Qt, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont

class Toggle_Switch(QWidget):
    toggled = Signal(bool)

    def __init__(self, label_text: str = "", parent: QWidget = None):
        super().__init__(parent)
        self._is_checked = False
        self._label_text = label_text
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignLeft)

        if self._label_text:
            self.label = QLabel(self._label_text)
            layout.addWidget(self.label)

        self.track = Toggle_Track(self)
        layout.addWidget(self.track)

    def _on_clicked(self):
        self.set_checked(not self._is_checked)

    def set_checked(self, checked: bool):
        if self._is_checked != checked:
            self._is_checked = checked
            self.track.set_checked(checked)
            self.toggled.emit(self._is_checked)

    def is_checked(self) -> bool:
        return self._is_checked

    def set_label_text(self, text: str):
        self._label_text = text
        if hasattr(self, 'label'):
            self.label.setText(text)
        else:
            self.label = QLabel(text)
            self.layout().insertWidget(0, self.label)


class Toggle_Track(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 26)
        self._checked = False
        self._handle_position = -8

        self.setCursor(Qt.PointingHandCursor)
        self.mousePressEvent = lambda _: parent._on_clicked() if parent else None

        self._animation = QPropertyAnimation(self, b"handle_position", self)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.setDuration(200)

    def get_handle_position(self):
        return self._handle_position

    def set_handle_position(self, pos):
        self._handle_position = pos - 10
        self.update()

    handle_position = Property(int, get_handle_position, set_handle_position)

    def set_checked(self, checked: bool):
        self._checked = checked
        end_pos = 38 if checked else 2  # Adjusted for new size
        if self._animation.state() == QPropertyAnimation.Running:
            self._animation.stop()
        self._animation.setEndValue(end_pos)
        self._animation.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        # Track background
        track_color = QColor("#4CAF50") if self._checked else QColor("#f44336")
        painter.setBrush(QBrush(track_color))
        painter.drawRoundedRect(self.rect().adjusted(2, 4, -2, -4), 6, 6)

        # Handle
        handle_rect = self.rect().adjusted(
            self._handle_position, 3,
            self._handle_position - 20, -3
        ).normalized()
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.setPen(QPen(QColor("#CCCCCC"), 1))
        painter.drawRoundedRect(handle_rect, 6, 6)

        # --- ON / OFF Text ---
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)

        # OFF (right)
        off_color = QColor("#FFFFFF") if not self._checked else QColor(255, 255, 255, 120)
        painter.setPen(off_color)
        painter.drawText(self.rect().adjusted(0, 0, -4, 0), Qt.AlignRight | Qt.AlignVCenter, "OFF")

        # ON (left)
        on_color = QColor("#FFFFFF") if self._checked else QColor(255, 255, 255, 120)
        painter.setPen(on_color)
        painter.drawText(self.rect().adjusted(4, 0, 0, 0), Qt.AlignLeft | Qt.AlignVCenter, "ON")