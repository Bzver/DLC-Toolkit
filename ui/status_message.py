from PySide6.QtWidgets import QLabel, QHBoxLayout, QWidget
from PySide6.QtCore import QTimer, Qt

class Status_Bar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 2, 5, 2)
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.layout.addWidget(self.label)
        self.setStyleSheet("""
            Status_Bar {
                background-color: #f0f0f0;
                border-top: 1px solid #ccc;
                padding: 2px;
            }
            QLabel {
                color: #333;
                font-size: 12px;
            }
        """)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._clear_message)

    def show_message(self, message: str, duration_ms: int = 3000):
        """Show a temporary status message."""
        self.label.setText(message)
        self.setVisible(True)
        if duration_ms > 0:
            self.timer.start(duration_ms)
        else:
            self.timer.stop()  # persistent until next message or manual clear

    def clear_message(self):
        """Manually clear the message."""
        self._clear_message()

    def on_hold_message(self):
        self.show_message("Processing... please wait", duration_ms=0)

    def _clear_message(self):
        self.label.setText("")
        self.setVisible(False)