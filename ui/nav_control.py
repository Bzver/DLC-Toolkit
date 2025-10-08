from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QFrame, QHBoxLayout, QPushButton, QLabel
from PySide6.QtGui import QFont

class Nav_Widget(QtWidgets.QWidget):
    frame_changed_sig = Signal(int)
    prev_marked_frame_sig = Signal()
    next_marked_frame_sig = Signal()

    def __init__(self, mark_name="Marked", parent=None):
        super().__init__(parent)
        self.marked_name = mark_name
        self.collapsed = False

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Header bar
        self.header_frame = QFrame()
        self.header_frame.setStyleSheet("""
            QFrame {
                background-color: #d3d7cf;
                border: 1px solid #a0a0a0;
                border-radius: 4px;
            }
        """)
        self.header_layout = QHBoxLayout(self.header_frame)
        self.header_layout.setContentsMargins(6, 4, 6, 4)
        self.header_layout.setSpacing(6)

        # Toggle button
        self.toggle_button = QPushButton("▼")
        self.toggle_button.setFixedSize(16, 16)
        font = QFont("Arial", 8)
        font.setBold(True)
        self.toggle_button.setFont(font)
        self.toggle_button.clicked.connect(self._toggle_collapsed)

        # Title label
        self.title_label = QLabel("Video Navigation")
        self.title_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.header_layout.addWidget(self.toggle_button)
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()

        # Content frame
        self.content_frame = QFrame()
        self.content_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #a0a0a0;
                border-top: none;
                background-color: #f8f8f8;
            }
        """)
        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.setSpacing(6)
        self.content_layout.setContentsMargins(8, 6, 8, 6)

        self._create_buttons()

        self.main_layout.addWidget(self.header_frame)
        self.main_layout.addWidget(self.content_frame)

    def setTitle(self, title_text):
        self.title_label.setText(title_text)

    def setTitleColor(self, color_hex):
        self.title_label.setStyleSheet(f"color: {color_hex}; font-weight: bold;")

    def _create_buttons(self):
        self.prev_10_frames_button = QPushButton("Prev 10 Frames (Shift + ←)")
        self.prev_frame_button = QPushButton("Prev Frame (←)")
        self.prev_marked_frame_button = QPushButton(f"◄ Prev {self.marked_name} (↑)")

        row1 = QHBoxLayout()
        row1.addWidget(self.prev_10_frames_button)
        row1.addWidget(self.prev_frame_button)
        row1.addWidget(self.prev_marked_frame_button)

        self.next_10_frames_button = QPushButton("Next 10 Frames (Shift + →)")
        self.next_frame_button = QPushButton("Next Frame (→)")
        self.next_marked_frame_button = QPushButton(f"► Next {self.marked_name} (↓)")

        row2 = QHBoxLayout()
        row2.addWidget(self.next_10_frames_button)
        row2.addWidget(self.next_frame_button)
        row2.addWidget(self.next_marked_frame_button)

        self.content_layout.addLayout(row1)
        self.content_layout.addLayout(row2)

        # Connect signals
        self.prev_10_frames_button.clicked.connect(lambda: self.frame_changed_sig.emit(-10))
        self.prev_frame_button.clicked.connect(lambda: self.frame_changed_sig.emit(-1))
        self.next_frame_button.clicked.connect(lambda: self.frame_changed_sig.emit(1))
        self.next_10_frames_button.clicked.connect(lambda: self.frame_changed_sig.emit(10))
        self.prev_marked_frame_button.clicked.connect(self.prev_marked_frame_sig.emit)
        self.next_marked_frame_button.clicked.connect(self.next_marked_frame_sig.emit)

    def _toggle_collapsed(self):
        """Toggle collapse/expand by showing/hiding content."""
        self.collapsed = not self.collapsed
        self.toggle_button.setText("►" if self.collapsed else "▼")
        self.content_frame.setVisible(not self.collapsed)

    def set_collapsed(self, collapsed: bool):
        """Allow external code to collapse or expand the widget."""
        if collapsed != self.collapsed:
            self._toggle_collapsed()