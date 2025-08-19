from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QFrame
from PySide6.QtGui import QFont

from utils.dtu_comp import Slider_With_Marks

class Menu_Widget(QtWidgets.QMenuBar):
    def __init__(self, parent=None):
        super().__init__(parent)

    def add_menu_from_config(self, menu_config):
        """
        Adds menus and their actions based on a configuration dictionary.
        Args:
            menu_config (dict): A dictionary defining the menu structure.
            Example:
            {
                "File": {
                    "display_name": "File",
                    "buttons": [
                        ("Load Video", load_video_function),
                        ("Load Config and Prediction", load_prediction_function),
                        ("Show Axis", toggle_axis, {"checkable": True, "checked": True}),
                    ]
                },
                ...
            }
        """
        for menu_name, config in menu_config.items():
            display_name = config.get("display_name", menu_name)
            menu = self.addMenu(display_name)

            buttons = config.get("buttons", [])
            for item in buttons:
                if len(item) == 2:
                    action_text, action_func = item
                    action = menu.addAction(action_text)
                    action.triggered.connect(action_func)
                elif len(item) == 3:
                    action_text, action_func, options = item
                    action = menu.addAction(action_text)
                    action.triggered.connect(action_func)
                    if options and options.get("checkable"):
                        action.setCheckable(True)
                        action.setChecked(options.get("checked", False))
                else:
                    raise ValueError(
                        "Menu button must be a tuple of length 2 (text, func) "
                        "or 3 (text, func, options)"
                    )

###################################################################################################################################################

class Progress_Bar_Widget(QtWidgets.QWidget):
    frame_changed = Signal(int)

    def __init__(self):
        super().__init__()
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        
        self.layout = QHBoxLayout(self)

        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(20)
        self.play_button.clicked.connect(self.toggle_playback)

        self.progress_slider = Slider_With_Marks(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setTracking(True)
        self.progress_slider.sliderMoved.connect(self.handle_slider_move)
        self.progress_slider.frame_changed.connect(self.handle_slider_move)

        self.layout.addWidget(self.play_button)
        self.layout.addWidget(self.progress_slider)

        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(int(1000/50)) # ~50 FPS
        self.playback_timer.timeout.connect(self.advance_frame)

    def set_frame_category(self, category_name, frames, color=None, priority=0): # Public API to pass the slider mark properties
        self.progress_slider.set_frame_category(category_name, frames, color, priority)

    def set_slider_range(self, total_frames):
        self.total_frames = total_frames
        self.progress_slider.setRange(0, self.total_frames - 1)
    
    def set_current_frame(self, frame_number):
        self.current_frame = frame_number
        self.progress_slider.setValue(self.current_frame)

    def handle_slider_move(self, value):
        self.current_frame = value
        self.frame_changed.emit(self.current_frame)
    
    def advance_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.set_current_frame(self.current_frame)
            self.frame_changed.emit(self.current_frame)
        else:
            self.stop_playback()

    def toggle_playback(self):
        if not self.is_playing:
            self.start_playback()
        else:
            self.stop_playback()
    
    def start_playback(self):
        if self.playback_timer:
            self.is_playing = True
            self.play_button.setText("■")
            self.playback_timer.start()

    def stop_playback(self):
        if self.playback_timer:
            self.is_playing = False
            self.play_button.setText("▶")
            self.playback_timer.stop()

###################################################################################################################################################

class Nav_Widget(QtWidgets.QWidget):
    """
    Custom collapsible navigation widget with collapse button beside the title.
    Built for PySide6.
    """
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

        # Header bar (acts like a styled group box title)
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

        # Toggle button (triangle arrow)
        self.toggle_button = QPushButton("▼")
        self.toggle_button.setFixedSize(16, 16)
        font = QFont("Arial", 8)
        font.setBold(True)
        self.toggle_button.setFont(font)
        self.toggle_button.clicked.connect(self._toggle_collapsed)

        # Title label
        self.title_label = QtWidgets.QLabel("Video Navigation")
        self.title_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Assemble header
        self.header_layout.addWidget(self.toggle_button)
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()

        # Content frame (holds navigation buttons)
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

        # Add header and content to main layout
        self.main_layout.addWidget(self.header_frame)
        self.main_layout.addWidget(self.content_frame)

        # Smooth animation for collapse/expand
        self.animation = QPropertyAnimation(self.content_frame, b"maximumHeight")
        self.animation.setDuration(240)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

    def setTitle(self, title_text):
        self.title_label.setText(title_text)

    def setTitleColor(self, color_hex):
        self.title_label.setStyleSheet(f"color: {color_hex}; font-weight: bold;")

    def _create_buttons(self):
        """Create navigation buttons and arrange them in a grid-like layout."""
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

        # Add rows to content layout
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
        """Toggle collapse/expand state with animation."""
        self.collapsed = not self.collapsed
        self.toggle_button.setText("►" if self.collapsed else "▼")

        self.animation.stop()

        if self.collapsed:
            start_height = self.content_frame.sizeHint().height()
            self.animation.setStartValue(start_height)
            self.animation.setEndValue(0)
            self.animation.start()
        else:
            self.content_frame.show()
            self.content_frame.setMaximumHeight(16777215)  # Large value instead of hiding

            height = self.content_frame.sizeHint().height()

            self.animation.setStartValue(0)
            self.animation.setEndValue(height)
            self.animation.start()

            def on_animation_finished():
                if not self.collapsed:
                    self.content_frame.setMaximumHeight(16777215)  # No limit
            self.animation.finished.connect(on_animation_finished, Qt.UniqueConnection)

    def set_collapsed(self, collapsed: bool):
        """Allow external code to collapse or expand the widget."""
        if collapsed != self.collapsed:
            self._toggle_collapsed()

###################################################################################################################################################