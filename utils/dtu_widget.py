from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QPushButton

from utils.dtu_comp import Slider_With_Marks

class Menu_Widget(QtWidgets.QMenuBar):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.menu_layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.menu_layout)

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
                                        ]
                                    },
                                    ...
                                }
        """
        for menu_name, config in menu_config.items():
            display_name = config.get("display_name", menu_name)
            menu = self.addMenu(display_name)
            
            for action_text, action_func in config["buttons"]:
                action = menu.addAction(action_text)
                action.triggered.connect(action_func)

###################################################################################################################################################

class Progress_Widget(QtWidgets.QWidget):
    frame_changed = Signal(int)

    def __init__(self):
        super().__init__()
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        
        self.layout = QtWidgets.QHBoxLayout(self)

        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(20)
        self.play_button.clicked.connect(self.toggle_playback)

        self.progress_slider = Slider_With_Marks(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setTracking(True)
        self.progress_slider.sliderMoved.connect(self.handle_slider_move)

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

class Nav_Widget(QtWidgets.QGroupBox):
    """A modular QGroupBox widget for video navigation controls."""
    frame_changed_sig = Signal(int)
    prev_marked_frame_sig = Signal()
    next_marked_frame_sig = Signal()

    def __init__(self, mark_name="Marked", parent=None):
        super().__init__(parent)
        self.title = "Video Navigation"
        self.marked_name = mark_name
        self.navigation_layout = QtWidgets.QGridLayout(self)
        self._create_buttons()

    def _create_buttons(self):
        self.prev_10_frames_button = QtWidgets.QPushButton("Prev 10 Frames (Shift + ←)")
        self.prev_frame_button = QtWidgets.QPushButton("Prev Frame (←)")
        self.next_frame_button = QtWidgets.QPushButton("Next Frame (→)")
        self.next_10_frames_button = QtWidgets.QPushButton("Next 10 Frames (Shift + →)")
        self.prev_marked_frame_button = QtWidgets.QPushButton(f"◄ Prev {self.marked_name} (↑)")
        self.next_marked_frame_button = QtWidgets.QPushButton(f"► Next {self.marked_name} (↓)")

        self.navigation_layout.addWidget(self.prev_10_frames_button, 0, 0)
        self.navigation_layout.addWidget(self.next_10_frames_button, 1, 0)
        self.navigation_layout.addWidget(self.prev_frame_button, 0, 1)
        self.navigation_layout.addWidget(self.next_frame_button, 1, 1)
        self.navigation_layout.addWidget(self.prev_marked_frame_button, 0, 2)
        self.navigation_layout.addWidget(self.next_marked_frame_button, 1, 2)

        # Connect internal button signals to our custom signals
        self.prev_10_frames_button.clicked.connect(lambda: self.frame_changed_sig.emit(-10))
        self.prev_frame_button.clicked.connect(lambda: self.frame_changed_sig.emit(-1))
        self.next_frame_button.clicked.connect(lambda: self.frame_changed_sig.emit(1))
        self.next_10_frames_button.clicked.connect(lambda: self.frame_changed_sig.emit(10))
        self.prev_marked_frame_button.clicked.connect(self.prev_marked_frame_sig.emit)
        self.next_marked_frame_button.clicked.connect(self.next_marked_frame_sig.emit)

###################################################################################################################################################