from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QPushButton, QMenu, QToolButton, QWidget

from utils.dtu_ui import Slider_With_Marks

class Menu_Comp(QtWidgets.QMenuBar):
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

class Progress_Bar_Comp(QWidget):
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

