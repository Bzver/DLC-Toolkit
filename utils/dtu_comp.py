from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QPushButton, QMenu, QToolButton, QFileDialog

class Menu_Comp(QtWidgets.QWidget):
    def __init__(self, parent):
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
            menu = QMenu(display_name, self)
            
            for action_text, action_func in config["buttons"]:
                action = menu.addAction(action_text)
                action.triggered.connect(action_func)
            
            self._create_menu_button(display_name, menu)
            
        self.menu_layout.addStretch(1)

    def _create_menu_button(self, button_text: str, menu: QMenu, alignment=Qt.AlignLeft):
        button = QToolButton()
        button.setText(button_text)
        button.setMenu(menu)
        button.setPopupMode(QToolButton.InstantPopup)
        self.menu_layout.addWidget(button, alignment=alignment)

###################################################################################################################################################

class Progress_Bar_Comp(QtWidgets.QWidget):
    frame_changed = Signal(int)
    request_total_frames = Signal()

    def __init__(self):
        super().__init__()

        self.progress_layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(self.progress_layout)

        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(40)
        self.progress_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal) # Use Qt.Orientation.Horizontal
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setTracking(True)

        self.progress_layout.addWidget(self.play_button)
        self.progress_layout.addWidget(self.progress_slider)
        
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_frame_for_autoplay)
        
        self.is_playing = False
        self._total_frames = 0

        self.progress_slider.sliderMoved.connect(self._handle_slider_moved)

        self.request_total_frames.emit()

    def set_slider_range(self, total_frames: int):
        self._total_frames = total_frames
        self.progress_slider.setRange(0, max(0, total_frames - 1))
        self.progress_slider.setValue(0)

    def set_current_frame(self, frame_idx: int):
        if 0 <= frame_idx < self._total_frames:
            self.progress_slider.setValue(frame_idx)
        else:
            if self.is_playing:
                self.toggle_playback()
            print(f"Warning: Attempted to set frame {frame_idx} which is out of range [0, {self._total_frames-1}]")

    def _handle_slider_moved(self, value: int):
        if self.is_playing:
            self.toggle_playback()
        self.frame_changed.emit(value)

    def _advance_frame_for_autoplay(self):
        current_frame = self.progress_slider.value()
        
        if current_frame < self._total_frames - 1:
            next_frame = current_frame + 1
            self.progress_slider.setValue(next_frame)
            self.frame_changed.emit(next_frame)
        else:
            self.toggle_playback()
            self.progress_slider.setValue(self._total_frames - 1)
            self.frame_changed.emit(self._total_frames - 1)

    def toggle_playback(self):
        if not self.is_playing:
            if self._total_frames == 0:
                print("Cannot play, no frames loaded.")
                return
            
            if self.progress_slider.value() >= self._total_frames - 1:
                self.set_current_frame(0)
                self.frame_changed.emit(0)

            self.playback_timer.start(int(1000 / 50))
            self.play_button.setText("■")
            self.is_playing = True
        else:
            self.playback_timer.stop()
            self.play_button.setText("▶")
            self.is_playing = False

###################################################################################################################################################

