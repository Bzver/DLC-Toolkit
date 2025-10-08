from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QPushButton, QHBoxLayout, QStyle, QStyleOptionSlider, QSlider
from PySide6.QtGui import QPainter, QColor
from typing import List

class Video_Slider_Widget(QtWidgets.QWidget):
    frame_changed = Signal(int)
    HexColor = str

    def __init__(self):
        super().__init__()
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        
        self.slider_layout = QHBoxLayout(self)

        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(20)
        self.play_button.clicked.connect(self.toggle_playback)

        self.progress_slider = Slider_With_Marks(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setTracking(True)
        self.progress_slider.sliderMoved.connect(self.handle_slider_move)
        self.progress_slider.frame_changed.connect(self.handle_slider_move)

        self.slider_layout.addWidget(self.play_button)
        self.slider_layout.addWidget(self.progress_slider)

        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(int(1000/50)) # ~50 FPS
        self.playback_timer.timeout.connect(self.advance_frame)

    def set_frame_category(self, category_name:str, frame_list:List[int], color:HexColor="#183539", priority:int=0):
        """
        Public API to pass the slider mark properties

        Args:
            category_name (str): The name of the category to assign to the specified frames.
            frame_list (List[int]): A list of frame indices to be associated with the category.
            color (HexColor): The hexadecimal color code (e.g., '#FF55A3') used to style the frames in this category
            priority (int): The rendering priority of the category. 
                The higher the priority, the more prominently the category will be displayed.

        """
        self.progress_slider.set_frame_category(category_name, frame_list, color, priority)

    def set_slider_range(self, total_frames:int):
        self.total_frames = total_frames
        self.progress_slider.setRange(0, self.total_frames - 1)
    
    def set_current_frame(self, frame_idx:int):
        self.current_frame = frame_idx
        self.progress_slider.setValue(self.current_frame)

    def handle_slider_move(self, value:int):
        self.current_frame = value
        self.frame_changed.emit(self.current_frame)

    def toggle_playback(self):
        if not self.is_playing:
            self._start_playback()
        else:
            self._stop_playback()
        
    def advance_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.set_current_frame(self.current_frame)
            self.frame_changed.emit(self.current_frame)
        else:
            self._stop_playback()
            
    def _start_playback(self):
        if self.playback_timer:
            self.is_playing = True
            self.play_button.setText("■")
            self.playback_timer.start()

    def _stop_playback(self):
        if self.playback_timer:
            self.is_playing = False
            self.play_button.setText("▶")
            self.playback_timer.stop()

class Slider_With_Marks(QSlider):
    frame_changed = Signal(int)

    def __init__(self, orientation):
        super().__init__(orientation)
        self.frame_categories = {} # {category_name: set_of_frames}
        self.category_colors = {} # {category_name: color_string}
        self.category_priorities = {} # {category_name: priority_int}
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #B1B1B1, stop:1 #B1B1B1);
                margin: 2px 0;
            }
            
            QSlider::handle:horizontal {
                background: transparent;
                border: 1px solid #5c5c5c;
                width: 5px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)

    def set_frame_category(self, category_name, frames, color, priority):
        self.frame_categories[category_name] = set(frames)
        self.category_colors[category_name] = color
        self.category_priorities[category_name] = priority # Store the priority
        self.update() # Request a repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        
        # Manually draw groove only
        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider,
            opt,
            QStyle.SC_SliderGroove,
            self
        )
        opt.subControls = QStyle.SC_SliderGroove
        self.style().drawComplexControl(QStyle.CC_Slider, opt, painter, self)

        # Draw marks on top of groove, but below handle
        if self.frame_categories:
            min_val = self.minimum()
            max_val = self.maximum()
            available_width = groove_rect.width()
            total_frames = max_val - min_val + 1
            mark_width = max(1, int(available_width / total_frames))

            frame_colors_to_plot = {}

            sorted_categories = sorted(
                self.frame_categories.keys(),
                key=lambda cat: self.category_priorities.get(cat, float('inf'))
            )

            for category_name in sorted_categories:
                frames = self.frame_categories.get(category_name, set())
                color = self.category_colors.get(category_name)
                if frames and color:
                    for frame in frames:
                        if min_val <= frame <= max_val:
                            frame_colors_to_plot[frame] = color

            painter.setPen(Qt.NoPen)
            for frame, color in frame_colors_to_plot.items():
                pos = QStyle.sliderPositionFromValue(
                    min_val, max_val, frame, available_width, opt.upsideDown
                ) + groove_rect.left()

                painter.setBrush(QColor(color))
                painter.drawRect(
                    int(pos) - mark_width // 2, # Center the mark
                    groove_rect.top(),
                    mark_width,  # Width of mark
                    groove_rect.height()
                )

        # Draw the handle
        opt.subControls = QStyle.SC_SliderHandle
        self.style().drawComplexControl(QStyle.CC_Slider, opt, painter, self)

        painter.end()

    def mousePressEvent(self, event):
        if self.orientation() == Qt.Orientation.Horizontal:
            pos = event.position().x()
            slider_length = self.width()

        new_value = (self.maximum() - self.minimum()) * pos / slider_length + self.minimum()
        self.setValue(int(new_value))
        self.frame_changed.emit(new_value)
        super().mousePressEvent(event)
