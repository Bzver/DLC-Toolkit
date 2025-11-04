from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QPushButton, QHBoxLayout, QStyle, QStyleOptionSlider, QSlider, QLineEdit, QLabel, QApplication
from PySide6.QtGui import QPainter, QColor, QIntValidator, QFont, QPixmap
from typing import List

class Video_Slider_Widget(QtWidgets.QWidget):
    frame_changed = Signal(int)
    HexColor = str

    def __init__(self):
        super().__init__()
        self.total_frames = 0
        self.current_frame_idx = 0
        self.is_playing = False
        
        self.slider_layout = QHBoxLayout(self)

        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(20)
        self.play_button.clicked.connect(self.toggle_playback)

        self.fin = Frame_Input()
        self.fin.frame_changed_sig.connect(self._handle_frame_input)

        self.progress_slider = Slider_With_Marks(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setTracking(True)
        self.progress_slider.sliderMoved.connect(self._handle_slider_move)
        self.progress_slider.frame_changed.connect(self._handle_slider_move)

        self.slider_layout.addLayout(self.fin)
        self.slider_layout.addWidget(self.play_button)
        self.slider_layout.addWidget(self.progress_slider)

        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(int(1000/50)) # ~50 FPS
        self.playback_timer.timeout.connect(self.advance_frame)

    def clear_frame_category(self):
        """Public API to clear existing categories"""
        self.progress_slider.clear_frame_category()

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

    def set_total_frames(self, total_frames:int):
        self.total_frames = total_frames
        self.progress_slider.setRange(0, self.total_frames - 1)
        self.fin.set_total_frames(self.total_frames - 1)
    
    def set_current_frame(self, frame_idx:int):
        self.current_frame_idx = frame_idx
        self.progress_slider.setValue(self.current_frame_idx)
        self.fin.set_current_frame(self.current_frame_idx)

    def export_background(self):
        self.progress_slider.copy_background_to_clipboard()

    def _handle_slider_move(self, value:int):
        self.current_frame_idx = value
        self.fin.set_current_frame(value)
        self.frame_changed.emit(self.current_frame_idx)

    def _handle_frame_input(self, value:int):
        self.current_frame_idx = value
        self.progress_slider.setValue(value)
        self.frame_changed.emit(self.current_frame_idx)

    def toggle_playback(self):
        if not self.is_playing:
            self._start_playback()
        else:
            self._stop_playback()
        
    def advance_frame(self):
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.set_current_frame(self.current_frame_idx)
            self.frame_changed.emit(self.current_frame_idx)
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
        self.frame_categories = {}
        self.category_colors = {}
        self.category_priorities = {}
        self._background_pixmap = None

        self.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #B1B1B1, stop:1 #B1B1B1);
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: rgba(255, 255, 255, 150);
                border: 2px solid #5c5c5c;
                width: 6px;
                margin: -2px -3px -2px -3px;
                border-radius: 3px;
            }
        """)

    def clear_frame_category(self):
        self.frame_categories.clear()
        self.category_colors.clear()
        self.category_priorities.clear()
        self._background_pixmap = None
        self.update()

    def set_frame_category(self, category_name, frames, color, priority):
        self.frame_categories[category_name] = set(frames)
        self.category_colors[category_name] = color
        self.category_priorities[category_name] = priority
        self._background_pixmap = None
        self.update()

    def copy_background_to_clipboard(self) -> bool:
        if self._background_pixmap is None:
            self._render_background()
        if self._background_pixmap and not self._background_pixmap.isNull():
            clipboard = QApplication.clipboard()
            clipboard.setPixmap(self._background_pixmap)

    def _render_background(self):
        if self.width() <= 0 or self.height() <= 0:
            return

        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        # Draw groove
        groove_rect = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        opt.subControls = QStyle.SC_SliderGroove
        self.style().drawComplexControl(QStyle.CC_Slider, opt, painter, self)

        # Draw marks
        if self.frame_categories:
            min_val = self.minimum()
            max_val = self.maximum()
            total_frames = max(1, max_val - min_val + 1)
            available_width = groove_rect.width()
            if available_width > 0:
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
                        int(pos) - mark_width // 2,
                        groove_rect.top(),
                        mark_width,
                        groove_rect.height()
                    )

        painter.end()
        self._background_pixmap = pixmap

    def paintEvent(self, event):
        if self._background_pixmap is None:
            self._render_background()

        painter = QPainter(self)

        if self._background_pixmap:
            painter.drawPixmap(0, 0, self._background_pixmap)

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        opt.subControls = QStyle.SC_SliderHandle
        self.style().drawComplexControl(QStyle.CC_Slider, opt, painter, self)

        painter.end()

    def resizeEvent(self, event):
        self._background_pixmap = None
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self.orientation() == Qt.Orientation.Horizontal:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            groove_rect = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
            handle_rect = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)

            pos = event.position().toPoint()

            if groove_rect.contains(pos) and not handle_rect.contains(pos):
                slider_min = self.minimum()
                slider_max = self.maximum()
                slider_range = slider_max - slider_min
                if slider_range == 0:
                    value = slider_min
                else:
                    groove_start = groove_rect.left()
                    groove_end = groove_rect.right()
                    groove_width = groove_end - groove_start
                    if groove_width <= 0:
                        value = slider_min
                    else:
                        ratio = (pos.x() - groove_start) / groove_width
                        value = int(round(slider_min + ratio * slider_range))
                        value = max(slider_min, min(value, slider_max))

                self.setValue(value)
                self.frame_changed.emit(value)
                return
            
        super().mousePressEvent(event)

class Frame_Input(QHBoxLayout):
    frame_changed_sig = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_frame = 1

        self.frame_input = QLineEdit("0")
        validator = QIntValidator(0, 2147483647, self)
        self.frame_input.setValidator(validator)
        self.frame_input.textChanged.connect(self._on_frame_idx_input)
        self.frame_input.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        separator = QLabel("|")
        self.total_line = QLineEdit("0")
        self.total_line.setReadOnly(True)

        self.frame_input.setFixedWidth(50)
        separator.setFixedWidth(10)
        separator.setFont(QFont("Arial", 10, QFont.Bold))
        self.total_line.setFixedWidth(50)

        self.addWidget(self.frame_input)
        self.addWidget(separator)
        self.addWidget(self.total_line)

    def set_current_frame(self, frame_idx:int):
        self.frame_input.blockSignals(True)
        self.frame_input.setText(str(frame_idx))
        self.frame_input.blockSignals(False)

    def set_total_frames(self, total_frames:int):
        self.total_line.setText(str(total_frames))
        self.max_frame = total_frames

    def _on_frame_idx_input(self):
        frame_input_text = self.frame_input.text()
        try:
            frame_idx = int(frame_input_text)
        except ValueError:
            frame_idx = 0

        if frame_idx > self.max_frame:
            self.frame_input.setText(str(self.max_frame))
            frame_idx = self.max_frame

        self.frame_changed_sig.emit(frame_idx)