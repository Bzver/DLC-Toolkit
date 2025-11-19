import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer, Signal, QRect
from PySide6.QtWidgets import (
    QPushButton, QHBoxLayout, QVBoxLayout, QStyle, QStyleOptionSlider, QSlider, QLineEdit, QLabel, QApplication)
from PySide6.QtGui import QPainter, QColor, QImage, QIntValidator, QFont, QPixmap
from typing import List, Dict

from utils.helper import indices_to_spans

class Video_Slider_Widget(QtWidgets.QWidget):
    frame_changed = Signal(int)
    HexColor = str

    def __init__(self):
        super().__init__()
        self.total_frames, self.current_frame_idx, self.current_behav_idx = 0, 0, 0
        self.category_array, self.priority_array = None, None
        self.idx_to_color = {}
        self.is_playing = False
        self.is_zoom_slider_shown = False
        
        self.main_layout = QVBoxLayout(self)

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
        self.show_zoom_btn = QPushButton("▲")
        self.show_zoom_btn.clicked.connect(self.toggle_zoom_slider)
        self.show_zoom_btn.setFixedWidth(20)
        self.show_zoom_btn.setVisible(not self.is_zoom_slider_shown)
        
        slider_layout = QHBoxLayout()
        slider_layout.addLayout(self.fin)
        slider_layout.addWidget(self.play_button)
        slider_layout.addWidget(self.progress_slider)
        slider_layout.addWidget(self.show_zoom_btn)
        self._setup_zoom_slider()
        self.main_layout.addLayout(slider_layout)

        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(int(1000/50)) # ~50 FPS
        self.playback_timer.timeout.connect(self.advance_frame)

    def _setup_zoom_slider(self):
        zoom_slider_layout = QHBoxLayout()
        self.zoom_radius_combo = QtWidgets.QComboBox()
        self.zoom_radius_combo.addItems(["50", "100", "250", "500", "1000"])
        self.zoom_radius_combo.setCurrentText("250")
        self.zoom_radius_combo.setFixedWidth(50)
        self.zoom_radius_combo.currentTextChanged.connect(self._on_zoom_radius_changed)
        self.zoom_radius_combo.setVisible(self.is_zoom_slider_shown)

        self.zoom_slider = Zoomed_zoom_slider(Qt.Orientation.Horizontal)
        self.zoom_slider.setVisible(self.is_zoom_slider_shown)

        self.hide_zoom_btn = QPushButton("▼")
        self.hide_zoom_btn.setFixedWidth(20)
        self.hide_zoom_btn.clicked.connect(self.toggle_zoom_slider)
        self.hide_zoom_btn.setVisible(self.is_zoom_slider_shown)

        zoom_slider_layout.addWidget(self.zoom_radius_combo)
        zoom_slider_layout.addWidget(self.zoom_slider)
        zoom_slider_layout.addWidget(self.hide_zoom_btn)
        self.main_layout.addLayout(zoom_slider_layout)

    def set_total_frames(self, total_frames:int):
        self.total_frames = total_frames
        self.progress_slider.setRange(0, self.total_frames - 1)
        self.fin.set_total_frames(self.total_frames - 1)
        self.clear_frame_category()
    
    def set_current_frame(self, frame_idx:int):
        self.current_frame_idx = frame_idx
        self.progress_slider.setValue(self.current_frame_idx)
        self.fin.set_current_frame(self.current_frame_idx)
        if self.is_zoom_slider_shown:
            self.zoom_slider.set_center_frame(self.current_frame_idx, self.total_frames)

    def clear_frame_category(self):
        self.category_array = np.full((self.total_frames,), 255, dtype=np.uint8)
        self.priority_array = np.zeros_like(self.category_array, dtype=np.uint8)
        self.current_behav_idx = 0
        self.idx_to_color.clear()
        self.progress_slider.reset_category()

    def set_frame_category(self, category_name:str, frame_list:List[int], color:HexColor="#183539", priority:int=0): # category_name kept for backward compat
        frame_array = np.array(frame_list)
        current_priority = self.priority_array[frame_list]
        mask = priority >= current_priority
        frames_to_update = frame_array[mask]
        if len(frames_to_update) > 0:
            self.category_array[frames_to_update] = self.current_behav_idx
            self.priority_array[frames_to_update] = priority

        self.idx_to_color[self.current_behav_idx] = color
        self.current_behav_idx += 1

    def set_frame_category_array(self, category_array:np.ndarray, idx_to_color:Dict[int, HexColor]):
        self.category_array = category_array
        self.idx_to_color = idx_to_color

    def commit_categories(self):
        self.progress_slider.set_frame_category(self.category_array, self.idx_to_color)
        self.zoom_slider.set_full_categories(self.category_array, self.idx_to_color)
        if self.is_zoom_slider_shown:
            self.zoom_slider.set_center_frame(self.current_frame_idx, self.total_frames)

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
        if self.is_zoom_slider_shown:        
            self.zoom_slider.set_center_frame(value, self.total_frames)

    def _on_zoom_radius_changed(self, value:str):
        self.zoom_slider.set_window_radius(int(value))

    def toggle_zoom_slider(self):
        self.is_zoom_slider_shown = not self.is_zoom_slider_shown
        self.hide_zoom_btn.setVisible(self.is_zoom_slider_shown)
        self.zoom_slider.setVisible(self.is_zoom_slider_shown)
        self.zoom_radius_combo.setVisible(self.is_zoom_slider_shown)
        self.show_zoom_btn.setVisible(not self.is_zoom_slider_shown)

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
        self._category_array = None
        self._idx_to_color = {}
        self._bg_image = None
        self._dirty_spans = []

        self.NO_CATEGORY = np.uint8(255)

        self.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 20px;
                background: #B1B1B1;
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

    def reset_category(self):
        self._category_array = None
        self._idx_to_color.clear()
        self._bg_image = None
        self._dirty_spans.clear()
        self.update()

    def set_frame_category(self, category_array: np.ndarray, idx_to_color: Dict[int, str], force_full_update: bool = False):
        old_array = self._category_array
        if category_array is None:
            return
        self._category_array = category_array
        self._idx_to_color = idx_to_color

        if force_full_update or old_array is None or old_array.shape != category_array.shape:
            self._bg_image = None
            self._dirty_spans = [(0, len(category_array) - 1)] if len(category_array) > 0 else []
        else:
            changed = np.where(old_array != category_array)[0]
            self._dirty_spans = indices_to_spans(changed)
            if not self._dirty_spans:
                return

        self.update()

    def copy_background_to_clipboard(self):
        self._ensure_bg_image()
        if self._bg_image and not self._bg_image.isNull():
            pixmap = QPixmap.fromImage(self._bg_image)
            QApplication.clipboard().setPixmap(pixmap)

    def _ensure_bg_image(self):
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        w, h = int(round(w)), int(round(h))
        if self._bg_image is None or self._bg_image.size() != self.size():
            self._bg_image = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
            self._bg_image.fill(Qt.transparent)
            painter = QPainter(self._bg_image)
            painter.setPen(Qt.NoPen)
            groove_rect = self._get_groove_rect()
            painter.fillRect(groove_rect, QColor("#B1B1B1"))
            painter.end()
            if self._category_array is not None:
                n = len(self._category_array)
                self._dirty_spans = [(0, max(0, n - 1))] if n > 0 else []
            else:
                self._dirty_spans = []

    def _paint_dirty_regions(self):
        if not self._dirty_spans or self._category_array is None:
            return

        painter = QPainter(self._bg_image)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(Qt.NoPen)

        groove_rect = self._get_groove_rect()
        available_width = groove_rect.width()
        min_val, max_val = self.minimum(), self.maximum()
        total_logical_frames = max(1, max_val - min_val + 1)
        mark_width = max(1, available_width // total_logical_frames)

        color_cache = {}
        for idx, hex_color in self._idx_to_color.items():
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            color_cache[idx] = QColor(r, g, b)

        no_category_color = QColor(0, 0, 0, 0)

        for start_f, end_f in self._dirty_spans:
            x1_clear = self._frame_to_x(start_f, groove_rect, available_width, min_val, max_val)
            x2_clear = self._frame_to_x(end_f, groove_rect, available_width, min_val, max_val)
            clear_rect = QRect(
                int(x1_clear) - mark_width // 2 - 1,
                groove_rect.top(),
                int(x2_clear - x1_clear) + mark_width + 2,
                groove_rect.height()
            )
            painter.fillRect(clear_rect, QColor("#B1B1B1"))

            if start_f >= len(self._category_array):
                continue
            end_clip = min(end_f + 1, len(self._category_array))
            sub_arr = self._category_array[start_f:end_clip]
            if len(sub_arr) == 0:
                continue

            current_cat = sub_arr[0]
            local_start = 0

            for i in range(1, len(sub_arr) + 1):
                cat = sub_arr[i] if i < len(sub_arr) else self.NO_CATEGORY
                if cat != current_cat or i == len(sub_arr):
                    frame_start = start_f + local_start
                    frame_end = start_f + i - 1

                    if current_cat != self.NO_CATEGORY:
                        color = color_cache.get(current_cat, no_category_color)
                        if color.alpha() > 0:
                            painter.setBrush(color)
                            x1 = self._frame_to_x(frame_start, groove_rect, available_width, min_val, max_val)
                            x2 = self._frame_to_x(frame_end, groove_rect, available_width, min_val, max_val)

                            x1_i = int(round(x1))
                            x2_i = int(round(x2))
                            rect_w = max(mark_width, x2_i - x1_i + mark_width)

                            painter.drawRect(
                                x1_i - mark_width // 2,
                                groove_rect.top(),
                                rect_w,
                                groove_rect.height()
                            )
                    current_cat = cat
                    local_start = i

        painter.end()
        self._dirty_spans.clear()

    def _get_groove_rect(self) -> QRect:
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        return self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)

    def _frame_to_x(self, frame_idx: int, groove_rect: QRect, available_width: int, min_val: int, max_val: int) -> float:
        return QStyle.sliderPositionFromValue(min_val, max_val, frame_idx, available_width, False) + groove_rect.left()

    def paintEvent(self, event):
        self._ensure_bg_image()
        if self._bg_image is None:
            super().paintEvent(event)
            return

        if self._dirty_spans:
            self._paint_dirty_regions()

        painter = QPainter(self)
        painter.drawImage(0, 0, self._bg_image)

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        opt.subControls = QStyle.SC_SliderHandle
        self.style().drawComplexControl(QStyle.CC_Slider, opt, painter, self)
        painter.end()

    def resizeEvent(self, event):
        self._bg_image = None
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self.orientation() == Qt.Orientation.Horizontal:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            groove_rect = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
            handle_rect = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
            pos = event.position().toPoint()
            if groove_rect.contains(pos) and not handle_rect.contains(pos):
                min_val, max_val = self.minimum(), self.maximum()
                groove_start = groove_rect.left()
                groove_width = groove_rect.width()
                if groove_width > 0:
                    ratio = (pos.x() - groove_start) / groove_width
                    value = int(round(min_val + ratio * (max_val - min_val)))
                    value = max(min_val, min(value, max_val))
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
        self.total_line.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

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

class Zoomed_zoom_slider(Slider_With_Marks):
    def __init__(self, orientation):
        super().__init__(orientation)
        self._window_radius = 250
        self._center_frame = 0
        self._total_frames = 0
        self._full_category_array = None
        self._full_idx_to_color = {}

        self.setEnabled(False)
        self.setFocusPolicy(Qt.NoFocus)

        self.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 10px;
                background: #E0E0E0;
                margin: 2px 0;
            }
        """)

    def set_full_categories(self, category_array: np.ndarray, idx_to_color: Dict[int, str]):
        self._full_category_array = category_array
        self._full_idx_to_color = idx_to_color
        self._update_range_and_background()

    def set_window_radius(self, radius: int):
        self._window_radius = radius
        self._update_range_and_background()

    def set_center_frame(self, frame_idx: int, total_frames: int):
        self._center_frame = np.clip(frame_idx, 0, total_frames - 1)
        self._total_frames = total_frames
        self._update_range_and_background()

    def _update_range_and_background(self):
        if self._total_frames <= 0:
            self.setRange(0, 0)
            self.reset_category()
            return

        start = max(0, self._center_frame - self._window_radius)
        end = min(self._total_frames - 1, self._center_frame + self._window_radius)
        logical_span = end - start
        self.setRange(0, max(0, logical_span))

        if self._full_category_array is not None:
            sub_cat = self._full_category_array[start:end+1]
            self.set_frame_category(sub_cat, self._full_idx_to_color, force_full_update=True)
        else:
            self.reset_category()