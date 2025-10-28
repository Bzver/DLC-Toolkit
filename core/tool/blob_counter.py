import cv2
import numpy as np
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QSizePolicy, 
    QSlider, QDialog, QSpinBox, QComboBox, QPushButton
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from typing import Optional, Tuple, List

from ui import Progress_Indicator_Dialog
from core.io import Frame_Extractor
from core.dataclass import Blob_Config

class Blob_Counter(QGroupBox):
    Frame_CV2 = np.ndarray
    parameters_changed = Signal()
    frame_processed = Signal(object, int)
    video_counted = Signal(object)
    config_ready = Signal()

    def __init__(self,
                 frame_extractor:Frame_Extractor,
                 config: Optional[Blob_Config]=None,
                 request:bool=False,
                 blob_array=False,
                 parent=None):
        super().__init__(parent)
        self.setTitle("Blob-based Animal Counting Controls")

        self.frame_extractor = frame_extractor
        self.current_frame = None
        self.total_frames, self.frame_idx = 0, 0
        self.sample_frame_count = 100
        self.vid_h, self.vid_w = self.frame_extractor.get_frame_dim()
        self.roi = None

        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # UI parameters
        self.threshold = 50
        self.min_blob_area = 2000
        self.blob_type = "Dark Blobs (Max)"

        self.blb_layout = QVBoxLayout(self)
        self.setFixedWidth(200)

        # Image Display
        self.bg_display = Blob_Background(self.frame_extractor)
        self.blb_layout.addWidget(self.bg_display)

        # Histogram for blob sizes
        self.blb_hist = Blob_Histogram(self.frame_extractor)
        self.blb_layout.addLayout(self.blb_hist)
        self.blb_hist.threshold_changed.connect(self._on_blb_hist_change)

        # Controls
        self.control_gbox = QGroupBox(self)
        self.controls_layout = QVBoxLayout(self.control_gbox)
        self.control_gbox.setTitle("Blob Counter Control")
        self.blb_layout.addWidget(self.control_gbox)

        # Threshold
        self.threshold_label = QLabel("Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(self.threshold)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.threshold_value_label = QLabel(str(self.threshold))
        self.controls_layout.addWidget(self.threshold_label)
        self.controls_layout.addWidget(self.threshold_slider)
        self.controls_layout.addWidget(self.threshold_value_label)

        # Min Blob Area
        self.min_area_label = QLabel("Min Blob Area:")
        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setRange(10, 20000)
        self.min_area_slider.setValue(self.min_blob_area)
        self.min_area_slider.valueChanged.connect(self._on_min_area_changed)
        self.min_area_value_label = QLabel(str(self.min_blob_area))
        self.controls_layout.addWidget(self.min_area_label)
        self.controls_layout.addWidget(self.min_area_slider)
        self.controls_layout.addWidget(self.min_area_value_label)

        # Blob type
        self.blob_type_label = QLabel("Blob Type:")
        self.blob_type_combo = QComboBox()
        self.blob_type_combo.addItems(["Dark Blobs (Max)", "Light Blobs (Min)"])
        self.blob_type_combo.currentTextChanged.connect(self._on_blob_type_changed)
        self.controls_layout.addWidget(self.blob_type_label)
        self.controls_layout.addWidget(self.blob_type_combo)

        # BG removal
        self.bg_removal_label = QLabel("Background Removal:")
        self.bg_removal_combo = QComboBox()
        self.bg_removal_combo.addItems(["None", "Min", "Max", "Median", "Mean"])
        self.bg_removal_combo.currentTextChanged.connect(self._on_bg_removal_changed)
        
        self.sample_count_label = QLabel("Sample Frames:")
        self.sample_count_spin = QSpinBox()
        self.sample_count_spin.setRange(10, 10000)
        self.sample_count_spin.setValue(100)  # default
        self.sample_count_spin.setSingleStep(100)
        self.sample_count_spin.valueChanged.connect(self._on_sample_count_changed)
        self.controls_layout.addWidget(self.sample_count_label)
        self.controls_layout.addWidget(self.sample_count_spin) 

        self.controls_layout.addWidget(self.bg_removal_label)
        self.controls_layout.addWidget(self.bg_removal_combo)

        # ROI
        self.select_roi_btn = QPushButton("Select ROI")
        self.select_roi_btn.clicked.connect(self._select_roi)
        self.blb_layout.addWidget(self.select_roi_btn)

        self.refresh_hist_btn = QPushButton("Refresh Histogram")
        self.refresh_hist_btn.clicked.connect(self._plot_blob_histogram)
        self.blb_layout.addWidget(self.refresh_hist_btn)

        self.count_all_btn = QPushButton("Count Animals in Entire Video")
        self.count_all_btn.clicked.connect(self._count_entire_video)
        self.blb_layout.addWidget(self.count_all_btn)
        self.count_all_btn.setVisible(not request)

        self.request_btn = QPushButton("Blob Config Ready")
        self.request_btn.clicked.connect(self._config_ready)
        self.blb_layout.addWidget(self.request_btn)
        self.request_btn.setVisible(request)

        self.count_label = QLabel("Animal Count: 0")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.blb_layout.addWidget(self.count_label)

        self._get_total_frames()

        self.parameters_changed.connect(self._reset_and_reprocess)
        self.bg_display.update_background_display(self.sample_frame_count)

        if config is not None:
            self._apply_config(config)
        else:
            self._reset_blob_array()

        self.blob_array = blob_array

    def set_current_frame(self, frame:Frame_CV2, frame_idx:int):
        self.current_frame = frame
        self.frame_idx = frame_idx
        self._reprocess_current_frame()

    def get_config(self) -> Blob_Config:
        config = Blob_Config(
            sample_frame_count = self.sample_frame_count,
            threshold = self.threshold,
            double_blob_area_threshold = self.blb_hist.double_blob_area_threshold,
            min_blob_area = self.min_blob_area,
            bg_removal_method = self.bg_display.bg_removal_method,
            blob_type = self.blob_type,
            background_frames = self.bg_display.background_frames,
            roi = self.roi
        )
        return config

    def _config_ready(self):
        self.config_ready.emit()

    def _apply_config(self, config: Blob_Config):
        self.sample_frame_count = config.sample_frame_count
        self.threshold = config.threshold
        self.min_blob_area = config.min_blob_area
        self.bg_removal_method = config.bg_removal_method
        self.blob_type = config.blob_type
        self.roi = config.roi

        self.bg_display.background_frames = config.background_frames or {}
        self.bg_display.bg_removal_method = config.bg_removal_method
        self.blb_hist.double_blob_area_threshold = config.double_blob_area_threshold
        
        self.threshold_slider.setValue(self.threshold)
        self.threshold_value_label.setText(str(self.threshold))
        self.min_area_slider.setValue(self.min_blob_area)
        self.min_area_value_label.setText(str(self.min_blob_area))
        self.blob_type_combo.setCurrentText(self.blob_type)
        self.bg_removal_combo.setCurrentText(config.bg_removal_method)
        self.sample_count_spin.setValue(self.sample_frame_count)

        self.parameters_changed.emit()

    def _reprocess_current_frame(self):
        if self.current_frame is None:
            self.count_label.setText("Animal Count: 0")
            return
        
        contours = self._process_contour_from_frame(self.current_frame)
        count, merge = self._perform_blob_counting(contours)
        x1, y1, x2, y2 = self._perform_bbox_calculation(contours)

        if self.blob_array is None:
            self._reset_blob_array()
        self._update_blob_array(self.frame_idx, count, merge, x1, y1, x2, y2)

        self.count_label.setText(f"Animal Count: {count}")
        display_frame = self._draw_mask(self.current_frame, contours)
        self.frame_processed.emit(display_frame, count)

    def _get_total_frames(self):
        self.total_frames = self.frame_extractor.get_total_frames()

    def _count_entire_video(self):
        self.frame_extractor.start_sequential_read(start=0)
        frame_idx = 0

        progress = Progress_Indicator_Dialog(0, self.total_frames, "Counting", "Blob counting...", self)

        while frame_idx < self.total_frames:
            if progress.wasCanceled():
                self.frame_extractor.finish_sequential_read()
                return

            result = self.frame_extractor.read_next_frame()
            if result is None:
                break
            actual_idx, frame = result
            assert actual_idx == frame_idx, "Frame index mismatch!"

            if self.blob_array[frame_idx, 5] == 0:
                contours = self._process_contour_from_frame(frame)
                count, merged = self._perform_blob_counting(contours)
                x1, y1, x2, y2 = self._perform_bbox_calculation(contours)
                self._update_blob_array(frame_idx, count, merged, x1, y1, x2, y2)

            progress.setValue(frame_idx)
            QtWidgets.QApplication.processEvents()
            frame_idx += 1

        self.frame_extractor.finish_sequential_read()
        progress.close()
        self.video_counted.emit(self.blob_array)

    def _perform_blob_counting(self, contours:List[np.ndarray]) -> Tuple[int, int]:
        if not contours:
            return 0, False

        merged = 0
        animal_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.blb_hist.double_blob_area_threshold:
                animal_count += 2
                merged = 1
            else:
                animal_count += 1

        return animal_count, merged

    def _perform_bbox_calculation(self, contours:List[np.ndarray]) -> Tuple[int, int, int, int]:
        if not contours:
            return (0, 0, self.vid_w, self.vid_h)

        min_x, max_x, min_y, max_y = self.vid_w, 0, self.vid_h, 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            min_x = max(min(min_x, x - 50), 0)
            max_x = min(max(max_x, x + w + 50), self.vid_w)
            min_y = max(min(min_y, y - 50), 0)
            max_y = min(max(max_y, y + h + 50), self.vid_h)

        return (int(min_x), int(min_y), int(max_x), int(max_y))

    def _process_contour_from_frame(self, frame:Frame_CV2) -> List[np.ndarray]:
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            frame_to_process = frame[y1:y2, x1:x2].copy()
            roi_offset = (x1, y1)
        else:
            frame_to_process = frame
            roi_offset = (0, 0)

        gray_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)

        processed_frame = gray_frame
        if self.bg_display.bg_removal_method != "None":
            background_frame = self.bg_display.get_background_frame(self.sample_frame_count)
            if background_frame is not None:
                if self.roi is not None:
                    bg_roi = background_frame[y1:y2, x1:x2]
                else:
                    bg_roi = background_frame
                bg_gray = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY) if len(bg_roi.shape) == 3 else bg_roi
                bg_gray = bg_gray.astype(gray_frame.dtype)
                processed_frame = cv2.absdiff(gray_frame, bg_gray)

        # Thresholding
        if self.blob_type == "Dark Blobs (Min)":
            _, thresh = cv2.threshold(processed_frame, self.threshold, 255, cv2.THRESH_BINARY_INV)
        else:  # Light Blobs (Max)
            _, thresh = cv2.threshold(processed_frame, self.threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_blob_area < area < self.min_blob_area * 25:
                if self.roi is not None:
                    cnt = cnt + np.array([[roi_offset]])
                filtered_contours.append(cnt)

        return filtered_contours

    def _draw_mask(self, current_frame:Frame_CV2, contours:List[np.ndarray]) -> Frame_CV2:
        mask = np.zeros_like(current_frame, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, (0, 255, 0), cv2.FILLED)
        alpha = 0.3
        display_frame = cv2.addWeighted(current_frame, 1 - alpha, mask, alpha, 0)
        cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 2)
        return display_frame

    def _select_roi(self):
        frame = self.frame_extractor.get_frame(0)
        if frame is None:
            return
        
        cv2.namedWindow("Select ROI ('space' to accept, 'c' to cancel)", cv2.WINDOW_NORMAL)
        roi = cv2.selectROI("Select ROI ('space' to accept, 'c' to cancel)", frame, fromCenter=False)
        cv2.destroyWindow("Select ROI ('space' to accept, 'c' to cancel)")
        
        if roi[2] > 0 and roi[3] > 0:
            x, y, w, h = roi
            self.roi = (x, y, x + w, y + h)
            print(f"ROI set to {self.roi}")
        else:
            self.roi = None
            print("ROI selection canceled.")
        
        self.parameters_changed.emit()

    def _plot_blob_histogram(self):
        areas = self._compute_blob_areas()
        self.blb_hist.plot_histogram(areas)
    
    def _compute_blob_areas(self) -> List[float]:
        if not self.frame_extractor:
            return []

        total_frames = self.frame_extractor.get_total_frames()
        if total_frames == 0:
            return []

        sample_count = min(self.sample_frame_count, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        all_areas = []
        progress_dialog = Progress_Indicator_Dialog(0, len(frame_indices), "Blob Analysis", "Analyzing blob sizes...", self)

        for i, idx in enumerate(frame_indices):
            if progress_dialog.wasCanceled():
                break
            frame = self.frame_extractor.get_frame(idx)
            if frame is None:
                continue

            contours = self._process_contour_from_frame(frame)
            areas = [cv2.contourArea(c) for c in contours]
            all_areas.extend(areas)

            progress_dialog.setValue(i + 1)

        progress_dialog.close()
        return all_areas

    def _update_blob_array(self, frame_idx, count, merged, x1, y1, x2, y2):
        self.blob_array[frame_idx, 0] = count
        self.blob_array[frame_idx, 1] = merged
        self.blob_array[frame_idx, 2] = x1
        self.blob_array[frame_idx, 3] = y1
        self.blob_array[frame_idx, 4] = x2
        self.blob_array[frame_idx, 5] = y2

    def _on_threshold_changed(self, value:int):
        self.threshold = value
        self.threshold_value_label.setText(str(value))
        self.parameters_changed.emit()

    def _on_min_area_changed(self, value:int):
        self.min_blob_area = value
        self.min_area_value_label.setText(str(value))
        self.parameters_changed.emit()

    def _on_blob_type_changed(self, text:str):
        self.blob_type = text
        self.parameters_changed.emit()

    def _on_bg_removal_changed(self, text:str):
        self.bg_display.bg_removal_method = text
        self.parameters_changed.emit()
        self.bg_display.update_background_display(self.sample_frame_count)

    def _on_sample_count_changed(self, value:int):
        self.sample_frame_count = value
        self.bg_display.background_frames.clear()
        self.bg_display.update_background_display(self.sample_frame_count)

    def _on_blb_hist_change(self, value:int):
        self.blb_hist.double_blob_area_threshold = value
        self._reprocess_current_frame()

    def _reset_and_reprocess(self):
        self._reset_blob_array()
        self._reprocess_current_frame()

    def _reset_blob_array(self):
        self.blob_array = np.zeros((self.total_frames, 6), dtype=np.uint16) # count, is_merged, x1, y1, x2, y2

class Blob_Background(QtWidgets.QWidget):
    def __init__(self, extractor:Frame_Extractor, parent=None):
        super().__init__(parent)
        self.bg_removal_method = "None"
        self.background_frames = {}
        self.frame_extractor = extractor

        layout = QHBoxLayout(self)
        self.bg_label = QLabel("Background Image:")
        self.bg_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.image_label = QLabel("None")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMaximumHeight(50)
        self.image_label.setCursor(Qt.PointingHandCursor)
        self.image_label.mousePressEvent = lambda e: self._show_background_in_dialog()

        layout.addWidget(self.bg_label, 1)
        layout.addWidget(self.image_label, 1)

    def get_background_frame(self, sample_frame_count:int):
        method = self.bg_removal_method
        if method in self.background_frames:
            return self.background_frames[method]

        total_frames = self.frame_extractor.get_total_frames()

        if total_frames == 0:
            self.background_frames[method] = None
            return None

        if total_frames <= sample_frame_count:
            frames_to_iter = range(total_frames)
        else:
            frames_to_iter = np.linspace(0, total_frames - 1, sample_frame_count, dtype=int)

        first_frame = self.frame_extractor.get_frame(0)
        if first_frame is None:
            self.background_frames[method] = None
            return None

        if method == "Min":
            accumulator = np.full_like(first_frame, 255, dtype=np.uint8)
        elif method == "Max":
            accumulator = np.zeros_like(first_frame, dtype=np.uint8)
        elif method == "Mean":
            accumulator = np.zeros_like(first_frame, dtype=np.float32)
        elif method == "Median":
            all_frames = []
        else:
            self.background_frames[method] = None
            return None

        progress_dialog = Progress_Indicator_Dialog(0, len(frames_to_iter), "Background", "Calculating background...", self)

        for i, idx in enumerate(frames_to_iter):
            if progress_dialog.wasCanceled():
                break
            frame = self.frame_extractor.get_frame(idx)
            if frame is None:
                continue

            if method == "Min":
                accumulator = np.minimum(accumulator, frame)
            elif method == "Max":
                accumulator = np.maximum(accumulator, frame)
            elif method == "Mean":
                accumulator += frame.astype(np.float32)
            elif method == "Median":
                all_frames.append(frame)

            progress_dialog.setValue(i + 1)

        progress_dialog.close()

        if method == "Mean":
            background_frame = (accumulator / len(frames_to_iter)).astype(np.uint8)
        elif method == "Median":
            if all_frames:
                background_frame = np.median(np.array(all_frames), axis=0).astype(np.uint8)
            else:
                background_frame = None
        else:
            background_frame = accumulator

        self.background_frames[method] = background_frame
        return background_frame

    def update_background_display(self, sample_frame_count:int):
        if self.bg_removal_method == "None":
            self.image_label.setText("None")
            return

        if self.bg_removal_method not in self.background_frames:
            self.get_background_frame(sample_frame_count)

        frame = self.background_frames.get(self.bg_removal_method)
        if frame is None:
            self.image_label.setText("Failed to compute background")
            return

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        self.image_label.setText("")

    def _show_background_in_dialog(self, event=None):
        if self.bg_removal_method == "None":
            return

        frame = self.background_frames.get(self.bg_removal_method)
        if frame is None:
            return

        # Convert to QImage
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Background Image — {self.bg_removal_method}")
        dialog_layout = QVBoxLayout(dialog)

        label = QLabel()
        label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(False)  # Keep aspect ratio
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(label)
        scroll_area.setWidgetResizable(True)

        dialog_layout.addWidget(scroll_area)
        dialog.showMaximized()
        dialog.exec()

class Blob_Histogram(QVBoxLayout):
    threshold_changed = Signal(int)

    def __init__(self, extractor:Frame_Extractor, parent=None):
        super().__init__(parent)
        self.double_blob_area_threshold = 6000
        self.frame_extractor = extractor

        self.blob_areas = []

        self.histogram_label = QLabel("Blob Size Distribution (computing...)")
        self.addWidget(self.histogram_label)

        # Matplotlib figure
        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(100)
        self.addWidget(self.canvas)
        self._dragging_threshold = False

        self.ax = self.fig.add_subplot(111)
        self.threshold_line = None

    def plot_histogram(self, areas):
        self.blob_areas = areas
        if not self.blob_areas:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No blobs found", transform=self.ax.transAxes, ha="center")
            self.canvas.draw()
            return
        self.ax.clear()

        _, _, _ = self.ax.hist(self.blob_areas, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        self.ax.set_xlabel("Blob Area (pixels²)")
        self.ax.set_ylabel("Frequency")
        self.ax.set_title("Blob Size Distribution")

        # Add draggable threshold line
        if self.threshold_line:
            self.threshold_line.remove()

        self.threshold_line = self.ax.axvline(
            x=self.double_blob_area_threshold,
            color='red',
            linestyle='--',
            linewidth=2,
        )

        self.canvas.mpl_connect('button_press_event', self._on_histogram_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_histogram_drag)
        self.canvas.mpl_connect('button_release_event', self._on_histogram_release)

        self.canvas.draw()
        self.histogram_label.setText("Blob Size Distribution (drag red line to set '2-animal' threshold)")

    def _on_histogram_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if self.threshold_line and abs(event.xdata - self.double_blob_area_threshold) < (max(self.blob_areas) - min(self.blob_areas)) / 20:
            self._dragging_threshold = True

    def _on_histogram_drag(self, event):
        if not self._dragging_threshold or event.inaxes != self.ax:
            return
        new_x = max(0, event.xdata)
        self.double_blob_area_threshold = new_x
        self.threshold_changed.emit(int(new_x))
        self.threshold_line.set_xdata([new_x, new_x])
        self.canvas.draw()

    def _on_histogram_release(self, event):
        if self._dragging_threshold:
            self._dragging_threshold = False