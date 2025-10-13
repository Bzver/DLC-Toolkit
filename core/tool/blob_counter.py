import cv2
import numpy as np
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from typing import Optional

from ui import Progress_Indicator_Dialog
from core.io import Frame_Extractor
from core.dataclass import Blob_Config

class Blob_Counter(QtWidgets.QGroupBox):
    parameters_changed = Signal()  # No args needed, store state internally
    frame_processed = Signal(object, int)  # emit processed QImage and count
    video_counted = Signal(list)

    def __init__(self, frame_extractor: Frame_Extractor, config: Optional[Blob_Config]=None,  parent=None):
        super().__init__(parent)
        self.setTitle("Blob-based Animal Counting Controls")

        self.bg_sample_frame_count = 100
        self.frame_extractor = frame_extractor
        self.current_frame = None
        self.background_frames = {}

        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # UI parameters
        self.threshold = 50
        self.double_blob_area_threshold = 6000
        self.min_blob_area = 500
        self.bg_removal_method = "None"
        self.blob_type = "Dark Blobs (Max)"
        self._dragging_threshold = False

        self.blb_layout = QtWidgets.QVBoxLayout(self)
        self.setFixedWidth(200)

        # Image display
        self.bg_display = QtWidgets.QHBoxLayout()
        self.bg_label = QtWidgets.QLabel("Background Image:")
        self.bg_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.image_label = QtWidgets.QLabel("None")
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.image_label.setCursor(Qt.PointingHandCursor)
        self.image_label.mousePressEvent = lambda e: self._show_background_in_dialog()

        self.bg_display.addWidget(self.bg_label, 1)
        self.bg_display.addWidget(self.image_label, 1)
        self.blb_layout.addLayout(self.bg_display)

        # Histogram for blob sizes
        self.histogram_layout = QtWidgets.QVBoxLayout()
        self.blb_layout.addLayout(self.histogram_layout)

        self.histogram_label = QtWidgets.QLabel("Blob Size Distribution (computing...)")
        self.histogram_layout.addWidget(self.histogram_label)

        # Matplotlib figure
        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(100)
        self.histogram_layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self.threshold_line = None
        self.blob_areas = []

        # Controls
        self.control_gbox = QtWidgets.QGroupBox(self)
        self.controls_layout = QtWidgets.QVBoxLayout(self.control_gbox)
        self.control_gbox.setTitle("Blob Counter Control")
        self.blb_layout.addWidget(self.control_gbox)

        # Threshold
        self.threshold_label = QtWidgets.QLabel("Threshold:")
        self.threshold_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(self.threshold)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.threshold_value_label = QtWidgets.QLabel(str(self.threshold))
        self.controls_layout.addWidget(self.threshold_label)
        self.controls_layout.addWidget(self.threshold_slider)
        self.controls_layout.addWidget(self.threshold_value_label)

        # Min Blob Area
        self.min_area_label = QtWidgets.QLabel("Min Blob Area:")
        self.min_area_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.min_area_slider.setRange(10, 5000)
        self.min_area_slider.setValue(self.min_blob_area)
        self.min_area_slider.valueChanged.connect(self._on_min_area_changed)
        self.min_area_value_label = QtWidgets.QLabel(str(self.min_blob_area))
        self.controls_layout.addWidget(self.min_area_label)
        self.controls_layout.addWidget(self.min_area_slider)
        self.controls_layout.addWidget(self.min_area_value_label)

        # Blob type
        self.blob_type_label = QtWidgets.QLabel("Blob Type:")
        self.blob_type_combo = QtWidgets.QComboBox()
        self.blob_type_combo.addItems(["Dark Blobs (Max)", "Light Blobs (Min)"])
        self.blob_type_combo.currentTextChanged.connect(self._on_blob_type_changed)
        self.controls_layout.addWidget(self.blob_type_label)
        self.controls_layout.addWidget(self.blob_type_combo)

        # BG removal
        self.bg_removal_label = QtWidgets.QLabel("Background Removal:")
        self.bg_removal_combo = QtWidgets.QComboBox()
        self.bg_removal_combo.addItems(["None", "Min", "Max", "Median", "Mean"])
        self.bg_removal_combo.currentTextChanged.connect(self._on_bg_removal_changed)
        
        self.bg_frame_count_label = QtWidgets.QLabel("Background Sample Frames:")
        self.bg_frame_count_spin = QtWidgets.QSpinBox()
        self.bg_frame_count_spin.setRange(10, 10000)
        self.bg_frame_count_spin.setValue(100)  # default
        self.bg_frame_count_spin.setSingleStep(100)
        self.bg_frame_count_spin.valueChanged.connect(self._on_bg_frame_count_changed)
        self.controls_layout.addWidget(self.bg_frame_count_label)
        self.controls_layout.addWidget(self.bg_frame_count_spin)

        self.controls_layout.addWidget(self.bg_removal_label)
        self.controls_layout.addWidget(self.bg_removal_combo)

        self.refresh_hist_btn = QtWidgets.QPushButton("Refresh Histogram")
        self.refresh_hist_btn.clicked.connect(self._plot_blob_histogram)
        self.blb_layout.addWidget(self.refresh_hist_btn)

        # Add button to count animals in entire video
        self.count_all_btn = QtWidgets.QPushButton("Count Animals in Entire Video")
        self.count_all_btn.clicked.connect(self._count_entire_video)
        self.blb_layout.addWidget(self.count_all_btn)

        # Count display
        self.count_label = QtWidgets.QLabel("Animal Count: 0")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.blb_layout.addWidget(self.count_label)

        # Connect parameter changes to reprocessing
        self.parameters_changed.connect(self._reprocess_current_frame)
        self._update_background_display()  # Initial background display

        if config is not None:
            self._apply_config(config)

    def set_current_frame(self, frame):
        self.current_frame = frame
        self._reprocess_current_frame()

    def get_config(self):
        config = Blob_Config(
            bg_sample_frame_count = self.bg_sample_frame_count,
            threshold = self.threshold,
            double_blob_area_threshold = self.double_blob_area_threshold,
            min_blob_area = self.min_blob_area,
            bg_removal_method = self.bg_removal_method,
            blob_type = self.blob_type,
            background_frames = self.background_frames,
        )
        return config

    def _apply_config(self, config: Blob_Config):
        self.bg_sample_frame_count = config.bg_sample_frame_count
        self.threshold = config.threshold
        self.double_blob_area_threshold = config.double_blob_area_threshold
        self.min_blob_area = config.min_blob_area
        self.bg_removal_method = config.bg_removal_method
        self.blob_type = config.blob_type
        self.background_frames = config.background_frames or {}

        # Update UI widgets to reflect config
        self.threshold_slider.setValue(self.threshold)
        self.threshold_value_label.setText(str(self.threshold))

        self.min_area_slider.setValue(self.min_blob_area)
        self.min_area_value_label.setText(str(self.min_blob_area))

        self.blob_type_combo.setCurrentText(self.blob_type)
        
        self.bg_removal_combo.setCurrentText(self.bg_removal_method)
        self.bg_frame_count_spin.setValue(self.bg_sample_frame_count)

        # Re-emit parameters_changed to trigger reprocessing (optional but safe)
        self.parameters_changed.emit()

    def _on_threshold_changed(self, value):
        self.threshold = value
        self.threshold_value_label.setText(str(value))
        self.parameters_changed.emit()

    def _on_min_area_changed(self, value):
        self.min_blob_area = value
        self.min_area_value_label.setText(str(value))
        self.parameters_changed.emit()

    def _on_blob_type_changed(self, text):
        self.blob_type = text
        self.parameters_changed.emit()

    def _on_bg_removal_changed(self, text):
        self.bg_removal_method = text
        self.parameters_changed.emit()
        self._update_background_display()

    def _on_bg_frame_count_changed(self, value):
        self.bg_sample_frame_count = value
        self.background_frames.clear() # Invalidate cached backgrounds since sampling changed
        self._update_background_display()

    def _update_background_display(self):
        method = self.bg_removal_method
        if method == "None":
            self.image_label.setText("None")
            return

        if method not in self.background_frames:
            self._get_background_frame(method)

        frame = self.background_frames.get(method)
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

    def _reprocess_current_frame(self):
        if self.current_frame is None:
            self.count_label.setText("Animal Count: 0")
            return
        result = self._perform_blob_counting(self.current_frame)
        if result is not None:
            display_frame, count = result
            # Convert to QImage for display if needed (optional)
            self.frame_processed.emit(display_frame, count)

    def _count_entire_video(self):
        total_frames = self.frame_extractor.get_total_frames()

        counts_per_frame = [0] * total_frames
        current_count = 0
        skip = 1
        max_skip = 40  # Don't skip more than 40 frmaes

        frame_idx = 0
        progress = Progress_Indicator_Dialog(0, total_frames, "Counting", "Adaptive counting...", self)

        while frame_idx < total_frames:
            if progress.wasCanceled():
                break

            frame = self.frame_extractor.get_frame(frame_idx)
            if frame is None:
                count = current_count
            else:
                _, count = self._perform_blob_counting(
                    frame,
                    skip_draw=True
                )

            # Fill all skipped frames with this count
            next_process_frame = min(frame_idx + skip, total_frames)
            for i in range(frame_idx, next_process_frame):
                counts_per_frame[i] = count

            progress.setValue(frame_idx)
            QtWidgets.QApplication.processEvents()

            # Adaptive logic
            if count == current_count:
                skip = min(skip * 2, max_skip)  # exponential backoff
            else:
                skip = 1

            current_count = count
            frame_idx = next_process_frame

        progress.close()
        self.video_counted.emit(counts_per_frame)
        
    def _perform_blob_counting(self, current_frame, skip_draw=False):
        if current_frame is None:
            self.count_label.setText("Animal Count: 0")
            return None, 0

        frame_to_process = current_frame
        gray_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)

        processed_frame = gray_frame
        if self.bg_removal_method != "None":
            background_frame = self._get_background_frame(self.bg_removal_method)
            if background_frame is not None:
                background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY) if len(background_frame.shape) == 3 else background_frame
                background_gray = background_gray.astype(gray_frame.dtype)
                processed_frame = cv2.absdiff(gray_frame, background_gray)

        # Thresholding
        if self.blob_type == "Dark Blobs (Min)":
            _, thresh = cv2.threshold(processed_frame, self.threshold, 255, cv2.THRESH_BINARY_INV)
        else:  # Light Blobs (Max)
            _, thresh = cv2.threshold(processed_frame, self.threshold, 255, cv2.THRESH_BINARY)

        # Find and filter contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_blob_area]

        animal_count = 0
        for cnt in filtered_contours:
            area = cv2.contourArea(cnt)
            if area >= self.double_blob_area_threshold:
                animal_count += 2
            else:
                animal_count += 1

        if skip_draw:
            return None, animal_count
        
        display_frame = self._draw_mask(current_frame, filtered_contours)
        self.count_label.setText(f"Animal Count: {animal_count}")
        return display_frame, animal_count

    def _draw_mask(self, current_frame, filtered_contours):
        # Draw results
        mask = np.zeros_like(current_frame, dtype=np.uint8)
        cv2.drawContours(mask, filtered_contours, -1, (0, 255, 0), cv2.FILLED)
        alpha = 0.3
        display_frame = cv2.addWeighted(current_frame, 1 - alpha, mask, alpha, 0)
        cv2.drawContours(display_frame, filtered_contours, -1, (0, 255, 0), 2)
        return display_frame

    def _get_background_frame(self, method):
        if method in self.background_frames:
            return self.background_frames[method]

        total_frames = self.frame_extractor.get_total_frames()
        if total_frames == 0:
            self.background_frames[method] = None
            return None

        # Sample frames
        if total_frames <= self.bg_sample_frame_count:
            frames_to_iter = range(total_frames)
        else:
            frames_to_iter = np.linspace(0, total_frames - 1, self.bg_sample_frame_count, dtype=int)

        # Initialize
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
        
    def _show_background_in_dialog(self, event=None):
        method = self.bg_removal_method
        if method == "None":
            return

        frame = self.background_frames.get(method)
        if frame is None:
            return

        # Convert to QImage
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

        # Create dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Background Image — {method}")
        dialog_layout = QtWidgets.QVBoxLayout(dialog)

        label = QtWidgets.QLabel()
        label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(False)  # Keep aspect ratio
        label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(label)
        scroll_area.setWidgetResizable(True)

        dialog_layout.addWidget(scroll_area)
        dialog.showMaximized()
        dialog.exec()

    def _compute_blob_areas(self):
        if not self.frame_extractor:
            return []

        total_frames = self.frame_extractor.get_total_frames()
        if total_frames == 0:
            return []

        # Sample frames (e.g., 200 frames max)
        sample_count = min(200, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        all_areas = []
        progress_dialog = Progress_Indicator_Dialog(0, len(frame_indices), "Blob Analysis", "Analyzing blob sizes...", self)

        for i, idx in enumerate(frame_indices):
            if progress_dialog.wasCanceled():
                break
            frame = self.frame_extractor.get_frame(idx)
            if frame is None:
                continue

            # Reuse your preprocessing logic
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.bg_removal_method != "None":
                bg = self._get_background_frame(self.bg_removal_method)
                if bg is not None:
                    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY) if len(bg.shape) == 3 else bg
                    processed = cv2.absdiff(gray, bg_gray.astype(gray.dtype))
                else:
                    processed = gray
            else:
                processed = gray

            if self.blob_type == "Dark Blobs (Min)":
                _, thresh = cv2.threshold(processed, self.threshold, 255, cv2.THRESH_BINARY_INV)
            else:
                _, thresh = cv2.threshold(processed, self.threshold, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > self.min_blob_area]
            all_areas.extend(areas)

            progress_dialog.setValue(i + 1)

        progress_dialog.close()
        return all_areas
    
    def _plot_blob_histogram(self):
        self.blob_areas = self._compute_blob_areas()
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
            label=f"Double Threshold: {self.double_blob_area_threshold}"
        )
        self.ax.legend()

        # Enable dragging
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
        self.double_blob_area_threshold = int(new_x)
        self.threshold_line.set_xdata([new_x, new_x])
        self.canvas.draw()

    def _on_histogram_release(self, event):
        if self._dragging_threshold:
            self._dragging_threshold = False
            if self.current_frame is not None:
                self.set_current_frame(self.current_frame)