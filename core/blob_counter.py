import cv2
import numpy as np
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal

from ui import Progress_Indicator_Dialog
from .io import Frame_Extractor

class Blob_Counter(QtWidgets.QGroupBox):
    parameters_changed = Signal()  # No args needed, store state internally
    frame_processed = Signal(object, int)  # emit processed QImage and count

    def __init__(self, frame_extractor: Frame_Extractor, parent=None):
        super().__init__(parent)
        self.setTitle("Blob-based Animal Counting Controls")

        self.animal_count = 0
        self.frame_extractor = frame_extractor
        self.current_frame = None
        self.background_frames = {}

        # UI parameters
        self.threshold = 100
        self.min_blob_area = 500
        self.bg_removal_method = "None"
        self.blob_type = "Dark Blobs (Max)"

        self.layout = QtWidgets.QVBoxLayout(self)
        self.setFixedWidth(200)

        # Image display
        self.image_label = QtWidgets.QLabel("No background image to display")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.layout.addWidget(self.image_label, 1)

        # Controls
        self.control_gbox = QtWidgets.QGroupBox(self)
        self.controls_layout = QtWidgets.QVBoxLayout(self.control_gbox)
        self.control_gbox.setTitle("Blob Counter Control")
        self.layout.addWidget(self.control_gbox)

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
        self.controls_layout.addWidget(self.bg_removal_label)
        self.controls_layout.addWidget(self.bg_removal_combo)

        # Count display
        self.count_label = QtWidgets.QLabel("Animal Count: 0")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.count_label)

        # Connect parameter changes to reprocessing
        self.parameters_changed.connect(self._reprocess_current_frame)
        self._update_background_display()  # Initial background display

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

    def _update_background_display(self):
        method = self.bg_removal_method
        if method == "None":
            self.image_label.setText("No background image to display")
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

    def set_current_frame(self, frame):
        self.current_frame = frame
        self._reprocess_current_frame()

    def _reprocess_current_frame(self):
        if self.current_frame is None:
            self.count_label.setText("Animal Count: 0")
            return
        result = self._perform_blob_counting(self.current_frame)
        if result is not None:
            display_frame, count = result
            # Convert to QImage for display if needed (optional)
            self.frame_processed.emit(display_frame, count)

    def _perform_blob_counting(self, current_frame):
        if current_frame is None:
            self.animal_count = 0
            self.count_label.setText("Animal Count: 0")
            return None

        frame_to_process = current_frame.copy()
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
        self.animal_count = len(filtered_contours)

        # Draw results
        mask = np.zeros_like(current_frame, dtype=np.uint8)
        cv2.drawContours(mask, filtered_contours, -1, (0, 255, 0), cv2.FILLED)
        alpha = 0.3
        display_frame = cv2.addWeighted(current_frame, 1 - alpha, mask, alpha, 0)
        cv2.drawContours(display_frame, filtered_contours, -1, (0, 255, 0), 2)

        self.count_label.setText(f"Animal Count: {self.animal_count}")
        return display_frame, self.animal_count

    def _get_background_frame(self, method):
        if method in self.background_frames:
            return self.background_frames[method]

        total_frames = self.frame_extractor.get_total_frames()
        if total_frames == 0:
            self.background_frames[method] = None
            return None

        # Sample frames
        if total_frames <= 1000:
            frames_to_iter = range(total_frames)
        else:
            frames_to_iter = np.linspace(0, total_frames - 1, 1000, dtype=int)

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