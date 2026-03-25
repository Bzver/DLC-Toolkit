import cv2
import numpy as np
import time
from tqdm import tqdm

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QSizePolicy, QSlider, QComboBox, QPushButton,
)
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import Optional, Tuple, List

from ui import Progress_Indicator_Dialog, Frame_Display_Dialog, Frame_Range_Dialog, Spinbox_With_Label
from utils.helper import get_roi_cv2, plot_roi, frame_to_qimage
from core.io import Frame_Extractor
from utils.logger import Loggerbox, QMessageBox, logger
from utils.dataclass import Blob_Config


class Blob_Counter(QGroupBox):
    Frame_CV2 = np.ndarray
    parameters_changed = Signal()
    frame_processed = Signal(object, int)
    video_counted = Signal(object)
    roi_set = Signal(object)

    def __init__(self,
                 video_filepath:str,
                 config: Optional[Blob_Config]=None,
                 roi:Optional[np.ndarray] = None,
                 blob_array:Optional[np.ndarray] = None,
                 parent=None):
        super().__init__(parent)
        self.setTitle("Blob Counting Controls")

        self.video_filepath = video_filepath
        self.extractor = Frame_Extractor(video_filepath)
        
        self.current_frame = None
        self.working_bg = None
        self.total_frames, self.frame_idx = 0, 0
        self.roi = roi

        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # UI parameters
        self.threshold = 50
        self.min_blob_area = 2000
        self.blob_type = "Dark Blobs"

        self.blb_layout = QVBoxLayout(self)
        self.setFixedWidth(200)

        # Image Display
        self.bg_display = Blob_Background(self.extractor)
        self.bg_display.roi_requested.connect(self._roi_to_bg_display)
        self.blb_layout.addWidget(self.bg_display)

        # Histogram for blob sizes
        self.blb_hist = Blob_Histogram(self.extractor)
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
        self.blob_type_combo.addItems(["Dark Blobs", "Light Blobs"])
        self.blob_type_combo.currentTextChanged.connect(self._on_blob_type_changed)
        self.controls_layout.addWidget(self.blob_type_label)
        self.controls_layout.addWidget(self.blob_type_combo)

        # BG removal
        self.bg_removal_label = QLabel("Background Removal:")
        self.bg_removal_combo = QComboBox()
        self.bg_removal_combo.addItems([
            "None",
            "Mean",
            "Min",
            "5th Percentile",
            "10th Percentile",
            "25th Percentile",
            "Median",
            "75th Percentile",
            "90th Percentile",
            "95th Percentile",
            "Max"
        ])
        self.bg_removal_combo.currentTextChanged.connect(self._on_bg_removal_changed)
        
        self.bg_regen_btn = QPushButton("Regenerate Background")
        self.bg_regen_btn.clicked.connect(self._regen_bg)
        self.blb_layout.addWidget(self.bg_regen_btn)

        self.controls_layout.addWidget(self.bg_removal_label)
        self.controls_layout.addWidget(self.bg_removal_combo)

        self.select_roi_btn = QPushButton("Select ROI")
        self.select_roi_btn.clicked.connect(self._select_roi)
        self.blb_layout.addWidget(self.select_roi_btn)

        self.refresh_hist_btn = QPushButton("Refresh Histogram")
        self.refresh_hist_btn.clicked.connect(self._plot_blob_histogram)
        self.blb_layout.addWidget(self.refresh_hist_btn)

        self.max_worker_spin = Spinbox_With_Label("Max Workers: ", (1,64), 4)
        self.max_worker_spin.setToolTip("Number of parallel processes when counting the entire video.")
        self.controls_layout.addWidget(self.max_worker_spin)

        self.count_all_btn = QPushButton("Count Animals")
        self.count_all_btn.clicked.connect(self._count_video)
        self.blb_layout.addWidget(self.count_all_btn)

        self.count_label = QLabel("Animal Count: 0")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.blb_layout.addWidget(self.count_label)

        self.total_frames = self.extractor.get_total_frames()

        self.parameters_changed.connect(self._reset_and_reprocess)
        self.bg_display.update_background_display()

        if config is not None:
            self._apply_config(config)
        else:
            self._reset_blob_array()

        self.blob_array = blob_array
        self.last_reset_query = 0

    def set_current_frame(self, frame:Frame_CV2, frame_idx:int):
        self.current_frame = frame
        self.frame_idx = frame_idx
        self._reprocess_current_frame()

    def get_config(self) -> Blob_Config:
        config = Blob_Config(
            threshold = self.threshold,
            double_blob_area_threshold = self.blb_hist.double_blob_area_threshold,
            min_blob_area = self.min_blob_area,
            bg_removal_method = self.bg_display.bg_removal_method,
            blob_type = self.blob_type,
            background_frames = self.bg_display.background_frames,
            roi = self.roi
        )
        return config

    def _apply_config(self, config: Blob_Config):
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

        self.parameters_changed.emit()

    def _reprocess_current_frame(self):
        if self.current_frame is None:
            self.count_label.setText("Animal Count: 0")
            return
        
        contours = self._process_contour_from_frame(self.current_frame)
        count, _ = self._count_contours(contours)

        self.count_label.setText(f"Animal Count: {count}")
        display_frame = self._draw_mask(self.current_frame, contours)
        self.frame_processed.emit(display_frame, count)

    def _count_video(self):
        fm_dialog = Frame_Range_Dialog(self.total_frames, parent=self)
        if fm_dialog.exec() == QtWidgets.QDialog.Accepted:
            max_workers = self.max_worker_spin.value()
            start_idx, end_idx = fm_dialog.selected_range #inclusive
            end_idx += 1

            segments = self._task_splitter(start_idx, end_idx, max_workers)

            title = f"BLOB COUNTER | {max_workers} workers | {len(segments)} segments"
            border = "═" * (len(title) + 2)
            logger.info(f"╔{border}╗")
            logger.info(f"║ {title} ║")
            logger.info(f"╚{border}╝")

            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(self._process_segment, idx, chunk[0], chunk[1]): idx
                        for idx, chunk in enumerate(segments)
                    }
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                start_idx, end_idx, chunk_array = result
                                self.blob_array[start_idx:end_idx] = chunk_array
                        except Exception as e:
                            logger.error(f"[THREAD_EXP] Segment failed: {e}")

            except Exception as e:
                logger.error(f"[THREAD_EXP] Critical failure: {e}")

            self.video_counted.emit(self.blob_array)

    def _process_segment(self, seg_idx:int, start_idx: int, end_idx: int) -> Tuple[int, int, np.ndarray]:
        pbar = tqdm(
            total=end_idx - start_idx,
            desc=f"Segment {seg_idx} [{start_idx}-{end_idx}]",
            leave=False, 
            ncols=200
        )

        extractor = Frame_Extractor(self.video_filepath)
        try:
            extractor.start_sequential_read(start=start_idx, end=end_idx)
            chunk_array = np.zeros((end_idx-start_idx, 2), dtype=np.uint8)

            frame_idx = 0
            while frame_idx < end_idx - start_idx:
                result = extractor.read_next_frame()
                if result is None:
                    break
                actual_idx, frame = result
                assert actual_idx == frame_idx + start_idx, "Frame index mismatch!"

                contours = self._process_contour_from_frame(frame)
                count, merged = self._count_contours(contours)

                chunk_array[frame_idx, 0] = count
                chunk_array[frame_idx, 1] = merged
                pbar.update(1)
                frame_idx += 1
        except Exception as e:
            logger.warning(f"[Segment_{seg_idx}] Failed: {e}")
            pbar.set_description(f"Segment {seg_idx} FAILED")
            return None
        finally:
            pbar.close()
            extractor.finish_sequential_read()

        return start_idx, end_idx, chunk_array

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

        if self.working_bg is not None:
            diff = gray_frame.astype(np.int16) - self.working_bg

            if self.blob_type == "Dark Blobs":
                thresh = (diff < -self.threshold).astype(np.uint8) * 255
            else:  # Light Blobs
                thresh = (diff > self.threshold).astype(np.uint8) * 255
        else:
            if self.blob_type == "Dark Blobs":
                _, thresh = cv2.threshold(processed_frame, self.threshold, 255, cv2.THRESH_BINARY_INV)
            else:  # Light Blobs (Max)
                _, thresh = cv2.threshold(processed_frame, self.threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_blob_area:
                if self.roi is not None:
                    cnt = cnt + np.array([[roi_offset]])
                filtered_contours.append(cnt)

        return filtered_contours

    def _count_contours(self, contours:List[np.ndarray]) -> Tuple[int, int]:
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

    def _draw_mask(self, current_frame:Frame_CV2, contours:List[np.ndarray]) -> Frame_CV2:
        mask = np.zeros_like(current_frame, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, (0, 255, 0), cv2.FILLED)
        alpha = 0.3
        display_frame = cv2.addWeighted(current_frame, 1 - alpha, mask, alpha, 0)
        cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 2)
        return display_frame

    def _process_background(self):
        if self.bg_display.bg_removal_method == "None":
            return None

        background_frame = self.bg_display.get_background_frame()
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            bg_roi = background_frame[y1:y2, x1:x2]
        else:
            bg_roi = background_frame
        bg_gray = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY) if len(bg_roi.shape) == 3 else bg_roi
        return bg_gray.astype(dtype=np.int16)

    def _regen_bg(self):
        self.bg_display.background_frames.clear()
        self.bg_display.update_background_display()

    def _select_roi(self):
        frame = self.extractor.get_frame(0)
        if frame is None:
            return
        
        roi = get_roi_cv2(frame)
        if roi is not None:
            self.roi = roi
            logger.info(f"[BLOB] ROI set to {self.roi}")
        else:
            logger.info("[BLOB] ROI selection canceled.")
        
        self.parameters_changed.emit()
        self.roi_set.emit(np.array(self.roi))
        self.working_bg = self._process_background()

    def _plot_blob_histogram(self):
        if not self.extractor:
            return
        
        self._reset_blob_array()
        self.video_counted.emit(self.blob_array) # Invalidated previous counting and refresh the state

        total_frames = self.extractor.get_total_frames()
        if total_frames == 0:
            return
        
        sample_segment_count = min(100, total_frames//100)
        frame_indices = np.unique(np.linspace(0, total_frames - 1, sample_segment_count, dtype=int))
        if len(frame_indices) == 1:
            sample_segment_length = min(100, total_frames)
        else:
            sample_segment_length = min(frame_indices[1] - frame_indices[0], 100)

        areas = []
        progress_dialog = Progress_Indicator_Dialog(0, len(frame_indices), "Blob Analysis", "Analyzing blob sizes...", self)

        for i, idx in enumerate(frame_indices):
            self.extractor.start_sequential_read(idx)
            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                self.extractor.finish_sequential_read()
                break

            for k in range(sample_segment_length):
                result = self.extractor.read_next_frame()
                if result is None:
                    break
                if k % 5 != 0:
                    continue
                actual_idx, frame = result
                assert actual_idx == idx + k, f"Frame index mismatch! actual_idx: {actual_idx} | idx + k: {idx + k}"

                contours = self._process_contour_from_frame(frame)
                count_result = self._count_contours(contours)
                frame_areas = [cv2.contourArea(c) for c in contours]
                areas.extend(frame_areas)
                self.blob_array[idx + k, 0], self.blob_array[idx+k, 1] = count_result
            
            self.extractor.finish_sequential_read()

        progress_dialog.close()
        self.blb_hist.plot_histogram(areas)
        self.video_counted.emit(self.blob_array)

    def _roi_to_bg_display(self):
        self.bg_display.display_background_roi_dialog(self.roi)

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
        self.bg_display.update_background_display()
        self.working_bg = self._process_background()

    def _on_blb_hist_change(self, value:int):
        self.blb_hist.double_blob_area_threshold = value
        self._reprocess_current_frame()

    def _reset_and_reprocess(self):
        self._reset_blob_array()
        self._reprocess_current_frame()

    def _reset_blob_array(self):
        if not hasattr(self, "blob_array") or self.blob_array is None:
            logger.info("[BCOUNT] Preallocating blob array with numeric zeros.")
            self.blob_array = np.zeros((self.total_frames, 2), dtype=np.uint8)
            return
        if np.any(self.blob_array != 0):
            now = time.time()
            elapsed = now - self.last_reset_query
            if elapsed < 60.0:
                logger.debug("[BCOUNT] Skipping reset prompt (cooldown active).")
                return
            self.last_reset_query = now
            reply = Loggerbox.question(
                self, "Blob Counting Reset", "Parameters have changed, do you want to reset the blob counting result?"
            )
            if reply == QMessageBox.No:
                return

        self.blob_array = np.zeros((self.total_frames, 2), dtype=np.uint8)

    @staticmethod
    def _task_splitter(start_idx: int, end_idx: int, max_workers:int):
        segment_size = min(5000, (end_idx - start_idx)//max_workers)
        chunk_start = start_idx
        chunks = []
        while chunk_start < end_idx:
            chunk_end = min(end_idx, chunk_start + segment_size)
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end
        return chunks

class Blob_Background(QtWidgets.QWidget):
    roi_requested = Signal()

    def __init__(self, extractor:Frame_Extractor, parent=None):
        super().__init__(parent)
        self.bg_removal_method = "None"
        self.background_frames = {}
        self.extractor = extractor

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

    def get_background_frame(self, bg_frame_list:Optional[List[int]]=None):
        method = self.bg_removal_method
        if method in self.background_frames:
            return self.background_frames[method]

        total_frames = self.extractor.get_total_frames()

        if total_frames == 0:
            self.background_frames[method] = None
            return None

        if total_frames <= 100:
            frames_to_iter = range(total_frames)
        elif bg_frame_list:
            frames_to_iter = bg_frame_list
        else:
            frames_to_iter = np.linspace(0, total_frames - 1, 100, dtype=int)

        first_frame = self.extractor.get_frame(0)
        if first_frame is None:
            self.background_frames[method] = None
            return None

        all_frames = []

        progress_dialog = Progress_Indicator_Dialog(0, len(frames_to_iter), "Background", "Calculating background...", self)

        for i, idx in enumerate(frames_to_iter):
            if progress_dialog.wasCanceled():
                break
            frame = self.extractor.get_frame(idx)
            if frame is None:
                continue

            all_frames.append(frame)
            progress_dialog.setValue(i + 1)

        progress_dialog.close()

        if not all_frames:
            return None

        if method == "None":
            return None

        frame_array = np.array(all_frames)
        if method == "Mean":
            bg = np.mean(np.array(all_frames), axis=0).astype(np.uint8)
        elif method == "Min":
            bg = np.min(frame_array, axis=0)
        elif method == "Max":
            bg = np.max(frame_array, axis=0)
        elif method == "5th Percentile":
            bg = np.percentile(frame_array, 5, axis=0)
        elif method == "10th Percentile":
            bg = np.percentile(frame_array, 10, axis=0)
        elif method == "25th Percentile":
            bg = np.percentile(frame_array, 25, axis=0)
        elif method == "Median":
            bg = np.median(frame_array, axis=0)
        elif method == "75th Percentile":
            bg = np.percentile(frame_array, 75, axis=0)
        elif method == "90th Percentile":
            bg = np.percentile(frame_array, 90, axis=0)
        elif method == "95th Percentile":
            bg = np.percentile(frame_array, 95, axis=0)
        else:
            raise ValueError(f"Unknown bg method: {method}")

        bg = bg.astype(np.uint8)
        self.background_frames[method] = bg
        return bg

    def update_background_display(self, bg_frame_list:Optional[List[int]]=None):
        if self.bg_removal_method == "None":
            self.image_label.setText("None")
            return

        if self.bg_removal_method not in self.background_frames:
            self.get_background_frame(bg_frame_list)

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
        self.roi_requested.emit()

    def display_background_roi_dialog(self, roi:np.ndarray):
        frame = self.background_frames.get(self.bg_removal_method)
        frame = plot_roi(frame, roi)
        qimage = frame_to_qimage(frame)
        dialog = Frame_Display_Dialog(title=f"Background Image — {self.bg_removal_method}", image=qimage)
        dialog.exec()

class Blob_Histogram(QVBoxLayout):
    threshold_changed = Signal(int)

    def __init__(self, extractor:Frame_Extractor, parent=None):
        super().__init__(parent)
        self.double_blob_area_threshold = 100000
        self.extractor = extractor

        self.blob_areas = []

        self.histogram_label = QLabel("Blob Size Distribution (computing...)")
        self.addWidget(self.histogram_label)

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

        upper_limit = float(np.percentile(self.blob_areas, 99))
        xlim_max = upper_limit * 1.2 + 500
        xlim_min = np.nanmin(self.blob_areas) * 0.95

        self.ax.hist(self.blob_areas, bins=100, range=(0, xlim_max), color='skyblue', edgecolor='black', alpha=0.7)
        self.ax.set_xlabel("Blob Area (pixels²)")
        self.ax.set_ylabel("Frequency")
        self.ax.set_title("Blob Size Distribution")
        self.ax.set_xlim(xlim_min, xlim_max)

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
        _, xlim_max = self.ax.get_xlim()
        if self.double_blob_area_threshold > xlim_max:
            new_lim = int(xlim_max*0.9)
            self.double_blob_area_threshold = new_lim
            self.threshold_changed.emit(new_lim)
            self.threshold_line.set_xdata([new_lim, new_lim])
            self.canvas.draw()
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