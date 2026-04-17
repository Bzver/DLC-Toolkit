
import cv2
import numpy as np
import time
import json
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QFileDialog, QSlider, QComboBox, QPushButton
)

from typing import Optional, Tuple, List, Dict

from ui import Progress_Indicator_Dialog, Frame_Display_Dialog, Frame_Range_Dialog, Spinbox_With_Label
from utils.helper import get_roi_cv2, plot_roi, frame_to_qimage
from core.io import Frame_Extractor, Frame_Extractor_Img
from utils.logger import Loggerbox, QMessageBox, logger
from utils.dataclass import Blob_Config

import traceback

class Blob_Counter(QGroupBox):
    parameters_changed = Signal()
    frame_processed = Signal(object, int)
    video_counted = Signal(object)
    roi_set = Signal(object)

    def __init__(self,
                 video_filepath: str,
                 config: Optional[Blob_Config] = None,
                 roi: Optional[np.ndarray] = None,
                 blob_array: Optional[np.ndarray] = None,
                 parent=None):

        super().__init__(parent)
        self.setTitle("Blob Counting Controls")

        self.video_filepath = video_filepath
        try:
            self.extractor = Frame_Extractor(video_filepath)
        except:
            self.extractor = Frame_Extractor_Img(video_filepath)

        self.total_frames = self.extractor.get_total_frames()
        self.current_frame = None
        self.frame_idx = 0
        self.roi = tuple(roi) if roi is not None else None
        self.blob_array = blob_array
        self.last_reset_query = 0

        self.core = BC_Core()

        self.threshold = 50
        
        self.min_blob_area = 2000
        self.blob_type = "Dark Blobs"
        self.bg_removal_method = "None"
        self.background_frames: Dict[str, np.ndarray] = {}
        self.double_blob_area_threshold = 100000
        self.working_bg: Optional[np.ndarray] = None

        self._setup_ui()

        if config is not None:
            self._apply_config(config)
        else:
            self._reset_blob_array()

        self.parameters_changed.connect(self._reset_and_reprocess)

        if config is not None and blob_array is not None and np.any(blob_array != 0):
            self._collapse_histogram()

    def _setup_ui(self):
        self.blb_layout = QVBoxLayout(self)
        self.setFixedWidth(200)
        self._setup_background_display()
        self._setup_histogram_section()
        self._setup_controls()
        self._setup_action_buttons()

    def _setup_background_display(self):
        self.bg_display_btn = QPushButton("View Background")
        self.bg_display_btn.clicked.connect(self._show_background_in_dialog)

        self.bg_regen_btn = QPushButton("Regenerate Background")
        self.bg_regen_btn.clicked.connect(self._regen_bg)

        self.select_roi_btn = QPushButton("Set ROI")
        self.select_roi_btn.clicked.connect(self._select_roi)

        self.blb_layout.addWidget(self.bg_display_btn)
        self.blb_layout.addWidget(self.bg_regen_btn)
        self.blb_layout.addWidget(self.select_roi_btn)

    def _setup_histogram_section(self):
        self.hist_section_widget = QtWidgets.QWidget()
        self.hist_section_layout = QVBoxLayout(self.hist_section_widget)
        self.hist_section_layout.setContentsMargins(0, 0, 0, 0)
        self.hist_section_layout.setSpacing(2)

        self.hist_header_layout = QHBoxLayout()
        self.hist_toggle_btn = QPushButton("▼ Blob Size Histogram")
        self.hist_toggle_btn.setCheckable(True)
        self.hist_toggle_btn.setChecked(True)
        self.hist_toggle_btn.clicked.connect(self._toggle_histogram)
        self.hist_header_layout.addWidget(self.hist_toggle_btn)
        self.hist_section_layout.addLayout(self.hist_header_layout)

        self.hist_content_widget = QtWidgets.QWidget()
        self.hist_content_layout = QVBoxLayout(self.hist_content_widget)
        self.hist_content_layout.setContentsMargins(0, 0, 0, 0)
        
        self.histogram_label = QLabel("Blob Size Distribution (computing...)")
        self.hist_content_layout.addWidget(self.histogram_label)
        
        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(100)
        self.ax = self.fig.add_subplot(111)
        self.threshold_line = None
        
        self.hist_content_layout.addWidget(self.canvas)
        self.refresh_hist_btn = QPushButton("Refresh Histogram")
        self.refresh_hist_btn.clicked.connect(self._plot_blob_histogram)

        db_threshold_gbox = QGroupBox("Double Blob Threshold")
        dbg_layout = QVBoxLayout()

        self.db_threshold_slider = QSlider(Qt.Horizontal)
        self.db_threshold_slider.setRange(0, max(100, self.double_blob_area_threshold+100))
        self.db_threshold_slider.setValue(self.double_blob_area_threshold)
        self.db_threshold_slider.valueChanged.connect(self._on_db_threshold_changed)

        dbg_layout.addWidget(self.db_threshold_slider)
        db_threshold_gbox.setLayout(dbg_layout)

        self.hist_content_layout.addWidget(self.refresh_hist_btn)
        self.hist_section_layout.addWidget(self.hist_content_widget) 
        self.hist_content_layout.addWidget(db_threshold_gbox)
        self.blb_layout.addWidget(self.hist_section_widget)

    def _setup_controls(self):
        self.control_gbox = QGroupBox("Blob Counter Control")
        self.controls_layout = QVBoxLayout(self.control_gbox)
        self.blb_layout.addWidget(self.control_gbox)

        self.threshold_label = QLabel("Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(self.threshold)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.threshold_value_label = QLabel(str(self.threshold))
        self.min_area_label = QLabel("Min Blob Area:")
        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setRange(10, 20000)
        self.min_area_slider.setValue(self.min_blob_area)
        self.min_area_slider.valueChanged.connect(self._on_min_area_changed)
        self.min_area_value_label = QLabel(str(self.min_blob_area))

        self.blob_type_label = QLabel("Blob Type:")
        self.blob_type_combo = QComboBox()
        self.blob_type_combo.addItems(["Dark Blobs", "Light Blobs"])
        self.blob_type_combo.currentTextChanged.connect(self._on_blob_type_changed)

        self.bg_removal_label = QLabel("Background Removal:")
        self.bg_removal_combo = QComboBox()
        self.bg_removal_combo.addItems([
            "None", "Mean", "Min", "5th Percentile", "10th Percentile", "25th Percentile",
            "Median", "75th Percentile", "90th Percentile", "95th Percentile", "Max"])
        self.bg_removal_combo.currentTextChanged.connect(self._on_bg_removal_changed)

        self.max_worker_spin = Spinbox_With_Label("Max Workers: ", (1, 64), 8)
        self.max_worker_spin.setToolTip("Parallel processes for video counting")

        for w in [
            self.threshold_label, self.threshold_slider, self.threshold_value_label, self.min_area_label, self.min_area_slider, self.min_area_value_label,
            self.blob_type_label, self.blob_type_combo, self.bg_removal_label, self.bg_removal_combo, self.max_worker_spin]:
            self.controls_layout.addWidget(w)

        config_layout = QHBoxLayout()
        self.export_config_btn = QPushButton("Save Config")
        self.export_config_btn.clicked.connect(self._export_config_json)
        self.import_config_btn = QPushButton("Load Config")
        self.import_config_btn.clicked.connect(self._import_config_json)
        config_layout.addWidget(self.export_config_btn)
        config_layout.addWidget(self.import_config_btn)
        self.blb_layout.addLayout(config_layout)

        self.count_label = QLabel("Animal Count: 0")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.blb_layout.addWidget(self.count_label)

    def _setup_action_buttons(self):
        self.count_all_btn = QPushButton("Count Animals")
        self.count_all_btn.clicked.connect(self._count_video)
        self.blb_layout.addWidget(self.count_all_btn)

    def _get_background_for_core(self) -> Optional[np.ndarray]:
        if self.bg_removal_method == "None":
            return None
        return self.background_frames.get(self.bg_removal_method)

    def set_current_frame(self, frame: np.ndarray, frame_idx: int):
        self.current_frame = frame
        self.frame_idx = frame_idx
        self._reprocess_current_frame()

    def _reprocess_current_frame(self):
        if self.current_frame is None:
            self.count_label.setText("Animal Count: 0")
            return
        
        config = self.get_config()
        bg_frame = self._get_background_for_core()
        count, _, contours = self.core.process_frame(self.current_frame, config, bg_frame)
        
        self.count_label.setText(f"Animal Count: {count}")
        display_frame = self.core.draw_mask(self.current_frame, contours)
        self.frame_processed.emit(display_frame, count)

    def _toggle_histogram(self, checked: bool):
        self.hist_content_widget.setVisible(checked)
        self.hist_toggle_btn.setText("▼ Histogram" if checked else "▶ Histogram")

    def _collapse_histogram(self):
        self.hist_toggle_btn.setChecked(False)
        self.hist_content_widget.setVisible(False)
        self.hist_toggle_btn.setText("▶ Histogram")

    def _plot_blob_histogram(self):
        if not self.hist_content_widget.isVisible():
            self._toggle_histogram(True)
        if not self.extractor:
            return
        
        max_workers = self.max_worker_spin.value()
        
        self._reset_blob_array()
        self.video_counted.emit(self.blob_array)

        total_frames = self.extractor.get_total_frames()
        if total_frames == 0:
            return
        
        sample_count = min(200, total_frames // 200) or 1
        frame_indices = np.unique(np.linspace(0, total_frames - 1, sample_count, dtype=int))
        segment_length = min(200, (frame_indices[1] - frame_indices[0]) if len(frame_indices) > 1 else total_frames)

        segments = []
        for idx in frame_indices:
            seg_start = max(0, idx - segment_length//2)
            seg_end = min(self.total_frames, idx + segment_length//2 + 1)
            segments.append((seg_start, seg_end))

        areas = []
        pbar = tqdm(total=len(segments), desc="Analyzing blobs", leave=False, ncols=100)
        config_dict = self.get_config().to_dict()

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        analyze_blob_areas_worker,
                        self.video_filepath,
                        chunk[0], chunk[1],
                        config_dict
                    ): idx for idx, chunk in enumerate(segments)
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            areas.extend(result)
                    except Exception as e:
                        logger.error(f"[THREAD] Segment failed: {e}")
                    pbar.update(1)
        except Exception as e:
            logger.error(f"[THREAD] Critical failure: {e}")
            traceback.print_exc()
        finally:
            pbar.close()
            
        self._render_histogram(areas)

    def _render_histogram(self, areas: List[float]):
        if not areas:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No blobs found", transform=self.ax.transAxes, ha="center")
            self.canvas.draw()
            return
            
        self.ax.clear()
        upper = float(np.percentile(areas, 99))
        xlim_max = upper * 1.2 + 500
        xlim_min = np.nanmin(areas) * 0.95

        self.ax.hist(areas, bins=100, range=(0, xlim_max), color='skyblue', edgecolor='black', alpha=0.7)
        self.ax.set_xlabel("Blob Area (pixels²)")
        self.ax.set_ylabel("Frequency")
        self.ax.set_title("Blob Size Distribution")
        self.ax.set_xlim(xlim_min, xlim_max)

        self.db_threshold_slider.setMinimum(xlim_min)
        self.db_threshold_slider.setMaximum(xlim_max)

        if self.threshold_line:
            self.threshold_line.remove()

        self.threshold_line = self.ax.axvline(
            x=self.double_blob_area_threshold, color='red', linestyle='--', linewidth=2
        )

        self.canvas.draw()
        self.histogram_label.setText("Blob Size Distribution")

    def _compute_background(self, method: str, frame_indices: Optional[List[int]] = None) -> Optional[np.ndarray]:
        if method == "None":
            return None
            
        total = self.extractor.get_total_frames()
        if total == 0:
            return None
            
        if total <= 100:
            indices = range(total)
        elif frame_indices:
            indices = frame_indices
        else:
            indices = np.linspace(0, total - 1, 100, dtype=int)

        frames = []
        progress = Progress_Indicator_Dialog(0, len(indices), "Background", "Computing...", self)
        
        for i, idx in enumerate(indices):
            if progress.wasCanceled():
                break
            frame = self.extractor.get_frame(idx)
            if frame is not None:
                frames.append(frame)
            progress.setValue(i + 1)
        progress.close()

        if not frames:
            return None
            
        frame_array = np.array(frames)
        methods = {
            "Mean": lambda a: np.mean(a, axis=0),
            "Min": lambda a: np.min(a, axis=0),
            "Max": lambda a: np.max(a, axis=0),
            "Median": lambda a: np.median(a, axis=0),
            "5th Percentile": lambda a: np.percentile(a, 5, axis=0),
            "10th Percentile": lambda a: np.percentile(a, 10, axis=0),
            "25th Percentile": lambda a: np.percentile(a, 25, axis=0),
            "75th Percentile": lambda a: np.percentile(a, 75, axis=0),
            "90th Percentile": lambda a: np.percentile(a, 90, axis=0),
            "95th Percentile": lambda a: np.percentile(a, 95, axis=0),
        }
        
        if method in methods:
            bg = methods[method](frame_array).astype(np.uint8)
            self.background_frames[method] = bg
            return bg
        return None

    def _update_background_display(self):
        if self.bg_removal_method == "None":
            return
        if self.bg_removal_method not in self.background_frames:
            self._compute_background(self.bg_removal_method)

    def _show_background_in_dialog(self, event=None):
        if self.bg_removal_method == "None":
            return
        frame = self.background_frames.get(self.bg_removal_method)
        if self.roi:
            frame = plot_roi(frame, np.array(self.roi))
        qimg = frame_to_qimage(frame)
        dialog = Frame_Display_Dialog(title=f"Background — {self.bg_removal_method}", image=qimg)
        dialog.exec()

    def _on_threshold_changed(self, value: int):
        self.threshold = value
        self.threshold_value_label.setText(str(value))
        self.parameters_changed.emit()

    def _on_db_threshold_changed(self, value: int):
        self.double_blob_area_threshold = value
        if self.threshold_line:
            self.threshold_line.set_xdata([self.double_blob_area_threshold, self.double_blob_area_threshold])
            self.canvas.draw()
        self.parameters_changed.emit()

    def _on_min_area_changed(self, value: int):
        self.min_blob_area = value
        self.min_area_value_label.setText(str(value))
        self.parameters_changed.emit()

    def _on_blob_type_changed(self, text: str):
        self.blob_type = text
        self.parameters_changed.emit()

    def _on_bg_removal_changed(self, text: str):
        self.bg_removal_method = text
        self.parameters_changed.emit()

    def _regen_bg(self):
        self.background_frames.clear()
        self._update_background_display()

    def _select_roi(self):
        frame = self.extractor.get_frame(0)
        if frame is None:
            return
        roi = get_roi_cv2(frame)
        if roi is not None:
            self.roi = tuple(int(x) for x in roi)
            logger.info(f"[BLOB] ROI set to {self.roi}")
        else:
            logger.info("[BLOB] ROI selection canceled.")
        self.parameters_changed.emit()
        self.roi_set.emit(np.array(self.roi) if self.roi else None)

    def _reset_and_reprocess(self):
        self._update_background_display()
        self.working_bg = self._get_background_for_core()
        self._reset_blob_array()
        self._reprocess_current_frame()

    def _reset_blob_array(self):
        if not hasattr(self, "blob_array") or self.blob_array is None:
            logger.info("[BCOUNT] Preallocating blob array")
            self.blob_array = np.zeros((self.total_frames, 2), dtype=np.uint8)
            return
        if np.any(self.blob_array != 0):
            now = time.time()
            if now - self.last_reset_query < 60.0:
                return
            self.last_reset_query = now
            if Loggerbox.question(self, "Reset Counting?", "Parameters changed. Reset results?") == QMessageBox.No:
                return
        self.blob_array = np.zeros((self.total_frames, 2), dtype=np.uint8)

    def _count_video(self):
        fm_dialog = Frame_Range_Dialog(self.total_frames, parent=self)
        if fm_dialog.exec() != QtWidgets.QDialog.Accepted:
            return
            
        max_workers = self.max_worker_spin.value()
        start_idx, end_idx = fm_dialog.selected_range
        end_idx += 1
        segments = self._task_splitter(start_idx, end_idx, max_workers)

        title = f"BLOB COUNTER | {max_workers} workers | {len(segments)} segments"
        border = "═" * (len(title) + 2)
        logger.info(f"\n╔{border}╗\n║ {title} ║\n╚{border}╝")

        pbar = tqdm(total=len(segments), desc="Processing segments", leave=False, ncols=100)
        config_dict = self.get_config().to_dict()

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_segment_worker,
                        self.video_filepath,
                        chunk[0], chunk[1],
                        config_dict
                    ): idx for idx, chunk in enumerate(segments)
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            s_start, s_end, chunk_array = result
                            self.blob_array[s_start:s_end] = chunk_array
                    except Exception as e:
                        logger.error(f"[THREAD] Segment failed: {e}")
                    pbar.update(1)
        except Exception as e:
            logger.error(f"[THREAD] Critical failure: {e}")
        finally:
            pbar.close()
            
        self.video_counted.emit(self.blob_array)

    @staticmethod
    def _task_splitter(start_idx: int, end_idx: int, max_workers: int) -> List[Tuple[int, int]]:
        segment_size = min(1000, (end_idx - start_idx) // max_workers) or 1
        chunks = []
        chunk_start = start_idx
        while chunk_start < end_idx:
            chunk_end = min(end_idx, chunk_start + segment_size)
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end
        return chunks

    def get_config(self) -> Blob_Config:
        return Blob_Config(
            threshold=self.threshold,
            double_blob_area_threshold=self.double_blob_area_threshold,
            min_blob_area=self.min_blob_area,
            bg_removal_method=self.bg_removal_method,
            blob_type=self.blob_type,
            background_frames=self.background_frames if self.background_frames else None,
            roi=np.array(self.roi) if self.roi else None
        )

    def _apply_config(self, config: Blob_Config):
        self.threshold = config.threshold
        self.min_blob_area = config.min_blob_area
        self.bg_removal_method = config.bg_removal_method
        self.blob_type = config.blob_type
        self.roi = tuple(config.roi) if config.roi is not None else None
        self.background_frames = config.background_frames or {}
        self.double_blob_area_threshold = config.double_blob_area_threshold

        self.threshold_slider.setValue(self.threshold)
        self.db_threshold_slider.setValue(self.double_blob_area_threshold)
        self.threshold_value_label.setText(str(self.threshold))
        self.min_area_slider.setValue(self.min_blob_area)
        self.min_area_value_label.setText(str(self.min_blob_area))
        self.blob_type_combo.setCurrentText(self.blob_type)
        self.bg_removal_combo.setCurrentText(config.bg_removal_method)
        
        self.parameters_changed.emit()

    def _export_config_json(self):
        config_dict = {
            "threshold": str(self.threshold),
            "double_blob_area_threshold": str(self.double_blob_area_threshold),
            "min_blob_area": str(self.min_blob_area),
            "bg_removal_method": self.bg_removal_method,
            "blob_type": self.blob_type,
            "roi": list(self.roi) if self.roi else None
        }
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Config", "", "JSON Files (*.json)")
        if file_path:
            if not file_path.endswith('.json'):
                file_path += '.json'
            try:
                with open(file_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                logger.info(f"[CONFIG] Exported to {file_path}")
            except Exception as e:
                logger.error(f"[CONFIG] Export failed: {e}")
                QMessageBox.critical(self, "Export Error", str(e))

    def _import_config_json(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Config", "", "JSON Files (*.json)")
        if not file_path:
            return
        try:
            with open(file_path, 'r') as f:
                d = json.load(f)
            if "threshold" in d:
                self.threshold = int(d["threshold"])
                self.threshold_slider.setValue(self.threshold)
                self.threshold_value_label.setText(str(self.threshold))
            if "double_blob_area_threshold" in d:
                self.double_blob_area_threshold = int(d["double_blob_area_threshold"])
                self.db_threshold_slider.setValue(self.double_blob_area_threshold)
            if "min_blob_area" in d:
                self.min_blob_area = int(d["min_blob_area"])
                self.min_area_slider.setValue(self.min_blob_area)
                self.min_area_value_label.setText(str(self.min_blob_area))
            if "bg_removal_method" in d:
                self.bg_removal_method = str(d["bg_removal_method"])
                self.bg_removal_combo.setCurrentText(self.bg_removal_method)
            if "blob_type" in d:
                self.blob_type = str(d["blob_type"])
                self.blob_type_combo.setCurrentText(self.blob_type)
            if "roi" in d and d["roi"]:
                self.roi = tuple(int(x) for x in d["roi"])
            logger.info(f"[CONFIG] Imported from {file_path}")
            self.parameters_changed.emit()
            self._reset_and_reprocess()
        except Exception as e:
            logger.error(f"[CONFIG] Import failed: {e}")
            QMessageBox.critical(self, "Import Error", str(e))
        

class BC_Core:
    KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    def __init__(self, config: Optional[Blob_Config] = None):
        self.config = config
    
    def process_contours_from_frame(
        self, 
        frame: np.ndarray, 
        config: Optional[Blob_Config] = None,
        background_frame: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        cfg = config or self.config

        if cfg.roi is not None:
            x1, y1, x2, y2 = cfg.roi
            frame_to_process = frame[y1:y2, x1:x2].copy()
            roi_offset = np.array([[x1, y1]], dtype=np.int32)
        else:
            frame_to_process = frame
            roi_offset = None

        gray_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)

        if cfg.bg_removal_method != "None" and background_frame is not None:
            if cfg.roi is not None:
                x1, y1, x2, y2 = cfg.roi
                bg_roi = background_frame[y1:y2, x1:x2]
            else:
                bg_roi = background_frame
            
            bg_gray = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY) if len(bg_roi.shape) == 3 else bg_roi
            working_bg = bg_gray.astype(np.int16)
            diff = gray_frame.astype(np.int16) - working_bg
            
            if cfg.blob_type == "Dark Blobs":
                thresh = (diff < -cfg.threshold).astype(np.uint8) * 255
            else:
                thresh = (diff > cfg.threshold).astype(np.uint8) * 255
        else:
            if cfg.blob_type == "Dark Blobs":
                _, thresh = cv2.threshold(gray_frame, cfg.threshold, 255, cv2.THRESH_BINARY_INV)
            else:
                _, thresh = cv2.threshold(gray_frame, cfg.threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > cfg.min_blob_area:
                if roi_offset is not None:
                    cnt = cnt + roi_offset
                filtered_contours.append(cnt)

        return filtered_contours

    def count_contours(
        self, 
        contours: List[np.ndarray], 
        double_threshold: Optional[int] = None
    ) -> Tuple[int, int]:
        if not contours:
            return 0, 0

        threshold = double_threshold if double_threshold is not None else self.config.double_blob_area_threshold
        merged = 0
        animal_count = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= threshold:
                animal_count += 2
                merged = 1
            else:
                animal_count += 1

        return animal_count, merged

    def process_frame(
        self, 
        frame: np.ndarray, 
        config: Optional[Blob_Config] = None,
        background_frame: Optional[np.ndarray] = None
    ) -> Tuple[int, int, List[np.ndarray]]:
        contours = self.process_contours_from_frame(frame, config, background_frame)
        count, merged = self.count_contours(contours, config.double_blob_area_threshold if config else None)
        return count, merged, contours

    @staticmethod
    def draw_mask(frame: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, (0, 255, 0), cv2.FILLED)
        display_frame = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)
        cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 2)
        return display_frame


def process_segment_worker(
    video_filepath: str,
    start_idx: int,
    end_idx: int,
    config_dict: dict,
) -> Optional[Tuple[int, int, np.ndarray]]:
    
    config = Blob_Config.from_dict(config_dict)
    core = BC_Core(config)

    bg_removal_method = config.bg_removal_method
    background_frame = None
    if bg_removal_method != "None":
        background_frame = config.background_frames[bg_removal_method]
    
    try:
        extractor = Frame_Extractor(video_filepath)
        extractor.start_sequential_read(start=start_idx, end=end_idx)
        chunk_array = np.zeros((end_idx - start_idx, 2), dtype=np.uint8)

        frame_idx = 0
        while frame_idx < end_idx - start_idx:
            result = extractor.read_next_frame()
            if result is None:
                break
            actual_idx, frame = result
            if actual_idx != frame_idx + start_idx:
                raise ValueError(f"Frame index mismatch: expected {frame_idx + start_idx}, got {actual_idx}")

            count, merged, _ = core.process_frame(frame, config, background_frame)
            chunk_array[frame_idx, 0] = count
            chunk_array[frame_idx, 1] = merged
            frame_idx += 1
            
        return start_idx, end_idx, chunk_array
        
    except Exception as e:
        try:
            logger.warning(f"[Segment {start_idx}-{end_idx}] Processing failed: {e}")
        except:
            pass
        return None
    finally:
        try:
            extractor.finish_sequential_read()
        except:
            pass

def analyze_blob_areas_worker(
    video_filepath: str,
    start_idx: int,
    end_idx: int,
    config_dict: dict,
    step: int = 3
) -> Optional[List[float]]:

    config = Blob_Config.from_dict(config_dict)
    core = BC_Core(config)

    bg_removal_method = config.bg_removal_method
    background_frame = None
    if bg_removal_method is not None:
        background_frame = config.background_frames[bg_removal_method]

    try:
        extractor = Frame_Extractor(video_filepath)
        extractor.start_sequential_read(start=start_idx, end=end_idx)
        
        areas = []
        frame_idx = 0
        
        while frame_idx < end_idx - start_idx:
            result = extractor.read_next_frame()
            if result is None:
                break
            _, frame = result
            
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            _, _, contours = core.process_frame(frame, config, background_frame)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > config.min_blob_area:
                    areas.append(float(area))
            
            frame_idx += 1
            
        extractor.finish_sequential_read()
        return areas

    except Exception as e:
        try:
            logger.error(f"[HIST_WORKER] Failed segment {start_idx}-{end_idx}: {e}")
        except:
            pass
        return []