import numpy as np
import random

from PySide6 import QtWidgets
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QGroupBox, QCheckBox
from PySide6.QtGui import QIntValidator

from typing import Optional, Dict

from .outlier_finder import Outlier_Container
from ui import Spinbox_With_Label
from utils.helper import get_instance_count_per_frame
from utils.logger import Loggerbox
from utils.dataclass import Loaded_DLC_Data

class Mark_Generator(QGroupBox):
    frame_list_replace = Signal(list)
    frame_list_subset = Signal(list)
    frame_list_combine = Signal(list)

    def __init__(self,
                 total_frames: int,
                 pred_data_array: Optional[np.ndarray] = None,
                 blob_array: Optional[np.ndarray] = None,
                 dlc_data: Optional[Loaded_DLC_Data] = None,
                 angle_map_data: Optional[Dict[str, int]] = None,
                 parent=None):
        super().__init__(parent)
        self.setTitle("Automatic Mark Generation")
        self.total_frames = total_frames
        self.pred_data_array = pred_data_array
        self.blob_array = blob_array

        layout = QVBoxLayout(self)

        self.mode_frame = QHBoxLayout()
        mode_label = QLabel("Mark Generation Mode:")
        self.mode_frame.addWidget(mode_label)

        self.mode_option = QtWidgets.QComboBox()
        self.mode_option.addItems(["Random", "Stride"])

        if self.pred_data_array is not None:
            self.mode_option.addItem("Outlier")
        else:
            self.mode_option.setToolTip("Outlier mode requires loaded DLC data")

        if (self.blob_array is not None and np.any(self.blob_array[:, 0])) or self.pred_data_array is not None:
            self.mode_option.addItem("Animal Num")

        self.mode_option.addItem("Clipboard")

        self.mode_option.setCurrentIndex(0)
        self.mode_option.currentTextChanged.connect(self._on_selection_changed)
        self.mode_frame.addWidget(self.mode_option)
        layout.addLayout(self.mode_frame)

        range_frame = QHBoxLayout()
        range_start_label = QLabel("Enter Frame Range: ")
        range_symbol_label = QLabel(" ~ ")

        self.start_frame_textbox = QLineEdit()
        self.end_frame_textbox = QLineEdit()
        self.start_frame_textbox.setText("0")
        self.end_frame_textbox.setText(str(self.total_frames - 1))

        validator_range = QIntValidator(0, self.total_frames - 1)
        self.start_frame_textbox.setValidator(validator_range)
        self.end_frame_textbox.setValidator(validator_range)

        range_frame.addWidget(range_start_label)
        range_frame.addWidget(self.start_frame_textbox)
        range_frame.addWidget(range_symbol_label)
        range_frame.addWidget(self.end_frame_textbox)
        layout.addLayout(range_frame)

        self.random_container = self._build_random_container()
        self.stride_container = self._build_stride_container()

        if self.pred_data_array is not None:
            self.outlier_container = Outlier_Container(
                self.pred_data_array, skele_list=dlc_data.skeleton,
                kp_to_idx=dlc_data.keypoint_to_idx, angle_map_data=angle_map_data)
        else:
            self.outlier_container = QtWidgets.QWidget()

        self.animal_num_container = self._build_animal_num_container()
        self.clipboard_container = self._build_clipboard_container()

        self._mode_containers = []
        self._mode_containers.append(self.random_container)
        self._mode_containers.append(self.stride_container)
        if self.pred_data_array is not None:
            self._mode_containers.append(self.outlier_container)
        if (self.blob_array is not None and np.any(self.blob_array[:, 0])) or self.pred_data_array is not None:
            self._mode_containers.append(self.animal_num_container)
        self._mode_containers.append(self.clipboard_container)

        self.stack = QtWidgets.QStackedWidget()
        for container in self._mode_containers:
            self.stack.addWidget(container)

        layout.addWidget(self.stack)

        mode_group = QGroupBox("Marking Behavior")
        mode_layout = QHBoxLayout(mode_group)

        self.replace_radio = QtWidgets.QRadioButton("REPLACE")
        self.subset_radio = QtWidgets.QRadioButton("SUBSET")
        self.combine_radio = QtWidgets.QRadioButton("COMBINE")

        self.combine_radio.setChecked(True)

        mode_layout.addWidget(self.replace_radio)
        mode_layout.addWidget(self.subset_radio)
        mode_layout.addWidget(self.combine_radio)
        layout.addWidget(mode_group)

        okay_button = QPushButton("Mark Frames")
        okay_button.clicked.connect(self.find_frames_to_mark)
        layout.addWidget(okay_button)

    def find_frames_to_mark(self):
        start_text = self.start_frame_textbox.text().strip()
        end_text = self.end_frame_textbox.text().strip()

        try:
            start_frame = int(start_text) if start_text else 0
            end_frame = int(end_text) if end_text else self.total_frames - 1
        except ValueError:
            Loggerbox.error(self, "Invalid Input", "Please enter valid frame numbers.")
            return

        if not (0 <= start_frame <= end_frame < self.total_frames):
            Loggerbox.error(self, "Invalid Range", f"Frame range must be 0–{self.total_frames - 1}")
            return

        frame_range = list(range(start_frame, end_frame + 1))
        selected_frames = []

        mode = self.mode_option.currentText()

        if mode == "Random":
            n = self.random_spin.value()
            selected_frames = random.sample(frame_range, min(n, len(frame_range)))

        elif mode == "Stride":
            stride = self.stride_spin.value()
            step = int(stride)
            selected_frames = frame_range[::step]

        elif mode == "Outlier":
            if self.pred_data_array is None:
                Loggerbox.error(self, "No DLC Data", "Outlier detection requires DLC data.")
                return

            outlier_mask = self.outlier_container.get_combined_mask()
            if outlier_mask is None or not np.any(outlier_mask):
                Loggerbox.info(self, "No Outliers Found", "No frames matched the selected outlier criteria in the given range.")
                return

            data_length = self.pred_data_array.shape[0]
            mask_range = np.zeros(self.total_frames, dtype=bool)

            clipped_range = np.clip(frame_range, 0, data_length - 1)
            mask_range[clipped_range] = True

            combined_mask = np.any(outlier_mask, axis=1) & mask_range
            if not np.any(combined_mask):
                Loggerbox.info(self, "No Outliers in Range", "No outliers found within the selected frame range.")
                return

            selected_frames = np.where(combined_mask)[0].tolist()

        elif mode == "Animal Num":
            if self.blob_array is not None:
                animal_count_array = self.blob_array[:, 0]

            animal_count_array = self._acquire_animal_count_source()
            if animal_count_array is None:
                return

            selected_options = []
            if self.zero_animal_checkbox.isChecked():
                selected_options.append(0)
            if self.one_animal_checkbox.isChecked():
                selected_options.append(1)
            if self.two_plus_animal_checkbox.isChecked():
                selected_options.extend(range(2, int(np.max(animal_count_array)) + 1))
            
            has_merged = hasattr(self, 'merged_animal_checkbox') and self.merged_animal_checkbox.isChecked()
            if has_merged:
                merged_frames = np.where(self.blob_array[:, 1] == 1)[0]
                selected_frames.extend(merged_frames.tolist())

            if selected_options:
                mask = np.isin(animal_count_array, selected_options)
                count_frames = np.where(mask)[0]
                selected_frames.extend(count_frames.tolist())
            elif not selected_options and not has_merged:
                Loggerbox.error(self, "No Selection", "Please select at least one animal count option.")
                return
            
        elif mode == "Clipboard":
            text = self.clipboard_textbox.toPlainText().strip()
            if not text:
                Loggerbox.error(self, "Empty Input", "Please enter a list of frame numbers.")
                return

            try:
                text = text.replace('\n', ',')
                parts = []
                for segment in text.split(','):
                    subparts = segment.split()
                    parts.extend(subparts)

                frame_nums = []
                for s in parts:
                    s = s.strip()
                    if s:
                        frame_nums.append(int(s))
                        
            except ValueError:
                Loggerbox.error(self, "Invalid Format", "Please enter only integers separated by commas, spaces, or newlines.")
                return

            selected_frames = [f for f in frame_nums if start_frame <= f <= end_frame]

            if not selected_frames:
                Loggerbox.info(self, "No Valid Frames", f"No frames fall within the range [{start_frame}, {end_frame}].")
                return

        else:
            Loggerbox.error(self, "Invalid Mode", "Unknown mode selected.")

        selected_frames = [f for f in selected_frames if start_frame <= f <= end_frame]
        selected_frames = list(set(selected_frames))
        selected_frames.sort()
        
        if not selected_frames:
            Loggerbox.info(self, "No Frames Found", "No frames matched the selected criteria in the given range.")
            return

        if self.replace_radio.isChecked():
            self.frame_list_replace.emit(selected_frames)
        elif self.subset_radio.isChecked():
            self.frame_list_subset.emit(selected_frames)
        elif self.combine_radio.isChecked():
            self.frame_list_combine.emit(selected_frames)

        return selected_frames

    def _build_random_container(self):
        container = QGroupBox("Random Frame Extraction")
        layout = QHBoxLayout(container)
        self.random_spin = Spinbox_With_Label("Number of Frames to Mark: ", (1, self.total_frames), 100)
        layout.addWidget(self.random_spin)
        return container

    def _build_stride_container(self):
        container = QGroupBox("Stride Frame Extraction")
        layout = QHBoxLayout(container)
        self.stride_spin = Spinbox_With_Label("Stride Interval: ", (1, self.total_frames-1), 1)
        layout.addWidget(self.stride_spin)
        return container

    def _build_animal_num_container(self):
        container = QtWidgets.QWidget()
        main_layout = QVBoxLayout(container)

        source_box = QGroupBox("Data Source")
        source_layout = QHBoxLayout()
        
        self.dlc_source_radio = QtWidgets.QRadioButton("DLC Prediction")
        self.blob_source_radio = QtWidgets.QRadioButton("Blob Counter")
        
        if self.pred_data_array is not None and self.blob_array is not None:
            self.dlc_source_radio.setChecked(True)
        elif self.pred_data_array is not None:
            self.dlc_source_radio.setChecked(True)
            self.blob_source_radio.setEnabled(False)
            self.blob_source_radio.setToolTip("Blob data not available")
        elif self.blob_array is not None and np.any(self.blob_array[:, 0]):
            self.blob_source_radio.setChecked(True)
            self.dlc_source_radio.setEnabled(False)
            self.dlc_source_radio.setToolTip("DLC prediction data not available")
        else:
            self.dlc_source_radio.setEnabled(False)
            self.blob_source_radio.setEnabled(False)
        
        source_layout.addWidget(self.dlc_source_radio)
        source_layout.addWidget(self.blob_source_radio)
        source_box.setLayout(source_layout)
        main_layout.addWidget(source_box)

        count_box = QGroupBox("Selection Based on Raw Count")
        count_layout = QHBoxLayout()
        self.zero_animal_checkbox = QCheckBox("0 Animal")
        self.one_animal_checkbox = QCheckBox("1 Animal")  
        self.two_plus_animal_checkbox = QCheckBox("2+ Animal")

        count_layout.addWidget(self.zero_animal_checkbox)
        count_layout.addWidget(self.one_animal_checkbox)
        count_layout.addWidget(self.two_plus_animal_checkbox)
        count_box.setLayout(count_layout)

        if self.blob_array is not None:
            self.merged_animal_checkbox = QCheckBox("Merged Animal")
            count_layout.addWidget(self.merged_animal_checkbox)

        main_layout.addWidget(count_box)

        if self.pred_data_array is not None or self.blob_array is not None:
            change_box = QGroupBox("Selection Based on Count Change")
            change_layout = QVBoxLayout()
            self.buffer_size_spin = Spinbox_With_Label("Buffer Frames: ", (1, 100), 2)
            self.short_seg_btn = QPushButton("Find Animal Count Change Frames")
            self.short_seg_btn.clicked.connect(self._mark_count_change_frames)
            change_layout.addWidget(self.buffer_size_spin)
            change_layout.addWidget(self.short_seg_btn)
            change_box.setLayout(change_layout)
            main_layout.addWidget(change_box)

        if self.pred_data_array is not None and self.blob_array is not None:
            dscr_box = QGroupBox("Selection Based on DLC-Blob Discrepancy")
            dscr_layout = QVBoxLayout()
            self.discrepancy_btn = QPushButton("Find DLC-Blob Discrepancy")
            self.discrepancy_btn.clicked.connect(self._mark_count_discrepancy_frames)
            dscr_layout.addWidget(self.discrepancy_btn)
            dscr_box.setLayout(dscr_layout)
            main_layout.addWidget(dscr_box)

        return container

    def _mark_count_discrepancy_frames(self):
        start_text = self.start_frame_textbox.text().strip()
        end_text = self.end_frame_textbox.text().strip()
        
        try:
            start_frame = int(start_text) if start_text else 0
            end_frame = int(end_text) if end_text else self.total_frames - 1
        except ValueError:
            start_frame, end_frame = 0, self.total_frames - 1

        dlc_count_array = get_instance_count_per_frame(self.pred_data_array)
        blob_count_array = self.blob_array[:, 0]

        min_len = min(len(dlc_count_array), len(blob_count_array), self.total_frames)
        dlc_count_array = dlc_count_array[:min_len]
        blob_count_array = blob_count_array[:min_len]

        discrepancy_frames = np.where((dlc_count_array != blob_count_array) & (blob_count_array != 0))[0].tolist()
        discrepancy_frames = [f for f in discrepancy_frames if start_frame <= f <= end_frame]

        if not discrepancy_frames:
            Loggerbox.info(self, "No Frames Found", "No frames matched the selected criteria in the given range.")
            return

        if self.replace_radio.isChecked():
            self.frame_list_replace.emit(discrepancy_frames)
        elif self.subset_radio.isChecked():
            self.frame_list_subset.emit(discrepancy_frames)
        elif self.combine_radio.isChecked():
            self.frame_list_combine.emit(discrepancy_frames)

        return discrepancy_frames

    def _acquire_animal_count_source(self):
        if self.dlc_source_radio.isChecked() and self.pred_data_array is not None:
            return get_instance_count_per_frame(self.pred_data_array)
        elif self.blob_source_radio.isChecked() and self.blob_array is not None:
            return self.blob_array[:, 0]
        elif self.pred_data_array is not None:
            return get_instance_count_per_frame(self.pred_data_array)
        elif self.blob_array is not None:
            return self.blob_array[:, 0]
        else:
            Loggerbox.error(self, "No Data", "Either DLC prediction or blob counting is required to find count changes.")
            return None

    def _mark_count_change_frames(self):
        if self.pred_data_array is None and self.blob_array is None:
            Loggerbox.error(self, "No Data", "Either DLC prediction or blob counting is required to find count changes.")
            return

        animal_count_array = self._acquire_animal_count_source()
        if animal_count_array is None:
            return

        start_text = self.start_frame_textbox.text().strip()
        end_text = self.end_frame_textbox.text().strip()
        
        try:
            range_start = int(start_text) if start_text else 0
            range_end = int(end_text) if end_text else self.total_frames - 1
        except ValueError:
            range_start, range_end = 0, self.total_frames - 1

        range_start = max(0, range_start)
        range_end = min(self.total_frames - 1, range_end)

        marked_frames = set()
        buffer_size = self.buffer_size_spin.value()

        changes = np.diff(animal_count_array) != 0
        change_indices = np.where(changes)[0] + 1

        if len(change_indices) == 0:
            Loggerbox.info(self, "No Changes Found", "Animal count remained constant throughout the video.")
            return

        for idx in change_indices:
            start_buf = max(0, idx - buffer_size)
            end_buf = min(self.total_frames - 1, idx + buffer_size)
            
            for f in range(start_buf, end_buf + 1):
                marked_frames.add(f)

        selected_frames = sorted(list(marked_frames))
        selected_frames = [f for f in selected_frames if range_start <= f <= range_end]

        if not selected_frames:
            Loggerbox.info(self, "No Frames in Range", "Count changes were found, but none (including their buffers) fall within the specified frame range.")
            return

        if self.replace_radio.isChecked():
            self.frame_list_replace.emit(selected_frames)
        elif self.subset_radio.isChecked():
            self.frame_list_subset.emit(selected_frames)
        elif self.combine_radio.isChecked():
            self.frame_list_combine.emit(selected_frames)

        return selected_frames

    def _build_clipboard_container(self):
        container = QGroupBox("Paste or Edit Frame List")
        layout = QVBoxLayout(container)

        self.clipboard_textbox = QtWidgets.QTextEdit()
        self.clipboard_textbox.setPlaceholderText("Enter comma-separated frame numbers (e.g., 10, 20, 30)\nYou can also paste from Excel or a list.")
        self.clipboard_textbox.setToolTip("Comma-separated list of frame numbers. Newlines are auto-converted to commas.")

        layout.addWidget(self.clipboard_textbox)
        return container

    def _on_selection_changed(self):
        mode = self.mode_option.currentText()
        mode_names = ["Random", "Stride"]
        if self.pred_data_array is not None:
            mode_names.append("Outlier")
        if (self.blob_array is not None and np.any(self.blob_array[:, 0])) or self.pred_data_array is not None:
            mode_names.append("Animal Num")
        mode_names.append("Clipboard")

        index = mode_names.index(mode)
        self.stack.setCurrentIndex(index)