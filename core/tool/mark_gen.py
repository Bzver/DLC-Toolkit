import numpy as np

from PySide6 import QtWidgets
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QGroupBox, QCheckBox
from PySide6.QtGui import QIntValidator

from typing import Optional, Dict

from .outlier_finder import Outlier_Container
from utils.helper import get_instance_count_per_frame
from utils.logger import Loggerbox
from utils.dataclass import Loaded_DLC_Data

class Mark_Generator(QGroupBox):
    clear_old = Signal(bool)
    frame_list_new = Signal(list)

    def __init__(self,
            total_frames:int,
            pred_data_array:Optional[np.ndarray]=None,
            blob_array:Optional[np.ndarray]=None,
            dlc_data:Optional[Loaded_DLC_Data]=None,
            angle_map_data:Optional[Dict[str, int]]=None,
            parent=None
            ):
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
                self.pred_data_array, skele_list=dlc_data.skeleton, kp_to_idx=dlc_data.keypoint_to_idx, angle_map_data=angle_map_data)
        else:
            self.outlier_container = QtWidgets.QWidget()
        
        self.animal_num_container = self._build_animal_num_container()
        
        self.random_container.setVisible(True)
        layout.addWidget(self.random_container)
        self.stride_container.setVisible(False)
        layout.addWidget(self.stride_container)
        self.outlier_container.setVisible(False)
        layout.addWidget(self.outlier_container)
        self.animal_num_container.setVisible(False)
        layout.addWidget(self.animal_num_container)

        confirm_frame = QVBoxLayout()
        self.keep_old_checkbox = QtWidgets.QCheckBox("Keep Existing Marks")
        self.keep_old_checkbox.setChecked(True)
        confirm_frame.addWidget(self.keep_old_checkbox)

        okay_button = QPushButton("Mark Frames")
        okay_button.clicked.connect(self.find_frames_to_mark)
        confirm_frame.addWidget(okay_button)
        layout.addLayout(confirm_frame)

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
            num_text = self.random_textbox.text().strip()
            if not num_text:
                Loggerbox.error(self, "Missing Input", "Please enter number of frames to mark.")
                return
            try:
                n = int(num_text)
                import random
                selected_frames = random.sample(frame_range, min(n, len(frame_range)))
            except (ValueError, TypeError):
                Loggerbox.error(self, "Invalid Number", "Please enter a valid positive number.")
                return

        elif mode == "Stride":
            stride_text = self.stride_textbox.text().strip()
            if not stride_text:
                Loggerbox.error(self, "Missing Input", "Please enter a stride interval.")
                return
            try:
                step = int(stride_text)
                selected_frames = frame_range[::step]
            except ValueError:
                Loggerbox.error(self, "Invalid Stride", "Stride must be a positive integer.")
                return

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
                merged_array = self.blob_array[:, 1]

            if self.pred_data_array is not None:
                animal_count_array = get_instance_count_per_frame(self.pred_data_array)

            selected_options = []
            if self.zero_animal_checkbox.isChecked():
                selected_options.append(0)
            if self.one_animal_checkbox.isChecked():
                selected_options.append(1)
            if self.two_plus_animal_checkbox.isChecked():
                selected_options.extend(range(2, int(np.max(animal_count_array)) + 1))
            if hasattr(self, 'merged_animal_checkbox') and self.merged_animal_checkbox.isChecked():
                merged_frames = np.where(merged_array == 1)[0]
                selected_frames.extend(merged_frames.tolist())

            if selected_options and not (hasattr(self, 'merged_animal_checkbox') and self.merged_animal_checkbox.isChecked()):
                mask = np.isin(animal_count_array, selected_options)
                count_frames = np.where(mask)[0]
                selected_frames.extend(count_frames.tolist())
            elif not selected_options and not (hasattr(self, 'merged_animal_checkbox') and self.merged_animal_checkbox.isChecked()):
                Loggerbox.error(self, "No Selection", "Please select at least one animal count option.")
                return
        else:
            Loggerbox.error(self, "Invalid Mode", "Unknown mode selected.")

        selected_frames = [f for f in selected_frames if start_frame <= f <= end_frame]
        selected_frames = list(set(selected_frames))
        selected_frames.sort()
        
        if not selected_frames:
            Loggerbox.info(self, "No Frames Found", "No frames matched the selected criteria in the given range.")
            return

        self.clear_old.emit(not self.keep_old_checkbox.isChecked())
        self.frame_list_new.emit(selected_frames)

    def _build_random_container(self):
        container = QGroupBox("Random Frame Extraction")
        layout = QHBoxLayout(container)
        label = QLabel("Number of Frames to Mark: ")
        layout.addWidget(label)
        self.random_textbox = QLineEdit()
        validator = QIntValidator(1, self.total_frames)
        self.random_textbox.setValidator(validator)
        layout.addWidget(self.random_textbox)
        container.setMaximumHeight(70)
        return container

    def _build_stride_container(self):
        container = QGroupBox("Stride Frame Extraction")
        layout = QHBoxLayout(container)
        label = QLabel("Stride Interval:")
        self.stride_textbox = QLineEdit()
        self.stride_textbox.setPlaceholderText("e.g., 5")
        validator = QIntValidator(1, self.total_frames)
        self.stride_textbox.setValidator(validator)
        layout.addWidget(label)
        layout.addWidget(self.stride_textbox)
        container.setMaximumHeight(70)
        return container

    def _build_animal_num_container(self):
        container = QGroupBox("Animal Count Selection")
        main_layout = QVBoxLayout(container)
        
        count_layout = QHBoxLayout()
        self.zero_animal_checkbox = QCheckBox("0 Animal")
        self.one_animal_checkbox = QCheckBox("1 Animal")  
        self.two_plus_animal_checkbox = QCheckBox("2+ Animal")
        
        count_layout.addWidget(self.zero_animal_checkbox)
        count_layout.addWidget(self.one_animal_checkbox)
        count_layout.addWidget(self.two_plus_animal_checkbox)
        
        if self.blob_array is not None:
            self.merged_animal_checkbox = QCheckBox("Merged Animal")
            count_layout.addWidget(self.merged_animal_checkbox)
        
        if self.pred_data_array is not None and self.blob_array is not None:
            self.discrepancy_btn = QPushButton("Find DLC-Blob Discrepancy")
            self.discrepancy_btn.clicked.connect(self._mark_count_discrepancy_frames)

            main_layout.addWidget(self.discrepancy_btn)

        main_layout.addLayout(count_layout)
            
        container.setMaximumHeight(100)
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

        discrepancy_frames = np.where(dlc_count_array != blob_count_array)[0].tolist()
        discrepancy_frames = [f for f in discrepancy_frames if start_frame <= f <= end_frame]

        if not discrepancy_frames:
            Loggerbox.info(self, "No Frames Found", "No frames matched the selected criteria in the given range.")
            return

        self.clear_old.emit(not self.keep_old_checkbox.isChecked())
        self.frame_list_new.emit(discrepancy_frames)

    def _on_selection_changed(self):
        """Show the appropriate container based on selected mode."""
        mode = self.mode_option.currentText()

        self.random_container.setVisible(mode == "Random")
        self.stride_container.setVisible(mode == "Stride")
        self.outlier_container.setVisible(mode == "Outlier")
        self.animal_num_container.setVisible(mode == "Animal Num")