
import numpy as np

from PySide6 import QtWidgets
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox
from PySide6.QtGui import QIntValidator

from typing import Optional

from core.dataclass import Loaded_DLC_Data
from .outlier_finder import Outlier_Container

class Mark_Generator(QtWidgets.QGroupBox):
    clear_old = Signal(bool)
    frame_list_new = Signal(list)

    def __init__(self,
            total_frames:int,
            dlc_data:Optional[Loaded_DLC_Data]=None,
            canon_pose:Optional[np.ndarray]=None,
            parent=None
            ):
        """
        Initializes the dialog for generating frame marks using various strategies: random, 
        strided, or outlier-based selection.

        Args:
            total_frames (int): Total number of frames in the video, used to constrain input ranges.
            dlc_data (Optional[Loaded_DLC_Data]): DLC prediction data; required for "Outlier" mode.
            canon_pose (Optional[np.ndarray]): Canonical pose for size-based outlier detection.
            parent: Parent widget for modal behavior.
        """ 
        super().__init__(parent)
        self.setTitle("Automatic Mark Generation")
        self.total_frames = total_frames
        self.dlc_data = dlc_data

        layout = QVBoxLayout(self)

        self.mode_frame = QHBoxLayout()
        mode_label = QLabel("Mark Generation Mode:")
        self.mode_frame.addWidget(mode_label)

        self.mode_option = QtWidgets.QComboBox()  # Make it an instance attribute
        self.mode_option.addItems(["Random", "Stride"])

        if self.dlc_data is not None:
            self.mode_option.addItem("Outlier")
        else:
            self.mode_option.setToolTip("Outlier mode requires loaded DLC data")

        self.mode_option.setCurrentIndex(0)
        self.mode_option.currentTextChanged.connect(self._on_selection_changed)
        self.mode_frame.addWidget(self.mode_option)
        layout.addLayout(self.mode_frame)

        # Frame range input
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

        # Build all containers
        self.random_container = self._build_random_container()
        self.stride_container = self._build_stride_container()

        if self.dlc_data:
            self.outlier_container = Outlier_Container(
                self.dlc_data.pred_data_array, canon_pose=canon_pose)
        else:
            self.outlier_container = QtWidgets.QWidget() # Dummy container
        
        # Add all to layout
        self.random_container.setVisible(True)
        layout.addWidget(self.random_container)
        self.stride_container.setVisible(False)
        layout.addWidget(self.stride_container)
        self.outlier_container.setVisible(False)
        layout.addWidget(self.outlier_container)

        # Confirmation buttons
        confirm_frame = QHBoxLayout()
        self.keep_old_checkbox = QtWidgets.QCheckBox("Keep Existing Marks")
        self.keep_old_checkbox.setChecked(True)
        confirm_frame.addWidget(self.keep_old_checkbox)

        okay_button = QPushButton("Mark Frames")
        okay_button.clicked.connect(self.find_frames_to_mark)
        confirm_frame.addWidget(okay_button)
        layout.addLayout(confirm_frame)

    def find_frames_to_mark(self):
        """
        Processes user input and generates a list of frames to mark based on the selected mode.

        Modes:
            - "Random": Selects a specified number of random frames within the range.
            - "Stride": Selects frames at a fixed interval (stride).
            - "Outlier": Selects frames flagged by outlier detection criteria (if DLC data is available).

        Validates inputs and emits signals with the result:
            - clear_old: Whether to remove existing marks.
            - frame_list_new: List of newly selected frame indices.

        Shows appropriate error messages for invalid input or conditions.
        """
        start_text = self.start_frame_textbox.text().strip()
        end_text = self.end_frame_textbox.text().strip()

        try:
            start_frame = int(start_text) if start_text else 0
            end_frame = int(end_text) if end_text else self.total_frames - 1
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Please enter valid frame numbers.")
            return

        if not (0 <= start_frame <= end_frame < self.total_frames):
            QMessageBox.critical(self, "Invalid Range", f"Frame range must be 0–{self.total_frames - 1}")
            return

        frame_range = list(range(start_frame, end_frame + 1))
        selected_frames = []

        mode = self.mode_option.currentText()

        if mode == "Random":
            num_text = self.random_textbox.text().strip()
            if not num_text:
                QMessageBox.critical(self, "Missing Input", "Please enter number of frames to mark.")
                return
            try:
                n = int(num_text)
                import random
                selected_frames = random.sample(frame_range, min(n, len(frame_range)))
            except (ValueError, TypeError):
                QMessageBox.critical(self, "Invalid Number", "Please enter a valid positive number.")
                return

        elif mode == "Stride":
            stride_text = self.stride_textbox.text().strip()
            if not stride_text:
                QMessageBox.critical(self, "Missing Input", "Please enter a stride interval.")
                return
            try:
                step = int(stride_text)
                selected_frames = frame_range[::step]
            except ValueError:
                QMessageBox.critical(self, "Invalid Stride", "Stride must be a positive integer.")
                return

        elif mode == "Outlier":
            if self.dlc_data is None:
                QMessageBox.critical(self, "No DLC Data", "Outlier detection requires DLC data.")
                return

            outlier_mask = self.outlier_container.get_combined_mask()
            if outlier_mask is None or not np.any(outlier_mask):
                QMessageBox.information(self, "No Outliers Found",
                                        "No frames matched the selected outlier criteria in the given range.")
                return

            frame_count_in_data = self.dlc_data.pred_data_array.shape[0]
            mask_range = np.zeros(self.total_frames, dtype=bool)
            mask_range[frame_range] = True

            # Trucate or pad mask_range in case frame counts are different
            if self.total_frames >= frame_count_in_data:
                mask_range_processed = mask_range[range(frame_count_in_data)]
            else:
                mask_range_processed = np.zeros(self.total_frames, dtype=bool)
                mask_range_processed[frame_range] = True

            combined_mask = np.any(outlier_mask, axis=1) & mask_range_processed
            if not np.any(combined_mask):
                QMessageBox.information(self, "No Outliers in Range",
                                        "No outliers found within the selected frame range.")
                return

            selected_frames = np.where(outlier_mask)[0].tolist()

        else:
            QMessageBox.critical(self, "Invalid Mode", "Unknown mode selected.")

        selected_frames.sort()
        self.clear_old.emit(not self.keep_old_checkbox.isChecked())
        self.frame_list_new.emit(selected_frames)

    def _build_random_container(self):
        container = QtWidgets.QGroupBox("Random Frame Extraction")
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
        container = QtWidgets.QGroupBox("Stride Frame Extraction")
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

    def _on_selection_changed(self):
        """Show the appropriate container based on selected mode."""
        mode = self.mode_option.currentText()

        self.random_container.setVisible(mode == "Random")
        self.stride_container.setVisible(mode == "Stride")
        self.outlier_container.setVisible(mode == "Outlier")