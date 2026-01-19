import numpy as np

from ui import Spinbox_With_Label

from PySide6 import QtWidgets
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel

from typing import Optional, Dict

from utils.pose import (
    outlier_bodypart,
    outlier_confidence,
    outlier_duplicate,
    outlier_flicker,
    outlier_size,
    outlier_pose,
)
from utils.logger import Loggerbox

class Outlier_Finder(QGroupBox):
    list_changed = Signal(list)
    mask_changed = Signal(object)

    def __init__(self,
                pred_data_array:np.ndarray,
                canon_pose:Optional[np.ndarray]=None,
                angle_map_data:Optional[Dict[str, int]]=None,
                parent=None):
        super().__init__(parent)
        self.setTitle("Outlier Finder")
        self.pred_data_array = pred_data_array
        self.outlier_mask = None
        
        layout = QVBoxLayout(self)
        self.outlier_container = Outlier_Container(
            self.pred_data_array, canon_pose=canon_pose, angle_map_data=angle_map_data)
        layout.addWidget(self.outlier_container)

        button_frame = self._build_button_frame()
        layout.addLayout(button_frame)

    def _build_button_frame(self):
        button_frame = QHBoxLayout()

        preview_button = QPushButton("Preview Outliers")
        preview_button.clicked.connect(self.outlier_preview)

        delete_button = QPushButton("Delete Outliers")
        delete_button.clicked.connect(self.outlier_delete)

        button_frame.addWidget(preview_button)
        button_frame.addWidget(delete_button)
        return button_frame

    def get_outlier_mask(self):
        outlier_mask = self.outlier_container.get_combined_mask()
        self.outliers = outlier_mask
        if outlier_mask is None or not np.any(outlier_mask):
            self.list_changed.emit([])
            return

    def outlier_preview(self):
        self.get_outlier_mask()
        if self.outliers is not None:
            outlier_frames = np.where(np.any(self.outliers, axis=1))[0].tolist()
            self.list_changed.emit(outlier_frames)
        else:
            self.list_changed.emit([])

    def outlier_delete(self):
        self.get_outlier_mask()
        if self.outliers is not None:
            self.mask_changed.emit(self.outliers)
            self.list_changed.emit([])

class Outlier_Container(QtWidgets.QWidget):
    def __init__(self,
                pred_data_array:np.ndarray,
                canon_pose:Optional[np.ndarray]=None,
                angle_map_data:Optional[Dict[str, int]]=None,
                parent=None):
        super().__init__(parent)
        self.pred_data_array = pred_data_array
        self.canon_pose = canon_pose
        self.angle_map_data = angle_map_data

        layout = QVBoxLayout(self)

        self.logic_widget = self._build_or_and_radio_widget()
        self.outlier_confidence_gbox = self._build_outlier_confidence_gbox()
        self.outlier_bodypart_gbox = self._build_outlier_bodypart_gbox()
        self.outlier_size_gbox = self._build_outlier_size_gbox()
        self.outlier_duplicate_gbox = self._build_outlier_duplicate_gbox()
        self.outlier_pose_gbox = self._build_outlier_pose_gbox()
        self.outlier_flicker_gbox = self._build_outlier_flicker_gbox()

        layout.addWidget(self.logic_widget)
        layout.addWidget(self.outlier_confidence_gbox)
        layout.addWidget(self.outlier_bodypart_gbox)
        layout.addWidget(self.outlier_size_gbox)
        layout.addWidget(self.outlier_duplicate_gbox)
        layout.addWidget(self.outlier_pose_gbox)
        layout.addWidget(self.outlier_flicker_gbox)

        if self.canon_pose is None:
            self.outlier_size_gbox.setEnabled(False)
        if self.angle_map_data is None:
            self.outlier_pose_gbox.setEnabled(False)

    def get_combined_mask(self) -> Optional[np.ndarray]:
        masks = []

        if self.outlier_confidence_gbox.isChecked():
            mask = outlier_confidence(
                self.pred_data_array,
                threshold=self.confidence_spinbox.value()
            )
            masks.append(mask)

        if self.outlier_bodypart_gbox.isChecked():
            mask = outlier_bodypart(
                self.pred_data_array,
                threshold=self.bodypart_spinbox.value()
            )
            masks.append(mask)

        if self.outlier_size_gbox.isChecked():
            mask = outlier_size(
                self.pred_data_array,
                canon_pose=self.canon_pose,
                min_ratio=self.min_size_spinbox.value(),
                max_ratio=self.max_size_spinbox.value()
            )
            masks.append(mask)

        if self.outlier_flicker_gbox.isChecked():
            mask = outlier_flicker(self.pred_data_array)
            masks.append(mask)

        if self.outlier_duplicate_gbox.isChecked():
            mask = outlier_duplicate(
                self.pred_data_array,
                bp_threshold=self.duplicate_bp_spinbox.value(),
                dist_threshold=self.duplicate_dist_spinbox.value()
            )
            masks.append(mask)

        if self.outlier_pose_gbox.isChecked():
            mask = outlier_pose(
                self.pred_data_array,
                angle_map_data=self.angle_map_data,
                quant_step=self.pose_step_spinbox.value(),
                min_samples=self.pose_sample_spinbox.value(),
            )
            masks.append(mask)
        if not masks:
            Loggerbox.warning(self, "No Detectors Enabled", "All outlier detectors are disabled.")
        else:
            combined = masks[0]
            if self.radio_or.isChecked():
                for m in masks[1:]:
                    combined = np.logical_or(combined, m)
            else:
                for m in masks[1:]:
                    combined = np.logical_and(combined, m)

            return combined

    def _build_outlier_confidence_gbox(self):
        gbox = QtWidgets.QGroupBox("Outlier Confidence")
        gbox.setCheckable(True)
        gbox.setChecked(True)
        layout = QHBoxLayout(gbox)

        label = QLabel("Confidence Threshold:")
        self.confidence_spinbox = QtWidgets.QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.0, 1.0)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setValue(0.5)
        self.confidence_spinbox.setDecimals(2)
        layout.addWidget(label)
        layout.addWidget(self.confidence_spinbox)
        return gbox

    def _build_outlier_bodypart_gbox(self):
        gbox = QtWidgets.QGroupBox("Outlier Bodypart")
        gbox.setCheckable(True)
        gbox.setChecked(False)
        layout = QHBoxLayout(gbox)

        label = QLabel("Bodypart Threshold:")
        self.bodypart_spinbox = QtWidgets.QSpinBox()
        self.bodypart_spinbox.setRange(0, 30)
        self.bodypart_spinbox.setValue(2)
        layout.addWidget(label)
        layout.addWidget(self.bodypart_spinbox)
        return gbox

    def _build_outlier_size_gbox(self):
        gbox = QtWidgets.QGroupBox("Outlier Size")
        gbox.setCheckable(True)
        gbox.setChecked(False)
        layout = QVBoxLayout(gbox)

        min_frame = QHBoxLayout()
        min_label = QLabel("Min Size Ratio:")
        self.min_size_spinbox = QtWidgets.QDoubleSpinBox()
        self.min_size_spinbox.setRange(0.0, 1.0)
        self.min_size_spinbox.setSingleStep(0.05)
        self.min_size_spinbox.setValue(0.5)
        self.min_size_spinbox.setDecimals(2)
        min_frame.addWidget(min_label)
        min_frame.addWidget(self.min_size_spinbox)
        layout.addLayout(min_frame)

        max_frame = QHBoxLayout()
        max_label = QLabel("Max Size Ratio:")
        self.max_size_spinbox = QtWidgets.QDoubleSpinBox()
        self.max_size_spinbox.setRange(1.0, 5.0)
        self.max_size_spinbox.setSingleStep(0.1)
        self.max_size_spinbox.setValue(2.0)
        self.max_size_spinbox.setDecimals(2)
        max_frame.addWidget(max_label)
        max_frame.addWidget(self.max_size_spinbox)
        layout.addLayout(max_frame)
        return gbox

    def _build_outlier_flicker_gbox(self):
        gbox = QtWidgets.QGroupBox("Outlier Flicker")
        gbox.setCheckable(True)
        gbox.setChecked(False)

        layout = QtWidgets.QVBoxLayout()
        gbox.setLayout(layout)
        gbox.setFixedHeight(15)

        return gbox

    def _build_outlier_duplicate_gbox(self):
        gbox = QtWidgets.QGroupBox("Outlier Duplicate")
        gbox.setCheckable(True)
        gbox.setChecked(False)
        layout = QVBoxLayout(gbox)

        label = QLabel("Duplcate Bodypart Threshold (ratio):")
        self.duplicate_bp_spinbox = QtWidgets.QDoubleSpinBox()
        self.duplicate_bp_spinbox.setRange(0.0, 1.0)
        self.duplicate_bp_spinbox.setSingleStep(0.1)
        self.duplicate_bp_spinbox.setValue(0.7)
        self.duplicate_bp_spinbox.setDecimals(2)
        layout.addWidget(label)
        layout.addWidget(self.duplicate_bp_spinbox)

        self.duplicate_dist_spinbox = Spinbox_With_Label("Distance Threshold:", (1, 100), 3)
        layout.addWidget(self.duplicate_dist_spinbox)

        return gbox

    def _build_outlier_pose_gbox(self):
        gbox = QtWidgets.QGroupBox("Outlier Pose")
        gbox.setCheckable(True)
        gbox.setChecked(False)
        layout = QVBoxLayout(gbox)

        step_frame = QHBoxLayout()
        step_label = QLabel("Quantization Step:")
        self.pose_step_spinbox = QtWidgets.QDoubleSpinBox()
        self.pose_step_spinbox.setRange(0.05, 2.0)
        self.pose_step_spinbox.setSingleStep(0.05)
        self.pose_step_spinbox.setValue(1.5)
        step_frame.addWidget(step_label)
        step_frame.addWidget(self.pose_step_spinbox)
        layout.addLayout(step_frame)

        sample_frame = QHBoxLayout()
        sample_label = QLabel("Min Samples:")
        self.pose_sample_spinbox = QtWidgets.QSpinBox()
        self.pose_sample_spinbox.setRange(2, 10)
        self.pose_sample_spinbox.setValue(2)
        sample_frame.addWidget(sample_label)
        sample_frame.addWidget(self.pose_sample_spinbox)
        layout.addLayout(sample_frame)

        return gbox

    def _build_or_and_radio_widget(self):
        logic_widget = QtWidgets.QWidget()
        layout = QHBoxLayout(logic_widget)
        layout.addWidget(QLabel("Combine:"))

        self.radio_or = QtWidgets.QRadioButton("OR")
        self.radio_and = QtWidgets.QRadioButton("AND")
        self.radio_or.setChecked(True)

        self.logic_button_group = QtWidgets.QButtonGroup()
        self.logic_button_group.addButton(self.radio_or)
        self.logic_button_group.addButton(self.radio_and)
        layout.addWidget(self.radio_or)
        layout.addWidget(self.radio_and)

        layout.addStretch()
        
        return logic_widget
