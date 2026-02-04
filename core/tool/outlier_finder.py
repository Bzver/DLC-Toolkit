import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel, QStackedWidget, QCheckBox
from typing import Optional, Dict, List

from ui import Spinbox_With_Label
from utils.pose import (
    outlier_bodypart, outlier_confidence, outlier_duplicate, outlier_speed, outlier_envelop,
    outlier_size, outlier_rotation, outlier_bad_to_the_bone, outlier_flicker)
from utils.logger import Loggerbox


class Outlier_Finder(QGroupBox):
    mask_changed = Signal(object, bool)

    def __init__(self,
                pred_data_array:np.ndarray,
                frame_list:Optional[List[int]]=None,
                skele_list:Optional[List[List[str]]]=None,
                kp_to_idx:Optional[Dict[str, int]]=None,
                angle_map_data:Optional[Dict[str, int]]=None,
                parent=None):
        super().__init__(parent)
        self.setTitle("Outlier Finder")
        self.pred_data_array = pred_data_array
        self.frame_list = frame_list

        F, I, _ = self.pred_data_array.shape
        self.outliers = np.zeros((F, I), dtype=bool)
        
        layout = QVBoxLayout(self)

        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Instance", "Keypoint"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)

        self.frame_range_checkbox = QCheckBox("Within Marked Frames Only:")
        self.frame_range_checkbox.setChecked(False)
        if self.frame_list:
            self.frame_range_checkbox.setVisible(True)
        else:
            self.frame_range_checkbox.setVisible(False)

        mode_layout.addWidget(self.frame_range_checkbox)

        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        self.outlier_stack = QStackedWidget()
        self.instance_container = Outlier_Container(self.pred_data_array, skele_list, kp_to_idx, angle_map_data)
        self.keypoint_container = Outlier_Container_KP(self.pred_data_array, skele_list, kp_to_idx, angle_map_data)
        self.outlier_stack.addWidget(self.instance_container)
        self.outlier_stack.addWidget(self.keypoint_container)

        layout.addWidget(self.outlier_stack)

        button_frame = QHBoxLayout()
        preview_button = QPushButton("Preview Outliers")
        preview_button.clicked.connect(self._outlier_preview)
        delete_button = QPushButton("Delete Outliers")
        delete_button.clicked.connect(self._outlier_delete)
        button_frame.addWidget(preview_button)
        button_frame.addWidget(delete_button)
        layout.addLayout(button_frame)

    def _outlier_preview(self):
        self._get_outlier_mask()
        if self.outliers is not None:
            self.mask_changed.emit(self.outliers, False)

    def _outlier_delete(self):
        self._get_outlier_mask()
        if self.outliers is not None:
            self.mask_changed.emit(self.outliers, True)

    def _get_outlier_mask(self):
        current_index = self.outlier_stack.currentIndex()
        if current_index == 0:
            outlier_mask = self.instance_container.get_combined_mask()
        else:
            outlier_mask = self.keypoint_container.get_combined_mask()

        if outlier_mask is None:
            self.outliers = np.zeros_like(self.outliers, dtype=bool)
            return

        if self.frame_range_checkbox.isChecked() and self.frame_list:
            restricted_mask = np.zeros_like(outlier_mask, dtype=bool)
            restricted_mask[self.frame_list] = outlier_mask[self.frame_list]
            self.outliers = restricted_mask
        else:
            self.outliers = outlier_mask.copy()

    def _on_mode_changed(self, index: int):
        self.outlier_stack.setCurrentIndex(index)
        self._refresh_outliers_shape()

    def _refresh_outliers_shape(self):
        F, I, D = self.pred_data_array.shape
        if self.outlier_stack.currentIndex() == 0:
            self.outliers = np.zeros((F, I), dtype=bool)
        else:
            K = D // 3
            self.outliers = np.zeros((F, I, K), dtype=bool)


class Outlier_Container(QtWidgets.QWidget):
    def __init__(self,
                pred_data_array:np.ndarray,
                skele_list:Optional[List[List[str]]]=None,
                kp_to_idx:Optional[Dict[str, int]]=None,
                angle_map_data:Optional[Dict[str, int]]=None,
                parent=None):
        super().__init__(parent)
        self.pred_data_array = pred_data_array
        self.skele_list = skele_list
        self.kp_to_idx = kp_to_idx
        self.angle_map_data = angle_map_data

        self.skele_list_idx = []
        for kp_1, kp_2 in skele_list:
            kp_1_idx = kp_to_idx[kp_1]
            kp_2_idx = kp_to_idx[kp_2]
            self.skele_list_idx.append([kp_1_idx, kp_2_idx])

        self.oc_layout = QVBoxLayout(self)

        self.logic_widget = self._build_or_and_radio_widget()
        self.outlier_confidence_gbox = self._build_outlier_confidence_gbox()
        self.outlier_bodypart_gbox = self._build_outlier_bodypart_gbox()
        self.outlier_size_gbox = self._build_outlier_size_gbox()
        self.outlier_rotation_gbox = self._build_outlier_rotation_gbox()
        self.outlier_duplicate_gbox = self._build_outlier_duplicate_gbox()
        self.outlier_bone_gbox = self._build_outlier_bone_gbox()
        self.outlier_speed_gbox = self._build_outlier_speed_gbox()
        self.outlier_flicker_gbox = self._build_outlier_flicker_gbox()

        self.oc_layout.addWidget(self.logic_widget)
        self.oc_layout.addWidget(self.outlier_confidence_gbox)
        self.oc_layout.addWidget(self.outlier_bodypart_gbox)
        self.oc_layout.addWidget(self.outlier_size_gbox)
        self.oc_layout.addWidget(self.outlier_rotation_gbox)
        self.oc_layout.addWidget(self.outlier_duplicate_gbox)
        self.oc_layout.addWidget(self.outlier_bone_gbox)
        self.oc_layout.addWidget(self.outlier_speed_gbox)
        self.oc_layout.addWidget(self.outlier_flicker_gbox)

        if self.skele_list is None:
            self.outlier_bone_gbox.setEnabled(False)
        if self.angle_map_data is None:
            self.outlier_rotation_gbox.setEnabled(False)

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
                min_ratio=self.min_size_spinbox.value(),
                max_ratio=self.max_size_spinbox.value()
            )
            masks.append(mask)

        if self.outlier_rotation_gbox.isChecked():
            mask = outlier_rotation(
                self.pred_data_array,
                angle_map_data=self.angle_map_data,
                threshold_deg=self.rotation_angle_spinbox.value()
            )
            masks.append(mask)

        if self.outlier_duplicate_gbox.isChecked():
            mask = outlier_duplicate(
                self.pred_data_array,
                bp_threshold=self.duplicate_bp_spinbox.value(),
                dist_threshold=self.duplicate_dist_spinbox.value()
            )
            masks.append(mask)

        if self.outlier_bone_gbox.isChecked():
            mask = outlier_bad_to_the_bone(
                self.pred_data_array,
                skele_list=self.skele_list_idx,
                threshold_max=self.bone_length_spinbox.value(),
                ignored_bones=self._get_ignored_bones_indices()
            )
            masks.append(mask)

        if self.outlier_flicker_gbox.isChecked():
            mask = outlier_flicker(self.pred_data_array)
            masks.append(mask)

        if self.outlier_speed_gbox.isChecked():
            mask = outlier_envelop(
                self.pred_data_array,
                padding=self.speed_spinbox.value()
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
        gbox = QGroupBox("Outlier Confidence")
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
        gbox = QGroupBox("Outlier Bodypart")
        gbox.setCheckable(True)
        gbox.setChecked(False)
        layout = QHBoxLayout(gbox)

        self.bodypart_spinbox = Spinbox_With_Label("Bodypart Threshold:", (0, 30), 4)
        self.bodypart_spinbox.setToolTip("Minimum number of detected keypoints required for an instance to be considered valid.")
        layout.addWidget(self.bodypart_spinbox)
        return gbox

    def _build_outlier_size_gbox(self):
        gbox = QGroupBox("Outlier Size")
        gbox.setCheckable(True)
        gbox.setChecked(False)
        layout = QHBoxLayout(gbox)

        min_frame = QHBoxLayout()
        min_label = QLabel("Min Size Ratio:")
        min_label.setToolTip("Instances smaller than this ratio of the average pose size are considered too small.")
        self.min_size_spinbox = QtWidgets.QDoubleSpinBox()
        self.min_size_spinbox.setRange(0.0, 1.0)
        self.min_size_spinbox.setSingleStep(0.05)
        self.min_size_spinbox.setValue(0.5)
        self.min_size_spinbox.setDecimals(2)
        self.min_size_spinbox.setToolTip("Minimum allowed size relative to the dataset-wide average pose radius (e.g., 0.5 = half the average size).")
        min_frame.addWidget(min_label)
        min_frame.addWidget(self.min_size_spinbox)
        layout.addLayout(min_frame)

        max_frame = QHBoxLayout()
        max_label = QLabel("Max Size Ratio:")
        max_label.setToolTip("Instances larger than this ratio of the average pose size are considered too large.")
        self.max_size_spinbox = QtWidgets.QDoubleSpinBox()
        self.max_size_spinbox.setRange(1.0, 5.0)
        self.max_size_spinbox.setSingleStep(0.1)
        self.max_size_spinbox.setValue(2.0)
        self.max_size_spinbox.setDecimals(2)
        self.max_size_spinbox.setToolTip("Maximum allowed size relative to the dataset-wide average pose radius (e.g., 2.0 = twice the average size).")
        max_frame.addWidget(max_label)
        max_frame.addWidget(self.max_size_spinbox)
        layout.addLayout(max_frame)
        return gbox

    def _build_outlier_rotation_gbox(self):
        gbox = QGroupBox("Outlier Rotation")
        gbox.setCheckable(True)
        gbox.setChecked(False)
        layout = QHBoxLayout(gbox)

        self.rotation_angle_spinbox = Spinbox_With_Label("Angle Threshold (°):", (1, 180), 50)
        self.rotation_angle_spinbox.setToolTip("Minimum allowed angle (in degrees) between head-center and tail-center vectors.")
        layout.addWidget(self.rotation_angle_spinbox)
        layout.addStretch()
        return gbox

    def _build_outlier_duplicate_gbox(self):
        gbox = QGroupBox("Outlier Duplicate")
        gbox.setCheckable(True)
        gbox.setChecked(False)
        layout = QHBoxLayout(gbox)

        dbp_lframe = QHBoxLayout()
        label = QLabel("Bodypart Ratio:")
        label.setToolTip("Fraction of overlapping keypoints required to consider two instances as duplicates.")
        self.duplicate_bp_spinbox = QtWidgets.QDoubleSpinBox()
        self.duplicate_bp_spinbox.setRange(0.0, 1.0)
        self.duplicate_bp_spinbox.setSingleStep(0.1)
        self.duplicate_bp_spinbox.setValue(0.7)
        self.duplicate_bp_spinbox.setDecimals(2)
        self.duplicate_bp_spinbox.setToolTip("Threshold (0.0–1.0): if the number of close keypoints exceeds this fraction of the smaller instance’s valid keypoints, duplication is suspected.")
        dbp_lframe.addWidget(label)
        dbp_lframe.addWidget(self.duplicate_bp_spinbox)

        self.duplicate_dist_spinbox = Spinbox_With_Label("Distance Threshold:", (1, 100), 3)
        self.duplicate_dist_spinbox.setToolTip("Maximum distance (in pixels) between two keypoints to be considered 'close' for duplicate detection.")
        layout.addLayout(dbp_lframe)
        layout.addWidget(self.duplicate_dist_spinbox)

        return gbox

    def _build_outlier_bone_gbox(self):
        gbox = QGroupBox("Outlier Bone Length")
        gbox.setCheckable(True)
        gbox.setChecked(False)
        layout = QVBoxLayout(gbox)

        first_level = QHBoxLayout()

        label = QLabel("Bone Length Threshold:")
        label.setToolTip("Relative threshold for flagging abnormally long bones (e.g., 2.0 = bone > 2× typical length).")
        self.bone_length_spinbox = QtWidgets.QDoubleSpinBox()
        self.bone_length_spinbox.setRange(1.0, 50.0)
        self.bone_length_spinbox.setSingleStep(0.1)
        self.bone_length_spinbox.setValue(2.0)
        self.bone_length_spinbox.setDecimals(2)
        self.bone_length_spinbox.setToolTip("Flag a pose if any bone exceeds this multiple of its median length across the dataset.")
        first_level.addWidget(label)
        first_level.addWidget(self.bone_length_spinbox)

        ignore_label = QLabel("Ignore these bones:")
        first_level.addWidget(ignore_label)

        layout.addLayout(first_level)

        self.ignore_bones_list = QtWidgets.QListWidget()
        self.ignore_bones_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for [a, b] in self.skele_list:
            item = QtWidgets.QListWidgetItem(f"{a} — {b}")
            item.setData(Qt.ItemDataRole.UserRole, [a, b])
            self.ignore_bones_list.addItem(item)
        
        layout.addWidget(self.ignore_bones_list)

        return gbox

    def _get_ignored_bones_indices(self) -> List[List[int]]:
        ignored = []
        for i in range(self.ignore_bones_list.count()):
            item = self.ignore_bones_list.item(i)
            if item.isSelected():
                pair = item.data(Qt.ItemDataRole.UserRole)
                ignored.append(pair)
        ignored_idx = []
        for name1, name2 in ignored:
            idx1 = self.kp_to_idx[name1]
            idx2 = self.kp_to_idx[name2]
            ignored_idx.append([idx1, idx2])
        return ignored_idx

    def _build_outlier_speed_gbox(self):
        gbox = QGroupBox("Outlier Nested Pose")
        gbox.setCheckable(True)
        gbox.setChecked(False)
        layout = QHBoxLayout(gbox)

        self.speed_spinbox = Spinbox_With_Label("Padding (%):", (0, 100), 20)
        self.speed_spinbox.setToolTip("Allow smaller pose to extend beyond larger pose by this percentage (e.g., 20 = 120% of radius).")
        layout.addWidget(self.speed_spinbox)
        layout.addStretch()
        return gbox

    def _build_outlier_flicker_gbox(self):
        gbox = QtWidgets.QGroupBox("Outlier Flicker")
        gbox.setCheckable(True)
        gbox.setChecked(False)

        layout = QtWidgets.QVBoxLayout()
        no_option_label = QLabel("No option applicable for this method.")
        layout.addWidget(no_option_label)

        gbox.setLayout(layout)

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
        logic_widget.setMaximumHeight(50)
        
        return logic_widget


class Outlier_Container_KP(Outlier_Container):
    def __init__(self, pred_data_array, canon_pose = None, angle_map_data = None, parent=None):
        super().__init__(pred_data_array, canon_pose, angle_map_data, parent)

        self.outlier_duplicate_gbox.setVisible(False)
        self.outlier_bodypart_gbox.setVisible(False)
        self.outlier_size_gbox.setVisible(False)
        self.outlier_rotation_gbox.setVisible(False)
        self.outlier_flicker_gbox.setVisible(False)

    def _build_outlier_speed_gbox(self):
        gbox = QGroupBox("Outlier Jump")
        gbox.setCheckable(True)
        gbox.setChecked(False)
        layout = QHBoxLayout(gbox)

        self.speed_spinbox = Spinbox_With_Label("Max Jump (px):", (1, 500), 50)
        self.speed_spinbox.setToolTip(
            "Flag keypoints that move more than this distance relative to centroids (in pixels) between consecutive frames. This detects sudden jumps, not true speed."
        )
        layout.addWidget(self.speed_spinbox)
        layout.addStretch()
        return gbox

    def get_combined_mask(self) -> Optional[np.ndarray]:
        masks = []

        if self.outlier_confidence_gbox.isChecked():
            mask = outlier_confidence(
                self.pred_data_array,
                threshold=self.confidence_spinbox.value(),
                kp_mode=True
            )
            masks.append(mask)

        if self.outlier_speed_gbox.isChecked():
            mask = outlier_speed(
                self.pred_data_array,
                angle_map_data=self.angle_map_data,
                max_speed_px=self.speed_spinbox.value(),
                kp_mode=True
            )
            masks.append(mask)

        if self.outlier_bone_gbox.isChecked():
            mask = outlier_bad_to_the_bone(
                self.pred_data_array,
                skele_list=self.skele_list_idx,
                threshold_max=self.bone_length_spinbox.value(),
                ignored_bones=self._get_ignored_bones_indices(),
                kp_mode=True
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