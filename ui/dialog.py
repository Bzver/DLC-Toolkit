from functools import partial
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QPushButton, QHBoxLayout, QVBoxLayout, QDial, QDialog,
    QLabel, QDialogButtonBox, QCheckBox, QSizePolicy, QScrollArea, QComboBox)

from typing import List, Dict, Tuple, Optional

from .component import Spinbox_With_Label
from .menu_shortcut import Shortcut_Manager
from utils.logger import Loggerbox


class Pose_Rotation_Dialog(QDialog):
    rotation_changed = Signal(int, float)  # (selected_instance_idx, angle_delta)

    def __init__(self, selected_instance_idx: int, initial_angle_deg:float=0.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Rotate Instance {selected_instance_idx}")
        self.selected_instance_idx = selected_instance_idx
        self.base_angle = initial_angle_deg - 90.0

        layout = QVBoxLayout(self)

        self.angle_label = QLabel(f"Angle: {self.base_angle:.1f}°")
        layout.addWidget(self.angle_label)

        self.dial = QDial()
        self.dial.setRange(0, 360)
        self.dial.setValue(self.base_angle)
        self.dial.setWrapping(True)
        self.dial.setNotchesVisible(True)
        layout.addWidget(self.dial)

        self.dial.valueChanged.connect(self._on_dial_change)

        self.setLayout(layout)
        self.resize(150, 150)

    def _on_dial_change(self, value:int):
        self.angle = float(value)
        angle_delta = self.angle - self.base_angle
        if abs(angle_delta) < 1e-1:
            return  # Skip tiny changes
        self.angle_label.setText(f"Angle: {self.angle:.1f}°")
        self.rotation_changed.emit(self.selected_instance_idx, angle_delta)
        self.base_angle = self.angle

    def get_angle(self) -> float:
        return self.angle

    def set_angle(self, angle:float):
        clamped_angle = angle % 360.0
        self.dial.setValue(int(clamped_angle))
        self.angle = clamped_angle
        self.angle_label.setText(f"Angle: {self.angle:.1f}°")


class Frame_List_Dialog(QDialog):
    frame_indices_acquired = Signal(list)
    categories_selected = Signal(list)

    def __init__(self, frame_categories: Dict[str, Tuple[str, List[int]]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Frame Categories")
        self.frame_categories = frame_categories

        self.checkboxes: Dict[str, QCheckBox] = {}
        main_layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setAlignment(Qt.AlignTop)

        for label, (_, indices) in self.frame_categories.items():
            count = len(indices)
            checkbox = QCheckBox(f"{label} — ({count} frames)")
            checkbox.setObjectName(label)
            self.checkboxes[label] = checkbox
            scroll_layout.addWidget(checkbox)

        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")

        self.ok_btn.clicked.connect(self._on_ok)
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)

        main_layout.addLayout(button_layout)

    def _on_ok(self):
        selected_categories = []
        combined_indices = []

        for label, checkbox in self.checkboxes.items():
            if checkbox.isChecked():
                cat, indices = self.frame_categories[label]
                selected_categories.append(cat)
                combined_indices.extend(indices)

        combined_indices = sorted(set(combined_indices))
        self.frame_indices_acquired.emit(combined_indices)
        self.categories_selected.emit(selected_categories)
        self.accept()


class Head_Tail_Dialog(QDialog):
    def __init__(self, keypoints, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Head and Tail Keypoints")
        self.keypoints = keypoints
        self.head_idx, self.tail_idx = None, None

        layout = QVBoxLayout(self)

        head_label = QLabel("Select Head Keypoint:")
        self.head_combo = QComboBox()
        self.head_combo.addItems(self.keypoints)
        layout.addWidget(head_label)
        layout.addWidget(self.head_combo)

        tail_label = QLabel("Select Tail Keypoint:")
        self.tail_combo = QComboBox()
        self.tail_combo.addItems(self.keypoints)
        layout.addWidget(tail_label)
        layout.addWidget(self.tail_combo)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def on_accept(self):
        self.head_idx = self.head_combo.currentIndex()
        self.tail_idx = self.tail_combo.currentIndex()
        if self.head_idx == self.tail_idx:
            Loggerbox.warning(self, "Invalid Selection", "Head and tail cannot be the same bodypart.")
            return
        self.accept()

    def get_selected_indices(self):
        return self.head_idx, self.tail_idx
    

class Inference_interval_Dialog(QDialog):
    intervals_selected = Signal(dict, bool)
    CATEGORIES = {
        "No Animals (0)": "interval_0_animal",
        "One Animal (1)": "interval_1_animal",
        "Multiple Animals (2+)": "interval_n_animals",
        "Animal Close Together": "interval_merged"
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Inference Intervals")
        self.setMinimumWidth(300)

        main_layout = QVBoxLayout(self)

        self.interval_widgets = {}

        for label_text, key in self.CATEGORIES.items():
            spin_box = Spinbox_With_Label(label_text, (1,1000), 1)
            main_layout.addWidget(spin_box)
            self.interval_widgets[key] = spin_box


        self.skip_existing = QCheckBox("Skip Inferenced Frames")
        self.skip_existing.setChecked(False)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")

        ok_button.clicked.connect(self._accept_input)
        cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.skip_existing)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)

    def _accept_input(self):
        intervals = {key: widget.value() for key, widget in self.interval_widgets.items()}
        self.intervals_selected.emit(intervals, self.skip_existing.isChecked())
        self.accept()


class Frame_Range_Dialog(QDialog):
    range_selected = Signal(tuple)

    def __init__(self, total_frames:int, parent=None):
        super().__init__(parent)
        self.total_frames = total_frames

        self.setWindowTitle("Set Frame Range")
        self.setMinimumWidth(300)

        main_layout = QVBoxLayout(self)

        self.start_spin = Spinbox_With_Label("Start:", (0, self.total_frames-1), 0)
        self.end_spin = Spinbox_With_Label("End:", (0, self.total_frames-1), self.total_frames-1)
        main_layout.addWidget(self.start_spin)
        main_layout.addWidget(self.end_spin)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")

        ok_button.clicked.connect(self._accept_input)
        cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)

    def _accept_input(self):
        start_idx = self.start_spin.value()
        end_idx = self.end_spin.value()
        if end_idx >= start_idx:
            self.hide()
            self.range_selected.emit((start_idx, end_idx))
            self.accept()
        else:
            Loggerbox(self, "Invalid Parameters", f"End frame ({end_idx}) cannot be lower than start frame ({start_idx}).")


class Frame_Display_Dialog(QDialog):
    def __init__(self, title:str, image:QtGui.QImage, parent=None):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.dialog_layout = QVBoxLayout(self)

        self.label = QLabel()
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(False)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.label)
        scroll_area.setWidgetResizable(True)

        self.dialog_layout.addWidget(scroll_area)
        self.showMaximized()


class Instance_Selection_Dialog(QDialog):
    inst_checked = Signal(int, bool)
    instances_selected = Signal(tuple)

    def __init__(self, inst_count:int, colormap:List[str], select_status:Optional[List[bool]]=None, dual_selection:bool=False, parent=None):
        super().__init__(parent)
        self.inst_count = inst_count
        self.colormap = colormap
        self.dual_selection = dual_selection

        if select_status is None or self.dual_selection:
            self.select_status = [False] * self.inst_count
        else:
            self.select_status = select_status

        self.setWindowTitle("Select Instance")
        layout = QHBoxLayout(self)

        self.buttons:List[QPushButton] = []
        self.shortcuts = Shortcut_Manager(self)
        sc_config = {}

        for inst_idx in range(self.inst_count):
            sc_config[inst_idx] = {"key": str(inst_idx+1), "callback": lambda idx=inst_idx: self._on_key_pressed(idx)}
            color = colormap[inst_idx % len(colormap)]
            status = self.select_status[inst_idx]
            btn = QPushButton(f"Inst {inst_idx+1}")
            btn.setStyleSheet(f"background-color: {color};")
            btn.setCheckable(True)
            btn.setChecked(status)
            btn.clicked.connect(partial(self._on_button_clicked, inst_idx))
            layout.addWidget(btn)
            self.buttons.append(btn)

        self.shortcuts.add_shortcuts_from_config(sc_config)

    def _on_button_clicked(self, idx: int):
        checked_status = self.buttons[idx].isChecked()
        self.select_status[idx] = checked_status

        if not self.dual_selection:
            self.inst_checked.emit(idx, checked_status)
            self.accept()
        elif sum(self.select_status) == 2:
            selected_indices = [i for i, x in enumerate(self.select_status) if x]
            self.instances_selected.emit(tuple(selected_indices))
            self.accept()

    def _on_key_pressed(self, idx: int):
        checked_status = self.buttons[idx].isChecked()
        self.buttons[idx].setChecked(not checked_status)