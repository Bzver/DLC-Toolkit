from functools import partial

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QDialog, QLabel, QMessageBox, QSpinBox

from typing import List, Dict
from time import time

class Pose_Rotation_Dialog(QDialog):
    rotation_changed = Signal(int, float)  # (selected_instance_idx, angle_delta)

    def __init__(self, selected_instance_idx: int, initial_angle_deg:float=0.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Rotate Instance {selected_instance_idx}")
        self.selected_instance_idx = selected_instance_idx
        self.base_angle = initial_angle_deg

        layout = QVBoxLayout(self)

        self.angle_label = QLabel(f"Angle: {self.base_angle:.1f}°")
        layout.addWidget(self.angle_label)

        self.dial = QtWidgets.QDial()
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
        if abs(angle_delta) < 1e-3:
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

###################################################################################################################################################

class Frame_List_Dialog(QDialog):
    frame_list_selected = Signal(str)
    frame_indices_acquired = Signal(list)

    def __init__(self, frame_categories:Dict[str, List[int]], indices_mode:bool=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Frame List")
        self.frame_categories = frame_categories
        self.indices_mode = indices_mode

        layout = QVBoxLayout(self)
        
        for label, frames in self.frame_categories.items():
            count = len(frames)
            btn = QPushButton(f"{label} ({count})")
            btn.clicked.connect(partial(self._on_button_clicked, label))
            layout.addWidget(btn)

    def _on_button_clicked(self, category_text:str):
        if self.indices_mode: 
            self.frame_indices_acquired.emit(self.frame_categories[category_text])
        else:
            self.frame_list_selected.emit(category_text)
        self.accept()

###################################################################################################################################################

class Head_Tail_Dialog(QtWidgets.QDialog):
    def __init__(self, keypoints, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Head and Tail Keypoints")
        self.keypoints = keypoints
        self.head_idx, self.tail_idx = None, None

        layout = QtWidgets.QVBoxLayout(self)

        head_label = QtWidgets.QLabel("Select Head Keypoint:")
        self.head_combo = QtWidgets.QComboBox()
        self.head_combo.addItems(self.keypoints)
        layout.addWidget(head_label)
        layout.addWidget(self.head_combo)

        tail_label = QtWidgets.QLabel("Select Tail Keypoint:")
        self.tail_combo = QtWidgets.QComboBox()
        self.tail_combo.addItems(self.keypoints)
        layout.addWidget(tail_label)
        layout.addWidget(self.tail_combo)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def on_accept(self):
        self.head_idx = self.head_combo.currentIndex()
        self.tail_idx = self.tail_combo.currentIndex()
        if self.head_idx == self.tail_idx:
            QMessageBox.warning(self, "Invalid Selection",
                "Head and tail cannot be the same bodypart.")
            return
        self.accept()

    def get_selected_indices(self):
        return self.head_idx, self.tail_idx
    
###################################################################################################################################################

class Progress_Indicator_Dialog(QtWidgets.QProgressDialog):
    def __init__(self, min_val, max_val, title, text, parent=None):
        super().__init__(parent)
        self.setLabelText(text)
        self.setMinimum(min_val)
        self.setMaximum(max_val)
        self.setCancelButtonText("Cancel")
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setValue(0)

        self._start_time = time()
        self._last_update_time = self._start_time
        self._last_value = min_val
        self._base_text = text

    def setValue(self, value):
        super().setValue(value)

        if value <= self.minimum():
            return

        current_time = time()
        elapsed = current_time - self._start_time
        if elapsed <= 0:
            return

        delta_value = value - self._last_value
        delta_time = current_time - self._last_update_time
        if delta_time > 0:
            it_per_sec = delta_value / delta_time
        else:
            it_per_sec = 0

        total_items = self.maximum() - self.minimum()
        completed_items = value - self.minimum()
        if it_per_sec > 0 and completed_items > 0:
            eta = (total_items - completed_items) / it_per_sec
        else:
            eta = float('inf')

        elapsed_str = self._format_time(elapsed)
        if eta == float('inf'):
            eta_str = "--:--:--"
        else:
            eta_str = self._format_time(eta)

        new_text = f"{self._base_text}\nElapsed: {elapsed_str} | Remaining: {eta_str} | {it_per_sec:.1f} it/s"
        self.setLabelText(new_text)

        self._last_update_time = current_time
        self._last_value = value

    def _format_time(self, seconds):
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60

        return f"{h:02d}:{m:02d}:{s:02d}"
    
###################################################################################################################################################

class Inference_interval_Dialog(QDialog):
    intervals_selected = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Inference Intervals")
        self.setMinimumWidth(300)

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # Labels and SpinBoxes for intervals
        self.interval_widgets = {}
        categories = {
            "No Animals (0)": "interval_0_animals",
            "One Animal (1)": "interval_1_animal",
            "Multiple Animals (2+)": "interval_n_animals"
        }

        for label_text, key in categories.items():
            h_layout = QHBoxLayout()
            label = QLabel(label_text)
            spin_box = QSpinBox()
            spin_box.setMinimum(1)
            spin_box.setMaximum(1000) # Arbitrary max, can be adjusted
            spin_box.setValue(1) # Default to 1 (every frame)
            h_layout.addWidget(label)
            h_layout.addStretch()
            h_layout.addWidget(spin_box)
            main_layout.addLayout(h_layout)
            self.interval_widgets[key] = spin_box

        # Buttons
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
        intervals = {key: widget.value() for key, widget in self.interval_widgets.items()}
        self.intervals_selected.emit(intervals)
        self.accept()