from functools import partial

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QDialog, QLabel, QMessageBox, QSpinBox

from typing import List

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

class Clear_Mark_Dialog(QDialog):
    frame_category_to_clear = Signal(str)

    def __init__(self, frame_category:List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clear Frame Marks")

        layout = QVBoxLayout(self)
        button_frame = QHBoxLayout()
        for category_text in frame_category:
            button = QPushButton(category_text)
            button_frame.addWidget(button)
            button.clicked.connect(partial(self._on_button_clicked, category_text))
        layout.addLayout(button_frame)

    def _on_button_clicked(self, category_text:str):
        self.frame_category_to_clear.emit(category_text)
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
    def __init__(self, min, max, title, text, parent=None):
        super().__init__(parent)
        self.setLabelText(text)
        self.setMinimum(min)
        self.setMaximum(max)
        self.setCancelButtonText("Cancel")
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setValue(0)

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