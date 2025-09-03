from functools import partial

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QDialog, QLabel, QMessageBox

from typing import List

class Adjust_Property_Dialog(QDialog):
    property_changed = Signal(float)

    def __init__(self, property_name, property_val, range:tuple, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Adjust {property_name}")
        self.property_name = property_name
        self.property_val = float(property_val)
        self.range = range
        range_length = (self.range[1] - self.range[0])
        self.slider_mult = range_length / 100
        layout = QtWidgets.QVBoxLayout(self)

        self.property_input = QtWidgets.QDoubleSpinBox()
        self.property_input.setRange(self.range[0], self.range[1])
        self.property_input.setValue(self.property_val)
        self.property_input.setSingleStep(self.slider_mult)
        layout.addWidget(self.property_input)

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        initial_slider_value = int((self.property_val - self.range[0]) / self.slider_mult)
        initial_slider_value = max(0, min(100, initial_slider_value)) 
        self.slider.setValue(initial_slider_value)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self._slider_changed)
        self.property_input.valueChanged.connect(self._spinbox_changed)

    def _spinbox_changed(self, value:int):
        self.property_val = value
        slider_value = int((value - self.range[0]) / self.slider_mult)
        slider_value = max(0, min(100, slider_value))
        self.slider.setValue(slider_value)
        self.property_changed.emit(self.property_val)

    def _slider_changed(self, value:int):
        # Map slider (0–100) to actual value
        self.property_val = self.range[0] + value * self.slider_mult
        self.property_input.setValue(self.property_val)
        self.property_changed.emit(self.property_val )

###################################################################################################################################################

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