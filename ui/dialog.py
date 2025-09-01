from functools import partial

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QDialog, QLabel, QLineEdit, QMessageBox
from PySide6.QtGui import QIntValidator
 
from typing import List, Optional

from utils.dataclass import Loaded_DLC_Data
from utils import track_edit as dute

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

class Generate_Mark_Dialog(QDialog):
    clear_old = Signal(bool)
    frame_list_new = Signal(list)

    def __init__(self, total_frames:int, dlc_data:Optional[Loaded_DLC_Data]=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Automatic Mark Generation")
        self.total_frames = total_frames
        self.dlc_data = dlc_data
        
        layout = QVBoxLayout(self)

        mode_frame = QHBoxLayout()
        mode_label = QLabel("Mark Gneration Mode:")
        mode_frame.addWidget(mode_label)

        self.mode_option = QtWidgets.QComboBox()
        self.mode_option.addItems(["Random", "Stride"])
        
        if self.dlc_data is not None: # Only add "Low Score" if DLC data is available
            self.mode_option.addItem("Low Score")
        else:
            self.mode_option.setToolTip("Low Score mode requires loaded DLC data")

        self.mode_option.setCurrentIndex(0)
        self.mode_option.currentTextChanged.connect(self._on_selection_changed)
        mode_frame.addWidget(self.mode_option)

        layout.addLayout(mode_frame)

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

        self.random_container = self.build_random_container()
        self.stride_container = self.build_stride_container()
        self.lowscore_container = self.build_lowscore_container()

        layout.addWidget(self.random_container)
        layout.addWidget(self.stride_container)
        layout.addWidget(self.lowscore_container)

        self.stride_container.setEnabled(False)
        self.lowscore_container.setEnabled(False)

        confirm_frame = QHBoxLayout()
        self.keep_old_checkbox = QtWidgets.QCheckBox("Keep Existing Marks")
        self.keep_old_checkbox.setChecked(True)
        confirm_frame.addWidget(self.keep_old_checkbox)
    
        okay_button = QPushButton("Mark Frames")
        okay_button.clicked.connect(self.find_frames_to_mark)
        confirm_frame.addWidget(okay_button)
        layout.addLayout(confirm_frame)

    def build_random_container(self):
        container = QtWidgets.QGroupBox("Random Frame Extraction")

        layout = QHBoxLayout(container)
        label = QLabel("Number of Frames to Mark: ")
        layout.addWidget(label)

        self.random_textbox = QLineEdit()

        validator = QIntValidator(1, self.total_frames)
        self.random_textbox.setValidator(validator)
        layout.addWidget(self.random_textbox)

        return container

    def build_stride_container(self):
        container = QtWidgets.QGroupBox("Stride Frame Extraction")
        layout = QHBoxLayout(container)

        label = QLabel("Stride Interval:")
        self.stride_textbox = QLineEdit()
        self.stride_textbox.setPlaceholderText("e.g., 5")

        validator = QIntValidator(1, self.total_frames)
        self.stride_textbox.setValidator(validator)

        layout.addWidget(label)
        layout.addWidget(self.stride_textbox)
        return container

    def build_lowscore_container(self):
        container = QtWidgets.QGroupBox("Low Score Frame Extraction")
        layout = QVBoxLayout(container)

        logic_frame = QHBoxLayout()
        logic_label = QLabel("Mark frame if:")
        
        self.or_radio = QtWidgets.QRadioButton("Any condition matches (OR)")
        self.and_radio = QtWidgets.QRadioButton("All conditions match (AND)")
        
        self.or_radio.setChecked(True)

        logic_frame.addWidget(logic_label)
        logic_frame.addWidget(self.or_radio)
        logic_frame.addWidget(self.and_radio)
        
        self.logic_button_group = QtWidgets.QButtonGroup()
        self.logic_button_group.addButton(self.or_radio)
        self.logic_button_group.addButton(self.and_radio)

        layout.addLayout(logic_frame)

        layout.addWidget(QtWidgets.QFrame())  # visual spacer

        score_frame = QHBoxLayout()
        score_label = QLabel("Low Confidence Threshold:")
        self.lowscore_spinbox = QtWidgets.QDoubleSpinBox()
        self.lowscore_spinbox.setRange(0.0, 1.0)
        self.lowscore_spinbox.setSingleStep(0.1)
        self.lowscore_spinbox.setValue(0.2)
        self.lowscore_spinbox.setDecimals(2)
        score_frame.addWidget(score_label)
        score_frame.addWidget(self.lowscore_spinbox)
        layout.addLayout(score_frame)

        bodypart_frame = QHBoxLayout()
        bodypart_label = QLabel("Low Bodyparts Detection Threshold (%):")
        self.lowbodypart_spinbox = QtWidgets.QSpinBox()
        self.lowbodypart_spinbox.setRange(0, 100)
        self.lowbodypart_spinbox.setValue(20)
        self.lowbodypart_spinbox.setSuffix("%")
        bodypart_frame.addWidget(bodypart_label)
        bodypart_frame.addWidget(self.lowbodypart_spinbox)
        layout.addLayout(bodypart_frame)

        animal_frame = QHBoxLayout()
        animal_label = QLabel("Min Animal Instances to Accept:")
        self.lowanimal_spinbox = QtWidgets.QSpinBox()
        self.lowanimal_spinbox.setRange(1, 10)
        self.lowanimal_spinbox.setValue(1)
        animal_frame.addWidget(animal_label)
        animal_frame.addWidget(self.lowanimal_spinbox)
        layout.addLayout(animal_frame)

        return container
    
    def find_frames_to_mark(self):
        """Process input and emit the list of frames to mark."""
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

        elif mode == "Low Score":
            lowscore_threshold = self.lowscore_spinbox.value()
            lowbodypart_threshold = self.lowbodypart_spinbox.value() / 100.0
            lowanimal_threshold = self.lowanimal_spinbox.value()
            use_or_logic = self.or_radio.isChecked()

            selected_frames = dute.filter_by_conf_bp_instance(
                pred_data_array = self.dlc_data.pred_data_array,
                confidence_threshold = lowscore_threshold,
                bodypart_threshold = lowbodypart_threshold,
                instance_threshold = lowanimal_threshold,
                use_or = use_or_logic,
                return_frame_list = True
                )
            
        selected_frames.sort()
        self.clear_old.emit(not self.keep_old_checkbox.isChecked())
        self.frame_list_new.emit(selected_frames)
        self.accept()

    def _on_selection_changed(self):
        """Show the appropriate container based on selected mode."""
        mode = self.mode_option.currentText()

        self.random_container.setEnabled(False)
        self.stride_container.setEnabled(False)
        self.lowscore_container.setEnabled(False)

        if mode == "Random":
            self.random_container.setEnabled(True)
        elif mode == "Stride":
            self.stride_container.setEnabled(True)
        elif mode == "Low Score":
            self.lowscore_container.setEnabled(True)
        
        self.center()

    def center(self):
        """Center the dialog on screen."""
        geo = self.frameGeometry()
        center = self.screen().availableGeometry().center()
        geo.moveCenter(center)
        self.move(geo.topLeft())

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