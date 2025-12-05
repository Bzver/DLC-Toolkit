import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QPushButton, QHBoxLayout, QVBoxLayout, QDialog, QLabel, QDialogButtonBox, QSpinBox, QDoubleSpinBox, QCheckBox, QSizePolicy)

from typing import List, Dict, Tuple

from .component import Spinbox_With_Label
from utils.logger import Loggerbox

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


class Frame_List_Dialog(QDialog):
    frame_indices_acquired = Signal(list)
    categories_selected = Signal(list)

    def __init__(self, frame_categories: Dict[str, Tuple[str, List[int]]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Frame Categories")
        self.frame_categories = frame_categories

        self.checkboxes: Dict[str, QCheckBox] = {}
        main_layout = QVBoxLayout(self)

        scroll_area = QtWidgets.QScrollArea()
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
            Loggerbox.warning(self, "Invalid Selection", "Head and tail cannot be the same bodypart.")
            return
        self.accept()

    def get_selected_indices(self):
        return self.head_idx, self.tail_idx
    

class Inference_interval_Dialog(QDialog):
    intervals_selected = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Inference Intervals")
        self.setMinimumWidth(300)

        main_layout = QVBoxLayout(self)

        self.interval_widgets = {}
        categories = {
            "No Animals (0)": "interval_0_animal",
            "One Animal (1)": "interval_1_animal",
            "Multiple Animals (2+)": "interval_n_animals",
            "Animal Close Together": "interval_merged"
        }

        for label_text, key in categories.items():
            spin_box = Spinbox_With_Label(label_text, (1,1000), 1)
            main_layout.addWidget(spin_box)
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
            self.range_selected.emit((start_idx, end_idx))
            self.accept()
        else:
            Loggerbox(self, "Invalid Parameters", f"End frame ({end_idx}) cannot be lower than start frame ({start_idx}).")


class Track_Fix_Dialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input Parameters For Track Fixing")
        self.setMinimumSize(500, 400)
        self.speeds_flat = None
        
        main_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Speed Distribution (px/frame)", fontsize=10)
        self.ax.set_xlabel("Speed (px/frame)")
        self.ax.set_ylabel("Density")
        self.ax.grid(True, alpha=0.3)
        self.canvas.setStyleSheet("background-color:white; border:1px solid #ccc;")
        main_layout.addWidget(self.canvas)
        
        max_dist_layout = QHBoxLayout()
        max_dist_label = QLabel("Max Distance:")
        self.max_dist_spinbox = QDoubleSpinBox()
        self.max_dist_spinbox.setRange(1.0, 100.0)
        self.max_dist_spinbox.setSingleStep(0.1)
        self.max_dist_spinbox.setValue(25.0)
        self.max_dist_spinbox.valueChanged.connect(self._max_dist_changed)
        max_dist_layout.addWidget(max_dist_label)
        max_dist_layout.addWidget(self.max_dist_spinbox)
        
        lookback_layout = QHBoxLayout()
        lookback_label = QLabel("Lookback Window:")
        self.lookback_spinbox = QSpinBox()
        self.lookback_spinbox.setRange(2, 1000)
        self.lookback_spinbox.setValue(11)
        lookback_layout.addWidget(lookback_label)
        lookback_layout.addWidget(self.lookback_spinbox)
        
        main_layout.addLayout(max_dist_layout)
        main_layout.addLayout(lookback_layout)
        
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        
        self.setLayout(main_layout)
    
    def set_histogram(self, speeds_flat:np.ndarray, max_dist_px_frame: float = None):
        self.speeds_flat = speeds_flat
        self._plot_histogram(max_dist_px_frame)
    
    def _max_dist_changed(self, max_dist_px_frame:float):
        self._plot_histogram(max_dist_px_frame)

    def _plot_histogram(self, max_dist_px_frame):
        self.ax.clear()
        self.ax.set_title("Speed Distribution (px/frame)", fontsize=10)
        self.ax.set_xlabel("Speed (px/frame)")
        self.ax.set_ylabel("Density")
        self.ax.grid(True, alpha=0.3)
        
        if self.speeds_flat is None or len(self.speeds_flat) == 0:
            self.ax.text(0.5, 0.5, 'No speed data available', 
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=12, color='gray')
            self.canvas.draw()
            return
        
        if len(self.speeds_flat) > 0:
            
            p95 = np.percentile(self.speeds_flat, 95)
            p99 = np.percentile(self.speeds_flat, 99)
            median = np.median(self.speeds_flat)

            speeds_filtered = self.speeds_flat[self.speeds_flat<=p99*1.2]

            self.ax.hist(speeds_filtered, bins=50, density=True, 
                        alpha=0.7, color='steelblue', edgecolor='white')
            
            self.ax.axvline(median, color='blue', linestyle='--', label=f'Median: {median:.1f}')
            self.ax.axvline(p95, color='green', linestyle='--', label=f'95th %: {p95:.1f}')
            self.ax.axvline(p99, color='orange', linestyle='--', label=f'99th %: {p99:.1f}')

            if max_dist_px_frame is not None:
                self.ax.axvline(max_dist_px_frame, color='red', linewidth=2,
                               label=f'Current max_dist: {max_dist_px_frame:.1f}')
            
            self.ax.set_xlim(0, p99*1.2)
            self.ax.legend(fontsize=8)
        
        self.canvas.draw()

    def get_values(self):
        return (self.max_dist_spinbox.value(), self.lookback_spinbox.value())


class Frame_Display_Dialog(QDialog):
    def __init__(self, title:str, image:QtGui.QImage, parent=None):
        super().__init__(parent)

        self.setWindowTitle(title)
        dialog_layout = QVBoxLayout(self)

        label = QLabel()
        label.setPixmap(QtGui.QPixmap.fromImage(image))
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(False)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(label)
        scroll_area.setWidgetResizable(True)

        dialog_layout.addWidget(scroll_area)
        self.showMaximized()