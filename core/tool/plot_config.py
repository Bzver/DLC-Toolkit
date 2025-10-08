from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGroupBox, QDialog, QVBoxLayout, QCheckBox

from core.dataclass import Plot_Config

class Plot_Config_Menu(QGroupBox):
    config_changed = Signal(object)

    def __init__(self, plot_config:Plot_Config, skip_opacity=False, parent=None):
        super().__init__(parent)
        self.setTitle("Plot Config Menu")
        self.plot_config = plot_config

        layout = QVBoxLayout(self)
        self.conf_box = Adjust_Property_Box(
            property_name="Confidence Cutoff", property_val=self.plot_config.confidence_cutoff, range=(0.00, 1.00), parent=self)
        layout.addWidget(self.conf_box)
        self.conf_box.property_changed.connect(self._on_conf_change)

        self.ps_box = Adjust_Property_Box(
            property_name="Point Size", property_val=self.plot_config.point_size, range=(0.1, 10.0), parent=self)
        layout.addWidget(self.ps_box)
        self.ps_box.property_changed.connect(self._on_ps_change)

        if not skip_opacity:
            self.po_box = Adjust_Property_Box(
                property_name="Point Opacity", property_val=self.plot_config.plot_opacity, range=(0.00, 1.00), parent=self)
            layout.addWidget(self.po_box)
            self.po_box.property_changed.connect(self._on_po_change)
        
        self.label_vis = QCheckBox("Label Text Visibility")
        self.label_vis.setChecked(True)
        layout.addWidget(self.label_vis)
        self.label_vis.toggled.connect(self._on_vis_toggled)
        
    def _on_conf_change(self, val):
        self.plot_config.confidence_cutoff = val
        self.config_changed.emit(self.plot_config)

    def _on_ps_change(self, val):
        self.plot_config.point_size = val
        self.config_changed.emit(self.plot_config)

    def _on_po_change(self, val):
        self.plot_config.plot_opacity = val
        self.config_changed.emit(self.plot_config)

    def _on_vis_toggled(self, checked):
        self.plot_config.hide_text_labels = not checked
        self.config_changed.emit(self.plot_config)

class Adjust_Property_Dialog(QDialog):
    def __init__(self, property_name, property_val, range:tuple, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Adjust {property_name}")
        layout = QVBoxLayout(self)
        self.gb = Adjust_Property_Box(property_name, property_val, range, parent=self)
        layout.addWidget(self.gb)

class Adjust_Property_Box(QtWidgets.QGroupBox):
    property_changed = Signal(float)

    def __init__(self, property_name, property_val, range:tuple, parent=None):
        super().__init__(parent)
        self.setTitle(f"Adjust {property_name}")
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