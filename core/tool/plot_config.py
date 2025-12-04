from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGroupBox, QDialog, QVBoxLayout

from ui import Toggle_Switch
from utils.dataclass import Plot_Config

class Plot_Config_Menu(QGroupBox):
    config_changed = Signal(object)

    def __init__(self, plot_config:Plot_Config, label_mode=False, parent=None):
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

        self.po_box = Adjust_Property_Box(
            property_name="Point Opacity", property_val=self.plot_config.plot_opacity, range=(0.00, 1.00), parent=self)
        self.po_box.property_changed.connect(self._on_po_change)
        self.po_box.setVisible(False)

        self.auto_snap = Toggle_Switch("Snap to Animals (E)", gbox=True, parent=self)
        self.auto_snap.toggled.connect(self._on_auto_snap_toggle)
        self.auto_snap.setVisible(False)

        self.roi_nav = Toggle_Switch("Navigate ROI Frames", gbox=True, parent=self)
        self.roi_nav.toggled.connect(self._on_roi_nav_toggle)
        self.roi_nav.setVisible(False)

        self.label_plot = Toggle_Switch("Plot Labeled Data", gbox=True, parent=self)
        self.label_plot.toggled.connect(self._on_label_plot_toggle)
        self.label_plot.setVisible(False)

        self.pred_plot = Toggle_Switch("Plot Prediction Data", gbox=True, parent=self)
        self.pred_plot.toggled.connect(self._on_pred_plot_toggle)
        self.pred_plot.setVisible(False)

        self.label_nav = Toggle_Switch("Navigate Labeled Frames", gbox=True, parent=self)
        self.label_nav.toggled.connect(self._on_label_nav_toggle)
        self.label_nav.setVisible(False)
        
        self.text_vis = Toggle_Switch("Label Text Visibility", gbox=True, parent=self)
        self.text_vis.toggled.connect(self._on_text_vis_toggle)

        if label_mode:
            layout.addWidget(self.po_box)
            self.po_box.setVisible(True)
            layout.addWidget(self.auto_snap)
            self.auto_snap.setVisible(True)
            layout.addWidget(self.roi_nav)
            self.roi_nav.setVisible(True)
        else:
            layout.addWidget(self.label_plot)
            self.label_plot.setVisible(True)
            layout.addWidget(self.pred_plot)
            self.pred_plot.setVisible(True)
            layout.addWidget(self.label_nav)
            self.label_nav.setVisible(True)

        layout.addWidget(self.text_vis)

        self.refresh_toggle_state()

    def refresh_toggle_state(self):
        self.auto_snap.set_checked(self.plot_config.auto_snapping)
        self.roi_nav.set_checked(self.plot_config.navigate_roi)
        self.label_plot.set_checked(self.plot_config.plot_labeled)
        self.pred_plot.set_checked(self.plot_config.plot_pred)
        self.label_nav.set_checked(self.plot_config.navigate_labeled)
        self.text_vis.set_checked(not self.plot_config.hide_text_labels)
        
    def _on_conf_change(self, val):
        self.plot_config.confidence_cutoff = val
        self.config_changed.emit(self.plot_config)

    def _on_ps_change(self, val):
        self.plot_config.point_size = val
        self.config_changed.emit(self.plot_config)

    def _on_po_change(self, val):
        self.plot_config.plot_opacity = val
        self.config_changed.emit(self.plot_config)

    def _on_auto_snap_toggle(self, checked):
        self.plot_config.auto_snapping = checked
        self.config_changed.emit(self.plot_config)

    def _on_roi_nav_toggle(self, checked):
        self.plot_config.navigate_roi = checked
        self.config_changed.emit(self.plot_config)

    def _on_label_plot_toggle(self, checked):
        self.plot_config.plot_labeled = checked
        self.config_changed.emit(self.plot_config)

    def _on_pred_plot_toggle(self, checked):
        self.plot_config.plot_pred = checked
        self.config_changed.emit(self.plot_config)

    def _on_label_nav_toggle(self, checked):
        self.plot_config.navigate_labeled = checked
        self.config_changed.emit(self.plot_config)

    def _on_text_vis_toggle(self, checked):
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