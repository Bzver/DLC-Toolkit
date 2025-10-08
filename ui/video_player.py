from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QHBoxLayout, QPushButton, QLabel
from PySide6.QtGui import QFont

from typing import Callable

from .video_slider import Video_Slider_Widget
from core.dataclass import Nav_Callback

class Video_Player_Widget(QtWidgets.QWidget):
    def __init__(self,
            slider_callback:Callable[[int], None],
            nav_callback:Nav_Callback,
            parent=None
            ):
        super().__init__(parent)
        self.vid_layout = QtWidgets.QVBoxLayout(self)

        self.video_side_panel_layout = QHBoxLayout()
        self.video_left_panel_widget = QtWidgets.QWidget()
        self.video_left_panel_widget.setVisible(False)  # Hidden by default
        self.video_left_panel_layout = QVBoxLayout(self.video_left_panel_widget)
        self.video_left_panel_layout.setContentsMargins(0, 0, 0, 0)

        self.video_right_panel_widget = QtWidgets.QWidget()
        self.video_right_panel_widget.setVisible(False)  # Hidden by default
        self.video_right_panel_layout = QVBoxLayout(self.video_right_panel_widget)
        self.video_right_panel_layout.setContentsMargins(0, 0, 0, 0)

        self.video_display = QVBoxLayout()
        self.display = QtWidgets.QLabel("No video loaded")
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setStyleSheet("background-color: black; color: white;")
        self.display.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.video_display.addWidget(self.display, 1)

        self.sld = Video_Slider_Widget()
        self.video_display.addWidget(self.sld)
        self.sld.frame_changed.connect(slider_callback)

        self.video_side_panel_layout.addWidget(self.video_left_panel_widget)
        self.video_side_panel_layout.addLayout(self.video_display, 1)
        self.video_side_panel_layout.addWidget(self.video_right_panel_widget)

        self.video_bottom_panel_widget = QtWidgets.QWidget()
        self.video_bottom_panel_widget.setVisible(False)  # Hidden by default
        self.video_bottom_panel_layout = QVBoxLayout(self.video_bottom_panel_widget)
        self.video_bottom_panel_layout.setContentsMargins(0, 0, 0, 0)

        self.nav = Nav_Widget(nav_callback)
        self.set_bottom_panel_widget(self.nav)

        self.vid_layout.addLayout(self.video_side_panel_layout)
        self.vid_layout.addWidget(self.video_bottom_panel_widget)

    def set_left_panel_widget(self, widget: QtWidgets.QWidget | None):
        # Remove existing widget if any
        if self.video_left_panel_layout.count() > 0:
            old_widget = self.video_left_panel_layout.takeAt(0).widget()
            if old_widget:
                old_widget.setParent(None)  # or deleteLater() if owned

        if widget is not None:
            self.video_left_panel_layout.addWidget(widget)
            self.video_left_panel_widget.setVisible(True)
        else:
            self.video_left_panel_widget.setVisible(False)

    def set_right_panel_widget(self, widget: QtWidgets.QWidget | None):
        if self.video_right_panel_layout.count() > 0:
            old_widget = self.video_right_panel_layout.takeAt(0).widget()
            if old_widget:
                old_widget.setParent(None)

        if widget is not None:
            self.video_right_panel_layout.addWidget(widget)
            self.video_right_panel_widget.setVisible(True)
        else:
            self.video_right_panel_widget.setVisible(False)
            
    def set_bottom_panel_widget(self, widget: QtWidgets.QWidget | None):
        if self.video_bottom_panel_layout.count() > 0:
            old_widget = self.video_bottom_panel_layout.takeAt(0).widget()
            if old_widget:
                old_widget.setParent(None)

        if widget is not None:
            self.video_bottom_panel_layout.addWidget(widget)
            self.video_bottom_panel_widget.setVisible(True)
        else:
            self.video_bottom_panel_widget.setVisible(False)

class Nav_Widget(QtWidgets.QWidget):
    HexColor = str

    def __init__(self, nav_callback:Nav_Callback, parent=None):
        super().__init__(parent)
        self.marked_name = "Marked"
        self.collapsed = True
        self.nvc = nav_callback

        self.header_layout = QHBoxLayout(self)
        self.header_layout.setContentsMargins(6, 4, 6, 4)
        self.header_layout.setSpacing(6)

        # Toggle button
        self.toggle_button = QPushButton("►")
        self.toggle_button.setFixedSize(16, 16)
        font = QFont("Arial", 8)
        font.setBold(True)
        self.toggle_button.setFont(font)
        self.toggle_button.clicked.connect(self._toggle_collapsed)

        # Title label
        self.title_label = QLabel("Video Navigation")
        self.title_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.header_layout.addWidget(self.toggle_button)
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()

        self.ctrl = None

    def set_marked_list_name(self, list_name:str):
        self.marked_name = list_name

    def setTitle(self, title_text:str):
        self.title_label.setText(title_text)

    def setTitleColor(self, color_hex:HexColor):
        self.title_label.setStyleSheet(f"color: {color_hex}; font-weight: bold;")

    def _update_dialog_buttons(self):
        if not self.ctrl:
            return
        buttons = self.ctrl.findChildren(QPushButton)
        for btn in buttons:
            if "Prev Marked" in btn.text() or "Next Marked" in btn.text():
                btn.setText(btn.text().replace("Marked", self.marked_name))

    def _show_control_dialog(self):
        if self.ctrl is None:
            self.ctrl = Nav_Control_Dialog(self.marked_name, self)
            self.ctrl.finished.connect(self._on_dialog_finished)
            self.ctrl.frame_changed_sig.connect(self.nvc.change_frame_callback)
            self.ctrl.prev_marked_frame_sig.connect(self.nvc.nav_prev_callback)
            self.ctrl.next_marked_frame_sig.connect(self.nvc.nav_next_callback)
        self.ctrl.show()
        self.ctrl.raise_()
        self.ctrl.activateWindow()
        self.collapsed = False
        self.toggle_button.setText("▲")

    def _on_dialog_finished(self):
        self.ctrl = None
        self.collapsed = True
        self.toggle_button.setText("►")
        
    def _toggle_collapsed(self):
        """Toggle collapse/expand by showing/hiding content."""
        self.collapsed = not self.collapsed
        if self.collapsed:
            self.toggle_button.setText("►")
            self.ctrl.accept()
            self.ctrl = None
        else:
            self.toggle_button.setText("▲")
            self._show_control_dialog()

class Nav_Control_Dialog(QtWidgets.QDialog):
    frame_changed_sig = Signal(int)
    prev_marked_frame_sig = Signal()
    next_marked_frame_sig = Signal()

    def __init__(self, marked_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Navigation Controls")
        self.setModal(False) 

        self.marked_name = marked_name
        self.btn_layout = self._create_buttons()

    def _create_buttons(self):
        self.prev_frame_button = QPushButton("  ◄ Frame (←)  ")
        self.next_frame_button = QPushButton("  ► Next Frame (→)  ")
        self.prev_marked_frame_button = QPushButton(f"  ◄ Prev {self.marked_name} (↑)  ")
        self.next_marked_frame_button = QPushButton(f"  ► Next {self.marked_name} (↓)  ")
        self.prev_10_frames_button = QPushButton("  ◄ Prev 10 (Shift + ←)  ")
        self.next_10_frames_button = QPushButton("  ► Next 10 (Shift + →)  ")

        btn_layout = QVBoxLayout(self)
        btn_layout.addWidget(self.prev_frame_button)
        btn_layout.addWidget(self.next_frame_button)
        btn_layout.addWidget(self.prev_marked_frame_button)
        btn_layout.addWidget(self.next_marked_frame_button)
        btn_layout.addWidget(self.prev_10_frames_button)
        btn_layout.addWidget(self.next_10_frames_button)

        # Connect signals
        self.prev_10_frames_button.clicked.connect(lambda: self.frame_changed_sig.emit(-10))
        self.prev_frame_button.clicked.connect(lambda: self.frame_changed_sig.emit(-1))
        self.next_frame_button.clicked.connect(lambda: self.frame_changed_sig.emit(1))
        self.next_10_frames_button.clicked.connect(lambda: self.frame_changed_sig.emit(10))
        self.prev_marked_frame_button.clicked.connect(self.prev_marked_frame_sig.emit)
        self.next_marked_frame_button.clicked.connect(self.next_marked_frame_sig.emit)

        return btn_layout