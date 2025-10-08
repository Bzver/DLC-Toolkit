from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout

from typing import Callable

from .video_slider import Video_Slider_Widget

class Video_Player_Widget(QtWidgets.QWidget):
    def __init__(self, slider_callback:Callable[[int], None]):
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

        self.progress_widget = Video_Slider_Widget()
        self.video_display.addWidget(self.progress_widget)
        self.progress_widget.frame_changed.connect(slider_callback)

        self.video_side_panel_layout.addWidget(self.video_left_panel_widget)
        self.video_side_panel_layout.addLayout(self.video_display, 1)
        self.video_side_panel_layout.addWidget(self.video_right_panel_widget)

        self.video_bottom_panel_widget = QtWidgets.QWidget()
        self.video_bottom_panel_widget.setVisible(False)  # Hidden by default
        self.video_bottom_panel_layout = QVBoxLayout(self.video_right_panel_widget)
        self.video_bottom_panel_layout.setContentsMargins(0, 0, 0, 0)

        self.vid_layout.addLayout(self.video_side_panel_layout)
        self.vid_layout.addLayout(self.video_bottom_panel_layout)

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