from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QLineEdit
from PySide6.QtGui import QFont,  QIntValidator

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
        self.slider_callback = slider_callback
        
        self.vid_layout = QVBoxLayout(self)
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

        self._setup_slider()

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

    def set_current_frame(self, frame_idx:int):
        self.nav.set_current_frame(frame_idx)
        self.sld.set_current_frame(frame_idx)

    def set_total_frames(self, total_frames:int):
        self.nav.set_total_frames(total_frames)
        self.sld.set_slider_range(total_frames)

    def set_left_panel_widget(self, widget: QtWidgets.QWidget | None):
        self.clear_layout(self.video_left_panel_layout)

        if widget is not None:
            self.video_left_panel_layout.addWidget(widget)
            self.video_left_panel_widget.setVisible(True)
        else:
            self.video_left_panel_widget.setVisible(False)

    def set_right_panel_widget(self, widget: QtWidgets.QWidget | None):
        self.clear_layout(self.video_right_panel_layout)

        if widget is not None:
            self.video_right_panel_layout.addWidget(widget)
            self.video_right_panel_widget.setVisible(True)
        else:
            self.video_right_panel_widget.setVisible(False)
            
    def set_bottom_panel_widget(self, widget: QtWidgets.QWidget | None):
        self.clear_layout(self.video_bottom_panel_layout)

        if widget is not None:
            self.video_bottom_panel_layout.addWidget(widget)
            self.video_bottom_panel_widget.setVisible(True)
        else:
            self.video_bottom_panel_widget.setVisible(False)

    def swap_display_for_graphics_view(self, graphics_view:QtWidgets.QWidget):
        self.clear_layout(self.video_display)

        self.video_display.addWidget(graphics_view)
        self._setup_slider()

    def clear_layout(self, layout: QGridLayout | QHBoxLayout | QVBoxLayout):
        while layout.count():
            item = layout.takeAt(0)

            if item.widget():
                widget = item.widget()
                widget.setParent(None)
                widget.deleteLater()

            elif item.layout(): # Recursively clear child layouts
                self.clear_layout(item.layout())
                item.layout().deleteLater()

    def _setup_slider(self):
        self.sld = Video_Slider_Widget()
        self.video_display.addWidget(self.sld)
        self.sld.frame_changed.connect(self.slider_callback)

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

        font = QFont("Arial", 8)
        font.setBold(True)
        self.toggle_button = QPushButton("►")
        self.toggle_button.setFixedSize(16, 16)
        self.toggle_button.setFont(font)
        self.toggle_button.clicked.connect(self._toggle_collapsed)

        self.title_label = QLabel("Video Navigation")
        self.title_label.setFont(QFont("Arial", 9, QFont.Bold))

        self.fin = Nav_Frame_Input()
        self.fin.frame_changed_sig.connect(self.nvc.change_frame_callback)

        self.header_layout.addWidget(self.toggle_button)
        self.header_layout.addLayout(self.fin)
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()

        self.ctrl = None

    def set_current_frame(self, frame_idx:int):
        self.fin.set_current_frame(frame_idx)

    def set_total_frames(self, total_frames:int):
        self.fin.set_total_frames(total_frames)

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

class Nav_Frame_Input(QHBoxLayout):
    frame_changed_sig = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_frame = 1

        self.frame_input = QLineEdit("0")
        validator = QIntValidator(0, 2147483647, self)
        self.frame_input.setValidator(validator)
        self.frame_input.textChanged.connect(self._on_frame_idx_input)

        separator = QLabel("/")
        self.total_line = QLineEdit("0")
        self.total_line.setReadOnly(True)

        self.frame_input.setFixedWidth(60)
        separator.setFixedWidth(10)
        self.total_line.setFixedWidth(60)

        self.addWidget(self.frame_input)
        self.addWidget(separator)
        self.addWidget(self.total_line)

    def set_current_frame(self, frame_idx:int):
        self.frame_input.blockSignals(True)
        self.frame_input.setText(str(frame_idx))
        self.frame_input.blockSignals(False)

    def set_total_frames(self, total_frames:int):
        self.total_line.setText(str(total_frames))
        self.max_frame = total_frames - 1

    def _on_frame_idx_input(self):
        frame_idx = int(self.frame_input.text())
        if frame_idx > self.max_frame:
            self.frame_input.setText(str(self.max_frame))

        self.frame_changed_sig.emit(0, frame_idx)

class Nav_Control_Dialog(QtWidgets.QDialog):
    frame_changed_sig = Signal(int)
    prev_marked_frame_sig = Signal()
    next_marked_frame_sig = Signal()

    def __init__(self, marked_name:str, parent=None):
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