from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QGroupBox
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
        self.slider_callback = slider_callback
        
        self.vid_layout = QVBoxLayout(self)
        self.video_side_panel_layout = QHBoxLayout()
        self.video_left_panel_widget = QtWidgets.QWidget()
        self.video_left_panel_widget.setVisible(False)
        self.video_left_panel_layout = QVBoxLayout(self.video_left_panel_widget)
        self.video_left_panel_layout.setContentsMargins(0, 0, 0, 0)

        self.video_right_panel_widget = QtWidgets.QWidget()
        self.video_right_panel_widget.setVisible(False)
        self.video_right_panel_layout = QVBoxLayout(self.video_right_panel_widget)
        self.video_right_panel_layout.setContentsMargins(0, 0, 0, 0)

        self.video_display = QVBoxLayout()
        self.display_stack = QtWidgets.QStackedWidget()
        self.video_display.addWidget(self.display_stack, 1)
        self._setup_display()

        self._setup_slider()

        self.video_side_panel_layout.addWidget(self.video_left_panel_widget)
        self.video_side_panel_layout.addLayout(self.video_display, 1)
        self.video_side_panel_layout.addWidget(self.video_right_panel_widget)

        self.video_bottom_panel_widget = QtWidgets.QWidget()
        self.video_bottom_panel_widget.setVisible(False)
        self.video_bottom_panel_layout = QVBoxLayout(self.video_bottom_panel_widget)
        self.video_bottom_panel_layout.setContentsMargins(0, 0, 0, 0)

        self.nav = Nav_Widget(nav_callback)
        self.set_bottom_panel_widget(self.nav)

        self.vid_layout.addLayout(self.video_side_panel_layout)
        self.vid_layout.addWidget(self.video_bottom_panel_widget)

    def set_current_frame(self, frame_idx:int):
        self.sld.set_current_frame(frame_idx)

    def set_total_frames(self, total_frames:int):
        self.sld.set_total_frames(total_frames)

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
        """Switch display to Graphics_View mode (labeler)."""
        if self.display_stack.count() > 1:
            old_widget = self.display_stack.widget(1)
            if old_widget != graphics_view:
                self.display_stack.removeWidget(old_widget)
                old_widget.deleteLater()

        self.display_stack.insertWidget(1, graphics_view)
        self.display_stack.setCurrentIndex(1)

    def swap_display_for_label(self):
        """Switch back to QLabel mode (viewer)."""
        self.display_stack.setCurrentIndex(0)

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

    def _setup_display(self):
        self.display = QtWidgets.QLabel("No video loaded")
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setStyleSheet("background-color: black; color: white;")
        self.display.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        
        self.display_stack.addWidget(self.display)
        self.display_stack.setCurrentWidget(self.display)

    def _setup_slider(self):
        self.sld = Video_Slider_Widget()
        self.video_display.addWidget(self.sld)
        self.sld.frame_changed.connect(self.slider_callback)

class Nav_Widget(QtWidgets.QWidget):
    HexColor = str

    def __init__(self, nav_callback:Nav_Callback, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.marked_name = "Marked"
        self.collapsed = True
        self.nvc = nav_callback

        self.header_layout = QHBoxLayout(self)
        self.header_layout.setContentsMargins(6, 4, 6, 4)
        self.header_layout.setSpacing(6)

        self.toggle_button = QPushButton("◀")
        self.toggle_button.setFixedSize(16, 16)
        self.toggle_button.setFont(QFont("Arial", 9, QFont.Bold))
        self.toggle_button.clicked.connect(self._toggle_collapsed)

        self.title_label = QLabel("Video Navigation")
        self.title_label.setFont(QFont("Arial", 10, QFont.Bold))

        self.control_btn_frame = self._build_control_btn_frame()
        self.control_btn_frame.setFixedHeight(26)
        self.control_btn_frame.setFixedWidth(140)

        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()
        self.header_layout.addWidget(self.control_btn_frame)
        self.header_layout.addWidget(self.toggle_button)

        self.ctrl_dialog = None

    def set_marked_list_name(self, list_name:str):
        self.marked_name = list_name

    def setTitle(self, title_text:str):
        self.title_label.setText(title_text)

    def setTitleColor(self, color_hex:HexColor):
        self.title_label.setStyleSheet(f"color: {color_hex}; font-weight: bold;")

    def _build_control_btn_frame(self):
        control_btn_frame = Nav_Control(abridged=True)
        control_btn_frame.frame_changed_sig.connect(self.nvc.change_frame_callback)
        control_btn_frame.prev_marked_frame_sig.connect(self.nvc.nav_prev_callback)
        control_btn_frame.next_marked_frame_sig.connect(self.nvc.nav_next_callback)
        return control_btn_frame

    def _update_dialog_buttons(self):
        if not self.ctrl_dialog:
            return
        buttons = self.ctrl_dialog.findChildren(QPushButton)
        for btn in buttons:
            if "Prev Marked" in btn.text() or "Next Marked" in btn.text():
                btn.setText(btn.text().replace("Marked", self.marked_name))

    def _show_control_dialog(self):
        self.control_btn_frame.setVisible(False)
        if self.ctrl_dialog is None:
            self.ctrl_dialog = QtWidgets.QDialog(self)
            self.ctrl_dialog.setWindowTitle("Navigation Control")
            nav_contol = Nav_Control(self.marked_name)
            nav_contol.frame_changed_sig.connect(self.nvc.change_frame_callback)
            nav_contol.prev_marked_frame_sig.connect(self.nvc.nav_prev_callback)
            nav_contol.next_marked_frame_sig.connect(self.nvc.nav_next_callback)
            dialog_layout = QVBoxLayout(self.ctrl_dialog)
            dialog_layout.addWidget(nav_contol)
            self.ctrl_dialog.finished.connect(self._on_dialog_finished)
        self.ctrl_dialog.show()
        self.ctrl_dialog.raise_()
        self.ctrl_dialog.activateWindow()
        self.collapsed = False
        self.toggle_button.setText("▲")

    def _on_dialog_finished(self):
        self.control_btn_frame.setVisible(True)
        self.ctrl_dialog = None
        self.collapsed = True
        self.toggle_button.setText("◀")
        
    def _toggle_collapsed(self):
        """Toggle collapse/expand by showing/hiding content."""
        self.collapsed = not self.collapsed
        if self.collapsed:
            self.toggle_button.setText("◀")
            self.ctrl_dialog.accept()
            self.ctrl_dialog = None
        else:
            self.toggle_button.setText("▲")
            self._show_control_dialog()

class Nav_Control(QGroupBox):
    frame_changed_sig = Signal(int)
    prev_marked_frame_sig = Signal()
    next_marked_frame_sig = Signal()

    def __init__(self, marked_name:str="Marked", abridged:bool=False, parent=None):
        super().__init__(parent)
        if abridged:
            self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
            self.setFlat(True) 
        
        self.marked_name = marked_name
        self.abridged = abridged
        self.btn_layout = self._create_buttons()

    def _create_buttons(self):
        self.prev_frame_button = QPushButton("←") if self.abridged else QPushButton("  ◄ Frame (←)  ")
        self.next_frame_button = QPushButton("→") if self.abridged else QPushButton("  ► Next Frame (→)  ")
        self.prev_marked_frame_button = QPushButton("⇤") if self.abridged else QPushButton(f"  ◄ Prev {self.marked_name} (↑)  ")
        self.next_marked_frame_button = QPushButton("⇥") if self.abridged else QPushButton(f"  ► Next {self.marked_name} (↓)  ")
        self.prev_10_frames_button = QPushButton("↞") if self.abridged else QPushButton("  ◄ Prev 10 (Shift + ←)  ")
        self.next_10_frames_button = QPushButton("↠") if self.abridged else QPushButton("  ► Next 10 (Shift + →)  ")

        if self.abridged:
            btn_layout = QHBoxLayout(self)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            btn_layout.setSpacing(1)
        else:
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