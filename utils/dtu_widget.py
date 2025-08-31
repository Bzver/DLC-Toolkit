from functools import partial
import numpy as np
import cv2

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QFrame, QDialog, QLabel, QLineEdit, QMessageBox
from PySide6.QtGui import QFont, QIntValidator, QPixmap, QImage
 
from typing import List, Optional

from utils.dtu_comp import Slider_With_Marks
from utils.dtu_plotter import DLC_Plotter
from utils.dtu_dataclass import Loaded_DLC_Data, Plot_Config
from . import dtu_track_edit as dute
from . import dtu_helper as duh

class Menu_Widget(QtWidgets.QMenuBar):
    def __init__(self, parent=None):
        super().__init__(parent)

    def add_menu_from_config(self, menu_config):
        """
        Adds menus and their actions based on a configuration dictionary.
        Args:
            menu_config (dict): A dictionary defining the menu structure.
            Example:
                "File": {
                    "display_name": "File",
                    "buttons": [
                        ("Load Video", load_video_function),
                        {
                            "submenu": "Import",
                            "display_name": "Import Data",
                            "items": [
                                ("From File", import_from_file_function),
                                ("From URL", import_from_url_function),
                                {
                                    "submenu": "Advanced",
                                    "items": [
                                        ("From Database", lambda: print("DB import")),
                                        ("From API", lambda: print("API import"))
                                    ]
                                }
                            ]
                        },
                        ("Exit", exit_function, {"checkable": False})
                    ]
                },
                "View": {
                    "display_name": "View",
                    "buttons": [
                        ("Show Axis", lambda: print("Toggle axis"), {"checkable": True, "checked": True}),
                        ("Fullscreen", lambda: print("Fullscreen"), {})
                    ]
                }
        """
        for menu_name, config in menu_config.items():
            display_name = config.get("display_name", menu_name)
            menu = self.addMenu(display_name)

            buttons = config.get("buttons", [])
            for item in buttons:
                self._add_menu_item(menu, item)

    def _add_menu_item(self, parent_menu, item):
        """Recursively adds an action or submenu to the given parent menu."""
        if isinstance(item, dict):
            # It's a submenu
            submenu_key = item.get("submenu")
            if not submenu_key:
                raise ValueError("Submenu dictionary must have 'submenu' key")
            submenu_display = item.get("display_name", submenu_key)
            submenu = parent_menu.addMenu(submenu_display)

            subitems = item.get("items", [])
            for subitem in subitems:
                self._add_menu_item(submenu, subitem)

        elif isinstance(item, (list, tuple)):
            if len(item) == 2:
                action_text, action_func = item
                options = {}
            elif len(item) == 3:
                action_text, action_func, options = item
            else:
                raise ValueError("Menu item must be tuple of length 2 (text, func) or 3 (text, func, options)")

            action = parent_menu.addAction(action_text)
            action.triggered.connect(action_func)

            if isinstance(options, dict):
                if options.get("checkable"):
                    action.setCheckable(True)
                    action.setChecked(options.get("checked", False))
        else:
            raise ValueError("Menu item must be a tuple or dict (submenu)")
        
###################################################################################################################################################

class Progress_Bar_Widget(QtWidgets.QWidget):
    frame_changed = Signal(int)
    HexColor = str

    def __init__(self):
        super().__init__()
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        
        self.layout = QHBoxLayout(self)

        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(20)
        self.play_button.clicked.connect(self.toggle_playback)

        self.progress_slider = Slider_With_Marks(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setTracking(True)
        self.progress_slider.sliderMoved.connect(self.handle_slider_move)
        self.progress_slider.frame_changed.connect(self.handle_slider_move)

        self.layout.addWidget(self.play_button)
        self.layout.addWidget(self.progress_slider)

        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(int(1000/50)) # ~50 FPS
        self.playback_timer.timeout.connect(self.advance_frame)

    def set_frame_category(self, category_name:str, frame_list:List[int], color:Optional[HexColor]="#183539", priority:int=0):
        """
        Public API to pass the slider mark properties

        Args:
            category_name (str): The name of the category to assign to the specified frames.
            frame_list (List[int]): A list of frame indices to be associated with the category.
            color (Optional[HexColor]): The hexadecimal color code (e.g., '#FF55A3') used to style the frames in this category
            priority (int): The rendering priority of the category. 
                The higher the priority, the more prominently the category will be displayed.

        """
        self.progress_slider.set_frame_category(category_name, frame_list, color, priority)

    def set_slider_range(self, total_frames:int):
        self.total_frames = total_frames
        self.progress_slider.setRange(0, self.total_frames - 1)
    
    def set_current_frame(self, frame_idx:int):
        self.current_frame = frame_idx
        self.progress_slider.setValue(self.current_frame)

    def handle_slider_move(self, value:int):
        self.current_frame = value
        self.frame_changed.emit(self.current_frame)

    def toggle_playback(self):
        if not self.is_playing:
            self._start_playback()
        else:
            self._stop_playback()
        
    def advance_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.set_current_frame(self.current_frame)
            self.frame_changed.emit(self.current_frame)
        else:
            self._stop_playback()
            
    def _start_playback(self):
        if self.playback_timer:
            self.is_playing = True
            self.play_button.setText("■")
            self.playback_timer.start()

    def _stop_playback(self):
        if self.playback_timer:
            self.is_playing = False
            self.play_button.setText("▶")
            self.playback_timer.stop()

###################################################################################################################################################

class Nav_Widget(QtWidgets.QWidget):
    """
    Custom collapsible navigation widget with collapse button beside the title.
    Built for PySide6.
    """
    frame_changed_sig = Signal(int)
    prev_marked_frame_sig = Signal()
    next_marked_frame_sig = Signal()

    def __init__(self, mark_name="Marked", parent=None):
        super().__init__(parent)
        self.marked_name = mark_name
        self.collapsed = False

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Header bar (acts like a styled group box title)
        self.header_frame = QFrame()
        self.header_frame.setStyleSheet("""
            QFrame {
                background-color: #d3d7cf;
                border: 1px solid #a0a0a0;
                border-radius: 4px;
            }
        """)
        self.header_layout = QHBoxLayout(self.header_frame)
        self.header_layout.setContentsMargins(6, 4, 6, 4)
        self.header_layout.setSpacing(6)

        # Toggle button (triangle arrow)
        self.toggle_button = QPushButton("▼")
        self.toggle_button.setFixedSize(16, 16)
        font = QFont("Arial", 8)
        font.setBold(True)
        self.toggle_button.setFont(font)
        self.toggle_button.clicked.connect(self._toggle_collapsed)

        # Title label
        self.title_label = QLabel("Video Navigation")
        self.title_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Assemble header
        self.header_layout.addWidget(self.toggle_button)
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()

        # Content frame (holds navigation buttons)
        self.content_frame = QFrame()
        self.content_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #a0a0a0;
                border-top: none;
                background-color: #f8f8f8;
            }
        """)
        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.setSpacing(6)
        self.content_layout.setContentsMargins(8, 6, 8, 6)

        self._create_buttons()

        # Add header and content to main layout
        self.main_layout.addWidget(self.header_frame)
        self.main_layout.addWidget(self.content_frame)

        # Smooth animation for collapse/expand
        self.animation = QPropertyAnimation(self.content_frame, b"maximumHeight")
        self.animation.setDuration(240)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

    def setTitle(self, title_text):
        self.title_label.setText(title_text)

    def setTitleColor(self, color_hex):
        self.title_label.setStyleSheet(f"color: {color_hex}; font-weight: bold;")

    def _create_buttons(self):
        """Create navigation buttons and arrange them in a grid-like layout."""
        self.prev_10_frames_button = QPushButton("Prev 10 Frames (Shift + ←)")
        self.prev_frame_button = QPushButton("Prev Frame (←)")
        self.prev_marked_frame_button = QPushButton(f"◄ Prev {self.marked_name} (↑)")

        row1 = QHBoxLayout()
        row1.addWidget(self.prev_10_frames_button)
        row1.addWidget(self.prev_frame_button)
        row1.addWidget(self.prev_marked_frame_button)

        self.next_10_frames_button = QPushButton("Next 10 Frames (Shift + →)")
        self.next_frame_button = QPushButton("Next Frame (→)")
        self.next_marked_frame_button = QPushButton(f"► Next {self.marked_name} (↓)")

        row2 = QHBoxLayout()
        row2.addWidget(self.next_10_frames_button)
        row2.addWidget(self.next_frame_button)
        row2.addWidget(self.next_marked_frame_button)

        # Add rows to content layout
        self.content_layout.addLayout(row1)
        self.content_layout.addLayout(row2)

        # Connect signals
        self.prev_10_frames_button.clicked.connect(lambda: self.frame_changed_sig.emit(-10))
        self.prev_frame_button.clicked.connect(lambda: self.frame_changed_sig.emit(-1))
        self.next_frame_button.clicked.connect(lambda: self.frame_changed_sig.emit(1))
        self.next_10_frames_button.clicked.connect(lambda: self.frame_changed_sig.emit(10))
        self.prev_marked_frame_button.clicked.connect(self.prev_marked_frame_sig.emit)
        self.next_marked_frame_button.clicked.connect(self.next_marked_frame_sig.emit)

    def _toggle_collapsed(self):
        """Toggle collapse/expand state with animation."""
        self.collapsed = not self.collapsed
        self.toggle_button.setText("►" if self.collapsed else "▼")

        self.animation.stop()

        if self.collapsed:
            start_height = self.content_frame.sizeHint().height()
            self.animation.setStartValue(start_height)
            self.animation.setEndValue(0)
            self.animation.start()
        else:
            self.content_frame.show()
            self.content_frame.setMaximumHeight(16777215)  # Large value instead of hiding

            height = self.content_frame.sizeHint().height()

            self.animation.setStartValue(0)
            self.animation.setEndValue(height)
            self.animation.start()

            def on_animation_finished():
                if not self.collapsed:
                    self.content_frame.setMaximumHeight(16777215)  # No limit
            self.animation.finished.connect(on_animation_finished, Qt.UniqueConnection)

    def set_collapsed(self, collapsed: bool):
        """Allow external code to collapse or expand the widget."""
        if collapsed != self.collapsed:
            self._toggle_collapsed()

###################################################################################################################################################

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

class Canonical_Pose_Dialog(QDialog):
    def __init__(self, dlc_data:Loaded_DLC_Data, canon_pose:np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Canonical Pose Viewer")
        self.setGeometry(200, 200, 600, 600)

        self.dlc_data = dlc_data
        self.canon_pose = canon_pose

        self.layout = QVBoxLayout(self)
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.draw_canonical_pose()

    def draw_canonical_pose(self):
        if self.canon_pose is None or self.dlc_data is None:
            self.image_label.setText("No canonical pose data available.")
            return

        img_height, img_width = 600, 600
        blank_image = np.full((img_height, img_width, 3), 255, dtype=np.uint8)
        
        min_x, min_y, max_x, max_y = duh.calculate_bbox(self.canon_pose[:, 0], self.canon_pose[:, 1])
        canon_len = max(max_y-min_y, max_x-min_x)

        zoom_factor = 600 // canon_len

        # Reshape canon_pose to be compatible with DLC_Plotter
        num_keypoints = self.canon_pose.shape[0]
        reshaped_pose = np.zeros((1, num_keypoints * 3))
        for i in range(num_keypoints):  # Center the pose
            reshaped_pose[0, i*3] = self.canon_pose[i, 0] * zoom_factor + img_width / 2
            reshaped_pose[0, i*3+1] = self.canon_pose[i, 1] * zoom_factor + img_height / 2
            reshaped_pose[0, i*3+2] = 1.0

        # Create a dummy dlc_data for the plotter to use the skeleton and keypoint names
        dummy_dlc_data = self.dlc_data
        dummy_dlc_data.instance_count = 1
        
        plot_config = Plot_Config(
            plot_opacity=1.0, point_size=6.0, confidence_cutoff=0.0, hide_text_labels=False, edit_mode=False)

        plotter = DLC_Plotter(
            dlc_data=dummy_dlc_data,
            current_frame_data=reshaped_pose,
            frame_cv2=blank_image,
            plot_config=plot_config,
        )
        plotted_image = plotter.plot_predictions()

        # Convert OpenCV image to QPixmap and display
        rgb_image = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)