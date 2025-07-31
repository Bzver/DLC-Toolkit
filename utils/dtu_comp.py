from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QPushButton, QMenu, QToolButton, QFileDialog

class Menu_Comp(QtWidgets.QWidget):
    def __init__(self, parent, host_type="Unknown"):
        super().__init__()

        self.gui = parent
        self.host = host_type
        self.menu_layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.menu_layout)

        self._setup_file_menu()
        if self.host == "Extractor":
            self._setup_export_menu()
        
        if self.host == "Refiner":
            self._setup_refiner_menu()
            self._setup_pref_menu()
            self._setup_save_menu()

        self.menu_layout.addStretch(1)

    def _create_menu_button(self, button_text: str, menu: QMenu, alignment=Qt.AlignLeft):
        button = QToolButton()
        button.setText(button_text)
        button.setMenu(menu)
        button.setPopupMode(QToolButton.InstantPopup)
        self.menu_layout.addWidget(button, alignment=alignment)
    
    def _setup_file_menu(self):
        self.load_menu = QMenu("File", self.gui)
        self.load_video_action = self.load_menu.addAction("Load Video")
        self.load_prediction_action = self.load_menu.addAction("Load Config and Prediction")
        if self.host == "Extractor":
            self.load_workplace_action = self.load_menu.addAction("Load Workplace")

        self._create_menu_button("File", self.load_menu)

        self.load_video_action.triggered.connect(self.gui.load_video)
        self.load_prediction_action.triggered.connect(self.gui.load_prediction)
        if self.host == "Extractor":
            self.load_workplace_action.triggered.connect(self.gui.load_workplace)

    def _setup_export_menu(self):
        self.export_menu = QMenu("Export", self.gui)
        self.save_workspace_action = self.export_menu.addAction("Save the Current Workspace")
        self.save_to_dlc_action = self.export_menu.addAction("Export to DLC")
        self.export_to_refiner_action = self.export_menu.addAction("Export to Refiner")
        self.merge_data_action = self.export_menu.addAction("Merge with Existing Data")

        self._create_menu_button("Save", self.export_menu)

        self.save_workspace_action.triggered.connect(self.gui.save_workspace)
        self.save_to_dlc_action.triggered.connect(self.gui.save_to_dlc)
        self.export_to_refiner_action.triggered.connect(self.gui.export_to_refiner)
        self.merge_data_action.triggered.connect(self.gui.merge_data)

    def _setup_refiner_menu(self):
        self.refiner_menu = QMenu("Adv. Refine", self.gui)
        self.direct_keypoint_edit_action = self.refiner_menu.addAction("Direct Keypoint Edit (Q)")
        self.purge_inst_by_conf_action = self.refiner_menu.addAction("Delete All Track Below Set Confidence")
        self.interpolate_all_action = self.refiner_menu.addAction("Interpolate All Frames for One Inst")
        self.designate_no_mice_zone_action = self.refiner_menu.addAction("Remove All Prediction Inside Area")
        self.segment_auto_correct_action = self.refiner_menu.addAction("Segmental Auto Correct")

        self._create_menu_button("Adv. Refine", self.refiner_menu)

        self.purge_inst_by_conf_action.triggered.connect(self.gui.purge_inst_by_conf)
        self.interpolate_all_action.triggered.connect(self.gui.interpolate_all)
        self.segment_auto_correct_action.triggered.connect(self.gui.segment_auto_correct)
        self.designate_no_mice_zone_action.triggered.connect(self.gui.designate_no_mice_zone)

    def _setup_pref_menu(self):
        self.pref_menu = QMenu("Preference", self.gui)
        self.adjust_point_size_action = self.pref_menu.addAction("Adjust Point Size")
        self.adjust_plot_visibilty_action = self.pref_menu.addAction("Adjust Plot Visibility")

        self._create_menu_button("Preference", self.pref_menu)

        self.adjust_point_size_action.triggered.connect(self.gui.adjust_point_size)
        self.adjust_plot_visibilty_action.triggered.connect(self.gui.adjust_plot_opacity)

    def _setup_save_menu(self):
        self.save_menu = QMenu("Save", self.gui)
        self.save_prediction_action = self.save_menu.addAction("Save Prediction")
        self.save_prediction_as_csv_action = self.save_menu.addAction("Save Prediction Into CSV")

        self._create_menu_button("Save", self.save_menu)

        self.save_prediction_action.triggered.connect(self.gui.save_prediction)
        self.save_prediction_as_csv_action.triggered.connect(self.gui.save_prediction_as_csv)

###################################################################################################################################################

class Progress_Bar_Comp(QtWidgets.QWidget):
    frame_changed = Signal(int)
    request_total_frames = Signal()

    def __init__(self):
        super().__init__()

        self.progress_layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(self.progress_layout)

        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(40)
        self.progress_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal) # Use Qt.Orientation.Horizontal
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setTracking(True)

        self.progress_layout.addWidget(self.play_button)
        self.progress_layout.addWidget(self.progress_slider)
        
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_frame_for_autoplay)
        
        self.is_playing = False
        self._total_frames = 0

        self.progress_slider.sliderMoved.connect(self._handle_slider_moved)

        self.request_total_frames.emit()

    def set_slider_range(self, total_frames: int):
        self._total_frames = total_frames
        self.progress_slider.setRange(0, max(0, total_frames - 1))
        self.progress_slider.setValue(0)

    def set_current_frame(self, frame_idx: int):
        if 0 <= frame_idx < self._total_frames:
            self.progress_slider.setValue(frame_idx)
        else:
            if self.is_playing:
                self.toggle_playback()
            print(f"Warning: Attempted to set frame {frame_idx} which is out of range [0, {self._total_frames-1}]")

    def _handle_slider_moved(self, value: int):
        if self.is_playing:
            self.toggle_playback()
        self.frame_changed.emit(value)

    def _advance_frame_for_autoplay(self):
        current_frame = self.progress_slider.value()
        
        if current_frame < self._total_frames - 1:
            next_frame = current_frame + 1
            self.progress_slider.setValue(next_frame)
            self.frame_changed.emit(next_frame)
        else:
            self.toggle_playback()
            self.progress_slider.setValue(self._total_frames - 1)
            self.frame_changed.emit(self._total_frames - 1)

    def toggle_playback(self):
        if not self.is_playing:
            if self._total_frames == 0:
                print("Cannot play, no frames loaded.")
                return
            
            if self.progress_slider.value() >= self._total_frames - 1:
                self.set_current_frame(0)
                self.frame_changed.emit(0)

            self.playback_timer.start(int(1000 / 50))
            self.play_button.setText("■")
            self.is_playing = True
        else:
            self.playback_timer.stop()
            self.play_button.setText("▶")
            self.is_playing = False

###################################################################################################################################################

