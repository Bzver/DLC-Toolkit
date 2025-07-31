from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QPushButton, QMenu, QToolButton

class Menu_Comp:
    def __init__(self, parent, host_type:str):
        self.gui = parent
        self.host = host_type
        self.menu_layout = QtWidgets.QHBoxLayout()

        self._setup_file_menu()
        if self.host == "Extractor":
            self._setup_export_menu()
        
        if self.host == "Refiner":
            self._setup_refiner_menu()
            self._setup_pref_menu()
            self._setup_save_menu()

        self.menu_layout.addStretch(1)
        self.gui.layout.addLayout(self.menu_layout)

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

class Progress_Bar_Comp:
    def __init__(self, parent):
        self.gui = parent
        self.progress_layout = QtWidgets.QHBoxLayout()

        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(40) # Slightly wider button
        self.progress_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 0) # Will be set dynamically
        self.progress_slider.setTracking(True)

        self.progress_layout.addWidget(self.play_button)
        self.progress_layout.addWidget(self.progress_slider)
        self.playback_timer = QTimer(self.gui) # Pass GUI to QTimer
        self.playback_timer.timeout.connect(self.autoplay_video)
        self.is_playing = False
        self.gui.layout.addLayout(self.progress_layout)
        
        self.progress_slider.sliderMoved.connect(self.set_frame_from_slider)
        self.play_button.clicked.connect(self.toggle_playback)

    def set_slider_range(self, total_frames):
        self.progress_slider.setRange(0, total_frames - 1)
        self.progress_slider.setValue(0)

    def set_slider_value(self, value):
        self.progress_slider.setValue(value)

    def set_frame_from_slider(self, value):
        if hasattr(self.gui, "selected_cam_idx"):
            self.gui.current_frame_idx = None
        self.gui.current_frame_idx = value
        self.gui.display_current_frame()
        self.gui.navigation_box_title_controller()

    def autoplay_video(self):
        if not hasattr(self.gui, 'total_frames') or self.gui.total_frames <= 0:
                    self.playback_timer.stop()
                    self.play_button.setText("▶")
                    self.is_playing = False
                    return
        
        if self.gui.current_frame_idx is None:
            self.gui.current_frame_idx = 0

        if self.gui.current_frame_idx < self.gui.total_frames - 1:
            if hasattr(self.gui, "selected_cam_idx"):
                self.gui.selected_cam_idx = None
            self.gui.current_frame_idx += 1
            self.gui.display_current_frame()
            self.gui.navigation_box_title_controller()
            self.set_slider_value(self.gui.current_frame_idx)
        else:
            self.playback_timer.stop()
            self.play_button.setText("▶")
            self.is_playing = False

    def toggle_playback(self):
        if not self.is_playing:
            self.playback_timer.start(1000/50) # 50 fps
            self.play_button.setText("■")
            self.is_playing = True
        else:
            self.playback_timer.stop()
            self.play_button.setText("▶")
            self.is_playing = False

###################################################################################################################################################

