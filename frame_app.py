from PySide6 import QtWidgets
from PySide6.QtCore import QEvent
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QMessageBox, QHBoxLayout, QApplication

from ui import Menu_Widget, Video_Player_Widget, Shortcut_Manager, Toggle_Switch, Status_Bar, Frame_List_Dialog
from utils.helper import handle_unsaved_changes_on_close
from core.runtime import Data_Manager, Video_Manager, Keypoint_Edit_Manager
from core.module import Frame_View, Frame_Label, Frame_Annotator
from core.tool import Canonical_Pose_Dialog, Plot_Config_Menu, navigate_to_marked_frame
from core.dataclass import Nav_Callback, Plot_Config

class Frame_App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BVT - Behavioral Video Toolkit")
        self.setGeometry(100, 100, 1200, 960)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.app_layout = QVBoxLayout(self.central_widget)

        self.open_config = False
        self.plot_config_widget = None

        self.dm = Data_Manager(
            init_vid_callback = self._initialize_loaded_video,
            refresh_callback = self._refresh_ui, parent = self)
        self.vm = Video_Manager(self)
        self.kem = Keypoint_Edit_Manager(self._on_kem_edit, self._on_amb_frames_return, self)

        nav_callback = Nav_Callback(
            change_frame_callback = self._change_frame,
            nav_prev_callback = self._navigate_prev,
            nav_next_callback = self._navigate_next,
        )
        self.vid_play = Video_Player_Widget(
            slider_callback = self._handle_frame_change_from_comp,
            nav_callback = nav_callback,
            parent = self,
            )
        self.app_layout.addWidget(self.vid_play)

        status_layout = QHBoxLayout()
        self.status_bar = Status_Bar(self)

        self.mode_toggle_flabel = Toggle_Switch("Labeling Mode")
        self.mode_toggle_fannot = Toggle_Switch("Annotation Mode")
        self.mode_toggle_flabel.toggled.connect(self._on_mode_toggle_flabel)
        self.mode_toggle_fannot.toggled.connect(self._on_mode_toggle_fannot)
        status_layout.addWidget(self.status_bar)
        status_layout.addStretch()
        status_layout.addWidget(self.mode_toggle_flabel)
        status_layout.addWidget(self.mode_toggle_fannot)
        self.app_layout.addLayout(status_layout)

        self.sc_comm = Shortcut_Manager(self)

        self._setup_menu()
        self._setup_shortcut()

        self.fview = Frame_View(self.dm, self.vm, self.vid_play, self.status_bar, self._handle_right_panel_menu_change, self)
        self.flabel = Frame_Label(
            self.dm, self.vm, self.kem, self.vid_play, self.status_bar, self._handle_right_panel_menu_change, self._plot_config_callback, self)
        self.fannot = Frame_Annotator(self.dm, self.vm, self.vid_play, self.status_bar, self._handle_right_panel_menu_change, self)

        self._switch_to_fview()

    def _setup_menu(self):
        self.menu_widget = Menu_Widget(self)
        self.setMenuBar(self.menu_widget)
        menu_config = {
            "Main": {
                "buttons": [
                    {
                        "submenu": "Load",
                        "items": [
                            ("Load Video", self._load_video),
                            ("Load Prediction", self._load_prediction),
                            ("Load DLC Config", self._load_dlc_config),
                            ("Load Workspace", self._load_workspace),
                            ("Load DLC Label Data", self._load_dlc_label_data),
                        ]
                    },
                    ("View Canonical Pose", self._view_canonical_pose),
                    ("Config Menu", self._open_plot_config_menu),
                    ("Save the Current Workspace", self._save_workspace),
                    {
                        "submenu": "Export",
                        "items": [
                            ("Copy Frame Lists To Clipboard", self._export_dm_lists),
                            ("Copy Slider To Clipboard", self._export_slider),
                        ]
                    },
                ]
            },
        }
        self.menu_widget.add_menu_from_config(menu_config)

    def _setup_shortcut(self):
        self.sc_comm.add_shortcuts_from_config({
            "prev_frame":{"key": "Left", "callback": lambda: self._change_frame(-1)},
            "next_frame":{"key": "Right", "callback": lambda: self._change_frame(1)},
            "prev_fast":{"key": "Shift+Left", "callback": lambda: self._change_frame(-10)},
            "next_fast":{"key": "Shift+Right", "callback": lambda: self._change_frame(10)},
            "prev_mark":{"key": "Up", "callback": self._navigate_prev},
            "next_mark":{"key": "Down", "callback": self._navigate_next},
            "playback":{"key": "Space", "callback": self._toggle_playback},
            "save":{"key": "Ctrl+S", "callback": self._save_workspace},
        })

    def _reset_state(self):
        self._switch_to_fview()
        self.dm.reset_dm()
        self.vm.reset_vm()
        self.kem.reset_kem()
        self.vid_play.set_total_frames(0)
        self.vid_play.nav.set_current_video_name("---")
        self.fview.reset_state()
        self.flabel.reset_state()
        self.fannot.reset_state()
        self._refresh_ui()

        self.open_config = False
        self.plot_config_widget = None
        self._reset_ui_during_mode_switch()

    def _refresh_ui(self):
        self.at.refresh_ui()

    def _switch_to_fview(self):
        if hasattr(self, "at"):
            self.at.deactivate(self.menu_widget)
        self.fview.activate(self.menu_widget)
        self.at = self.fview
        self.mode_toggle_flabel.set_checked(False)
        self.mode_toggle_fannot.set_checked(False)

    def _switch_to_flabel(self):
        if self.dm.dlc_data is None:
            QMessageBox.information(self, )
            self._switch_to_fview()
            raise Exception("DLC data not loaded, you need to load it before labeling.")
        self.fview.deactivate(self.menu_widget)
        if self.kem.pred_data_array is None and self.dm.dlc_data is not None:
            self.kem.pred_data_array = self.dm.dlc_data.pred_data_array
        self.flabel.activate(self.menu_widget)
        self.at = self.flabel
        self.dm.handle_mode_switch_fview_to_flabel()
        self.mode_toggle_flabel.set_checked(True)

    def _switch_to_fannot(self):
        self.fview.deactivate(self.menu_widget)
        self.at = self.fannot
        self.fannot.activate(self.menu_widget)
        self.mode_toggle_fannot.set_checked(True)

    def _on_mode_toggle_flabel(self, is_checked:bool):
        self._reset_ui_during_mode_switch()
        if is_checked:
            try:
                self._switch_to_flabel()
            except:
                return
            self.mode_toggle_fannot.set_locked(True)
        else:
            self._switch_to_fview()
            self.mode_toggle_fannot.set_locked(False)
        if self.dm.video_file:
            self.at.refresh_and_display()

    def _on_mode_toggle_fannot(self, is_checked:bool):
        self._reset_ui_during_mode_switch()
        if is_checked:
            self._switch_to_fannot()
            self.mode_toggle_flabel.set_locked(True)
        else:
            self._switch_to_fview()
            self.mode_toggle_flabel.set_locked(False)
        if self.dm.video_file:
            self.at.refresh_and_display()

    ###################################################################################################

    def _load_video(self):
        self._reset_state()
        video_path = self.vm.load_video_dialog()
        if video_path:
            self.dm.update_video_path(video_path)
            if self.dm.auto_loader_workspace():
                self.status_bar.show_message("Automatically loaded workspace file.", duration_ms=10000)
                return
            self._initialize_loaded_video(video_path)

    def _initialize_loaded_video(self, video_path:str):
        dlc_config_path, pred_path = self.dm.auto_loader()
        if dlc_config_path and pred_path:
            if dlc_config_path == "Workspace":
                return
            self.dm.load_pred_to_dm(dlc_config_path, pred_path)
            self.kem.set_pred_data(self.dm.dlc_data.pred_data_array)
        self.vm.init_extractor(video_path)
        self.dm.total_frames = self.vm.get_frame_counts()
        self.vid_play.set_total_frames(self.dm.total_frames)
        self.at.init_loaded_vid()

        self.at.refresh_and_display()
        self.vid_play.nav.set_current_video_name(self.dm.video_name)
        self.status_bar.show_message(f"Video loaded: {self.dm.video_file}", duration_ms=2000)

    def _load_prediction(self):
        if not self.vm.check_status_msg():
            return
        if self.dm.pred_file_dialog():
            self.kem.set_pred_data(self.dm.dlc_data.pred_data_array)
            self.at.display_current_frame()
            self.flabel.reset_zoom()

    def _load_dlc_label_data(self):
        self._reset_state()
        image_folder = self.vm.load_label_folder_dialog()
        if not image_folder:
            return
        self.dm.load_dlc_label(image_folder)
        if self.vm.load_img_from_folder(image_folder):
            self.dm.total_frames = len(self.vm.image_files)
            self.vid_play.set_total_frames(self.dm.total_frames)
        self.at.display_current_frame()
        self.flabel.reset_zoom()

    def _load_dlc_config(self):
        if not self.vm.check_status_msg():
            return
        dlc_config_file = self.dm.config_file_dialog()
        if dlc_config_file:
            self.dm.load_metadata_to_dm(dlc_config_file)

    def _load_workspace(self):
        self._reset_state()
        self.dm.load_workspace()
        if self.dm.video_file:
            self.at.display_current_frame()

    def _save_workspace(self):
        if self.at == self.flabel:
            self.flabel.save_prediction()
        if self.dm.video_file:
            self.status_bar.show_message(f"Workspace Saved to {self.dm.video_file}")
            self.dm.save_workspace()

    def _export_dm_lists(self):
        if not self.dm.video_file:
            self.status_bar.show_message("No frame list to export.")
            return
        
        if hasattr(self.at, "is_counting") and self.at.is_counting:
            frame_categories = self.dm.get_frame_categories_counting()
        elif hasattr(self.at, "open_outlier") and any[self.at.open_outlier, self.dm.plot_config.navigate_roi]:
            frame_categories = self.dm.get_frame_categories_flabel()
        else:
            frame_categories = self.dm.get_frame_categories_fview()

        if frame_categories:
            list_select_dialog = Frame_List_Dialog(frame_categories, parent=self)
            list_select_dialog.frame_indices_acquired.connect(self._frame_list_selected)
            list_select_dialog.exec()

    def _frame_list_selected(self, frame_list):
        clipboard = QApplication.clipboard()
        clipboard.setText(', '.join(map(str, frame_list)))
        self.status_bar.show_message(f"Frame List copied to clipboard.")

    def _export_slider(self):
        if self.dm.video_file:
            self.vid_play.sld.export_background()
            self.status_bar.show_message(f"Frame slider copied to clipboard.")

   ###################################################################################################

    def _change_frame(self, delta):
        if self.vm.get_frame(0) is None:
            return
        new_frame_idx = self.dm.current_frame_idx + delta
        if 0 <= new_frame_idx < self.dm.total_frames:
            self.dm.current_frame_idx = new_frame_idx
            self.kem.last_selected_idx = None
            self.at.display_current_frame()
            self.at.navigation_title_controller()

    def _navigate_prev(self):
        list_to_nav = self.at.determine_list_to_nav()
        navigate_to_marked_frame(
            self, list_to_nav, self.dm.current_frame_idx, self._handle_frame_change_from_comp, "prev")

    def _navigate_next(self):
        list_to_nav = self.at.determine_list_to_nav()
        navigate_to_marked_frame(
            self, list_to_nav, self.dm.current_frame_idx, self._handle_frame_change_from_comp, "next")

    def _toggle_playback(self):
        self.vid_play.sld.toggle_playback()

    ###################################################################################################

    def _view_canonical_pose(self):
        if not self.vm.check_status_msg():
            return
        dialog = Canonical_Pose_Dialog(self.dm.dlc_data, self.dm.canon_pose)
        dialog.exec()

    def _open_plot_config_menu(self):
        if not self.vm.check_status_msg():
            return
        if not self.dm.dlc_data:
            QMessageBox.warning(self, "No Prediction", "No prediction has been loaded, please load prediction first.")
            return
        
        self.open_config = not self.open_config

        if self.open_config:
            self.at.sync_menu_state()
            label_mode = self.mode_toggle_flabel.is_checked()
            plot_config_widget = Plot_Config_Menu(plot_config=self.dm.plot_config, label_mode=label_mode)
            plot_config_widget.config_changed.connect(self._handle_config_from_config)
            self.vid_play.set_right_panel_widget(plot_config_widget)
            self.plot_config_widget = plot_config_widget
        else:
            self.vid_play.set_right_panel_widget(None)
            self.plot_config_widget = None

    def _reset_ui_during_mode_switch(self):
        self.at.sync_menu_state(close_all=True)
        self.vid_play.set_right_panel_widget(None)
        self.vid_play.set_left_panel_widget(None)
        self.plot_config_widget = None
        if self.open_config:
            self.open_config = False
            self._open_plot_config_menu()

    def _plot_config_callback(self):
        if self.plot_config_widget:
            self.plot_config_widget.refresh_toggle_state()

    def _handle_config_from_config(self, new_config:Plot_Config):
        self.dm.plot_config = new_config
        if not self.dm.plot_config.auto_snapping:
            self.flabel.reset_zoom()
        self.at.refresh_and_display()

    def _handle_right_panel_menu_change(self):
        self.open_config = False
        self.plot_config_widget = None

    def _handle_frame_change_from_comp(self, new_frame_idx: int):
        self.dm.current_frame_idx = new_frame_idx
        self.kem.last_selected_idx = None
        self.at.navigation_title_controller()
        self.at.display_current_frame()

    def _on_kem_edit(self):
        self.flabel.on_track_data_changed()

    def _on_amb_frames_return(self, frame_list):
        self.dm.add_frames("ambiguous", frame_list)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            self.flabel.reset_zoom()
        super().changeEvent(event)

    def closeEvent(self, event: QCloseEvent):
        if not self.vm.video_file:
            return
        if self.vm.check_status_msg():
            handle_unsaved_changes_on_close(self, event, True, self._save_workspace)

#######################################################################################################################################################

if __name__ == "__main__":
    app = QApplication([])
    window = Frame_App()
    window.show()
    app.exec()