import os
from PySide6 import QtWidgets
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QApplication, QFileDialog

from core.runtime import Data_Manager, Video_Manager
from core.module import Frame_View, Frame_Label, Frame_Annotator
from core.tool import Canonical_Pose_Dialog, Plot_Config_Menu, DLC_Save_Dialog, Load_Label_Dialog, navigate_to_marked_frame
from ui import Menu_Widget, Video_Player_Widget, Shortcut_Manager, Toggle_Switch, Status_Bar, Frame_List_Dialog, Frame_Display_Dialog
from utils.helper import frame_to_qimage, get_roi_cv2, plot_roi, validate_crop_coord
from utils.logger import Loggerbox, QMessageBox
from utils.dataclass import Nav_Callback, Plot_Config


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
            refresh_callback = self._refresh_ui,
            parent = self
            )
        self.vm = Video_Manager(self)

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

        self.fview = Frame_View(
            self.dm, self.vm, self.vid_play, self.status_bar, self._handle_right_panel_menu_change, self._load_dlc_config, self)
        self.flabel = Frame_Label(
            self.dm, self.vm, self.vid_play, self.status_bar, self._handle_right_panel_menu_change, self._plot_config_callback, self)
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
                            ("Load Workspace", self._load_workspace),
                            ("Load DLC Label Data", self._load_dlc_label_data),
                            ("Load Prediction", self._load_prediction),
                            ("Load DLC Config", self._load_dlc_config),
                            ("Reset", self._reset_state),
                        ]
                    },
                    {
                        "submenu": "View",
                        "items": [
                            ("Canonical Pose", self._view_canonical_pose),
                            ("Config Menu", self._open_plot_config_menu),
                            ("Toggle Smart Masking", self._toggle_bg_masking),
                            ("ROI Region", self._check_roi),
                        ]
                    },
                    {
                        "submenu": "Save",
                        "items": [
                            ("Save the Current Workspace", self._save_workspace),
                            ("Save Prediction as H5", self._save_prediction),
                            ("Save Prediction as CSV", self._save_prediction_as_csv),
                            ("Save to DeepLabCut", self._save_to_dlc),
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
        self.at.display_current_frame(reset=True)
        
        self.dm.reset_dm()
        self.vm.reset_vm()
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
        if self.dm.dlc_data is None or self.dm.dlc_data.pred_data_array is None:
            self._switch_to_fview()
            Loggerbox.warning(self, "Prediction Not Loaded", "You need to load prediction before labeling.")
            return False
        self.fview.deactivate(self.menu_widget)
        self.flabel.activate(self.menu_widget)
        self.at = self.flabel
        self.dm.handle_mode_switch_fview_to_flabel()
        self.mode_toggle_flabel.set_checked(True)
        return True

    def _switch_to_fannot(self):
        self.fview.deactivate(self.menu_widget)
        self.at = self.fannot
        self.fannot.activate(self.menu_widget)
        self.mode_toggle_fannot.set_checked(True)

    def _on_mode_toggle_flabel(self, is_checked:bool):
        self._reset_ui_during_mode_switch()
        if is_checked:
            status = self._switch_to_flabel()
            if not status:
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
        file_dialog = QFileDialog(self)
        video_path, _ = file_dialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if not video_path:
            return
        self._reset_state()

        self.vm.update_video_path(video_path)
        self.dm.update_video_path(video_path)
        if self.dm.auto_loader_workspace():
            self.status_bar.show_message("Automatically loaded workspace file.", duration_ms=10000)
            return
        dlc_config_path, pred_path = self.dm.auto_loader()
        if dlc_config_path and pred_path:
            self.dm.load_pred_to_dm(dlc_config_path, pred_path)
        self._initialize_loaded_video(video_path)

    def _initialize_loaded_video(self, video_path:str):
        if self.dm.dlc_label_mode:
            try:
                self.vm.load_img_from_folder(self.dm.video_file)
            except Exception as e:
                Loggerbox.error(self, "Error Opening DLC Label", e, exc=e)
                return
        else:
            try:
                self.vm.init_extractor(video_path)
            except Exception as e:
                Loggerbox.error(self, "Error Opening Video", e, exc=e)
                return

        self.dm.total_frames = self.vm.get_frame_counts()
        self.vid_play.set_total_frames(self.dm.total_frames)
        self.vid_play.nav.set_current_video_name(self.dm.video_name)
        self.at.init_loaded_vid()
        self.at.refresh_and_display()
        self.status_bar.show_message(f"Video loaded: {self.dm.video_file}", duration_ms=2000)

    def _load_prediction(self):
        if not self.vm.check_status_msg():
            return
        try:
            file_dialog = QFileDialog(self)
            prediction_path, _ = file_dialog.getOpenFileName(self, "Select Prediction", "", "HDF5 Files (*.h5)")
            if not prediction_path:
                return

            if self.dm.dlc_data is None:
                Loggerbox.info(self, "Prediction Selected", "Prediction selected, now loading DLC config.")
                self._load_dlc_config()
                if self.dm.dlc_data is None:
                    return

            self.dm.load_pred_to_dm(self.dm.dlc_data.dlc_config_filepath, prediction_path)
        except Exception as e:
            Loggerbox.error(self, "Error Loading Prediction", f"Unexpected error during prediction loading: {e}.", exc=e)
        
        self.at.display_current_frame()
        self.flabel.reset_zoom()

    def _load_dlc_label_data(self):
        if self.dm.dlc_data is None:
            Loggerbox.info(self, "DLC Config Not Loaded", "DLC Config is needed before loading DLC label.")
            self._load_dlc_config()
            if self.dm.dlc_data is None:
                return

        folder_dialog = Load_Label_Dialog(self.dm.dlc_data, roi=self.dm.roi, video_file=self.dm.video_file, parent=self)
        folder_dialog.folder_selected.connect(self._on_label_folder_return)
        folder_dialog.exec()

    def _load_dlc_config(self):
        file_dialog = QFileDialog(self)
        dlc_config_path, _ = file_dialog.getOpenFileName(self, "Select DLC Config", "", "YAML Files (config.yaml);;All Files (*)")
        if dlc_config_path:
            try:
                self.dm.load_metadata_to_dm(dlc_config_path)
            except Exception as e:
                Loggerbox.error(self, "Error Loading DLC Config", f"Unexpected error during DLC Config loading: {e}.", exc=e)

    def _load_workspace(self):
        workspace_path, _ = QFileDialog.getOpenFileName(
            self, "Load Workspace", "", "Pickle Files (*.pkl);;All Files (*)"
        )
        if not workspace_path:
            return
        self._reset_state()
        try:
            self.dm.load_workspace(workspace_path)
        except Exception as e:
            Loggerbox.error(self, "Error Loading Workspace", f"Failed to load workspace:\n{e}", exc=e)

        if self.dm.video_file:
            self.at.display_current_frame()

    def _save_workspace(self):
        if not self.vm.check_status_msg():
            return
        self.status_bar.show_message(f"Workspace Saved to {self.dm.video_file}")
        if self.flabel.pred_data_array is not None:
            self.dm.dlc_data.pred_data_array = self.flabel.pred_data_array.copy()
        self.dm.save_workspace()

    def _save_prediction(self, to_dlc:bool=False):
        if not self._save_blocker():
            return
        if self.dm.dlc_label_mode:
            save_path = os.path.join(self.dm.video_file, f"CollectedData_{self.dm.dlc_data.scorer}.h5")
        else:
            default_path, _ = os.path.splitext(self.dm.video_file)
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Prediction as H5", f"{default_path}.h5",  "HDF5 Files (*.h5)"
            )
        if not save_path:
            return
        if not save_path.lower().endswith(('.h5','.hdf5')):
            save_path += '.h5'
        try:
            self.dm.save_pred(self.dm.dlc_data.pred_data_array, save_path, to_dlc)
        except Exception as e:
            Loggerbox.error(self, "Saving Error", f"An error occurred during saving: {e}", exc=e)
        else:
            self.dm.dlc_data.prediction_filepath = save_path
            Loggerbox.info(self, "Save Successful", f"Prediction saved in {save_path}.")

    def _save_prediction_as_csv(self):
        if not self._save_blocker():
            return
        default_path, _ = os.path.splitext(self.dm.video_file)
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Prediction as CSV", f"{default_path}.csv",  "CSV Files (*.csv)"
        )
        if not save_path:
            return
        if not save_path.lower().endswith('.csv'):
            save_path += '.csv'
        try:
            self.dm.save_pred_to_csv(self.dm.dlc_data.pred_data_array, save_path)
        except Exception as e:
            Loggerbox.error(self, "Saving Error", f"An error occurred during csv saving: {e}", exc=e)
        else:
            Loggerbox.info(self, "Save Successful", f"Prediction saved in {save_path}.")

    def _save_to_dlc(self):
        if not self._save_blocker():
            return
        if self.dm.dlc_label_mode:
            self.dm.add_frames("marked", range(self.dm.total_frames))
        if not self.dm.frames_in_any({"marked", "refined", "rejected", "approved"}):
            Loggerbox.warning(self, "No Marked Frames", "Mark some frames for export.")
            return

        if self.flabel.pred_data_array is not None:
            self.dm.dlc_data.pred_data_array = self.flabel.pred_data_array.copy()
        self.dm.save_workspace()

        if self.dm.dlc_label_mode:
            self._save_prediction(to_dlc=True)
        else:
            save_dialog = DLC_Save_Dialog(self.dm.dlc_data, self.dm.roi, self.dm.video_file, self)
            save_dialog.folder_selected.connect(self._on_save_folder_return)
            save_dialog.exec()

    def _save_blocker(self):
        if self.dm.dlc_data is not None and self.dm.dlc_data.pred_data_array is not None:
            return True

        Loggerbox.warning(self, "No Prediction", "No prediction to be saved.")
        return False

    def ask_crop_before_export(self) -> bool:
        reply = Loggerbox.question(
            self, "Crop Frame For Export?", "Crop the frames before exporting to DLC?")
        if reply == QMessageBox.Yes:
            if self.dm.roi is None:
                frame = self.vm.get_frame(self.dm.current_frame_idx)
                roi = get_roi_cv2(frame)
                if roi is not None:
                    self.dm.roi = roi
                    return True
                else:
                    raise RuntimeError("User cancel the ROI selection.")
            else:
                return True
        else:
            return False

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

    def _toggle_bg_masking(self):
        if not self.vm.check_status_msg():
            return
        if self.dm.blob_config is None:
            Loggerbox.warning("Smart masking requires background and threshold from Animal Counter.")
            return
        self.dm.background_masking = not self.dm.background_masking

        if self.dm.background_masking:
            self.fview.get_mask_from_blob_config()

        self.at.display_current_frame()

    def _check_roi(self):
        if not self.vm.check_status_msg():
            return
        frame = self.vm.get_frame(self.dm.current_frame_idx)

        roi = validate_crop_coord(self.dm.roi)
        if roi is None:
            roi = get_roi_cv2(frame)
            if roi is None:
                return
            self.dm.roi = roi
        
        frame = plot_roi(frame, self.dm.roi)
        qimage = frame_to_qimage(frame)
        dialog = Frame_Display_Dialog(title=f"Crop Region", image=qimage)
        dialog.exec()

    def _open_plot_config_menu(self):
        if not self.vm.check_status_msg():
            return
        if not self.dm.dlc_data:
            Loggerbox.warning(self, "No Prediction", "No prediction has been loaded, please load prediction first.")
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
        self.at.navigation_title_controller()
        self.at.display_current_frame()

    def _on_label_folder_return(self, image_folder):
        if not image_folder:
            return
        if self.dm.dlc_data.pred_data_array is None or self.dm.dlc_label_mode: # No existing predictions or loading label already
            dlc_config_filepath = self.dm.dlc_data.dlc_config_filepath
            self._reset_state()
            self.dm.load_metadata_to_dm(dlc_config_filepath)
            self.vm.update_video_path(image_folder)
            self.dm.load_dlc_label(image_folder)
        else:
            label_file = os.path.join(image_folder, f"CollectedData_{self.dm.dlc_data.scorer}.h5")
            if not os.path.isfile(label_file):
                Loggerbox.error(self, "Error Opening DLC Label", f"{image_folder} does not seem to have predictions!")
                return
            try:
                self.dm.load_labeled_overlay(label_file)
            except Exception as e:
                Loggerbox.error(self, "Error Opening DLC Label", e, exc=e)
                return
            self.dm.plot_config.plot_labeled = True
            self.dm.plot_config.navigate_labeled = True
            if not self.open_config:
                self._open_plot_config_menu()

        self.at.display_current_frame()
        self.flabel.reset_zoom()

    def _on_save_folder_return(self, save_folder):
        try:
            crop_status = self.ask_crop_before_export()
            self.dm.save_to_dlc(save_folder, crop_status)
        except Exception as e:
            Loggerbox.error(self, "Failed to Save to DLC", e, exc=e)
        else:
            self.at.refresh_and_display()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            self.flabel.reset_zoom()
        super().changeEvent(event)

#######################################################################################################################################################

if __name__ == "__main__":
    app = QApplication([])
    window = Frame_App()
    window.show()
    app.exec()