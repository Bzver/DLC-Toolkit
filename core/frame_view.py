import pandas as pd
import numpy as np

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QShortcut, QKeySequence, QCloseEvent
from PySide6.QtWidgets import QMessageBox

import traceback

from utils.helper import handle_unsaved_changes_on_close, frame_to_pixmap
from core import Data_Manager, Video_Manager
from core.dataclass import Plot_Config, Nav_Callback
from core.palette import (
    NAV_COLOR_PALETTE as nvp, NAV_COLOR_PALETTE_COUNTING as nvpc,
    LABEL_INST_PALETTE as lip)
from ui import (
    Menu_Widget, Clear_Mark_Dialog, Video_Player_Widget
    )
from core.tool import (
    Prediction_Plotter, Mark_Generator, Canonical_Pose_Dialog,
    Plot_Config_Menu, Blob_Counter, navigate_to_marked_frame,
    )

class Frame_View(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Frame Viewer")
        self.setGeometry(100, 100, 1200, 960)

        self.dm = Data_Manager(
            init_vid_callback = self._initialize_loaded_video,
            refresh_callback = self._refresh_ui, parent = self)
        self.vm = Video_Manager(parent=self)

        self._setup_menu()

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.app_layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Video display area
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
        self._setup_shortcut()        
        self.reset_state()

    def _setup_menu(self):
        self.menu_widget = Menu_Widget(self)
        self.setMenuBar(self.menu_widget)
        menu_config = {
            "File": {
                "buttons": [
                    ("Load Video", self.load_video),
                    ("Load Prediction", self.load_prediction),
                    ("Load Workspace", self.load_workspace),
                ]
            },
            "View": {
                "buttons": [
                    ("Toggle Labeled Predictions Visiblity", self._toggle_labeled_vis, {"checkable": True, "checked": True}),
                    ("Toggle Predictions Visiblity", self._toggle_pred_vis, {"checkable": True, "checked": True}),
                    ("Toggle Navigating Labeled Frames", self._toggle_labeled_nav, {"checkable": True, "checked": False}),
                    ("Toggle Animal Counting", self._toggle_animal_counting, {"checkable": True, "checked": False}),
                    ("View Canonical Pose", self.view_canonical_pose),
                    ("Animal Counting Menu", self.count_animals_options),
                ]
            },
            "Mark": {
                "buttons": [
                    ("Mark / Unmark Current Frame (X)", self._toggle_frame_status),
                    ("Clear Frame Marks of Category", self.show_clear_mark_dialog),
                    ("Automatic Mark Generation", self.toggle_mark_gen_menu),
                    ("Plot Config Menu", self.open_plot_config_menu),
                ]
            },
            "Edit": {
                "buttons": [
                    ("Call Labeler - Track Correction", lambda: self.call_labeler(track_only=True)),
                    ("Call Labeler - Edit Marked Frames", lambda: self.call_labeler(track_only=False)),
                    ("Call DeepLabCut - Run Predictions of Marked Frames", self.dlc_inference_marked),
                    ("Call DeepLabCut - Run Predictions on Entire Video", self.dlc_inference_all),
                ]
            },
            "Export": {
                "display_name": "Save",
                "buttons": [
                    ("Save the Current Workspace", self.save_workspace),
                    ("Export to DeepLabCut", self.save_to_dlc),
                    ("Export Marked Frame Indices to Clipboard", self.export_marked_to_clipboard),
                    ("Merge with Existing Label in DeepLabCut", self.merge_data)
                ]
            }
        }
        self.menu_widget.add_menu_from_config(menu_config)

    def _setup_shortcut(self):
        QShortcut(QKeySequence(Qt.Key_Left | Qt.ShiftModifier), self).activated.connect(lambda: self._change_frame(-10))
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(lambda: self._change_frame(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self._change_frame(1))
        QShortcut(QKeySequence(Qt.Key_Right | Qt.ShiftModifier), self).activated.connect(lambda: self._change_frame(10))
        QShortcut(QKeySequence(Qt.Key_X), self).activated.connect(self._toggle_frame_status)
        QShortcut(QKeySequence(Qt.Key_Up), self).activated.connect(self._navigate_prev)
        QShortcut(QKeySequence(Qt.Key_Down), self).activated.connect(self._navigate_next)
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.vid_play.sld.toggle_playback)
        QShortcut(QKeySequence(Qt.Key_S | Qt.ControlModifier), self).activated.connect(self.save_workspace)

    def reset_state(self):
        if self.dm.video_file:
            self.save_workspace()
        self.dm.reset_dm_vars()
        self.vm.reset_vm()
        self.blob_counter = None
        self.vid_play.set_total_frames(0)

        self.open_mark_gen, self.open_config = False, False
        self.plot_labeled, self.plot_pred = True, True
        self.navigate_labeled = False
        self.is_counting = False
        self._refresh_ui()

    def load_video(self):
        self.reset_state()
        video_path = self.vm.load_video_dialog()
        if video_path:
            self._initialize_loaded_video(video_path)

    def _initialize_loaded_video(self, video_path:str):
        self.dm.update_video_path(video_path)
        dlc_config_path, pred_path = self.dm.auto_loader()
        if dlc_config_path and pred_path:
            self.dm.load_pred_to_dm(dlc_config_path, pred_path)
        self.vm.init_extractor(video_path)
        self.dm.total_frames = self.vm.get_frame_counts()
        self.vid_play.set_total_frames(self.dm.total_frames)

        self.blob_counter = Blob_Counter(frame_extractor=self.vm.extractor, parent=self)
        self.blob_counter.frame_processed.connect(self._plot_current_frame)
        self.blob_counter.video_counted.connect(self._handle_counter_from_counter)
        if self.is_counting:
            self.vid_play.set_left_panel_widget(self.blob_counter)

        self._refresh_and_display()
        print(f"Video loaded: {self.dm.video_file}")

    def load_prediction(self):
        if self.vm.check_status_msg():
            self.dm.pred_file_dialog()
            self.display_current_frame()

    def initialize_plotter(self):
        current_frame_data = np.full((self.dm.dlc_data.instance_count, self.dm.dlc_data.num_keypoint*3), np.nan)
        self.plotter = Prediction_Plotter(
            dlc_data = self.dm.dlc_data, current_frame_data = current_frame_data,
            plot_config = self.dm.plot_config, frame_cv2 = self.vm.current_frame)

    def count_animals_options(self):
        if self.vm.check_status_msg():  
            if self.blob_counter:
                self.blob_counter.show()

###################################################################################################################################################

    def display_current_frame(self):
        if not self.vm.check_status_msg():
            self.vid_play.display.setText("No video loaded")

        frame = self.vm.get_frame(self.dm.current_frame_idx)
        if frame is None:
            self.vid_play.display.setText("Failed to load current frame.")
            return
        
        if self.is_counting:
            self.blob_counter.set_current_frame(frame)
        else:
            self._plot_current_frame(frame)

    def _plot_current_frame(self, frame, count=None):
        if self.dm.dlc_data is not None:
            if not hasattr(self, "plotter"):
                self.initialize_plotter()

            if self.plot_pred:
                self.plotter.frame_cv2 = frame
                self.plotter.current_frame_data = self.dm.dlc_data.pred_data_array[self.dm.current_frame_idx,:,:]
                frame = self.plotter.plot_predictions()

            if self.dm.current_frame_idx in self.dm.labeled_frame_list and self.plot_labeled:
                self.plotter.frame_cv2 = frame
                self.plotter.current_frame_data = self.dm.label_data_array[self.dm.current_frame_idx,:,:]
                old_colors = self.plotter.color.copy()
                self.plotter.color = lip
                frame = self.plotter.plot_predictions()
                self.plotter.color = old_colors

        pixmap, _, _ = frame_to_pixmap(frame)

        # Scale pixmap to fit label
        scaled_pixmap = pixmap.scaled(self.vid_play.display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.vid_play.display.setPixmap(scaled_pixmap)
        self.vid_play.display.setText("")
        self.vid_play.set_current_frame(self.dm.current_frame_idx)

    ###################################################################################################################################################

    def _refresh_and_display(self):
        self._refresh_ui()
        self.display_current_frame()

    def _refresh_ui(self):
        self._navigation_title_controller()
        self._refresh_slider()

    def _navigation_title_controller(self):
        title_text = self.dm.get_title_text()
        self.vid_play.nav.setTitle(title_text)
        
        color = self.dm.determine_nav_color_counting() if self.is_counting else self.dm.determine_nav_color_fview()
        if color:
            self.vid_play.nav.setTitleColor(color)
        else:
            self.vid_play.nav.setTitleColor("black")

    def _refresh_slider(self):
        if self.is_counting:
            self.vid_play.sld.set_frame_category("zero_animal_frames", self.dm.animal_0_list, nvpc[1], priority=1)
            self.vid_play.sld.set_frame_category("one_animal_frames", self.dm.animal_1_list, nvpc[2], priority=2)
            self.vid_play.sld.set_frame_category("muliple_animal_frames", self.dm.animal_n_list, nvpc[3], priority=3)
        else:
            self.vid_play.sld.set_frame_category("marked_frames", self.dm.frame_list, nvp[1], 1)
            self.vid_play.sld.set_frame_category("rejected_frames", self.dm.rejected_frame_list, nvp[2], 2)
            self.vid_play.sld.set_frame_category("approved_frames", self.dm.approved_frame_list, nvp[3], 3)
            self.vid_play.sld.set_frame_category("refined_frames", self.dm.refined_frame_list, nvp[4], 4)
            self.vid_play.sld.set_frame_category("labeled_frames", self.dm.labeled_frame_list, nvp[5], 5)

    ###################################################################################################################################################

    def _change_frame(self, delta, absolute=None):
        if self.vm.get_frame(0) is None:
            return
        if absolute is None:
            new_frame_idx = self.dm.current_frame_idx + delta
        else:
            new_frame_idx = absolute
        if 0 <= new_frame_idx < self.dm.total_frames:
            self.dm.current_frame_idx = new_frame_idx
            self._refresh_and_display()

    def _navigate_prev(self):
        list_to_nav = self._determine_list_to_nav()
        navigate_to_marked_frame(
            self, list_to_nav, self.dm.current_frame_idx, self._handle_frame_change_from_comp, "prev")

    def _navigate_next(self):
        list_to_nav = self._determine_list_to_nav()
        navigate_to_marked_frame(
            self, list_to_nav, self.dm.current_frame_idx, self._handle_frame_change_from_comp, "next")

    def _determine_list_to_nav(self):
        return self.dm.labeled_frame_list if self.navigate_labeled else self.dm.frame_list

    def _toggle_frame_status(self):
        if self.vm.check_status_msg():
            self.dm.toggle_frame_status_fview()

    ###################################################################################################################################################

    def open_plot_config_menu(self):
        if not self.vm.check_status_msg():
            return
        if not self.dm.dlc_data:
            QtWidgets.QMessageBox.warning(self, "No Prediction", "No prediction has been loaded, please load prediction first.")
            return
        
        self.open_config = not self.open_config

        if self.open_config:
            self.open_mark_gen = False
            plot_config_widget = Plot_Config_Menu(plot_config=self.dm.plot_config, skip_opacity=True)
            plot_config_widget.config_changed.connect(self._handle_config_from_config)
            self.vid_play.set_right_panel_widget(plot_config_widget)
        else:
            self.vid_play.set_right_panel_widget(None)

    def toggle_mark_gen_menu(self):
        if not self.vm.check_status_msg():
            return
        
        self.open_mark_gen = not self.open_mark_gen
        if self.open_mark_gen:
            self.open_config = False
            mark_gen = Mark_Generator(self.dm.total_frames, self.dm.dlc_data, self.dm.canon_pose, parent=self)
            mark_gen.clear_old.connect(self._on_clear_old_command)
            mark_gen.frame_list_new.connect(self._handle_frame_marks_from_comp)
            self.vid_play.set_right_panel_widget(mark_gen)
        else:
            self.vid_play.set_right_panel_widget(None)

    def show_clear_mark_dialog(self):
        frame_categories = self.dm.get_frame_cat()
        if frame_categories:
            mark_clear_dialog = Clear_Mark_Dialog(frame_categories, parent=self)
            mark_clear_dialog.frame_category_to_clear.connect(self._clear_category)
            mark_clear_dialog.exec()

    def view_canonical_pose(self):
        dialog = Canonical_Pose_Dialog(self.dm.dlc_data, self.dm.canon_pose)
        dialog.exec()

    ###################################################################################################################################################

    def _toggle_labeled_vis(self):
        self.plot_labeled = not self.plot_labeled
        self.display_current_frame()

    def _toggle_pred_vis(self):
        self.plot_pred = not self.plot_pred
        self.display_current_frame()

    def _toggle_labeled_nav(self):
        self.navigate_labeled = not self.navigate_labeled
        self.display_current_frame()

    def _toggle_animal_counting(self):
        self.is_counting = not self.is_counting
        if self.is_counting:
            self.vid_play.set_left_panel_widget(self.blob_counter)
        else:
            self.vid_play.set_left_panel_widget(None)
        self.display_current_frame()

    def _clear_category(self, frame_category):
        self.dm.clear_frame_cat(frame_category)

    def _on_clear_old_command(self, clear_old:bool):
        self.dm.clear_old_cat(clear_old)

    ###################################################################################################################################################

    def _handle_rerun_frames_exported(self, frame_tuple):
        self.dm.approved_frame_list, self.dm.rejected_frame_list = frame_tuple
        self._refresh_and_display()

    def _handle_frame_change_from_comp(self, new_frame_idx: int):
        self.dm.current_frame_idx = new_frame_idx
        self._refresh_and_display()

    def _handle_frame_marks_from_comp(self, frame_list):
        frame_set = set(self.dm.frame_list) | set(frame_list) - set(self.dm.labeled_frame_list)
        self.dm.frame_list[:] = list(frame_set)
        self._refresh_and_display()

    def _handle_counter_from_counter(self, count_list):
        count_array = np.array(count_list)
        self.dm.animal_0_list = list(np.where(count_array==0)[0])
        self.dm.animal_1_list = list(np.where(count_array==1)[0])
        self.dm.animal_n_list = list(np.where((count_array!=1) & (count_array!=0))[0])
        self._refresh_ui()

    def _handle_config_from_config(self, new_config:Plot_Config):
        self.dm.plot_config = new_config
        self.display_current_frame()

    ###################################################################################################################################################

    def pre_saving_sanity_check(self):
        if not self.vm.check_status_msg():
            return False
        if not self.dm.frame_list:
            QMessageBox.warning(self, "No Marked Frame", "No frame has been marked, please mark some frames first.")
            return False
        return True

    def load_workspace(self):
        self.reset_state()
        self.dm.load_workspace()
        self.display_current_frame()

    def save_workspace(self):
        if self.dm.video_file:
            self.statusBar().showMessage(f"Workspace Saved to {self.dm.video_file}")
            self.dm.save_workspace()

    def call_labeler(self, track_only=False):
        pass

    def dlc_inference_marked(self):
        inference_list = self.dm.get_inference_list()
        if not inference_list:
            self.statusBar().showMessage("No unapproved / unrejected/ unrefined marked frames to inference.")
            return
        
        self.call_inference(inference_list)

    def dlc_inference_all(self):
        pass
    
    def call_inference(self, inference_list:list):
        if not self.dm.video_file:
            QMessageBox.warning(self, "Video Not Loaded", "No video is loaded, load a video first!")
            return
        if not self.dm.frame_list:
            QMessageBox.warning(self, "No Marked Frame", "No frame has been marked, please mark some frames first.")
            return
        if self.dm.dlc_data is None:
            QMessageBox.information(self, "Load DLC Config", "You need to load DLC config to inference with DLC models.")

            dlc_config = self.dm.config_file_dialog()
            if not dlc_config:
                return

            self.dm.load_metadata_to_dm(dlc_config)

        from core.tool import DLC_Inference
        try:
            self.inference_window = DLC_Inference(
                dlc_data=self.dm.dlc_data, frame_list=inference_list, video_filepath=self.dm.video_file, parent=self)
            self.inference_window.show()
            self.inference_window.frames_exported.connect(self._handle_rerun_frames_exported)
            self.inference_window.prediction_saved.connect(self.reload_prediction)
        except Exception as e:
            error_message = f"Inference Process failed to initialize. Exception: {e}"
            detailed_message = f"{error_message}\n\nTraceback:\n{traceback.format_exc()}"
            QMessageBox.warning(self, "Inference Failed", detailed_message)
            return

    def reload_prediction(self, prediction_path):
        """Reload prediction data from file and update visualization"""
        self.dm.reload_pred_to_dm(prediction_path)
        self._refresh_and_display()
        self.statusBar().showMessage("Prediction successfully reloaded")
        if hasattr(self, "inference_window") and self.inference_window:
            self.inference_window.close()
            self.inference_window = None
        if hasattr(self, 'plotter'):
            delattr(self, 'plotter')

    def export_marked_to_clipboard(self):
        df = pd.DataFrame([self.dm.frame_list])
        df.to_clipboard(sep=',', index=False, header=False)
        self.statusBar().showMessage("Marked frames exported to clipboard.")

    def save_to_dlc(self):
        if not self.pre_saving_sanity_check():
            return
        self.dm.save_to_dlc()
        self._refresh_and_display()

    def merge_data(self):
        if not self.pre_saving_sanity_check():
            return
        self.dm.merge_data()
        self._refresh_ui()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            self.display_current_frame()
        super().changeEvent(event)

    def closeEvent(self, event: QCloseEvent):
        if self.vm.check_status_msg():
            handle_unsaved_changes_on_close(self, event, False, self.save_workspace)

#######################################################################################################################################################

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = Frame_View()
    window.show()
    app.exec()