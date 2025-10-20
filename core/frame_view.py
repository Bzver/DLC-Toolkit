import pandas as pd
import numpy as np

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

import traceback

from ui import (
    Menu_Widget, Video_Player_Widget, Clear_Mark_Dialog, Shortcut_Manager, Status_Bar,
      Inference_interval_Dialog, Progress_Indicator_Dialog)
from utils.helper import frame_to_pixmap
from .data_man import Data_Manager
from .video_man import  Video_Manager
from .tool import Mark_Generator, Blob_Counter, Prediction_Plotter
from .palette import (NAV_COLOR_PALETTE as nvp, NAV_COLOR_PALETTE_COUNTING as nvpc, LABEL_INST_PALETTE as lip)

class Frame_View:
    def __init__(self,
                 data_manager: Data_Manager,
                 video_manager: Video_Manager,
                 video_play_widget: Video_Player_Widget,
                 status_bar: Status_Bar,
                 menu_slot_callback: callable,
                 parent: QtWidgets.QWidget):
        self.dm = data_manager
        self.vm = video_manager
        self.vid_play = video_play_widget
        self.status_bar = status_bar
        self.menu_slot_callback = menu_slot_callback
        self.main = parent

        self._setup_shortcuts()
        self.reset_state()

    def activate(self, menu_widget:Menu_Widget):
        menu_widget.add_menu_from_config(self.viewer_menu_config)
        self.shortcuts.set_enabled(True)
        self.vid_play.swap_display_for_label()
        self.vid_play.nav.set_marked_list_name("Labeled")

    def deactivate(self, menu_widget:Menu_Widget):
        self._remove_menu(menu_widget)
        self.shortcuts.set_enabled(False)

    def _remove_menu(self, menu_widget: Menu_Widget):
        for menu in self.viewer_menu_config.keys():
            menu_widget.remove_entire_menu(menu)

    def _setup_shortcuts(self):
        self.shortcuts = Shortcut_Manager(self.main)
        self.shortcuts.add_shortcut("mark", "X", self._toggle_frame_status)
        self.shortcuts.set_enabled(True)

    def reset_state(self):
        if self.dm.video_file:
            self.save_workspace()
        self.vid_play.set_total_frames(0)

        self.open_mark_gen = False
        self.is_counting = False
        self.skip_counting = False

        self.viewer_menu_config = {
            "View":{
                "buttons": [
                    ("Toggle Animal Counting", self._toggle_animal_counting, {"checkable": True, "checked": False}),
                ]
            },
            "Mark": {
                "buttons": [
                    ("Mark / Unmark Current Frame (X)", self._toggle_frame_status),
                    ("Clear Frame Marks of Category", self.show_clear_mark_dialog),
                    ("Automatic Mark Generation", self.toggle_mark_gen_menu),
                ]
            },
            "Inference": {
                "buttons": [
                    ("Call DeepLabCut - Run Predictions of Marked Frames", self.dlc_inference_marked),
                    ("Call DeepLabCut - Run Predictions on Entire Video", self.dlc_inference_all),
                ]
            },
            "Save":{
                "buttons": [
                    ("Export to DeepLabCut", self.save_to_dlc),
                    ("Export Marked Frame Indices to Clipboard", self.export_marked_to_clipboard),
                    ("Merge with Existing Label in DeepLabCut", self.merge_data),
                ]
            },
        }
        
        self.refresh_ui()

    def init_loaded_vid(self):
        self._init_blob_counter()

    def _init_blob_counter(self):
        self.blob_counter = Blob_Counter(frame_extractor=self.vm.extractor, config=self.dm.blob_config, parent=self.main)
        self.blob_counter.frame_processed.connect(self._plot_current_frame)
        self.blob_counter.parameters_changed.connect(self._handle_counter_config_change)
        self.blob_counter.video_counted.connect(self._handle_counter_from_counter)
        if self.is_counting:
            self.vid_play.set_left_panel_widget(self.blob_counter)

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

    def _initialize_plotter(self):
        current_frame_data = np.full((self.dm.dlc_data.instance_count, self.dm.dlc_data.num_keypoint*3), np.nan)
        self.plotter = Prediction_Plotter(
            dlc_data = self.dm.dlc_data, current_frame_data = current_frame_data,
            plot_config = self.dm.plot_config, frame_cv2 = self.vm.current_frame)

    def _plot_current_frame(self, frame, count=None):
        if self.dm.dlc_data is not None:
            if not hasattr(self, "plotter"):
                self._initialize_plotter()

            if self.dm.plot_config.plot_pred:
                self.plotter.frame_cv2 = frame
                self.plotter.current_frame_data = self.dm.dlc_data.pred_data_array[self.dm.current_frame_idx,:,:]
                frame = self.plotter.plot_predictions()

            if self.dm.current_frame_idx in self.dm.labeled_frame_list and self.dm.plot_config.plot_labeled:
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

    def refresh_and_display(self):
        self.refresh_ui()
        self.display_current_frame()

    def reset_zoom(self): # Kept for frame_app complatibility
        pass

    def refresh_ui(self):
        self.navigation_title_controller()
        self._refresh_slider()

    def navigation_title_controller(self):
        title_text = self.dm.get_title_text()
        self.status_bar.show_message(title_text, duration_ms=0)
        color = self.dm.determine_nav_color_counting() if self.is_counting else self.dm.determine_nav_color_fview()
        if color:
            self.vid_play.nav.set_title_color(color)
        else:
            self.vid_play.nav.set_title_color("black")

    def _refresh_slider(self):
        self.vid_play.sld.clear_frame_category()
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

    def determine_list_to_nav(self):
        return self.dm.labeled_frame_list if self.dm.plot_config.navigate_labeled else self.dm.frame_list

    def _toggle_frame_status(self):
        if self.vm.check_status_msg():
            self.dm.toggle_frame_status_fview()

    def _toggle_animal_counting(self):
        self.is_counting = not self.is_counting
        if not self.dm.video_file:
            return
        if self.is_counting:
            self._init_blob_counter()
        else:
            self.vid_play.set_left_panel_widget(None)
        self.refresh_and_display()

    def _clear_category(self, frame_category):
        self.dm.clear_frame_cat(frame_category)

    def _on_clear_old_command(self, clear_old:bool):
        self.dm.clear_old_cat(clear_old)

    ###################################################################################################################################################

    def sync_menu_state(self, close_all:bool=False):
        self.open_mark_gen = False
        if close_all:
            self.is_counting = False

    def toggle_mark_gen_menu(self):
        if not self.vm.check_status_msg():
            return
        
        self.open_mark_gen = not self.open_mark_gen
        if self.open_mark_gen:
            self.menu_slot_callback()
            mark_gen = Mark_Generator(self.dm.total_frames, self.dm.dlc_data, self.dm.canon_pose, parent=self.main)
            mark_gen.clear_old.connect(self._on_clear_old_command)
            mark_gen.frame_list_new.connect(self._handle_frame_list_from_comp)
            self.vid_play.set_right_panel_widget(mark_gen)
        else:
            self.vid_play.set_right_panel_widget(None)

    def show_clear_mark_dialog(self):
        frame_categories = self.dm.get_frame_cat()
        if frame_categories:
            mark_clear_dialog = Clear_Mark_Dialog(frame_categories, parent=self.main)
            mark_clear_dialog.frame_category_to_clear.connect(self._clear_category)
            mark_clear_dialog.exec()

    ###################################################################################################################################################

    def _handle_rerun_frames_exported(self, frame_tuple):
        self.dm.approved_frame_list, self.dm.rejected_frame_list = frame_tuple
        self.refresh_and_display()

    def _handle_frame_list_from_comp(self, frame_list):
        frame_set = set(self.dm.frame_list) | set(frame_list) - set(self.dm.labeled_frame_list)
        self.dm.frame_list[:] = list(frame_set)
        self.refresh_and_display()

    def _handle_counter_from_counter(self, count_list):
        count_array = np.array(count_list)
        self.dm.animal_0_list = list(np.where(count_array==0)[0])
        self.dm.animal_1_list = list(np.where(count_array==1)[0])
        self.dm.animal_n_list = list(np.where((count_array!=1) & (count_array!=0))[0])
        self.dm.inst_count_per_frame_vid = count_list
        self.refresh_ui()

    def _handle_counter_config_change(self):
        self.dm.blob_config = self.blob_counter.get_config()

    ###################################################################################################################################################

    def pre_saving_sanity_check(self):
        if not self.vm.check_status_msg():
            return False
        if not self.dm.frame_list:
            QMessageBox.warning(self.main, "No Marked Frame", "No frame has been marked, please mark some frames first.")
            return False
        return True

    def dlc_inference_marked(self):
        inference_list = self.dm.get_inference_list()
        if not inference_list:
            self.status_bar.show_message("No unapproved / unrejected/ unrefined marked frames to inference.")
            return
        
        self.call_inference(inference_list)

    def dlc_inference_all(self):
        if self.dm.total_frames > 9000:
            self.status_bar.show_message("It's over nine thousands!", duration_ms=500)
            self._suggest_animal_counting()
            if self.dm.inst_count_per_frame_vid is not None:
                dialog = Inference_interval_Dialog(self.main)
                dialog.intervals_selected.connect(self._handle_inference_intervals)
                dialog.exec()
            elif self.skip_counting:
                inference_list = list(range(self.dm.total_frames))
                self.call_inference(inference_list)
        else:
            inference_list = list(range(self.dm.total_frames))
            self.call_inference(inference_list)
    
    def call_inference(self, inference_list:list):
        if not self.dm.video_file:
            QMessageBox.warning(self.main, "Video Not Loaded", "No video is loaded, load a video first!")
            return
        if not self.dm.frame_list and not inference_list:
            QMessageBox.warning(self.main, "No Marked Frame", "No frame has been marked, please mark some frames first.")
            return
        if self.is_counting:
            self._toggle_animal_counting()
        if self.dm.dlc_data is None:
            QMessageBox.information(self.main, "Load DLC Config", "You need to load DLC config to inference with DLC models.")

            dlc_config = self.dm.config_file_dialog()
            if not dlc_config:
                return

            self.dm.load_metadata_to_dm(dlc_config)
            if not inference_list:
                inference_list = self.dm.frame_list

        from core.tool import DLC_Inference
        try:
            self.inference_window = DLC_Inference(
                dlc_data=self.dm.dlc_data, frame_list=inference_list, video_filepath=self.dm.video_file, parent=self.main)
            self.inference_window.show()
            self.inference_window.frames_exported.connect(self._handle_rerun_frames_exported)
            self.inference_window.prediction_saved.connect(self._reload_prediction)
            self.inference_window.crop_coords_requested.connect(self._update_inference_crop_coords)
        except Exception as e:
            error_message = f"Inference Process failed to initialize. Exception: {e}"
            detailed_message = f"{error_message}\n\nTraceback:\n{traceback.format_exc()}"
            QMessageBox.warning(self.main, "Inference Failed", detailed_message)
            return

    def _suggest_animal_counting(self):
        if self.dm.inst_count_per_frame_vid is None and not self.skip_counting and not self.is_counting:
            reply = QMessageBox.question(
                self.main, "Animal Not Counted",
                "Animal counting has not been performed for this video. For videos with a large "
                "number of frames, skipping animal counting may lead to a significantly slower "
                "inference process. Do you want to count animals now?"
            )
            if reply == QMessageBox.Yes:
                self._toggle_animal_counting()
            else:
                self.skip_counting = True

    def _handle_inference_intervals(self, intervals: dict):
        inference_list = []
        last_inferenced_frame = 0

        for frame_idx in range(self.dm.total_frames):
            animal_count = self.dm.inst_count_per_frame_vid[frame_idx]
            
            current_interval = 1

            if animal_count == 0:
                current_interval = intervals["interval_0_animals"]
            elif animal_count == 1:
                current_interval = intervals["interval_1_animal"]
            else: # animal_count >= 2
                current_interval = intervals["interval_n_animals"]
            
            if frame_idx - last_inferenced_frame >= current_interval:
                inference_list.append(frame_idx)
                last_inferenced_frame = frame_idx
        
        inference_set = set(inference_list)
        reply = QMessageBox.question(
            self.main, "Inference List Calculated",
            f"A total of {len(inference_set)} frames out of {self.dm.total_frames} will be inferenced, confirm?"
        )
        if reply == QMessageBox.Yes:
            self.call_inference(sorted(list(inference_set)))

    def _update_inference_crop_coords(self, frame_list:list):
        if not frame_list:
             return
        
        crop_dict = {}
        self._init_blob_counter()
        progress = Progress_Indicator_Dialog(
            0, len(frame_list), "Getting Crop Coords", "Acquring crop coordinates from Blob_Counter...", self.main)
        for i, frame_idx in enumerate(frame_list):
            progress.setValue(i)
            if progress.wasCanceled():
                return
            frame = self.vm.get_frame(frame_idx)
            bbox = self.blob_counter.get_blob_bbox(frame)
            crop_dict[frame_idx] = bbox
            
        progress.close()
        if crop_dict:
            self.inference_window.crop_coords = crop_dict
            self.inference_window.inference_workflow()
        else:
            QMessageBox.critical(self.main, "Failed", "Failed to Extract Crop Coords.")

    def _reload_prediction(self, prediction_path:str):
        """Reload prediction data from file and update visualization"""
        self.dm.reload_pred_to_dm(prediction_path)
        self.refresh_and_display()
        self.status_bar.show_message("Prediction successfully reloaded")
        if hasattr(self, "inference_window") and self.inference_window:
            self.inference_window.close()
            self.inference_window = None
        if hasattr(self, 'plotter'):
            delattr(self, 'plotter')

    def export_marked_to_clipboard(self):
        df = pd.DataFrame([self.dm.frame_list])
        df.to_clipboard(sep=',', index=False, header=False)
        self.status_bar.show_message("Marked frames exported to clipboard.")

    def save_to_dlc(self):
        if not self.pre_saving_sanity_check():
            return
        self.dm.save_to_dlc()
        self.refresh_and_display()

    def merge_data(self):
        if not self.pre_saving_sanity_check():
            return
        self.dm.merge_data()
        self.refresh_ui()