import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from core.runtime import Data_Manager, Video_Manager
from core.tool import Mark_Generator, Blob_Counter, Prediction_Plotter
from ui import Menu_Widget, Video_Player_Widget, Frame_List_Dialog, Status_Bar, Inference_interval_Dialog, Shortcut_Manager
from utils.helper import frame_to_pixmap, calculate_blob_inference_intervals, get_smart_bg_masking, frame_to_grayscale
from utils.logger import Loggerbox, QMessageBox


class Frame_View:
    def __init__(self,
                 data_manager: Data_Manager,
                 video_manager: Video_Manager,
                 video_play_widget: Video_Player_Widget,
                 status_bar: Status_Bar,
                 menu_slot_callback: callable,
                 request_config_callback: callable,
                 parent: QtWidgets.QWidget):
        self.dm = data_manager
        self.vm = video_manager
        self.vid_play = video_play_widget
        self.status_bar = status_bar
        self.menu_slot_callback = menu_slot_callback
        self.request_config_callback = request_config_callback
        self.main = parent

        self.viewer_menu_config = {
            "Animal Counter":{
                "buttons": [
                    ("Toggle Animal Counting", self._toggle_animal_counting),
                    ("Select Counter List to Navigate", self._select_counter_list),
                ]
            },
            "Mark Generator": {
                "buttons": [
                    ("Automatic Mark Generation", self.toggle_mark_gen_menu),
                    ("Mark / Unmark Current Frame (X)", self._toggle_frame_status),
                    ("Clear Frame Marks of Category", self.show_clear_mark_dialog),
                ]
            },
            "DLC Inference": {
                "buttons": [
                    ("Call DeepLabCut - Run Predictions of Marked Frames", self.dlc_inference_marked),
                    ("Call DeepLabCut - Run Predictions on Entire Video", self.dlc_inference_all),
                ]
            },
        }

        self.sc_viewer = Shortcut_Manager(self.main)
        self.reset_state()

    def activate(self, menu_widget:Menu_Widget):
        menu_widget.add_menu_from_config(self.viewer_menu_config)
        self.vid_play.swap_display_for_label()
        self._setup_shortcuts()
        self.vid_play.nav.set_marked_list_name("Labeled")

    def deactivate(self, menu_widget:Menu_Widget):
        self._remove_menu(menu_widget)
        self.sc_viewer.clear()

    def _remove_menu(self, menu_widget: Menu_Widget):
        for menu in self.viewer_menu_config.keys():
            menu_widget.remove_entire_menu(menu)

    def _setup_shortcuts(self):
        self.sc_viewer.add_shortcuts_from_config(
            {"mark": {"key": "X", "callback": self._toggle_frame_status}})

    def reset_state(self):
        self.counter_list = []
        self.open_mark_gen = False
        self.is_counting = False
        self.skip_counting = False

    def init_loaded_vid(self):
        if self.dm.dlc_data is None or self.dm.dlc_data.pred_data_array is None:
            self.is_counting = True
        self._init_blob_counter()

    def _init_blob_counter(self):
        self.blob_counter = Blob_Counter(
            frame_extractor=self.vm.extractor,
            config=self.dm.blob_config,
            blob_array=self.dm.blob_array,
            roi=self.dm.roi,
            parent=self.main)
        self.blob_counter.frame_processed.connect(self._plot_current_frame)
        self.blob_counter.parameters_changed.connect(self._handle_counter_config_change)
        self.blob_counter.video_counted.connect(self._handle_counter_from_counter)
        self.blob_counter.roi_set.connect(self._handle_roi_from_comp)
        self.dm.blob_config = self.blob_counter.get_config() # Get the config on every init
        if self.is_counting:
            self.vid_play.set_left_panel_widget(self.blob_counter)
            if self.dm.blob_array is not None and np.any(self.dm.blob_array):
                self._handle_counter_from_counter(self.dm.blob_array)

###################################################################################################################################################

    def display_current_frame(self, reset:bool=False):
        if reset:
            self.vid_play.display.setPixmap(QPixmap())
            self.vid_play.display.setText("No video loaded")
            return

        if not self.vm.check_status_msg():
            self.vid_play.display.setText("No video loaded")
            return

        if self.dm.dlc_data is not None and not hasattr(self, "plotter"):
            self.plotter = Prediction_Plotter(dlc_data=self.dm.dlc_data, plot_config=self.dm.plot_config)

        frame = self.vm.get_frame(self.dm.current_frame_idx)
        if frame is None:
            self.vid_play.display.setText("Failed to load current frame.")
            return
        
        if self.dm.background_masking:
            mask = self.dm.background_mask
            if mask is None:
                mask = self.get_mask_from_blob_config()
        
            frame =  np.clip(frame.astype(np.int16) + mask, 0, 255).astype(np.uint8)

        if self.dm.use_grayscale:
            frame = frame_to_grayscale(frame, keep_as_bgr=True)

        if self.is_counting:
            self.blob_counter.set_current_frame(frame, self.dm.current_frame_idx)
        else:
            self._plot_current_frame(frame)

    def get_mask_from_blob_config(self):
        if not self.dm.blob_config:
            return
        try:
            frame_batch = self.vm.get_random_frame_samples(sample_count=20)
            mask = get_smart_bg_masking(
                frame_batched = frame_batch,
                background = self.dm.blob_config.background_frames[self.dm.blob_config.bg_removal_method], 
                threshold = self.dm.blob_config.threshold,
                polarity = self.dm.blob_config.blob_type
                )
        except KeyError:
            pass
        except Exception as e:
            raise RuntimeError(f"[VIEW] Failed to get masking from workspace file: {e}.")
        else:
            self.dm.background_mask = mask

    def _plot_current_frame(self, frame, count=None):
        if self.dm.dlc_data is not None and self.dm.dlc_data.pred_data_array is not None:
            if self.dm.plot_config.plot_pred:
                frame = self.plotter.plot_predictions(frame, self.dm.dlc_data.pred_data_array[self.dm.current_frame_idx,:,:])

            if self.dm.has_current_frame_cat("labeled") and self.dm.plot_config.plot_labeled and self.dm.label_data_array is not None:
                old_colors = self.plotter.color_rgb.copy()
                self.plotter.color_rgb = [(200, 130, 0), (40, 200, 40), (40, 120, 200), (200, 40, 40), (200, 200, 80)]
                frame = self.plotter.plot_predictions(frame, self.dm.label_data_array[self.dm.current_frame_idx,:,:])
                self.plotter.color_rgb = old_colors

        pixmap = frame_to_pixmap(frame)

        # Scale pixmap to fit label
        scaled_pixmap = pixmap.scaled(self.vid_play.display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.vid_play.display.setPixmap(scaled_pixmap)
        self.vid_play.display.setText("")
        self.vid_play.set_current_frame(self.dm.current_frame_idx)

    ###################################################################################################################################################

    def refresh_and_display(self):
        self.refresh_ui()
        self.display_current_frame()

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
        group = "counting" if self.is_counting else "fview"
        grouped_cat = self.dm.get_cat_in_group(group)
        for cat in grouped_cat:
            priority = 5 if cat == "blob_merged" else 0
            self.vid_play.sld.set_frame_category(*self.dm.get_cat_metadata(cat), priority=priority)
        self.vid_play.sld.commit_categories()

    ###################################################################################################################################################

    def determine_list_to_nav(self) -> list:
        if self.is_counting and self.counter_list:
            return self.counter_list
        else:
            return self.dm.determine_list_to_nav_fview()

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

    def _clear_category(self, frame_categories):
        for cat in frame_categories: 
            self.dm.clear_frame_cat(cat)

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
            mark_gen = Mark_Generator(
                total_frames=self.dm.total_frames,
                pred_data_array=self.dm.dlc_data.pred_data_array,
                blob_array=self.dm.blob_array,
                canon_pose=self.dm.canon_pose,
                angle_map_data=self.dm.angle_map_data,
                parent=self.main)
            mark_gen.clear_old.connect(self._on_clear_old_command)
            mark_gen.frame_list_new.connect(self._handle_frame_list_from_mark_gen)
            self.vid_play.set_right_panel_widget(mark_gen)
        else:
            self.vid_play.set_right_panel_widget(None)

    def show_clear_mark_dialog(self):
        frame_categories = self.dm.get_frame_categories_fview()
        if frame_categories:
            mark_clear_dialog = Frame_List_Dialog(frame_categories, parent=self.main)
            mark_clear_dialog.categories_selected.connect(self._clear_category)
            mark_clear_dialog.exec()

    def _select_counter_list(self):
        frame_categories = self.dm.get_frame_categories_counting()
        if frame_categories:
            list_select_dialog = Frame_List_Dialog(frame_categories, parent=self.main)
            list_select_dialog.frame_indices_acquired.connect(self._counter_list_selected)
            list_select_dialog.exec()

    ###################################################################################################################################################

    def _handle_rerun_frames_exported(self, frame_tuple):
        self.dm.handle_rurun_frame_tuple(frame_tuple)
        self.display_current_frame()

    def _handle_frame_list_from_mark_gen(self, frame_list):
        self.dm.handle_mark_gen_list(frame_list)
        self.display_current_frame()

    def _handle_counter_from_counter(self, blob_array:np.ndarray):
        self.counter_list = self.dm.handle_blob_counter_array(blob_array)

    def _handle_counter_config_change(self):
        self.dm.blob_config = self.blob_counter.get_config()

    def _handle_roi_from_comp(self, roi:np.ndarray):
        self.dm.roi = roi

    def _counter_list_selected(self, counter_list):
        self.counter_list = counter_list
        if counter_list:
            self.dm.current_frame_idx = counter_list[0]
            self.display_current_frame()

    ###################################################################################################################################################

    def dlc_inference_marked(self):
        inference_list = self.dm.get_inference_list()
        if not inference_list:
            Loggerbox.warning(self.main, "No Inference List", "No unapproved / unrejected / unrefined marked frames to inference.")
            return
        
        self.call_inference(inference_list)

    def dlc_inference_all(self):
        if self.dm.total_frames > 9000:
            self.status_bar.show_message("It's over nine thousands!", duration_ms=500)
            self._suggest_animal_counting()
            if self.dm.blob_array is not None:
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
        if not self.vm.check_status_msg():
            return
        fm_list = self.dm.get_frames("marked")
        if not fm_list and not inference_list:
            Loggerbox.warning(self.main, "No Marked Frame", "No frame has been marked, please mark some frames first.")
            return
        if self.is_counting:
            self._toggle_animal_counting()
        if self.dm.dlc_data is None:
            Loggerbox.info(self.main, "Load DLC Config", "You need to load DLC config to inference with DLC models.")
            self.request_config_callback()
            if self.dm.dlc_data is None:
                return

        if not inference_list:
            inference_list = fm_list

        mask = self.dm.background_mask
        if mask is None:
            mask = self.get_mask_from_blob_config()

        from core.tool import DLC_Inference
        try:
            self.inference_window = DLC_Inference(
                dlc_data=self.dm.dlc_data,
                frame_list=inference_list,
                video_filepath=self.dm.video_file,
                roi=self.dm.roi,
                mask=mask,
                parent=self.main)
        except Exception as e:
            Loggerbox.error(self.main, "Inference Failed", f"Inference Process failed to initialize. Exception: {e}", exc=e)
        else:
            self.inference_window.show()
            self.inference_window.frames_exported.connect(self._handle_rerun_frames_exported)
            self.inference_window.prediction_saved.connect(self._reload_prediction)

    def _suggest_animal_counting(self):
        if self.dm.blob_array is None and not self.skip_counting and not self.is_counting:
            reply = Loggerbox.question(
                self.main, "Animal Not Counted",
                "Animal counting has not been performed for this video. For videos with a large "
                "number of frames, skipping animal counting may lead to a significantly slower "
                "inference process. Do you want to count animals now?"
            )
            if reply == QMessageBox.Yes:
                self._toggle_animal_counting()
            else:
                self.skip_counting = True

    def _handle_inference_intervals(self, intervals:dict, skip_existing:bool):
        existing_frames = []

        if skip_existing and self.dm.dlc_data is not None and self.dm.dlc_data.pred_data_array is not None:
            existing_frames = np.where(np.any(~np.isnan(self.dm.dlc_data.pred_data_array), axis=(1,2)))[0].tolist()

        inference_list= calculate_blob_inference_intervals(self.dm.blob_array, intervals, existing_frames)

        if not inference_list:
            Loggerbox.info(self, "Inference List Empty", "No additional frames are to be inferenced, skipping...")
            return

        reply = Loggerbox.question(
            self.main, "Inference List Calculated",
            f"A total of {len(inference_list)} frames out of {self.dm.total_frames} will be inferenced, confirm?"
        )
        if reply == QMessageBox.Yes:
            self.call_inference(inference_list)

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