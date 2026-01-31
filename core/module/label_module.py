import numpy as np
import pandas as pd

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTransform

from typing import Optional, Tuple

from core.runtime import Data_Manager, Video_Manager
from core.tool import (Outlier_Finder, Canvas, Prediction_Plotter, Uno_Stack,
                       Track_Correction_Dialog, Iteration_Review_Dialog)
from ui import (Menu_Widget, Video_Player_Widget, Pose_Rotation_Dialog, Status_Bar, Instance_Selection_Dialog,
                Shortcut_Manager, Progress_Indicator_Dialog, Frame_Range_Dialog, Keypoint_Num_Dialog)
from utils.pose import (
    rotate_selected_inst, generate_missing_inst, generate_missing_kp_for_inst, 
    generate_missing_kp_batch, calculate_pose_centroids, calculate_pose_rotations, outlier_removal
    )
from utils.track import (
    Track_Fixer, interpolate_track_all, delete_track, swap_track, interpolate_track,
    )
from utils.helper import (
    frame_to_pixmap, calculate_snapping_zoom_level, get_instances_on_current_frame,
    get_instance_count_per_frame, clean_inconsistent_nans, frame_to_grayscale
    )
from utils.dataclass import Plot_Config, Plotter_Callbacks
from utils.logger import Loggerbox


class Frame_Label:
    def __init__(
            self,
            data_manager: Data_Manager,
            video_manager: Video_Manager,
            video_play_widget: Video_Player_Widget,
            status_bar: Status_Bar,
            menu_slot_callback: callable,
            plot_config_callback: callable,
            parent: QtWidgets.QWidget
            ):
        self.dm = data_manager
        self.vm = video_manager
        self.vid_play = video_play_widget
        self.status_bar = status_bar
        self.menu_slot_callback = menu_slot_callback
        self.plot_config_callback = plot_config_callback
        self.main = parent

        self.labeler_menu_config = {
            "Zoom":{
                "buttons": [
                    ("Toggle Zoom Mode (Z)", self._toggle_zoom_mode, {"checkable": True, "checked": False}),
                    ("Reset Zoom", self.reset_zoom),
                ]
            },
            "Refine": {
                "buttons": [
                    ("Open Outlier Cleaning Menu", self._call_outlier_finder),
                    {
                        "submenu": "Interpolate",
                        "items": [
                            ("Interpolate Selected Instance on Current Frame (T)", self._interpolate_track),
                            ("Interpolate Missing Keypoints for Selected Instance (Shift+T)", self._interpolate_missing_kp),
                            ("Interpolate Selected Instance Across All Frames", self._interpolate_all),     
                            ("Interpolate Missing Keypoints for All Frames", self._interpolate_all_missing_kp),
                        ]
                    },
                    {
                        "submenu": "Delete",
                        "items": [
                            ("Delete Selected Instance On Current Frame (X)", self._delete_inst),
                            ("Delete Selected Instance On Frames Between Seleted Range", self._delete_track),
                            ("Delete All Prediction Inside Selected Area", self._designate_no_mice_zone),
                        ]
                    },
                    {
                        "submenu": "Swap",
                        "items": [
                            ("Swap Instances On Current Frame (W)", self._swap_track_single),
                            ("Swap Instances On Frames Between Selecte Range", self._swap_track_free),
                            ("Swap Until The End (Shift + W)", self._swap_track_continous),
                        ]
                    },
                    {
                        "submenu": "Duplicate",
                        "items": [
                            ("Copy Selected Instance On Current Frame (Ctrl + C)", self._copy_inst),
                            ("Paste Inst On Current Frame (Ctrl + V)", self._paste_inst),
                        ]
                    },
                    ("Semi Automatic Track Correction", self._temporal_track_correct),
                    ("Generate Instance (G)", self._generate_inst),
                    ("Rotate Selected Instance (R)", self._rotate_inst),
                ]
            },
            "Edit": {
                "buttons": [
                    ("Direct Keypoint Edit (Q)", self._direct_keypoint_edit),
                    ("Mark Current Frame As Unrefined", self._toggle_frame_status),
                    ("Mark All As Refined", self._mark_all_as_refined),
                    ("Undo Changes (Ctrl+Z)", self._undo_changes),
                    ("Redo Changes (Ctrl+Y)", self._redo_changes),
                ]
            },
        }

        self._init_gview()
        self.plotter_callback = Plotter_Callbacks(
            keypoint_coords_callback = self._update_keypoint_position,
            keypoint_object_callback = self.gview.set_dragged_keypoint,
            box_coords_callback = self._update_instance_position,
            box_object_callback = self.gview._handle_box_selection
        )

        self.sc_label = Shortcut_Manager(self.main)
        self.reset_state()

    def activate(self, menu_widget:Menu_Widget):
        menu_widget.add_menu_from_config(self.labeler_menu_config)
        if self.gview is None:
            self._init_gview()
        self.vid_play.nav.set_marked_list_name("ROI")
        self.vid_play.swap_display_for_graphics_view(self.gview)
        self._setup_shortcuts()
        
        self.pred_data_array = self.dm.dlc_data.pred_data_array.copy()
        if not self.gview.is_kp_edit:
            self._direct_keypoint_edit()

    def deactivate(self, menu_widget:Menu_Widget):
        self._remove_menu(menu_widget)
        self.sc_label.clear()
        self.dm.dlc_data.pred_data_array = self.pred_data_array.copy()
        self.pred_data_array = None

    def _remove_menu(self, menu_widget:Menu_Widget):
        for menu in self.labeler_menu_config.keys():
            menu_widget.remove_entire_menu(menu)

    def _setup_shortcuts(self):
        self.sc_label.add_shortcuts_from_config({
            "swp_trk_sg": {"key": "W", "callback": self._swap_track_single},
            "swp_trk_ct": {"key": "Shift+W", "callback": self._swap_track_continous},
            "del_trk": {"key": "X", "callback": self._delete_inst},
            "intp_trk": {"key": "T", "callback": self._interpolate_track},
            "intp_ms_kp": {"key": "Shift+T", "callback": self._interpolate_missing_kp},
            "gen_inst": {"key": "G", "callback": self._generate_inst},
            "rot_inst": {"key": "R", "callback": self._rotate_inst},
            "kp_edit": {"key": "Q", "callback": self._direct_keypoint_edit},
            "del_kp": {"key": "Backspace", "callback": self._on_keypoint_delete},
            "copy": {"key": "Ctrl+C", "callback": self._copy_inst},
            "paste": {"key": "Ctrl+V", "callback": self._paste_inst},
            "undo": {"key": "Ctrl+Z", "callback": self._undo_changes},
            "redo": {"key": "Ctrl+Y", "callback": self._redo_changes},
            "zoom": {"key": "Z", "callback": self._toggle_zoom_mode},
            "snap_to_inst": {"key": "E", "callback": self._toggle_snap_to_instances},
        })

    def reset_state(self):
        self.uno = Uno_Stack()
        self.last_selected_idx = None
        self.inst_pastebin = None
        self.pred_data_array = None
        self.open_outlier = False
        self.outlier_mask = None
        self.reset_zoom()
            
    def init_loaded_vid(self):
        self.reset_zoom()

    def _init_gview(self):
        self.gview = Canvas(parent=self.main)
        self.gview.instance_selected.connect(self._update_last_selected_inst)
        self.gview.rect_finished.connect(self._on_canvas_rect_return)
        self.vid_play.swap_display_for_graphics_view(self.gview)

    ###################################################################################################################################################

    def display_current_frame(self):
        self.gview.sbox = None

        if self.dm.dlc_data is not None and not hasattr(self, "plotter"):
            self.plotter = Prediction_Plotter(
                dlc_data = self.dm.dlc_data, plot_config=self.dm.plot_config, plot_callback=self.plotter_callback, fast_mode=False)

        frame = self.vm.get_frame(self.dm.current_frame_idx)
        if frame is None:
            self.gview.clear_graphic_scene()
            return

        if self.dm.background_masking:
            mask = self.dm.background_mask
            if mask is None:
                mask = self.get_mask_from_blob_config()
        
            frame = np.clip(frame.astype(np.int16) + mask, 0, 255).astype(np.uint8)

        if self.dm.use_grayscale:
            frame = frame_to_grayscale(frame)

        self.gview.clear_graphic_scene()
        self._plot_current_frame(frame)

    def _plot_current_frame(self, frame):
        pixmap, w, h = frame_to_pixmap(frame, request_dim=True)
        pixmap_item = self.gview.gscene.addPixmap(pixmap)
        pixmap_item.setZValue(-1)
        self.gview.gscene.setSceneRect(0, 0, w, h)
        
        current_frame_data = self.pred_data_array[self.dm.current_frame_idx]
        
        if self.dm.plot_config.auto_snapping:
            view_width, view_height = self.gview.get_graphic_scene_dim()
            if not np.all(np.isnan(current_frame_data)):
                self.gview.zoom_factor, center_x, center_y = \
                    calculate_snapping_zoom_level(current_frame_data, view_width, view_height)
                self.gview.centerOn(center_x, center_y)

        new_transform = QTransform()
        new_transform.scale(self.gview.zoom_factor, self.gview.zoom_factor)
        self.gview.setTransform(new_transform)

        if self.pred_data_array is not None:
            self.plotter.plot_config = self.dm.plot_config
            if self.outlier_mask is None:
                self.plotter.plot_predictions(self.gview.gscene, current_frame_data)
            elif self.outlier_mask.ndim == 2:
                self.plotter.plot_predictions(self.gview.gscene, current_frame_data, marked_frame_instance=self.outlier_mask[self.dm.current_frame_idx])
            elif self.outlier_mask.ndim == 3:
                self.plotter.plot_predictions(self.gview.gscene, current_frame_data, marked_frame_kp=self.outlier_mask[self.dm.current_frame_idx])

        self.gview.update()
        self.vid_play.set_current_frame(self.dm.current_frame_idx)

    ###################################################################################################################################################

    def refresh_and_display(self):
        self.refresh_ui()
        self.display_current_frame()

    def reset_zoom(self):
        self.gview.reset_zoom()

    def refresh_ui(self):
        self.navigation_title_controller()
        self._refresh_slider()

    def navigation_title_controller(self):
        title_text = self.dm.get_title_text(labeler=True, kp_edit=self.gview.is_kp_edit)
        self.status_bar.show_message(title_text, duration_ms=0)

        if self.open_outlier or self.dm.plot_config.navigate_roi:
            color = self.dm.determine_nav_color_flabel()
        else:
            color = self.dm.determine_nav_color_fview()
        if color:
            self.vid_play.nav.set_title_color(color)
        else:
            self.vid_play.nav.set_title_color("black")

    def _refresh_slider(self):
        self.vid_play.sld.clear_frame_category()
        if self.open_outlier:
            self.vid_play.sld.set_frame_category(*self.dm.get_cat_metadata("outlier"))
        elif self.dm.plot_config.navigate_roi:
            self.vid_play.sld.set_frame_category(*self.dm.get_cat_metadata("roi_change"))
            self.dm.handle_cat_update("roi_change", self._update_roi())
        else:
            self.vid_play.sld.set_frame_category(*self.dm.get_cat_metadata("marked"))
            self.vid_play.sld.set_frame_category(*self.dm.get_cat_metadata("refined"))
        self.vid_play.sld.commit_categories()

    ###################################################################################################################################################

    def determine_list_to_nav(self):
        if self.open_outlier:
            return self.dm.get_frames("outlier")
        else:
            return self.dm.determine_list_to_nav_flabel()

    def _toggle_frame_status(self):
        self.dm.toggle_frame_status_flabel()
        self.refresh_ui()

    def _mark_refined(self):
        self.dm.mark_refined_flabel(self.dm.current_frame_idx)

    def _mark_all_as_refined(self):
        self.dm.mark_all_refined_flabel()

    def _toggle_zoom_mode(self):
        self.gview.toggle_zoom_mode()
        self.navigation_title_controller()

    def _toggle_snap_to_instances(self):
        self.dm.plot_config.auto_snapping = not self.dm.plot_config.auto_snapping
        self.plot_config_callback()
        self.display_current_frame()

    ###################################################################################################################################################

    def sync_menu_state(self, close_all:bool=False):
        self.open_outlier = False
        if close_all:
            self.reset_zoom()

    def _call_outlier_finder(self):
        if not self.vm.check_status_msg():
            return
        if self.pred_data_array is None:
            return
        
        self.open_outlier = not self.open_outlier
        self.reset_zoom()

        if self.open_outlier:
            self.menu_slot_callback()
            self.outlier_finder = Outlier_Finder(
                self.pred_data_array,
                skele_list=self.dm.dlc_data.skeleton,
                kp_to_idx=self.dm.dlc_data.keypoint_to_idx,
                angle_map_data=self.dm.angle_map_data, parent=self.main)
            self.outlier_finder.mask_changed.connect(self._handle_outlier_mask_from_comp)
            self.vid_play.set_right_panel_widget(self.outlier_finder)
        else:
            self.vid_play.set_right_panel_widget(None)

    def _track_edit_blocker(self):
        if self.pred_data_array is None:
            return
        if self.open_outlier:
            Loggerbox.warning(self.main, "Outlier Cleaning Pending", "An outlier cleaning operation is pending. Please dismiss the outlier widget first.")
            return False
        if self.gview.is_kp_edit:
            self._direct_keypoint_edit()
        return True

    ###################################################################################################################################################

    def _direct_keypoint_edit(self):
        if self.pred_data_array is None:
            return

        self.gview.toggle_kp_edit()
        self.dm.plot_config.edit_mode = self.gview.is_kp_edit
        self.navigation_title_controller()

        if self.gview.is_kp_edit:
            self.status_bar.show_message(
                "Keypoint editing mode is ON. Drag to adjust. Press Backspace to delete a keypoint."
            )
        else:
            self.status_bar.show_message("Keypoint editing mode is OFF.")

        self.navigation_title_controller()
        self.display_current_frame() # Refresh the frame to get the kp edit status through to plotter

    def _designate_no_mice_zone(self):
        if not self._track_edit_blocker():
            return
        self.gview.setCursor(Qt.CrossCursor)
        self.gview.is_drawing_zone = True
        self._save_state_for_undo()
        Loggerbox.info(self.main, "Designate No Mice Zone", "Click and drag on the video to select a zone. Release to apply.")

    def _delete_inst(self):
        if self.pred_data_array is None:
            return
        selected_instance_idx = self._instance_multi_select()
        if selected_instance_idx is None:
            return
        self._save_state_for_undo()
        try:
            self.pred_data_array = delete_track(self.pred_data_array, self.dm.current_frame_idx, selected_instance_idx)
        except ValueError as e:
            Loggerbox.error(self.main, "Deletion Error", str(e), exc=e)
        else:
            self.display_current_frame()

    def _delete_track(self, selected_range:Optional[Tuple[int,int]]):
        if self.pred_data_array is None:
            return
        selected_instance_idx = self._instance_multi_select()
        if selected_instance_idx is None:
            return

        if selected_range is None or selected_range is False:
            fm_dialog = Frame_Range_Dialog(self.dm.total_frames, parent=self.main)
            fm_dialog.range_selected.connect(self._delete_track) # Recursive black magic
            fm_dialog.exec()
        else:
            start, end = selected_range
            self._save_state_for_undo()
            try:
                self.pred_data_array = delete_track(
                    self.pred_data_array, self.dm.current_frame_idx, selected_instance_idx, deletion_range=range(start, end+1))
            except ValueError as e:
                Loggerbox.error(self.main, "Deletion Error", str(e), exc=e)
            else:
                self.status_bar.show_message(f"Deleted inst {selected_instance_idx} between frame {start} to {end}.")
                self.display_current_frame()

    def _swap_track_single(self, swap_target:Tuple[int,int]=None):
        if self.pred_data_array is None:
            return
        if self.dm.dlc_data.instance_count > 2 and not swap_target:
            colormap = self.plotter.get_current_color_map()
            inst_dialog = Instance_Selection_Dialog(self.pred_data_array.shape[1], colormap, dual_selection=True)
            inst_dialog.instances_selected.connect(self._swap_track_single)
            inst_dialog.exec()
        else:
            self._save_state_for_undo()
            try:
                self.pred_data_array = swap_track(self.pred_data_array, self.dm.current_frame_idx, swap_target=swap_target)
            except ValueError as e:
                Loggerbox.error(self.main, "Swap Error", str(e), exc=e)
                return
            self.display_current_frame()

    def _swap_track_free(self, selected_range:Optional[Tuple[int,int]]=None):
        if self.pred_data_array is None:
            return
        
        if not selected_range:
            fm_dialog = Frame_Range_Dialog(self.dm.total_frames, parent=self.main)
            fm_dialog.range_selected.connect(self._swap_track_free)
            fm_dialog.exec()
        else:
            self._swap_range = selected_range
            self._swap_track_free_worker()

    def _swap_track_free_worker(self, swap_target:Optional[Tuple[int,int]]=None):
        start, end = self._swap_range

        if self.dm.dlc_data.instance_count > 2 and not swap_target:
            colormap = self.plotter.get_current_color_map()
            inst_dialog = Instance_Selection_Dialog(self.pred_data_array.shape[1], colormap, dual_selection=True)
            inst_dialog.instances_selected.connect(self._swap_track_free_worker)
            inst_dialog.exec()
        else:
            self._save_state_for_undo()
            try:
                self.pred_data_array = swap_track(self.pred_data_array, self.dm.current_frame_idx, swap_range=range(start, end+1), swap_target=swap_target)
            except ValueError as e:
                Loggerbox.error(self.main, "Swap Error", str(e), exc=e)
            else:
                self.status_bar.show_message(f"Swap insts between frame {start} to {end}.")
                self.display_current_frame()

    def _swap_track_continous(self, swap_target:Optional[Tuple[int,int]]=None):
        if self.pred_data_array is None:
            return
        if self.dm.dlc_data.instance_count > 2 and not swap_target:
            colormap = self.plotter.get_current_color_map()
            inst_dialog = Instance_Selection_Dialog(self.pred_data_array.shape[1], colormap, dual_selection=True)
            inst_dialog.instances_selected.connect(self._swap_track_continous)
            inst_dialog.exec()
        else:
            self._save_state_for_undo()
            try:
                self.pred_data_array = swap_track(self.pred_data_array, self.dm.current_frame_idx, swap_range=[-1], swap_target=swap_target)
            except ValueError as e:
                Loggerbox.error(self.main, "Swap Error", str(e), exc=e)
                return
            self.display_current_frame()

    def _copy_inst(self):
        frame_idx = self.dm.current_frame_idx
        if self.pred_data_array is None:
            return
        selected_instance_idx = self._instance_multi_select()
        if selected_instance_idx is None:
            return
        self.inst_pastebin = self.pred_data_array[frame_idx, selected_instance_idx, :].copy()
        self.status_bar.show_message(f"Inst {selected_instance_idx} on frame {frame_idx} has been copied into the pastebin.")

    def _paste_inst(self):
        frame_idx = self.dm.current_frame_idx
        if self.pred_data_array is None:
            return
        if self.inst_pastebin is None:
            Loggerbox.warning(self.main, "Pose pastebin is still empty, no pose to paste.")
            return
        selected_instance_idx = self._instance_select_inverted()
        if selected_instance_idx is None:
            return
        self.pred_data_array[frame_idx, selected_instance_idx, :] = self.inst_pastebin.copy()
        self.display_current_frame()
        self.status_bar.show_message(f"Inst {selected_instance_idx} on frame {frame_idx} has been replaced by the pastebin pose.")

    def _interpolate_track(self):
        frame_idx = self.dm.current_frame_idx
        if self.pred_data_array is None:
            return
        selected_instance_idx = self._instance_multi_select()
        if selected_instance_idx is None:
            return
        self._save_state_for_undo()
        iter_frame_idx = frame_idx + 1
        frames_to_interpolate = []
        while np.all(np.isnan(self.pred_data_array[iter_frame_idx, selected_instance_idx, :])):
            frames_to_interpolate.append(iter_frame_idx)
            iter_frame_idx += 1
            if iter_frame_idx >= self.dm.total_frames:
                Loggerbox.info(self.main, "Interpolation Failed", "No valid subsequent keypoint data found for this instance to interpolate to.")
                return

        if not frames_to_interpolate:
            Loggerbox.info(self.main, "Interpolation Info", "No gaps found to interpolate for the selected instance.")
            return
        
        frames_to_interpolate.sort()
        self.pred_data_array = interpolate_track(self.pred_data_array, frames_to_interpolate, selected_instance_idx)
        self.display_current_frame()

    def _interpolate_all(self):
        if not self._track_edit_blocker():
            return

        if self.gview.sbox is None:
            Loggerbox.info(self.main, "No Instance Selected", "Please select a track to interpolate all frames for one instance.")
            return False
        
        max_gap_allowed, ok = QtWidgets.QInputDialog.getInt(
            self.main,"Set Max Gap For Interpolation","Will not interpolate gap beyond this limit, set to 0 to ignore the limit.",
            value=10, minValue=0, maxValue=1000
        )
        if not ok:
            Loggerbox.info(self.main, "Input Cancelled", "Max Gap input cancelled.")
            return

        self._save_state_for_undo()
        self.pred_data_array = interpolate_track_all(self.pred_data_array, self.gview.sbox.instance_id, max_gap_allowed)
        self.display_current_frame()
        self.reset_zoom()

    def _interpolate_missing_kp(self):
        if self.pred_data_array is None:
            return
        
        selected_instance_idx = self._instance_multi_select()
        if selected_instance_idx is None:
            return
        self._save_state_for_undo()
        self.pred_data_array = generate_missing_kp_for_inst(
            pred_data_array=self.pred_data_array,
            current_frame_idx=self.dm.current_frame_idx,
            selected_instance_idx=selected_instance_idx,
            canon_pose=self.dm.canon_pose)
        self.display_current_frame()

    def _interpolate_all_missing_kp(self):
        if self.pred_data_array is None:
            return
        
        self._save_state_for_undo()
        min_visible_kp = self.dm.dlc_data.num_keypoint // 2
        dialog = Keypoint_Num_Dialog(init_bp=min_visible_kp, max_bp=self.dm.dlc_data.num_keypoint )
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            min_bodyparts = dialog.bp_spin.value()
        self.pred_data_array = generate_missing_kp_batch(self.pred_data_array, self.dm.canon_pose, min_bodyparts)
        self.display_current_frame()

    def _generate_inst(self):
        if self.pred_data_array is None:
            return
        self._save_state_for_undo()

        current_frame_inst = get_instances_on_current_frame(self.pred_data_array, self.dm.current_frame_idx)
        missing_instances = [inst for inst in range(self.dm.dlc_data.instance_count) if inst not in current_frame_inst]
        if missing_instances is None:
            Loggerbox.info(self.main, "No Missing Instances", "No missing instances found in the current frame to fill.")
            return

        self.pred_data_array = generate_missing_inst(
            pred_data_array=self.pred_data_array,
            current_frame_idx=self.dm.current_frame_idx,
            missing_instances=missing_instances,
            angle_map_data=self.dm.angle_map_data,
            canon_pose=self.dm.canon_pose)
        self.display_current_frame()

    def _rotate_inst(self):
        if self.pred_data_array is None:
            return
        
        selected_instance_idx = self._instance_multi_select()
        if selected_instance_idx is None:
            return None, None
        _, local_coords = calculate_pose_centroids(self.pred_data_array, self.dm.current_frame_idx)
        local_x = local_coords[selected_instance_idx, 0::2]
        local_y = local_coords[selected_instance_idx, 1::2]
        current_rotation = np.degrees(calculate_pose_rotations(local_x, local_y, self.dm.angle_map_data))

        if np.isnan(current_rotation) or np.isinf(current_rotation):
            current_rotation = 0.0
        else:
            current_rotation = current_rotation % 360.0 
        self._save_state_for_undo()
        
        if selected_instance_idx is not None:
            self.rotation_dialog = Pose_Rotation_Dialog(selected_instance_idx, current_rotation, parent=self.main)
            self.rotation_dialog.rotation_changed.connect(self._on_rotation_changed)
            self.rotation_dialog.show()

    ###################################################################################################################################################

    def _on_canvas_rect_return(self, x1, y1, x2, y2):
        self.gview.sbox = None
        self.pred_data_array = clean_inconsistent_nans(self.pred_data_array)

        all_x_kps = self.pred_data_array[:,:,0::3]
        all_y_kps = self.pred_data_array[:,:,1::3]

        x_in_range = (all_x_kps >= x1) & (all_x_kps <= x2)
        y_in_range = (all_y_kps >= y1) & (all_y_kps <= y2)
        points_in_bbox_mask = x_in_range & y_in_range

        self.pred_data_array[np.repeat(points_in_bbox_mask, 3, axis=-1)] = np.nan
        self.refresh_and_display()

    def _on_keypoint_delete(self):
        if self.gview.is_kp_edit and self.gview.drag_kp:
            self._mark_refined()
            instance_id = self.gview.drag_kp.instance_id
            keypoint_id = self.gview.drag_kp.keypoint_id

            self._save_state_for_undo()
            self.pred_data_array[self.dm.current_frame_idx, instance_id, keypoint_id*3:keypoint_id*3+3] = np.nan

            self.status_bar.show_message(f"{self.dm.dlc_data.keypoints[keypoint_id]} of instance {instance_id} deleted.")
            self.gview.drag_kp = None
            self.refresh_and_display()

    def _on_rotation_changed(self, instance_idx, angle_delta: float):
        angle_delta = np.radians(angle_delta)
        self._save_state_for_undo()
        self.pred_data_array = rotate_selected_inst(self.pred_data_array, self.dm.current_frame_idx, instance_idx, angle_delta)
        self.display_current_frame()

    def _handle_outlier_mask_from_comp(self, outlier_mask:np.ndarray, delete:bool):
        frame_list = np.where(np.any(outlier_mask, axis=1))[0].tolist()
        self.dm.handle_cat_update("outlier", frame_list)
        if frame_list:
            self.dm.current_frame_idx = frame_list[0]
            self.outlier_mask = outlier_mask
        self.refresh_and_display()

        if delete:
            self._save_state_for_undo()
            self.pred_data_array = outlier_removal(self.pred_data_array, self.outlier_mask)
            if hasattr(self, "outlier_finder"):
                self.outlier_finder.pred_data_array = self.pred_data_array.copy()
            self.display_current_frame()
            self.dm.handle_cat_update("outlier", [])

    def _handle_config_from_config(self, new_config:Plot_Config):
        self.dm.plot_config = new_config
        self.display_current_frame()

    def _update_last_selected_inst(self, instance_id):
        self.last_selected_idx = instance_id

    def _update_keypoint_position(self, instance_id, keypoint_id, new_x, new_y):
        self._mark_refined()

        self._save_state_for_undo()
        frame_idx = self.dm.current_frame_idx
        current_conf = self.pred_data_array[frame_idx, instance_id, keypoint_id*3+2]
        self.pred_data_array[frame_idx, instance_id, keypoint_id*3] += new_x
        self.pred_data_array[frame_idx, instance_id, keypoint_id*3+1] += new_y
        if pd.isna(current_conf) and not (pd.isna(new_x) or pd.isna(new_y)):
            self.pred_data_array[frame_idx, instance_id, keypoint_id*3+2] = 1.0

        self.status_bar.show_message(f"{self.dm.dlc_data.keypoints[keypoint_id]} of instance {instance_id} moved by ({new_x}, {new_y})")
        self.refresh_ui()
        QTimer.singleShot(0, self.display_current_frame)

    def _update_instance_position(self, instance_id, dx, dy):
        self._mark_refined()

        self._save_state_for_undo()
        frame_idx = self.dm.current_frame_idx
        
        for kp_idx in range(self.pred_data_array.shape[2]//3): # Update all keypoints for the given instance in the current frame
            x_coord_idx, y_coord_idx = kp_idx * 3, kp_idx * 3 + 1
            current_x = self.pred_data_array[frame_idx, instance_id, x_coord_idx]
            current_y = self.pred_data_array[frame_idx, instance_id, y_coord_idx]

            if not pd.isna(current_x) and not pd.isna(current_y):
                self.pred_data_array[frame_idx, instance_id, x_coord_idx] = current_x + dx
                self.pred_data_array[frame_idx, instance_id, y_coord_idx] = current_y + dy
    
        self.status_bar.show_message(f"Instance {instance_id} moved by ({dx}, {dy})")
        self.refresh_ui()
        QTimer.singleShot(0, self.display_current_frame)

    ###################################################################################################################################################

    def _undo_changes(self):
        data_array = self.uno.undo(self.pred_data_array)
        if data_array is not None:
            self.pred_data_array = data_array
        self.refresh_and_display()

    def _redo_changes(self):
        data_array = self.uno.redo(self.pred_data_array)
        if data_array is not None:
            self.pred_data_array = data_array
        self.refresh_and_display()

    def _save_state_for_undo(self):
        self.uno.save_state_for_undo(self.pred_data_array)

    ###################################################################################################################################################

    def _update_roi(self) -> list:
        self.inst_count_per_frame_pred = get_instance_count_per_frame(self.pred_data_array)
        return list(np.where(np.diff(self.inst_count_per_frame_pred)!=0)[0]+1)

    def _instance_multi_select(self) -> Optional[int]:
        current_frame_inst = get_instances_on_current_frame(self.pred_data_array, self.dm.current_frame_idx)
        if current_frame_inst is None:
            return
        if len(current_frame_inst) == 1:
            return current_frame_inst[0]
        if self.gview.sbox is None:
            if self.last_selected_idx is None:
                Loggerbox.info(self.main, "No Instance Selected",
                    "When there are more than one instance present, "
                    "you need to click one of the instance bounding box to specify which to delete.")
                return
            else:
                return self.last_selected_idx
        return self.gview.sbox.instance_id
    
    def _instance_select_inverted(self) -> Optional[int]:
        current_frame_inst = get_instances_on_current_frame(self.pred_data_array, self.dm.current_frame_idx)
        if not current_frame_inst:
            return 0
        if len(current_frame_inst) < len(self.dm.dlc_data.individuals):
            available_inst = [inst for inst in range(self.dm.dlc_data.instance_count) if inst not in current_frame_inst]
            return available_inst[0]
        if self.gview.sbox is None:
            if self.last_selected_idx is None:
                Loggerbox.info(self.main, "No Instance Selected",
                    "Click the instance you wish to overwrite with the pasted pose.")
                return
            else:
                return self.last_selected_idx
        return self.gview.sbox.instance_id

    ##############################################################################################

    def _temporal_track_correct(self):
        if not self._track_edit_blocker():
            return

        self._save_state_for_undo()

        is_entertained = False
        current_crp_weight = (0.75, 0.15, 0.1)
        sigma, kappa = (75, 0.2), None
        min_sim, gap_thresh = 0.10, 0.10
        used_starts = []

        progress = Progress_Indicator_Dialog(0, self.dm.total_frames, "Supervised Track Fixing", "", self.main)
        self.tf = Track_Fixer(self.pred_data_array, self.dm.angle_map_data, progress,
                                crp_weight=current_crp_weight, cr_sigma=sigma, kappa=kappa, 
                                minimum_similarity=min_sim, gap_threshold=gap_thresh,
                                lookback_window=3, used_starts=used_starts)

        # while not is_entertained:

        #     self.tf = Track_Fixer(self.pred_data_array, self.dm.angle_map_data, progress,
        #                           crp_weight=current_crp_weight, cr_sigma=sigma, kappa=kappa, 
        #                           minimum_similarity=min_sim, gap_threshold=gap_thresh,
        #                           lookback_window=3, used_starts=used_starts)

        #     pred_data_array, blasted_frames, amongus_frames, frame_list, used_starts = self.tf.iter_orchestrator()
        #     dialog = Iteration_Review_Dialog(self.dm.dlc_data, self.vm.extractor, pred_data_array, frame_list, blasted_frames, amongus_frames, parent=self.main)
        #     dialog.exec()

        #     if dialog.was_cancelled:
        #         return

        #     corrected_pred, status_array, is_entertained = dialog.get_result()

        #     self.tf.process_labels(corrected_pred, frame_list, status_array)
        #     current_crp_weight, sigma, min_sim, gap_thresh, kappa = self.tf.get_params()

        pred_data_array, amongus_frames = self.tf.fit_full_video()
        dialog = Track_Correction_Dialog(
            self.dm.dlc_data, self.vm.extractor, pred_data_array, list(range(self.dm.total_frames)), [], amongus_frames, parent=self.main)

        dialog.pred_data_exported.connect(self._get_pred_data_from_manual_correction)
        dialog.exec()

    def _get_pred_data_from_manual_correction(self, pred_data_array, frame_tuple):
        self._save_state_for_undo()
        self.dm.dlc_data.pred_data_array = pred_data_array
        self.pred_data_array = self.dm.dlc_data.pred_data_array
        self.display_current_frame()