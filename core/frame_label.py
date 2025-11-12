import numpy as np

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import QMessageBox

from ui import Menu_Widget, Video_Player_Widget, Pose_Rotation_Dialog, Status_Bar
from utils.helper import frame_to_pixmap, calculate_snapping_zoom_level
from .data_man import Data_Manager
from .video_man import Video_Manager
from .edit_man import Keypoint_Edit_Manager
from .tool import Outlier_Finder, Canvas, Prediction_Plotter
from .dataclass import Plot_Config, Plotter_Callbacks

class Frame_Label:
    def __init__(self,
                 data_manager: Data_Manager,
                 video_manager: Video_Manager,
                 keypoint_manager: Keypoint_Edit_Manager,
                 video_play_widget: Video_Player_Widget,
                 status_bar: Status_Bar,
                 menu_slot_callback: callable,
                 plot_config_callback: callable,
                 parent: QtWidgets.QWidget):
        self.dm = data_manager
        self.vm = video_manager
        self.kem = keypoint_manager
        self.vid_play = video_play_widget
        self.status_bar = status_bar
        self.menu_slot_callback = menu_slot_callback
        self.plot_config_callback = plot_config_callback
        self.main = parent

        self._init_gview()
        self.reset_state()

    def activate(self, menu_widget:Menu_Widget):
        menu_widget.add_menu_from_config(self.labeler_menu_config)
        if self.gview is None:
            self._init_gview()
        self.vid_play.nav.set_marked_list_name("ROI")
        self.vid_play.swap_display_for_graphics_view(self.gview)

    def deactivate(self, menu_widget:Menu_Widget):
        self._remove_menu(menu_widget)

        if not np.array_equal(self.kem.pred_data_array, self.dm.dlc_data.pred_data_array, equal_nan=True):
            self.save_prediction()
        
    def _remove_menu(self, menu_widget:Menu_Widget):
        for menu in self.labeler_menu_config.keys():
            menu_widget.remove_entire_menu(menu)

    def reset_state(self):
        self.open_outlier = False
        self.skip_outlier_clean= False
        self.is_saved = True

        self.plotter_callback = Plotter_Callbacks(
            keypoint_coords_callback = self._update_keypoint_position,
            keypoint_object_callback = self.gview.set_dragged_keypoint,
            box_coords_callback = self._update_instance_position,
            box_object_callback = self.gview._handle_box_selection
        )

        self.labeler_menu_config = {
            "View":{
                "buttons": [
                    ("Toggle Zoom Mode (Z)", self.toggle_zoom_mode, {"checkable": True, "checked": False}),
                    ("Reset Zoom", self.reset_zoom),
                ]
            },
            "Refine": {
                "buttons": [
                    {
                        "submenu": "Interpolate",
                        "items": [
                            ("Interpolate Selected Instance on Current Frame (T)", self.interpolate_track),
                            ("Interpolate Missing Keypoints for Selected Instance (Shift+T)", self.interpolate_missing_kp),
                            ("Interpolate Selected Instance Across All Frames", self._interpolate_all),     
                        ]
                    },
                    {
                        "submenu": "Delete",
                        "items": [
                            ("Delete Selected Instance On Current Frame (X)", self.delete_track),
                            ("Delete All Prediction Inside Selected Area", self._designate_no_mice_zone),
                        ]
                    },
                    {
                        "submenu": "Swap",
                        "items": [
                            ("Swap Instances On Current Frame (W)", self.swap_track_single),
                            ("Swap Until The End (Shift + W)", self.swap_track_continous)
                        ]
                    },
                    ("Correct Track Using Temporal Consistency", self._temporal_track_correct),
                    ("Generate Instance (G)", self.generate_inst),
                    ("Rotate Selected Instance (R)", self.rotate_inst),
                ]
            },
            "Edit": {
                "buttons": [
                    ("Direct Keypoint Edit (Q)", self.direct_keypoint_edit),
                    ("Open Outlier Cleaning Menu", self._call_outlier_finder),
                    ("Remove Current Frame From Refine Task", self.toggle_frame_status),
                    ("Mark All As Refined", self._mark_all_as_refined),
                    ("Undo Changes (Ctrl+Z)", self.undo_changes),
                    ("Redo Changes (Ctrl+Y)", self.redo_changes),
                ]
            },
            "Save":{
                "buttons": [
                    ("Save Prediction", self.save_prediction),
                    ("Save Prediction Into CSV", self.save_prediction_as_csv),
                ]
            },
        }

        self.reset_zoom()
        self.refresh_ui()
            
    def init_loaded_vid(self):
        self.reset_zoom()

    def _init_gview(self):
        self.gview = Canvas(track_edit_callback=self.on_track_data_changed, parent=self.main)
        self.vid_play.nav.set_marked_list_name("ROI")
        self.vid_play.swap_display_for_graphics_view(self.gview)

    ###################################################################################################################################################

    def display_current_frame(self):
        self.gview.sbox = None # Ensure the selected instance is unselected

        frame = self.vm.get_frame(self.dm.current_frame_idx)
        if frame is None:
            self.gview.clear_graphic_scene()
            return

        self.gview.clear_graphic_scene()
        self._plot_current_frame(frame)

    def _initialize_plotter(self):
        current_frame_data = np.full((self.dm.dlc_data.instance_count, self.dm.dlc_data.num_keypoint*3), np.nan)
        self.plotter = Prediction_Plotter(
            dlc_data = self.dm.dlc_data, current_frame_data = current_frame_data,
            plot_config = self.dm.plot_config, graphics_scene = self.gview.gscene, plot_callback=self.plotter_callback)

    def _plot_current_frame(self, frame):
        if self.dm.dlc_data is not None:
            if not hasattr(self, "plotter"):
                self._initialize_plotter()

        pixmap, w, h = frame_to_pixmap(frame)
        pixmap_item = self.gview.gscene.addPixmap(pixmap)
        pixmap_item.setZValue(-1)
        self.gview.gscene.setSceneRect(0, 0, w, h)
        
        if self.dm.plot_config.auto_snapping:
            view_width, view_height = self.gview.get_graphic_scene_dim()
            current_frame_data = self.kem.get_current_frame_data(self.dm.current_frame_idx)
            if not np.all(np.isnan(current_frame_data)):
                self.gview.zoom_factor, center_x, center_y = \
                    calculate_snapping_zoom_level(current_frame_data, view_width, view_height)
                self.gview.centerOn(center_x, center_y)

        new_transform = QTransform()
        new_transform.scale(self.gview.zoom_factor, self.gview.zoom_factor)
        self.gview.setTransform(new_transform)

        if self.kem.pred_data_array is not None:
            self.plotter.current_frame_data = self.kem.get_current_frame_data(self.dm.current_frame_idx)
            self.plotter.plot_config = self.dm.plot_config
            self.plotter.plot_predictions()

        self.gview.update() # Force update of the graphics view
        self.vid_play.set_current_frame(self.dm.current_frame_idx) # Update slider handle's position

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
            self.dm.handle_cat_update("roi_change", self.kem.update_roi())
        else:
            self.vid_play.sld.set_frame_category(*self.dm.get_cat_metadata("marked"))
            self.vid_play.sld.set_frame_category(*self.dm.get_cat_metadata("refined"))

    ###################################################################################################################################################

    def determine_list_to_nav(self):
        if self.open_outlier:
            return self.dm.get_frames("outlier")
        else:
            return self.dm.determine_list_to_nav_flabel()

    def toggle_frame_status(self):
        self.dm.toggle_frame_status_flabel()
        self.refresh_ui()

    def _mark_refined(self):
        self.dm.mark_refined_flabel()

    def _mark_all_as_refined(self):
        self.dm.mark_all_refined_flabel()
        self.refresh_ui()

    def toggle_zoom_mode(self):
        self.gview.toggle_zoom_mode()
        self.navigation_title_controller()

    def toggle_snap_to_instances(self):
        self.dm.plot_config.auto_snapping = not self.dm.plot_config.auto_snapping
        self.plot_config_callback()
        self.display_current_frame()

    ###################################################################################################################################################

    def sync_menu_state(self, close_all:bool=False):
        self.open_outlier = False
        if close_all:
            self.open_outlier = False
            self.reset_zoom()

    def _call_outlier_finder(self):
        if not self.vm.check_status_msg():
            return
        if not self.kem.check_pred_data():
            return
        
        self.open_outlier = not self.open_outlier
        self.reset_zoom()

        if self.open_outlier:
            self.menu_slot_callback()
            self.outlier_finder = Outlier_Finder(
                self.kem.pred_data_array, angle_map_data=self.dm.angle_map_data,
                canon_pose=self.dm.canon_pose, parent=self.main)
            self.outlier_finder.mask_changed.connect(self._handle_outlier_mask_from_comp)
            self.outlier_finder.list_changed.connect(self._handle_frame_list_from_comp)
            self.vid_play.set_right_panel_widget(self.outlier_finder)
        else:
            self.vid_play.set_right_panel_widget(None)

    def _track_edit_blocker(self):
        if not self.kem.check_pred_data():
            return
        if self.gview.is_kp_edit:
            QMessageBox.warning(self.main, "Not Allowed", "Please finish editing keypoints before using this function.")
            return False
        if self.open_outlier:
            QMessageBox.warning(self.main, "Outlier Cleaning Pending",
            "An outlier cleaning operation is pending.\n"
            "Please dismiss the outlier widget first."
        )
            return False
        return True

    ###################################################################################################################################################

    def direct_keypoint_edit(self):
        if not self.kem.check_pred_data():
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
        self.kem._save_state_for_undo()
        QMessageBox.information(self.main, "Designate No Mice Zone", "Click and drag on the video to select a zone. Release to apply.")

    def _temporal_track_correct(self):
        if not self._track_edit_blocker():
            return
        self.kem.correct_track_using_temporal(self.dm.canon_pose, self.dm.angle_map_data)

    def delete_track(self):
        self.kem.del_trk(self.dm.current_frame_idx, self.gview.sbox)

    def swap_track_single(self):
        if self._tri_swap_not_implemented:
            self.kem.swp_trk(self.dm.current_frame_idx)

    def swap_track_continous(self):
        if self._tri_swap_not_implemented:
            self.kem.swp_trk(self.dm.current_frame_idx, [-1])

    def _tri_swap_not_implemented(self):
        if self.dm.dlc_data.instance_count > 2:
            QMessageBox.information(self.main, "Not Implemented",
                "Swapping while instance count is larger than 2 has not been implemented.")
            return False
        return True

    def interpolate_track(self):
        self.kem.intp_trk(self.dm.current_frame_idx, self.gview.sbox)

    def _interpolate_all(self):
        if not self._track_edit_blocker():
            return
        if self.kem.interpolate_all_for_inst(self.gview.sbox):
            self.reset_zoom()

    def interpolate_missing_kp(self):
        self.kem.intp_ms_kp(self.dm.current_frame_idx, self.gview.sbox,
                            self.dm.angle_map_data, self.dm.canon_pose)
        
    def generate_inst(self):
        self.kem.gen_inst(self.dm.current_frame_idx, self.dm.dlc_data.instance_count, self.dm.angle_map_data, self.dm.canon_pose)

    def rotate_inst(self):
        if not self.kem.check_pred_data():
            return
        selected_instance_idx, current_rotation = self.kem.rot_inst_prep(
            self.dm.current_frame_idx, self.gview.sbox, self.dm.angle_map_data)
        if selected_instance_idx:
            self.rotation_dialog = Pose_Rotation_Dialog(selected_instance_idx, current_rotation, parent=self.main)
            self.rotation_dialog.rotation_changed.connect(self._on_rotation_changed)
            self.rotation_dialog.show()

    ###################################################################################################################################################

    def on_track_data_changed(self):
        self.gview.sbox = None
        self.is_saved = False
        self.refresh_and_display()

    def on_keypoint_delete(self):
        if self.gview.is_kp_edit and self.gview.drag_kp:
            self._mark_refined()
            instance_id = self.gview.drag_kp.instance_id
            keypoint_id = self.gview.drag_kp.keypoint_id
            self.kem.del_kp(self.dm.current_frame_idx, instance_id, keypoint_id)
            self.status_bar.show_message(f"{self.dm.dlc_data.keypoints[keypoint_id]} of instance {instance_id} deleted.")
            self.gview.drag_kp = None
            self.is_saved = False
            self.refresh_and_display()

    def _on_rotation_changed(self, instance_idx, angle_delta: float):
        angle_delta = np.radians(angle_delta)
        self.kem.rot_inst(self.dm.current_frame_idx, instance_idx, angle_delta)

    def _handle_frame_list_from_comp(self, frame_list):
        self.dm.handle_cat_update("outlier", frame_list)
        if frame_list:
            self.dm.current_frame_idx = frame_list[0]
        self.refresh_and_display()

    def _handle_outlier_mask_from_comp(self, outlier_mask:np.ndarray):
        self.kem.del_outlier(outlier_mask)
        self.dm.handle_cat_update("outlier", [])

    def _handle_config_from_config(self, new_config:Plot_Config):
        self.dm.plot_config = new_config
        self.display_current_frame()

    def _update_keypoint_position(self, instance_id, keypoint_id, new_x, new_y):
        self._mark_refined()
        self.kem.update_kp_pos(
            self.dm.current_frame_idx, instance_id, keypoint_id, new_x, new_y)
        self.status_bar.show_message(f"{self.dm.dlc_data.keypoints[keypoint_id]} of instance {instance_id} moved by ({new_x}, {new_y})")
        self.is_saved = False
        self.refresh_ui()
        QTimer.singleShot(0, self.display_current_frame)

    def _update_instance_position(self, instance_id, dx, dy):
        self._mark_refined()
        self.kem.update_inst_pos(self.dm.current_frame_idx, instance_id, dx, dy)
        self.status_bar.show_message(f"Instance {instance_id} moved by ({dx}, {dy})")
        self.is_saved = False
        self.refresh_ui()
        QTimer.singleShot(0, self.display_current_frame)

    ###################################################################################################################################################

    def save_prediction(self):
        self.kem.check_pred_data()
        is_label_file = True if self.vm.image_files else False
        save_path, status, msg = self.dm.save_pred(self.kem.pred_data_array, is_label_file)
        if not status:
            QMessageBox.critical(self.main, "Saving Error", f"An error occurred during saving: {msg}")
            return
        self._reload_prediction(save_path)
        QMessageBox.information(self.main, "Save Successful", str(msg))
        self.is_saved = True

    def save_prediction_as_csv(self):
        self.dm.save_pred_to_csv()

    def undo_changes(self):
        self.kem.undo()
        self.refresh_and_display()
        self.is_saved = False

    def redo_changes(self):
        self.kem.redo()
        self.refresh_and_display()
        self.is_saved = False

    def _reload_prediction(self, prediction_path):
        self.dm.reload_pred_to_dm(prediction_path)
        self.refresh_and_display()
        self.status_bar.show_message("Prediction successfully reloaded")
        self.kem.set_pred_data(self.dm.dlc_data.pred_data_array)
        if hasattr(self, 'plotter'):
            delattr(self, 'plotter')