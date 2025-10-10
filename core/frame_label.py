import os
import numpy as np

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, QEvent, QTimer, Signal
from PySide6.QtGui import QShortcut, QKeySequence, QCloseEvent
from PySide6.QtWidgets import QMessageBox

import utils.helper as duh
from ui import Menu_Widget, Video_Player_Widget, Pose_Rotation_Dialog
from core.palette import NAV_COLOR_PALETTE as nvp, NAV_COLOR_PALETTE_FLAB as nvpl
from core.tool import (
    Outlier_Finder, Canonical_Pose_Dialog, Prediction_Plotter,
    Plot_Config_Menu, Canvas, navigate_to_marked_frame
)
from core import Data_Manager, Video_Manager, Keypoint_Edit_Manager
from core.dataclass import Plot_Config, Plotter_Callbacks, Nav_Callback

class Frame_Label(QtWidgets.QMainWindow):
    prediction_saved = Signal(str)          # Signal to emit the path of the saved prediction file
    refined_frames_exported = Signal(list)  # Signal to emit the refined_roi_frame_list back to Extractor

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Frame Labeler")
        self.setGeometry(100, 100, 1200, 960)

        self.dm = Data_Manager(
            init_vid_callback = self._initialize_loaded_video,
            refresh_callback = self._refresh_ui, parent=self)
        self.vm = Video_Manager(self)
        self.kem = Keypoint_Edit_Manager(
            edit_callback = self._on_track_data_changed,
            parent = self)
        
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
        
        self.gview = Canvas(track_edit_callback=self._on_track_data_changed, parent=self)
        self.vid_play.nav.set_marked_list_name("ROI")
        self.vid_play.swap_display_for_graphics_view(self.gview)

        self.app_layout.addWidget(self.vid_play)
        self._setup_shortcut()
        self.reset_state()

    def _setup_menu(self):
        self.menu_widget = Menu_Widget(self)
        self.setMenuBar(self.menu_widget)
        labeler_menu_config = {
            "File": {
                "buttons": [
                    ("Load Video", self.load_video),
                    ("Load Prediction", self.load_prediction),
                    ("Load DLC Label Data", self.load_dlc_label_data),
                ]
            },
            "Labeler": {
                "display_name": "Refine",
                "buttons": [
                    ("Direct Keypoint Edit (Q)", self._direct_keypoint_edit),
                    {
                        "submenu": "Interpolate",
                        "display_name": "Interpolate",
                        "items": [
                            ("Interpolate Selected Instance on Current Frame (T)", self._interpolate_track),
                            ("Interpolate Missing Keypoints for Selected Instance (Shift + T)", self._interpolate_missing_kp),
                            ("Interpolate Selected Instance Across All Frames", self._interpolate_all),     
                        ]
                    },
                    {
                        "submenu": "Delete",
                        "display_name": "Delete",
                        "items": [
                            ("Delete Selected Instance On Current Frame (X)", self._delete_track),
                            ("Delete All Prediction Inside Selected Area", self._designate_no_mice_zone),
                        ]
                    },
                    {
                        "submenu": "Swap",
                        "display_name": "Swap",
                        "items": [
                            ("Swap Instances On Current Frame (W)", self._swap_track_single),
                            ("Swap Until The End (Shift + W)", self._swap_track_continous)
                        ]
                    },
                    {
                        "submenu": "Correct",
                        "display_name": "Auto Correction",
                        "items": [
                            ("Correct Track Using Temporal Consistency", self._temporal_track_correct),
                            ("Correct Track Using Idtrackerai Trajectories", self._idtrackerai_track_correct),
                        ]
                    },
                    ("Generate Instance (G)", self._generate_inst),
                    ("Rotate Selected Instance (R)", self._rotate_inst),
                ]
            },
            "View": {
                "buttons": [
                    ("Toggle Navigating Frames With Animal Count Changes", self._toggle_roi_nav, {"checkable": True, "checked": False}),
                    ("Toggle Zoom Mode (Z)", self._toggle_zoom_mode, {"checkable": True, "checked": False}),
                    ("Toggle Snap to Instances (E)", self._toggle_snap_to_instances, {"checkable": True, "checked": False}),
                    ("View Canonical Pose", self.view_canonical_pose),
                    ("Outliers Menu", self._call_outlier_finder),
                    ("Reset Zoom", self._reset_zoom),
                ]
            },
            "Preference": {
                "buttons": [
                    ("Undo Changes", self._undo_changes),
                    ("Redo Changes", self._redo_changes),
                    ("Remove Current Frame From Refine Task", self._unmark_frame),
                    ("Mark All As Refined", self._mark_all_as_refined),
                    ("Plot Config Menu", self.open_plot_config_menu),
                ]
            },
            "Save": {
                "buttons": [
                    ("Save Prediction", self.save_prediction),
                    ("Save Prediction Into CSV", self.save_prediction_as_csv)
                ]
            }
        }
        self.menu_widget.add_menu_from_config(labeler_menu_config)

    def _setup_shortcut(self):
        QShortcut(QKeySequence(Qt.Key_Left | Qt.ShiftModifier), self).activated.connect(lambda: self._change_frame(-10))
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(lambda: self._change_frame(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self._change_frame(1))
        QShortcut(QKeySequence(Qt.Key_Right | Qt.ShiftModifier), self).activated.connect(lambda: self._change_frame(10))
        QShortcut(QKeySequence(Qt.Key_Up), self).activated.connect(lambda:self._navigate_prev)
        QShortcut(QKeySequence(Qt.Key_Down), self).activated.connect(lambda:self._navigate_next)
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.vid_play.sld.toggle_playback)

        QShortcut(QKeySequence(Qt.Key_W), self).activated.connect(self._swap_track_single)
        QShortcut(QKeySequence(Qt.Key_X), self).activated.connect(self._delete_track)
        QShortcut(QKeySequence(Qt.Key_W | Qt.ShiftModifier), self).activated.connect(self._swap_track_continous)
        QShortcut(QKeySequence(Qt.Key_T), self).activated.connect(self._interpolate_track)
        QShortcut(QKeySequence(Qt.Key_T | Qt.ShiftModifier), self).activated.connect(self._interpolate_missing_kp)
        QShortcut(QKeySequence(Qt.Key_G), self).activated.connect(self._generate_inst)
        QShortcut(QKeySequence(Qt.Key_R), self).activated.connect(self._rotate_inst)
        QShortcut(QKeySequence(Qt.Key_Q), self).activated.connect(self._direct_keypoint_edit)
        QShortcut(QKeySequence(Qt.Key_Backspace), self).activated.connect(self._on_keypoint_delete)

        QShortcut(QKeySequence(Qt.Key_Z | Qt.ControlModifier), self).activated.connect(self._undo_changes)
        QShortcut(QKeySequence(Qt.Key_Y | Qt.ControlModifier), self).activated.connect(self._redo_changes)
        QShortcut(QKeySequence(Qt.Key_S | Qt.ControlModifier), self).activated.connect(self.save_prediction)
        QShortcut(QKeySequence(Qt.Key_Z), self).activated.connect(self._toggle_zoom_mode)
        QShortcut(QKeySequence(Qt.Key_E), self).activated.connect(self._toggle_snap_to_instances)

    def reset_state(self):
        self.dm.reset_dm_vars()
        self.vm.reset_vm()
        self.kem.reset_kem_vars()
        self.vid_play.set_total_frames(0)

        self.open_config, self.open_outlier = False, False
        self.skip_outlier_clean, self.is_cleaned = False, False

        self.is_saved = True
        self.auto_snapping = False
        self.navigate_roi = False

        self.plotter_callback = Plotter_Callbacks(
            keypoint_coords_callback = self._update_keypoint_position,
            keypoint_object_callback = self.gview.set_dragged_keypoint,
            box_coords_callback = self._update_instance_position,
            box_object_callback = self.gview._handle_box_selection
        )
        self._reset_zoom()
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
            self.kem.set_pred_data(self.dm.dlc_data.pred_data_array)
        self.vm.init_extractor(video_path)
        self.dm.total_frames = self.vm.get_frame_counts()
        self.vid_play.set_total_frames(self.dm.total_frames)

        self._reset_zoom()

        self._refresh_and_display()
        print(f"Video loaded: {self.dm.video_file}")

    def load_prediction(self):
        if not self.vm.check_status_msg:
            return
        if self.dm.pred_file_dialog():
            self.kem.set_pred_data(self.dm.dlc_data.pred_data_array)
            self.display_current_frame()
            self._reset_zoom()

    def initialize_plotter(self):
        current_frame_data = np.full((self.dm.dlc_data.instance_count, self.dm.dlc_data.num_keypoint*3), np.nan)
        self.plotter = Prediction_Plotter(
            dlc_data = self.dm.dlc_data, current_frame_data = current_frame_data,
            plot_config = self.dm.plot_config, graphics_scene = self.gview.gscene, plot_callback=self.plotter_callback)

    def load_dlc_label_data(self):
        self.reset_state()
        image_folder = self.vm.load_label_folder_dialog()
        if not image_folder:
            return
        self.dm.load_dlc_label(image_folder)
        if not self.vm.image_files:
            QMessageBox.warning(self, "No Images", "No image files found in the selected folder.")
            return
        self.vm.load_img_from_folder(image_folder)
        self.dm.total_frames = len(self.vm.image_files)
        self.dm.frame_list = list(range(self.dm.total_frames))

        self.vid_play.set_total_frames(self.dm.total_frames) # Initialize slider range
        self.display_current_frame()
        self._reset_zoom()

    ###################################################################################################################################################

    def display_current_frame(self):
        self.gview.sbox = None # Ensure the selected instance is unselected

        frame = self.vm.get_frame(self.dm.current_frame_idx)
        if frame is None:
            self.gview.clear_graphic_scene()
            return

        self.gview.clear_graphic_scene() # Clear previous graphics items
        self._plot_current_frame(frame)

    def _plot_current_frame(self, frame):
        if self.dm.dlc_data is not None:
            if not hasattr(self, "plotter"):
                self.initialize_plotter()

        pixmap, w, h = duh.frame_to_pixmap(frame)
        
        # Add pixmap to the scene
        pixmap_item = self.gview.gscene.addPixmap(pixmap)
        pixmap_item.setZValue(-1)
        self.gview.gscene.setSceneRect(0, 0, w, h)
        
        if self.auto_snapping:
            view_width, view_height = self.gview.get_graphic_scene_dim()
            current_frame_data = self.kem.get_current_frame_data(self.dm.current_frame_idx)
            if not np.all(np.isnan(current_frame_data)):
                self.gview.zoom_factor, center_x, center_y = \
                    duh.calculate_snapping_zoom_level(current_frame_data, view_width, view_height)
                self.gview.centerOn(center_x, center_y)

        new_transform = QtGui.QTransform()
        new_transform.scale(self.gview.zoom_factor, self.gview.zoom_factor)
        self.gview.setTransform(new_transform)

        if self.kem.pred_data_array is not None:
            self.plotter.current_frame_data = self.kem.get_current_frame_data(self.dm.current_frame_idx)
            self.plotter.plot_config = self.dm.plot_config
            self.plotter.plot_predictions()

        self.gview.update() # Force update of the graphics view
        self.vid_play.set_current_frame(self.dm.current_frame_idx) # Update slider handle's position

    ###################################################################################################################################################

    def _refresh_and_display(self):
        self._refresh_ui()
        self.display_current_frame()

    def _refresh_ui(self):
        self._navigation_title_controller()
        self._refresh_slider()

    def _navigation_title_controller(self):
        title_text = self.dm.get_title_text(labeler=True, kp_edit=self.gview.is_kp_edit)
        self.vid_play.nav.setTitle(title_text)

        if self.open_config or self.navigate_roi:
            color = self.dm.determine_nav_color_flabel()
        else:
            color = self.dm.determine_nav_color_fview()
        if color:
            self.vid_play.nav.setTitleColor(color)
        else:
            self.vid_play.nav.setTitleColor("black")

    def _refresh_slider(self):
        if self.open_outlier:
            self.vid_play.sld.set_frame_category("Outlier frames", self.dm.outlier_frame_list, nvpl[2], 2)
        elif self.navigate_roi:
            self.dm.roi_frame_list = self.kem.update_roi()
            self.vid_play.sld.set_frame_category("ROI frames", self.dm.roi_frame_list, nvpl[1], 1)
        else:
            self.vid_play.sld.set_frame_category("Marked frames", self.dm.frame_list, nvp[1], 1)
            self.vid_play.sld.set_frame_category("Refined frames", self.dm.refined_frame_list, nvp[4], 4)

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
        navigate_to_marked_frame(self, list_to_nav, self.dm.current_frame_idx, self._handle_frame_change_from_comp, "prev")

    def _navigate_next(self):
        list_to_nav = self._determine_list_to_nav()
        navigate_to_marked_frame(self, list_to_nav, self.dm.current_frame_idx, self._handle_frame_change_from_comp, "next")

    def _determine_list_to_nav(self):
        if self.navigate_roi:
            return self.dm.roi_frame_list
        if self.open_outlier:
            return self.dm.outlier_frame_list
        return self.dm.frame_list

    def _toggle_frame_status(self):
        self.dm.toggle_frame_status_flabel()

    def _toggle_roi_nav(self):
        self.navigate_roi = not self.navigate_roi
        self._refresh_and_display()

    def _toggle_zoom_mode(self):
        self.gview.toggle_zoom_mode()
        self._navigation_title_controller()

    def _toggle_snap_to_instances(self):
        self.auto_snapping = not self.auto_snapping
        self.display_current_frame()

    def _reset_zoom(self):
        self.gview.reset_zoom()
        
    ###################################################################################################################################################

    def open_plot_config_menu(self):
        if not self.vm.check_status_msg():
            return
        if not self.dm.dlc_data:
            QtWidgets.QMessageBox.warning(self, "No Prediction", "No prediction has been loaded, please load prediction first.")
            return
        
        self.open_config = not self.open_config

        if self.open_config:
            self.open_outlier = False
            plot_config_widget = Plot_Config_Menu(plot_config=self.dm.plot_config)
            plot_config_widget.config_changed.connect(self._handle_config_from_config)
            self.vid_play.set_right_panel_widget(plot_config_widget)
        else:
            self.vid_play.set_right_panel_widget(None)

    def _call_outlier_finder(self):
        if not self.vm.check_status_msg():
            return
        if not self.kem.check_pred_data():
            return
        
        self.open_outlier = not self.open_outlier
        self._reset_zoom()

        if self.open_outlier:
            self.open_config = False
            self.outlier_finder = Outlier_Finder(self.kem.pred_data_array, canon_pose=self.dm.canon_pose, parent=self)
            self.outlier_finder.mask_changed.connect(self._handle_outlier_mask_from_comp)
            self.outlier_finder.list_changed.connect(self._handle_frame_list_from_comp)
            self.vid_play.set_right_panel_widget(self.outlier_finder)
        else:
            self.vid_play.set_right_panel_widget(None)

    def _track_edit_blocker(self):
        if not self.kem.check_pred_data():
            return
        if self.gview.is_kp_edit:
            QMessageBox.warning(self, "Not Allowed", "Please finish editing keypoints before using this function.")
            return False
        if self.open_outlier:
            QMessageBox.warning(self, "Outlier Cleaning Pending",
            "An outlier cleaning operation is pending.\n"
            "Please dismiss the outlier widget first."
        )
            return False
        return True

    def _suggest_outlier_clean(self):
        if not self.is_cleaned and not self.skip_outlier_clean:
            reply = QMessageBox.question(
                self, "Outliers Not Cleaned",
            "You are about to apply temporal correction on uncleaned tracking data.\n"
            "This may lead to error propagation or inaccurate smoothing.\n"
            "It is strongly recommended cleaning outliers first.\n\n"
            "Do you still want to continue without cleaning?",
            )

            if reply == QMessageBox.No:
                self._call_outlier_finder()
                return
            else:
                self.skip_outlier_clean = True

    ###################################################################################################################################################

    def _direct_keypoint_edit(self):
        if not self.kem.check_pred_data():
            return

        self.gview.toggle_kp_edit()
        self.dm.plot_config.edit_mode = self.gview.is_kp_edit
        self._navigation_title_controller()
        
        if self.gview.is_kp_edit:
            self.statusBar().showMessage(
                "Keypoint editing mode is ON. Drag to adjust. Press Backspace to delete a keypoint."
            )
        else:
            self.statusBar().showMessage("Keypoint editing mode is OFF.")

        self._navigation_title_controller()
        self.display_current_frame() # Refresh the frame to get the kp edit status through to plotter

    def _designate_no_mice_zone(self):
        if not self._track_edit_blocker():
            return
        self.gview.setCursor(Qt.CrossCursor)
        self.gview.is_drawing_zone = True
        self.kem._save_state_for_undo()
        QMessageBox.information(self, "Designate No Mice Zone", "Click and drag on the video to select a zone. Release to apply.")

    def _temporal_track_correct(self):
        if not self._track_edit_blocker():
            return
        self._suggest_outlier_clean()
        self.kem.correct_track_using_temporal()

    def _idtrackerai_track_correct(self):
        if not self._track_edit_blocker():
            return
        self._suggest_outlier_clean()
        self.kem.correct_track_using_idtrackerai()

    def _delete_track(self):
        self.kem.del_trk(self.dm.current_frame_idx, self.gview.sbox)

    def _swap_track_single(self):
        if self._tri_swap_not_implemented:
            self.kem.swp_trk(self.dm.current_frame_idx)

    def _swap_track_continous(self):
        if self._tri_swap_not_implemented:
            self.kem.swp_trk(self.dm.current_frame_idx, [-1])

    def _tri_swap_not_implemented(self):
        if self.dm.dlc_data.instance_count > 2:
            QMessageBox.information(self, "Not Implemented",
                "Swapping while instance count is larger than 2 has not been implemented.")
            return False
        return True

    def _interpolate_track(self):
        self.kem.intp_trk(self.dm.current_frame_idx, self.gview.sbox)

    def _interpolate_all(self):
        if not self._track_edit_blocker():
            return
        if self.kem.interpolate_all_for_inst(self.gview.sbox):
            self._reset_zoom()

    def _interpolate_missing_kp(self):
        self.kem.intp_ms_kp(self.dm.current_frame_idx, self.gview.sbox,
                            self.dm.angle_map_data, self.dm.canon_pose)
        
    def _generate_inst(self):
        self.kem.gen_inst(self.dm.current_frame_idx, self.dm.dlc_data.instance_count, self.dm.angle_map_data)

    def _rotate_inst(self):
        if not self.kem.check_pred_data():
            return
        selected_instance_idx, current_rotation = self.kem.rot_inst_prep(
            self.dm.current_frame_idx, self.gview.sbox, self.dm.angle_map_data)
        if selected_instance_idx:
            self.rotation_dialog = Pose_Rotation_Dialog(selected_instance_idx, current_rotation, parent=self)
            self.rotation_dialog.rotation_changed.connect(self._on_rotation_changed)
            self.rotation_dialog.show()

    ###################################################################################################################################################

    def _on_track_data_changed(self):
        self.gview.sbox = None
        self.is_saved = False
        self._refresh_and_display()

    def _on_keypoint_delete(self):
        if self.gview.is_kp_edit and self.gview.drag_kp:
            self._toggle_frame_status()
            instance_id = self.gview.drag_kp.instance_id
            keypoint_id = self.gview.drag_kp.keypoint_id
            self.kem.del_kp()
            print(f"{self.dm.dlc_data.keypoints[keypoint_id]} of instance {instance_id} deleted.")
            self.gview.drag_kp = None
            self.is_saved = False
            self._refresh_and_display(self.dm.current_frame_idx, instance_id, keypoint_id)

    def _on_rotation_changed(self, instance_idx, angle_delta: float):
        angle_delta = np.radians(angle_delta)
        self.kem.rot_inst(self.dm.current_frame_idx, instance_idx, angle_delta)

    def _handle_frame_change_from_comp(self, new_frame_idx: int):
        self.dm.current_frame_idx = new_frame_idx
        self._refresh_and_display()

    def _handle_frame_list_from_comp(self, frame_list:list):
        self.dm.outlier_frame_list = frame_list
        if frame_list:
            self.dm.current_frame_idx = frame_list[0]
        self._refresh_and_display()

    def _handle_outlier_mask_from_comp(self, outlier_mask:np.ndarray):
        self.kem.del_outlier(outlier_mask)
        self.dm.outlier_frame_list.clear()
        self.is_cleaned = True

    def _handle_config_from_config(self, new_config:Plot_Config):
        self.dm.plot_config = new_config
        self.display_current_frame()

    def _update_keypoint_position(self, instance_id, keypoint_id, new_x, new_y):
        self._toggle_frame_status()
        self.kem.update_kp_pos(
            self.dm.current_frame_idx, instance_id, keypoint_id, new_x, new_y)
        print(f"{self.dm.dlc_data.keypoints[keypoint_id]} of instance {instance_id} moved by ({new_x}, {new_y})")
        self.is_saved = False
        self._refresh_ui()
        QTimer.singleShot(0, self.display_current_frame)

    def _update_instance_position(self, instance_id, dx, dy):
        self._toggle_frame_status()
        self.kem.update_inst_pos(self.dm.current_frame_idx, instance_id, dx, dy)
        print(f"Instance {instance_id} moved by ({dx}, {dy})")
        self.is_saved = False
        self._refresh_ui()
        QTimer.singleShot(0, self.display_current_frame)

    ###################################################################################################################################################

    def view_canonical_pose(self):
        dialog = Canonical_Pose_Dialog(self.dm.dlc_data, self.dm.canon_pose)
        dialog.exec()

    def save_prediction(self):
        self.kem.check_pred_data()
        is_label_file = True if self.vm.image_files else False
        save_path, status, msg = self.dm.save_pred(self.kem.pred_data_array, is_label_file)
        if not status:
            QMessageBox.critical(self, "Saving Error", f"An error occurred during saving: {msg}")
            print(f"An error occurred during saving: {msg}")
            return
        self._reload_prediction(save_path)
        QMessageBox.information(self, "Save Successful", str(msg))
        self.is_saved = True

    def save_prediction_as_csv(self):
        self.dm.save_pred_to_csv()

    def _mark_all_as_refined(self):
        self.dm.mark_all_refined_flabel()
        self._refresh_ui()

    def _unmark_frame(self):
        if self.dm.current_frame_idx in self.dm.frame_list:
            self.dm.toggle_frame_status_fview()
        self._refresh_ui()

    def _undo_changes(self):
        self.kem.undo()
        self._refresh_and_display()
        self.is_saved = False

    def _redo_changes(self):
        self.kem.redo()
        self._refresh_and_display()
        self.is_saved = False

    def _reload_prediction(self, prediction_path):
        self.dm.reload_pred_to_dm(prediction_path)
        self._refresh_and_display()
        self.statusBar().showMessage("Prediction successfully reloaded")
        self.kem.set_pred_data(self.dm.dlc_data.pred_data_array)
        if hasattr(self, 'plotter'):
            delattr(self, 'plotter')

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            self._reset_zoom()
        super().changeEvent(event)
    
    def closeEvent(self, event: QCloseEvent):
        duh.handle_unsaved_changes_on_close(self, event, self.is_saved, self.save_prediction)

#######################################################################################################################################################

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = Frame_Label()
    window.show()
    app.exec()