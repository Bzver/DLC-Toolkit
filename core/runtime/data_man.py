import os
import pickle
import numpy as np
from PySide6.QtWidgets import QFileDialog, QDialog
from typing import Callable, Tuple, List, Optional, Dict

from .frame_man import Frame_Manager
from core.io import (
    Prediction_Loader, Exporter,
    backup_existing_prediction, save_predictions_to_new_h5, get_frame_list_from_h5,
    prediction_to_csv, remove_confidence_score)
from ui import Head_Tail_Dialog
from utils.helper import infer_head_tail_indices, build_angle_map
from utils.pose import calculate_canonical_pose, calculate_pose_bbox
from utils.logger import logger, Loggerbox
from utils.dataclass import Plot_Config, Blob_Config, Loaded_DLC_Data


class Data_Manager:
    HexColor = str

    def __init__(
            self,
            init_vid_callback:Callable[[str], None],
            refresh_callback:Callable[[], None],
            parent=None
            ):
        self.main = parent
        self.init_vid_callback = init_vid_callback
        self.refresh_callback = refresh_callback
        self.reset_dm()

    def reset_dm(self):
        self.fm = Frame_Manager(refresh_callback=self.refresh_callback)
        self.total_frames, self.current_frame_idx = 0, 0
        self.video_file, self.video_name = None, None
        self.dlc_data, self.label_file, self.canon_pose = None, None, None
        self.dlc_label_mode = False

        self.plot_config = Plot_Config(
            plot_opacity =1.0, point_size = 6.0, confidence_cutoff = 0.0, hide_text_labels = False, edit_mode = False,
            plot_labeled = True, plot_pred = True, navigate_labeled = False, auto_snapping = False, navigate_roi = False)

        # fview only
        self.blob_config:Blob_Config = None
        self.label_data_array, self.blob_array = None, None
        self.roi = None

        # flabel only
        self.angle_map_data = None
        self.inst_count_per_frame_pred = None

    def update_video_path(self, video_path:str):
        self.video_file = video_path
        self.video_name = os.path.splitext(os.path.basename(self.video_file))[0]

    def load_pred_to_dm(self, dlc_config_path:str, prediction_path:str):
        data_loader = Prediction_Loader(dlc_config_path, prediction_path)
        self.dlc_data = data_loader.load_data()
        self._init_loaded_data()

    def load_metadata_to_dm(self, dlc_config_path:str):
        data_loader = Prediction_Loader(dlc_config_path)
        self.dlc_data = data_loader.load_data(metadata_only=True)
        self.dlc_data.pred_frame_count = self.total_frames

    def load_dlc_label(self, image_folder:str, prediction_path:Optional[str]=None):
        """Load DLC Label without a preexisting prediction"""
        if not prediction_path:
            h5_candidates = [f for f in os.listdir(image_folder) if f.startswith("CollectedData_") and f.endswith(".h5")]
            if not h5_candidates:
                Loggerbox.warning(self.main, "No H5 File", "No 'CollectedData_*.h5' file found in the selected folder.")
                return
            prediction_path = os.path.join(image_folder, h5_candidates[0])

        self.dlc_label_mode = True
        logger.info("[MODE] Load DLC Label Mode.")
        self.dlc_data.prediction_filepath = prediction_path

        self.video_file = image_folder
        self.video_name = os.path.basename(image_folder)

        data_loader = Prediction_Loader(self.dlc_data.dlc_config_filepath, prediction_path)
        try:
            self.dlc_data = data_loader.load_data(force_load_pred=True)
        except Exception as e:
            Loggerbox.error(self.main, "Error Loading Prediction", f"Failed to load prediction: {e}", exc=e)
            return
        
        self.init_vid_callback(self.video_file)
        self._init_loaded_data()

    def _init_loaded_data(self):
        if self.label_file:
            self.load_labeled_overlay(self.label_file)
        self._init_canon_pose()

    def auto_loader_workspace(self):
        video_folder = os.path.dirname(self.video_file)

        if f"{self.video_name}_workspace.pkl" in os.listdir(video_folder):
            file_path = os.path.join(video_folder, f"{self.video_name}_workspace.pkl")
            self.load_workspace(file_path)
            return True

        return False

    def auto_loader(self):
        if self.dlc_data is not None and self.dlc_data.pred_data_array is not None:
           logger.info("[DATAMAN] Prediction already loaded in the current dlc_data. Skipping auto load.")
           return None, None
        video_folder = os.path.dirname(self.video_file)
        pred_candidates = []
        for f in os.listdir(video_folder):
            if f.endswith(".h5") and self.video_name in f:
                full_path = os.path.join(video_folder, f)
                pred_candidates.append(full_path)
        if not pred_candidates:
            logger.info("[DATAMAN] No prediction found to auto load.")
            return None, None
        newest_pred = max(pred_candidates, key=os.path.getmtime)
        logger.info(f"[DATAMAN] Automatically fetched the newest prediction: {newest_pred}")

        dlc_sub_folders = ["dlc-models-pytorch", "evaluation-results-pytorch", "labeled-data", "training-datasets", "videos"]
        found = False
        for fn in dlc_sub_folders:
            if fn in self.video_file:
                found = True
                break
        if not found:
            return None, newest_pred
        dlc_dir = self.video_file.split(fn)[0]
        dlc_config = os.path.join(dlc_dir, "config.yaml")
        logger.info(f"[DATAMAN] DLC config found: {dlc_config}")
        return dlc_config, newest_pred

    ###################################################################################################################################################

    def toggle_frame_status_fview(self):
        if self.has_current_frame_cat("labeled"):
            self.fm.move_frame("refined", "labeled", self.current_frame_idx)
            return

        if self.has_current_frame_cat("refined"):
            self.fm.move_frame("marked", "refined", self.current_frame_idx)
            return

        if self.current_frame_idx not in self.frames_in_any(self.fm.fview_cats):
            self.add_current_frame_cat("marked")
        else:
            self.remove_current_frame_cat("marked")
            self.remove_current_frame_cat("rejected")
            self.remove_current_frame_cat("approved")

        self.refresh_callback()

    def toggle_frame_status_flabel(self):
        if self.has_current_frame_cat("marked") or self.has_current_frame_cat("refined"):
            self.toggle_frame_status_fview()

    def mark_refined_flabel(self, frame_idx):
        self.fm.move_frame(new_category="refined", old_category="marked", frame_idx=frame_idx)

    def mark_all_refined_flabel(self):
        self.fm.move_category("refined", "marked")

    def get_inference_list(self) -> List[int]:
        return self.get_frames("marked")

    def get_frame_categories_fview(self) -> Dict[str, Tuple[str, List[int]]]:
        frame_options = {}
        pop_cats = self.fm.all_populated_categories()
        for cat in pop_cats:
            if cat in self.fm.fview_cats and cat != "labeled":
                frame_options[self.fm.get_display_name(cat)] = (cat, self.get_frames(cat))

        return frame_options
    
    def get_frame_categories_counting(self) -> Dict[str, Tuple[str, List[int]]]:
        frame_options = {}
        pop_cats = self.fm.all_populated_categories()
        for cat in pop_cats:
            if cat in self.fm.count_cats:
                frame_options[self.fm.get_display_name(cat)] = (cat, self.get_frames(cat))

        return frame_options
    
    def get_frame_categories_flabel(self) -> Dict[str, Tuple[str, List[int]]]:
        frame_options = {}
        pop_cats = self.fm.all_populated_categories()
        for cat in pop_cats:
            if cat in self.fm.flabel_cats:
                frame_options[self.fm.get_display_name(cat)] = (cat, self.get_frames(cat))

        return frame_options
    
    def clear_frame_cat(self, frame_category:str):
        self.fm.clear_category(frame_category)

    def clear_old_cat(self, clear_old:bool):
        if not clear_old:
            return
        self.fm.clear_category("marked")

    def get_frames(self, category:str) -> List[int]:
        return self.fm.get_frames(category)

    def frames_in_any(self, categories:List[str]) -> List[int]:
        return self.fm.frames_in_any(categories)

    def get_cat_in_group(self, group:str) -> List[str]:
        return self.fm.get_group_categories(group)

    def add_current_frame_cat(self, category:str):
        self.fm.add_frame(category, self.current_frame_idx)

    def remove_current_frame_cat(self, category:str):
        self.fm.remove_frame(category, self.current_frame_idx)

    def has_current_frame_cat(self, category:str) -> bool:
        return self.fm.has_frame(category, self.current_frame_idx)

    def add_frames(self, category:str, frame_list:List[int]):
        self.fm.add_frames(category, frame_list)

    def get_cat_metadata(self, category:str) -> Tuple[str, List[int], HexColor]:
        return self.fm.get_display_name(category), self.get_frames(category), self.fm.get_color(category)

    ###################################################################################################################################################

    def determine_nav_color_fview(self) -> HexColor:
        for cat in self.fm.fview_cats:
            if self.has_current_frame_cat(cat):
                return self.fm.get_color(cat)
        return "#000000"

    def determine_nav_color_counting(self) -> HexColor:
        if self.has_current_frame_cat("blob_merged"):
            return self.fm.get_color("blob_merged")
        for cat in self.fm.count_cats:
            if self.has_current_frame_cat(cat):
                return self.fm.get_color(cat)
        return "#000000"
    
    def determine_nav_color_flabel(self) -> HexColor:
        for cat in self.fm.flabel_cats:
            if self.has_current_frame_cat(cat):
                return self.fm.get_color(cat)
        return "#000000"

    def get_title_text(self, labeler:bool=False, kp_edit:bool=False):
        title_text = ""
        refd_count = self.fm.get_len("refined")
        pend_count = self.fm.get_len("marked")

        if labeler:
            if refd_count or pend_count:
                title_text += f"    Manual Refining Progress: {refd_count} | {refd_count+pend_count} Frames Refined    "
            if kp_edit and self.current_frame_idx:
                title_text += "    ----- KEYPOINTS EDITING MODE -----    "
        elif pend_count:
            title_text += f"    Marked Frame Count: {pend_count}    "
        return title_text

    def determine_list_to_nav_fview(self) -> List[int]:
        lb_list = self.get_frames("labeled")
        if self.plot_config.navigate_labeled and lb_list:
            return lb_list
        else:
            return self.frames_in_any(["marked", "rejected", "approved"])
        
    def determine_list_to_nav_flabel(self) -> List[int]:
        if self.plot_config.navigate_roi:
            return self.get_frames("roi_change")
        else:
            return self.frames_in_any(["marked", "refined"])

    ###################################################################################################################################################

    def load_labeled_overlay(self, label_file:str=None):
        """Load labeled frames as an separate overlay to the current prediction file."""
        if self.dlc_label_mode:
            return
        self.label_file = label_file
        self.label_data_array = np.full_like(self.dlc_data.pred_data_array, np.nan)
        data_loader = Prediction_Loader(self.dlc_data.dlc_config_filepath, label_file)
        label_data = data_loader.load_data()
        label_array = label_data.pred_data_array
        self.label_data_array[range(label_array.shape[0])] = label_array

        labeled_frame_list = np.where(np.any(~np.isnan(self.label_data_array), axis=(1, 2)))[0].tolist()
        self.fm.clear_category("labeled", no_refresh=True)
        self.fm.add_frames("labeled", labeled_frame_list, no_refresh=True)
        self.fm.remove_frames("marked", labeled_frame_list, no_refresh=True)
        self.fm.remove_frames("rejected", labeled_frame_list, no_refresh=True)
        self.fm.remove_frames("refined", labeled_frame_list, no_refresh=True)
        self.fm.remove_frames("approved", labeled_frame_list, no_refresh=True)

    def _init_canon_pose(self):
        head_idx, tail_idx = infer_head_tail_indices(self.dlc_data.keypoints)

        if head_idx is None or tail_idx is None:
            dialog = Head_Tail_Dialog(self.dlc_data.keypoints, self)
            if dialog.exec() == QDialog.Accepted:
                head_idx, tail_idx = dialog.get_selected_indices()
            else:
                Loggerbox.warning(self.main, "Head/Tail Not Set", "Canonical pose and angle map will not be available.")
                self.canon_pose = None
                self.angle_map_data = None
                return

        self.canon_pose, _ = calculate_canonical_pose(self.dlc_data.pred_data_array, head_idx, tail_idx)
        self.angle_map_data = build_angle_map(self.canon_pose, head_idx, tail_idx)

    def get_crop_coords_from_pred(self, frame_list:List[int], max_x:int, max_y:int) -> np.ndarray:
        coords_x = self.dlc_data.pred_data_array[frame_list, :, 0::3]
        coords_y = self.dlc_data.pred_data_array[frame_list, :, 1::3]
        x1_array, y1_array, x2_array, y2_array = calculate_pose_bbox(coords_x, coords_y, 30)
        crop_coords = np.column_stack(
            (np.nanmin(x1_array, axis=1), np.nanmin(y1_array, axis=1), np.nanmax(x2_array, axis=1), np.nanmax(y2_array, axis=1))
            )
        crop_coords = np.clip(crop_coords, 0, [max_x, max_y, max_x, max_y]).astype(int)
        return crop_coords

    def handle_mode_switch_fview_to_flabel(self):
        self.fm.clear_category("rejected")
        self.fm.move_category("marked", "approved")

    def handle_rurun_frame_tuple(self, frame_tuple:Tuple[List[int], List[int]]):
        self.fm.add_frames("approved", frame_tuple[0])
        self.fm.add_frames("rejected", frame_tuple[1])
        self.fm.remove_frames("marked", frame_tuple[0])
        self.fm.remove_frames("marked", frame_tuple[1])

    def handle_mark_gen_list(self, frame_list:List[int]):
        frame_set = set(frame_list) - set(self.get_frames("labeled")) - set(self.get_frames("rejected")) - set(self.get_frames("approved"))
        self.fm.add_frames("marked", list(frame_set))

    def handle_blob_counter_array(self, blob_array:np.ndarray) -> List[int]:
        self.blob_array = blob_array
        animal_0_list = list(np.where(blob_array[:, 0]==0)[0])
        animal_1_list = list(np.where(blob_array[:, 0]==1)[0])
        animal_n_list = list(np.where(blob_array[:, 0]>1)[0])
        blob_merged_list = list(np.where(blob_array[:, 1]==1)[0])
        self.fm.clear_group("counting")
        self.fm.add_frames("animal_0", animal_0_list)
        self.fm.add_frames("animal_1", animal_1_list)
        self.fm.add_frames("animal_n", animal_n_list)
        self.fm.add_frames("blob_merged", blob_merged_list)
        self.save_workspace()

        for frame_list in [blob_merged_list, animal_n_list, animal_1_list, animal_0_list]:
            if frame_list:
                return frame_list

    def handle_cat_update(self, category:str, frame_list:List[int]):
        self.fm.clear_category(category)
        self.fm.add_frames(category, frame_list)

    ###################################################################################################################################################

    def save_workspace(self):
        """Save the current workspace state (all vars from reset_dm_vars) to a pickle file."""
        default_name = f"{self.video_name}_workspace.pkl"
        file_path = os.path.join(os.path.dirname(self.video_file), default_name)

        workspace_state = {
            'total_frames': self.total_frames,
            'current_frame_idx': self.current_frame_idx,
            'video_file': self.video_file,
            'video_name': self.video_name,
            'dlc_data': self.dlc_data.to_dict()if self.dlc_data is not None else None,
            'canon_pose': self.canon_pose,
            'label_file': self.label_file, # corresponding DLC labels
            'frame_store': self.fm.to_dict(),
            'plot_config': self.plot_config.to_dict(),
            'blob_config': self.blob_config.to_dict() if self.blob_config is not None else None,
            'angle_map_data': self.angle_map_data,
            'inst_count_per_frame_pred': self.inst_count_per_frame_pred,
            'blob_array': self.blob_array,
            'roi': self.roi,
            'dlc_label_mode': self.dlc_label_mode,
        }
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(workspace_state, f)
        except Exception as e:
            Loggerbox.error(self.main, "Error Saving Workspace", f"Failed to save workspace:\n{e}", exc=e)

    def load_workspace(self, file_path:str):
        with open(file_path, 'rb') as f:
            workspace_state = pickle.load(f)

        # Restore all attributes
        self.dlc_label_mode = workspace_state.get('dlc_label_mode', False)
        self.total_frames = workspace_state.get('total_frames', 0)
        self.current_frame_idx = workspace_state.get('current_frame_idx', 0)
        self.video_file = workspace_state.get('video_file')
        self.video_name = workspace_state.get('video_name')
        self.canon_pose = workspace_state.get('canon_pose')
        self.angle_map_data = workspace_state.get('angle_map_data')
        self.inst_count_per_frame_pred = workspace_state.get('inst_count_per_frame_pred')
        self.blob_array = workspace_state.get('blob_array')
        self.label_file = workspace_state.get('label_file')
        self.roi = workspace_state.get('roi')

        if 'frame_store' in workspace_state.keys():
            self.fm = Frame_Manager.from_dict(workspace_state.get('frame_store'), self.refresh_callback)
            self.fm.move_category("animal_n", "animal_2")
        else: # For backward compatibility 
            self.fm = Frame_Manager(self.refresh_callback)
            self.fm.add_frames("marked", workspace_state.get('frame_list', []))
            self.fm.add_frames("refined", workspace_state.get('refined_frame_list', []))
            self.fm.add_frames("approved", workspace_state.get('approved_frame_list', []))
            self.fm.add_frames("rejected", workspace_state.get('rejected_frame_list', []))
            self.fm.add_frames("animal_0", workspace_state.get('animal_0_list', []))
            self.fm.add_frames("animal_1", workspace_state.get('animal_1_list', []))
            self.fm.add_frames("animal_n", workspace_state.get('animal_n_list', []))
            self.fm.add_frames("blob_merged", workspace_state.get('blob_merged_list', []))
            self.fm.add_frames("roi_change", workspace_state.get('roi_frame_list', []))
            self.fm.add_frames("outlier", workspace_state.get('outlier_frame_list', []))

        dlc_data = workspace_state.get('dlc_data')
        self.dlc_data = Loaded_DLC_Data.from_dict(dlc_data) if isinstance(dlc_data, dict) else dlc_data

        update_needed = False
        plot_config = workspace_state.get('plot_config')
        if isinstance(plot_config, dict):
            self.plot_config = Plot_Config.from_dict(plot_config) 
        else:
            self.plot_config = plot_config
            update_needed = True

        blob_config = workspace_state.get('blob_config')
        self.blob_config = Blob_Config.from_dict(blob_config) if isinstance(blob_config, dict) else blob_config
        if self.blob_config and self.roi is None:
            self.roi = np.array(self.blob_config.roi)

        if self.dlc_data is not None and self.dlc_data.pred_data_array is not None:
            self._init_loaded_data()
    
        if self.dlc_label_mode:
            self.load_dlc_label(self.video_file)
            return

        if not os.path.isfile(self.video_file):
            Loggerbox.error(self.main, "Video File Missing", f"Cannot find video at {self.video_file}")
            self._select_missing_video()
            if not self.video_file:
                return

        self.init_vid_callback(self.video_file)

        if update_needed:
            self.save_workspace()

    def _select_missing_video(self):
        file_dialog = QFileDialog(self.main)
        video_path, _ = file_dialog.getOpenFileName(self.main, "Load Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        self.video_file = video_path

    ###################################################################################################################################################

    def save_pred(self, pred_data_array:np.ndarray, save_path:str, to_dlc:bool=False):
        frame_list = None
        if to_dlc:
            frame_list = get_frame_list_from_h5(save_path)

        if os.path.isfile(save_path):
            backup_existing_prediction(save_path)

        data_array = remove_confidence_score(pred_data_array) if to_dlc else pred_data_array
        save_predictions_to_new_h5(self.dlc_data, data_array, save_path, frame_list, to_dlc)

    def save_pred_to_csv(self, pred_data_array:np.ndarray, save_path:str):
        if os.path.isfile(save_path):
            backup_existing_prediction(save_path)
        prediction_to_csv(self.dlc_data, pred_data_array, save_path)

    def reload_pred_to_dm(self, prediction_path:str):
        data_loader = Prediction_Loader(self.dlc_data.dlc_config_filepath, prediction_path)
        self.dlc_data = data_loader.load_data()
        self._init_canon_pose()

    ###################################################################################################################################################

    def save_to_dlc(self, save_folder:str, crop_mode:bool=False):
        label_file = os.path.join(save_folder, f"CollectedData_{self.dlc_data.scorer}.h5")

        crop_coord = self.roi if crop_mode else None
        refd_list = self.get_frames("refined")
        frame_list = refd_list if refd_list else self.frames_in_any(["marked", "approved"])

        if os.path.isfile(label_file):
            backup_existing_prediction(label_file)
            if self.label_data_array is None:
                self.load_labeled_overlay(label_file)
                lb_list = self.get_frames("labeled")
            self.label_data_array[frame_list, ...] = self.dlc_data.pred_data_array[frame_list, ...]
            frame_list = list(set(lb_list) | set(frame_list))

        pred_data_to_use = self.dlc_data.pred_data_array if self.label_data_array is None else self.label_data_array

        try:
            exporter = Exporter(
                self.dlc_data, save_folder, self.video_file, frame_list, pred_data_array=pred_data_to_use, crop_coord=crop_coord)
            exporter.export_data_to_DLC()
        except Exception as e:
            Loggerbox.error(self.main, "Error Save Data", f"Error saving data to DLC: {e}", exc=e)
        else:
            self.label_file = label_file
            self.load_labeled_overlay(self.label_file)