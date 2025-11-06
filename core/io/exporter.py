import os
import numpy as np
import cv2

from typing import List, Optional
from PySide6.QtWidgets import QProgressDialog
import traceback

from .csv_op import prediction_to_csv, csv_to_h5
from .io_helper import append_new_video_to_dlc_config
from utils.helper import crop_coords_to_array
from core.dataclass import Loaded_DLC_Data, Export_Settings

DEBUG = False

class Exporter:
    """A class to handle saving or merging predictions back to DLC"""
    Frame_CV2 = np.ndarray

    def __init__(self, dlc_data: Loaded_DLC_Data, export_settings: Export_Settings,
            frame_list: List[int], pred_data_array:Optional[np.ndarray]=None,
            progress_callback:Optional[QProgressDialog]=None,
            crop_coords:Optional[np.ndarray]=None,
            ):
        self.dlc_data = dlc_data
        self.export_settings = export_settings
        self.frame_list = frame_list
        self.pred_data_array = pred_data_array
        self.progress_callback = progress_callback
        self.crop_coords = crop_coords

        os.makedirs(self.export_settings.save_path, exist_ok=True)

    def export_data_to_DLC(self, frame_only:bool=False):
        if DEBUG:
            os.startfile(self.export_settings.save_path)
        self._extract_frame()
        if frame_only and not DEBUG:
            return
        self._extract_pred()

    def _extract_frame(self):
        try:
            if os.path.isdir(self.export_settings.video_filepath): # Loading DLC labels
                if self.export_settings.save_path == self.export_settings.video_filepath:
                    self.export_settings.video_filepath += "_cropped"
                    video_name = self.export_settings.video_name + "_cropped"
                else:
                    video_name = self.export_settings.video_name
                image_folder = self.export_settings.video_filepath
                img_exts = ('.png', '.jpg')
                image_files = sorted([
                        os.path.join(image_folder,f) for f in os.listdir(image_folder)
                        if f.lower().endswith(img_exts) and f.startswith("img")
                    ])
                for i, frame_idx in enumerate(self.frame_list):
                    image_output_path = os.path.join(self.export_settings.save_path, f"img{str(int(frame_idx)).zfill(8)}.png")
                    image_input_path = image_files[i]
                    frame = cv2.imread(image_input_path)

                    if self.crop_coords is not None:
                        frame = self._apply_crop(frame, frame_idx)

                    cv2.imwrite(image_output_path, frame)
                append_new_video_to_dlc_config(self.dlc_data.dlc_config_filepath, video_name)
                return
                
            cap = cv2.VideoCapture(self.export_settings.video_filepath)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video {self.export_settings.video_filepath}")

            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_needed = max(self.frame_list) if self.frame_list else -1
            append_new_video_to_dlc_config(self.dlc_data.dlc_config_filepath, self.export_settings.video_name)

            if self.progress_callback:
                self.progress_callback.setMaximum(len(self.frame_list))

            if len(self.frame_list) < 0.1 * total_video_frames: # sparse extraction
                for i, frame_idx in enumerate(self.frame_list):

                    if self.progress_callback:
                        self.progress_callback.setValue(i)
                        if self.progress_callback.wasCanceled():
                            cap.release()
                            self.progress_callback.close()
                            raise Exception("Frame extraction canceled by user.")

                    image_output_path = os.path.join(self.export_settings.save_path, f"img{str(int(frame_idx)).zfill(8)}.png")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                    if self.crop_coords is not None:
                        frame = self._apply_crop(frame, frame_idx)

                    if ret:
                        cv2.imwrite(image_output_path, frame)
                
                if self.progress_callback:
                    self.progress_callback.close()
            else:
                extracted_count = 0
                current_frame_idx = 0
                while current_frame_idx <= max_needed:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if current_frame_idx in self.frame_list:
                        image_path = f"img{str(current_frame_idx).zfill(8)}.png"
                        image_output_path = os.path.join(self.export_settings.save_path, image_path)

                        if self.crop_coords is not None:
                            frame = self._apply_crop(frame, current_frame_idx)

                        cv2.imwrite(image_output_path, frame)
                        extracted_count += 1

                        if self.progress_callback:
                            self.progress_callback.setValue(extracted_count)
                            if self.progress_callback.wasCanceled():
                                cap.release()
                                self.progress_callback.close()
                                raise Exception("Frame extraction canceled by user.")

                    current_frame_idx += 1

                cap.release()
                if self.progress_callback:
                    self.progress_callback.close()

                if extracted_count != len(self.frame_list):
                    raise Exception(f"Only extracted {extracted_count}/{len(self.frame_list)} frames.")

        except Exception as e:
            if 'cap' in locals():
                cap.release()
            if self.progress_callback:
                self.progress_callback.close()
            traceback.print_exc()
            raise RuntimeError(f"Error extracting frames: {e}")

    def _extract_pred(self):
        if self.pred_data_array is None:
            pred_data_array = self.dlc_data.pred_data_array[self.frame_list, :, :]
        else:
            pred_data_array = self.pred_data_array[self.frame_list, :, :] # (F, I, K*3)

        if self.crop_coords is not None:
            coords_array = crop_coords_to_array(self.crop_coords, pred_data_array.shape)
            pred_data_array = pred_data_array - coords_array
        try:
            csv_name = prediction_to_csv(self.dlc_data, pred_data_array, self.export_settings, self.frame_list)
            if not csv_name:
                raise RuntimeError("Error exporting predictions to csv.")

            if not csv_to_h5(self.export_settings.save_path, self.dlc_data.multi_animal, self.dlc_data.scorer, csv_name=csv_name):
                raise RuntimeError("Error transforming to h5.")
        except Exception as e:
            raise RuntimeError(f"Error extracting prediction: {e}") from e
        
    def _apply_crop(self, frame:Frame_CV2, frame_idx:int):
        if self.crop_coords is None:
            return frame
        x1, y1, x2, y2 = self.crop_coords[frame_idx]
        return frame[y1:y2, x1:x2]