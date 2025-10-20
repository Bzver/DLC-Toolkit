import os
import numpy as np
import cv2

from typing import Tuple, List, Optional, Dict

from .csv_op import prediction_to_csv, csv_to_h5
from ui import Progress_Indicator_Dialog
from core.dataclass import Loaded_DLC_Data, Export_Settings

DEBUG = False

class Exporter:
    """A class to handle saving or merging predictions back to DLC"""
    Frame_CV2 = np.ndarray

    def __init__(self, dlc_data: Loaded_DLC_Data, export_settings: Export_Settings,
            frame_list: List[int], pred_data_array:Optional[np.ndarray]=None,
            progress_callback:Optional[Progress_Indicator_Dialog]=None,
            crop_coords:Optional[Dict[int, Tuple[int, int, int, int]]]=None,
            ):
        self.dlc_data = dlc_data
        self.export_settings = export_settings
        self.frame_list = frame_list
        self.pred_data_array = pred_data_array
        self.progress_callback = progress_callback
        self.crop_coords = crop_coords

        if DEBUG:
            self.export_settings.export_mode = "Append"

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
            cap = cv2.VideoCapture(self.export_settings.video_filepath)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video {self.export_settings.video_filepath}")

            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_needed = max(self.frame_list) if self.frame_list else -1

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

                    image_path = f"img{str(int(frame_idx)).zfill(8)}.png"
                    image_output_path = os.path.join(self.export_settings.save_path, image_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                    if self.crop_coords:
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

                        if self.crop_coords:
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
            raise RuntimeError(f"Error extracting frames: {e}")

    def _extract_pred(self):
        if self.pred_data_array is None:
            pred_data_array = self.dlc_data.pred_data_array[self.frame_list, :, :]
        else:
            pred_data_array = self.pred_data_array[self.frame_list, :, :] # (F, I, K*3)

        if self.crop_coords:
            coords_array_final = np.zeros_like(pred_data_array)
            sorted_coords = dict(sorted(self.crop_coords.items()))
            crop_offsets = np.array(list(sorted_coords.values()))
            
            x_per_frame = crop_offsets[:, 0][:, np.newaxis, np.newaxis]
            y_per_frame = crop_offsets[:, 1][:, np.newaxis, np.newaxis]

            coords_array_final[:, :, 0::3] = x_per_frame
            coords_array_final[:, :, 1::3] = y_per_frame
            
            pred_data_array = pred_data_array - coords_array_final

        try:
            if not prediction_to_csv(self.dlc_data, pred_data_array, self.export_settings, self.frame_list):
                raise RuntimeError("Error exporting predictions to csv.")

            csv_name = f"CollectedData_{self.dlc_data.scorer}"
            if not csv_to_h5(self.export_settings.save_path, self.dlc_data.multi_animal, self.dlc_data.scorer, csv_name=csv_name):
                raise RuntimeError("Error transforming to h5.")
        except Exception as e:
            raise RuntimeError(f"Error extracting prediction: {e}") from e
        
    def _apply_crop(self, frame:Frame_CV2, frame_idx:int):
        if self.crop_coords is None or frame_idx not in self.crop_coords:
            return frame
        x1, y1, x2, y2 = self.crop_coords[frame_idx]
        return frame[y1:y2, x1:x2]