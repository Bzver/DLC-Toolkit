import os
import numpy as np
import cv2

from typing import Tuple, List, Optional

from .csv_op import prediction_to_csv, csv_to_h5
from ui import Progress_Indicator_Dialog
from core.dataclass import Loaded_DLC_Data, Export_Settings

class Exporter:
    """A class to handle saving or merging predictions back to DLC"""
    def __init__(self, dlc_data: Loaded_DLC_Data, export_settings: Export_Settings,
            frame_list: List[int], pred_data_array:Optional[np.ndarray]=None,
            progress_callback:Optional[Progress_Indicator_Dialog]=None
            ):
        self.dlc_data = dlc_data
        self.export_settings = export_settings
        self.frame_list = frame_list
        self.pred_data_array = pred_data_array
        self.progress_callback = progress_callback

        os.makedirs(self.export_settings.save_path, exist_ok=True)

    def export_data_to_DLC(self, frame_only:bool=False):
        self._extract_frame()
        if frame_only:
            return
        self._extract_pred()

    def _extract_frame(self):
        try:
            cap = cv2.VideoCapture(self.export_settings.video_filepath)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video {self.export_settings.video_filepath}")

            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_set = set(self.frame_list)
            max_needed = max(frame_set) if frame_set else -1

            if len(frame_set) < 0.1 * total_video_frames: # sparse extraction
                for i, frame in enumerate(frame_set):
                    if self.progress_callback:
                        self.progress_callback.setValue(i)
                        self.progress_callback.setMaximum(len(frame_set))
                        if self.progress_callback.wasCanceled():
                            break
                    image_path = f"img{str(int(frame)).zfill(8)}.png"
                    image_output_path = os.path.join(self.export_settings.save_path, image_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(image_output_path, frame)
                
                if self.progress_callback:
                    self.progress_callback.close()
            else:
                valid_frames = {f for f in frame_set if 0 <= f < total_video_frames}
                if self.progress_callback:
                    self.progress_callback.setMaximum(len(valid_frames))

                extracted_count = 0
                current_frame = 0

                while current_frame <= max_needed:
                    ret, img = cap.read()
                    if not ret:
                        break  # End of video

                    if current_frame in valid_frames:
                        image_path = f"img{str(current_frame).zfill(8)}.png"
                        image_output_path = os.path.join(self.export_settings.save_path, image_path)
                        cv2.imwrite(image_output_path, img)
                        extracted_count += 1

                        if self.progress_callback:
                            self.progress_callback.setValue(extracted_count)
                            if self.progress_callback.wasCanceled():
                                cap.release()
                                if self.progress_callback:
                                    self.progress_callback.close()
                                raise Exception("Frame extraction canceled by user.")

                    current_frame += 1

                cap.release()
                if self.progress_callback:
                    self.progress_callback.close()

                if extracted_count != len(valid_frames):
                    raise Exception(f"Only extracted {extracted_count}/{len(valid_frames)} frames.")

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
            pred_data_array = self.pred_data_array[self.frame_list, :, :]

        try:
            if not prediction_to_csv(self.dlc_data, pred_data_array, self.export_settings, self.frame_list):
                raise RuntimeError("Error exporting predictions to csv.")

            csv_name = f"CollectedData_{self.dlc_data.scorer}"
            if not csv_to_h5(self.export_settings.save_path, self.dlc_data.multi_animal, self.dlc_data.scorer, csv_name=csv_name):
                raise RuntimeError("Error transforming to h5.")
        except Exception as e:
            raise RuntimeError(f"Error extracting prediction: {e}") from e