import os
import numpy as np
import cv2

from typing import Tuple, List, Optional
from PySide6.QtWidgets import QProgressBar

from .csv_op import prediction_to_csv, csv_to_h5
from core.dataclass import Loaded_DLC_Data, Export_Settings

class Exporter:
    """A class to handle saving or merging predictions back to DLC"""
    def __init__(self, dlc_data: Loaded_DLC_Data, export_settings: Export_Settings,
            frame_list: List[int], pred_data_array:Optional[np.ndarray]=None,
            progress_callback:Optional[QProgressBar]=None
            ):
        self.dlc_data = dlc_data
        self.export_settings = export_settings
        self.frame_list = frame_list
        self.pred_data_array = pred_data_array
        self.progress_callback = progress_callback

        os.makedirs(self.export_settings.save_path, exist_ok=True)

    def export_data_to_DLC(self, frame_only:bool=False) -> Tuple[bool, str]:
        self._extract_frame()
        if frame_only:
            return
        self._extract_pred()

    def _extract_frame(self) -> Tuple[bool, str]:
        try:
            cap = cv2.VideoCapture(self.export_settings.video_filepath)
            if not cap.isOpened():
                raise RuntimeError(f"Error: Could not open video {self.export_settings.video_filepath}")
            
            frames_to_extract = set(self.frame_list)
            if self.progress_callback:
                self.progress_callback.setMaximum(len(frames_to_extract))

            for i, frame in enumerate(frames_to_extract):
                if self.progress_callback:
                    self.progress_callback.setValue(i)
                image_path = f"img{str(int(frame)).zfill(8)}.png"
                image_output_path = os.path.join(self.export_settings.save_path, image_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(image_output_path, frame)
            
            if self.progress_callback:
                self.progress_callback.close()
                    
        except Exception as e:
            raise RuntimeError(f"Error extracting frame: {e}") from e

    def _extract_pred(self) -> Tuple[bool, str]:
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