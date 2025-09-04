import os
import numpy as np
import cv2

from typing import Tuple, List

from .csv_op import prediction_to_csv, csv_to_h5
from core.dataclass import Loaded_DLC_Data, Export_Settings

class Exporter:
    """A class to handle saving or merging predictions back to DLC"""
    def __init__(self, dlc_data: Loaded_DLC_Data, export_settings: Export_Settings,
        frame_list: List[int], pred_data_array: np.ndarray=None):
        self.dlc_data = dlc_data
        self.export_settings = export_settings
        self.frame_list = frame_list
        self.pred_data_array = pred_data_array

    def export_data_to_DLC(self, frame_only:bool=False) -> Tuple[bool, str]:
        status, msg = self._extract_frame()
        if not status:
            return False, msg

        if frame_only:
            msg = "Successfully exported marked frames to DLC for labeling!"
            return status, msg
    
        status, msg = self._extract_pred()
        if not status:
            return False, msg
        
        msg = "Successfully exported frames and prediction to DLC!"
        return status, msg

    def _extract_frame(self) -> Tuple[bool, str]:
        cap = cv2.VideoCapture(self.export_settings.video_filepath)
        if not cap.isOpened():
            return False, f"Error: Could not open video {self.export_settings.video_filepath}"
        
        frames_to_extract = set(self.frame_list)

        for frame in frames_to_extract:
            image_path = f"img{str(int(frame)).zfill(8)}.png"
            image_output_path = os.path.join(self.export_settings.save_path, image_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(image_output_path, frame)
        return True, "Success"

    def _extract_pred(self) -> Tuple[bool, str]:
        if self.pred_data_array is None:
            pred_data_array = self.dlc_data.pred_data_array[self.frame_list, :, :]
        else:
            pred_data_array = self.pred_data_array[self.frame_list, :, :]
        
        if not prediction_to_csv(self.dlc_data, pred_data_array, self.export_settings, self.frame_list):
            return False, "Error exporting predictions to csv."

        csv_name = f"CollectedData_{self.dlc_data.scorer}"
        if not csv_to_h5(self.export_settings.save_path, self.dlc_data.multi_animal, self.dlc_data.scorer, csv_name=csv_name):
            return False, "Error transforming to h5."
        
        return True, f"Label extracted to {self.export_settings.save_path}"