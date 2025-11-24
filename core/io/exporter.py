import os
import numpy as np
import cv2

from typing import List, Optional
from PySide6.QtWidgets import QProgressDialog
import traceback

from .csv_op import prediction_to_csv, csv_to_h5
from .io_helper import append_new_video_to_dlc_config
from .frame_loader import Frame_Extractor
from utils.helper import crop_coord_to_array
from core.dataclass import Loaded_DLC_Data, Export_Settings

class Exporter:
    """A class to handle saving or merging predictions back to DLC"""
    Frame_CV2 = np.ndarray

    def __init__(self, dlc_data: Loaded_DLC_Data, export_settings: Export_Settings,
            frame_list: List[int], pred_data_array:Optional[np.ndarray]=None,
            progress_callback:Optional[QProgressDialog]=None,
            crop_coord:Optional[np.ndarray]=None,
            ):
        self.dlc_data = dlc_data
        self.export_settings = export_settings
        self.frame_list = frame_list
        self.pred_data_array = pred_data_array
        self.progress_callback = progress_callback
        self.crop_coord = crop_coord
        self.extractor = Frame_Extractor(self.export_settings.video_filepath)

        os.makedirs(self.export_settings.save_path, exist_ok=True)

    def export_data_to_DLC(self, frame_only:bool=False):
        self._extract_frame()
        if frame_only:
            return
        self._extract_pred()

    def export_frame_to_video(self):
        try:
            self._continuous_frame_extraction(to_video=True)
        except Exception as e:
            if self.progress_callback:
                self.progress_callback.close()
            traceback.print_exc()
            raise RuntimeError(f"Error extracting frames: {e}")

    def _extract_frame(self):
        try:
            if os.path.isdir(self.export_settings.video_filepath): # Loading DLC labels
                self._extract_dlc_label()
                return

            total_video_frames = self.extractor.get_total_frames()
            if os.path.dirname(self.dlc_data.dlc_config_filepath) in self.export_settings.save_path:
                append_new_video_to_dlc_config(self.dlc_data.dlc_config_filepath, self.export_settings.video_name)

            if len(self.frame_list) < total_video_frames // 10: # sparse extraction
                self._sparse_frame_extraction()
            else:
                self._continuous_frame_extraction()

        except Exception as e:
            if self.progress_callback:
                self.progress_callback.close()
            traceback.print_exc()
            raise RuntimeError(f"Error extracting frames: {e}")

    def _extract_pred(self):
        if self.pred_data_array is None:
            pred_data_array = self.dlc_data.pred_data_array[self.frame_list, :, :]
        else:
            pred_data_array = self.pred_data_array[self.frame_list, :, :] # (F, I, K*3)

        if self.crop_coord is not None:
            coords_array = crop_coord_to_array(self.crop_coord, pred_data_array.shape, self.frame_list)
            pred_data_array = pred_data_array - coords_array
        try:
            csv_name = prediction_to_csv(self.dlc_data, pred_data_array, self.export_settings, self.frame_list)
            if not csv_name:
                raise RuntimeError("Error exporting predictions to csv.")

            if not csv_to_h5(self.export_settings.save_path, self.dlc_data.multi_animal, self.dlc_data.scorer, csv_name=csv_name):
                raise RuntimeError("Error transforming to h5.")
        except Exception as e:
            raise RuntimeError(f"Error extracting prediction: {e}") from e
        
    def _apply_crop(self, frame:Frame_CV2):
        if self.crop_coord is None:
            return frame
        x1, y1, x2, y2 = self.crop_coord
        return frame[y1:y2, x1:x2]
    
    def _extract_dlc_label(self):
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
            frame = self._apply_crop(frame)

            cv2.imwrite(image_output_path, frame)
        append_new_video_to_dlc_config(self.dlc_data.dlc_config_filepath, video_name)

    def _sparse_frame_extraction(self):
        if self.progress_callback:
            self.progress_callback.setMaximum(len(self.frame_list))
        for i, frame_idx in enumerate(self.frame_list):
            if self.progress_callback:
                self.progress_callback.setValue(i)
                if self.progress_callback.wasCanceled():
                    self.progress_callback.close()
                    raise Exception("Frame extraction canceled by user.")

            image_output_path = os.path.join(self.export_settings.save_path, f"img{str(int(frame_idx)).zfill(8)}.png")
            frame = self.extractor.get_frame(frame_idx)
            if frame is None:
                continue

            frame = self._apply_crop(frame)
            cv2.imwrite(image_output_path, frame)
        
        if self.progress_callback:
            self.progress_callback.close()
    
    def _continuous_frame_extraction(self, to_video:bool=False):
        if self.progress_callback:
            self.progress_callback.setMaximum(max(self.frame_list))

        extracted_count = 0
        current_frame_idx = 0
        writer = None
        frame_set = set(self.frame_list)

        self.extractor.start_sequential_read(0)
        while current_frame_idx <= max(self.frame_list) if self.frame_list else -1:
            idx, frame = self.extractor.read_next_frame() 
            if idx != current_frame_idx:
                self.extractor.finish_sequential_read()
                raise RuntimeError(f"Mismatch between frame indices: {idx} != {current_frame_idx}")
            
            if frame is None:
                frame = np.zeros((*self.extractor.get_frame_dim(), 3), dtype=np.uint8)

            if current_frame_idx in frame_set:
                frame = self._apply_crop(frame)
                
                if to_video and not writer:
                    video_output_path = os.path.join(self.export_settings.save_path, "temp_extract.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(filename=video_output_path, fourcc=fourcc, fps=10.0, frameSize=frame.shape[1::-1])
                    if not writer.isOpened():
                        self.extractor.finish_sequential_read()
                        raise RuntimeError(f"Failed to open VideoWriter for {video_output_path}")
    
                if not to_video:
                    image_path = f"img{str(current_frame_idx).zfill(8)}.png"
                    image_output_path = os.path.join(self.export_settings.save_path, image_path)
                    cv2.imwrite(image_output_path, frame)
                else:
                    writer.write(frame)

                if self.progress_callback:
                    self.progress_callback.setValue(current_frame_idx)
                    if self.progress_callback.wasCanceled():
                        if writer:
                            writer.release()
                        self.extractor.finish_sequential_read()
                        self.progress_callback.close()
                        raise Exception("Frame extraction canceled by user.")
                    
            extracted_count += 1
            current_frame_idx += 1

        if writer:
            writer.release()
        self.extractor.finish_sequential_read()
        if self.progress_callback:
            self.progress_callback.close()

        if extracted_count != len(self.frame_list):
            raise Exception(f"Only extracted {extracted_count}/{len(self.frame_list)} frames.")