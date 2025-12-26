import os
import numpy as np
import cv2

from typing import List, Optional
from PySide6.QtWidgets import QProgressDialog

from .csv_op import prediction_to_csv, csv_to_h5
from .io_helper import append_new_video_to_dlc_config, generate_crop_coord_notations, remove_confidence_score
from .frame_loader import Frame_Extractor
from utils.helper import crop_coord_to_array, validate_crop_coord
from utils.logger import logger
from utils.dataclass import Loaded_DLC_Data


class Exporter:
    """A class to handle saving or merging predictions back to DLC"""
    Frame_CV2 = np.ndarray

    def __init__(
            self,
            dlc_data: Loaded_DLC_Data,
            save_folder:str,
            video_filepath: str,
            frame_list: List[int],
            pred_data_array:Optional[np.ndarray]=None,
            progress_callback:Optional[QProgressDialog]=None,
            crop_coord:Optional[np.ndarray]=None,
            ):
        logger.info(f"[EXPORTER] Initializing Exporter for save path: {save_folder}")
        self.dlc_data = dlc_data
        self.save_folder = save_folder
        self.video_filepath = video_filepath

        self.frame_list = frame_list
        self.pred_data_array = pred_data_array
        self.progress_callback = progress_callback

        self.video_name, _ = os.path.splitext(os.path.basename(self.video_filepath))

        self.crop_coord = validate_crop_coord(crop_coord)
        self.extractor = Frame_Extractor(self.video_filepath)

        os.makedirs(self.save_folder, exist_ok=True)

    def export_data_to_DLC(self, frame_only:bool=False) -> Optional[List[int]]:
        corrected_indices = self._extract_frame()
        if corrected_indices:
            self.frame_list = corrected_indices
            logger.info(f"[EXPORTER] Frame list updated with corrected indices. Total frames: {len(self.frame_list)}")
        else:
            logger.debug("[EXPORTER] No corrected indices returned from frame extraction.")

        if frame_only:
            return corrected_indices

        self._extract_pred()
        logger.info("[EXPORTER] Prediction data extracted successfully.")
        return corrected_indices

    def export_frame_to_video(self):
        logger.info("[EXPORTER] Starting frame export to video.")
        return self._continuous_frame_extraction(to_video=True)

    def _extract_frame(self) -> Optional[List[int]]:
        logger.debug(f"[EXPORTER] Entering _extract_frame. Video filepath: {self.video_filepath}")
        if os.path.isdir(self.video_filepath): # Loading DLC labels
            logger.info("[EXPORTER] Video filepath detected as a directory. Extracting DLC labels.")
            self._extract_dlc_label()
            return None # _extract_dlc_label handles the writing, no indices to return here

        total_video_frames = self.extractor.get_total_frames()
        logger.info(f"[EXPORTER] Total frames in video: {total_video_frames}")
        if os.path.dirname(self.dlc_data.dlc_config_filepath) in self.save_folder:
            append_new_video_to_dlc_config(self.dlc_data.dlc_config_filepath, os.path.basename(self.save_folder))
            logger.info(f"[EXPORTER] Appended new video '{os.path.basename(self.save_folder)}' to DLC config.")
        else:
            logger.debug("[EXPORTER] DLC config path not within save path. Skipping config update.")

        if not self.frame_list:
            logger.warning("[EXPORTER] Frame list is empty. No frames to extract.")
            return []

        if len(self.frame_list) < total_video_frames // 10: # sparse extraction
            logger.info(f"[EXPORTER] Performing sparse frame extraction for {len(self.frame_list)} frames.")
            corrected_indices = self._sparse_frame_extraction()
        else:
            logger.info(f"[EXPORTER] Performing continuous frame extraction for {len(self.frame_list)} frames.")
            corrected_indices = self._continuous_frame_extraction()

        return corrected_indices

    def _extract_pred(self):
        logger.debug("[EXPORTER] Entering _extract_pred.")
        if self.pred_data_array is None:
            pred_data_array = self.dlc_data.pred_data_array[self.frame_list, :, :]
            logger.debug("[EXPORTER] Using dlc_data.pred_data_array for predictions.")
        else:
            pred_data_array = self.pred_data_array[self.frame_list, :, :] # (F, I, K*3)
            logger.debug("[EXPORTER] Using provided pred_data_array for predictions.")

        if self.crop_coord is not None:
            logger.info(f"[EXPORTER] Applying crop coordinates {self.crop_coord} to prediction data.")
            coords_array = crop_coord_to_array(self.crop_coord, pred_data_array.shape)
            pred_data_array = pred_data_array - coords_array
            x1, y1, _, _ = self.crop_coord
            generate_crop_coord_notations((x1, y1), self.save_folder, self.frame_list)
            logger.debug("[EXPORTER] Generated crop coordinate notations.")
        else:
            logger.debug("[EXPORTER] No crop coordinates to apply to prediction data.")

        logger.info("[EXPORTER] Removing confidence scores from prediction data.")
        pred_data_array = remove_confidence_score(pred_data_array)

        csv_name = f"CollectedData_{self.dlc_data.scorer}.csv"
        save_path = os.path.join(self.save_folder, csv_name)
        prediction_to_csv(self.dlc_data, pred_data_array, save_path, self.frame_list, to_dlc=True)
        if not csv_name:
            raise RuntimeError("Error exporting predictions to csv.")
        logger.info(f"[EXPORTER] Predictions exported to CSV: {save_path}")

        csv_to_h5(save_path, self.dlc_data.multi_animal, self.dlc_data.scorer)
        logger.info("[EXPORTER] Converted CSV to H5 format.")
        
    def _apply_crop(self, frame:Frame_CV2):
        if self.crop_coord is None:
            logger.debug("[EXPORTER] No crop coordinates, returning original frame.")
            return frame
        try:
            x1, y1, x2, y2 = self.crop_coord
            cropped_frame = frame[y1:y2, x1:x2]
            logger.debug(f"[EXPORTER] Applied crop {self.crop_coord} to frame. New shape: {cropped_frame.shape}")
            return cropped_frame
        except Exception as e:
            logger.error(f"[EXPORTER] Error applying crop {self.crop_coord} to frame. Returning original frame. Error: {e}")
            return frame
    
    def _extract_dlc_label(self):
        logger.debug("[EXPORTER] Entering _extract_dlc_label.")
        if self.save_folder == self.video_filepath:
            self.video_filepath += "_cropped"
            logger.warning(f"[EXPORTER] Save path is same as video filepath. Appending '_cropped'. New video path: {self.video_filepath}")

        image_folder = self.video_filepath
        img_exts = ('.png', '.jpg')
        image_files = sorted([
                os.path.join(image_folder,f) for f in os.listdir(image_folder)
                if f.lower().endswith(img_exts) and f.startswith("img")
            ])
        logger.info(f"[EXPORTER] Found {len(image_files)} image files in {image_folder} for DLC label extraction.")

        for i, frame_idx in enumerate(self.frame_list):
            image_output_path = os.path.join(self.save_folder, f"img{str(int(frame_idx)).zfill(8)}.png")
            image_input_path = image_files[i]
            logger.debug(f"[EXPORTER] Processing DLC labeled frame {frame_idx} from {image_input_path}.")
            frame = cv2.imread(image_input_path)
            if frame is None:
                logger.warning(f"[EXPORTER] Failed to read image: {image_input_path}. Skipping frame {frame_idx}.")
                continue
            frame = self._apply_crop(frame)

            cv2.imwrite(image_output_path, frame)
            logger.debug(f"[EXPORTER] Wrote cropped DLC labeled image to {image_output_path}.")
        
        append_new_video_to_dlc_config(self.dlc_data.dlc_config_filepath, os.path.basename(self.save_folder))
        logger.info(f"[EXPORTER] Appended video '{os.path.basename(self.save_folder)}' to DLC config after DLC label extraction.")

    def _sparse_frame_extraction(self):
        logger.debug(f"[EXPORTER] Entering _sparse_frame_extraction for {len(self.frame_list)} frames.")
        if self.progress_callback:
            self.progress_callback.setMaximum(len(self.frame_list))
            logger.debug(f"[EXPORTER] Progress callback maximum set to {len(self.frame_list)} for sparse extraction.")

        extracted_indices = []

        for i, frame_idx in enumerate(self.frame_list):
            if self.progress_callback:
                self.progress_callback.setValue(i)
                if self.progress_callback.wasCanceled():
                    logger.warning("[EXPORTER] Sparse frame extraction canceled by user.")
                    self.progress_callback.close()
                    raise Exception("Frame extraction canceled by user.")

            image_output_path = os.path.join(self.save_folder, f"img{str(int(frame_idx)).zfill(8)}.png")
            logger.debug(f"[EXPORTER] Attempting to extract sparse frame {frame_idx}.")
            frame = self.extractor.get_frame(frame_idx)
            if frame is None:
                logger.warning(f"[EXPORTER] Frame {frame_idx} not found during sparse extraction. Skipping.")
                continue

            frame = self._apply_crop(frame)
            cv2.imwrite(image_output_path, frame)
            logger.debug(f"[EXPORTER] Wrote sparse frame {frame_idx} to {image_output_path}.")
            extracted_indices.append(frame_idx)
        
        extracted_set = set(extracted_indices)
        frame_set =set(self.frame_list)
        missing = sorted(frame_set - extracted_set)

        if self.progress_callback:
            self.progress_callback.close()

        if missing:
            logger.error(f"[EXPORTER] Frame count mismatch! Extracted frames: {len(extracted_set)} | Expected: {len(frame_set)}"
                         f"\nThe following frames are not extracted: {missing}")
            return list(extracted_indices)
    
    def _continuous_frame_extraction(self, to_video:bool=False) -> Optional[List[int]]:
        logger.debug(f"[EXPORTER] Entering _continuous_frame_extraction. to_video: {to_video}.")
        if self.progress_callback:
            max_frame_in_list = max(self.frame_list) if self.frame_list else 0
            self.progress_callback.setMaximum(max_frame_in_list)
            logger.debug(f"[EXPORTER] Progress callback maximum set to {max_frame_in_list} for continuous extraction.")

        empty_count = 0
        current_frame_idx = 0
        writer = None
        frame_set = set(self.frame_list)
        extracted_indices = []

        self.extractor.start_sequential_read(0)
        logger.info("[EXPORTER] Started sequential frame read for continuous extraction.")
        max_frame_in_list = max(self.frame_list) if self.frame_list else -1

        while current_frame_idx <= max_frame_in_list:
            result = self.extractor.read_next_frame()
            if result is None:
                logger.warning(f"[EXPORTER] End of video stream reached at frame {current_frame_idx} during continuous extraction.")
                break

            idx, frame = result
            if idx != current_frame_idx:
                self.extractor.finish_sequential_read()
                logger.critical(f"[EXPORTER] Mismatch between frame indices: Expected {current_frame_idx}, Got {idx}. Aborting continuous extraction.")
                raise RuntimeError(f"Mismatch between frame indices: {idx} != {current_frame_idx}")
            
            if frame is None:
                frame = np.zeros((*self.extractor.get_frame_dim(), 3), dtype=np.uint8)
                empty_count += 1
                logger.warning(f"[EXPORTER] Frame {current_frame_idx} is empty during continuous extraction. Substituted with placeholder.")

            if current_frame_idx in frame_set:
                frame = self._apply_crop(frame)

                if to_video and not writer:
                    video_output_path = os.path.join(self.save_folder, "temp_extract.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(filename=video_output_path, fourcc=fourcc, fps=10.0, frameSize=frame.shape[1::-1])
                    if not writer.isOpened():
                        self.extractor.finish_sequential_read()
                        logger.critical(f"[EXPORTER] Failed to open VideoWriter for {video_output_path}. Aborting continuous extraction.")
                        raise RuntimeError(f"Failed to open VideoWriter for {video_output_path}")
                    logger.info(f"[EXPORTER] VideoWriter opened for {video_output_path}.")

                if not to_video:
                    image_path = f"img{str(current_frame_idx).zfill(8)}.png"
                    image_output_path = os.path.join(self.save_folder, image_path)
                    cv2.imwrite(image_output_path, frame)
                else:
                    writer.write(frame)
                
                extracted_indices.append(current_frame_idx)

            if self.progress_callback:
                self.progress_callback.setValue(current_frame_idx)
                if self.progress_callback.wasCanceled():
                    logger.warning("[EXPORTER] Continuous frame extraction canceled by user.")
                    if writer:
                        writer.release()
                        logger.debug("[EXPORTER] VideoWriter released due to cancellation.")
                    self.extractor.finish_sequential_read()
                    self.progress_callback.close()
                    raise Exception("Frame extraction canceled by user.")

            current_frame_idx += 1

        if writer:
            writer.release()
            logger.debug("[EXPORTER] VideoWriter released after continuous extraction.")
        self.extractor.finish_sequential_read()
        logger.info("[EXPORTER] Finished sequential frame read after continuous extraction.")
        if self.progress_callback:
            self.progress_callback.close()
            logger.debug("[EXPORTER] Progress callback closed after continuous extraction.")

        extracted_set = set(extracted_indices)

        if empty_count:
            logger.warning(f"[EXPORTER] {empty_count} frames are empty and substituded with placeholders.")

        missing = sorted(frame_set - extracted_set)

        if missing:
            logger.error(f"[EXPORTER] Frame count mismatch! Extracted frames: {len(extracted_set)} | Expected: {len(frame_set)}"
                         f"\nThe following frames are not extracted: {missing}")
            return sorted(extracted_indices)