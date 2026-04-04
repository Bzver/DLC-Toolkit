import os
import numpy as np
import cv2
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from typing import List, Optional

from .csv_op import prediction_to_csv, csv_to_h5
from .io_helper import append_new_video_to_dlc_config, generate_crop_coord_notations, remove_confidence_score
from .frame_loader import Frame_Extractor, Frame_Extractor_Img
from .helper_temp import Temp_Manager
from utils.helper import crop_coord_to_array, validate_crop_coord, frame_to_grayscale
from utils.logger import logger
from utils.dataclass import Loaded_DLC_Data, Exporter_Augments, Cutout_Augments


class DLC_Exporter:
    def __init__(
            self,
            dlc_data:Loaded_DLC_Data,
            save_folder:str,
            video_filepath:str,
            frame_list:List[int],
            pred_data_array:Optional[np.ndarray]=None,
            crop_coord:Optional[np.ndarray]=None,
            mask:Optional[np.ndarray]=None,
            grayscaling:bool=False,
            ):

        logger.info(f"[EXPORTER] Initializing Exporter for save path: {save_folder}")
        self.dlc_data = dlc_data
        self.save_folder = save_folder
        self.video_filepath = video_filepath
        self.frame_list = frame_list
        self.pred_data_array = pred_data_array

        if not os.path.exists(self.video_filepath):
            raise FileNotFoundError(f"Invalid video filepath: {self.video_filepath}")

        self.video_name, _ = os.path.splitext(os.path.basename(self.video_filepath))

        self.ea = Exporter_Augments(
            crop_coord = validate_crop_coord(crop_coord),
            mask = None if mask is None else True, grayscaling = grayscaling
        )

        os.makedirs(self.save_folder, exist_ok=True)

    def export_data_to_DLC(self) -> Optional[List[int]]:
        corrected_indices = self._extract_frame()
        if corrected_indices:
            self.frame_list = corrected_indices
            logger.info(f"[EXPORTER] Frame list updated with corrected indices. Total frames: {len(self.frame_list)}")
        else:
            logger.debug("[EXPORTER] No corrected indices returned from frame extraction.")

        self._extract_pred()
        logger.info("[EXPORTER] Prediction data extracted successfully.")
        return corrected_indices

    def _extract_frame(self) -> Optional[List[int]]:
        logger.debug(f"[EXPORTER] Entering _extract_frame. Video filepath: {self.video_filepath}")
        if os.path.dirname(self.dlc_data.dlc_config_filepath) in self.save_folder:
            append_new_video_to_dlc_config(self.dlc_data.dlc_config_filepath, os.path.basename(self.save_folder))
            logger.info(f"[EXPORTER] Appended new video '{os.path.basename(self.save_folder)}' to DLC config.")
        else:
            logger.debug("[EXPORTER] DLC config path not within save path. Skipping config update.")

        if not self.frame_list:
            logger.warning("[EXPORTER] Frame list is empty. No frames to extract.")
            return []

        fp = Frame_Exporter_Threaded(self.video_filepath, self.save_folder, self.frame_list)
        return fp.extract_frames(self.ea)

    def _extract_pred(self):
        logger.debug("[EXPORTER] Entering _extract_pred.")
        if self.pred_data_array is None:
            pred_data_array = self.dlc_data.pred_data_array[self.frame_list, :, :]
        else:
            pred_data_array = self.pred_data_array[self.frame_list, :, :]

        if self.ea.crop_coord is not None:
            logger.debug(f"[EXPORTER] Applying crop coordinates {self.ea.crop_coord} to prediction data.")
            coords_array = crop_coord_to_array(self.ea.crop_coord, pred_data_array.shape)
            pred_data_array = pred_data_array - coords_array
            x1, y1, _, _ = self.ea.crop_coord
            generate_crop_coord_notations((x1, y1), self.save_folder, self.frame_list)

        pred_data_array = remove_confidence_score(pred_data_array)

        csv_name = f"CollectedData_{self.dlc_data.scorer}.csv"
        save_path = os.path.join(self.save_folder, csv_name)
        prediction_to_csv(self.dlc_data, pred_data_array, save_path, self.frame_list, to_dlc=True)
        if not csv_name:
            raise RuntimeError("Error exporting predictions to csv.")
        logger.info(f"[EXPORTER] Predictions exported to CSV: {save_path}")

        csv_to_h5(save_path, self.dlc_data.multi_animal, self.dlc_data.scorer)
        logger.info("[EXPORTER] Converted CSV to H5 format.")


class Frame_Exporter_Threaded:
    def __init__(
            self,
            video_filepath:str,
            output_folder:str,
            frame_list:List[int],
            max_workers: int = 8,
            max_segment_size: int=5000
            ):
        self.video_filepath = video_filepath
        self.save_folder = output_folder
        self.frame_list = sorted(frame_list)
        self.max_workers = max_workers
        self.segment_size = min(max_segment_size, len(frame_list)//max_workers) 

        assert self.video_filepath != self.save_folder, "Invalid destination."
        self.worker_zero = None

        if os.path.isdir(self.video_filepath) or len(frame_list) < 100:
            self.worker_zero = Frame_Exporter(video_filepath, output_folder, frame_list)

        tm = Temp_Manager(video_filepath)
        self.temp_dir = tm.create("export")

    def extract_frames(self, aug, chunked_segments:List[List[int]]=[], use_cache:bool=False):
        if self.worker_zero:
            return self.worker_zero.extract_frames(aug, use_cache)

        if not chunked_segments:
            segments = self._task_splitter()
        else:
            segments = chunked_segments

        all_indices = []

        self._job_info_verbose(segments, aug)

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._worker_process_segment, idx, chunk, aug, self.temp_dir, to_video=False, use_cache=use_cache): idx
                    for idx, chunk in enumerate(segments)
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            indices = result
                            all_indices.extend(indices)
                    except Exception as e:
                        logger.error(f"[THREAD_EXP] Segment failed: {e}")
            
            logger.info(f"[THREAD_EXP] Image extraction complete. {len(all_indices)} frames.")
            return sorted(all_indices)

        except Exception as e:
            logger.error(f"[THREAD_EXP] Critical failure: {e}")

    def extract_frames_into_video(self, ea, video_name:str="temp_extract.mp4"):
        if self.worker_zero:
            return self.worker_zero.extract_frames_into_video(ea, video_name)
        if not self._check_ffmpeg():
            fe = Frame_Exporter(self.video_filepath, self.save_folder, self.frame_list)
            return fe.extract_frames_into_video(ea, video_name)
        
        video_output_path = os.path.join(self.save_folder, video_name)
        segments = self._task_splitter()
        all_indices = []

        self._job_info_verbose(segments, ea)

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._worker_process_segment, idx, chunk, ea, self.temp_dir, to_video=True, use_cache=False): idx
                    for idx, chunk in enumerate(segments)
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            indices = result
                            all_indices.extend(indices)
                    except Exception as e:
                        logger.error(f"[THREAD_EXP] Segment failed: {e}")

                self._concat_videos(video_output_path)

        except Exception as e:
            logger.error(f"[THREAD_EXP] Critical failure: {e}")

    def _job_info_verbose(self, segments, aug:Exporter_Augments|Cutout_Augments):
        line1 = f"FRAME EXPORTOR | {self.max_workers} workers | {len(segments)} segments | MODE: {aug.mode}"
        if aug.mode == "ea":
            line2 = f"Crop Coords: {aug.crop_coord} | Masking: {aug.mask is not None} | Grayscaling: {aug.grayscaling}"
        else:
            line2 = f"Cutout Dim: ({aug.cutout_dim}, {aug.cutout_dim}) | To Image: {aug.to_image} | Grayscaling: {aug.grayscaling}"

        content_width = max(len(line1), len(line2))
        border = "═" * (content_width + 2)

        logger.info(f"╔{border}╗")
        logger.info(f"║ {line1:^{content_width}} ║")
        logger.info(f"║ {line2:^{content_width}} ║")
        logger.info(f"╚{border}╝")

    def _worker_process_segment(
            self, 
            seg_idx: int, 
            frame_list: List[int], 
            aug: Exporter_Augments|Cutout_Augments,
            temp_dir: str,
            to_video: bool,
            use_cache: bool,
    ) -> Optional[List[int]]:

        pbar = tqdm(
            total=len(frame_list),
            desc=f"Segment {seg_idx} [{min(frame_list)}-{max(frame_list)}]",
            leave=False, 
            ncols=200
        )
        worker_fe = Frame_Exporter(self.video_filepath, temp_dir if to_video else self.save_folder, frame_list, pbar)
        try:
            if to_video:
                seg_filename = f"batch_extract_{seg_idx:04d}.mp4"
                indices = worker_fe.extract_frames_into_video(aug, video_name=seg_filename)
                return indices
            else:
                indices = worker_fe.extract_frames(aug, use_cache)
                return indices
        except Exception as e:
            logger.warning(f"[Segment_{seg_idx}] Failed: {e}")
            pbar.set_description(f"Segment {seg_idx} FAILED")
            if to_video:
                seg_path = os.path.join(temp_dir, seg_filename)
                os.remove(seg_path)

                actual_h, actual_w = worker_fe.get_vid_dim_to_export(aug)
                if self._generate_black_placeholder(seg_path, actual_w, actual_h, fps=10, frame_count=len(frame_list)):
                    logger.warning(f"[WORKER_{seg_idx}] Substituting failed segment with placeholder.")
                else:
                    logger.error(f"[WORKER_{seg_idx}] Failed to generate placeholder.")
        finally:
            pbar.close()

    def _task_splitter(self, probe_length:int=100) -> List[List[int]]:
        num_probe = (len(self.frame_list) - 1) // probe_length + 1

        chunks = []
        last_probe = "s" 
        last_start = 0

        for i in range(num_probe):
            start = i*probe_length
            end = min((i+1)*probe_length, len(self.frame_list))

            if self.frame_list[end - 1] - self.frame_list[start] >= 10 * probe_length: # Very sparse
                last_start = start
                last_probe = "s"
                chunks.append((start, end))
            elif self.frame_list[end - 1] - self.frame_list[start] < 2 * probe_length: # Very dense
                if last_probe == "d" and end - last_start < self.segment_size and len(chunks) > 1:
                    chunks.pop()
                    chunks.append((last_start, end))
                else:
                    last_start = start
                    last_probe = "d"
                    chunks.append((start, end))
            else: # Normie
                if end - last_start < self.segment_size//10 and len(chunks) > 1:
                    chunks.pop()
                    chunks.append((last_start, end))
                else:
                    last_start = start
                    chunks.append((last_start, end))
        
        chunked_list = [self.frame_list[start:end] for start, end in chunks]
            
        return chunked_list

    def _concat_videos(self, video_output_path):
        all_vid = []
        for f in os.listdir(self.temp_dir):
            if f.startswith("batch_extract") and f.endswith(".mp4"):
                f_path = os.path.join(self.temp_dir, f)
                if os.path.getsize(f_path) > 0:
                    all_vid.append(f_path)
                else:
                    logger.warning(f"[CONCAT] Skipping empty/corrupt segment: {f}")

        all_vid.sort()

        if not all_vid:
            return None
    
        logger.info(f"[CONCAT] About to concat {len(all_vid)} segments.")
        list_file = os.path.join(self.temp_dir, 'concat_list.txt')
        try:
            with open(list_file, 'w', encoding='utf-8') as f:
                for video in all_vid:
                    path = video.replace('\\', '/')
                    f.write(f"file '{path}'\n")

            cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_file, '-c', 'copy', '-y', video_output_path]
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr.decode('utf-8', errors='replace') if e.stderr else "Unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg failed in {self.temp_dir}:\n{stderr_msg}")
        finally:
            if os.path.exists(list_file):
                try:
                    os.remove(list_file)
                except Exception:
                    pass

    def _generate_black_placeholder(
        self, 
        output_path: str, 
        width: int, 
        height: int, 
        fps: int, 
        frame_count: int
    ) -> bool:
        duration = frame_count / fps
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', f'color=c=black:s={width}x{height}:r={fps}:d={duration}',
            '-c:v', 'mpeg4', '-pix_fmt', 'yuv420p',
            '-q:v', '5', '-preset', 'ultrafast',
            '-an', output_path
        ]
        try:
            subprocess.run(
                cmd,
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            return True
        except Exception as e:
            logger.error(f"Failed to generate black placeholder: {e}")
            return False

    @staticmethod
    def _check_ffmpeg():
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


class Frame_Exporter:
    def __init__(
            self,
            video_filepath:str,
            output_folder:str,
            frame_list:List[int],
            progress_bar:tqdm=None
            ):
        self.video_filepath = video_filepath
        self.save_folder = output_folder
        self.frame_list = frame_list

        self.homegrown_pbar = progress_bar is None
        if self.homegrown_pbar:
            self.pbar = tqdm(total=len(frame_list), desc=f"Extracting [{min(frame_list)}-{max(frame_list)}]", leave=False, ncols=200)
        else:
            self.pbar = progress_bar

        assert self.video_filepath != self.save_folder, "Invalid destination."

        self.label_mode = os.path.isdir(video_filepath)
        self.extractor = Frame_Extractor_Img(video_filepath) if self.label_mode else Frame_Extractor(video_filepath)
        self.extracted_indices = []

    def extract_frames(self, aug:Exporter_Augments|Cutout_Augments, use_cache:bool=False):
        if aug.mode == "ca" and not aug.to_image:
            d = aug.cutout_dim
            f = len(self.frame_list)
            i = aug.centroids.shape[1]
            chunk_path = os.path.join(self.save_folder, f"chunk_{self.frame_list[0]:08d}.npz")
            if os.path.isfile(chunk_path) and use_cache:
                return self.frame_list
            self.cutout_images = np.zeros((f, i, d, d, 3), dtype=np.uint8)
            self.cutout_frames = np.array(sorted(self.frame_list))
            self.frame_to_arr_idx = {fid: idx for idx, fid in enumerate(self.cutout_frames)}
        if aug.mode == "ea":
            self._process_frame_mask(aug.mask)

        sparse_mode = self._determine_continous_or_sparse()
        if sparse_mode or self.label_mode:
            self._sparse_frame_extraction(aug)
        else:
            self._continuous_frame_extraction(aug)

        if hasattr(self, "cutout_images"):
            chunk_path = os.path.join(self.save_folder, f"chunk_{self.frame_list[0]:08d}.npz")
            np.savez_compressed(
                chunk_path,
                images=self.cutout_images,
                frame_indices=self.cutout_frames
            )

        self._validate_extracted_indices()

        if self.homegrown_pbar:
            self.pbar.close()

        return self.extracted_indices

    def extract_frames_into_video(
            self,
            ea:Exporter_Augments,
            video_name:str="temp_extract.mp4"
            ):
        assert ea.mode == "ea", "Invalid augmentation."

        self._process_frame_mask(ea.mask)
        video_output_path = os.path.join(self.save_folder, video_name)

        actual_h, actual_w = self.get_vid_dim_to_export(ea)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_output_path, fourcc, fps=10.0, frameSize=(actual_w, actual_h))
        if not writer.isOpened():
            logger.error(f"[EXPORTER] Failed to open VideoWriter for {video_output_path}.")
        logger.debug(f"[EXPORTER] VideoWriter opened for {video_output_path} (size: {actual_w}x{actual_h}).")

        sparse_mode = self._determine_continous_or_sparse()

        try:
            if sparse_mode:
                self._sparse_frame_extraction(ea, writer=writer)
            else:
                self._continuous_frame_extraction(ea, writer=writer)
        finally:
            writer.release()

        logger.debug(f"[EXPORTER] Video saved to {video_output_path}")

        self._validate_extracted_indices()
        if self.homegrown_pbar:
            self.pbar.close()
        return self.extracted_indices

    def get_vid_dim_to_export(self, ea:Exporter_Augments):
        h, w = self.extractor.get_frame_dim()
        if ea.crop_coord is not None:
            x1, y1, x2, y2 = ea.crop_coord
            if x2 > w or y2 > h or x1 < 0 or y1 < 0:
                logger.warning(
                    f"Crop coordinates out of bounds: ({x1}, {y1}, {x2}, {y2}) for frame size ({w}, {h})")
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

            actual_h = y2 - y1
            actual_w = x2 - x1
        else:
            actual_h = h
            actual_w = w
        return actual_h, actual_w

    def _process_frame_mask(self, mask:np.ndarray|None):
        if mask is not None:
            self.mask_pos = (mask == 255)
            self.mask_neg = (mask == -255)
        else:
            self.mask_neg = self.mask_pos = None

    def _determine_continous_or_sparse(self):
        f = self.frame_list
        return max(f) - min(f) > 50 * len(f)

    def _validate_extracted_indices(self):
        extracted_set = set(self.extracted_indices)
        frame_set = set(self.frame_list)
        missing = sorted(frame_set - extracted_set)

        if missing:
            logger.warning(f"[EXPORTER] Frame count mismatch! Extracted frames: {len(extracted_set)} | Expected: {len(frame_set)}"
                         f"\nThe following frames are not extracted: {missing}")

    def _export_augmentation(self, ea:Exporter_Augments, frame:np.ndarray, frame_idx:int, writer:Optional[cv2.VideoWriter]=None):
        if ea.mask is not None:
            frame[self.mask_pos] = 255
            frame[self.mask_neg] = 0

        if ea.crop_coord is not None:
            try:
                x1, y1, x2, y2 = ea.crop_coord
                frame = frame[y1:y2, x1:x2]
            except Exception as e:
                logger.error(f"[EXPORTER] Error applying crop {ea.crop_coord} to frame. Error: {e}")

        if ea.grayscaling:
            frame = frame_to_grayscale(frame, keep_as_bgr=True)

        if writer:
            writer.write(frame)
        else:
            image_path = f"img{str(frame_idx).zfill(8)}.png"
            image_output_path = os.path.join(self.save_folder, image_path)
            cv2.imwrite(image_output_path, frame)

        self.pbar.update(1)
        self.extracted_indices.append(frame_idx)

    def _cutout_augmentation(self, ca:Cutout_Augments, frame:np.ndarray, frame_idx:int):
        if ca.grayscaling:
            frame = frame_to_grayscale(frame, keep_as_bgr=True)

        for inst_idx in range(ca.centroids.shape[1]):
            frame_inst = frame.copy()

            x, y = ca.centroids[frame_idx, inst_idx]
            x = int(x)
            y = int(y)

            if ca.angle_array is not None:
                angle = ca.angle_array[frame_idx, inst_idx]
                frame_inst = self._rotate(frame_inst, angle, (x, y))
                h_rot, w_rot = frame_inst.shape[:2]
                x = w_rot // 2
                y = h_rot // 2

            x1 = x - ca.cutout_dim // 2
            x2 = x + ca.cutout_dim // 2
            y1 = y - ca.cutout_dim // 2
            y2 = y + ca.cutout_dim // 2

            frame_inst = frame_inst[y1:y2, x1:x2]

            if ca.to_image:
                image_output_path = os.path.join(self.save_folder, f"{frame_idx}_{inst_idx}.png")
                cv2.imwrite(image_output_path, frame_inst)
            else:
                arr_idx = self.frame_to_arr_idx[frame_idx]
                self.cutout_images[arr_idx, inst_idx, ...] = frame_inst

        self.pbar.update(1)
        self.extracted_indices.append(frame_idx)

    def _sparse_frame_extraction(self, aug:Exporter_Augments|Cutout_Augments, writer=None):
        for frame_idx in self.frame_list:
            frame = self.extractor.get_frame(frame_idx)
            if frame is None:
                logger.warning(f"[EXPORTER] Frame {frame_idx} not found during sparse extraction. Skipping.")
                continue
            if aug.mode == "ca":
                self._cutout_augmentation(aug, frame, frame_idx)
            else:
                self._export_augmentation(aug, frame, frame_idx, writer)
    
    def _continuous_frame_extraction(self, aug:Exporter_Augments|Cutout_Augments, writer=None):
        min_frame_in_list = min(self.frame_list) if self.frame_list else 0
        max_frame_in_list = max(self.frame_list) if self.frame_list else -1

        current_frame_idx = 0
        frame_set = set(self.frame_list)

        current_frame_idx += min_frame_in_list
        self.extractor.start_sequential_read(start=min_frame_in_list, end=max_frame_in_list+1)
        logger.debug("[EXPORTER] Started sequential frame read for continuous extraction.")

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
                logger.warning(f"[EXPORTER] Frame {current_frame_idx} is empty during continuous extraction. Substituted with placeholder.")

            if current_frame_idx in frame_set:
                if aug.mode == "ca":
                    self._cutout_augmentation(aug, frame, current_frame_idx)
                else:
                    self._export_augmentation(aug, frame, current_frame_idx, writer)

            current_frame_idx += 1

        self.extractor.finish_sequential_read()
        logger.debug("[EXPORTER] Finished sequential frame read after continuous extraction.")

    @staticmethod
    def _rotate(frame:np.ndarray, angle, center=None, scale=1.0):
        (h, w) = frame.shape[:2]
        if not center:
            center = (w / 2, h / 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(frame, rotation_matrix, (new_w, new_h))

        return rotated