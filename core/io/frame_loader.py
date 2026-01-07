import os
import numpy as np
import random
import cv2
from PIL import Image
from collections import OrderedDict
from typing import Optional, Tuple

from utils.logger import logger


class Frame_Extractor:
    def __init__(self, video_path: str):
        logger.info(f"[FLOADER] Initializing Frame_Extractor for video: {video_path}")
        self.video_path = os.path.abspath(video_path)
        if not os.path.isfile(self.video_path):
            logger.error(f"[FLOADER] Video file not found: {self.video_path}")
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logger.error(f"[FLOADER] Failed to open video with OpenCV: {self.video_path}")
            raise RuntimeError(f"Failed to open video with OpenCV: {self.video_path}")
        logger.info(f"[FLOADER] Video {self.video_path} opened successfully.")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._frame_cache = OrderedDict()
        self._cache_size = 100

        self._seq_cap = None
        self._seq_next_index = 0
        self._seq_end = 0

    def get_total_frames(self) -> int:
        return self.total_frames

    def get_frame_dim(self) -> Tuple[int, int]:
        return self.height, self.width

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        if frame_index < 0 or frame_index >= self.total_frames:
            logger.warning(f"[FLOADER] Frame index {frame_index} out of bounds (0-{self.total_frames-1}).")
            return None

        if frame_index in self._frame_cache:
            self._frame_cache.move_to_end(frame_index)
            return self._frame_cache[frame_index].copy()

        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret and frame is not None:
            if len(self._frame_cache) >= self._cache_size:
                self._frame_cache.popitem(last=False)
            self._frame_cache[frame_index] = frame.copy()
            return frame
        else:
            logger.warning(f"[FLOADER] OpenCV failed to read frame {frame_index}.")
            return None
        
    def sample_frames(self, frame_count:int=100):
        logger.info(f"[FLOADER] Randomly sampling {frame_count} frames from video.")
        if self.total_frames < frame_count:
            frame_count = self.total_frames        
        
        frames_to_sample = set(random.sample(range(self.total_frames), frame_count)) # Convert to set for efficient seeking
        frame_batched_array = np.zeros((frame_count, self.height, self.width, 3), dtype=np.uint8)

        if frame_count < 1000:
            for i, frame_idx in enumerate(frames_to_sample):
                frame = self.get_frame(frame_idx)
                if frame is not None:
                    frame_batched_array[i] = frame
            
            return frame_batched_array
        else:
            frame_idx = 0
            i = 0
            self.start_sequential_read()
            while frame_idx < self.total_frames:
                _, frame = self.read_next_frame()
                if frame_idx in frames_to_sample:
                    frame_batched_array[i] = frame
                    i += 1
                frame_idx += 1
            return frame_batched_array

    def start_sequential_read(self, start: int = 0, end: Optional[int] = None):
        logger.info(f"[FLOADER] Starting sequential read from frame {start} to {end}.")
        if end is None:
            end = self.total_frames
        if not (0 <= start < self.total_frames):
            logger.error(f"[FLOADER] Start frame {start} out of range (0-{self.total_frames-1}).")
            raise ValueError(f"Start frame {start} out of range")
        if end <= start:
            logger.warning(f"[FLOADER] End frame {end} is less than or equal to start frame {start}. Adjusting end to {start + 1}.")
            end = start + 1
        end = min(end, self.total_frames)
        logger.info(f"[FLOADER] Actual sequential read range: {start} to {end}.")

        if self._seq_cap is not None:
            logger.debug("[FLOADER] Releasing existing sequential capture.")
            self._seq_cap.release()

        self._seq_cap = cv2.VideoCapture(self.video_path)
        if not self._seq_cap.isOpened():
            logger.error(f"[FLOADER] Failed to open video for sequential reading: {self.video_path}")
            raise RuntimeError("Failed to open video for sequential reading")
        logger.info(f"[FLOADER] Sequential capture for {self.video_path} opened successfully.")

        self._seq_cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        self._seq_next_index = start
        self._seq_end = end

    def read_next_frame(self) -> Optional[Tuple[int, np.ndarray]]:
        if self._seq_cap is None:
            logger.warning("[FLOADER] Sequential capture not started.")
            return None
        if self._seq_next_index >= self._seq_end:
            logger.debug(f"[FLOADER] Reached end of sequential read at index {self._seq_next_index}.")
            return None

        ret, frame = self._seq_cap.read()
        if not ret:
            logger.warning(f"[FLOADER] Failed to read frame {self._seq_next_index} during sequential read.")
            return None

        idx = self._seq_next_index
        self._seq_next_index += 1
        return idx, frame

    def finish_sequential_read(self):
        logger.info("[FLOADER] Finishing sequential read.")
        if self._seq_cap is not None:
            self._seq_cap.release()
            self._seq_cap = None
            logger.debug("[FLOADER] Sequential capture released.")

    def clear_cache(self):
        logger.info("[FLOADER] Clearing frame cache.")
        self._frame_cache.clear()

    def close(self):
        logger.info("[FLOADER] Closing Frame_Extractor.")
        self.clear_cache()
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.debug("[FLOADER] Main video capture released.")
        self.finish_sequential_read()

    def __del__(self):
        logger.debug("[FLOADER] Frame_Extractor instance being deleted.")
        self.close()


class Frame_Extractor_Img:
    def __init__(self, img_folder: str):
        logger.info(f"[FLOADER] Initializing Frame_Extractor_Img for folder: {img_folder}")
        if not os.path.isdir(img_folder):
            raise FileNotFoundError(f"Image folder not found: {img_folder}")

        self.img_files = [os.path.join(img_folder,f) for f in os.listdir(img_folder)
                            if f.endswith((".png", ".jpg")) and f.startswith("img")]

        if len(self.img_files) == 0:
            logger.error(f"[FLOADER] No eligible image can be found in {img_folder}.")
            raise RuntimeError(f"No eligible image can be found in {img_folder}.")
        logger.info(f"[FLOADER] Found {len(self.img_files)} image files in {img_folder}.")
        self.img_files.sort()

        self.total_frames = len(self.img_files)
        self.width, self.height = self.get_largest_dim()

    def get_total_frames(self) -> int:
        return self.total_frames
    
    def get_frame_dim(self) -> Tuple[int, int]:
        return self.height, self.width

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        logger.debug(f"[FLOADER] Attempting to get image frame {frame_index}")
        if frame_index < 0 or frame_index >= self.total_frames:
            logger.warning(f"[FLOADER] Image frame index {frame_index} out of bounds (0-{self.total_frames-1}).")
            return None

        frame = cv2.imread(self.img_files[frame_index])
        if frame is not None:
            return frame
        else:
            logger.warning(f"[FLOADER] OpenCV failed to read image frame {frame_index}.")
            return None

    def get_largest_dim(self) -> Tuple[int, int]:
        logger.info("[FLOADER] Calculating largest image dimensions.")
        max_x, max_y = 0, 0
        for path in self.img_files:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    max_x = max(max_x, w)
                    max_y = max(max_y, h)
            except Exception as e:
                logger.exception(f"[FLOADER] Image {path} cannot be opened with PIL.Image due to {e}")
        logger.info(f"[FLOADER] Largest dimensions found: {max_x}x{max_y}.")
        return int(max_x), int(max_y)
    
    def close(self):
        logger.info("[FLOADER] Closing Frame_Extractor_Img.")
        pass