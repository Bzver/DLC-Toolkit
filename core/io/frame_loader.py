import os
import numpy as np
from collections import OrderedDict

import cv2
from PIL import Image

from typing import Optional, Tuple

class Frame_Extractor:
    def __init__(self, video_path: str):
        self.video_path = os.path.abspath(video_path)
        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video with OpenCV: {self.video_path}")

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
            print(f"OpenCV failed to read frame {frame_index}")
            return None

    def start_sequential_read(self, start: int = 0, end: Optional[int] = None):
        if end is None:
            end = self.total_frames
        if not (0 <= start < self.total_frames):
            raise ValueError(f"Start frame {start} out of range")
        if end <= start:
            end = start + 1
        end = min(end, self.total_frames)

        if self._seq_cap is not None:
            self._seq_cap.release()

        self._seq_cap = cv2.VideoCapture(self.video_path)
        if not self._seq_cap.isOpened():
            raise RuntimeError("Failed to open video for sequential reading")

        self._seq_cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        self._seq_next_index = start
        self._seq_end = end

    def read_next_frame(self) -> Optional[Tuple[int, np.ndarray]]:
        if self._seq_cap is None or self._seq_next_index >= self._seq_end:
            return None

        ret, frame = self._seq_cap.read()
        if not ret:
            return None

        idx = self._seq_next_index
        self._seq_next_index += 1
        return idx, frame

    def finish_sequential_read(self):
        if self._seq_cap is not None:
            self._seq_cap.release()
            self._seq_cap = None

    def clear_cache(self):
        self._frame_cache.clear()

    def close(self):
        self.clear_cache()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.finish_sequential_read()

    def __del__(self):
        self.close()


class Frame_Extractor_Img:
    def __init__(self, img_folder: str):
        if not os.path.isdir(img_folder):
            raise FileNotFoundError(f"Image folder not found: {img_folder}")

        self.img_files = [os.path.join(img_folder,f) for f in os.listdir(img_folder) 
                          if f.endswith((".png", ".jpg")) and f.startswith("img")]

        if len(self.img_files) == 0:
            raise RuntimeError(f"No eligible image can be found in {img_folder}.")
        
        self.img_files.sort()

        self.total_frames = len(self.img_files)
        self.width, self.height = self.get_largest_dim()

    def get_total_frames(self) -> int:
        return self.total_frames
    
    def get_frame_dim(self) -> Tuple[int, int]:
        return self.height, self.width

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        if frame_index < 0 or frame_index >= self.total_frames:
            return None

        frame = cv2.imread(self.img_files[frame_index])
        if frame is not None:
            return frame
        else:
            print(f"OpenCV failed to read frame {frame_index}")
            return None

    def get_largest_dim(self) -> Tuple[int, int]:
        max_x, max_y = 0, 0
        for path in self.img_files:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    max_x = max(max_x, w)
                    max_y = max(max_y, h)
            except Exception as e:
                print(f"Skipping {path}: {e}")
        return int(max_x), int(max_y)
    
    def close(self):
        pass