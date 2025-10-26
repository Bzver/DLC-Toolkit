import os

import numpy as np
from collections import OrderedDict
import cv2

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

        # Cache
        self._frame_cache = OrderedDict()
        self._cache_size = 100

    def get_total_frames(self) -> int:
        return self.total_frames

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        if frame_index < 0 or frame_index >= self.total_frames:
            return None

        # Check cache first
        if frame_index in self._frame_cache:
            self._frame_cache.move_to_end(frame_index)
            return self._frame_cache[frame_index].copy()

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

    def get_frame_dim(self) -> Tuple[int, int]:
        frame = self.get_frame(0)
        if frame is not None:
            return frame.shape[:2]
        else:
            return 0, 0

    def clear_cache(self):
        self._frame_cache.clear()

    def close(self):
        self.clear_cache()
        if self.cap:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.close()