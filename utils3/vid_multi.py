import os
import numpy as np
import cv2

from PySide6.QtCore import QRunnable, Signal, QObject
from PySide6.QtGui import QPixmap

from typing import List

from core.tool import Prediction_Plotter
from core.io import Frame_Extractor
from utils.helper import frame_to_pixmap


class Video_Manager_3D:
    def __init__(self):
        self.reset_vm3()

    def reset_vm3(self):
        self.num_cam = 0
        self.total_frames = None
        self.video_folders = []
        self.extractors:List[Frame_Extractor|None] = []

    def load_video_folder(self, folder_path:str):
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file != "0.mp4":
                    continue
                cam_idx = root.split("Camera")[-1] # 1-based
                if self.num_cam < cam_idx:
                    gap = cam_idx - self.num_cam
                    self.video_folders.extend([None] * gap)
                    self.extractors.extend([None] * gap)
                    self.num_cam = cam_idx
                self.video_folders[cam_idx-1] = root
                vid_path = os.path.join(root, file)
                self.extractors[cam_idx-1] = Frame_Extractor(vid_path)

        self._determine_com_duration()

    def _determine_com_duration(self):
        all_durations = []
        for extractor in self.extractors:
            if extractor is None:
                continue
            view_duration = extractor.get_total_frames()
            all_durations.append(view_duration)
        self.total_frames = min(all_durations)


class Video_Signal(QObject):
    frame_processed = Signal(int, QPixmap, str)  # cam_idx, pixmap, text


class VnP_Runner(QRunnable):
    def __init__(
            self,
            cam_idx:int,
            extractor: Frame_Extractor|None,
            current_frame_idx: int,
            target_size: tuple,
            plotter: Prediction_Plotter,
            pred_data_array: np.ndarray|None,
            signal_obj: Video_Signal
            ):
        super().__init__()
        self.cam_idx = cam_idx
        self.extractor = extractor
        self.current_frame_idx = current_frame_idx
        self.target_width, self.target_height = target_size
        self.plotter = plotter
        self.pred_data_array = pred_data_array
        self.signal_obj = signal_obj
        self.setAutoDelete(True)
    
    def run(self):
        if not self.extractor:
            pixmap = QPixmap()
            text = f"Video {self.cam_idx+1} Not Loaded"
            self.signal_obj.frame_processed.emit(self.cam_idx, pixmap, text)
            return
        
        frame = self.extractor.get_frame(self.current_frame_idx)
        
        if frame is None:
            pixmap = QPixmap()
            text = f"End of Video {self.cam_idx+1} / Error"
            self.signal_obj.frame_processed.emit(self.cam_idx, pixmap, text)
            return

        if self.plotter and self.pred_data_array is not None:
            current_frame_data = self.pred_data_array[self.current_frame_idx, self.cam_idx, :, :]
            if not np.all(np.isnan(current_frame_data)):
                frame = self.plotter.plot_predictions(frame, current_frame_data)

        resized_frame = cv2.resize(frame, (self.target_width, self.target_height),  interpolation=cv2.INTER_AREA)
        rgb_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        pixmap = frame_to_pixmap(rgb_image)

        self.signal_obj.frame_processed.emit(self.cam_idx, pixmap, "")