from typing import Optional, Tuple
from numpy.typing import NDArray

from core.io import Frame_Extractor, Frame_Extractor_Img
from utils.logger import Loggerbox


class Video_Manager:
    def __init__(self, parent=None):
        self.main = parent
        self.reset_vm()
    
    def reset_vm(self):
        self.extractor = None
        self.current_frame = None
        self.video_file = None
        self.image_mode = False

    def update_video_path(self, video_path):
        self.video_file = video_path

    def load_img_from_folder(self, image_folder):
        self.extractor = Frame_Extractor_Img(image_folder)
        self.image_mode = True

    def init_extractor(self, video_path:str):
        self.video_file = video_path
        self.extractor = Frame_Extractor(video_path)

    def get_frame(self, frame_idx:int) -> Optional[NDArray]:
        return self.get_frame_extractor(frame_idx)

    def get_frame_dim(self) -> Tuple[int, int]:
        return self.extractor.get_frame_dim()

    def get_frame_extractor(self, frame_idx:int) -> Optional[NDArray]:
        self.current_frame = self.extractor.get_frame(frame_idx)
        return self.current_frame

    def get_extractor_status(self) -> bool:
        return self.extractor.get_frame(0) is not None

    def check_status_msg(self) -> bool:
        if self.extractor is not None and self.get_extractor_status():
            return True
        else:
            Loggerbox.warning(self.main, "No Video Loaded", "Please load a video via Load Video, Load Workspace or Load DLC Label first.")
            return False

    def get_frame_counts(self) -> int:
        return self.extractor.get_total_frames()

    def get_random_frame_samples(self, sample_count:int):
        return self.extractor.sample_frames(sample_count)