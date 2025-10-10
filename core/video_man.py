from PySide6.QtWidgets import QFileDialog
from .io import Frame_Extractor

class Video_Manager:
    def __init__(self, parent=None):
        self.main = parent
        self.reset_vm()
    
    def reset_vm(self):
        self.extractor = None
        self.current_frame = None

    def load_video_dialog(self) -> str:
        file_dialog = QFileDialog(self)
        video_path, _ = file_dialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        return video_path

    def init_extractor(self, video_path:str):
        self.extractor = Frame_Extractor(video_path)

    def get_status(self):
        return self.current_frame is not None

    def get_frame(self, frame_idx:int):
        self.current_frame = self.extractor.get_frame(frame_idx)
        return self.current_frame

    def get_frame_counts(self) -> int:
        return self.extractor.get_total_frames()