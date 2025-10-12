import os
import cv2
from PySide6.QtWidgets import QFileDialog, QMessageBox

from typing import Optional
from numpy.typing import NDArray

from .io import Frame_Extractor

class Video_Manager:
    def __init__(self, parent=None):
        self.main = parent
        self.reset_vm()
    
    def reset_vm(self):
        self.extractor = None
        self.current_frame = None
        self.video_file = None
        self.image_files = []

    def load_video_dialog(self) -> Optional[str]:
        file_dialog = QFileDialog(self.main)
        video_path, _ = file_dialog.getOpenFileName(self.main, "Load Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        self.video_file = video_path
        return video_path

    def load_label_folder_dialog(self) -> Optional[str]:
        folder_dialog = QFileDialog(self.main)
        image_folder = folder_dialog.getExistingDirectory(self.main, "Select Image Folder")
        self.video_file = image_folder
        return image_folder

    def load_img_from_folder(self, image_folder):
        img_exts = ('.png', '.jpg')
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(img_exts) and f.startswith("img")])
        if not self.vm.image_files:
            return True
        else:
            QMessageBox.warning(self.main, "No Images", "No image files found in the selected folder.")
            return False

    def init_extractor(self, video_path:str):
        self.video_file = video_path
        self.extractor = Frame_Extractor(video_path)

    def get_frame(self, frame_idx:int) -> Optional[NDArray]:
        if self.image_files:
            return self.get_frame_img(frame_idx)
        if self.get_extractor_status():
            return self.get_frame_extractor(frame_idx)

    def get_frame_extractor(self, frame_idx:int) -> Optional[NDArray]:
        if not self.extractor:
            return None
        self.current_frame = self.extractor.get_frame(frame_idx)
        return self.current_frame

    def get_frame_img(self, frame_idx:int) -> NDArray:
        img_path = os.path.join(self.video_file, self.image_files[frame_idx])
        self.current_frame = cv2.imread(img_path)
        return self.current_frame 

    def get_extractor_status(self) -> bool:
        self.get_frame_extractor(0)
        return self.current_frame is not None

    def check_status_msg(self) -> bool:
        if self.current_frame is not None or self.get_extractor_status():
            return True
        else:
            QMessageBox.warning(self.main, "No Video", "No video has been loaded, please load a video first.")
            return False

    def get_frame_counts(self) -> int:
        return self.extractor.get_total_frames()