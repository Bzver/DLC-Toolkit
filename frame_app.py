
from PySide6 import QtWidgets
from PySide6.QtWidgets import QMainWindow, QVBoxLayout

from ui import Menu_Widget, Video_Player_Widget
from core import Data_Manager, Video_Manager, Keypoint_Edit_Manager
from core.dataclass import Plot_Config, Nav_Callback

class Frame_App(QMainWindow):
    def __init__(self):
        self.setWindowTitle("Frame Label Multitool for DeepLabCut")
        self.setGeometry(100, 100, 1200, 960)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.app_layout = QVBoxLayout(self.central_widget)

        self.dm = Data_Manager(
            init_vid_callback = self._initialize_loaded_video,
            refresh_callback = self._refresh_ui, parent = self)
        self.vm = Video_Manager(self)

        self._setup_menu()

        nav_callback = Nav_Callback(
            change_frame_callback = self._change_frame,
            nav_prev_callback = self._navigate_prev,
            nav_next_callback = self._navigate_next,
        )
        self.vid_play = Video_Player_Widget(
            slider_callback = self._handle_frame_change_from_comp,
            nav_callback = nav_callback,
            parent = self,
            )
        self.app_layout.addWidget(self.vid_play)
        self._setup_shortcut()        
        self.reset_state()
        
    def _setup_menu(self):
        self.menu_widget = Menu_Widget(self)
        self.setMenuBar(self.menu_widget)
        menu_config = {
            "File": {
                "buttons": [
                    ("Load Video", self.load_video),
                    ("Load Prediction", self.load_prediction),
                    ("Load Workspace", self.load_workspace),
                    ("Load DLC Label Data", self.load_dlc_label_data),
                ]
            },
        }
        self.menu_widget.add_menu_from_config(menu_config)

    def load_video(self):
        self.reset_state()
        video_path = self.vm.load_video_dialog()
        if video_path:
            self._initialize_loaded_video(video_path)

    def _initialize_loaded_video(self, video_path:str):
        self.dm.update_video_path(video_path)
        dlc_config_path, pred_path = self.dm.auto_loader()
        if dlc_config_path and pred_path:
            self.dm.load_pred_to_dm(dlc_config_path, pred_path)
        self.vm.init_extractor(video_path)
        self.dm.total_frames = self.vm.get_frame_counts()
        self.vid_play.set_total_frames(self.dm.total_frames)


        self._refresh_and_display()
        print(f"Video loaded: {self.dm.video_file}")
