from PySide6 import QtWidgets

from ui import Menu_Widget, Video_Player_Widget, Frame_List_Dialog, Status_Bar
from .data_man import Data_Manager
from .video_man import  Video_Manager

# Use hardcoded behavior list for now

class Frame_Annotator:
    def __init__(self,
                 data_manager: Data_Manager,
                 video_manager: Video_Manager,
                 video_play_widget: Video_Player_Widget,
                 status_bar: Status_Bar,
                 menu_slot_callback: callable,
                 parent: QtWidgets.QWidget):
        self.dm = data_manager
        self.vm = video_manager
        self.vid_play = video_play_widget
        self.status_bar = status_bar
        self.menu_slot_callback = menu_slot_callback
        self.main = parent

        self.reset_state()

    def activate(self, menu_widget:Menu_Widget):
        menu_widget.add_menu_from_config(self.annot_menu_config)

    def deactivate(self, menu_widget:Menu_Widget):
        for menu in self.annot_menu_config.keys():
            menu_widget.remove_entire_menu(menu)

    def reset_state(self):
        self.vid_play.set_total_frames(0)
        self.annot_menu_config = {
            "Import":{
                "buttons": [
                    ("Import Annotation", self._unimplemented),
                    ("Import Frame List As Annotation Basis", self._unimplemented),
                    ("Import List Group As Annotation Basis", self._unimplemented),
                ]
            },
            "Save":{
                "buttons": [
                    ("Export in Text", self._unimplemented),
                    ("Export in Mat", self._unimplemented),
                ]
            },
        }
        self.refresh_ui()

    def _unimplemented(self):
        pass

    def refresh_ui(self):
        pass