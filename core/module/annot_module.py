import numpy as np

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from ui import Menu_Widget, Video_Player_Widget, Frame_List_Dialog, Status_Bar
from utils.helper import frame_to_pixmap
from core import Data_Manager, Video_Manager
from core.tool import Annotation_Config

class Frame_Annotator:
    # Use hardcoded behavior map for now
    BEHAVIORS_MAP = {
        "allogrooming": "a",
        "anogenital": "s",
        "co-sleeping": "e",
        "cuddling": "c",
        "receptive": "r", 
        "mounting": "m",
        "copulation": "p",
        "proximal": "o"
        }

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

        self.annot_menu_config = {
            "View":{
                "buttons": [
                    ("Toggle Annotation Key Display", self._toggle_annotation_config, {"checkable": True, "checked": False}),
                ]
            },
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

        self.reset_state()

    def activate(self, menu_widget:Menu_Widget):
        menu_widget.add_menu_from_config(self.annot_menu_config)
        self._toggle_annotation_config()

    def deactivate(self, menu_widget:Menu_Widget):
        for menu in self.annot_menu_config.keys():
            menu_widget.remove_entire_menu(menu)
        self.vid_play.set_right_panel_widget(None)

    def reset_state(self):
        self.open_annot = True
        self.behav_map = self.BEHAVIORS_MAP
        self.annot_array = None

        self._setup_shortcuts()

    def _setup_shortcuts(self):
        pass

    def _unimplemented(self):
        QMessageBox.information(self.main, "Unimplemented", "This feature is not yet implemented.")

    def init_loaded_vid(self):
        frame_count = self.vm.get_frame_counts()
        self.annot_array = np.zeros((frame_count,), dtype=np.int8)

    def _toggle_annotation_config(self):
        self.open_annot = not self.open_annot

        if self.open_annot:
            self.annot_conf = Annotation_Config(self.BEHAVIORS_MAP, parent=self.main)
            self.annot_conf.category_removed.connect(self._handle_annot_category_change)
            self.annot_conf.map_change.connect(self._handle_annot_key_change)
            self.vid_play.set_right_panel_widget(self.annot_conf)
            self.menu_slot_callback()
        else:
            self.vid_play.set_right_panel_widget(None)
            self.menu_slot_callback()
            
    def sync_menu_state(self, close_all:bool=False):
        pass

    def display_current_frame(self):
        if not self.vm.check_status_msg():
            self.vid_play.display.setText("No video loaded")

        frame = self.vm.get_frame(self.dm.current_frame_idx)
        if frame is None:
            self.vid_play.display.setText("Failed to load current frame.")
            return

        pixmap, _, _ = frame_to_pixmap(frame)
        scaled_pixmap = pixmap.scaled(self.vid_play.display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.vid_play.display.setPixmap(scaled_pixmap)
        self.vid_play.display.setText("")
        self.vid_play.set_current_frame(self.dm.current_frame_idx)

    ###################################################################################################################################################

    def refresh_and_display(self):
        self.refresh_ui()
        self.display_current_frame()

    def refresh_ui(self):
        self.navigation_title_controller()
        self._refresh_slider()

    def navigation_title_controller(self):
        title_text = self.dm.get_title_text()
        self.status_bar.show_message(title_text, duration_ms=0)
        self.vid_play.nav.set_title_color("black")

    def _refresh_slider(self):
        self.vid_play.sld.clear_frame_category()

    ###################################################################################################################################################

    def _handle_annot_category_change(self, dest_category, src_category):
        pass

    def _handle_annot_key_change(self, new_map):
        self.behav_map = new_map
        self._setup_shortcuts()