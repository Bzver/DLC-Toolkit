import numpy as np

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QMessageBox

from ui import Menu_Widget, Video_Player_Widget, Shortcut_Manager, Status_Bar
from utils.helper import frame_to_pixmap
from core import Data_Manager, Video_Manager
from core.tool import Annotation_Config, get_next_frame_in_list

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
        "proximal": "l",
        "in-cage": "q",
        "other": "o",
        "roi": "i"
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
                    ("Toggle Annotation Key Display", self._toggle_annotation_config),
                ]
            },
            "Import":{
                "buttons": [
                    ("Import Annotation From File", self._unimplemented),
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

        self.extra_shorts = Shortcut_Manager(parent=self.main)
        self.reset_state()

    def activate(self, menu_widget:Menu_Widget):
        menu_widget.add_menu_from_config(self.annot_menu_config)
        self.open_annot = True
        QTimer.singleShot(0, self._init_annot_config)

    def deactivate(self, menu_widget:Menu_Widget):
        for menu in self.annot_menu_config.keys():
            menu_widget.remove_entire_menu(menu)
        self.vid_play.set_right_panel_widget(None)
        self.open_annot = False

    def reset_state(self):
        self.open_annot = False
        self.behav_map = self.BEHAVIORS_MAP
        self.annot_array = None
        self._refresh_annot_numeric()
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        for category, key in self.behav_map.items():
            if not key.strip():
                continue
            self.extra_shorts.add_shortcut(
                name=category, key=key.lower(), callback=lambda cat=category: self._annotate(cat)
            )

    def _unimplemented(self):
        QMessageBox.information(self.main, "Unimplemented", "This feature is not yet implemented.")

    def init_loaded_vid(self):
        frame_count = self.vm.get_frame_counts()
        self.annot_array = np.zeros((frame_count,), dtype=np.int8)
        self.open_annot = True
        QTimer.singleShot(0, self._init_annot_config)

    def _init_annot_config(self):
        self.annot_conf = Annotation_Config(self.behav_map, parent=self.main)
        self.annot_conf.category_removed.connect(self._handle_annot_category_remove)
        self.annot_conf.map_change.connect(self._handle_annot_key_change)
        if self.open_annot:
            print("twying to set panel uwu")
            self.vid_play.set_right_panel_widget(self.annot_conf)

    ###################################################################################################################################################

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

        if self.annot_array is not None:
            idx = int(self.annot_array[self.dm.current_frame_idx])
            if idx < len(self.behav_map):
                cat = list(self.behav_map.keys())[idx]
                key = self.behav_map[cat]
                self.status_bar.show_message(f"Current frame annotation: {cat} ({key})", 0)
            else:
                self.status_bar.show_message("", 0)
        else:
            self.status_bar.show_message("", 0)
            
    def _annotate(self, category: str):
        if self.annot_array is None:
            self.init_loaded_vid()

        if category not in self.behav_map:
            self.status_bar.show_message(f"Invalid category: {category}", 2000)
            return

        frame_idx = self.dm.current_frame_idx
        old_idx = self.annot_array[frame_idx]
        new_idx = self.annot_num[category]

        if old_idx == new_idx:
            return

        next_change = self._find_next_annot_change()
        self.annot_array[frame_idx:next_change] = new_idx
        self._refresh_slider() 

    def _find_next_annot_change(self) -> int:
        diffs = np.diff(self.annot_array)
        change_locs = np.where(diffs != 0)[0] + 1
        total_frames = self.vm.get_frame_counts()
        if not change_locs:
            return total_frames # Intentional, total_frames = last_frame + 1, used for array slicing
        
        next_change = get_next_frame_in_list(change_locs.tolist(), self.dm.current_frame_idx)
        if not next_change:
            return total_frames
        else:
            return next_change

    ###################################################################################################################################################

    def _toggle_annotation_config(self):
        self.open_annot = not self.open_annot
        if self.open_annot:
            self._init_annot_config()
            self.menu_slot_callback()
        else:
            self.vid_play.set_right_panel_widget(None)
            self.menu_slot_callback()
            
    def sync_menu_state(self, close_all:bool=False):
        self.open_annot = False

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

    def _refresh_annot_numeric(self):
        self.annot_num = {item:i for i, item in enumerate(self.behav_map.keys())}

    ###################################################################################################################################################

    def _handle_annot_category_remove(self, dest_category, src_category):
        if dest_category == src_category:
            return
        dest_idx = self.annot_num[dest_category]
        src_idx = self.annot_num[src_category]
        self.annot_array[self.annot_array == src_idx] = dest_idx
        self.annot_array[self.annot_array > src_idx] -= 1
        self._refresh_slider()

    def _handle_annot_key_change(self, new_map):
        self.behav_map = new_map
        self._setup_shortcuts()
        self._refresh_annot_numeric()