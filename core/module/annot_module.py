import numpy as np

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QFileDialog

from typing import List, Dict, Optional

from ui import Menu_Widget, Video_Player_Widget, Shortcut_Manager, Status_Bar, Frame_List_Dialog
from utils.helper import frame_to_pixmap
from core import Data_Manager, Video_Manager
from core.tool import Annotation_Config, Annotation_Summary_Table, get_next_frame_in_list
from core.io import load_annotation

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

    COLOR_HEX_EXPANDED = (
        "#D50000", "#00BCD4", "#FF9800", "#4CAF50", "#FFB3AD", "#3F51B5", "#FFEB3B",
        "#009688", "#607D8B", "#FF5722", "#795548", "#2196F3", "#CDDC39", "#FFC107",
        "#8BC34A", "#673AB7", "#03A9F4", "#E91E63", "#00E676", "#BD34A6", "#9C27B0"
    )

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
                    ("Toggle Annotation Config", self._toggle_annotation_config),
                    ("Choose Annotation Category to Navigate", self._choose_nav_cat,)
                ]
            },
            "Import":{
                "buttons": [
                    ("Import Annotation From File", self._load_annotation),
                    ("Import Frame List As Annotation", self._import_frame_list),
                ]
            },
            "Save":{
                "buttons": [
                    ("Export in Text", self._export_annotation_to_text),
                    ("Export in Mat", self._unimplemented),
                ]
            },
        }

        self.extra_shorts = Shortcut_Manager(parent=self.main)
        self.reset_state()

    def activate(self, menu_widget:Menu_Widget):
        menu_widget.add_menu_from_config(self.annot_menu_config)
        self.open_annot = True
        self._init_annot_config()
        self._refresh_slider()
        self._setup_shortcuts()
        if not self.vid_play.sld.is_zoom_slider_shown:
            self.vid_play.sld.toggle_zoom_slider()

    def deactivate(self, menu_widget:Menu_Widget):
        for menu in self.annot_menu_config.keys():
            menu_widget.remove_entire_menu(menu)
        self.vid_play.set_right_panel_widget(None)
        if self.vid_play.sld.is_zoom_slider_shown:
            self.vid_play.sld.toggle_zoom_slider()
        self.open_annot = False
        self.extra_shorts.clear()

    def reset_state(self, hardcore=False):
        self.open_annot = False
        if hardcore:
            self.behav_map.clear()
        else:
            self.behav_map = self.BEHAVIORS_MAP
        if hasattr(self, "annot_conf"):
            self.annot_conf.sync_behaviors_map(self.behav_map)
        self.annot_array = None
        self.nav_list = []
        self._refresh_annot_numeric()

    def _setup_shortcuts(self):
        self.extra_shorts.clear()
        for category, key in self.behav_map.items():
            if not key.strip():
                continue
            self.extra_shorts.add_shortcut(
                name=category, key=key.lower(), callback=lambda cat=category: self._annotate(cat)
            )

    def _unimplemented(self):
        QMessageBox.information(self.main, "Unimplemented", "This feature is not yet implemented.")

    def _export_annotation_to_text(self):
        if self.annot_array is None:
            QMessageBox.warning(self.main, "No Annotation", "No annotation data to export.")
            return

        file_dialog = QFileDialog(self.main)
        file_path, _ = file_dialog.getSaveFileName(self.main, "Export Annotation to Text File", "", "Text Files (*.txt);;All Files (*)")

        if not file_path:
            return

        try:
            content = "Caltech Behavior Annotator - Annotation File\n\n"
            content += "Configuration file:\n"
            for category, key in self.behav_map.items():
                content += f"{category}\t{key}\n"
            content += "\n"
            content += "S1:\tstart\tend\ttype\n"
            content += "-----------------------------\n"
            if not hasattr(self, "annot_sum"):
                self._init_annot_config()
            segments = self.annot_sum.extract_segments(include_other=True)
            for category, start, end in segments:
                content += f"\t{start}\t{end}\t{category}\n"

            with open(file_path, 'w') as f:
                f.write(content)
            QMessageBox.information(self.main, "Export Successful", f"Annotation exported to {file_path}")

        except Exception as e:
            QMessageBox.critical(self.main, "Export Error", f"Failed to export annotation: {e}")

    def init_loaded_vid(self):
        frame_count = self.vm.get_frame_counts()
        self.annot_array = np.full((frame_count,), 255, dtype=np.uint8)
        self.open_annot = True

    def _load_annotation(self):
        file_dialog = QFileDialog(self.main)
        annot_path, _ = file_dialog.getOpenFileName(self.main, "Select Annotation File", "", "Text Files (*.txt);;All Files (*)")
        if annot_path:
            self.reset_state(True)
            behav_map, frame_dict = load_annotation(annot_path)
            self._frame_list_to_new_annot_cat(frame_dict.keys(), behav_map=behav_map, frame_dict=frame_dict)

    def _import_frame_list(self):
        frame_categories = {
            cat: (self.dm.fm.get_display_name(cat), self.dm.fm.get_frames(cat))
            for cat in self.dm.fm.all_populated_categories()
        }
        if not frame_categories:
            QMessageBox.information(self.main, "No Frames", "No frame categories with frames found.")
            return

        dialog = Frame_List_Dialog(frame_categories, parent=self.main)
        dialog.categories_selected.connect(self._frame_list_to_new_annot_cat)
        dialog.exec()

    def _frame_list_to_new_annot_cat(
            self, categories:List[str], behav_map:Optional[Dict[str, str]]=None, frame_dict:Optional[Dict[str, List[int]]]=None):
        if self.annot_array is None:
            self.init_loaded_vid()
        if behav_map:
            self._handle_annot_key_change(behav_map)
            self.annot_conf.sync_behaviors_map(behav_map)
        for cat in categories:
            if cat not in self.behav_map.keys():
                self.annot_conf.add_category_external(cat)
            frame_list = frame_dict[cat] if frame_dict else self.dm.get_frames(cat)
            frame_arr = np.array(frame_list)
            frame_list_final = frame_arr[frame_arr < self.dm.total_frames].tolist()
            self.annot_array[frame_list_final] = self.cat_to_idx[cat]
        self.refresh_ui()

    def _init_annot_config(self):
        self.annot_conf = Annotation_Config(self.behav_map, parent=self.main)
        self.annot_conf.category_removed.connect(self._handle_annot_category_remove)
        self.annot_conf.map_change.connect(self._handle_annot_key_change)

        self.annot_sum = Annotation_Summary_Table(self.main)
        self.annot_sum.update_data(self.annot_array, self.behav_map, self.idx_to_cat)
        self.annot_sum.row_clicked.connect(self._handle_frame_change_from_comp)

        self.combined_annot = QtWidgets.QWidget()
        combined_layout = QVBoxLayout(self.combined_annot)
        combined_layout.addWidget(self.annot_conf)
        combined_layout.addWidget(self.annot_sum)

        if self.open_annot:
            self.vid_play.set_right_panel_widget(self.combined_annot)

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
            cat = "other" if idx == 255 else self.idx_to_cat[idx] 
            if hasattr(self, "annot_sum") and hasattr(self, "annot_conf"):
                self.annot_conf.highlight_current_category(cat)
                self.annot_sum.highlight_current_frame(self.dm.current_frame_idx)
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
        new_idx = self.cat_to_idx[category]

        if old_idx == new_idx:
            return

        next_change = self._find_next_annot_change()
        self.annot_array[frame_idx:next_change] = new_idx
        self.refresh_ui()

    def _find_next_annot_change(self) -> int:
        diffs = np.diff(self.annot_array)
        change_locs = np.where(diffs != 0)[0] + 1
        total_frames = self.vm.get_frame_counts()
        if not np.any(change_locs):
            return total_frames # Intentional, total_frames = last_frame + 1, used for array slicing
        
        next_change = get_next_frame_in_list(change_locs.tolist(), self.dm.current_frame_idx)
        if not next_change:
            return total_frames
        else:
            return next_change

    def determine_list_to_nav(self):
        return self.nav_list

    def _choose_nav_cat(self):
        frame_categories = {key: (key, self._get_cat_list_from_array(key)) for key in self.cat_to_idx.keys()}
        if frame_categories:
            list_select_dialog = Frame_List_Dialog(frame_categories, parent=self.main)
            list_select_dialog.frame_indices_acquired.connect(self._nav_cat_selected)
            list_select_dialog.exec()

    def _nav_cat_selected(self, frame_list:List[int]):
        self.nav_list = frame_list
        if self.nav_list:
            self.dm.current_frame_idx = self.nav_list[0]

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
        if self.annot_array is None:
            color = "black"
        else:
            current_behav_idx = self.annot_array[self.dm.current_frame_idx]
            color = "black" if current_behav_idx == 255 else self.COLOR_HEX_EXPANDED[current_behav_idx % len(self.COLOR_HEX_EXPANDED)]
        self.vid_play.nav.set_title_color(color)

    def _refresh_slider(self):
        self.annot_sum.update_data(self.annot_array, self.behav_map, self.idx_to_cat)
        self.vid_play.sld.clear_frame_category()
        idx_to_color = {idx:self.COLOR_HEX_EXPANDED[idx] for idx in range(len(self.cat_to_idx))}
        self.vid_play.sld.set_frame_category_array(self.annot_array, idx_to_color)
        self.vid_play.sld.commit_categories()

    def _refresh_annot_numeric(self):
        self.cat_to_idx = {item:i for i, item in enumerate(self.behav_map.keys())}
        self.idx_to_cat = {idx:key for key, idx in self.cat_to_idx.items()}

    def _get_cat_list_from_array(self, category:str) -> List[int]:
        idx = self.cat_to_idx[category]
        return np.where(self.annot_array == idx)[0]

    ###################################################################################################################################################

    def _handle_frame_change_from_comp(self, frame_idx):
        self.dm.current_frame_idx = frame_idx
        self.refresh_and_display()

    def _handle_annot_category_remove(self, dest_category, src_category):
        if dest_category == src_category:
            return
        dest_idx = self.cat_to_idx[dest_category]
        src_idx = self.cat_to_idx[src_category]
        self.annot_array[self.annot_array == src_idx] = dest_idx
        self.annot_array[self.annot_array > src_idx] -= 1
        self._refresh_slider()

    def _handle_annot_key_change(self, new_map):
        self.behav_map = new_map
        self._setup_shortcuts()
        self._refresh_annot_numeric()