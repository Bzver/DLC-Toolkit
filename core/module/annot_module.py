import os
import json
import scipy.io as sio
import numpy as np
import pandas as pd

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QFileDialog

from typing import List, Dict, Optional, Tuple

from core.runtime import Data_Manager, Video_Manager
from core.tool import Annotation_Config, Annotation_Summary_Table, Prediction_Plotter, get_next_frame_in_list
from core.io import load_annotation, prediction_to_csv, load_onehot_csv
from ui import Menu_Widget, Video_Player_Widget, Shortcut_Manager, Status_Bar, Frame_List_Dialog
from utils.track import interpolate_track_all
from utils.helper import frame_to_pixmap, frame_to_grayscale, get_instance_count_per_frame, array_to_iterable_runs
from utils.logger import Loggerbox


class Frame_Annotator:
    BEHAVIORS_MAP = {
        "other": ("o", "#9BB0BB"),
        "allogrooming": ("g", "#D50000"),
        "sniffing": ("s", "#00BCD4"),
        "anogenital": ("a", "#FF9800"),
        "huddling": ("h", "#FFEB3B"),
        "mounting": ("m", "#795548"),
        "copulation": ("p", "#FFB3AD"),
        "receptive": ("l", "#4CAF50"),
        "fleeing": ("f", "#3F51B5"),
    }

    COLOR_HEX_EXPANDED = (
        "#009688", "#FF5722", "#795548", "#2196F3", "#CDDC39", "#FFC107",
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
            "Import":{
                "buttons": [
                    ("Import Annotation From File", self._load_annotation),
                    ("Import Frame List As Annotation", self._import_frame_list),
                    ("Import and Remap ASOID Predictions", self._load_one_hot),
                    ("Import Config From Annotation or JSON Config", self._load_annotation_config),
                    ("Filter Out Short Bout", self._filter_short_bout),
                ]
            },
            "Save":{
                "buttons": [
                    ("Export in Text", self._export_annotation_to_text),
                    ("Export in Mat", self._export_annotation_to_mat),
                    ("Export Truncated in BORIS", self._export_truncated_package),
                ]
            },
        }

        self.sc_annot = Shortcut_Manager(parent=self.main)
        self.reset_state()

    def activate(self, menu_widget:Menu_Widget):
        menu_widget.add_menu_from_config(self.annot_menu_config)
        self.open_annot = True
        self._init_annot_config()
        self._refresh_slider()
        self._setup_shortcuts()
        self._auto_load()
        if not self.vid_play.sld.is_zoom_slider_shown:
            self.vid_play.sld.toggle_zoom_slider()

    def deactivate(self, menu_widget:Menu_Widget):
        for menu in self.annot_menu_config.keys():
            menu_widget.remove_entire_menu(menu)
        self.vid_play.set_right_panel_widget(None)
        if self.vid_play.sld.is_zoom_slider_shown:
            self.vid_play.sld.toggle_zoom_slider()
        self.open_annot = False
        self.annot_conf = None
        self.sc_annot.clear()

    def reset_state(self, hardcore=False):
        self.open_annot = False
        if hardcore:
            self.behav_map.clear()
        else:
            self.behav_map = self.BEHAVIORS_MAP.copy()
        if hasattr(self, "annot_conf") and self.annot_conf is not None:
            self.annot_conf.sync_behaviors_map(self.behav_map)
        self.annot_array = None
        self._refresh_annot_numeric()

    def _setup_shortcuts(self):
        self.sc_annot.clear()
        for category, (key, _) in self.behav_map.items():
            if not key.strip():
                continue
            self.sc_annot.add_shortcut(
                name=category, key=key.lower(), callback=lambda cat=category: self._annotate(cat)
            )

    def init_loaded_vid(self):
        frame_count = self.vm.get_frame_counts()
        self.annot_array = np.zeros((frame_count,), dtype=np.uint8)
        self.open_annot = True

    def _load_annotation(self):
        file_dialog = QFileDialog(self.main)
        annot_path, _ = file_dialog.getOpenFileName(self.main, "Select Annotation File", "", "Text Files (*.txt);;All Files (*)")
        if not annot_path:
            return
        
        self.reset_state(True)
        json_path, _ = file_dialog.getOpenFileName(self.main, "Select JSON Color Map File (Optional)", "", "Json Files (*.json)")
        if json_path:
            behav_map, frame_dict = load_annotation(annot_path, json_path)
        else:
            behav_map, frame_dict = load_annotation(annot_path)

        self._frame_list_to_new_annot_cat(frame_dict.keys(), behav_map=behav_map, frame_dict=frame_dict)

    def _load_annotation_config(self):
        file_dialog = QFileDialog(self.main)
        self.reset_state(True)
        config_path, _ = file_dialog.getOpenFileName(self.main, "Select JSON Color Map File (Optional)", "", "Json Files (*.json);;Text Files (*.txt)")

        if config_path.endswith(".json"):
            with open(config_path, "r") as f:
                meta = json.load(f)
            behav_map = meta["behav_map"]
            frame_dict = {}
        else:
            behav_map, frame_dict = load_annotation(config_path, config_only=True)

        self._frame_list_to_new_annot_cat(behav_map.keys(), behav_map=behav_map, frame_dict=frame_dict)

    def _load_one_hot(self):
        file_dialog = QFileDialog(self.main)
        csv_path, _ = file_dialog.getOpenFileName(
            self.main, "Select One-Hot CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not csv_path:
            return

        json_path, _ = file_dialog.getOpenFileName(
            self.main, "Select Associated JSON Metadata", "", "Json Files (*.json)"
        )
        if not json_path:
            Loggerbox.warning(self.main, "No Metadata", "JSON metadata is required for one-hot import.")
            return

        self.reset_state(True)

        try:
            behav_map, frame_dict = load_onehot_csv(csv_path, json_path)
        except Exception as e:
            Loggerbox.error(self.main, "Load Error", f"Failed to load one-hot annotation: {e}")
            return

        self._frame_list_to_new_annot_cat(list(frame_dict.keys()), behav_map=behav_map, frame_dict=frame_dict)

    def _import_frame_list(self):
        frame_categories = {
            cat: (self.dm.fm.get_display_name(cat), self.dm.fm.get_frames(cat))
            for cat in self.dm.fm.all_populated_categories()
        }
        if not frame_categories:
            Loggerbox.info(self.main, "No Frames", "No frame categories with frames found.")
            return

        dialog = Frame_List_Dialog(frame_categories, parent=self.main)
        dialog.categories_selected.connect(self._frame_list_to_new_annot_cat)
        dialog.exec()

    def _frame_list_to_new_annot_cat(
            self, categories:List[str], behav_map:Optional[Dict[str, Tuple[str, str]]]=None, frame_dict:Optional[Dict[str, List[int]]]=None):
        if self.annot_array is None:
            self.init_loaded_vid()
        if behav_map:
            self._handle_annot_key_change(behav_map)
            self.annot_conf.sync_behaviors_map(behav_map)
        for cat in categories:
            if cat not in self.behav_map:
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
            return

        if self.dm.dlc_data is not None and not hasattr(self, "plotter"):
            self.plotter = Prediction_Plotter(dlc_data=self.dm.dlc_data, plot_config=self.dm.plot_config)

        frame = self.vm.get_frame(self.dm.current_frame_idx)
        if frame is None:
            self.vid_play.display.setText("Failed to load current frame.")
            return
        
        if self.dm.background_masking:
            mask = self.dm.background_mask
            if mask is None:
                mask = self.get_mask_from_blob_config()
        
            frame =  np.clip(frame.astype(np.int16) + mask, 0, 255).astype(np.uint8)

        if self.dm.use_grayscale:
            frame = frame_to_grayscale(frame, keep_as_bgr=True)
    
        if self.annot_array is not None:
            idx = int(self.annot_array[self.dm.current_frame_idx])
            cat = self.idx_to_cat[idx] 
            if hasattr(self, "annot_sum") and hasattr(self, "annot_conf"):
                self.annot_conf.highlight_current_category(cat)
                self.annot_sum.highlight_current_frame(self.dm.current_frame_idx)
        else:
            self.status_bar.show_message("", 0)

        self._plot_current_frame(frame)
    
    def _plot_current_frame(self, frame, count=None):
        if self.dm.dlc_data is not None and self.dm.dlc_data.pred_data_array is not None:
            frame = self.plotter.plot_predictions(frame, self.dm.dlc_data.pred_data_array[self.dm.current_frame_idx,:,:])

        pixmap = frame_to_pixmap(frame)
        scaled_pixmap = pixmap.scaled(self.vid_play.display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.vid_play.display.setPixmap(scaled_pixmap)
        self.vid_play.display.setText("")
        self.vid_play.set_current_frame(self.dm.current_frame_idx)
            
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
        self._auto_save()

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
        if self.annot_array is not None:
            return np.insert(np.where(np.diff(self.annot_array)!=0)[0]+1, 0, 0).tolist()

    ###################################################################################################################################################
            
    def sync_menu_state(self, close_all:bool=False):
        pass

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
            current_behav = self.idx_to_cat[current_behav_idx]
            color = self.behav_map.get(current_behav, ("", "#000000"))[1]
        self.vid_play.nav.set_title_color(color)

    def _refresh_slider(self):
        if hasattr(self, "annot_sum"):
            self.annot_sum.update_data(self.annot_array, self.behav_map, self.idx_to_cat)
        self.vid_play.sld.clear_frame_category()
    
        idx_to_color = {
            self.cat_to_idx[cat]: color 
            for cat, (_, color) in self.behav_map.items()
            if cat in self.cat_to_idx
        }

        for idx in range(len(self.cat_to_idx)):
            if idx not in idx_to_color:
                idx_to_color[idx] = self.COLOR_HEX_EXPANDED[idx % len(self.COLOR_HEX_EXPANDED)]
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
        self._auto_save()

    def _handle_annot_key_change(self, new_map: Dict[str, Tuple[str, str]]):
        self.behav_map = new_map
        self._setup_shortcuts()
        self._refresh_annot_numeric()
        self.annot_sum.update_data(self.annot_array, self.behav_map, self.idx_to_cat)
        self._auto_save()

    ###################################################################################################################################################

    def _filter_short_bout(self):
        value, ok = QtWidgets.QInputDialog.getText(self.main, 
            "Behavior Bout Filtering", "Bouts with shorter duration (frames) will be subsumed into neighboring behaviors:")
        if value.strip() and ok:
            try:
                min_duration = max(2, int(value))
            except ValueError:
                Loggerbox.warning(self.main, "Error", "Please input numbers of minimum frames.")
                return
            
            before_counts = np.bincount(self.annot_array)

            short_mask = np.array([True])
            while np.any(short_mask):
                bout_start = np.insert(np.where(np.diff(self.annot_array)!=0)[0]+1, 0, 0)
                bout_end = np.append(bout_start[1:], len(self.annot_array))
                bout_len = bout_end - bout_start

                short_mask = bout_len < min_duration
                short_mask[0] = False
                short_mask[-1] = False

                for i in np.where(short_mask)[0]:
                    start, end = bout_start[i], bout_end[i]
                    if bout_len[i-1] >= bout_len[i+1]:
                        self.annot_array[start:end] = self.annot_array[start-1]
                    else:
                        self.annot_array[start:end] = self.annot_array[end]

            after_counts = np.bincount(self.annot_array)

            dialog = QtWidgets.QDialog(self.main)
            dialog.setWindowTitle("Composition Change")
            layout = QtWidgets.QVBoxLayout()

            table = QtWidgets.QTableWidget()
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels(["Behavior", "Before (%)", "After (%)", "Δ (%)"])
            table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

            labels = np.union1d(np.where(before_counts)[0], np.where(after_counts)[0])
            table.setRowCount(len(labels))

            for row_idx, lbl in enumerate(np.sort(labels)):
                b = before_counts[lbl] if lbl < len(before_counts) else 0
                a = after_counts[lbl] if lbl < len(after_counts) else 0
                pct_b = 100 * b / self.dm.total_frames
                pct_a = 100 * a / self.dm.total_frames
                delta = pct_a - pct_b
                lbl_text = self.idx_to_cat[lbl]

                table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(str(lbl_text)))
                table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(f"{pct_b:.2f}"))
                table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(f"{pct_a:.2f}"))
                item = QtWidgets.QTableWidgetItem(f"{delta:+.2f}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(row_idx, 3, item)

            table.resizeColumnsToContents()
            layout.addWidget(table)
            layout.addWidget(QtWidgets.QPushButton("OK", clicked=dialog.accept))
            dialog.setLayout(layout)
            dialog.adjustSize()
            dialog.exec()

            self.refresh_ui()

    def _auto_load(self):
        if self.dm.video_file is None:
            return
        if self.annot_array is not None: # No load when already loaded
            return
        vid_path, _ = os.path.splitext(self.dm.video_file)
        annot_path = f"{vid_path}_annot_backup.txt"
        if os.path.isfile(annot_path):
            self.reset_state(True)
            behav_map, frame_dict = load_annotation(annot_path)
            self._frame_list_to_new_annot_cat(frame_dict.keys(), behav_map=behav_map, frame_dict=frame_dict)

    def _auto_save(self):
        if self.annot_array is None:
            return
        
        vid_path, _ = os.path.splitext(self.dm.video_file)
        save_path = f"{vid_path}_annot_backup.txt"
        try:
            self._export_txt_worker(save_path)
            json_path = save_path.replace(".txt", ".json")
            with open(json_path, 'w') as f:
                json.dump({"behav_map": self.behav_map}, f, indent=2)

        except Exception as e:
            self.status_bar.show_message(f"Failed to auto save annotation: {e}")

    def _export_annotation_to_text(self):
        if self.annot_array is None:
            Loggerbox.warning(self.main, "No Annotation", "No annotation data to export.")
            return

        file_dialog = QFileDialog(self.main)
        file_path, _ = file_dialog.getSaveFileName(self.main, "Export Annotation to Text File", "", "Text Files (*.txt);;All Files (*)")

        if not file_path:
            return

        try:
            self._export_txt_worker(file_path)
            json_path = file_path.replace(".txt", ".json")
            with open(json_path, 'w') as f:
                json.dump({"behav_map": self.behav_map}, f, indent=2)

            Loggerbox.info(self.main, "Export Successful", f"Annotation exported to {file_path}")
        except Exception as e:
            Loggerbox.error(self.main, "Export Error", f"Failed to export annotation: {e}", exc=e)

    def _export_txt_worker(self, file_path):
        content = "Caltech Behavior Annotator - Annotation File\n\n"
        content += "Configuration file:\n"
        for category, (key, _) in self.behav_map.items():
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

    def _export_annotation_to_mat(self):
        if self.annot_array is None:
            Loggerbox.warning(self.main, "No Annotation", "No annotation data to export.")
            return

        file_dialog = QFileDialog(self.main)
        file_path, _ = file_dialog.getSaveFileName(self.main, "Export Annotation to Mat File", "", "Matlab Files (*.mat);;All Files (*)")

        if not file_path:
            return
        
        try:
            behavior_struct = self.annot_array.copy()
            annotation_struct = {
                "streamID": 1,
                "annotation": behavior_struct.reshape(-1, 1),
                "behaviors": self.cat_to_idx
            }
            mat_to_save = {"annotation": annotation_struct}
            sio.savemat(file_path, mat_to_save)

            json_path = file_path.replace(".mat", ".json")
            with open(json_path, 'w') as f:
                json.dump({"behav_map": self.behav_map}, f, indent=2)

            Loggerbox.info(self.main, "Success", f"Successfully saved to {file_path}")
        except Exception as e:
            Loggerbox.error(self.main, "Failed", f"Failed to save {file_path}, Exception: {e}", exc=e)

    def _export_truncated_package(self, min_duration:int=10, max_gap:int=2, fps:int=10):
        if self.dm.dlc_data is None or self.dm.dlc_data.pred_data_array is None:
            return
        
        file_dialog = QFileDialog(self.main)
        file_path, _ = file_dialog.getSaveFileName(self.main, "Export Truncated Annotation to BORIS Format", "", "CSV Files (*.csv);;All Files (*)")

        if not file_path:
            return

        pred_data_array = self.dm.dlc_data.pred_data_array.copy()
        I = pred_data_array.shape[1]
        for inst_idx in range(I):
            pred_data_array = interpolate_track_all(pred_data_array, inst_idx, max_gap)

        instance_array = get_instance_count_per_frame(pred_data_array)
        
        frames_to_use = np.zeros_like(instance_array, dtype=bool)
        truncated_arrays = []

        for start, end, value in array_to_iterable_runs(instance_array != I):
            if value:
                continue
            if end - start + 1 < min_duration:
                continue

            frames_to_use[start:end+1] = True
            truncated_arrays.append(self.annot_array[start:end+1])

        frame_list = np.where(frames_to_use)[0].tolist()
        truncated_behavior_array = np.concatenate(truncated_arrays)
        truncated_pred_array = pred_data_array[frames_to_use]

        behavior_list = [b for b in self.idx_to_cat.values()]

        truncated_one_hot = np.zeros((truncated_behavior_array.size, len(behavior_list)), dtype=int)
        truncated_one_hot[np.arange(truncated_behavior_array.size), truncated_behavior_array] = 1

        df_annot = pd.DataFrame(truncated_one_hot, columns=behavior_list)
        if "other" in behavior_list:
            behavior_cols = [col for col in df_annot.columns if col != "other"]
            new_column_order = behavior_cols + ["other"]
            df_annot = df_annot[new_column_order]

        df_annot.insert(0, "time", np.arange(len(frame_list)) / fps)

        try:
            df_annot.to_csv(file_path, sep=',', index=False, float_format='%.6f')

            pred_path = file_path.replace(".csv", "_pred.csv")
            prediction_to_csv(self.dm.dlc_data, truncated_pred_array, pred_path, keep_conf=True)

            json_path = file_path.replace(".csv", ".json")
            with open(json_path, 'w') as f:
                json.dump({
                    "used_frames": frame_list,
                    "total_frames": self.dm.total_frames,
                    "total_exported_frames": len(frame_list),
                    "behav_map": self.behav_map,
                }, f, indent=2)
        except Exception as e:
            Loggerbox.error(self.main, "Failed", f"Failed to save {file_path}, Exception: {e}", exc=e)
        else:
            Loggerbox.info(self.main, "Success", f"Successfully saved to {file_path}")