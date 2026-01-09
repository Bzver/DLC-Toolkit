import numpy as np
import cv2

from PySide6 import QtGui
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox

from typing import List, Tuple, Optional

from .plot import Prediction_Plotter
from .undo_redo import Uno_Stack, Uno_Stack_Dict
from .mark_nav import navigate_to_marked_frame
from core.io import Frame_Extractor, Frame_Extractor_Img
from ui import Clickable_Video_Label, Video_Slider_Widget, Shortcut_Manager
from utils.helper import (
    frame_to_pixmap, handle_unsaved_changes_on_close, crop_coord_to_array, validate_crop_coord, indices_to_spans
    )
from utils.track import swap_track
from utils.dataclass import Loaded_DLC_Data, Plot_Config
from utils.logger import Loggerbox, QMessageBox


class Parallel_Review_Dialog(QDialog):
    pred_data_exported = Signal(object, tuple)
    RERUN_PALLETTE = {0: "#959595", 1: "#68b3ff", 2: "#F749C6"}

    def __init__(self,
                 dlc_data:Loaded_DLC_Data,
                 extractor:Frame_Extractor|Frame_Extractor_Img,
                 new_data_array:np.ndarray,
                 frame_list:List[int],
                 crop_coord:Optional[Tuple[int,int,int,int]]=None,
                 parent=None):
        super().__init__(parent)
        self.dlc_data = dlc_data
        self.extractor = extractor
        self.new_data_array = new_data_array

        self.backup_data_array = dlc_data.pred_data_array.copy()
        self.pred_data_array = dlc_data.pred_data_array.copy()

        self.total_frames = self.pred_data_array.shape[0]
        self.frame_list = frame_list
        self.total_marked_frames = len(self.frame_list)

        crop_coord_good = validate_crop_coord(crop_coord)
        if crop_coord_good is not None:
            self.crop_array = crop_coord_to_array(crop_coord_good, new_data_array.shape)
        else:
            self.crop_array = None

        if self.crop_array is not None:
            self.backup_data_array -= self.crop_array
            self.pred_data_array -= self.crop_array
            self.new_data_array -= self.crop_array

        self.frame_status_array = np.zeros((self.total_marked_frames,), dtype=np.uint8)
        self.current_frame_idx = 0 # local_idx
        self.is_saved = False

        plot_config = Plot_Config(
            plot_opacity =0.7, point_size = 6.0, confidence_cutoff = 0.0, hide_text_labels = False, edit_mode = False,
            plot_labeled = True, plot_pred = True, navigate_labeled = False, auto_snapping = False, navigate_roi = False)
        self.plotter = Prediction_Plotter(dlc_data=self.dlc_data, plot_config=plot_config)

        self.layout_reviewer = QVBoxLayout(self)
        frame_info_layout, video_label_layout = self._setup_video_display()
        self.layout_reviewer.addLayout(frame_info_layout)
        self.layout_reviewer.addLayout(video_label_layout)
        self.layout_reviewer.addLayout(self.video_layout)

        self.progress_slider = Video_Slider_Widget()
        self.progress_slider.set_total_frames(self.total_marked_frames)
        self.progress_slider.set_current_frame(0)
        self.progress_slider.frame_changed.connect(self._handle_frame_change_from_comp)
        self.layout_reviewer.addWidget(self.progress_slider)

        self.approval_box = self._setup_control()
        self.layout_reviewer.addWidget(self.approval_box)
        
        self._build_shortcut()
        self._navigation_title_controller()
        self._update_button_states()
        self._setup_uno()

        self._display_current_frame()
        self._refresh_ui()

    def _setup_video_display(self):
        frame_info_layout = QHBoxLayout()
        self.global_frame_label = QLabel()
        self.selected_frame_label = QLabel()
        font = QtGui.QFont(); font.setBold(True)
        self.global_frame_label.setFont(font); self.selected_frame_label.setFont(font)
        self.global_frame_label.setStyleSheet("color: black;")
        self.selected_frame_label.setStyleSheet("color: #1E90FF;")
        frame_info_layout.addStretch()
        frame_info_layout.addWidget(self.global_frame_label)
        frame_info_layout.addWidget(self.selected_frame_label)
        frame_info_layout.addStretch()

        video_label_layout = QHBoxLayout()
        self.old_label = QLabel("Old Predictions")
        self.old_label.setAlignment(Qt.AlignCenter)
        self.old_label.setFont(font)
        self.old_label.setStyleSheet("color: #4B4B4B; background: #F0F0F0; padding: 6px; border-radius: 4px;")
        self.old_label.setFixedHeight(30)

        self.new_label = QLabel("New Predictions")
        self.new_label.setAlignment(Qt.AlignCenter)
        self.new_label.setFont(font)
        self.new_label.setStyleSheet("color: #4B4B4B; background: #F0F0F0; padding: 6px; border-radius: 4px;")
        self.new_label.setFixedHeight(30)

        video_label_layout.addWidget(self.old_label)
        video_label_layout.addWidget(self.new_label)

        self.video_layout = QHBoxLayout()
        self.video_labels:List[Clickable_Video_Label] = []
        for col in range(2):
            label = Clickable_Video_Label(col, self)
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(720, 540)
            label.setStyleSheet("border: 1px solid gray;")
            self.video_labels.append(label)
            self.video_layout.addWidget(label)

        return frame_info_layout, video_label_layout

    def _setup_control(self):
        approval_box = QGroupBox("Frame Approval")
        approval_layout = QHBoxLayout()

        self.reject_button = QPushButton("Reject (R)")
        self.approve_button = QPushButton("Approve (A)")
        self.approve_all_button = QPushButton("Approve All")
        self.reject_all_button = QPushButton("Reject All")
        self.apply_button = QPushButton("Apply Changes (Ctrl + S)")

        self.approve_button.clicked.connect(lambda: self._mark_frame_status(True))
        self.reject_button.clicked.connect(lambda: self._mark_frame_status(False))
        self.approve_all_button.clicked.connect(self._approve_all_remaining)
        self.reject_all_button.clicked.connect(self._reject_all_remaining)
        self.apply_button.clicked.connect(self._save_prediction)

        for btn in [self.reject_button, self.approve_button, self.approve_all_button,
                    self.reject_all_button, self.apply_button]:
            approval_layout.addWidget(btn)
        approval_box.setLayout(approval_layout)

        return approval_box

    def _setup_uno(self):
        self.uno = Uno_Stack()
        self.uno.save_state_for_undo(self.frame_status_array)

    def _build_shortcut(self):
        self.shortcut_man = Shortcut_Manager(self)
        self.shortcut_man.add_shortcuts_from_config({
            "prev_frame":{"key": "Left", "callback": lambda: self._change_frame(-1)},
            "next_frame":{"key": "Right", "callback": lambda: self._change_frame(1)},
            "prev_fast":{"key": "Shift+Left", "callback": lambda: self._change_frame(-10)},
            "next_fast":{"key": "Shift+Right", "callback": lambda: self._change_frame(10)},
            "prev_mark":{"key": "Up", "callback": self._navigate_prev},
            "next_mark":{"key": "Down", "callback": self._navigate_next},
            "playback":{"key": "Space", "callback": self._toggle_playback},
            "undo": {"key": "Ctrl+Z", "callback": self._undo_changes},
            "redo": {"key": "Ctrl+Y", "callback": self._redo_changes},
            "save":{"key": "Ctrl+S", "callback": self._save_prediction},
            "approve":{"key": "A", "callback": lambda: self._mark_frame_status(True)},
            "reject":{"key": "R", "callback": lambda: self._mark_frame_status(False)},
        })

    ###################################################################################################

    def _display_current_frame(self):
        global_idx = self.frame_list[self.current_frame_idx]
        frame = self.extractor.get_frame(self.current_frame_idx)

        if frame is None:
            for i, label in enumerate(self.video_labels):
                label.setText(f"Frame {global_idx} unavailable")
                label.setPixmap(QtGui.QPixmap())
            return

        for i in range(2):
            view = frame.copy()
            pred_data = None
            if i == 0 and self.pred_data_array is not None:
                pred_data = self.pred_data_array[global_idx]
            elif i == 1 and self.new_data_array is not None:
                pred_data = self.new_data_array[global_idx]

            if pred_data is not None and not np.all(np.isnan(pred_data)):
                view = self.plotter.plot_predictions(view, pred_data)

            h, w = self.video_labels[i].height(), self.video_labels[i].width()
            resized = cv2.resize(view, (w, h), interpolation=cv2.INTER_AREA)
            pixmap = frame_to_pixmap(resized)
            self.video_labels[i].setPixmap(pixmap)
            self.video_labels[i].setText("")

        self.progress_slider.set_current_frame(self.current_frame_idx)
        self._update_button_states()

    def _change_frame(self, delta):
        new_frame_idx = self.current_frame_idx + delta
        if 0 <= new_frame_idx < self.total_marked_frames:
            self.current_frame_idx = new_frame_idx
            self._display_current_frame()
            self._navigation_title_controller()
            self._update_button_states()

    def _navigate_prev(self):
        list_to_nav = self._determine_list_to_nav()
        navigate_to_marked_frame(
            self, list_to_nav, self.current_frame_idx, self._handle_frame_change_from_comp, "prev")

    def _navigate_next(self):
        list_to_nav = self._determine_list_to_nav()
        navigate_to_marked_frame(
            self, list_to_nav, self.current_frame_idx, self._handle_frame_change_from_comp, "next")

    def _determine_list_to_nav(self):
        return np.where(self.frame_status_array==1)[0].tolist()

    def _toggle_playback(self):
        self.progress_slider.toggle_playback()

    ###################################################################################################

    def _mark_frame_status(self, approved:bool):
        self._save_state_for_undo()
        self.frame_status_array[self.current_frame_idx] = 1 if approved else 2
        self.apply_button.setEnabled(np.any(self.frame_status_array==1))
        global_frame_idx = self.frame_list[self.current_frame_idx]
        pred_data_to_use = self.new_data_array if approved else self.backup_data_array
        self.pred_data_array[global_frame_idx, :, :] = pred_data_to_use[global_frame_idx, :, :]
        self._refresh_ui()

    def _approve_all_remaining(self):
        self._save_state_for_undo()
        unproc_mask, global_mask = self._acquire_unproc_mask()
        self.pred_data_array[global_mask] = self.new_data_array[global_mask]
        self.frame_status_array[unproc_mask] = 1
        self.apply_button.setEnabled(True)
        self._refresh_ui()

    def _reject_all_remaining(self):
        self._save_state_for_undo()
        unproc_mask, global_mask = self._acquire_unproc_mask()
        self.pred_data_array[global_mask] = self.backup_data_array[global_mask]
        self.frame_status_array[unproc_mask] = 2
        self._refresh_ui()

    ###################################################################################################

    def _navigation_title_controller(self):
        global_idx = self.frame_list[self.current_frame_idx]
        self.global_frame_label.setText(f"Global: {global_idx} / {self.total_frames - 1}")
        self.selected_frame_label.setText(f"Selected: {self.current_frame_idx} / {self.total_marked_frames - 1}")

    def _update_button_states(self):
        current_status = self.frame_status_array[self.current_frame_idx]
        self.approve_button.setEnabled(current_status != 1)
        self.reject_button.setEnabled(current_status != 2)
        self.approve_all_button.setEnabled(np.any(self.frame_status_array==0))
        self.reject_all_button.setEnabled(np.any(self.frame_status_array==0))

    def _acquire_unproc_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        unproc_mask = self.frame_status_array == 0
        global_mask = np.zeros((self.total_frames), dtype=bool)
        global_mask[self.frame_list] = unproc_mask
        return unproc_mask, global_mask

    def _handle_frame_change_from_comp(self, new_frame_idx:int):
        self.current_frame_idx = new_frame_idx
        self._display_current_frame()
        self._navigation_title_controller()
        self._update_button_states()

    def _refresh_ui(self):
        self._display_current_frame()
        self._update_button_states()
        self._refresh_slider()
    
    def _refresh_slider(self):
        self.is_saved = False
        self.progress_slider.clear_frame_category()
        self.progress_slider.set_frame_category_array(self.frame_status_array, self.RERUN_PALLETTE)
        self.progress_slider.commit_categories()

    ###################################################################################################
    
    def _save_state_for_undo(self):
        data_array = self.frame_status_array
        self.uno.save_state_for_undo(data_array)

    def _undo_changes(self):
        data_array = self.frame_status_array
        data_array = self.uno.undo(data_array)
        self._undo_redo_worker(data_array)

    def _redo_changes(self):
        data_array = self.frame_status_array
        data_array = self.uno.redo(data_array)
        self._undo_redo_worker(data_array)

    def _undo_redo_worker(self, data_array):
        if data_array is None:
            return

        if np.any(self.frame_status_array!=data_array):
            for frame_idx in np.where(self.frame_status_array!=data_array)[0]:
                global_idx = self.frame_list[frame_idx]
                status = data_array[frame_idx]
                if status == 1:
                    self.pred_data_array[global_idx] = self.new_data_array[global_idx]
                else:
                    self.pred_data_array[global_idx] = self.backup_data_array[global_idx]

            self.frame_status_array = data_array.copy()

        self._display_current_frame()
        self._refresh_ui()

    def _save_prediction(self):
        self._refresh_slider()

        if np.any(self.frame_status_array==0):
            unprocessed_count = np.sum(self.frame_status_array==0)
            reply = Loggerbox.question(
                parent=self,
                title="Unprocessed Frames - Save and Exit?",
                text=(
                    f"You have {unprocessed_count} unprocessed frame(s).\n\n"
                    "This action will save your progress and close the window.\n"
                    "All unprocessed frames will remain as original and will not be marked as approved.\n\n"
                    "Are you sure you want to continue?"
                ),
                buttons=QMessageBox.Yes | QMessageBox.No,
                default=QMessageBox.No
            )

            if reply == QMessageBox.No:
                return

        self.is_saved = True

        if self.crop_array is not None:
            self.new_data_array += self.crop_array
            self.pred_data_array += self.crop_array
            self.backup_data_array += self.crop_array

        if self.total_marked_frames < 1000:
            frame_arr = np.array(self.frame_list)
            approve_list_global = frame_arr[self.frame_status_array==1].tolist()
            rejected_list_global = frame_arr[self.frame_status_array==2].tolist()
        else:
            approved_slices = [slice(start, end + 1) for (start,end) in indices_to_spans(np.where(self.frame_status_array==1)[0])]
            rejected_slices = [slice(start, end + 1) for (start,end) in indices_to_spans(np.where(self.frame_status_array==2)[0])]

            approve_list_global, rejected_list_global = [], []

            for sl in approved_slices:
                approve_list_global.extend(self.frame_list[sl])
            for sl in rejected_slices:
                rejected_list_global.extend(self.frame_list[sl]) 
    
        list_tuple = (approve_list_global, rejected_list_global)
        self.pred_data_exported.emit(self.pred_data_array, list_tuple)
        self.accept()

    def closeEvent(self, event):
        handle_unsaved_changes_on_close(self, event, self.is_saved, self._save_prediction)


class Track_Correction_Dialog(Parallel_Review_Dialog):
    TC_PALLETTE    = {0: "#D81687", 1: "#CF5300"}

    def __init__(
            self,
            dlc_data:Loaded_DLC_Data,
            extractor:Frame_Extractor,
            new_data_array:np.ndarray,
            frame_list: List[int],
            tc_frame_tuple:Optional[Tuple[List[int], List[int]]]=None,
            crop_coord:Optional[Tuple[int, int, int, int]]= None,
            parent=None
            ):

        self.frame_list = frame_list

        if tc_frame_tuple is None:
            blasted_frames_global, ambiguous_frames_global = [], []
        else:
            blasted_frames_global, ambiguous_frames_global = tc_frame_tuple

        blasted_set_global = set(blasted_frames_global)
        ambiguous_set_global = set(ambiguous_frames_global)
        frame_set = set(frame_list)

        if not blasted_set_global.issubset(frame_set) or not ambiguous_set_global.issubset(frame_set):
            blasted_frames_global = [f for f in blasted_frames_global if f in frame_set]
            ambiguous_frames_global = [f for f in ambiguous_frames_global if f in frame_set]

        self.blasted_set_global = set(blasted_frames_global)
        self.ambiguous_set_global = set(ambiguous_frames_global)
        
        self.global_to_local = {g: i for i, g in enumerate(self.frame_list)}
        self.blasted_local = [self.global_to_local[g] for g in self.blasted_set_global if g in self.global_to_local]
        self.ambiguous_local = [self.global_to_local[g] for g in self.ambiguous_set_global if g in self.global_to_local]
        self.list_to_nav = self.ambiguous_local if self.ambiguous_local else self.blasted_local

        super().__init__(dlc_data, extractor, new_data_array, frame_list=frame_list, crop_coord=crop_coord, parent=parent)

        self.nonempty_local = set(np.where(np.any(~np.isnan(new_data_array[self.frame_list]), axis=(1, 2)))[0].tolist())

        self.old_label.setText("Last Frame With Prediction")
        self.new_label.setText("Current Frame")

    def _setup_control(self):
        swap_box = QGroupBox("Manual Track Correction")
        swap_layout = QHBoxLayout()

        self.swap_button = QPushButton("Swap Instance (W)")
        self.big_swap_button = QPushButton("Swap Track (Shift + W)")
        self.apply_button = QPushButton("Apply Changes (Ctrl + S)")

        self.swap_button.clicked.connect(self._swap_instance)
        self.big_swap_button.clicked.connect(self._swap_track)
        self.apply_button.clicked.connect(self._save_prediction)

        self.swap_button.setToolTip("Swap current frame instance only.")
        self.big_swap_button.setToolTip("Swap current frame + all preceding frames.")

        for btn in [self.swap_button, self.big_swap_button, self.apply_button]:
            swap_layout.addWidget(btn)
        swap_box.setLayout(swap_layout)

        return swap_box

    def _build_shortcut(self):
        self.shortcut_man = Shortcut_Manager(self)
        self.shortcut_man.add_shortcuts_from_config({
            "prev_frame":{"key": "Left", "callback": lambda: self._change_frame(-1)},
            "next_frame":{"key": "Right", "callback": lambda: self._change_frame(1)},
            "prev_fast":{"key": "Shift+Left", "callback": lambda: self._change_frame(-10)},
            "next_fast":{"key": "Shift+Right", "callback": lambda: self._change_frame(10)},
            "prev_mark":{"key": "Up", "callback": self._navigate_prev},
            "next_mark":{"key": "Down", "callback": self._navigate_next},
            "playback":{"key": "Space", "callback": self._toggle_playback},
            "undo": {"key": "Ctrl+Z", "callback": self._undo_changes},
            "redo": {"key": "Ctrl+Y", "callback": self._redo_changes},
            "save":{"key": "Ctrl+S", "callback": self._save_prediction},
            "swap":{"key": "W", "callback": self._swap_instance},
            "big_swap":{"key": "Shift+W", "callback": self._swap_track},
        })

    def _setup_uno(self):
        self.uno = Uno_Stack()
        self.inst_array = np.array(range(self.dlc_data.pred_data_array.shape[1]))
        self.uno.save_state_for_undo(self.new_data_array)

    def _display_current_frame(self):
        global_idx = self.frame_list[self.current_frame_idx]
        frame = self.extractor.get_frame(global_idx)

        last_frame_local = self._back_search_nonempty_frames()
        last_frame_global = self.frame_list[last_frame_local]

        last_frame = self.extractor.get_frame(last_frame_global)
        if frame is None:
            self.video_labels[1].setText(f"Frame {global_idx} unavailable")
            return
        if last_frame is None:
            self.video_labels[0].setText(f"Frame {last_frame_global} unavailable")

        pred_data_now, pred_data_last = None, None
        pred_data_now = self.new_data_array[global_idx]
        if global_idx != 0:
            pred_data_last = self.new_data_array[last_frame_global]
        
        if pred_data_now is not None:
            frame = self.plotter.plot_predictions(frame, pred_data_now)
        if pred_data_last is not None:
            last_frame = self.plotter.plot_predictions(last_frame, pred_data_last)

        for i in range(2):
            view = frame if i == 1 else last_frame
            if view is None:
                continue
            h, w = self.video_labels[i].height(), self.video_labels[i].width()
            resized = cv2.resize(view, (w, h), interpolation=cv2.INTER_AREA)
            pixmap = frame_to_pixmap(resized)
            self.video_labels[i].setPixmap(pixmap)
            self.video_labels[i].setText("")

        self.progress_slider.set_current_frame(self.current_frame_idx)
        self.apply_button.setEnabled(True)

    def _determine_list_to_nav(self):
        return self.list_to_nav
    
    def _swap_instance(self):
        self._save_state_for_undo()
        self.new_data_array = swap_track(self.new_data_array, self.frame_list[self.current_frame_idx])
        self._display_current_frame()
        self._refresh_ui()

    def _swap_track(self):
        self._save_state_for_undo()
        self.new_data_array = swap_track(self.new_data_array, self.frame_list[self.current_frame_idx], swap_range=[-1])
        self._display_current_frame()
        self._refresh_ui()

    def _back_search_nonempty_frames(self, lookback_window: int = 10) -> int:
        start_local = self.current_frame_idx
        for offset in range(1, min(lookback_window + 1, start_local + 1)):
            candidate_local = start_local - offset
            if candidate_local in self.nonempty_local:
                return candidate_local
        return max(0, start_local - 1)

    def _navigation_title_controller(self):
        super()._navigation_title_controller()
        global_idx = self.frame_list[self.current_frame_idx]
        if global_idx in self.ambiguous_set_global:
            self.selected_frame_label.setStyleSheet(f"color: white; background-color: {self.TC_PALLETTE[1]};")
        elif global_idx in self.blasted_set_global:
            self.selected_frame_label.setStyleSheet(f"color: white; background-color: {self.TC_PALLETTE[0]};")
        else:
            self.selected_frame_label.setStyleSheet(f"color: #1E90FF; background-color: transparent;")

    def _update_button_states(self):
        pass
    
    def _refresh_slider(self):
        self.is_saved = False
        self.progress_slider.clear_frame_category()
        self.progress_slider.set_frame_category("ambiguous", self.ambiguous_local, self.TC_PALLETTE[1], priority=5)
        self.progress_slider.set_frame_category("changed", self.blasted_local, self.TC_PALLETTE[0])
        self.progress_slider.commit_categories()

    def _save_state_for_undo(self):
        data_array = self.new_data_array
        self.uno.save_state_for_undo(data_array)

    def _undo_changes(self):
        data_array = self.new_data_array
        data_array = self.uno.undo(data_array)
        self._undo_redo_worker(data_array)

    def _redo_changes(self):
        data_array = self.new_data_array
        data_array = self.uno.redo(data_array)
        self._undo_redo_worker(data_array)

    def _undo_redo_worker(self, data_array):
        if data_array is None:
            return
        self.new_data_array = data_array.copy()
        self._display_current_frame()
        self._refresh_ui()

    def _save_prediction(self):
        self._refresh_slider()

        self.is_saved = True
        if self.crop_array is not None:
            self.new_data_array += self.crop_array
            self.pred_data_array += self.crop_array
            self.backup_data_array += self.crop_array

        self.pred_data_exported.emit(self.new_data_array, ())
        self.accept()

    def closeEvent(self, event):
        self.extractor.close()
        handle_unsaved_changes_on_close(self, event, self.is_saved, self._save_prediction)


class Iteration_Review_Dialog(Track_Correction_Dialog):
    IR_PALLETTE = {0: "#959595", 1: "#2EDA04", 2: "#CF5300", 3: "#AF01A6", 4: "#4938E4"}

    def __init__(self, dlc_data, extractor, new_data_array, frame_list, ir_frame_tuple, crop_coord = None, parent=None):
        super().__init__(dlc_data, extractor, new_data_array, frame_list, ir_frame_tuple, crop_coord, parent)

        self.blasted_set_global = self.blasted_set_global
        self.blasted_local = self.blasted_local

        self.was_cancelled = False 
        self.is_entertained = False

         # 0: skip (default approved), 1: approve, 2: reject(and swap), 3: blasted, 4: ambiguous
        self.frame_status_array[self.blasted_local] = 3
        self.frame_status_array[self.ambiguous_local] = 4
        self._refresh_slider()

    def get_result(self):
        return self.new_data_array[self.frame_list], self.frame_status_array, self.is_entertained

    def _setup_control(self):
        swap_box = QGroupBox("Manual Track Correction")
        swap_layout = QHBoxLayout()

        self.approve_button = QPushButton("Approve Current Frame (A)")
        self.reject_button = QPushButton("Reject and Swap Track (R)")
        self.blast_button = QPushButton("Mark | Unmark Faulty Frame (D)")
        self.apply_button = QPushButton("Continue Training (Ctrl + S)")
        self.finish_button = QPushButton("Accept Current Weight and Fit the Whole Video")

        self.approve_button.clicked.connect(self._mark_approved)
        self.reject_button.clicked.connect(self._mark_rejected_swap)
        self.blast_button.clicked.connect(self._mark_blasted)
        self.apply_button.clicked.connect(self._export_label)
        self.finish_button.clicked.connect(self._i_am_entertained)

        for btn in [self.approve_button, self.reject_button, self.blast_button, self.apply_button, self.finish_button]:
            swap_layout.addWidget(btn)
        swap_box.setLayout(swap_layout)

        return swap_box

    def _build_shortcut(self):
        self.shortcut_man = Shortcut_Manager(self)
        self.shortcut_man.add_shortcuts_from_config({
            "prev_frame":{"key": "Left", "callback": lambda: self._change_frame(-1)},
            "next_frame":{"key": "Right", "callback": lambda: self._change_frame(1)},
            "prev_fast":{"key": "Shift+Left", "callback": lambda: self._change_frame(-10)},
            "next_fast":{"key": "Shift+Right", "callback": lambda: self._change_frame(10)},
            "prev_mark":{"key": "Up", "callback": self._navigate_prev},
            "next_mark":{"key": "Down", "callback": self._navigate_next},
            "playback":{"key": "Space", "callback": self._toggle_playback},
            "undo": {"key": "Ctrl+Z", "callback": self._undo_changes},
            "redo": {"key": "Ctrl+Y", "callback": self._redo_changes},
            "save":{"key": "Ctrl+S", "callback": self._export_label},
            "approve":{"key": "A", "callback": self._mark_approved},
            "big_swap":{"key": "R", "callback": self._mark_rejected_swap},
            "blast":{"key": "D", "callback": self._mark_blasted},
        })

    def _mark_approved(self):
        self._save_state_for_undo()
        if self.frame_status_array[self.current_frame_idx] == 2:
            self._swap_track()
        self.frame_status_array[self.current_frame_idx] = 1
        self._refresh_ui()

    def _mark_rejected_swap(self):
        if self.frame_status_array[self.current_frame_idx] == 2:
            return
        self._save_state_for_undo()
        self.frame_status_array[self.current_frame_idx] = 2
        self._refresh_ui()
        self._swap_track()

    def _mark_blasted(self):
        self._save_state_for_undo()
        if self.frame_status_array[self.current_frame_idx] == 3:
            self.frame_status_array[self.current_frame_idx] = 1
        else:
            self.frame_status_array[self.current_frame_idx] = 3
        self._refresh_ui()

    def _refresh_ui(self):
        super()._refresh_ui()
        self._navigation_title_controller()

    def _navigation_title_controller(self):
        super()._navigation_title_controller()
        match self.frame_status_array[self.current_frame_idx]:
            case 1: self.selected_frame_label.setStyleSheet(f"color: white; background-color: {self.IR_PALLETTE[1]};")
            case 2: self.selected_frame_label.setStyleSheet(f"color: white; background-color: {self.IR_PALLETTE[2]};")
            case 3: self.selected_frame_label.setStyleSheet(f"color: white; background-color: {self.IR_PALLETTE[3]};")
            case 4: self.selected_frame_label.setStyleSheet(f"color: white; background-color: {self.IR_PALLETTE[4]};")
            case _: self.selected_frame_label.setStyleSheet(f"color: #1E90FF; background-color: transparent;")

    def _refresh_slider(self):
        self.is_saved = False
        self.progress_slider.clear_frame_category()
        status_arr = self.frame_status_array
        palette = self.IR_PALLETTE
        self.progress_slider.set_frame_category_array(status_arr, palette)
        self.progress_slider.commit_categories()

    def _determine_list_to_nav(self):
        return np.where(self.frame_status_array != 0)[0].tolist()

    def _setup_uno(self):
        self.uno = Uno_Stack_Dict()
        self.uno.save_state_for_undo(self._compose_dict_for_uno())

    def _update_button_states(self):
        current_status = self.frame_status_array[self.current_frame_idx]
        self.approve_button.setEnabled(current_status != 1)
        self.reject_button.setEnabled(current_status != 2)

    def _export_label(self):
        self.is_saved = True
        self.accept()

    def _i_am_entertained(self):
        self.is_entertained = True
        self._export_label()

    def _save_state_for_undo(self):
        self._compose_dict_for_uno()
        self.uno.save_state_for_undo(self._compose_dict_for_uno())

    def _compose_dict_for_uno(self):
        return {"pred": self.new_data_array, "status": self.frame_status_array}

    def _undo_changes(self):
        self._compose_dict_for_uno()
        uno_dict = self.uno.undo(self._compose_dict_for_uno())
        self.new_data_array = uno_dict["pred"]
        self.frame_status_array = uno_dict["status"]
        self._display_current_frame()
        self._refresh_ui()

    def _redo_changes(self):
        self._compose_dict_for_uno()
        uno_dict = self.uno.redo(self._compose_dict_for_uno())
        self.new_data_array = uno_dict["pred"]
        self.frame_status_array = uno_dict["status"]
        self._display_current_frame()
        self._refresh_ui()

    def reject(self):
        self.was_cancelled = True
        super().reject()

    def closeEvent(self, event):
        self.reject()