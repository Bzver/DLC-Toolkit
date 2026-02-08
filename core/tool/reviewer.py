import numpy as np
import cv2

from PySide6 import QtGui
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox

from typing import List, Tuple, Optional, Literal

from .plot import Prediction_Plotter
from .undo_redo import Uno_Stack
from .mark_nav import navigate_to_marked_frame
from core.io import Frame_Extractor, Frame_Extractor_Img
from ui import Clickable_Video_Label, Video_Slider_Widget, Shortcut_Manager
from utils.helper import (
    frame_to_pixmap, handle_unsaved_changes_on_close, crop_coord_to_array, validate_crop_coord, indices_to_spans
    )
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
                 grayscaling:bool=False,
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

        self.grayscaling = grayscaling

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
    def __init__(
            self,
            dlc_data:Loaded_DLC_Data,
            extractor:Frame_Extractor,
            pred_data_array:np.ndarray,
            current_frame_idx:int,
            mode: Literal["swap", "exit", "return"]="swap",
            last_event_idx:Optional[int]=None,
            parent=None
            ):
        total_frames = pred_data_array.shape[0]
        frame_list_start = last_event_idx if last_event_idx is not None else current_frame_idx-20
        self.frame_list = sorted(range(max(0, frame_list_start), min(current_frame_idx+21, total_frames)))
        self.mode = mode
        super().__init__(dlc_data, extractor, new_data_array=pred_data_array, frame_list=self.frame_list, crop_coord=None, parent=parent)
        self.nonempty_local = set(np.where(np.any(~np.isnan(pred_data_array[self.frame_list]), axis=(1, 2)))[0].tolist())
        self.current_frame_idx = self.frame_list.index(current_frame_idx)

        self.old_label.setText("Last Frame With Prediction")
        self.new_label.setText("Current Frame")
        self._display_current_frame()

    def _setup_control(self):
        mode_cap = self.mode.capitalize()

        ctrl_box = QGroupBox(f"{mode_cap} Confirmation")
        ctrl_layout = QHBoxLayout()

        self.yes_button = QPushButton(f"{mode_cap}")
        self.no_button = QPushButton(f"No {mode_cap}")

        self.yes_button.clicked.connect(lambda:self._return_status(True))
        self.no_button.clicked.connect(lambda:self._return_status(False))

        ctrl_layout.addWidget(self.yes_button)
        ctrl_layout.addWidget(self.no_button)
        ctrl_box.setLayout(ctrl_layout)

        return ctrl_box

    def _setup_uno(self):
        pass

    def _build_shortcut(self):
        self.shortcut_man = Shortcut_Manager(self)
        self.shortcut_man.add_shortcuts_from_config({
            "prev_frame":{"key": "Left", "callback": lambda: self._change_frame(-1)},
            "next_frame":{"key": "Right", "callback": lambda: self._change_frame(1)},
            "prev_fast":{"key": "Shift+Left", "callback": lambda: self._change_frame(-10)},
            "next_fast":{"key": "Shift+Right", "callback": lambda: self._change_frame(10)},
            "playback":{"key": "Space", "callback": self._toggle_playback},
        })

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

    def _return_status(self, status:bool):
        self.user_decision = (status, self.frame_list[self.current_frame_idx])
        self.accept()

    def _back_search_nonempty_frames(self, lookback_window: int = 10) -> int:
        start_local = self.current_frame_idx
        for offset in range(1, min(lookback_window + 1, start_local + 1)):
            candidate_local = start_local - offset
            if candidate_local in self.nonempty_local:
                return candidate_local
        return max(0, start_local - 1)

    def _update_button_states(self):
        pass

    def closeEvent(self, event):
        pass