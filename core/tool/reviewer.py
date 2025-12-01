import numpy as np
import cv2

from PySide6 import QtGui
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox, QMessageBox)

from typing import List, Tuple, Optional

from .plot import Prediction_Plotter
from .undo_redo import Uno_Stack
from .mark_nav import navigate_to_marked_frame
from ui import Clickable_Video_Label, Video_Slider_Widget, Shortcut_Manager
from utils.helper import frame_to_pixmap, handle_unsaved_changes_on_close
from utils.track import swap_track
from core.dataclass import Loaded_DLC_Data
from core.io import Frame_Extractor, Frame_Extractor_Img

class Parallel_Review_Dialog(QDialog):
    pred_data_exported = Signal(object, tuple)
    RERUN_PALLETTE = {0: "#959595", 1: "#68b3ff", 2: "#F749C6"}
    TC_PALLETTE    = {0: "#009353", 1: "#FF0040"}

    def __init__(self,
                 dlc_data:Loaded_DLC_Data,
                 extractor:Frame_Extractor|Frame_Extractor_Img,
                 new_data_array:np.ndarray,
                 frame_list:List[int], # Used to map local idx to global idx, irrelevant if tc_mode
                 tc_frame_tuple:Optional[Tuple[List[int], List[int]]]=None,
                 tc_mode:bool=False,
                 parent=None):
        super().__init__(parent)
        self.dlc_data = dlc_data
        self.extractor = extractor
        self.new_data_array = new_data_array
        self.tc_mode = tc_mode

        self.backup_data_array = dlc_data.pred_data_array.copy()
        self.pred_data_array = dlc_data.pred_data_array.copy()

        self.total_frames = self.pred_data_array.shape[0]
        self.frame_list = list(range(self.total_frames)) if tc_mode else frame_list
        self.total_marked_frames = len(self.frame_list)

        self.current_frame_idx = 0
        self.is_saved = False
        self.uno = Uno_Stack()

        if self.tc_mode:
            if tc_frame_tuple is None:
                raise TypeError("Missing 1 required positional argument: 'tc_frame_tuple")
            self.inst_array = np.array(range(self.dlc_data.pred_data_array.shape[1]))
            self.uno.save_state_for_undo(self.new_data_array)
            self.corrected_frames, self.ambiguous_frames = tc_frame_tuple
        else:
            self.frame_status_array = np.zeros((self.total_marked_frames,), dtype=np.uint8)
            self.uno.save_state_for_undo(self.frame_status_array)

        self.plotter = Prediction_Plotter(dlc_data=self.dlc_data)

        layout = QVBoxLayout(self)
        frame_info_layout, video_label_layout = self._setup_video_display()
        layout.addLayout(frame_info_layout)
        layout.addLayout(video_label_layout)
        layout.addLayout(self.video_layout)

        self.progress_slider = Video_Slider_Widget()
        self.progress_slider.set_total_frames(self.total_marked_frames)
        self.progress_slider.set_current_frame(0)
        self.progress_slider.frame_changed.connect(self._handle_frame_change_from_comp)
        layout.addWidget(self.progress_slider)

        approval_box = self._setup_control_tc() if self.tc_mode else self._setup_control()

        layout.addWidget(approval_box)
        
        self._build_shortcut()
        self._navigation_title_controller()
        self._update_button_states()

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

        if self.tc_mode:
            self.global_frame_label.setVisible(False)

        video_label_layout = QHBoxLayout()
        old_label = QLabel("Old Predictions")
        old_label.setAlignment(Qt.AlignCenter)
        old_label.setFont(font)
        old_label.setStyleSheet("color: #4B4B4B; background: #F0F0F0; padding: 6px; border-radius: 4px;")
        old_label.setFixedHeight(30)

        new_label = QLabel("New Predictions")
        new_label.setAlignment(Qt.AlignCenter)
        new_label.setFont(font)
        new_label.setStyleSheet("color: #4B4B4B; background: #F0F0F0; padding: 6px; border-radius: 4px;")
        new_label.setFixedHeight(30)

        video_label_layout.addWidget(old_label)
        video_label_layout.addWidget(new_label)

        self.video_layout = QHBoxLayout()
        self.video_labels = []
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

    def _setup_control_tc(self):
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

        swap_layout.addWidget(self.swap_button)
        swap_layout.addWidget(self.big_swap_button)
        swap_layout.addWidget(self.apply_button)
        swap_box.setLayout(swap_layout)

        return swap_box

    def _build_shortcut(self):
        self.shortcut_man = Shortcut_Manager(self)
        common_sc={
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
        }
        ntc_sc={
            **common_sc,
            "approve":{"key": "A", "callback": lambda: self._mark_frame_status(True)},
            "reject":{"key": "R", "callback": lambda: self._mark_frame_status(False)},
        }
        tc_sc={
            **common_sc,
            "swap":{"key": "W", "callback": self._swap_instance},
            "big_swap":{"key": "Shift+W", "callback": self._swap_track},
        }
        shortcuts = tc_sc if self.tc_mode else ntc_sc
        self.shortcut_man.add_shortcuts_from_config(shortcuts)

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
            pixmap, _, _ = frame_to_pixmap(resized)
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
        if self.tc_mode:
            return self.ambiguous_frames if self.ambiguous_frames else self.corrected_frames
        else:
            return np.where(self.frame_status_array==1)[0]

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

    def _swap_instance(self):
        self._save_state_for_undo()
        self.new_data_array = swap_track(self.new_data_array, self.current_frame_idx)
        self._display_current_frame()
        self._refresh_ui()

    def _swap_track(self):
        self._save_state_for_undo()
        self.new_data_array = swap_track(self.new_data_array, self.current_frame_idx, swap_range=[-1])
        self._display_current_frame()
        self._refresh_ui()

    ###################################################################################################

    def _navigation_title_controller(self):
        global_idx = self.frame_list[self.current_frame_idx]
        self.global_frame_label.setText(f"Global: {global_idx} / {self.total_frames - 1}")
        self.selected_frame_label.setText(f"Selected: {self.current_frame_idx} / {self.total_marked_frames - 1}")

    def _update_button_states(self):
        if self.tc_mode:
            self.apply_button.setEnabled(True)
            return
        
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
        if self.tc_mode:
            self.progress_slider.set_frame_category("ambiguous", self.ambiguous_frames, self.TC_PALLETTE[1], priority=5)
            self.progress_slider.set_frame_category("changed", self.corrected_frames, self.TC_PALLETTE[0])
        else:
            self.progress_slider.set_frame_category_array(self.frame_status_array, self.RERUN_PALLETTE)
        self.progress_slider.commit_categories()

    ###################################################################################################
    
    def _save_state_for_undo(self):
        data_array = self.new_data_array if self.tc_mode else self.frame_status_array
        self.uno.save_state_for_undo(data_array)

    def _undo_changes(self):
        data_array = self.uno.undo()
        self._undo_redo_worker(data_array)

    def _redo_changes(self):
        data_array = self.uno.redo()
        self._undo_redo_worker(data_array)

    def _undo_redo_worker(self, data_array):
        if data_array is None:
            return

        if self.tc_mode:
            self.new_data_array = data_array.copy()
        elif np.any(self.frame_status_array!=data_array):
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
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Unprocessed Frames - Save and Exit?")
            msg_box.setText(
                f"You have {np.sum(self.frame_status_array==0)} unprocessed frame(s).\n\n"
                "This action will save your progress and close the window.\n"
                "All unprocessed frames will remain as original and will not be marked as approved.\n\n"
                "Are you sure you want to continue?"
            )
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
            reply = msg_box.exec_()

            if reply == QMessageBox.No:
                return

        self.is_saved = True
        if self.tc_mode:
            self.pred_data_exported.emit(self.new_data_array, ())
        else:
            approve_list_global = np.array(self.frame_list)[self.frame_status_array==1].tolist()
            rejected_list_global = np.array(self.frame_list)[self.frame_status_array==2].tolist()
            list_tuple = (approve_list_global, rejected_list_global)
            self.pred_data_exported.emit(self.pred_data_array, list_tuple)
        self.accept()

    def closeEvent(self, event):
        if not self.tc_mode:
            self.extractor.close()
        handle_unsaved_changes_on_close(self, event, self.is_saved, self._save_prediction)