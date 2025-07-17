import os

import h5py
import yaml

import pandas as pd
from itertools import islice
import bisect

import cv2

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QShortcut, QKeySequence, QCloseEvent
from PySide6.QtWidgets import QMessageBox, QPushButton

#################   W   ##################   I   ##################   P   ##################   

DLC_CONFIG_DEBUG = "D:/Project/DLC-Models/NTD/config.yaml"
VIDEO_FILE_DEBUG = "D:/Project/DLC-Models/NTD/videos/jobs/20250626C1-first3h-conv/20250626C1-first3h-D.mp4"
PRED_FILE_DEBUG = "D:/Project/DLC-Models/NTD/videos/jobs/20250626C1-first3h-conv/20250626C1-first3h-DDLC_HrnetW32_bezver-SD-20250605M-cam52025-06-26shuffle1_detector_090_snapshot_080_el.h5"

class DLC_Track_Refiner(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLC Track Refiner")
        self.setGeometry(100, 100, 1200, 960)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.load_video_button = QPushButton("Load Video")
        self.load_DLC_config_button = QPushButton("Load DLC Config")
        self.load_prediction_button = QPushButton("Load Prediction")
        self.save_prediction_button = QPushButton("Save Prediction")

        self.button_layout.addWidget(self.load_video_button)
        self.button_layout.addWidget(self.load_DLC_config_button)
        self.button_layout.addWidget(self.load_prediction_button)
        self.button_layout.addWidget(self.save_prediction_button)
        self.layout.addLayout(self.button_layout)

        # Video display area
        self.video_label = QtWidgets.QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.layout.addWidget(self.video_label, 1)

        # Progress bar
        self.progress_layout = QtWidgets.QHBoxLayout()
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(20)
        self.undo_button = QPushButton("⮌")
        self.undo_button.setFixedWidth(20)
        self.redo_button = QPushButton("⮎")
        self.redo_button.setFixedWidth(20)
        self.progress_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.progress_slider.setTracking(True)

        self.progress_layout.addWidget(self.play_button)
        self.progress_layout.addWidget(self.progress_slider)
        self.progress_layout.addWidget(self.undo_button)
        self.progress_layout.addWidget(self.redo_button)
        self.playback_timer = QTimer()
        # self.playback_timer.timeout.connect(self.autoplay_video)
        self.is_playing = False
        self.layout.addLayout(self.progress_layout)

        # Navigation controls
        self.navigation_group_box = QtWidgets.QGroupBox("Video Navigation")
        self.navigation_layout = QtWidgets.QGridLayout(self.navigation_group_box)
        self.prev_10_frames_button = QPushButton("Prev 10 Frames (Shift + ←)")
        self.prev_frame_button = QPushButton("Prev Frame (←)")
        self.next_frame_button = QPushButton("Next Frame (→)")
        self.next_10_frames_button = QPushButton("Next 10 Frames (Shift + →)")

        self.prev_instance_change_button = QPushButton("◄ Prev ROI (↓)")
        self.next_instance_change_button = QPushButton("► Next ROI (↑)")
        self.swap_track_button = QPushButton("Swap Track")
        self.delete_track_button = QPushButton("Delete Track")

        self.navigation_layout.addWidget(self.prev_10_frames_button, 0, 0)
        self.navigation_layout.addWidget(self.prev_frame_button, 0, 1)
        self.navigation_layout.addWidget(self.next_frame_button, 0, 2)
        self.navigation_layout.addWidget(self.next_10_frames_button, 0, 3)

        self.navigation_layout.addWidget(self.prev_instance_change_button, 1, 1)
        self.navigation_layout.addWidget(self.next_instance_change_button, 1, 2)
        self.navigation_layout.addWidget(self.swap_track_button, 1, 0)
        self.navigation_layout.addWidget(self.delete_track_button, 1, 3)

        self.layout.addWidget(self.navigation_group_box)

        # # Connect buttons to events
        # self.load_video_button.clicked.connect(self.load_video)
        # self.load_DLC_config_button.clicked.connect(self.load_DLC_config)
        # self.load_prediction_button.clicked.connect(self.load_prediction)
        # self.save_prediction_button.clicked.connect(self.save_prediction)

        # self.progress_slider.sliderMoved.connect(self.set_frame_from_slider)
        # self.play_button.clicked.connect(self.toggle_playback)
        # self.undo_button.clicked.connect(self.undo_changes)
        # self.redo_button.clicked.connect(self.redo_changes)

        # self.prev_10_frames_button.clicked.connect(lambda: self.change_frame(-10))
        # self.prev_frame_button.clicked.connect(lambda: self.change_frame(-1))
        # self.next_frame_button.clicked.connect(lambda: self.change_frame(1))
        # self.next_10_frames_button.clicked.connect(lambda: self.change_frame(10))

        # self.prev_instance_change_button.clicked.connect(self.prev_instance_change)
        # self.next_instance_change_button.clicked.connect(self.next_instance_change)
        # self.swap_track_button.clicked.connect(self.swap_track)
        # self.delete_track_button.clicked.connect(self.delete_track)

        # QShortcut(QKeySequence(Qt.Key_Left | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(-10))
        # QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(lambda: self.change_frame(-1))
        # QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(lambda: self.change_frame(1))
        # QShortcut(QKeySequence(Qt.Key_Right | Qt.ShiftModifier), self).activated.connect(lambda: self.change_frame(10))
        # QShortcut(QKeySequence(Qt.Key_W), self).activated.connect(self.swap_track)
        # QShortcut(QKeySequence(Qt.Key_X), self).activated.connect(self.delete_track)
        # QShortcut(QKeySequence(Qt.Key_Z | Qt.ShiftModifier), self).activated.connect(self.undo_changes)
        # QShortcut(QKeySequence(Qt.Key_Y | Qt.ShiftModifier), self).activated.connect(self.redo_changes)
        # QShortcut(QKeySequence(Qt.Key_Down), self).activated.connect(self.prev_instance_change)
        # QShortcut(QKeySequence(Qt.Key_Up), self).activated.connect(self.next_instance_change)
        # QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_playback)
        # QShortcut(QKeySequence(Qt.Key_S | Qt.ControlModifier), self).activated.connect(self.save_frame_mark)

        self.reset_state()


    def reset_state(self):
        self.original_vid, self.prediction, self.dlc_dir, self.video_name = None, None, None, None
        self.keypoints, self.skeleton, self.individuals, self.project_dir = None, None, None, None

        self.instance_count = 1
        self.multi_animal = False
        self.pred_data = None

        self.labeled_frame_list, self.frame_list = [], []

        self.cap, self.current_frame = None, None
        self.confidence_cutoff = 0 # Default confidence cutoff

        self.is_playing = False
        self.is_saved = True
        self.last_saved = []

        self.progress_slider.setRange(0, 0)
        self.navigation_group_box.hide()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = DLC_Track_Refiner()
    window.show()
    app.exec()