import pandas as pd
import numpy as np

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

import traceback

from ui import Menu_Widget, Video_Player_Widget, Frame_List_Dialog, Shortcut_Manager, Status_Bar, Inference_interval_Dialog
from utils.helper import frame_to_pixmap
from .data_man import Data_Manager
from .video_man import  Video_Manager
from .tool import Mark_Generator, Blob_Counter, Prediction_Plotter

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


    