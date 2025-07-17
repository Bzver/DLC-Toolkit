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

class DLC_Track_Refiner(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLC Manual Frame Extractor")
        self.setGeometry(100, 100, 1200, 960)
