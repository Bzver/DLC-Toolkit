import os
import yaml
import numpy as np
from PySide6 import QtGui
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QComboBox, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QLineEdit
from typing import Tuple, Optional

from .plot import Prediction_Plotter
from core.io import Frame_Extractor, Prediction_Loader, get_existing_projects, generate_crop_coord_notations
from ui import Frame_Display_Dialog, Spinbox_With_Label
from utils.helper import frame_to_qimage, validate_crop_coord
from utils.logger import Loggerbox, logger
from utils.dataclass import Loaded_DLC_Data


class DLC_Save_Dialog(QDialog):
    folder_selected = Signal(str)

    def __init__(self, dlc_data:Loaded_DLC_Data, roi, video_file, parent=None):
        super().__init__(parent)

        self.dlc_data = dlc_data
        self.roi = roi
        self.video_file = video_file

        self.video_name, _ = os.path.splitext(os.path.basename(video_file))
        
        btn_layout = QHBoxLayout()

        self.old_btn = QPushButton("Save to Existing Project")
        self.old_btn.clicked.connect(self._save_old_proj)
        self.new_btn = QPushButton("Save to New Project")
        self.new_btn.clicked.connect(self._save_new_proj)

        btn_layout.addWidget(self.old_btn)
        btn_layout.addWidget(self.new_btn)

        self.setLayout(btn_layout)
        self.setWindowTitle("Select Saving Option")

    def _save_old_proj(self):
        old_dd = Load_Label_Dialog(self.dlc_data, self.roi, self.video_file)
        old_dd.folder_selected.connect(self._folder_selected)
        old_dd.exec()

    def _save_new_proj(self):
        new_dd = New_Folder_Name_Dialog(self.video_name, self)
        new_dd.folder_selected.connect(self._folder_selected)
        new_dd.exec()
    
    def _folder_selected(self, folder_path:str):
        dlc_folder = os.path.dirname(self.dlc_data.dlc_config_filepath)
        save_path = os.path.join(dlc_folder, "labeled-data", folder_path)
        os.makedirs(save_path, exist_ok=True)
        self.folder_selected.emit(save_path)
        self.accept()


class New_Folder_Name_Dialog(QDialog):
    folder_selected = Signal(str)

    def __init__(self, video_name=None, parent=None):
        super().__init__(parent)

        dialog_layout = QHBoxLayout()

        self.input = QLineEdit()
        self.input.setText(video_name)
        self.okay_btn = QPushButton("Confirm")
        self.okay_btn.clicked.connect(self._on_confirm)

        dialog_layout.addWidget(self.input)
        dialog_layout.addWidget(self.okay_btn)

        self.setLayout(dialog_layout)
        self.setWindowTitle("Enter New Project Name")

    def _on_confirm(self):
        name = self.input.text().strip()
        if not name:
            Loggerbox.warning(self, "Invalid Name", "Project name cannot be empty.")
            return
        if os.path.sep in name or (':' in name and os.name == 'nt'):
            Loggerbox.warning(self, "Invalid Name", "Project name cannot contain path separators.")
            return
        self.folder_selected.emit(name)
        self.accept()


class Load_Label_Dialog(QDialog):
    folder_selected = Signal(str)

    def __init__(
            self,
            dlc_data:Loaded_DLC_Data,
            roi:Optional[Tuple[int,int,int,int]]=None,
            video_file:Optional[str]=None,
            parent=None
            ):
        super().__init__(parent)
        self.setWindowTitle("Load DLC Label")
        self.dlc_data = dlc_data
        if video_file is not None:
            self.video_name, _ = os.path.splitext(os.path.basename(video_file))
        else:
            self.video_name = None

        self._swap_time = 0

        self.config_path = self.dlc_data.dlc_config_filepath
        self.project_folders = get_existing_projects(self.config_path)
        if not self.project_folders:
            Loggerbox.info(self, "No Existing Labels Found", "No valid existing labels found in /labeled-data/.")
            self.accept()
            return
        
        self.project_dict = {os.path.basename(f):f for f in self.project_folders}
        
        self.selected_folder = None
        self.roi = validate_crop_coord(roi)

        if video_file is not None and os.path.isfile(video_file):
            self.extractor = Frame_Extractor(video_file)
        else:
            self.extractor = None

        label_layout = QVBoxLayout()

        combo_frame = QHBoxLayout()
        combo_label = QLabel("Select DLC Label")
        self.combo = QComboBox()
        self.combo.addItems(self.project_dict.keys())
        index = self.combo.findText(self.video_name)
        self.combo.currentIndexChanged.connect(self._on_selection_changed)
        self.combo.setCurrentIndex(index)

        combo_frame.addWidget(combo_label)
        combo_frame.addWidget(self.combo)

        self.okay_frame = QHBoxLayout()
        self.offset_btn = QPushButton("Check Label Offset")
        self.offset_btn.clicked.connect(self._check_offset)
        self.okay_btn = QPushButton("Accept")
        self.okay_btn.clicked.connect(self._okay)

        self.okay_frame.addWidget(self.offset_btn)
        self.okay_frame.addWidget(self.okay_btn)

        label_layout.addLayout(combo_frame)
        label_layout.addLayout(self.okay_frame)
        self.setLayout(label_layout)

    def _check_offset(self):
        if not self.selected_folder or not self.extractor:
            self._swap_buttons()
            return
        
        crop_file = os.path.join(self.selected_folder, "crop.yaml")
        if os.path.isfile(crop_file):
            with open(crop_file, 'r') as f:
                crop_f = yaml.safe_load(f)
            regions = crop_f.get("crop_regions", [])
            if regions:
                x1 = regions[0].get("x", 0)
                y1 = regions[0].get("y", 0)
            else:
                x1 = y1 = 0
        elif self.roi is not None:
            x1, y1, _, _ = self.roi
        else:
            x1, y1 = 0, 0

        self.label_offset = (x1, y1)
        self._get_sample_frame_data()
        image = frame_to_qimage(self.frame)
        self.od_dialog = Offset_Display_Dialog("Offset Adjustment", image, self.label_offset, self)
        self.od_dialog.offset_changed.connect(self._on_odd_change)
        self.od_dialog.exec()
        self._plot_frame()

    def _get_sample_frame_data(self):
        pred_path = os.path.join(self.selected_folder, f"CollectedData_{self.dlc_data.scorer}.h5")
        try:
            loader = Prediction_Loader(self.dlc_data.dlc_config_filepath, pred_path)
            loaded_data = loader.load_data()
            loaded_data_array = loaded_data.pred_data_array
        except Exception as e:
            logger.exception(f"Failed to read data from DLC label: {e}")
            return

        self.frame_list = np.where(np.any(~np.isnan(loaded_data_array), axis=(1,2)))[0].tolist()

        frame_idx = self.frame_list[0]
        self.frame = self.extractor.get_frame(frame_idx)
        self.current_frame_data = loaded_data_array[frame_idx]

        self.plotter = Prediction_Plotter(self.dlc_data)

    def _plot_frame(self):
        frame_data = self.current_frame_data.copy()
        frame = self.frame.copy()
        frame_data[..., 0::3] += self.label_offset[0]
        frame_data[..., 1::3] += self.label_offset[1]
        frame = self.plotter.plot_predictions(frame, frame_data)
        image = frame_to_qimage(frame)
        self.od_dialog.update_image(image)

    def _swap_buttons(self):
        self._swap_time += 1

        item_j = self.okay_frame.takeAt(1)
        widget_j = item_j.widget()
        item_i = self.okay_frame.takeAt(0)
        widget_i = item_i.widget()
        self.okay_frame.insertWidget(0, widget_j)
        self.okay_frame.insertWidget(1, widget_i)

        taunts = [
            None,
            "Try again.",
            "Hmm. Maybe this button is for another use case?",
            "Perhaps come back when you've already loaded an existing prediction?",
            "Admiring your persistence. Truly.",
            "Warning: Excessive clicking may cause\n   - mild confusion  - sudden awareness  - urge to read the README",
            "I give up. You win.",
            (
            "<pre>  The cow says: *mooove* a prediction in place first.\n"
            "        ^__^\n"
            "        (oo)\\_______\n"
            "        (__)\\       )\\/\\\n"
            "            ||---ww |\n"
            "            ||     ||\n\n"
            "</pre>"
            )
        ]
        if self._swap_time < len(taunts):
            self.offset_btn.setToolTip(taunts[self._swap_time])
        else:
            self.offset_btn.setVisible(False)

    def _on_selection_changed(self, index):
        if index >= 0:
            text = self.combo.itemText(index)
            self.selected_folder = self.project_dict[text]
            
    def _on_odd_change(self, offset:Tuple[int,int]):
        self.label_offset = offset
        self._plot_frame()

    def _okay(self):
        if hasattr(self, "label_offset") and self.label_offset != (0,0):
            generate_crop_coord_notations(
                label_offset=self.label_offset,
                project_dir=self.selected_folder,
                frame_list=self.frame_list
                )
        self.folder_selected.emit(self.selected_folder)
        self.accept()


class Offset_Display_Dialog(Frame_Display_Dialog):
    offset_changed = Signal(tuple)

    def __init__(self, title, image, offset:Tuple[int, int], parent=None):
        super().__init__(title, image, parent)
        self.x, self.y = offset

        self.x_spin = Spinbox_With_Label("X:", (0, 10000), self.x)
        self.y_spin = Spinbox_With_Label("Y:", (0, 10000), self.y)
        self.x_spin.value_changed.connect(self._on_spinbox_spin)
        self.y_spin.value_changed.connect(self._on_spinbox_spin)

        spin_frame = QHBoxLayout()
        spin_frame.addWidget(self.x_spin)
        spin_frame.addWidget(self.y_spin)

        self.dialog_layout.addLayout(spin_frame)
        self.setWindowTitle("Offset Viewer")

    def update_image(self, image):
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def _on_spinbox_spin(self):
        self.x = self.x_spin.value()
        self.y = self.y_spin.value()
        self.offset_changed.emit((self.x, self.y))
