import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QDialog, QLabel
from PySide6.QtGui import QTransform

from .plot import Prediction_Plotter
from .graphic_view import Canvas
from utils.helper import frame_to_pixmap
from utils.dataclass import Loaded_DLC_Data

class Canonical_Pose_Dialog(QDialog):
    def __init__(self, dlc_data:Loaded_DLC_Data, canon_pose:np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Canonical Pose Viewer")
        self.setGeometry(200, 200, 600, 600)

        self.dlc_data = dlc_data
        self.canon_pose = canon_pose

        self.pose_layout = QVBoxLayout(self)
        self.placeholder_label = QLabel("No canonical pose data available.")
        self.placeholder_label.setVisible(False)
        self.gview = Canvas(self)
        self.pose_layout.addWidget(self.placeholder_label)
        self.pose_layout.addWidget(self.gview)

        self.draw_canonical_pose()

    def draw_canonical_pose(self):
        if self.canon_pose is None or self.dlc_data is None:
            self.placeholder_label.setVisible(True)
            return

        img_height, img_width = 600, 600
        blank_image = np.full((img_height, img_width, 3), 255, dtype=np.uint8)
        pixmap, w, h = frame_to_pixmap(blank_image, request_dim=True)
        pixmap_item = self.gview.gscene.addPixmap(pixmap)
        pixmap_item.setZValue(-1)
        self.gview.gscene.setSceneRect(0, 0, w, h)

        new_transform = QTransform()
        new_transform.scale(self.gview.zoom_factor, self.gview.zoom_factor)
        self.gview.setTransform(new_transform)
        self.gview.toggle_zoom_mode()

        dummy_dlc_data = self.dlc_data
        dummy_dlc_data.instance_count = 1

        flatten_canon = np.full((2, self.dlc_data.num_keypoint*3,), np.nan)
        flatten_canon[0, 0::3] = self.canon_pose[:, 0] + 0.5 * w
        flatten_canon[0, 1::3] = self.canon_pose[:, 1] + 0.5 * h

        plotter = Prediction_Plotter(dlc_data=dummy_dlc_data, fast_mode=False)
        plotter.plot_predictions(self.gview.gscene, flatten_canon)