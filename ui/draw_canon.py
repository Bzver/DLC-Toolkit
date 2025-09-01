import numpy as np
import cv2

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QDialog, QLabel
from PySide6.QtGui import QPixmap, QImage

from .plot import Prediction_Plotter
from utils.dataclass import Loaded_DLC_Data, Plot_Config
from utils import helper as duh

class Canonical_Pose_Dialog(QDialog):
    def __init__(self, dlc_data:Loaded_DLC_Data, canon_pose:np.ndarray, parent=None):

        """
        Initializes a dialog window to visualize the canonical (average) pose in a centered and
        zoomed-in view using 2D keypoint layout.

        Args:
            dlc_data (Loaded_DLC_Data): DLC dataset object containing keypoint names, 
                skeleton structure, and individual information needed for plotting.
            canon_pose (np.ndarray): Array of shape (num_keypoints, 2) representing the 
                canonical 2D coordinates (x, y) of each keypoint. Confidence is assumed to be 1.0.
            parent: Parent widget (typically a QMainWindow).
        """
        super().__init__(parent)
        self.setWindowTitle("Canonical Pose Viewer")
        self.setGeometry(200, 200, 600, 600)

        self.dlc_data = dlc_data
        self.canon_pose = canon_pose

        self.layout = QVBoxLayout(self)
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.draw_canonical_pose()

    def draw_canonical_pose(self):
        """
        Renders the canonical pose onto a blank canvas using the Prediction_Plotter and displays 
        it in the dialog. The pose is automatically scaled and centered for clear visualization.

        Returns:
            None: The method directly updates the image_label with the rendered pose.
                  If no canonical pose data is available, a placeholder text is shown instead.
        """
        if self.canon_pose is None or self.dlc_data is None:
            self.image_label.setText("No canonical pose data available.")
            return

        img_height, img_width = 600, 600
        blank_image = np.full((img_height, img_width, 3), 255, dtype=np.uint8)
        
        min_x, min_y, max_x, max_y = duh.calculate_bbox(self.canon_pose[:, 0], self.canon_pose[:, 1])
        canon_len = max(max_y-min_y, max_x-min_x)

        zoom_factor = 600 // canon_len

        # Reshape canon_pose to be compatible with DLC_Plotter
        num_keypoints = self.canon_pose.shape[0]
        reshaped_pose = np.zeros((1, num_keypoints * 3))
        for i in range(num_keypoints):  # Center the pose
            reshaped_pose[0, i*3] = self.canon_pose[i, 0] * zoom_factor + img_width / 2
            reshaped_pose[0, i*3+1] = self.canon_pose[i, 1] * zoom_factor + img_height / 2
            reshaped_pose[0, i*3+2] = 1.0

        # Create a dummy dlc_data for the plotter to use the skeleton and keypoint names
        dummy_dlc_data = self.dlc_data
        dummy_dlc_data.instance_count = 1
        
        plot_config = Plot_Config(
            plot_opacity=1.0, point_size=6.0, confidence_cutoff=0.0, hide_text_labels=False, edit_mode=False)

        plotter = Prediction_Plotter(
            dlc_data=dummy_dlc_data,
            current_frame_data=reshaped_pose,
            frame_cv2=blank_image,
            plot_config=plot_config,
        )
        plotted_image = plotter.plot_predictions()

        # Convert OpenCV image to QPixmap and display
        rgb_image = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)