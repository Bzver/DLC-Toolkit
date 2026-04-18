import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout

from utils.dataclass import Loaded_DLC_Data


class Canvas_3D(QVBoxLayout):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setMouseTracking(True)
        self.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111, projection='3d')

        self.reset_c3d()

    def reset_c3d(self):
        self.ax.clear()
        self.ax.set_title("3D Skeleton Plot - No data loaded")
        self.canvas.draw_idle()

    def plot_3d_points(self, dlc_data:Loaded_DLC_Data, point_3d_array:np.ndarray):
        if dlc_data is None or point_3d_array is None:
            self.reset_c3d()
            return

        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f"3D Skeleton Plot - Frame {self.current_frame_idx}")
        self.ax.set_xlim([-self.plot_lim, self.plot_lim])
        self.ax.set_ylim([-self.plot_lim, self.plot_lim])
        self.ax.set_zlim([-self.plot_lim // 5, self.plot_lim])

        for inst in range(self.dlc_data.instance_count):
            color = self.instance_color[inst % len(self.instance_color)]
            for kp_idx in range(self.dlc_data.num_keypoint):
                point_3d = point_3d_array[inst, kp_idx, :]
                if self._check_point_validity(point_3d):
                    self.ax.scatter(point_3d[0], point_3d[1], point_3d[2], color=np.array(color)/255, s=50)

            if self.dlc_data.skeleton:
                for start_kp, end_kp in self.dlc_data.skeleton:
                    start_kp_idx = self.keypoint_to_idx.get(start_kp)
                    end_kp_idx = self.keypoint_to_idx.get(end_kp)
                    
                    if start_kp_idx is not None and end_kp_idx is not None:
                        start_point = point_3d_array[inst, start_kp_idx, :]
                        end_point = point_3d_array[inst, end_kp_idx, :]
                        if self._check_point_validity(start_point) and self._check_point_validity(end_point):
                            self.ax.plot([start_point[0], end_point[0]],
                                         [start_point[1], end_point[1]],
                                         [start_point[2], end_point[2]],
                                         color=np.array(color)/255)
        self.canvas.draw_idle()

    def plot_camera_geometry(self):
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f"3D Camera Geometry")
        self.ax.set_xlim([-self.plot_lim * 3, self.plot_lim * 3])
        self.ax.set_ylim([-self.plot_lim * 3, self.plot_lim * 3])
        self.ax.set_zlim([-self.plot_lim * 3 // 5, self.plot_lim * 3])

        for i in range(self.num_cam_from_calib):
            self.ax.scatter(*self.cam_pos[i], s=100, label=f"Camera {i+1} Pos")
            self.ax.quiver(*self.cam_pos[i], *self.cam_dir[i], length=100, color='blue', normalize=True)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
        Z = np.zeros_like(X)
        self.ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')
        self.canvas.draw_idle()

    def _check_point_validity(self, pts3d:np.ndarray):
        return np.all(~np.isnan(pts3d))