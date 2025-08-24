import numpy as np

from PySide6 import QtGui
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsLineItem, QGraphicsScene

from typing import Optional

from .dtu_dataclass import Loaded_DLC_Data, Plot_Config, Refiner_Plotter_Callbacks
from .dtu_comp import Selectable_Instance, Draggable_Keypoint

class DLC_Plotter:
    def __init__(self, dlc_data:Loaded_DLC_Data, current_frame_data:np.ndarray, graphics_scene:QGraphicsScene,
            plot_config:Plot_Config, plot_callback:Optional[Refiner_Plotter_Callbacks]=None):
        
        self.dlc_data = dlc_data
        self.current_frame_data = current_frame_data
        self.graphics_scene = graphics_scene
        self.plot_config = plot_config
        self.plot_callback = plot_callback

        self.color = [(255, 165, 0), (51, 255, 51), (51, 153, 255), (255, 51, 51), (255, 255, 102)]
        self.keypoint_coords = {}

    def plot_predictions(self):
        for inst in range(self.dlc_data.instance_count):
            self.keypoint_coords = {} # Cleanup the keypoint coords
            color = self.color[inst % len(self.color)]
            
            for kp_idx in range(self.dlc_data.num_keypoint):
                kp = self.current_frame_data[inst,kp_idx*3:kp_idx*3+3]
                if np.isnan(kp[0]):
                    continue
                x, y, conf = kp[0], kp[1], kp[2]
                self.keypoint_coords[kp_idx] = (float(x),float(y),float(conf))

                # QGraphicsEllipseItem dot representing the keypoints 
                keypoint_item = Draggable_Keypoint(
                    x - self.plot_config.point_size / 2, y - self.plot_config.point_size / 2,
                    self.plot_config.point_size, self.plot_config.point_size, inst, kp_idx, default_color_rgb=color)
                keypoint_item.setOpacity(self.plot_config.plot_opacity)

                if isinstance(keypoint_item, Draggable_Keypoint):
                    keypoint_item.setFlag(QGraphicsEllipseItem.ItemIsMovable, self.plot_config.edit_mode)

                self.graphics_scene.addItem(keypoint_item)
                keypoint_item.setZValue(1) # Ensure keypoints are on top of the video frame
                keypoint_item.keypoint_moved.connect(self.plot_callback.keypoint_coords_callback)
                keypoint_item.keypoint_drag_started.connect(self.plot_callback.keypoint_object_callback)

            if not self.plot_config.hide_text_labels:
                self._plot_keypoint_label(color)

            if self.dlc_data.individuals is not None and len(self.keypoint_coords) >= 2:
                self._plot_bounding_box(color, inst)

            if self.dlc_data.skeleton:
                self._plot_skeleton(color)

    def _plot_bounding_box(self, color, inst, padding = 10):
        x_coords = [self.keypoint_coords[p][0] for p in self.keypoint_coords if self.keypoint_coords[p] is not None]
        y_coords = [self.keypoint_coords[p][1] for p in self.keypoint_coords if self.keypoint_coords[p] is not None]
        kp_confidence = [self.keypoint_coords[p][2] for p in self.keypoint_coords if self.keypoint_coords[p] is not None]

        if not x_coords or not y_coords: # Skip if the mice has no keypoint
            return

        kp_inst_mean = sum(kp_confidence) / len(kp_confidence)
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        view_width = self.graphics_scene.sceneRect().width()
        view_height = self.graphics_scene.sceneRect().height()
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(view_width - 1, max_x + padding)
        max_y = min(view_height - 1, max_y + padding) # Or use frame.shape[0]

        # Draw bounding box using QGraphicsRectItem
        rect_item = Selectable_Instance(min_x, min_y, max_x - min_x, max_y - min_y, inst, default_color_rgb=color)
        rect_item.setOpacity(self.plot_config.plot_opacity)

        if isinstance(rect_item, Selectable_Instance):
            rect_item.setFlag(QGraphicsRectItem.ItemIsMovable, self.plot_config.edit_mode)

        self.graphics_scene.addItem(rect_item)
        rect_item.bounding_box_moved.connect(self.plot_callback.box_coords_callback)
        rect_item.bounding_box_clicked.connect(self.plot_callback.box_object_callback)

        if not self.plot_config.hide_text_labels: # Add individual label and keypoint labels
            text_item_inst = QGraphicsTextItem(f"Inst: {self.dlc_data.individuals[inst]} | Conf:{kp_inst_mean:.4f}")
            text_item_inst.setPos(min_x, min_y - 20) # Adjust position to be above the bounding box
            text_item_inst.setDefaultTextColor(QtGui.QColor(*color))
            text_item_inst.setOpacity(self.plot_config.plot_opacity)
            text_item_inst.setFlag(QGraphicsTextItem.ItemIgnoresTransformations) # Keep text size constant
            self.graphics_scene.addItem(text_item_inst)

    def _plot_keypoint_label(self, color):
        for kp_idx, (x, y, conf) in self.keypoint_coords.items():
            keypoint_label = self.dlc_data.keypoints[kp_idx]

            text_item = QGraphicsTextItem(f"{keypoint_label}")

            font = text_item.font() # Get the default font of the QGraphicsTextItem
            fm = QtGui.QFontMetrics(font)
            text_rect = fm.boundingRect(keypoint_label)
            
            text_width = text_rect.width()
            text_height = text_rect.height()

            text_x = x - text_width / 2 + 5
            text_y = y - text_height / 2 + 5

            text_item.setPos(text_x, text_y)
            text_item.setDefaultTextColor(QtGui.QColor(*color))
            text_item.setOpacity(self.plot_config.plot_opacity)
            text_item.setFlag(QGraphicsTextItem.ItemIgnoresTransformations) # Keep text size constant
            self.graphics_scene.addItem(text_item)

    def _plot_skeleton(self, color):
        for start_kp, end_kp in self.dlc_data.skeleton:
            start_kp_idx = self.dlc_data.keypoint_to_idx[start_kp]
            end_kp_idx = self.dlc_data.keypoint_to_idx[end_kp]
            start_coord = self.keypoint_coords.get(start_kp_idx)
            end_coord = self.keypoint_coords.get(end_kp_idx)

            if start_coord and end_coord:
                line = QGraphicsLineItem(start_coord[0], start_coord[1], end_coord[0], end_coord[1])
                line.setPen(QtGui.QPen(QtGui.QColor(*color), self.plot_config.point_size / 3))
                line.setOpacity(self.plot_config.plot_opacity)
                self.graphics_scene.addItem(line)