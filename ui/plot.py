import numpy as np

import cv2
from PySide6 import QtGui
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsLineItem, QGraphicsScene

from typing import Optional, Tuple

from utils.dataclass import Loaded_DLC_Data, Plot_Config, Refiner_Plotter_Callbacks
from ui import Selectable_Instance, Draggable_Keypoint

class Prediction_Plotter:
    def __init__(
            self, dlc_data:Loaded_DLC_Data,
            current_frame_data:np.ndarray,
            plot_config:Optional[Plot_Config]=None,
            frame_cv2:Optional[np.ndarray]=None,
            graphics_scene:Optional[QGraphicsScene]=None,
            plot_callback:Optional[Refiner_Plotter_Callbacks]=None
            ):
        """
        Initializes the prediction plotter for visualizing 2D pose predictions either on a CV2 image 
        or a QGraphicsScene. Supports interactive editing and real-time rendering.

        Args:
            dlc_data (Loaded_DLC_Data): Metadata including instance count, keypoint names, skeleton
                structure, and individual IDs.
            current_frame_data (np.ndarray): Flattened array of shape (num_instances * num_keypoints * 3,) 
                containing x, y, confidence for each keypoint of each instance.
            plot_config (Plot_Config, optional): Configuration for plotting appearance and behavior. 
                Uses default values if not provided.
            frame_cv2 (np.ndarray, optional): BGR image frame for OpenCV-based drawing. 
                Used when plotting in "CV" mode.
            graphics_scene (QGraphicsScene, optional): Qt graphics scene for interactive rendering. 
                Used when plotting in "GS" mode.
            plot_callback (Refiner_Plotter_Callbacks, optional): Callback handler for keypoint drag 
                events; required for interactive editing in graphics scene mode.

        Raises:
            ValueError: If neither or both of 'frame_cv2' and 'graphics_scene' are provided.
        """
        self.dlc_data = dlc_data
        self.current_frame_data = current_frame_data
        self.frame_cv2 = frame_cv2
        self.graphics_scene = graphics_scene
        self.plot_callback = plot_callback

        self.color = [(255, 165, 0), (51, 255, 51), (51, 153, 255), (255, 51, 51), (255, 255, 102)] # RGB

        if plot_config is None: # Defualt plot config
            self.plot_config = Plot_Config(
                plot_opacity=1.0, point_size = 5.0, confidence_cutoff = 0.0, hide_text_labels = False, edit_mode = False)
        else:
            self.plot_config = plot_config

        if frame_cv2 is None and graphics_scene is not None:
            self.mode = "GS"
        elif graphics_scene is None and frame_cv2 is not None:
            self.mode = "CV"
            self.color = [color[::-1] for color in self.color] # RGB to BGR
        else:
            raise ValueError("Either 'frame_cv2' or 'graphics_scene' must be provided, but not both.")

        self.keypoint_coords = {}

    def plot_predictions(self) -> Optional[np.ndarray]:
        """
        Renders keypoints, labels, bounding boxes, and skeleton connections for all instances 
        in the current frame based on the selected plotting mode.

        Drawing modes:
            - "GS" (QGraphicsScene): Draws interactive, draggable keypoints with Qt items.
              Connects drag events to the provided callback for refinement workflows.
            - "CV" (OpenCV): Draws static circles on the input image using cv2.

        Additional elements:
            - Keypoint text labels (if not hidden).
            - Bounding box around each instance (if individuals are defined and >=2 keypoints visible).
            - Skeleton lines connecting keypoints according to the skeleton definition.

        Returns:
            Optional[np.ndarray]: Modified frame_cv2 with overlays if in "CV" mode; 
                                 otherwise None (drawing is done directly on the scene in "GS" mode).
        """
        for inst in range(self.dlc_data.instance_count):
            self.keypoint_coords = {} # Cleanup the keypoint coords of other insts
            color = self.color[inst % len(self.color)]
            
            for kp_idx in range(self.dlc_data.num_keypoint):
                kp = self.current_frame_data[inst,kp_idx*3:kp_idx*3+3]
                x, y, conf = kp[0], kp[1], kp[2]

                if np.isnan(x) or np.isnan(y) or conf <= self.plot_config.confidence_cutoff:
                    continue

                if self.mode == "GS": # QGraphicsEllipseItem dot representing the keypoints 
                    self.keypoint_coords[kp_idx] = (float(x),float(y),float(conf))
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

                elif self.mode == "CV": # Draw the dot representing the keypoints
                    self.keypoint_coords[kp_idx] = (int(x),int(y),float(conf))
                    cv2.circle(self.frame_cv2, (int(x), int(y)), int(self.plot_config.point_size//2), color, -1) 

            if not self.plot_config.hide_text_labels:
                self._plot_keypoint_label(color)

            if self.dlc_data.individuals is not None and len(self.keypoint_coords) >= 2:
                self._plot_bounding_box(color, inst)

            if self.dlc_data.skeleton:
                self._plot_skeleton(color)

        if self.mode == "CV":
            return self.frame_cv2

    def _plot_bounding_box(self, color:Tuple[int, int, int], inst_idx:int, padding:int=10):
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

        if self.mode == "GS":
            view_width = self.graphics_scene.sceneRect().width()
            view_height = self.graphics_scene.sceneRect().height()
        elif self.mode == "CV":
            view_width = self.frame_cv2.shape[1]
            view_height = self.frame_cv2.shape[0]

        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(view_width - 1, max_x + padding)
        max_y = min(view_height - 1, max_y + padding)

        bounding_box_label = f"Inst: {self.dlc_data.individuals[inst_idx]} | Conf:{kp_inst_mean:.4f}"

        if self.mode == "GS": # Draw bounding box using QGraphicsRectItem
            rect_item = Selectable_Instance(min_x, min_y, max_x - min_x, max_y - min_y, inst_idx, default_color_rgb=color)
            rect_item.setOpacity(self.plot_config.plot_opacity)

            if isinstance(rect_item, Selectable_Instance):
                rect_item.setFlag(QGraphicsRectItem.ItemIsMovable, self.plot_config.edit_mode)

            self.graphics_scene.addItem(rect_item)
            rect_item.bounding_box_moved.connect(self.plot_callback.box_coords_callback)
            rect_item.bounding_box_clicked.connect(self.plot_callback.box_object_callback)

            if not self.plot_config.hide_text_labels:
                text_item_inst = QGraphicsTextItem(f"{bounding_box_label}")
                text_item_inst.setPos(min_x, min_y - 20) # Adjust position to be above the bounding box
                text_item_inst.setDefaultTextColor(QtGui.QColor(*color))
                text_item_inst.setOpacity(self.plot_config.plot_opacity)
                text_item_inst.setFlag(QGraphicsTextItem.ItemIgnoresTransformations) # Keep text size constant
                self.graphics_scene.addItem(text_item_inst)
        
        elif self.mode == "CV":
            cv2.rectangle(self.frame_cv2, (min_x, min_y), (max_x, max_y), color, 1) # Draw the bounding box
            
            if not self.plot_config.hide_text_labels:
                cv2.putText(self.frame_cv2, f"{bounding_box_label}", (min_x, min_y),
                    cv2.FONT_HERSHEY_SIMPLEX, self.plot_config.point_size/20, color, 1, cv2.LINE_AA)

    def _plot_keypoint_label(self, color:Tuple[int, int, int]):
        for kp_idx, (x, y, _) in self.keypoint_coords.items():
            keypoint_label = self.dlc_data.keypoints[kp_idx]

            if self.mode == "GS":
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
            
            elif self.mode == "CV":
                cv2.putText(self.frame_cv2, str(keypoint_label), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, self.plot_config.point_size/20, color, 1, cv2.LINE_AA)

    def _plot_skeleton(self, color:Tuple[int, int, int]):
        for start_kp, end_kp in self.dlc_data.skeleton:
            start_kp_idx = self.dlc_data.keypoint_to_idx[start_kp]
            end_kp_idx = self.dlc_data.keypoint_to_idx[end_kp]
            start_coord = self.keypoint_coords.get(start_kp_idx)
            end_coord = self.keypoint_coords.get(end_kp_idx)

            if not start_coord or not end_coord:
                continue
            
            if self.mode == "GS":
                line = QGraphicsLineItem(start_coord[0], start_coord[1], end_coord[0], end_coord[1])
                line.setPen(QtGui.QPen(QtGui.QColor(*color), self.plot_config.point_size/3))
                line.setOpacity(self.plot_config.plot_opacity)
                self.graphics_scene.addItem(line)

            elif self.mode == "CV":
                start_coord = start_coord[:2]
                end_coord = end_coord[:2]
                line_thick = 1 if self.plot_config.point_size < 5 else 2
                cv2.line(self.frame_cv2, start_coord, end_coord, color, line_thick)