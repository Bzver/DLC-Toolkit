import numpy as np
import cv2

from PySide6 import QtGui
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsLineItem, QGraphicsScene

from typing import Optional, Tuple

from core.dataclass import Loaded_DLC_Data, Plot_Config, Plotter_Callbacks
from ui import Selectable_Instance, Draggable_Keypoint

class Prediction_Plotter:
    Frame_CV2 = np.ndarray

    def __init__(
            self, dlc_data:Loaded_DLC_Data,
            plot_config:Optional[Plot_Config] = None,
            plot_callback:Optional[Plotter_Callbacks] = None,
            fast_mode:bool = True,
            ):
        self.dlc_data = dlc_data
        self.plot_callback = plot_callback
        self.cv_mode = fast_mode

        self.current_frame_data = None
        self.frame_cv2 = None
        self.graphics_scene = None

        self.color = [(255, 165, 0), (51, 255, 51), (51, 153, 255), (255, 51, 51), (255, 255, 102)] # RGB

        if plot_config is None: # Defualt plot config
            self.plot_config = Plot_Config(
                plot_opacity =1.0, point_size = 6.0, confidence_cutoff = 0.0, hide_text_labels = False, edit_mode = False,
                plot_labeled = True, plot_pred = True, navigate_labeled = False, auto_snapping = False, navigate_roi = False)
        else:
            self.plot_config = plot_config

        if fast_mode:
            self.color = [color[::-1] for color in self.color] # RGB to BGR

        self.keypoint_coords = {}

    def plot_predictions(self, frame:Frame_CV2|QGraphicsScene, current_frame_data:np.ndarray) -> Frame_CV2:
        if frame is None:
            raise ValueError("Nonetype cannot be used as plotting canvas.")

        if isinstance(frame, np.ndarray):
            frame_type = "Frame_CV2"
        elif isinstance(frame, QGraphicsScene):
            frame_type = "QGraphicsScene"
        else:
            frame_type = "Unknown"

        if self.cv_mode and frame_type == "Frame_CV2":
            self.frame_cv2 = frame
        elif not self.cv_mode and frame_type == "QGraphicsScene":
            self.graphics_scene = frame
        else:
            plotter_mode = "CV2" if self.cv_mode else "GraphicScene"
            raise ValueError(
                f"Frame and plotter mode mismatch! Plotter mode:{plotter_mode} - frame: {frame_type}")
        
        if current_frame_data is None or np.all(np.isnan(current_frame_data)) or self.dlc_data is None:
            return frame
        
        self.current_frame_data = current_frame_data
        return self._plot_worker()

    def _plot_worker(self) -> Optional[np.ndarray]:
        for inst in range(self.dlc_data.instance_count):
            self.keypoint_coords = {} # Cleanup the keypoint coords of other insts
            color = self.color[inst % len(self.color)]
            
            for kp_idx in range(self.dlc_data.num_keypoint):
                kp = self.current_frame_data[inst,kp_idx*3:kp_idx*3+3]
                x, y, conf = kp[0], kp[1], kp[2]

                if np.isnan(x) or np.isnan(y) or conf <= self.plot_config.confidence_cutoff:
                    continue

                if self.cv_mode:
                    self.keypoint_coords[kp_idx] = (int(x),int(y),float(conf))
                    cv2.circle(self.frame_cv2, (int(x), int(y)), int(self.plot_config.point_size//2), color, -1)
                else:
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

            if not self.plot_config.hide_text_labels:
                self._plot_keypoint_label(color)

            if self.dlc_data.individuals is not None and len(self.keypoint_coords) >= 2:
                self._plot_bounding_box(color, inst)

            if self.dlc_data.skeleton:
                self._plot_skeleton(color)

        if self.cv_mode:
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

        if self.cv_mode:
            view_width = self.frame_cv2.shape[1]
            view_height = self.frame_cv2.shape[0]
        else:
            view_width = self.graphics_scene.sceneRect().width()
            view_height = self.graphics_scene.sceneRect().height()

        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(view_width - 1, max_x + padding)
        max_y = min(view_height - 1, max_y + padding)

        bounding_box_label = f"Inst: {self.dlc_data.individuals[inst_idx]} | Conf:{kp_inst_mean:.4f}"
        
        if self.cv_mode:
            cv2.rectangle(self.frame_cv2, (min_x, min_y), (max_x, max_y), color, 1) # Draw the bounding box
            
            if not self.plot_config.hide_text_labels:
                cv2.putText(self.frame_cv2, f"{bounding_box_label}", (min_x, min_y),
                    cv2.FONT_HERSHEY_SIMPLEX, self.plot_config.point_size/20, color, 1, cv2.LINE_AA)
                
        else: # Draw bounding box using QGraphicsRectItem
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


    def _plot_keypoint_label(self, color:Tuple[int, int, int]):
        for kp_idx, (x, y, _) in self.keypoint_coords.items():
            keypoint_label = self.dlc_data.keypoints[kp_idx]

            if self.cv_mode:
                cv2.putText(self.frame_cv2, str(keypoint_label), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, self.plot_config.point_size/20, color, 1, cv2.LINE_AA)
            else:
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
            
    def _plot_skeleton(self, color:Tuple[int, int, int]):
        for start_kp, end_kp in self.dlc_data.skeleton:
            start_kp_idx = self.dlc_data.keypoint_to_idx[start_kp]
            end_kp_idx = self.dlc_data.keypoint_to_idx[end_kp]
            start_coord = self.keypoint_coords.get(start_kp_idx)
            end_coord = self.keypoint_coords.get(end_kp_idx)

            if not start_coord or not end_coord:
                continue
            
            if self.cv_mode:
                start_coord = start_coord[:2]
                end_coord = end_coord[:2]
                line_thick = 1 if self.plot_config.point_size < 5 else 2
                cv2.line(self.frame_cv2, start_coord, end_coord, color, line_thick)
            else:
                line = QGraphicsLineItem(start_coord[0], start_coord[1], end_coord[0], end_coord[1])
                line.setPen(QtGui.QPen(QtGui.QColor(*color), self.plot_config.point_size/3))
                line.setOpacity(self.plot_config.plot_opacity)
                self.graphics_scene.addItem(line)