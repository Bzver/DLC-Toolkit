import numpy as np

from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsView, QGraphicsScene, QFrame
from PySide6.QtGui import QPainter, QColor, QPen, QTransform

from typing import Callable

from ui import Draggable_Keypoint, Selectable_Instance
import utils.helper as duh

class Canvas(QGraphicsView):
    instance_selected = Signal(int)

    def __init__(self, track_edit_callback:Callable[[object], None], parent=None):
        self.gscene = QGraphicsScene(parent)
        super().__init__(self.gscene)

        self.track_edit_callback = track_edit_callback

        self.setRenderHint(QPainter.Antialiasing)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background-color: black;")
        self.mousePressEvent = self._mouse_press_event
        self.mouseMoveEvent = self._mouse_move_event
        self.mouseReleaseEvent = self._mouse_release_event

        self.zoom_factor = 1.0

        self.is_zoom_mode = False
        self.is_drawing_zone = False
        self.is_kp_edit = False
        self.sbox, self.drag_kp = None, None

    def set_dragged_keypoint(self, keypoint_item:Draggable_Keypoint):
        self.drag_kp = keypoint_item

    def _handle_box_selection(self, clicked_box:Selectable_Instance):
        if self.sbox and self.sbox != clicked_box and self.sbox.scene() is not None:
            self.sbox.toggle_selection() # Remove old box
        clicked_box.toggle_selection()

        if clicked_box.is_selected:
            self.sbox = clicked_box
            self.instance_selected.emit(self.sbox.instance_id)
        else:
            self.sbox = None

    def reset_zoom(self):
        if self.zoom_factor == 1.0: # Don't do anything when there has not been any zoom in/out yet
            return
        self.zoom_factor = 1.0
        self.fitInView(self.gscene.sceneRect(), Qt.KeepAspectRatio)

    def toggle_zoom_mode(self):
        self.is_zoom_mode = not self.is_zoom_mode
        if self.is_zoom_mode:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.wheelEvent = self._mouse_wheel_event
        else:
            self.setDragMode(QGraphicsView.NoDrag)
            self.wheelEvent = super(QGraphicsView, self).wheelEvent

    def toggle_kp_edit(self):
        self.is_kp_edit = not self.is_kp_edit
        for item in self.gscene.items():
            if isinstance(item, Draggable_Keypoint):
                item.setFlag(QtWidgets.QGraphicsEllipseItem.ItemIsMovable, self.is_kp_edit)
            elif isinstance(item, Selectable_Instance):
                item.setFlag(QGraphicsRectItem.ItemIsMovable, self.is_kp_edit)

    def clear_graphic_scene(self):
        self.gscene.clear()

    def get_graphic_scene_dim(self):
        view_width = self.gscene.sceneRect().width()
        view_height = self.gscene.sceneRect().height()
        return view_width, view_height

    def _mouse_press_event(self, event):
        if self.is_zoom_mode:
            QGraphicsView.mousePressEvent(self, event)
            return
        if self.is_drawing_zone:
            self.start_point = self.mapToScene(event.position().toPoint())
            if self.current_rect_item:
                self.gscene.removeItem(self.current_rect_item)
            self.current_rect_item = QGraphicsRectItem(self.start_point.x(), self.start_point.y(), 0, 0)
            self.current_rect_item.setPen(QPen(QColor(255, 0, 0), 2))
            self.gscene.addItem(self.current_rect_item)
        else:
            item = self.itemAt(event.position().toPoint())
            
            if (isinstance(item, Draggable_Keypoint) or isinstance(item, Selectable_Instance)) and self.is_kp_edit:
                pass 
            elif isinstance(item, Selectable_Instance):
                pass
            else:
                if self.sbox:
                    self.sbox.toggle_selection()
                    self.sbox = None
            QGraphicsView.mousePressEvent(self, event)

    def _mouse_move_event(self, event):
        if self.is_zoom_mode:
            QGraphicsView.mouseMoveEvent(self, event)
            return
        if self.is_drawing_zone and self.start_point:
            current_point = self.mapToScene(event.position().toPoint())
            rect = QtCore.QRectF(self.start_point, current_point).normalized()
            self.current_rect_item.setRect(rect)
        QGraphicsView.mouseMoveEvent(self, event)

    def _mouse_release_event(self, event):
        if self.is_drawing_zone:
            self.is_drawing_zone = False
            if self.start_point and self.current_rect_item:
                self.setCursor(Qt.ArrowCursor)
                
                rect = self.current_rect_item.rect()
                self.gscene.removeItem(self.current_rect_item)
                self.current_rect_item = None
                self.start_point = None

                x1, y1, x2, y2 = int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())

                self.pred_data_array = duh.clean_inconsistent_nans(self.pred_data_array) # Cleanup ghost points (NaN for x,y yet non-nan in confidence)

                all_x_kps = self.pred_data_array[:,:,0::3]
                all_y_kps = self.pred_data_array[:,:,1::3]

                x_in_range = (all_x_kps >= x1) & (all_x_kps <= x2)
                y_in_range = (all_y_kps >= y1) & (all_y_kps <= y2)
                points_in_bbox_mask = x_in_range & y_in_range

                self.pred_data_array[np.repeat(points_in_bbox_mask, 3, axis=-1)] = np.nan

                self.track_edit_callback()
        
        QGraphicsView.mouseReleaseEvent(self, event)

    def _mouse_wheel_event(self, event):
        if self.is_zoom_mode:
            zoom_in_factor = 1.15
            zoom_out_factor = 1 / zoom_in_factor

            mouse_pos_view = event.position()
            mouse_pos_scene = self.mapToScene(QtCore.QPoint(int(mouse_pos_view.x()), int(mouse_pos_view.y())))

            if event.angleDelta().y() > 0: # Zoom in
                self.zoom_factor *= zoom_in_factor
            else: # Zoom out
                self.zoom_factor *= zoom_out_factor
            
            self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0)) # Limit zoom to prevent extreme values

            new_transform = QTransform()
            new_transform.scale(self.zoom_factor, self.zoom_factor)
            self.setTransform(new_transform)

            self.centerOn(mouse_pos_scene)
        else:
            self.wheelEvent(event)