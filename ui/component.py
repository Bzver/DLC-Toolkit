from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsRectItem


class Draggable_Keypoint(QtCore.QObject, QGraphicsEllipseItem):
    keypoint_moved = Signal(int, int, float, float) # instance_id, keypoint_id, new_x, new_y, emit when the keypoint is moved
    keypoint_drag_started = Signal(object) # Emits the Draggable_Keypoint object itself

    def __init__(self, x, y, width, height, instance_id, keypoint_id, default_color_rgb, parent=None):
        QtCore.QObject.__init__(self, None)
        QGraphicsEllipseItem.__init__(self, x, y, width, height, parent)
        self.instance_id = instance_id
        self.keypoint_id = keypoint_id
        self.default_color_rgb = default_color_rgb
        self.setBrush(QtGui.QBrush(QtGui.QColor(*default_color_rgb)))
        self.setPen(QtGui.QPen(QtGui.QColor(*default_color_rgb), 1))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, False) # Initially not movable, enabled by direct_keypoint_edit
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.original_pos = self.pos() # Store initial position on press

    def hoverEnterEvent(self, event):
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 0))) # Yellow on hover
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QtGui.QBrush(QtGui.QColor(*self.default_color_rgb))) # Revert to default
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.flags() & QGraphicsEllipseItem.ItemIsMovable:
            self.keypoint_drag_started.emit(self)
            self.original_pos = self.pos() # Store position at the start of the drag
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.flags() & QGraphicsEllipseItem.ItemIsMovable:
            new_pos = self.pos()
            if new_pos != self.original_pos: # Calculate the delta from the original position
                delta_x = new_pos.x() - self.original_pos.x()
                delta_y = new_pos.y() - self.original_pos.y()
                self.keypoint_moved.emit(self.instance_id, self.keypoint_id, delta_x, delta_y)
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange and self.scene():
            # The actual data array update will happen on mouse release
            return value
        return super().itemChange(change, value)


class Selectable_Instance(QtCore.QObject, QGraphicsRectItem):
    bounding_box_clicked = Signal(object)                        # Signal to emit when this box is clicked
    bounding_box_moved = Signal(int, float, float)  # Signal to emit when the bounding box is moved, instance_id, dx, dy

    def __init__(self, x, y, width, height, instance_id, default_color_rgb, parent=None):
        QtCore.QObject.__init__(self, parent)
        QGraphicsRectItem.__init__(self, x, y, width, height, parent)
        self.instance_id = instance_id
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, False) # Initially not movable, enabled by direct_keypoint_edit
        self.setAcceptHoverEvents(True)

        self.default_pen = QPen(QColor(*default_color_rgb), 1) # Use passed color
        self.selected_pen = QPen(QColor(255, 0, 0), 2) # Red, 2px
        self.hover_pen = QPen(QColor(255, 255, 0), 1) # Yellow, 1px

        self.setPen(self.default_pen)
        self.is_selected = False
        self.last_mouse_pos = None # To track mouse movement for dragging

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.bounding_box_clicked.emit(self) # Emit the signal for selection
            if self.flags() & QGraphicsRectItem.ItemIsMovable:
                self.last_mouse_pos = event.scenePos() # Store the initial mouse position for dragging
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.flags() & QGraphicsRectItem.ItemIsMovable and self.last_mouse_pos is not None:
            current_pos = event.scenePos()
            dx = current_pos.x() - self.last_mouse_pos.x()
            dy = current_pos.y() - self.last_mouse_pos.y()
            
            self.setPos(self.pos().x() + dx, self.pos().y() + dy)
            self.last_mouse_pos = current_pos # Update last position for next move event

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.flags() & QGraphicsRectItem.ItemIsMovable and self.last_mouse_pos is not None:

            if hasattr(self, 'initial_pos_on_press'):
                dx = self.pos().x() - self.initial_pos_on_press.x()
                dy = self.pos().y() - self.initial_pos_on_press.y()
                if dx != 0 or dy != 0:
                    self.bounding_box_moved.emit(self.instance_id, dx, dy)
                del self.initial_pos_on_press # Clean up
            
            self.last_mouse_pos = None # Reset
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.ItemPositionChange and self.flags() & QGraphicsRectItem.ItemIsMovable:
            if not hasattr(self, 'initial_pos_on_press'):
                self.initial_pos_on_press = self.pos()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        if not self.is_selected:
            self.setPen(self.hover_pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if not self.is_selected:
            self.setPen(self.default_pen)
        super().hoverLeaveEvent(event)

    def toggle_selection(self):
        self.is_selected = not self.is_selected
        self.update_visual()

    def update_visual(self):
        if self.is_selected:
            self.setPen(self.selected_pen)
        else:
            self.setPen(self.default_pen)


class Clickable_Video_Label(QtWidgets.QLabel):
    clicked = Signal(int) # Signal to emit cam_idx when clicked

    def __init__(self, cam_idx, parent=None):
        super().__init__(parent)
        self.cam_idx = cam_idx
        self.setMouseTracking(True) # Enable mouse tracking for hover effects

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.cam_idx)
        super().mousePressEvent(event)


class Spinbox_With_Label(QtWidgets.QWidget):
    value_changed = Signal(int)

    def __init__(self, label_text: str, spinbox_range: tuple, initial_val: int, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        label = QtWidgets.QLabel(label_text)
        self.spinbox = QtWidgets.QSpinBox()
        self.spinbox.setRange(*spinbox_range)
        self.spinbox.setValue(initial_val)
        self.spinbox.valueChanged.connect(lambda v: self.value_changed.emit(v))

        layout.addWidget(label)
        layout.addWidget(self.spinbox)
        layout.setContentsMargins(0, 0, 0, 0)
