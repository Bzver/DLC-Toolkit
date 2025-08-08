from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsRectItem

class Slider_With_Marks(QtWidgets.QSlider):
    def __init__(self, orientation):
        super().__init__(orientation)
        self.frame_categories = {} # Stores {category_name: set_of_frames}
        self.category_colors = {} # Stores {category_name: color_string}
        self.category_priorities = {} # Stores {category_name: priority_int}
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #B1B1B1, stop:1 #B1B1B1);
                margin: 2px 0;
            }
            
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 10px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)

    def set_frame_category(self, category_name, frames, color, priority):
        self.frame_categories[category_name] = set(frames)
        if color:
            self.category_colors[category_name] = color
        elif category_name not in self.category_colors:
            self.category_colors[category_name] = "#183539"  # default color if None
        self.category_priorities[category_name] = priority # Store the priority
        self.update() # Request a repaint

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.frame_categories:
            return
        
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        # Get slider geometry
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        groove_rect = self.style().subControlRect(
            QtWidgets.QStyle.CC_Slider, 
            opt, 
            QtWidgets.QStyle.SC_SliderGroove, 
            self
        )
        # Calculate available width and range
        min_val = self.minimum()
        max_val = self.maximum()
        available_width = groove_rect.width()

        frame_colors_to_plot = {} # {frame: color} Decide which color to use for a frame in multiple category

        sorted_categories = sorted(
            self.frame_categories.keys(),
            key=lambda cat_name: self.category_priorities.get(cat_name, float('inf')) # Default high priority for safety
        )
        
        for category_name in sorted_categories:
            frames = self.frame_categories.get(category_name, set())
            color = self.category_colors.get(category_name)

            if frames and color:
                for frame in frames:
                    if min_val <= frame <= max_val:
                        frame_colors_to_plot[frame] = color

        painter.setPen(QtCore.Qt.NoPen)

        # Draw each frame on slider
        for frame, color in frame_colors_to_plot.items():
            pos = QtWidgets.QStyle.sliderPositionFromValue(
                min_val, 
                max_val, 
                frame, 
                available_width,
                opt.upsideDown
            ) + groove_rect.left()
            
            painter.setBrush(QtGui.QColor(color))
            painter.drawRect(
                int(pos) - 1,  # Center the mark
                groove_rect.top(),
                1,             # Width
                groove_rect.height()
            )
        
        painter.end()

#######################################################################################################################################################

class Draggable_Keypoint(QtCore.QObject, QGraphicsEllipseItem):
    # Signal to emit when the keypoint is moved
    keypoint_moved = Signal(int, int, float, float) # instance_id, keypoint_id, new_x, new_y
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

#######################################################################################################################################################

class Selectable_Instance(QtCore.QObject, QGraphicsRectItem):
    clicked = Signal(object) # Signal to emit when this box is clicked
    # Signal to emit when the bounding box is moved
    bounding_box_moved = Signal(int, float, float) # instance_id, dx, dy

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
            self.clicked.emit(self) # Emit the signal for selection
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

###################################################################################################################################################

class Clickable_Video_Label(QtWidgets.QLabel):
    clicked = Signal(int) # Signal to emit cam_idx when clicked

    def __init__(self, cam_idx, parent=None):
        super().__init__(parent)
        self.cam_idx = cam_idx
        self.setMouseTracking(True) # Enable mouse tracking for hover effects if needed

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.cam_idx)
        super().mousePressEvent(event)

###################################################################################################################################################

class Adjust_Property_Dialog(QtWidgets.QDialog):
    property_changed = QtCore.Signal(float)

    def __init__(self, property_name, property_val, range_mult, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Adjust {property_name}")
        self.property_name = property_name
        self.property_val = property_val
        self.range_mult = range_mult

        layout = QtWidgets.QVBoxLayout(self)

        self.property_label = QtWidgets.QLabel(f"{self.property_name}: {self.property_val:.2f}")
        self.property_input = QtWidgets.QDoubleSpinBox()

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(self.property_val * self.range_mult))
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self._emit_adjusted_signal)

    def _emit_adjusted_signal(self, value):
        new_property_val =  value / self.range_mult
        self.property_label.setText(f"{self.property_name} {new_property_val:.2f}")
        self.property_changed.emit(new_property_val)