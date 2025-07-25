from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsRectItem, QPushButton

###################################################################################################################################################

class Progress_Bar_Comp:
    def __init__(self, parent):
        self.gui = parent
        self.progress_layout = QtWidgets.QHBoxLayout()
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(40) # Slightly wider button
        self.progress_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 0) # Will be set dynamically
        self.progress_slider.setTracking(True)

        self.progress_layout.addWidget(self.play_button)
        self.progress_layout.addWidget(self.progress_slider)
        self.playback_timer = QTimer(self.gui) # Pass GUI to QTimer
        self.playback_timer.timeout.connect(self.autoplay_video)
        self.is_playing = False
        self.gui.layout.addLayout(self.progress_layout)
        
        self.progress_slider.sliderMoved.connect(self.set_frame_from_slider)
        self.play_button.clicked.connect(self.toggle_playback)

    def set_slider_range(self, total_frames):
        self.progress_slider.setRange(0, total_frames - 1)
        self.progress_slider.setValue(0)

    def set_slider_value(self, value):
        self.progress_slider.setValue(value)

    def set_frame_from_slider(self, value):
        if hasattr(self.gui, "selected_cam_idx"):
            self.gui.current_frame_idx = None
        self.gui.current_frame_idx = value
        self.gui.display_current_frame()
        self.gui.navigation_box_title_controller()

    def autoplay_video(self):
        if not hasattr(self.gui, 'total_frames') or self.gui.total_frames <= 0:
                    self.playback_timer.stop()
                    self.play_button.setText("▶")
                    self.is_playing = False
                    return
        
        if self.gui.current_frame_idx is None:
            self.gui.current_frame_idx = 0

        if self.gui.current_frame_idx < self.gui.total_frames - 1:
            if hasattr(self.gui, "selected_cam_idx"):
                self.gui.selected_cam_idx = None
            self.gui.current_frame_idx += 1
            self.gui.display_current_frame()
            self.gui.navigation_box_title_controller()
            self.set_slider_value(self.gui.current_frame_idx)
        else:
            self.playback_timer.stop()
            self.play_button.setText("▶")
            self.is_playing = False

    def toggle_playback(self):
        if not self.is_playing:
            self.playback_timer.start(1000/50) # 50 fps
            self.play_button.setText("■")
            self.is_playing = True
        else:
            self.playback_timer.stop()
            self.play_button.setText("▶")
            self.is_playing = False

###################################################################################################################################################

class Slider_With_Marks(QtWidgets.QSlider):
    def __init__(self, orientation):
        super().__init__(orientation)
        self.frame_categories = {} # Stores {category_name: set_of_frames}
        self.category_colors = {}
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

    def set_frame_category(self, category_name, frames, color=None):
        self.frame_categories[category_name] = set(frames)
        if color:
            self.category_colors[category_name] = color
        elif category_name not in self.category_colors:
            self.category_colors[category_name] = "#183539"  # default color if not specified
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.frame_categories:
            return
        for category_name, frames in self.frame_categories.items():
            color = self.category_colors.get(category_name)
            if frames and color:
                self.paintEvent_painter(frames, color)
        
    def paintEvent_painter(self, frames, color):
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
        # Draw each frame on slider
        for frame in frames:
            if frame < min_val or frame > max_val:
                continue  
            pos = QtWidgets.QStyle.sliderPositionFromValue(
                min_val, 
                max_val, 
                frame, 
                available_width,
                opt.upsideDown
            ) + groove_rect.left()
            # Draw marker
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(color))
            painter.drawRect(
                int(pos) - 1,  # Center the mark
                groove_rect.top(),
                3,  # Width
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
            if new_pos != self.original_pos:
                center_x = new_pos.x() + self.rect().width() / 2
                center_y = new_pos.y() + self.rect().height() / 2
                self.keypoint_moved.emit(self.instance_id, self.keypoint_id, center_x, center_y)
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