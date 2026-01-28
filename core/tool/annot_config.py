import numpy as np
import string
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout, QHeaderView,
    QDialog, QLineEdit, QPushButton, QFormLayout, QLabel, QComboBox, QDialogButtonBox,
    QColorDialog
)

from utils.helper import indices_to_spans
from utils.logger import Loggerbox

TABLE_STYLESHEET = """
        QTableWidget {
            selection-background-color: #1E03B6;
            selection-color: white;
            alternate-background-color: #FAFAFA;
        }
        QTableWidget::item:selected {
            font-weight: bold;
        }
    """

class Annotation_Config(QtWidgets.QWidget):
    category_removed = Signal(str, str)
    map_change = Signal(dict)

    COLOR_HEX_EXPANDED = (
        "#009688", "#FF5722", "#795548", "#2196F3", "#CDDC39", "#FFC107",
        "#8BC34A", "#673AB7", "#03A9F4", "#E91E63", "#00E676", "#BD34A6", "#9C27B0"
    )

    def __init__(self, behaviors_map: dict, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self._behaviors_map = behaviors_map
        self.layout = QVBoxLayout(self)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["Category", "Key", "Color"])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setEditTriggers(QTableWidget.DoubleClicked)
        self.table_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_widget.setSelectionMode(QTableWidget.SingleSelection)
        self.table_widget.itemChanged.connect(self._handle_item_changed)
        self.table_widget.cellClicked.connect(self._handle_cell_clicked)

        self.table_widget.setStyleSheet(TABLE_STYLESHEET)
        self.layout.addWidget(self.table_widget)

        button_layout = QVBoxLayout()
        self.add_button = QPushButton("Add Category")
        self.remove_button = QPushButton("Remove Category")

        self.add_button.clicked.connect(self._add_category)
        self.remove_button.clicked.connect(self._remove_category)

        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        self.layout.addLayout(button_layout)
        self._populate_table()

    def add_category_external(self, new_category: str):
        existing_keys = {v[0] for v in self._behaviors_map.values()}
        pool = list(string.ascii_lowercase)
        available_keys = [k for k in pool if k not in existing_keys]
        if not available_keys:
            raise RuntimeError("No valid keys left, remove some behaviors first!")

        color_pool = self.COLOR_HEX_EXPANDED
        color = color_pool[len(self._behaviors_map) % len(color_pool)]
        self._behaviors_map[new_category] = (available_keys[0], color)
        self.map_change.emit(self._behaviors_map)
        self._populate_table()

    def highlight_current_category(self, category: str):
        for row in range(self.table_widget.rowCount()):
            cat_item = self.table_widget.item(row, 0)
            if cat_item and cat_item.text() == category:
                self.table_widget.clearSelection()
                self.table_widget.selectRow(row)
                self.table_widget.scrollToItem(
                    self.table_widget.item(row, 0),
                    QTableWidget.PositionAtCenter
                )
                return
        self.table_widget.clearSelection()

    def sync_behaviors_map(self, behaviors_map):
        self._behaviors_map = behaviors_map
        self._populate_table()

    def _populate_table(self):
        self.table_widget.setRowCount(len(self._behaviors_map))
        for row, (category, (key, color)) in enumerate(self._behaviors_map.items()):

            category_item = QTableWidgetItem(category)
            category_item.setFlags(category_item.flags() & ~Qt.ItemIsEditable)
            self.table_widget.setItem(row, 0, category_item)

            key_item = QTableWidgetItem(key.upper())
            self.table_widget.setItem(row, 1, key_item)

            color_item = QTableWidgetItem()
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
            self._set_color_item_background(color_item, color)
            self.table_widget.setItem(row, 2, color_item)

    def _set_color_item_background(self, item: QTableWidgetItem, color_hex: str):
        """Set background color of table item based on hex string"""
        try:
            color = QColor(color_hex)
            if not color.isValid():
                color = QColor("#808080")
        except Exception:
            color = QColor("#808080")
        item.setBackground(color)
        item.setText("")

    def _handle_cell_clicked(self, row: int, column: int):
        if column == 2:
            category = self.table_widget.item(row, 0).text()
            current_color = self._behaviors_map[category][1]
            color_dialog = QColorDialog(QColor(current_color), self)
            color_dialog.setWindowTitle(f"Select Color for '{category}'")
            if color_dialog.exec() == QDialog.Accepted:
                new_color = color_dialog.selectedColor()
                if new_color.isValid():
                    new_color_hex = new_color.name(QColor.HexArgb)
                    old_key, _ = self._behaviors_map[category]
                    self._behaviors_map[category] = (old_key, new_color_hex)
                    self.map_change.emit(self._behaviors_map)
                    self._set_color_item_background(self.table_widget.item(row, 2), new_color_hex)

    def _handle_cell_clicked(self, row: int, column: int):
        if not column == 2:
            return
        
        category = self.table_widget.item(row, 0).text()
        qcolor = self._behaviors_map[category][1]

        color_dialog = QColorDialog(qcolor, self)
        color_dialog.setWindowTitle(f"Select Color for '{category}'")
        color_dialog.setOption(QColorDialog.ShowAlphaChannel, False)
        
        if color_dialog.exec() == QDialog.Accepted:
            new_color = color_dialog.selectedColor()
            if new_color.isValid():
                new_color.setAlpha(255)
                new_color_hex = new_color.name(QColor.HexRgb)
                old_key, _ = self._behaviors_map[category]
                self._behaviors_map[category] = (old_key, new_color_hex)
                self.map_change.emit(self._behaviors_map)
                self._set_color_item_background(self.table_widget.item(row, 2), new_color_hex)

    def _handle_item_changed(self, item: QTableWidgetItem):
        if item.column() != 1:
            return

        row = item.row()
        category = self.table_widget.item(row, 0).text()
        new_key = item.text().strip()

        if len(new_key) != 1 or not new_key.isalpha():
            Loggerbox.warning(self, "Invalid Input", "Key must be a single alphabet character.")
            current_key = self._behaviors_map[category][0]
            item.setText(current_key.upper())
            return

        new_key_lower = new_key.lower()

        for cat, (key, _) in self._behaviors_map.items():
            if cat != category and key == new_key_lower:
                Loggerbox.warning(
                    self,
                    "Duplicate Key",
                    f"Key '{new_key}' is already assigned to category '{cat}'.\n"
                    "Each key must be unique."
                )
                current_key = self._behaviors_map[category][0]
                item.setText(current_key.upper())
                return

        _, current_color = self._behaviors_map[category]
        self._behaviors_map[category] = (new_key_lower, current_color)
        self.map_change.emit(self._behaviors_map)
        item.setText(new_key.upper())

    def _add_category(self):
        dialog = Add_Category_Dialog(self)
        if dialog.exec() == QDialog.Accepted:
            category, key, color = dialog.get_inputs()
            if not category:
                Loggerbox.warning(self, "Input Error", "Category name cannot be empty.")
                return
            if len(key) != 1 or not key.isalpha():
                Loggerbox.warning(self, "Input Error", "Key must be a single alphabet character.")
                return

            if category in self._behaviors_map:
                Loggerbox.warning(self, "Duplicate Category", f"Category '{category}' already exists.")
                return

            if key in [v[0].lower() for v in self._behaviors_map.values()]:
                existing = [k for k, v in self._behaviors_map.items() if v[0].lower() == key][0]
                Loggerbox.warning(self, "Duplicate Key", f"Key '{key.upper()}' is already used by '{existing}'.")
                return

            self._behaviors_map[category] = (key.lower(), color)
            self.map_change.emit(self._behaviors_map)
            self._populate_table()

    def _remove_category(self):
        selected = self.table_widget.selectedItems()
        if not selected:
            Loggerbox.info(self, "No Selection", "Please select a category to remove.")
            return

        row = selected[0].row()
        category_to_remove = self.table_widget.item(row, 0).text()

        if len(self._behaviors_map) <= 1:
            Loggerbox.warning(self, "Cannot Remove", "At least one category must remain.")
            return

        other_categories = [cat for cat in self._behaviors_map.keys() if cat != category_to_remove]
        if not other_categories:
            Loggerbox.warning(self, "No Target", "No other categories available for reassignment.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Reassign Annotations")
        layout = QVBoxLayout(dialog)

        label = QLabel(f"Category '{category_to_remove}' will be removed.\n"
                       "Which category should receive its annotated frames?")
        layout.addWidget(label)

        combo = QComboBox()
        combo.addItems(other_categories)
        layout.addWidget(combo)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() != QDialog.Accepted:
            return

        target_category = combo.currentText()

        self.category_removed.emit(target_category, category_to_remove)
        del self._behaviors_map[category_to_remove]
        self.map_change.emit(self._behaviors_map)
        self._populate_table()
        
        Loggerbox.info(self, "Success",
                                f"Category '{category_to_remove}' removed.\n"
                                f"Frames reassigned to '{target_category}'.")


class Add_Category_Dialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Category")
        self.resize(350, 150)

        self.category_edit = QLineEdit()
        self.key_edit = QLineEdit()
        self.key_edit.setMaxLength(1)
        self.key_edit.setPlaceholderText("e.g. H")

        self.color_preview = QLabel()
        self.color_preview.setFixedSize(30, 20)
        self.color_preview.setStyleSheet("border: 1px solid #ccc;")
        self.color_button = QPushButton("Select Color")
        self.color_button.clicked.connect(self._pick_color)

        self._current_color = "#FF0000"
        self._update_color_preview()

        form_layout = QFormLayout()
        form_layout.addRow("Category Name:", self.category_edit)
        form_layout.addRow("Key (single letter):", self.key_edit)
        color_layout = QHBoxLayout()
        color_layout.addWidget(self.color_preview)
        color_layout.addWidget(self.color_button)
        color_layout.addStretch()
        form_layout.addRow("Color:", color_layout)

        button_box = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_box.addWidget(self.ok_button)
        button_box.addWidget(self.cancel_button)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addLayout(button_box)
        self.setLayout(layout)

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def _pick_color(self):
        color_dialog = QColorDialog(QColor(self._current_color), self)
        if color_dialog.exec() == QDialog.Accepted:
            color = color_dialog.selectedColor()
            if color.isValid():
                self._current_color = color.name(QColor.HexArgb)
                self._update_color_preview()

    def _update_color_preview(self):
        self.color_preview.setStyleSheet(
            f"background-color: {self._current_color}; border: 1px solid #999;"
        )

    def get_inputs(self):
        category = self.category_edit.text().strip()
        key = self.key_edit.text().strip().lower()
        color = self._current_color
        return category, key, color


class Annotation_Summary_Table(QtWidgets.QWidget):
    row_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.category_array = None
        self.behaviors_map = {}
        self.idx_to_cat = {}
        self.layout = QVBoxLayout(self)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["Behavior", "Start", "End"])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_widget.setSelectionMode(QTableWidget.SingleSelection)
        self.table_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_widget.cellClicked.connect(self._on_row_clicked)

        self.table_widget.setStyleSheet(TABLE_STYLESHEET)
        self.layout.addWidget(self.table_widget)

    def update_data(self, category_array: np.ndarray, behaviors_map: dict, idx_to_cat: dict):
        self.category_array = category_array
        self.behaviors_map = behaviors_map
        self.idx_to_cat = idx_to_cat
        self._populate_table()

    def highlight_current_frame(self, current_frame: int):
        if not hasattr(self, '_segments') or not self._segments:
            return
        
        matching_row = -1
        for row, (_, start, end) in enumerate(self._segments):
            if start <= current_frame <= end:
                matching_row = row
                break
        
        self.table_widget.clearSelection()
        
        if matching_row >= 0:
            self.table_widget.selectRow(matching_row)
            self.table_widget.scrollToItem(
                self.table_widget.item(matching_row, 0),
                QTableWidget.PositionAtCenter
            )

    def extract_segments(self, include_other: bool = False):
        if self.category_array is None:
            return []
        
        segments = []
        NO_CATEGORY = 255
        
        unique_idxs = np.unique(self.category_array)
    
        if not include_other:
            unique_idxs = unique_idxs[unique_idxs != NO_CATEGORY]
        
        for idx in unique_idxs:
            frame_indices = np.where(self.category_array == idx)[0]
            if frame_indices.size == 0:
                continue
                
            spans = indices_to_spans(frame_indices)
            
            if idx == NO_CATEGORY:
                category = "other"
            else:
                cat_name = self.idx_to_cat.get(int(idx), '?')
                category = next((cat for cat in self.behaviors_map.keys() if cat == cat_name), f'Unknown({idx})')
            
            for start, end in spans:
                segments.append((category, start, end))
        
        return sorted(segments, key=lambda x: x[1])

    def _on_row_clicked(self, row: int, column: int):
        if 0 <= row < len(self._segments):
            _, start, _ = self._segments[row]
            self.row_clicked.emit(start)

    def _populate_table(self):
        if self.category_array is None or len(self.category_array) == 0:
            self.table_widget.setRowCount(0)
            return

        segments = self.extract_segments()
        self.table_widget.setRowCount(len(segments))
        self._segments = segments
        
        for row, (category, start, end) in enumerate(segments):
            self.table_widget.setItem(row, 0, self._centered_item(category))
            self.table_widget.setItem(row, 1, self._centered_item(str(start)))
            self.table_widget.setItem(row, 2, self._centered_item(str(end)))

    def _centered_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        return item