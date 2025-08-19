from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog
from typing import Optional, Callable, List

from .dtu_io import DLC_Loader, DLC_Exporter
from .dtu_dataclass import Loaded_DLC_Data
from . import dtu_helper as duh

def navigate_to_marked_frame(parent:object, frame_list:List[int], current_frame_idx:int,
        change_frame_callback:Callable[[int], None], mode:str):
    if not frame_list:
        QMessageBox.warning(parent, "No Marked Frames", "No marked frames to navigate.")
        return

    if mode == "prev":
        dest_frame_idx = duh.get_prev_frame_in_list(frame_list, current_frame_idx)
        no_frame_message = "No previous marked frame found."
    elif mode == "next":
        dest_frame_idx = duh.get_next_frame_in_list(frame_list, current_frame_idx)
        no_frame_message = "No next marked frame found."
    else:
        QMessageBox.warning(parent, "Invalid Mode", "Expected mode: 'prev' or 'next'.")
        return
    
    if not dest_frame_idx:
        QMessageBox.warning(parent, "Navigation", no_frame_message)
        return

    try:
        change_frame_callback(dest_frame_idx)
    except Exception as e:
        QMessageBox.critical(parent, "Exception", f"Enountering exception: {e}.")

def load_and_show_message(parent, data_loader: DLC_Loader, metadata_only=False, mute=False) -> Optional[Loaded_DLC_Data]:
    loaded_data, msg = data_loader.load_data(metadata_only)

    if loaded_data is None:
        QMessageBox.critical(parent, "Error", str(msg))
    elif not mute:
        QMessageBox.information(parent, "Success", str(msg))
    
    return loaded_data

def export_and_show_message(parent, exporter: DLC_Exporter, frame_only=False, mute=False) -> bool:
    status, msg = exporter.export_data_to_DLC(frame_only)

    if not status:
        QMessageBox.critical(parent, "Error", str(msg))
    elif not mute:
        QMessageBox.information(parent, "Success", str(msg))

    return status

def handle_unsaved_changes_on_close(parent, event, is_saved: bool, save_callback: Callable[[], bool]):
    if is_saved:
        event.accept()
        return
    
    close_call = QMessageBox(parent)
    close_call.setWindowTitle("Changes Unsaved")
    close_call.setText("Do you want to save your changes before closing?")
    close_call.setIcon(QMessageBox.Icon.Question)

    save_btn = close_call.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
    discard_btn = close_call.addButton("Don't Save", QMessageBox.ButtonRole.DestructiveRole)
    close_btn = close_call.addButton("Close", QMessageBox.RejectRole)
    
    close_call.setDefaultButton(close_btn)

    close_call.exec()
    clicked_button = close_call.clickedButton()
    
    if clicked_button == save_btn:
        success = save_callback()
        if success:
            event.accept()
        else:
            event.ignore()
    elif clicked_button == discard_btn:
        event.accept()  # Close without saving
    else:
        event.ignore()  # Cancel the close action

def get_progress_dialog(parent_gui, start_frame:int, end_frame:int, title:str, dialog:str,
        parent_progress:QProgressDialog=None) -> QProgressDialog:
    
    progress = QProgressDialog(dialog, "Cancel",  start_frame, end_frame, parent_gui)
    progress.setWindowTitle(title)
    progress.setWindowModality(Qt.WindowModal)
    progress.setValue(0)

    if parent_progress:
        # Position it below and slightly to the side of the parent dialog
        x = parent_progress.x()
        y = parent_progress.y() + parent_progress.height() + 30
        progress.move(x, y)

    return progress