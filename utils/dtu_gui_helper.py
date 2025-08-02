from PySide6.QtWidgets import QMessageBox
from typing import Optional, Callable

from .dtu_io import DLC_Loader, DLC_Exporter
from .dtu_dataclass import Loaded_DLC_Data

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

def handle_unsaved_changes_on_close(parent, event, has_unsaved_changes: bool, save_callback: Callable[[], bool]):
    if not has_unsaved_changes:
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