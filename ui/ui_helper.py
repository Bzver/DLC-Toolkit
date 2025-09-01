import bisect

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog
from typing import Optional, Callable, List, Literal

from utils.io import DLC_Loader, DLC_Exporter
from utils.dataclass import Loaded_DLC_Data

def format_title(base_title: str, debug_status: bool) -> str:
    return f"{base_title} --- DEBUG MODE" if debug_status else base_title

def load_and_show_message(
        parent,
        data_loader:DLC_Loader,
        metadata_only:bool=False,
        mute:bool=False
        ) -> Optional[Loaded_DLC_Data]:
    """
    Loads DLC data using the provided loader and displays an appropriate message to the user 
    based on the outcome. Shows error, success, or status messages via QMessageBox or statusBar.

    Args:
        parent: Parent widget (e.g., QMainWindow) used for displaying dialogs or status messages.
        data_loader (DLC_Loader): Loader instance responsible for loading the DLC data.
        metadata_only (bool): If True, only metadata (e.g., config, skeleton) is loaded; 
                              otherwise, full prediction data is loaded.
        mute (bool): If True, suppresses pop-up messages and only updates the status bar.

    Returns:
        Optional[Loaded_DLC_Data]: Loaded data object if successful; None if loading failed.
    """
    loaded_data, msg = data_loader.load_data(metadata_only)

    if loaded_data is None:
        QMessageBox.critical(parent, "Error", str(msg))
    elif not mute:
        QMessageBox.information(parent, "Success", str(msg))
    else:
        parent.statusBar().showMessage(msg)
    
    return loaded_data

def export_and_show_message(
        parent,
        exporter:DLC_Exporter,
        frame_only:bool=False,
        mute:bool=False
        ) -> bool:
    """
    Exports current pose estimation data using the provided exporter and informs the user 
    of the result via message dialogs or status bar.

    Args:
        parent: Parent widget used for displaying messages.
        exporter (DLC_Exporter): Exporter instance handling the write operation.
        frame_only (bool): If True, exports only the current frame; otherwise, exports all data.
        mute (bool): If True, avoids pop-up dialogs and only shows the result in the status bar.

    Returns:
        bool: True if export was successful; False otherwise.
    """
    status, msg = exporter.export_data_to_DLC(frame_only)

    if not status:
        QMessageBox.critical(parent, "Error", str(msg))
    elif not mute:
        QMessageBox.information(parent, "Success", str(msg))
    else:
        parent.statusBar().showMessage(msg)

    return status

def handle_unsaved_changes_on_close(
        parent,
        event,
        is_saved:bool,
        save_callback:Callable[[], bool]
        ):
    """
    Prompts the user when attempting to close a window with unsaved changes, offering 
    options to save, discard, or cancel the close action.

    Args:
        parent: Parent widget (e.g., QMainWindow) used for modal dialog positioning.
        event: Close event object that will be accepted or ignored based on user choice.
        is_saved (bool): Flag indicating whether the current state is already saved. 
                         If True, the window closes immediately without prompting.
        save_callback (Callable[[], bool]): Function to call when the user chooses to save. 
                                            Should return True on successful save, False otherwise.

    Returns:
        None: This function directly controls the event's acceptance or rejection.
              It does not return a value but affects application flow by accepting 
              or ignoring the close event based on user interaction.
    """
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

def get_progress_dialog(
        parent,
        start_frame:int,
        end_frame:int,
        title:str,
        dialog:str,
        parent_progress:QProgressDialog=None
        ) -> QProgressDialog:
    """
    Creates and configures a QProgressDialog for long-running frame-based operations.

    Args:
        parent: Parent widget for modal behavior.
        start_frame (int): Starting value of the progress range.
        end_frame (int): Ending value of the progress range.
        title (str): Window title for the dialog.
        dialog (str): Label text displayed inside the dialog.
        parent_progress (QProgressDialog, optional): Reference to a parent dialog for positioning.

    Returns:
        QProgressDialog: Configured progress dialog with modal behavior and optional positioning 
                         below the parent dialog.
    """
    progress = QProgressDialog(dialog, "Cancel",  start_frame, end_frame, parent)
    progress.setWindowTitle(title)
    progress.setWindowModality(Qt.WindowModal)
    progress.setValue(0)

    if parent_progress:
        # Position it below and slightly to the side of the parent dialog
        x = parent_progress.x()
        y = parent_progress.y() + parent_progress.height() + 30
        progress.move(x, y)

    return progress

###########################################################################################

def navigate_to_marked_frame(
        parent,
        frame_list:List[int],
        current_frame_idx:int,
        change_frame_callback:Callable[[int], None],
        mode:Literal["prev","next"]
        ):
    """
    Navigates to the previous or next frame in a sorted list of frames.

    Args:
        parent: Parent widget for displaying warning or error messages.
        frame_list (List[int]): List of frame indices that are marked.
        current_frame_idx (int): Index of the currently displayed frame.
        change_frame_callback (Callable[[int], None]): Function to call with the destination frame index.
        mode (Literal["prev", "next"]): Direction of navigation — either "prev" or "next".

    Behavior:
        - Sorts the frame list and finds the nearest previous or next frame.
        - If no such frame exists, shows a warning.
        - Otherwise, calls the frame change callback with the target frame.
        - On exception during callback, shows a critical error message.
    """
    if not frame_list:
        QMessageBox.warning(parent, "No Marked Frames", "No marked frames to navigate.")
        return
    
    frame_list.sort()

    if mode == "prev":
        dest_frame_idx = _get_prev_frame_in_list(frame_list, current_frame_idx)
        no_frame_message = "No previous marked frame found."
    elif mode == "next":
        dest_frame_idx = _get_next_frame_in_list(frame_list, current_frame_idx)
        no_frame_message = "No next marked frame found."
    
    if dest_frame_idx is None:
        QMessageBox.warning(parent, "Navigation", no_frame_message)
        return

    try:
        change_frame_callback(dest_frame_idx)
    except Exception as e:
        QMessageBox.critical(parent, "Exception", f"Enountering exception: {e}.")

def _get_prev_frame_in_list(frame_list:List[int], current_frame_idx:int) -> Optional[int]:
    try:
        current_idx_in_list = frame_list.index(current_frame_idx)
        prev_idx = current_idx_in_list - 1
    except ValueError:
        insertion_point = bisect.bisect_left(frame_list, current_frame_idx)
        prev_idx = insertion_point - 1

    if prev_idx >= 0:
        return frame_list[prev_idx]
    
    return None

def _get_next_frame_in_list(frame_list:List[int], current_frame_idx:int) -> Optional[int]:
    try:
        current_idx_in_list = frame_list.index(current_frame_idx)
        next_idx = current_idx_in_list + 1
    except ValueError:
        insertion_point = bisect.bisect_right(frame_list, current_frame_idx)
        next_idx = insertion_point

    if next_idx < len(frame_list):
        return frame_list[next_idx]
    
    return None