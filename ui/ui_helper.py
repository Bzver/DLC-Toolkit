import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog
from typing import Optional, Callable, Tuple

from utils.io import DLC_Loader, Exporter
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
        exporter:Exporter,
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

###########################################################################################

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

def calculate_snapping_zoom_level(
        current_frame_data:np.ndarray,
        view_width:float,
        view_height:float
        )->Tuple[float,float,float]:
    """
    Calculates an optimal zoom level and center position to fit all visible keypoints 
    in the current frame within the view, with padding.

    The function computes the bounding box of all non-NaN 2D keypoint coordinates, 
    applies uniform padding, and determines the maximum zoom level that fits the padded 
    box within the given view dimensions. The result centers the keypoints in the view.

    Args:
        current_frame_data (np.ndarray): Array of shape (num_instances * num_keypoints, 3) 
            containing flattened x, y, confidence values for all keypoints in the frame.
        view_width (float): Width of the target view (e.g., graphics scene or display window).
        view_height (float): Height of the target view.

    Returns:
        Tuple[float, float, float]:
            - new_zoom_level (float): Scaling factor to apply (clamped between 0.1 and 10.0).
            - center_x (float): X-coordinate of the center of the bounding box.
            - center_y (float): Y-coordinate of the center of the bounding box.
    """
    x_vals_current_frame = current_frame_data[:, 0::3]
    y_vals_current_frame = current_frame_data[:, 1::3]

    if np.all(np.isnan(x_vals_current_frame)):
        return
    
    min_x = np.nanmin(x_vals_current_frame)
    max_x = np.nanmax(x_vals_current_frame)
    min_y = np.nanmin(y_vals_current_frame)
    max_y = np.nanmax(y_vals_current_frame)

    padding_factor = 1.25 # 25% padding
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    padded_width = width * padding_factor
    padded_height = height * padding_factor
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Calculate new zoom level
    if padded_width > 0 and padded_height > 0:
        zoom_x = view_width / padded_width
        zoom_y = view_height / padded_height
        new_zoom_level = min(zoom_x, zoom_y)
    else:
        new_zoom_level = 1.0

    # Apply zoom limits
    new_zoom_level = max(0.1, min(new_zoom_level, 10.0))

    return new_zoom_level, center_x, center_y