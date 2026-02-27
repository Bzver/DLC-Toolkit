import warnings
from contextlib import contextmanager
from typing import Callable

from utils.logger import Loggerbox, QMessageBox


def handle_unsaved_changes_on_close(
        parent,
        event,
        is_saved:bool,
        save_callback:Callable[[], bool]
        ):
    if is_saved:
        event.accept()
        return
    
    reply = Loggerbox.question(
        parent,
        "Changes Unsaved",
        "Do you want to save your changes before closing?",
        buttons=QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        default=QMessageBox.Cancel
    )

    if reply == QMessageBox.Save:
        if save_callback():
            event.accept()
        else:
            event.ignore()
    elif reply == QMessageBox.Discard:
        event.accept()
    else:
        event.ignore()

@contextmanager
def bye_bye_runtime_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        yield