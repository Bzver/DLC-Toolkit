import os
import sys
import logging

from PySide6.QtWidgets import QMessageBox

def setup_logging(name='BVT', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'bvt.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_logging()

_HEADLESS_MODE = False
_HEADLESS_RESPONSE = QMessageBox.No  # Default auto-response for questions

def set_headless_mode(enable: bool = True, auto_response: QMessageBox.StandardButton = QMessageBox.No):
    global _HEADLESS_MODE, _HEADLESS_RESPONSE
    _HEADLESS_MODE = enable
    _HEADLESS_RESPONSE = auto_response
    mode_str = "enabled" if enable else "disabled"
    logger.info(f"[LOGGER] Headless mode {mode_str}. Auto-response for questions: {auto_response}")

def is_headless() -> bool:
    """Check if headless mode is active."""
    return _HEADLESS_MODE

class Loggerbox:
    @staticmethod
    def show(parent=None,
             title:str = "",
             text:str = "",
             icon:QMessageBox.Icon = QMessageBox.Information,
             buttons:QMessageBox.StandardButtons = QMessageBox.Ok,
             default_button:QMessageBox.StandardButton = QMessageBox.Ok,
             detailed_text:str|None = None) -> QMessageBox.StandardButton:
        msg_box = QMessageBox(parent)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setIcon(icon)
        msg_box.setStandardButtons(buttons)
        msg_box.setDefaultButton(default_button)

        if is_headless():
            if icon == QMessageBox.Question:
                logger.info(f"[AUTO-ANSWER] Question: {title}, {text} - No.")
                return _HEADLESS_RESPONSE
            else:
                return default_button

        if detailed_text:
            msg_box.setDetailedText(detailed_text)

        return msg_box.exec()

    @classmethod
    def info(cls, parent=None, title="Info", text=""):
        logger.info(f"{title}, {text}.")
        return cls.show(parent, title, text, QMessageBox.Information)

    @classmethod
    def warning(cls, parent=None, title="Warning", text=""):
        logger.warning(f"{title}, {text}.")
        return cls.show(parent, title, text, QMessageBox.Warning)

    @classmethod
    def error(cls, parent=None, title="Error", text="", exc:BaseException|None =None):
        if exc is None:
            logger.error(f"{title}, {text}.")
        else:
            logger.exception(f"{title}, {text}.")
            raise RuntimeError(f"{title}, {text}.")
            
        return cls.show(parent, title, text, QMessageBox.Critical)

    @classmethod
    def question(cls, parent=None, title="Confirmation", text="",
                 buttons=QMessageBox.Yes | QMessageBox.No,
                 default=QMessageBox.No) -> QMessageBox.StandardButton:
        return cls.show(parent, title, text, QMessageBox.Question, buttons, default)
