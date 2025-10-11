from PySide6.QtWidgets import QMenuBar, QWidget
from PySide6.QtGui import QShortcut, QKeySequence

class Menu_Widget(QMenuBar):
    def __init__(self, parent:QWidget=None):
        super().__init__(parent)

    def add_menu_from_config(self, menu_config, clear_menu:bool=True):
        """
        Adds menus and their actions based on a configuration dictionary.
        Args:
            menu_config (dict): A dictionary defining the menu structure.
            Example:
                "File": {
                    "display_name": "File",
                    "buttons": [
                        ("Load Video", load_video_function),
                        {
                            "submenu": "Import",
                            "display_name": "Import Data",
                            "items": [
                                ("From File", import_from_file_function),
                                ("From URL", import_from_url_function),
                                {
                                    "submenu": "Advanced",
                                    "items": [
                                        ("From Database", lambda: print("DB import")),
                                        ("From API", lambda: print("API import"))
                                    ]
                                }
                            ]
                        },
                        ("Exit", exit_function, {"checkable": False})
                    ]
                },
                "View": {
                    "display_name": "View",
                    "buttons": [
                        ("Show Axis", lambda: print("Toggle axis"), {"checkable": True, "checked": True}),
                        ("Fullscreen", lambda: print("Fullscreen"), {})
                    ]
                }
        """
        if clear_menu:
            self.clear()
        for menu_name, config in menu_config.items():
            display_name = config.get("display_name", menu_name)
            menu = self.addMenu(display_name)

            buttons = config.get("buttons", [])
            for item in buttons:
                self._add_menu_item(menu, item)

    def append_to_menu(self, menu_title: str, items: list):
        """
        Append new items to an existing top-level menu.
        
        Args:
            menu_title (str): Display name of the existing menu (e.g., "View").
            items (list): List of button definitions (same format as in menu_config).
                        Each item can be:
                            - ("Label", callback)
                            - ("Label", callback, {options})
                            - {submenu dict...}
        """
        menu = self.find_menu_by_title(menu_title)
        if menu is None:
            raise ValueError(f"No top-level menu found with title '{menu_title}'")
        
        for item in items:
            self._add_menu_item(menu, item)

    def _add_menu_item(self, parent_menu, item):
        """Recursively adds an action or submenu to the given parent menu."""
        if isinstance(item, dict):
            # It's a submenu
            submenu_key = item.get("submenu")
            if not submenu_key:
                raise ValueError("Submenu dictionary must have 'submenu' key")
            submenu_display = item.get("display_name", submenu_key)
            submenu = parent_menu.addMenu(submenu_display)

            subitems = item.get("items", [])
            for subitem in subitems:
                self._add_menu_item(submenu, subitem)

        elif isinstance(item, (list, tuple)):
            if len(item) == 2:
                action_text, action_func = item
                options = {}
            elif len(item) == 3:
                action_text, action_func, options = item
            else:
                raise ValueError("Menu item must be tuple of length 2 (text, func) or 3 (text, func, options)")

            action = parent_menu.addAction(action_text)
            action.triggered.connect(action_func)

            if isinstance(options, dict):
                if options.get("checkable"):
                    action.setCheckable(True)
                    action.setChecked(options.get("checked", False))
        else:
            raise ValueError("Menu item must be a tuple or dict (submenu)")

class Shortcut_Manager:
    def __init__(self, parent: QWidget):
        """
        Manages keyboard shortcuts for a widget.
        
        Args:
            parent (QWidget): The widget that will be the parent/context for shortcuts.
        """
        self.parent = parent
        self._shortcuts = {}

    def add_shortcuts_from_config(self, shortcut_config, clear_first: bool = True):
        """
        Adds shortcuts from a configuration dict.
        
        Args:
            shortcut_config (dict): Mapping of shortcut names to config.
                Format:
                {
                    "prev_frame_fast": {
                        "key": "Shift+Left",
                        "callback": lambda: self._change_frame(-10)
                    },
                    "toggle_play": {
                        "key": "Space",
                        "callback": self.vid_play.sld.toggle_playback
                    },
                    "save": {
                        "key": "Ctrl+S",
                        "callback": self.save_file
                    }
                }
            clear_first (bool): If True, removes all existing managed shortcuts.
        """
        if clear_first:
            self.clear()

        for name, config in shortcut_config.items():
            key = config["key"]
            callback = config["callback"]

            shortcut = QShortcut(self._parse_shortcut(key), self.parent)
            shortcut.activated.connect(callback)
            self._shortcuts[name] = shortcut

    def clear(self):
        """Remove and delete all managed shortcuts."""
        for shortcut in self._shortcuts.values():
            shortcut.deleteLater()
        self._shortcuts.clear()

    def set_enabled(self, enabled: bool):
        for sc in self._shortcuts.values():
            sc.setEnabled(enabled)

    def remove_shortcut(self, name:str):
        """Remove a specific shortcut by name."""
        if name in self._shortcuts:
            self._shortcuts[name].deleteLater()
            del self._shortcuts[name]

    def add_shortcut(self, name: str, key, callback):
        """Add a single shortcut by name."""
        shortcut = QShortcut(QKeySequence(key), self.parent)
        shortcut.activated.connect(callback)
        self._shortcuts[name] = shortcut

    def _parse_shortcut(self, key) -> QKeySequence:
        seq = QKeySequence(key)
        if seq.isEmpty():
            raise ValueError(f"Invalid key sequence: {key!r}")
        return seq