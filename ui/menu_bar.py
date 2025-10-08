from PySide6.QtWidgets import QMenuBar

class Menu_Widget(QMenuBar):
    def __init__(self, parent=None):
        super().__init__(parent)

    def add_menu_from_config(self, menu_config):
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
        for menu_name, config in menu_config.items():
            display_name = config.get("display_name", menu_name)
            menu = self.addMenu(display_name)

            buttons = config.get("buttons", [])
            for item in buttons:
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