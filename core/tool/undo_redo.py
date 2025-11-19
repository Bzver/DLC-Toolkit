import numpy as np
from typing import Optional

class Uno_Stack:
    def __init__(self, max_undo_stack_size:int=100):
        self._data_array = None
        self.undo_stack, self.redo_stack = [], []
        self.max_undo_stack_size = max_undo_stack_size

    def save_state_for_undo(self, data_array):
        self._data_array = data_array.copy()
        self.redo_stack = []
        self.undo_stack.append(self._data_array)
        if len(self.undo_stack) > self.max_undo_stack_size:
            self.undo_stack.pop(0)

    def undo(self) -> Optional[np.ndarray]:
        if not self.undo_stack:
            print("DEBUG: Undo failed — undo stack is empty.")
            return None
        if self._data_array is None:
            print("DEBUG: Undo failed — no current state.")
            return None
        
        self.redo_stack.append(self._data_array.copy())
        self._data_array = self.undo_stack.pop()
        print("Undo performed.")
        return self._data_array

    def redo(self) -> Optional[np.ndarray]:
        if not self.redo_stack:
            print("DEBUG: Redo failed — redo stack is empty.")
            return None
        if self._data_array is None:
            print("DEBUG: Redo failed — no current state.")
            return None
        
        self.undo_stack.append(self._data_array.copy())
        self._data_array = self.redo_stack.pop()
        print("Redo performed.")
        return self._data_array