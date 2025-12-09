import numpy as np
from typing import Optional

from utils.logger import logger

class Uno_Stack:
    def __init__(self, max_undo_stack_size: int = 100):
        self.undo_stack, self.redo_stack = [], []
        self.max_undo_stack_size = max_undo_stack_size
        logger.info(f"[UNO] Initialized Uno_Stack with max_undo_stack_size={max_undo_stack_size}")

    def save_state_for_undo(self, data_array):
        shape_info = getattr(data_array, 'shape', 'unknown')
        dtype_info = getattr(data_array, 'dtype', 'unknown')
        logger.info(f"[UNO] Saving state for undo: shape={shape_info}, dtype={dtype_info}")

        self.redo_stack = []
        logger.debug("[UNO] Redo stack cleared due to new state save.")
        
        self.undo_stack.append(data_array.copy())
        logger.debug(f"[UNO] State saved to undo stack. Undo stack size: {len(self.undo_stack)}")
        
        if len(self.undo_stack) > self.max_undo_stack_size:
            self.undo_stack.pop(0)
            logger.warning(
                f"[UNO] Undo stack exceeded max size ({self.max_undo_stack_size}); oldest state removed. "
                f"New size: {len(self.undo_stack)}"
            )

    def undo(self, data_array) -> Optional[np.ndarray]:
        if not self.undo_stack:
            logger.warning("[UNO] Undo requested, but undo stack is empty.")
            return None

        self.redo_stack.append(data_array.copy())
        data_array = self.undo_stack.pop()
        logger.info(
            f"[UNO] Undo performed. "
            f"Undo stack size: {len(self.undo_stack)}, Redo stack size: {len(self.redo_stack)}"
        )
        return data_array

    def redo(self, data_array) -> Optional[np.ndarray]:
        if not self.redo_stack:
            logger.warning("[UNO] Redo requested, but redo stack is empty.")
            return None

        self.undo_stack.append(data_array.copy())
        data_array = self.redo_stack.pop()
        logger.info(
            f"[UNO] Redo performed. "
            f"Undo stack size: {len(self.undo_stack)}, Redo stack size: {len(self.redo_stack)}"
        )
        return data_array