import numpy as np
import copy
from typing import Optional, Dict, Any

from utils.logger import logger


class Uno_Stack:
    def __init__(self, max_undo_stack_size: int = 100):
        self.undo_stack, self.redo_stack = [], []
        self.max_undo_stack_size = max_undo_stack_size
        logger.debug(f"[UNO] Initialized Uno_Stack with max_undo_stack_size={max_undo_stack_size}")

    def save_state_for_undo(self, data_array):
        self.redo_stack = []
        logger.debug("[UNO] Redo stack cleared due to new state save.")
        
        self.undo_stack.append(data_array.copy())
        logger.debug(f"[UNO] State saved to undo stack. Undo stack size: {len(self.undo_stack)}")
        
        if len(self.undo_stack) > self.max_undo_stack_size:
            self.undo_stack.pop(0)
            logger.debug(
                f"[UNO] Undo stack exceeded max size ({self.max_undo_stack_size}); oldest state removed. "
                f"New size: {len(self.undo_stack)}"
            )

    def undo(self, data_array) -> Optional[np.ndarray]:
        if not self.undo_stack:
            logger.warning("[UNO] Undo requested, but undo stack is empty.")
            return None

        self.redo_stack.append(data_array.copy())
        data_array = self.undo_stack.pop()
        logger.debug(
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
        logger.debug(
            f"[UNO] Redo performed. "
            f"Undo stack size: {len(self.undo_stack)}, Redo stack size: {len(self.redo_stack)}"
        )
        return data_array
    

class Uno_Stack_Dict:
    def __init__(self, max_undo_stack_size: int = 10):
        self.undo_stack: list[Dict[str, Any]] = []
        self.redo_stack: list[Dict[str, Any]] = []
        self.max_undo_stack_size = max_undo_stack_size
        logger.debug(f"[UNDO] Initialized Uno_Stack_Dict with max_undo_stack_size={max_undo_stack_size}")

    def _deep_copy_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return copy.deepcopy(state)

    def save_state_for_undo(self, state: Dict[str, Any]) -> None:
        self.redo_stack.clear()
        copied_state = self._deep_copy_state(state)
        self.undo_stack.append(copied_state)
        logger.debug(f"[UNDO] State saved. Undo stack size: {len(self.undo_stack)}")
        
        if len(self.undo_stack) > self.max_undo_stack_size:
            self.undo_stack.pop(0)
            logger.debug(
                f"[UNDO] Undo stack trimmed. New size: {len(self.undo_stack)}"
            )

    def undo(self, current_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.undo_stack:
            logger.warning("[UNDO] Undo requested, but undo stack is empty.")
            return None

        self.redo_stack.append(self._deep_copy_state(current_state))
        prev_state = self.undo_stack.pop()
        logger.debug(
            f"[UNDO] Performed. Undo: {len(self.undo_stack)}, Redo: {len(self.redo_stack)}"
        )
        return prev_state

    def redo(self, current_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.redo_stack:
            logger.warning("[UNDO] Redo requested, but redo stack is empty.")
            return None

        self.undo_stack.append(self._deep_copy_state(current_state))
        next_state = self.redo_stack.pop()
        logger.debug(
            f"[UNDO] Redo performed. Undo: {len(self.undo_stack)}, Redo: {len(self.redo_stack)}"
        )
        return next_state