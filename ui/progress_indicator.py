from time import time
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from utils.logger import logger

class Progress_Indicator_Dialog(QtWidgets.QProgressDialog):
    def __init__(self, min_val, max_val, title, text, parent=None):
        super().__init__(parent)
        self.setLabelText(text)
        self.setMinimum(min_val)
        self.setMaximum(max_val)
        self.setCancelButtonText("Cancel")
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setValue(min_val)

        self._start_time = time()
        self._last_update_time = self._start_time
        self._last_value = min_val

        self._last_text_update = self._start_time
        self._last_text_value = min_val
        self._text_update_min_interval = 0.2
        self._text_update_min_delta_ratio = 0.05

        self._ema_alpha = 0.3
        self._ema_rate = None

    def setLabelText(self, text):
        self._base_text = text
        return super().setLabelText(text)

    def setValue(self, value):
        super().setValue(value)

        if value <= self.minimum() or self.maximum() <= self.minimum():
            return

        current_time = time()
        elapsed = current_time - self._start_time
        if elapsed <= 0:
            return

        delta_value = value - self._last_value
        delta_time = current_time - self._last_update_time

        self._last_update_time = current_time
        self._last_value = value

        instant_rate = delta_value / delta_time if delta_time > 0 else 0

        if self._ema_rate is None:
            self._ema_rate = instant_rate
        else:
            self._ema_rate = (
                self._ema_alpha * instant_rate + (1 - self._ema_alpha) * self._ema_rate
            )
        it_per_sec = self._ema_rate

        total_range = self.maximum() - self.minimum()
        progress_delta = abs(value - self._last_text_value)
        progress_delta_ratio = progress_delta / total_range if total_range > 0 else 1.0

        should_update = (
            (current_time - self._last_text_update >= self._text_update_min_interval)
            or (progress_delta_ratio >= self._text_update_min_delta_ratio)
            or (value >= self.maximum())
        )

        if not should_update:
            return

        self._last_text_update = current_time
        self._last_text_value = value

        completed_items = value - self.minimum()
        remaining_items = total_range - completed_items
        if it_per_sec > 0 and remaining_items > 0:
            eta = remaining_items / it_per_sec
        else:
            eta = float('inf')

        elapsed_str = self._format_time(elapsed)
        eta_str = "--:--:--" if eta == float('inf') else self._format_time(eta)

        rate_str = self._format_rate(it_per_sec)

        new_text = (
            f"{self._base_text}\n"
            f"Elapsed: {elapsed_str} | Remaining: {eta_str} | {rate_str}"
        )
        super().setLabelText(new_text)
        self._log_milestone(value)

    def _log_milestone(self, value):
        total = self.maximum() - self.minimum()
        percent = 100 * (value - self.minimum()) / total

        milestone = int(percent // 10) * 10
        if not hasattr(self, '_last_milestone'):
            self._last_milestone = -1

        if milestone != self._last_milestone and milestone % 10 == 0:
            self._last_milestone = milestone
            elapsed = time() - self._start_time
            msg = (
                f"[PROGRESS] {self._base_text} — "
                f"{milestone}% ({value - self.minimum()}/{total}) "
                f"after {elapsed:.1f}s"
            )
            logger.info(msg)

    def _format_time(self, seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _format_rate(self, rate: float) -> str:
        if rate < 0.01:
            return "— it/s"
        elif rate < 0.1:
            return f"{rate * 60:.1f} it/min"
        elif rate < 1000:
            return f"{rate:.1f} it/s"
        elif rate < 1_000_000:
            return f"{rate / 1_000:.1f} kit/s"
        else:
            return f"{rate / 1_000_000:.1f} Mit/s"