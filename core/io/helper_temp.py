import os
import shutil
import time
import datetime
from utils.logger import logger


class Temp_Manager:
    def __init__(self, video_filepath: str, force_clean:bool=False):
        self.video_filepath = video_filepath
        self.video_name = os.path.splitext(os.path.basename(self.video_filepath))[0]

        self.temp_dir_root = os.path.join(os.path.dirname(self.video_filepath), "bvt_temp")
        os.makedirs(self.temp_dir_root, exist_ok=True)

        limit= 0 if force_clean else 1
        self._clean(limit = limit)

    def create(self, sub_name:str, use_existing:bool=False) -> str:
        self.temp_dir = os.path.join(self.temp_dir_root, self.video_name, sub_name)
        if os.path.exists(self.temp_dir):
            if use_existing:
                return self.temp_dir
            try:
                shutil.rmtree(self.temp_dir)
            except (PermissionError, OSError):
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                self.temp_dir = f"{self.temp_dir}_{timestamp}"
                logger.warning(f"[TEMP] Existing dir locked. Using fallback: {self.temp_dir}")

        os.makedirs(self.temp_dir, exist_ok=True)
        return self.temp_dir

    def _clean(self, limit:int = 1):
        if not os.path.isdir(self.temp_dir_root):
            return

        cutoff_time = time.time() - (limit * 24 * 60 * 60)
        
        logger.info(f"[TEMP] Scanning for folders older than {limit} days...")
        cleaned_count = 0

        for entry in os.listdir(self.temp_dir_root):
            full_path = os.path.join(self.temp_dir_root, entry)
            
            if not os.path.isdir(full_path):
                continue
            try:
                mtime = os.path.getmtime(full_path)
                if mtime < cutoff_time:
                    shutil.rmtree(full_path)
                    logger.info(f"[TEMP] Auto-removed old dir ({entry}): {datetime.datetime.fromtimestamp(mtime).isoformat()}")
                    cleaned_count += 1
                else:
                    pass
            except (PermissionError, OSError) as e:
                logger.warning(f"[TEMP] Failed to check/remove old dir {entry}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"[TEMP] Cleanup complete. Removed {cleaned_count} old folders.")