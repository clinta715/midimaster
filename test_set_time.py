import os
import time
from datetime import datetime, timedelta
file_path = r'test_temp\old_preview.mid'
os.utime(file_path, (time.time() - 7200, time.time() - 7200))  # 2 hours old
print(f"Set {file_path} to 2 hours old")