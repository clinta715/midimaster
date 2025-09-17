import os
import time
from datetime import datetime, timedelta

file_path = r'test_temp\old_preview.mid'

# Ensure directory exists
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Ensure file exists before updating timestamp
if not os.path.exists(file_path):
    # Create an empty file
    with open(file_path, 'wb') as f:
        pass

# Set file mtime/atime to 2 hours ago
two_hours_ago = time.time() - 7200
os.utime(file_path, (two_hours_ago, two_hours_ago))
print(f"Set {file_path} to 2 hours old")