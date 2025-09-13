from gui.config_manager import ConfigManager
import time
import os
config_manager = ConfigManager()
temp_dir = os.path.abspath('./test_temp')
print("Using temp_dir:", temp_dir)
print("Files to check:")
for root, dirs, files in os.walk(temp_dir):
  for file in files:
    if file.endswith('.mid'):
      file_path = os.path.join(root, file)
      file_stat = os.stat(file_path)
      file_age = time.time() - file_stat.st_mtime
      print(f"{file_path}: age {file_age} s")
result = config_manager.cleanup_temp_files(temp_dir=temp_dir, retention_hours=1, max_size_mb=1)
print(result)