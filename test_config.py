from gui.config_manager import ConfigManager
from datetime import datetime
config_manager = ConfigManager()
settings = config_manager.load_temp_settings()
print("Loaded settings:", settings)
temp_dir = settings.get('temp_directory', '')
retention_hours = settings.get('retention_hours', 24)
max_size_mb = settings.get('max_size_mb', 500)
last_cleanup_str = settings.get('last_cleanup', '')
if last_cleanup_str:
  last_cleanup = datetime.fromisoformat(last_cleanup_str.replace('Z', '+00:00'))
  print("Last cleanup:", last_cleanup)
else:
  print("No last cleanup")
print("Temp dir:", temp_dir)
print("Retention hours:", retention_hours)
print("Max size MB:", max_size_mb)
result = config_manager.cleanup_temp_files()
print("Cleanup result:", result)