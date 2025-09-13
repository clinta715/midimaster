"""
Configuration Manager for MIDI Master GUI

This module handles saving and loading of GUI configurations and settings,
including output paths, temp directories, and generation parameters.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ConfigManager:
    """
    Manages saving and loading of application configurations.
    """

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.temp_config_file = self.config_dir / "temp_settings.json"

    def save_configuration(self, params: Dict[str, Any], name: str = "", description: str = "") -> str:
        """
        Save current parameters to a configuration file.

        Args:
            params: Current generation parameters
            name: Configuration name (auto-generated if empty)
            description: Configuration description

        Returns:
            Path to the saved configuration file
        """
        if not name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            genre = params.get('genre', 'unknown')
            mood = params.get('mood', 'unknown')
            name = f"config_{genre}_{mood}_{timestamp}"

        config_data = {
            "version": "1.0",
            "name": name,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "parameters": params.copy()
        }

        # Ensure output folder is included
        if 'output_folder' not in config_data['parameters']:
            config_data['parameters']['output_folder'] = 'output/'

        config_path = self.config_dir / f"{name}.json"

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        return str(config_path)

    def load_configuration(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration parameters
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # Validate configuration structure
        if 'parameters' not in config_data:
            raise ValueError("Invalid configuration file: missing parameters")

        return config_data['parameters']

    def list_configurations(self) -> list:
        """
        List all available configuration files.

        Returns:
            List of configuration file information
        """
        configs = []
        for config_file in self.config_dir.glob("*.json"):
            if config_file.name == "temp_settings.json":
                continue

            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                configs.append({
                    'path': str(config_file),
                    'name': config_data.get('name', config_file.stem),
                    'description': config_data.get('description', ''),
                    'timestamp': config_data.get('timestamp', ''),
                    'version': config_data.get('version', '1.0')
                })
            except (json.JSONDecodeError, KeyError):
                # Skip invalid configuration files
                continue

        # Sort by timestamp (most recent first)
        configs.sort(key=lambda x: x['timestamp'], reverse=True)
        return configs

    def save_temp_settings(self, temp_dir: str, auto_cleanup: bool = True,
                          retention_hours: int = 24, max_size_mb: int = 500) -> None:
        """
        Save temporary file settings.

        Args:
            temp_dir: Temporary directory path
            auto_cleanup: Enable automatic cleanup
            retention_hours: Hours to retain temp files
            max_size_mb: Maximum size of temp directory in MB
        """
        settings = {
            "temp_directory": temp_dir,
            "auto_cleanup": auto_cleanup,
            "retention_hours": retention_hours,
            "max_size_mb": max_size_mb,
            "last_cleanup": datetime.now().isoformat()
        }

        with open(self.temp_config_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)

    def load_temp_settings(self) -> Dict[str, Any]:
        """
        Load temporary file settings.

        Returns:
            Dictionary with temp settings, using defaults if file doesn't exist
        """
        defaults = {
            "temp_directory": "",
            "auto_cleanup": True,
            "retention_hours": 24,
            "max_size_mb": 500,
            "last_cleanup": datetime.now().isoformat()
        }

        if not self.temp_config_file.exists():
            return defaults

        try:
            with open(self.temp_config_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                defaults.update(settings)
                return defaults
        except json.JSONDecodeError:
            return defaults

    def delete_configuration(self, config_path: str) -> bool:
        """
        Delete a configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            Path(config_path).unlink()
            return True
        except (OSError, FileNotFoundError):
            return False

    def cleanup_temp_files(self, temp_dir: Optional[str] = None, retention_hours: int = 24, max_size_mb: int = 100) -> Dict[str, Any]:
        """
        Clean up temporary files in the temp directory.

        Args:
            temp_dir: Temporary directory path (uses settings if None)
            retention_hours: Delete files older than this many hours
            max_size_mb: Delete files if total size exceeds this many MB

        Returns:
            Dictionary with cleanup statistics
        """
        if temp_dir is None:
            settings = self.load_temp_settings()
            temp_dir = settings.get('temp_directory', '')
            if not temp_dir:
                return {'error': 'No temp directory configured'}

        temp_path = Path(temp_dir)
        if not temp_path.exists():
            return {'error': f'Temp directory does not exist: {temp_dir}'}

        current_time = time.time()
        max_age_seconds = retention_hours * 3600
        max_size_bytes = max_size_mb * 1024 * 1024

        deleted_files = []
        total_size_deleted = 0
        errors = []

        # Collect files to potentially delete
        files_to_check = []
        total_current_size = 0

        for root, dirs, files in os.walk(temp_path):
            for file in files:
                if file.endswith('.mid'):
                    file_path = Path(root) / file
                    try:
                        file_stat = file_path.stat()
                        file_age = current_time - file_stat.st_mtime
                        file_size = file_stat.st_size
                        total_current_size += file_size

                        files_to_check.append({
                            'path': file_path,
                            'age': file_age,
                            'size': file_size
                        })
                    except OSError as e:
                        errors.append(f"Could not access {file_path}: {e}")

        # Sort files by age (oldest first)
        files_to_check.sort(key=lambda x: x['age'], reverse=True)

        # Delete files based on age or size limits
        for file_info in files_to_check:
            should_delete = False

            # Delete if older than retention period
            if file_info['age'] > max_age_seconds:
                should_delete = True

            # Delete if total size exceeds limit (delete oldest files first)
            elif total_current_size > max_size_bytes:
                should_delete = True

            if should_delete:
                try:
                    file_info['path'].unlink()
                    deleted_files.append(str(file_info['path']))
                    total_size_deleted += file_info['size']
                    total_current_size -= file_info['size']
                except OSError as e:
                    errors.append(f"Could not delete {file_info['path']}: {e}")

        # Update last cleanup timestamp
        settings = self.load_temp_settings()
        settings['last_cleanup'] = datetime.now().isoformat()
        self.save_temp_settings(
            settings['temp_directory'],
            settings['auto_cleanup'],
            settings['retention_hours'],
            settings['max_size_mb']
        )

        return {
            'files_deleted': len(deleted_files),
            'total_size_deleted_mb': total_size_deleted / (1024 * 1024),
            'remaining_size_mb': total_current_size / (1024 * 1024),
            'errors': errors,
            'deleted_files': deleted_files[:10]  # Limit to first 10 for summary
        }

    def validate_configuration(self, config_path: str) -> bool:
        """
        Validate a configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            True if valid, False otherwise
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            required_keys = ['version', 'parameters']
            for key in required_keys:
                if key not in config_data:
                    return False

            # Validate parameters
            params = config_data['parameters']
            required_params = ['genre', 'mood', 'tempo', 'bars', 'density']
            for param in required_params:
                if param not in params:
                    return False

            return True
        except (json.JSONDecodeError, OSError):
            return False