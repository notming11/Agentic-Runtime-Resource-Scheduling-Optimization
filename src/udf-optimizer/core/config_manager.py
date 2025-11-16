"""
Configuration management for DeerFlow parallelization.

Handles loading and managing configuration from YAML files and environment.
"""

import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .workflow_types import Configuration

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Manages loading and accessing parallelization configuration."""
    
    DEFAULT_CONFIG = {
        "parallelization": {
            "enabled": True,
            "max_concurrent_tasks": 10,
            "max_tasks_per_second": 5.0,
            "task_timeout_seconds": 300,
            "batch_timeout_seconds": 900,
            "retry_on_failure": True,
            "max_retries": 3,
            "retry_backoff_seconds": [2, 10, 30],
            "failure_mode": "partial_completion",
            "dependency_strategy": "llm_based"
        }
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file (optional)
        """
        self.config_path = config_path
        self._config_dict: Dict[str, Any] = {}
        self._parallelization_config: Optional[Configuration] = None
        
        if config_path and config_path.exists():
            self.load_from_file(config_path)
        else:
            logger.info("No config file provided, using defaults")
            self._config_dict = self.DEFAULT_CONFIG.copy()
    
    def load_from_file(self, config_path: Path) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
            
            if loaded_config:
                self._config_dict = loaded_config
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Empty config file, using defaults")
                self._config_dict = self.DEFAULT_CONFIG.copy()
                
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Using default configuration")
            self._config_dict = self.DEFAULT_CONFIG.copy()
    
    def get_parallelization_config(self) -> Configuration:
        """
        Get the parallelization configuration object.
        
        Returns:
            Configuration object with parallelization settings
        """
        if self._parallelization_config is None:
            parallel_dict = self._config_dict.get("parallelization", {})
            
            # Merge with defaults
            merged_config = self.DEFAULT_CONFIG["parallelization"].copy()
            merged_config.update(parallel_dict)
            
            self._parallelization_config = Configuration.from_dict(merged_config)
            
            logger.info(f"Parallelization config: enabled={self._parallelization_config.enabled}, "
                       f"max_concurrent={self._parallelization_config.max_concurrent_tasks}")
        
        return self._parallelization_config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        if "parallelization" in updates:
            parallel_updates = updates["parallelization"]
            current_parallel = self._config_dict.get("parallelization", {})
            current_parallel.update(parallel_updates)
            self._config_dict["parallelization"] = current_parallel
            
            # Reset cached config to force reload
            self._parallelization_config = None
            logger.info(f"Updated parallelization config: {parallel_updates}")
    
    def save_to_file(self, output_path: Path) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path where configuration should be saved
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config_dict, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved configuration to {output_path}")
        except Exception as e:
            logger.error(f"Error saving config to {output_path}: {e}")
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self._config_dict.copy()
    
    @staticmethod
    def create_default_config_file(output_path: Path) -> None:
        """
        Create a default configuration file with documentation.
        
        Args:
            output_path: Path where default config should be created
        """
        config_content = """# DeerFlow Parallelization Configuration

parallelization:
  # Enable or disable parallelization
  enabled: true
  
  # Concurrency limits
  max_concurrent_tasks: 10      # Maximum tasks executing simultaneously
  max_tasks_per_second: 5.0     # Rate limit for launching tasks
  
  # Timeout settings
  task_timeout_seconds: 300     # 5 minutes per task
  batch_timeout_seconds: 900    # 15 minutes per batch
  
  # Error handling
  retry_on_failure: true
  max_retries: 3
  retry_backoff_seconds: [2, 10, 30]  # Exponential backoff delays
  failure_mode: "partial_completion"  # Options: "fail_fast", "partial_completion"
  
  # Dependency analysis
  dependency_strategy: "llm_based"  # Options: "llm_based", "heuristic", "explicit"

# Example workflow configuration
workflow:
  enable_parallelization: true
  log_level: "INFO"
"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            logger.info(f"Created default configuration file at {output_path}")
        except Exception as e:
            logger.error(f"Error creating default config: {e}")


def load_configuration(config_path: Optional[Path] = None) -> Configuration:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Optional path to configuration file
    
    Returns:
        Configuration object
    """
    manager = ConfigurationManager(config_path)
    return manager.get_parallelization_config()


# Example configuration for different scenarios
EXAMPLE_CONFIGS = {
    "speed_optimized": {
        "parallelization": {
            "enabled": True,
            "max_concurrent_tasks": 20,
            "max_tasks_per_second": 10.0,
            "task_timeout_seconds": 180,
            "dependency_strategy": "heuristic",
            "retry_on_failure": False
        }
    },
    
    "reliability_optimized": {
        "parallelization": {
            "enabled": True,
            "max_concurrent_tasks": 5,
            "max_tasks_per_second": 2.0,
            "task_timeout_seconds": 600,
            "dependency_strategy": "llm_based",
            "retry_on_failure": True,
            "max_retries": 5
        }
    },
    
    "cost_optimized": {
        "parallelization": {
            "enabled": True,
            "max_concurrent_tasks": 3,
            "dependency_strategy": "heuristic",
            "retry_on_failure": False
        }
    },
    
    "sequential_fallback": {
        "parallelization": {
            "enabled": False
        }
    }
}


def get_example_config(config_name: str) -> Configuration:
    """
    Get a pre-configured example configuration.
    
    Args:
        config_name: Name of example config (e.g., "speed_optimized")
    
    Returns:
        Configuration object
    
    Raises:
        KeyError: If config_name not found
    """
    if config_name not in EXAMPLE_CONFIGS:
        available = ", ".join(EXAMPLE_CONFIGS.keys())
        raise KeyError(f"Unknown config '{config_name}'. Available: {available}")
    
    config_dict = EXAMPLE_CONFIGS[config_name]["parallelization"]
    
    # Merge with defaults
    merged = ConfigurationManager.DEFAULT_CONFIG["parallelization"].copy()
    merged.update(config_dict)
    
    return Configuration.from_dict(merged)
