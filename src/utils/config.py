"""
Configuration Loading Utilities
================================

This module provides functions for loading and validating YAML configuration files.

Why Configuration Files?
------------------------
Hardcoding values in Python code is problematic:
- Changing settings requires editing source code
- Hard to track what settings were used for experiments
- Easy to accidentally commit sensitive data (API keys)

Configuration files solve these problems by separating settings from code.

Usage
-----
>>> from src.utils.config import load_config
>>> config = load_config('configs/data_config.yaml')
>>> print(config['assets']['SPY']['name'])
'S&P 500 ETF'
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing the configuration.
    
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the YAML file is malformed.
    
    Examples
    --------
    >>> config = load_config('configs/training_config.yaml')
    >>> config['ppo']['learning_rate']
    0.0003
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Make sure you're running from the project root directory."
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_all_configs(config_dir: str = 'configs') -> Dict[str, Dict[str, Any]]:
    """
    Load all configuration files from a directory.
    
    Parameters
    ----------
    config_dir : str
        Path to the directory containing configuration files.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping config names to their contents.
        Keys are filenames without the .yaml extension.
    
    Examples
    --------
    >>> configs = load_all_configs()
    >>> configs['data_config']['assets']['SPY']['name']
    'S&P 500 ETF'
    """
    config_dir = Path(config_dir)
    configs = {}
    
    for config_file in config_dir.glob('*.yaml'):
        config_name = config_file.stem  # Filename without extension
        configs[config_name] = load_config(config_file)
    
    return configs


def get_env_variable(var_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an environment variable, with optional default.
    
    This is useful for loading sensitive data like API keys that shouldn't
    be stored in configuration files.
    
    Parameters
    ----------
    var_name : str
        Name of the environment variable.
    default : Optional[str]
        Default value if the variable is not set.
    
    Returns
    -------
    Optional[str]
        The value of the environment variable, or the default.
    
    Examples
    --------
    >>> api_key = get_env_variable('FRED_API_KEY')
    >>> if api_key is None:
    ...     print("Please set FRED_API_KEY environment variable")
    """
    return os.environ.get(var_name, default)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Values in `override` take precedence over values in `base`.
    This is useful for applying experiment-specific overrides to base configs.
    
    Parameters
    ----------
    base : Dict[str, Any]
        Base configuration dictionary.
    override : Dict[str, Any]
        Override configuration dictionary.
    
    Returns
    -------
    Dict[str, Any]
        Merged configuration dictionary.
    
    Examples
    --------
    >>> base = {'learning_rate': 0.001, 'batch_size': 64}
    >>> override = {'learning_rate': 0.0001}
    >>> merged = merge_configs(base, override)
    >>> merged
    {'learning_rate': 0.0001, 'batch_size': 64}
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override the value
            result[key] = value
    
    return result


# Make key functions available at package level
__all__ = ['load_config', 'load_all_configs', 'get_env_variable', 'merge_configs']
