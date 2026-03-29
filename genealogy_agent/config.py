"""
Configuration loader — reads config.yaml and provides defaults.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable overrides.

    Checks in order:
    1. Explicit path argument
    2. CONFIG_FILE environment variable
    3. config.yaml in current directory
    4. Falls back to defaults
    """
    config: Dict[str, Any] = _defaults()

    # Find config file
    path = config_path or os.environ.get("CONFIG_FILE", "config.yaml")
    config_file = Path(path)

    if config_file.exists():
        try:
            import yaml

            with open(config_file) as f:
                user_config = yaml.safe_load(f) or {}
            _deep_merge(config, user_config)
        except ImportError:
            # No PyYAML — try simple key=value parsing or skip
            pass

    # Environment overrides
    if os.environ.get("OLLAMA_URL"):
        config["ollama"]["url"] = os.environ["OLLAMA_URL"]
    if os.environ.get("GEDCOM_FILE"):
        config["app"]["gedcom"] = os.environ["GEDCOM_FILE"]
    if os.environ.get("WS_PORT"):
        config["server"]["ws_port"] = int(os.environ["WS_PORT"])
    if os.environ.get("WEB_PORT"):
        config["server"]["web_port"] = int(os.environ["WEB_PORT"])
    if os.environ.get("APP_TITLE"):
        config["app"]["title"] = os.environ["APP_TITLE"]

    return config


def _defaults() -> Dict[str, Any]:
    return {
        "server": {
            "host": "0.0.0.0",
            "ws_port": 8765,
            "web_port": 8766,
        },
        "app": {
            "title": "Genealogy Agent",
            "gedcom": "data/Toll Family Tree.ged",
            "knowledge_db": "data/knowledge.db",
        },
        "ollama": {
            "url": "http://localhost:11434",
            "models": {
                "researcher": "llama3.2:3b",
                "fact_checker": "qwen2.5:7b",
                "narrator": "llama3.1:8b",
            },
        },
        "theme": {
            "primary": "#e94560",
            "background": "#1a1a2e",
            "surface": "#16213e",
            "border": "#0f3460",
            "text": "#e0e0e0",
            "text_muted": "#666666",
            "success": "#4ecca3",
        },
    }


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
