"""Tests for configuration loading — defaults, YAML merge, env overrides."""

import os
from unittest.mock import patch

from genealogy_agent.config import load_config


class TestConfigDefaults:
    """Verify default configuration values."""

    def test_has_server_section(self):
        config = load_config("/nonexistent.yaml")
        assert "server" in config
        assert config["server"]["ws_port"] == 8765
        assert config["server"]["web_port"] == 8766

    def test_has_app_section(self):
        config = load_config("/nonexistent.yaml")
        assert "app" in config
        assert "gedcom" in config["app"]
        assert "knowledge_db" in config["app"]

    def test_has_ollama_section(self):
        config = load_config("/nonexistent.yaml")
        assert "ollama" in config
        assert "models" in config["ollama"]
        assert "researcher" in config["ollama"]["models"]
        assert "fact_checker" in config["ollama"]["models"]
        assert "narrator" in config["ollama"]["models"]

    def test_has_personalities_section(self):
        config = load_config("/nonexistent.yaml")
        assert "personalities" in config
        assert config["personalities"]["enabled"] is True

    def test_has_consensus_section(self):
        config = load_config("/nonexistent.yaml")
        assert "consensus" in config
        assert config["consensus"]["enabled"] is True
        assert config["consensus"]["timeout"] == 30
        assert config["consensus"]["debate_enabled"] is True
        assert config["consensus"]["debate_rounds"] == 2
        assert config["consensus"]["disagreement_threshold"] == 0.6

    def test_has_training_section(self):
        config = load_config("/nonexistent.yaml")
        assert "training" in config
        assert config["training"]["feedback_enabled"] is True
        assert config["training"]["heuristics_enabled"] is True

    def test_has_theme_section(self):
        config = load_config("/nonexistent.yaml")
        assert "theme" in config


class TestConfigEnvOverrides:
    """Verify environment variable overrides."""

    @patch.dict(os.environ, {"OLLAMA_URL": "http://remote:11434"})
    def test_ollama_url_override(self):
        config = load_config("/nonexistent.yaml")
        assert config["ollama"]["url"] == "http://remote:11434"

    @patch.dict(os.environ, {"GEDCOM_FILE": "/custom/path.ged"})
    def test_gedcom_file_override(self):
        config = load_config("/nonexistent.yaml")
        assert config["app"]["gedcom"] == "/custom/path.ged"

    @patch.dict(os.environ, {"WS_PORT": "9999"})
    def test_ws_port_override(self):
        config = load_config("/nonexistent.yaml")
        assert config["server"]["ws_port"] == 9999


class TestConfigYamlMerge:
    """Verify YAML file merging."""

    def test_yaml_merge(self, tmp_path):
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(
            "app:\n  title: Custom Title\nserver:\n  ws_port: 7777\n"
        )
        config = load_config(str(yaml_file))
        assert config["app"]["title"] == "Custom Title"
        assert config["server"]["ws_port"] == 7777
        # Defaults still present for unset values
        assert config["ollama"]["url"] == "http://localhost:11434"
