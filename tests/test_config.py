"""Tests for configuration management."""

import pytest

from config import InferenceConfig, ModelConfig


class TestModelConfig:
    """Test ModelConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.model_name == "rf-detr-large.pth"
        assert config.model_type == "rfdetr-large"
        assert config.resolution == 560
        assert config.confidence_threshold == 0.25
        assert config.optimize is True
        assert config.compile is True
        assert config.labels == "coco"

    def test_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("RFDETR_MODEL", "custom-model.pth")
        monkeypatch.setenv("RFDETR_MODEL_TYPE", "rfdetr-base")
        monkeypatch.setenv("RFDETR_RESOLUTION", "448")
        monkeypatch.setenv("RFDETR_CONF", "0.5")
        monkeypatch.setenv("RFDETR_OPTIMIZE", "false")

        config = ModelConfig.from_env()
        assert config.model_name == "custom-model.pth"
        assert config.model_type == "rfdetr-base"
        assert config.resolution == 448
        assert config.confidence_threshold == 0.5
        assert config.optimize is False


class TestInferenceConfig:
    """Test InferenceConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = InferenceConfig()
        assert config.confidence is None
        assert config.classes is None
        assert config.max_detections is None
        assert config.min_box_area is None

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        config = InferenceConfig(confidence=0.5, max_detections=10, min_box_area=100.0)
        config.validate()  # Should not raise

    def test_validate_invalid_confidence(self):
        """Test validation with invalid confidence."""
        config = InferenceConfig(confidence=1.5)
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            config.validate()

    def test_validate_invalid_max_detections(self):
        """Test validation with invalid max_detections."""
        config = InferenceConfig(max_detections=-1)
        with pytest.raises(ValueError, match="max_detections must be positive"):
            config.validate()

    def test_validate_invalid_min_box_area(self):
        """Test validation with invalid min_box_area."""
        config = InferenceConfig(min_box_area=-10.0)
        with pytest.raises(ValueError, match="min_box_area must be non-negative"):
            config.validate()
