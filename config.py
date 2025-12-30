"""Configuration management for RF-DETR SageMaker deployment."""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for RF-DETR model."""

    model_name: str = "rf-detr-large.pth"
    model_type: str = "rfdetr-large"
    resolution: int = 560
    confidence_threshold: float = 0.25
    optimize: bool = True
    compile: bool = True
    labels: str = "coco"

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Create configuration from environment variables."""
        return cls(
            model_name=os.getenv("RFDETR_MODEL", cls.model_name),
            model_type=os.getenv("RFDETR_MODEL_TYPE", cls.model_type),
            resolution=int(os.getenv("RFDETR_RESOLUTION", str(cls.resolution))),
            confidence_threshold=float(
                os.getenv("RFDETR_CONF", str(cls.confidence_threshold))
            ),
            optimize=os.getenv("RFDETR_OPTIMIZE", "true").lower() == "true",
            compile=os.getenv("RFDETR_COMPILE", "true").lower() == "true",
            labels=os.getenv("RFDETR_LABELS", cls.labels),
        )


@dataclass
class InferenceConfig:
    """Configuration for inference requests."""

    confidence: Optional[float] = None
    classes: Optional[List[str]] = None
    max_detections: Optional[int] = None
    min_box_area: Optional[float] = None

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.confidence is not None:
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError(
                    f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
                )

        if self.max_detections is not None:
            if self.max_detections <= 0:
                raise ValueError(
                    f"max_detections must be positive, got {self.max_detections}"
                )

        if self.min_box_area is not None:
            if self.min_box_area < 0:
                raise ValueError(
                    f"min_box_area must be non-negative, got {self.min_box_area}"
                )


# Constants
SUPPORTED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/jpg", "application/json")
MAX_IMAGE_DIMENSION = 10000
DEFAULT_CONF_THRESHOLD = 0.25

MODEL_CLASSES = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-base": "RFDETRBase",
    "rfdetr-large": "RFDETRLarge",
}
