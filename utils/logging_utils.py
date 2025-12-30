"""Logging utilities for RF-DETR inference."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string
        include_timestamp: Whether to include timestamp in logs

    Returns:
        Configured logger instance
    """
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()), format=format_string, stream=sys.stdout
    )

    return logging.getLogger(__name__)


def log_inference_metrics(
    logger: logging.Logger,
    image_shape: tuple,
    inference_time_ms: float,
    detection_count: int,
    confidence_threshold: float,
) -> None:
    """Log inference performance metrics.

    Args:
        logger: Logger instance
        image_shape: Shape of input image (H, W, C)
        inference_time_ms: Inference time in milliseconds
        detection_count: Number of detections found
        confidence_threshold: Confidence threshold used
    """
    logger.info(
        "Inference completed - Image: %dx%d, Time: %.2fms, "
        "Detections: %d, Threshold: %.3f",
        image_shape[1],
        image_shape[0],  # W, H
        inference_time_ms,
        detection_count,
        confidence_threshold,
    )
