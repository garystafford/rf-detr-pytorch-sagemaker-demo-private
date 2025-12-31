# Demonstrates different video detection and segmentation options
# using RFDETRBase model and Supervision library with Object Tracking
# Author: Gary Stafford
# Date: December 2025

import os
import time
import warnings
from collections import Counter

# Suppress PyTorch meshgrid warning from rfdetr internals
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

import numpy as np
import supervision as sv
import cv2

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.util.coco_classes import COCO_CLASSES

# ========= CONFIG =========
SOURCE_VIDEO_PATH = (
    "../sample_images/kling_traffic_10s_03.mp4"  # <-- set your input video
)
TARGET_VIDEO_PATH = "output_annotated.mp4"  # <-- set your output video
MODEL_VARIANT = "small"  # Options: "nano", "small", "medium", "base", "large"
THRESHOLD = 0.50  # confidence threshold for displaying boxes
SHOW_OBJECT_COUNT = True  # Display object count overlay
ENABLE_TRACKING = True  # Enable object tracking across frames
# Filter to specific classes (empty list = track all)
# Examples: ["person"], ["car", "truck", "bus"], ["person", "bicycle"]
TRACK_CLASSES = ["car"]  # <-- set classes to track, or [] for all
# Only show moving objects (requires ENABLE_TRACKING=True)
ONLY_MOVING = True  # <-- set True to filter out stationary objects
MOTION_THRESHOLD = 10  # pixels - minimum movement to be considered "moving"
# Pre-scale frames to model resolution (may improve performance)
PRE_SCALE = True  # <-- set True to scale frames before inference
# ==========================

assert os.path.exists(SOURCE_VIDEO_PATH), f"Source video not found: {SOURCE_VIDEO_PATH}"

print("Loading RF-DETR model...")
model_classes = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}
if MODEL_VARIANT not in model_classes:
    raise ValueError(
        f"Invalid MODEL_VARIANT: {MODEL_VARIANT}. Use: {list(model_classes.keys())}"
    )
model = model_classes[MODEL_VARIANT]()
print(
    f"Using model: RF-DETR {MODEL_VARIANT.capitalize()} (resolution: {model.model.resolution})"
)
model_resolution = model.model.resolution
print("Optimizing model for inference...")
model.optimize_for_inference(compile=False)
print("Model loaded and optimized.")

# Initialize tracker (ByteTrack is fast and accurate)
tracker = sv.ByteTrack() if ENABLE_TRACKING else None

# Track unique objects seen across all frames
unique_tracker_ids = set()

# Track object positions for motion detection
# tracker_id -> list of recent positions [(x, y), ...]
object_position_history = {}

box_annotator = sv.BoxAnnotator(
    thickness=2,
)
label_annotator = sv.LabelAnnotator(
    smart_position=True,
)
# Trace annotator shows object paths
trace_annotator = (
    sv.TraceAnnotator(
        thickness=2,
        trace_length=30,  # Number of frames to show trail
    )
    if ENABLE_TRACKING
    else None
)


def draw_object_count(
    frame: np.ndarray, detections: sv.Detections, unique_count: int
) -> np.ndarray:
    """Draw object count overlay on the frame."""
    # Count objects by class
    class_counts = (
        Counter(COCO_CLASSES[class_id] for class_id in detections.class_id)
        if len(detections) > 0
        else Counter()
    )

    # Build count text
    current_count = len(detections)
    count_lines = [
        f"Current Frame: {current_count}",
        f"Unique Objects: {unique_count}",
        "---",
    ]
    for class_name, count in class_counts.most_common(8):
        count_lines.append(f"  {class_name}: {count}")

    # Draw background rectangle (50% larger)
    padding = 15
    line_height = 34
    text_width = 270
    text_height = len(count_lines) * line_height + padding * 2

    try:
        import cv2

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (padding, padding),
            (text_width + padding, text_height + padding),
            (0, 0, 0),
            -1,
        )
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Draw text (50% larger font)
        for i, line in enumerate(count_lines):
            y_pos = padding + 26 + i * line_height
            color = (0, 255, 255) if "Unique" in line else (255, 255, 255)
            cv2.putText(
                frame,
                line,
                (padding + 8, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                color,
                1,
                cv2.LINE_AA,
            )
    except ImportError:
        pass

    return frame


def callback(frame, index: int):
    global unique_tracker_ids, object_position_history

    # supervision gives BGR; RF-DETR expects RGB
    rgb_frame = frame[:, :, ::-1].copy()
    orig_h, orig_w = rgb_frame.shape[:2]

    # Pre-scale frame to model resolution if enabled
    if PRE_SCALE:
        # Scale longest dimension to model resolution, maintain aspect ratio
        scale = model_resolution / max(orig_h, orig_w)
        if scale < 1.0:  # Only downscale, never upscale
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            rgb_frame = cv2.resize(
                rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA
            )

    # Detect objects
    detections = model.predict(rgb_frame, threshold=THRESHOLD)

    # Scale bounding boxes back to original frame size if pre-scaled
    if PRE_SCALE and scale < 1.0:
        detections.xyxy = detections.xyxy / scale

    # Filter to specific classes if configured
    if TRACK_CLASSES:
        class_ids_to_keep = [
            i
            for i, cid in enumerate(detections.class_id)
            if COCO_CLASSES[cid] in TRACK_CLASSES
        ]
        detections = detections[class_ids_to_keep]

    # Apply tracking to assign consistent IDs across frames
    if ENABLE_TRACKING and tracker is not None:
        detections = tracker.update_with_detections(detections)

        # Track unique objects
        if detections.tracker_id is not None:
            unique_tracker_ids.update(detections.tracker_id.tolist())

        # Filter to only moving objects if enabled
        if ONLY_MOVING and detections.tracker_id is not None and len(detections) > 0:
            moving_indices = []
            for i, (tracker_id, xyxy) in enumerate(
                zip(detections.tracker_id, detections.xyxy)
            ):
                # Calculate center of bounding box
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = (xyxy[1] + xyxy[3]) / 2

                # Initialize history for new objects
                if tracker_id not in object_position_history:
                    object_position_history[tracker_id] = []

                history = object_position_history[tracker_id]
                history.append((x_center, y_center))

                # Keep only last 10 positions
                if len(history) > 10:
                    history.pop(0)

                # Check if object has moved enough from its oldest tracked position
                if len(history) >= 2:
                    first_x, first_y = history[0]
                    distance = np.sqrt(
                        (x_center - first_x) ** 2 + (y_center - first_y) ** 2
                    )
                    if distance >= MOTION_THRESHOLD:
                        moving_indices.append(i)
                else:
                    # New object - include it
                    moving_indices.append(i)

            # Filter detections to only moving objects
            if moving_indices:
                detections = detections[moving_indices]
            else:
                # Keep detections but mark as empty by filtering to empty list
                detections = detections[[]]

    # Build labels with tracker IDs
    if ENABLE_TRACKING and detections.tracker_id is not None:
        labels = [
            f"#{tracker_id} {COCO_CLASSES[class_id]} {confidence:.2f}"
            for tracker_id, class_id, confidence in zip(
                detections.tracker_id, detections.class_id, detections.confidence
            )
        ]
    else:
        labels = [
            f"{COCO_CLASSES[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

    # Annotate frame
    annotated_frame = frame.copy()

    # Draw traces (object paths) if tracking enabled
    if ENABLE_TRACKING and trace_annotator is not None:
        annotated_frame = trace_annotator.annotate(annotated_frame, detections)

    annotated_frame = box_annotator.annotate(annotated_frame, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

    # Add object count overlay
    if SHOW_OBJECT_COUNT:
        annotated_frame = draw_object_count(
            annotated_frame, detections, len(unique_tracker_ids)
        )

    if index % 50 == 0 or index == 0:
        print(
            f"Frame {index} - "
            f"Current: {len(detections)}, "
            f"Unique total: {len(unique_tracker_ids)}",
            flush=True,
        )
    elif index < 10:
        # Show first few frames to confirm it's working
        print(f"Frame {index}...", flush=True)

    return annotated_frame


print("Starting video processing...")
print(f"Tracking enabled: {ENABLE_TRACKING}")
print(f"Pre-scaling enabled: {PRE_SCALE}")
if TRACK_CLASSES:
    print(f"Filtering to classes: {TRACK_CLASSES}")
if ONLY_MOVING:
    print(f"Only showing moving objects (threshold: {MOTION_THRESHOLD}px)")

start_time = time.time()
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback,
)
elapsed_time = time.time() - start_time

print("Processing complete.")
print(f"Processing time: {elapsed_time:.2f} seconds")
print(f"Total unique objects tracked: {len(unique_tracker_ids)}")
print("Final file size (bytes):", os.path.getsize(TARGET_VIDEO_PATH))
