# Object Detection Video V2
# Enhanced video detection with frame skipping, line counting, and heatmap
# using RF-DETR model and Supervision library with Object Tracking
# Author: Gary Stafford
# Date: December 2025

import os
import time
import warnings
from collections import Counter, defaultdict

# Suppress PyTorch meshgrid warning from rfdetr internals
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
# Suppress optimization warning for seg model
warnings.filterwarnings("ignore", message=".*Model is not optimized for inference.*")

import torch
import numpy as np
import supervision as sv
import cv2

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall, RFDETRSegPreview
from rfdetr.util.coco_classes import COCO_CLASSES

# Display device information
print("=== Device Information ===")
if torch.cuda.is_available():
    print(f"Using device: CUDA - {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
elif torch.backends.mps.is_available():
    print("Using device: MPS (Apple Silicon)")
else:
    print("Using device: CPU")
print("==========================\n")

# ========= CONFIG =========
SOURCE_VIDEO_PATH = "chinatown-nyc-optimized.mp4"  # <-- set your input video
TARGET_VIDEO_PATH = "chinatown-nyc-optimized-annotated.mp4"  # <-- set your output video
MODEL_VARIANT = "nano"  # Options: "nano", "small", "medium", "base", "large", "seg"
THRESHOLD = 0.50  # confidence threshold for displaying boxes
OPTIMIZE_MODEL = True  # Optimize model for inference (faster, may not work with seg)

# Annotation toggles
SHOW_BOXES = True  # Draw bounding boxes around detections
SHOW_LABELS = True  # Draw labels (class name, confidence, tracker ID)
SHOW_MASKS = False  # Draw segmentation masks (only works with MODEL_VARIANT="seg")
SHOW_TRACES = False  # Draw object movement trails (requires ENABLE_TRACKING)
SHOW_OBJECT_COUNT = True  # Display object count overlay

# Tracking options
ENABLE_TRACKING = True  # Enable object tracking across frames

# Filter to specific classes (empty list = track all)
# Examples: ["person"], ["car", "truck", "bus"], ["person", "bicycle"]
TRACK_CLASSES = []  # <-- set classes to track, or [] for all

# Only show moving objects (requires ENABLE_TRACKING=True)
ONLY_MOVING = False  # <-- set True to filter out stationary objects
MOTION_THRESHOLD = 10  # pixels - minimum movement to be considered "moving"

# Pre-scale frames to model resolution (may improve performance)
PRE_SCALE = True  # <-- set True to scale frames before inference

# Frame skipping - process every Nth frame (speeds up processing)
SKIP_FRAMES = True  # <-- set True to enable frame skipping
FRAME_SKIP_COUNT = 2  # Process every Nth frame (2 = every other frame)

# Entry/Exit line counting - count objects crossing a virtual line
COUNT_LINE = False  # <-- set True to enable line counting
# Line coordinates as percentage of frame (0.0 to 1.0)
# Default: horizontal line across middle of frame
COUNT_LINE_START = (0.0, 0.75)  # (x%, y%) - left point
COUNT_LINE_END = (1.0, 0.75)  # (x%, y%) - right point
COUNT_LINE_COLOR = (0, 255, 0)  # Green line

# Heatmap overlay - show where objects appear most frequently
SHOW_HEATMAP = False  # <-- set True to show heatmap overlay
HEATMAP_ALPHA = 0.4  # Transparency of heatmap (0.0 to 1.0)
# ==========================

# Validation
assert os.path.exists(SOURCE_VIDEO_PATH), f"Source video not found: {SOURCE_VIDEO_PATH}"
if SHOW_MASKS and MODEL_VARIANT != "seg":
    raise ValueError("SHOW_MASKS=True requires MODEL_VARIANT='seg'")

# Get video info for line counting coordinates
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frame_width = video_info.width
frame_height = video_info.height
total_frames = video_info.total_frames
print(
    f"Video: {frame_width}x{frame_height}, {total_frames} frames, {video_info.fps} fps"
)

# Convert line coordinates from percentages to pixels
if COUNT_LINE:
    line_start = (
        int(COUNT_LINE_START[0] * frame_width),
        int(COUNT_LINE_START[1] * frame_height),
    )
    line_end = (
        int(COUNT_LINE_END[0] * frame_width),
        int(COUNT_LINE_END[1] * frame_height),
    )
    print(f"Count line: {line_start} -> {line_end}")

print("Loading RF-DETR model...")
model_classes = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
    "seg": RFDETRSegPreview,
}
if MODEL_VARIANT not in model_classes:
    raise ValueError(
        f"Invalid MODEL_VARIANT: {MODEL_VARIANT}. Use: {list(model_classes.keys())}"
    )
model = model_classes[MODEL_VARIANT]()

# Seg model may not have resolution attribute in same place
if MODEL_VARIANT == "seg":
    model_resolution = 560  # Default for seg preview
    print("Using model: RF-DETR Segmentation Preview")
else:
    print(
        f"Using model: RF-DETR {MODEL_VARIANT.capitalize()} (resolution: {model.model.resolution})"
    )
    model_resolution = model.model.resolution

# Optimize model if enabled
if OPTIMIZE_MODEL:
    print("Optimizing model for inference...")
    try:
        model.optimize_for_inference(compile=False)
        print("Model optimized.")
    except Exception as e:
        print(f"Optimization failed: {e}")
        print("Continuing without optimization...")
else:
    print("Model optimization disabled.")

# Initialize tracker (ByteTrack is fast and accurate)
tracker = sv.ByteTrack() if ENABLE_TRACKING else None

# Track unique objects seen across all frames
unique_tracker_ids = set()

# Track object positions for motion detection
object_position_history = {}

# Line crossing tracking
objects_crossed = {"in": set(), "out": set()}  # Track which objects crossed
line_cross_counts = {"in": 0, "out": 0}
object_last_side = {}  # tracker_id -> "above" or "below" the line

# Heatmap accumulator
heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)

# Cache for skipped frames
last_detections = None
last_labels = None

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(smart_position=True)
mask_annotator = sv.MaskAnnotator()
trace_annotator = (
    sv.TraceAnnotator(thickness=2, trace_length=40) if ENABLE_TRACKING else None
)


def point_side_of_line(point, line_start, line_end):
    """Determine which side of a line a point is on.
    Returns positive for one side, negative for other, 0 if on line."""
    return (line_end[0] - line_start[0]) * (point[1] - line_start[1]) - (
        line_end[1] - line_start[1]
    ) * (point[0] - line_start[0])


def check_line_crossing(tracker_id, center_x, center_y):
    """Check if an object crossed the counting line."""
    global line_cross_counts, object_last_side, objects_crossed

    current_side = point_side_of_line((center_x, center_y), line_start, line_end)
    side_name = "above" if current_side > 0 else "below"

    if tracker_id in object_last_side:
        last_side = object_last_side[tracker_id]
        if (
            last_side != side_name
            and tracker_id not in objects_crossed["in"]
            and tracker_id not in objects_crossed["out"]
        ):
            # Object crossed the line
            if last_side == "above":
                line_cross_counts["in"] += 1
                objects_crossed["in"].add(tracker_id)
            else:
                line_cross_counts["out"] += 1
                objects_crossed["out"].add(tracker_id)

    object_last_side[tracker_id] = side_name


def update_heatmap(detections):
    """Add detection centers to heatmap accumulator."""
    global heatmap_accumulator

    for xyxy in detections.xyxy:
        center_x = int((xyxy[0] + xyxy[2]) / 2)
        center_y = int((xyxy[1] + xyxy[3]) / 2)

        # Add gaussian blob at detection center
        radius = 30
        y_min = max(0, center_y - radius)
        y_max = min(frame_height, center_y + radius)
        x_min = max(0, center_x - radius)
        x_max = min(frame_width, center_x + radius)

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if dist < radius:
                    heatmap_accumulator[y, x] += 1.0 - (dist / radius)


def draw_heatmap_overlay(frame):
    """Draw heatmap overlay on frame."""
    if heatmap_accumulator.max() == 0:
        return frame

    # Normalize heatmap
    normalized = heatmap_accumulator / heatmap_accumulator.max()

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    # Blend with frame
    return cv2.addWeighted(frame, 1 - HEATMAP_ALPHA, heatmap_colored, HEATMAP_ALPHA, 0)


def draw_count_line(frame):
    """Draw the counting line on the frame."""
    cv2.line(frame, line_start, line_end, COUNT_LINE_COLOR, 3)

    # Draw counts on the left side of the line
    left_x = min(line_start[0], line_end[0]) + 10
    left_y = (line_start[1] + line_end[1]) // 2

    cv2.putText(
        frame,
        f"In: {line_cross_counts['in']}",
        (left_x, left_y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Out: {line_cross_counts['out']}",
        (left_x, left_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_object_count(frame, detections, unique_count):
    """Draw object count overlay on the frame."""
    class_counts = (
        Counter(COCO_CLASSES[class_id] for class_id in detections.class_id)
        if len(detections) > 0
        else Counter()
    )

    current_count = len(detections)
    count_lines = [
        f"Current Objects: {current_count}",
        f"Unique Objects: {unique_count}",
        "---",
    ]
    for class_name, count in class_counts.most_common(8):
        count_lines.append(f"  {class_name}: {count}")

    padding = 15
    line_height = 34
    text_width = 270
    text_height = len(count_lines) * line_height + padding * 2

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (padding, padding),
        (text_width + padding, text_height + padding),
        (0, 0, 0),
        -1,
    )
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

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

    return frame


def callback(frame, index: int):
    global unique_tracker_ids, object_position_history
    global last_detections, last_labels

    # Frame skipping - reuse last detections for skipped frames
    if SKIP_FRAMES and index % FRAME_SKIP_COUNT != 0 and last_detections is not None:
        detections = last_detections
        labels = last_labels
    else:
        # Process this frame
        rgb_frame = frame[:, :, ::-1].copy()
        orig_h, orig_w = rgb_frame.shape[:2]

        # Pre-scale frame to model resolution if enabled (skip for seg model)
        scale = 1.0
        if PRE_SCALE and MODEL_VARIANT != "seg":
            scale = model_resolution / max(orig_h, orig_w)
            if scale < 1.0:
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                rgb_frame = cv2.resize(
                    rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                )

        # Detect objects
        detections = model.predict(rgb_frame, threshold=THRESHOLD)

        # Scale bounding boxes back to original frame size if pre-scaled
        if PRE_SCALE and MODEL_VARIANT != "seg" and scale < 1.0:
            detections.xyxy = detections.xyxy / scale

        # Filter to specific classes if configured
        if TRACK_CLASSES:
            class_ids_to_keep = [
                i
                for i, cid in enumerate(detections.class_id)
                if COCO_CLASSES[cid] in TRACK_CLASSES
            ]
            detections = detections[class_ids_to_keep]

        # Apply tracking
        if ENABLE_TRACKING and tracker is not None:
            detections = tracker.update_with_detections(detections)

            if detections.tracker_id is not None:
                unique_tracker_ids.update(detections.tracker_id.tolist())

            # Motion filtering
            if (
                ONLY_MOVING
                and detections.tracker_id is not None
                and len(detections) > 0
            ):
                moving_indices = []
                for i, (tracker_id, xyxy) in enumerate(
                    zip(detections.tracker_id, detections.xyxy)
                ):
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2

                    if tracker_id not in object_position_history:
                        object_position_history[tracker_id] = []

                    history = object_position_history[tracker_id]
                    history.append((x_center, y_center))

                    if len(history) > 10:
                        history.pop(0)

                    if len(history) >= 2:
                        first_x, first_y = history[0]
                        distance = np.sqrt(
                            (x_center - first_x) ** 2 + (y_center - first_y) ** 2
                        )
                        if distance >= MOTION_THRESHOLD:
                            moving_indices.append(i)
                    else:
                        moving_indices.append(i)

                if moving_indices:
                    detections = detections[moving_indices]
                else:
                    detections = detections[[]]

        # Build labels
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
                for class_id, confidence in zip(
                    detections.class_id, detections.confidence
                )
            ]

        # Cache for frame skipping
        last_detections = detections
        last_labels = labels

    # Line crossing detection
    if COUNT_LINE and ENABLE_TRACKING and detections.tracker_id is not None:
        for tracker_id, xyxy in zip(detections.tracker_id, detections.xyxy):
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2
            check_line_crossing(tracker_id, center_x, center_y)

    # Update heatmap
    if SHOW_HEATMAP and len(detections) > 0:
        update_heatmap(detections)

    # Annotate frame
    annotated_frame = frame.copy()

    # Draw heatmap first (underneath everything)
    if SHOW_HEATMAP:
        annotated_frame = draw_heatmap_overlay(annotated_frame)

    # Draw segmentation masks (underneath boxes/labels)
    if SHOW_MASKS and detections.mask is not None:
        annotated_frame = mask_annotator.annotate(annotated_frame, detections)

    # Draw traces
    if SHOW_TRACES and ENABLE_TRACKING and trace_annotator is not None:
        annotated_frame = trace_annotator.annotate(annotated_frame, detections)

    # Draw bounding boxes
    if SHOW_BOXES:
        annotated_frame = box_annotator.annotate(annotated_frame, detections)

    # Draw labels
    if SHOW_LABELS:
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

    # Draw counting line
    if COUNT_LINE:
        annotated_frame = draw_count_line(annotated_frame)

    # Draw object count overlay
    if SHOW_OBJECT_COUNT:
        annotated_frame = draw_object_count(
            annotated_frame, detections, len(unique_tracker_ids)
        )

    # if index % 50 == 0 or index == 0:
    #     msg = f"Frame {index} - Current: {len(detections)}, Unique: {len(unique_tracker_ids)}"
    #     if COUNT_LINE:
    #         msg += f", In: {line_cross_counts['in']}, Out: {line_cross_counts['out']}"
    #     print(msg, flush=True)
    # elif index < 10:
    #     print(f"Frame {index}...", flush=True)

    if index % 50 == 0 or index == 0:
        msg = f"Frame {index} - Current: {len(detections)}, Unique: {len(unique_tracker_ids)}"
        if COUNT_LINE:
            msg += f", In: {line_cross_counts['in']}, Out: {line_cross_counts['out']}"
        print(msg, flush=True)

    return annotated_frame


# Print configuration
print("\n=== Configuration ===")
print(f"Source video: {SOURCE_VIDEO_PATH}")
print(f"Target video: {TARGET_VIDEO_PATH}")
print(f"Model variant: {MODEL_VARIANT}")
print(f"Confidence threshold: {THRESHOLD}")
print(f"Model optimized: {OPTIMIZE_MODEL}")
print(f"---")
print(f"Show boxes: {SHOW_BOXES}")
print(f"Show labels: {SHOW_LABELS}")
print(f"Show masks: {SHOW_MASKS}")
print(f"Show traces: {SHOW_TRACES}")
print(f"Show object count: {SHOW_OBJECT_COUNT}")
print(f"Tracking enabled: {ENABLE_TRACKING}")
print(f"Pre-scaling enabled: {PRE_SCALE}")
if SKIP_FRAMES:
    print(f"Frame skipping: every {FRAME_SKIP_COUNT} frames")
else:
    print(f"Frame skipping: disabled")
if TRACK_CLASSES:
    print(f"Filtering to classes: {TRACK_CLASSES}")
if ONLY_MOVING:
    print(f"Only moving objects (threshold: {MOTION_THRESHOLD}px)")
if COUNT_LINE:
    print(f"Line counting enabled: {line_start} -> {line_end}")
if SHOW_HEATMAP:
    print(f"Heatmap overlay enabled (alpha: {HEATMAP_ALPHA})")
print("=====================\n")

print("Starting video processing...")
start_time = time.time()
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback,
)
elapsed_time = time.time() - start_time
fps_processed = total_frames / elapsed_time if elapsed_time > 0 else 0
avg_time_per_frame = elapsed_time / total_frames if total_frames > 0 else 0

print("\n=== Results ===")
print(f"Total frames: {total_frames}")
print(f"Processing time: {elapsed_time:.2f} seconds")
print(f"Average FPS: {fps_processed:.2f} frames/sec")
print(f"Average time per frame: {avg_time_per_frame*1000:.2f} ms")
print(f"Total unique objects tracked: {len(unique_tracker_ids)}")
if COUNT_LINE:
    print(
        f"Line crossings - In: {line_cross_counts['in']}, Out: {line_cross_counts['out']}"
    )
print(f"Output file: {TARGET_VIDEO_PATH}")
print(f"File size: {os.path.getsize(TARGET_VIDEO_PATH):,} bytes")
print("=================")
