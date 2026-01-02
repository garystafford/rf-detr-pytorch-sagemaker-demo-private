"""
Extract a frame from video and overlay a percentage grid for ROI configuration
Author: Gary Stafford
Date: December 2025
"""

import cv2
import numpy as np

# Configuration
VIDEO_PATH = "chinatown-nyc-optimized-v2.mp4"
OUTPUT_PATH = "grid_reference_frame.jpg"
FRAME_NUMBER = 0  # Which frame to extract (0 = first frame)
GRID_SPACING = 0.05  # 5% increments (0.05 = 5%)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

# Set to desired frame
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)

# Read frame
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"Error: Could not read frame {FRAME_NUMBER} from {VIDEO_PATH}")
    exit(1)

height, width = frame.shape[:2]
print(f"Frame size: {width}x{height}")

# Create a copy to draw on
grid_frame = frame.copy()

# Grid color (bright cyan for visibility)
grid_color = (255, 255, 0)  # Cyan in BGR
text_color = (0, 255, 255)  # Yellow in BGR
grid_thickness = 2
text_thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6

# Draw vertical lines (x-axis)
x = 0.0
while x <= 1.0:
    x_pixel = int(x * width)
    cv2.line(grid_frame, (x_pixel, 0), (x_pixel, height), grid_color, grid_thickness)

    # Add label at top and bottom
    label = f"{x:.2f}"
    text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]

    # Top label
    cv2.putText(
        grid_frame,
        label,
        (x_pixel - text_size[0] // 2, 30),
        font,
        font_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA,
    )

    # Bottom label
    cv2.putText(
        grid_frame,
        label,
        (x_pixel - text_size[0] // 2, height - 10),
        font,
        font_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA,
    )

    x += GRID_SPACING

# Draw horizontal lines (y-axis)
y = 0.0
while y <= 1.0:
    y_pixel = int(y * height)
    cv2.line(grid_frame, (0, y_pixel), (width, y_pixel), grid_color, grid_thickness)

    # Add label on left and right
    label = f"{y:.2f}"
    text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]

    # Left label
    cv2.putText(
        grid_frame,
        label,
        (10, y_pixel + text_size[1] // 2),
        font,
        font_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA,
    )

    # Right label
    cv2.putText(
        grid_frame,
        label,
        (width - text_size[0] - 10, y_pixel + text_size[1] // 2),
        font,
        font_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA,
    )

    y += GRID_SPACING

# Highlight the edges with thicker lines
cv2.rectangle(grid_frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 4)

# Save the result
cv2.imwrite(OUTPUT_PATH, grid_frame)

print(f"\nGrid reference frame saved to: {OUTPUT_PATH}")
print(
    f"Grid spacing: {GRID_SPACING * 100}% ({int(GRID_SPACING * width)}px horizontal, {int(GRID_SPACING * height)}px vertical)"
)
print(f"\nUse this image to determine your ROI_POLYGON coordinates.")
print(f"Example: Point at x=0.20, y=0.40 means 20% from left, 40% from top")
