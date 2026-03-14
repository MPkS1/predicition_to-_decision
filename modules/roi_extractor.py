"""
Step 5 — ROI Extraction
Crops the pothole region from a frame with configurable padding,
saves the image to disk, and returns both the array and file path.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from modules.config import ROI_DIR

# Padding fraction around each bounding box
ROI_PAD = 0.12


def extract_roi(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = ROI_PAD,
) -> Optional[np.ndarray]:
    """
    Return the padded ROI crop from *frame* for the given *bbox*.

    Parameters
    ----------
    frame   : BGR image (H x W x 3)
    bbox    : (x1, y1, x2, y2) pixel coordinates
    padding : fractional padding relative to bbox dimensions

    Returns
    -------
    BGR numpy array or None if the crop is degenerate.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(w, x2 + pad_x)
    ny2 = min(h, y2 + pad_y)

    if nx2 <= nx1 or ny2 <= ny1:
        return None

    return frame[ny1:ny2, nx1:nx2].copy()


def save_roi(
    roi: np.ndarray,
    video_id: int,
    frame_number: int,
    detection_idx: int,
) -> str:
    """
    Save the ROI image to disk and return its relative path string.

    Naming convention: roi_{video_id}_{frame_number}_{detection_idx}.jpg
    """
    if roi is None or roi.size == 0:
        return ""

    filename = f"roi_{video_id}_{frame_number}_{detection_idx}.jpg"
    save_path = ROI_DIR / filename
    cv2.imwrite(str(save_path), roi, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(save_path)


def draw_detections(
    frame: np.ndarray,
    detections: list,
    severity_colors: dict,
) -> np.ndarray:
    """
    Draw bounding boxes and severity labels on a copy of *frame*.

    Parameters
    ----------
    detections : list of dicts with keys bbox, severity_level, confidence, risk_score
    severity_colors : mapping of level → BGR tuple
    """
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (
            int(det["bbox_x1"]), int(det["bbox_y1"]),
            int(det["bbox_x2"]), int(det["bbox_y2"]),
        )
        level = det.get("severity_level", "Low")
        conf  = det.get("confidence", 0.0)
        risk  = det.get("risk_score", 0.0)

        # Colour in BGR
        color_hex = severity_colors.get(level, "#2ECC71").lstrip("#")
        r, g, b = (int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        color_bgr = (b, g, r)

        # Bounding box
        thickness = 3 if level == "High" else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, thickness)

        # Label background
        label = f"{level} | {conf:.0%} | Risk:{risk:.0f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - baseline - 6), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(
            annotated, label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated


def save_frame(
    annotated_frame: np.ndarray,
    video_id: int,
    frame_number: int,
) -> str:
    """Save an annotated frame and return its path string."""
    from modules.config import FRAMES_DIR
    filename = f"frame_{video_id}_{frame_number}.jpg"
    path = FRAMES_DIR / filename
    cv2.imwrite(str(path), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(path)
