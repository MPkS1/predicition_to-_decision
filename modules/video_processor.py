"""
Main Video Analysis Pipeline
Orchestrates Steps 1–8:
  1. Open video (OpenCV)
  2. Adaptive frame extraction (1 FPS base, 3 FPS burst when potholes detected)
  3. YOLO inference
  4. ROI extraction (Step 5)
  5. Severity estimation (Step 6)
  6. Degradation prediction (Step 7)
  7. Risk prioritisation (Step 8)
  8. Persist results to SQLite
  9. Run temporal aggregation + compute video-level priority score

Yields progress dicts so the Streamlit UI can display a live progress bar.
"""
from __future__ import annotations

import cv2
import heapq
import numpy as np
from pathlib import Path
from typing import Generator, Dict, Any, List

from modules.config import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, IMGSZ,
    BASE_FPS, ADAPTIVE_FPS, ADAPTIVE_SECS,
    SEVERITY_COLORS,
)
from modules.roi_extractor        import extract_roi, save_roi, draw_detections, save_frame
from modules.severity_estimator   import estimate_severity
from modules.degradation_predictor import predict_degradation
from modules.risk_prioritizer     import calculate_pothole_risk
from modules.temporal_aggregator  import aggregate_detections, compute_video_priority_score
from modules.database import (
    add_detection, add_frame_result, update_video_analysis,
    get_video_detections,
)


# ─── Model loader (cached at module level so it loads once per process) ──────
_MODEL = None

def _load_model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(MODEL_PATH))
    return _MODEL


def _build_frame_index(total_frames: int, native_fps: float) -> List[int]:
    """
    Build the list of frame indices to sample at BASE_FPS.
    Adaptive burst frames are inserted dynamically during processing.
    """
    if native_fps <= 0:
        return list(range(0, total_frames, 1))
    step = max(1, int(round(native_fps / BASE_FPS)))
    return list(range(0, total_frames, step))


def analyze_video(
    video_path: str,
    video_id:   int,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator that processes a video file frame by frame.

    Yields progress dicts:
        { 'type': 'progress', 'done': int, 'total': int,
          'frame_ts': float, 'new_detections': int, 'current_frame_img': ndarray|None }
        { 'type': 'complete', 'video_summary': dict }
        { 'type': 'error',    'message': str }
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        yield {"type": "error", "message": f"Cannot open video: {video_path}"}
        return

    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / native_fps

    model = _load_model()

    # ── Build frame heap (min-heap so frames always process in ascending order) ─
    base_indices = set(_build_frame_index(total_frames, native_fps))
    seen_frames: set = set(base_indices)

    frame_heap: List[int] = list(base_indices)
    heapq.heapify(frame_heap)

    all_detections_db: List[Dict] = []   # flat list for aggregation after loop

    processed        = 0
    total_to_process = len(frame_heap)   # grows as burst frames are added
    det_idx_counter  = 0

    while frame_heap:
        frame_pos = heapq.heappop(frame_heap)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            processed += 1
            continue

        timestamp = frame_pos / native_fps

        # ── YOLO inference ────────────────────────────────────────────────────
        results = model.predict(
            source    = frame,
            conf      = CONFIDENCE_THRESHOLD,
            iou       = IOU_THRESHOLD,
            imgsz     = IMGSZ,
            verbose   = False,
        )

        frame_dets: List[Dict] = []

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            for bi in range(len(boxes)):
                # Raw coords
                xyxy = boxes.xyxy[bi].cpu().numpy().tolist()
                x1, y1, x2, y2 = xyxy
                conf = float(boxes.conf[bi].cpu().numpy())

                bbox = (x1, y1, x2, y2)

                # ── Step 5: ROI ───────────────────────────────────────────────
                roi = extract_roi(frame, bbox)

                # ── Step 6: Severity ──────────────────────────────────────────
                sev = estimate_severity(roi, conf, frame, bbox)

                # ── Step 7: Degradation ───────────────────────────────────────
                deg = predict_degradation(sev.score, sev.level)

                # ── Step 8: Risk ──────────────────────────────────────────────
                risk = calculate_pothole_risk(
                    severity_score   = sev.score,
                    degradation_rate = deg.degradation_rate,
                    area_ratio       = sev.area_ratio,
                )

                # Save ROI image
                roi_path = save_roi(roi, video_id, frame_pos, det_idx_counter)
                det_idx_counter += 1

                det_record: Dict = {
                    "frame_number":       frame_pos,
                    "timestamp":          round(timestamp, 3),
                    "bbox_x1":            round(x1, 1),
                    "bbox_y1":            round(y1, 1),
                    "bbox_x2":            round(x2, 1),
                    "bbox_y2":            round(y2, 1),
                    "confidence":         round(conf, 4),
                    "area_ratio":         round(sev.area_ratio,    5),
                    "edge_density":       round(sev.edge_density,  4),
                    "darkness_ratio":     round(sev.darkness_ratio,4),
                    "severity_score":     round(sev.score,         4),
                    "severity_level":     sev.level,
                    "pci_score":          sev.pci_score,
                    "risk_score":         round(risk.risk_score,   2),
                    "degradation_rate":   deg.degradation_rate,
                    "predicted_score_30d":deg.sigmoid_scores[0],
                    "predicted_score_60d":deg.sigmoid_scores[1],
                    "predicted_score_90d":deg.sigmoid_scores[2],
                    "predicted_level_30d":deg.sigmoid_levels[0],
                    "predicted_level_60d":deg.sigmoid_levels[1],
                    "predicted_level_90d":deg.sigmoid_levels[2],
                    "roi_image_path":     roi_path,
                }

                # Persist detection
                add_detection(video_id, det_record)
                det_record["id"] = det_idx_counter  # local id for aggregation
                frame_dets.append(det_record)
                all_detections_db.append(det_record)

            # ── Adaptive burst: sample denser around detected potholes ─────────
            burst_end_frame = min(int(frame_pos + ADAPTIVE_SECS * native_fps), total_frames)
            burst_step      = max(1, int(native_fps / ADAPTIVE_FPS))
            for bf in range(frame_pos + 1, burst_end_frame, burst_step):
                if bf not in seen_frames:
                    seen_frames.add(bf)
                    heapq.heappush(frame_heap, bf)
                    total_to_process += 1

        # ── Annotate & save frame (only frames with detections) ───────────────
        frame_img_path = ""
        if frame_dets:
            annotated = draw_detections(frame, frame_dets, SEVERITY_COLORS)
            frame_img_path = save_frame(annotated, video_id, frame_pos)
            # Attach frame path to detections for aggregation
            for d in frame_dets:
                d["frame_image_path"] = frame_img_path

        # ── Persist frame result ──────────────────────────────────────────────
        frame_record = {
            "frame_number":   frame_pos,
            "timestamp":      round(timestamp, 3),
            "potholes_count": len(frame_dets),
            "low_count":      sum(1 for d in frame_dets if d["severity_level"] == "Low"),
            "medium_count":   sum(1 for d in frame_dets if d["severity_level"] == "Medium"),
            "high_count":     sum(1 for d in frame_dets if d["severity_level"] == "High"),
            "frame_image_path": frame_img_path,
        }
        add_frame_result(video_id, frame_record)

        # Yield progress (include a small preview frame for live display)
        preview = None
        if frame_dets:
            sm = cv2.resize(annotated if frame_dets else frame, (320, 180))
            preview = cv2.cvtColor(sm, cv2.COLOR_BGR2RGB)

        processed += 1
        yield {
            "type":               "progress",
            "done":               processed,
            "total":              max(total_to_process, processed),
            "frame_ts":           timestamp,
            "new_detections":     len(frame_dets),
            "preview_frame":      preview,
        }

    cap.release()

    # ── Temporal aggregation ──────────────────────────────────────────────────
    unique_potholes, agg_summary = aggregate_detections(all_detections_db)
    priority = compute_video_priority_score(unique_potholes, duration)

    # ── Update video record in DB ─────────────────────────────────────────────
    low_cnt = agg_summary["low_potholes"]
    med_cnt = agg_summary["medium_potholes"]
    hi_cnt  = agg_summary["high_potholes"]
    total_d = len(all_detections_db)

    update_video_analysis(video_id, {
        "duration":         round(duration,  2),
        "fps":              round(native_fps, 2),
        "total_frames":     total_frames,
        "analyzed_frames":  processed,
        "video_risk_score": priority["priority_score"],
        "risk_level":       priority["risk_level"],
        "total_potholes":   total_d,
        "unique_potholes":  agg_summary["unique_potholes"],
        "low_potholes":     low_cnt,
        "medium_potholes":  med_cnt,
        "high_potholes":    hi_cnt,
    })

    video_summary = {
        "video_id":           video_id,
        "duration":           duration,
        "fps":                native_fps,
        "analyzed_frames":    processed,
        "total_potholes":     total_d,
        "unique_potholes":    agg_summary["unique_potholes"],
        "low_potholes":       low_cnt,
        "medium_potholes":    med_cnt,
        "high_potholes":      hi_cnt,
        "video_risk_score":   priority["priority_score"],
        "risk_level":         priority["risk_level"],
        "unique_pothole_list": unique_potholes,
    }

    yield {"type": "complete", "video_summary": video_summary}
