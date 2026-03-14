"""
Temporal Aggregation Engine
Deduplicates per-frame pothole detections into unique pothole events and
computes a video-level priority score.

Two detections are considered the SAME pothole if:
  • They occur within DEDUP_TIME_WINDOW seconds of each other, AND
  • Their bounding-box centres are within DEDUP_DIST_PX pixels of each other
    (works for slow-moving inspection vehicles at 1–3 FPS sampling).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


from modules.config import DEDUP_TIME_WINDOW, DEDUP_DIST_PX


@dataclass
class UniquePothole:
    """Tracks a single unique real-world pothole across frames."""
    uid:            int
    first_seen:     float          # seconds
    last_seen:      float          # seconds
    detections:     List[Dict] = field(default_factory=list)

    # Aggregated stats (updated after all frames processed)
    max_severity_score:  float = 0.0
    avg_severity_score:  float = 0.0
    dominant_level:      str   = "Low"
    max_risk_score:      float = 0.0
    avg_confidence:      float = 0.0
    best_roi_path:       str   = ""
    best_frame_path:     str   = ""
    best_det_idx:        int   = 0        # index of most-confident detection
    duration_secs:       float = 0.0      # visible window length


def _bbox_centre(det: Dict) -> Tuple[float, float]:
    return (
        (det["bbox_x1"] + det["bbox_x2"]) / 2.0,
        (det["bbox_y1"] + det["bbox_y2"]) / 2.0,
    )


def _centre_dist(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return math.sqrt(dx * dx + dy * dy)


def aggregate_detections(
    all_detections: List[Dict],
) -> Tuple[List[UniquePothole], Dict]:
    """
    Group per-frame detections into unique potholes and return a
    video-level summary dict.

    Parameters
    ----------
    all_detections : list of detection dicts sorted by timestamp ASC

    Returns
    -------
    (unique_potholes, summary_dict)
    """
    unique: List[UniquePothole] = []
    uid_counter = 0

    for det in all_detections:
        ts      = det["timestamp"]
        centre  = _bbox_centre(det)
        matched = False

        # Try to match against an existing unique pothole
        for up in reversed(unique):          # search recent ones first
            if ts - up.last_seen > DEDUP_TIME_WINDOW:
                continue                      # too old → can't be same one

            # Check spatial proximity using LAST detection in this unique pothole
            last_det    = up.detections[-1]
            last_centre = _bbox_centre(last_det)
            if _centre_dist(centre, last_centre) <= DEDUP_DIST_PX:
                up.detections.append(det)
                up.last_seen = ts
                matched = True
                break

        if not matched:
            uid_counter += 1
            up = UniquePothole(
                uid        = uid_counter,
                first_seen = ts,
                last_seen  = ts,
            )
            up.detections.append(det)
            unique.append(up)

    # Compute aggregated stats for each unique pothole
    for up in unique:
        dets = up.detections
        sev_scores   = [d["severity_score"] for d in dets]
        risk_scores  = [d["risk_score"]      for d in dets]
        confs        = [d["confidence"]      for d in dets]
        sev_levels   = [d["severity_level"]  for d in dets]

        up.max_severity_score = max(sev_scores)
        up.avg_severity_score = sum(sev_scores) / len(sev_scores)
        up.max_risk_score     = max(risk_scores)
        up.avg_confidence     = sum(confs) / len(confs)
        up.duration_secs      = up.last_seen - up.first_seen

        # Dominant level = most frequent
        from collections import Counter
        up.dominant_level = Counter(sev_levels).most_common(1)[0][0]

        # Best detection = highest risk score
        best_idx          = risk_scores.index(max(risk_scores))
        up.best_det_idx   = best_idx
        up.best_roi_path  = dets[best_idx].get("roi_image_path", "")
        up.best_frame_path= dets[best_idx].get("frame_image_path", "")

    # Summary
    if not unique:
        summary = {
            "unique_potholes":  0,
            "low_potholes":     0,
            "medium_potholes":  0,
            "high_potholes":    0,
        }
    else:
        from collections import Counter
        level_counts = Counter(up.dominant_level for up in unique)
        summary = {
            "unique_potholes":  len(unique),
            "low_potholes":     level_counts.get("Low",    0),
            "medium_potholes":  level_counts.get("Medium", 0),
            "high_potholes":    level_counts.get("High",   0),
        }

    return unique, summary


def compute_video_priority_score(
    unique_potholes: List[UniquePothole],
    video_duration:  float,
) -> Dict:
    """
    Compute overall video priority score (0-100) using Pareto rank-weighting
    and Gini coefficient concentration bonus.

    Theory
    ──────
    Pareto rank-weighting  (UK STATS19 / Highway Asset Management)
      The worst 20% of potholes cause ~80% of vehicle damage (Pareto 80/20).
      Rank-1 (worst) gets weight n, rank-2 gets n-1, …, rank-n gets 1.
      This naturally down-weights roads with many low-severity potholes
      while heavily penalising a single high-risk one.

      pareto_score = Σ w_i × r_i  /  Σ w_i
      where w_i = (n − i + 1), r_i = risk scores sorted descending (0-100)

    Gini coefficient  (EVT / concentration of damage, range [0, 1])
      A road with all potholes at the same severity has Gini ≈ 0.
      A road with one extreme pothole (Critical) and the rest Low has Gini → 1.
      High concentration = higher accident risk than mean risk suggests.

      Gini = (2 × Σ i × r_i_sorted_asc) / (n × Σ r_i)  −  (n + 1) / n

    Final score
      score = pareto_score × (1  +  VID_GINI_GAMMA × Gini)   clamped [0, 100]
      VID_GINI_GAMMA = 0.20 → up to 20% bonus when damage is maximally concentrated
    """
    from modules.config import RISK_COLORS, VID_GINI_GAMMA, RISK_LOW_MAX, RISK_MEDIUM_MAX, RISK_HIGH_MAX

    if not unique_potholes:
        return {
            "priority_score": 0.0,
            "risk_level":     "Low",
            "risk_color":     RISK_COLORS["Low"],
        }

    risk_scores = [up.max_risk_score for up in unique_potholes]
    n = len(risk_scores)

    # ── Pareto rank-weighted score ────────────────────────────────────────────
    sorted_desc = sorted(risk_scores, reverse=True)     # best→worst rank
    weights     = list(range(n, 0, -1))                 # n, n-1, …, 1
    w_sum       = sum(weights)
    pareto_score = sum(w * r for w, r in zip(weights, sorted_desc)) / w_sum

    # ── Gini coefficient ──────────────────────────────────────────────────────
    sorted_asc  = sorted(risk_scores)                   # ascending for Gini formula
    total_risk  = sum(sorted_asc)
    if total_risk == 0:
        gini = 0.0
    else:
        gini = (
            2.0 * sum((i + 1) * v for i, v in enumerate(sorted_asc))
            / (n * total_risk)
        ) - (n + 1) / n
    gini = max(0.0, min(1.0, gini))   # clamp numerically

    # ── Final score (Pareto × Gini concentration bonus) ───────────────────────
    score = pareto_score * (1.0 + VID_GINI_GAMMA * gini)
    score = round(min(100.0, max(0.0, score)), 2)

    if score < RISK_LOW_MAX:
        level = "Low"
    elif score < RISK_MEDIUM_MAX:
        level = "Medium"
    elif score < RISK_HIGH_MAX:
        level = "High"
    else:
        level = "Critical"

    return {
        "priority_score": score,
        "gini":           round(gini, 4),
        "pareto_score":   round(pareto_score, 2),
        "risk_level":     level,
        "risk_color":     RISK_COLORS[level],
    }
