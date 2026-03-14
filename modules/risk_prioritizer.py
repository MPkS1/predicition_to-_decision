"""
Step 8 — Risk Prioritisation
Computes a 0-100 risk score for each pothole detection and
an aggregate video-level road risk score.

Individual risk = 0.571 × severity  +  0.286 × degradation urgency  +  0.143 × size
  Weights from AHP pairwise matrix CR=0 (Saaty 1977, Moazami 2011)

Video risk = Pareto rank-weighted score × (1 + 0.20 × Gini coefficient)
  Reflects: worst 20% of potholes cause 80% of accidents (UK STATS19)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

from modules.config import (
    RISK_SEV_WEIGHT, RISK_DEG_WEIGHT, RISK_SIZE_WEIGHT,
    RISK_SIZE_SAT, RISK_LOW_MAX, RISK_MEDIUM_MAX, RISK_HIGH_MAX,
    DEGRADATION_RATES,
    VID_GINI_GAMMA,
    RISK_COLORS,
)


@dataclass
class RiskResult:
    risk_score:              float
    risk_level:              str     # Low | Medium | High | Critical
    risk_color:              str
    severity_contribution:   float   # component score (out of 50)
    degradation_contribution:float   # component score (out of 30)
    size_contribution:       float   # component score (out of 20)
    action_required:         str


_ACTIONS = {
    "Low":      "Schedule routine maintenance (>90 days)",
    "Medium":   "Plan repair within 30 days",
    "High":     "Urgent repair within 7 days",
    "Critical": "Emergency repair IMMEDIATELY",
}

_MAX_DEG_RATE = max(DEGRADATION_RATES.values())   # 0.010


def _score_to_level(score: float) -> str:
    if score < RISK_LOW_MAX:
        return "Low"
    if score < RISK_MEDIUM_MAX:
        return "Medium"
    if score < RISK_HIGH_MAX:
        return "High"
    return "Critical"


def calculate_pothole_risk(
    severity_score:   float,
    degradation_rate: float,
    area_ratio:       float,
) -> RiskResult:
    """
    Compute a 0-100 risk score for a single pothole detection.
    """
    sev_cont  = severity_score  * (RISK_SEV_WEIGHT  * 100)
    deg_cont  = min(degradation_rate / _MAX_DEG_RATE, 1.0) * (RISK_DEG_WEIGHT * 100)
    size_cont = min(area_ratio / RISK_SIZE_SAT, 1.0)       * (RISK_SIZE_WEIGHT * 100)

    risk_score = round(min(100.0, sev_cont + deg_cont + size_cont), 2)
    level      = _score_to_level(risk_score)

    return RiskResult(
        risk_score               = risk_score,
        risk_level               = level,
        risk_color               = RISK_COLORS[level],
        severity_contribution    = round(sev_cont,  2),
        degradation_contribution = round(deg_cont,  2),
        size_contribution        = round(size_cont, 2),
        action_required          = _ACTIONS[level],
    )


def calculate_video_risk(detections: List[Dict]) -> Dict:
    """
    Aggregate individual detection risks into a single video road-risk score.

    Parameters
    ----------
    detections : list of detection dicts from the database (require 'risk_score',
                 'severity_level').

    Returns
    -------
    Dict with video_risk_score, risk_level, risk_color, and summary counts.
    """
    if not detections:
        return {
            "video_risk_score": 0.0,
            "risk_level":       "Low",
            "risk_color":       RISK_COLORS["Low"],
            "total_potholes":   0,
            "high_potholes":    0,
            "avg_risk_score":   0.0,
            "max_risk_score":   0.0,
        }

    risk_scores = [d["risk_score"] for d in detections]
    n           = len(risk_scores)
    total       = n
    high_cnt    = sum(1 for d in detections if d["severity_level"] == "High")
    avg_risk    = sum(risk_scores) / n
    max_risk    = max(risk_scores)

    # Pareto rank-weighted mean
    sorted_desc  = sorted(risk_scores, reverse=True)
    weights      = list(range(n, 0, -1))
    pareto_score = sum(w * r for w, r in zip(weights, sorted_desc)) / sum(weights)

    # Gini concentration coefficient
    sorted_asc = sorted(risk_scores)
    total_risk = sum(sorted_asc)
    if total_risk == 0:
        gini = 0.0
    else:
        gini = (
            2.0 * sum((i + 1) * v for i, v in enumerate(sorted_asc))
            / (n * total_risk)
        ) - (n + 1) / n
    gini = max(0.0, min(1.0, gini))

    video_risk = round(min(100.0, max(0.0,
        pareto_score * (1.0 + VID_GINI_GAMMA * gini)
    )), 2)
    level = _score_to_level(video_risk)

    return {
        "video_risk_score": video_risk,
        "risk_level":       level,
        "risk_color":       RISK_COLORS[level],
        "total_potholes":   total,
        "high_potholes":    high_cnt,
        "avg_risk_score":   round(avg_risk, 2),
        "max_risk_score":   round(max_risk, 2),
    }
