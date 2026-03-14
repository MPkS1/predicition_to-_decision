"""
Step 7 — Degradation Prediction  (Logistic Sigmoid + Markov Chain)
==================================================================
Literature basis:
  · HDM-4 World Bank (2000) — Highway Development & Management model
  · AASHTO Pavement Design Guide — deterioration curves
  · FAA Advisory Circular AC 150/5370-11 — Markov-based PMS
  · Butt et al. 1987 — Markov chain pavement deterioration

Two parallel forecasting methods:

1. LOGISTIC SIGMOID  (deterministic, physics-based)
   ─────────────────
   s(t) = 1 / (1 + ((1−s₀)/s₀) × exp(−k × CF × t))

   Why better than exponential:
   • Exponential is unbounded — predicts s > 1.0 at large t (physically impossible)
   • Sigmoid saturates at 1.0, models the real S-shaped deterioration curve:
     slow start (surface cracking) → accelerating phase (water/frost infiltration)
     → saturation (complete structural failure, can't get "worse than failed")
   • CF = climate factor: 1.4 = freeze-thaw climates, 1.2 = tropical, 1.0 = default

2. MARKOV CHAIN  (probabilistic, state-based)
   ──────────────
   4 states: Low(0), Medium(1), High(2), Critical(3)
   30-day transition matrix P calibrated from AASHTO:

         Low   Med   High  Crit
   Low  [0.70  0.25  0.04  0.01]   stays Low 70% of the time
   Med  [0.00  0.65  0.28  0.07]   NO recovery (asymmetric — roads don't self-heal)
   High [0.00  0.00  0.60  0.40]   40% chance Critical in 30 days if already High
   Crit [0.00  0.00  0.00  1.00]   Critical is absorbing (requires repair to exit)

   P60 = P30 @ P30  |  P90 = P30 @ P30 @ P30
   Expected severity = dot(state_dist, [0.175, 0.50, 0.75, 0.925])

PCI = 100 × (1 − score)^1.7   [ASTM D6433 nonlinear, exponent for potholes]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

from modules.config import (
    DEGRADATION_RATES, CLIMATE_FACTOR, FORECAST_DAYS,
    MARKOV_P30, MARKOV_STATE_MIDPOINTS, MARKOV_STATE_LABELS,
    SEV_LOW_MAX, SEV_MEDIUM_MAX, PCI_EXPONENT,
)

# Precompute P60 and P90 once at module load
_P30 = np.array(MARKOV_P30, dtype=np.float64)
_P60 = _P30 @ _P30
_P90 = _P30 @ _P30 @ _P30
_MIDPOINTS = np.array(MARKOV_STATE_MIDPOINTS)


@dataclass
class DegradationResult:
    # ── Current state ──────────────────────────────────────────────────────
    current_score:    float
    current_level:    str
    pci_current:      float
    degradation_rate: float
    climate_factor:   float

    # ── Sigmoid forecasts ──────────────────────────────────────────────────
    sigmoid_scores:   List[float]   # s at [30, 60, 90] days
    sigmoid_levels:   List[str]
    sigmoid_pci:      List[float]

    # ── Markov forecasts ───────────────────────────────────────────────────
    markov_state_probs: List[List[float]]  # 3 × 4 matrix (horizons × states)
    markov_exp_scores:   List[float]       # expected severity at each horizon
    markov_levels:       List[str]         # dominant state at each horizon
    markov_pci:          List[float]

    # ── Days ───────────────────────────────────────────────────────────────
    forecast_days:   List[int]

    # ── Maintenance ────────────────────────────────────────────────────────
    urgency:          str
    recommendation:   str
    time_to_critical: float   # days until Markov P(Critical) > 0.5; -1 if never


_URGENCY = {
    "Low":      "Monitor (routine inspection in 90 days)",
    "Medium":   "Plan Repair (within 30 days)",
    "High":     "Urgent Repair (within 7 days)",
    "Critical": "EMERGENCY — repair immediately",
}

_RECOMMENDATIONS = {
    "Low": (
        "• Apply preventive crack sealer to halt water ingress\n"
        "• Schedule monthly visual monitoring\n"
        "• Flag for routine maintenance cycle in 90 days\n"
        "• Markov model: 30% probability of worsening within 30 days"
    ),
    "Medium": (
        "• Install road hazard warning signs immediately\n"
        "• Schedule hot-mix asphalt throw-and-roll patch within 30 days\n"
        "• Inspect 50 m radius for related pavement distress\n"
        "• Sigmoid model: severity will reach High within ~60 days if untreated\n"
        "• Re-evaluate after patching — full-depth repair may be needed"
    ),
    "High": (
        "• Place emergency road hazard markers NOW\n"
        "• Full-depth reclamation or mill-and-fill repair within 7 days\n"
        "• Markov model: 40% probability of Critical failure within 30 days\n"
        "• Assess structural base integrity before repaving\n"
        "• Document with photographs for road authority reporting\n"
        "• Consider temporary speed restriction or lane closure"
    ),
}


def _score_to_level(score: float) -> str:
    if score < SEV_LOW_MAX:    return "Low"
    if score < SEV_MEDIUM_MAX: return "Medium"
    if score < 0.85:           return "High"
    return "Critical"


def _pci(score: float) -> float:
    return round(max(0.0, 100.0 * ((1.0 - score) ** PCI_EXPONENT)), 1)


def _sigmoid(s0: float, k: float, cf: float, t: float) -> float:
    """Logistic sigmoid growth: s(t) = 1/[1 + ((1-s0)/s0)·exp(-k·cf·t)]"""
    if s0 <= 0.0: return 0.0
    if s0 >= 1.0: return 1.0
    ratio = (1.0 - s0) / s0
    return min(1.0, 1.0 / (1.0 + ratio * math.exp(-k * cf * t)))


def _score_to_initial_state(score: float) -> np.ndarray:
    """Convert a continuous severity score to a Markov initial state vector."""
    v = np.zeros(4)
    if score < SEV_LOW_MAX:
        v[0] = 1.0
    elif score < SEV_MEDIUM_MAX:
        # Interpolate between Low and Medium states
        frac = (score - SEV_LOW_MAX) / (SEV_MEDIUM_MAX - SEV_LOW_MAX)
        v[0] = 1.0 - frac
        v[1] = frac
    elif score < 0.85:
        frac = (score - SEV_MEDIUM_MAX) / (0.85 - SEV_MEDIUM_MAX)
        v[1] = 1.0 - frac
        v[2] = frac
    else:
        frac = (score - 0.85) / (1.0 - 0.85)
        v[2] = 1.0 - min(frac, 1.0)
        v[3] = min(frac, 1.0)
    return v


def _time_to_critical(v0: np.ndarray, threshold: float = 0.5) -> float:
    """Estimate days until P(Critical) exceeds threshold via discrete stepping."""
    v = v0.copy()
    step_days = 5
    P5 = np.linalg.matrix_power(_P30, 0)
    # Use continuous matrix power approximation via eigen-decomposition not needed;
    # just step month by month
    for month in range(1, 49):  # up to 4 years
        v = v @ _P30
        if v[3] >= threshold:
            return float(month * 30)
    return -1.0   # never reaches threshold within 4 years


def predict_degradation(severity_score: float, severity_level: str) -> DegradationResult:
    """
    Predict pothole degradation via two parallel methods.

    Parameters
    ----------
    severity_score  : current AHP composite score in [0, 1]
    severity_level  : 'Low' | 'Medium' | 'High'
    """
    k  = DEGRADATION_RATES.get(severity_level, DEGRADATION_RATES["Medium"])
    cf = CLIMATE_FACTOR

    # ── Sigmoid forecasts ──────────────────────────────────────────────────────
    sig_scores = []
    sig_levels = []
    sig_pci    = []
    for days in FORECAST_DAYS:
        s = round(_sigmoid(severity_score, k, cf, days), 4)
        sig_scores.append(s)
        sig_levels.append(_score_to_level(s))
        sig_pci.append(_pci(s))

    # ── Markov forecasts ───────────────────────────────────────────────────────
    v0 = _score_to_initial_state(severity_score)
    P_matrices = [_P30, _P60, _P90]

    markov_probs  = []
    markov_scores = []
    markov_levels = []
    markov_pci    = []

    for P in P_matrices:
        vt      = v0 @ P
        exp_s   = float(np.dot(vt, _MIDPOINTS))
        dom_idx = int(np.argmax(vt))
        markov_probs.append([round(float(p), 4) for p in vt])
        markov_scores.append(round(exp_s, 4))
        markov_levels.append(MARKOV_STATE_LABELS[dom_idx])
        markov_pci.append(_pci(exp_s))

    ttc = _time_to_critical(v0)

    return DegradationResult(
        current_score       = severity_score,
        current_level       = severity_level,
        pci_current         = _pci(severity_score),
        degradation_rate    = k,
        climate_factor      = cf,
        sigmoid_scores      = sig_scores,
        sigmoid_levels      = sig_levels,
        sigmoid_pci         = sig_pci,
        markov_state_probs  = markov_probs,
        markov_exp_scores   = markov_scores,
        markov_levels       = markov_levels,
        markov_pci          = markov_pci,
        forecast_days       = list(FORECAST_DAYS),
        urgency             = _URGENCY.get(severity_level, _URGENCY["Low"]),
        recommendation      = _RECOMMENDATIONS.get(severity_level, _RECOMMENDATIONS["Low"]),
        time_to_critical    = ttc,
    )

