"""
Visualiser — all Plotly charts used in the Streamlit dashboard.
Every function returns a plotly Figure object ready for st.plotly_chart().
"""
from __future__ import annotations

import math
from typing import List, Dict, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from modules.config import SEVERITY_COLORS, RISK_COLORS, FORECAST_DAYS


# ─── Palette helpers ──────────────────────────────────────────────────────────

_SEV_ORDER  = ["Low", "Medium", "High"]
_RISK_ORDER = ["Low", "Medium", "High", "Critical"]

_PIE_COLORS = {
    "Low":      "#2ECC71",
    "Medium":   "#F39C12",
    "High":     "#E74C3C",
    "Critical": "#8B0000",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Severity Distribution  (donut)
# ═══════════════════════════════════════════════════════════════════════════════

def create_severity_donut(
    low: int, medium: int, high: int,
    title: str = "Pothole Severity Distribution",
) -> go.Figure:
    labels  = ["Low",         "Medium",     "High"]
    values  = [low,           medium,       high]
    colors  = [_PIE_COLORS["Low"], _PIE_COLORS["Medium"], _PIE_COLORS["High"]]

    # Filter out zero-count slices for cleanliness
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not filtered:
        filtered = [("No Detections", 1, "#cccccc")]

    fl, fv, fc = zip(*filtered)

    fig = go.Figure(go.Pie(
        labels       = fl,
        values       = fv,
        hole         = 0.55,
        marker       = dict(colors=list(fc), line=dict(color="#ffffff", width=2)),
        textinfo     = "label+percent+value",
        textfont     = dict(size=13),
        hovertemplate = "<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title       = dict(text=title, x=0.5, font=dict(size=16)),
        showlegend  = True,
        legend      = dict(orientation="h", y=-0.15),
        margin      = dict(t=60, b=40, l=20, r=20),
        height      = 340,
        plot_bgcolor= "rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Detection Timeline  (area chart over video duration)
# ═══════════════════════════════════════════════════════════════════════════════

def create_timeline_chart(
    frame_results: List[Dict],
    title: str = "Potholes Detected Over Video Timeline",
) -> go.Figure:
    if not frame_results:
        fig = go.Figure()
        fig.update_layout(title=title, height=280)
        return fig

    ts   = [r["timestamp"]      for r in frame_results]
    tot  = [r["potholes_count"] for r in frame_results]
    low  = [r["low_count"]      for r in frame_results]
    med  = [r["medium_count"]   for r in frame_results]
    hi   = [r["high_count"]     for r in frame_results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=hi,  name="High",   mode="lines", fill="tozeroy",
        line=dict(color=_PIE_COLORS["High"],   width=1),
        fillcolor="rgba(231,76,60,0.25)",
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=med, name="Medium", mode="lines", fill="tozeroy",
        line=dict(color=_PIE_COLORS["Medium"], width=1),
        fillcolor="rgba(243,156,18,0.20)",
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=low, name="Low",    mode="lines", fill="tozeroy",
        line=dict(color=_PIE_COLORS["Low"],    width=1),
        fillcolor="rgba(46,204,113,0.20)",
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=tot, name="Total",  mode="lines+markers",
        line=dict(color="#3498DB", width=2, dash="dot"),
        marker=dict(size=5),
    ))

    fig.update_layout(
        title        = dict(text=title, x=0.5),
        xaxis_title  = "Time (seconds)",
        yaxis_title  = "Potholes",
        height       = 300,
        margin       = dict(t=50, b=40, l=50, r=20),
        legend       = dict(orientation="h", y=1.12),
        plot_bgcolor = "rgba(248,249,250,1)",
        paper_bgcolor= "rgba(0,0,0,0)",
        hovermode    = "x unified",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Risk Score Gauge  (single value 0-100)
# ═══════════════════════════════════════════════════════════════════════════════

def create_risk_gauge(
    score: float,
    level: str,
    title: str = "Road Risk Score",
) -> go.Figure:
    color = _PIE_COLORS.get(level, "#cccccc")

    fig = go.Figure(go.Indicator(
        mode   = "gauge+number+delta",
        value  = round(score, 1),
        title  = {"text": title, "font": {"size": 16}},
        delta  = {"reference": 50, "increasing": {"color": "#E74C3C"},
                  "decreasing": {"color": "#2ECC71"}},
        gauge  = {
            "axis":     {"range": [0, 100], "tickwidth": 1, "tickcolor": "#666"},
            "bar":      {"color": color, "thickness": 0.25},
            "bgcolor":  "white",
            "borderwidth": 2,
            "bordercolor": "#ccc",
            "steps": [
                {"range": [0,  25], "color": "rgba(46,204,113,0.15)"},
                {"range": [25, 50], "color": "rgba(243,156,18,0.15)"},
                {"range": [50, 75], "color": "rgba(231,76,60,0.15)"},
                {"range": [75,100], "color": "rgba(139,0,0,0.15)"},
            ],
            "threshold": {
                "line":      {"color": "#1a1a2e", "width": 4},
                "thickness": 0.75,
                "value":     score,
            },
        },
        number = {"suffix": "/100", "font": {"size": 28, "color": color}},
    ))
    # Level annotation
    fig.add_annotation(
        x=0.5, y=0.20, xref="paper", yref="paper",
        text=f"<b>{level} RISK</b>",
        font=dict(size=18, color=color),
        showarrow=False,
    )
    fig.update_layout(
        height        = 280,
        margin        = dict(t=40, b=20, l=40, r=40),
        paper_bgcolor = "rgba(0,0,0,0)",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Degradation Prediction Chart  — smooth sigmoid curves (0 → +90 days)
# ═══════════════════════════════════════════════════════════════════════════════

def _sigmoid_curve(s0: float, k: float, days: List[float]) -> List[float]:
    """Logistic sigmoid: s(t) = 1 / (1 + ((1-s0)/s0) × exp(-k × t))"""
    if s0 <= 0: return [0.0] * len(days)
    if s0 >= 1: return [1.0] * len(days)
    ratio = (1.0 - s0) / s0
    import math
    return [min(1.0, 1.0 / (1.0 + ratio * math.exp(-k * t))) for t in days]


def create_degradation_chart(
    detections: List[Dict],
    max_lines: int = 8,
    title: str = "Pothole Degradation Forecast — Logistic Sigmoid (Current → +90 days)",
) -> go.Figure:
    """
    Smooth sigmoid degradation curves for the top-risk potholes.
    Uses the HDM-4 logistic model: s(t) = 1/(1+((1-s0)/s0)·exp(-k·t))
    Sampled every 5 days for a continuous-looking curve (vs. 4 discrete scatter points).
    """
    if not detections:
        fig = go.Figure()
        fig.update_layout(title=title, height=350)
        return fig

    from modules.config import DEGRADATION_RATES

    sorted_dets = sorted(detections, key=lambda d: d["risk_score"], reverse=True)[:max_lines]
    day_samples = list(range(0, 91, 5))   # 0, 5, 10, … 90

    fig = go.Figure()

    for i, det in enumerate(sorted_dets):
        s0    = det["severity_score"]
        k     = DEGRADATION_RATES.get(det["severity_level"], 0.015)
        ts    = det["timestamp"]
        lvl   = det["severity_level"]
        risk  = det["risk_score"]
        color = _PIE_COLORS.get(lvl, "#3498DB")

        y_vals = _sigmoid_curve(s0, k, day_samples)

        fig.add_trace(go.Scatter(
            x    = day_samples,
            y    = y_vals,
            name = f"P{i+1} @{ts:.0f}s [{lvl}] Risk:{risk:.0f}",
            mode = "lines",
            line = dict(color=color, width=2.5,
                        dash="dot" if lvl == "Low" else "solid"),
            hovertemplate=(
                f"<b>Pothole {i+1}</b> (t={ts:.0f}s, {lvl})<br>"
                "Day %{x}: Severity %{y:.3f}<extra></extra>"
            ),
        ))
        # Mark current point
        fig.add_trace(go.Scatter(
            x=[0], y=[s0],
            mode="markers",
            marker=dict(color=color, size=10, symbol="circle"),
            showlegend=False,
            hovertemplate=f"<b>Current</b>: {s0:.3f}<extra></extra>",
        ))

    # Threshold bands
    fig.add_hrect(y0=0.65, y1=1.05, fillcolor="rgba(231,76,60,0.06)",
                  layer="below", line_width=0)
    fig.add_hrect(y0=0.35, y1=0.65, fillcolor="rgba(243,156,18,0.06)",
                  layer="below", line_width=0)
    fig.add_hline(y=0.35, line_dash="dash", line_color="#F39C12",
                  annotation_text="Medium threshold", annotation_position="right")
    fig.add_hline(y=0.65, line_dash="dash", line_color="#E74C3C",
                  annotation_text="High threshold",   annotation_position="right")

    fig.update_layout(
        title       = dict(text=title, x=0.5),
        xaxis       = dict(title="Days from Today",
                           tickvals=[0, 15, 30, 45, 60, 75, 90]),
        yaxis       = dict(title="Severity Score (0=healthy, 1=failed)", range=[0, 1.05]),
        height      = 400,
        margin      = dict(t=60, b=50, l=70, r=100),
        legend      = dict(orientation="h", y=-0.28, font=dict(size=10)),
        plot_bgcolor= "rgba(248,249,250,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode   = "x unified",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Confidence Histogram
# ═══════════════════════════════════════════════════════════════════════════════

def create_confidence_histogram(
    detections: List[Dict],
    title: str = "Detection Confidence Distribution",
) -> go.Figure:
    if not detections:
        fig = go.Figure()
        fig.update_layout(title=title, height=280)
        return fig

    confs  = [d["confidence"] for d in detections]
    levels = [d["severity_level"] for d in detections]

    fig = px.histogram(
        x           = confs,
        color       = levels,
        color_discrete_map = _PIE_COLORS,
        nbins       = 20,
        title       = title,
        labels      = {"x": "Confidence", "color": "Severity"},
        opacity     = 0.8,
        barmode     = "stack",
        category_orders={"color": _SEV_ORDER},
    )
    fig.update_layout(
        xaxis_title  = "YOLO Confidence Score",
        yaxis_title  = "Count",
        height       = 280,
        margin       = dict(t=50, b=40, l=50, r=20),
        legend_title_text = "Severity",
        plot_bgcolor = "rgba(248,249,250,1)",
        paper_bgcolor= "rgba(0,0,0,0)",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Severity Over Time  (scatter + rolling avg)
# ═══════════════════════════════════════════════════════════════════════════════

def create_severity_over_time(
    detections: List[Dict],
    title: str = "Severity Score Over Video Timeline",
) -> go.Figure:
    if not detections:
        fig = go.Figure()
        fig.update_layout(title=title, height=280)
        return fig

    ts     = [d["timestamp"]     for d in detections]
    scores = [d["severity_score"] for d in detections]
    levels = [d["severity_level"] for d in detections]
    colors = [_PIE_COLORS[l]      for l in levels]

    fig = go.Figure()

    # Scatter dots coloured by severity
    fig.add_trace(go.Scatter(
        x           = ts,
        y           = scores,
        mode        = "markers",
        name        = "Detections",
        marker      = dict(color=colors, size=9, opacity=0.8,
                           line=dict(color="white", width=1)),
        hovertemplate = "Time: %{x:.1f}s<br>Severity: %{y:.3f}<extra></extra>",
    ))

    # Rolling mean line (window ~5 points)
    if len(scores) >= 3:
        win = min(5, len(scores))
        rolling = np.convolve(scores, np.ones(win)/win, mode="valid")
        rolling_ts = ts[win-1:]
        fig.add_trace(go.Scatter(
            x    = rolling_ts,
            y    = rolling,
            mode = "lines",
            name = f"{win}-pt Rolling Avg",
            line = dict(color="#3498DB", width=2.5),
        ))

    fig.add_hline(y=0.35, line_dash="dash", line_color=_PIE_COLORS["Medium"],
                  annotation_text="Medium", annotation_position="right")
    fig.add_hline(y=0.65, line_dash="dash", line_color=_PIE_COLORS["High"],
                  annotation_text="High",   annotation_position="right")

    fig.update_layout(
        title        = dict(text=title, x=0.5),
        xaxis_title  = "Time (seconds)",
        yaxis_title  = "Severity Score",
        yaxis        = dict(range=[0, 1.05]),
        height       = 300,
        margin       = dict(t=50, b=40, l=60, r=70),
        plot_bgcolor = "rgba(248,249,250,1)",
        paper_bgcolor= "rgba(0,0,0,0)",
        hovermode    = "x unified",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Risk Comparison Bar (all videos)
# ═══════════════════════════════════════════════════════════════════════════════

def create_risk_comparison_bar(
    videos: List[Dict],
    title: str = "Road Risk Score — All Analyzed Videos",
) -> go.Figure:
    if not videos:
        fig = go.Figure()
        fig.update_layout(title=title, height=350)
        return fig

    sorted_vids = sorted(videos, key=lambda v: v.get("video_risk_score", 0), reverse=True)
    names  = [v["original_filename"] or v["filename"] for v in sorted_vids]
    scores = [v.get("video_risk_score", 0)             for v in sorted_vids]
    levels = [v.get("risk_level", "Low")               for v in sorted_vids]
    colors = [_PIE_COLORS.get(l, "#cccccc")            for l in levels]

    fig = go.Figure(go.Bar(
        x            = scores,
        y            = names,
        orientation  = "h",
        marker_color = colors,
        text         = [f"{s:.1f} ({l})" for s, l in zip(scores, levels)],
        textposition = "outside",
        hovertemplate= "<b>%{y}</b><br>Risk Score: %{x:.1f}<extra></extra>",
    ))

    # Threshold lines
    for val, label, color in [
        (25,  "Low/Med",  _PIE_COLORS["Medium"]),
        (50,  "Med/High", _PIE_COLORS["High"]),
        (75,  "High/Crit",_PIE_COLORS["Critical"]),
    ]:
        fig.add_vline(x=val, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="top")

    fig.update_layout(
        title        = dict(text=title, x=0.5),
        xaxis        = dict(range=[0, 110], title="Risk Score (0-100)"),
        yaxis        = dict(title=""),
        height       = max(300, 60 * len(names) + 80),
        margin       = dict(t=60, b=40, l=200, r=80),
        plot_bgcolor = "rgba(248,249,250,1)",
        paper_bgcolor= "rgba(0,0,0,0)",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  8. PCI Forecast Comparison (before/after degradation per video)
# ═══════════════════════════════════════════════════════════════════════════════

def create_pci_forecast_chart(
    detections: List[Dict],
    title: str = "Pavement Condition Index — Before & After Degradation",
) -> go.Figure:
    """
    Bar chart of average PCI now (+30d/60d/90d).
    Uses stored pci_score (ASTM D6433: 100 × (1-s)^1.7) for current,
    and recalculates from predicted severity scores for forecasts.
    """
    if not detections:
        fig = go.Figure()
        fig.update_layout(title=title, height=300)
        return fig

    import math
    PCI_EXP = 1.7   # mirrors config.PCI_EXPONENT

    def pci_from_score(s: float) -> float:
        return max(0.0, 100.0 * ((1.0 - max(0, min(1, s))) ** PCI_EXP))

    def avg_pci(col: str) -> float:
        vals = [d.get(col, 0) for d in detections]
        return round(sum(pci_from_score(v) for v in vals) / len(vals), 1)

    # Use stored pci_score for current (if available), fallback to recalc
    pci_now_vals = [d.get("pci_score") for d in detections]
    if all(v is not None and v > 0 for v in pci_now_vals):
        pci_now = round(sum(pci_now_vals) / len(pci_now_vals), 1)
    else:
        pci_now = avg_pci("severity_score")
    pci_30  = avg_pci("predicted_score_30d")
    pci_60  = avg_pci("predicted_score_60d")
    pci_90  = avg_pci("predicted_score_90d")

    def pci_color(pci):
        if pci >= 70:  return _PIE_COLORS["Low"]
        if pci >= 40:  return _PIE_COLORS["Medium"]
        return _PIE_COLORS["High"]

    days   = ["Current",  "+30 Days", "+60 Days", "+90 Days"]
    values = [pci_now,    pci_30,     pci_60,     pci_90]
    colors = [pci_color(v) for v in values]

    fig = go.Figure(go.Bar(
        x            = days,
        y            = values,
        marker_color = colors,
        text         = [f"{v:.1f}" for v in values],
        textposition = "outside",
        hovertemplate= "<b>%{x}</b><br>Avg PCI: %{y:.1f}/100<extra></extra>",
    ))

    fig.add_hline(y=70, line_dash="dash", line_color=_PIE_COLORS["Low"],
                  annotation_text="Good (PCI≥70)", annotation_position="right")
    fig.add_hline(y=40, line_dash="dash", line_color=_PIE_COLORS["Medium"],
                  annotation_text="Fair (PCI≥40)", annotation_position="right")

    fig.update_layout(
        title        = dict(text=title, x=0.5),
        yaxis        = dict(title="Avg Pavement Condition Index (0-100)", range=[0, 115]),
        height       = 320,
        margin       = dict(t=60, b=40, l=60, r=80),
        plot_bgcolor = "rgba(248,249,250,1)",
        paper_bgcolor= "rgba(0,0,0,0)",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  9. Risk contribution stacked bar  (severity / degradation / size)
# ═══════════════════════════════════════════════════════════════════════════════

def create_risk_breakdown_chart(
    detections: List[Dict],
    max_items: int = 15,
    title: str = "Risk Score Breakdown — Top Detections",
) -> go.Figure:
    if not detections:
        fig = go.Figure()
        fig.update_layout(title=title, height=300)
        return fig

    from modules.config import RISK_SEV_WEIGHT, RISK_DEG_WEIGHT, RISK_SIZE_WEIGHT, DEGRADATION_RATES, RISK_SIZE_SAT
    _max_deg = max(DEGRADATION_RATES.values())

    sorted_d = sorted(detections, key=lambda d: d["risk_score"], reverse=True)[:max_items]

    labels = [f"P{i+1}@{d['timestamp']:.0f}s" for i, d in enumerate(sorted_d)]

    sev_parts  = [round(d["severity_score"]  * (RISK_SEV_WEIGHT  * 100), 1) for d in sorted_d]
    deg_parts  = [round(min(d["degradation_rate"]/_max_deg, 1) * (RISK_DEG_WEIGHT * 100), 1) for d in sorted_d]
    size_parts = [round(min(d["area_ratio"]/RISK_SIZE_SAT, 1) * (RISK_SIZE_WEIGHT * 100), 1) for d in sorted_d]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Severity",    x=labels, y=sev_parts,  marker_color="#E74C3C"))
    fig.add_trace(go.Bar(name="Degradation", x=labels, y=deg_parts,  marker_color="#F39C12"))
    fig.add_trace(go.Bar(name="Size",        x=labels, y=size_parts, marker_color="#3498DB"))

    fig.update_layout(
        barmode      = "stack",
        title        = dict(text=title, x=0.5),
        xaxis_title  = "Detection",
        yaxis_title  = "Risk Score Component",
        height       = 320,
        margin       = dict(t=60, b=60, l=60, r=20),
        legend       = dict(orientation="h", y=1.10),
        plot_bgcolor = "rgba(248,249,250,1)",
        paper_bgcolor= "rgba(0,0,0,0)",
        xaxis        = dict(tickangle=-45),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Markov State Probability Chart  (stacked bar at t=0, 30, 60, 90 days)
# ═══════════════════════════════════════════════════════════════════════════════

def create_markov_probability_chart(
    severity_score: float,
    severity_level: str,
    title: str = "Markov Chain — State Probability Evolution",
) -> go.Figure:
    """
    Stacked 100% bar chart showing P(Low), P(Medium), P(High), P(Critical)
    at t = 0, 30, 60, 90 days for a single pothole detection.

    Uses the AASHTO-calibrated 4-state Markov P30 transition matrix from config.
    Visualises WHY urgency increases over time even if the current state is Low.
    """
    import numpy as np
    from modules.config import (
        MARKOV_P30, MARKOV_STATE_LABELS, SEV_LOW_MAX, SEV_MEDIUM_MAX,
    )

    P30 = np.array(MARKOV_P30, dtype=np.float64)
    P60 = P30 @ P30
    P90 = P30 @ P30 @ P30

    # Initial state vector (interpolated)
    v0 = np.zeros(4)
    if severity_score < SEV_LOW_MAX:
        v0[0] = 1.0
    elif severity_score < SEV_MEDIUM_MAX:
        frac = (severity_score - SEV_LOW_MAX) / (SEV_MEDIUM_MAX - SEV_LOW_MAX)
        v0[0] = 1.0 - frac; v0[1] = frac
    elif severity_score < 0.85:
        frac = (severity_score - SEV_MEDIUM_MAX) / (0.85 - SEV_MEDIUM_MAX)
        v0[1] = 1.0 - frac; v0[2] = frac
    else:
        frac = min((severity_score - 0.85) / 0.15, 1.0)
        v0[2] = 1.0 - frac; v0[3] = frac

    horizons = {"Today (t=0)": v0,
                "+30 Days":    v0 @ P30,
                "+60 Days":    v0 @ P60,
                "+90 Days":    v0 @ P90}

    state_colors = {
        "Low":      _PIE_COLORS["Low"],
        "Medium":   _PIE_COLORS["Medium"],
        "High":     _PIE_COLORS["High"],
        "Critical": _PIE_COLORS["Critical"],
    }

    x_labels = list(horizons.keys())
    fig = go.Figure()

    for si, state in enumerate(MARKOV_STATE_LABELS):
        probs = [round(float(v[si]) * 100, 1) for v in horizons.values()]
        fig.add_trace(go.Bar(
            name             = state,
            x                = x_labels,
            y                = probs,
            marker_color     = state_colors[state],
            text             = [f"{p:.0f}%" if p >= 4 else "" for p in probs],
            textposition     = "inside",
            insidetextanchor = "middle",
            hovertemplate    = f"<b>{state}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        barmode       = "stack",
        title         = dict(text=title, x=0.5, font=dict(size=15)),
        xaxis_title   = "Time Horizon",
        yaxis         = dict(title="State Probability (%)", range=[0, 101]),
        height        = 320,
        margin        = dict(t=60, b=50, l=70, r=20),
        legend        = dict(orientation="h", y=-0.20),
        plot_bgcolor  = "rgba(248,249,250,1)",
        paper_bgcolor = "rgba(0,0,0,0)",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Overall dashboard — severity stacked bar across all videos
# ═══════════════════════════════════════════════════════════════════════════════

def create_all_videos_severity_bar(
    videos: List[Dict],
    title: str = "Pothole Severity Breakdown — All Videos",
) -> go.Figure:
    if not videos:
        fig = go.Figure()
        fig.update_layout(title=title, height=300)
        return fig

    sorted_vids = sorted(videos, key=lambda v: v.get("video_risk_score", 0), reverse=True)
    names   = [v["original_filename"] or v["filename"] for v in sorted_vids]
    lows    = [v.get("low_potholes",    0) for v in sorted_vids]
    meds    = [v.get("medium_potholes", 0) for v in sorted_vids]
    highs   = [v.get("high_potholes",   0) for v in sorted_vids]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="High",   x=names, y=highs, marker_color=_PIE_COLORS["High"]))
    fig.add_trace(go.Bar(name="Medium", x=names, y=meds,  marker_color=_PIE_COLORS["Medium"]))
    fig.add_trace(go.Bar(name="Low",    x=names, y=lows,  marker_color=_PIE_COLORS["Low"]))

    fig.update_layout(
        barmode      = "stack",
        title        = dict(text=title, x=0.5),
        xaxis_title  = "Video",
        yaxis_title  = "Pothole Count",
        height       = 350,
        margin       = dict(t=60, b=80, l=60, r=20),
        legend       = dict(orientation="h", y=1.10),
        xaxis        = dict(tickangle=-30),
        plot_bgcolor = "rgba(248,249,250,1)",
        paper_bgcolor= "rgba(0,0,0,0)",
    )
    return fig
