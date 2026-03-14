"""
RoadSense AI — Streamlit Application
Full pothole detection, severity analysis, degradation forecasting,
risk prioritisation, and persistent video dashboard.

Run: streamlit run app.py
"""
from __future__ import annotations

import os
import io
import time
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from modules.config    import INPUT_VIDEOS_DIR, RISK_COLORS, SEVERITY_COLORS
from modules.database  import (
    initialize_db, add_video, get_all_videos,
    get_video_by_id, get_video_detections,
    get_video_frame_results, delete_video,
)
from modules.video_processor import analyze_video
from modules.visualizer import (
    create_severity_donut,
    create_timeline_chart,
    create_risk_gauge,
    create_degradation_chart,
    create_confidence_histogram,
    create_severity_over_time,
    create_risk_comparison_bar,
    create_pci_forecast_chart,
    create_risk_breakdown_chart,
    create_all_videos_severity_bar,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "RoadSense AI",
    page_icon  = "🛣️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global font & background */
body { font-family: 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.5rem; }

/* ── Header gradient */
.rs-header {
    background: linear-gradient(135deg,#0f3460,#16213e,#1a1a2e);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    color: #fff;
    margin-bottom: 1.5rem;
    text-align: center;
}
.rs-header h1 { margin:0; font-size:2.2rem; letter-spacing:1px; }
.rs-header p  { margin:0.3rem 0 0; opacity:0.75; font-size:1rem; }

/* ── Metric cards */
.metric-row { display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1rem; }
.metric-card {
    flex:1; min-width:130px;
    background:#fff;
    border-radius:10px;
    padding:1rem 1.2rem;
    box-shadow:0 2px 8px rgba(0,0,0,.08);
    text-align:center;
}
.metric-card .mc-value { font-size:2rem; font-weight:700; }
.metric-card .mc-label { font-size:.78rem; color:#666; margin-top:.2rem; }

/* ── Severity badges */
.badge {
    display:inline-block;
    padding:2px 10px;
    border-radius:20px;
    font-size:.78rem;
    font-weight:600;
    color:#fff;
}
.badge-Low      { background:#2ECC71; }
.badge-Medium   { background:#F39C12; }
.badge-High     { background:#E74C3C; }
.badge-Critical { background:#8B0000; }
.badge-Unknown  { background:#95a5a6; }

/* ── Video list sidebar items */
.vid-item {
    padding:.6rem .8rem;
    border-radius:8px;
    margin:.25rem 0;
    cursor:pointer;
    border-left:4px solid transparent;
    background:#f8f9fa;
    transition:background .15s;
}
.vid-item:hover { background:#eef1f5; }

/* ── Risk card borders */
.risk-card {
    border-radius:8px;
    padding:1rem;
    margin:.5rem 0;
}
.risk-Low      { border-left:5px solid #2ECC71; background:#f0fff4; }
.risk-Medium   { border-left:5px solid #F39C12; background:#fffbf0; }
.risk-High     { border-left:5px solid #E74C3C; background:#fff5f5; }
.risk-Critical { border-left:5px solid #8B0000; background:#fff0f0; }

/* ── Progress bar colour override */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg,#0f3460,#e94560);
}

/* ── ROI gallery */
.roi-img-container {
    border-radius:8px;
    overflow:hidden;
    box-shadow:0 2px 8px rgba(0,0,0,.15);
    position:relative;
}

/* ── Section dividers */
.section-title {
    font-size:1.15rem;
    font-weight:700;
    color:#1a1a2e;
    border-bottom:2px solid #e94560;
    padding-bottom:.3rem;
    margin:1.2rem 0 .8rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Bootstrap DB ─────────────────────────────────────────────────────────────
initialize_db()

# ─── Session state defaults ───────────────────────────────────────────────────
if "selected_video_id" not in st.session_state:
    st.session_state.selected_video_id = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_seconds(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec:02d}s"

def _risk_badge(level: str) -> str:
    return f'<span class="badge badge-{level}">{level}</span>'

def _severity_badge(level: str) -> str:
    return f'<span class="badge badge-{level}">{level}</span>'

def _load_image_safe(path: str, size: tuple = None) -> Optional[np.ndarray]:
    """Load an image from disk safely; return None if missing."""
    if not path or not Path(path).is_file():
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size:
        img_rgb = cv2.resize(img_rgb, size)
    return img_rgb

def _save_uploaded_video(uploaded_file) -> tuple:
    """Save uploaded file to input_videos and return (video_id, save_path)."""
    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts_str}_{uploaded_file.name}"
    save_path= INPUT_VIDEOS_DIR / filename
    with open(str(save_path), "wb") as f:
        f.write(uploaded_file.read())
    vid_id = add_video(filename, uploaded_file.name, str(save_path))
    return vid_id, str(save_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:.5rem 0 1rem;">
          <span style="font-size:2.5rem;">🛣️</span>
          <h2 style="margin:.2rem 0 0;color:#0f3460;font-size:1.4rem;">RoadSense AI</h2>
          <p style="font-size:.78rem;color:#666;margin:0;">Pothole Detection & Analysis</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        all_vids = get_all_videos()
        completed = [v for v in all_vids if v["status"] == "completed"]
        pending   = [v for v in all_vids if v["status"] != "completed"]

        # Quick stats
        if completed:
            total_ph = sum(v.get("total_potholes", 0) for v in completed)
            avg_risk = sum(v.get("video_risk_score", 0) for v in completed) / len(completed)
            c1, c2 = st.columns(2)
            c1.metric("Videos", len(completed))
            c2.metric("Potholes", total_ph)
            st.metric("Avg Road Risk", f"{avg_risk:.1f}/100")
            st.markdown("---")

        # Navigation buttons
        if st.button("🏠  Home / Upload", use_container_width=True):
            st.session_state.selected_video_id = None
            st.session_state.active_tab = 0
            st.rerun()
        if st.button("📊  All-Videos Dashboard", use_container_width=True):
            st.session_state.selected_video_id = None
            st.session_state.active_tab = 2
            st.rerun()

        st.markdown("---")
        st.markdown("**📁 Analyzed Videos**")

        if not completed:
            st.info("No videos analyzed yet.")
        else:
            for v in completed:
                lvl = v.get("risk_level", "Unknown")
                color = RISK_COLORS.get(lvl, "#95a5a6")
                name  = v["original_filename"] or v["filename"]
                score = v.get("video_risk_score", 0)
                date  = (v.get("analysis_date") or "")[:10]

                if st.button(
                    f"🎬 {name[:28]}…" if len(name) > 28 else f"🎬 {name}",
                    key        = f"vid_{v['id']}",
                    use_container_width=True,
                    help       = f"Risk: {score:.1f}/100 | {lvl} | {date}",
                ):
                    st.session_state.selected_video_id = v["id"]
                    st.session_state.active_tab = 1
                    st.rerun()

                st.markdown(
                    f'<div style="font-size:.7rem;color:#888;margin:-6px 0 4px 4px;">'
                    f'Risk {score:.0f}/100 • {_risk_badge(lvl)} • {date}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Pending (failed / never completed)
        if pending:
            st.markdown("---")
            st.markdown("**⏳ Pending / Incomplete**")
            for v in pending:
                name = v["original_filename"] or v["filename"]
                st.markdown(f"<small>⚠️ {name[:30]}</small>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 0  —  Upload & Analyse
# ═══════════════════════════════════════════════════════════════════════════════

def render_upload_tab():
    st.markdown("""
    <div class="rs-header">
      <h1>🛣️ RoadSense AI</h1>
      <p>Upload a road inspection video to automatically detect and analyse potholes,
      estimate severity, forecast degradation, and prioritise repairs.</p>
    </div>
    """, unsafe_allow_html=True)

    col_up, col_info = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown('<div class="section-title">📤 Upload Road Video</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop or browse video file",
            type=["mp4", "avi", "mov", "mkv", "wmv"],
            help="Supported: MP4, AVI, MOV, MKV, WMV",
        )

        if uploaded:
            st.video(uploaded)
            file_mb = uploaded.size / (1024*1024)
            st.caption(f"📦 File size: {file_mb:.1f} MB | Type: {uploaded.type}")

            st.markdown('<div class="section-title">⚙️ Analysis Settings</div>', unsafe_allow_html=True)
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                conf_thresh = st.slider("Min Confidence", 0.10, 0.80, 0.25, 0.05,
                                        help="YOLO confidence threshold")
            with col_s2:
                base_fps = st.selectbox("Sample Rate", [1, 2, 3], index=0,
                                        help="Frames per second to analyse (1 = fast, 3 = thorough)")

            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                # Reset uploaded file position for saving
                uploaded.seek(0)
                vid_id, vid_path = _save_uploaded_video(uploaded)
                st.success(f"✅ Video saved → Running analysis on **{uploaded.name}**")
                _run_analysis(vid_id, vid_path)
                st.session_state.selected_video_id = vid_id
                st.session_state.active_tab = 1
                st.rerun()

    with col_info:
        st.markdown('<div class="section-title">ℹ️ What the System Does</div>', unsafe_allow_html=True)
        steps = [
            ("5️⃣", "ROI Extraction",        "Crops each detected pothole with padding for detailed inspection."),
            ("6️⃣", "Severity Estimation",    "Multi-factor score: area (40%) + edge-texture (35%) + confidence (25%)."),
            ("7️⃣", "Degradation Forecast",   "Exponential growth model predicts severity at +30/60/90 days."),
            ("8️⃣", "Risk Prioritisation",    "Weighted risk score (0-100): severity + degradation urgency + size."),
            ("⏱️", "Temporal Aggregation",   "Deduplicates per-frame detections into unique real-world potholes."),
            ("📊", "Interactive Dashboard",  "Full charts, gallery, timeline, and road-level priority score."),
        ]
        for icon, name, desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:.8rem;align-items:flex-start;margin:.5rem 0;
                        background:#f8f9fa;border-radius:8px;padding:.6rem .8rem;">
              <span style="font-size:1.4rem;line-height:1;">{icon}</span>
              <div>
                <strong style="color:#0f3460">{name}</strong>
                <br><small style="color:#555">{desc}</small>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">📋 Severity Levels</div>', unsafe_allow_html=True)
        for level, color, desc in [
            ("Low",    "#2ECC71", "Small depression •  area < ~5% frame  •  Low vehicle hazard"),
            ("Medium", "#F39C12", "Moderate damage  •  5-12% frame       •  Schedule repair"),
            ("High",   "#E74C3C", "Severe damage    •  >12% frame        •  Immediate action"),
        ]:
            st.markdown(
                f'<div style="border-left:4px solid {color};padding:.4rem .8rem;'
                f'margin:.3rem 0;background:{color}18;border-radius:0 6px 6px 0;">'
                f'<b style="color:{color}">{level}</b>  <small>— {desc}</small></div>',
                unsafe_allow_html=True,
            )


def _run_analysis(vid_id: int, vid_path: str):
    """Run the analysis generator and display live progress."""
    progress_bar   = st.progress(0.0)
    status_text    = st.empty()
    preview_slot   = st.empty()
    live_metrics   = st.empty()

    total_dets = 0
    high_dets  = 0
    try:
        for update in analyze_video(vid_path, vid_id):
            if update["type"] == "progress":
                done  = update["done"]
                total = max(update["total"], 1)
                pct   = done / total
                progress_bar.progress(pct)

                total_dets += update["new_detections"]
                ts_str      = _fmt_seconds(update["frame_ts"])

                status_text.markdown(
                    f"🔍 Analysing… frame **{done}/{total}** | "
                    f"⏱️ `{ts_str}` | "
                    f"🕳️ Detections so far: **{total_dets}**"
                )

                # Live preview
                if update.get("preview_frame") is not None:
                    preview_slot.image(
                        update["preview_frame"],
                        caption="Live — latest frame with detections",
                        use_container_width=False, width=360,
                    )

                # Live mini-metrics
                live_metrics.markdown(
                    f'<div style="display:flex;gap:1rem;flex-wrap:wrap;">'
                    f'<div style="background:#f8f9fa;border-radius:6px;padding:.4rem .8rem;">'
                    f'Frames:{done}/{total}</div>'
                    f'<div style="background:#f8f9fa;border-radius:6px;padding:.4rem .8rem;">'
                    f'Detections:{total_dets}</div></div>',
                    unsafe_allow_html=True,
                )

            elif update["type"] == "complete":
                progress_bar.progress(1.0)
                vs = update["video_summary"]
                status_text.success(
                    f"✅ Analysis complete!  "
                    f"| Unique potholes: {vs['unique_potholes']} "
                    f"| Road Risk: {vs['video_risk_score']:.1f}/100 ({vs['risk_level']})"
                )
                preview_slot.empty()
                time.sleep(0.5)

            elif update["type"] == "error":
                st.error(f"❌ Analysis error: {update['message']}")
                return

    except Exception as e:
        st.error(f"❌ Unexpected error during analysis: {e}")
        st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1  —  Single Video Results
# ═══════════════════════════════════════════════════════════════════════════════

def render_video_results(video_id: int):
    video  = get_video_by_id(video_id)
    if not video or video["status"] != "completed":
        st.warning("Video not found or not yet analysed.")
        return

    dets   = get_video_detections(video_id)
    frames = get_video_frame_results(video_id)

    fname  = video["original_filename"] or video["filename"]
    risk_s = video.get("video_risk_score", 0)
    risk_l = video.get("risk_level", "Unknown")
    risk_c = RISK_COLORS.get(risk_l, "#95a5a6")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="rs-header"><h1>📋 {fname}</h1>'
        f'<p>Analysed: {(video.get("analysis_date") or "")[:19]}  |  '
        f'Duration: {_fmt_seconds(video.get("duration",0))}  |  '
        f'FPS: {video.get("fps",0):.1f}</p></div>',
        unsafe_allow_html=True,
    )

    # ── Summary metric cards ──────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    def _mc(col, val, label, color="#1a1a2e"):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="mc-value" style="color:{color}">{val}</div>'
            f'<div class="mc-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    _mc(m1, video.get("total_potholes",  0),  "Total Detections")
    _mc(m2, video.get("unique_potholes", 0),  "Unique Potholes")
    _mc(m3, video.get("low_potholes",    0),  "🟢 Low Severity",    "#2ECC71")
    _mc(m4, video.get("medium_potholes", 0),  "🟡 Med Severity",    "#F39C12")
    _mc(m5, video.get("high_potholes",   0),  "🔴 High Severity",   "#E74C3C")
    _mc(m6, f"{risk_s:.0f}/100",              f"Road Risk ({risk_l})", risk_c)

    # ── Action recommendation ─────────────────────────────────────────────────
    _ACTIONS = {
        "Low":      "✅ **No immediate action required.** Schedule routine maintenance.",
        "Medium":   "⚠️ **Plan repairs within 30 days.** Multiple potholes detected.",
        "High":     "🚨 **Urgent repair within 7 days.** Significant road damage.",
        "Critical": "🆘 **EMERGENCY REPAIR REQUIRED.** Road poses immediate hazard to vehicles.",
    }
    st.markdown(
        f'<div class="risk-card risk-{risk_l}">'
        f'{_ACTIONS.get(risk_l, "")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Analysis Charts</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.6, 1])

    with c1:
        fig_donut = create_severity_donut(
            video.get("low_potholes",0),
            video.get("medium_potholes",0),
            video.get("high_potholes",0),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        fig_timeline = create_timeline_chart(frames)
        st.plotly_chart(fig_timeline, use_container_width=True)

    with c3:
        fig_gauge = create_risk_gauge(risk_s, risk_l)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    c4, c5 = st.columns(2)
    with c4:
        fig_sev_time = create_severity_over_time(dets)
        st.plotly_chart(fig_sev_time, use_container_width=True)
    with c5:
        fig_conf = create_confidence_histogram(dets)
        st.plotly_chart(fig_conf, use_container_width=True)

    # ── Degradation & PCI (before/after) ─────────────────────────────────────
    st.markdown('<div class="section-title">🔮 Degradation Forecast (Before vs After)</div>',
                unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        fig_deg = create_degradation_chart(dets)
        st.plotly_chart(fig_deg, use_container_width=True)
    with d2:
        fig_pci = create_pci_forecast_chart(dets)
        st.plotly_chart(fig_pci, use_container_width=True)

    # ── Risk Breakdown ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⚖️ Risk Score Components</div>',
                unsafe_allow_html=True)
    fig_rb = create_risk_breakdown_chart(dets)
    st.plotly_chart(fig_rb, use_container_width=True)

    # ── High-Severity Gallery ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">🖼️ High & Medium Severity Pothole Gallery</div>',
                unsafe_allow_html=True)

    priority_dets = sorted(
        [d for d in dets if d["severity_level"] in ("High", "Medium")],
        key=lambda d: d["risk_score"], reverse=True
    )[:20]

    if not priority_dets:
        st.info("No medium or high severity potholes detected in this video.")
    else:
        cols_per_row = 4
        for row_start in range(0, len(priority_dets), cols_per_row):
            row_dets = priority_dets[row_start: row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, det in zip(cols, row_dets):
                with col:
                    roi_img = _load_image_safe(det.get("roi_image_path"), size=(220, 165))
                    if roi_img is not None:
                        st.image(roi_img, use_container_width=True)
                    else:
                        st.markdown(
                            '<div style="background:#f0f0f0;height:120px;border-radius:6px;'
                            'display:flex;align-items:center;justify-content:center;">'
                            '<span>No image</span></div>',
                            unsafe_allow_html=True,
                        )

                    lvl   = det["severity_level"]
                    conf  = det["confidence"]
                    risk  = det["risk_score"]
                    ts    = det["timestamp"]
                    color = SEVERITY_COLORS.get(lvl, "#888")

                    st.markdown(
                        f'<div style="text-align:center;font-size:.8rem;margin-top:.3rem;">'
                        f'<span class="badge badge-{lvl}">{lvl}</span> '
                        f'Conf: <b>{conf:.0%}</b><br>'
                        f'⏱️ {_fmt_seconds(ts)} | Risk: <b style="color:{color}">{risk:.0f}/100</b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Detailed Detections Table ─────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Detailed Detection Log</div>',
                unsafe_allow_html=True)

    if not dets:
        st.info("No detections found.")
    else:
        # Filters
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        with fc1:
            sev_filter = st.multiselect(
                "Filter Severity", ["Low", "Medium", "High"],
                default=["Low", "Medium", "High"],
                key=f"sev_filter_{video_id}",
            )
        with fc2:
            min_risk_filter = st.slider(
                "Min Risk Score", 0, 100, 0, 5,
                key=f"risk_filter_{video_id}",
            )
        with fc3:
            sort_col = st.selectbox(
                "Sort By",
                ["timestamp", "risk_score", "severity_score", "confidence"],
                index=1,
                key=f"sort_{video_id}",
            )

        filtered = [
            d for d in dets
            if d["severity_level"] in sev_filter
            and d["risk_score"] >= min_risk_filter
        ]
        filtered.sort(key=lambda d: d.get(sort_col, 0), reverse=(sort_col != "timestamp"))

        if filtered:
            df = pd.DataFrame([
                {
                    "Time":        _fmt_seconds(d["timestamp"]),
                    "Frame":       d["frame_number"],
                    "Severity":    d["severity_level"],
                    "Sev Score":   f'{d["severity_score"]:.3f}',
                    "Confidence":  f'{d["confidence"]:.1%}',
                    "Risk Score":  f'{d["risk_score"]:.1f}',
                    "Area %":      f'{d["area_ratio"]*100:.2f}%',
                    "Edge Dens":   f'{d["edge_density"]*100:.1f}%',
                    "Pred +30d":   d["predicted_level_30d"],
                    "Pred +60d":   d["predicted_level_60d"],
                    "Pred +90d":   d["predicted_level_90d"],
                    "Action":      {"Low":"Monitor","Medium":"Schedule","High":"Urgent Repair"}.get(d["severity_level"],""),
                }
                for d in filtered
            ])

            # Colour rows
            def _row_color(row):
                lvl = row["Severity"]
                if lvl == "High":   return ["background-color: #fff5f5"] * len(row)
                if lvl == "Medium": return ["background-color: #fffbf0"] * len(row)
                return ["background-color: #f0fff4"] * len(row)

            st.dataframe(
                df.style.apply(_row_color, axis=1),
                use_container_width=True,
                height=min(400, 50 + 35 * len(df)),
            )
            st.caption(f"Showing {len(filtered)} of {len(dets)} detections.")
        else:
            st.info("No detections match the current filters.")

    # ── Per-detection degradation recommendations ─────────────────────────────
    high_dets = [d for d in dets if d["severity_level"] == "High"]
    if high_dets:
        with st.expander(f"🛠️ Repair Recommendations ({len(high_dets)} HIGH severity potholes)"):
            for i, det in enumerate(sorted(high_dets, key=lambda d: d["risk_score"], reverse=True)[:10], 1):
                ts = _fmt_seconds(det["timestamp"])
                prv30 = det["predicted_level_30d"]
                prv90 = det["predicted_level_90d"]
                st.markdown(f"""
                <div class="risk-card risk-High">
                <b>#{i}  @{ts}  |  Risk {det['risk_score']:.0f}/100  |  Conf {det['confidence']:.0%}</b><br>
                📐 Area: {det['area_ratio']*100:.2f}%  •  Edge Density: {det['edge_density']*100:.1f}%<br>
                📅 Predicted: <b>+30d→{prv30}</b>  |  <b>+90d→{prv90}</b><br>
                🔧 <em>Emergency repair required — full-depth reclamation recommended within 7 days.</em>
                </div>
                """, unsafe_allow_html=True)

    # ── Delete video option ───────────────────────────────────────────────────
    with st.expander("⚠️ Danger Zone"):
        st.warning("Deleting a video record removes all its analysis data from the database.")
        if st.button(f"🗑️ Delete '{fname}' and all its data", key=f"del_{video_id}"):
            delete_video(video_id)
            st.session_state.selected_video_id = None
            st.session_state.active_tab = 0
            st.success("Video record deleted.")
            time.sleep(0.5)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2  —  All-Videos Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def render_dashboard():
    all_vids   = get_all_videos()
    completed  = [v for v in all_vids if v["status"] == "completed"]

    st.markdown("""
    <div class="rs-header">
      <h1>📊 All-Roads Dashboard</h1>
      <p>Aggregate view of all analysed road videos, ranked by risk.</p>
    </div>
    """, unsafe_allow_html=True)

    if not completed:
        st.info("No completed analyses yet. Upload and analyse a video first.")
        return

    # ── Global summary ────────────────────────────────────────────────────────
    total_vids  = len(completed)
    total_ph    = sum(v.get("total_potholes",  0) for v in completed)
    total_hi    = sum(v.get("high_potholes",   0) for v in completed)
    total_med   = sum(v.get("medium_potholes", 0) for v in completed)
    total_low   = sum(v.get("low_potholes",    0) for v in completed)
    avg_risk    = sum(v.get("video_risk_score",0) for v in completed) / total_vids
    worst_risk  = max(v.get("video_risk_score",0) for v in completed)

    mc = st.columns(6)
    def _mc(col, val, label, color="#1a1a2e"):
        col.markdown(
            f'<div class="metric-card"><div class="mc-value" style="color:{color}">{val}</div>'
            f'<div class="mc-label">{label}</div></div>', unsafe_allow_html=True)

    _mc(mc[0], total_vids,             "Videos Analysed")
    _mc(mc[1], total_ph,               "Total Detections")
    _mc(mc[2], total_low,              "🟢 Low",     "#2ECC71")
    _mc(mc[3], total_med,              "🟡 Medium",  "#F39C12")
    _mc(mc[4], total_hi,               "🔴 High",    "#E74C3C")
    _mc(mc[5], f"{avg_risk:.1f}/100",  "Avg Road Risk")

    st.markdown("---")

    # ── Risk comparison bar chart ─────────────────────────────────────────────
    st.markdown('<div class="section-title">🏆 Road Risk Ranking</div>', unsafe_allow_html=True)
    fig_compare = create_risk_comparison_bar(completed)
    st.plotly_chart(fig_compare, use_container_width=True)

    # ── Severity stacked bar ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Severity Breakdown by Video</div>',
                unsafe_allow_html=True)
    fig_sev = create_all_videos_severity_bar(completed)
    st.plotly_chart(fig_sev, use_container_width=True)

    # ── Overall severity donut ────────────────────────────────────────────────
    dc1, dc2 = st.columns(2)
    with dc1:
        fig_total_donut = create_severity_donut(
            total_low, total_med, total_hi,
            title="Overall Severity Distribution (All Videos)",
        )
        st.plotly_chart(fig_total_donut, use_container_width=True)

    with dc2:
        # Risk level distribution pie
        import plotly.graph_objects as go
        risk_lev_counts = {}
        for v in completed:
            lvl = v.get("risk_level", "Unknown")
            risk_lev_counts[lvl] = risk_lev_counts.get(lvl, 0) + 1

        labels = list(risk_lev_counts.keys())
        values = list(risk_lev_counts.values())
        colors_pie = [RISK_COLORS.get(l, "#95a5a6") for l in labels]

        fig_risk_dist = go.Figure(go.Pie(
            labels    = labels,
            values    = values,
            hole      = 0.5,
            marker    = dict(colors=colors_pie, line=dict(color="#fff", width=2)),
            textinfo  = "label+value+percent",
        ))
        fig_risk_dist.update_layout(
            title  = dict(text="Road Risk Level Distribution", x=0.5),
            height = 340,
            margin = dict(t=50, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_risk_dist, use_container_width=True)

    # ── Per-video card table ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Video Summary Table</div>',
                unsafe_allow_html=True)

    sort_dash = st.selectbox(
        "Sort Dashboard By",
        ["Risk Score ↓", "Date ↓", "Total Potholes ↓", "High Potholes ↓"],
        key="dash_sort",
    )
    sort_map = {
        "Risk Score ↓":     lambda v: -v.get("video_risk_score", 0),
        "Date ↓":           lambda v: -(v.get("analysis_date") or ""),
        "Total Potholes ↓": lambda v: -v.get("total_potholes",  0),
        "High Potholes ↓":  lambda v: -v.get("high_potholes",   0),
    }
    sorted_vids = sorted(completed, key=sort_map[sort_dash])

    for v in sorted_vids:
        lvl   = v.get("risk_level", "Unknown")
        risk  = v.get("video_risk_score", 0)
        color = RISK_COLORS.get(lvl, "#95a5a6")
        name  = v["original_filename"] or v["filename"]
        date  = (v.get("analysis_date") or "")[:19]
        dur   = _fmt_seconds(v.get("duration", 0))
        fps   = v.get("fps", 0)

        with st.container():
            st.markdown(
                f'<div class="risk-card risk-{lvl}">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:1rem;font-weight:700;color:#1a1a2e">🎬 {name}</span>'
                f'<span class="badge badge-{lvl}" style="font-size:.9rem;">'
                f'{lvl} RISK  {risk:.1f}/100</span></div>'
                f'<div style="font-size:.8rem;color:#555;margin-top:.4rem;display:flex;gap:1.5rem;">'
                f'<span>📅 {date}</span>'
                f'<span>⏱️ {dur}</span>'
                f'<span>🎞️ {fps:.1f} FPS</span>'
                f'<span>🕳️ Total: <b>{v.get("total_potholes",0)}</b></span>'
                f'<span>🟢 {v.get("low_potholes",0)}</span>'
                f'<span>🟡 {v.get("medium_potholes",0)}</span>'
                f'<span>🔴 {v.get("high_potholes",0)}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            if st.button(f"View Details →", key=f"dash_view_{v['id']}", use_container_width=False):
                st.session_state.selected_video_id = v["id"]
                st.session_state.active_tab = 1
                st.rerun()

            st.markdown("<div style='margin-bottom:.3rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    render_sidebar()

    # Determine active view from session state
    tab_idx = st.session_state.active_tab
    vid_id  = st.session_state.selected_video_id

    if tab_idx == 1 and vid_id is not None:
        render_video_results(vid_id)
    elif tab_idx == 2:
        render_dashboard()
    else:
        render_upload_tab()


if __name__ == "__main__":
    main()
