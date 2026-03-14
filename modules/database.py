"""
SQLite database layer for RoadSense AI.
Handles all persistence: videos, frame results, individual detections.
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from modules.config import DB_PATH


# ─── Connection helper ────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ─── Schema initialisation ────────────────────────────────────────────────────

def initialize_db() -> None:
    """Create all tables if they do not already exist."""
    conn = _get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS videos (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        filename            TEXT    NOT NULL,
        original_filename   TEXT,
        file_path           TEXT    NOT NULL,
        upload_date         TEXT,
        analysis_date       TEXT,
        duration            REAL    DEFAULT 0,
        fps                 REAL    DEFAULT 0,
        total_frames        INTEGER DEFAULT 0,
        analyzed_frames     INTEGER DEFAULT 0,
        status              TEXT    DEFAULT 'pending',
        video_risk_score    REAL    DEFAULT 0,
        risk_level          TEXT    DEFAULT 'Unknown',
        total_potholes      INTEGER DEFAULT 0,
        unique_potholes     INTEGER DEFAULT 0,
        low_potholes        INTEGER DEFAULT 0,
        medium_potholes     INTEGER DEFAULT 0,
        high_potholes       INTEGER DEFAULT 0,
        notes               TEXT    DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS detections (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id            INTEGER NOT NULL,
        frame_number        INTEGER NOT NULL,
        timestamp           REAL    NOT NULL,
        bbox_x1             REAL,
        bbox_y1             REAL,
        bbox_x2             REAL,
        bbox_y2             REAL,
        confidence          REAL,
        area_ratio          REAL,
        edge_density        REAL,
        darkness_ratio      REAL,
        severity_score      REAL,
        severity_level      TEXT,
        risk_score          REAL,
        degradation_rate    REAL,
        predicted_score_30d REAL,
        predicted_score_60d REAL,
        predicted_score_90d REAL,
        predicted_level_30d TEXT,
        predicted_level_60d TEXT,
        predicted_level_90d TEXT,
        roi_image_path      TEXT,
        pci_score           REAL    DEFAULT 0,
        FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS frame_results (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id        INTEGER NOT NULL,
        frame_number    INTEGER NOT NULL,
        timestamp       REAL    NOT NULL,
        potholes_count  INTEGER DEFAULT 0,
        low_count       INTEGER DEFAULT 0,
        medium_count    INTEGER DEFAULT 0,
        high_count      INTEGER DEFAULT 0,
        frame_image_path TEXT,
        FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
    );
    """)
    # Migrate existing databases — silently skip if column already exists
    for col_def in [("pci_score", "REAL", "0")]:
        col, ctype, default = col_def
        try:
            conn.execute(f"ALTER TABLE detections ADD COLUMN {col} {ctype} DEFAULT {default}")
        except sqlite3.OperationalError:
            pass   # column already exists
    conn.commit()
    conn.close()


# ─── Video CRUD ───────────────────────────────────────────────────────────────

def add_video(filename: str, original_filename: str, file_path: str) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO videos (filename, original_filename, file_path, upload_date, status)
           VALUES (?, ?, ?, ?, 'pending')""",
        (filename, original_filename, file_path, datetime.now().isoformat()),
    )
    vid_id = cur.lastrowid
    conn.commit()
    conn.close()
    return vid_id


def update_video_analysis(video_id: int, data: Dict[str, Any]) -> None:
    conn = _get_conn()
    data["analysis_date"] = datetime.now().isoformat()
    data["status"] = "completed"
    cols = ", ".join(f"{k} = ?" for k in data)
    vals = list(data.values()) + [video_id]
    conn.execute(f"UPDATE videos SET {cols} WHERE id = ?", vals)
    conn.commit()
    conn.close()


def get_all_videos() -> List[Dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM videos ORDER BY upload_date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_video_by_id(video_id: int) -> Optional[Dict]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_video(video_id: int) -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
    conn.commit()
    conn.close()


# ─── Detection CRUD ───────────────────────────────────────────────────────────

def add_detection(video_id: int, det: Dict[str, Any]) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO detections (
               video_id, frame_number, timestamp,
               bbox_x1, bbox_y1, bbox_x2, bbox_y2,
               confidence, area_ratio, edge_density, darkness_ratio,
               severity_score, severity_level, risk_score, degradation_rate,
               predicted_score_30d, predicted_score_60d, predicted_score_90d,
               predicted_level_30d, predicted_level_60d, predicted_level_90d,
               roi_image_path, pci_score
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            video_id,
            det["frame_number"],    det["timestamp"],
            det["bbox_x1"],         det["bbox_y1"],
            det["bbox_x2"],         det["bbox_y2"],
            det["confidence"],      det["area_ratio"],
            det["edge_density"],    det["darkness_ratio"],
            det["severity_score"],  det["severity_level"],
            det["risk_score"],      det["degradation_rate"],
            det["predicted_score_30d"], det["predicted_score_60d"], det["predicted_score_90d"],
            det["predicted_level_30d"], det["predicted_level_60d"], det["predicted_level_90d"],
            det.get("roi_image_path", ""),
            det.get("pci_score", 0.0),
        ),
    )
    det_id = cur.lastrowid
    conn.commit()
    conn.close()
    return det_id


def get_video_detections(video_id: int) -> List[Dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM detections WHERE video_id = ? ORDER BY timestamp ASC",
        (video_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Frame results CRUD ───────────────────────────────────────────────────────

def add_frame_result(video_id: int, fr: Dict[str, Any]) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT INTO frame_results
               (video_id, frame_number, timestamp,
                potholes_count, low_count, medium_count, high_count, frame_image_path)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            video_id, fr["frame_number"], fr["timestamp"],
            fr["potholes_count"], fr["low_count"],
            fr["medium_count"],   fr["high_count"],
            fr.get("frame_image_path", ""),
        ),
    )
    conn.commit()
    conn.close()


def get_video_frame_results(video_id: int) -> List[Dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM frame_results WHERE video_id = ? ORDER BY timestamp ASC",
        (video_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
