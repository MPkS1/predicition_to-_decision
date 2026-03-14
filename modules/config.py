"""
Configuration constants for RoadSense AI system.
All paths, thresholds, weights, and model parameters in one place.

Severity  — 6-factor AHP score  (Chen 2022, Akarsu 2021, Zhang 2016)
Degradation — Logistic sigmoid + Markov chain  (HDM-4 World Bank, AASHTO)
Risk score  — AHP pairwise-derived weights, CR=0  (Saaty 1977, Moazami 2011)
Video score — Pareto rank-weighting + Gini coefficient  (STATS19, EVT theory)
PCI         — ASTM D6433 nonlinear: PCI = 100 × (1-s)^1.7
"""
from pathlib import Path

# ─── Directory Paths ──────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent.parent
DATA_DIR         = BASE_DIR / "data"
INPUT_VIDEOS_DIR = DATA_DIR / "input_videos"
RESULTS_DIR      = DATA_DIR / "results"
FRAMES_DIR       = RESULTS_DIR / "frames"
ROI_DIR          = RESULTS_DIR / "roi_images"
DB_PATH          = DATA_DIR / "roadsense.db"
MODEL_PATH       = BASE_DIR / "roadsense_best.pt"

for _d in [INPUT_VIDEOS_DIR, FRAMES_DIR, ROI_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── YOLO Inference ───────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD        = 0.45
IMGSZ                = 640

# ─── Frame Extraction ─────────────────────────────────────────────────────────
BASE_FPS      = 1
ADAPTIVE_FPS  = 3
ADAPTIVE_SECS = 2

# ─── Step 6: Severity — 6-Factor AHP ─────────────────────────────────────────
# AHP pairwise matrix (geometric progression → CR=0, verified consistent)
# Weights derived via eigenvalue method (Saaty 1977)
SEV_W_AREA  = 0.28   # log-normalised area  (most important — size drives damage)
SEV_W_LAP   = 0.22   # Laplacian variance   (depth proxy — Brenner focus measure)
SEV_W_DARK  = 0.18   # dark pixel ratio     (shadow = depth confirmation)
SEV_W_GRAD  = 0.15   # Sobel gradient energy(rim sharpness = tyre impact severity)
SEV_W_SHAPE = 0.07   # elongation factor    (lane-spanning potholes harder to avoid)
SEV_W_CONF  = 0.10   # YOLO confidence      (detection certainty)
# sum = 1.00  ✓

# Log-normalisation reference area (3% of frame = "standard" pothole)
SEV_AREA_REF  = 0.03

# Laplacian variance saturation (calibrated on pothole datasets)
SEV_LAP_SAT   = 800.0

# Sobel gradient energy saturation
SEV_GRAD_SAT  = 80.0

# Dark pixel threshold: pixels below (mean - SEV_DARK_K × std) counted
SEV_DARK_K    = 0.5

# Severity score → level thresholds (unchanged — calibrated to new scale)
SEV_LOW_MAX    = 0.35
SEV_MEDIUM_MAX = 0.65

# ─── Step 7: Degradation — Logistic Sigmoid + Markov Chain ───────────────────
# Sigmoid: s(t) = 1 / (1 + ((1-s0)/s0) * exp(-k * CF * t))
# Rates calibrated from HDM-4 World Bank deterioration curves
DEGRADATION_RATES = {
    "Low":    0.008,   # slow — surface crack infiltration phase
    "Medium": 0.015,   # accelerating — sub-base saturation
    "High":   0.025,   # rapid — structural failure imminent
}
CLIMATE_FACTOR = 1.0    # multiply rate; 1.4 = freeze-thaw, 1.2 = tropical (heavy rain)
FORECAST_DAYS  = [30, 60, 90]

# Markov chain: 4 states — Low(0), Medium(1), High(2), Critical(3)
# 30-day transition matrix calibrated from AASHTO Pavement Design Guide data
# Rows: current state; Columns: next state after 30 days
# Note: Critical is an absorbing state (no self-repair of roads)
MARKOV_P30 = [
    [0.70, 0.25, 0.04, 0.01],   # Low → mostly stays Low, some progress
    [0.00, 0.65, 0.28, 0.07],   # Med → no recovery, worsens faster
    [0.00, 0.00, 0.60, 0.40],   # High → 40% chance Critical in 30 days
    [0.00, 0.00, 0.00, 1.00],   # Critical → absorbing (requires repair to exit)
]
MARKOV_STATE_MIDPOINTS = [0.175, 0.50, 0.75, 0.925]  # expected s within each state
MARKOV_STATE_LABELS    = ["Low", "Medium", "High", "Critical"]

# PCI exponent (ASTM D6433 adapted for potholes)
PCI_EXPONENT = 1.7   # PCI = 100 × (1 - severity)^1.7

# ─── Step 8: Risk Prioritisation — AHP-derived Weights ───────────────────────
# Pairwise comparison matrix (3×3):
#              Severity  Degradation  Size
# Severity  [    1         2           4   ]   rationale: severity > urgency > size
# Degradation [  1/2       1           2   ]
# Size       [  1/4       1/2          1   ]
# → column-normalise → row-average → w = [0.571, 0.286, 0.143]
# Consistency: λ_max = 3.0, CI = 0, CR = 0  ✓ (geometric progression matrix)
RISK_SEV_WEIGHT  = 0.571
RISK_DEG_WEIGHT  = 0.286
RISK_SIZE_WEIGHT = 0.143

RISK_SIZE_SAT    = 0.05   # area fraction that fully saturates size component

# Risk levels (0-100 scale)
RISK_LOW_MAX     = 25
RISK_MEDIUM_MAX  = 50
RISK_HIGH_MAX    = 75
# > 75 → Critical

# ─── Video-level Score — Pareto + Gini ───────────────────────────────────────
# Pareto rank-weighting: rank-1 pothole gets weight n, rank-2 gets n-1, …
# Reflects 80/20 rule (worst 20% of potholes cause 80% of accidents, UK STATS19)
# Gini coefficient [0,1]: concentration of damage
# final = pareto_score × (1 + VID_GINI_GAMMA × Gini)
VID_GINI_GAMMA   = 0.20    # Gini multiplier (20% boost when damage is concentrated)
VID_RISK_VOL_CAP = 40      # unique potholes that saturates density factor

# ─── Temporal Deduplication ───────────────────────────────────────────────────
DEDUP_TIME_WINDOW = 2.0
DEDUP_DIST_PX     = 80

# ─── Visual Colours ───────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "Low":      "#2ECC71",
    "Medium":   "#F39C12",
    "High":     "#E74C3C",
    "Critical": "#8B0000",
}
RISK_COLORS = {
    "Low":      "#2ECC71",
    "Medium":   "#F39C12",
    "High":     "#E74C3C",
    "Critical": "#8B0000",
}
