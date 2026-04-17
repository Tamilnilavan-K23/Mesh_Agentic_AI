# =============================================================
#  config.py — Central configuration for Disaster Mesh Agent AI
#  Edit the PATHS block below to match your local file locations
# =============================================================

import os

# -------------------------------------------------------------
#  FILE PATHS — edit these two lines
# -------------------------------------------------------------
CSV_PATH   = r"F:\sankar project\Agentic model\Mesh_Agentic_AI\data\validation_dataset.csv"   # your telemetry CSV
MODEL_PATH = r"F:\sankar project\Agentic model\Mesh_Agentic_AI\model\random_forest_model_fixed.pkl"   # your trained RF model
# -------------------------------------------------------------

# Feature names expected by the model (must match training order)
FEATURES = [
    "battery_v",
    "rssi_dbm",
    "pkt_loss_pct",
    "temp_c",
    "uptime_hrs",
    "hop_count",
    "queue_len_pkts",
    "tx_rate_kbps",
]

# Physical min/max ranges for normalization
FEATURE_RANGES = {
    "battery_v":      (2.9,  4.2),
    "rssi_dbm":       (-120, -40),
    "pkt_loss_pct":   (0,    100),
    "temp_c":         (20,   85),
    "uptime_hrs":     (0,    720),
    "hop_count":      (1,    10),
    "queue_len_pkts": (0,    50),
    "tx_rate_kbps":   (1,    250),
}

# Warning thresholds — a breached feature counts toward severity
FEATURE_THRESHOLDS = {
    "battery_v":      ("lt", 3.4),
    "rssi_dbm":       ("lt", -90),
    "pkt_loss_pct":   ("gt", 15.0),
    "temp_c":         ("gt", 55.0),
    "uptime_hrs":     ("lt", 0.5),
    "hop_count":      ("gt", 5),
    "queue_len_pkts": ("gt", 15),
    "tx_rate_kbps":   ("lt", 20.0),
}

# Routing score weights (lower total score = better relay node)
ROUTING_WEIGHTS = {
    "rssi_dbm":       0.30,
    "hop_count":      0.25,
    "battery_v":      0.20,
    "pkt_loss_pct":   0.15,
    "queue_len_pkts": 0.10,
}

# Severity bands (0-100)
SEVERITY_CRITICAL  = 70   # → alert HQ + quarantine
SEVERITY_WARNING   = 40   # → reroute traffic
SEVERITY_WATCH     = 20   # → monitor trend

# Prediction confidence floor — below this treated as uncertain
MIN_CONFIDENCE = 0.65

# Rolling window for trend analysis (number of 5-min readings)
TREND_WINDOW = 6   # = 30 minutes

# If this fraction of nodes are label=1, escalate to satellite
SATELLITE_THRESHOLD = 0.40

# Column rename map — handles different CSV naming conventions
# Left = possible CSV column names, Right = what the model expects
COLUMN_RENAME_MAP = {
    "packet_loss_pct":  "pkt_loss_pct",
    "pkt_loss_%":       "pkt_loss_pct",
    "queue_length":     "queue_len_pkts",
    "queue_len":        "queue_len_pkts",
    "node_id":          "node_id",
    "Node_ID":          "node_id",
    "Timestamp":        "timestamp",
    "Label":            "label",
}
